"""Offline CPU text-to-speech, engine-switchable.

Primary: Supertonic 3 (ONNX) — 31 languages including Hindi, 44.1 kHz output,
preset female voice styles (F1-F5; her skin is a 65/35 F1+F5 blend in config). Text is cut into
sentence-sized pieces; each piece renders in under a second (RTF ~0.2 on CPU)
and goes to the speakers the moment it is done while the next piece renders
ahead. Language is sniffed per piece from the script: Devanagari-only -> `hi`,
mixed Devanagari + Latin -> `na` (auto), plain Latin -> `en`.

Without a usable sounddevice the clip renders to a wav and a system player
(paplay/pw-play/aplay/ffplay) plays it instead. Missing files or audio
problems print one warning and the bot stays silent; Ctrl+C during playback
is torn down and re-raised so the caller can tell a cancel from silence.
"""

import asyncio
import contextlib
import importlib.util
import random
import re
import shutil
import subprocess
import tempfile
import threading
import unicodedata
from pathlib import Path

import numpy as np

from config import (
    TTS_ENABLED, TTS_GAIN, TTS_JITTER_CHANCE, TTS_JITTER_SPEED, TTS_MAX_CHARS,
    TTS_ONNX_DIR, TTS_REALTIME, TTS_SPEED, TTS_STEPS, TTS_VOICE_STYLE
)

ENABLED = TTS_ENABLED
STYLE_FILE = Path(TTS_VOICE_STYLE)  # supertonic voice preset, loaded with the engine
_PLAYERS = ("paplay", "pw-play", "aplay", "ffplay")
_SLICE_SECONDS = 0.25  # how often playback looks for a stop
_PIECE_CHARS = 120  # replies are rendered one sentence-sized piece at a time
_QUEUE_SIZE = 2  # pieces held ready while the current one plays
_CANCEL_JOIN = 1.0  # seconds to wait for the pump after a stop
_S3_FILES = (
    "duration_predictor.onnx", "text_encoder.onnx", "vector_estimator.onnx",
    "vocoder.onnx", "tts.json", "unicode_indexer.json",
)

_warned: set[str] = set()
_model = None
_style = None
_model_lock = threading.Lock()
_speak_lock = threading.Lock()
_stop = threading.Event()
_stream_failed = False


def set_enabled(enabled: bool) -> None:
    global ENABLED
    ENABLED = enabled


def flags() -> dict:
    """Every runtime-settable tts flag with its current value."""
    return {
        "enabled": ENABLED,
        "realtime": TTS_REALTIME,
        "speed": TTS_SPEED,
        "gain": TTS_GAIN,
        "steps": TTS_STEPS,
        "max_chars": TTS_MAX_CHARS,
        "style": STYLE_FILE.stem,  # supertonic preset or blend
    }


def _as_bool(raw: str) -> bool:
    low = str(raw).strip().lower()
    if low in ("on", "true", "1", "yes"):
        return True
    if low in ("off", "false", "0", "no"):
        return False
    raise ValueError(f"{raw!r} is not on/off")


def _as_float(low: float, high: float):
    def parse(raw: str, name: str) -> float:
        try:
            value = float(str(raw).strip())
        except ValueError:
            raise ValueError(f"{name} takes a number, not {raw!r}") from None
        if not low <= value <= high:
            raise ValueError(f"{name} must be {low}–{high}, not {value}")
        return value
    return parse


def _as_int(low: int, high: int):
    def parse(raw: str, name: str) -> int:
        try:
            value = int(float(str(raw).strip()))
        except ValueError:
            raise ValueError(f"{name} takes a number, not {raw!r}") from None
        if not low <= value <= high:
            raise ValueError(f"{name} must be {low}–{high}, not {value}")
        return value
    return parse


def _drop_engine() -> None:
    """Forget loaded onnx sessions so the next speak reloads the current dials."""
    global _model, _style
    _model = None
    _style = None


def set_flag(name: str, raw: str) -> str:
    """Set one runtime flag (session only; config.py still rules at startup).

    Raises ValueError on an unknown flag or a bad value.
    """
    global TTS_REALTIME, TTS_SPEED, TTS_GAIN, TTS_STEPS, TTS_MAX_CHARS

    if name not in flags():
        raise ValueError(f"tts flags: {', '.join(sorted(flags()))}")
    if name == "enabled":
        set_enabled(_as_bool(raw))
        return "voice on" if ENABLED else "voice off"
    if name == "realtime":
        TTS_REALTIME = _as_bool(raw)
    elif name == "speed":
        TTS_SPEED = _as_float(0.5, 2.0)(raw, "speed")
    elif name == "gain":
        TTS_GAIN = _as_float(0.5, 3.0)(raw, "gain")
    elif name == "steps":
        TTS_STEPS = _as_int(1, 50)(raw, "steps")
    elif name == "max_chars":
        TTS_MAX_CHARS = _as_int(100, 2000)(raw, "max_chars")
    else:
        return _set_style_dial(name, raw)
    return f"{name} = {flags()[name]}"


def _set_style_dial(name: str, raw: str) -> str:
    """Style dial — the one that must reload the engine."""
    global STYLE_FILE
    # a preset/blend file next to the current one, matched by stem
    want = str(raw).strip()
    available = {p.stem.lower(): p for p in STYLE_FILE.parent.glob("*.json")}
    hit = available.get(want.lower())
    if hit is None:
        raise ValueError(f"style is one of {', '.join(sorted(available))}, not {raw!r}")
    if hit == STYLE_FILE:
        return f"style = {hit.stem} (already)"
    STYLE_FILE = hit
    _drop_engine()
    return f"style = {hit.stem} (reloads on the next speak)"


def stop() -> None:
    """Cut playback short (safe from any thread, safe while silent)."""
    _stop.set()


def shutdown() -> None:
    """Stop talking and drop the engine so exit leaves no RAM behind."""
    global _model, _style
    stop()
    with _model_lock:
        _model = None
        _style = None


def speak(text: str) -> str | None:
    """Say `text` out loud (blocking); returns the engine name when audio played.

    Ctrl+C during playback is re-raised once the audio is torn down.
    """
    engine, state = _pick_engine()
    if engine is None:
        return None

    clean = _clean_for_speech(text)
    if not ENABLED or not clean:
        return None

    broken = _missing(state)
    if broken:
        _warn_once(f"[tts] {engine} cannot speak ({', '.join(broken)} missing) -- silent")
        return None

    with _speak_lock:
        _stop.clear()
        try:
            if TTS_REALTIME and state["sounddevice"]:
                played = _play_stream(clean)
                if not played and not _stop.is_set():
                    _note_stream_failure()
            else:
                played = _play_file(clean)
        except KeyboardInterrupt:
            stop()  # tell the pump to abort at its next slice
            raise
        except Exception as exc:
            _warn_once(f"[tts] {engine} failed: {type(exc).__name__}: {exc}")
            return None
        finally:
            _stop.clear()
        return engine if played else None


def _can_stream() -> bool:
    """True when sounddevice can open an output device on this machine."""
    try:
        import sounddevice as sd

        sd.query_devices(kind="output")
        return True
    except Exception:
        return False


def _audio_state() -> dict[str, bool]:
    """Audio stack pieces and engine readiness, as they stand right now."""
    supertonic_ok = importlib.util.find_spec("onnxruntime") is not None and all(
        (TTS_ONNX_DIR / name).exists() for name in _S3_FILES
    ) and STYLE_FILE.exists()
    return {
        "sounddevice": importlib.util.find_spec("sounddevice") is not None
        and _can_stream(),
        "soundfile": importlib.util.find_spec("soundfile") is not None,
        "player": any(shutil.which(name) for name in _PLAYERS),
        "supertonic-ok": supertonic_ok,
    }


def _pick_engine() -> tuple[str | None, dict[str, bool] | None]:
    """The one engine and its state, or (None, None) when it cannot run."""
    state = _audio_state()
    if not state["supertonic-ok"]:
        _warn_once("[tts] supertonic files missing -- silent")
        return None, None
    return "supertonic", state


def _missing(state: dict[str, bool]) -> list[str]:
    """What is missing before the chosen engine can play anything."""
    broken: list[str] = []
    if not state["sounddevice"]:
        if not state["player"]:
            broken.append("player")
        if not state["soundfile"]:
            broken.append("soundfile")
    return broken


def _warn_once(message: str) -> None:
    if message not in _warned:
        _warned.add(message)
        print(message)


def _note_stream_failure() -> None:
    """Realtime playback blew up: use the wav player for the rest of the run."""
    global _stream_failed
    if not _stream_failed:
        _stream_failed = True
        _warn_once("[tts] realtime playback failed -- playing wav files instead")


def _load_engine():
    """Load the engine once, on first use (~1s)."""
    global _model, _style
    with _model_lock:
        if _model is None:
            from supertonic_helper import load_text_to_speech, load_voice_style

            _model = load_text_to_speech(str(TTS_ONNX_DIR))
            _style = load_voice_style([str(STYLE_FILE)])
        return _model


def _render_piece(piece: str, model) -> tuple[np.ndarray, int]:
    """Render one piece whole: (float32 samples, sample_rate)."""
    speed = TTS_SPEED
    if random.random() < TTS_JITTER_CHANCE:  # slow drag on only some pieces
        speed = round(random.uniform(*TTS_JITTER_SPEED), 2)
    wav, dur = model(piece, _s3_lang(piece), _style, total_step=TTS_STEPS, speed=speed)
    n = int(model.sample_rate * float(np.asarray(dur).ravel()[0]))
    return wav[0, :n], model.sample_rate


# Modern texting Hindi arrives as Latin script; render it as Devanagari so it
# is pronounced correctly. Keys match whole words case-insensitively (so "chal"
# never catches "chalo"), and the standalone English "sun" is safe because only
# the phrase "sun na" is listed. Anything unlisted stays Latin and reads
# English-style — the system prompt keeps her Hindi inside this map for that
# reason.
_HINDI_WORDS = {
    "sun na": "सुन ना",
    "chalo": "चलो",
    "chal": "चल",
    "arre": "अरे",
    "acha": "अच्छा",
    "haan": "हाँ",
    "jaldi": "जल्दी",
    "yaar": "यार",
    "ruko": "रुको",
    "bas": "बस",
    "abhi": "अभी",
    "theek": "ठीक",
    "thoda": "थोड़ा",
    "aaj": "आज",
    "kal": "कल",
    "aao": "आओ",
    "kaam": "काम",
    "kya": "क्या",
    "kyun": "क्यों",
    "bata": "बता",
    "na": "ना",
}

_DEVA_RE = re.compile(
    "|".join(rf"\b{re.escape(key)}\b" for key in sorted(_HINDI_WORDS, key=len, reverse=True)),
    flags=re.IGNORECASE,
)


def _deva(text: str) -> str:
    """Romanized Hindi words -> Devanagari, for the engine's ears (not the screen)."""
    return _DEVA_RE.sub(lambda m: _HINDI_WORDS[m.group(0).lower()], text)


def _s3_lang(piece: str) -> str:
    """Supertonic routing from the piece's script: en / hi / na (auto)."""
    if not any("\u0900" <= ch <= "\u097F" for ch in piece):
        return "en"
    mixed = any(ch.isascii() and ch.isalpha() for ch in piece)
    return "na" if mixed else "hi"


def _loud(samples):
    """Gain + limiter: louder output, never past full scale (no clip grit)."""
    return (samples * TTS_GAIN).clip(-1.0, 1.0).astype("float32", copy=False)


def _pieces(text: str) -> list[str]:
    """Sentence-sized pieces, so speech starts (and stops) without waiting.

    Short pieces keep every render quick and every stop cheap; supertonic also
    re-chunks anything over 300 chars itself, which per-piece inputs never hit.
    """
    pieces: list[str] = []
    for sentence in re.split(r"(?<=[.!?…])\s+", text):
        sentence = sentence.strip()
        while len(sentence) > _PIECE_CHARS:
            cut = max(sentence.rfind(mark, 0, _PIECE_CHARS) for mark in (",", ";", ":"))
            if cut <= 0:
                cut = sentence.rfind(" ", 0, _PIECE_CHARS)
            if cut <= 0:
                cut = _PIECE_CHARS
            pieces.append(sentence[: cut + 1].strip())
            sentence = sentence[cut + 1 :].strip()
        if sentence:
            pieces.append(sentence)
    return pieces


def _clean_for_speech(text: str) -> str:
    """Strip markdown so it is not read aloud, and cap the length."""
    text = re.sub(r"```.*?```", " code block omitted. ", text, flags=re.DOTALL)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"[*_#>]+", "", text)
    # Drop emojis/symbols so TTS reads text cleanly; keep ASCII and unicode
    # letters/markers/numbers for non-English scripts.
    text = "".join(
        ch
        for ch in text
        if (ch.isascii() and (ch.isalnum() or ch.isspace() or ch in ".,!?;:'\"()-+/=%@&"))
        or (not ch.isascii() and unicodedata.category(ch)[0] in "LMN")
    )
    # Supertonic pronounces Devanagari natively; _deva() adds it so romanized
    # Hindi comes out as its real pronunciation.
    text = _deva(text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > TTS_MAX_CHARS:
        text = text[:TTS_MAX_CHARS].rsplit(" ", 1)[0] + " ..."
    return text


def _cancelled(abort: threading.Event) -> bool:
    """Latch a cancel for this call, so a later `_stop.clear()` cannot revive it."""
    if abort.is_set() or _stop.is_set():
        abort.set()
        return True
    return False


def _play_stream(text: str) -> bool:
    """Realtime playback: the next piece renders while the current one plays."""
    outcome: list[bool] = []
    abort = threading.Event()

    def worker() -> None:
        outcome.append(asyncio.run(_pump_stream(text, abort)))

    thread = threading.Thread(target=worker, name="naoki-tts", daemon=True)
    thread.start()
    try:
        while thread.is_alive():
            thread.join(_SLICE_SECONDS)
            if _stop.is_set():
                # The pump aborts the stream on its own; do not hold the
                # prompt hostage for the piece the model is still finishing.
                abort.set()
                thread.join(_CANCEL_JOIN)
                break
    except KeyboardInterrupt:
        stop()
        abort.set()
        thread.join(_CANCEL_JOIN)
        raise
    return bool(outcome and outcome[0])


async def _pump_stream(text: str, abort: threading.Event) -> bool:
    """Render pieces ahead of playback and push each one straight to the DAC."""
    import sounddevice as sd

    queue: asyncio.Queue = asyncio.Queue(maxsize=_QUEUE_SIZE)

    async def render() -> None:
        """Synthesize piece by piece, so a stop only waits for one piece."""
        model = None
        try:
            for piece in _pieces(text):
                if _cancelled(abort):
                    return
                if model is None:
                    model = _load_engine()
                # Supertonic renders a piece whole in a fraction of a
                # second; the thread pool keeps playback moving meanwhile.
                samples, rate = await asyncio.to_thread(_render_piece, piece, model)
                await queue.put((samples, rate))
        finally:
            with contextlib.suppress(asyncio.QueueFull):
                queue.put_nowait(None)

    producer = asyncio.create_task(render())
    stream = None
    try:
        while True:
            if _cancelled(abort):
                return False
            item = await queue.get()
            if item is None:
                break
            samples, sample_rate = item
            if stream is None:
                stream = sd.RawOutputStream(
                    samplerate=sample_rate, channels=1, dtype="float32"
                )
                stream.start()
            raw = _loud(samples)
            slice_size = max(1, int(sample_rate * _SLICE_SECONDS))
            for start in range(0, raw.size, slice_size):
                if _cancelled(abort):
                    return False
                stream.write(raw[start : start + slice_size].tobytes())
        return stream is not None
    finally:
        # Silence first: waiting for the abandoned render must not keep talking.
        if stream is not None:
            if abort.is_set():
                stream.abort(ignore_errors=True)  # drop the queue, cut the tail
            else:
                stream.stop(ignore_errors=True)  # let the buffered tail play out
            stream.close(ignore_errors=True)
        producer.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await producer


def _play_file(text: str) -> bool:
    """Fallback: render the whole clip, then hand a wav to a system player."""
    import soundfile as sf

    player = next((name for name in _PLAYERS if shutil.which(name)), None)
    if player is None:
        _warn_once(f"[tts] no audio player found ({', '.join(_PLAYERS)}) -- silent")
        return False

    samples, sample_rate = _render_piece(text, _load_engine())
    with tempfile.NamedTemporaryFile(
        prefix="naoki_tts_", suffix=".wav", delete=False
    ) as handle:
        wav_path = Path(handle.name)
    try:
        sf.write(str(wav_path), _loud(samples), sample_rate)
        command = [player, str(wav_path)]
        if player == "ffplay":
            command = [
                "ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet",
                str(wav_path),
            ]
        process = subprocess.Popen(
            command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        try:
            while process.poll() is None:
                if _stop.wait(_SLICE_SECONDS):
                    process.terminate()
                    return False
            return True
        finally:
            if process.poll() is None:
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
    finally:
        wav_path.unlink(missing_ok=True)
