"""Speech to text for push-to-talk.

arecord captures from the default mic (no Python audio deps), faster-whisper
transcribes fully offline in English and Hindi. Everything raises RuntimeError
with a human message the UI shows as a red line; a Ctrl+C is never swallowed.
"""

import shutil
import subprocess
import tempfile
import threading
from pathlib import Path

from config import STT_MODEL, STT_MAX_SECONDS

# 16_000 Hz * 2 bytes = 32_000 bytes/s of S16_LE mono wav
_BYTES_PER_SECOND = 32_000

_proc: subprocess.Popen | None = None
_wav: Path | None = None
_cap_timer: threading.Timer | None = None
_model = None
_model_lock = threading.Lock()
_loaded = False  # True once the whisper model is resident (first use downloads it)


def start() -> None:
    """Begin mic capture. Raises RuntimeError with a human message on failure."""
    global _proc, _wav, _cap_timer
    if _proc is not None:
        raise RuntimeError("already recording")
    if shutil.which("arecord") is None:
        raise RuntimeError("arecord not found (install alsa-utils to use push-to-talk)")
    _wav = Path(tempfile.mkdtemp(prefix="naoki_stt_")) / "mic.wav"
    try:
        _proc = subprocess.Popen(
            ["arecord", "-q", "-f", "S16_LE", "-r", "16000", "-c", "1", str(_wav)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError as exc:
        _proc = None
        raise RuntimeError(f"mic capture failed to start: {exc}") from exc
    # Hard cap so a forgotten Enter cannot eat the drive all afternoon.
    _cap_timer = threading.Timer(
        STT_MAX_SECONDS + 2, lambda: _proc and _proc.terminate() if _proc else None
    )
    _cap_timer.daemon = True
    _cap_timer.start()


def stop() -> tuple[Path, float, bool]:
    """End capture; returns (wav path, seconds recorded, did the hard cap end it)."""
    global _proc, _wav, _cap_timer
    if _proc is None or _wav is None:
        raise RuntimeError("not recording")
    if _cap_timer is not None:
        _cap_timer.cancel()
        _cap_timer = None
    proc, _proc = _proc, _proc and None
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
    if not _wav.is_file() or _wav.stat().st_size < 1000:
        raise RuntimeError("captured silence -- check the mic and try again")
    seconds = _wav.stat().st_size / _BYTES_PER_SECOND
    return _wav, seconds, _cap_capped(seconds)


def abort() -> None:
    """Kill capture and discard the clip (Ctrl+C mid-recording)."""
    global _proc, _wav, _cap_timer
    if _cap_timer is not None:
        _cap_timer.cancel()
        _cap_timer = None
    if _proc is not None:
        proc, _proc = _proc, None
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
    _wav = None


def warm() -> None:
    """Load the whisper model once; the first call fetches ~500MB."""
    global _model
    if _model is None:
        from faster_whisper import WhisperModel

        _model = WhisperModel(STT_MODEL, compute_type="int8")  # fast on CPU
    _loaded = True


def loaded() -> bool:
    return _loaded


def flags() -> dict:
    """Runtime-settable stt flags with their current values."""
    return {"model": STT_MODEL, "max": STT_MAX_SECONDS}


def set_flag(name: str, raw: str) -> str:
    """Set one runtime flag (session only; config.py still rules at startup).

    Raises ValueError on an unknown flag or a bad value.
    """
    global STT_MODEL, STT_MAX_SECONDS, _model, _loaded
    if name not in flags():
        raise ValueError(f"stt flags: {', '.join(sorted(flags()))}")
    if name == "model":
        value = str(raw).strip().lower()
        if not value:
            raise ValueError("model needs a whisper size, e.g. tiny / base / small")
        if value == STT_MODEL:
            return f"model = {value} (already)"
        STT_MODEL = value
        _model, _loaded = None, False  # next transcribe warms the new model
        return f"model = {value} (first use downloads it)"
    # max seconds per recording; the Enter-wait window keeps its +5 grace
    try:
        value = int(float(str(raw).strip()))
    except ValueError:
        raise ValueError(f"max takes a number, not {raw!r}") from None
    if not 3 <= value <= 120:
        raise ValueError(f"max must be 3–120 seconds, not {value}")
    STT_MAX_SECONDS = value
    return f"max = {value} seconds (next recording starts using it)"


def transcribe(wav: Path) -> str:
    """Transcribe a wav (any language Whisper knows). Returns the text."""
    try:
        warm()
    except Exception as exc:
        raise RuntimeError(f"whisper model failed to load: {exc}") from exc
    segments, _info = _model.transcribe(str(wav))
    text = " ".join(seg.text for seg in segments).strip()
    if not text:
        raise RuntimeError("couldn't make out any words -- try again, closer to the mic")
    return text


def _cap_capped(seconds: float) -> bool:
    """Did this recording run into the hard cap (vs a manual Enter stop)?"""
    return seconds >= STT_MAX_SECONDS
