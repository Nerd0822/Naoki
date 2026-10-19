# Naoki

Personal assistant with an offline voice: terminal chat over a local LLM
(LangGraph + Ollama) with Supertonic-on-CPU text-to-speech, push-to-talk
microphone input, and per-tool utilities (email, files, shell jobs, web,
memory). Works fully offline except model downloads.

Layout is a uv package:

```
src/naoki/
├── agent.py        # LangGraph agent
├── config.py       # every dial + .env loading
├── helper.py       # shared helpers (archive, input handling)
├── rag.py          # chroma-indexed memory
├── run.py          # terminal loop, /commands, push-to-talk
├── stt.py          # arecord capture + faster-whisper transcription
├── tts.py          # Supertonic 3 voice, romanized-Hindi handling
├── supertonic_helper.py  # vendored Supertone inference core (MIT)
├── tools/          # agent tools (email, files, memory, shell, web)
└── models/         # gitignored: supertonic onnx + voice styles
```

## Setup

1. Sync the environment (creates `.venv/` from `pyproject.toml` + `uv.lock`):

   ```bash
   uv sync
   ```

2. Keep Ollama running locally (default port 11434).

3. Secrets live in `src/naoki/.env` (gitignored):

   ```
   model=<ollama chat model>
   embed_model=<ollama embedding model>
   gmail_user=<optional, email tool>
   gmail_pass=<optional>
   imap_host=imap.gmail.com
   ```

4. Model weights are not in git — download once into
   `src/naoki/models/supertonic-3/` from pinned revision
   `aafc6e3` of [supertone-oss-archive/supertonic-3](https://huggingface.co/supertone-oss-archive/supertonic-3):

   ```bash
   BASE=https://huggingface.co/supertone-oss-archive/supertonic-3/resolve/aafc6e32416a594460b32413efc49d7fe4ce6d46
   DEST=src/naoki/models/supertonic-3
   mkdir -p $DEST/onnx $DEST/voice_styles
   for f in duration_predictor text_encoder vector_estimator vocoder; do
     curl -L -C - --retry 5 --retry-all-errors -o $DEST/onnx/$f.onnx $BASE/onnx/$f.onnx
   done
   curl -L -o $DEST/onnx/tts.json $BASE/onnx/tts.json
   curl -L -o $DEST/onnx/unicode_indexer.json $BASE/onnx/unicode_indexer.json
   curl -L -o $DEST/config.json $BASE/config.json
   for f in F1 F2 F3 F4 F5; do
     curl -L -o $DEST/voice_styles/$f.json $BASE/voice_styles/$f.json
   done
   ```

   `F1F5.json` (the 65/35 voice blend used by default) regenerates from the
   two presets by averaging both style tensors; first `/mic` use downloads
   the ~500MB whisper model on its own.

5. Microphone capture uses ALSA's `arecord` (on Arch: `pacman -S alsa-utils`).

## Run

```bash
uv run python src/naoki/run.py
```

End a turn with Enter; `quit` / `exit` / `bye` leaves, Ctrl+C interrupts
(streaming, speech, recording — each safe to interrupt).

## Commands

| command | action |
|---------|--------|
| `/voice` | mute / unmute her |
| `/again` | speak the last reply again |
| `/mic` | push-to-talk round (Enter or alt+x stops, auto-caps at the limit) |
| `/stop` | cut current speech |
| `/tts [<flag> <value>]` | list or set voice flags for the session |
| `/stt [<flag> <value>]` | same for transcription flags |

## Voice

- **Engine:** Supertonic 3, ONNX Runtime, CPU, realtime per-sentence pieces
  (~0.2 RTF). Language is sniffed per sentence: English, Devanagari Hindi,
  or auto-mixed.
- **Skin:** `F1F5.json` — a 65/35 blend of presets F1 (calm, low) and F5
  (soft, breathy).
- **Mood jitter:** half of all pieces render at a random slower speed
  (0.90–1.00 vs the 1.1 base), so delivery drags in different places every
  time.
- **Romanized Hindi:** display text stays romanized; the speech path maps
  the word list (see `_HINDI_WORDS` in `tts.py`) to Devanagari so it is
  pronounced as Hindi. New words go in both `tts.py` and the system prompt
  in `config.py`.
- **Live dials (session-only):** `enabled, realtime, speed, gain, steps,
  max_chars, style` — `style` accepts any preset/blend stem in
  `voice_styles/`; persistence is `config.py`.

## Notes

- `src/naoki/models/` and `src/naoki/data/` (chroma db, backups) are
  gitignored; everything else is source.
- The `naoki` console script is a placeholder for now — launch with
  `uv run python src/naoki/run.py`.
