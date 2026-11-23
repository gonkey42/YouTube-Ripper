# YouTube Ripper upgrade status

Implementation and local verification complete. Changes are uncommitted. Existing user edits were preserved. The user subsequently requested an app restart. The local YouTube Ripper launchd service was restarted and verified over HTTP. No commit or PR was created, and no YouTube cookies or downloads were used for verification.

## Scope

Completed the requested downloader update/checks, transcription controls, YouTube captions, and download validation/progress. User explicitly excluded file-isolation/overwrite changes and persistent jobs. Transcription speed was the priority, including MLX on Apple Silicon.

## Delivered

- yt-dlp 2026.10.26 and its required certifi update; conditional MLX Whisper 0.4.3 dependency for Apple Silicon.
- Default local engine is MLX on Apple Silicon. Explicit faster-whisper selection and environment overrides remain available. MLX errors do not silently switch engines.
- UI defaults to available YouTube captions, with local transcription fallback. Text Only with captions skips media download and inference.
- Language detection/selection, model selection, source labels, TXT plus optional SRT/VTT exports.
- Video transcription reads the video directly rather than making an intermediate AAC file.
- Actual final download filenames, nonempty-file validation, audio/video progress, and processing keepalives.
- Startup/page/request dependency diagnostics for ffmpeg and Node.js.

## Verification

- `uv sync --locked`: passed after dependency changes.
- `.venv/bin/python -m unittest discover -q`: 105 tests passed.
- `uv lock --check`: passed.
- Rendered Flask template and `node --check` on its script: passed.
- Local ffmpeg/Node diagnostics: no issues. Automatic engine resolves to `mlx`.
- `git diff --check`: passed.
- No live YouTube end-to-end test or browser interaction was performed. HTTP, download metadata, captions, and failure paths were verified with local fixtures/mocks.

## Real transcription smoke test

The transcription agent ran both engines on the same locally generated MP4 with roughly 19 seconds of English speech, using the base model and automatic language detection. Both returned timed English segments ending at 18.7 seconds and read the MP4 directly.

MLX first inference including import/model load took 3.832 seconds; the warm run took 0.293 seconds. Faster-whisper CPU/int8 took 3.133 seconds first and 1.021 seconds warm. These are a single short smoke test with different native decoder settings, not a general performance guarantee. CPU used beam size 5 and VAD; MLX used native defaults. Model download/cache checks were excluded. The generated color-video container lasted 70 seconds, while speech lasted about 19 seconds.

## Operational handoff

Restarted `com.claudebot.youtube-ripper` with launchctl at the user's request. The replacement process serves HTTP 200 at http://127.0.0.1:4039/ with the MLX, caption, and subtitle controls present and no dependency warnings. No automatic resume task was created: the user subsequently resumed work directly. All four subagent tasks have finished. No further implementation is pending within this scope.

## UI simplification requested after rollout

- Renamed local source to Transcribe locally. It uses the chosen engine, never forces faster-whisper.
- Removed the language picker; web requests default to English.
- Removed Automatic from the engine menu. Apple MLX is selected by default; faster-whisper is the only other choice. Explicit choices bypass the environment's automatic selection.
- Internal/API automatic selection and optional language overrides remain compatible.
- Updated request-contract tests verify default English and both explicit engine choices. All 34 request/transcription tests passed for this follow-up.

## Default model update

The user selected Turbo as the default for both engines. The menu now reads Server default (turbo); explicit model choices and environment overrides remain supported. Historical base-model benchmark results above do not measure Turbo.

The user subsequently requested removal of the Also save subtitles controls. SRT/VTT checkboxes and form submission fields were removed; browser requests now use the existing TXT-only default.

## Turbo only and per-job memory release (2026-11-23)

Turbo is now the only model, and the engine and model menus are gone from the page. `TranscriptionOptions` no longer has `model` or `backend`. The `/rip` endpoint ignores those two settings if a caller still sends them. `YOUTUBE_RIPPER_TRANSCRIPTION_BACKEND` is the only engine override and now applies to web requests too. Nothing reads `YOUTUBE_RIPPER_WHISPER_MODEL` any more. This replaces the engine and model statements in the sections above.

The server used to keep several gigabytes of GPU memory after the first transcription, because mlx_whisper caches the model and MLX keeps a buffer cache. `transcription.py` now clears `ModelHolder`, runs `gc.collect()`, and calls `mx.clear_cache()` after every MLX job, on success and on failure. The release runs after the `except` block has ended. A release inside `finally` frees nothing while the exception is still propagating, because the traceback frames hold the model. faster-whisper creates its model as a local variable for each call. Each job reloads the model, which takes a few seconds.

Verification: `uv run python -m unittest discover -q` runs 107 tests and all pass. A separate process transcribed a 7 second speech clip with the real Turbo model and then a missing file. MLX reported 0.1 MB active and 0.0 MB cached after each call. The peak was about 2040 MB during the successful job and about 1540 MB during the failed one.
