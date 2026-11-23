# YouTube Ripper

Local web utility that extracts audio and video from YouTube, transcribes with Whisper, and generates clean text transcripts.

## Features

- **Five output modes:** Audio Only, Text Only, Audio + Text, Video, Video + Text
- **Video quality selection:** 480p, 720p, 1080p, or Best Available
- **Audio format:** M4A (AAC) — plays on Mac, Android, iOS
- **Video format:** MP4 with merged audio
- **Transcription:** Apple MLX selected by default in the web interface, with faster-whisper available as the other engine choice. Media stays local during transcription.
- **Captions first:** The UI tries uploaded YouTube captions, then automatic English captions, with the selected local engine as fallback. Select Transcribe locally to always transcribe the audio.
- **Transcript controls:** English transcription, Apple MLX or faster-whisper engine selection, and model selection.
- **Text output:** Clean `.txt` transcript with title, source URL, transcript source, language, and paragraph breaks.
- **Real-time progress** via Server-Sent Events, including audio/video download percentage, speed, ETA, and keepalives during processing.
- **Download checks:** Saved-file links use the actual final extension, and missing or empty downloads report an error.
- **Supported URLs:** `youtube.com/watch`, `youtu.be`, `youtube.com/shorts`, `youtube.com/live`

## Output Modes

| Mode | What you get | Quality picker? |
|------|-------------|-----------------|
| Audio Only | M4A | No |
| Text Only | TXT transcript | No |
| Audio + Text | M4A + TXT | No |
| Video | MP4 | Yes |
| Video + Text | MP4 + TXT | Yes |

**Video + Text** downloads the video once. If captions are unavailable or Transcribe locally is selected, the transcription engine reads the video's audio stream directly, without an intermediate AAC conversion.

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager
- ffmpeg (`brew install ffmpeg`)
- Node.js 22+ for YouTube JavaScript challenges. On Homebrew systems, this app prefers `/opt/homebrew/opt/node/bin/node` without changing the system `node` link.

## Quick Start

```bash
cd tools/youtube-ripper
uv sync --locked
uv run python app.py
```

Open http://localhost:4039, paste a YouTube URL, pick your mode, and hit **Rip It**.

For development with auto-reload:
```bash
FLASK_DEBUG=1 uv run python app.py
```

## Stack

| Component | Tool |
|-----------|------|
| YouTube download | yt-dlp |
| Transcription | MLX Whisper on Apple Silicon; faster-whisper elsewhere; Turbo model by default |
| Text generation | Python pathlib |
| Web server | Flask |
| Frontend | Vanilla HTML/CSS/JS |

## Output

Files are saved to `output/` by default (gitignored). The UI shows clickable saved-file links after each run, so you can download the generated files directly from the browser.

Generated filenames include the YouTube id to avoid collisions between same-title videos:

- `Title [video_id].m4a`
- `Title [video_id].mp4`
- `Title [video_id].txt`

In "Text Only" mode, available captions produce the transcript without downloading media or loading Whisper. When local transcription is needed, the downloaded audio file is automatically deleted afterward. Video + Text transcribes the video directly when captions are unavailable.

The web form saves TXT transcripts for text modes. SRT/VTT export remains available through the API, but is not offered in the form. The transcript and progress messages identify whether the source is uploaded captions, automatic captions, MLX, or faster-whisper. The web interface uses English without a language picker. Caption selection excludes YouTube's machine-translated tracks; local transcription does not translate.

## Transcription speed

The web interface offers two engines: Apple MLX, selected by default, and faster-whisper. Apple MLX uses the Apple GPU. The engine does not switch automatically based on the video; an MLX error is reported rather than silently switching to faster-whisper. Both engines run Whisper speech models locally, so choosing Transcribe locally does not force the faster-whisper engine.

Turbo is the default model. Larger models require more memory and can be slower. A model downloads on first use and stays cached for reuse. Each process keeps one active model, so changing engines or models requires loading it again. faster-whisper uses voice activity detection; MLX uses its native no-speech filtering.

Choose captions when speed matters most. Choose Transcribe locally when you want a fresh transcription of the audio. English is supplied explicitly, avoiding language detection.

## Validation and maintenance

```bash
uv run python -m unittest discover -q
```

Startup and the web page check ffmpeg and Node.js and show actionable setup errors. Captions-only requests can work without ffmpeg; fallback to media download checks it before proceeding. The lockfile includes yt-dlp 2026.10.26 and MLX Whisper 0.4.3. To update yt-dlp later:

```bash
uv lock --upgrade-package yt-dlp
uv sync --locked
```

## Notes

- First local transcription downloads the selected Whisper model from Hugging Face. Model size varies by engine and model.
- No database, application accounts, or cloud transcription service.
- Port: 4039
- The server binds to `0.0.0.0` for Tailscale reachability, but requests are accepted only from loopback and Tailscale client ranges by default.
- YouTube downloads use your local Chrome `Profile 1` cookies by default to avoid bot/sign-in checkpoints.
- To override auth, set `YOUTUBE_RIPPER_COOKIES_FROM_BROWSER`, for example `firefox:default`, or set `YOUTUBE_RIPPER_COOKIES` to a Netscape-format cookies file.

## Configuration

Optional environment variables:

| Variable | Default | Purpose |
|----------|---------|---------|
| `YOUTUBE_RIPPER_OUTPUT_DIR` | `output/` | Directory for generated files |
| `YOUTUBE_RIPPER_HOST` | `0.0.0.0` | Flask bind host |
| `YOUTUBE_RIPPER_PORT` | `4039` | Flask port |
| `YOUTUBE_RIPPER_ALLOWED_CLIENTS` | `127.0.0.0/8,::1/128,100.64.0.0/10,fd7a:115c:a1e0::/48` | Comma-separated allowed client IPs/CIDR ranges |
| `YOUTUBE_RIPPER_WHISPER_MODEL` | `turbo` | Default model name for the selected engine |
| `YOUTUBE_RIPPER_TRANSCRIPTION_BACKEND` | `auto` | Used only by API/internal callers requesting `auto`; the web interface sends an explicit engine and overrides this setting |
| `YOUTUBE_RIPPER_WHISPER_DEVICE` | `cpu` | Device for the faster-whisper backend only |
| `YOUTUBE_RIPPER_WHISPER_COMPUTE_TYPE` | `int8` | Compute type for the faster-whisper backend only |
| `YOUTUBE_RIPPER_COOKIES_FROM_BROWSER` | `chrome:Profile 1` | Browser cookies source for yt-dlp |
| `YOUTUBE_RIPPER_COOKIES` | unset | Netscape-format cookies file path |

The UI model selector overrides `YOUTUBE_RIPPER_WHISPER_MODEL`. MLX supports `tiny`, `base`, `small`, `medium`, `large-v3`, and `turbo`; custom faster-whisper models require selecting that backend.
