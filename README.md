# YouTube Ripper

Local web utility that extracts audio and video from YouTube, transcribes with Whisper, and generates clean text transcripts.

## Features

- **Five output modes:** Audio Only, Text Only, Audio + Text, Video, Video + Text
- **Video quality selection:** 480p, 720p, 1080p, or Best Available
- **Audio format:** M4A (AAC) — plays on Mac, Android, iOS
- **Video format:** MP4 with merged audio
- **Transcription:** Whisper Turbo, run locally. It is the only model. The app uses MLX on Apple Silicon and faster-whisper on other machines. Media stays local during transcription.
- **Captions first:** The UI tries uploaded YouTube captions, then automatic English captions, and falls back to local transcription. Select Transcribe locally to always transcribe the audio.
- **Transcript controls:** One setting, the transcript source. Transcription is in English.
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
| Transcription | Whisper Turbo, with MLX Whisper on Apple Silicon and faster-whisper elsewhere |
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

Turbo is the only model. There is no engine or model menu. The app picks the engine automatically. It uses MLX on the Apple GPU on Apple Silicon and faster-whisper on other machines. Set `YOUTUBE_RIPPER_TRANSCRIPTION_BACKEND` to `mlx` or `faster-whisper` to override that choice. The setting applies to every request, including requests from the web interface. The engine never switches during a job. If MLX fails, the app reports the error and does not fall back to faster-whisper.

The model downloads from Hugging Face on first use and stays on disk after that. The app unloads the model and releases its GPU memory when each job ends, so the server does not hold several gigabytes between jobs. Each job reloads the model, which takes a few seconds. faster-whisper uses voice activity detection. MLX uses its native no-speech filtering.

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

- The first local transcription downloads the Whisper Turbo model from Hugging Face. The download is about 1.5 GB for MLX.
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
| `YOUTUBE_RIPPER_TRANSCRIPTION_BACKEND` | `auto` | Transcription engine for every request, including the web interface. One of `auto`, `mlx`, or `faster-whisper` |
| `YOUTUBE_RIPPER_WHISPER_DEVICE` | `cpu` | Device for the faster-whisper backend only |
| `YOUTUBE_RIPPER_WHISPER_COMPUTE_TYPE` | `int8` | Compute type for the faster-whisper backend only |
| `YOUTUBE_RIPPER_COOKIES_FROM_BROWSER` | `chrome:Profile 1` | Browser cookies source for yt-dlp |
| `YOUTUBE_RIPPER_COOKIES` | unset | Netscape-format cookies file path |

`auto` selects MLX on Apple Silicon and faster-whisper elsewhere. Setting `mlx` on any other machine is an error.
