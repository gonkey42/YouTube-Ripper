# YouTube Ripper

Local web utility that extracts audio and video from YouTube, transcribes with Whisper, and generates clean text transcripts.

## Features

- **Five output modes:** Audio Only, Text Only, Audio + Text, Video, Video + Text
- **Video quality selection:** 480p, 720p, 1080p, or Best Available
- **Audio format:** M4A (AAC) — plays on Mac, Android, iOS
- **Video format:** MP4 with merged audio
- **Transcription:** Local Whisper (faster-whisper) — no cloud APIs, no data leaves your machine
- **Text output:** Clean `.txt` transcript with title, source URL, and paragraph breaks
- **Real-time progress** via Server-Sent Events (download %, speed, ETA for video)
- **Supported URLs:** `youtube.com/watch`, `youtu.be`, `youtube.com/shorts`, `youtube.com/live`

## Output Modes

| Mode | What you get | Quality picker? |
|------|-------------|-----------------|
| Audio Only | M4A | No |
| Text Only | TXT transcript | No |
| Audio + Text | M4A + TXT | No |
| Video | MP4 | Yes |
| Video + Text | MP4 + TXT | Yes |

**Video + Text** downloads the video, extracts the audio track, runs Whisper on it, generates the text transcript, and cleans up the intermediate audio file.

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager
- ffmpeg (`brew install ffmpeg`)

## Quick Start

```bash
cd tools/youtube-ripper
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
| Transcription | faster-whisper (base model, int8) |
| Text generation | Python pathlib |
| Web server | Flask |
| Frontend | Vanilla HTML/CSS/JS |

## Output

Files are saved to `output/` by default (gitignored). The UI shows clickable saved-file links after each run, so you can download the generated files directly from the browser.

Generated filenames include the YouTube id to avoid collisions between same-title videos:

- `Title [video_id].m4a`
- `Title [video_id].mp4`
- `Title [video_id].txt`

In "Text Only" mode, the downloaded audio file is automatically deleted after transcription. In "Video + Text" mode, the app downloads the video once, extracts temporary transcription audio locally with `ffmpeg`, and removes that temporary audio after transcription.

## Notes

- First run downloads the Whisper `base` model (~150MB)
- No database, no accounts, no cloud services
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
| `YOUTUBE_RIPPER_WHISPER_MODEL` | `base` | faster-whisper model name |
| `YOUTUBE_RIPPER_WHISPER_DEVICE` | `cpu` | faster-whisper device |
| `YOUTUBE_RIPPER_WHISPER_COMPUTE_TYPE` | `int8` | faster-whisper compute type |
| `YOUTUBE_RIPPER_COOKIES_FROM_BROWSER` | `chrome:Profile 1` | Browser cookies source for yt-dlp |
| `YOUTUBE_RIPPER_COOKIES` | unset | Netscape-format cookies file path |
