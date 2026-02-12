# YouTube Ripper

Local web utility that extracts audio from YouTube videos, transcribes them with Whisper, and generates clean formatted PDFs.

## Features

- **Three output modes:** Audio Only, Text Only, or Both
- **Audio format:** M4A (AAC) — plays on Mac, Android, iOS
- **Transcription:** Local Whisper (faster-whisper) — no cloud APIs, no data leaves your machine
- **PDF output:** Clean document with title, source URL, paragraph breaks
- **Real-time progress** via Server-Sent Events

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager
- ffmpeg (`brew install ffmpeg`)
- pango + glib (`brew install pango glib`) — for WeasyPrint PDF rendering

## Quick Start

```bash
cd tools/youtube-ripper
DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib uv run python app.py
```

Open http://localhost:4039, paste a YouTube URL, pick your mode, and hit **Rip It**.

For development with auto-reload:
```bash
FLASK_DEBUG=1 DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib uv run python app.py
```

## Stack

| Component | Tool |
|-----------|------|
| YouTube download | yt-dlp |
| Transcription | faster-whisper (base model, int8) |
| PDF generation | WeasyPrint |
| Web server | Flask |
| Frontend | Vanilla HTML/CSS/JS |

## Output

Files are saved to `output/` (gitignored). In "Text Only" mode, the audio file is automatically deleted after transcription.

## Notes

- First run downloads the Whisper `base` model (~150MB)
- No database, no accounts, no cloud services
- Run manually — no launchd service
- Port: 4039
