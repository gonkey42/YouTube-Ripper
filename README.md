# YouTube Ripper

Local web utility that extracts audio and video from YouTube, transcribes with Whisper, and generates clean formatted PDFs.

## Features

- **Five output modes:** Audio Only, Text Only, Audio + Text, Video, Video + Text
- **Video quality selection:** 480p, 720p, 1080p, or Best Available
- **Audio format:** M4A (AAC) — plays on Mac, Android, iOS
- **Video format:** MP4 with merged audio
- **Transcription:** Local Whisper (faster-whisper) — no cloud APIs, no data leaves your machine
- **PDF output:** Clean document with title, source URL, paragraph breaks
- **Real-time progress** via Server-Sent Events (download %, speed, ETA for video)
- **Supported URLs:** `youtube.com/watch`, `youtu.be`, `youtube.com/shorts`, `youtube.com/live`

## Output Modes

| Mode | What you get | Quality picker? |
|------|-------------|-----------------|
| Audio Only | M4A | No |
| Text Only | PDF transcript | No |
| Audio + Text | M4A + PDF | No |
| Video | MP4 | Yes |
| Video + Text | MP4 + PDF | Yes |

**Video + Text** downloads the video, extracts the audio track, runs Whisper on it, generates the PDF, and cleans up the intermediate audio file.

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

Files are saved to `output/` (gitignored). The UI shows filenames of saved files — grab them from the output folder. In "Text Only" mode, the audio file is automatically deleted after transcription. In "Video + Text" mode, the temporary audio extraction is cleaned up after transcription.

## Notes

- First run downloads the Whisper `base` model (~150MB)
- No database, no accounts, no cloud services
- Port: 4039
