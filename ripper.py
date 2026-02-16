"""Core logic for YouTube audio/video extraction, transcription, and PDF generation."""

import queue
import re
import subprocess
import threading
from datetime import datetime
from pathlib import Path

import yt_dlp
from weasyprint import HTML

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

_whisper_model = None


def _get_whisper_model():
    """Lazy-load the faster-whisper model (singleton)."""
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel

        _whisper_model = WhisperModel("base", compute_type="int8", device="cpu")
    return _whisper_model


def _sanitize_filename(title: str) -> str:
    """Convert a video title into a safe filename."""
    name = re.sub(r'[<>:"/\\|?*]', "", title)
    name = re.sub(r"\s+", " ", name).strip()
    return name[:200] if name else "untitled"


def _format_bytes(num_bytes) -> str:
    """Format bytes into human-readable string."""
    if not num_bytes:
        return "?"
    for unit in ("B", "KB", "MB", "GB"):
        if abs(num_bytes) < 1024:
            return f"{num_bytes:.1f}{unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f}TB"


def _format_speed(bps) -> str:
    """Format bytes/sec into human-readable speed."""
    if not bps:
        return "?"
    return f"{_format_bytes(bps)}/s"


def _format_eta(seconds) -> str:
    """Format seconds into M:SS or H:MM:SS."""
    if seconds is None:
        return "?"
    seconds = int(seconds)
    if seconds < 3600:
        return f"{seconds // 60}:{seconds % 60:02d}"
    return f"{seconds // 3600}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}"


QUALITY_FORMATS = {
    "1080p": "bestvideo[height<=1080]+bestaudio/best[height<=1080]",
    "720p": "bestvideo[height<=720]+bestaudio/best[height<=720]",
    "480p": "bestvideo[height<=480]+bestaudio/best[height<=480]",
    "best": "bestvideo+bestaudio/best",
}


def download_video(
    url: str,
    quality: str = "1080p",
    progress_queue: queue.Queue | None = None,
) -> tuple[Path, str]:
    """Download video from a YouTube URL as MP4. Returns (filepath, title)."""
    info_opts = {"quiet": True, "no_warnings": True, "skip_download": True}
    with yt_dlp.YoutubeDL(info_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        title = info.get("title", "Untitled")
        video_id = info.get("id", "unknown")

    safe_name = _sanitize_filename(title)
    output_path = OUTPUT_DIR / f"{safe_name} [{video_id}].mp4"

    def progress_hook(d):
        if progress_queue is None:
            return
        status = d.get("status")
        if status == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
            downloaded = d.get("downloaded_bytes", 0)
            percent = (downloaded / total * 100) if total > 0 else 0
            progress_queue.put({
                "status": "downloading",
                "percent": round(percent, 1),
                "speed": _format_speed(d.get("speed")),
                "eta": _format_eta(d.get("eta")),
            })
        elif status == "finished":
            progress_queue.put({"status": "finished"})

    def postprocessor_hook(d):
        if progress_queue is None:
            return
        if d.get("status") == "started" and "Merger" in d.get("postprocessor", ""):
            progress_queue.put({"status": "merging"})

    fmt = QUALITY_FORMATS.get(quality, QUALITY_FORMATS["1080p"])
    dl_opts = {
        "format": fmt,
        "merge_output_format": "mp4",
        "outtmpl": str(OUTPUT_DIR / f"{safe_name} [{video_id}].%(ext)s"),
        "progress_hooks": [progress_hook],
        "postprocessor_hooks": [postprocessor_hook],
        "quiet": True,
        "no_warnings": True,
    }
    with yt_dlp.YoutubeDL(dl_opts) as ydl:
        ydl.download([url])

    return output_path, title


def download_audio(url: str) -> tuple[Path, str]:
    """Download audio from a YouTube URL as M4A. Returns (filepath, title)."""
    info_opts = {"quiet": True, "no_warnings": True, "skip_download": True}
    with yt_dlp.YoutubeDL(info_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        title = info.get("title", "Untitled")

    safe_name = _sanitize_filename(title)
    output_path = OUTPUT_DIR / f"{safe_name}.m4a"

    dl_opts = {
        "format": "bestaudio[ext=m4a]/bestaudio",
        "outtmpl": str(OUTPUT_DIR / f"{safe_name}.%(ext)s"),
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "m4a",
            }
        ],
        "quiet": True,
        "no_warnings": True,
    }
    with yt_dlp.YoutubeDL(dl_opts) as ydl:
        ydl.download([url])

    # Find the actual output file (extension may vary depending on source)
    if not output_path.exists():
        # Look for any audio file with the sanitized name
        candidates = list(OUTPUT_DIR.glob(f"{safe_name}.*"))
        audio_exts = {".m4a", ".webm", ".opus", ".mp3", ".ogg", ".wav"}
        for c in candidates:
            if c.suffix.lower() in audio_exts:
                output_path = c
                break

    return output_path, title


def transcribe_audio(filepath: Path):
    """Transcribe an audio file using faster-whisper. Returns list of segments."""
    model = _get_whisper_model()
    segments, _info = model.transcribe(str(filepath), beam_size=5, language="en")
    return list(segments)


def format_transcript(segments, title: str, url: str) -> str:
    """Format transcript segments into clean HTML for PDF rendering."""
    full_text = " ".join(seg.text.strip() for seg in segments if seg.text.strip())

    if not full_text:
        full_text = "(No speech detected in this video.)"

    # Split into sentences
    sentences = re.split(r"(?<=[.!?])\s+", full_text)

    # Group into paragraphs of ~4 sentences each
    paragraphs = []
    for i in range(0, len(sentences), 4):
        chunk = " ".join(sentences[i : i + 4])
        if chunk.strip():
            paragraphs.append(chunk.strip())

    date_str = datetime.now().strftime("%B %d, %Y")

    para_html = "\n".join(f"    <p>{p}</p>" for p in paragraphs)

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
    body {{
        font-family: Georgia, 'Times New Roman', serif;
        line-height: 1.6;
        max-width: 700px;
        margin: 40px auto;
        padding: 0 20px;
        color: #1a1a1a;
    }}
    h1 {{
        font-size: 1.6em;
        margin-bottom: 0.3em;
        line-height: 1.3;
    }}
    .meta {{
        color: #666;
        font-size: 0.85em;
        margin-bottom: 2em;
        border-bottom: 1px solid #ddd;
        padding-bottom: 1em;
    }}
    p {{
        margin-bottom: 1em;
        text-align: justify;
    }}
</style>
</head>
<body>
    <h1>{title}</h1>
    <p class="meta">Source: {url}<br>Transcribed: {date_str}</p>
{para_html}
</body>
</html>"""
    return html


def generate_pdf(html: str, title: str) -> Path:
    """Render HTML transcript to PDF. Returns filepath."""
    safe_name = _sanitize_filename(title)
    pdf_path = OUTPUT_DIR / f"{safe_name}.pdf"
    HTML(string=html).write_pdf(str(pdf_path))
    return pdf_path


def _run_video_download_with_progress(url: str, quality: str):
    """Run download_video in a thread, yielding SSE status messages for progress."""
    prog_queue = queue.Queue()
    result_holder = {"path": None, "title": None, "error": None}

    def _download():
        try:
            path, title = download_video(url, quality, progress_queue=prog_queue)
            result_holder["path"] = path
            result_holder["title"] = title
        except Exception as e:
            result_holder["error"] = str(e)
        finally:
            prog_queue.put(None)  # sentinel

    thread = threading.Thread(target=_download, daemon=True)
    thread.start()

    last_percent = -1
    while True:
        try:
            update = prog_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        if update is None:
            break

        status = update.get("status")
        if status == "downloading":
            percent = update["percent"]
            if percent - last_percent >= 2 or percent >= 99:
                last_percent = percent
                msg = f"Downloading video... {percent}%"
                if update["speed"] != "?":
                    msg += f" ({update['speed']}"
                    if update["eta"] != "?":
                        msg += f", ETA {update['eta']}"
                    msg += ")"
                yield ("status", msg)
        elif status == "merging":
            yield ("status", "Merging video and audio streams...")
        elif status == "finished":
            yield ("status", "Download complete, finalizing...")

    thread.join(timeout=10)

    if result_holder["error"]:
        yield ("error", f"Video download failed: {result_holder['error']}")
        return

    video_path = result_holder["path"]
    title = result_holder["title"]

    if video_path and video_path.exists():
        size = _format_bytes(video_path.stat().st_size)
        yield ("status", f"Video saved: {title} ({size})")

    yield ("_done", {"path": video_path, "title": title})


def _extract_audio_from_video(video_path: Path) -> Path:
    """Extract audio track from MP4 to M4A using ffmpeg."""
    audio_path = video_path.with_suffix(".temp.m4a")
    subprocess.run(
        ["ffmpeg", "-i", str(video_path), "-vn", "-acodec", "copy",
         str(audio_path), "-y"],
        capture_output=True,
        check=True,
    )
    return audio_path


def process(url: str, mode: str, quality: str = "1080p"):
    """Main orchestrator. Yields (type, data) tuples for SSE streaming.

    mode: 'audio', 'text', 'both', 'video', or 'video_text'
    """
    # --- Video modes ---
    if mode in ("video", "video_text"):
        yield ("status", "Fetching video info...")

        video_path = None
        title = None
        for msg_type, msg_data in _run_video_download_with_progress(url, quality):
            if msg_type == "_done":
                video_path = msg_data["path"]
                title = msg_data["title"]
            else:
                yield (msg_type, msg_data)
                if msg_type == "error":
                    return

        if video_path is None:
            yield ("error", "Video download produced no output.")
            return

        result = {"video": video_path.name}

        if mode == "video_text":
            yield ("status", "Extracting audio for transcription...")
            try:
                audio_path = _extract_audio_from_video(video_path)
            except Exception as e:
                yield ("error", f"Audio extraction failed: {e}")
                return

            yield ("status", "Transcribing with Whisper... (this may take a minute)")
            try:
                segments = transcribe_audio(audio_path)
            except Exception as e:
                yield ("error", f"Transcription failed: {e}")
                return

            yield ("status", "Generating PDF...")
            try:
                html = format_transcript(segments, title, url)
                pdf_path = generate_pdf(html, title)
                result["pdf"] = pdf_path.name
            except Exception as e:
                yield ("error", f"PDF generation failed: {e}")
                return

            # Clean up temp audio
            if audio_path.exists():
                audio_path.unlink()

        yield ("status", "Done! Your files are ready.")
        yield ("result", result)
        return

    # --- Audio / Text / Both modes (unchanged) ---
    yield ("status", "Fetching video info and downloading audio...")

    try:
        audio_path, title = download_audio(url)
    except Exception as e:
        yield ("error", f"Download failed: {e}")
        return

    yield ("status", f"Downloaded: {title}")

    result = {}

    if mode in ("audio", "both"):
        result["audio"] = audio_path.name

    if mode in ("text", "both"):
        yield ("status", "Transcribing with Whisper... (this may take a minute)")

        try:
            segments = transcribe_audio(audio_path)
        except Exception as e:
            yield ("error", f"Transcription failed: {e}")
            return

        yield ("status", "Generating PDF...")

        try:
            html = format_transcript(segments, title, url)
            pdf_path = generate_pdf(html, title)
            result["pdf"] = pdf_path.name
        except Exception as e:
            yield ("error", f"PDF generation failed: {e}")
            return

    # If text-only, clean up the audio file
    if mode == "text" and audio_path.exists():
        audio_path.unlink()
        yield ("status", "Audio file cleaned up (text-only mode).")

    yield ("status", "Done! Your files are ready.")
    yield ("result", result)
