"""Core logic for YouTube audio/video extraction, transcription, and text generation."""

import os
import queue
import re
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import yt_dlp

DEFAULT_OUTPUT_DIR = Path(__file__).parent / "output"


def _configured_output_dir() -> Path:
    configured = os.environ.get("YOUTUBE_RIPPER_OUTPUT_DIR")
    if configured:
        return Path(configured).expanduser()
    return DEFAULT_OUTPUT_DIR


def _whisper_model_name() -> str:
    return os.environ.get("YOUTUBE_RIPPER_WHISPER_MODEL", "base")


def _whisper_device() -> str:
    return os.environ.get("YOUTUBE_RIPPER_WHISPER_DEVICE", "cpu")


def _whisper_compute_type() -> str:
    return os.environ.get("YOUTUBE_RIPPER_WHISPER_COMPUTE_TYPE", "int8")


OUTPUT_DIR = _configured_output_dir()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_COOKIES_FROM_BROWSER = "chrome:Profile 1"


@dataclass(frozen=True)
class MediaInfo:
    title: str
    video_id: str


@dataclass(frozen=True)
class DownloadedMedia:
    path: Path
    info: MediaInfo


_whisper_model = None


def _get_whisper_model():
    """Lazy-load the faster-whisper model (singleton)."""
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel

        _whisper_model = WhisperModel(
            _whisper_model_name(),
            compute_type=_whisper_compute_type(),
            device=_whisper_device(),
        )
    return _whisper_model


def _sanitize_filename(title: str) -> str:
    """Convert a video title into a safe filename."""
    name = re.sub(r'[<>:"/\\|?*]', "", title)
    name = re.sub(r"\s+", " ", name).strip()
    return name[:200] if name else "untitled"


def _output_stem(info: MediaInfo) -> str:
    """Return the shared output filename stem for a YouTube item."""
    return f"{_sanitize_filename(info.title)} [{info.video_id}]"


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


def _parse_cookies_from_browser(value: str) -> tuple[str, str | None, str | None, str | None] | None:
    value = value.strip()
    if not value or value.lower() in {"0", "false", "none", "off"}:
        return None

    browser, _, profile = value.partition(":")
    browser = browser.strip()
    profile = profile.strip() or None
    return (browser, profile, None, None)


def _yt_dlp_opts(extra_opts: dict | None = None) -> dict:
    """Build common yt-dlp options, including YouTube auth cookies."""
    opts = dict(extra_opts or {})

    cookie_file = (
        os.environ.get("YOUTUBE_RIPPER_COOKIES")
        or os.environ.get("YT_DLP_COOKIES")
    )
    if cookie_file:
        opts["cookiefile"] = cookie_file
        return opts

    cookies_from_browser = (
        os.environ.get("YOUTUBE_RIPPER_COOKIES_FROM_BROWSER")
        or os.environ.get("YT_DLP_COOKIES_FROM_BROWSER")
        or DEFAULT_COOKIES_FROM_BROWSER
    )
    parsed = _parse_cookies_from_browser(cookies_from_browser)
    if parsed:
        opts["cookiesfrombrowser"] = parsed

    return opts


def _fetch_media_info(url: str) -> MediaInfo:
    """Fetch title and id without downloading media."""
    info_opts = _yt_dlp_opts({"quiet": True, "no_warnings": True, "skip_download": True})
    with yt_dlp.YoutubeDL(info_opts) as ydl:
        info = ydl.extract_info(url, download=False)

    return MediaInfo(
        title=info.get("title", "Untitled"),
        video_id=info.get("id", "unknown"),
    )


def download_video(
    url: str,
    quality: str = "1080p",
    progress_queue: queue.Queue | None = None,
) -> DownloadedMedia:
    """Download video from a YouTube URL as MP4."""
    info = _fetch_media_info(url)
    output_stem = _output_stem(info)
    output_path = OUTPUT_DIR / f"{output_stem}.mp4"

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
    dl_opts = _yt_dlp_opts({
        "format": fmt,
        "merge_output_format": "mp4",
        "outtmpl": str(OUTPUT_DIR / f"{output_stem}.%(ext)s"),
        "progress_hooks": [progress_hook],
        "postprocessor_hooks": [postprocessor_hook],
        "quiet": True,
        "no_warnings": True,
    })
    with yt_dlp.YoutubeDL(dl_opts) as ydl:
        ydl.download([url])

    return DownloadedMedia(path=output_path, info=info)


def download_audio(url: str) -> DownloadedMedia:
    """Download audio from a YouTube URL as M4A."""
    info = _fetch_media_info(url)
    output_stem = _output_stem(info)
    output_path = OUTPUT_DIR / f"{output_stem}.m4a"

    dl_opts = _yt_dlp_opts({
        "format": "bestaudio[ext=m4a]/bestaudio",
        "outtmpl": str(OUTPUT_DIR / f"{output_stem}.%(ext)s"),
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "m4a",
            }
        ],
        "quiet": True,
        "no_warnings": True,
    })
    with yt_dlp.YoutubeDL(dl_opts) as ydl:
        ydl.download([url])

    # Find the actual output file (extension may vary depending on source)
    if not output_path.exists():
        # Look for any audio file with the sanitized name
        candidates = list(OUTPUT_DIR.glob(f"{output_stem}.*"))
        audio_exts = {".m4a", ".webm", ".opus", ".mp3", ".ogg", ".wav"}
        for candidate in candidates:
            if candidate.suffix.lower() in audio_exts:
                output_path = candidate
                break

    return DownloadedMedia(path=output_path, info=info)


def transcribe_audio(filepath: Path):
    """Transcribe an audio file using faster-whisper. Returns list of segments."""
    model = _get_whisper_model()
    segments, _info = model.transcribe(str(filepath), beam_size=5, language="en")
    return list(segments)


def _run_transcription_with_keepalive(filepath: Path):
    """Run transcription in a thread, yielding keepalive SSE messages."""
    result_holder = {"segments": None, "error": None}
    done_event = threading.Event()

    def _transcribe():
        try:
            result_holder["segments"] = transcribe_audio(filepath)
        except Exception as e:
            result_holder["error"] = str(e)
        finally:
            done_event.set()

    thread = threading.Thread(target=_transcribe, daemon=True)
    thread.start()

    while not done_event.wait(timeout=3):
        yield ("status", "Still transcribing...")

    thread.join(timeout=5)

    if result_holder["error"]:
        yield ("error", f"Transcription failed: {result_holder['error']}")
        return

    yield ("_done", result_holder["segments"])


def format_transcript(segments, title: str, url: str) -> str:
    """Format transcript segments into a readable plain-text transcript."""
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

    body = "\n\n".join(paragraphs)
    return f"{title}\n\nSource: {url}\nTranscribed: {date_str}\n\n{body}\n"


def generate_text(transcript: str, info: MediaInfo) -> Path:
    """Write a plain-text transcript."""
    text_path = OUTPUT_DIR / f"{_output_stem(info)}.txt"
    text_path.write_text(transcript, encoding="utf-8")
    return text_path


def _run_video_download_with_progress(url: str, quality: str):
    """Run download_video in a thread, yielding SSE status messages for progress."""
    prog_queue = queue.Queue()
    result_holder = {"media": None, "error": None}

    def _download():
        try:
            result_holder["media"] = download_video(url, quality, progress_queue=prog_queue)
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

    media = result_holder["media"]

    if media and media.path.exists():
        size = _format_bytes(media.path.stat().st_size)
        yield ("status", f"Video saved: {media.info.title} ({size})")

    yield ("_done", media)



def process(url: str, mode: str, quality: str = "1080p"):
    """Main orchestrator. Yields (type, data) tuples for SSE streaming.

    mode: 'audio', 'text', 'both', 'video', or 'video_text'
    """
    # --- Video only mode ---
    if mode == "video":
        yield ("status", "Fetching video info...")

        downloaded_video = None
        for msg_type, msg_data in _run_video_download_with_progress(url, quality):
            if msg_type == "_done":
                downloaded_video = msg_data
            else:
                yield (msg_type, msg_data)
                if msg_type == "error":
                    return

        if downloaded_video is None:
            yield ("error", "Video download produced no output.")
            return

        yield ("status", "Done! Your files are ready.")
        yield ("result", {"video": downloaded_video.path.name})
        return

    # --- Video + Text mode ---
    if mode == "video_text":
        # Step 1: Download audio separately (fast, reuses existing download_audio)
        yield ("status", "Downloading audio for transcription...")
        try:
            downloaded_audio = download_audio(url)
            audio_path = downloaded_audio.path
            info = downloaded_audio.info
        except Exception as e:
            yield ("error", f"Audio download failed: {e}")
            return

        # Step 2: Transcribe (threaded with keepalive to prevent SSE timeout)
        yield ("status", "Transcribing with Whisper... (this may take a minute)")
        segments = None
        for msg_type, msg_data in _run_transcription_with_keepalive(audio_path):
            if msg_type == "_done":
                segments = msg_data
            else:
                yield (msg_type, msg_data)
                if msg_type == "error":
                    if audio_path.exists():
                        audio_path.unlink()
                    return

        # Step 3: Generate text transcript
        yield ("status", "Generating text file...")
        try:
            transcript = format_transcript(segments, info.title, url)
            text_path = generate_text(transcript, info)
        except Exception as e:
            yield ("error", f"Text generation failed: {e}")
            if audio_path.exists():
                audio_path.unlink()
            return

        # Step 4: Download video (slow, with progress bar)
        yield ("status", "Fetching video info...")
        downloaded_video = None
        for msg_type, msg_data in _run_video_download_with_progress(url, quality):
            if msg_type == "_done":
                downloaded_video = msg_data
            else:
                yield (msg_type, msg_data)
                if msg_type == "error":
                    if audio_path.exists():
                        audio_path.unlink()
                    return

        if downloaded_video is None:
            yield ("error", "Video download produced no output.")
            if audio_path.exists():
                audio_path.unlink()
            return

        # Step 5: Clean up audio file (user gets video + text)
        if audio_path.exists():
            audio_path.unlink()

        yield ("status", "Done! Your files are ready.")
        yield ("result", {"video": downloaded_video.path.name, "text": text_path.name})
        return

    # --- Audio / Text / Both modes (unchanged) ---
    yield ("status", "Fetching video info and downloading audio...")

    try:
        downloaded_audio = download_audio(url)
        audio_path = downloaded_audio.path
        info = downloaded_audio.info
    except Exception as e:
        yield ("error", f"Download failed: {e}")
        return

    yield ("status", f"Downloaded: {info.title}")

    result = {}

    if mode in ("audio", "both"):
        result["audio"] = audio_path.name

    if mode in ("text", "both"):
        yield ("status", "Transcribing with Whisper... (this may take a minute)")

        segments = None
        for msg_type, msg_data in _run_transcription_with_keepalive(audio_path):
            if msg_type == "_done":
                segments = msg_data
            else:
                yield (msg_type, msg_data)
                if msg_type == "error":
                    return

        yield ("status", "Generating text file...")

        try:
            transcript = format_transcript(segments, info.title, url)
            text_path = generate_text(transcript, info)
            result["text"] = text_path.name
        except Exception as e:
            yield ("error", f"Text generation failed: {e}")
            return

    # If text-only, clean up the audio file
    if mode == "text" and audio_path.exists():
        audio_path.unlink()
        yield ("status", "Audio file cleaned up (text-only mode).")

    yield ("status", "Done! Your files are ready.")
    yield ("result", result)
