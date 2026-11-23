"""Core logic for YouTube audio/video extraction, transcription, and text generation."""

import os
import queue
import re
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path

import yt_dlp
from yt_dlp.utils import DownloadError

from captions import fetch_captions
from diagnostics import dependency_issues
from media_types import Transcript, TranscriptionOptions
import transcription

DEFAULT_OUTPUT_DIR = Path(__file__).parent / "output"
HOMEBREW_NODE_PATH = Path("/opt/homebrew/opt/node/bin/node")


def _configured_output_dir() -> Path:
    configured = os.environ.get("YOUTUBE_RIPPER_OUTPUT_DIR")
    if configured:
        return Path(configured).expanduser()
    return DEFAULT_OUTPUT_DIR


def _whisper_model_name() -> str:
    return os.environ.get("YOUTUBE_RIPPER_WHISPER_MODEL", "turbo")


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


def _yt_dlp_opts(extra_opts: dict | None = None, *, use_cookies: bool = True) -> dict:
    """Build common yt-dlp options, including YouTube auth cookies."""
    opts = dict(extra_opts or {})
    node_config = {"path": str(HOMEBREW_NODE_PATH)} if os.access(HOMEBREW_NODE_PATH, os.X_OK) else {}
    opts.setdefault("js_runtimes", {"node": node_config})
    opts.setdefault("remote_components", ["ejs:github"])

    if not use_cookies:
        return opts

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


def _has_cookie_options(opts: dict) -> bool:
    return "cookiefile" in opts or "cookiesfrombrowser" in opts


def _run_ytdlp_with_cookie_fallback(base_opts: dict, action):
    opts = _yt_dlp_opts(base_opts)
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            return action(ydl)
    except DownloadError:
        if not _has_cookie_options(opts):
            raise
        retry_opts = _yt_dlp_opts(base_opts, use_cookies=False)
        with yt_dlp.YoutubeDL(retry_opts) as ydl:
            return action(ydl)


def _fetch_media_info(url: str) -> MediaInfo:
    """Fetch title and id without downloading media."""
    base_opts = {"quiet": True, "no_warnings": True, "skip_download": True}
    info = _run_ytdlp_with_cookie_fallback(
        base_opts,
        lambda ydl: ydl.extract_info(url, download=False),
    )

    return MediaInfo(
        title=info.get("title", "Untitled"),
        video_id=info.get("id", "unknown"),
    )


def _validate_downloaded_path(path: Path) -> Path:
    """Require a usable file before announcing a completed download."""
    try:
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(
                f"Download did not produce a nonempty media file: {path.name}. "
                "Retry the download and check that ffmpeg is installed and the output directory is writable."
            )
    except OSError as exc:
        raise RuntimeError(f"Cannot access downloaded file {path.name}: {exc}") from exc
    return path


def _download_media(url: str, info: MediaInfo, dl_opts: dict, progress_queue=None) -> DownloadedMedia:
    """Track this download's final path, including extension changes by ffmpeg."""
    final_paths = []

    def progress_hook(update):
        if progress_queue is None:
            return
        status = update.get("status")
        if status == "downloading":
            total = update.get("total_bytes") or update.get("total_bytes_estimate") or 0
            downloaded = update.get("downloaded_bytes", 0)
            progress_queue.put({
                "status": "downloading",
                "phase": update.get("filename") or update.get("info_dict", {}).get("format_id"),
                "percent": round(min(100, downloaded / total * 100), 1) if total > 0 else None,
                "speed": _format_speed(update.get("speed")),
                "eta": _format_eta(update.get("eta")),
            })
        elif status == "finished":
            progress_queue.put({"status": "finished"})

    def postprocessor_hook(update):
        processor = update.get("postprocessor", "")
        # MoveFiles runs after conversion/merging and supplies the final filepath.
        if update.get("status") == "finished" and processor == "MoveFiles":
            filepath = update.get("info_dict", {}).get("filepath")
            if filepath:
                final_paths.append(filepath)
        if progress_queue is not None and update.get("status") == "started":
            progress_queue.put({"status": "merging" if "Merger" in processor else "processing"})

    opts = dict(dl_opts, progress_hooks=[progress_hook], postprocessor_hooks=[postprocessor_hook])

    def download(ydl):
        # A cookie-free retry must not inherit an earlier attempt's completed path.
        final_paths.clear()
        if progress_queue is not None:
            progress_queue.put({"status": "starting"})
        return ydl.extract_info(url, download=True)

    result = _run_ytdlp_with_cookie_fallback(opts, download)
    reported_paths = []
    if isinstance(result, dict):
        reported_paths = [
            item["filepath"] for item in result.get("requested_downloads", [])
            if isinstance(item, dict) and item.get("filepath")
        ]
        if not reported_paths and result.get("filepath"):
            reported_paths.append(result["filepath"])
    paths = reported_paths or final_paths
    if not paths:
        raise RuntimeError(
            "yt-dlp reported no completed media file. Retry the download; "
            "if this continues, update yt-dlp and check ffmpeg."
        )
    output_path = _validate_downloaded_path(Path(paths[-1]))
    return DownloadedMedia(path=output_path, info=info)


def download_video(
    url: str,
    quality: str = "1080p",
    progress_queue: queue.Queue | None = None,
) -> DownloadedMedia:
    """Download video, preferring MP4 and preserving the actual final extension."""
    info = _fetch_media_info(url)
    dl_opts = {
        "format": QUALITY_FORMATS.get(quality, QUALITY_FORMATS["1080p"]),
        "merge_output_format": "mp4",
        "outtmpl": str(OUTPUT_DIR / f"{_output_stem(info)}.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
    }
    return _download_media(url, info, dl_opts, progress_queue)


def extract_audio_from_video(video_path: Path) -> Path:
    """Extract temporary audio from a downloaded video for transcription."""
    audio_path = video_path.with_name(f"{video_path.stem}.transcription.m4a")

    def _remove_partial_audio():
        if audio_path.exists():
            audio_path.unlink()

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-acodec",
        "aac",
        "-b:a",
        "128k",
        str(audio_path),
    ]

    try:
        subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except FileNotFoundError as exc:
        _remove_partial_audio()
        raise RuntimeError("ffmpeg is required to extract audio from downloaded video.") from exc
    except subprocess.CalledProcessError as exc:
        _remove_partial_audio()
        message = (exc.stderr or exc.stdout or str(exc)).strip()
        raise RuntimeError(f"ffmpeg audio extraction failed: {message}") from exc

    return audio_path


def download_audio(url: str, progress_queue: queue.Queue | None = None) -> DownloadedMedia:
    """Download audio as M4A and verify the path reported after conversion."""
    info = _fetch_media_info(url)
    dl_opts = {
        "format": "bestaudio[ext=m4a]/bestaudio",
        "outtmpl": str(OUTPUT_DIR / f"{_output_stem(info)}.%(ext)s"),
        "postprocessors": [
            {"key": "FFmpegExtractAudio", "preferredcodec": "m4a"}
        ],
        "quiet": True,
        "no_warnings": True,
    }
    return _download_media(url, info, dl_opts, progress_queue)


def _run_with_keepalive(action, status: str, error_prefix: str):
    """Keep the current response alive during a blocking local or network step."""
    result = {"value": None, "error": None}
    done = threading.Event()

    def run():
        try:
            result["value"] = action()
        except Exception as exc:
            result["error"] = str(exc)
        finally:
            done.set()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    while not done.wait(timeout=3):
        yield ("status", status)
    thread.join()
    if result["error"] is not None:
        yield ("error", f"{error_prefix}: {result['error']}")
    else:
        yield ("_done", result["value"])


def _run_transcription_with_keepalive(filepath: Path, options: TranscriptionOptions):
    yield from _run_with_keepalive(
        lambda: transcription.transcribe(filepath, options),
        "Still transcribing...", "Transcription failed",
    )


def _write_transcript(transcript: Transcript, info: MediaInfo, url: str, options: TranscriptionOptions):
    return transcription.write_transcript_files(
        transcript, title=info.title, url=url, stem=_output_stem(info),
        output_dir=OUTPUT_DIR, subtitle_formats=options.subtitle_formats,
    )


def _transcribe_to_text_file(
    audio_path: Path, info: MediaInfo, url: str, options: TranscriptionOptions | None = None,
):
    options = options or TranscriptionOptions()
    try:
        backend = transcription.resolve_backend(options)
    except Exception as exc:
        yield ("error", f"Transcription setup failed: {exc}")
        return
    engine = "MLX on the Apple GPU" if backend == "mlx" else "faster-whisper"
    yield ("status", f"Transcribing with {engine}... The first use of a model includes its download.")
    transcript = None
    for msg_type, msg_data in _run_transcription_with_keepalive(audio_path, options):
        if msg_type == "_done":
            transcript = msg_data
        else:
            yield (msg_type, msg_data)
            if msg_type == "error":
                return
    if transcript is None:
        yield ("error", "Transcription produced no result.")
        return
    yield ("status", f"Transcript source: {transcript.source}. Language: {transcript.language or 'unknown'}.")
    yield ("status", "Generating transcript files...")
    try:
        files = _write_transcript(transcript, info, url, options)
    except Exception as exc:
        yield ("error", f"Text generation failed: {exc}")
        return
    yield ("_done", files)


def _run_download_with_progress(download, media_kind: str):
    """Yield stream progress, idle keepalives, and only verified completed media."""
    prog_queue = queue.Queue()
    result_holder = {"media": None, "error": None}

    def run_download():
        try:
            media = download(prog_queue)
            if media is None:
                raise RuntimeError("yt-dlp returned no media file. Retry the download.")
            _validate_downloaded_path(media.path)
            result_holder["media"] = media
        except Exception as exc:
            result_holder["error"] = str(exc)
        finally:
            prog_queue.put(None)

    thread = threading.Thread(target=run_download, daemon=True)
    thread.start()
    last_percent = -2
    last_phase = None
    idle_message = f"Still downloading {media_kind}..."
    while True:
        try:
            update = prog_queue.get(timeout=10)
        except queue.Empty:
            yield ("status", idle_message)
            continue
        if update is None:
            break
        status = update.get("status")
        if status == "starting":
            last_percent = -2
            last_phase = None
            idle_message = f"Still downloading {media_kind}..."
        elif status == "downloading":
            percent = update["percent"]
            phase = update.get("phase")
            phase_changed = phase != last_phase
            if phase_changed or (percent is not None and percent < last_percent):
                last_percent = -2
            last_phase = phase
            idle_message = f"Still downloading {media_kind}..."
            if percent is None or percent - last_percent >= 2 or percent >= 99:
                last_percent = percent if percent is not None else -2
                msg = f"Downloading {media_kind}..."
                if percent is not None:
                    msg += f" {percent}%"
                if update["speed"] != "?":
                    msg += f" ({update['speed']}"
                    if update["eta"] != "?":
                        msg += f", ETA {update['eta']}"
                    msg += ")"
                yield ("status", msg)
        elif status == "merging":
            idle_message = "Still merging video and audio streams..."
            yield ("status", "Merging video and audio streams...")
        elif status == "processing":
            idle_message = f"Still processing {media_kind} with ffmpeg..."
            yield ("status", f"Processing {media_kind} with ffmpeg...")
        elif status == "finished":
            last_percent = -2
            idle_message = "Still finalizing downloaded media..."
            yield ("status", "Stream downloaded, finalizing...")

    thread.join()
    if result_holder["error"]:
        yield ("error", f"{media_kind.capitalize()} download failed: {result_holder['error']}")
        return
    media = result_holder["media"]
    size = _format_bytes(media.path.stat().st_size)
    yield ("status", f"{media_kind.capitalize()} saved: {media.info.title} ({size})")
    yield ("_done", media)


def _run_video_download_with_progress(url: str, quality: str):
    """Run download_video in a thread, yielding SSE status messages for progress."""
    yield from _run_download_with_progress(
        lambda progress_queue: download_video(url, quality, progress_queue=progress_queue), "video"
    )


def _run_audio_download_with_progress(url: str):
    """Run download_audio in a thread, yielding SSE status messages for progress."""
    yield from _run_download_with_progress(
        lambda progress_queue: download_audio(url, progress_queue=progress_queue), "audio"
    )


def process(
    url: str, mode: str, quality: str = "1080p", options: TranscriptionOptions | None = None,
):
    """Download requested media and produce transcripts within this request."""
    options = options or TranscriptionOptions()
    needs_text = mode in ("text", "both", "video_text")
    is_video = mode in ("video", "video_text")
    captions = None
    result = {}

    if needs_text and options.prefer_captions:
        yield ("status", "Checking YouTube captions...")
        for msg_type, msg_data in _run_with_keepalive(
            lambda: fetch_captions(url, options.language, _run_ytdlp_with_cookie_fallback),
            "Still checking YouTube captions...", "Caption lookup failed",
        ):
            if msg_type == "_done":
                captions = msg_data
            elif msg_type == "status":
                yield (msg_type, msg_data)
        if captions:
            metadata, transcript = captions
            info = MediaInfo(title=metadata.get("title", "Untitled"), video_id=metadata.get("id", "unknown"))
            yield ("status", f"Transcript source: {transcript.source}. Language: {transcript.language or 'unknown'}.")
            try:
                result.update(_write_transcript(transcript, info, url, options))
            except Exception as exc:
                yield ("error", f"Text generation failed: {exc}")
                return
            if mode == "text":
                yield ("status", "Done! Captions saved without downloading audio or running Whisper.")
                yield ("result", result)
                return
        else:
            yield ("status", "No usable captions in the requested language. Transcribing locally with your selected engine.")

    # Captions-only requests can succeed without ffmpeg. Check it here too so
    # falling back to a media download gives a useful setup error.
    issues = dependency_issues()
    if issues:
        yield ("error", "\n".join(issues))
        return
    yield ("status", "Fetching video info and downloading media...")
    download = _run_video_download_with_progress(url, quality) if is_video else _run_audio_download_with_progress(url)
    media = None
    for msg_type, msg_data in download:
        if msg_type == "_done":
            media = msg_data
        else:
            yield (msg_type, msg_data)
            if msg_type == "error":
                return
    if media is None:
        yield ("error", "Download produced no media file.")
        return
    if mode != "text":
        result["video" if is_video else "audio"] = media.path.name

    if needs_text and not captions:
        # Both engines decode the audio stream of a video directly. Avoid the
        # previous extra AAC conversion and temporary audio file.
        files = None
        for msg_type, msg_data in _transcribe_to_text_file(media.path, media.info, url, options):
            if msg_type == "_done":
                files = msg_data
            else:
                yield (msg_type, msg_data)
                if msg_type == "error":
                    return
        if files is None:
            yield ("error", "Text generation produced no output.")
            return
        result.update(files)

    if mode == "text" and media.path.exists():
        media.path.unlink()
        yield ("status", "Audio file cleaned up (text-only mode).")
    yield ("status", "Done! Your files are ready.")
    yield ("result", result)
