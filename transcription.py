"""Local speech recognition and plain-text/subtitle transcript exports."""

import gc
import html
import logging
import math
import os
import platform
import re
import threading
from datetime import datetime
from pathlib import Path

from media_types import Transcript, TranscriptSegment, TranscriptionOptions


logger = logging.getLogger(__name__)

# One transcription runs at a time. No model is cached between jobs, so the
# memory a job uses is returned when the job ends.
_model_lock = threading.Lock()

MODEL_NAME = "turbo"
MLX_MODEL_REPO = "mlx-community/whisper-large-v3-turbo"


def resolve_backend() -> str:
    """Select the environment override or the platform default engine."""
    backend = os.environ.get("YOUTUBE_RIPPER_TRANSCRIPTION_BACKEND", "auto")
    if backend not in {"auto", "mlx", "faster-whisper"}:
        raise ValueError("Transcription backend must be auto, mlx, or faster-whisper")
    apple_silicon = platform.system() == "Darwin" and platform.machine() == "arm64"
    if backend == "auto":
        return "mlx" if apple_silicon else "faster-whisper"
    if backend == "mlx" and not apple_silicon:
        raise RuntimeError(
            "MLX requires an Apple Silicon Mac. "
            "Set YOUTUBE_RIPPER_TRANSCRIPTION_BACKEND=faster-whisper on this machine."
        )
    return backend


def _transcribe_mlx(filepath: Path, options: TranscriptionOptions) -> Transcript:
    """Run MLX Whisper Turbo, then unload the model and free its GPU memory."""
    try:
        import mlx.core as mx
        import mlx_whisper
        from mlx_whisper.transcribe import ModelHolder
    except (ImportError, OSError) as error:
        raise RuntimeError(
            "MLX Whisper could not load. Run 'uv sync --locked' on an Apple Silicon Mac, "
            "then retry, or set YOUTUBE_RIPPER_TRANSCRIPTION_BACKEND=faster-whisper."
        ) from error

    failure = None
    segments = language = None
    try:
        result = mlx_whisper.transcribe(
            str(filepath),
            path_or_hf_repo=MLX_MODEL_REPO,
            language=options.language,
            verbose=None,
        )
        segments = tuple(
            TranscriptSegment(start=float(segment["start"]), end=float(segment["end"]), text=segment["text"].strip())
            for segment in result["segments"]
        )
        language = result.get("language")
        del result
    except Exception as error:
        # Log the stack here. The error raised to the caller is unchained.
        logger.exception("MLX transcription failed")
        failure = RuntimeError(
            f"MLX transcription failed: {error}. Check that ffmpeg is available and "
            "the model can download from Hugging Face, or set "
            "YOUTUBE_RIPPER_TRANSCRIPTION_BACKEND=faster-whisper."
        )

    # Release here, after the except block has ended, and not in a finally.
    # While an exception is propagating, its traceback frames still reference
    # the model, so a release at that point frees nothing. The failure is kept
    # as a new, unchained error for the same reason.
    try:
        ModelHolder.model = None
        ModelHolder.model_path = None
        gc.collect()
        mx.clear_cache()
    except Exception:
        logger.exception("Could not release MLX model memory after transcription")

    if failure is not None:
        raise failure
    return Transcript(segments=segments, language=language, source="Local Whisper (MLX)")


def _transcribe_faster_whisper(filepath: Path, options: TranscriptionOptions) -> Transcript:
    """Run faster-whisper with a model that lives only for this call."""
    from faster_whisper import WhisperModel

    model = WhisperModel(
        MODEL_NAME,
        device=os.environ.get("YOUTUBE_RIPPER_WHISPER_DEVICE", "cpu"),
        compute_type=os.environ.get("YOUTUBE_RIPPER_WHISPER_COMPUTE_TYPE", "int8"),
    )
    segments, info = model.transcribe(
        str(filepath), beam_size=5, language=options.language, vad_filter=True
    )
    # Inference happens while iterating the lazy segments, so finish it here.
    timed_segments = tuple(
        TranscriptSegment(start=float(seg.start), end=float(seg.end), text=seg.text.strip())
        for seg in segments
    )
    return Transcript(segments=timed_segments, language=info.language, source="Local Whisper")


def transcribe(filepath: Path, options: TranscriptionOptions) -> Transcript:
    """Transcribe locally and retain segment times and the detected language."""
    backend = resolve_backend()
    with _model_lock:
        if backend == "mlx":
            return _transcribe_mlx(filepath, options)
        return _transcribe_faster_whisper(filepath, options)


def _plain_text(transcript: Transcript, title: str, url: str) -> str:
    text = " ".join(" ".join(seg.text.split()) for seg in transcript.segments if seg.text.strip())
    if not text:
        text = "(No speech detected in this video.)"
    sentences = re.split(r"(?<=[.!?])\s+", text)
    body = "\n\n".join(" ".join(sentences[i : i + 4]) for i in range(0, len(sentences), 4))
    date = datetime.now().strftime("%B %d, %Y")
    return (
        f"{title}\n\nSource: {url}\nTranscript source: {transcript.source}\n"
        f"Language: {transcript.language or 'unknown'}\nTranscribed: {date}\n\n{body}\n"
    )


def _milliseconds(seconds: float) -> int:
    if not math.isfinite(seconds):
        raise ValueError("Subtitle timestamps must be finite numbers")
    return max(0, math.floor(seconds * 1000 + 0.5))


def _timestamp(milliseconds: int, separator: str) -> str:
    seconds, fraction = divmod(milliseconds, 1000)
    minutes, second = divmod(seconds, 60)
    hours, minute = divmod(minutes, 60)
    return f"{hours:02d}:{minute:02d}:{second:02d}{separator}{fraction:03d}"


def _subtitle_text(text: str) -> str:
    # Blank lines terminate cues. Keep meaningful line breaks while removing
    # blank lines and control bytes, and render markup as literal speech.
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    return html.escape("\n".join(lines), quote=False)


def _subtitles(transcript: Transcript, format: str) -> str:
    cues = []
    separator = "," if format == "srt" else "."
    for segment in transcript.segments:
        text = _subtitle_text(segment.text)
        if not text:
            continue
        start = _milliseconds(segment.start)
        end = max(start + 1, _milliseconds(segment.end))
        timing = f"{_timestamp(start, separator)} --> {_timestamp(end, separator)}"
        prefix = f"{len(cues) + 1}\n" if format == "srt" else ""
        cues.append(f"{prefix}{timing}\n{text}\n\n")
    return ("WEBVTT\n\n" if format == "vtt" else "") + "".join(cues)


def write_transcript_files(
    transcript: Transcript,
    *,
    title: str,
    url: str,
    stem: str,
    output_dir: Path,
    subtitle_formats: tuple[str, ...] = (),
) -> dict[str, str]:
    """Write TXT and requested subtitle files, returning downloadable filenames."""
    if not stem or stem in {".", ".."} or any(char in stem for char in ("/", "\\", "\x00")):
        raise ValueError("Transcript filename stem must be a single filename")
    if any(format not in {"srt", "vtt"} for format in subtitle_formats):
        raise ValueError("Subtitle formats must be srt or vtt")

    contents = {"text": ("txt", _plain_text(transcript, title, url))}
    for format in subtitle_formats:
        contents[format] = (format, _subtitles(transcript, format))

    output_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    for kind, (extension, content) in contents.items():
        path = output_dir / f"{stem}.{extension}"
        path.write_text(content, encoding="utf-8")
        files[kind] = path.name
    return files
