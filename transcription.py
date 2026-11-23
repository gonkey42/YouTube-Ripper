"""Local speech recognition and plain-text/subtitle transcript exports."""

import html
import math
import os
import platform
import re
import threading
from datetime import datetime
from pathlib import Path

from media_types import Transcript, TranscriptSegment, TranscriptionOptions


# Both backends share one lock and release their cached model when switching.
# Faster-whisper inference happens while iterating its lazy segment results.
_model_lock = threading.Lock()
_model = None
_model_key: tuple[str, str, str] | None = None
_mlx_model_holder = None

MLX_MODELS = {
    "tiny": "mlx-community/whisper-tiny-mlx",
    "base": "mlx-community/whisper-base-mlx",
    "small": "mlx-community/whisper-small-mlx",
    "medium": "mlx-community/whisper-medium-mlx",
    "large-v3": "mlx-community/whisper-large-v3-mlx",
    "turbo": "mlx-community/whisper-large-v3-turbo",
}


def resolve_backend(options: TranscriptionOptions) -> str:
    """Select the explicit backend, environment override, or platform default."""
    backend = options.backend
    if backend == "auto":
        backend = os.environ.get("YOUTUBE_RIPPER_TRANSCRIPTION_BACKEND", "auto")
    if backend not in {"auto", "mlx", "faster-whisper"}:
        raise ValueError("Transcription backend must be auto, mlx, or faster-whisper")
    apple_silicon = platform.system() == "Darwin" and platform.machine() == "arm64"
    if backend == "auto":
        return "mlx" if apple_silicon else "faster-whisper"
    if backend == "mlx" and not apple_silicon:
        raise RuntimeError("MLX requires an Apple Silicon Mac. Select the faster-whisper backend on this machine.")
    return backend


def _transcribe_mlx(filepath: Path, options: TranscriptionOptions, model_name: str) -> Transcript:
    """Use upstream's single-model cache while holding our shared model lock."""
    global _model, _model_key, _mlx_model_holder
    if model_name not in MLX_MODELS:
        raise ValueError(f"MLX model must be one of: {', '.join(MLX_MODELS)}")
    try:
        import mlx_whisper
        from mlx_whisper.transcribe import ModelHolder
    except (ImportError, OSError) as error:
        raise RuntimeError(
            "MLX Whisper could not load. Run 'uv sync --locked' on an Apple Silicon Mac, "
            "then retry, or select the faster-whisper backend."
        ) from error

    repo = MLX_MODELS[model_name]
    _model = None
    _model_key = None
    _mlx_model_holder = ModelHolder
    if ModelHolder.model_path != repo:
        ModelHolder.model = None
        ModelHolder.model_path = None

    try:
        result = mlx_whisper.transcribe(
            str(filepath),
            path_or_hf_repo=repo,
            language=options.language,
            verbose=None,
        )
    except Exception as error:
        raise RuntimeError(
            f"MLX transcription failed: {error}. Check that ffmpeg is available and "
            "the selected model can download from Hugging Face, or select the faster-whisper backend."
        ) from error
    segments = tuple(
        TranscriptSegment(start=float(segment["start"]), end=float(segment["end"]), text=segment["text"].strip())
        for segment in result["segments"]
    )
    return Transcript(segments=segments, language=result.get("language"), source="Local Whisper (MLX)")


def transcribe(filepath: Path, options: TranscriptionOptions) -> Transcript:
    """Transcribe locally and retain segment times and the detected language."""
    global _model, _model_key

    model_name = options.model or os.environ.get("YOUTUBE_RIPPER_WHISPER_MODEL", "turbo")
    backend = resolve_backend(options)
    device = os.environ.get("YOUTUBE_RIPPER_WHISPER_DEVICE", "cpu")
    compute_type = os.environ.get("YOUTUBE_RIPPER_WHISPER_COMPUTE_TYPE", "int8")
    key = (model_name, device, compute_type)

    with _model_lock:
        if backend == "mlx":
            return _transcribe_mlx(filepath, options, model_name)
        if _mlx_model_holder is not None:
            _mlx_model_holder.model = None
            _mlx_model_holder.model_path = None
        if _model is None or _model_key != key:
            from faster_whisper import WhisperModel

            # Release the previous model before loading another large model.
            _model = None
            _model_key = None
            _model = WhisperModel(model_name, device=device, compute_type=compute_type)
            _model_key = key

        segments, info = _model.transcribe(
            str(filepath), beam_size=5, language=options.language, vad_filter=True
        )
        timed_segments = tuple(
            TranscriptSegment(start=float(seg.start), end=float(seg.end), text=seg.text.strip())
            for seg in segments
        )
        language = info.language

    return Transcript(segments=timed_segments, language=language, source="Local Whisper")


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
