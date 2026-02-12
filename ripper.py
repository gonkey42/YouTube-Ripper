"""Core logic for YouTube audio extraction, transcription, and PDF generation."""

import re
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


def process(url: str, mode: str):
    """Main orchestrator. Yields (type, data) tuples for SSE streaming.

    mode: 'audio', 'text', or 'both'
    """
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
