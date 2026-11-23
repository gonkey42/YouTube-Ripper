"""Fetch timed YouTube captions without downloading audio or video."""

import json
import math
from collections.abc import Callable
from urllib.parse import parse_qs, urlparse

from yt_dlp.networking import Request

from media_types import Transcript, TranscriptSegment

MAX_CAPTION_BYTES = 5 * 1024 * 1024
CAPTION_TIMEOUT = 15
MAX_TRACK_ATTEMPTS = 4


def _language(value: str) -> str:
    return value.lower().replace("_", "-").removesuffix("-orig")


def _matches(actual: str, requested: str) -> bool:
    actual, requested = _language(actual), _language(requested)
    # A broad request accepts regional variants; a regional request stays specific.
    return actual == requested or ("-" not in requested and actual.startswith(requested + "-"))


def _tracks(info: dict, key: str):
    for language, formats in (info.get(key) or {}).items():
        if not isinstance(language, str) or not isinstance(formats, list):
            continue
        for track in formats:
            if isinstance(track, dict) and track.get("ext") == "json3" and isinstance(track.get("url"), str):
                yield language, track


def _translated(track: dict) -> bool:
    return bool(parse_qs(urlparse(track["url"]).query).get("tlang"))


def _source_language(info: dict, uploaded: list, automatic: list) -> str | None:
    if isinstance(info.get("language"), str) and info["language"] != "und":
        return info["language"]
    original = {_language(lang) for lang, track in automatic if lang.endswith("-orig") and not _translated(track)}
    if len(original) == 1:
        return original.pop()
    for fmt in info.get("formats") or []:
        if isinstance(fmt, dict) and "original" in (fmt.get("format_note") or "").lower() and fmt.get("language"):
            return fmt["language"]
    native = {_language(lang) for lang, track in automatic if not _translated(track)}
    if len(native) == 1:
        return native.pop()
    # A lone uploaded language is useful when the extractor has no audio language.
    manual = {_language(lang) for lang, track in uploaded if not _translated(track)}
    return manual.pop() if len(manual) == 1 else None


def _candidates(info: dict, requested: str | None):
    uploaded = list(_tracks(info, "subtitles"))
    automatic = list(_tracks(info, "automatic_captions"))
    language = requested or _source_language(info, uploaded, automatic)
    if not language:
        return
    seen = set()
    for tracks, source in ((uploaded, "YouTube uploaded captions"), (automatic, "YouTube automatic captions")):
        matched = [(lang, track) for lang, track in tracks if _matches(lang, language) and not _translated(track)]
        matched.sort(key=lambda item: (_language(item[0]) != _language(language), _translated(item[1]), not item[0].endswith("-orig")))
        for lang, track in matched:
            if track["url"] not in seen:
                seen.add(track["url"])
                yield _language(lang), track, source


def _number(value) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) and value >= 0 else None


def _parse_json3(payload: bytes) -> tuple[TranscriptSegment, ...]:
    data = json.loads(payload)
    if not isinstance(data, dict) or not isinstance(data.get("events"), list):
        return ()
    segments = []
    windows = {}
    for event in data["events"]:
        if not isinstance(event, dict) or not isinstance(event.get("segs"), list):
            continue
        start, duration = _number(event.get("tStartMs")), _number(event.get("dDurationMs"))
        if start is None or duration is None or duration <= 0:
            continue
        raw_text = "".join(part["utf8"] for part in event["segs"] if isinstance(part, dict) and isinstance(part.get("utf8"), str))
        text = " ".join(raw_text.split())
        if not text:
            continue
        start, end = start / 1000, (start + duration) / 1000
        window = event.get("wWinId")
        # Window IDs distinguish rolling ASR updates from separate spoken cues.
        prior_index = windows.get(window) if isinstance(window, (str, int)) else None
        if prior_index is not None:
            prior = segments[prior_index]
            if prior.start <= start < prior.end:
                if event.get("aAppend") == 1:
                    joiner = " " if raw_text[:1].isspace() else ""
                    segments[prior_index] = TranscriptSegment(prior.start, max(prior.end, end), prior.text + joiner + text)
                    continue
                if text == prior.text or text.startswith(prior.text + " "):
                    segments[prior_index] = TranscriptSegment(prior.start, max(prior.end, end), text)
                    continue
                old, new = prior.text.split(), text.split()
                # Require two repeated words, preserving single-word repetition.
                for count in range(min(len(old), len(new)), 1, -1):
                    if old[-count:] == new[:count]:
                        text = " ".join(new[count:])
                        break
                if not text:
                    continue
                segments[prior_index] = TranscriptSegment(prior.start, min(prior.end, start), prior.text)
        if isinstance(window, (str, int)):
            windows[window] = len(segments)
        segments.append(TranscriptSegment(start, end, text))
    return tuple(sorted((segment for segment in segments if segment.end > segment.start), key=lambda segment: segment.start))


def fetch_captions(url: str, language: str | None, run_ytdlp: Callable) -> tuple[dict, Transcript] | None:
    """Return captions and metadata, or None so the caller can use Whisper.

    ``language=None`` selects source captions. Explicit language requests select
    native captions in that language. The supplied runner owns cookie fallback.
    """
    options = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "noplaylist": True,
        "socket_timeout": CAPTION_TIMEOUT,
        "retries": 1,
        "extractor_retries": 1,
        # Request the full track inventory without writing subtitle files.
        "writeautomaticsub": True,
        "writesubtitles": True,
    }

    def retrieve(ydl):
        info = ydl.extract_info(url, download=False)
        if not isinstance(info, dict):
            return None
        for attempt, (selected_language, track, source) in enumerate(_candidates(info, language)):
            if attempt >= MAX_TRACK_ATTEMPTS:
                break
            try:
                headers = {**(info.get("http_headers") or {}), **(track.get("http_headers") or {})}
                request = Request(track["url"], headers=headers)
                with ydl.urlopen(request) as response:
                    payload = response.read(MAX_CAPTION_BYTES + 1)
                if len(payload) > MAX_CAPTION_BYTES:
                    continue
                segments = _parse_json3(payload)
                if segments:
                    return info, Transcript(segments, selected_language, source)
            except Exception:
                # A missing, expired, or malformed caption track must not stop
                # the user's transcription. Try another track, then Whisper.
                continue
        return None

    try:
        return run_ytdlp(options, retrieve)
    except Exception:
        return None
