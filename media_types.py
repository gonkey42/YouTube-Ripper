"""Shared options and timed transcript data for local speech and captions."""

from dataclasses import dataclass


@dataclass(frozen=True)
class TranscriptSegment:
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class Transcript:
    segments: tuple[TranscriptSegment, ...]
    language: str | None
    source: str


@dataclass(frozen=True)
class TranscriptionOptions:
    model: str | None = None
    language: str | None = None
    prefer_captions: bool = False
    subtitle_formats: tuple[str, ...] = ()
    backend: str = "auto"
