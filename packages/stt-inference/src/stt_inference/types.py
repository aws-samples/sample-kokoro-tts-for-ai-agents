"""Enums and Pydantic models for STT inference."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, Field


class STTModelName(StrEnum):
    """Supported STT model identifiers."""

    WHISPER_LARGE_V3 = "whisper-large-v3"
    QWEN3_ASR = "qwen3-asr"


class ExecutionMode(StrEnum):
    """Where inference runs."""

    LOCAL = "local"
    SAGEMAKER = "sagemaker"
    AUTO = "auto"


class AudioFormat(StrEnum):
    """Supported audio input formats."""

    WAV = "wav"
    MP3 = "mp3"
    FLAC = "flac"
    OGG = "ogg"
    M4A = "m4a"


class WordSegment(BaseModel):
    """A single word with timing information."""

    word: str
    start: float = Field(description="Start time in seconds")
    end: float = Field(description="End time in seconds")
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class TranscriptionSegment(BaseModel):
    """A segment (sentence/phrase) of transcribed audio."""

    text: str
    start: float = Field(description="Start time in seconds")
    end: float = Field(description="End time in seconds")
    words: list[WordSegment] | None = None
    language: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class TranscriptionResult(BaseModel):
    """Complete STT output for an audio file."""

    source: str
    model_name: STTModelName
    text: str
    segments: list[TranscriptionSegment]
    language: str | None = None
    duration_seconds: float = Field(description="Audio duration in seconds")
    elapsed_seconds: float = Field(description="Inference wall-clock time")

    @property
    def has_word_timestamps(self) -> bool:
        return any(s.words is not None for s in self.segments)

    @property
    def word_count(self) -> int:
        return len(self.text.split())

    def get_text(self, separator: str = " ") -> str:
        return separator.join(s.text for s in self.segments)


class AudioInput(BaseModel):
    """Normalized audio input for STT models."""

    path: str | None = None
    audio_bytes: bytes | None = None
    sample_rate: int = 16000
    format: AudioFormat | None = None

    @property
    def resolved_path(self) -> Path | None:
        if self.path:
            return Path(self.path)
        return None


class STTOutputParser(Protocol):
    """Protocol for model-specific raw output parsers."""

    def parse(self, raw_output: dict, source: str, elapsed: float) -> TranscriptionResult:
        ...
