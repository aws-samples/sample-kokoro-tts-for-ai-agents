"""Enums and Pydantic models for TTS inference."""

from __future__ import annotations

from enum import StrEnum
from typing import Protocol

from pydantic import BaseModel, Field


class TTSModelName(StrEnum):
    """Supported TTS model identifiers."""

    KOKORO_82M = "kokoro-82m"
    KOKORO_82M_CPU = "kokoro-82m-cpu"
    MAYA_VEENA = "maya-veena"
    CHATTERBOX_TURBO = "chatterbox-turbo"
    ORPHEUS_3B = "orpheus-3b"


class ExecutionMode(StrEnum):
    """Where inference runs."""

    LOCAL = "local"
    SAGEMAKER = "sagemaker"
    AUTO = "auto"


class AudioEncoding(StrEnum):
    """Output audio encoding formats."""

    WAV = "wav"
    MP3 = "mp3"
    FLAC = "flac"
    RAW_PCM = "raw_pcm"


class VoiceConfig(BaseModel):
    """Configuration for TTS voice selection and style."""

    voice_id: str | None = None
    language: str = "en"
    speed: float = Field(default=1.0, ge=0.5, le=3.0)
    reference_audio_path: str | None = Field(
        default=None,
        description="Path to reference audio for voice cloning (Chatterbox)",
    )


class SynthesisRequest(BaseModel):
    """Input request for TTS synthesis."""

    text: str
    voice: VoiceConfig = Field(default_factory=VoiceConfig)
    encoding: AudioEncoding = AudioEncoding.WAV
    sample_rate: int = 24000


class SynthesisResult(BaseModel):
    """Complete TTS output for a text input."""

    source_text: str
    model_name: TTSModelName
    audio_bytes: bytes
    sample_rate: int = Field(default=24000, description="Output sample rate in Hz")
    duration_seconds: float = Field(description="Generated audio duration")
    elapsed_seconds: float = Field(description="Inference wall-clock time")
    encoding: AudioEncoding = AudioEncoding.WAV
    voice_config: VoiceConfig | None = None

    @property
    def realtime_factor(self) -> float:
        """RTF: time to generate / audio duration. <1.0 means faster than real-time."""
        if self.duration_seconds == 0:
            return float("inf")
        return self.elapsed_seconds / self.duration_seconds

    @property
    def audio_size_bytes(self) -> int:
        return len(self.audio_bytes)


class TTSOutputParser(Protocol):
    """Protocol for model-specific raw output parsers."""

    def parse(
        self, raw_audio: bytes, metadata: dict, source_text: str, elapsed: float
    ) -> SynthesisResult:
        ...
