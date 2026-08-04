"""Request/response types for the TTS synthesis API.

Standalone from the rest of the workspace on purpose: this package has no
dependency on ``tts_inference`` or ``shared``, so an external consumer can
install just ``tts-client`` and get everything it needs from one import.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class AudioFormat(StrEnum):
    """Audio encoding requested from the endpoint.

    Matches the container's own ``format`` values (``kokoro/serve.py``:
    ``FORMAT_WAV``, ``FORMAT_MP3``) so a request built here is accepted
    unmodified.
    """

    WAV = "wav"
    MP3 = "mp3"


class Transport(StrEnum):
    """Which of the two wire protocols a call used."""

    RESPONSE_STREAM = "response-stream"
    """HTTP/1.1 chunked binary body — :meth:`TTSClient.synthesize`."""

    BIDI = "bidi"
    """SageMaker bidirectional streaming (HTTP/2) — raw PCM, no format
    choice — :meth:`TTSClient.synthesize_bidi`."""


class SynthesisRequest(BaseModel):
    """One synthesis call.

    ``audio_format`` is ignored by :meth:`TTSClient.synthesize_bidi`: that
    transport has no format field server-side and always returns raw PCM.
    """

    text: str
    voice: str
    speed: float = 1.0
    audio_format: AudioFormat = AudioFormat.WAV
    request_timestamp: float | None = Field(
        default=None,
        description="Epoch seconds the request was sent. Stamped by the client "
        "at call time if left unset — the container measures request age "
        "against this to reject stale work.",
    )


class SynthesisResult(BaseModel):
    """One synthesis response, uniform across both transports."""

    audio_bytes: bytes
    audio_format: AudioFormat
    sample_rate: int
    duration_s: float
    latency_ms: float
    ttfab_ms: float | None = None
    chars: int
    chunks: int = 0
