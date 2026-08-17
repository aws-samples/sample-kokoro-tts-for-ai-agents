# Copyright (c) 2026 Amazon.com
# This file is licensed under the MIT License.
# See the LICENSE file in the project root for full license information.

"""Request/response types for the TTS synthesis API.

Standalone from the rest of the workspace on purpose: this package has no
dependency on ``tts_inference`` or ``shared``, so an external consumer can
install just ``tts-client`` and get everything it needs from one import.
"""

from __future__ import annotations

from enum import IntEnum, StrEnum

from pydantic import BaseModel, Field


class AudioFormat(StrEnum):
    """Audio encoding requested from the endpoint.

    Matches the container's own ``format`` values (``kokoro/serve.py``:
    ``FORMAT_WAV``, ``FORMAT_MP3``) so a request built here is accepted
    unmodified.
    """

    WAV = "wav"
    MP3 = "mp3"


class SampleRate(IntEnum):
    """Output sample rate requested from the endpoint.

    Matches the container's own ``SUPPORTED_SAMPLE_RATES`` (``kokoro/serve.py``).
    ``HZ_24000`` (Kokoro's native rate) is the ceiling, not one option among
    several: producing a higher rate from a 24kHz source would be pure
    interpolation with no added fidelity, so nothing above it is offered.
    Every other member is a real downsample, done server-side; requesting one
    adds resampling latency over the native rate.
    """

    HZ_8000 = 8000
    HZ_16000 = 16000
    HZ_22050 = 22050
    HZ_24000 = 24000


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
    sample_rate: SampleRate | None = Field(
        default=None,
        description="Output sample rate. Omitted (the default) requests the "
        "endpoint's native rate with no resampling; the container 400s a "
        "value it doesn't support.",
    )
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


class SynthesisChunk(BaseModel):
    """One text chunk's audio, from a :meth:`TTSClient.synthesize_bidi_stream` session.

    ``audio_bytes`` is raw PCM, not WAV-wrapped: a caller consuming several
    chunks concatenates them in ``seq`` order and wraps once at the end
    (see ``streaming.concat_chunks_to_wav``), rather than paying a WAV header
    per chunk. ``ttfab_ms`` is set only on the first chunk of a session,
    measured from session-open — later chunks reuse the same connection, so
    their first-byte time is not a meaningful "time to first audio" signal.
    """

    seq: int
    text: str
    audio_bytes: bytes
    ttfab_ms: float | None = None
    duration_s: float
    sample_rate: int
