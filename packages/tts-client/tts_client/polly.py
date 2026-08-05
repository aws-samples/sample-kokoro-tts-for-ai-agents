"""Client for Amazon Polly.

A different AWS service from :class:`tts_client.client.TTSClient`'s
SageMaker-endpoint contract, kept as its own class rather than a mode on
that one: Polly is a managed, unbounded API (no instance to size, no queue
to saturate), so none of ``TTSClient``'s pooling/retry/error-taxonomy design
— all of it built around SageMaker's specific failure modes — applies here.
Conflating the two into one class's error handling would blur a design that
was deliberately kept narrow to one contract.

Requires the ``polly`` extra (``librosa``, for measuring the duration of the
MP3 Polly returns — its response carries no duration field of its own). A
caller who only needs :class:`TTSClient` never imports this module and never
pays for that dependency.
"""

from __future__ import annotations

import io
import time

import boto3
import librosa

from tts_client.types import AudioFormat, SynthesisResult

#: Polly's synthesize_speech has no sample-rate-agnostic mode; this repo's
#: contract with it is fixed at 24kHz to match the self-hosted models it is
#: compared against.
DEFAULT_SAMPLE_RATE = 24000


class PollyClient:
    """Blocking client for Amazon Polly speech synthesis.

    Make as many instances as you like, same as :class:`TTSClient` — one
    boto3 client per instance, safe to share across threads or not, no
    sizing decision required.
    """

    def __init__(self, region: str = "us-east-1") -> None:
        self._client = boto3.client("polly", region_name=region)

    def synthesize(
        self,
        *,
        voice_id: str,
        engine: str,
        text: str,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
    ) -> SynthesisResult:
        """Synthesize text via Amazon Polly. Always returns MP3.

        Args:
            voice_id: Polly voice, e.g. ``"Joanna"``.
            engine: Polly engine, e.g. ``"neural"``.
            text: Text to synthesize.
            sample_rate: Output sample rate in Hz.

        Duration is measured by decoding the returned MP3 with ``librosa``:
        Polly's response has no duration field of its own.
        """
        t0 = time.perf_counter()
        response = self._client.synthesize_speech(
            Text=text,
            Engine=engine,
            VoiceId=voice_id,
            OutputFormat="mp3",
            SampleRate=str(sample_rate),
        )
        stream = response["AudioStream"]
        first_chunk = stream.read(1024)
        ttfab_ms = (time.perf_counter() - t0) * 1000.0
        rest = stream.read()
        audio_bytes = first_chunk + rest
        latency_ms = (time.perf_counter() - t0) * 1000.0

        y, sr_actual = librosa.load(io.BytesIO(audio_bytes), sr=None)
        duration_s = len(y) / sr_actual

        return SynthesisResult(
            audio_bytes=audio_bytes,
            audio_format=AudioFormat.MP3,
            sample_rate=sample_rate,
            duration_s=duration_s,
            latency_ms=latency_ms,
            ttfab_ms=ttfab_ms,
            chars=len(text),
        )
