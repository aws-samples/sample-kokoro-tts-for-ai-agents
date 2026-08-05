"""Tests for PollyClient.

Ported from tts_eval/tests/test_synthesize.py's TestPollyIntegration, which
tested the pre-tts-client SynthesisClient._synthesize_polly. The catalog
data (which voice_id/engine a model resolves to) stays in
tts_eval.synthesize.POLLY_VOICES -- PollyClient itself takes voice_id/engine
directly and knows nothing about this repo's model catalog, same as
TTSClient knows nothing about ENDPOINT_MAP.
"""

from __future__ import annotations

import struct

from tts_client.polly import DEFAULT_SAMPLE_RATE, PollyClient
from tts_client.types import AudioFormat


def _make_minimal_wav(duration_s: float = 1.0, sr: int = 24000) -> bytes:
    """A minimal valid WAV file -- librosa can decode WAV as well as MP3, so
    this stands in for Polly's real MP3 response without needing an MP3
    encoder in the test; the class's call sequence and field mapping are
    what's under test, not the codec itself."""
    channels = 1
    bits_per_sample = 16
    num_samples = int(sr * duration_s)
    data_size = num_samples * channels * (bits_per_sample // 8)
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + data_size,
        b"WAVE",
        b"fmt ",
        16,
        1,
        channels,
        sr,
        sr * channels * (bits_per_sample // 8),
        channels * (bits_per_sample // 8),
        bits_per_sample,
        b"data",
        data_size,
    )
    return header + b"\x00" * data_size


class _FakeAudioStream:
    def __init__(self, data: bytes) -> None:
        self._data = data
        self._pos = 0

    def read(self, n: int | None = None) -> bytes:
        if n is None:
            chunk = self._data[self._pos :]
            self._pos = len(self._data)
        else:
            chunk = self._data[self._pos : self._pos + n]
            self._pos += len(chunk)
        return chunk


class _FakePollyBoto3Client:
    def __init__(self, audio_bytes: bytes) -> None:
        self._audio_bytes = audio_bytes
        self.call_kwargs: dict = {}

    def synthesize_speech(self, **kwargs) -> dict:
        self.call_kwargs = kwargs
        return {"AudioStream": _FakeAudioStream(self._audio_bytes)}


def _client_with(audio_bytes: bytes) -> tuple[PollyClient, _FakePollyBoto3Client]:
    client = PollyClient.__new__(PollyClient)
    fake = _FakePollyBoto3Client(audio_bytes)
    client._client = fake  # type: ignore[attr-defined]
    return client, fake


class TestSynthesize:
    def test_returns_mp3_with_audio_metrics(self) -> None:
        wav = _make_minimal_wav(duration_s=2.5)
        client, _ = _client_with(wav)

        result = client.synthesize(voice_id="Joanna", engine="neural", text="hello world")

        assert result.audio_format == AudioFormat.MP3
        assert result.sample_rate == DEFAULT_SAMPLE_RATE
        assert abs(result.duration_s - 2.5) < 0.05
        assert result.chars == len("hello world")
        assert result.latency_ms >= 0
        assert result.ttfab_ms is not None
        assert result.ttfab_ms <= result.latency_ms

    def test_sends_voice_id_engine_and_mp3_format(self) -> None:
        client, fake = _client_with(_make_minimal_wav())
        client.synthesize(voice_id="Ruth", engine="generative", text="hi")

        assert fake.call_kwargs["VoiceId"] == "Ruth"
        assert fake.call_kwargs["Engine"] == "generative"
        assert fake.call_kwargs["OutputFormat"] == "mp3"
        assert fake.call_kwargs["Text"] == "hi"

    def test_sample_rate_is_sent_as_a_string(self) -> None:
        # boto3's Polly client requires SampleRate as a string, not an int.
        client, fake = _client_with(_make_minimal_wav())
        client.synthesize(voice_id="Salli", engine="standard", text="hi", sample_rate=16000)
        assert fake.call_kwargs["SampleRate"] == "16000"

    def test_custom_sample_rate_is_reflected_on_the_result(self) -> None:
        wav = _make_minimal_wav(sr=16000)
        client, _ = _client_with(wav)
        result = client.synthesize(
            voice_id="Salli", engine="standard", text="hi", sample_rate=16000
        )
        assert result.sample_rate == 16000

    def test_ttfab_reads_only_the_first_chunk_not_the_whole_stream(self) -> None:
        # The class reads 1024 bytes to stamp ttfab, then the rest -- proving
        # both halves are stitched back into one audio_bytes payload.
        wav = _make_minimal_wav(duration_s=1.0)
        assert len(wav) > 1024, "fixture must exceed the first-chunk read size"
        client, _ = _client_with(wav)

        result = client.synthesize(voice_id="Joanna", engine="neural", text="hi")
        assert len(result.audio_bytes) == len(wav)
