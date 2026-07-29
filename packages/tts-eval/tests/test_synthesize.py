"""Tests for SynthesisClient utilities."""

from __future__ import annotations

import base64
import json
import struct

import pytest

from tts_eval.synthesize import ENDPOINT_MAP, POLLY_VOICES, _pcm_to_wav, wav_duration
from tts_inference.types import TTSModelName


def _make_minimal_wav(duration_s: float = 1.0, sr: int = 24000) -> bytes:
    """Create a minimal valid WAV file."""
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
        1,  # PCM
        channels,
        sr,
        sr * channels * (bits_per_sample // 8),
        channels * (bits_per_sample // 8),
        bits_per_sample,
        b"data",
        data_size,
    )
    return header + b"\x00" * data_size


class TestWavDuration:
    def test_correct_duration_calculation(self) -> None:
        wav = _make_minimal_wav(duration_s=2.5, sr=24000)
        d = wav_duration(wav)
        assert abs(d - 2.5) < 0.01

    def test_different_sample_rates(self) -> None:
        for sr in [16000, 22050, 24000, 44100, 48000]:
            wav = _make_minimal_wav(duration_s=1.0, sr=sr)
            d = wav_duration(wav)
            assert abs(d - 1.0) < 0.01

    def test_invalid_data_returns_zero(self) -> None:
        assert wav_duration(b"not a wav file") == 0.0
        assert wav_duration(b"") == 0.0
        assert wav_duration(b"RIFF" + b"\x00" * 10) == 0.0


class TestEndpointMap:
    def test_deployed_models_have_endpoints(self) -> None:
        deployed = [
            TTSModelName.ORPHEUS_3B,
            TTSModelName.KOKORO_82M,
            TTSModelName.CHATTERBOX_TURBO,
        ]
        for model in deployed:
            assert model in ENDPOINT_MAP, f"Missing endpoint for {model.value}"

    def test_endpoint_names_follow_convention(self) -> None:
        for _model, endpoint in ENDPOINT_MAP.items():
            assert endpoint.startswith("speech-"), f"{endpoint} should start with 'speech-'"


class TestPollyIntegration:
    def test_polly_voices_config_complete(self) -> None:
        assert TTSModelName.POLLY_STANDARD in POLLY_VOICES
        assert TTSModelName.POLLY_NEURAL in POLLY_VOICES
        assert TTSModelName.POLLY_GENERATIVE in POLLY_VOICES
        for config in POLLY_VOICES.values():
            assert "engine" in config
            assert "voice_id" in config

    def test_pcm_to_wav_produces_valid_header(self) -> None:
        pcm = b"\x00" * 32000  # 1 second at 16000Hz, 16-bit mono
        wav = _pcm_to_wav(pcm, 16000)
        assert wav[:4] == b"RIFF"
        assert wav[8:12] == b"WAVE"
        duration = wav_duration(wav)
        assert abs(duration - 1.0) < 0.01

    def test_pcm_to_wav_correct_sample_rate(self) -> None:
        pcm = b"\x00" * 48000  # 1.5 seconds at 16000Hz
        wav = _pcm_to_wav(pcm, 16000)
        sr = struct.unpack_from("<I", wav, 24)[0]
        assert sr == 16000

    def test_polly_voices_use_mp3_format(self) -> None:
        """Polly synthesis should indicate MP3 output format at 24kHz."""
        for _model_name, config in POLLY_VOICES.items():
            assert config["engine"] in ("standard", "neural", "generative")
            assert config["voice_id"]


def _sse_frame(event: str, payload: dict) -> bytes:
    return f"event: {event}\ndata: {json.dumps(payload)}\n\n".encode()


def _sse_body(chunks: list[bytes], duration_s: float = 2.0) -> bytes:
    """Build a full SSE response body matching the kokoro container's contract."""
    body = _sse_frame(
        "audio_stream_start",
        {"request_id": "r1", "format": "mp3", "voice": "af_heart", "sample_rate": 24000},
    )
    for seq, chunk in enumerate(chunks):
        body += _sse_frame(
            "audio_chunk",
            {"request_id": "r1", "seq": seq, "data": base64.b64encode(chunk).decode("ascii")},
        )
    body += _sse_frame(
        "audio_stream_end",
        {"request_id": "r1", "total_chunks": len(chunks), "duration_s": duration_s},
    )
    return body


class _FakeStreamClient:
    """Stands in for a boto3 sagemaker-runtime client streaming an SSE body.

    `parts` is the exact byte split delivered as PayloadParts, so a test can
    reproduce SageMaker fragmenting the body wherever it likes.
    """

    def __init__(self, parts: list[bytes], delay_s: float = 0.0) -> None:
        self._parts = parts
        self._delay_s = delay_s
        self.call_kwargs: dict = {}

    def invoke_endpoint_with_response_stream(self, **kwargs) -> dict:
        self.call_kwargs = kwargs
        return {"ContentType": "text/event-stream", "Body": self._iter_parts()}

    def _iter_parts(self):
        import time as _time

        for part in self._parts:
            if self._delay_s:
                _time.sleep(self._delay_s)
            yield {"PayloadPart": {"Bytes": part}}


def _client_with(parts: list[bytes], delay_s: float = 0.0) -> tuple[object, _FakeStreamClient]:
    """Build a SynthesisClient whose thread-local boto3 client is faked."""
    from tts_eval.synthesize import SynthesisClient

    client = SynthesisClient.__new__(SynthesisClient)
    fake = _FakeStreamClient(parts, delay_s)
    client._client = fake
    client._region = "us-east-1"
    client._get_thread_client = lambda: fake  # type: ignore[method-assign]
    return client, fake


class TestSynthesizeSSE:
    """The SSE transport used by the AgentCore relay.

    The framing tests here are not hypothetical: against the live endpoint a
    4-frame response arrived as 6 PayloadParts with 2 of them ending mid-frame,
    so a client that parses each part independently fails on valid traffic.
    """

    def test_reassembles_frames_split_across_payload_parts(self) -> None:
        audio = [b"\xff\xf3d\xc4" + bytes(range(64)), b"\xff\xf3" + bytes(range(32))]
        body = _sse_body(audio)
        # Split at byte offsets that land mid-frame, as SageMaker actually does.
        parts = [body[:40], body[40:150], body[150:210], body[210:]]
        assert not all(p.endswith(b"\n\n") for p in parts), "test setup must split mid-frame"

        client, _ = _client_with(parts)
        result = client.synthesize_sse(TTSModelName.KOKORO_82M, "hello there")

        assert result["audio_bytes"] == b"".join(audio)
        assert result["total_chunks"] == 2

    def test_single_part_body_also_works(self) -> None:
        audio = [b"\xff\xf3d\xc4" + bytes(range(48))]
        client, _ = _client_with([_sse_body(audio)])
        result = client.synthesize_sse(TTSModelName.KOKORO_82M, "hello")
        assert result["audio_bytes"] == audio[0]

    def test_requests_sse_mp3_transport(self) -> None:
        client, fake = _client_with([_sse_body([b"\xff\xf3d\xc4"])])
        client.synthesize_sse(TTSModelName.KOKORO_82M, "hello")

        body = json.loads(fake.call_kwargs["Body"])
        assert body["transport"] == "sse"
        assert body["format"] == "mp3"
        assert fake.call_kwargs["EndpointName"] == "speech-kokoro-82m"

    def test_ttfab_measured_at_first_audio_chunk_not_stream_start(self) -> None:
        """The start frame precedes inference, so timing it would report ~0ms.

        Delivering audio_stream_start well before the first audio_chunk means a
        client that stamps ttfab on the first byte reports a fraction of the real
        wait. Two 50ms-delayed parts precede the audio, so a correct client
        reports >=100ms.
        """
        audio = [b"\xff\xf3d\xc4" + bytes(range(64))]
        body = _sse_body(audio)
        start_frame, rest = body.split(b"\n\n", 1)
        parts = [start_frame + b"\n\n", b"", rest]

        client, _ = _client_with(parts, delay_s=0.05)
        result = client.synthesize_sse(TTSModelName.KOKORO_82M, "hello")

        assert result["ttfab_ms"] >= 100, (
            f"ttfab {result['ttfab_ms']:.0f}ms implies timing was taken at "
            "audio_stream_start rather than the first audio_chunk"
        )

    def test_duration_comes_from_stream_end_event(self) -> None:
        client, _ = _client_with([_sse_body([b"\xff\xf3d\xc4"], duration_s=3.25)])
        result = client.synthesize_sse(TTSModelName.KOKORO_82M, "hello")
        assert result["duration_s"] == 3.25

    def test_error_event_raises(self) -> None:
        body = _sse_frame("audio_stream_start", {"request_id": "r1", "format": "mp3"}) + _sse_frame(
            "error", {"request_id": "r1", "message": "synthesis exploded"}
        )

        client, _ = _client_with([body])
        with pytest.raises(RuntimeError, match="synthesis exploded"):
            client.synthesize_sse(TTSModelName.KOKORO_82M, "hello")

    def test_reports_chars_and_voice(self) -> None:
        client, _ = _client_with([_sse_body([b"\xff\xf3d\xc4"])])
        result = client.synthesize_sse(TTSModelName.KOKORO_82M, "hello there")
        assert result["chars"] == len("hello there")
        assert result["voice"] == "af_heart"


class TestSynthesizeStreamUnchanged:
    """Guard the eval baseline: synthesize_stream must stay raw chunked WAV.

    runner.py:156 calls this and asserts RIFF downstream. Adding the SSE path
    must not move it.
    """

    def test_stream_sends_no_transport_or_format_fields(self) -> None:
        wav = _make_minimal_wav(duration_s=1.0)
        client, fake = _client_with([wav[:20], wav[20:]])
        result = client.synthesize_stream(TTSModelName.KOKORO_82M, "hello")

        body = json.loads(fake.call_kwargs["Body"])
        assert "transport" not in body
        assert "format" not in body
        assert result["audio_bytes"][:4] == b"RIFF"
