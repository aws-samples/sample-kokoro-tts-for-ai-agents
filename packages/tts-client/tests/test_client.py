"""Tests for TTSClient across both transports.

Fakes mirror the patterns already proven in tts_eval/tests/test_synthesize.py
(response-stream) and tts_bench/tests/test_bidi.py (bidi). The bidi
fakes patch ``tts_client._bidi_transport.SageMakerRuntimeHTTP2Client``
directly rather than injecting a client, because :meth:`TTSClient.synthesize_bidi`
deliberately builds its own client per call and accepts none — see the
client module's docstring for why sharing one is unsafe with the installed
SDK.
"""

from __future__ import annotations

import json
import struct
from unittest.mock import patch

import pytest
from aws_sdk_sagemaker_runtime_http2.models import (
    ModelStreamError,
    ResponsePayloadPart,
    ResponseStreamEventInternalStreamFailure,
    ResponseStreamEventModelStreamError,
    ResponseStreamEventPayloadPart,
)

from tts_client.client import TTSClient, _pcm_to_wav, wav_duration
from tts_client.errors import QueueSaturatedError, ServerError, TTSClientError
from tts_client.types import AudioFormat, SampleRate, SynthesisRequest

PCM_100MS = b"\x00\x01" * 2400


def _make_wav(duration_s: float = 1.0, sr: int = 24000) -> bytes:
    channels, bits = 1, 16
    num_samples = int(sr * duration_s)
    data_size = num_samples * channels * (bits // 8)
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
        sr * channels * (bits // 8),
        channels * (bits // 8),
        bits,
        b"data",
        data_size,
    )
    return header + b"\x00" * data_size


class TestWavDuration:
    def test_computes_duration_from_header(self) -> None:
        wav = _make_wav(duration_s=2.5)
        assert abs(wav_duration(wav) - 2.5) < 0.001

    def test_non_riff_data_returns_zero(self) -> None:
        assert wav_duration(b"not a wav") == 0.0


class TestPcmToWav:
    def test_wraps_with_correct_sample_rate(self) -> None:
        pcm = b"\x00" * 48000
        wav = _pcm_to_wav(pcm, sample_rate=16000)
        sr = struct.unpack_from("<I", wav, 24)[0]
        assert sr == 16000
        assert wav[:4] == b"RIFF"


# --------------------------------------------------------------------------- #
# Response-stream transport
# --------------------------------------------------------------------------- #


class _FakeStreamClient:
    """Stands in for a boto3 sagemaker-runtime client streaming a chunked body."""

    def __init__(self, parts: list[bytes]) -> None:
        self._parts = parts
        self.call_kwargs: dict = {}

    def invoke_endpoint_with_response_stream(self, **kwargs) -> dict:
        self.call_kwargs = kwargs
        return {"Body": ({"PayloadPart": {"Bytes": p}} for p in self._parts)}


class _RawEventStreamClient:
    """Yields raw event dicts, for mid-stream fault events that aren't
    ``PayloadPart``-shaped."""

    def __init__(self, events: list[dict]) -> None:
        self._events = events

    def invoke_endpoint_with_response_stream(self, **kwargs) -> dict:
        return {"Body": iter(self._events)}


def _client_with(parts: list[bytes]) -> tuple[TTSClient, _FakeStreamClient]:
    client = TTSClient.__new__(TTSClient)
    fake = _FakeStreamClient(parts)
    client._client = fake  # type: ignore[attr-defined]
    client._region = "us-east-1"  # type: ignore[attr-defined]
    return client, fake


class TestSynthesizeResponseStream:
    def test_reassembles_chunks_and_reads_duration(self) -> None:
        wav = _make_wav(duration_s=2.0)
        client, fake = _client_with([wav[:20], wav[20:]])

        result = client.synthesize(
            "speech-kokoro-82m", SynthesisRequest(text="hello there", voice="af_heart")
        )

        assert result.audio_bytes == wav
        assert abs(result.duration_s - 2.0) < 0.01
        assert result.sample_rate == 24000
        assert result.chars == len("hello there")
        assert result.chunks == 2

    def test_sends_requested_format_and_no_transport_field(self) -> None:
        """No `transport` field: the container's default (TRANSPORT_BINARY)
        is exactly what this method wants."""
        wav = _make_wav()
        client, fake = _client_with([wav])
        client.synthesize("speech-kokoro-82m", SynthesisRequest(text="hi", voice="af_heart"))

        body = json.loads(fake.call_kwargs["Body"])
        assert "transport" not in body
        assert body["format"] == "wav"
        assert fake.call_kwargs["EndpointName"] == "speech-kokoro-82m"
        assert fake.call_kwargs["Accept"] == "audio/wav"

    def test_mp3_format_sets_mpeg_accept_header(self) -> None:
        client, fake = _client_with([b"\xff\xf3d\xc4"])
        client.synthesize(
            "speech-kokoro-82m",
            SynthesisRequest(text="hi", voice="af_heart", audio_format=AudioFormat.MP3),
        )
        assert fake.call_kwargs["Accept"] == "audio/mpeg"

    def test_empty_response_raises(self) -> None:
        client, _ = _client_with([])
        with pytest.raises(TTSClientError, match="no audio bytes"):
            client.synthesize("speech-kokoro-82m", SynthesisRequest(text="hi", voice="af_heart"))

    def test_sample_rate_is_omitted_by_default(self) -> None:
        """The default (no resampling requested) must produce the exact same
        body as before this field existed -- the eval baseline depends on it."""
        wav = _make_wav()
        client, fake = _client_with([wav])
        client.synthesize("speech-kokoro-82m", SynthesisRequest(text="hi", voice="af_heart"))

        body = json.loads(fake.call_kwargs["Body"])
        assert "sample_rate" not in body

    def test_sample_rate_is_sent_when_set(self) -> None:
        wav = _make_wav()
        client, fake = _client_with([wav])
        client.synthesize(
            "speech-kokoro-82m",
            SynthesisRequest(text="hi", voice="af_heart", sample_rate=SampleRate.HZ_16000),
        )

        body = json.loads(fake.call_kwargs["Body"])
        assert body["sample_rate"] == 16000

    def test_model_stream_error_mid_stream_raises_server_error(self) -> None:
        """Mid-stream faults arrive as events on a response that already
        returned HTTP 200 -- the failure mode behind the torch.Tensor
        regression in commit caa4dcf, per invoke.py's docstring."""
        client = TTSClient.__new__(TTSClient)
        client._client = _RawEventStreamClient(  # type: ignore[attr-defined]
            [{"ModelStreamError": {"Message": "boom"}}]
        )
        client._region = "us-east-1"  # type: ignore[attr-defined]

        with pytest.raises(ServerError, match="boom"):
            client.synthesize("speech-kokoro-82m", SynthesisRequest(text="hi", voice="af_heart"))

    def test_internal_stream_failure_mid_stream_raises_server_error(self) -> None:
        client = TTSClient.__new__(TTSClient)
        client._client = _RawEventStreamClient(  # type: ignore[attr-defined]
            [{"InternalStreamFailure": {"Message": "infra blip"}}]
        )
        client._region = "us-east-1"  # type: ignore[attr-defined]

        with pytest.raises(ServerError, match="infra blip"):
            client.synthesize("speech-kokoro-82m", SynthesisRequest(text="hi", voice="af_heart"))


# --------------------------------------------------------------------------- #
# Bidi transport
# --------------------------------------------------------------------------- #


def _payload_event(data: bytes):
    return ResponseStreamEventPayloadPart(value=ResponsePayloadPart(bytes_=data))


def _control_frame(**fields):
    return _payload_event(json.dumps(fields).encode("utf-8"))


class _FakeInputStream:
    def __init__(self) -> None:
        self.sent: list = []
        self.closed = False

    async def send(self, event) -> None:
        self.sent.append(event)

    async def close(self) -> None:
        self.closed = True


class _FakeOutputStream:
    def __init__(self, events: list, *, on_receive=None) -> None:
        self._events = list(events)
        self._on_receive = on_receive
        self.receives = 0

    async def receive(self):
        self.receives += 1
        if self._on_receive is not None:
            self._on_receive(self.receives)
        if not self._events:
            return None
        return self._events.pop(0)


class _FakeStream:
    def __init__(self, events: list, *, on_receive=None) -> None:
        self.input_stream = _FakeInputStream()
        self.output_stream = _FakeOutputStream(events, on_receive=on_receive)
        self.closed = False

    async def await_output(self):
        return (object(), self.output_stream)

    async def close(self) -> None:
        self.closed = True


class _FakeBidiClientCtor:
    """Replaces ``SageMakerRuntimeHTTP2Client`` at the constructor call site."""

    def __init__(self, stream: _FakeStream) -> None:
        self._stream = stream
        self.config = None

    def __call__(self, config=None):
        self.config = config
        return self

    async def invoke_endpoint_with_bidirectional_stream(self, input_):
        return self._stream


class TestSynthesizeBidi:
    def test_success_wraps_pcm_in_wav(self) -> None:
        events = [_payload_event(PCM_100MS), _control_frame(type="synthesis_complete")]
        fake_ctor = _FakeBidiClientCtor(_FakeStream(events))

        with patch("tts_client._bidi_transport.SageMakerRuntimeHTTP2Client", fake_ctor):
            client = TTSClient()
            result = client.synthesize_bidi(
                "speech-kokoro-82m", SynthesisRequest(text="hello", voice="af_heart")
            )

        assert result.audio_bytes[:4] == b"RIFF"
        assert result.audio_format == AudioFormat.WAV
        assert result.sample_rate == 24000
        assert result.chunks == 1

    def test_error_frame_raises_queue_saturated(self) -> None:
        events = [_control_frame(type="error", message="queue_saturated: try later")]
        fake_ctor = _FakeBidiClientCtor(_FakeStream(events))

        with patch("tts_client._bidi_transport.SageMakerRuntimeHTTP2Client", fake_ctor):
            client = TTSClient()
            with pytest.raises(QueueSaturatedError):
                client.synthesize_bidi(
                    "speech-kokoro-82m", SynthesisRequest(text="hello", voice="af_heart")
                )

    def test_model_stream_error_mid_stream_raises_server_error(self) -> None:
        events = [
            _payload_event(PCM_100MS),
            ResponseStreamEventModelStreamError(value=ModelStreamError("boom")),
        ]
        fake_ctor = _FakeBidiClientCtor(_FakeStream(events))

        with patch("tts_client._bidi_transport.SageMakerRuntimeHTTP2Client", fake_ctor):
            client = TTSClient()
            with pytest.raises(ServerError, match="boom"):
                client.synthesize_bidi(
                    "speech-kokoro-82m", SynthesisRequest(text="hello", voice="af_heart")
                )

    def test_internal_stream_failure_raises_server_error(self) -> None:
        from aws_sdk_sagemaker_runtime_http2.models import InternalStreamFailure

        events = [
            _payload_event(PCM_100MS),
            ResponseStreamEventInternalStreamFailure(value=InternalStreamFailure("infra blip")),
        ]
        fake_ctor = _FakeBidiClientCtor(_FakeStream(events))

        with patch("tts_client._bidi_transport.SageMakerRuntimeHTTP2Client", fake_ctor):
            client = TTSClient()
            with pytest.raises(ServerError, match="infra blip"):
                client.synthesize_bidi(
                    "speech-kokoro-82m", SynthesisRequest(text="hello", voice="af_heart")
                )

    def test_empty_stream_raises(self) -> None:
        events = [_control_frame(type="synthesis_complete")]
        fake_ctor = _FakeBidiClientCtor(_FakeStream(events))

        with patch("tts_client._bidi_transport.SageMakerRuntimeHTTP2Client", fake_ctor):
            client = TTSClient()
            with pytest.raises(TTSClientError, match="no audio bytes"):
                client.synthesize_bidi(
                    "speech-kokoro-82m", SynthesisRequest(text="hello", voice="af_heart")
                )

    def test_sends_text_voice_and_timestamp(self) -> None:
        events = [_payload_event(PCM_100MS), _control_frame(type="synthesis_complete")]
        stream = _FakeStream(events)
        fake_ctor = _FakeBidiClientCtor(stream)

        with patch("tts_client._bidi_transport.SageMakerRuntimeHTTP2Client", fake_ctor):
            client = TTSClient()
            client.synthesize_bidi(
                "speech-kokoro-82m", SynthesisRequest(text="hello world", voice="af_bella")
            )

        sent = stream.input_stream.sent
        assert len(sent) == 1
        message = json.loads(sent[0].value.bytes_.decode("utf-8"))
        assert message["text"] == "hello world"
        assert message["voice"] == "af_bella"
        assert "request_timestamp" in message

    def test_sample_rate_is_sent_in_the_bidi_message_when_set(self) -> None:
        events = [_payload_event(PCM_100MS), _control_frame(type="synthesis_complete")]
        stream = _FakeStream(events)
        fake_ctor = _FakeBidiClientCtor(stream)

        with patch("tts_client._bidi_transport.SageMakerRuntimeHTTP2Client", fake_ctor):
            client = TTSClient()
            client.synthesize_bidi(
                "speech-kokoro-82m",
                SynthesisRequest(text="hello", voice="af_heart", sample_rate=SampleRate.HZ_16000),
            )

        message = json.loads(stream.input_stream.sent[0].value.bytes_.decode("utf-8"))
        assert message["sample_rate"] == 16000

    def test_sample_rate_omitted_from_message_by_default(self) -> None:
        events = [_payload_event(PCM_100MS), _control_frame(type="synthesis_complete")]
        stream = _FakeStream(events)
        fake_ctor = _FakeBidiClientCtor(stream)

        with patch("tts_client._bidi_transport.SageMakerRuntimeHTTP2Client", fake_ctor):
            client = TTSClient()
            client.synthesize_bidi(
                "speech-kokoro-82m", SynthesisRequest(text="hello", voice="af_heart")
            )

        message = json.loads(stream.input_stream.sent[0].value.bytes_.decode("utf-8"))
        assert "sample_rate" not in message

    def test_sample_rate_is_reflected_on_the_result_when_set(self) -> None:
        # Raw PCM carries no self-describing rate, so the client reports back
        # whatever it asked for -- there is nothing on the wire to read
        # it from. duration_s must divide by the requested rate, not the
        # BIDI_SAMPLE_RATE default, or a resampled response reports a wrong
        # duration.
        events = [_payload_event(PCM_100MS), _control_frame(type="synthesis_complete")]
        fake_ctor = _FakeBidiClientCtor(_FakeStream(events))

        with patch("tts_client._bidi_transport.SageMakerRuntimeHTTP2Client", fake_ctor):
            client = TTSClient()
            result = client.synthesize_bidi(
                "speech-kokoro-82m",
                SynthesisRequest(text="hello", voice="af_heart", sample_rate=SampleRate.HZ_16000),
            )

        assert result.sample_rate == 16000
        assert abs(result.duration_s - len(PCM_100MS) / (16000 * 2)) < 1e-9
        sr_in_header = struct.unpack_from("<I", result.audio_bytes, 24)[0]
        assert sr_in_header == 16000

    def test_does_not_hold_a_client_on_self(self) -> None:
        """The whole point of this transport: two calls build two independent
        HTTP/2 clients, never sharing state — see the client module docstring
        for the unlocked-connection-dict race this avoids."""
        events = [_payload_event(PCM_100MS), _control_frame(type="synthesis_complete")]

        seen_instances = []

        class _TrackingCtor(_FakeBidiClientCtor):
            def __call__(self, config=None):
                instance = super().__call__(config=config)
                seen_instances.append(instance)
                return instance

        fake_ctor = _TrackingCtor(_FakeStream(events))
        # Reset the stream's consumed events between calls.
        with patch("tts_client._bidi_transport.SageMakerRuntimeHTTP2Client", fake_ctor):
            client = TTSClient()
            fake_ctor._stream = _FakeStream(events)
            client.synthesize_bidi(
                "speech-kokoro-82m", SynthesisRequest(text="one", voice="af_heart")
            )
            fake_ctor._stream = _FakeStream(events)
            client.synthesize_bidi(
                "speech-kokoro-82m", SynthesisRequest(text="two", voice="af_heart")
            )

        assert len(seen_instances) == 2, "each call must construct its own HTTP/2 client"

    def test_input_stream_is_not_closed_before_the_output_is_read(self) -> None:
        """Closing the input right after send loses the whole request.

        SageMaker tears the WebSocket down when the input half closes, and it
        does so before the container reads the payload -- verified live:
        closing early gave 0 bytes, holding the input open gave a full
        synthesis for the same request. So the close must happen in
        teardown, after the audio is drained, not right after send.
        """
        closed_at_first_receive: list[bool] = []
        stream = _FakeStream(
            [_payload_event(PCM_100MS), _control_frame(type="synthesis_complete")],
            on_receive=lambda _n: closed_at_first_receive.append(stream.input_stream.closed),
        )
        fake_ctor = _FakeBidiClientCtor(stream)

        with patch("tts_client._bidi_transport.SageMakerRuntimeHTTP2Client", fake_ctor):
            client = TTSClient()
            client.synthesize_bidi(
                "speech-kokoro-82m", SynthesisRequest(text="hello", voice="af_heart")
            )

        assert closed_at_first_receive, "the output stream was never read"
        assert not closed_at_first_receive[0], (
            "the input half was closed before the first output read; SageMaker "
            "drops the session before the container reads the payload"
        )

    def test_input_stream_is_closed_by_the_time_the_call_returns(self) -> None:
        # A ladder that leaks a half-open input per request runs out of
        # connections before its highest step.
        stream = _FakeStream([_payload_event(PCM_100MS), _control_frame(type="synthesis_complete")])
        fake_ctor = _FakeBidiClientCtor(stream)

        with patch("tts_client._bidi_transport.SageMakerRuntimeHTTP2Client", fake_ctor):
            client = TTSClient()
            client.synthesize_bidi(
                "speech-kokoro-82m", SynthesisRequest(text="hello", voice="af_heart")
            )

        assert stream.input_stream.closed

    def test_undecodable_chunk_starting_with_a_brace_is_treated_as_audio(self) -> None:
        # PCM can legitimately start with byte 0x7b ('{'). Treating such a
        # chunk as a control frame would silently drop real audio and
        # understate throughput at exactly the rates where every sample
        # matters.
        pcm = b"{" + b"\xff\xfe" * 100
        events = [_payload_event(pcm), _control_frame(type="synthesis_complete")]
        fake_ctor = _FakeBidiClientCtor(_FakeStream(events))

        with patch("tts_client._bidi_transport.SageMakerRuntimeHTTP2Client", fake_ctor):
            client = TTSClient()
            result = client.synthesize_bidi(
                "speech-kokoro-82m", SynthesisRequest(text="hello", voice="af_heart")
            )

        assert result.chunks == 1
        assert len(result.audio_bytes) == len(pcm) + 44  # WAV header

    def test_decodable_dict_without_a_known_type_is_a_control_frame_not_audio(self) -> None:
        # A decodable dict starting with `{` is treated as a control frame
        # even without a recognized `type` -- it's silently skipped (the
        # `continue` after the frame_type checks), not counted as audio.
        events = [_payload_event(b'{"just": "text"}'), _control_frame(type="synthesis_complete")]
        fake_ctor = _FakeBidiClientCtor(_FakeStream(events))

        with patch("tts_client._bidi_transport.SageMakerRuntimeHTTP2Client", fake_ctor):
            client = TTSClient()
            with pytest.raises(TTSClientError, match="no audio bytes"):
                client.synthesize_bidi(
                    "speech-kokoro-82m", SynthesisRequest(text="hello", voice="af_heart")
                )


# --------------------------------------------------------------------------- #
# Retries flag
# --------------------------------------------------------------------------- #


class TestRetriesFlag:
    """The one constructor knob, and the one with a proven correctness reason:
    Kokoro's admission-gate 503 is one of botocore's retryable statuses (see
    the client module docstring), so a caller measuring capacity must disable
    retries or a genuine rejection can be retried into a false success."""

    def test_default_leaves_boto3_retries_on(self) -> None:
        client = TTSClient()
        config = client._client.meta.config  # type: ignore[attr-defined]
        # botocore normalizes an unset retries config to just {"mode": ...},
        # with no total_max_attempts cap -- i.e. its own default behavior.
        assert "total_max_attempts" not in config.retries

    def test_retries_false_disables_botocore_retries(self) -> None:
        client = TTSClient(retries=False)
        config = client._client.meta.config  # type: ignore[attr-defined]
        # max_attempts=0 (zero retries) normalizes to total_max_attempts=1
        # (one attempt total, no retry).
        assert config.retries["total_max_attempts"] == 1


class TestMultipleInstances:
    def test_two_instances_are_independent(self) -> None:
        """Making several TTSClient() instances is the documented, supported
        way to get concurrency without any pool-size decision."""
        a = TTSClient()
        b = TTSClient()
        assert a is not b
        assert a._client is not b._client  # type: ignore[attr-defined]


class TestErrorHttpStatus:
    def test_client_error_carries_http_status(self) -> None:
        try:
            raise QueueSaturatedError("boom", http_status=503)
        except TTSClientError as e:
            assert e.http_status == 503
