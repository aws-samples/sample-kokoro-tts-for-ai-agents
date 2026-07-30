"""Tests for the bidirectional-streaming transport.

Two things are being protected here, and neither is the wire format.

The first is **outcome classification**. On the response-stream path saturation
is an HTTP status code; on bidi it is a JSON frame inside a stream that already
succeeded. ``tts_eval/bidi_client.py:140-146`` logs such a frame and breaks,
which for a benchmark means a rejected request is counted as a short success —
the exact failure that makes an overloaded endpoint look healthy and produces a
C_max that is too high. Every frame type therefore has a test.

The second is **comparability with the other transport**. A ``C_max`` measured
on bidi is only interpretable next to one measured on response-stream if both
populate ``InvokeResult`` the same way, so there is a test asserting the two
agree field by field on an equivalent successful stream.

The coroutines are driven through ``asyncio.run`` rather than ``pytest-asyncio``
(not a dependency here), which is also how ``loadgen`` calls them: one loop per
session, in the worker thread that owns it.
"""

from __future__ import annotations

import asyncio
import json
import struct
import time
from typing import Any
from unittest.mock import MagicMock

import pytest

from tts_bench.bidi import (
    BIDI_PORT,
    BIDI_SAMPLE_RATE,
    BidiTransportError,
    Boto3CredentialsResolver,
    Transport,
    _classify_exception,
    build_bidi_message,
    classify_error_frame,
    invoke_bidi,
    invoke_bidi_async,
    invoke_for,
    make_bidi_client,
    make_client_for,
)
from tts_bench.invoke import DEFAULT_READ_TIMEOUT_S, InvokeOutcome, invoke_stream

ENDPOINT = "speech-kokoro-82m"

#: 0.1s of 16-bit mono PCM at 24kHz, the shape the containers actually send.
PCM_100MS = b"\x00\x01" * 2400


# --------------------------------------------------------------------------- #
# Fake bidi stream
# --------------------------------------------------------------------------- #


def _payload_event(data: bytes) -> Any:
    from aws_sdk_sagemaker_runtime_http2.models import (
        ResponsePayloadPart,
        ResponseStreamEventPayloadPart,
    )

    return ResponseStreamEventPayloadPart(value=ResponsePayloadPart(bytes_=data))


def _model_stream_error(message: str = "boom") -> Any:
    from aws_sdk_sagemaker_runtime_http2.models import (
        ModelStreamError,
        ResponseStreamEventModelStreamError,
    )

    return ResponseStreamEventModelStreamError(value=ModelStreamError(message))


def _internal_stream_failure(message: str = "infra blip") -> Any:
    from aws_sdk_sagemaker_runtime_http2.models import (
        InternalStreamFailure,
        ResponseStreamEventInternalStreamFailure,
    )

    return ResponseStreamEventInternalStreamFailure(value=InternalStreamFailure(message))


def _frame(**fields: Any) -> Any:
    """A JSON control frame, as the containers interleave with PCM."""
    return _payload_event(json.dumps(fields).encode("utf-8"))


class FakeInputStream:
    """Records what the benchmark sent, so payload contents are assertable."""

    def __init__(self) -> None:
        self.sent: list[Any] = []
        self.closed = False

    async def send(self, event: Any) -> None:
        self.sent.append(event)

    async def close(self) -> None:
        self.closed = True

    @property
    def messages(self) -> list[dict]:
        return [json.loads(e.value.bytes_.decode("utf-8")) for e in self.sent]


class FakeOutputStream:
    """Yields scripted events, then ``None`` for end-of-stream."""

    def __init__(self, events: list[Any], *, on_receive: Any = None) -> None:
        self._events = list(events)
        self._on_receive = on_receive
        self.receives = 0

    async def receive(self) -> Any:
        self.receives += 1
        if self._on_receive is not None:
            self._on_receive(self.receives)
        if not self._events:
            return None
        return self._events.pop(0)


class FakeStream:
    """Stands in for the SDK's duplex stream handle."""

    def __init__(
        self,
        events: list[Any] | None = None,
        *,
        on_receive: Any = None,
        send_error: BaseException | None = None,
        close_error: BaseException | None = None,
    ) -> None:
        self.input_stream = FakeInputStream()
        self.output_stream = FakeOutputStream(events or [], on_receive=on_receive)
        self._send_error = send_error
        self._close_error = close_error
        self.closed = False
        self.awaited_output = False

        if send_error is not None:

            async def _raise(_event: Any) -> None:
                raise send_error

            self.input_stream.send = _raise  # type: ignore[method-assign]

    async def await_output(self) -> tuple[Any, FakeOutputStream]:
        self.awaited_output = True
        return (object(), self.output_stream)

    async def close(self) -> None:
        self.closed = True
        if self._close_error is not None:
            raise self._close_error


class FakeBidiClient:
    """Client whose ``invoke_endpoint_with_bidirectional_stream`` is scripted."""

    def __init__(self, stream: FakeStream | None = None, *, error: BaseException | None = None):
        self.stream = stream if stream is not None else FakeStream([_payload_event(PCM_100MS)])
        self._error = error
        self.inputs: list[Any] = []

    async def invoke_endpoint_with_bidirectional_stream(self, input_: Any) -> FakeStream:
        self.inputs.append(input_)
        if self._error is not None:
            raise self._error
        return self.stream


def _run(
    client: FakeBidiClient,
    *,
    text: str = "hello world",
    voice: str = "af_heart",
    deadline_ts: float | None = None,
) -> Any:
    return asyncio.run(invoke_bidi_async(client, ENDPOINT, text, voice, deadline_ts=deadline_ts))


# --------------------------------------------------------------------------- #
# Transport enum
# --------------------------------------------------------------------------- #


class TestTransport:
    def test_values_match_the_cli_choices(self) -> None:
        # The CLI passes its raw --transport string straight into Transport(),
        # so a rename here silently breaks the flag.
        assert {t.value for t in Transport} == {"response-stream", "bidi"}

    def test_is_a_str_enum_so_it_serializes_into_the_artifact(self) -> None:
        # CMaxReport.transport is a plain str field; str(Transport.BIDI) must be
        # the wire value, not "Transport.BIDI".
        assert str(Transport.BIDI) == "bidi"
        assert f"{Transport.RESPONSE_STREAM}" == "response-stream"

    def test_accepts_its_own_value_back(self) -> None:
        assert Transport("bidi") is Transport.BIDI


class TestInvokeFor:
    def test_bidi_selects_the_bidi_transport(self) -> None:
        assert invoke_for("bidi") is invoke_bidi
        assert invoke_for(Transport.BIDI) is invoke_bidi

    def test_response_stream_stays_the_default_path(self) -> None:
        assert invoke_for("response-stream") is invoke_stream

    def test_rejects_an_unknown_transport(self) -> None:
        with pytest.raises(ValueError):
            invoke_for("grpc")


# --------------------------------------------------------------------------- #
# Client construction
# --------------------------------------------------------------------------- #


class TestMakeBidiClient:
    def test_disables_retries(self) -> None:
        # Same reasoning as make_runtime_client: a retry fabricates load exactly
        # when the server is saturated. smithy counts *total* attempts, so 1
        # means one attempt and no retry (botocore's max_attempts=0).
        client = make_bidi_client("us-east-1")
        assert client._config.retry_strategy.max_attempts == 1

    def test_targets_the_bidirectional_port(self) -> None:
        # 8443, not 443 — bidirectional streaming is served on its own port.
        client = make_bidi_client("us-west-2")
        assert client._config.endpoint_uri == (
            f"https://runtime.sagemaker.us-west-2.amazonaws.com:{BIDI_PORT}"
        )
        assert client._config.region == "us-west-2"

    def test_read_timeout_exceeds_sagemaker_ceiling(self) -> None:
        # The server's 60s limit must be the binding one, so a server-side
        # timeout is not misattributed to the client.
        client = make_bidi_client("us-east-1")
        assert client._config.http_request_config.read_timeout == DEFAULT_READ_TIMEOUT_S
        assert DEFAULT_READ_TIMEOUT_S > 60.0

    def test_uses_the_boto3_credential_resolver(self) -> None:
        # Not EnvironmentCredentialsResolver: it raises on an IAM-role box, which
        # is how this benchmark runs.
        client = make_bidi_client("us-east-1")
        assert isinstance(
            client._config.aws_credentials_identity_resolver, Boto3CredentialsResolver
        )


class TestMakeClientFor:
    def test_bidi_builds_the_http2_client(self) -> None:
        from aws_sdk_sagemaker_runtime_http2.client import SageMakerRuntimeHTTP2Client

        client = make_client_for("bidi", "us-east-1", max_pool=64)
        assert isinstance(client, SageMakerRuntimeHTTP2Client)

    def test_max_pool_is_accepted_and_ignored_for_bidi(self) -> None:
        # HTTP/2 multiplexes over one connection, so there is no pool to size.
        # Accepting the argument keeps callers from having to branch.
        client = make_client_for("bidi", "us-east-1", max_pool=1)
        assert client is not None

    def test_response_stream_builds_the_boto3_client_with_the_pool(self) -> None:
        client = make_client_for(Transport.RESPONSE_STREAM, "us-east-1", max_pool=128)
        assert client.meta.config.max_pool_connections == 128
        assert client.meta.config.retries["total_max_attempts"] == 1


# --------------------------------------------------------------------------- #
# Credentials
# --------------------------------------------------------------------------- #


class _FrozenCreds:
    def __init__(self, key: str, secret: str, token: str | None) -> None:
        self.access_key = key
        self.secret_key = secret
        self.token = token


class _RotatingCredentials:
    """Mimics botocore's refreshable credentials: a new tuple per freeze."""

    def __init__(self) -> None:
        self.freezes = 0

    def get_frozen_credentials(self) -> _FrozenCreds:
        self.freezes += 1
        return _FrozenCreds(f"AKIA{self.freezes}", f"secret{self.freezes}", f"token{self.freezes}")


class TestBoto3CredentialsResolver:
    def test_resolves_through_boto3s_chain(self) -> None:
        session = MagicMock()
        session.get_credentials.return_value = _RotatingCredentials()
        identity = asyncio.run(Boto3CredentialsResolver(session=session).get_identity())

        assert identity.access_key_id == "AKIA1"
        assert identity.secret_access_key == "secret1"
        # The session token is what makes role credentials work at all; dropping
        # it produces a signature the service rejects.
        assert identity.session_token == "token1"

    def test_refreezes_on_every_resolve_rather_than_caching(self) -> None:
        # This is the whole reason the class exists. A cmax ladder runs ~45
        # minutes and role credentials rotate inside that window; a cached tuple
        # starts failing at the high-rate steps that run last, where a wave of
        # auth failures is most easily misread as saturation.
        credentials = _RotatingCredentials()
        session = MagicMock()
        session.get_credentials.return_value = credentials
        resolver = Boto3CredentialsResolver(session=session)

        first = asyncio.run(resolver.get_identity())
        second = asyncio.run(resolver.get_identity())

        assert credentials.freezes == 2
        assert first.access_key_id != second.access_key_id

    def test_re_reads_the_session_each_time_so_a_new_role_is_picked_up(self) -> None:
        session = MagicMock()
        session.get_credentials.return_value = _RotatingCredentials()
        resolver = Boto3CredentialsResolver(session=session)

        asyncio.run(resolver.get_identity())
        asyncio.run(resolver.get_identity())

        assert session.get_credentials.call_count == 2

    def test_missing_credentials_raise_rather_than_measure(self) -> None:
        # An unauthenticated run is a setup fault, not a capacity finding: it
        # must not be folded into the run's error counts as if the endpoint
        # had rejected the request.
        session = MagicMock()
        session.get_credentials.return_value = None
        with pytest.raises(BidiTransportError, match="no AWS credentials"):
            asyncio.run(Boto3CredentialsResolver(session=session).get_identity())


# --------------------------------------------------------------------------- #
# Request payload
# --------------------------------------------------------------------------- #


class TestBuildBidiMessage:
    def test_includes_request_timestamp_for_the_containers_staleness_check(self) -> None:
        # No container's bidi handler reads this today (kokoro checks it on the
        # HTTP path only, serve.py:276), but Phase 6's dequeue-time re-check
        # needs the field present on both transports the day it lands.
        message = json.loads(build_bidi_message("hi", "af_heart", request_ts=1234.5))
        assert message["request_timestamp"] == 1234.5

    def test_carries_text_and_voice(self) -> None:
        message = json.loads(build_bidi_message("hello", "af_bella", request_ts=1.0))
        assert message["text"] == "hello"
        assert message["voice"] == "af_bella"

    def test_generates_a_request_id_when_none_is_given(self) -> None:
        # The containers echo request_id back on every control frame, which is
        # what makes a container log line traceable to a benchmark request.
        first = json.loads(build_bidi_message("hi", "af_heart", request_ts=1.0))["request_id"]
        second = json.loads(build_bidi_message("hi", "af_heart", request_ts=1.0))["request_id"]
        assert first != second

    def test_honors_an_explicit_request_id(self) -> None:
        message = json.loads(
            build_bidi_message("hi", "af_heart", request_ts=1.0, request_id="fixed-1")
        )
        assert message["request_id"] == "fixed-1"

    def test_is_utf8_bytes(self) -> None:
        assert isinstance(build_bidi_message("café", "af_heart", request_ts=1.0), bytes)


# --------------------------------------------------------------------------- #
# Error-frame classification
# --------------------------------------------------------------------------- #


class TestClassifyErrorFrame:
    @pytest.mark.parametrize(
        ("message", "expected"),
        [
            ("queue_saturated", InvokeOutcome.SATURATED_503),
            ("queue_full", InvokeOutcome.SATURATED_503),
            ("request_stale", InvokeOutcome.STALE_408),
            ("too_many_requests", InvokeOutcome.THROTTLED_429),
        ],
    )
    def test_maps_capacity_refusals_onto_the_same_outcomes_as_http(
        self, message: str, expected: InvokeOutcome
    ) -> None:
        # These must land in the same buckets the status codes produce, or the
        # two transports' saturation counts cannot be compared.
        assert classify_error_frame(message) is expected

    def test_matches_the_token_inside_prose(self) -> None:
        # vllm/streaming_proxy.py:330 sends the bare token, but the wording is
        # not contractual and other handlers wrap it in a sentence.
        assert classify_error_frame("rejected: queue_saturated (depth 64)") is (
            InvokeOutcome.SATURATED_503
        )

    def test_is_case_insensitive(self) -> None:
        assert classify_error_frame("Queue_Saturated") is InvokeOutcome.SATURATED_503

    def test_unrecognized_message_is_a_server_side_model_error(self) -> None:
        # MODEL_ERROR rather than ERROR: the container answered and reported a
        # failure of its own. ERROR means *our* bug, and misfiling a container
        # failure there would hide it from the saturation analysis.
        assert classify_error_frame("something new exploded") is InvokeOutcome.MODEL_ERROR

    def test_empty_message_still_classifies(self) -> None:
        assert classify_error_frame("") is InvokeOutcome.MODEL_ERROR


# --------------------------------------------------------------------------- #
# Successful stream
# --------------------------------------------------------------------------- #


class TestInvokeBidiSuccess:
    def test_returns_ok_with_audio_metrics(self) -> None:
        client = FakeBidiClient(
            FakeStream([_payload_event(PCM_100MS[:100]), _payload_event(PCM_100MS[100:])])
        )
        result = _run(client)

        assert result.outcome is InvokeOutcome.OK
        assert result.ok
        assert result.chars == len("hello world")
        assert result.audio_bytes == len(PCM_100MS)
        assert result.chunks == 2
        # Bidi sends headerless PCM, so duration comes from the containers'
        # uniform SAMPLE_RATE rather than from a RIFF header.
        assert result.sample_rate == BIDI_SAMPLE_RATE
        assert result.audio_duration_s == pytest.approx(0.1, abs=1e-6)
        assert result.ttfab_ms is not None
        assert result.first_byte_ts is not None

    def test_ttfab_is_the_first_audio_chunk_not_the_last(self) -> None:
        client = FakeBidiClient(
            FakeStream([_payload_event(PCM_100MS[:100]), _payload_event(PCM_100MS[100:])])
        )
        result = _run(client)
        assert result.ttfab_ms is not None
        assert result.ttfab_ms <= result.latency_ms

    def test_control_frames_do_not_start_the_ttfab_clock(self) -> None:
        # synthesis_start arrives before any audio. Timing TTFAB from it would
        # measure acknowledgement rather than audible output — and TTFAB is the
        # quantity the whole ladder's knee is defined on.
        def _slow(_n: int) -> None:
            time.sleep(0.02)

        client = FakeBidiClient(
            FakeStream(
                [
                    _frame(type="synthesis_start", request_id="r1"),
                    _payload_event(PCM_100MS),
                ],
                on_receive=_slow,
            )
        )
        result = _run(client)

        assert result.outcome is InvokeOutcome.OK
        assert result.chunks == 1
        assert result.audio_bytes == len(PCM_100MS)

    def test_completion_frame_does_not_count_as_audio(self) -> None:
        client = FakeBidiClient(
            FakeStream(
                [
                    _payload_event(PCM_100MS),
                    _frame(type="synthesis_complete", request_id="r1", total_duration_s=0.1),
                ]
            )
        )
        result = _run(client)
        assert result.outcome is InvokeOutcome.OK
        assert result.chunks == 1
        assert result.audio_bytes == len(PCM_100MS)

    def test_empty_payload_parts_are_skipped(self) -> None:
        client = FakeBidiClient(FakeStream([_payload_event(b""), _payload_event(PCM_100MS)]))
        result = _run(client)
        assert result.outcome is InvokeOutcome.OK
        assert result.chunks == 1

    def test_sends_exactly_one_request_then_closes_the_input(self) -> None:
        # One session == one request. Leaving the input open would make the
        # container wait on a request that is never coming, and would make a
        # bidi step incomparable with a response-stream step.
        client = FakeBidiClient(FakeStream([_payload_event(PCM_100MS)]))
        before = time.time()
        _run(client, text="hello", voice="af_bella")
        after = time.time()

        assert len(client.stream.input_stream.sent) == 1
        assert client.stream.input_stream.closed
        message = client.stream.input_stream.messages[0]
        assert message["text"] == "hello"
        assert message["voice"] == "af_bella"
        # Actual send time, not a deadline: the containers compute age as
        # time.time() - request_timestamp, so a future value would make the age
        # negative and silently disable their staleness check.
        assert before <= message["request_timestamp"] <= after

    def test_targets_the_requested_endpoint(self) -> None:
        client = FakeBidiClient(FakeStream([_payload_event(PCM_100MS)]))
        _run(client)
        assert client.inputs[0].endpoint_name == ENDPOINT

    def test_closes_the_stream_on_the_happy_path(self) -> None:
        # A ladder that leaks one stream per request exhausts the connection
        # before it reaches its highest step.
        client = FakeBidiClient(FakeStream([_payload_event(PCM_100MS)]))
        _run(client)
        assert client.stream.closed


# --------------------------------------------------------------------------- #
# Failures
# --------------------------------------------------------------------------- #


class TestInvokeBidiFailures:
    def test_error_frame_is_a_failure_not_a_short_success(self) -> None:
        # The defect this transport exists to avoid: bidi_client.py:140-146
        # logs a warning and breaks, so a rejected request lands in the results
        # as a fast OK — which is what makes an overloaded endpoint look fast.
        client = FakeBidiClient(
            FakeStream([_frame(type="error", request_id="r1", message="queue_saturated")])
        )
        result = _run(client)

        assert result.outcome is InvokeOutcome.SATURATED_503
        assert not result.ok
        assert result.error_class == "ErrorFrame"
        assert result.error_message == "queue_saturated"

    def test_error_frame_after_partial_audio_keeps_the_partial_measurement(self) -> None:
        # A request that streamed for a while and then failed is a different
        # capacity signal from one refused immediately; zeroing the fields
        # would erase the distinction.
        client = FakeBidiClient(
            FakeStream(
                [
                    _payload_event(PCM_100MS),
                    _frame(type="error", request_id="r1", message="queue_saturated"),
                ]
            )
        )
        result = _run(client)

        assert result.outcome is InvokeOutcome.SATURATED_503
        assert result.chunks == 1
        assert result.audio_bytes == len(PCM_100MS)
        assert result.ttfab_ms is not None

    def test_stops_reading_at_the_error_frame(self) -> None:
        stream = FakeStream(
            [
                _frame(type="error", request_id="r1", message="queue_full"),
                _payload_event(PCM_100MS),
            ]
        )
        result = _run(FakeBidiClient(stream))
        assert result.outcome is InvokeOutcome.SATURATED_503
        assert stream.output_stream.receives == 1

    def test_mid_stream_model_stream_error(self) -> None:
        # The caa4dcf shape on this transport: the stream was accepted, then the
        # generator raised inside the container.
        client = FakeBidiClient(
            FakeStream(
                [
                    _payload_event(PCM_100MS),
                    _model_stream_error("AttributeError astype"),
                ]
            )
        )
        result = _run(client)

        assert result.outcome is InvokeOutcome.SERVER_5XX
        assert result.error_class == "ModelStreamError"
        assert result.ttfab_ms is not None
        assert result.chunks == 1

    def test_mid_stream_internal_failure(self) -> None:
        client = FakeBidiClient(FakeStream([_internal_stream_failure()]))
        result = _run(client)
        assert result.outcome is InvokeOutcome.SERVER_5XX
        assert result.error_class == "InternalStreamFailure"

    def test_stream_with_no_audio_is_a_model_error(self) -> None:
        # Distinct from a rejection: the endpoint accepted the work and returned
        # nothing, which no outcome count would otherwise reveal.
        client = FakeBidiClient(FakeStream([]))
        result = _run(client)
        assert result.outcome is InvokeOutcome.MODEL_ERROR
        assert result.error_class == "EmptyResponse"

    def test_control_frames_only_is_a_model_error(self) -> None:
        client = FakeBidiClient(
            FakeStream(
                [
                    _frame(type="synthesis_start", request_id="r1"),
                    _frame(type="synthesis_complete", request_id="r1"),
                ]
            )
        )
        result = _run(client)
        assert result.outcome is InvokeOutcome.MODEL_ERROR
        assert result.error_class == "EmptyResponse"

    def test_deadline_abandons_mid_stream(self) -> None:
        client = FakeBidiClient(
            FakeStream([_payload_event(PCM_100MS[:100]), _payload_event(PCM_100MS[100:])])
        )
        result = _run(client, deadline_ts=0.0)
        assert result.outcome is InvokeOutcome.CLIENT_TIMEOUT
        assert not result.ok

    def test_deadline_timeout_keeps_the_partial_audio(self) -> None:
        client = FakeBidiClient(FakeStream([_payload_event(PCM_100MS)]))
        result = _run(client, deadline_ts=0.0)
        assert result.audio_bytes == len(PCM_100MS)
        assert result.audio_duration_s > 0

    def test_open_failure_is_classified_not_raised(self) -> None:
        from aws_sdk_sagemaker_runtime_http2.models import ModelError

        client = FakeBidiClient(error=ModelError("full", original_status_code=503))
        result = _run(client)
        assert result.outcome is InvokeOutcome.SATURATED_503
        assert result.error_class == "ModelError"

    def test_send_failure_is_classified_not_raised(self) -> None:
        from aws_sdk_sagemaker_runtime_http2.models import InternalServerError

        client = FakeBidiClient(FakeStream(send_error=InternalServerError("nope")))
        result = _run(client)
        assert result.outcome is InvokeOutcome.SERVER_5XX

    def test_a_close_failure_does_not_lose_the_result(self) -> None:
        # The request already has an outcome by teardown; losing it to a close
        # fault would silently remove a sample from the run.
        client = FakeBidiClient(
            FakeStream([_payload_event(PCM_100MS)], close_error=RuntimeError("closed twice"))
        )
        result = _run(client)
        assert result.outcome is InvokeOutcome.OK

    def test_stream_is_closed_after_an_abandoned_request(self) -> None:
        client = FakeBidiClient(FakeStream([_payload_event(PCM_100MS)]))
        _run(client, deadline_ts=0.0)
        assert client.stream.closed

    def test_failures_still_record_chars_and_timing(self) -> None:
        # Reconciliation depends on this: the totals must account for every
        # scheduled request, including the failed ones.
        from aws_sdk_sagemaker_runtime_http2.models import ServiceUnavailableError

        client = FakeBidiClient(error=ServiceUnavailableError("busy"))
        result = _run(client, text="hello world")
        assert result.chars == len("hello world")
        assert result.latency_ms >= 0
        assert result.end_ts >= result.dispatch_ts


class TestPcmThatLooksLikeJson:
    def test_undecodable_chunk_starting_with_a_brace_is_treated_as_audio(self) -> None:
        # PCM can legitimately start with byte 0x7b ('{'). Treating such a chunk
        # as a control frame would silently drop real audio and understate
        # throughput at exactly the rates where every sample matters.
        pcm = b"{" + b"\xff\xfe" * 100
        client = FakeBidiClient(FakeStream([_payload_event(pcm)]))
        result = _run(client)

        assert result.outcome is InvokeOutcome.OK
        assert result.audio_bytes == len(pcm)
        assert result.chunks == 1

    def test_valid_json_that_is_not_an_object_is_treated_as_audio(self) -> None:
        # b"[1,2]" decodes but is not a control frame; only dicts are.
        client = FakeBidiClient(FakeStream([_payload_event(b'{"just": "text"}')]))
        result = _run(client)
        # A decodable dict without type=error is a control frame and yields no
        # audio, hence EmptyResponse rather than OK.
        assert result.outcome is InvokeOutcome.MODEL_ERROR


# --------------------------------------------------------------------------- #
# Exception mapping
# --------------------------------------------------------------------------- #


class TestClassifyException:
    @pytest.mark.parametrize(
        ("original_status", "expected"),
        [
            (408, InvokeOutcome.STALE_408),
            (429, InvokeOutcome.THROTTLED_429),
            (503, InvokeOutcome.SATURATED_503),
            (500, InvokeOutcome.SERVER_5XX),
            (502, InvokeOutcome.SERVER_5XX),
            (400, InvokeOutcome.MODEL_ERROR),
        ],
    )
    def test_unwraps_the_containers_status_from_model_error(
        self, original_status: int, expected: InvokeOutcome
    ) -> None:
        # Same reasoning as the response-stream path: without unwrapping, queue
        # full / too old / crashed all collapse into one bucket.
        from aws_sdk_sagemaker_runtime_http2.models import ModelError

        outcome, name = _classify_exception(ModelError("x", original_status_code=original_status))
        assert outcome is expected
        assert name == "ModelError"

    def test_model_error_without_a_status_stays_a_model_error(self) -> None:
        from aws_sdk_sagemaker_runtime_http2.models import ModelError

        assert _classify_exception(ModelError("x"))[0] is InvokeOutcome.MODEL_ERROR

    @pytest.mark.parametrize(
        ("factory", "expected"),
        [
            ("ServiceUnavailableError", InvokeOutcome.SATURATED_503),
            ("InternalServerError", InvokeOutcome.SERVER_5XX),
            ("InternalStreamFailure", InvokeOutcome.SERVER_5XX),
            ("ModelStreamError", InvokeOutcome.SERVER_5XX),
        ],
    )
    def test_maps_modeled_sdk_errors(self, factory: str, expected: InvokeOutcome) -> None:
        from aws_sdk_sagemaker_runtime_http2 import models

        assert _classify_exception(getattr(models, factory)("x"))[0] is expected

    def test_validation_error_is_our_bug_not_saturation(self) -> None:
        # A malformed payload must never read as a capacity finding.
        from aws_sdk_sagemaker_runtime_http2.models import InputValidationError

        assert _classify_exception(InputValidationError("bad"))[0] is InvokeOutcome.ERROR

    def test_asyncio_timeout_is_a_client_timeout(self) -> None:
        assert _classify_exception(TimeoutError())[0] is InvokeOutcome.CLIENT_TIMEOUT

    def test_throttling_flag_is_the_fallback_for_unnamed_errors(self) -> None:
        class Weird(Exception):
            is_throttling_error = True

        assert _classify_exception(Weird())[0] is InvokeOutcome.THROTTLED_429

    def test_timeout_flag_is_the_fallback_for_unnamed_errors(self) -> None:
        class Weird(Exception):
            is_timeout_error = True

        assert _classify_exception(Weird())[0] is InvokeOutcome.CLIENT_TIMEOUT

    def test_the_named_taxonomy_outranks_the_generic_flags(self) -> None:
        # Every modeled SDK error defaults both flags to False, so consulting
        # them first would let a generic hint override a specific mapping.
        from aws_sdk_sagemaker_runtime_http2.models import ServiceUnavailableError

        exc = ServiceUnavailableError("busy")
        assert exc.is_throttling_error is False
        assert _classify_exception(exc)[0] is InvokeOutcome.SATURATED_503

    def test_unknown_exception_is_error_not_silently_ok(self) -> None:
        outcome, name = _classify_exception(RuntimeError("who knows"))
        assert outcome is InvokeOutcome.ERROR
        assert name == "RuntimeError"


# --------------------------------------------------------------------------- #
# Cross-transport comparability
# --------------------------------------------------------------------------- #


def _wav(n_samples: int = 2400, sample_rate: int = BIDI_SAMPLE_RATE) -> bytes:
    """Minimal 16-bit mono WAV, matching PCM_100MS's audio content."""
    data = b"\x00\x01" * n_samples
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + len(data),
        b"WAVE",
        b"fmt ",
        16,
        1,
        1,
        sample_rate,
        sample_rate * 2,
        2,
        16,
        b"data",
        len(data),
    )
    return header + data


class TestCrossTransportComparability:
    """A C_max on one transport is only readable next to one on the other.

    Both paths must populate ``InvokeResult`` identically for equivalent audio,
    because ``loadgen`` and ``cmax`` consume these fields without knowing which
    transport produced them.
    """

    def test_equivalent_successful_streams_agree_field_by_field(self) -> None:
        text = "hello world"
        bidi = _run(FakeBidiClient(FakeStream([_payload_event(PCM_100MS)])), text=text)

        rs_client = MagicMock()
        rs_client.invoke_endpoint_with_response_stream.return_value = {
            "Body": iter([{"PayloadPart": {"Bytes": _wav()}}])
        }
        rs = invoke_stream(rs_client, ENDPOINT, text, "af_heart")

        assert bidi.outcome is rs.outcome is InvokeOutcome.OK
        assert bidi.chars == rs.chars
        assert bidi.chunks == rs.chunks
        assert bidi.sample_rate == rs.sample_rate
        # Identical audio content; bidi omits the 44-byte RIFF header.
        assert bidi.audio_duration_s == pytest.approx(rs.audio_duration_s, abs=1e-6)
        assert bidi.audio_bytes == rs.audio_bytes - 44

    def test_the_same_optional_fields_are_populated_on_success(self) -> None:
        bidi = _run(FakeBidiClient(FakeStream([_payload_event(PCM_100MS)])))

        rs_client = MagicMock()
        rs_client.invoke_endpoint_with_response_stream.return_value = {
            "Body": iter([{"PayloadPart": {"Bytes": _wav()}}])
        }
        rs = invoke_stream(rs_client, ENDPOINT, "hello world", "af_heart")

        for field in ("ttfab_ms", "first_byte_ts", "error_class", "error_message"):
            assert (getattr(bidi, field) is None) == (getattr(rs, field) is None), field

    def test_both_transports_report_saturation_as_the_same_outcome(self) -> None:
        # One arrives as an HTTP status, the other as a JSON frame. If these
        # diverged, a bidi ladder's knee could not be compared with a
        # response-stream one.
        import botocore.exceptions

        bidi = _run(
            FakeBidiClient(
                FakeStream([_frame(type="error", request_id="r", message="queue_saturated")])
            )
        )

        rs_client = MagicMock()
        rs_client.invoke_endpoint_with_response_stream.side_effect = (
            botocore.exceptions.ClientError(
                {
                    "Error": {"Code": "ModelError", "Message": "503"},
                    "ResponseMetadata": {"HTTPStatusCode": 424},
                    "OriginalStatusCode": 503,
                },
                "InvokeEndpointWithResponseStream",
            )
        )
        rs = invoke_stream(rs_client, ENDPOINT, "hello world", "af_heart")

        assert bidi.outcome is rs.outcome is InvokeOutcome.SATURATED_503

    def test_rtf_is_computable_on_the_bidi_path(self) -> None:
        # RTF depends on audio_duration_s, which bidi derives from SAMPLE_RATE
        # rather than a header — if that were zero, RTF would silently be None
        # for every bidi request.
        result = _run(FakeBidiClient(FakeStream([_payload_event(PCM_100MS)])))
        assert result.rtf is not None


# --------------------------------------------------------------------------- #
# Sync wrapper
# --------------------------------------------------------------------------- #


class TestInvokeBidiSync:
    def test_matches_invoke_streams_signature(self) -> None:
        # loadgen injects one or the other as `invoke=`, positionally, with
        # deadline_ts by keyword. A signature drift would only show up at
        # runtime, mid-ladder.
        import inspect

        assert list(inspect.signature(invoke_bidi).parameters) == list(
            inspect.signature(invoke_stream).parameters
        )

    def test_runs_a_session_to_completion(self) -> None:
        client = FakeBidiClient(FakeStream([_payload_event(PCM_100MS)]))
        result = invoke_bidi(client, ENDPOINT, "hello world", "af_heart")
        assert result.outcome is InvokeOutcome.OK
        assert result.audio_bytes == len(PCM_100MS)

    def test_each_call_gets_its_own_loop(self) -> None:
        # One shared loop would serialize sessions behind whichever is furthest
        # behind, re-creating the closed-loop behavior loadgen exists to avoid.
        # Captured from inside the session, so it is the loop that actually ran
        # the request rather than one a helper created afterwards.
        loops: list[Any] = []

        def _record(_n: int) -> None:
            loops.append(asyncio.get_running_loop())

        for _ in range(2):
            client = FakeBidiClient(FakeStream([_payload_event(PCM_100MS)], on_receive=_record))
            assert invoke_bidi(client, ENDPOINT, "hi", "af_heart").ok

        assert len(loops) >= 2
        assert loops[0] is not loops[-1]

    def test_works_from_a_worker_thread(self) -> None:
        # loadgen dispatches into a ThreadPoolExecutor, so asyncio.run must be
        # reached from a thread with no loop of its own.
        from concurrent.futures import ThreadPoolExecutor

        def _one() -> Any:
            client = FakeBidiClient(FakeStream([_payload_event(PCM_100MS)]))
            return invoke_bidi(client, ENDPOINT, "hi", "af_heart")

        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(lambda _i: _one(), range(4)))

        assert all(r.outcome is InvokeOutcome.OK for r in results)
