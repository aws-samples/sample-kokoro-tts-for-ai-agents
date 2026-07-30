"""Tests for the benchmarking invocation path.

The classification tests matter more than they look: every capacity conclusion
is built on counting outcomes, so a request landing in the wrong bucket turns
into a wrong C_max, which turns into a wrong fleet size.
"""

from __future__ import annotations

import json
import struct
from unittest.mock import MagicMock

import botocore.exceptions
import pytest

from tts_bench.invoke import (
    DEFAULT_READ_TIMEOUT_S,
    InvokeOutcome,
    InvokeResult,
    build_payload,
    classify_client_error,
    invoke_stream,
    make_runtime_client,
    resolve_endpoint,
    resolve_voice,
)
from tts_inference.types import TTSModelName


def _wav(n_samples: int = 2400, sample_rate: int = 24000) -> bytes:
    """Minimal valid 16-bit mono WAV; 2400 samples at 24kHz = 0.1s."""
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


def _client_error(
    code: str,
    *,
    http_status: int | None = None,
    original_status: int | None = None,
) -> botocore.exceptions.ClientError:
    response: dict = {"Error": {"Code": code, "Message": f"{code} happened"}}
    if http_status is not None:
        response["ResponseMetadata"] = {"HTTPStatusCode": http_status}
    if original_status is not None:
        response["OriginalStatusCode"] = original_status
    return botocore.exceptions.ClientError(response, "InvokeEndpointWithResponseStream")


def _streaming_client(events: list[dict]) -> MagicMock:
    client = MagicMock()
    client.invoke_endpoint_with_response_stream.return_value = {"Body": iter(events)}
    return client


class TestMakeRuntimeClient:
    def test_disables_retries(self) -> None:
        # Retries would fabricate load precisely when the server is saturated.
        # botocore normalizes max_attempts=0 to total_max_attempts=1, i.e. one
        # attempt and no retry — assert the resolved semantics, not the input key.
        client = make_runtime_client("us-east-1", max_pool=64)
        assert client.meta.config.retries["total_max_attempts"] == 1

    def test_honors_pool_size(self) -> None:
        # The default of 10 silently serializes above ten in-flight streams,
        # which would make the client the measured bottleneck.
        client = make_runtime_client("us-east-1", max_pool=128)
        assert client.meta.config.max_pool_connections == 128

    def test_read_timeout_exceeds_sagemaker_ceiling(self) -> None:
        # Must sit above the 60s invocation limit so the server's own limit is
        # the binding one and timeouts are not misattributed to us.
        client = make_runtime_client("us-east-1", max_pool=16)
        assert client.meta.config.read_timeout == DEFAULT_READ_TIMEOUT_S
        assert DEFAULT_READ_TIMEOUT_S > 60.0

    def test_region_is_applied(self) -> None:
        client = make_runtime_client("us-west-2", max_pool=16)
        assert client.meta.region_name == "us-west-2"


class TestBuildPayload:
    def test_sends_actual_send_time_as_request_timestamp(self) -> None:
        # Containers compute age as time.time() - request_timestamp. A future
        # value makes the age negative and disables their staleness check.
        payload = json.loads(build_payload("hello", "af_heart", request_ts=1234.5))
        assert payload == {"text": "hello", "voice": "af_heart", "request_timestamp": 1234.5}

    def test_is_utf8_bytes(self) -> None:
        assert isinstance(build_payload("café", "af_heart", request_ts=1.0), bytes)


class TestClassifyClientError:
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
    def test_unwraps_model_error_original_status(
        self, original_status: int, expected: InvokeOutcome
    ) -> None:
        # SageMaker wraps every container non-2xx in ModelError/424. Classifying
        # on the outer 424 would merge queue-full, too-old, and crashed.
        exc = _client_error("ModelError", http_status=424, original_status=original_status)
        outcome, status = classify_client_error(exc)
        assert outcome is expected
        assert status == original_status

    def test_container_503_is_saturation_not_generic_failure(self) -> None:
        # This is the signal admission control produces under overload; it must
        # be countable on its own.
        exc = _client_error("ModelError", http_status=424, original_status=503)
        assert classify_client_error(exc)[0] is InvokeOutcome.SATURATED_503

    def test_container_408_is_staleness_not_saturation(self) -> None:
        exc = _client_error("ModelError", http_status=424, original_status=408)
        assert classify_client_error(exc)[0] is InvokeOutcome.STALE_408

    @pytest.mark.parametrize(
        ("code", "expected"),
        [
            ("ThrottlingException", InvokeOutcome.THROTTLED_429),
            ("TooManyRequestsException", InvokeOutcome.THROTTLED_429),
            ("ServiceUnavailable", InvokeOutcome.SATURATED_503),
            ("InternalFailure", InvokeOutcome.SERVER_5XX),
            ("InternalStreamFailure", InvokeOutcome.SERVER_5XX),
        ],
    )
    def test_maps_sagemaker_error_codes(self, code: str, expected: InvokeOutcome) -> None:
        assert classify_client_error(_client_error(code))[0] is expected

    def test_validation_error_is_our_bug_not_saturation(self) -> None:
        # A malformed payload must never read as a capacity finding.
        outcome, status = classify_client_error(_client_error("ValidationError"))
        assert outcome is InvokeOutcome.ERROR
        assert status == 400

    def test_model_error_without_original_status(self) -> None:
        outcome, _ = classify_client_error(_client_error("ModelError", http_status=424))
        assert outcome is InvokeOutcome.MODEL_ERROR

    def test_falls_back_to_http_status(self) -> None:
        outcome, status = classify_client_error(_client_error("Weird", http_status=503))
        assert outcome is InvokeOutcome.SATURATED_503
        assert status == 503

    def test_unrecognized_is_error_not_silently_ok(self) -> None:
        outcome, _ = classify_client_error(_client_error("SomethingNew", http_status=418))
        assert outcome is InvokeOutcome.ERROR

    def test_tolerates_non_integer_original_status(self) -> None:
        exc = _client_error("ModelError", http_status=424, original_status=None)
        exc.response["OriginalStatusCode"] = "not-a-number"
        outcome, _ = classify_client_error(exc)
        assert outcome is InvokeOutcome.MODEL_ERROR


class TestInvokeStreamSuccess:
    def test_returns_ok_with_audio_metrics(self) -> None:
        audio = _wav()
        client = _streaming_client(
            [{"PayloadPart": {"Bytes": audio[:100]}}, {"PayloadPart": {"Bytes": audio[100:]}}]
        )
        result = invoke_stream(client, "speech-kokoro-82m", "hello world", "af_heart")

        assert result.outcome is InvokeOutcome.OK
        assert result.ok
        assert result.chars == len("hello world")
        assert result.audio_bytes == len(audio)
        assert result.chunks == 2
        assert result.sample_rate == 24000
        assert result.audio_duration_s == pytest.approx(0.1, abs=1e-6)
        assert result.ttfab_ms is not None
        assert result.first_byte_ts is not None

    def test_ttfab_is_first_chunk_not_last(self) -> None:
        audio = _wav()
        client = _streaming_client(
            [{"PayloadPart": {"Bytes": audio[:100]}}, {"PayloadPart": {"Bytes": audio[100:]}}]
        )
        result = invoke_stream(client, "speech-kokoro-82m", "hello", "af_heart")
        assert result.ttfab_ms is not None
        assert result.ttfab_ms <= result.latency_ms

    def test_empty_chunks_do_not_start_the_ttfab_clock(self) -> None:
        # A zero-length part is not audible output; counting it as first byte
        # would understate TTFAB, the metric the SLO is defined on.
        audio = _wav()
        client = _streaming_client(
            [{"PayloadPart": {"Bytes": b""}}, {"PayloadPart": {"Bytes": audio}}]
        )
        result = invoke_stream(client, "speech-kokoro-82m", "hello", "af_heart")
        assert result.outcome is InvokeOutcome.OK
        assert result.chunks == 1

    def test_sends_request_timestamp_in_body(self) -> None:
        client = _streaming_client([{"PayloadPart": {"Bytes": _wav()}}])
        invoke_stream(client, "speech-kokoro-82m", "hello", "af_heart")

        kwargs = client.invoke_endpoint_with_response_stream.call_args.kwargs
        body = json.loads(kwargs["Body"])
        assert "request_timestamp" in body
        assert kwargs["EndpointName"] == "speech-kokoro-82m"
        assert kwargs["Accept"] == "audio/wav"


class TestInvokeStreamFailures:
    def test_never_raises_on_client_error(self) -> None:
        client = MagicMock()
        client.invoke_endpoint_with_response_stream.side_effect = _client_error(
            "ModelError", http_status=424, original_status=503
        )
        result = invoke_stream(client, "speech-kokoro-82m", "hello", "af_heart")
        assert result.outcome is InvokeOutcome.SATURATED_503
        assert result.http_status == 503
        assert result.error_class == "ClientError"

    def test_read_timeout_is_client_timeout(self) -> None:
        client = MagicMock()
        client.invoke_endpoint_with_response_stream.side_effect = (
            botocore.exceptions.ReadTimeoutError(endpoint_url="https://x")
        )
        result = invoke_stream(client, "speech-kokoro-82m", "hello", "af_heart")
        assert result.outcome is InvokeOutcome.CLIENT_TIMEOUT

    def test_connection_pool_exhaustion_is_not_server_saturation(self) -> None:
        # Attributing a client-side pool limit to the server is exactly the
        # mistake make_runtime_client exists to prevent.
        client = MagicMock()
        client.invoke_endpoint_with_response_stream.side_effect = (
            botocore.exceptions.ConnectionError(error="pool is full")
        )
        result = invoke_stream(client, "speech-kokoro-82m", "hello", "af_heart")
        assert result.outcome is InvokeOutcome.ERROR

    def test_mid_stream_model_error_after_http_200(self) -> None:
        # The caa4dcf failure shape: 200 + headers already sent, then the
        # generator raises. Must not be recorded as a success.
        audio = _wav()
        client = _streaming_client(
            [
                {"PayloadPart": {"Bytes": audio[:100]}},
                {"ModelStreamError": {"Message": "AttributeError astype", "ErrorCode": "500"}},
            ]
        )
        result = invoke_stream(client, "speech-kokoro-82m", "hello", "af_heart")

        assert result.outcome is InvokeOutcome.SERVER_5XX
        assert result.error_class == "ModelStreamError"
        assert result.ttfab_ms is not None  # first byte did arrive
        assert result.chunks == 1

    def test_mid_stream_internal_failure(self) -> None:
        client = _streaming_client([{"InternalStreamFailure": {"Message": "infra blip"}}])
        result = invoke_stream(client, "speech-kokoro-82m", "hello", "af_heart")
        assert result.outcome is InvokeOutcome.SERVER_5XX
        assert result.error_class == "InternalStreamFailure"

    def test_http_200_with_no_audio_is_model_error(self) -> None:
        client = _streaming_client([])
        result = invoke_stream(client, "speech-kokoro-82m", "hello", "af_heart")
        assert result.outcome is InvokeOutcome.MODEL_ERROR
        assert result.error_class == "EmptyResponse"

    def test_deadline_abandons_mid_stream(self) -> None:
        audio = _wav()
        client = _streaming_client(
            [{"PayloadPart": {"Bytes": audio[:10]}}, {"PayloadPart": {"Bytes": audio[10:]}}]
        )
        result = invoke_stream(client, "speech-kokoro-82m", "hello", "af_heart", deadline_ts=0.0)
        assert result.outcome is InvokeOutcome.CLIENT_TIMEOUT

    def test_failed_results_still_record_chars_and_timing(self) -> None:
        # Reconciliation depends on this: a run's totals must account for every
        # scheduled request, including the ones that failed.
        client = MagicMock()
        client.invoke_endpoint_with_response_stream.side_effect = _client_error(
            "ServiceUnavailable", http_status=503
        )
        result = invoke_stream(client, "speech-kokoro-82m", "hello world", "af_heart")
        assert result.chars == len("hello world")
        assert result.latency_ms >= 0
        assert result.end_ts >= result.dispatch_ts


class TestInvokeResult:
    def test_rtf_is_none_without_audio(self) -> None:
        # Averaging a failed request in as RTF 0.0 would flatter the result.
        result = InvokeResult(
            outcome=InvokeOutcome.SATURATED_503,
            dispatch_ts=0.0,
            end_ts=1.0,
            latency_ms=1000.0,
        )
        assert result.rtf is None
        assert not result.ok

    def test_rtf_is_wall_clock_over_audio_duration(self) -> None:
        result = InvokeResult(
            outcome=InvokeOutcome.OK,
            dispatch_ts=0.0,
            end_ts=1.0,
            latency_ms=500.0,
            audio_duration_s=5.0,
        )
        assert result.rtf == pytest.approx(0.1)


class TestOutcomeEnum:
    def test_every_member_is_distinct(self) -> None:
        values = [m.value for m in InvokeOutcome]
        assert len(values) == len(set(values))

    def test_covers_the_documented_set(self) -> None:
        # The report renders one row per outcome; adding a member without
        # updating the reports would silently drop requests from the totals.
        assert {m.value for m in InvokeOutcome} == {
            "ok",
            "stale_408",
            "saturated_503",
            "throttled_429",
            "server_5xx",
            "model_error",
            "client_timeout",
            "dispatch_skipped",
            "error",
        }


class TestResolvers:
    def test_resolves_deployed_endpoints(self) -> None:
        assert resolve_endpoint(TTSModelName.KOKORO_82M) == "speech-kokoro-82m"
        assert resolve_endpoint("kokoro-82m-cpu") == "speech-kokoro-82m-cpu"

    def test_rejects_managed_polly(self) -> None:
        # There is no instance to size, so capacity planning does not apply.
        with pytest.raises(ValueError, match="no SageMaker endpoint"):
            resolve_endpoint(TTSModelName.POLLY_NEURAL)

    def test_voice_defaults_per_model(self) -> None:
        assert resolve_voice(TTSModelName.KOKORO_82M) == "af_heart"
        assert resolve_voice(TTSModelName.KOKORO_82M, "af_bella") == "af_bella"
