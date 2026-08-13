"""Tests for the benchmarking invocation path.

The classification tests matter more than they look: every capacity conclusion
is built on counting outcomes, so a request landing in the wrong bucket turns
into a wrong C_max, which turns into a wrong fleet size.

``invoke_stream`` calls :class:`tts_client.client.TTSClient` internally now,
so these mock at that boundary (``client.synthesize``) rather than at the raw
boto3 ``invoke_endpoint_with_response_stream`` call ``TTSClient`` itself
already has coverage for.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from tts_bench.invoke import (
    InvokeOutcome,
    InvokeResult,
    invoke_stream,
    resolve_endpoint,
    resolve_voice,
)
from tts_client.client import TTSClient
from tts_client.errors import (
    ModelError,
    QueueSaturatedError,
    RequestStaleError,
    ServerError,
    ThrottledError,
    TTSClientError,
    TTSTimeoutError,
)
from tts_client.types import AudioFormat, SynthesisResult
from tts_inference.types import TTSModelName


def _result(**overrides) -> SynthesisResult:
    base = {
        "audio_bytes": b"\x00\x01" * 2400,
        "audio_format": AudioFormat.WAV,
        "sample_rate": 24000,
        "duration_s": 0.1,
        "latency_ms": 120.0,
        "ttfab_ms": 50.0,
        "chars": 11,
        "chunks": 2,
    }
    base.update(overrides)
    return SynthesisResult(**base)


class TestInvokeStreamSuccess:
    def test_returns_ok_with_audio_metrics(self) -> None:
        client = MagicMock(spec=TTSClient)
        client.synthesize.return_value = _result(chars=len("hello world"))

        result = invoke_stream(client, "speech-kokoro-82m", "hello world", "af_heart")

        assert result.outcome is InvokeOutcome.OK
        assert result.ok
        assert result.chars == len("hello world")
        assert result.audio_bytes == 2400 * 2
        assert result.chunks == 2
        assert result.sample_rate == 24000
        assert result.audio_duration_s == pytest.approx(0.1, abs=1e-6)
        assert result.ttfab_ms is not None
        assert result.first_byte_ts is not None

    def test_ttfab_is_first_chunk_not_last(self) -> None:
        client = MagicMock(spec=TTSClient)
        client.synthesize.return_value = _result(latency_ms=200.0, ttfab_ms=50.0)

        result = invoke_stream(client, "speech-kokoro-82m", "hello", "af_heart")
        assert result.ttfab_ms is not None
        assert result.ttfab_ms <= result.latency_ms

    def test_sends_request_via_synthesis_request(self) -> None:
        client = MagicMock(spec=TTSClient)
        client.synthesize.return_value = _result()

        invoke_stream(client, "speech-kokoro-82m", "hello", "af_heart")

        (endpoint, request), _ = client.synthesize.call_args
        assert endpoint == "speech-kokoro-82m"
        assert request.text == "hello"
        assert request.voice == "af_heart"
        assert request.request_timestamp is not None


class TestInvokeStreamFailures:
    @pytest.mark.parametrize(
        ("exc", "expected_outcome"),
        [
            (QueueSaturatedError("full", http_status=503), InvokeOutcome.SATURATED_503),
            (RequestStaleError("old", http_status=408), InvokeOutcome.STALE_408),
            (ThrottledError("slow down", http_status=429), InvokeOutcome.THROTTLED_429),
            (ServerError("boom", http_status=500), InvokeOutcome.SERVER_5XX),
            (ModelError("bad", http_status=422), InvokeOutcome.MODEL_ERROR),
            (TTSTimeoutError("timed out"), InvokeOutcome.CLIENT_TIMEOUT),
        ],
    )
    def test_maps_each_tts_client_error_to_its_outcome(
        self, exc: TTSClientError, expected_outcome: InvokeOutcome
    ) -> None:
        client = MagicMock(spec=TTSClient)
        client.synthesize.side_effect = exc

        result = invoke_stream(client, "speech-kokoro-82m", "hello", "af_heart")
        assert result.outcome is expected_outcome
        assert result.http_status == exc.http_status

    def test_never_raises_on_client_error(self) -> None:
        client = MagicMock(spec=TTSClient)
        client.synthesize.side_effect = QueueSaturatedError("full", http_status=503)

        result = invoke_stream(client, "speech-kokoro-82m", "hello", "af_heart")
        assert result.outcome is InvokeOutcome.SATURATED_503
        assert result.http_status == 503
        assert result.error_class == "QueueSaturatedError"

    def test_mid_stream_model_stream_error_after_http_200(self) -> None:
        # The caa4dcf failure shape: 200 + headers already sent, then the
        # generator raises. Must not be recorded as a success. TTSClient
        # raises ServerError for this; the mid-stream distinction lives there
        # now, not in this benchmark.
        client = MagicMock(spec=TTSClient)
        client.synthesize.side_effect = ServerError("AttributeError astype")

        result = invoke_stream(client, "speech-kokoro-82m", "hello", "af_heart")
        assert result.outcome is InvokeOutcome.SERVER_5XX
        assert result.error_class == "ServerError"

    def test_http_200_with_no_audio_is_model_error(self) -> None:
        # TTSClient raises the bare base class (not a coded subclass) with
        # http_status=200 for this case -- invoke_stream must still name it
        # EmptyResponse, matching the pre-TTSClient contract.
        client = MagicMock(spec=TTSClient)
        client.synthesize.side_effect = TTSClientError(
            "stream completed with no audio bytes", http_status=200
        )

        result = invoke_stream(client, "speech-kokoro-82m", "hello", "af_heart")
        assert result.outcome is InvokeOutcome.MODEL_ERROR
        assert result.error_class == "EmptyResponse"

    def test_unclassified_error_is_error_not_silently_ok(self) -> None:
        client = MagicMock(spec=TTSClient)
        client.synthesize.side_effect = TTSClientError("weird", http_status=418)

        result = invoke_stream(client, "speech-kokoro-82m", "hello", "af_heart")
        assert result.outcome is InvokeOutcome.ERROR
        assert result.error_class == "TTSClientError"

    def test_deadline_ts_is_accepted_but_no_longer_abandons_mid_stream(self) -> None:
        # TTSClient.synthesize has no mid-stream hook to enforce a deadline;
        # its own read timeout is the only backstop now (see invoke_stream's
        # docstring). deadline_ts stays an accepted parameter for
        # loadgen.run_step(invoke=...) signature compatibility, but a
        # deadline already in the past does not change a successful result.
        client = MagicMock(spec=TTSClient)
        client.synthesize.return_value = _result()

        result = invoke_stream(client, "speech-kokoro-82m", "hello", "af_heart", deadline_ts=0.0)
        assert result.outcome is InvokeOutcome.OK

    def test_failed_results_still_record_chars_and_timing(self) -> None:
        # Reconciliation depends on this: a run's totals must account for every
        # scheduled request, including the ones that failed.
        client = MagicMock(spec=TTSClient)
        client.synthesize.side_effect = QueueSaturatedError("full", http_status=503)

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
            "error",
        }


class TestResolvers:
    def test_resolves_deployed_endpoints(self) -> None:
        assert resolve_endpoint(TTSModelName.KOKORO_82M) == "speech-kokoro-82m"
        assert resolve_endpoint("kokoro-82m") == "speech-kokoro-82m"

    def test_rejects_managed_polly(self) -> None:
        # There is no instance to size, so capacity planning does not apply.
        with pytest.raises(ValueError, match="no SageMaker endpoint"):
            resolve_endpoint(TTSModelName.POLLY_NEURAL)

    def test_voice_defaults_per_model(self) -> None:
        assert resolve_voice(TTSModelName.KOKORO_82M) == "af_heart"
        assert resolve_voice(TTSModelName.KOKORO_82M, "af_bella") == "af_bella"
