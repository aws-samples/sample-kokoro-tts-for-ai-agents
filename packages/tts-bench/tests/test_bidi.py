"""Tests for the bidirectional-streaming transport.

``invoke_bidi`` calls :class:`tts_client.client.TTSClient.synthesize_bidi`
internally now, so these mock at that boundary rather than at the raw
``invoke_endpoint_with_bidirectional_stream`` SDK call. The stream mechanics
that used to be tested here (the ``synthesis_complete``-ends-the-request
fix, the input-stream-stays-open-until-drained fix, the
starts-with-brace-but-is-audio edge case, and the credential resolver) now
live inside ``tts_client`` and are covered by ``packages/tts-client/tests``
instead — see that package's ``test_client.py`` for the equivalents.

What's still tested here is what's still tts-bench's own responsibility:
mapping ``TTSClient``'s typed exceptions onto ``InvokeOutcome``, keeping the
two transports' ``InvokeResult`` fields comparable, and the sync wrapper
``loadgen`` calls.
"""

from __future__ import annotations

import inspect
from unittest.mock import MagicMock

import pytest

from tts_bench.bidi import Transport, invoke_bidi, invoke_for, make_client_for
from tts_bench.invoke import InvokeOutcome, invoke_stream
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

ENDPOINT = "speech-kokoro-82m"


def _result(**overrides) -> SynthesisResult:
    base = {
        "audio_bytes": b"\x00\x01" * 2400,
        "audio_format": AudioFormat.WAV,
        "sample_rate": 24000,
        "duration_s": 0.1,
        "latency_ms": 120.0,
        "ttfab_ms": 50.0,
        "chars": 11,
        "chunks": 1,
    }
    base.update(overrides)
    return SynthesisResult(**base)


def _patched_bidi_client(**kwargs):
    """Patch the TTSClient invoke_bidi() builds internally, per call."""
    fake = MagicMock(spec=TTSClient)
    if "return_value" in kwargs:
        fake.synthesize_bidi.return_value = kwargs["return_value"]
    if "side_effect" in kwargs:
        fake.synthesize_bidi.side_effect = kwargs["side_effect"]
    return fake


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


class TestMakeClientFor:
    def test_bidi_builds_a_ttsclient(self) -> None:
        client = make_client_for("bidi", "us-east-1", max_pool=64)
        assert isinstance(client, TTSClient)

    def test_response_stream_also_builds_a_ttsclient(self) -> None:
        # Both transports go through TTSClient now; bidi's own
        # synthesize_bidi() never actually holds the client this returns
        # (see invoke_bidi's docstring) but response-stream's invoke_stream
        # does use this one.
        client = make_client_for(Transport.RESPONSE_STREAM, "us-east-1", max_pool=128)
        assert isinstance(client, TTSClient)

    def test_max_pool_and_read_timeout_are_accepted_and_unused(self) -> None:
        # TTSClient takes no pooling knobs -- see its module docstring for why
        # sizing a pool buys nothing worth the API surface. Accepting these
        # keeps call sites from having to branch on transport.
        client = make_client_for("bidi", "us-east-1", max_pool=1, read_timeout=5.0)
        assert client is not None


# --------------------------------------------------------------------------- #
# Successful stream
# --------------------------------------------------------------------------- #


class TestInvokeBidiSuccess:
    def test_returns_ok_with_audio_metrics(self) -> None:
        client = _patched_bidi_client(return_value=_result(chars=len("hello world")))
        with pytest.MonkeyPatch().context() as mp:
            mp.setattr("tts_bench.bidi.TTSClient", lambda: client)
            result = invoke_bidi(None, ENDPOINT, "hello world", "af_heart")

        assert result.outcome is InvokeOutcome.OK
        assert result.ok
        assert result.chars == len("hello world")
        assert result.audio_bytes == 2400 * 2
        assert result.chunks == 1
        assert result.sample_rate == 24000
        assert result.audio_duration_s == pytest.approx(0.1, abs=1e-6)
        assert result.ttfab_ms is not None
        assert result.first_byte_ts is not None

    def test_sends_a_synthesis_request_with_text_voice_and_timestamp(self) -> None:
        client = _patched_bidi_client(return_value=_result())
        with pytest.MonkeyPatch().context() as mp:
            mp.setattr("tts_bench.bidi.TTSClient", lambda: client)
            invoke_bidi(None, ENDPOINT, "hello", "af_bella")

        (endpoint, request), _ = client.synthesize_bidi.call_args
        assert endpoint == ENDPOINT
        assert request.text == "hello"
        assert request.voice == "af_bella"
        assert request.request_timestamp is not None


# --------------------------------------------------------------------------- #
# Failures
# --------------------------------------------------------------------------- #


class TestInvokeBidiFailures:
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
        client = _patched_bidi_client(side_effect=exc)
        with pytest.MonkeyPatch().context() as mp:
            mp.setattr("tts_bench.bidi.TTSClient", lambda: client)
            result = invoke_bidi(None, ENDPOINT, "hello", "af_heart")

        assert result.outcome is expected_outcome
        assert result.http_status == exc.http_status

    def test_error_frame_is_a_failure_not_a_short_success(self) -> None:
        # The defect this transport exists to avoid: a naive client that logs a
        # warning and breaks on an in-band error frame would count a rejected
        # request as a fast OK, which is what makes an overloaded endpoint
        # look fast. TTSClient.synthesize_bidi raises for this; invoke_bidi
        # must classify the raise, not swallow it.
        client = _patched_bidi_client(
            side_effect=QueueSaturatedError("queue_saturated", http_status=503)
        )
        with pytest.MonkeyPatch().context() as mp:
            mp.setattr("tts_bench.bidi.TTSClient", lambda: client)
            result = invoke_bidi(None, ENDPOINT, "hello", "af_heart")

        assert result.outcome is InvokeOutcome.SATURATED_503
        assert not result.ok

    def test_http_200_with_no_audio_is_model_error(self) -> None:
        client = _patched_bidi_client(
            side_effect=TTSClientError("bidi stream completed with no audio bytes", http_status=200)
        )
        with pytest.MonkeyPatch().context() as mp:
            mp.setattr("tts_bench.bidi.TTSClient", lambda: client)
            result = invoke_bidi(None, ENDPOINT, "hello", "af_heart")

        assert result.outcome is InvokeOutcome.MODEL_ERROR
        assert result.error_class == "EmptyResponse"

    def test_unclassified_error_is_error_not_silently_ok(self) -> None:
        client = _patched_bidi_client(side_effect=TTSClientError("weird", http_status=418))
        with pytest.MonkeyPatch().context() as mp:
            mp.setattr("tts_bench.bidi.TTSClient", lambda: client)
            result = invoke_bidi(None, ENDPOINT, "hello", "af_heart")

        assert result.outcome is InvokeOutcome.ERROR
        assert result.error_class == "TTSClientError"

    def test_deadline_ts_is_accepted_but_no_longer_abandons_mid_stream(self) -> None:
        # TTSClient.synthesize_bidi has no mid-stream hook to enforce a
        # deadline; its own read timeout is the only backstop now. deadline_ts
        # stays an accepted parameter for loadgen.run_step(invoke=...)
        # signature compatibility, but a deadline already in the past does
        # not change a successful result.
        client = _patched_bidi_client(return_value=_result())
        with pytest.MonkeyPatch().context() as mp:
            mp.setattr("tts_bench.bidi.TTSClient", lambda: client)
            result = invoke_bidi(None, ENDPOINT, "hello", "af_heart", deadline_ts=0.0)

        assert result.outcome is InvokeOutcome.OK

    def test_failures_still_record_chars_and_timing(self) -> None:
        # Reconciliation depends on this: the totals must account for every
        # scheduled request, including the failed ones.
        client = _patched_bidi_client(side_effect=QueueSaturatedError("busy", http_status=503))
        with pytest.MonkeyPatch().context() as mp:
            mp.setattr("tts_bench.bidi.TTSClient", lambda: client)
            result = invoke_bidi(None, ENDPOINT, "hello world", "af_heart")

        assert result.chars == len("hello world")
        assert result.latency_ms >= 0
        assert result.end_ts >= result.dispatch_ts


# --------------------------------------------------------------------------- #
# Cross-transport comparability
# --------------------------------------------------------------------------- #


class TestCrossTransportComparability:
    """A C_max on one transport is only readable next to one on the other.

    Both paths must populate ``InvokeResult`` identically for equivalent audio,
    because ``loadgen`` and ``cmax`` consume these fields without knowing which
    transport produced them.
    """

    def test_equivalent_successful_streams_agree_field_by_field(self) -> None:
        text = "hello world"
        shared_result = _result(chars=len(text))

        bidi_client = _patched_bidi_client(return_value=shared_result)
        with pytest.MonkeyPatch().context() as mp:
            mp.setattr("tts_bench.bidi.TTSClient", lambda: bidi_client)
            bidi = invoke_bidi(None, ENDPOINT, text, "af_heart")

        rs_client = MagicMock(spec=TTSClient)
        rs_client.synthesize.return_value = shared_result
        rs = invoke_stream(rs_client, ENDPOINT, text, "af_heart")

        assert bidi.outcome is rs.outcome is InvokeOutcome.OK
        assert bidi.chars == rs.chars
        assert bidi.chunks == rs.chunks
        assert bidi.sample_rate == rs.sample_rate
        assert bidi.audio_duration_s == pytest.approx(rs.audio_duration_s, abs=1e-6)
        assert bidi.audio_bytes == rs.audio_bytes

    def test_the_same_optional_fields_are_populated_on_success(self) -> None:
        bidi_client = _patched_bidi_client(return_value=_result())
        with pytest.MonkeyPatch().context() as mp:
            mp.setattr("tts_bench.bidi.TTSClient", lambda: bidi_client)
            bidi = invoke_bidi(None, ENDPOINT, "hello world", "af_heart")

        rs_client = MagicMock(spec=TTSClient)
        rs_client.synthesize.return_value = _result()
        rs = invoke_stream(rs_client, ENDPOINT, "hello world", "af_heart")

        for field in ("ttfab_ms", "first_byte_ts", "error_class", "error_message"):
            assert (getattr(bidi, field) is None) == (getattr(rs, field) is None), field

    def test_both_transports_report_saturation_as_the_same_outcome(self) -> None:
        # One arrives as an HTTP status inside TTSClient.synthesize, the other
        # as a JSON frame inside TTSClient.synthesize_bidi -- both are raised
        # as QueueSaturatedError, so both transports classify the same way.
        bidi_client = _patched_bidi_client(
            side_effect=QueueSaturatedError("queue_saturated", http_status=503)
        )
        with pytest.MonkeyPatch().context() as mp:
            mp.setattr("tts_bench.bidi.TTSClient", lambda: bidi_client)
            bidi = invoke_bidi(None, ENDPOINT, "hello world", "af_heart")

        rs_client = MagicMock(spec=TTSClient)
        rs_client.synthesize.side_effect = QueueSaturatedError("full", http_status=503)
        rs = invoke_stream(rs_client, ENDPOINT, "hello world", "af_heart")

        assert bidi.outcome is rs.outcome is InvokeOutcome.SATURATED_503

    def test_rtf_is_computable_on_the_bidi_path(self) -> None:
        client = _patched_bidi_client(return_value=_result())
        with pytest.MonkeyPatch().context() as mp:
            mp.setattr("tts_bench.bidi.TTSClient", lambda: client)
            result = invoke_bidi(None, ENDPOINT, "hello", "af_heart")

        assert result.rtf is not None


# --------------------------------------------------------------------------- #
# Sync wrapper
# --------------------------------------------------------------------------- #


class TestInvokeBidiSync:
    def test_matches_invoke_streams_signature(self) -> None:
        # loadgen injects one or the other as `invoke=`, positionally, with
        # deadline_ts by keyword. A signature drift would only show up at
        # runtime, mid-ladder.
        assert list(inspect.signature(invoke_bidi).parameters) == list(
            inspect.signature(invoke_stream).parameters
        )

    def test_works_from_a_worker_thread(self) -> None:
        # loadgen dispatches into a ThreadPoolExecutor. invoke_bidi calls
        # TTSClient.synthesize_bidi, which runs its own asyncio.run()
        # internally, so this must work from a thread with no loop of its own.
        from concurrent.futures import ThreadPoolExecutor

        client = _patched_bidi_client(return_value=_result())

        def _one():
            with pytest.MonkeyPatch().context() as mp:
                mp.setattr("tts_bench.bidi.TTSClient", lambda: client)
                return invoke_bidi(None, ENDPOINT, "hi", "af_heart")

        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(lambda _i: _one(), range(4)))

        assert all(r.outcome is InvokeOutcome.OK for r in results)
