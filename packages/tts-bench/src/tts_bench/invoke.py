"""Invocation path for capacity benchmarking.

Calls :class:`tts_client.client.TTSClient` with ``retries=False`` — a retry
turns one measured request into several attempts under one latency, hiding
the failure behind an eventual success, so a saturating endpoint would report
as slow rather than as saturated, which is the distinction the whole
measurement rests on. ``TTSClient`` exists precisely so this benchmark and
every other caller share one wire implementation instead of each building
its own boto3 client.

Nothing here is swallowed: every request resolves to exactly one
:class:`InvokeOutcome`, so a run's totals always reconcile against the requests
it dispatched.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import StrEnum

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
from tts_client.types import SynthesisRequest
from tts_eval.synthesize import DEFAULT_VOICES, ENDPOINT_MAP
from tts_inference.types import TTSModelName


class InvokeOutcome(StrEnum):
    """Terminal classification of one invocation attempt.

    Closed set: every request lands in exactly one, and each is counted
    separately in reports. Collapsing these into "errors" is what makes an
    overloaded endpoint look merely flaky — a queue rejection (``saturated_503``)
    and a model crash (``model_error``) call for opposite responses.
    """

    OK = "ok"
    STALE_408 = "stale_408"
    """Container rejected the request as older than its ``MAX_REQUEST_AGE_S``."""

    SATURATED_503 = "saturated_503"
    """Admission control refused: queue full or backpressure tripped."""

    THROTTLED_429 = "throttled_429"
    """SageMaker or the container rate-limited the caller."""

    SERVER_5XX = "server_5xx"
    """Infrastructure failure, including mid-stream ``InternalStreamFailure``."""

    MODEL_ERROR = "model_error"
    """Container returned a non-2xx that is not one of the cases above."""

    CLIENT_TIMEOUT = "client_timeout"
    """We stopped waiting. Not evidence the server failed."""

    ERROR = "error"
    """Unclassified. Investigate rather than aggregate."""


@dataclass(frozen=True, slots=True)
class InvokeResult:
    """One invocation attempt, successful or not.

    A dataclass rather than a Pydantic model: this is allocated once per request
    on the hot path, and none of it is external input needing validation.
    Timings are epoch seconds so they correlate directly with CloudWatch
    datapoints and container log timestamps.
    """

    outcome: InvokeOutcome
    dispatch_ts: float
    end_ts: float
    latency_ms: float
    first_byte_ts: float | None = None
    ttfab_ms: float | None = None
    chars: int = 0
    audio_bytes: int = 0
    audio_duration_s: float = 0.0
    sample_rate: int = 0
    http_status: int | None = None
    error_class: str | None = None
    error_message: str | None = None
    chunks: int = 0

    @property
    def ok(self) -> bool:
        return self.outcome is InvokeOutcome.OK

    @property
    def rtf(self) -> float | None:
        """Real-time factor: wall-clock spent per second of audio produced.

        ``None`` when no audio came back, which is not the same as zero — a
        failed request has no RTF, and averaging it in as 0.0 would flatter the
        result.
        """
        if self.audio_duration_s <= 0:
            return None
        return (self.latency_ms / 1000.0) / self.audio_duration_s


#: Maps each TTSClientError subclass onto the InvokeOutcome it corresponds to.
#: Built to mirror tts_client.errors's classification one-to-one, since that
#: module already does the SageMaker-ModelError-unwrapping/status-code work
#: this benchmark used to do itself. All of tts_client's exceptions are flat
#: subclasses of TTSClientError (no further hierarchy), so an exact type
#: lookup is enough -- a caught exception whose exact class is not here (only
#: the base TTSClientError itself) falls through to InvokeOutcome.ERROR.
_ERROR_OUTCOMES: dict[type[TTSClientError], InvokeOutcome] = {
    RequestStaleError: InvokeOutcome.STALE_408,
    QueueSaturatedError: InvokeOutcome.SATURATED_503,
    ThrottledError: InvokeOutcome.THROTTLED_429,
    ServerError: InvokeOutcome.SERVER_5XX,
    ModelError: InvokeOutcome.MODEL_ERROR,
    TTSTimeoutError: InvokeOutcome.CLIENT_TIMEOUT,
}


def _outcome_for(exc: TTSClientError) -> InvokeOutcome:
    """The InvokeOutcome for a raised TTSClientError.

    A bare ``TTSClientError`` (none of the coded subclasses) with
    ``http_status=200`` is ``TTSClient``'s "stream completed with no audio
    bytes" case — an HTTP 200 the endpoint answered but produced nothing
    useful from, which is a model failure, not an unclassified one.
    """
    if type(exc) is TTSClientError and exc.http_status == 200:
        return InvokeOutcome.MODEL_ERROR
    return _ERROR_OUTCOMES.get(type(exc), InvokeOutcome.ERROR)


def invoke_stream(
    client: TTSClient,
    endpoint: str,
    text: str,
    voice: str,
    *,
    deadline_ts: float | None = None,
) -> InvokeResult:
    """Stream one synthesis and classify the result. Never raises.

    Args:
        client: A :class:`TTSClient`, built with ``retries=False``.
        endpoint: SageMaker endpoint name.
        text: Text to synthesize.
        voice: Voice id.
        deadline_ts: Unused. ``TTSClient.synthesize`` has no mid-stream hook to
            abandon a request early, so the read timeout it already builds
            in (just above SageMaker's 60s ceiling) is the only backstop now.
            Kept as a parameter so this still matches
            :func:`tts_bench.bidi.invoke_bidi`'s signature for
            ``loadgen.run_step(invoke=...)``.

    Returns:
        An :class:`InvokeResult`. Every failure path is classified rather than
        propagated, so a caller's accounting cannot silently lose requests.
    """
    dispatch_ts = time.time()
    t0 = time.perf_counter()
    request = SynthesisRequest(text=text, voice=voice, request_timestamp=dispatch_ts)

    try:
        result = client.synthesize(endpoint, request)
    except TTSClientError as exc:
        # TTSClient raises the bare base class, not a coded subclass, for a
        # 200 that produced no audio -- name it EmptyResponse to match what
        # this benchmark called that case before the client existed.
        is_empty = type(exc) is TTSClientError and exc.http_status == 200
        return InvokeResult(
            outcome=_outcome_for(exc),
            dispatch_ts=dispatch_ts,
            end_ts=time.time(),
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            chars=len(text),
            http_status=exc.http_status,
            error_class="EmptyResponse" if is_empty else type(exc).__name__,
            error_message=str(exc)[:500],
        )

    end_ts = time.time()
    first_byte_ts = dispatch_ts + result.ttfab_ms / 1000.0 if result.ttfab_ms is not None else None

    return InvokeResult(
        outcome=InvokeOutcome.OK,
        dispatch_ts=dispatch_ts,
        end_ts=end_ts,
        latency_ms=result.latency_ms,
        first_byte_ts=first_byte_ts,
        ttfab_ms=result.ttfab_ms,
        chars=result.chars,
        audio_bytes=len(result.audio_bytes),
        audio_duration_s=result.duration_s,
        sample_rate=result.sample_rate,
        http_status=200,
        chunks=result.chunks,
    )


def resolve_endpoint(model: str | TTSModelName) -> str:
    """Endpoint name for a model, reusing ``tts_eval``'s mapping.

    Raises:
        ValueError: If the model has no SageMaker endpoint (the Polly variants
            are managed, so there is no instance to size).
    """
    model = TTSModelName(model)
    endpoint = ENDPOINT_MAP.get(model)
    if endpoint is None:
        raise ValueError(
            f"{model.value} has no SageMaker endpoint; capacity planning "
            "applies to self-hosted endpoints only"
        )
    return str(endpoint)


def resolve_voice(model: str | TTSModelName, voice: str | None = None) -> str:
    """Voice for a model, defaulting to ``tts_eval``'s per-model choice."""
    model = TTSModelName(model)
    return voice or DEFAULT_VOICES[model]
