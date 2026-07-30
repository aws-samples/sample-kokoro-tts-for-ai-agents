"""Invocation path for capacity benchmarking.

Separate from :class:`tts_eval.synthesize.SynthesisClient` for two reasons that
both distort measurements:

1. That client builds boto3 clients with no ``botocore.Config``, so
   ``max_pool_connections`` defaults to 10. Above ten in-flight streams urllib3
   silently queues connections, and the benchmark measures the client's
   connection pool instead of the server's concurrency ceiling.
2. boto3 retries by default. Retries fabricate load exactly when the server is
   saturated, inflating the offered rate and hiding errors behind eventual
   successes — the two things a saturation measurement most needs to be honest
   about.

Nothing here is swallowed: every request resolves to exactly one
:class:`InvokeOutcome`, so a run's totals always reconcile against the requests
that were scheduled.
"""

from __future__ import annotations

import json
import struct
import time
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import botocore.exceptions
from botocore.client import BaseClient
from botocore.config import Config

from tts_eval.synthesize import DEFAULT_VOICES, ENDPOINT_MAP, wav_duration
from tts_inference.types import TTSModelName

#: Just above SageMaker's 60s invocation ceiling. A client timeout below the
#: server's own limit would report client_timeout for requests the server was
#: still entitled to finish, misattributing the failure.
DEFAULT_READ_TIMEOUT_S = 65.0

DEFAULT_CONNECT_TIMEOUT_S = 5.0


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

    DISPATCH_SKIPPED = "dispatch_skipped"
    """No worker was free at the scheduled arrival time; never sent.

    Set by the load generator, not here. Its presence means the *client* ran out
    of capacity, so any saturation conclusion from that step is unsafe.
    """

    ERROR = "error"
    """Unclassified. Investigate rather than aggregate."""


#: Container status codes mapped through SageMaker's ``ModelError`` wrapper.
_ORIGINAL_STATUS_OUTCOMES: dict[int, InvokeOutcome] = {
    408: InvokeOutcome.STALE_408,
    429: InvokeOutcome.THROTTLED_429,
    503: InvokeOutcome.SATURATED_503,
}


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


def make_runtime_client(
    region: str,
    *,
    max_pool: int,
    read_timeout: float = DEFAULT_READ_TIMEOUT_S,
    connect_timeout: float = DEFAULT_CONNECT_TIMEOUT_S,
) -> BaseClient:
    """Build a ``sagemaker-runtime`` client that will not throttle the benchmark.

    Args:
        region: AWS region.
        max_pool: Connection pool size. Must exceed the highest concurrency the
            run will reach, or urllib3 queues connections and the measured knee
            is the pool's, not the model's. Callers should pass their peak
            in-flight bound with headroom.
        read_timeout: Per-read timeout. Defaults just above SageMaker's 60s
            ceiling so the server's limit is always the binding one.
        connect_timeout: TCP connect timeout.

    Returns:
        A configured client. **Retries are disabled** (``max_attempts=0``).
    """
    config = Config(
        region_name=region,
        max_pool_connections=max_pool,
        read_timeout=read_timeout,
        connect_timeout=connect_timeout,
        retries={"max_attempts": 0},
        tcp_keepalive=True,
    )
    import boto3

    return boto3.client("sagemaker-runtime", config=config)


def classify_client_error(exc: botocore.exceptions.ClientError) -> tuple[InvokeOutcome, int | None]:
    """Map a botocore ``ClientError`` to an outcome and the meaningful status.

    SageMaker wraps any container non-2xx in ``ModelError`` (HTTP 424) and puts
    the container's real status in ``OriginalStatusCode``. Classifying on the
    outer 424 would collapse "queue full", "request too old", and "model
    crashed" into one bucket, so the wrapper is unwrapped first.

    Returns:
        ``(outcome, status)`` where ``status`` is the container's original status
        when SageMaker supplied one, otherwise the HTTP status of the response.
    """
    response: dict[str, Any] = getattr(exc, "response", {}) or {}
    error = response.get("Error", {})
    code = error.get("Code", "")
    metadata = response.get("ResponseMetadata", {})
    http_status = metadata.get("HTTPStatusCode")

    original = response.get("OriginalStatusCode")
    if original is not None:
        try:
            original_int = int(original)
        except (TypeError, ValueError):
            original_int = None
        if original_int is not None:
            mapped = _ORIGINAL_STATUS_OUTCOMES.get(original_int)
            if mapped is not None:
                return mapped, original_int
            if 500 <= original_int < 600:
                return InvokeOutcome.SERVER_5XX, original_int
            return InvokeOutcome.MODEL_ERROR, original_int

    if code in ("ThrottlingException", "TooManyRequestsException"):
        return InvokeOutcome.THROTTLED_429, http_status or 429
    if code == "ServiceUnavailable":
        return InvokeOutcome.SATURATED_503, http_status or 503
    if code in ("InternalFailure", "InternalServerError", "InternalStreamFailure"):
        return InvokeOutcome.SERVER_5XX, http_status or 500
    if code == "ModelError":
        return InvokeOutcome.MODEL_ERROR, http_status
    if code in ("ValidationError", "ValidationException"):
        # A malformed payload is a benchmark bug, not a capacity finding. It
        # stays distinct from model_error so it cannot be read as saturation.
        return InvokeOutcome.ERROR, http_status or 400

    if isinstance(http_status, int):
        mapped = _ORIGINAL_STATUS_OUTCOMES.get(http_status)
        if mapped is not None:
            return mapped, http_status
        if 500 <= http_status < 600:
            return InvokeOutcome.SERVER_5XX, http_status

    return InvokeOutcome.ERROR, http_status


def build_payload(text: str, voice: str, *, request_ts: float) -> bytes:
    """Serialize the invocation body.

    ``request_timestamp`` is the **actual send time**, which is what the
    containers subtract from ``time.time()`` to decide whether a request has
    aged past ``MAX_REQUEST_AGE_S`` (``kokoro/serve.py:215`` and equivalents).
    Sending anything else — a deadline, say — would make that age negative and
    silently disable the server-side staleness check we want exercised.
    """
    return json.dumps({"text": text, "voice": voice, "request_timestamp": request_ts}).encode(
        "utf-8"
    )


def invoke_stream(
    client: BaseClient,
    endpoint: str,
    text: str,
    voice: str,
    *,
    deadline_ts: float | None = None,
) -> InvokeResult:
    """Stream one synthesis and classify the result. Never raises.

    Args:
        client: A client from :func:`make_runtime_client`.
        endpoint: SageMaker endpoint name.
        text: Text to synthesize.
        voice: Voice id.
        deadline_ts: Optional epoch time after which the client abandons the
            stream, reporting ``CLIENT_TIMEOUT``. Independent of the server's
            own staleness check: this bounds how long *we* wait, while
            ``request_timestamp`` in the payload lets the *container* decide the
            request is too old to be worth serving.

    Returns:
        An :class:`InvokeResult`. Every failure path is classified rather than
        propagated, so a caller's accounting cannot silently lose requests.
    """
    dispatch_ts = time.time()
    t0 = time.perf_counter()
    payload = build_payload(text, voice, request_ts=dispatch_ts)

    def _elapsed_ms() -> float:
        return (time.perf_counter() - t0) * 1000.0

    def _failure(
        outcome: InvokeOutcome,
        *,
        status: int | None = None,
        exc: BaseException | None = None,
    ) -> InvokeResult:
        return InvokeResult(
            outcome=outcome,
            dispatch_ts=dispatch_ts,
            end_ts=time.time(),
            latency_ms=_elapsed_ms(),
            chars=len(text),
            http_status=status,
            error_class=type(exc).__name__ if exc is not None else None,
            error_message=str(exc)[:500] if exc is not None else None,
        )

    try:
        response = client.invoke_endpoint_with_response_stream(
            EndpointName=endpoint,
            ContentType="application/json",
            Accept="audio/wav",
            Body=payload,
        )
    except botocore.exceptions.ReadTimeoutError as exc:
        return _failure(InvokeOutcome.CLIENT_TIMEOUT, exc=exc)
    except botocore.exceptions.ConnectTimeoutError as exc:
        return _failure(InvokeOutcome.CLIENT_TIMEOUT, exc=exc)
    except botocore.exceptions.ClientError as exc:
        outcome, status = classify_client_error(exc)
        return _failure(outcome, status=status, exc=exc)
    except botocore.exceptions.BotoCoreError as exc:
        # Covers connection-pool exhaustion and endpoint resolution problems:
        # client-side faults that must not be counted as server saturation.
        return _failure(InvokeOutcome.ERROR, exc=exc)

    chunks: list[bytes] = []
    ttfab_ms: float | None = None
    first_byte_ts: float | None = None

    try:
        for event in response["Body"]:
            if "PayloadPart" in event:
                chunk = event["PayloadPart"]["Bytes"]
                if not chunk:
                    continue
                if ttfab_ms is None:
                    ttfab_ms = _elapsed_ms()
                    first_byte_ts = time.time()
                chunks.append(chunk)
                if deadline_ts is not None and time.time() > deadline_ts:
                    # Abandon mid-stream. Reported as our timeout, not a server
                    # failure, and the partial audio is discarded.
                    return _failure(InvokeOutcome.CLIENT_TIMEOUT)
                continue

            # Mid-stream faults arrive as events on a response that already
            # returned HTTP 200 — the failure mode behind the torch.Tensor
            # regression in commit caa4dcf.
            if "ModelStreamError" in event:
                detail = event["ModelStreamError"]
                return InvokeResult(
                    outcome=InvokeOutcome.SERVER_5XX,
                    dispatch_ts=dispatch_ts,
                    end_ts=time.time(),
                    latency_ms=_elapsed_ms(),
                    first_byte_ts=first_byte_ts,
                    ttfab_ms=ttfab_ms,
                    chars=len(text),
                    audio_bytes=sum(len(c) for c in chunks),
                    chunks=len(chunks),
                    error_class="ModelStreamError",
                    error_message=str(detail.get("Message", ""))[:500],
                )
            if "InternalStreamFailure" in event:
                detail = event["InternalStreamFailure"]
                return InvokeResult(
                    outcome=InvokeOutcome.SERVER_5XX,
                    dispatch_ts=dispatch_ts,
                    end_ts=time.time(),
                    latency_ms=_elapsed_ms(),
                    first_byte_ts=first_byte_ts,
                    ttfab_ms=ttfab_ms,
                    chars=len(text),
                    audio_bytes=sum(len(c) for c in chunks),
                    chunks=len(chunks),
                    error_class="InternalStreamFailure",
                    error_message=str(detail.get("Message", ""))[:500],
                )
    except botocore.exceptions.ReadTimeoutError as exc:
        return _failure(InvokeOutcome.CLIENT_TIMEOUT, exc=exc)
    except botocore.exceptions.ClientError as exc:
        outcome, status = classify_client_error(exc)
        return _failure(outcome, status=status, exc=exc)
    except botocore.exceptions.BotoCoreError as exc:
        return _failure(InvokeOutcome.ERROR, exc=exc)

    latency_ms = _elapsed_ms()
    end_ts = time.time()
    audio = b"".join(chunks)

    if not audio:
        # HTTP 200 with an empty body. Distinct from a rejection: the endpoint
        # accepted the work and produced nothing.
        return InvokeResult(
            outcome=InvokeOutcome.MODEL_ERROR,
            dispatch_ts=dispatch_ts,
            end_ts=end_ts,
            latency_ms=latency_ms,
            chars=len(text),
            http_status=200,
            error_class="EmptyResponse",
            error_message="stream completed with no audio bytes",
        )

    return InvokeResult(
        outcome=InvokeOutcome.OK,
        dispatch_ts=dispatch_ts,
        end_ts=end_ts,
        latency_ms=latency_ms,
        first_byte_ts=first_byte_ts,
        ttfab_ms=ttfab_ms,
        chars=len(text),
        audio_bytes=len(audio),
        audio_duration_s=wav_duration(audio),
        sample_rate=_sample_rate(audio),
        http_status=200,
        chunks=len(chunks),
    )


def _sample_rate(audio: bytes) -> int:
    """Read the sample rate from a RIFF header, 0 if unreadable."""
    if len(audio) < 28 or audio[:4] != b"RIFF":
        return 0
    try:
        return int(struct.unpack_from("<I", audio, 24)[0])
    except struct.error:
        return 0


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
    return endpoint


def resolve_voice(model: str | TTSModelName, voice: str | None = None) -> str:
    """Voice for a model, defaulting to ``tts_eval``'s per-model choice."""
    model = TTSModelName(model)
    return voice or DEFAULT_VOICES[model]
