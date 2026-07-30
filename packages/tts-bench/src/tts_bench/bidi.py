"""Bidirectional-streaming transport for capacity benchmarking.

All five self-hosted models are configured ``streaming_mode=BIDIRECTIONAL``
(``config.py:100-149``), so this is the transport production actually uses — but
every number ``cmax`` produced before this module came from
``invoke_endpoint_with_response_stream``. The two are not interchangeable, and
the difference is not a detail of the wire format:

* **The containers hold their inference lock differently per transport.**
  ``kokoro/serve.py:323`` holds ``_inference_lock`` across an entire bidi
  session — every request on that socket, not just one — while the
  response-stream path takes it per generator (``:229``). A ``C_max`` measured
  on one transport therefore does not transfer to the other, and for kokoro the
  bidi number should be *lower*.
* **Saturation looks completely different.** On the response-stream path a full
  queue is an HTTP 503 that SageMaker wraps in ``ModelError``. On bidi it is a
  JSON *frame* — ``{"type": "error", ..., "message": "queue_saturated"}``
  (``vllm/streaming_proxy.py:330``) — delivered inside a stream that already
  returned success. :func:`invoke_bidi` classifies those frames, because
  counting one as a short success is how an overloaded endpoint comes to look
  healthy.

:func:`invoke_bidi` returns the same :class:`~tts_bench.invoke.InvokeResult` as
:func:`~tts_bench.invoke.invoke_stream`, so ``loadgen``'s accounting, ``cmax``'s
summarization, and the ``InvokeOutcome`` enum all work unchanged. Like that
function, nothing here raises: every request resolves to exactly one outcome.

Plain non-streaming ``invoke_endpoint`` is deliberately not offered. It cannot
measure TTFAB at all, which is the quantity the whole ladder is keyed on.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from loguru import logger

from tts_bench.invoke import (
    DEFAULT_READ_TIMEOUT_S,
    InvokeOutcome,
    InvokeResult,
)

if TYPE_CHECKING:  # pragma: no cover - import cycle only matters to type checkers
    from collections.abc import Callable

#: Bidirectional streaming is served on 8443, not the usual 443.
BIDI_PORT = 8443

#: Bidi returns raw 16-bit mono PCM with no RIFF header, so the rate cannot be
#: read off the payload the way :func:`invoke._sample_rate` does. Every
#: container defines ``SAMPLE_RATE = 24000`` (verified in all four), so audio
#: duration is computed from that rather than guessed.
BIDI_SAMPLE_RATE = 24000

_BYTES_PER_SAMPLE = 2


class Transport(StrEnum):
    """Which wire protocol a measurement was taken over.

    Recorded on the artifact because a ``C_max`` is only meaningful alongside
    its transport — see the module docstring on kokoro's per-session lock.
    """

    RESPONSE_STREAM = "response-stream"
    """``invoke_endpoint_with_response_stream``: HTTP/1.1 chunked event stream."""

    BIDI = "bidi"
    """``invoke_endpoint_with_bidirectional_stream``: HTTP/2 duplex event stream."""


#: Container error-frame messages that mean "refused for capacity reasons",
#: mapped to the same outcomes the HTTP status codes produce on the other
#: transport. Matched as substrings: the containers send bare tokens today
#: (``vllm/streaming_proxy.py:330``) but wrap them in prose elsewhere.
_ERROR_MESSAGE_OUTCOMES: tuple[tuple[str, InvokeOutcome], ...] = (
    ("queue_saturated", InvokeOutcome.SATURATED_503),
    ("queue_full", InvokeOutcome.SATURATED_503),
    ("request_stale", InvokeOutcome.STALE_408),
    ("too_many_requests", InvokeOutcome.THROTTLED_429),
)


def classify_error_frame(message: str) -> InvokeOutcome:
    """Map a container error frame's message onto an outcome.

    The bidi handlers have no status code to carry the reason, so the message is
    the only signal distinguishing "queue full" from "the model crashed". An
    unrecognized message becomes :attr:`InvokeOutcome.MODEL_ERROR` rather than
    :attr:`InvokeOutcome.ERROR`: the container answered and reported a failure
    of its own, which is a server-side event even when we cannot name it.
    """
    lowered = message.lower()
    for token, outcome in _ERROR_MESSAGE_OUTCOMES:
        if token in lowered:
            return outcome
    return InvokeOutcome.MODEL_ERROR


class Boto3CredentialsResolver:
    """Resolve SigV4 credentials through boto3's full credential chain.

    The SDK's own ``EnvironmentCredentialsResolver`` (which
    ``tts_eval/bidi_client.py:71`` uses) reads ``AWS_ACCESS_KEY_ID`` and
    ``AWS_SECRET_ACCESS_KEY`` directly and raises ``SmithyIdentityError`` when
    they are unset. On an EC2 instance with an IAM role — which is how this
    benchmark runs — they *are* unset, so that resolver cannot authenticate at
    all. The other bundled resolvers do not substitute cleanly either:
    ``IMDSCredentialsResolver`` requires an ``http_client`` argument, and
    ``StaticCredentialsResolver`` reads from auth properties rather than config.

    Deferring to boto3 also means the benchmark and the response-stream path
    authenticate identically, so a permissions difference between transports
    cannot masquerade as a capacity difference.

    ``get_frozen_credentials()`` is called on **every** resolve rather than
    cached. A ``cmax`` ladder runs for 45 minutes and role credentials rotate
    inside that window; caching the first frozen tuple (as the SDK's resolver
    does) would start failing mid-ladder, at the high-rate steps that run last —
    exactly where a wave of auth failures is most easily misread as saturation.
    """

    def __init__(self, session: Any | None = None) -> None:
        if session is None:
            import boto3

            session = boto3.Session()
        self._session = session

    async def get_identity(self, *, properties: Any = None) -> Any:
        """Return current credentials as an ``AWSCredentialsIdentity``."""
        from smithy_aws_core.identity import AWSCredentialsIdentity

        credentials = self._session.get_credentials()
        if credentials is None:
            raise BidiTransportError(
                "no AWS credentials found by boto3's credential chain; bidi streaming "
                "cannot be signed. Check the instance role or AWS_PROFILE."
            )
        frozen = credentials.get_frozen_credentials()
        return AWSCredentialsIdentity(
            access_key_id=frozen.access_key,
            secret_access_key=frozen.secret_key,
            session_token=frozen.token,
        )


class BidiTransportError(RuntimeError):
    """Raised for setup faults that are benchmark bugs, not measurements.

    A missing credential or an unresolvable endpoint is not a capacity finding,
    and returning it as an :class:`InvokeResult` would fold it into the run's
    error counts. Per-request failures never come through here — those are
    classified.
    """


def make_bidi_client(
    region: str,
    *,
    read_timeout: float = DEFAULT_READ_TIMEOUT_S,
    session: Any | None = None,
) -> Any:
    """Build a ``SageMakerRuntimeHTTP2Client`` configured for benchmarking.

    Args:
        region: AWS region.
        read_timeout: Per-read timeout, defaulting just above SageMaker's 60s
            ceiling so the server's limit is the binding one — same reasoning as
            :func:`invoke.make_runtime_client`.
        session: Optional boto3 session for credential resolution.

    Returns:
        A client with **retries disabled** (``max_attempts=1``, i.e. the initial
        attempt only). Retries would fabricate load exactly when the server is
        saturated, which is the same reason the boto3 client passes
        ``max_attempts=0``. The two SDKs count attempts differently: botocore's
        value is retries-after-the-first, smithy's is total attempts.
    """
    from aws_sdk_sagemaker_runtime_http2.client import SageMakerRuntimeHTTP2Client
    from aws_sdk_sagemaker_runtime_http2.config import Config, HTTPAuthSchemeResolver
    from smithy_aws_core.auth.sigv4 import SigV4AuthScheme
    from smithy_core.retries import RetryStrategyOptions
    from smithy_http.interfaces import HTTPRequestConfiguration

    config = Config(
        endpoint_uri=f"https://runtime.sagemaker.{region}.amazonaws.com:{BIDI_PORT}",
        region=region,
        aws_credentials_identity_resolver=Boto3CredentialsResolver(session=session),
        auth_scheme_resolver=HTTPAuthSchemeResolver(),
        auth_schemes={"aws.auth#sigv4": SigV4AuthScheme(service="sagemaker")},
        http_request_config=HTTPRequestConfiguration(read_timeout=read_timeout),
        retry_strategy=RetryStrategyOptions(max_attempts=1),
    )
    return SageMakerRuntimeHTTP2Client(config=config)


def build_bidi_message(
    text: str,
    voice: str,
    *,
    request_ts: float,
    request_id: str | None = None,
) -> bytes:
    """Serialize one synthesis request for the bidi input stream.

    ``request_timestamp`` is included for parity with
    :func:`invoke.build_payload`, but note what it does and does not buy here:
    **no container's bidi handler checks it today** — kokoro reads it at
    ``serve.py:276`` on the HTTP path only, and the other three are the same.
    Sending it anyway means the Phase 6 dequeue-time deadline re-check has the
    field available on both transports the day it lands, rather than needing a
    coordinated client change. Until then a stale bidi request is served in
    full, which is itself a finding worth reporting rather than hiding.
    """
    return json.dumps(
        {
            "text": text,
            "voice": voice,
            "request_timestamp": request_ts,
            "request_id": request_id or f"bench-{uuid.uuid4().hex[:8]}",
        }
    ).encode("utf-8")


@dataclass(slots=True)
class _StreamState:
    """Mutable accumulator for one bidi session's audio and timing."""

    pcm_bytes: int = 0
    chunks: int = 0
    ttfab_ms: float | None = None
    first_byte_ts: float | None = None


async def invoke_bidi_async(
    client: Any,
    endpoint: str,
    text: str,
    voice: str,
    *,
    deadline_ts: float | None = None,
) -> InvokeResult:
    """One bidi synthesis, classified. Never raises except :class:`BidiTransportError`.

    Args:
        client: A client from :func:`make_bidi_client`.
        endpoint: SageMaker endpoint name.
        text: Text to synthesize.
        voice: Voice id.
        deadline_ts: Epoch time after which we abandon the stream and report
            ``CLIENT_TIMEOUT``, mirroring ``invoke_stream``'s mid-stream check.
            This bounds how long *we* wait; it says nothing about the server.

    Returns:
        An :class:`InvokeResult` with the same fields the response-stream path
        populates, so the two transports' events are directly comparable.
    """
    from aws_sdk_sagemaker_runtime_http2.models import (
        InvokeEndpointWithBidirectionalStreamInput,
        RequestPayloadPart,
        RequestStreamEventPayloadPart,
    )

    dispatch_ts = time.time()
    t0 = time.perf_counter()
    state = _StreamState()

    def _elapsed_ms() -> float:
        return (time.perf_counter() - t0) * 1000.0

    def _result(
        outcome: InvokeOutcome,
        *,
        error_class: str | None = None,
        error_message: str | None = None,
    ) -> InvokeResult:
        """Build the result, carrying whatever audio and timing did arrive.

        Partial audio is reported on failures too: a request that streamed for
        50s and then failed is a different capacity signal from one that failed
        immediately, and zeroing the fields would erase that.
        """
        return InvokeResult(
            outcome=outcome,
            dispatch_ts=dispatch_ts,
            end_ts=time.time(),
            latency_ms=_elapsed_ms(),
            first_byte_ts=state.first_byte_ts,
            ttfab_ms=state.ttfab_ms,
            chars=len(text),
            audio_bytes=state.pcm_bytes,
            audio_duration_s=_pcm_duration_s(state.pcm_bytes),
            sample_rate=BIDI_SAMPLE_RATE if state.pcm_bytes else 0,
            error_class=error_class,
            error_message=error_message[:500] if error_message else None,
            chunks=state.chunks,
        )

    message = build_bidi_message(text, voice, request_ts=dispatch_ts)

    try:
        stream = await client.invoke_endpoint_with_bidirectional_stream(
            InvokeEndpointWithBidirectionalStreamInput(endpoint_name=endpoint)
        )
    except Exception as exc:  # noqa: BLE001 - the SDK's error taxonomy is open
        outcome, error_class = _classify_exception(exc)
        return _result(outcome, error_class=error_class, error_message=str(exc))

    try:
        await stream.input_stream.send(
            RequestStreamEventPayloadPart(value=RequestPayloadPart(bytes_=message))
        )
        # Closed immediately: the benchmark sends one synthesis per session, so
        # holding the input open would make the container wait on a request that
        # is never coming. One session == one request keeps this transport's
        # results comparable with the response-stream path's.
        await stream.input_stream.close()

        _, output_stream = await stream.await_output()
        return await _consume_output(
            output_stream,
            state=state,
            elapsed_ms=_elapsed_ms,
            result=_result,
            deadline_ts=deadline_ts,
        )
    except Exception as exc:  # noqa: BLE001 - as above
        outcome, error_class = _classify_exception(exc)
        return _result(outcome, error_class=error_class, error_message=str(exc))
    finally:
        # Aborting mid-stream (deadline, error frame) leaves the HTTP/2 stream
        # open otherwise, and a ladder that leaks one per abandoned request
        # exhausts the connection before it reaches its highest step.
        await _close_quietly(stream)


async def _consume_output(
    output_stream: Any,
    *,
    state: _StreamState,
    elapsed_ms: Callable[[], float],
    result: Callable[..., InvokeResult],
    deadline_ts: float | None,
) -> InvokeResult:
    """Drain the output stream, classifying frames as they arrive."""
    from aws_sdk_sagemaker_runtime_http2.models import (
        ResponseStreamEventInternalStreamFailure,
        ResponseStreamEventModelStreamError,
    )

    while True:
        event = await output_stream.receive()
        if event is None:
            break

        # Mid-stream service faults are modeled as event variants here rather
        # than dict keys, but mean what they do on the other transport: a
        # failure that arrives *after* the stream was accepted.
        if isinstance(event, ResponseStreamEventModelStreamError):
            return result(
                InvokeOutcome.SERVER_5XX,
                error_class="ModelStreamError",
                error_message=str(getattr(event.value, "message", "")),
            )
        if isinstance(event, ResponseStreamEventInternalStreamFailure):
            return result(
                InvokeOutcome.SERVER_5XX,
                error_class="InternalStreamFailure",
                error_message=str(getattr(event.value, "message", "")),
            )

        payload = getattr(event, "value", None)
        chunk = getattr(payload, "bytes_", None)
        if not chunk:
            continue

        # The containers multiplex JSON control frames and raw PCM on one
        # stream, distinguished only by whether the payload starts with `{`.
        # A chunk that starts that way but does not decode is audio, not a
        # malformed frame — PCM can legitimately begin with byte 0x7b, and
        # dropping it would understate throughput at exactly the rates where
        # every sample counts. Hence the fall-through rather than `continue`.
        decoded = _decode_control_frame(chunk) if chunk[0:1] == b"{" else None
        if decoded is not None:
            if decoded.get("type") == "error":
                message = str(decoded.get("message", ""))
                return result(
                    classify_error_frame(message),
                    error_class="ErrorFrame",
                    error_message=message,
                )
            continue

        if state.ttfab_ms is None:
            state.ttfab_ms = elapsed_ms()
            state.first_byte_ts = time.time()
        state.pcm_bytes += len(chunk)
        state.chunks += 1

        if deadline_ts is not None and time.time() > deadline_ts:
            # Our timeout, not a server failure. The partial audio stays on the
            # result but the outcome is not OK, so it cannot count as throughput.
            return result(InvokeOutcome.CLIENT_TIMEOUT)

    if not state.pcm_bytes:
        # The stream completed cleanly and produced nothing. Distinct from a
        # rejection: the endpoint accepted the work and returned no audio.
        return result(
            InvokeOutcome.MODEL_ERROR,
            error_class="EmptyResponse",
            error_message="bidi stream completed with no audio bytes",
        )

    return result(InvokeOutcome.OK)


def _decode_control_frame(chunk: bytes) -> dict[str, Any] | None:
    """Parse a JSON control frame, or ``None`` if it is not one.

    A chunk starting with ``{`` is *probably* JSON, but PCM can begin with byte
    0x7b too. Treating an undecodable chunk as a control frame would drop real
    audio, so this fails soft and the caller counts it as audio.
    """
    try:
        decoded = json.loads(chunk.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    return decoded if isinstance(decoded, dict) else None


def _classify_exception(exc: BaseException) -> tuple[InvokeOutcome, str]:
    """Map an SDK exception to an outcome and its class name.

    The bidi SDK raises modeled errors (``ModelError``, ``ServiceUnavailableError``)
    rather than botocore's single ``ClientError`` carrying a code, so this is the
    structural analogue of :func:`invoke.classify_client_error`. ``ModelError``
    still carries the container's real status in ``original_status_code``, and it
    is unwrapped for the same reason: otherwise "queue full", "too old", and
    "crashed" collapse into one bucket.

    Precedence runs most-specific first: the container's own status code, then
    the modeled error's name, then ``is_throttling_error`` / ``is_timeout_error``
    last. Those flags are the fallback for errors this mapping does not name —
    every modeled error in the SDK defaults them to ``False``, so consulting
    them before the name would let a generic hint override a specific taxonomy.
    """
    name = type(exc).__name__

    if isinstance(exc, TimeoutError | asyncio.TimeoutError):
        return InvokeOutcome.CLIENT_TIMEOUT, name

    original = getattr(exc, "original_status_code", None)
    if isinstance(original, int):
        mapped = _STATUS_OUTCOMES.get(original)
        if mapped is not None:
            return mapped, name
        if 500 <= original < 600:
            return InvokeOutcome.SERVER_5XX, name
        return InvokeOutcome.MODEL_ERROR, name

    outcome = _EXCEPTION_OUTCOMES.get(name)
    if outcome is not None:
        return outcome, name

    if getattr(exc, "is_throttling_error", False):
        return InvokeOutcome.THROTTLED_429, name
    if getattr(exc, "is_timeout_error", False):
        return InvokeOutcome.CLIENT_TIMEOUT, name

    return InvokeOutcome.ERROR, name


#: Container status codes, as carried in ``ModelError.original_status_code``.
_STATUS_OUTCOMES: dict[int, InvokeOutcome] = {
    408: InvokeOutcome.STALE_408,
    429: InvokeOutcome.THROTTLED_429,
    503: InvokeOutcome.SATURATED_503,
}

#: Modeled SDK exceptions, keyed by class name so the mapping does not import
#: the SDK at module scope — ``tts-bench`` must stay importable without it.
_EXCEPTION_OUTCOMES: dict[str, InvokeOutcome] = {
    "ServiceUnavailableError": InvokeOutcome.SATURATED_503,
    "InternalServerError": InvokeOutcome.SERVER_5XX,
    "InternalStreamFailure": InvokeOutcome.SERVER_5XX,
    "ModelStreamError": InvokeOutcome.SERVER_5XX,
    "ModelError": InvokeOutcome.MODEL_ERROR,
    "InputValidationError": InvokeOutcome.ERROR,
    "SerializationError": InvokeOutcome.ERROR,
}


def _pcm_duration_s(pcm_bytes: int) -> float:
    """Seconds of audio in a count of 16-bit mono PCM bytes."""
    if pcm_bytes <= 0:
        return 0.0
    return pcm_bytes / float(BIDI_SAMPLE_RATE * _BYTES_PER_SAMPLE)


async def _close_quietly(stream: Any) -> None:
    """Close a stream, logging rather than raising on failure.

    A close error must not overwrite the measurement: the request already has an
    outcome by this point, and losing it to a teardown fault would silently
    remove a sample from the run.
    """
    close = getattr(stream, "close", None)
    if close is None:
        return
    try:
        outcome = close()
        if _is_awaitable(outcome):
            await outcome
    except Exception as exc:  # noqa: BLE001 - teardown must never lose a result
        logger.debug("bidi stream close failed: {}", exc)


def _is_awaitable(value: Any) -> bool:
    return hasattr(value, "__await__")


def invoke_bidi(
    client: Any,
    endpoint: str,
    text: str,
    voice: str,
    *,
    deadline_ts: float | None = None,
) -> InvokeResult:
    """Synchronous :func:`invoke_bidi_async`, for the threaded load generator.

    ``loadgen`` dispatches into a ``ThreadPoolExecutor`` and the bidi SDK is
    asyncio, so each session gets its own event loop in its own worker thread —
    the same arrangement as ``tts_eval.bidi_client.synthesize_bidirectional``,
    which is documented thread-safe. Deliberately *not* one shared loop: a
    single loop would serialize sessions behind whichever one is furthest
    behind, turning the open-loop dispatcher into the closed-loop design this
    package exists to replace.

    Matches :func:`invoke.invoke_stream`'s signature exactly, so it can be
    passed as ``loadgen.run_step(invoke=...)`` with no adapter.
    """
    return asyncio.run(invoke_bidi_async(client, endpoint, text, voice, deadline_ts=deadline_ts))


def make_client_for(
    transport: Transport | str,
    region: str,
    *,
    max_pool: int,
    read_timeout: float = DEFAULT_READ_TIMEOUT_S,
) -> Any:
    """Build whichever client ``transport`` needs.

    ``max_pool`` is meaningful only for the boto3 path — the HTTP/2 client
    multiplexes streams over one connection rather than drawing from a pool — so
    it is accepted and ignored for bidi rather than making callers branch.
    """
    transport = Transport(transport)
    if transport is Transport.BIDI:
        return make_bidi_client(region, read_timeout=read_timeout)

    from tts_bench.invoke import make_runtime_client

    return make_runtime_client(region, max_pool=max_pool, read_timeout=read_timeout)


def invoke_for(transport: Transport | str) -> Callable[..., InvokeResult]:
    """The invoke function for ``transport``, ready to inject into ``loadgen``."""
    from tts_bench.invoke import invoke_stream

    transport = Transport(transport)
    return invoke_bidi if transport is Transport.BIDI else invoke_stream


__all__ = [
    "BIDI_PORT",
    "BIDI_SAMPLE_RATE",
    "BidiTransportError",
    "Boto3CredentialsResolver",
    "Transport",
    "build_bidi_message",
    "classify_error_frame",
    "invoke_bidi",
    "invoke_bidi_async",
    "invoke_for",
    "make_bidi_client",
    "make_client_for",
]
