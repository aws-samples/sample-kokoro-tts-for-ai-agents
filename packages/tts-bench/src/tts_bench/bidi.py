"""Bidirectional-streaming transport for capacity benchmarking.

All five self-hosted models are configured ``streaming_mode=BIDIRECTIONAL``
(``config.py:100-149``), so this is the transport production actually uses — but
the ladder's first measurements came from
``invoke_endpoint_with_response_stream``. The two are not interchangeable, and
the difference is not a detail of the wire format:

* **The containers hold their inference lock differently per transport.**
  ``kokoro/serve.py:323`` holds ``_inference_lock`` across an entire bidi
  session — every request on that socket, not just one — while the
  response-stream path takes it per generator (``:229``). A ``Q_max`` measured
  on one transport therefore does not transfer to the other, and for kokoro the
  bidi number should be *lower*.
* **Saturation looks completely different.** On the response-stream path a full
  queue is an HTTP 503 that SageMaker wraps in ``ModelError``. On bidi it is a
  JSON *frame* — ``{"type": "error", ..., "message": "queue_saturated"}``
  (``vllm/streaming_proxy.py:330``) — delivered inside a stream that already
  returned success. :class:`tts_client.client.TTSClient` classifies those
  frames on the way through :func:`tts_client.errors.raise_for_error_frame`,
  because counting one as a short success is how an overloaded endpoint comes
  to look healthy.

:func:`invoke_bidi` returns the same :class:`~tts_bench.invoke.InvokeResult` as
:func:`~tts_bench.invoke.invoke_stream`, so ``loadgen``'s accounting, ``cmax``'s
summarization, and the ``InvokeOutcome`` enum all work unchanged. Like that
function, nothing here raises: every request resolves to exactly one outcome.

Plain non-streaming ``invoke_endpoint`` is deliberately not offered. It cannot
measure TTFAB at all, which is the quantity the whole ladder is keyed on.

``invoke_bidi`` builds its own :class:`TTSClient` per call rather than
sharing one across a ladder run — see that class's module docstring for why:
the installed HTTP/2 SDK caches connections in an unlocked dict, so sharing
one instance across the worker threads ``loadgen`` uses to drive concurrency
can race it. The cost is a fresh HTTP/2 handshake per request instead of a
warm connection; no currently-trusted bidi Q_max/T_total numbers exist to
invalidate, since those were measured on response-stream only, so this is a
new baseline rather than a regression.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from enum import StrEnum

from tts_bench.invoke import InvokeOutcome, InvokeResult
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

#: Bidi returns raw 16-bit mono PCM with no RIFF header. Every container
#: defines ``SAMPLE_RATE = 24000`` (verified in all four).
BIDI_SAMPLE_RATE = 24000


class Transport(StrEnum):
    """Which wire protocol a measurement was taken over.

    Recorded on the artifact because a ``Q_max`` is only meaningful alongside
    its transport — see the module docstring on kokoro's per-session lock.
    """

    RESPONSE_STREAM = "response-stream"
    """``invoke_endpoint_with_response_stream``: HTTP/1.1 chunked event stream."""

    BIDI = "bidi"
    """``invoke_endpoint_with_bidirectional_stream``: HTTP/2 duplex event stream."""


#: Maps each TTSClientError subclass onto the InvokeOutcome it corresponds to.
#: Identical table to tts_bench.invoke's -- both transports raise the same
#: typed exceptions now, since both go through TTSClient.
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

    A bare ``TTSClientError`` with ``http_status=200`` is ``TTSClient``'s
    "stream completed with no audio bytes" case -- a model failure, not an
    unclassified one.
    """
    if type(exc) is TTSClientError and exc.http_status == 200:
        return InvokeOutcome.MODEL_ERROR
    return _ERROR_OUTCOMES.get(type(exc), InvokeOutcome.ERROR)


def invoke_bidi(
    client: TTSClient,
    endpoint: str,
    text: str,
    voice: str,
    *,
    deadline_ts: float | None = None,
) -> InvokeResult:
    """One bidi synthesis, classified. Never raises.

    ``loadgen`` dispatches into a ``ThreadPoolExecutor``, and
    ``TTSClient.synthesize_bidi`` runs its own ``asyncio.run()`` internally —
    that call is itself blocking, so this function needs no event loop of its
    own; each worker thread just calls it directly. Deliberately not one
    shared client or loop: a single one would serialize sessions behind
    whichever is furthest behind, turning the open-loop dispatcher into the
    closed-loop design this package exists to replace.

    Args:
        client: Unused -- kept only so this matches ``invoke_stream``'s
            signature for ``loadgen.run_step(invoke=...)``. A fresh
            ``TTSClient`` is built for every call instead; see the module
            docstring for why sharing one is unsafe with the installed SDK.
        endpoint: SageMaker endpoint name.
        text: Text to synthesize.
        voice: Voice id.
        deadline_ts: Unused. ``TTSClient.synthesize_bidi`` has no mid-stream
            hook to abandon a request early, so its own read timeout (just
            above SageMaker's 60s ceiling) is the only backstop now. Kept as
            a parameter so this still matches ``invoke_stream``'s signature.

    Returns:
        An :class:`InvokeResult` with the same fields the response-stream path
        populates, so the two transports' events are directly comparable.
    """
    del client, deadline_ts
    dispatch_ts = time.time()
    t0 = time.perf_counter()
    request = SynthesisRequest(text=text, voice=voice, request_timestamp=dispatch_ts)
    bidi_client = TTSClient()

    try:
        result = bidi_client.synthesize_bidi(endpoint, request)
    except TTSClientError as exc:
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


def make_client_for(
    transport: Transport | str,
    region: str,
    *,
    max_pool: int,
    read_timeout: float | None = None,
) -> TTSClient:
    """Build whichever client ``transport`` needs.

    Both transports now build a :class:`TTSClient`. ``max_pool`` and
    ``read_timeout`` are accepted for call-site compatibility but unused --
    ``TTSClient`` takes no pooling knobs (see its module docstring) and bidi
    never holds this client anyway, building its own per call.
    """
    del max_pool, read_timeout
    Transport(transport)
    return TTSClient(region=region, retries=False)


def invoke_for(transport: Transport | str) -> Callable[..., InvokeResult]:
    """The invoke function for ``transport``, ready to inject into ``loadgen``."""
    from tts_bench.invoke import invoke_stream

    transport = Transport(transport)
    return invoke_bidi if transport is Transport.BIDI else invoke_stream


__all__ = [
    "BIDI_SAMPLE_RATE",
    "Transport",
    "invoke_bidi",
    "invoke_for",
    "make_client_for",
]
