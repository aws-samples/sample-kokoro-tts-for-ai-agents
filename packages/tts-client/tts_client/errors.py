# Copyright (c) 2026 Amazon.com
# This file is licensed under the MIT License.
# See the LICENSE file in the project root for full license information.

"""Typed exceptions for TTS synthesis failures.

Ported from ``tts_bench.invoke.classify_client_error`` and
``tts_bench.bidi.classify_error_frame``/``_classify_exception``, which return
``(InvokeOutcome, ...)`` tuples for benchmark accounting. A general-purpose
client raises instead: callers that just want audio can let these propagate,
and callers that need benchmark-style outcome classification (``tts-bench``)
catch by type and map onto their own ``InvokeOutcome`` enum, same as they
already do today for botocore's exceptions.

Every exception carries ``http_status`` (the container's real status once
SageMaker's wrapping is unwrapped, where one exists) so a caller that wants
the raw code without a match/case over exception types still can.
"""

from __future__ import annotations

import asyncio
from typing import Any


class TTSClientError(Exception):
    """Base for every exception this package raises."""

    def __init__(self, message: str, *, http_status: int | None = None) -> None:
        super().__init__(message)
        self.http_status = http_status


class RequestStaleError(TTSClientError):
    """The container rejected the request as older than its staleness bound."""


class QueueSaturatedError(TTSClientError):
    """Admission control refused: the container's queue is full."""


class ThrottledError(TTSClientError):
    """SageMaker or the container rate-limited the caller."""


class ServerError(TTSClientError):
    """Infrastructure failure — 5xx, or a mid-stream ``InternalStreamFailure``/
    ``ModelStreamError``."""


class ModelError(TTSClientError):
    """The container returned a non-2xx that is not one of the cases above."""


class TTSTimeoutError(TTSClientError):
    """The client gave up waiting. Not evidence the server failed."""


#: Container status codes, once SageMaker's `ModelError` (424) wrapper is
#: unwrapped or the bidi SDK's modeled error is asked for its own status.
_STATUS_ERRORS: dict[int, type[TTSClientError]] = {
    408: RequestStaleError,
    429: ThrottledError,
    503: QueueSaturatedError,
}


def raise_for_client_error(exc: Exception) -> None:
    """Classify a ``botocore.exceptions.ClientError`` and raise the matching
    :class:`TTSClientError` subclass.

    SageMaker wraps any container non-2xx in ``ModelError`` (HTTP 424) and puts
    the container's real status in ``OriginalStatusCode``. Classifying on the
    outer 424 would collapse "queue full", "request too old", and "model
    crashed" into one bucket, so the wrapper is unwrapped first.
    """
    response: dict[str, Any] = getattr(exc, "response", {}) or {}
    error = response.get("Error", {})
    code = error.get("Code", "")
    metadata = response.get("ResponseMetadata", {})
    http_status = metadata.get("HTTPStatusCode")

    original = response.get("OriginalStatusCode")
    if original is not None:
        try:
            original_int: int | None = int(original)
        except (TypeError, ValueError):
            original_int = None
        if original_int is not None:
            error_cls = _STATUS_ERRORS.get(original_int)
            if error_cls is not None:
                raise error_cls(str(exc), http_status=original_int) from exc
            if 500 <= original_int < 600:
                raise ServerError(str(exc), http_status=original_int) from exc
            raise ModelError(str(exc), http_status=original_int) from exc

    if code in ("ThrottlingException", "TooManyRequestsException"):
        raise ThrottledError(str(exc), http_status=http_status or 429) from exc
    if code == "ServiceUnavailable":
        raise QueueSaturatedError(str(exc), http_status=http_status or 503) from exc
    if code in ("InternalFailure", "InternalServerError", "InternalStreamFailure"):
        raise ServerError(str(exc), http_status=http_status or 500) from exc
    if code == "ModelError":
        raise ModelError(str(exc), http_status=http_status) from exc
    if code in ("ValidationError", "ValidationException"):
        # A malformed payload is a caller bug, not a server failure. Kept
        # distinct from ModelError so it cannot be misread as saturation.
        raise TTSClientError(str(exc), http_status=http_status or 400) from exc

    if isinstance(http_status, int):
        error_cls = _STATUS_ERRORS.get(http_status)
        if error_cls is not None:
            raise error_cls(str(exc), http_status=http_status) from exc
        if 500 <= http_status < 600:
            raise ServerError(str(exc), http_status=http_status) from exc

    raise TTSClientError(str(exc), http_status=http_status) from exc


#: Bidi container error-frame messages that mean "refused for capacity
#: reasons". Matched as substrings: the containers send bare tokens today but
#: wrap them in prose elsewhere.
_ERROR_MESSAGE_ERRORS: tuple[tuple[str, type[TTSClientError]], ...] = (
    ("queue_saturated", QueueSaturatedError),
    ("queue_full", QueueSaturatedError),
    ("request_stale", RequestStaleError),
    ("too_many_requests", ThrottledError),
)


def raise_for_error_frame(message: str) -> None:
    """Classify a bidi control frame's ``{"type": "error", "message": ...}``
    and raise the matching :class:`TTSClientError` subclass.

    The bidi handlers have no status code to carry the reason, so the message
    is the only signal distinguishing "queue full" from "the model crashed".
    An unrecognized message raises :class:`ModelError` rather than the base
    class: the container answered and reported a failure of its own, which is
    a server-side event even when it cannot be named more precisely.
    """
    lowered = message.lower()
    for token, error_cls in _ERROR_MESSAGE_ERRORS:
        if token in lowered:
            raise error_cls(message)
    raise ModelError(message)


#: Modeled bidi SDK exceptions, keyed by class name so this module never
#: imports the SDK — a caller using only the response-stream transport
#: should not need it installed.
_BIDI_EXCEPTION_ERRORS: dict[str, type[TTSClientError]] = {
    "ServiceUnavailableError": QueueSaturatedError,
    "InternalServerError": ServerError,
    "InternalStreamFailure": ServerError,
    "ModelStreamError": ServerError,
    "ModelError": ModelError,
    "InputValidationError": TTSClientError,
    "SerializationError": TTSClientError,
}


def raise_for_bidi_exception(exc: BaseException) -> None:
    """Classify an exception raised by the bidi HTTP/2 SDK and raise the
    matching :class:`TTSClientError` subclass.

    The bidi SDK raises modeled errors (``ModelError``, ``ServiceUnavailableError``)
    rather than botocore's single ``ClientError`` carrying a code, so this is the
    structural analogue of :func:`raise_for_client_error`. ``ModelError`` still
    carries the container's real status in ``original_status_code``, and it is
    unwrapped for the same reason: otherwise "queue full", "too old", and
    "crashed" collapse into one bucket.

    Precedence runs most-specific first: the container's own status code, then
    the modeled error's name, then ``is_throttling_error``/``is_timeout_error``
    last. Those flags are the fallback for errors this mapping does not name —
    every modeled error in the SDK defaults them to ``False``, so consulting
    them before the name would let a generic hint override a specific taxonomy.
    """
    if isinstance(exc, TimeoutError | asyncio.TimeoutError):
        raise TTSTimeoutError(str(exc)) from exc

    original = getattr(exc, "original_status_code", None)
    if isinstance(original, int):
        error_cls = _STATUS_ERRORS.get(original)
        if error_cls is not None:
            raise error_cls(str(exc), http_status=original) from exc
        if 500 <= original < 600:
            raise ServerError(str(exc), http_status=original) from exc
        raise ModelError(str(exc), http_status=original) from exc

    name = type(exc).__name__
    error_cls = _BIDI_EXCEPTION_ERRORS.get(name)
    if error_cls is not None:
        raise error_cls(str(exc)) from exc

    if getattr(exc, "is_throttling_error", False):
        raise ThrottledError(str(exc)) from exc
    if getattr(exc, "is_timeout_error", False):
        raise TTSTimeoutError(str(exc)) from exc

    raise TTSClientError(str(exc)) from exc
