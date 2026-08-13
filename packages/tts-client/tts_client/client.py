"""Blocking client for the TTS synthesis API.

Two independently-callable methods, one per transport the Kokoro-style
SageMaker contract exposes: :meth:`TTSClient.synthesize` (binary
response-stream, chunked WAV/MP3) and :meth:`TTSClient.synthesize_bidi`
(SageMaker bidirectional streaming, raw PCM).

Pooling was investigated directly against the installed SDKs rather than
assumed (see the plan this package was built from):

* :meth:`synthesize` shares one boto3 client per ``TTSClient`` instance.
  botocore always builds its connection pool with urllib3's default
  ``block=False``, so an undersized pool never blocks or errors — it just
  skips connection reuse on the excess calls. There is therefore no
  pool-size knob here: making several ``TTSClient()`` instances is exactly
  as safe as sharing one, and requires no sizing decision.
* :meth:`synthesize_bidi` never holds a client on ``self``. The installed
  ``aws_sdk_sagemaker_runtime_http2`` SDK (v0.7.0) caches HTTP/2 connections
  in a plain unlocked dict (``smithy_http/aio/crt.py``, commented
  "TODO: Use CRT connection pooling instead of this basic kind"), so sharing
  one instance across threads can race that dict. ``synthesize_bidi`` builds
  a fresh client and runs a fresh ``asyncio.run()`` every call instead.
"""

from __future__ import annotations

import asyncio
import json
import struct
import time
from collections.abc import Iterable
from typing import Any

import boto3
import botocore.exceptions
from aws_sdk_sagemaker_runtime_http2.models import (
    RequestPayloadPart,
    RequestStreamEventPayloadPart,
)
from botocore.config import Config

from tts_client._bidi_transport import (
    BIDI_SAMPLE_RATE,
    _build_bidi_message,
    _close_quietly,
    _drain_until_complete,
    _open_bidi_stream,
    _pcm_to_wav,
    wav_duration,
)
from tts_client.errors import (
    ServerError,
    TTSClientError,
    TTSTimeoutError,
    raise_for_bidi_exception,
    raise_for_client_error,
)
from tts_client.streaming import BidiChunkStream
from tts_client.types import AudioFormat, SampleRate, SynthesisRequest, SynthesisResult

#: Just above SageMaker's 60s invocation ceiling. A client timeout below the
#: server's own limit would report a client timeout for requests the server
#: was still entitled to finish.
DEFAULT_READ_TIMEOUT_S = 65.0

DEFAULT_CONNECT_TIMEOUT_S = 5.0

_MEDIA_TYPES = {AudioFormat.WAV: "audio/wav", AudioFormat.MP3: "audio/mpeg"}


def _build_payload(request: SynthesisRequest) -> bytes:
    """Serialize a :class:`SynthesisRequest` for ``/invocations``.

    No ``transport`` field: the container's default (raw chunked bytes) is
    exactly what :meth:`TTSClient.synthesize` wants.

    ``request_timestamp`` is the actual send time, which is what the
    containers subtract from ``time.time()`` to decide whether a request has
    aged past their staleness bound. Stamped here if the caller left it unset.
    """
    body: dict[str, Any] = {
        "text": request.text,
        "voice": request.voice,
        "speed": request.speed,
        "format": request.audio_format.value,
        "request_timestamp": request.request_timestamp
        if request.request_timestamp is not None
        else time.time(),
    }
    if request.sample_rate is not None:
        body["sample_rate"] = request.sample_rate.value
    return json.dumps(body).encode("utf-8")


class TTSClient:
    """Blocking client for the TTS synthesis API.

    Make as many instances as you like — each is independent, and each is
    safe to call concurrently from multiple threads. See the module docstring
    for the pooling reasoning behind that claim.
    """

    def __init__(self, region: str = "us-east-1", *, retries: bool = True) -> None:
        """
        Args:
            region: AWS region the SageMaker endpoint lives in.
            retries: Whether botocore retries transient failures (its own
                default: up to 5 attempts on 503/429/5xx with backoff).
                Callers measuring capacity must pass ``False`` — the
                container's queue-saturated 503 is one of botocore's
                retryable statuses, so a retry can turn a genuine rejection
                into what looks like a slow success, corrupting exactly the
                distinction a capacity measurement depends on. Most callers
                want the default.
        """
        config = Config(
            region_name=region,
            read_timeout=DEFAULT_READ_TIMEOUT_S,
            connect_timeout=DEFAULT_CONNECT_TIMEOUT_S,
            retries={"max_attempts": 0} if not retries else None,
            tcp_keepalive=True,
        )
        self._client = boto3.client("sagemaker-runtime", config=config)
        self._region = region

    def synthesize(self, endpoint: str, request: SynthesisRequest) -> SynthesisResult:
        """Binary response-stream: chunked WAV or MP3, depending on
        ``request.audio_format``.

        Raises:
            TTSClientError: or a subclass, for any non-2xx response or
                mid-stream server fault.
        """
        payload = _build_payload(request)
        t0 = time.perf_counter()

        try:
            response = self._client.invoke_endpoint_with_response_stream(
                EndpointName=endpoint,
                ContentType="application/json",
                Accept=_MEDIA_TYPES[request.audio_format],
                Body=payload,
            )
        except botocore.exceptions.ReadTimeoutError as exc:
            raise TTSTimeoutError(str(exc)) from exc
        except botocore.exceptions.ConnectTimeoutError as exc:
            raise TTSTimeoutError(str(exc)) from exc
        except botocore.exceptions.ClientError as exc:
            raise_for_client_error(exc)
            raise AssertionError("unreachable") from exc

        chunks: list[bytes] = []
        ttfab_ms: float | None = None

        try:
            for event in response["Body"]:
                if "PayloadPart" in event:
                    chunk = event["PayloadPart"]["Bytes"]
                    if not chunk:
                        continue
                    if ttfab_ms is None:
                        ttfab_ms = (time.perf_counter() - t0) * 1000.0
                    chunks.append(chunk)
                    continue
                # Mid-stream faults arrive as events on a response that
                # already returned HTTP 200.
                if "ModelStreamError" in event:
                    detail = event["ModelStreamError"]
                    raise ServerError(str(detail.get("Message", "")), http_status=None)
                if "InternalStreamFailure" in event:
                    detail = event["InternalStreamFailure"]
                    raise ServerError(str(detail.get("Message", "")), http_status=None)
        except botocore.exceptions.ReadTimeoutError as exc:
            raise TTSTimeoutError(str(exc)) from exc
        except botocore.exceptions.ClientError as exc:
            raise_for_client_error(exc)
            raise AssertionError("unreachable") from exc

        latency_ms = (time.perf_counter() - t0) * 1000.0
        audio = b"".join(chunks)
        if not audio:
            raise TTSClientError("stream completed with no audio bytes", http_status=200)

        sample_rate = (
            struct.unpack_from("<I", audio, 24)[0]
            if request.audio_format == AudioFormat.WAV and len(audio) >= 28
            else BIDI_SAMPLE_RATE
        )
        duration_s = wav_duration(audio) if request.audio_format == AudioFormat.WAV else 0.0

        return SynthesisResult(
            audio_bytes=audio,
            audio_format=request.audio_format,
            sample_rate=sample_rate,
            duration_s=duration_s,
            latency_ms=latency_ms,
            ttfab_ms=ttfab_ms or latency_ms,
            chars=len(request.text),
            chunks=len(chunks),
        )

    def synthesize_bidi(self, endpoint: str, request: SynthesisRequest) -> SynthesisResult:
        """Bidirectional streaming over SageMaker's HTTP/2 transport.

        Builds a fresh HTTP/2 client and event loop for this call alone —
        see the module docstring for why this transport never shares state
        across calls. Always raw PCM server-side; the result wraps it in a
        WAV header, regardless of ``request.audio_format``.

        Raises:
            TTSClientError: or a subclass, for any modeled SDK error or an
                in-band ``error`` frame.
        """
        return asyncio.run(self._synthesize_bidi_async(endpoint, request))

    async def _synthesize_bidi_async(
        self, endpoint: str, request: SynthesisRequest
    ) -> SynthesisResult:
        resolved_rate = (
            request.sample_rate.value if request.sample_rate is not None else BIDI_SAMPLE_RATE
        )
        message = _build_bidi_message(
            request.text,
            request.voice,
            request.request_timestamp,
            sample_rate=request.sample_rate.value if request.sample_rate is not None else None,
        )
        t0 = time.perf_counter()
        stream = await _open_bidi_stream(self._region, endpoint)

        try:
            await stream.input_stream.send(
                RequestStreamEventPayloadPart(value=RequestPayloadPart(bytes_=message))
            )
            # The input stream stays open until the audio is drained. Closing
            # it right after send makes SageMaker tear the connection down
            # before the container reads the payload.
            _, output_stream = await stream.await_output()
            drained = await _drain_until_complete(output_stream, t0)
        except TTSClientError:
            # Already classified above (a control-frame error, or the
            # ModelStreamError/InternalStreamFailure cases) — reclassifying it
            # through raise_for_bidi_exception would collapse it back to a
            # generic TTSClientError, since that function does not know these
            # exception types.
            raise
        except Exception as exc:  # noqa: BLE001 - the SDK's error taxonomy is open
            raise_for_bidi_exception(exc)
            raise AssertionError("unreachable") from exc
        finally:
            await _close_quietly(stream)

        latency_ms = (time.perf_counter() - t0) * 1000.0
        if not drained.pcm_bytes:
            raise TTSClientError("bidi stream completed with no audio bytes", http_status=200)

        audio = _pcm_to_wav(drained.pcm_bytes, sample_rate=resolved_rate)
        duration_s = len(drained.pcm_bytes) / (resolved_rate * 2)

        return SynthesisResult(
            audio_bytes=audio,
            audio_format=AudioFormat.WAV,
            sample_rate=resolved_rate,
            duration_s=duration_s,
            latency_ms=latency_ms,
            ttfab_ms=drained.ttfab_ms or latency_ms,
            chars=len(request.text),
            chunks=drained.chunk_count,
        )

    def synthesize_bidi_stream(
        self,
        endpoint: str,
        voice: str,
        text_source: str | Iterable[str],
        *,
        speed: float = 1.0,
        sample_rate: SampleRate | None = None,
    ) -> BidiChunkStream:
        """Stream text to ``endpoint`` as one message per sentence, on one bidi session.

        ``text_source`` is either a complete string (split into sentences and
        sent one at a time) or an iterable of fragments arriving over time
        (e.g. an upstream LLM's token stream) -- see the ``streaming`` module
        docstring for why one chunker handles both without two code paths.

        Chunk N+1 is not sent until chunk N's audio has fully arrived: every
        bidi container's handler is strictly sequential, so this does not
        overlap synthesis across chunks. What it buys is not waiting for the
        entire text to be assembled before sending anything -- chunk 1 starts
        synthesizing as soon as it is available.

        Returns:
            A :class:`~tts_client.streaming.BidiChunkStream`: an iterator of
            :class:`~tts_client.types.SynthesisChunk`. Use as a context
            manager for deterministic cleanup on early exit::

                with client.synthesize_bidi_stream(endpoint, voice, text) as stream:
                    for chunk in stream:
                        ...

        Raises:
            ValueError: if ``text_source`` produces no sentences to synthesize.
            TTSClientError: or a subclass, for any modeled SDK error, an
                in-band ``error`` frame, or the connection dropping mid-session.
        """
        return BidiChunkStream(
            self._region,
            endpoint,
            voice,
            text_source,
            speed,
            sample_rate.value if sample_rate is not None else None,
        )
