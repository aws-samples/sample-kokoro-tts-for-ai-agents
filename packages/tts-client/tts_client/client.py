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
from typing import Any

import boto3
import botocore.exceptions
from aws_sdk_sagemaker_runtime_http2.client import SageMakerRuntimeHTTP2Client
from aws_sdk_sagemaker_runtime_http2.config import Config as Http2Config
from aws_sdk_sagemaker_runtime_http2.config import HTTPAuthSchemeResolver
from aws_sdk_sagemaker_runtime_http2.models import (
    InvokeEndpointWithBidirectionalStreamInput,
    RequestPayloadPart,
    RequestStreamEventPayloadPart,
    ResponseStreamEventInternalStreamFailure,
    ResponseStreamEventModelStreamError,
)
from botocore.config import Config
from smithy_aws_core.auth.sigv4 import SigV4AuthScheme

from tts_client.bidi_credentials import Boto3CredentialsResolver
from tts_client.errors import (
    ServerError,
    TTSClientError,
    TTSTimeoutError,
    raise_for_bidi_exception,
    raise_for_client_error,
    raise_for_error_frame,
)
from tts_client.types import AudioFormat, SynthesisRequest, SynthesisResult

#: Just above SageMaker's 60s invocation ceiling. A client timeout below the
#: server's own limit would report a client timeout for requests the server
#: was still entitled to finish.
DEFAULT_READ_TIMEOUT_S = 65.0

DEFAULT_CONNECT_TIMEOUT_S = 5.0

#: SageMaker bidirectional streaming is served on 8443, not the usual 443.
BIDI_PORT = 8443

#: Bidi returns raw 16-bit mono PCM with no RIFF header, so the rate cannot be
#: read off the payload. Every container defines SAMPLE_RATE = 24000.
BIDI_SAMPLE_RATE = 24000

_MEDIA_TYPES = {AudioFormat.WAV: "audio/wav", AudioFormat.MP3: "audio/mpeg"}


def wav_duration(data: bytes) -> float:
    """Audio duration in seconds, read from a WAV file's RIFF header."""
    if len(data) < 44 or data[:4] != b"RIFF":
        return 0.0
    sr: int = struct.unpack_from("<I", data, 24)[0]
    bits: int = struct.unpack_from("<H", data, 34)[0]
    channels: int = struct.unpack_from("<H", data, 22)[0]
    data_size = len(data) - 44
    return float(data_size / (sr * channels * (bits // 8)))


def _pcm_to_wav(pcm_bytes: bytes, sample_rate: int = BIDI_SAMPLE_RATE) -> bytes:
    """Wrap raw 16-bit mono PCM in a WAV header."""
    data_size = len(pcm_bytes)
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + data_size,
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
        data_size,
    )
    return header + pcm_bytes


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
        http2_config = Http2Config(
            endpoint_uri=f"https://runtime.sagemaker.{self._region}.amazonaws.com:{BIDI_PORT}",
            region=self._region,
            aws_credentials_identity_resolver=Boto3CredentialsResolver(),
            auth_scheme_resolver=HTTPAuthSchemeResolver(),
            auth_schemes={"aws.auth#sigv4": SigV4AuthScheme(service="sagemaker")},
        )
        client = SageMakerRuntimeHTTP2Client(config=http2_config)

        message = json.dumps(
            {
                "text": request.text,
                "voice": request.voice,
                "request_timestamp": request.request_timestamp
                if request.request_timestamp is not None
                else time.time(),
            }
        ).encode("utf-8")

        t0 = time.perf_counter()

        try:
            stream = await client.invoke_endpoint_with_bidirectional_stream(
                InvokeEndpointWithBidirectionalStreamInput(endpoint_name=endpoint)
            )
        except Exception as exc:  # noqa: BLE001 - the SDK's error taxonomy is open
            raise_for_bidi_exception(exc)
            raise AssertionError("unreachable") from exc

        try:
            await stream.input_stream.send(
                RequestStreamEventPayloadPart(value=RequestPayloadPart(bytes_=message))
            )
            # The input stream stays open until the audio is drained. Closing
            # it right after send makes SageMaker tear the connection down
            # before the container reads the payload.
            _, output_stream = await stream.await_output()

            audio_chunks: list[bytes] = []
            ttfab_ms: float | None = None

            while True:
                event = await output_stream.receive()
                if event is None:
                    break
                if isinstance(event, ResponseStreamEventModelStreamError):
                    raise ServerError(str(getattr(event.value, "message", "")))
                if isinstance(event, ResponseStreamEventInternalStreamFailure):
                    raise ServerError(str(getattr(event.value, "message", "")))

                payload = getattr(event, "value", None)
                chunk = getattr(payload, "bytes_", None)
                if not chunk:
                    continue

                # The containers multiplex JSON control frames and raw PCM on
                # one stream, distinguished only by whether the payload
                # starts with `{`. A chunk that starts that way but does not
                # decode is audio, not a malformed frame.
                decoded = _decode_control_frame(chunk) if chunk[0:1] == b"{" else None
                if decoded is not None:
                    frame_type = decoded.get("type")
                    if frame_type == "error":
                        raise_for_error_frame(str(decoded.get("message", "")))
                    if frame_type == "synthesis_complete":
                        break
                    continue

                if ttfab_ms is None:
                    ttfab_ms = (time.perf_counter() - t0) * 1000.0
                audio_chunks.append(chunk)
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
        pcm_bytes = b"".join(audio_chunks)
        if not pcm_bytes:
            raise TTSClientError("bidi stream completed with no audio bytes", http_status=200)

        audio = _pcm_to_wav(pcm_bytes)
        duration_s = len(pcm_bytes) / (BIDI_SAMPLE_RATE * 2)

        return SynthesisResult(
            audio_bytes=audio,
            audio_format=AudioFormat.WAV,
            sample_rate=BIDI_SAMPLE_RATE,
            duration_s=duration_s,
            latency_ms=latency_ms,
            ttfab_ms=ttfab_ms or latency_ms,
            chars=len(request.text),
            chunks=len(audio_chunks),
        )


def _decode_control_frame(chunk: bytes) -> dict[str, Any] | None:
    """Parse a JSON control frame, or ``None`` if it is not one.

    A chunk starting with ``{`` is *probably* JSON, but PCM can begin with
    byte ``0x7b`` too. Treating an undecodable chunk as a control frame would
    drop real audio, so this fails soft and the caller counts it as audio.
    """
    try:
        decoded = json.loads(chunk.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    return decoded if isinstance(decoded, dict) else None


async def _close_quietly(stream: Any) -> None:
    """Close a stream and its input half, swallowing teardown faults.

    A close error must not overwrite a result already produced or an
    exception already being raised.
    """
    input_stream = getattr(stream, "input_stream", None)
    for target in (input_stream, stream):
        if target is None:
            continue
        close = getattr(target, "close", None)
        if close is None:
            continue
        try:
            outcome = close()
            if hasattr(outcome, "__await__"):
                await outcome
        except Exception:  # noqa: BLE001 - teardown must never mask the real result
            pass
