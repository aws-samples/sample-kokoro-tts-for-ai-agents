"""Low-level SageMaker bidirectional-streaming wire protocol.

Shared leaf module: everything here is a primitive for one message on one
HTTP/2 bidi session (open a stream, build a message, drain one response,
close cleanly). Both :mod:`tts_client.client` (single-shot ``synthesize_bidi``)
and :mod:`tts_client.streaming` (multi-message ``synthesize_bidi_stream``)
are peers built on top of this layer -- neither imports the other, which is
what keeps the module graph acyclic once both need these primitives.
"""

from __future__ import annotations

import json
import struct
import time
from typing import Any, NamedTuple

from aws_sdk_sagemaker_runtime_http2.client import SageMakerRuntimeHTTP2Client
from aws_sdk_sagemaker_runtime_http2.config import Config as Http2Config
from aws_sdk_sagemaker_runtime_http2.config import HTTPAuthSchemeResolver
from aws_sdk_sagemaker_runtime_http2.models import (
    InvokeEndpointWithBidirectionalStreamInput,
    ResponseStreamEventInternalStreamFailure,
    ResponseStreamEventModelStreamError,
)
from smithy_aws_core.auth.sigv4 import SigV4AuthScheme

from tts_client.bidi_credentials import Boto3CredentialsResolver
from tts_client.errors import ServerError, raise_for_bidi_exception, raise_for_error_frame

#: SageMaker bidirectional streaming is served on 8443, not the usual 443.
BIDI_PORT = 8443

#: Bidi returns raw 16-bit mono PCM with no RIFF header, so the rate cannot be
#: read off the payload. Every container defines SAMPLE_RATE = 24000.
BIDI_SAMPLE_RATE = 24000


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


def _build_bidi_message(
    text: str,
    voice: str,
    request_timestamp: float | None,
    *,
    speed: float | None = None,
    sample_rate: int | None = None,
) -> bytes:
    """Serialize one bidi text message.

    ``speed``/``sample_rate`` are omitted unless explicitly passed, matching
    the wire shape :meth:`TTSClient.synthesize_bidi` has always sent. Kokoro's
    bidi handler defaults an absent ``speed`` to 1.0 and an absent
    ``sample_rate`` to its native rate server-side, so omitting either is not
    a behavior change. Plain ``int``, not :class:`tts_client.types.SampleRate`
    -- this is a leaf module with no dependency on that higher-level type;
    callers pass the already-validated ``.value``.
    """
    body: dict[str, Any] = {
        "text": text,
        "voice": voice,
        "request_timestamp": request_timestamp if request_timestamp is not None else time.time(),
    }
    if speed is not None:
        body["speed"] = speed
    if sample_rate is not None:
        body["sample_rate"] = sample_rate
    return json.dumps(body).encode("utf-8")


async def _open_bidi_stream(region: str, endpoint: str) -> Any:
    """Open a fresh HTTP/2 bidirectional stream to ``endpoint``.

    See :mod:`tts_client.client`'s module docstring for why a fresh client is
    built per call rather than sharing one across ``TTSClient`` instances.
    """
    http2_config = Http2Config(
        endpoint_uri=f"https://runtime.sagemaker.{region}.amazonaws.com:{BIDI_PORT}",
        region=region,
        aws_credentials_identity_resolver=Boto3CredentialsResolver(),
        auth_scheme_resolver=HTTPAuthSchemeResolver(),
        auth_schemes={"aws.auth#sigv4": SigV4AuthScheme(service="sagemaker")},
    )
    client = SageMakerRuntimeHTTP2Client(config=http2_config)
    try:
        return await client.invoke_endpoint_with_bidirectional_stream(
            InvokeEndpointWithBidirectionalStreamInput(endpoint_name=endpoint)
        )
    except Exception as exc:  # noqa: BLE001 - the SDK's error taxonomy is open
        raise_for_bidi_exception(exc)
        raise AssertionError("unreachable") from exc


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


class _DrainResult(NamedTuple):
    """One message's worth of drained bidi output.

    ``ended_cleanly`` is ``True`` only for an explicit ``synthesis_complete``
    frame, ``False`` when the stream ended with ``event is None`` instead. A
    single-message caller can ignore this (the "no audio bytes" check is its
    safety net), but a multi-message session needs it: a connection dropped
    between chunk 2 and chunk 3 must raise rather than be mistaken for "chunk
    2 done, send chunk 3."
    """

    pcm_bytes: bytes
    chunk_count: int
    ttfab_ms: float | None
    ended_cleanly: bool


async def _drain_until_complete(output_stream: Any, t0: float) -> _DrainResult:
    """Read one message's response frames until ``synthesis_complete`` or EOF."""
    audio_chunks: list[bytes] = []
    ttfab_ms: float | None = None
    ended_cleanly = False

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

        # The containers multiplex JSON control frames and raw PCM on one
        # stream, distinguished only by whether the payload starts with `{`.
        # A chunk that starts that way but does not decode is audio, not a
        # malformed frame.
        decoded = _decode_control_frame(chunk) if chunk[0:1] == b"{" else None
        if decoded is not None:
            frame_type = decoded.get("type")
            if frame_type == "error":
                raise_for_error_frame(str(decoded.get("message", "")))
            if frame_type == "synthesis_complete":
                ended_cleanly = True
                break
            continue

        if ttfab_ms is None:
            ttfab_ms = (time.perf_counter() - t0) * 1000.0
        audio_chunks.append(chunk)

    return _DrainResult(
        pcm_bytes=b"".join(audio_chunks),
        chunk_count=len(audio_chunks),
        ttfab_ms=ttfab_ms,
        ended_cleanly=ended_cleanly,
    )


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
