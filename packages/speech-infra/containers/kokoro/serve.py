"""Kokoro-82M TTS serving (PyTorch GPU).

Provides:
- GET /ping: health check
- POST /invocations: streaming TTS inference (per-sentence chunks)
- WS /invocations-bidirectional-stream: streaming TTS over WebSocket

Uses the PyTorch `kokoro` package with KPipeline for native CUDA
inference on A10G GPU. Single model instance with asyncio.Lock
serialization — the model is fast enough (0.12s/inference) that
multi-session adds negligible benefit.

/invocations selects its wire shape from two body fields, both defaulting to
today's behaviour so existing callers are unaffected:
- transport: "binary" (raw chunked bytes) | "sse" (text/event-stream)
- format:    "wav" (raw PCM frames) | "mp3" (48 kbps mono)

SSE exists because audio reaches the browser over an AgentCore relay that
already multiplexes other agent event types on one stream. SSE is UTF-8 only,
so audio is base64-encoded (+33%); MP3 keeps that affordable and every prefix
of an MP3 frame stream is independently decodable, which is what lets the
client start playing before synthesis finishes.
"""

import asyncio
import base64
import json
import logging
import os
import struct
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import lameenc
import numpy as np
import torch
import uvicorn
from kokoro import KPipeline
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route, WebSocketRoute
from starlette.websockets import WebSocket, WebSocketDisconnect

MAX_REQUEST_AGE_S = float(os.environ.get("MAX_REQUEST_AGE_S", "56"))
SAMPLE_RATE = 24000
DEFAULT_VOICE = "af_heart"
MP3_BITRATE_KBPS = 48
MP3_QUALITY = 2

FORMAT_WAV = "wav"
FORMAT_MP3 = "mp3"
TRANSPORT_BINARY = "binary"
TRANSPORT_SSE = "sse"

_MEDIA_TYPES = {FORMAT_WAV: "audio/wav", FORMAT_MP3: "audio/mpeg"}

_logger = logging.getLogger("kokoro_serve")

_pipeline: KPipeline | None = None
_inference_lock: asyncio.Lock | None = None


def _load_pipeline() -> KPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(
            f"[kokoro] Pipeline loaded, device={device}, CUDA={torch.cuda.is_available()}",
            flush=True,
        )
        if torch.cuda.is_available():
            print(f"[kokoro] GPU: {torch.cuda.get_device_name(0)}", flush=True)
    return _pipeline


def _synthesize_full(text: str, voice: str, speed: float) -> np.ndarray:
    pipeline = _load_pipeline()
    chunks = []
    for _, _, audio in pipeline(text, voice=voice, speed=speed):
        if audio is not None:
            chunks.append(audio)
    if not chunks:
        return np.array([], dtype=np.float32)
    return np.concatenate(chunks)


def _samples_to_wav(samples: np.ndarray, sample_rate: int) -> bytes:
    pcm = (samples * 32767).astype(np.int16).tobytes()
    data_size = len(pcm)
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
    return header + pcm


@asynccontextmanager
async def _lifespan(app: Starlette) -> AsyncGenerator[None, None]:
    global _inference_lock
    _inference_lock = asyncio.Lock()
    _load_pipeline()
    yield


async def ping(request: Request) -> Response:
    return Response(content="OK", status_code=200)


def _wav_header_placeholder(sample_rate: int = SAMPLE_RATE) -> bytes:
    """44-byte WAV header with 0xFFFFFFFF data size for streaming."""
    return struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        0xFFFFFFFF,
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
        0xFFFFFFFF,
    )


def _to_numpy(audio: object) -> np.ndarray:
    """Convert a KPipeline audio segment (torch.Tensor) to a float32 numpy array."""
    if hasattr(audio, "detach"):
        return audio.detach().cpu().numpy()
    return np.asarray(audio, dtype=np.float32)


def _samples_to_pcm(samples: np.ndarray) -> bytes:
    return (samples * 32767).astype(np.int16).tobytes()


class Mp3StreamEncoder:
    """Incremental MP3 encoder for one request.

    lameenc is used instead of an ffmpeg subprocess because ffmpeg buffers
    ~1.5-2.1s of audio before emitting its first byte, which would defeat
    progressive playback for short replies. This emits after ~0.5s of audio at
    roughly 3ms per call.

    A single instance is valid for exactly one utterance: lameenc raises
    "Encoder not initialised" if encode() is called after flush().
    """

    def __init__(self, sample_rate: int = SAMPLE_RATE) -> None:
        self._encoder = lameenc.Encoder()
        self._encoder.set_bit_rate(MP3_BITRATE_KBPS)
        self._encoder.set_in_sample_rate(sample_rate)
        self._encoder.set_channels(1)
        self._encoder.set_quality(MP3_QUALITY)
        self._encoder.silence()
        self._closed = False

    def encode(self, samples: np.ndarray) -> bytes:
        """Encode one segment. Returns b"" when LAME is still filling its frame buffer."""
        if self._closed:
            raise RuntimeError("Mp3StreamEncoder already flushed")
        return bytes(self._encoder.encode(_samples_to_pcm(samples)))

    def flush(self) -> bytes:
        """Emit LAME's remaining frames.

        This tail carries real audio, not just padding, and for utterances short
        enough that encode() never returned anything it carries *all* of the
        audio. Callers must always send it before ending the stream.
        """
        if self._closed:
            return b""
        self._closed = True
        return bytes(self._encoder.flush())


def _generate_sentences(text: str, voice: str, speed: float) -> list[np.ndarray]:
    """Generate all sentence audio chunks (runs in executor)."""
    pipeline = _load_pipeline()
    results = []
    for _, _, audio in pipeline(text, voice=voice, speed=speed):
        if audio is not None:
            results.append(_to_numpy(audio))
    return results


def _next_segment(segments: object) -> np.ndarray | None:
    """Advance the KPipeline generator to the next audio segment (runs in executor).

    Returns the segment as float32 numpy, or None once the generator is exhausted.
    The passed iterator resumes where it left off, so repeated calls walk the
    segments one at a time - enabling true incremental streaming.
    """
    for _, _, audio in segments:
        if audio is not None:
            return _to_numpy(audio)
    return None


async def _stream_sentences_generator(
    text: str, voice: str, speed: float, audio_format: str = FORMAT_WAV, stats: dict | None = None
) -> AsyncGenerator[bytes, None]:
    """Yield encoded audio chunks per KPipeline segment as they are produced.

    For WAV this is a placeholder header followed by raw PCM; for MP3 it is a
    bare frame stream, so any prefix the client has received is playable.

    The inference lock is held for the whole stream: KPipeline is a single stateful
    instance, so two interleaved segment generators would corrupt each other. The
    model is fast enough that serializing whole requests costs nothing meaningful.
    """
    header_sent = False
    encoder = Mp3StreamEncoder() if audio_format == FORMAT_MP3 else None
    sample_count = 0

    loop = asyncio.get_event_loop()
    pipeline = _load_pipeline()

    async with _inference_lock:
        segments = iter(pipeline(text, voice=voice, speed=speed))
        while True:
            audio = await loop.run_in_executor(None, _next_segment, segments)
            if audio is None:
                break
            sample_count += len(audio)
            if encoder is None:
                pcm = _samples_to_pcm(audio)
                if not header_sent:
                    yield _wav_header_placeholder() + pcm
                    header_sent = True
                else:
                    yield pcm
            else:
                # Empty output is normal: LAME is still filling its frame buffer.
                chunk = encoder.encode(audio)
                if chunk:
                    yield chunk

        if encoder is not None:
            tail = encoder.flush()
            if tail:
                yield tail

    if stats is not None:
        stats["samples"] = sample_count


def _sse_frame(event: str, payload: dict) -> bytes:
    return f"event: {event}\ndata: {json.dumps(payload)}\n\n".encode()


async def _sse_generator(
    text: str, voice: str, speed: float, audio_format: str, request_id: str
) -> AsyncGenerator[bytes, None]:
    """Wrap the audio byte stream in the AgentCore relay's SSE event contract.

    start is emitted before inference begins so the client can build its decoder
    while the model runs. A failure mid-stream becomes an in-band `error` event:
    on the raw binary path the same failure just truncates the chunked body and
    reaches the caller as an opaque ModelStreamError.
    """
    yield _sse_frame(
        "audio_stream_start",
        {
            "request_id": request_id,
            "format": audio_format,
            "voice": voice,
            "sample_rate": SAMPLE_RATE,
        },
    )

    stats: dict = {}
    seq = 0
    try:
        async for chunk in _stream_sentences_generator(text, voice, speed, audio_format, stats):
            yield _sse_frame(
                "audio_chunk",
                {
                    "request_id": request_id,
                    "seq": seq,
                    "data": base64.b64encode(chunk).decode("ascii"),
                },
            )
            seq += 1
    except Exception as e:
        _logger.exception("SSE synthesis failed")
        yield _sse_frame("error", {"request_id": request_id, "message": str(e)})
        return

    yield _sse_frame(
        "audio_stream_end",
        {
            "request_id": request_id,
            "total_chunks": seq,
            "duration_s": round(stats.get("samples", 0) / SAMPLE_RATE, 3),
        },
    )


def _samples_to_mp3(samples: np.ndarray) -> bytes:
    encoder = Mp3StreamEncoder()
    return encoder.encode(samples) + encoder.flush()


async def _invocations_sync(
    text: str, voice: str, speed: float, audio_format: str = FORMAT_WAV
) -> Response:
    """Original synchronous path: complete audio file in one response."""
    t0 = time.perf_counter()

    async with _inference_lock:
        samples = await asyncio.get_event_loop().run_in_executor(
            None, _synthesize_full, text, voice, speed
        )

    if audio_format == FORMAT_MP3:
        content = _samples_to_mp3(samples)
    else:
        content = _samples_to_wav(samples, SAMPLE_RATE)
    elapsed = time.perf_counter() - t0
    audio_duration = len(samples) / SAMPLE_RATE

    return Response(
        content=content,
        media_type=_MEDIA_TYPES[audio_format],
        headers={
            "X-Audio-Duration": f"{audio_duration:.3f}",
            "X-Inference-Time-Ms": str(int(elapsed * 1000)),
            "X-RTF": f"{elapsed / audio_duration:.3f}" if audio_duration > 0 else "0",
            "X-Characters-Processed": str(len(text)),
        },
    )


async def invocations(request: Request) -> Response:
    """TTS: accept text, return audio (chunked WAV by default)."""
    body = json.loads(await request.body())
    text = body.get("text", "")
    voice = body.get("voice", DEFAULT_VOICE)
    speed = body.get("speed", 1.0)
    use_stream = body.get("stream", True)
    audio_format = body.get("format", FORMAT_WAV)
    transport = body.get("transport", TRANSPORT_BINARY)

    request_ts = body.get("request_timestamp")
    if request_ts is not None and time.time() - request_ts > MAX_REQUEST_AGE_S:
        return JSONResponse(status_code=408, content={"error": "request_stale"})

    if not text:
        return JSONResponse(status_code=400, content={"error": "text is required"})

    if audio_format not in _MEDIA_TYPES:
        return JSONResponse(
            status_code=400,
            content={"error": f"format must be one of {sorted(_MEDIA_TYPES)}"},
        )

    if transport not in (TRANSPORT_BINARY, TRANSPORT_SSE):
        return JSONResponse(
            status_code=400,
            content={"error": f"transport must be '{TRANSPORT_BINARY}' or '{TRANSPORT_SSE}'"},
        )

    if transport == TRANSPORT_SSE:
        return StreamingResponse(
            _sse_generator(text, voice, speed, audio_format, body.get("request_id", "unknown")),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    if not use_stream:
        return await _invocations_sync(text, voice, speed, audio_format)

    return StreamingResponse(
        _stream_sentences_generator(text, voice, speed, audio_format),
        media_type=_MEDIA_TYPES[audio_format],
    )


async def bidirectional_stream(websocket: WebSocket) -> None:
    await websocket.accept()

    try:
        while True:
            raw = await websocket.receive_text()
            msg = json.loads(raw)

            if msg.get("type") == "close":
                break

            text = msg.get("text", "")
            voice = msg.get("voice", DEFAULT_VOICE)
            speed = msg.get("speed", 1.0)
            request_id = msg.get("request_id", "unknown")

            if not text:
                await websocket.send_text(
                    json.dumps(
                        {"type": "error", "request_id": request_id, "message": "text required"}
                    )
                )
                continue

            await websocket.send_text(
                json.dumps({"type": "synthesis_start", "request_id": request_id})
            )

            t0 = time.monotonic()
            loop = asyncio.get_event_loop()

            async with _inference_lock:
                sentence_audios = await loop.run_in_executor(
                    None, _generate_sentences, text, voice, speed
                )

            cumulative_duration = 0.0
            for audio in sentence_audios:
                pcm = (audio * 32767).astype(np.int16).tobytes()
                await websocket.send_bytes(pcm)
                segment_duration = len(audio) / SAMPLE_RATE
                cumulative_duration += segment_duration

            elapsed = time.monotonic() - t0

            await websocket.send_text(
                json.dumps(
                    {
                        "type": "synthesis_complete",
                        "request_id": request_id,
                        "total_duration_s": round(cumulative_duration, 3),
                        "elapsed_s": round(elapsed, 3),
                    }
                )
            )

    except WebSocketDisconnect:
        pass
    except Exception as e:
        _logger.exception("WebSocket error")
        try:
            await websocket.send_text(
                json.dumps({"type": "error", "request_id": "unknown", "message": str(e)})
            )
        except Exception:
            pass


app = Starlette(
    routes=[
        Route("/ping", ping, methods=["GET"]),
        Route("/invocations", invocations, methods=["POST"]),
        WebSocketRoute("/invocations-bidirectional-stream", bidirectional_stream),
    ],
    lifespan=_lifespan,
)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")
