"""Kokoro-82M TTS serving (PyTorch GPU).

Provides:
- GET /ping: health check
- POST /invocations: streaming TTS inference (per-sentence chunks)
- WS /invocations-bidirectional-stream: streaming TTS over WebSocket

Uses the PyTorch `kokoro` package with KPipeline for native CUDA
inference on A10G GPU. Single model instance with asyncio.Lock
serialization — the model is fast enough (0.12s/inference) that
multi-session adds negligible benefit.
"""

import asyncio
import json
import logging
import os
import struct
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

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
WARMUP_TEXT = os.environ.get("WARMUP_TEXT", "Warming up.")

_logger = logging.getLogger("kokoro_serve")

_pipeline: KPipeline | None = None
_inference_lock: asyncio.Lock | None = None

#: Container start, for `elapsed_s` on the stage markers below. Taken from the
#: entrypoint via `CONTAINER_START_EPOCH` where one exists; this container is
#: started directly by `CMD`, so process start is the earliest point observable
#: from inside. Image pull is bounded externally by the log stream's first event.
_STAGE_EPOCH = float(os.environ.get("CONTAINER_START_EPOCH") or time.time())


def _stage(name: str) -> None:
    """Emit a startup-stage marker, parsed by `tts-bench ttotal`.

    T_total is the scaling lag the whole capacity plan is most sensitive to, and
    it is only actionable when attributed to a stage. The format is
    byte-identical across all four containers so a single parser reads them all;
    `time.strftime` rather than `datetime.UTC` because this image and
    chatterbox's are ubuntu22.04-based (Python 3.10, no `datetime.UTC`), and
    the emitters must not diverge between containers.

    `elapsed_s` runs from container start, so a log stream whose earlier lines
    aged out is still partially usable.
    """
    now = time.time()
    stamp = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(now))
    print(
        f"=== STAGE {name} t={stamp}.{int(now % 1 * 1000):03d}Z "
        f"elapsed_s={now - _STAGE_EPOCH:.3f} ===",
        flush=True,
    )


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


def _warmup() -> int:
    """Run one discarded synthesis so the first real request does not pay for JIT.

    The pipeline being loaded is not the same as it being ready: CUDA kernel
    autotune happens on first inference, so without this the first caller after a
    scale-out absorbs it. T_total should measure time-to-serving-*good*-traffic,
    which makes paying that cost here — before uvicorn accepts connections — the
    right trade.

    Returns:
        Samples generated, for the log line. Zero is not fatal: a container that
        cannot warm up can still serve, and failing startup over it would turn a
        latency problem into an outage.
    """
    samples = _synthesize_full(WARMUP_TEXT, DEFAULT_VOICE, 1.0)
    return int(samples.size)


@asynccontextmanager
async def _lifespan(app: Starlette) -> AsyncGenerator[None, None]:
    global _inference_lock
    _inference_lock = asyncio.Lock()

    _stage("framework_init")
    _load_pipeline()
    _stage("weights_ready")

    try:
        count = _warmup()
        print(f"[kokoro] Warm-up complete, {count} samples discarded", flush=True)
    except Exception:
        # Logged and swallowed: see _warmup. The marker is still emitted so
        # ttotal's stage sequence stays complete and the warm-up cost is visible
        # even when the warm-up itself failed.
        _logger.exception("Warm-up inference failed; serving anyway")
    _stage("warmup_done")

    _stage("ready")
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
    text: str, voice: str, speed: float
) -> AsyncGenerator[bytes, None]:
    """Yield WAV header + PCM chunks per KPipeline segment as they are produced.

    The inference lock is held for the whole stream: KPipeline is a single stateful
    instance, so two interleaved segment generators would corrupt each other. The
    model is fast enough that serializing whole requests costs nothing meaningful.
    """
    header_sent = False
    wav_header = _wav_header_placeholder()

    loop = asyncio.get_event_loop()
    pipeline = _load_pipeline()

    async with _inference_lock:
        segments = iter(pipeline(text, voice=voice, speed=speed))
        while True:
            audio = await loop.run_in_executor(None, _next_segment, segments)
            if audio is None:
                break
            pcm = (audio * 32767).astype(np.int16).tobytes()
            if not header_sent:
                yield wav_header + pcm
                header_sent = True
            else:
                yield pcm


async def _invocations_sync(text: str, voice: str, speed: float) -> Response:
    """Original synchronous path: full WAV in one response."""
    t0 = time.perf_counter()

    async with _inference_lock:
        samples = await asyncio.get_event_loop().run_in_executor(
            None, _synthesize_full, text, voice, speed
        )

    wav_bytes = _samples_to_wav(samples, SAMPLE_RATE)
    elapsed = time.perf_counter() - t0
    audio_duration = len(samples) / SAMPLE_RATE

    return Response(
        content=wav_bytes,
        media_type="audio/wav",
        headers={
            "X-Audio-Duration": f"{audio_duration:.3f}",
            "X-Inference-Time-Ms": str(int(elapsed * 1000)),
            "X-RTF": f"{elapsed / audio_duration:.3f}" if audio_duration > 0 else "0",
            "X-Characters-Processed": str(len(text)),
        },
    )


async def invocations(request: Request) -> Response:
    """TTS: accept text, return WAV audio (streaming by default)."""
    body = json.loads(await request.body())
    text = body.get("text", "")
    voice = body.get("voice", DEFAULT_VOICE)
    speed = body.get("speed", 1.0)
    use_stream = body.get("stream", True)

    request_ts = body.get("request_timestamp")
    if request_ts is not None and time.time() - request_ts > MAX_REQUEST_AGE_S:
        return JSONResponse(status_code=408, content={"error": "request_stale"})

    if not text:
        return JSONResponse(status_code=400, content={"error": "text is required"})

    if not use_stream:
        return await _invocations_sync(text, voice, speed)

    return StreamingResponse(
        _stream_sentences_generator(text, voice, speed),
        media_type="audio/wav",
    )


async def _receive_message(websocket: WebSocket) -> str:
    """Read one client frame as text, whatever frame type it arrived as.

    SageMaker's bidirectional transport forwards ``RequestPayloadPart`` as a
    *binary* WebSocket frame, so ``receive_text()`` raises ``KeyError: 'text'``
    on every request from ``invoke_endpoint_with_bidirectional_stream`` — the
    only way this endpoint is invoked in production. Browsers and the local
    test client send text frames. Accept both rather than picking one.
    """
    message = await websocket.receive()
    if message["type"] == "websocket.disconnect":
        raise WebSocketDisconnect(message.get("code", 1000))
    payload = message.get("text")
    if payload is None:
        payload = message.get("bytes", b"").decode("utf-8")
    return payload


async def bidirectional_stream(websocket: WebSocket) -> None:
    await websocket.accept()

    try:
        while True:
            raw = await _receive_message(websocket)
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
