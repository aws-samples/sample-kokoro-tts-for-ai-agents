"""Kokoro-82M TTS serving (PyTorch GPU).

Provides:
- GET /ping: health check
- POST /invocations: synchronous TTS inference
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
from starlette.responses import JSONResponse, Response
from starlette.routing import Route, WebSocketRoute
from starlette.websockets import WebSocket, WebSocketDisconnect

MAX_REQUEST_AGE_S = float(os.environ.get("MAX_REQUEST_AGE_S", "56"))
SAMPLE_RATE = 24000
DEFAULT_VOICE = "af_heart"

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


async def invocations(request: Request) -> Response:
    body = json.loads(await request.body())
    text = body.get("text", "")
    voice = body.get("voice", DEFAULT_VOICE)
    speed = body.get("speed", 1.0)

    request_ts = body.get("request_timestamp")
    if request_ts is not None and time.time() - request_ts > MAX_REQUEST_AGE_S:
        return JSONResponse(status_code=408, content={"error": "request_stale"})

    if not text:
        return JSONResponse(status_code=400, content={"error": "text is required"})

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

            async with _inference_lock:
                samples = await asyncio.get_event_loop().run_in_executor(
                    None, _synthesize_full, text, voice, speed
                )

            pcm = (samples * 32767).astype(np.int16).tobytes()
            await websocket.send_bytes(pcm)

            audio_duration = len(samples) / SAMPLE_RATE
            elapsed = time.monotonic() - t0

            await websocket.send_text(
                json.dumps(
                    {
                        "type": "synthesis_complete",
                        "request_id": request_id,
                        "total_duration_s": round(audio_duration, 3),
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
