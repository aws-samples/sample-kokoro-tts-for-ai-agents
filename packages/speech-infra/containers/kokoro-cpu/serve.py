"""Kokoro-82M TTS serving.

Provides:
- GET /ping: health check
- POST /invocations: synchronous TTS inference
- WS /invocations-bidirectional-stream: streaming TTS over WebSocket

Uses kokoro-onnx with TensorRT execution provider and misaki G2P.
Requests are processed sequentially — the Kokoro ONNX model only
supports batch_size=1 (variable-length duration expansion).
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
import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route, WebSocketRoute
from starlette.websockets import WebSocket, WebSocketDisconnect

MODEL_DIR = os.environ.get("MODEL_DIR", "/app/models")
MODEL_NAME = os.environ.get("MODEL_NAME", "kokoro-v1.0.onnx")
VOICES_NAME = os.environ.get("VOICES_NAME", "voices-v1.0.bin")
MAX_REQUEST_AGE_S = float(os.environ.get("MAX_REQUEST_AGE_S", "56"))
SAMPLE_RATE = 24000
DEFAULT_VOICE = "af_heart"

_logger = logging.getLogger("kokoro_serve")

_kokoro = None
_g2p = None
_inference_lock: asyncio.Lock | None = None


def _get_model():
    global _kokoro
    if _kokoro is None:
        from kokoro_onnx import Kokoro

        model_path = os.path.join(MODEL_DIR, MODEL_NAME)
        voices_path = os.path.join(MODEL_DIR, VOICES_NAME)
        _kokoro = Kokoro(model_path, voices_path)
        _logger.info("Kokoro model loaded: %s", model_path)
    return _kokoro


def _get_g2p():
    global _g2p
    if _g2p is None:
        from misaki import en, espeak

        fallback = espeak.EspeakFallback(british=False)
        _g2p = en.G2P(trf=False, british=False, fallback=fallback)
        _logger.info("Misaki G2P initialized with espeak fallback")
    return _g2p


def _phonemize(text: str) -> str:
    g2p = _get_g2p()
    phonemes, _ = g2p(text)
    return phonemes


def _synthesize(model, phonemes: str, voice: str, speed: float):
    return model.create(phonemes, voice, speed=speed, is_phonemes=True)


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
    _get_model()
    _get_g2p()
    _logger.info("Model and G2P loaded, ready to serve")
    yield


async def ping(request: Request) -> Response:
    return Response(content="OK", status_code=200)


async def invocations(request: Request) -> Response:
    """Synchronous TTS: accept text, return WAV audio."""
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
    phonemes = _phonemize(text)

    model = _get_model()
    async with _inference_lock:
        samples, sr = await asyncio.get_event_loop().run_in_executor(
            None, _synthesize, model, phonemes, voice, speed
        )

    wav_bytes = _samples_to_wav(samples, sr)
    elapsed = time.perf_counter() - t0
    audio_duration = len(samples) / sr

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
    """WebSocket handler for streaming TTS."""
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
            phonemes = _phonemize(text)

            model = _get_model()
            async with _inference_lock:
                samples, sr = await asyncio.get_event_loop().run_in_executor(
                    None, _synthesize, model, phonemes, voice, speed
                )

            pcm = (samples * 32767).astype(np.int16).tobytes()
            await websocket.send_bytes(pcm)

            audio_duration = len(samples) / sr
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
