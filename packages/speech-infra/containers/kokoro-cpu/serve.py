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
WARMUP_TEXT = os.environ.get("WARMUP_TEXT", "Warming up.")

_logger = logging.getLogger("kokoro_serve")

_kokoro = None
_g2p = None
_inference_lock: asyncio.Lock | None = None

#: See the identical block in ../kokoro/serve.py. Each container is its own
#: Docker build context, so this cannot be imported from a shared module until
#: the DockerImageAsset context changes.
_STAGE_EPOCH = float(os.environ.get("CONTAINER_START_EPOCH") or time.time())


def _stage(name: str) -> None:
    """Emit a startup-stage marker, parsed by `tts-bench ttotal`."""
    now = time.time()
    stamp = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(now))
    print(
        f"=== STAGE {name} t={stamp}.{int(now % 1 * 1000):03d}Z "
        f"elapsed_s={now - _STAGE_EPOCH:.3f} ===",
        flush=True,
    )


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


def _warmup() -> int:
    """Run one discarded synthesis, exercising both G2P and the ONNX session.

    Both are lazily initialized, so without this the first real caller pays for
    espeak dictionary loading and the ONNX provider's first-run graph
    optimization. Returns samples generated; see ../kokoro/serve.py:_warmup for
    why a failure here is not fatal.
    """
    model = _get_model()
    phonemes = _phonemize(WARMUP_TEXT)
    samples, _ = _synthesize(model, phonemes, DEFAULT_VOICE, 1.0)
    return int(np.asarray(samples).size)


@asynccontextmanager
async def _lifespan(app: Starlette) -> AsyncGenerator[None, None]:
    global _inference_lock
    _inference_lock = asyncio.Lock()

    _stage("framework_init")
    _get_model()
    _get_g2p()
    _stage("weights_ready")

    try:
        count = _warmup()
        _logger.info("Warm-up complete, %d samples discarded", count)
    except Exception:
        _logger.exception("Warm-up inference failed; serving anyway")
    _stage("warmup_done")

    _logger.info("Model and G2P loaded, ready to serve")
    _stage("ready")
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
    loop = asyncio.get_event_loop()

    # G2P is synchronous CPU work (espeak shells out), so running it inline would
    # block the event loop for every concurrent request - including /ping, which
    # SageMaker reads as an unhealthy instance. It is deliberately outside the
    # inference lock: phonemization is stateless, so it can overlap other requests'
    # synthesis and only the ONNX session needs serializing.
    phonemes = await loop.run_in_executor(None, _phonemize, text)

    model = _get_model()
    async with _inference_lock:
        samples, sr = await loop.run_in_executor(None, _synthesize, model, phonemes, voice, speed)

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
    """WebSocket handler for streaming TTS."""
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

            # Off the event loop for the same reason as the /invocations path.
            phonemes = await loop.run_in_executor(None, _phonemize, text)

            model = _get_model()
            async with _inference_lock:
                samples, sr = await loop.run_in_executor(
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
