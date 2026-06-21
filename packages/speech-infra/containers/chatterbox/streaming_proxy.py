"""Streaming proxy for Chatterbox-Turbo TTS via chatterbox-vllm.

Uses the chatterbox-vllm library which runs the T3 autoregressive stage
through vLLM (continuous batching) and S3Gen+HiFiGAN as post-processing.

Provides:
- GET /ping: health check
- POST /invocations: synchronous TTS (text + voice_id -> WAV)
- WS /invocations-bidirectional-stream: streaming TTS over WebSocket
"""

import json
import logging
import os
import struct
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import torch
import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route, WebSocketRoute
from starlette.websockets import WebSocket, WebSocketDisconnect

VOICES_DIR = os.environ.get("VOICES_DIR", "/app/voices")
DEFAULT_VOICE = os.environ.get("DEFAULT_VOICE", "female_shadowheart4")
MODEL_DIR = os.environ.get("MODEL_DIR", "/app/model")
MAX_REQUEST_AGE_S = float(os.environ.get("MAX_REQUEST_AGE_S", "51"))
SAMPLE_RATE = 24000

_logger = logging.getLogger("chatterbox_proxy")
_model = None


def _get_model():
    global _model
    if _model is None:
        from chatterbox_vllm.tts import ChatterboxTTS

        gpu_util = float(os.environ.get("GPU_MEMORY_UTILIZATION", "0.7"))
        max_len = int(os.environ.get("MAX_MODEL_LEN", "1000"))

        model_dir = Path(MODEL_DIR)
        if (model_dir / "t3_cfg.safetensors").exists():
            _model = ChatterboxTTS.from_local(
                model_dir,
                max_model_len=max_len,
                gpu_memory_utilization=gpu_util,
                enforce_eager=True,
            )
        else:
            _model = ChatterboxTTS.from_pretrained(
                gpu_memory_utilization=gpu_util,
                max_model_len=max_len,
                enforce_eager=True,
            )
        _logger.info("ChatterboxTTS loaded (gpu_util=%.2f, max_len=%d)", gpu_util, max_len)
    return _model


def _resolve_voice(voice_id: str) -> str:
    """Map voice_id to reference audio file path."""
    for ext in (".flac", ".wav", ".mp3", ".ogg"):
        path = os.path.join(VOICES_DIR, f"{voice_id}{ext}")
        if os.path.exists(path):
            return path
    available = [f.rsplit(".", 1)[0] for f in os.listdir(VOICES_DIR) if not f.startswith(".")]
    raise FileNotFoundError(f"Voice '{voice_id}' not found in {VOICES_DIR}. Available: {available}")


def _tensor_to_wav(wav_tensor: torch.Tensor, sample_rate: int) -> bytes:
    """Convert torch audio tensor to WAV bytes."""
    audio_np = wav_tensor.squeeze().cpu().numpy()
    pcm = (audio_np * 32767).astype(np.int16).tobytes()
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


async def ping(request: Request) -> Response:
    return Response(content="OK", status_code=200)


async def invocations(request: Request) -> Response:
    """Synchronous TTS: accept text + voice_id, return WAV audio."""
    body = json.loads(await request.body())
    text = body.get("text", "")
    voice_id = body.get("voice", DEFAULT_VOICE)

    request_ts = body.get("request_timestamp")
    if request_ts is not None and time.time() - request_ts > MAX_REQUEST_AGE_S:
        return JSONResponse(status_code=408, content={"error": "request_stale"})

    if not text:
        return JSONResponse(status_code=400, content={"error": "text is required"})

    try:
        audio_path = _resolve_voice(voice_id)
    except FileNotFoundError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

    t0 = time.perf_counter()
    model = _get_model()
    wav = model.generate(text, audio_prompt_path=audio_path)[0]
    elapsed = time.perf_counter() - t0

    wav_bytes = _tensor_to_wav(wav, SAMPLE_RATE)
    audio_duration = wav.shape[-1] / SAMPLE_RATE

    return Response(
        content=wav_bytes,
        media_type="audio/wav",
        headers={
            "X-Audio-Duration": f"{audio_duration:.3f}",
            "X-Inference-Time-Ms": str(int(elapsed * 1000)),
            "X-RTF": f"{elapsed / audio_duration:.3f}" if audio_duration > 0 else "0",
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
            voice_id = msg.get("voice", DEFAULT_VOICE)
            request_id = msg.get("request_id", "unknown")

            if not text:
                await websocket.send_text(
                    json.dumps(
                        {"type": "error", "request_id": request_id, "message": "text required"}
                    )
                )
                continue

            try:
                audio_path = _resolve_voice(voice_id)
            except FileNotFoundError as e:
                await websocket.send_text(
                    json.dumps({"type": "error", "request_id": request_id, "message": str(e)})
                )
                continue

            await websocket.send_text(
                json.dumps({"type": "synthesis_start", "request_id": request_id})
            )

            t0 = time.monotonic()
            model = _get_model()
            wav = model.generate(text, audio_prompt_path=audio_path)[0]

            audio_np = wav.squeeze().cpu().numpy()
            pcm = (audio_np * 32767).astype(np.int16).tobytes()
            await websocket.send_bytes(pcm)

            audio_duration = len(audio_np) / SAMPLE_RATE
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


@asynccontextmanager
async def lifespan(app: Starlette) -> AsyncGenerator[None, None]:
    _logger.info("Preloading model at startup...")
    _get_model()
    _logger.info("Model preloaded and ready for inference")
    yield


app = Starlette(
    routes=[
        Route("/ping", ping, methods=["GET"]),
        Route("/invocations", invocations, methods=["POST"]),
        WebSocketRoute("/invocations-bidirectional-stream", bidirectional_stream),
    ],
    lifespan=lifespan,
)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")
