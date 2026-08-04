"""Streaming proxy for Chatterbox Turbo TTS.

Uses the standard chatterbox-tts library with AlignmentStreamAnalyzer for
repetition-free generation. Single model instance with asyncio.Lock
serialization — the model uses ~18GB VRAM on A10G, so only one instance
fits and parallelism provides no benefit.

Provides:
- GET /ping: health check
- POST /invocations: streaming TTS (per-sentence chunks)
- WS /invocations-bidirectional-stream: streaming TTS over WebSocket
"""

import asyncio
import json
import logging
import os
import re
import struct
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import torch
import uvicorn
from chatterbox.tts_turbo import ChatterboxTurboTTS, Conditionals
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route, WebSocketRoute
from starlette.websockets import WebSocket, WebSocketDisconnect

VOICES_DIR = os.environ.get("VOICES_DIR", "/app/voices")
DEFAULT_VOICE = os.environ.get("DEFAULT_VOICE", "ENG_US_F_KimW")
MODEL_DIR = os.environ.get("MODEL_DIR", "/app/model")
MAX_REQUEST_AGE_S = float(os.environ.get("MAX_REQUEST_AGE_S", "51"))
SAMPLE_RATE = 24000
WARMUP_TEXT = os.environ.get("WARMUP_TEXT", "Warming up.")

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")

_logger = logging.getLogger("chatterbox_proxy")
_model: ChatterboxTurboTTS | None = None
_inference_lock: asyncio.Lock | None = None
_voice_cache: dict[str, Conditionals] = {}

#: See the identical block in ../kokoro/serve.py. `CONTAINER_START_EPOCH` is
#: exported by entrypoint.sh here, so `elapsed_s` covers the S3 model sync too.
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


def _patch_float64():
    """Fix chatterbox dtype bug: librosa returns float64, model expects float32."""
    try:
        from chatterbox.models.s3tokenizer import S3Tokenizer

        _orig = S3Tokenizer.log_mel_spectrogram

        def _patched(self, audio, padding=0):
            if not torch.is_tensor(audio):
                audio = torch.from_numpy(audio)
            audio = audio.to(dtype=torch.float32)
            return _orig(self, audio, padding)

        S3Tokenizer.log_mel_spectrogram = _patched
    except (ImportError, AttributeError):
        pass

    try:
        from chatterbox.models.voice_encoder import VoiceEncoder

        _orig_inf = VoiceEncoder.inference

        def _patched_inf(self, mels, *args, **kwargs):
            mels = mels.to(dtype=torch.float32)
            return _orig_inf(self, mels, *args, **kwargs)

        VoiceEncoder.inference = _patched_inf
    except (ImportError, AttributeError):
        pass


def _load_model() -> ChatterboxTurboTTS:
    global _model
    if _model is None:
        _patch_float64()

        model_dir = Path(MODEL_DIR)
        if (model_dir / "t3_turbo_v1.safetensors").exists():
            _model = ChatterboxTurboTTS.from_local(str(model_dir), device="cuda")
        else:
            _model = ChatterboxTurboTTS.from_pretrained(device="cuda")

        _logger.info("ChatterboxTurboTTS loaded")
    return _model


def _resolve_voice(voice_id: str) -> str:
    for ext in (".flac", ".wav", ".mp3", ".ogg"):
        path = os.path.join(VOICES_DIR, f"{voice_id}{ext}")
        if os.path.exists(path):
            return path
    available = [f.rsplit(".", 1)[0] for f in os.listdir(VOICES_DIR) if not f.startswith(".")]
    raise FileNotFoundError(f"Voice '{voice_id}' not found in {VOICES_DIR}. Available: {available}")


def _precompute_voices(model: ChatterboxTurboTTS) -> None:
    """Cache conditionals for all voice files at startup."""
    voices_path = Path(VOICES_DIR)
    for audio_file in voices_path.iterdir():
        if audio_file.suffix in (".flac", ".wav", ".mp3", ".ogg"):
            voice_id = audio_file.stem
            try:
                model.prepare_conditionals(str(audio_file), exaggeration=0.5)
                _voice_cache[voice_id] = model.conds
                _logger.info("Cached voice: %s", voice_id)
            except Exception as e:
                _logger.warning("Failed to cache voice %s: %s", voice_id, e)


def _get_conditionals(voice_id: str) -> Conditionals:
    """Get cached conditionals or compute on demand."""
    if voice_id in _voice_cache:
        return _voice_cache[voice_id]

    audio_path = _resolve_voice(voice_id)
    model = _load_model()
    model.prepare_conditionals(audio_path, exaggeration=0.5)
    _voice_cache[voice_id] = model.conds
    return model.conds


def _synthesize(text: str, voice_id: str) -> torch.Tensor:
    model = _load_model()
    model.conds = _get_conditionals(voice_id)
    return model.generate(text)


def _tensor_to_wav(wav_tensor: torch.Tensor, sample_rate: int) -> bytes:
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


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences for per-sentence streaming."""
    sentences = _SENTENCE_RE.split(text.strip())
    return [s for s in sentences if s.strip()]


async def _stream_sentences_generator(text: str, voice_id: str) -> AsyncGenerator[bytes, None]:
    """Yield WAV header + PCM chunks per sentence."""
    header_sent = False
    wav_header = _wav_header_placeholder()
    sentences = _split_sentences(text)
    loop = asyncio.get_event_loop()

    for sentence in sentences:
        async with _inference_lock:
            wav = await loop.run_in_executor(None, _synthesize, sentence, voice_id)

        audio_np = wav.squeeze().cpu().numpy()
        pcm = (audio_np * 32767).astype(np.int16).tobytes()

        if not header_sent:
            yield wav_header + pcm
            header_sent = True
        else:
            yield pcm


async def _invocations_sync(text: str, voice_id: str) -> Response:
    """Original synchronous path: full WAV in one response."""
    t0 = time.perf_counter()

    async with _inference_lock:
        wav = await asyncio.get_event_loop().run_in_executor(None, _synthesize, text, voice_id)

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
            "X-Characters-Processed": str(len(text)),
        },
    )


async def invocations(request: Request) -> Response:
    """TTS: accept text, return WAV audio (streaming by default)."""
    body = json.loads(await request.body())
    text = body.get("text", "")
    voice_id = body.get("voice", DEFAULT_VOICE)
    use_stream = body.get("stream", True)

    request_ts = body.get("request_timestamp")
    if request_ts is not None and time.time() - request_ts > MAX_REQUEST_AGE_S:
        return JSONResponse(status_code=408, content={"error": "request_stale"})

    if not text:
        return JSONResponse(status_code=400, content={"error": "text is required"})

    try:
        _resolve_voice(voice_id)
    except FileNotFoundError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

    if not use_stream:
        return await _invocations_sync(text, voice_id)

    return StreamingResponse(
        _stream_sentences_generator(text, voice_id),
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
                _resolve_voice(voice_id)
            except FileNotFoundError as e:
                await websocket.send_text(
                    json.dumps({"type": "error", "request_id": request_id, "message": str(e)})
                )
                continue

            await websocket.send_text(
                json.dumps({"type": "synthesis_start", "request_id": request_id})
            )

            t0 = time.monotonic()
            sentences = _split_sentences(text)
            loop = asyncio.get_event_loop()
            cumulative_duration = 0.0

            for sentence in sentences:
                async with _inference_lock:
                    wav = await loop.run_in_executor(None, _synthesize, sentence, voice_id)

                audio_np = wav.squeeze().cpu().numpy()
                pcm = (audio_np * 32767).astype(np.int16).tobytes()
                await websocket.send_bytes(pcm)
                cumulative_duration += len(audio_np) / SAMPLE_RATE

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


def _warmup() -> int:
    """Run one discarded generation so the first real request does not pay for JIT.

    Voice conditionals are already precomputed at startup, but CUDA kernel autotune
    for the T3 backbone happens on first `generate`. See ../kokoro/serve.py:_warmup
    for why a failure here is logged rather than fatal.

    Returns:
        Samples generated, for the log line.
    """
    wav = _synthesize(WARMUP_TEXT, DEFAULT_VOICE)
    return int(wav.numel())


@asynccontextmanager
async def lifespan(app: Starlette) -> AsyncGenerator[None, None]:
    global _inference_lock
    _inference_lock = asyncio.Lock()

    _stage("framework_init")
    _logger.info("Loading model at startup...")
    model = _load_model()
    _precompute_voices(model)
    _logger.info("Model loaded, %d voices cached", len(_voice_cache))
    _stage("weights_ready")

    try:
        count = _warmup()
        _logger.info("Warm-up complete, %d samples discarded", count)
    except Exception:
        _logger.exception("Warm-up inference failed; serving anyway")
    _stage("warmup_done")

    _stage("ready")
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
