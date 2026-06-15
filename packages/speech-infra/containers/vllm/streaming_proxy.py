"""Streaming proxy: HTTP + WebSocket gateway between SageMaker and vLLM.

Provides:
- GET /ping: health check passthrough
- POST /invocations: synchronous TTS (backpressure + full audio response)
- WS /invocations-bidirectional-stream: streaming TTS over WebSocket

Monitors vLLM's queue depth for backpressure (503 / WS close 1013).
"""

import json
import logging
import os
import re
import time
from io import BytesIO

import httpx
import numpy as np
import soundfile as sf
import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route, WebSocketRoute
from starlette.websockets import WebSocket, WebSocketDisconnect

from tts_orpheus.prompt import STOP_TOKEN_ID, build_prompt, token_ids_to_snac_codes
from tts_orpheus.snac_decode import SnacDecoder

_CUSTOM_TOKEN_RE = re.compile(r"<custom_token_(\d+)>")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_CLAUSE_RE = re.compile(r"(?<=[,;:])\s+")

VLLM_BACKEND = "http://localhost:8000"
MAX_QUEUE_DEPTH = int(os.environ.get("MAX_QUEUE_DEPTH", "24"))
SNAC_MODEL_PATH = os.environ.get("SNAC_MODEL_PATH", "hubertsiuzdak/snac_24khz")
SAMPLE_RATE = 24000

_logger = logging.getLogger("streaming_proxy")

_NUM_WAITING_RE = re.compile(
    r"^vllm:num_requests_waiting\{.*?\}\s+(\d+(?:\.\d+)?)", re.MULTILINE
)
_NUM_RUNNING_RE = re.compile(
    r"^vllm:num_requests_running\{.*?\}\s+(\d+(?:\.\d+)?)", re.MULTILINE
)

_client = httpx.AsyncClient(base_url=VLLM_BACKEND, timeout=120.0)
_snac_decoder: SnacDecoder | None = None


def _get_snac_decoder() -> SnacDecoder:
    """Lazy-load SNAC decoder on first use (requires GPU init)."""
    global _snac_decoder
    if _snac_decoder is None:
        _snac_decoder = SnacDecoder(model_path=SNAC_MODEL_PATH)
    return _snac_decoder


def _parse_metric(text: str, pattern: re.Pattern[str]) -> float:
    match = pattern.search(text)
    return float(match.group(1)) if match else 0.0


async def _check_backpressure() -> tuple[bool, float]:
    """Returns (should_reject, num_waiting)."""
    try:
        resp = await _client.get("/metrics")
        if resp.status_code != 200:
            return False, 0.0
        text = resp.text
        num_waiting = _parse_metric(text, _NUM_WAITING_RE)
        if num_waiting > MAX_QUEUE_DEPTH:
            return True, num_waiting
        return False, num_waiting
    except httpx.HTTPError:
        return False, 0.0


def _extract_token_ids(text: str) -> list[int]:
    """Extract audio token IDs from vLLM output text.

    vLLM outputs '<custom_token_N>' strings concatenated with no spaces.
    Token ID = N + 128256 (added_tokens base in the Orpheus tokenizer).
    """
    return [int(n) + 128256 for n in _CUSTOM_TOKEN_RE.findall(text)]


def _split_text(text: str, max_chars: int = 300) -> list[str]:
    """Split text into segments under max_chars, preferring natural boundaries."""
    sentences = _SENTENCE_RE.split(text.strip())
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        if current and len(current) + len(sentence) + 1 > max_chars:
            chunks.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip() if current else sentence
    if current:
        chunks.append(current)
    result: list[str] = []
    for chunk in chunks:
        if len(chunk) <= max_chars:
            result.append(chunk)
        else:
            result.extend(_break_long(chunk, max_chars))
    return result


def _break_long(text: str, max_chars: int) -> list[str]:
    """Break a long segment at clause boundaries, falling back to word boundaries."""
    parts = _CLAUSE_RE.split(text)
    if len(parts) == 1:
        parts = text.split(" ")
    chunks: list[str] = []
    current = ""
    for part in parts:
        if current and len(current) + len(part) + 1 > max_chars:
            chunks.append(current)
            current = part
        else:
            current = f"{current} {part}".strip() if current else part
    if current:
        chunks.append(current)
    return chunks


def _make_vllm_payload(prompt: str, stream: bool) -> dict:
    return {
        "model": os.environ.get("SM_VLLM_MODEL", "/tmp/model"),
        "prompt": prompt,
        "max_tokens": 1200,
        "temperature": 0.6,
        "top_p": 0.95,
        "repetition_penalty": 1.1,
        "stop_token_ids": [STOP_TOKEN_ID],
        "skip_special_tokens": False,
        "stream": stream,
    }


async def ping(request: Request) -> Response:
    """Health check passthrough to vLLM."""
    resp = await _client.get("/health")
    return Response(content=resp.content, status_code=resp.status_code)


async def invocations(request: Request) -> Response:
    """Synchronous TTS: accept text, return full WAV audio."""
    should_reject, num_waiting = await _check_backpressure()
    if should_reject:
        return JSONResponse(
            status_code=503,
            content={
                "error": "queue_saturated",
                "queue_depth": int(num_waiting),
                "max_queue_depth": MAX_QUEUE_DEPTH,
            },
            headers={"Retry-After": "5"},
        )

    body = json.loads(await request.body())
    text = body.get("text", "")
    voice = body.get("voice", "tara")

    if not text:
        return JSONResponse(status_code=400, content={"error": "text is required"})

    segments = _split_text(text)
    decoder = _get_snac_decoder()
    all_audio_np: list[np.ndarray] = []

    for segment_text in segments:
        prompt = build_prompt(segment_text, voice)
        vllm_payload = _make_vllm_payload(prompt, stream=False)

        resp = await _client.post("/v1/completions", json=vllm_payload, timeout=60.0)
        if resp.status_code != 200:
            return Response(content=resp.content, status_code=resp.status_code)

        result = resp.json()
        generated_text = result["choices"][0]["text"]

        token_ids = _extract_token_ids(generated_text)
        audio_codes = token_ids_to_snac_codes(token_ids)
        if not audio_codes:
            continue

        audio_bytes = decoder.decode_frames(audio_codes)
        if audio_bytes:
            segment_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            all_audio_np.append(segment_np)

    if not all_audio_np:
        return JSONResponse(
            status_code=500,
            content={"error": "no audio tokens generated"},
        )

    combined = np.concatenate(all_audio_np)
    buffer = BytesIO()
    sf.write(buffer, combined, SAMPLE_RATE, format="WAV")

    return Response(
        content=buffer.getvalue(),
        media_type="audio/wav",
        headers={"X-Audio-Duration": str(len(combined) / SAMPLE_RATE)},
    )


async def bidirectional_stream(websocket: WebSocket) -> None:
    """WebSocket handler for bidirectional streaming TTS."""
    await websocket.accept()

    try:
        while True:
            raw = await websocket.receive_text()
            msg = json.loads(raw)

            if msg.get("type") == "close":
                break

            text = msg.get("text", "")
            voice = msg.get("voice", "tara")
            request_id = msg.get("request_id", "unknown")

            if not text:
                await websocket.send_text(
                    json.dumps({"type": "error", "request_id": request_id, "message": "text required"})
                )
                continue

            should_reject, _ = await _check_backpressure()
            if should_reject:
                await websocket.send_text(
                    json.dumps({"type": "error", "request_id": request_id, "message": "queue_saturated"})
                )
                continue

            segments = _split_text(text)
            t0 = time.monotonic()

            await websocket.send_text(
                json.dumps({"type": "synthesis_start", "request_id": request_id, "segments": len(segments)})
            )

            decoder = _get_snac_decoder()
            cumulative_audio_ms = 0.0
            total_audio_bytes = 0

            for seq, segment_text in enumerate(segments):
                await websocket.send_text(
                    json.dumps({
                        "type": "segment_start",
                        "request_id": request_id,
                        "seq": seq,
                        "offset_ms": round(cumulative_audio_ms),
                        "text": segment_text,
                    })
                )

                prompt = build_prompt(segment_text, voice)
                vllm_payload = _make_vllm_payload(prompt, stream=True)
                token_ids: list[int] = []
                decoded_frames = 0
                segment_audio_bytes = 0

                async with _client.stream(
                    "POST", "/v1/completions", json=vllm_payload, timeout=60.0
                ) as resp:
                    async for line in resp.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        data_str = line[6:]
                        if data_str == "[DONE]":
                            break

                        chunk = json.loads(data_str)
                        token_text = chunk["choices"][0].get("text", "")
                        if not token_text:
                            continue

                        new_ids = _extract_token_ids(token_text)
                        if not new_ids:
                            continue
                        token_ids.extend(new_ids)

                        audio_codes = token_ids_to_snac_codes(token_ids)
                        while len(audio_codes) - decoded_frames * 7 >= 28:
                            start = decoded_frames * 7
                            frame_codes = audio_codes[start : start + 28]
                            audio_chunk = decoder.decode_frames(frame_codes)
                            if audio_chunk:
                                await websocket.send_bytes(audio_chunk)
                                segment_audio_bytes += len(audio_chunk)
                            decoded_frames += 4

                segment_duration_ms = (segment_audio_bytes / (SAMPLE_RATE * 2)) * 1000
                cumulative_audio_ms += segment_duration_ms
                total_audio_bytes += segment_audio_bytes

                await websocket.send_text(
                    json.dumps({
                        "type": "segment_complete",
                        "request_id": request_id,
                        "seq": seq,
                        "duration_ms": round(segment_duration_ms),
                    })
                )

            elapsed = time.monotonic() - t0

            await websocket.send_text(
                json.dumps({
                    "type": "synthesis_complete",
                    "request_id": request_id,
                    "total_duration_s": round(cumulative_audio_ms / 1000, 3),
                    "elapsed_s": round(elapsed, 3),
                    "segments": len(segments),
                })
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
)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="warning")
