"""Event dicts the agent's entrypoint yields.

``BedrockAgentCoreApp`` serializes each yielded object as one
``data: {json}\\n\\n`` line (confirmed by reading
``bedrock_agentcore.runtime.app``'s source directly, not the ``event:``/
``data:`` pair the deleted SageMaker-layer SSE relay used) -- these are
plain dicts, not raw SSE bytes. The ``type`` field is the discriminator a
consumer switches on.
"""

from __future__ import annotations

import base64


def agent_start(request_id: str, mode: str) -> dict:
    return {"type": "agent_start", "request_id": request_id, "mode": mode}


def text_delta(request_id: str, seq: int, text: str) -> dict:
    return {"type": "text_delta", "request_id": request_id, "seq": seq, "text": text}


def audio_stream_start(request_id: str, voice: str, sample_rate: int) -> dict:
    return {
        "type": "audio_stream_start",
        "request_id": request_id,
        "format": "pcm16",
        "voice": voice,
        "sample_rate": sample_rate,
    }


def audio_chunk(request_id: str, seq: int, pcm_bytes: bytes) -> dict:
    return {
        "type": "audio_chunk",
        "request_id": request_id,
        "seq": seq,
        "data": base64.b64encode(pcm_bytes).decode("ascii"),
    }


def audio_stream_end(request_id: str, total_chunks: int, duration_s: float) -> dict:
    return {
        "type": "audio_stream_end",
        "request_id": request_id,
        "total_chunks": total_chunks,
        "duration_s": round(duration_s, 3),
    }


def agent_end(request_id: str) -> dict:
    return {"type": "agent_end", "request_id": request_id}


def error(request_id: str, message: str) -> dict:
    return {"type": "error", "request_id": request_id, "message": message}
