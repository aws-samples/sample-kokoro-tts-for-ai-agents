# Copyright (c) 2026 Amazon.com
# This file is licensed under the MIT License.
# See the LICENSE file in the project root for full license information.

"""bidi vs. batch TTS delivery mode implementations.

``run_bidi`` feeds the agent's own streaming text deltas straight into
``TTSClient.synthesize_bidi_stream`` as they're produced -- sentence 1 starts
synthesizing while the agent is still generating sentence 3. ``run_batch``
collects the agent's full response first, then makes one
``TTSClient.synthesize_bidi`` call with the complete text -- the "wait for
everything" baseline the demo measures ``run_bidi`` against.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator

import events
from bidi_bridge import SyncTextQueue, bidi_stream_to_async
from strands_agent import stream_agent_text

from tts_client.client import TTSClient
from tts_client.types import SynthesisRequest

#: Purely for progressive delivery of batch mode's one WAV blob over the
#: event stream -- there is no concurrency win here by design.
_BATCH_CHUNK_BYTES = 32_768

#: BidiChunkStream's raw-PCM sample rate (tts_client._bidi_transport.BIDI_SAMPLE_RATE);
#: duplicated here rather than imported to keep this container's import surface
#: to public tts_client modules only.
_BIDI_SAMPLE_RATE = 24000


async def run_bidi(
    agent,
    prompt: str,
    tts_client: TTSClient,
    endpoint: str,
    voice: str,
    speed: float,
    request_id: str,
) -> AsyncGenerator[dict, None]:
    """Interleave the agent's ``text_delta`` events with TTS's ``audio_chunk``
    events as both actually occur, by merging two concurrent producers
    through one tagged ``asyncio.Queue``.

    Each producer puts ``("event", dict)``, ``("done", None)``, or
    ``("error", exc)`` -- never lets an exception propagate as an unhandled
    task failure, which would otherwise hang the consumer loop below waiting
    on a ``"done"`` that never arrives.
    """
    text_queue = SyncTextQueue()
    out_queue: asyncio.Queue = asyncio.Queue()

    async def _pump_text() -> None:
        seq = 0
        try:
            async for delta in stream_agent_text(agent, prompt):
                text_queue.put(delta)
                await out_queue.put(("event", events.text_delta(request_id, seq, delta)))
                seq += 1
        except Exception as exc:  # noqa: BLE001 - relayed via the queue, not swallowed
            await out_queue.put(("error", exc))
            return
        finally:
            text_queue.close()
        await out_queue.put(("done", None))

    async def _pump_audio() -> None:
        seq = 0
        total_duration = 0.0
        await out_queue.put(
            ("event", events.audio_stream_start(request_id, voice, _BIDI_SAMPLE_RATE))
        )

        def _open_stream():
            return tts_client.synthesize_bidi_stream(endpoint, voice, text_queue, speed=speed)

        try:
            async for chunk in bidi_stream_to_async(_open_stream):
                await out_queue.put(
                    ("event", events.audio_chunk(request_id, seq, chunk.audio_bytes))
                )
                total_duration += chunk.duration_s
                seq += 1
        except Exception as exc:  # noqa: BLE001 - relayed via the queue, not swallowed
            await out_queue.put(("error", exc))
            return
        await out_queue.put(("event", events.audio_stream_end(request_id, seq, total_duration)))
        await out_queue.put(("done", None))

    tasks = [asyncio.ensure_future(_pump_text()), asyncio.ensure_future(_pump_audio())]
    try:
        remaining = len(tasks)
        while remaining:
            kind, value = await out_queue.get()
            if kind == "done":
                remaining -= 1
                continue
            if kind == "error":
                raise value
            yield value
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()


async def run_batch(
    agent,
    prompt: str,
    tts_client: TTSClient,
    endpoint: str,
    voice: str,
    speed: float,
    request_id: str,
) -> AsyncGenerator[dict, None]:
    seq = 0
    parts: list[str] = []
    async for delta in stream_agent_text(agent, prompt):
        parts.append(delta)
        yield events.text_delta(request_id, seq, delta)
        seq += 1
    full_text = "".join(parts)

    loop = asyncio.get_event_loop()
    request = SynthesisRequest(text=full_text, voice=voice, speed=speed)
    result = await loop.run_in_executor(None, tts_client.synthesize_bidi, endpoint, request)

    yield events.audio_stream_start(request_id, voice, result.sample_rate)
    # result.audio_bytes is one WAV-wrapped blob (synthesize_bidi's own
    # contract), unlike run_bidi's raw-PCM-per-sentence chunks -- fine here
    # since this demo only measures timing, not audio playback.
    audio_seq = 0
    audio = result.audio_bytes
    for offset in range(0, len(audio), _BATCH_CHUNK_BYTES):
        yield events.audio_chunk(request_id, audio_seq, audio[offset : offset + _BATCH_CHUNK_BYTES])
        audio_seq += 1
    yield events.audio_stream_end(request_id, audio_seq, result.duration_s)
