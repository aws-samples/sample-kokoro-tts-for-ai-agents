# Copyright (c) 2026 Amazon.com
# This file is licensed under the MIT License.
# See the LICENSE file in the project root for full license information.

"""Tests for modes.py's bidi/batch TTS delivery implementations.

No skip needed: modes.py only imports tts_client (a workspace package) and
strands_agent (which itself only needs ``strands``, not ``bedrock_agentcore``)
-- none of the container-only deps server.py needs.
"""

from __future__ import annotations

import asyncio

import modes
import pytest


class _FakeChunk:
    def __init__(self, text: str) -> None:
        self.audio_bytes = text.encode()
        self.duration_s = 0.01


class _FakeBidiChunkStream:
    """Stands in for BidiChunkStream: consumes text_source synchronously,
    exactly as the real bridge's worker thread would."""

    def __init__(self, text_source) -> None:
        self._iter = iter(text_source)

    def __enter__(self) -> _FakeBidiChunkStream:
        return self

    def __exit__(self, *exc_info: object) -> None:
        pass

    def __iter__(self) -> _FakeBidiChunkStream:
        return self

    def __next__(self) -> _FakeChunk:
        return _FakeChunk(next(self._iter))


class _FakeTTSClient:
    def __init__(self) -> None:
        self.bidi_calls: list[tuple] = []
        self.batch_calls: list[tuple] = []

    def synthesize_bidi_stream(self, endpoint, voice, text_source, *, speed=1.0):
        self.bidi_calls.append((endpoint, voice, speed))
        return _FakeBidiChunkStream(text_source)

    def synthesize_bidi(self, endpoint, request):
        self.batch_calls.append((endpoint, request))
        from tts_client.types import AudioFormat, SynthesisResult

        return SynthesisResult(
            audio_bytes=request.text.encode(),
            audio_format=AudioFormat.WAV,
            sample_rate=24000,
            duration_s=0.1,
            latency_ms=5.0,
            chars=len(request.text),
        )


class _FakeAgent:
    def __init__(self, deltas: list[str]) -> None:
        self._deltas = deltas

    async def stream_async(self, prompt: str):
        for delta in self._deltas:
            yield {"data": delta}


async def _drain(agen) -> list[dict]:
    return [event async for event in agen]


class TestRunBidi:
    def test_interleaves_text_and_audio_events_in_a_sane_sequence(self) -> None:
        agent = _FakeAgent(["Hello. ", "World."])
        tts_client = _FakeTTSClient()
        events_out = asyncio.run(
            _drain(
                modes.run_bidi(agent, "prompt", tts_client, "endpoint", "af_heart", 1.0, "req-1")
            )
        )
        types = [e["type"] for e in events_out]
        assert types.count("text_delta") == 2
        assert "audio_stream_start" in types
        assert "audio_stream_end" in types
        if "audio_chunk" in types:
            assert types.index("audio_stream_start") < types.index("audio_chunk")

    def test_calls_synthesize_bidi_stream_with_the_right_endpoint_and_voice(self) -> None:
        agent = _FakeAgent(["One sentence."])
        tts_client = _FakeTTSClient()
        asyncio.run(
            _drain(
                modes.run_bidi(agent, "prompt", tts_client, "my-endpoint", "af_bella", 1.2, "req-2")
            )
        )
        assert tts_client.bidi_calls == [("my-endpoint", "af_bella", 1.2)]

    def test_tts_failure_propagates_as_an_exception_not_a_hang(self) -> None:
        class _BrokenTTSClient:
            def synthesize_bidi_stream(self, *args, **kwargs):
                raise RuntimeError("endpoint unavailable")

        agent = _FakeAgent(["Hello."])
        with pytest.raises(RuntimeError, match="endpoint unavailable"):
            asyncio.run(
                _drain(
                    modes.run_bidi(
                        agent, "prompt", _BrokenTTSClient(), "endpoint", "af_heart", 1.0, "req-3"
                    )
                )
            )


class TestRunBatch:
    def test_collects_full_text_before_synthesizing(self) -> None:
        agent = _FakeAgent(["Part one. ", "Part two."])
        tts_client = _FakeTTSClient()
        events_out = asyncio.run(
            _drain(
                modes.run_batch(agent, "prompt", tts_client, "endpoint", "af_heart", 1.0, "req-4")
            )
        )
        types = [e["type"] for e in events_out]
        assert types.count("text_delta") == 2
        assert types.index("text_delta") < types.index("audio_stream_start")
        assert len(tts_client.batch_calls) == 1
        _, request = tts_client.batch_calls[0]
        assert request.text == "Part one. Part two."
