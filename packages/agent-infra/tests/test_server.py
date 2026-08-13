"""Tests for the AgentCore entrypoint in container/server.py.

Exercises ``agent_invocation()`` directly as an async generator, bypassing
``BedrockAgentCoreApp``'s own HTTP layer -- that's the SDK's concern, already
confirmed by reading its source (see stacks/runtime.py's module docstring),
not ours to re-test.

Self-skips like speech-infra's own container tests (``test_kokoro_serve.py``):
container-only deps (``bedrock_agentcore``, ``strands``) are not installed in
this lean workspace dev environment. Runs for real inside the container
image, or via ``uv run --with bedrock-agentcore pytest packages/agent-infra/tests``.
"""

from __future__ import annotations

import asyncio
import importlib.util
import sys

import pytest

_REQUIRED_DEPS = ("bedrock_agentcore", "strands")

_HAS_DEPS = all(importlib.util.find_spec(mod) is not None for mod in _REQUIRED_DEPS)

pytestmark = pytest.mark.skipif(
    not _HAS_DEPS,
    reason=f"container deps ({', '.join(_REQUIRED_DEPS)}) not installed in this env",
)


class _FakeAgent:
    def __init__(self, deltas: list[str]) -> None:
        self._deltas = deltas

    async def stream_async(self, prompt: str):
        for delta in self._deltas:
            yield {"data": delta}


class _FakeChunk:
    def __init__(self, text: str) -> None:
        self.audio_bytes = text.encode()
        self.duration_s = 0.01


class _FakeBidiChunkStream:
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
    def synthesize_bidi_stream(self, endpoint, voice, text_source, *, speed=1.0):
        return _FakeBidiChunkStream(text_source)

    def synthesize_bidi(self, endpoint, request):
        from tts_client.types import AudioFormat, SynthesisResult

        return SynthesisResult(
            audio_bytes=request.text.encode(),
            audio_format=AudioFormat.WAV,
            sample_rate=24000,
            duration_s=0.1,
            latency_ms=5.0,
            chars=len(request.text),
        )


class _BrokenTTSClient:
    def synthesize_bidi_stream(self, *args, **kwargs):
        raise RuntimeError("endpoint unavailable")


@pytest.fixture
def server_module(monkeypatch):
    """Import container/server.py with a fake agent and TTS client.

    Environment variables server.py reads at import time are set first; the
    module is re-imported fresh per test so each test's monkeypatches (on
    ``strands_agent.build_agent``, applied before the import) take effect.
    """
    monkeypatch.setenv("AGENT_TTS_ENDPOINT", "speech-kokoro-82m")
    monkeypatch.setenv("AGENT_BEDROCK_MODEL_ID", "anthropic.claude-opus-5")
    monkeypatch.setenv("AWS_REGION", "us-east-1")

    sys.modules.pop("server", None)

    import strands_agent

    monkeypatch.setattr(strands_agent, "build_agent", lambda: _FakeAgent(["Hi there."]))

    import server as loaded_server

    loaded_server.tts_client = _FakeTTSClient()
    yield loaded_server
    sys.modules.pop("server", None)


async def _drain(agen) -> list[dict]:
    return [event async for event in agen]


class TestAgentInvocation:
    def test_bidi_mode_event_sequence(self, server_module) -> None:
        events_out = asyncio.run(
            _drain(server_module.agent_invocation({"prompt": "hello", "mode": "bidi"}))
        )
        types = [e["type"] for e in events_out]
        assert types[0] == "agent_start"
        assert types[-1] == "agent_end"
        assert "audio_stream_start" in types
        assert "audio_stream_end" in types
        assert "error" not in types

    def test_batch_mode_event_sequence(self, server_module) -> None:
        events_out = asyncio.run(
            _drain(server_module.agent_invocation({"prompt": "hello", "mode": "batch"}))
        )
        types = [e["type"] for e in events_out]
        assert types[0] == "agent_start"
        assert types[-1] == "agent_end"
        assert "error" not in types

    def test_defaults_to_bidi_mode(self, server_module) -> None:
        events_out = asyncio.run(_drain(server_module.agent_invocation({"prompt": "hello"})))
        assert events_out[0]["mode"] == "bidi"

    def test_invalid_mode_yields_in_band_error_before_agent_start(self, server_module) -> None:
        events_out = asyncio.run(
            _drain(server_module.agent_invocation({"prompt": "hello", "mode": "not-a-mode"}))
        )
        assert [e["type"] for e in events_out] == ["error"]

    def test_missing_prompt_yields_in_band_error(self, server_module) -> None:
        events_out = asyncio.run(_drain(server_module.agent_invocation({})))
        assert [e["type"] for e in events_out] == ["error"]

    def test_tts_failure_mid_stream_yields_in_band_error_not_a_raise(self, server_module) -> None:
        server_module.tts_client = _BrokenTTSClient()
        events_out = asyncio.run(
            _drain(server_module.agent_invocation({"prompt": "hello", "mode": "bidi"}))
        )
        assert events_out[-1]["type"] == "error"
