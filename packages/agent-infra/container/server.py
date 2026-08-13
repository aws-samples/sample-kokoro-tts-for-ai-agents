"""AgentCore Runtime entrypoint.

Streams the Strands agent's text into one of two TTS delivery modes and
relays the resulting audio (and the agent's own text) as a sequence of
structured events. ``BedrockAgentCoreApp`` satisfies AgentCore Runtime's HTTP
contract (``/invocations`` POST, ``/ping`` GET, port 8080) with no code of
ours -- confirmed by reading its source, not assumed.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import AsyncGenerator

import events
from bedrock_agentcore.runtime import BedrockAgentCoreApp
from modes import run_batch, run_bidi
from strands_agent import build_agent

from tts_client.client import TTSClient

app = BedrockAgentCoreApp()
agent = build_agent()
tts_client = TTSClient(region=os.environ.get("AWS_REGION", "us-east-1"))

DEFAULT_TTS_ENDPOINT = os.environ["AGENT_TTS_ENDPOINT"]
DEFAULT_VOICE = os.environ.get("AGENT_TTS_VOICE", "af_heart")

_MODES = {"bidi": run_bidi, "batch": run_batch}


@app.entrypoint
async def agent_invocation(payload: dict) -> AsyncGenerator[dict, None]:
    request_id = str(uuid.uuid4())

    try:
        prompt = payload["prompt"]
        mode = payload.get("mode", "bidi")
        voice = payload.get("voice", DEFAULT_VOICE)
        speed = payload.get("speed", 1.0)
        endpoint = payload.get("endpoint", DEFAULT_TTS_ENDPOINT)
        if mode not in _MODES:
            raise ValueError(f"mode must be one of {sorted(_MODES)}, got {mode!r}")
    except Exception as exc:  # noqa: BLE001 - malformed payload becomes an in-band error event
        yield events.error(request_id, str(exc))
        return

    yield events.agent_start(request_id, mode)
    try:
        run = _MODES[mode]
        async for event in run(agent, prompt, tts_client, endpoint, voice, speed, request_id):
            yield event
    except Exception as exc:  # noqa: BLE001 - relayed in-band, not a raw 500
        yield events.error(request_id, str(exc))
        return
    yield events.agent_end(request_id)


if __name__ == "__main__":
    app.run()
