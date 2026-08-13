"""Strands agent construction and text-delta extraction from its event stream."""

from __future__ import annotations

import os
from collections.abc import AsyncGenerator

from strands import Agent
from strands.models import BedrockModel


def build_agent() -> Agent:
    model = BedrockModel(
        model_id=os.environ["AGENT_BEDROCK_MODEL_ID"],
        region_name=os.environ.get("AWS_REGION", "us-east-1"),
    )
    return Agent(model=model)


def extract_text_delta(event: dict) -> str | None:
    """Pick out a plain text delta from one of ``agent.stream_async()``'s events.

    ``TextStreamEvent`` (``strands.types._events``) is the only event shape
    with a bare ``data`` key and no ``reasoning``/``citation`` marker --
    confirmed by reading strands' own ``_events.py`` source, not guessed
    from the docs.
    """
    data = event.get("data")
    if not isinstance(data, str):
        return None
    if "reasoning" in event or "citation" in event:
        return None
    return data


async def stream_agent_text(agent: Agent, prompt: str) -> AsyncGenerator[str, None]:
    async for event in agent.stream_async(prompt):
        delta = extract_text_delta(event)
        if delta:
            yield delta
