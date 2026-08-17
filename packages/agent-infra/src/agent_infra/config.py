# Copyright (c) 2026 Amazon.com
# This file is licensed under the MIT License.
# See the LICENSE file in the project root for full license information.

"""Agent runtime configuration for AgentCore deployment."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class AgentRuntimeConfig(BaseModel):
    """Everything the CDK stacks need to deploy the demo agent.

    ``bedrock_model_id`` has no default on purpose: a docs search for the
    latest Claude Opus ID this session returned inconsistent, unverifiable
    version strings, so the real ID is a deploy-time decision, confirmed via
    ``aws bedrock list-foundation-models --by-provider anthropic --region
    <region>`` (and whether an inference-profile ID is required for on-demand
    throughput), not a value baked into this file.
    """

    model_config = ConfigDict(frozen=True)

    bedrock_model_id: str = Field(
        ...,
        min_length=1,
        description="Bedrock model ID for the Strands agent's BedrockModel. "
        "Confirm via `aws bedrock list-foundation-models` before deploy — "
        "do not hardcode a guessed value here.",
    )

    #: The already-deployed speech-infra TTS endpoint this agent calls.
    tts_endpoint_name: str = "speech-kokoro-82m"
    tts_voice: str = "af_heart"

    agent_runtime_name: str = "agent-infra-demo"

    @property
    def stack_id(self) -> str:
        return "AgentRuntime"
