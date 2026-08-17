# Copyright (c) 2026 Amazon.com
# This file is licensed under the MIT License.
# See the LICENSE file in the project root for full license information.

"""Tests for AgentRuntimeConfig.

``bedrock_model_id`` has no default: a docs search for the latest Claude
Opus ID returned inconsistent version strings this session, so the model ID
is a required, deploy-time-confirmed input, not a value baked into this
file. What's worth testing here is exactly that requiredness, plus the
frozen/immutable contract the rest of the stacks rely on.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from agent_infra.config import AgentRuntimeConfig


class TestAgentRuntimeConfig:
    def test_bedrock_model_id_is_required(self) -> None:
        with pytest.raises(ValidationError):
            AgentRuntimeConfig()  # type: ignore[call-arg]

    def test_bedrock_model_id_cannot_be_empty(self) -> None:
        with pytest.raises(ValidationError):
            AgentRuntimeConfig(bedrock_model_id="")

    def test_defaults(self) -> None:
        config = AgentRuntimeConfig(bedrock_model_id="anthropic.claude-opus-5")
        assert config.tts_endpoint_name == "speech-kokoro-82m"
        assert config.tts_voice == "af_heart"
        assert config.agent_runtime_name == "agent-infra-demo"

    def test_frozen(self) -> None:
        config = AgentRuntimeConfig(bedrock_model_id="anthropic.claude-opus-5")
        with pytest.raises(ValidationError):
            config.bedrock_model_id = "something-else"  # type: ignore[misc]
