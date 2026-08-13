"""Tests for speech-infra configuration."""

import pytest
from pydantic import ValidationError

from speech_infra.config import (
    TTS_MODEL_CONFIGS,
    ContainerType,
    ModelEndpointConfig,
    StreamingMode,
    get_model_config,
)


class TestModelEndpointConfig:
    def test_endpoint_name(self) -> None:
        cfg = ModelEndpointConfig(
            model_name="test-model",
            hf_model_id="org/test",
            instance_type="ml.g5.xlarge",
            container_type=ContainerType.VLLM,
        )
        assert cfg.endpoint_name == "speech-test-model"

    def test_stack_id(self) -> None:
        cfg = ModelEndpointConfig(
            model_name="test-model",
            hf_model_id="org/test",
            instance_type="ml.g5.xlarge",
            container_type=ContainerType.VLLM,
        )
        assert cfg.stack_id == "Speech-test-model"

    def test_all_model_ids_without_codecs(self) -> None:
        cfg = ModelEndpointConfig(
            model_name="kokoro",
            hf_model_id="hexgrad/Kokoro-82M",
            instance_type="ml.g4dn.xlarge",
            container_type=ContainerType.PYTORCH_CUSTOM,
        )
        assert cfg.all_model_ids == ["hexgrad/Kokoro-82M"]

    def test_all_model_ids_with_codecs(self) -> None:
        cfg = ModelEndpointConfig(
            model_name="test-model",
            hf_model_id="org/test",
            instance_type="ml.g5.xlarge",
            container_type=ContainerType.VLLM,
            codec_model_ids=["hubertsiuzdak/snac_24khz"],
        )
        assert cfg.all_model_ids == [
            "org/test",
            "hubertsiuzdak/snac_24khz",
        ]

    def test_scaling_enabled_when_can_scale(self) -> None:
        cfg = ModelEndpointConfig(
            model_name="test",
            hf_model_id="org/test",
            instance_type="ml.g5.xlarge",
            container_type=ContainerType.VLLM,
            min_instances=0,
            max_instances=4,
        )
        assert cfg.scaling_enabled is True

    def test_scaling_disabled_when_max_equals_effective_min(self) -> None:
        cfg = ModelEndpointConfig(
            model_name="test",
            hf_model_id="org/test",
            instance_type="ml.g5.xlarge",
            container_type=ContainerType.VLLM,
            min_instances=0,
            max_instances=1,
        )
        assert cfg.scaling_enabled is False

    def test_scaling_disabled_when_min_equals_max(self) -> None:
        cfg = ModelEndpointConfig(
            model_name="test",
            hf_model_id="org/test",
            instance_type="ml.g5.xlarge",
            container_type=ContainerType.VLLM,
            min_instances=1,
            max_instances=1,
        )
        assert cfg.scaling_enabled is False

    def test_frozen_config(self) -> None:
        cfg = TTS_MODEL_CONFIGS["kokoro-82m"]
        with pytest.raises(ValidationError):
            cfg.model_name = "changed"


class TestInstanceTypeValidation:
    """The `ml.` prefix, checked at config-load time.

    `instance_type` is changed routinely now — the benchmark harness re-measures per
    configuration and a candidate type arrives via `app.py`'s context override — and a
    typo is expensive to diagnose: CloudFormation accepts it and the endpoint then sits
    in `Updating` with no `FailureReason`.
    """

    @pytest.mark.parametrize("bad", ["g6.xlarge", "ML.g6.xlarge", "ml-g6.xlarge", ""])
    def test_a_type_without_the_prefix_is_rejected(self, bad: str) -> None:
        with pytest.raises(ValidationError, match="must start with 'ml.'"):
            ModelEndpointConfig(
                model_name="test",
                hf_model_id="org/test",
                instance_type=bad,
                container_type=ContainerType.VLLM,
            )

    @pytest.mark.parametrize(
        "good", ["ml.g5.xlarge", "ml.g6.xlarge", "ml.g6.12xlarge", "ml.c5.2xlarge"]
    )
    def test_prefixed_types_are_accepted(self, good: str) -> None:
        # Not checked against a list: AWS adds instance types faster than this file
        # changes, and rejecting an unknown-but-real type would block the next sweep.
        cfg = ModelEndpointConfig(
            model_name="test",
            hf_model_id="org/test",
            instance_type=good,
            container_type=ContainerType.VLLM,
        )
        assert cfg.instance_type == good

    def test_every_configured_model_passes_it(self) -> None:
        for name, cfg in TTS_MODEL_CONFIGS.items():
            assert cfg.instance_type.startswith("ml."), name


class TestConfigRegistry:
    def test_kokoro_config_exists(self) -> None:
        cfg = TTS_MODEL_CONFIGS["kokoro-82m"]
        assert cfg.container_type == ContainerType.PYTORCH_CUSTOM
        assert cfg.streaming_mode == StreamingMode.RESPONSE_STREAM
        assert cfg.codec_model_ids == []

    def test_all_tts_configs_have_required_fields(self) -> None:
        for name, cfg in TTS_MODEL_CONFIGS.items():
            assert cfg.model_name, f"{name} missing model_name"
            assert cfg.hf_model_id, f"{name} missing hf_model_id"
            assert cfg.instance_type, f"{name} missing instance_type"
            assert cfg.container_type in ContainerType
            assert cfg.streaming_mode in StreamingMode

    def test_get_model_config_found(self) -> None:
        cfg = get_model_config("kokoro-82m")
        assert cfg.model_name == "kokoro-82m"

    def test_get_model_config_not_found(self) -> None:
        with pytest.raises(KeyError, match="Unknown model"):
            get_model_config("nonexistent-model")
