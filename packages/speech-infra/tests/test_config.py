"""Tests for speech-infra configuration."""

import pytest

from speech_infra.config import (
    STT_MODEL_CONFIGS,
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
            model_name="orpheus-3b",
            hf_model_id="canopylabs/orpheus-3b-0.1-ft",
            instance_type="ml.g5.xlarge",
            container_type=ContainerType.VLLM,
        )
        assert cfg.stack_id == "Speech-orpheus-3b"

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
            model_name="orpheus-3b",
            hf_model_id="canopylabs/orpheus-3b-0.1-ft",
            instance_type="ml.g5.xlarge",
            container_type=ContainerType.VLLM,
            codec_model_ids=["hubertsiuzdak/snac_24khz"],
        )
        assert cfg.all_model_ids == [
            "canopylabs/orpheus-3b-0.1-ft",
            "hubertsiuzdak/snac_24khz",
        ]

    def test_frozen_config(self) -> None:
        cfg = TTS_MODEL_CONFIGS["orpheus-3b"]
        with pytest.raises(Exception):
            cfg.model_name = "changed"


class TestConfigRegistry:
    def test_orpheus_config_exists(self) -> None:
        cfg = TTS_MODEL_CONFIGS["orpheus-3b"]
        assert cfg.container_type == ContainerType.VLLM
        assert cfg.streaming_mode == StreamingMode.BIDIRECTIONAL
        assert "hubertsiuzdak/snac_24khz" in cfg.codec_model_ids

    def test_all_tts_configs_have_required_fields(self) -> None:
        for name, cfg in TTS_MODEL_CONFIGS.items():
            assert cfg.model_name, f"{name} missing model_name"
            assert cfg.hf_model_id, f"{name} missing hf_model_id"
            assert cfg.instance_type, f"{name} missing instance_type"
            assert cfg.container_type in ContainerType
            assert cfg.streaming_mode in StreamingMode

    def test_all_stt_configs_have_required_fields(self) -> None:
        for name, cfg in STT_MODEL_CONFIGS.items():
            assert cfg.model_name, f"{name} missing model_name"
            assert cfg.hf_model_id, f"{name} missing hf_model_id"
            assert cfg.instance_type, f"{name} missing instance_type"

    def test_get_model_config_found(self) -> None:
        cfg = get_model_config("orpheus-3b")
        assert cfg.model_name == "orpheus-3b"

    def test_get_model_config_not_found(self) -> None:
        with pytest.raises(KeyError, match="Unknown model"):
            get_model_config("nonexistent-model")

    def test_vllm_models_have_codec_ids(self) -> None:
        for name, cfg in TTS_MODEL_CONFIGS.items():
            if cfg.container_type == ContainerType.VLLM:
                assert len(cfg.codec_model_ids) > 0, (
                    f"vLLM model {name} should have codec_model_ids"
                )
