"""Model endpoint configuration for SageMaker deployment."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class ContainerType(StrEnum):
    VLLM = "vllm"
    PYTORCH_CUSTOM = "pytorch"


class StreamingMode(StrEnum):
    BIDIRECTIONAL = "bidirectional"
    RESPONSE_STREAM = "response"
    NONE = "none"


class ModelEndpointConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    model_name: str
    hf_model_id: str
    instance_type: str
    container_type: ContainerType
    streaming_mode: StreamingMode = StreamingMode.BIDIRECTIONAL

    min_instances: int = 1
    max_instances: int = 4

    scaling_metric_namespace: str = "Speech/vLLM"
    scaling_metric_name: str = "vllm:num_requests_running"
    scaling_target_value: int = 8

    container_env: dict[str, str] = Field(default_factory=dict)
    codec_model_ids: list[str] = Field(default_factory=list)

    @property
    def endpoint_name(self) -> str:
        return f"speech-{self.model_name}"

    @property
    def stack_id(self) -> str:
        return f"Speech-{self.model_name}"

    @property
    def all_model_ids(self) -> list[str]:
        """All HF model IDs to cache (main model + codecs)."""
        return [self.hf_model_id] + self.codec_model_ids


def get_model_config(name: str) -> ModelEndpointConfig:
    """Look up a model config by name from all registries."""
    all_configs = {**STT_MODEL_CONFIGS, **TTS_MODEL_CONFIGS}
    if name not in all_configs:
        available = ", ".join(sorted(all_configs.keys()))
        raise KeyError(f"Unknown model '{name}'. Available: {available}")
    return all_configs[name]


STT_MODEL_CONFIGS: dict[str, ModelEndpointConfig] = {
    "whisper-large-v3": ModelEndpointConfig(
        model_name="whisper-large-v3",
        hf_model_id="openai/whisper-large-v3",
        instance_type="ml.g5.2xlarge",
        container_type=ContainerType.PYTORCH_CUSTOM,
        streaming_mode=StreamingMode.NONE,
        min_instances=0,
        max_instances=2,
        scaling_target_value=4,
    ),
    "qwen3-asr": ModelEndpointConfig(
        model_name="qwen3-asr",
        hf_model_id="Qwen/Qwen3-ASR-1.7B",
        instance_type="ml.g5.xlarge",
        container_type=ContainerType.PYTORCH_CUSTOM,
        streaming_mode=StreamingMode.NONE,
        min_instances=0,
        max_instances=2,
        scaling_target_value=4,
    ),
}

TTS_MODEL_CONFIGS: dict[str, ModelEndpointConfig] = {
    "kokoro-82m": ModelEndpointConfig(
        model_name="kokoro-82m",
        hf_model_id="hexgrad/Kokoro-82M",
        instance_type="ml.g4dn.xlarge",
        container_type=ContainerType.PYTORCH_CUSTOM,
        streaming_mode=StreamingMode.RESPONSE_STREAM,
        min_instances=0,
        max_instances=2,
        scaling_target_value=4,
    ),
    "maya-veena": ModelEndpointConfig(
        model_name="maya-veena",
        hf_model_id="maya-research/veena-tts",
        instance_type="ml.g5.xlarge",
        container_type=ContainerType.VLLM,
        streaming_mode=StreamingMode.BIDIRECTIONAL,
        codec_model_ids=["hubertsiuzdak/snac_24khz"],
        container_env={
            "SM_VLLM_PORT": "8000",
            "SM_VLLM_MAX_MODEL_LEN": "2048",
            "SM_VLLM_GPU_MEMORY_UTILIZATION": "0.85",
            "MAX_QUEUE_DEPTH": "24",
        },
    ),
    "chatterbox-turbo": ModelEndpointConfig(
        model_name="chatterbox-turbo",
        hf_model_id="ResembleAI/chatterbox-turbo",
        instance_type="ml.g5.xlarge",
        container_type=ContainerType.PYTORCH_CUSTOM,
        streaming_mode=StreamingMode.BIDIRECTIONAL,
        min_instances=0,
        max_instances=2,
        scaling_target_value=4,
    ),
    "orpheus-3b": ModelEndpointConfig(
        model_name="orpheus-3b",
        hf_model_id="canopylabs/orpheus-3b-0.1-ft",
        instance_type="ml.g5.xlarge",
        container_type=ContainerType.VLLM,
        streaming_mode=StreamingMode.BIDIRECTIONAL,
        min_instances=1,
        max_instances=4,
        scaling_target_value=8,
        codec_model_ids=["hubertsiuzdak/snac_24khz"],
        container_env={
            "SM_VLLM_PORT": "8000",
            "SM_VLLM_MAX_MODEL_LEN": "2048",
            "SM_VLLM_MAX_NUM_SEQS": "16",
            "SM_VLLM_GPU_MEMORY_UTILIZATION": "0.85",
            "SM_VLLM_ENFORCE_EAGER": "true",
            "MAX_QUEUE_DEPTH": "24",
        },
    ),
}
