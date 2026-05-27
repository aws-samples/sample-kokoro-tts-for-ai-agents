"""Model endpoint configuration for SageMaker deployment."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class ModelEndpointConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    model_name: str
    hf_model_id: str
    instance_type: str
    min_instances: int = 0
    max_instances: int = 1
    target_invocations: int = 10
    max_concurrency_per_instance: int = 4


STT_MODEL_CONFIGS: dict[str, ModelEndpointConfig] = {
    "whisper-large-v3": ModelEndpointConfig(
        model_name="whisper-large-v3",
        hf_model_id="openai/whisper-large-v3",
        instance_type="ml.g5.xlarge",
    ),
    "qwen3-asr": ModelEndpointConfig(
        model_name="qwen3-asr",
        hf_model_id="Qwen/Qwen3-ASR-1.7B",
        instance_type="ml.g5.xlarge",
    ),
}

TTS_MODEL_CONFIGS: dict[str, ModelEndpointConfig] = {
    "kokoro-82m": ModelEndpointConfig(
        model_name="kokoro-82m",
        hf_model_id="hexgrad/Kokoro-82M",
        instance_type="ml.g4dn.xlarge",
    ),
    "maya-veena": ModelEndpointConfig(
        model_name="maya-veena",
        hf_model_id="maya-research/veena-tts",
        instance_type="ml.g5.xlarge",
    ),
    "chatterbox-turbo": ModelEndpointConfig(
        model_name="chatterbox-turbo",
        hf_model_id="ResembleAI/chatterbox-turbo",
        instance_type="ml.g4dn.xlarge",
    ),
    "orpheus-3b": ModelEndpointConfig(
        model_name="orpheus-3b",
        hf_model_id="canopylabs/orpheus-3b-0.1-ft",
        instance_type="ml.g5.xlarge",
    ),
}
