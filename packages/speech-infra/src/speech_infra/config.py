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

    #: Metric the scale-in step policy reads. The native SageMaker metric, not a
    #: container-published one: ``Speech/vLLM`` published nothing, so the policies
    #: built on it sat in ``INSUFFICIENT_DATA`` and could never fire.
    scaling_metric_namespace: str = "AWS/SageMaker"
    scaling_metric_name: str = "ConcurrentRequestsPerModel"

    #: Per-instance concurrency the scale-out policy tracks. A float because
    #: ``derate x C_max / k`` rarely lands on an integer: kokoro's measured
    #: C_max of 1.63 at k=2 gives 0.713, which an int cannot express.
    scaling_target_value: float = 8.0

    #: Concurrency at or below which one instance is removed. Well under
    #: ``scaling_target_value`` so the two policies do not oscillate around it.
    scale_in_threshold: float = 0.2

    #: Cooldowns are deliberately asymmetric. Scaling out costs money and is
    #: reversible; scaling in drops capacity that takes a full T_total to get
    #: back, so it waits long enough to be sure the load is really gone.
    scale_out_cooldown_s: int = 30
    scale_in_cooldown_s: int = 600

    #: Opt-in steep step-out for models that cannot wait for target tracking to
    #: converge one step at a time. Off until a measurement justifies it.
    emergency_step_enabled: bool = False

    #: The p95 TTFAB budget ``scaling_target_value`` was derived against. Recorded
    #: beside the target because a target without its budget is unfalsifiable —
    #: you cannot tell later which SLO it was meant to hold.
    ttfab_budget_ms: int = 300

    #: W_max: added queueing wait a request may absorb. Hard-capped well under
    #: SageMaker's 60s invocation ceiling.
    max_added_wait_s: float = 2.0

    #: Q_max per instance = Lambda_cap x W_max. Consumed by the container
    #: admission queue; 0 means unbounded, i.e. not yet planned for this model.
    queue_max_depth: int = 0

    container_startup_health_check_timeout_s: int = 600

    cache_model_weights: bool = False

    container_env: dict[str, str] = Field(default_factory=dict)
    codec_model_ids: list[str] = Field(default_factory=list)

    @property
    def endpoint_name(self) -> str:
        return f"speech-{self.model_name}"

    @property
    def stack_id(self) -> str:
        return f"Speech-{self.model_name}"

    @property
    def scaling_enabled(self) -> bool:
        """Whether autoscaling is meaningful for this config."""
        effective_min = max(self.min_instances, 1)
        return self.max_instances > effective_min

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
    # The one model with a measured C_max, so the one model configured to scale.
    # From artifacts/cmax-kokoro-bidi.json (bidi transport, frozen, 1 instance):
    # C_max 1.63 concurrent at p95 TTFAB 276ms, S mean 110ms, so Lambda_cap 14.82 rps.
    # At the chosen k=2: C_target = 0.875 x 1.63 / 2 = 0.713, or 44% utilization.
    #
    # C_target below 1 is not a mistake. Kokoro holds its inference lock for a whole
    # bidi session, so an instance serves about one stream and one sustained request
    # is enough to scale out. For this model max_instances, not the target, is the
    # lever that sizes the fleet.
    "kokoro-82m": ModelEndpointConfig(
        model_name="kokoro-82m",
        hf_model_id="hexgrad/Kokoro-82M",
        instance_type="ml.g5.xlarge",
        container_type=ContainerType.PYTORCH_CUSTOM,
        streaming_mode=StreamingMode.BIDIRECTIONAL,
        # min=1 rather than 0 because SageMaker real-time variants cannot scale to
        # zero; max(min_instances, 1) already coerced it, so 1 is what deploys.
        min_instances=1,
        # Placeholder: enough to observe a real scale-out, which `tts-bench ttotal`
        # needs. Size it properly once `tts-bench plan` runs against a stated peak.
        max_instances=4,
        scaling_target_value=0.713,
        scale_in_threshold=0.2,
        ttfab_budget_ms=300,
        max_added_wait_s=20.0,
        queue_max_depth=296,
    ),
    "kokoro-82m-cpu": ModelEndpointConfig(
        model_name="kokoro-82m-cpu",
        hf_model_id="hexgrad/Kokoro-82M",
        instance_type="ml.c5.2xlarge",
        container_type=ContainerType.PYTORCH_CUSTOM,
        streaming_mode=StreamingMode.BIDIRECTIONAL,
        min_instances=0,
        max_instances=1,
        scaling_target_value=4,
    ),
    # min/max stated explicitly rather than inherited. Taking the class defaults
    # (1-4) made `scaling_enabled` true for a model with no endpoint and no stack,
    # which is the whole reason `tts-bench drift` reported a missing_target for it.
    # No model gets scaling until its own C_max is measured.
    "maya-veena": ModelEndpointConfig(
        model_name="maya-veena",
        hf_model_id="maya-research/veena-tts",
        instance_type="ml.g5.xlarge",
        container_type=ContainerType.VLLM,
        streaming_mode=StreamingMode.BIDIRECTIONAL,
        min_instances=1,
        max_instances=1,
        cache_model_weights=True,
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
        cache_model_weights=True,
        min_instances=0,
        max_instances=1,
        scaling_target_value=4,
        container_env={
            "DEFAULT_VOICE": "ENG_US_F_KimW",
        },
    ),
    "orpheus-3b": ModelEndpointConfig(
        model_name="orpheus-3b",
        hf_model_id="canopylabs/orpheus-3b-0.1-ft",
        instance_type="ml.g5.xlarge",
        container_type=ContainerType.VLLM,
        streaming_mode=StreamingMode.BIDIRECTIONAL,
        cache_model_weights=True,
        min_instances=1,
        max_instances=1,
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
