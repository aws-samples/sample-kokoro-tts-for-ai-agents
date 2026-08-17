# Copyright (c) 2026 Amazon.com
# This file is licensed under the MIT License.
# See the LICENSE file in the project root for full license information.

"""Model endpoint configuration for SageMaker deployment."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator


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

    #: SageMaker instance type, e.g. ``ml.g6.xlarge``. Validated for the ``ml.`` prefix
    #: rather than against a list, since AWS adds types faster than this file changes.
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

    #: ``C_scale_max``: the concurrency (queued + executing) at which one instance is
    #: added. Derived, never chosen — with ``h = max_scaling_per_T_total - 1`` it is
    #: ``(1 - h) x Q_max``, so the fleet still has ``h`` of queue headroom when the
    #: request for capacity goes out. A float because that product rarely lands on an
    #: integer.
    #:
    #: **In CloudWatch ``Maximum`` units, not client units.** The policy compares this
    #: against ``ConcurrentRequestsPerModel`` / ``Maximum`` over 10s, which is a
    #: different quantity from a client-measured mean in-flight: on one kokoro ladder
    #: the two differed by 1.35x to 9.8x, shrinking as load rose. The 0.713 that used
    #: to sit in the kokoro block below was the client figure deployed unconverted, and
    #: no positive arrival rate satisfies it. ``tts-bench plan`` measures the ratio on
    #: the same ladder that measures ``Q_max`` and emits this field already converted.
    scaling_target_value: float = 8.0

    #: ``C_scale_min``: the concurrency at or below which one instance is removed, in the
    #: same CloudWatch units. ``(1 - 2h) x Q_max`` — one surge of *excess* headroom, so
    #: scale-in needs the load to have really gone rather than merely dipped.
    #:
    #: Removing 1 of N instances multiplies per-instance concurrency by ``N/(N-1)``, so
    #: this is only stable while ``N/(N-1) <= C_scale_max/C_scale_min``, i.e. ``N >= 3``
    #: at ``h=0.25``. Below that a scale-in lands the survivors at or above
    #: ``scaling_target_value`` and they scale straight back out; ``plan`` emits a
    #: ``scale_in_safety`` finding with the smallest safe N, and the long
    #: ``scale_in_cooldown_s`` damps what remains.
    scale_in_threshold: float = 0.2

    #: Cooldowns are deliberately asymmetric. Scaling out costs money and is
    #: reversible; scaling in drops capacity that takes a full T_total to get
    #: back, so it waits long enough to be sure the load is really gone.
    scale_out_cooldown_s: int = 30
    scale_in_cooldown_s: int = 600

    #: Opt-in steep step-out for models that cannot wait for target tracking to
    #: converge one step at a time. Off until a measurement justifies it.
    emergency_step_enabled: bool = False

    #: End-to-end p95 first-byte SLO, milliseconds: queue wait *plus* service, the whole
    #: promise to the client. **One SLO, one number, one field.** Every other queueing
    #: number descends from it: it is the pass/fail line the ``Q_max`` ladder walks, so
    #: ``queue_max_depth`` is *defined* by it, and both scaling thresholds are fractions
    #: of that ``Q_max``.
    #:
    #: It is the only latency figure stated here, deliberately. A second one used to sit
    #: beside it (a 300ms "budget" naming which knee of a multi-budget curve to read), and
    #: two independent latency fields can disagree with the promise: that pair is how this
    #: model came to declare a 20s queue allowance under a 300ms budget, where a request
    #: spending its allowance reached first byte at 20.2s while the config claimed 0.3s.
    #: The consumers that need a *tighter* threshold than the promise — the
    #: ``FirstChunkLatencyP95`` alarm, which watches service time on an instance already
    #: serving and so has spent none of the queue allowance — read the ladder's own N=1
    #: rung (``ttfab_p95_at_c1_ms``) instead. Measured on every rerun, not hand-set.
    ttfab_slo_ms: int = 3000

    #: ``Q_max``: the largest concurrency (queued + executing) at which a request still
    #: reaches first byte inside ``ttfab_slo_ms``. Consumed by the container admission
    #: queue, which sheds past it — a request that cannot make the SLO is better refused
    #: than served late. 0 means unbounded, i.e. not yet measured for this model.
    #:
    #: Measured, not computed: ``tts-bench qmax`` steps concurrency against a frozen
    #: single instance until p95 first byte crosses the SLO. It cannot be derived here
    #: because it depends on the queue discipline and the service-time distribution, not
    #: just their means — and because a value that is only ever *stated* cannot be shown
    #: wrong by a rerun on a different instance type.
    queue_max_depth: int = 0

    container_startup_health_check_timeout_s: int = 600

    cache_model_weights: bool = False

    container_env: dict[str, str] = Field(default_factory=dict)
    codec_model_ids: list[str] = Field(default_factory=list)

    @field_validator("instance_type")
    @classmethod
    def _instance_type_looks_like_sagemaker(cls, value: str) -> str:
        """Reject anything without the ``ml.`` prefix SageMaker requires.

        Worth the check because ``instance_type`` is now changed routinely — the
        harness re-measures per configuration, and a candidate type arrives by flag
        (see ``app.py``'s ``instance_type`` context override). A typo like
        ``g6.xlarge`` is accepted by CloudFormation and then sits in ``Updating``
        with no ``FailureReason``, which costs far more to diagnose than to prevent.
        """
        if not value.startswith("ml."):
            raise ValueError(
                f"instance_type must start with 'ml.' (SageMaker's prefix), got {value!r}"
            )
        return value

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
    if name not in TTS_MODEL_CONFIGS:
        available = ", ".join(sorted(TTS_MODEL_CONFIGS.keys()))
        raise KeyError(f"Unknown model '{name}'. Available: {available}")
    return TTS_MODEL_CONFIGS[name]


TTS_MODEL_CONFIGS: dict[str, ModelEndpointConfig] = {
    # The one model whose Q_max and T_total are measured, so the one model configured to
    # scale. The model these numbers come from -- six variables, two chosen, two measured,
    # two derived, and the rerun constraints -- is written down once, in
    # docs/autoscaling-capacity-model.md. The operator recipe is packages/tts-bench/README.md.
    #
    # Do not hand-edit the derived fields. `uv run tts-bench plan --qmax <artifact> --ttotal
    # <artifact>` prints this block ready to paste, and `tts-bench drift` then confirms that
    # what deployed is what was computed.
    "kokoro-82m": ModelEndpointConfig(
        model_name="kokoro-82m",
        hf_model_id="hexgrad/Kokoro-82M",
        instance_type="ml.g5.xlarge",
        container_type=ContainerType.PYTORCH_CUSTOM,
        streaming_mode=StreamingMode.RESPONSE_STREAM,
        min_instances=1,  # trough 10 rps; 3 is the smallest safe for scale-in
        max_instances=9,  # peak 5000 rps
        scaling_target_value=37.500,  # C_scale_max 37.50 x 1.00 CW units; = (1-h) x Q_max 50 at h=0.25
        scale_in_threshold=25.000,  # C_scale_min 25.00 x 1.00; = (1-2h) x Q_max, one surge of excess headroom
        ttfab_slo_ms=3000,  # end-to-end promise; W_max 2.93s + p95 service 0.07s fits the 60s invocation ceiling
        queue_max_depth=50,  # = Q_max: past it a request cannot reach first byte inside the 3.0s SLO
        scale_out_cooldown_s=30,  # short: target tracking adds one instance at a time
        scale_in_cooldown_s=960,  # long: removed capacity costs a full 320s to replace
    ),
}
