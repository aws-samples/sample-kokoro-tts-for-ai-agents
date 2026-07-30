"""Types for TTS performance benchmarking.

The capacity-planning models here enforce the distinction the whole deliverable
rests on: what was **measured** against a live endpoint versus what was
**assumed** as a scenario argument. We have no production traffic — the ~26k
Kokoro invocations in CloudWatch are our own synthetic load — so the growth
factor ``k`` and the peak/trough rates are inputs, not observations. Tagging them
with :class:`Provenance` is what stops a scenario output being read later as a
traffic study.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tts_inference.types import TTSModelName


class LatencyStats(BaseModel):
    """Latency statistics for a benchmark run."""

    p50_ms: float
    p90_ms: float
    p95_ms: float
    p99_ms: float
    mean_ms: float
    min_ms: float
    max_ms: float


class ThroughputResult(BaseModel):
    """Throughput benchmark result."""

    model_name: TTSModelName
    chars_per_second: float = Field(description="Characters synthesized per wall-clock second")
    realtime_factor: float = Field(description="RTF: processing_time / audio_duration")
    latency: LatencyStats
    num_samples: int
    total_chars: int
    total_audio_seconds: float
    total_elapsed_seconds: float


class BenchmarkReport(BaseModel):
    """Aggregate benchmark report."""

    model_name: TTSModelName
    throughput: ThroughputResult | None = None
    latency: LatencyStats | None = None


class Origin(StrEnum):
    """Where a number came from."""

    MEASURED = "measured"
    """Observed against a live endpoint by ``cmax`` or ``ttotal``."""

    ASSUMPTION = "assumption"
    """Supplied as a scenario argument. Reported as such, never as a finding."""

    DERIVED = "derived"
    """Computed from the two above by ``shared.capacity``."""


class Provenance(BaseModel):
    """How a value was obtained, carried alongside the value itself.

    Every figure in the report is tagged. A plan built on assumed inputs is
    still useful, but only if the reader can tell which parts are which.
    """

    origin: Origin
    run_id: str | None = Field(default=None, description="Run that produced a measured value")
    measured_at: str | None = Field(default=None, description="ISO 8601 timestamp")
    endpoint: str | None = None
    note: str | None = Field(default=None, description="Why this value, in one line")

    @property
    def is_measured(self) -> bool:
        return self.origin is Origin.MEASURED


class Measured(BaseModel):
    """Inputs measured against a live endpoint.

    Produced by ``cmax`` and ``ttotal``, consumed by ``planner``. Serialized to
    an artifact JSON so a plan can be regenerated without re-running load.
    """

    model_config = ConfigDict(frozen=True)

    model_name: TTSModelName
    endpoint: str
    instance_type: str

    c_max_curve: dict[int, float] = Field(
        description=(
            "TTFAB budget in ms -> per-instance concurrency at the knee. A curve "
            "rather than a scalar because the knee moves with the SLO, and "
            "emitting all budgets from one run makes the SLO decision a table "
            "lookup instead of a re-run."
        )
    )
    s_mean_s: float = Field(gt=0, description="Mean service time, seconds")
    s_p95_s: float = Field(gt=0, description="p95 service time, seconds")
    t_total_s: float = Field(
        gt=0,
        description=(
            "Scaling lag: metric publication through to an instance serving good "
            "traffic. The single number the whole plan is most sensitive to."
        ),
    )

    frozen: bool = Field(
        default=False,
        description=(
            "Whether autoscaling was suspended and capacity pinned during the "
            "C_max run. False means the number may be N x C_max — see fixture.py."
        ),
    )
    instance_counts_observed: tuple[int, ...] = Field(
        default=(),
        description="Distinct instance counts seen across steps. More than one invalidates C_max.",
    )
    transport: str = Field(
        default="response-stream",
        description=(
            "Wire protocol c_max_curve was measured on. Carried this far because "
            "the plan derived from it configures a real fleet: a C_max measured "
            "on response-stream does not describe capacity for bidi traffic."
        ),
    )
    provenance: Provenance = Field(
        default_factory=lambda: Provenance(origin=Origin.MEASURED),
    )

    @model_validator(mode="after")
    def _check_curve(self) -> Measured:
        if not self.c_max_curve:
            raise ValueError("c_max_curve must not be empty")
        for budget, concurrency in self.c_max_curve.items():
            if budget <= 0:
                raise ValueError(f"TTFAB budget must be positive, got {budget}")
            if concurrency <= 0:
                raise ValueError(f"concurrency at budget {budget} must be positive")
        if self.s_p95_s < self.s_mean_s:
            raise ValueError(
                f"s_p95_s ({self.s_p95_s}) < s_mean_s ({self.s_mean_s}); percentiles disagree"
            )
        return self

    @property
    def trustworthy(self) -> bool:
        """Whether this measurement is safe to build a per-instance plan on."""
        return self.frozen and len(set(self.instance_counts_observed)) <= 1

    def c_max_for(self, ttfab_budget_ms: int) -> float:
        """Concurrency at the knee for a budget.

        Falls back to the nearest measured budget **below** the request, since
        interpolating upward would claim a knee we did not observe.

        Raises:
            ValueError: If no measured budget is at or below the request.
        """
        if ttfab_budget_ms in self.c_max_curve:
            return self.c_max_curve[ttfab_budget_ms]
        below = [b for b in self.c_max_curve if b <= ttfab_budget_ms]
        if not below:
            raise ValueError(
                f"no measured budget at or below {ttfab_budget_ms}ms; "
                f"measured: {sorted(self.c_max_curve)}"
            )
        return self.c_max_curve[max(below)]


class KneePoint(BaseModel):
    """Where the latency SLO was crossed, for one TTFAB budget.

    ``concurrency`` is the *measured* mean in-flight count at that step, not the
    offered rate: at the knee the two differ, and the scaling policy will see the
    measured quantity.
    """

    model_config = ConfigDict(frozen=True)

    ttfab_budget_ms: int = Field(gt=0)
    concurrency: float = Field(gt=0, description="Measured mean concurrency at the knee step")
    offered_rps: float = Field(gt=0)
    p95_ttfab_ms: float = Field(ge=0)
    step_index: int = Field(ge=0)
    bracketed: bool = Field(
        description=(
            "Whether the next step up actually failed the budget. False means the "
            "ladder ran out while still passing, so this is a lower bound on the "
            "knee rather than the knee."
        )
    )

    @property
    def is_lower_bound(self) -> bool:
        """A knee we never bracketed. Planning on it understates required capacity."""
        return not self.bracketed


class StepSummary(BaseModel):
    """One ladder step, flattened for the artifact.

    The in-process path uses dataclasses (``loadgen.WindowStats``); this is the
    serializable projection, carrying only what a later reader needs to judge
    whether the step was a valid measurement.
    """

    model_config = ConfigDict(frozen=True)

    run_index: int = Field(ge=0)
    step_index: int = Field(ge=0)
    target_concurrency: float = Field(gt=0)
    offered_rps: float = Field(gt=0)
    achieved_rps: float = Field(ge=0)
    completed: int = Field(ge=0)
    ok: int = Field(ge=0)
    outcome_counts: dict[str, int] = Field(default_factory=dict)

    ttfab_p50_ms: float | None = None
    ttfab_p95_ms: float | None = None
    ttfab_p99_ms: float | None = None
    latency_p95_ms: float | None = None
    s_mean_s: float | None = None
    s_p95_s: float | None = None
    concurrency_mean: float | None = None
    concurrency_p95: float | None = None
    concurrency_slope_per_s: float | None = None
    chars_per_hour: float = 0.0

    saturated: bool
    settled: bool
    usable: bool = Field(
        description=(
            "Whether this step may inform the knee. False when the fleet resized, "
            "or when the client skipped dispatches and so was itself the limit."
        )
    )
    unusable_reason: str | None = None
    dispatch_skipped: int = Field(default=0, ge=0)
    capacity_changed: bool = False
    instance_counts: tuple[int, ...] = ()

    # Joined from CloudWatch after a settle delay; absent when --no-cloudwatch.
    server_concurrency_mean: float | None = None
    server_model_latency_p95_ms: float | None = None
    server_5xx_total: float | None = None
    gpu_utilization_mean: float | None = None
    cpu_utilization_mean: float | None = None
    concurrency_agreement: str | None = None


class CMaxReport(BaseModel):
    """What one ``cmax`` invocation measured. The Phase 2 artifact.

    Deliberately *not* a :class:`Measured`: that requires ``t_total_s``, which
    only ``ttotal`` can supply. Keeping them separate means neither artifact
    claims a number it did not measure; :meth:`to_measured` is the single place
    the two are joined.

    The curve here is **raw**. ``derate`` is carried for the record but not
    applied — :func:`shared.capacity.c_target` applies it once, and applying it
    here as well would quietly shrink every planned fleet by a second 0.875.
    """

    model_config = ConfigDict(frozen=True)

    model_name: TTSModelName
    endpoint: str
    instance_type: str
    run_id: str

    c_max_curve: dict[int, float] = Field(
        description="TTFAB budget in ms -> per-instance concurrency at the knee, median of runs"
    )
    knees: list[KneePoint] = Field(
        default_factory=list, description="Per-budget knee detail from the last run"
    )
    curve_spread: dict[int, float] = Field(
        default_factory=dict,
        description=(
            "Per budget, (max - min) / median across runs. Above ~0.2 the ladder "
            "is measuring noise and the derate will not cover it."
        ),
    )

    s_mean_s: float = Field(
        gt=0,
        description=(
            "Uncontended mean service time, from the lowest usable step. Not from "
            "the knee step: service time there already includes queueing, and "
            "C_slo_cap = W_max / S would then double-count the wait it is bounding."
        ),
    )
    s_p95_s: float = Field(gt=0, description="Uncontended p95 service time, seconds")

    derate: float = Field(
        default=0.875,
        gt=0,
        le=1.0,
        description="Carried for the record. Applied downstream, never here.",
    )
    frozen: bool = Field(
        default=False,
        description="Whether autoscaling was suspended and capacity pinned for the run",
    )
    instance_counts_observed: tuple[int, ...] = ()
    ladder_truncated_at: int | None = Field(
        default=None,
        description=(
            "Step index where the ladder stopped early after repeated saturation. "
            "Recorded so a curve is never read as covering rates that were skipped."
        ),
    )

    transport: str = Field(
        default="response-stream",
        description=(
            "Wire protocol the ladder ran on. Not decoration: the containers hold "
            "their inference lock differently per transport - kokoro holds it "
            "across an entire bidi session but per-generator on response-stream - "
            "so a C_max from one does not transfer to the other."
        ),
    )

    runs: int = Field(default=1, ge=1)
    hold_s: float = Field(gt=0)
    measure_window_s: float = Field(gt=0)
    arrival_process: str = "poisson"
    seed: int | None = None
    steps: list[StepSummary] = Field(default_factory=list)
    provenance: Provenance = Field(default_factory=lambda: Provenance(origin=Origin.MEASURED))

    @model_validator(mode="after")
    def _check_report(self) -> CMaxReport:
        if not self.c_max_curve:
            raise ValueError(
                "c_max_curve is empty: no step met any TTFAB budget without saturating. "
                "Either the endpoint is unhealthy or the lowest ladder rate is already "
                "past capacity — lower --target-concurrency and re-run."
            )
        for budget, concurrency in self.c_max_curve.items():
            if budget <= 0:
                raise ValueError(f"TTFAB budget must be positive, got {budget}")
            if concurrency <= 0:
                raise ValueError(f"concurrency at budget {budget} must be positive")
        if self.s_p95_s < self.s_mean_s:
            raise ValueError(
                f"s_p95_s ({self.s_p95_s}) < s_mean_s ({self.s_mean_s}); percentiles disagree"
            )
        return self

    @property
    def trustworthy(self) -> bool:
        """Whether this curve is safe to read as *per-instance*."""
        return self.frozen and len(set(self.instance_counts_observed)) <= 1

    @property
    def unbracketed_budgets(self) -> list[int]:
        """Budgets whose knee is only a lower bound, in ascending order."""
        return sorted(k.ttfab_budget_ms for k in self.knees if k.is_lower_bound)

    def to_measured(
        self,
        *,
        t_total_s: float,
        t_total_provenance: Provenance | None = None,
    ) -> Measured:
        """Join this curve with a measured ``T_total`` into a planner input.

        The one place the two phases meet. ``frozen`` and
        ``instance_counts_observed`` are carried through rather than defaulted,
        so a curve measured without the freeze stays identifiable as such after
        the join.

        Args:
            t_total_s: Scaling lag from ``ttotal``, seconds.
            t_total_provenance: Provenance of ``t_total_s``. Its ``note`` is
                folded into the result so an assumed lag cannot be mistaken for
                a measured one.
        """
        note = self.provenance.note
        if t_total_provenance is not None and not t_total_provenance.is_measured:
            note = f"C_max measured; T_total {t_total_provenance.origin.value}"
        return Measured(
            model_name=self.model_name,
            endpoint=self.endpoint,
            instance_type=self.instance_type,
            c_max_curve=dict(self.c_max_curve),
            s_mean_s=self.s_mean_s,
            s_p95_s=self.s_p95_s,
            t_total_s=t_total_s,
            frozen=self.frozen,
            instance_counts_observed=self.instance_counts_observed,
            transport=self.transport,
            provenance=Provenance(
                origin=Origin.MEASURED,
                run_id=self.run_id,
                measured_at=self.provenance.measured_at,
                endpoint=self.endpoint,
                note=note,
            ),
        )


class Scenario(BaseModel):
    """Expected load and policy choices. Supplied as CLI arguments.

    Everything here is an assumption. ``k`` in particular is *not* measured: it
    is the factor by which traffic might grow within one ``T_total``, and with no
    production history there is nothing to measure it from.
    """

    model_config = ConfigDict(frozen=True)

    peak_rps: float | None = Field(default=None, ge=0)
    trough_rps: float | None = Field(default=None, ge=0)
    peak_streams: float | None = Field(
        default=None,
        ge=0,
        description=(
            "Concurrent long-lived sessions. Bypasses the lambda x S conversion, "
            "which is the honest input for bidirectional streaming where one "
            "session is not one request."
        ),
    )
    trough_streams: float | None = Field(default=None, ge=0)

    growth_factor_k: float = Field(
        default=2.0,
        ge=1.0,
        description="Traffic growth within one T_total. k=1 means flat.",
    )
    ttfab_budget_ms: int = Field(default=300, gt=0, description="p95 TTFAB SLO")
    max_added_wait_s: float = Field(
        default=2.0,
        ge=0,
        description=(
            "W_max: added wait a queued request may absorb. Hard-capped by "
            "SageMaker's 60s invocation ceiling."
        ),
    )
    derate: float = Field(default=0.875, gt=0, le=1.0)
    min_instances_floor: int = Field(
        default=1,
        ge=1,
        description="Never plan below this, whatever the trough says.",
    )
    provenance: Provenance = Field(
        default_factory=lambda: Provenance(
            origin=Origin.ASSUMPTION,
            note="supplied as a CLI argument; no production traffic to measure",
        ),
    )

    @model_validator(mode="after")
    def _check_load_given(self) -> Scenario:
        if self.peak_rps is None and self.peak_streams is None:
            raise ValueError("supply either peak_rps or peak_streams")
        if self.peak_rps is not None and self.trough_rps is not None:
            if self.trough_rps > self.peak_rps:
                raise ValueError(f"trough_rps ({self.trough_rps}) > peak_rps ({self.peak_rps})")
        if self.peak_streams is not None and self.trough_streams is not None:
            if self.trough_streams > self.peak_streams:
                raise ValueError(
                    f"trough_streams ({self.trough_streams}) > peak_streams ({self.peak_streams})"
                )
        return self


class Verdict(StrEnum):
    """Whether a feasibility condition holds."""

    OK = "ok"
    WARN = "warn"
    """Feasible, but the configuration is fragile or expensive."""

    INFEASIBLE = "infeasible"
    """The stated SLO cannot be met with this configuration."""

    SUPPRESSED = "suppressed"
    """Not evaluated for want of data. Distinct from OK — it is not a pass."""


class Finding(BaseModel):
    """One feasibility statement in the report."""

    name: str
    verdict: Verdict
    detail: str
    recommendation: str | None = None


class ScalingPlan(BaseModel):
    """The deliverable: the configuration needed to hold the SLO under a scenario.

    Field names mirror the ``ModelEndpointConfig`` fields they map to, so the
    report can print a paste-ready block.
    """

    model_config = ConfigDict(frozen=True)

    model_name: TTSModelName
    endpoint: str
    instance_type: str

    c_max: float = Field(description="Per-instance concurrency at the knee, at the chosen budget")
    c_target: float = Field(
        description=(
            "Per-instance concurrency for the scaling policy. Float, not int: "
            "Kokoro's C_max of 1 at k=2 gives 0.44, which an int cannot express."
        )
    )
    binding_constraint: str = Field(
        description="'surge_headroom' or 'slo_wait_budget' — the fix differs by which binds"
    )

    min_instances: int = Field(ge=1)
    max_instances: int = Field(ge=1)
    peak_instances: int = Field(ge=0)
    trough_instances: int = Field(ge=0)

    queue_max_depth: int = Field(ge=0, description="Q_max per instance = Lambda_cap x W_max")
    scale_out_cooldown_s: int = Field(ge=0)
    scale_in_cooldown_s: int = Field(ge=0)

    utilization_at_target: float = Field(
        description="derate / k — the standing price of the surge reserve"
    )
    w_absorbed_s: float = Field(description="Scaling lag the queue hides from clients")
    headroom_lag_s: float = Field(description="Lag standing headroom must cover after the queue")

    peak_cost_per_hour: float
    peak_cost_per_m_chars: float
    relative_fleet_cost_vs_k1: float = Field(
        description="Fleet size at this k divided by fleet size at k=1"
    )

    findings: list[Finding] = Field(default_factory=list)
    measured: Measured
    scenario: Scenario

    @property
    def infeasible(self) -> bool:
        return any(f.verdict is Verdict.INFEASIBLE for f in self.findings)

    @property
    def warnings(self) -> list[Finding]:
        return [f for f in self.findings if f.verdict is Verdict.WARN]
