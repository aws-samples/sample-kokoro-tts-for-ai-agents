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
from typing import Any

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
    c_max_throughput: float | None = Field(
        default=None,
        gt=0,
        description=(
            "Per-instance concurrency at the throughput ceiling, independent of any "
            "latency budget. None means no cmax run measured it — older artifacts, and "
            "ladders that never saturated. Distinct from the curve because the two can "
            "disagree: a model may hold latency inside a generous budget while already "
            "failing to keep up, and then this is the number that binds."
        ),
    )
    c_max_throughput_bracketed: bool = Field(
        default=True,
        description=(
            "Whether a saturated step was seen above the ceiling. False makes the "
            "ceiling a lower bound, which matters because the planner takes the "
            "*minimum* of the two C_max kinds: a lower-bound minimum understates the "
            "fleet, and the report has to say so rather than print a bare number."
        ),
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
    t_total_measured: bool = Field(
        default=True,
        description=(
            "Whether t_total_s came from a ttotal run or from --assume-t-total. "
            "Separate from `provenance.origin`, which describes this object as a "
            "whole: C_max and S are measured even when the lag was stated, so the "
            "origin stays MEASURED and cannot answer this. Without the distinction "
            "the planner reports a command-line argument as 'used exactly as "
            "measured', which is the one claim the provenance machinery exists to "
            "prevent."
        ),
    )
    chars_per_request: float = Field(
        default=0.0,
        ge=0,
        description=(
            "Mean characters per request across the ladder. Carried because the "
            "planner prices a fleet in $/M chars and has no corpus to measure it "
            "from; hard-coding an average would misprice any run against a "
            "different sample set. 0 means unknown, which surfaces as an infinite "
            "unit cost rather than a plausible wrong one."
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
    deployed_config: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Fingerprint of the configuration measured: instance type, container "
            "image digest, container env. See fixture.DeployedConfig. Checked "
            "before this artifact is replayed, because every number here is a "
            "property of a configuration rather than of a model. Empty means the "
            "artifact predates fingerprinting, which counts as a mismatch."
        ),
    )
    curve_spread: dict[int, float] = Field(
        default_factory=dict,
        description=(
            "Per-budget run-to-run spread of the curve, carried from CMaxReport. "
            "The planner reads it to say whether C_max is repeatable: every fleet "
            "size divides by C_max, so a curve that resolved noise produces a "
            "confidently wrong instance count."
        ),
    )
    runs_contributing: dict[int, int] = Field(
        default_factory=dict,
        description=(
            "Per budget, how many ladder runs found a knee, out of runs_total. "
            "Carried alongside curve_spread because a spread of 0% means agreement "
            "only when more than one run contributed."
        ),
    )
    c_max_bracketed: dict[int, bool] = Field(
        default_factory=dict,
        description=(
            "Per budget, whether the ladder observed a step above the knee that "
            "actually failed. False makes that budget's C_max a lower bound, which the "
            "planner has to say out loud: it takes the *minimum* of the knee and the "
            "throughput ceiling, and a lower-bound minimum understates the fleet. "
            "A budget absent from this dict is unknown rather than bracketed — an "
            "artifact predating the field has not passed the check."
        ),
    )
    runs_total: int = Field(
        default=1,
        ge=1,
        description="Ladder passes the curve was measured over.",
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

    def budget_at_or_below(self, ttfab_budget_ms: int) -> int | None:
        """The measured budget this request resolves to, or ``None``.

        Extracted because three callers need the resolved key rather than the value —
        :meth:`c_max_for` for the concurrency, and the spread and bracketing lookups for
        their own dicts keyed the same way. Each doing its own ``max(b for b <= ...)``
        was how a knee at 500ms came to be reported beside a spread from 300ms.
        """
        if ttfab_budget_ms in self.c_max_curve:
            return ttfab_budget_ms
        below = [b for b in self.c_max_curve if b <= ttfab_budget_ms]
        return max(below) if below else None

    def c_max_for(self, ttfab_budget_ms: int) -> float:
        """Concurrency at the knee for a budget.

        Falls back to the nearest measured budget **below** the request, since
        interpolating upward would claim a knee we did not observe.

        Raises:
            ValueError: If no measured budget is at or below the request.
        """
        budget = self.budget_at_or_below(ttfab_budget_ms)
        if budget is None:
            raise ValueError(
                f"no measured budget at or below {ttfab_budget_ms}ms; "
                f"measured: {sorted(self.c_max_curve)}"
            )
        return self.c_max_curve[budget]

    def knee_bracketed(self, ttfab_budget_ms: int) -> bool | None:
        """Whether the knee at this budget was bracketed, or ``None`` if unrecorded.

        Three-valued on purpose. ``False`` is a measured lower bound — the ladder ran
        out while still passing — and ``None`` is an artifact that never recorded the
        question. Collapsing them to a boolean would either warn about old artifacts as
        though they had failed the check, or silently pass a genuine lower bound.
        """
        budget = self.budget_at_or_below(ttfab_budget_ms)
        if budget is None:
            return None
        return self.c_max_bracketed.get(budget)

    def binding_c_max(self, ttfab_budget_ms: int) -> tuple[float, str]:
        """The C_max to plan on, and which measurement produced it.

        Takes the **lower** of the latency knee and the throughput ceiling. Both are
        real per-instance limits and an instance is bound by whichever it reaches first,
        so planning on the higher one sizes a fleet for capacity that does not exist.

        Returns:
            ``(c_max, source)`` where ``source`` is ``"latency_knee"``,
            ``"throughput_ceiling"``, or ``"latency_knee_only"`` when no ceiling was
            measured. The third value is not the same claim as the first: it says the
            comparison never happened, which for a model like kokoro — throughput-bound
            well below its 3s latency knee — is the difference between a checked answer
            and an unchecked one.

        Raises:
            ValueError: Via :meth:`c_max_for`, if no budget is at or below the request.
        """
        knee = self.c_max_for(ttfab_budget_ms)
        if self.c_max_throughput is None:
            return knee, "latency_knee_only"
        if self.c_max_throughput < knee:
            return self.c_max_throughput, "throughput_ceiling"
        return knee, "latency_knee"

    def binding_is_lower_bound(self, ttfab_budget_ms: int) -> bool | None:
        """Whether the C_max the plan will use understates the instance's capacity.

        Asked of whichever measurement :meth:`binding_c_max` selected, because that is
        the only one the fleet size divides by. A bracketed ceiling sitting under an
        unbracketed knee is a *bounded* answer, and warning about the knee there would
        send the operator to extend a ladder that would not change the plan.

        ``None`` means the selected measurement did not record the question — an older
        artifact for the knee, which is not the same as a pass.

        Raises:
            ValueError: Via :meth:`binding_c_max`, if no budget is at or below the request.
        """
        _, source = self.binding_c_max(ttfab_budget_ms)
        if source == "throughput_ceiling":
            return not self.c_max_throughput_bracketed
        bracketed = self.knee_bracketed(ttfab_budget_ms)
        return None if bracketed is None else not bracketed


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


class ThroughputCeiling(BaseModel):
    """The highest rate an instance sustained, and the concurrency it implies.

    A second, independent kind of ``C_max``. :class:`KneePoint` answers "where does
    latency degrade"; this answers "where does the server stop keeping up", and they are
    not the same question. A model can hold p95 TTFAB well inside a generous budget
    while already refusing to drain its queue — kokoro on bidi does exactly that, since
    it holds the inference lock for a whole session, so its throughput ceiling binds long
    before any 3s latency knee.

    Reading only the latency knee in that regime is actively misleading: the ladder's
    high steps show *rising* measured concurrency, but that concurrency is accumulated
    backlog rather than useful work, and planning on it would size a fleet for capacity
    the instance does not have.

    ``concurrency`` is ``max_sustained_rps x S_uncontended`` — Little's Law on the highest
    rate that did not saturate. That is *useful* concurrency: the in-flight count the work
    itself accounts for, with no queueing in it. Defining it this way makes
    ``lambda_cap_per_instance(concurrency, S)`` return ``max_sustained_rps`` exactly, so the
    rate the plan permits is the rate that was measured rather than one inferred from it.

    **It is not the same unit as a knee's concurrency**, and that is worth stating plainly
    because :meth:`Measured.binding_c_max` compares the two. :class:`KneePoint` reports the
    *observed* mean in-flight count, which includes queue residence; at kokoro's ceiling step
    observed was 2.93 against a useful 1.382, so the two derivations diverge by 2x exactly
    where it matters. Observed is never below useful, so taking the lower of the two errs
    toward a larger fleet, and ``observed_concurrency`` is recorded here so a reader can see
    the gap instead of assuming there is none.
    """

    model_config = ConfigDict(frozen=True)

    max_sustained_rps: float = Field(
        gt=0, description="Highest achieved rate at which the server kept up with the offer"
    )
    concurrency: float = Field(
        gt=0, description="max_sustained_rps x S: the ceiling as *useful* concurrency"
    )
    observed_concurrency: float | None = Field(
        default=None,
        gt=0,
        description=(
            "Measured mean in-flight count at the ceiling step — what a "
            "scaling_target_value policy would actually see. Divided by concurrency it "
            "gives the queueing multiple at the ceiling (2.1x for kokoro on bidi), which "
            "is how much of the instance's residence time is already wait rather than "
            "work. None when the 1Hz monitor produced no samples."
        ),
    )
    offered_rps: float = Field(gt=0, description="What was offered at that step")
    p95_ttfab_ms: float = Field(
        ge=0,
        description=(
            "Latency at the ceiling step. Recorded because a ceiling reached with "
            "latency still inside the budget is the whole point: it proves the limit "
            "was throughput and not the SLO."
        ),
    )
    step_index: int = Field(ge=0)
    bracketed: bool = Field(
        description=(
            "Whether a saturated step was observed above this one. False means the "
            "ladder ran out while the server was still keeping up, so this is a lower "
            "bound on the ceiling rather than the ceiling."
        )
    )
    dispatch_skipped: int = Field(
        default=0,
        ge=0,
        description=(
            "Dispatches the client could not issue at the ceiling step. Nonzero makes "
            "max_sustained_rps a lower bound for a second reason: the server was never "
            "handed the full offered rate, so it may sustain more. Not grounds to discard "
            "the step — the rate it did deliver is still a rate it delivered — and heavy "
            "skipping tends to exclude itself, since it drags achieved below "
            "SATURATION_RATIO x offered and the step reads as saturated."
        ),
    )
    runs_contributing: int = Field(
        default=1,
        ge=1,
        description=(
            "Ladder runs that produced a ceiling. Attached here rather than keyed on the "
            "report because a ceiling has no budget to key it by. 1 means the number "
            "below is a single sample, whatever the spread says."
        ),
    )
    spread: float = Field(
        default=0.0,
        ge=0,
        description=(
            "(max - min) / median of max_sustained_rps across runs. 0.0 with "
            "runs_contributing of 1 means there was nothing to disagree with, not agreement."
        ),
    )

    @property
    def is_lower_bound(self) -> bool:
        """A ceiling we never bracketed, or one the client throttled.

        Either way, planning on it understates capacity. The two causes want different
        fixes — extend the ladder, or raise ``--max-workers`` — so the report says which.
        """
        return not self.bracketed or self.dispatch_skipped > 0

    @property
    def queueing_multiple(self) -> float | None:
        """Observed residence over useful residence at the ceiling, or None.

        Above ~1.5 the instance is spending more time queueing than working at the very
        rate the plan is about to treat as its capacity — a signal that the deployed
        ``scaling_target_value`` (which tracks the observed figure) and this ``C_max``
        describe the same instance in different units.
        """
        if self.observed_concurrency is None or self.concurrency <= 0:
            return None
        return self.observed_concurrency / self.concurrency


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
    throughput_ceiling: ThroughputCeiling | None = Field(
        default=None,
        description=(
            "Highest sustained rate, independent of any latency budget. Optional "
            "because artifacts written before it existed have no value to report, and "
            "absent must read as 'not measured' rather than as 'no ceiling found' — "
            "the planner falls back to the latency knee alone and says so."
        ),
    )
    curve_spread: dict[int, float] = Field(
        default_factory=dict,
        description=(
            "Per budget, (max - min) / median across runs. Above ~0.2 the ladder "
            "is measuring noise and the derate will not cover it."
        ),
    )
    runs_contributing: dict[int, int] = Field(
        default_factory=dict,
        description=(
            "Per budget, how many runs found a knee. The denominator for "
            "curve_spread: a spread of 0% across three runs is agreement, but a "
            "spread of 0% from one contributing run is a single sample with nothing "
            "to disagree with, and the two must not read the same."
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
    deployed_config: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Fingerprint of the configuration the ladder ran against, read from the "
            "endpoint rather than from a registry. See fixture.DeployedConfig."
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
    def config_slug(self) -> str:
        """Short identifier of the configuration measured, for an artifact filename.

        Defaulting the output path to include this stops a second configuration's
        measurement from overwriting the first because ``--output`` was forgotten —
        which would leave the two indistinguishable after the fact.
        """
        from tts_bench.fixture import DeployedConfig

        if self.deployed_config:
            return DeployedConfig.from_dict(self.deployed_config).slug
        return DeployedConfig(instance_type=self.instance_type, image_digest=None).slug

    @property
    def chars_per_request(self) -> float:
        """Mean characters per request across the usable steps.

        Derived from ``chars_per_hour / achieved_rps`` per step rather than stored,
        so it cannot drift from the steps it summarizes. Only the planner needs it —
        to price a fleet in $/M chars, which requires knowing what a request *is* —
        and hard-coding a corpus average there would misprice any run against a
        different sample set.

        Returns 0.0 when no step recorded both figures, which the planner surfaces as
        an infinite unit cost: visibly absent rather than plausibly wrong.
        """
        per_step = [
            s.chars_per_hour / (s.achieved_rps * 3600.0)
            for s in self.steps
            if s.chars_per_hour > 0 and s.achieved_rps > 0
        ]
        if not per_step:
            return 0.0
        return sum(per_step) / len(per_step)

    @property
    def unbracketed_budgets(self) -> list[int]:
        """Budgets whose knee is only a lower bound, in ascending order."""
        return sorted(k.ttfab_budget_ms for k in self.knees if k.is_lower_bound)

    @property
    def _top_step_index(self) -> int:
        """Highest step index the run that produced :attr:`knees` actually reached.

        Scoped to that one run because ``steps`` holds every run and they can
        truncate at different points: a global maximum would call a knee
        "inconclusive" on the strength of a step a *different* run ran.
        """
        if not self.steps:
            return -1
        last_run = max(s.run_index for s in self.steps)
        return max(s.step_index for s in self.steps if s.run_index == last_run)

    @property
    def exhausted_budgets(self) -> list[int]:
        """Unbracketed budgets whose knee sits at the very top of the ladder.

        These are the ones a longer ladder would actually resolve. Split from
        :attr:`inconclusive_budgets` because the two need opposite responses and
        a single "extend --target-concurrency" note sent operators to re-run a
        45-minute ladder in the case where extending it changes nothing.
        """
        top = self._top_step_index
        return sorted(
            k.ttfab_budget_ms for k in self.knees if k.is_lower_bound and k.step_index >= top
        )

    @property
    def inconclusive_budgets(self) -> list[int]:
        """Unbracketed budgets where higher rates ran but measured nothing usable.

        The ladder did reach past this knee; those steps just produced no p95 to
        judge — every request failed, or none completed inside the window. A
        longer ladder cannot help, so the fix is upstream of the rate schedule.
        """
        top = self._top_step_index
        return sorted(
            k.ttfab_budget_ms for k in self.knees if k.is_lower_bound and k.step_index < top
        )

    @property
    def throughput_bound_budgets(self) -> list[int]:
        """Budgets whose latency knee sits above the measured throughput ceiling.

        For these, the reported knee is not this instance's capacity: the server had
        already stopped keeping up with the offered rate before latency crossed the budget,
        so the extra concurrency is queue backlog. Nothing *failed* at those steps, which
        is exactly why a latency-only reading passes them silently.

        Empty when no ceiling was measured — absent must read as "not checked" rather than
        as "checked and clean", so callers should test :attr:`throughput_ceiling` for
        ``None`` separately rather than treat an empty list as reassurance.
        """
        if self.throughput_ceiling is None:
            return []
        limit = self.throughput_ceiling.concurrency
        return sorted(b for b, value in self.c_max_curve.items() if value > limit)

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
                appended to this curve's, and its ``origin`` sets
                ``t_total_measured``, so an assumed lag cannot be mistaken for a
                measured one. Omitting it means the caller is supplying a measured
                lag — the default, and what every existing caller does.
        """
        # Both notes, not a summary of one. The caller's note carries the caveats
        # that cannot be reconstructed from `origin` -- that recovery was inferred so
        # the lag is a floor, or that the trigger skipped detection -- and a
        # `Measured` read back from a plan artifact is all a reader has. Substituting
        # "T_total assumption" for it drops exactly the part worth reading.
        parts = [
            part
            for part in (
                self.provenance.note,
                t_total_provenance.note if t_total_provenance is not None else None,
            )
            if part
        ]
        note = "; ".join(parts) or None
        return Measured(
            model_name=self.model_name,
            endpoint=self.endpoint,
            instance_type=self.instance_type,
            deployed_config=dict(self.deployed_config),
            c_max_curve=dict(self.c_max_curve),
            # Flattened to two scalars rather than carried as the whole object: `Measured`
            # is the planner's input and needs the number plus whether to trust it. The
            # full ceiling stays on the CMaxReport for anyone asking how it was reached.
            c_max_throughput=(
                self.throughput_ceiling.concurrency if self.throughput_ceiling else None
            ),
            c_max_throughput_bracketed=(
                # `is_lower_bound`, not `bracketed`: a client-throttled ceiling understates
                # capacity just as much as an unbracketed one, and the planner's question is
                # only "may I trust this as the ceiling", not which cause spoiled it.
                not self.throughput_ceiling.is_lower_bound if self.throughput_ceiling else True
            ),
            s_mean_s=self.s_mean_s,
            s_p95_s=self.s_p95_s,
            t_total_s=t_total_s,
            t_total_measured=t_total_provenance is None or t_total_provenance.is_measured,
            chars_per_request=self.chars_per_request,
            curve_spread=dict(self.curve_spread),
            runs_contributing=dict(self.runs_contributing),
            # From `knees` rather than a second stored dict, so the planner's warning and
            # the knee the CLI prints for that budget can never disagree. `knees` is one
            # representative per budget (`cmax._representative_knees`), so this is that
            # run's answer, not a vote: where runs disagree the newest wins, and a
            # spurious lower-bound warning costs a re-run while a missed one costs an
            # undersized fleet.
            c_max_bracketed={k.ttfab_budget_ms: k.bracketed for k in self.knees},
            runs_total=self.runs,
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

    **``W_max`` is deliberately absent.** It used to sit here beside
    ``ttfab_budget_ms`` as an independent field, and the two could disagree without
    anything noticing — which is how kokoro shipped a 20 s queue allowance under a
    300 ms budget, a request taking 20.2 s to first byte while the config claimed
    0.3 s. The SLO is the whole promise, queue included, so ``W_max`` is derived from
    :attr:`ttfab_slo_ms` by ``shared.capacity.w_max_for_slo`` at plan time. One derived
    field cannot contradict the promise; two independent ones always can.
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
    ttfab_slo_ms: int = Field(
        default=3000,
        gt=0,
        description=(
            "End-to-end p95 first-byte SLO: queue wait plus service, the whole promise "
            "to the client. W_max = SLO - S_p95 follows from it, so this is the single "
            "field that decides the queueing budget and the queue depth."
        ),
    )
    ttfab_budget_ms: int = Field(
        default=300,
        gt=0,
        description=(
            "Which measured budget to read the latency knee at. A *measurement* "
            "selector, not the SLO: `cmax` sweeps 300/500/1000/3000 in one ladder and "
            "this picks the column. Distinct from ttfab_slo_ms because the knee at a "
            "tight budget is the conservative number to size a fleet from even when "
            "the promise to the client is looser."
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

    c_max: float = Field(
        description=(
            "Per-instance concurrency the plan is built on: the lower of the latency "
            "knee at the chosen budget and the measured throughput ceiling."
        )
    )
    c_max_source: str = Field(
        default="latency_knee_only",
        description=(
            "Which measurement produced c_max: 'latency_knee', 'throughput_ceiling', or "
            "'latency_knee_only' when no ceiling was measured to compare against. Kept "
            "separate from binding_constraint rather than folded into it, because the "
            "two answer orthogonal questions and are both true at once: this one says "
            "what limits an instance, that one says what limits the target we track "
            "against that limit."
        ),
    )
    c_max_is_lower_bound: bool | None = Field(
        default=None,
        description=(
            "Whether the measurement c_max came from understates the instance's real "
            "capacity — the ladder never bracketed it, or the client throttled it. True "
            "means every fleet size here is an over-estimate, which is the safe "
            "direction but still wrong. None means the artifact did not record the "
            "question, which is not a pass."
        ),
    )
    c_target: float = Field(
        description=(
            "Per-instance concurrency for the scaling policy. Float, not int: "
            "Kokoro's C_max of 1 at k=2 gives 0.44, which an int cannot express."
        )
    )
    binding_constraint: str = Field(
        description="'surge_headroom' or 'slo_wait_budget' — the fix differs by which binds"
    )
    w_max_s: float = Field(
        default=0.0,
        ge=0,
        description=(
            "Queueing budget, derived as SLO - S_p95 rather than stated. Recorded on the "
            "plan because it is an output here, not an input: the report shows the "
            "arithmetic, and queue_max_depth below is this number times Lambda_cap. 0 "
            "means the model's own tail already misses the SLO — see the findings."
        ),
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
