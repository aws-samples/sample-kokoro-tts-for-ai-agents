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

import statistics
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

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
    """Observed against a live endpoint by ``qmax`` or ``ttotal``."""

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
    """The two measured variables, joined. Produced by ``qmax`` + ``ttotal``.

    Consumed by ``planner``. Serialized to an artifact JSON so a plan can be
    regenerated without re-running load.

    Only ``Q_max`` and ``T_total`` are measurements the model needs; ``S`` and
    ``chars_per_request`` come along because the SLO arithmetic and the cost arithmetic
    respectively need them, not because anything divides capacity by them.
    """

    model_config = ConfigDict(frozen=True)

    model_name: TTSModelName
    endpoint: str
    instance_type: str

    q_max: int = Field(
        gt=0,
        description=(
            "Highest per-instance concurrency (queued + executing) whose p95 first-byte "
            "time met slo_ms. Both scaling thresholds are fractions of this, so it is "
            "the one measured number the deployed policy is built on."
        ),
    )
    q_max_bracketed: bool = Field(
        default=True,
        description=(
            "Whether a rung above q_max was measured and missed the SLO. False makes "
            "q_max a lower bound: the thresholds derived from it then fire earlier than "
            "necessary, which is the safe direction but still costs instances."
        ),
    )
    slo_ms: int = Field(
        gt=0,
        description=(
            "The SLO q_max was measured against. Carried so `plan` can refuse a "
            "scenario asking for a different one: Q_max is defined by the SLO, and "
            "re-reading one ladder against another line is exactly the mistake that "
            "two independent latency fields used to permit."
        ),
    )
    ttfab_p95_at_c1_ms: float | None = Field(
        default=None,
        ge=0,
        description=(
            "p95 first-byte time at one outstanding request: service time with no queue "
            "in it. Not a capacity input — it is the FirstChunkLatencyP95 alarm's "
            "threshold, which watches an instance already serving and so cannot use the "
            "queue-inclusive SLO. None when the ladder had no N=1 rung."
        ),
    )
    s_mean_s: float = Field(gt=0, description="Mean service time, seconds")
    s_p95_s: float = Field(
        gt=0,
        description=(
            "p95 service time, seconds. Feeds W_max = SLO - S_p95, the queueing budget, "
            "and the 60s invocation-ceiling check. A tail, not a mean, because a "
            "mean-sized deadline is missed by half the requests that reach it."
        ),
    )
    t_total_s: float = Field(
        gt=0,
        description=(
            "Scaling lag: trigger through to an instance serving good traffic. The "
            "single number the whole plan is most sensitive to."
        ),
    )
    t_total_measured: bool = Field(
        default=True,
        description=(
            "Whether t_total_s came from a ttotal run or from --assume-t-total. "
            "Separate from `provenance.origin`, which describes this object as a "
            "whole: Q_max and S are measured even when the lag was stated, so the "
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
            "Q_max run. False means the number may be N x Q_max — see fixture.py."
        ),
    )
    unbounded_queue: bool | None = Field(
        default=None,
        description=(
            "Whether the container was verified to have no queue depth bound during "
            "the Q_max run. None means unchecked, which is not a pass."
        ),
    )
    instance_counts_observed: tuple[int, ...] = Field(
        default=(),
        description="Distinct instance counts seen across rungs. More than one invalidates Q_max.",
    )
    transport: str = Field(
        default="response-stream",
        description=(
            "Wire protocol q_max was measured on. Carried this far because the plan "
            "derived from it configures a real fleet: a Q_max measured on "
            "response-stream does not describe capacity for bidi traffic."
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
    q_max_spread: float = Field(
        default=0.0,
        ge=0,
        description=(
            "(max - min) / min of the per-run Q_max, carried from QMaxReport. The "
            "planner reports it because both thresholds are fractions of Q_max: a "
            "ladder that resolved noise produces confidently wrong thresholds."
        ),
    )
    runs_contributing: int = Field(
        default=1,
        ge=1,
        description=(
            "Ladder passes that produced a Q_max. The denominator for q_max_spread: a "
            "spread of 0% means agreement only when more than one run contributed."
        ),
    )
    ladder_p95_ms: dict[int, float] = Field(
        default_factory=dict,
        description=(
            "Concurrency -> median p95 first-byte time, the whole ladder. Carried past "
            "the two named scalars because ttotal's recovery test needs a *pair* of "
            "rungs — it declares a scale-out complete when a probe held at one "
            "concurrency drops to the p95 of another — and a plan artifact read back "
            "later should not have to re-run a 40-minute ladder to answer that."
        ),
    )
    cw_units_ratio_by_rung: dict[int, float] = Field(
        default_factory=dict,
        description=(
            "Concurrency -> CloudWatch ConcurrentRequestsPerModel/Maximum divided by the "
            "client's mean in-flight at that rung. What converts a measured occupancy "
            "into the units the deployed alarm compares against. Empty when the ladder "
            "ran without --cloudwatch, in which case the plan reports the conversion as "
            "unavailable rather than assuming 1:1 — see QMaxReport.cw_units_ratio_by_rung."
        ),
    )
    provenance: Provenance = Field(
        default_factory=lambda: Provenance(origin=Origin.MEASURED),
    )

    @model_validator(mode="after")
    def _check_measured(self) -> Measured:
        if self.s_p95_s < self.s_mean_s:
            raise ValueError(
                f"s_p95_s ({self.s_p95_s}) < s_mean_s ({self.s_mean_s}); percentiles disagree"
            )
        for rung, p95 in self.ladder_p95_ms.items():
            if rung <= 0:
                raise ValueError(f"ladder rung must be positive, got {rung}")
            if p95 < 0:
                raise ValueError(f"p95 at rung {rung} must be non-negative, got {p95}")
        for rung, ratio in self.cw_units_ratio_by_rung.items():
            if rung <= 0:
                raise ValueError(f"ratio rung must be positive, got {rung}")
            if ratio <= 0:
                raise ValueError(
                    f"cw units ratio at rung {rung} must be positive, got {ratio}; a "
                    "non-positive conversion would deploy a threshold no traffic satisfies"
                )
        return self

    @property
    def trustworthy(self) -> bool:
        """Whether this measurement is safe to build a per-instance plan on."""
        return self.frozen and len(set(self.instance_counts_observed)) <= 1


class StepSummary(BaseModel):
    """One ladder rung, flattened for the artifact.

    The in-process path uses dataclasses (``loadgen.WindowStats``); this is the
    serializable projection, carrying only what a later reader needs to judge
    whether the rung was a valid measurement.

    ``concurrency`` is the rung's ``N`` — set exactly by the closed-loop driver, not
    approached via an arrival rate. ``concurrency_mean`` is what the client *observed*
    in flight, and it should equal ``N``; the gap between them is
    :attr:`concurrency_shortfall`, and a large one means the client was the limit.
    """

    model_config = ConfigDict(frozen=True)

    run_index: int = Field(ge=0)
    step_index: int = Field(ge=0)
    concurrency: int = Field(gt=0, description="N: outstanding requests held for this rung")
    achieved_rps: float = Field(ge=0, description="Completions per second. An output, not a target")
    completed: int = Field(ge=0)
    ok: int = Field(ge=0)
    rejected: int = Field(
        default=0,
        ge=0,
        description=(
            "Completions the server refused (503/408/429). Counted separately from ok "
            "because a fast rejection is not a fast success: it lowers p95 while the "
            "endpoint is failing, which is what made a saturated step read as healthy."
        ),
    )
    chars: int = Field(
        default=0,
        ge=0,
        description=(
            "Characters synthesized by the completions in the window. A raw count, not "
            "a rate: the planner needs chars-per-request to price a fleet in $/M chars, "
            "and dividing a count by a count needs no units conversion."
        ),
    )
    outcome_counts: dict[str, int] = Field(default_factory=dict)

    ttfab_p50_ms: float | None = None
    ttfab_p95_ms: float | None = None
    ttfab_p99_ms: float | None = None
    latency_p95_ms: float | None = None
    s_mean_s: float | None = None
    s_p95_s: float | None = None
    concurrency_mean: float | None = Field(
        default=None, description="Client-observed mean in-flight. Should equal `concurrency`"
    )
    concurrency_peak: int | None = Field(
        default=None,
        description=(
            "Client-observed peak in-flight. Recorded beside the mean because the "
            "deployed alarm reads CloudWatch's *Maximum* statistic, and the two "
            "diverge by up to 9.8x — see server_concurrency_peak."
        ),
    )
    ttfab_drift_ms: float | None = Field(
        default=None,
        description=(
            "Fitted change in TTFAB across the window. The closed-loop replacement for "
            "an in-flight trend, which is pinned at N here and would always read settled."
        ),
    )

    meets_slo: bool = Field(
        description=(
            "Whether p95 TTFAB was inside the SLO on a rung that was a valid "
            "measurement. Q_max is the highest rung where this is true."
        )
    )
    saturated: bool
    settled: bool
    client_bound: bool = Field(
        default=False,
        description=(
            "Whether mean in-flight fell far enough below N that the client, not the "
            "server, was the limit. The closed-loop replacement for dispatch_skipped."
        ),
    )
    usable: bool = Field(
        description=(
            "Whether this rung may inform Q_max. False when the fleet resized, nothing "
            "completed, or the client was the limit."
        )
    )
    unusable_reason: str | None = None
    capacity_changed: bool = False
    instance_counts: tuple[int, ...] = ()

    # Joined from CloudWatch after a settle delay; absent when --no-cloudwatch.
    server_concurrency_mean: float | None = None
    server_concurrency_peak: float | None = Field(
        default=None,
        description=(
            "ConcurrentRequestsPerModel / *Maximum* over the window — the exact "
            "statistic the deployed target-tracking alarm reads. Carried because a "
            "threshold derived from client occupancy and compared against this one is "
            "the defect that shipped 0.713: Maximum divided by client mean ran from "
            "9.8x at low load to 1.35x at high load, so the ratio is not a constant "
            "the plan can correct for after the fact. High-resolution datapoints "
            "retain 3 hours, so this cannot be backfilled — hence --cloudwatch."
        ),
    )
    server_model_latency_p95_ms: float | None = None
    server_5xx_total: float | None = None
    gpu_utilization_mean: float | None = None
    cpu_utilization_mean: float | None = None
    concurrency_agreement: str | None = None

    @property
    def concurrency_shortfall(self) -> float | None:
        """How far observed mean in-flight fell below ``N``, as a fraction of ``N``."""
        if self.concurrency_mean is None or self.concurrency <= 0:
            return None
        return max(0.0, (self.concurrency - self.concurrency_mean) / self.concurrency)


class QMaxReport(BaseModel):
    """What one ``qmax`` invocation measured. The ``Q_max`` artifact.

    Deliberately *not* a :class:`Measured`: that requires ``t_total_s``, which only
    ``ttotal`` can supply. Keeping them separate means neither artifact claims a number
    it did not measure.

    ``q_max`` is a **rung**, not a fit: the highest ``N`` whose p95 first-byte time stayed
    inside :attr:`slo_ms`. That is the whole reason the ladder is closed-loop — ``N`` is
    the independent variable and is held exactly, so the answer is a concurrency that was
    actually run rather than one inferred through ``lambda = C/S``.

    The ladder's other rungs are not scaffolding to be discarded. Two downstream consumers
    read them, and neither can use the 3 s SLO: the ``FirstChunkLatencyP95`` alarm needs
    service time on a healthy instance (:attr:`ttfab_p95_at_c1_ms`), and ``ttotal``'s
    recovery test needs the p95 at two concurrencies to watch one halve into the other
    (:meth:`ttfab_p95_at`).
    """

    model_config = ConfigDict(frozen=True)

    model_name: TTSModelName
    endpoint: str
    instance_type: str
    run_id: str

    slo_ms: int = Field(
        gt=0,
        description=(
            "The p95 first-byte SLO this ladder was judged against, queue time included. "
            "Recorded on the artifact because Q_max is *defined by* it: a Q_max measured "
            "at 3000ms says nothing about a 1000ms promise, and `plan` refuses a mismatch "
            "rather than silently re-reading the ladder against a different line."
        ),
    )
    q_max: int = Field(
        gt=0,
        description=(
            "Highest concurrency whose p95 TTFAB met the SLO. Per-instance, because the "
            "ladder runs pinned at one instance — see `frozen`."
        ),
    )
    q_max_per_run: tuple[int, ...] = Field(
        default=(),
        description=(
            "Each run's own answer, in run order. `q_max` is the minimum of these rather "
            "than a median: the median of two rungs is a concurrency no run tested, while "
            "the minimum is both a real rung and the conservative one. Carried so the "
            "disagreement is visible instead of averaged away."
        ),
    )
    q_max_bracketed: bool = Field(
        description=(
            "Whether a rung above `q_max` was measured and actually missed the SLO. False "
            "means the ladder ran out while still passing, so `q_max` is a lower bound: "
            "both scaling thresholds derive from it, so an unbracketed value makes the "
            "policy scale out earlier than necessary rather than later."
        )
    )
    ttfab_p95_at_q_max_ms: float = Field(
        ge=0,
        description=(
            "p95 first-byte time at `q_max`. How much of the SLO was actually left over: "
            "a q_max that passed at 2900ms against a 3000ms SLO is on the edge, and one "
            "that passed at 400ms means the ladder stopped short."
        ),
    )

    s_mean_s: float = Field(
        gt=0,
        description=(
            "Mean service time from the *lowest* rung, seconds. The closed loop at N=1 is "
            "an uncontended probe by construction, which is what replaced the separate "
            "probe phase. Includes client-to-endpoint round trip (~34ms measured against "
            "kokoro), so it is service time as a client experiences it — which is the "
            "right quantity for an SLO derived from client-observed first byte, and a "
            "slight over-estimate anywhere it stands in for server-side work."
        ),
    )
    s_p95_s: float = Field(gt=0, description="p95 service time from the lowest rung, seconds")

    frozen: bool = Field(
        default=False,
        description=(
            "Whether autoscaling was suspended and capacity pinned for the run. False "
            "means `q_max` may be N x Q_max, with nothing else in the number saying so."
        ),
    )
    unbounded_queue: bool | None = Field(
        default=None,
        description=(
            "Whether the container was verified to have no queue depth bound. Q_max is "
            "the depth at which the SLO breaks, so a container that sheds first measures "
            "its own MAX_QUEUE_DEPTH instead. None means the check did not run, which is "
            "not the same as a pass."
        ),
    )
    instance_counts_observed: tuple[int, ...] = ()
    ladder_truncated_at: int | None = Field(
        default=None,
        description=(
            "Step index where the ladder stopped early after repeated saturation. "
            "Recorded so the ladder is never read as covering rungs that were skipped."
        ),
    )

    transport: str = Field(
        default="response-stream",
        description=(
            "Wire protocol the ladder ran on. Not decoration: the containers hold their "
            "inference lock differently per transport — kokoro holds it across an entire "
            "bidi session but per-generator on response-stream — so a Q_max from one does "
            "not transfer to the other."
        ),
    )
    deployed_config: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Fingerprint of the configuration the ladder ran against, read from the "
            "endpoint rather than from a registry. See fixture.DeployedConfig. This is "
            "what makes a rerun on a different instance type a new measurement rather "
            "than an overwrite of the old one."
        ),
    )

    runs: int = Field(default=1, ge=1)
    hold_s: float = Field(gt=0)
    measure_window_s: float = Field(gt=0)
    steps: list[StepSummary] = Field(default_factory=list)
    provenance: Provenance = Field(default_factory=lambda: Provenance(origin=Origin.MEASURED))

    @model_validator(mode="after")
    def _check_report(self) -> QMaxReport:
        if self.s_p95_s < self.s_mean_s:
            raise ValueError(
                f"s_p95_s ({self.s_p95_s}) < s_mean_s ({self.s_mean_s}); percentiles disagree"
            )
        for value in self.q_max_per_run:
            if value <= 0:
                raise ValueError(f"per-run q_max must be positive, got {value}")
        if self.q_max_per_run and self.q_max != min(self.q_max_per_run):
            raise ValueError(
                f"q_max ({self.q_max}) must be the minimum of q_max_per_run "
                f"({list(self.q_max_per_run)}): the cross-run answer is the conservative "
                "rung, and a q_max that is not one of the per-run answers is not a rung "
                "that was measured"
            )
        return self

    @property
    def trustworthy(self) -> bool:
        """Whether ``q_max`` is safe to read as *per-instance*.

        The freeze and a stable instance count are both required. ``unbounded_queue`` is
        not folded in: an unchecked queue bound is a separate warning with a separate fix,
        and collapsing them would make one re-run look like the other's.
        """
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
    def q_max_spread(self) -> float:
        """``(max - min) / min`` of the per-run answers.

        Relative to the *minimum* because that is the value the plan uses, so this reads
        directly as "how much capacity the other runs claimed on top of what we planned
        for". 0.0 from a single run means there was nothing to disagree with rather than
        agreement — check :attr:`runs_contributing` before reading it as repeatability.
        """
        if len(self.q_max_per_run) < 2:
            return 0.0
        low = min(self.q_max_per_run)
        return (max(self.q_max_per_run) - low) / low if low > 0 else 0.0

    @property
    def runs_contributing(self) -> int:
        """How many runs produced a ``Q_max``. The denominator for :attr:`q_max_spread`."""
        return len(self.q_max_per_run)

    @property
    def rungs(self) -> list[int]:
        """Concurrencies the ladder measured, ascending and deduplicated.

        Includes rungs that failed the SLO: what was *offered* is the question a caller
        asking "is c=10 on this ladder" needs answered, and truncation is reported
        separately by :attr:`ladder_truncated_at`.
        """
        return sorted({step.concurrency for step in self.steps})

    @property
    def ladder_p95_ms(self) -> dict[int, float]:
        """Concurrency -> median p95 TTFAB across runs, for rungs that measured one.

        The table ``ttotal`` reads to turn "the p95 halved" into a number. Median across
        runs, since with ``--runs 2`` one noisy pass should not move a threshold that
        decides when a scale-out is declared complete.
        """
        by_rung: dict[int, list[float]] = {}
        for step in self.steps:
            if step.ttfab_p95_ms is None or not step.usable:
                continue
            by_rung.setdefault(step.concurrency, []).append(step.ttfab_p95_ms)
        return {rung: statistics.median(values) for rung, values in sorted(by_rung.items())}

    @property
    def cw_units_ratio_by_rung(self) -> dict[int, float]:
        """Concurrency -> CloudWatch ``Maximum`` divided by the client's own mean in-flight.

        The conversion factor between what this tool measures and what the deployed alarm
        reads. ``ConcurrentRequestsPerModel`` / *Maximum* is a peak over a 10s period; the
        client's ``concurrency_mean`` is an average over a whole rung. Different
        quantities, and the ratio is **not** a constant — across one kokoro ladder it ran
        from 9.8x at the bottom to 1.35x at the top, because a lightly loaded endpoint's
        peak is many multiples of its average while a saturated one's is barely above it.

        So the planner interpolates on this table rather than on a single fitted number,
        and a threshold deployed without it is the 0.713 defect: a client occupancy
        compared against a server peak, satisfiable by no positive arrival rate.

        Only rungs where both figures are present and positive. Median across runs, and
        the ratio is taken per step before the median so a run whose server metrics went
        missing drops out of that rung rather than skewing it.
        """
        by_rung: dict[int, list[float]] = {}
        for step in self.steps:
            if not step.usable:
                continue
            client = step.concurrency_mean
            server = step.server_concurrency_peak
            if client is None or server is None or client <= 0 or server <= 0:
                continue
            by_rung.setdefault(step.concurrency, []).append(server / client)
        return {rung: statistics.median(values) for rung, values in sorted(by_rung.items())}

    def ttfab_p95_at(self, concurrency: int) -> float | None:
        """Median p95 TTFAB at exactly this concurrency, or ``None`` if not measured.

        Exact rather than nearest: ``ttotal`` compares a probe held at ``N`` against this
        table, and a nearest-rung fallback would silently compare a probe at 10 against a
        rung at 20. Callers wanting the comparison must ask for a rung that exists, and
        refuse when it does not.
        """
        return self.ladder_p95_ms.get(concurrency)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def ttfab_p95_at_c1_ms(self) -> float | None:
        """p95 first-byte time with one request outstanding.

        Service time on an unqueued instance, which is what the ``FirstChunkLatencyP95``
        alarm watches: that alarm fires on an instance already serving, where the request
        has spent none of its queue allowance, so an SLO-sized threshold there would only
        trip once the endpoint was ~10x past keeping up.

        ``None`` when the ladder had no ``N=1`` rung — an alarm threshold has to come from
        somewhere real, so the absence is reported rather than substituted for.

        A :func:`computed_field` rather than a bare property because the CDK stack reads it
        off the artifact JSON at synth time, and ``speech-infra`` cannot import this module
        (``tts-bench`` depends on it, not the other way round). Derived on the way out and
        ignored on the way in, so it cannot disagree with the ladder it summarizes.
        """
        return self.ttfab_p95_at(1)

    @property
    def chars_per_request(self) -> float:
        """Mean characters per request over the usable rungs.

        Derived from the raw counts rather than stored, so it cannot drift from the steps
        it summarizes. Only the planner needs it — to price a fleet in $/M chars, which
        requires knowing what a request *is* — and hard-coding a corpus average there
        would misprice any run against a different sample set.

        Returns 0.0 when no rung recorded both figures, which the planner surfaces as an
        infinite unit cost: visibly absent rather than plausibly wrong.
        """
        chars = sum(s.chars for s in self.steps if s.usable)
        completed = sum(s.completed for s in self.steps if s.usable)
        if chars <= 0 or completed <= 0:
            return 0.0
        return chars / completed

    @property
    def saturated_rungs(self) -> list[int]:
        """Rungs where the server refused work, ascending.

        Non-empty against a queue that was supposed to be unbounded means the run's
        preconditions were violated: something shed load, so the ladder measured that
        thing's threshold and not the concurrency at which the SLO breaks.
        """
        return sorted({s.concurrency for s in self.steps if s.saturated})

    @property
    def client_bound_rungs(self) -> list[int]:
        """Rungs where the client, not the server, was the limit, ascending."""
        return sorted({s.concurrency for s in self.steps if s.client_bound})

    def to_measured(
        self,
        *,
        t_total_s: float,
        t_total_provenance: Provenance | None = None,
    ) -> Measured:
        """Join this ``Q_max`` with a measured ``T_total`` into a planner input.

        The one place the two measured variables meet. ``frozen``,
        ``unbounded_queue`` and ``instance_counts_observed`` are carried through
        rather than defaulted, so a ladder run without a precondition stays
        identifiable as such after the join.

        Args:
            t_total_s: Scaling lag from ``ttotal``, seconds.
            t_total_provenance: Provenance of ``t_total_s``. Its ``note`` is
                appended to this run's, and its ``origin`` sets
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
            q_max=self.q_max,
            q_max_bracketed=self.q_max_bracketed,
            slo_ms=self.slo_ms,
            ttfab_p95_at_c1_ms=self.ttfab_p95_at_c1_ms,
            s_mean_s=self.s_mean_s,
            s_p95_s=self.s_p95_s,
            t_total_s=t_total_s,
            t_total_measured=t_total_provenance is None or t_total_provenance.is_measured,
            chars_per_request=self.chars_per_request,
            q_max_spread=self.q_max_spread,
            # `max(1, ...)`: the field is the denominator for the spread and must be a
            # count of passes, and a report with no per-run detail still had one run.
            runs_contributing=max(1, self.runs_contributing),
            ladder_p95_ms=dict(self.ladder_p95_ms),
            cw_units_ratio_by_rung=dict(self.cw_units_ratio_by_rung),
            frozen=self.frozen,
            unbounded_queue=self.unbounded_queue,
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
    """The two chosen variables plus the expected load. Supplied as CLI arguments.

    Everything here is an assumption. ``max_scaling_per_t_total`` in particular is *not*
    measured: it is the factor by which traffic might grow within one ``T_total``, and
    with no production history there is nothing to measure it from.

    **``W_max`` is deliberately absent.** It used to sit here beside a second latency
    field, and the two could disagree without anything noticing — which is how kokoro
    shipped a 20 s queue allowance under a 300 ms budget, a request taking 20.2 s to
    first byte while the config claimed 0.3 s. The SLO is the whole promise, queue
    included, so ``W_max`` is derived from :attr:`ttfab_slo_ms` by
    ``shared.capacity.w_max_for_slo`` at plan time. One derived field cannot contradict
    the promise; two independent ones always can.

    **So is any second latency field.** ``ttfab_slo_ms`` is the only one: it is the
    pass/fail line the ``Q_max`` ladder was judged against, and the planner refuses an
    artifact measured against a different value rather than re-reading that ladder
    against this one.
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

    max_scaling_per_t_total: float = Field(
        default=1.25,
        ge=1.0,
        lt=1.5,
        description=(
            "Surge ratio to survive within one T_total. 1.25 means traffic may grow 25% "
            "while a replacement instance is arriving. Both thresholds derive from it: "
            "with h = ratio - 1, C_scale_max = (1-h) x Q_max and C_scale_min = "
            "(1-2h) x Q_max. Capped below 1.5 because C_scale_min reaches zero there, "
            "and a non-positive scale-in threshold deploys as 'never scale in' — see "
            "shared.capacity.scale_thresholds."
        ),
    )
    ttfab_slo_ms: int = Field(
        default=3000,
        gt=0,
        description=(
            "End-to-end p95 first-byte SLO: queue wait plus service, the whole promise "
            "to the client. The one latency input. Q_max is measured against it, W_max = "
            "SLO - S_p95 follows from it, and `plan` refuses a Q_max artifact measured "
            "against a different value."
        ),
    )
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

    The six variables appear here in one place: ``slo_ms`` and
    ``max_scaling_per_t_total`` chosen (on the scenario), ``q_max`` and ``t_total_s``
    measured (on the measurement), ``c_scale_max`` and ``c_scale_min`` derived.
    """

    model_config = ConfigDict(frozen=True)

    model_name: TTSModelName
    endpoint: str
    instance_type: str

    q_max: int = Field(
        gt=0,
        description=(
            "Measured per-instance concurrency the plan is built on: the highest that held the SLO."
        ),
    )
    q_max_is_lower_bound: bool = Field(
        default=False,
        description=(
            "Whether the ladder ran out while still passing, so real capacity is at "
            "least this. True means both thresholds are conservative — the policy adds "
            "instances sooner than it needs to, which costs money rather than SLO."
        ),
    )
    c_scale_max: float = Field(
        gt=0,
        description=(
            "Scale out at or above this concurrency. (1-h) x Q_max, h = "
            "max_scaling_per_t_total - 1. Float because Q_max x 0.75 usually is."
        ),
    )
    c_scale_min: float = Field(
        ge=0,
        description=(
            "Scale in at or below this concurrency. (1-2h) x Q_max — the point where "
            "there is a full surge of excess headroom."
        ),
    )
    c_scale_max_in_cw_units: float | None = Field(
        default=None,
        description=(
            "c_scale_max converted to ConcurrentRequestsPerModel / *Maximum*, the "
            "statistic the deployed alarm actually reads, using the ratio measured on "
            "the same ladder. The threshold that ships is this number, not the client "
            "occupancy beside it: deploying the client figure is the defect that put "
            "0.713 on the endpoint, a value no positive arrival rate satisfies. None "
            "when the ladder ran without --cloudwatch, in which case the plan says the "
            "conversion is unavailable rather than assuming it is 1:1."
        ),
    )
    cw_units_ratio: float | None = Field(
        default=None,
        gt=0,
        description=(
            "server Maximum / client mean at the rung nearest c_scale_max. Recorded "
            "beside the converted threshold so a reader can see the size of the "
            "correction — it ran 1.35x to 9.8x across one kokoro ladder, so it is a "
            "measurement per configuration and not a constant."
        ),
    )
    min_safe_instances: int | None = Field(
        default=None,
        description=(
            "Smallest fleet where removing one instance does not push the survivors "
            "back over c_scale_max. Below it scale-in flaps; None means no fleet size "
            "is safe at this surge ratio. See the scale_in_safety finding."
        ),
    )
    w_max_s: float = Field(
        default=0.0,
        ge=0,
        description=(
            "Queueing budget, derived as SLO - S_p95 rather than stated. Recorded on the "
            "plan because it is an output here, not an input: the report shows the "
            "arithmetic. 0 means the model's own tail already misses the SLO — see the "
            "findings."
        ),
    )
    ceiling_s: float = Field(
        gt=0,
        description=(
            "The invocation ceiling W_max + S_p95 was judged against, seconds. Stored "
            "rather than re-read from SAGEMAKER_INVOCATION_CEILING_S because that "
            "constant is only the default: `--ceiling-s` moves it, and the config block "
            "is pasted into config.py verbatim, so its ttfab_slo_ms comment has to name "
            "the ceiling this plan was actually judged against. Naming 60s on a plan "
            "checked against something else contradicts the invocation_ceiling finding "
            "printed directly above it, which is how a policy gets deployed against a "
            "limit nobody tested. Required, not defaulted to the constant, so a plan "
            "that failed to record its ceiling cannot render as one judged at 60s."
        ),
    )

    min_instances: int = Field(ge=1)
    max_instances: int = Field(ge=1)
    peak_instances: int = Field(ge=0)
    trough_instances: int = Field(ge=0)

    queue_max_depth: int = Field(
        ge=0,
        description=(
            "Per-instance admission bound, set to Q_max: past it a request cannot reach "
            "first byte inside the SLO, so admitting it produces a late success instead "
            "of an honest rejection."
        ),
    )
    scale_out_cooldown_s: int = Field(ge=0)
    scale_in_cooldown_s: int = Field(ge=0)

    utilization_at_c_scale_max: float = Field(
        default=0.0,
        ge=0,
        description=(
            "Single-server utilization implied by holding c_scale_max in the queue, "
            "L/(1+L). Recorded because the relationship is steeply non-linear and that "
            "is not visible from the threshold itself: 0.75 x Q_max is not three "
            "quarters of the way to trouble but 97% utilized. Feeds surge_survival."
        ),
    )
    shed_probability_at_c_scale_max: float | None = Field(
        default=None,
        ge=0,
        le=1,
        description=(
            "Simulated P(queue reaches Q_max) while waiting out one T_total from "
            "c_scale_max. The falsifiable form of 'is this threshold early enough'. None "
            "when it could not be simulated."
        ),
    )

    peak_cost_per_hour: float
    peak_cost_per_m_chars: float

    findings: list[Finding] = Field(default_factory=list)
    measured: Measured
    scenario: Scenario

    @property
    def infeasible(self) -> bool:
        return any(f.verdict is Verdict.INFEASIBLE for f in self.findings)

    @property
    def warnings(self) -> list[Finding]:
        return [f for f in self.findings if f.verdict is Verdict.WARN]
