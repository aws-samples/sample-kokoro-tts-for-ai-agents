"""Step-and-hold ``C_max`` harness: per-instance concurrency at the latency knee.

Four decisions separate this from :func:`tts_bench.cost.find_saturation_concurrency`,
which returns 4 for a model whose real capacity is 1:

**The knee is an SLO crossing, not a throughput plateau.** Throughput plateaus
long after latency has become unacceptable — on a serialized model it plateaus
immediately, which is why the legacy plateau break trips on the first comparison
and never tests the interesting rates. Here the knee is the highest rate whose
p95 TTFAB still meets a budget, and it is reported for every budget in
:data:`DEFAULT_TTFAB_BUDGETS` from a single run: four numbers from one ladder
turns choosing an SLO into a table lookup rather than a re-run.

**Step and hold, never ramp.** Each rate is held for ``hold_s`` and only the
trailing ``measure_window_s`` is measured. A continuous ramp smears the knee
across rates, because the queue built at rate *n* is still draining at *n+1*.

**The ladder is in expected concurrency, fine at the bottom.** A geometric
``[2,4,8,16]`` ladder cannot resolve 1 from 2, and Kokoro's ``C_max`` is ~1. The
default ladder walks ``lambda x S`` of ``0.5, 1, 1.5, 2, 3, 4, 6, 8, 12, 16``, so
the bottom is resolved to half a request in flight.

**Saturation is an outcome, not an error.** A step whose completions fall below
95% of offered, or whose in-flight count is still trending upward at the end of
the window, cannot host a knee whatever its latency percentiles read — a step
that never settles *is* the measurement.

The whole run happens inside :class:`tts_bench.fixture.frozen`, so the fleet
cannot grow mid-measurement; see that module for why this is enforced rather
than advised.
"""

from __future__ import annotations

import statistics
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
from botocore.client import BaseClient
from loguru import logger

from tts_bench import observe
from tts_bench.bidi import Transport, invoke_for, make_client_for
from tts_bench.invoke import InvokeResult, invoke_stream, resolve_endpoint, resolve_voice
from tts_bench.loadgen import (
    SYSTEM_CLOCK,
    ArrivalProcess,
    Clock,
    LoadEvent,
    StepResult,
    WindowStats,
    build_text_pool,
    make_instance_count_fetcher,
    run_step,
    summarize_window,
)
from tts_bench.types import CMaxReport, KneePoint, Origin, Provenance, StepSummary
from tts_inference.types import TTSModelName

#: Ladder in *expected* concurrency (``lambda x S``) rather than in raw rate: a
#: rate ladder means something different for a 40ms model than for a 4s one,
#: while a concurrency ladder means the same thing for both. Fine at the bottom
#: because ``C_max`` can legitimately be 1.
DEFAULT_TARGET_CONCURRENCIES: tuple[float, ...] = (
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    8.0,
    12.0,
    16.0,
)

#: Budgets reported from every run. 300ms is the plan's default SLO; the others
#: bracket it, so how sensitive the answer is to the SLO is visible at a glance.
DEFAULT_TTFAB_BUDGETS: tuple[int, ...] = (50, 150, 300, 500)

DEFAULT_HOLD_S = 240.0
DEFAULT_MEASURE_WINDOW_S = 60.0

#: Idle seconds between steps so the previous step's queue drains. Without it a
#: step inherits backlog, and its knee then lands low for the wrong reason.
DEFAULT_SETTLE_BETWEEN_STEPS_S = 30.0

#: Requests used to estimate ``S`` before the ladder starts. Sequential and
#: uncontended: the point is a service time with no queueing in it.
DEFAULT_PROBE_REQUESTS = 5

#: Stop the ladder after this many consecutive saturated steps. Past the knee,
#: higher rates cost money and measure only how badly the endpoint fails.
DEFAULT_SATURATED_STEPS_TO_STOP = 2

#: Worker and connection headroom over the highest expected in-flight count.
#: Without it the client's pool saturates first and the measurement describes
#: the benchmark rather than the model.
WORKER_HEADROOM = 4.0

#: Run-to-run spread above which the ladder resolved noise rather than a knee.
#: The 0.875 derate is sized for ordinary variance, not for this.
SPREAD_WARN_THRESHOLD = 0.2


class CMaxError(RuntimeError):
    """The run could not produce a usable curve."""


def rps_for_concurrency(target_concurrency: float, s_mean_s: float) -> float:
    """Arrival rate that puts ``target_concurrency`` in flight, by Little's Law.

    ``L = lambda x W``, so ``lambda = L / S``. Uses the *uncontended* ``S``: once
    queueing starts, ``W`` exceeds ``S`` and achieved concurrency overshoots the
    target — which is exactly the signal the ladder is looking for, and why the
    knee is read from measured concurrency rather than from this figure.

    Raises:
        ValueError: If either argument is non-positive.
    """
    if target_concurrency <= 0:
        raise ValueError(f"target_concurrency must be positive, got {target_concurrency}")
    if s_mean_s <= 0:
        raise ValueError(f"s_mean_s must be positive, got {s_mean_s}")
    return target_concurrency / s_mean_s


def worker_count(max_target_concurrency: float) -> int:
    """Thread-pool and connection-pool size for the ladder's highest step.

    Sized above the *target* in-flight count, which is only the same as the peak
    in-flight count while the endpoint is keeping up. Once a step is past
    capacity, residence time grows without bound and real in-flight concurrency
    overshoots the target by however much the server is behind — measured at 30
    against a target of 2 on kokoro's bidi transport, where the inference lock is
    held for the whole session. No fixed multiple of the target can cover that,
    because the overshoot is a property of the server's backlog, not of the
    schedule.

    So this bounds the pool for the steps that can still *be* a knee, and past
    the knee ``dispatch_skipped`` is expected rather than a defect: the step is
    marked unusable (:func:`_unusable_reason`) and still brackets the knee from
    above (:func:`_brackets_from_above`). What must never happen is a *passing*
    step that was silently client-limited, and that is what the headroom buys.
    """
    return max(2, int((max_target_concurrency + WORKER_HEADROOM) * 2))


@dataclass(frozen=True, slots=True)
class ProbeResult:
    """Uncontended service time, measured one request at a time."""

    s_mean_s: float
    s_p95_s: float
    samples: int
    failures: int

    @property
    def usable(self) -> bool:
        return self.samples > 0 and self.s_mean_s > 0


def probe_service_time(
    # `Any`, not `BaseClient`: the bidi transport's client is a
    # `SageMakerRuntimeHTTP2Client` with no botocore ancestry. It is only ever
    # handed to `invoke`, never called directly.
    client: Any,
    *,
    endpoint: str,
    voice: str,
    texts: Sequence[str],
    requests: int = DEFAULT_PROBE_REQUESTS,
    invoke: Callable[..., InvokeResult] = invoke_stream,
) -> ProbeResult:
    """Measure ``S`` with one request in flight at a time.

    The ladder needs ``S`` *before* it can turn a concurrency target into a rate,
    and it has to be the uncontended value: an ``S`` that already contains
    queueing would compress the whole ladder toward low rates and put the knee
    below where it really is.

    Raises:
        CMaxError: If no probe request succeeded. A ladder built on a guessed
            ``S`` would be arbitrary, so this fails rather than defaulting.
    """
    latencies: list[float] = []
    failures = 0
    for i in range(requests):
        result = invoke(client, endpoint, texts[i % len(texts)], voice)
        if result.ok:
            latencies.append(result.latency_ms / 1000.0)
        else:
            failures += 1
            logger.warning(
                "Probe {}/{} failed: {} {}",
                i + 1,
                requests,
                result.outcome,
                result.error_message or "",
            )

    if not latencies:
        raise CMaxError(
            f"all {requests} probe request(s) to {endpoint} failed; the ladder cannot be "
            "sized without a service time. Check the endpoint is InService and healthy."
        )

    return ProbeResult(
        s_mean_s=statistics.fmean(latencies),
        s_p95_s=float(np.percentile(latencies, 95)),
        samples=len(latencies),
        failures=failures,
    )


def _unusable_reason(stats: WindowStats, result: StepResult) -> str | None:
    """Why this step may not inform the knee, or ``None`` if it may."""
    if stats.capacity_changed:
        return (
            f"instance count changed mid-step {result.instance_counts}; achieved throughput "
            "rose for a reason unrelated to the knee"
        )
    if stats.completed == 0:
        return "no requests completed inside the measure window"
    if result.dispatch_skipped:
        # The client, not the server, was the limit here. A knee measured under
        # that condition belongs to the thread pool, and it looks exactly like a
        # real one, so the step is excluded rather than merely flagged.
        return (
            f"{result.dispatch_skipped}/{result.scheduled_count} dispatches skipped; the "
            "client ran out of workers, so this step measures the benchmark"
        )
    return None


def summarize_step(
    result: StepResult,
    stats: WindowStats,
    *,
    run_index: int,
    target_concurrency: float,
) -> StepSummary:
    """Project one measured step into its serializable artifact form."""
    reason = _unusable_reason(stats, result)
    return StepSummary(
        run_index=run_index,
        step_index=result.step_index,
        target_concurrency=target_concurrency,
        offered_rps=result.offered_rps,
        achieved_rps=stats.achieved_rps,
        completed=stats.completed,
        ok=stats.ok,
        outcome_counts=dict(stats.outcome_counts),
        ttfab_p50_ms=stats.ttfab_p50_ms,
        ttfab_p95_ms=stats.ttfab_p95_ms,
        ttfab_p99_ms=stats.ttfab_p99_ms,
        latency_p95_ms=stats.latency_p95_ms,
        s_mean_s=stats.s_mean_s,
        s_p95_s=stats.s_p95_s,
        concurrency_mean=stats.concurrency_mean,
        concurrency_p95=stats.concurrency_p95,
        concurrency_slope_per_s=stats.concurrency_slope_per_s,
        chars_per_hour=stats.chars_per_hour,
        saturated=stats.saturated,
        settled=stats.settled,
        usable=reason is None,
        unusable_reason=reason,
        dispatch_skipped=result.dispatch_skipped,
        capacity_changed=stats.capacity_changed,
        instance_counts=result.instance_counts,
    )


def _meets_budget(step: StepSummary, ttfab_budget_ms: int) -> bool:
    """Whether a step met the latency budget *and* was a valid measurement."""
    return (
        step.ttfab_p95_ms is not None
        and step.ttfab_p95_ms <= ttfab_budget_ms
        and not step.saturated
        and step.settled
    )


def _brackets_from_above(step: StepSummary, ttfab_budget_ms: int) -> bool:
    """Whether this step is evidence that the budget is unmeetable above the knee.

    Weaker than "could be the knee" on purpose. A step excluded by
    :func:`_unusable_reason` cannot *be* a knee, but it can still bound one, and
    dropping it entirely is what made a ladder that hit the wall report its knee
    as a lower bound.

    An unusable step is judged on its measured p95 alone. ``saturated`` compares
    achieved against *offered* rps, and a skipped dispatch lowers achieved
    without the server ever seeing the request — so a client-limited step reads
    as saturated even when the endpoint kept up fine. TTFAB percentiles have no
    such problem: they are computed only over requests that really were
    dispatched, so a p95 past the budget there is a fact about the endpoint.
    """
    if step.usable:
        return not _meets_budget(step, ttfab_budget_ms)
    return step.ttfab_p95_ms is not None and step.ttfab_p95_ms > ttfab_budget_ms


def find_knee(steps: Sequence[StepSummary], ttfab_budget_ms: int) -> KneePoint | None:
    """Highest usable step meeting a p95 TTFAB budget without saturating.

    Takes the *highest* passing step rather than returning at the first failure,
    so one noisy step cannot truncate the curve. The result records whether a
    step above the knee actually failed the budget: if the ladder simply ran out
    while still passing, the knee is a lower bound, and planning on it would
    understate the capacity needed.

    Returns:
        The knee, or ``None`` if no usable step met the budget.
    """
    ordered = sorted(steps, key=lambda s: (s.target_concurrency, s.offered_rps))
    passing = [s for s in ordered if s.usable and _meets_budget(s, ttfab_budget_ms)]
    if not passing:
        return None

    best = passing[-1]
    # Bracketed relative to the knee we chose, not to any earlier pass: on a
    # pass/fail/pass ladder, accumulating the flag while walking upward would
    # claim a bracket that actually sits *below* the reported knee.
    #
    # Unusable steps bracket but cannot *be* the knee — see _brackets_from_above.
    bracketed = any(
        s.target_concurrency > best.target_concurrency and _brackets_from_above(s, ttfab_budget_ms)
        for s in ordered
    )

    # Prefer measured concurrency. The fallback only fires when the 1Hz monitor
    # produced no samples at all, which would otherwise discard a good step.
    concurrency = best.concurrency_mean
    if concurrency is None or concurrency <= 0:
        concurrency = best.target_concurrency
        logger.warning(
            "Step {} has no concurrency samples; falling back to the target {:.2f}",
            best.step_index,
            concurrency,
        )

    return KneePoint(
        ttfab_budget_ms=ttfab_budget_ms,
        concurrency=concurrency,
        offered_rps=best.offered_rps,
        p95_ttfab_ms=best.ttfab_p95_ms or 0.0,
        step_index=best.step_index,
        bracketed=bracketed,
    )


def median_curve(
    per_run_knees: Sequence[Sequence[KneePoint]],
    budgets: Sequence[int],
) -> tuple[dict[int, float], dict[int, float]]:
    """Median concurrency per budget across runs, plus relative spread.

    Median rather than mean: with ``runs=3`` one bad run should not move the
    answer, and the derate is not sized to absorb an outlier. Spread is
    ``(max - min) / median``, which is what the caller warns on — a wide spread
    means the ladder resolved noise, and no derate fixes that.

    Returns:
        ``(curve, spread)``, both keyed by budget. Budgets no run found are
        absent from both rather than present as zero.
    """
    curve: dict[int, float] = {}
    spread: dict[int, float] = {}
    for budget in budgets:
        values = [
            knee.concurrency
            for knees in per_run_knees
            for knee in knees
            if knee.ttfab_budget_ms == budget
        ]
        if not values:
            continue
        median = statistics.median(values)
        curve[budget] = median
        spread[budget] = ((max(values) - min(values)) / median) if median > 0 else 0.0
    return curve, spread


def uncontended_service_time(
    steps: Sequence[StepSummary],
    fallback: ProbeResult,
) -> tuple[float, float]:
    """``(s_mean_s, s_p95_s)`` from the lowest usable, unsaturated step.

    Deliberately not from the knee step: service time at the knee already
    contains queueing, and ``C_slo_cap = W_max / S`` would then be bounding a
    wait it had already counted, understating how much concurrency the SLO
    permits.

    Falls back to the probe when no step qualifies. The probe is uncontended by
    construction, so that is the more conservative source, not a worse one.
    """
    candidates = sorted(
        (
            s
            for s in steps
            if s.usable and not s.saturated and s.s_mean_s is not None and s.s_mean_s > 0
        ),
        key=lambda s: s.target_concurrency,
    )
    if not candidates:
        return fallback.s_mean_s, max(fallback.s_p95_s, fallback.s_mean_s)

    lowest = candidates[0]
    s_mean = lowest.s_mean_s or fallback.s_mean_s
    s_p95 = lowest.s_p95_s or s_mean
    return s_mean, max(s_p95, s_mean)


@dataclass(slots=True)
class LadderRun:
    """One pass up the ladder."""

    run_index: int
    steps: list[StepSummary] = field(default_factory=list)
    results: list[StepResult] = field(default_factory=list)
    truncated_at: int | None = None

    @property
    def instance_counts(self) -> tuple[int, ...]:
        """Distinct instance counts seen in this run, in first-seen order."""
        seen: list[int] = []
        for result in self.results:
            for count in result.instance_counts:
                if count not in seen:
                    seen.append(count)
        return tuple(seen)


def run_ladder(
    client: Any,  # See probe_service_time: bidi's client is not a BaseClient.
    *,
    model: str,
    endpoint: str,
    voice: str,
    texts: Sequence[str],
    s_mean_s: float,
    target_concurrencies: Sequence[float] = DEFAULT_TARGET_CONCURRENCIES,
    hold_s: float = DEFAULT_HOLD_S,
    measure_window_s: float = DEFAULT_MEASURE_WINDOW_S,
    settle_between_steps_s: float = DEFAULT_SETTLE_BETWEEN_STEPS_S,
    run_index: int = 0,
    run_id: str | None = None,
    arrival: ArrivalProcess | str = ArrivalProcess.POISSON,
    seed: int | None = None,
    saturated_steps_to_stop: int = DEFAULT_SATURATED_STEPS_TO_STOP,
    instance_count_fetch: Callable[[], int] | None = None,
    event_sink: Callable[[LoadEvent], None] | None = None,
    clock: Clock = SYSTEM_CLOCK,
    step_runner: Callable[..., StepResult] = run_step,
    invoke: Callable[..., InvokeResult] = invoke_stream,
) -> LadderRun:
    """Walk the ladder once, measuring only each step's trailing window.

    Stops early after ``saturated_steps_to_stop`` consecutive saturated steps:
    past the knee, higher rates cost money and measure only the shape of the
    failure. Where it stopped is recorded, so the curve is never read as covering
    rates that were never offered.

    Args:
        s_mean_s: Uncontended service time, used to convert each concurrency
            target into an arrival rate.
        measure_window_s: Trailing part of each step that is measured. The rest
            is warm-up and is discarded.
        step_runner: Injected for tests; defaults to :func:`loadgen.run_step`.
        invoke: Transport for each request. Must be the same one the probe used,
            or ``S`` and the ladder describe different wire protocols and every
            rate on the ladder is wrong.

    Raises:
        ValueError: If the measure window does not fit inside the hold.
    """
    if measure_window_s > hold_s:
        raise ValueError(
            f"measure_window_s ({measure_window_s}) must not exceed hold_s ({hold_s}): the "
            "window is the trailing part of the step, not an addition to it"
        )

    run_id = run_id or uuid.uuid4().hex[:12]
    ladder = LadderRun(run_index=run_index)
    ordered_targets = sorted(target_concurrencies)
    workers = worker_count(max(ordered_targets))
    consecutive_saturated = 0

    for step_index, target in enumerate(ordered_targets):
        offered_rps = rps_for_concurrency(target, s_mean_s)
        logger.info(
            "Run {} step {}: target concurrency {:.2f} -> {:.2f} rps for {:.0f}s "
            "(measuring the last {:.0f}s)",
            run_index,
            step_index,
            target,
            offered_rps,
            hold_s,
            measure_window_s,
        )

        result = step_runner(
            client,
            model=model,
            endpoint=endpoint,
            voice=voice,
            texts=texts,
            offered_rps=offered_rps,
            duration_s=hold_s,
            max_workers=workers,
            step_index=step_index,
            run_id=run_id,
            arrival=arrival,
            # Vary the seed per step and per run so each is individually
            # reproducible without every step drawing the same arrival pattern.
            seed=None if seed is None else seed + step_index + 1000 * run_index,
            instance_count_fetch=instance_count_fetch,
            event_sink=event_sink,
            clock=clock,
            invoke=invoke,
        )

        stats = summarize_window(
            result,
            start_ts=result.ended_ts - measure_window_s,
            end_ts=result.ended_ts,
        )
        summary = summarize_step(
            result,
            stats,
            run_index=run_index,
            target_concurrency=target,
        )
        ladder.steps.append(summary)
        ladder.results.append(result)

        logger.info(
            "Run {} step {}: achieved {:.2f} rps, concurrency {}, p95 TTFAB {} "
            "(saturated={} settled={} usable={})",
            run_index,
            step_index,
            stats.achieved_rps,
            f"{stats.concurrency_mean:.2f}" if stats.concurrency_mean is not None else "n/a",
            f"{stats.ttfab_p95_ms:.0f}ms" if stats.ttfab_p95_ms is not None else "n/a",
            stats.saturated,
            stats.settled,
            summary.usable,
        )
        if summary.unusable_reason:
            logger.warning("Run {} step {}: {}", run_index, step_index, summary.unusable_reason)

        consecutive_saturated = consecutive_saturated + 1 if stats.saturated else 0
        if consecutive_saturated >= saturated_steps_to_stop:
            logger.warning(
                "Run {}: {} consecutive saturated steps; stopping the ladder at step {} "
                "rather than paying for rates past the knee",
                run_index,
                consecutive_saturated,
                step_index,
            )
            ladder.truncated_at = step_index
            break

        if settle_between_steps_s > 0 and step_index < len(ordered_targets) - 1:
            logger.info("Draining {:.0f}s before the next step", settle_between_steps_s)
            clock.sleep(settle_between_steps_s)

    return ladder


def join_cloudwatch(
    cloudwatch: BaseClient,
    *,
    endpoint: str,
    variant: str,
    steps: Sequence[StepSummary],
    results: Sequence[StepResult],
    measure_window_s: float,
    settle_delay_s: float = observe.DEFAULT_SETTLE_DELAY_S,
    period_s: int = observe.HIGH_RES_PERIOD_S,
    sleep: Callable[[float], None] | None = None,
    now: Callable[[], datetime] | None = None,
) -> list[StepSummary]:
    """Attach server-side metrics to each step. Never raises.

    Server-side data answers two questions the client cannot. Whether the load
    reached the endpoint at all — :func:`observe.concurrency_agreement` catches a
    dispatcher or connection-pool bottleneck, which from the client side looks
    identical to server saturation. And what actually saturated: a GPU-bound knee
    and a lock-bound knee are indistinguishable from latency alone but call for
    opposite fixes.

    The settle wait is taken once, against the **newest** window. Waiting on the
    oldest instead returns immediately — its window is already older than the
    delay — and the newest step would then be read before CloudWatch had
    aggregated it, which surfaces as missing data rather than as a skipped wait.

    A failure here degrades the report but must never lose a measured curve, so
    every fetch is caught and the unjoined summary is kept.
    """
    if not results:
        return list(steps)

    def _window(result: StepResult, *, settle: bool) -> observe.WindowMetrics:
        end = datetime.fromtimestamp(result.ended_ts, tz=UTC)
        start = end - timedelta(seconds=measure_window_s)
        if settle:
            return observe.settle_and_fetch_window(
                cloudwatch,
                endpoint=endpoint,
                variant=variant,
                start=start,
                end=end,
                period_s=period_s,
                settle_delay_s=settle_delay_s,
                sleep=sleep,
                now=now,
            )
        return observe.fetch_window(
            cloudwatch,
            endpoint=endpoint,
            variant=variant,
            start=start,
            end=end,
            period_s=period_s,
        )

    newest = max(results, key=lambda r: r.ended_ts)
    windows: dict[int, observe.WindowMetrics] = {}
    for result in sorted(results, key=lambda r: r.ended_ts, reverse=True):
        try:
            windows[result.step_index] = _window(result, settle=result is newest)
        except Exception as exc:  # noqa: BLE001 - telemetry must not lose a measured curve
            logger.warning(
                "Could not join CloudWatch for step {}: {}. Client-side numbers stand; the "
                "server cross-check is missing.",
                result.step_index,
                exc,
            )

    joined: list[StepSummary] = []
    for summary in steps:
        window = windows.get(summary.step_index)
        if window is None:
            joined.append(summary)
            continue

        agreement = None
        if summary.concurrency_mean is not None:
            agreement = observe.concurrency_agreement(summary.concurrency_mean, window).diagnosis
        cpu = window.get("CPUUtilization")

        joined.append(
            summary.model_copy(
                update={
                    "server_concurrency_mean": window.concurrency_mean,
                    "server_model_latency_p95_ms": window.model_latency_p95_ms,
                    "server_5xx_total": window.error_5xx_total,
                    "gpu_utilization_mean": window.gpu_utilization_mean,
                    "cpu_utilization_mean": cpu.mean() if cpu is not None else None,
                    "concurrency_agreement": agreement,
                }
            )
        )
    return joined


def build_report(
    *,
    model: str | TTSModelName,
    endpoint: str,
    instance_type: str,
    run_id: str,
    ladders: Sequence[LadderRun],
    probe: ProbeResult,
    budgets: Sequence[int] = DEFAULT_TTFAB_BUDGETS,
    hold_s: float = DEFAULT_HOLD_S,
    measure_window_s: float = DEFAULT_MEASURE_WINDOW_S,
    derate: float = 0.875,
    frozen: bool = False,
    arrival: str = str(ArrivalProcess.POISSON),
    seed: int | None = None,
    transport: Transport | str = Transport.RESPONSE_STREAM,
    joined_steps: Sequence[StepSummary] | None = None,
    measured_at: str | None = None,
) -> CMaxReport:
    """Assemble the artifact from one or more ladder runs.

    The curve is the per-budget median across runs; ``knees`` carries the last
    run's detail so a reader can see which step each budget landed on. ``derate``
    is recorded but **not** applied — :func:`shared.capacity.c_target` applies it
    once, and doing it here as well would shrink every planned fleet twice.

    Raises:
        CMaxError: If no step met any budget. That is a real finding — the lowest
            rate is already past capacity, or the endpoint is unhealthy — but it
            is not a curve, so it fails rather than serializing an empty one.
    """
    per_run_knees = [
        [knee for budget in budgets if (knee := find_knee(ladder.steps, budget)) is not None]
        for ladder in ladders
    ]
    curve, spread = median_curve(per_run_knees, budgets)

    if not curve:
        raise CMaxError(
            "no ladder step met any TTFAB budget without saturating. Either the endpoint is "
            "unhealthy, or the lowest ladder rate is already past its capacity — re-run with "
            "a lower --target-concurrency, or raise --ttfab-budgets if the SLO allows it."
        )

    if joined_steps is not None:
        all_steps = list(joined_steps)
    else:
        all_steps = [step for ladder in ladders for step in ladder.steps]
    s_mean, s_p95 = uncontended_service_time(all_steps, probe)

    observed: list[int] = []
    for ladder in ladders:
        for count in ladder.instance_counts:
            if count not in observed:
                observed.append(count)

    truncated = next(
        (ladder.truncated_at for ladder in ladders if ladder.truncated_at is not None),
        None,
    )

    report = CMaxReport(
        model_name=TTSModelName(model),
        endpoint=endpoint,
        instance_type=instance_type,
        run_id=run_id,
        c_max_curve=curve,
        knees=list(per_run_knees[-1]) if per_run_knees else [],
        curve_spread=spread,
        s_mean_s=s_mean,
        s_p95_s=s_p95,
        derate=derate,
        frozen=frozen,
        instance_counts_observed=tuple(observed),
        ladder_truncated_at=truncated,
        runs=len(ladders),
        hold_s=hold_s,
        measure_window_s=measure_window_s,
        arrival_process=arrival,
        seed=seed,
        transport=str(Transport(transport)),
        steps=all_steps,
        provenance=Provenance(
            origin=Origin.MEASURED,
            run_id=run_id,
            measured_at=measured_at,
            endpoint=endpoint,
            note=(
                f"C_max at the p95 TTFAB knee on {Transport(transport)}, autoscaling frozen"
                if frozen
                else f"C_max measured on {Transport(transport)} WITHOUT the autoscaling "
                "freeze; may be N x C_max"
            ),
        ),
    )

    for budget, value in sorted(spread.items()):
        if value > SPREAD_WARN_THRESHOLD:
            logger.warning(
                "Budget {}ms: run spread {:.0%} exceeds {:.0%}; the {:.3f} derate is not sized "
                "to absorb this, so treat C_max={:.2f} as provisional",
                budget,
                value,
                SPREAD_WARN_THRESHOLD,
                derate,
                curve[budget],
            )
    if report.exhausted_budgets:
        logger.warning(
            "Budgets {} were still passing at the top of the ladder, so their knee is a lower "
            "bound. Extend --target-concurrency to bracket them.",
            report.exhausted_budgets,
        )
    if report.inconclusive_budgets:
        logger.warning(
            "Budgets {} are lower bounds, but higher rates were offered and produced no usable "
            "latency, so a longer ladder will not help; see unusable_reason on the steps above "
            "the knee",
            report.inconclusive_budgets,
        )
    if not report.trustworthy:
        logger.error(
            "This curve is NOT safe to read as per-instance: frozen={}, instance counts {}",
            report.frozen,
            report.instance_counts_observed,
        )
    return report


def dry_run_plan(
    *,
    s_mean_s: float,
    target_concurrencies: Sequence[float] = DEFAULT_TARGET_CONCURRENCIES,
    hold_s: float = DEFAULT_HOLD_S,
    runs: int = 1,
) -> list[dict[str, float]]:
    """The schedule the ladder *would* run, with no AWS calls.

    Sizing a run is the question this answers: at ``hold_s=240`` a ten-step
    ladder times three runs is over two hours, and that is worth seeing before
    committing to it.
    """
    plan: list[dict[str, float]] = []
    for run_index in range(runs):
        for step_index, target in enumerate(sorted(target_concurrencies)):
            rps = rps_for_concurrency(target, s_mean_s)
            plan.append(
                {
                    "run_index": float(run_index),
                    "step_index": float(step_index),
                    "target_concurrency": target,
                    "offered_rps": rps,
                    "hold_s": hold_s,
                    "expected_requests": rps * hold_s,
                }
            )
    return plan


def total_duration_s(
    *,
    target_concurrencies: Sequence[float] = DEFAULT_TARGET_CONCURRENCIES,
    hold_s: float = DEFAULT_HOLD_S,
    settle_between_steps_s: float = DEFAULT_SETTLE_BETWEEN_STEPS_S,
    runs: int = 1,
) -> float:
    """Wall-clock estimate for a full run, excluding the CloudWatch settle wait."""
    steps = len(target_concurrencies)
    per_run = steps * hold_s + max(0, steps - 1) * settle_between_steps_s
    return per_run * runs


def measure(
    *,
    model: str | TTSModelName,
    texts: Sequence[str],
    region: str = "us-east-1",
    variant: str = observe.DEFAULT_VARIANT,
    voice: str | None = None,
    target_concurrencies: Sequence[float] = DEFAULT_TARGET_CONCURRENCIES,
    budgets: Sequence[int] = DEFAULT_TTFAB_BUDGETS,
    hold_s: float = DEFAULT_HOLD_S,
    measure_window_s: float = DEFAULT_MEASURE_WINDOW_S,
    settle_between_steps_s: float = DEFAULT_SETTLE_BETWEEN_STEPS_S,
    runs: int = 1,
    derate: float = 0.875,
    arrival: ArrivalProcess | str = ArrivalProcess.POISSON,
    seed: int | None = 1234,
    probe_requests: int = DEFAULT_PROBE_REQUESTS,
    require_frozen: bool = True,
    pin_to: int = 1,
    cloudwatch_join: bool = True,
    transport: Transport | str = Transport.RESPONSE_STREAM,
    event_sink: Callable[[LoadEvent], None] | None = None,
    # See probe_service_time: bidi's client is not a BaseClient. Callers passing
    # one must build it for the same transport they ask for.
    runtime_client: Any | None = None,
    cloudwatch: BaseClient | None = None,
    appscaling: BaseClient | None = None,
    sagemaker: BaseClient | None = None,
    clock: Clock = SYSTEM_CLOCK,
) -> CMaxReport:
    """Measure ``C_max`` end to end: freeze, probe, ladder, join, report.

    The freeze wraps everything, the probe included, so no part of the
    measurement can run against a fleet that is free to grow. With
    ``require_frozen=True`` (the default) the run refuses to start unless
    scale-out is suspended and capacity is pinned: a warning would be ignored,
    and the resulting number would look entirely normal.

    Args:
        require_frozen: When true, freeze the endpoint and verify before sending
            any load. When false, run against whatever state exists and mark the
            artifact ``frozen=False``, so the number stays identifiable as
            possibly fleet-wide.
        cloudwatch_join: Join server-side metrics after the ladder. Costs a
            settle wait (~2 min) and buys the client/server cross-check.
        transport: Wire protocol to measure on. Recorded on the report, because
            the containers hold their inference lock differently per transport
            (``bidi.py`` module docstring) and a ``C_max`` from one does not
            transfer to the other.

    Raises:
        CMaxError: If the probe fails, or no step met any budget.
        fixture.FixtureError: If the freeze cannot be established or verified.
    """
    from tts_bench import fixture

    model = TTSModelName(model)
    endpoint = resolve_endpoint(model)
    voice = resolve_voice(model, voice)
    instance_type = _instance_type_for(model)
    pool = build_text_pool(texts, seed=seed)
    run_id = uuid.uuid4().hex[:12]
    transport = Transport(transport)

    client = runtime_client or make_client_for(
        transport, region, max_pool=worker_count(max(target_concurrencies))
    )
    # One transport for the probe and every ladder step. S is what converts each
    # concurrency target into a rate, so probing on one protocol and laddering on
    # another would misprice every step on the ladder.
    invoke = invoke_for(transport)
    # The mid-run tripwire runs whether or not we froze — it matters *most* when
    # we did not, since that is the run whose fleet is actually free to change.
    fetch = _instance_count_fetcher(sagemaker, region=region, endpoint=endpoint, variant=variant)

    def _run_ladders() -> tuple[list[LadderRun], ProbeResult]:
        probe = probe_service_time(
            client,
            endpoint=endpoint,
            voice=voice,
            texts=pool,
            requests=probe_requests,
            invoke=invoke,
        )
        logger.info(
            "Probe: S mean {:.3f}s p95 {:.3f}s over {} sample(s); ladder spans {:.2f}-{:.2f} rps",
            probe.s_mean_s,
            probe.s_p95_s,
            probe.samples,
            rps_for_concurrency(min(target_concurrencies), probe.s_mean_s),
            rps_for_concurrency(max(target_concurrencies), probe.s_mean_s),
        )
        ladders = [
            run_ladder(
                client,
                model=model.value,
                endpoint=endpoint,
                voice=voice,
                texts=pool,
                s_mean_s=probe.s_mean_s,
                target_concurrencies=target_concurrencies,
                hold_s=hold_s,
                measure_window_s=measure_window_s,
                settle_between_steps_s=settle_between_steps_s,
                run_index=run_index,
                run_id=f"{run_id}-r{run_index}",
                arrival=arrival,
                seed=seed,
                instance_count_fetch=fetch,
                event_sink=event_sink,
                clock=clock,
                invoke=invoke,
            )
            for run_index in range(runs)
        ]
        return ladders, probe

    if require_frozen:
        with fixture.frozen(
            endpoint,
            region=region,
            variant=variant,
            pin_to=pin_to,
            appscaling=appscaling,
            sagemaker=sagemaker,
        ):
            # freeze() verified suspension and current count; this also checks
            # the *desired* count, and it is the last thing to run before any
            # load is sent.
            fixture.require_frozen(
                endpoint,
                region=region,
                variant=variant,
                expect_instances=pin_to,
                appscaling=appscaling,
                sagemaker=sagemaker,
            )
            ladders, probe = _run_ladders()
    else:
        logger.warning(
            "Running WITHOUT the autoscaling freeze. If the fleet grows mid-run the result is "
            "N x C_max, with nothing in the number marking it as such."
        )
        ladders, probe = _run_ladders()

    all_steps = [step for ladder in ladders for step in ladder.steps]
    if cloudwatch_join:
        all_steps = join_cloudwatch(
            _cloudwatch_client(cloudwatch, region=region),
            endpoint=endpoint,
            variant=variant,
            steps=all_steps,
            results=[result for ladder in ladders for result in ladder.results],
            measure_window_s=measure_window_s,
        )

    return build_report(
        model=model,
        endpoint=endpoint,
        instance_type=instance_type,
        run_id=run_id,
        ladders=ladders,
        probe=probe,
        budgets=budgets,
        hold_s=hold_s,
        measure_window_s=measure_window_s,
        derate=derate,
        frozen=require_frozen,
        arrival=str(ArrivalProcess(arrival)),
        seed=seed,
        transport=transport,
        joined_steps=all_steps,
        measured_at=datetime.now(UTC).isoformat(),
    )


def _instance_count_fetcher(
    sagemaker: BaseClient | None,
    *,
    region: str,
    endpoint: str,
    variant: str,
) -> Callable[[], int]:
    """Build the mid-run capacity tripwire, creating a client only if needed."""
    if sagemaker is None:
        import boto3

        sagemaker = boto3.client("sagemaker", region_name=region)
    return make_instance_count_fetcher(sagemaker, endpoint, variant)


def _cloudwatch_client(cloudwatch: BaseClient | None, *, region: str) -> BaseClient:
    if cloudwatch is not None:
        return cloudwatch
    import boto3

    return boto3.client("cloudwatch", region_name=region)


def _instance_type_for(model: TTSModelName) -> str:
    """Instance type from the benchmark registry, not from ``speech_infra``.

    ``cost.MODEL_INSTANCE_TYPES`` is kept in step with ``TTS_MODEL_CONFIGS`` by a
    consistency test, so this avoids pulling ``aws-cdk-lib`` in for one lookup.
    """
    from tts_bench.cost import DEFAULT_INSTANCE_TYPE, MODEL_INSTANCE_TYPES

    instance_type = MODEL_INSTANCE_TYPES.get(model.value)
    if instance_type is None:
        logger.warning(
            "{} is not in MODEL_INSTANCE_TYPES; costing against {}",
            model.value,
            DEFAULT_INSTANCE_TYPE,
        )
        return DEFAULT_INSTANCE_TYPE
    return instance_type
