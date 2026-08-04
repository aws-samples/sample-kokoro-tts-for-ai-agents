"""``Q_max`` harness: the highest concurrency that still meets the first-byte SLO.

``Q_max`` is one of the two measured variables in the scaling model, and the whole
policy hangs off it: both thresholds are fractions of it, and the admission bound is
it. So it is measured as directly as possible.

**The ladder walks concurrency, and holds it exactly.** Each rung runs ``N`` closed-loop
workers, so queued + executing is ``N`` by construction (``loadgen.run_step``). There is
no arrival rate anywhere in this module — no ``lambda = C/S``, no offered-versus-achieved
comparison, no service-time probe to size a schedule from. That deletion is the point:
every units defect found on this measurement path existed only because a concurrency had
to be converted into a rate first.

**The pass condition is p95 first byte inside the SLO.** One line, one latency field. A
rung passes or it does not; ``Q_max`` is the highest rung that passed, and
``q_max_bracketed`` records whether a rung above it was actually measured and failed —
without that, a ladder that simply ran out while passing reports a lower bound as if it
were an answer.

**Step and hold, never ramp.** Each rung is held for ``hold_s`` and only the trailing
``measure_window_s`` is measured. A ramp smears the crossing, because the queue built at
``N`` is still draining at ``N+1``.

**Saturation is a precondition failure, not a result.** Against the unbounded queue this
ladder requires, the server has nothing to refuse with; sustained rejections mean
something shed load and the rung measured that thing's threshold instead
(:attr:`~tts_bench.loadgen.WindowStats.saturated`).

The run happens inside :class:`tts_bench.fixture.frozen` pinned at one instance, so
``Q_max`` is per-instance rather than fleet-wide; see that module for why this is
enforced rather than advised.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from botocore.client import BaseClient
from loguru import logger

from tts_bench import observe
from tts_bench.bidi import Transport, invoke_for, make_client_for
from tts_bench.invoke import InvokeResult, invoke_stream, resolve_endpoint, resolve_voice
from tts_bench.loadgen import (
    SYSTEM_CLOCK,
    Clock,
    LoadEvent,
    StepResult,
    WindowStats,
    build_text_pool,
    make_instance_count_fetcher,
    run_step,
    summarize_window,
)
from tts_bench.types import Origin, Provenance, QMaxReport, StepSummary
from tts_inference.types import TTSModelName

#: Default rungs. Fine at the bottom and coarse at the top, for two different
#: reasons. ``1`` is not decoration: it is the uncontended service-time sample (what
#: the separate probe phase used to produce) *and* the ``FirstChunkLatencyP95`` alarm
#: threshold. ``5`` and ``10`` are the pair ``ttotal``'s halving test compares — a
#: second instance splits a concurrency-10 probe 5/5, so both must be measured here
#: or that test has nothing to compare against. Above that the spacing widens
#: because a rung costs ``hold_s`` and the SLO crossing is what is being bracketed,
#: not resolved to the request.
DEFAULT_CONCURRENCIES: tuple[int, ...] = (1, 5, 10, 20, 30, 40, 50, 60)

#: Concurrencies that must be on any ladder whose artifact `ttotal` will read. Kept
#: here rather than in `ttotal` because this is where a missing rung is cheap to fix.
RECOVERY_RUNGS: tuple[int, int] = (5, 10)

DEFAULT_HOLD_S = 240.0
DEFAULT_MEASURE_WINDOW_S = 60.0

#: Idle seconds between rungs so the previous rung's queue drains. Without it a rung
#: inherits backlog and crosses the SLO for the wrong reason.
DEFAULT_SETTLE_BETWEEN_STEPS_S = 30.0

#: Stop the ladder after this many consecutive saturated rungs. Saturation against an
#: unbounded queue means the run's preconditions broke, and higher rungs then cost
#: money to measure the same broken thing.
DEFAULT_SATURATED_STEPS_TO_STOP = 2

#: Run-to-run spread above which the ladder resolved noise rather than a crossing.
#: Relative to the minimum, matching :attr:`QMaxReport.q_max_spread`.
SPREAD_WARN_THRESHOLD = 0.2

#: Connection-pool headroom over the highest rung. The pool must exceed ``N`` or
#: urllib3 queues connections and the ladder measures the client — the one confound
#: that looks exactly like a real SLO crossing.
POOL_HEADROOM = 4


class QMaxError(RuntimeError):
    """The run could not produce a usable ``Q_max``."""


def pool_size(max_concurrency: int) -> int:
    """Connection-pool size for a ladder topping out at ``max_concurrency``.

    Unlike the open-loop version this needs no estimate of overshoot: closed-loop
    in-flight cannot exceed ``N``, so the pool only has to clear the highest rung with
    a little room. There is no case where a *passing* rung was silently client-limited
    and no override to reason about — the shortfall check
    (:attr:`~tts_bench.loadgen.WindowStats.client_bound`) catches it directly.
    """
    if max_concurrency < 1:
        raise ValueError(f"max_concurrency must be >= 1, got {max_concurrency}")
    return max_concurrency + POOL_HEADROOM


def _unusable_reason(stats: WindowStats, result: StepResult) -> str | None:
    """Why this rung may not inform ``Q_max``, or ``None`` if it may."""
    if stats.capacity_changed:
        return (
            f"instance count changed mid-step {result.instance_counts}; N outstanding "
            "requests spread over two instances is not a per-instance measurement"
        )
    if stats.completed == 0:
        return "no requests completed inside the measure window"
    if result.worker_overlaps:
        # Falsifies the driver's one structural guarantee. If a worker dispatched
        # before its previous request finished, outstanding requests exceeded N and
        # the rung's concurrency is not the number it claims — which is the only
        # number this ladder is measuring.
        return (
            f"{result.worker_overlaps} overlapping request(s) on a single worker; "
            f"outstanding requests exceeded the requested concurrency of {stats.concurrency}"
        )
    if stats.client_bound:
        # The closed-loop replacement for dispatch_skipped: mean in-flight fell far
        # enough below N that our own dispatch overhead, not the server's queue, set
        # the latency. Excluded rather than flagged, because it looks like a pass.
        shortfall = stats.concurrency_shortfall or 0.0
        return (
            f"mean in-flight was {shortfall:.0%} below N={stats.concurrency}; the client "
            "was the limit, so this rung measures the benchmark"
        )
    if stats.saturated:
        # The queue was supposed to be unbounded. Something refused work, so the
        # latency percentiles describe requests that got through a shed filter.
        return (
            f"{stats.rejected}/{stats.completed} completions were rejections "
            f"({dict(stats.outcome_counts)}); the queue was supposed to be unbounded, so "
            "this rung measures whatever shed the load"
        )
    if not stats.settled:
        # On a serial server a queue still filling grows latency without bound, so
        # whatever p95 the window reports is a snapshot of a moving number rather
        # than the SLO verdict at this concurrency.
        return (
            f"TTFAB was still climbing at the end of the window (drift "
            f"{stats.ttfab_drift_ms:.0f}ms); the queue had not reached steady state"
            if stats.ttfab_drift_ms is not None
            else "the window did not reach steady state"
        )
    return None


def summarize_step(
    result: StepResult,
    stats: WindowStats,
    *,
    run_index: int,
    slo_ms: int,
) -> StepSummary:
    """Project one measured rung into its serializable artifact form.

    ``meets_slo`` is decided here rather than downstream, so the artifact carries the
    verdict against the SLO it was actually judged on. A rung the ladder could not
    trust never meets the SLO whatever its p95 reads: ``Q_max`` is the highest rung
    that *both* passed and was a valid measurement.
    """
    reason = _unusable_reason(stats, result)
    usable = reason is None
    meets_slo = usable and stats.ttfab_p95_ms is not None and stats.ttfab_p95_ms <= slo_ms
    return StepSummary(
        run_index=run_index,
        step_index=result.step_index,
        concurrency=stats.concurrency,
        achieved_rps=stats.achieved_rps,
        completed=stats.completed,
        ok=stats.ok,
        rejected=stats.rejected,
        chars=stats.chars,
        outcome_counts=dict(stats.outcome_counts),
        ttfab_p50_ms=stats.ttfab_p50_ms,
        ttfab_p95_ms=stats.ttfab_p95_ms,
        ttfab_p99_ms=stats.ttfab_p99_ms,
        latency_p95_ms=stats.latency_p95_ms,
        s_mean_s=stats.s_mean_s,
        s_p95_s=stats.s_p95_s,
        concurrency_mean=stats.concurrency_mean,
        concurrency_peak=stats.concurrency_peak,
        ttfab_drift_ms=stats.ttfab_drift_ms,
        meets_slo=meets_slo,
        saturated=stats.saturated,
        settled=stats.settled,
        client_bound=stats.client_bound,
        usable=usable,
        unusable_reason=reason,
        capacity_changed=stats.capacity_changed,
        instance_counts=result.instance_counts,
    )


def find_q_max(steps: Sequence[StepSummary]) -> tuple[int | None, bool]:
    """``(q_max, bracketed)`` from one run's rungs.

    The *highest* passing rung rather than the one below the first failure, so a single
    noisy rung cannot truncate the ladder. ``bracketed`` asks whether some rung above
    the answer was measured and missed the SLO — which is what separates "50 is the
    limit" from "50 is as far as we looked".

    A rung excluded by :func:`_unusable_reason` can still bracket, on its measured p95
    alone. Exclusion says the rung cannot be trusted as a *pass*; a p95 past the SLO
    there is still a fact about the endpoint, and dropping it entirely is what makes a
    ladder that hit the wall report its answer as a lower bound.

    Returns:
        ``(None, False)`` when no rung met the SLO — the honest answer for a ladder
        whose lowest rung is already past the line, and not something to substitute a
        default for.
    """
    ordered = sorted(steps, key=lambda s: s.concurrency)
    passing = [s for s in ordered if s.meets_slo]
    if not passing:
        return None, False

    best = passing[-1]
    bracketed = any(
        s.concurrency > best.concurrency
        and s.ttfab_p95_ms is not None
        and s.ttfab_p95_ms > 0
        and not s.meets_slo
        for s in ordered
    )
    return best.concurrency, bracketed


def service_time_at_lowest_rung(steps: Sequence[StepSummary]) -> tuple[float, float] | None:
    """``(s_mean_s, s_p95_s)`` from the lowest usable rung, or ``None``.

    The lowest rung, not the rung at ``Q_max``: service time there already contains
    queueing, and ``W_max = SLO - S_p95`` would then subtract a wait it had already
    counted. At ``N=1`` the closed loop is an uncontended probe by construction, which
    is what let the separate probe phase go.

    Includes the client-to-endpoint round trip (~34 ms measured against kokoro), so it
    is service time as a client experiences it. That is the right quantity for an SLO
    defined on client-observed first byte, and a slight over-estimate anywhere it
    stands in for server-side work.
    """
    candidates = sorted(
        (s for s in steps if s.usable and s.s_mean_s is not None and s.s_mean_s > 0),
        key=lambda s: s.concurrency,
    )
    if not candidates:
        return None
    lowest = candidates[0]
    s_mean = lowest.s_mean_s or 0.0
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
    # `Any`, not `BaseClient`: the bidi transport's client is a
    # `SageMakerRuntimeHTTP2Client` with no botocore ancestry. It is only ever handed
    # to `invoke`, never called directly.
    client: Any,
    *,
    model: str,
    endpoint: str,
    voice: str,
    texts: Sequence[str],
    slo_ms: int,
    concurrencies: Sequence[int] = DEFAULT_CONCURRENCIES,
    hold_s: float = DEFAULT_HOLD_S,
    measure_window_s: float = DEFAULT_MEASURE_WINDOW_S,
    settle_between_steps_s: float = DEFAULT_SETTLE_BETWEEN_STEPS_S,
    run_index: int = 0,
    run_id: str | None = None,
    saturated_steps_to_stop: int = DEFAULT_SATURATED_STEPS_TO_STOP,
    instance_count_fetch: Callable[[], int] | None = None,
    event_sink: Callable[[LoadEvent], None] | None = None,
    clock: Clock = SYSTEM_CLOCK,
    step_runner: Callable[..., StepResult] = run_step,
    invoke: Callable[..., InvokeResult] = invoke_stream,
) -> LadderRun:
    """Walk the ladder once, measuring only each rung's trailing window.

    Stops early after ``saturated_steps_to_stop`` consecutive saturated rungs, and
    records where. A ladder that stopped is never read as covering rungs it skipped.

    Note it does **not** stop at the first SLO miss: one rung past the crossing is what
    brackets ``Q_max`` from above, and a noisy rung in the middle should not truncate
    the run.

    Args:
        slo_ms: The p95 first-byte line each rung is judged against.
        measure_window_s: Trailing part of each rung that is measured. The rest is
            warm-up — including the burst of all ``N`` workers starting together — and
            is discarded.
        step_runner: Injected for tests; defaults to :func:`loadgen.run_step`.
        invoke: Transport for each request. Recorded on the report, because the
            containers hold their inference lock differently per transport.

    Raises:
        ValueError: If the measure window does not fit inside the hold, or a rung is
            not a positive concurrency.
    """
    if measure_window_s > hold_s:
        raise ValueError(
            f"measure_window_s ({measure_window_s}) must not exceed hold_s ({hold_s}): the "
            "window is the trailing part of the rung, not an addition to it"
        )
    if not concurrencies:
        raise ValueError("concurrencies must not be empty")
    for value in concurrencies:
        if value < 1:
            raise ValueError(f"every concurrency must be >= 1, got {value}")

    run_id = run_id or uuid.uuid4().hex[:12]
    ladder = LadderRun(run_index=run_index)
    rungs = sorted({int(c) for c in concurrencies})
    consecutive_saturated = 0

    for step_index, concurrency in enumerate(rungs):
        logger.info(
            "Run {} rung {}: holding N={} for {:.0f}s (measuring the last {:.0f}s), "
            "SLO p95 TTFAB <= {}ms",
            run_index,
            step_index,
            concurrency,
            hold_s,
            measure_window_s,
            slo_ms,
        )

        result = step_runner(
            client,
            model=model,
            endpoint=endpoint,
            voice=voice,
            texts=texts,
            concurrency=concurrency,
            duration_s=hold_s,
            step_index=step_index,
            run_id=run_id,
            instance_count_fetch=instance_count_fetch,
            event_sink=event_sink,
            clock=clock,
            invoke=invoke,
        )

        # Windowed off `started_ts`, not `ended_ts`. `run_step` lets requests already
        # in flight finish past the duration, so a window anchored at the end of the
        # step reaches into a drain tail where in-flight is falling toward zero — and
        # then reports a mean concurrency below N for a rung that held N perfectly.
        window_end = min(result.started_ts + hold_s, result.ended_ts)
        stats = summarize_window(
            result,
            start_ts=window_end - measure_window_s,
            end_ts=window_end,
        )
        summary = summarize_step(result, stats, run_index=run_index, slo_ms=slo_ms)
        ladder.steps.append(summary)
        ladder.results.append(result)

        logger.info(
            "Run {} rung {}: N={} -> p95 TTFAB {}, {:.2f} rps, in-flight {} "
            "(meets_slo={} usable={})",
            run_index,
            step_index,
            concurrency,
            f"{stats.ttfab_p95_ms:.0f}ms" if stats.ttfab_p95_ms is not None else "n/a",
            stats.achieved_rps,
            f"{stats.concurrency_mean:.2f}" if stats.concurrency_mean is not None else "n/a",
            summary.meets_slo,
            summary.usable,
        )
        if summary.unusable_reason:
            logger.warning("Run {} rung {}: {}", run_index, step_index, summary.unusable_reason)

        consecutive_saturated = consecutive_saturated + 1 if stats.saturated else 0
        if consecutive_saturated >= saturated_steps_to_stop:
            logger.warning(
                "Run {}: {} consecutive saturated rung(s); stopping at rung {}. Against an "
                "unbounded queue nothing should be refusing work, so higher rungs would "
                "only re-measure whatever is shedding.",
                run_index,
                consecutive_saturated,
                step_index,
            )
            ladder.truncated_at = step_index
            break

        if settle_between_steps_s > 0 and step_index < len(rungs) - 1:
            logger.info("Draining {:.0f}s before the next rung", settle_between_steps_s)
            clock.sleep(settle_between_steps_s)

    return ladder


def join_cloudwatch(
    cloudwatch: BaseClient,
    *,
    endpoint: str,
    variant: str,
    steps: Sequence[StepSummary],
    results: Sequence[tuple[int, StepResult]],
    measure_window_s: float,
    hold_s: float,
    settle_delay_s: float = observe.DEFAULT_SETTLE_DELAY_S,
    period_s: int = observe.HIGH_RES_PERIOD_S,
    sleep: Callable[[float], None] | None = None,
    now: Callable[[], datetime] | None = None,
) -> list[StepSummary]:
    """Attach server-side metrics to each rung. Never raises.

    Two things only the server can answer. Whether the load reached the endpoint at all
    — :func:`observe.concurrency_agreement` catches a connection-pool bottleneck, which
    from the client side looks like server queueing. And **what unit the threshold
    deploys in**: ``server_concurrency_peak`` is ``ConcurrentRequestsPerModel`` /
    *Maximum*, the exact statistic the target-tracking alarm reads, and the ratio
    between it and client-observed mean in-flight ran 9.8x at low load to 1.35x at
    high. A threshold derived from the client figure and compared against the server's
    is how ``0.713`` — a value no positive arrival rate satisfies — reached production.
    High-resolution datapoints retain **3 hours**, so this join cannot be backfilled.

    ``results`` is keyed by ``(run_index, step_index)``. Keying by step index alone
    made every run after the first overwrite the previous one, and the survivor was
    then applied to all of them — visible as byte-identical server metrics for runs 35
    minutes apart.

    The settle wait is taken once, against the **newest** window. Waiting on the oldest
    returns immediately (its window is already older than the delay) and the newest rung
    would then be read before CloudWatch had aggregated it.

    A failure here degrades the report but must never lose a measured ladder, so every
    fetch is caught and the unjoined summary kept.
    """
    if not results:
        return list(steps)

    def _window(result: StepResult, *, settle: bool) -> observe.WindowMetrics:
        # The same window the client summarized, for the same reason: anchored to the
        # start of the rung so the drain tail past `hold_s` stays out of it.
        end = datetime.fromtimestamp(min(result.started_ts + hold_s, result.ended_ts), tz=UTC)
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

    newest = max(results, key=lambda pair: pair[1].ended_ts)
    windows: dict[tuple[int, int], observe.WindowMetrics] = {}
    for run_index, result in sorted(results, key=lambda pair: pair[1].ended_ts, reverse=True):
        key = (run_index, result.step_index)
        try:
            windows[key] = _window(result, settle=(run_index, result) == newest)
        except Exception as exc:  # noqa: BLE001 - telemetry must not lose a measured ladder
            logger.warning(
                "Could not join CloudWatch for run {} rung {}: {}. Client-side numbers "
                "stand; the server cross-check and the CW-unit conversion are missing.",
                run_index,
                result.step_index,
                exc,
            )

    joined: list[StepSummary] = []
    for summary in steps:
        window = windows.get((summary.run_index, summary.step_index))
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
                    "server_concurrency_peak": window.concurrency_peak,
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
    slo_ms: int,
    hold_s: float = DEFAULT_HOLD_S,
    measure_window_s: float = DEFAULT_MEASURE_WINDOW_S,
    frozen: bool = False,
    unbounded_queue: bool | None = None,
    transport: Transport | str = Transport.RESPONSE_STREAM,
    joined_steps: Sequence[StepSummary] | None = None,
    measured_at: str | None = None,
    deployed_config: dict[str, Any] | None = None,
) -> QMaxReport:
    """Assemble the artifact from one or more ladder runs.

    Cross-run ``Q_max`` is the **minimum** of the per-run answers, not the median: the
    median of two rungs is a concurrency no run tested, while the minimum is both a
    rung that really ran and the conservative choice for a number that sets an
    admission bound. Every run's answer is carried in ``q_max_per_run`` so the
    disagreement stays visible instead of being averaged away.

    Raises:
        QMaxError: If no rung met the SLO, or no rung yielded a service time. Both are
            real findings, but neither is a ``Q_max``, so this fails rather than
            serializing a number nothing measured.
    """
    if joined_steps is not None:
        all_steps = list(joined_steps)
    else:
        all_steps = [step for ladder in ladders for step in ladder.steps]

    per_run: list[int] = []
    bracketed_any = False
    for ladder in ladders:
        # Per ladder, not over the pooled steps: Q_max is the crossing *one* pass
        # found, and pooling first would let the luckiest run stand in for all of them.
        answer, bracketed = find_q_max(ladder.steps)
        if answer is not None:
            per_run.append(answer)
            bracketed_any = bracketed_any or bracketed

    if not per_run:
        highest = max((s.concurrency for s in all_steps), default=0)
        lowest_p95 = min(
            (s.ttfab_p95_ms for s in all_steps if s.ttfab_p95_ms is not None), default=None
        )
        unusable = [s for s in all_steps if not s.usable]
        detail = (
            f" Every rung was excluded as a measurement; the first reason was: "
            f"{unusable[0].unusable_reason}"
            if len(unusable) == len(all_steps) and unusable
            else ""
        )
        raise QMaxError(
            f"no ladder rung met the {slo_ms}ms p95 TTFAB SLO across {highest} concurrency "
            f"(best p95 "
            f"{f'{lowest_p95:.0f}ms' if lowest_p95 is not None else 'n/a'}).{detail} Either "
            "the endpoint is unhealthy, or its service time alone already exceeds the SLO — "
            "in which case no queue depth is admissible and the fix is a faster "
            "configuration, not a lower Q_max."
        )

    q_max = min(per_run)
    service = service_time_at_lowest_rung(all_steps)
    if service is None:
        raise QMaxError(
            "no ladder rung produced a service time, so W_max = SLO - S_p95 cannot be "
            "computed and the artifact would carry a Q_max with no latency behind it. "
            "Check that the lowest rung completed requests."
        )
    s_mean, s_p95 = service

    p95_at_q_max = next(
        (
            s.ttfab_p95_ms
            for s in sorted(all_steps, key=lambda s: s.concurrency)
            if s.concurrency == q_max and s.meets_slo and s.ttfab_p95_ms is not None
        ),
        0.0,
    )

    observed: list[int] = []
    for ladder in ladders:
        for count in ladder.instance_counts:
            if count not in observed:
                observed.append(count)

    truncated = next(
        (ladder.truncated_at for ladder in ladders if ladder.truncated_at is not None),
        None,
    )

    transport = Transport(transport)
    report = QMaxReport(
        model_name=TTSModelName(model),
        endpoint=endpoint,
        instance_type=instance_type,
        run_id=run_id,
        slo_ms=slo_ms,
        q_max=q_max,
        q_max_per_run=tuple(per_run),
        q_max_bracketed=bracketed_any,
        ttfab_p95_at_q_max_ms=p95_at_q_max,
        s_mean_s=s_mean,
        s_p95_s=s_p95,
        frozen=frozen,
        unbounded_queue=unbounded_queue,
        instance_counts_observed=tuple(observed),
        ladder_truncated_at=truncated,
        transport=str(transport),
        deployed_config=dict(deployed_config) if deployed_config else {},
        runs=len(ladders),
        hold_s=hold_s,
        measure_window_s=measure_window_s,
        steps=all_steps,
        provenance=Provenance(
            origin=Origin.MEASURED,
            run_id=run_id,
            measured_at=measured_at,
            endpoint=endpoint,
            note=(
                f"Q_max at the {slo_ms}ms p95 TTFAB SLO on {transport}, autoscaling frozen"
                if frozen
                else f"Q_max measured on {transport} WITHOUT the autoscaling freeze; may be "
                "N x Q_max"
            ),
        ),
    )

    _warn_about(report, ladders)
    return report


def _warn_about(report: QMaxReport, ladders: Sequence[LadderRun]) -> None:
    """Log everything that makes this ``Q_max`` weaker than it looks."""
    if not report.q_max_bracketed:
        logger.warning(
            "Q_max={} is a LOWER bound: no rung above it was measured and failed the {}ms "
            "SLO (p95 there was {:.0f}ms, leaving {:.0f}ms of the SLO unused). Both scaling "
            "thresholds are fractions of this, so the policy will add instances earlier "
            "than needed. Extend --concurrency past {} to bracket it.",
            report.q_max,
            report.slo_ms,
            report.ttfab_p95_at_q_max_ms,
            report.slo_ms - report.ttfab_p95_at_q_max_ms,
            report.q_max,
        )
    if report.runs_contributing > 1 and report.q_max_spread > SPREAD_WARN_THRESHOLD:
        logger.warning(
            "Per-run Q_max spread is {:.0%} across {} run(s) ({}); planning on the minimum, "
            "but a spread this wide means the ladder resolved noise as much as a crossing. "
            "Lengthen --hold or add rungs around {}.",
            report.q_max_spread,
            report.runs_contributing,
            list(report.q_max_per_run),
            report.q_max,
        )
    if report.runs_contributing < len(ladders):
        logger.warning(
            "Only {}/{} run(s) produced a Q_max, so the {:.0%} spread is not a repeatability claim",
            report.runs_contributing,
            len(ladders),
            report.q_max_spread,
        )
    if report.saturated_rungs:
        logger.error(
            "Rungs {} saw the server refuse work. This ladder is supposed to run against "
            "an unbounded queue, so something shed load and those rungs measure its "
            "threshold rather than the SLO crossing. Check MAX_QUEUE_DEPTH and "
            "MAX_REQUEST_AGE_S on the container.",
            report.saturated_rungs,
        )
    if report.client_bound_rungs:
        logger.warning(
            "Rungs {} were client-limited and excluded; the benchmark host, not the "
            "endpoint, set their latency",
            report.client_bound_rungs,
        )
    missing = [rung for rung in RECOVERY_RUNGS if report.ttfab_p95_at(rung) is None]
    if missing:
        logger.warning(
            "Rungs {} produced no usable p95, so `ttotal` cannot run its halving test "
            "against this artifact: it holds a probe at {} and declares recovery when p95 "
            "reaches the {} value. Re-run with those rungs on --concurrency.",
            missing,
            RECOVERY_RUNGS[1],
            RECOVERY_RUNGS[0],
        )
    if report.ttfab_p95_at_c1_ms is None:
        logger.warning(
            "No usable N=1 rung, so ttfab_p95_at_c1_ms is absent and the "
            "FirstChunkLatencyP95 alarm has no measured threshold. That alarm watches "
            "service time on an instance already serving, where the request has spent "
            "none of its queue allowance, so the {}ms SLO cannot stand in for it.",
            report.slo_ms,
        )
    if report.unbounded_queue is None:
        logger.warning(
            "The unbounded-queue precondition was not checked. A container that sheds at "
            "a fixed depth reports that depth as Q_max."
        )
    if not report.trustworthy:
        logger.error(
            "This Q_max is NOT safe to read as per-instance: frozen={}, instance counts {}",
            report.frozen,
            report.instance_counts_observed,
        )


def dry_run_plan(
    *,
    concurrencies: Sequence[int] = DEFAULT_CONCURRENCIES,
    hold_s: float = DEFAULT_HOLD_S,
    s_mean_s: float | None = None,
    runs: int = 1,
) -> list[dict[str, float]]:
    """The schedule the ladder *would* run, with no AWS calls.

    Sizing a run is the question this answers: at ``hold_s=240`` an eight-rung ladder
    times two runs is over an hour, and that is worth seeing before committing to it.

    ``expected_requests`` is present only when ``s_mean_s`` is supplied, and is a
    forecast rather than a schedule: a closed loop completes ``N/S`` per second, so it
    follows from the service time rather than driving anything. Absent, the rung still
    runs — which is the difference from the open-loop version, where a wrong service
    time misplaced every rung.
    """
    plan: list[dict[str, float]] = []
    for run_index in range(runs):
        for step_index, concurrency in enumerate(sorted(set(concurrencies))):
            row: dict[str, float] = {
                "run_index": float(run_index),
                "step_index": float(step_index),
                "concurrency": float(concurrency),
                "hold_s": hold_s,
            }
            if s_mean_s is not None and s_mean_s > 0:
                row["expected_requests"] = concurrency * hold_s / s_mean_s
            plan.append(row)
    return plan


def total_duration_s(
    *,
    concurrencies: Sequence[int] = DEFAULT_CONCURRENCIES,
    hold_s: float = DEFAULT_HOLD_S,
    settle_between_steps_s: float = DEFAULT_SETTLE_BETWEEN_STEPS_S,
    runs: int = 1,
) -> float:
    """Wall-clock estimate for a full run, excluding the CloudWatch settle wait."""
    rungs = len(set(concurrencies))
    per_run = rungs * hold_s + max(0, rungs - 1) * settle_between_steps_s
    return per_run * runs


def measure(
    *,
    model: str | TTSModelName,
    texts: Sequence[str],
    slo_ms: int,
    region: str = "us-east-1",
    variant: str = observe.DEFAULT_VARIANT,
    voice: str | None = None,
    concurrencies: Sequence[int] = DEFAULT_CONCURRENCIES,
    hold_s: float = DEFAULT_HOLD_S,
    measure_window_s: float = DEFAULT_MEASURE_WINDOW_S,
    settle_between_steps_s: float = DEFAULT_SETTLE_BETWEEN_STEPS_S,
    runs: int = 1,
    seed: int | None = 1234,
    require_frozen: bool = True,
    require_unbounded_queue: bool = True,
    pin_to: int = 1,
    cloudwatch_join: bool = True,
    transport: Transport | str = Transport.RESPONSE_STREAM,
    event_sink: Callable[[LoadEvent], None] | None = None,
    # See run_ladder: bidi's client is not a BaseClient. Callers passing one must
    # build it for the same transport they ask for.
    runtime_client: Any | None = None,
    cloudwatch: BaseClient | None = None,
    appscaling: BaseClient | None = None,
    sagemaker: BaseClient | None = None,
    clock: Clock = SYSTEM_CLOCK,
) -> QMaxReport:
    """Measure ``Q_max`` end to end: freeze, preflight, ladder, join, report.

    The freeze wraps the whole ladder, so no part of the measurement runs against a
    fleet free to grow. With ``require_frozen=True`` (the default) the run refuses to
    start unless scale-out is suspended and capacity is pinned: a warning would be
    ignored and the resulting number would look entirely normal.

    Args:
        slo_ms: The p95 first-byte line. ``Q_max`` is *defined by* it and the artifact
            records it, so a plan built against a different SLO can be refused rather
            than silently re-reading this ladder.
        require_frozen: Freeze and verify before sending load. When false, run against
            whatever state exists and mark the artifact ``frozen=False``, so the number
            stays identifiable as possibly fleet-wide.
        require_unbounded_queue: Refuse when the container bounds its admission queue.
            When false the check still runs and its result is recorded, so a bounded
            run stays identifiable; see :func:`fixture.require_unbounded_queue`.
        cloudwatch_join: Join server-side metrics after the ladder. Costs a settle wait
            (~2 min) and buys the client/server cross-check *and* the CloudWatch-unit
            conversion for the deployed threshold. High-resolution datapoints retain 3
            hours, so skipping it cannot be undone later.
        transport: Wire protocol to measure on. Recorded on the report, because the
            containers hold their inference lock differently per transport
            (``bidi.py`` module docstring) and a ``Q_max`` from one does not transfer.

    Raises:
        QMaxError: If no rung met the SLO or none produced a service time.
        fixture.FixtureError: If the freeze cannot be established, or the container
            bounds its queue while ``require_unbounded_queue`` is set.
    """
    from tts_bench import fixture

    model = TTSModelName(model)
    endpoint = resolve_endpoint(model)
    voice = resolve_voice(model, voice)
    deployed = fixture.fingerprint_or_registry(
        model.value, endpoint=endpoint, region=region, variant=variant, sagemaker=sagemaker
    )
    instance_type = deployed.instance_type or fixture.registry_instance_type(model.value)

    # Before the freeze: a queue bound makes the whole run pointless, and finding that
    # out after suspending production autoscaling costs a thaw for nothing.
    unbounded: bool | None = None
    try:
        fixture.require_unbounded_queue(
            endpoint, region=region, variant=variant, sagemaker=sagemaker, deployed=deployed
        )
        unbounded = True
    except fixture.FixtureError as exc:
        if require_unbounded_queue:
            raise
        unbounded = False
        logger.error(
            "{} Proceeding because --no-require-unbounded-queue was passed; the artifact "
            "records unbounded_queue=False.",
            exc,
        )

    pool = build_text_pool(texts, seed=seed)
    run_id = uuid.uuid4().hex[:12]
    transport = Transport(transport)
    rungs = sorted({int(c) for c in concurrencies})
    if not rungs:
        raise ValueError("concurrencies must not be empty")

    client = runtime_client or make_client_for(transport, region, max_pool=pool_size(max(rungs)))
    invoke = invoke_for(transport)
    # The mid-run tripwire runs whether or not we froze — it matters *most* when we did
    # not, since that is the run whose fleet is actually free to change.
    fetch = _instance_count_fetcher(sagemaker, region=region, endpoint=endpoint, variant=variant)

    def _run_ladders() -> list[LadderRun]:
        return [
            run_ladder(
                client,
                model=model.value,
                endpoint=endpoint,
                voice=voice,
                texts=pool,
                slo_ms=slo_ms,
                concurrencies=rungs,
                hold_s=hold_s,
                measure_window_s=measure_window_s,
                settle_between_steps_s=settle_between_steps_s,
                run_index=run_index,
                run_id=f"{run_id}-r{run_index}",
                instance_count_fetch=fetch,
                event_sink=event_sink,
                clock=clock,
                invoke=invoke,
            )
            for run_index in range(runs)
        ]

    if require_frozen:
        with fixture.frozen(
            endpoint,
            region=region,
            variant=variant,
            pin_to=pin_to,
            appscaling=appscaling,
            sagemaker=sagemaker,
        ):
            # freeze() verified suspension and current count; this also checks the
            # *desired* count, and it is the last thing to run before any load is sent.
            fixture.require_frozen(
                endpoint,
                region=region,
                variant=variant,
                expect_instances=pin_to,
                appscaling=appscaling,
                sagemaker=sagemaker,
            )
            ladders = _run_ladders()
    else:
        logger.warning(
            "Running WITHOUT the autoscaling freeze. If the fleet grows mid-run the result "
            "is N x Q_max, with nothing in the number marking it as such."
        )
        ladders = _run_ladders()

    all_steps = [step for ladder in ladders for step in ladder.steps]
    if cloudwatch_join:
        all_steps = join_cloudwatch(
            _cloudwatch_client(cloudwatch, region=region),
            endpoint=endpoint,
            variant=variant,
            steps=all_steps,
            results=[(ladder.run_index, result) for ladder in ladders for result in ladder.results],
            measure_window_s=measure_window_s,
            hold_s=hold_s,
        )
    else:
        logger.warning(
            "Skipping the CloudWatch join, so no rung records ConcurrentRequestsPerModel / "
            "Maximum — the statistic the deployed threshold is compared against. "
            "High-resolution datapoints retain 3 hours, so `plan` will have no unit "
            "conversion and this cannot be backfilled."
        )

    return build_report(
        model=model,
        endpoint=endpoint,
        instance_type=instance_type,
        run_id=run_id,
        ladders=ladders,
        slo_ms=slo_ms,
        hold_s=hold_s,
        measure_window_s=measure_window_s,
        frozen=require_frozen,
        unbounded_queue=unbounded,
        transport=transport,
        joined_steps=all_steps,
        measured_at=datetime.now(UTC).isoformat(),
        deployed_config=deployed.to_dict(),
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
