"""Tests for the closed-loop ``Q_max`` ladder.

``Q_max`` is the highest concurrency whose p95 first-byte time stayed inside one SLO,
and every number the deployed policy carries is a fraction of it. So these tests
concentrate on the ways this module can produce a *plausible* value for something else:

- **A rung that was not a measurement must not read as a pass.** Saturation, a
  client-limited window and a mid-run resize each disqualify a rung, and each looks
  exactly like a healthy one from the latency percentiles alone.
- **A ladder that ran out while passing is a lower bound.** ``q_max_bracketed`` is the
  only thing separating "50 is the limit" from "50 is as far as we looked", and both
  scaling thresholds inherit the difference.
- **The units the answer converts into are measured, not assumed.** ``cw_units_ratio_by_rung``
  exists because ``ConcurrentRequestsPerModel``/*Maximum* and client mean in-flight are
  different quantities whose ratio is not constant.

Four defects that shipped are pinned here by name, because all four were silent:

- ``join_cloudwatch`` keyed its windows by ``step_index`` alone, so with ``--runs 3``
  each run overwrote the previous and the survivor was applied to all of them — visible
  live as byte-identical server metrics for runs 35 minutes apart.
  :class:`TestTheJoinIsKeyedByRunAndRung` is the regression lock.
- ``saturated`` compared achieved against offered throughput while counting any event
  with an ``end_ts`` as a completion, so a 1 ms 503 *raised* measured throughput and a
  fully-shedding endpoint read as healthy. :class:`TestRejectionsCountAsSaturation`.
- ``server_concurrency_peak`` was never read off ``WindowMetrics.concurrency_peak``, so
  nothing carried the statistic the deployed alarm compares against.
  :class:`TestTheJoinRecordsBothConcurrencyUnits`.
- ``find_q_max`` brackets on ``not meets_slo`` rather than on a p95 past the SLO, so a
  rung *excluded* above ``Q_max`` claims a bracket the ladder never measured. Left as an
  ``xfail`` in :class:`TestFindQMax` — see that test's reason.

Synthetic ``StepResult``s drive the ladder so saturation, drift and in-flight are exact
rather than timing-dependent; ``run_step`` itself is covered by ``test_loadgen.py``,
including the ``Q + E = N`` property. One test runs the real ``run_step`` against a fake
server to keep the two wired together. AWS reads go through botocore ``Stubber``, which
validates every response against the real service model. ``caplog`` is not used: loguru
does not propagate to the stdlib logging tree, so an assertion against it would pass
whether or not anything was logged.
"""

from __future__ import annotations

import functools
import threading
import time
from collections.abc import Iterator, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

import boto3
import pytest
from botocore.stub import Stubber
from loguru import logger

from tts_bench import fixture as fixture_mod
from tts_bench import qmax as qmax_mod
from tts_bench.bidi import Transport
from tts_bench.fixture import SCALABLE_DIMENSION, SERVICE_NAMESPACE, SUSPEND_ALL, FixtureError
from tts_bench.invoke import InvokeOutcome, InvokeResult
from tts_bench.loadgen import (
    SYSTEM_CLOCK,
    Clock,
    ConcurrencySample,
    LoadEvent,
    StepResult,
    run_step,
    summarize_window,
)
from tts_bench.observe import HIGH_RES_PERIOD_S
from tts_bench.qmax import (
    DEFAULT_CONCURRENCIES,
    POOL_HEADROOM,
    RECOVERY_RUNGS,
    SPREAD_WARN_THRESHOLD,
    LadderRun,
    QMaxError,
    build_report,
    dry_run_plan,
    find_q_max,
    join_cloudwatch,
    measure,
    pool_size,
    run_ladder,
    service_time_at_lowest_rung,
    summarize_step,
    total_duration_s,
)
from tts_bench.types import Origin, StepSummary

MODEL = "kokoro-82m"
ENDPOINT = "speech-kokoro-82m"
VARIANT = "primary"
RID = f"endpoint/{ENDPOINT}/variant/{VARIANT}"
TEXTS = ["Let me check that for you.", "Your appointment is confirmed.", "One moment please."]

#: The one line every rung is judged against, queue time included.
SLO_MS = 3000
HOLD_S = 240.0
WINDOW_S = 60.0

NOW = datetime(2026, 7, 29, 12, 30, 0, tzinfo=UTC)
T0_TS = NOW.timestamp()

#: The measured kokoro ladder, shared verbatim with ``test_ttotal.py`` and
#: ``test_cli_plan.py`` so one shape of endpoint is described the same way everywhere.
LADDER = {1: 92.0, 5: 379.0, 10: 667.0, 20: 1243.0, 50: 2910.0}

#: The rung above ``Q_max`` that actually missed the SLO. Without one measured and failed,
#: 50 is only where the ladder stopped.
FAILING_RUNG = {60: 3400.0}

#: CloudWatch *Maximum* over client mean in-flight, per rung, off one live kokoro ladder.
#: Not a constant, and that is the whole point: a lightly loaded endpoint's 10s peak is
#: many multiples of its 60s average while a saturated one's is barely above it.
RATIOS = {1: 9.8, 5: 5.1, 10: 3.0, 20: 2.0, 50: 1.35}


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def logged() -> Iterator[list[str]]:
    """Captured loguru messages at WARNING and above.

    ``caplog`` does not see these — loguru writes to its own sinks and does not
    propagate to the stdlib ``logging`` tree — so a ``caplog`` assertion would pass
    whether or not the warning was ever emitted.
    """
    records: list[str] = []
    sink_id = logger.add(lambda message: records.append(message.record["message"]), level="WARNING")
    try:
        yield records
    finally:
        logger.remove(sink_id)


@pytest.fixture
def errors() -> Iterator[list[str]]:
    """Captured loguru messages at ERROR only.

    Separate from ``logged`` so "this is logged at ERROR" is provable rather than
    inferred: the level is the difference between a caveat and a broken precondition.
    """
    records: list[str] = []
    sink_id = logger.add(lambda message: records.append(message.record["message"]), level="ERROR")
    try:
        yield records
    finally:
        logger.remove(sink_id)


@pytest.fixture
def cloudwatch() -> Iterator[tuple[Any, Stubber]]:
    client = boto3.client("cloudwatch", region_name="us-east-1")
    stub = Stubber(client)
    stub.activate()
    yield client, stub
    stub.deactivate()


@pytest.fixture
def sagemaker() -> Iterator[tuple[Any, Stubber]]:
    client = boto3.client("sagemaker", region_name="us-east-1")
    stub = Stubber(client)
    stub.activate()
    yield client, stub
    stub.deactivate()


@pytest.fixture
def appscaling() -> Iterator[tuple[Any, Stubber]]:
    client = boto3.client("application-autoscaling", region_name="us-east-1")
    stub = Stubber(client)
    stub.activate()
    yield client, stub
    stub.deactivate()


# --------------------------------------------------------------------------- #
# Builders: measured rungs
# --------------------------------------------------------------------------- #


def _event(
    *,
    seq: int,
    worker_index: int,
    concurrency: int,
    step_index: int,
    dispatch_ts: float,
    end_ts: float,
    ttfab_ms: float | None,
    latency_ms: float,
    outcome: InvokeOutcome,
    run_id: str,
) -> LoadEvent:
    return LoadEvent(
        run_id=run_id,
        step_index=step_index,
        seq=seq,
        worker_index=worker_index,
        concurrency=concurrency,
        model=MODEL,
        endpoint=ENDPOINT,
        dispatch_ts=dispatch_ts,
        first_byte_ts=dispatch_ts + (ttfab_ms or 0.0) / 1000.0,
        end_ts=end_ts,
        ttfab_ms=ttfab_ms,
        latency_ms=latency_ms,
        outcome=outcome.value,
        http_status=200 if outcome is InvokeOutcome.OK else 503,
        error_class=None if outcome is InvokeOutcome.OK else "ClientError",
        error_message=None if outcome is InvokeOutcome.OK else "queue_saturated",
        chars=26,
        audio_bytes=4800,
        audio_duration_s=0.2,
        rtf=0.5,
        in_flight_at_dispatch=concurrency,
        instance_count=1,
    )


def _result(
    *,
    concurrency: int = 5,
    step_index: int = 0,
    window_end_ts: float = T0_TS,
    window_s: float = WINDOW_S,
    hold_s: float = HOLD_S,
    drain_s: float = 0.0,
    per_worker: int = 4,
    ttfab_ms: float = 100.0,
    ttfab_slope_ms_per_s: float = 0.0,
    latency_ms: float = 200.0,
    outcome: InvokeOutcome = InvokeOutcome.OK,
    in_flight: Sequence[int] | None = None,
    instance_counts: Sequence[int] = (1,),
    overlapping: bool = False,
    run_id: str = "run",
) -> StepResult:
    """A closed-loop ``StepResult`` whose trailing window yields chosen statistics.

    Anchored on ``window_end_ts`` rather than on the end of the step, because that is
    where ``run_ladder`` and ``join_cloudwatch`` both anchor: ``min(started + hold_s,
    ended)``. ``drain_s`` extends the step past that point with in-flight falling toward
    zero and slow completions, which is what a window anchored at ``ended_ts`` would pull
    in — and it reports a mean concurrency below ``N`` for a rung that held ``N`` exactly.

    ``per_worker`` requests are issued per worker, sequentially, so ``worker_overlaps`` is
    zero by construction unless ``overlapping`` asks for the falsifying case.
    """
    window_start = window_end_ts - window_s
    started_ts = window_end_ts - hold_s
    ended_ts = window_end_ts + drain_s
    spacing = window_s / max(per_worker, 1)

    events: list[LoadEvent] = []
    seq = 0
    for worker_index in range(concurrency):
        for k in range(per_worker):
            end = window_start + (k + 0.5) * spacing
            events.append(
                _event(
                    seq=seq,
                    worker_index=worker_index,
                    concurrency=concurrency,
                    step_index=step_index,
                    dispatch_ts=end - latency_ms / 1000.0,
                    end_ts=end,
                    ttfab_ms=ttfab_ms + ttfab_slope_ms_per_s * (end - window_start),
                    latency_ms=latency_ms,
                    outcome=outcome,
                    run_id=run_id,
                )
            )
            seq += 1

    if overlapping and events:
        first_end = events[0].end_ts or window_start
        events.append(
            _event(
                seq=seq,
                worker_index=0,
                concurrency=concurrency,
                step_index=step_index,
                dispatch_ts=first_end - latency_ms / 2000.0,
                end_ts=first_end + latency_ms / 2000.0,
                ttfab_ms=ttfab_ms,
                latency_ms=latency_ms,
                outcome=outcome,
                run_id=run_id,
            )
        )
        seq += 1

    levels = list(in_flight) if in_flight is not None else [concurrency] * 12
    samples = [
        ConcurrencySample(
            ts=window_start + (i + 0.5) * window_s / len(levels),
            in_flight=level,
            instance_count=instance_counts[
                min(i * len(instance_counts) // len(levels), len(instance_counts) - 1)
            ],
        )
        for i, level in enumerate(levels)
    ]
    if drain_s > 0:
        # The drain tail: requests still finishing after the hold, on an emptying client.
        for i, level in enumerate(reversed(range(concurrency))):
            samples.append(
                ConcurrencySample(
                    ts=window_end_ts + (i + 0.5) * drain_s / concurrency,
                    in_flight=level,
                    instance_count=instance_counts[-1],
                )
            )
        events.append(
            _event(
                seq=seq,
                worker_index=0,
                concurrency=concurrency,
                step_index=step_index,
                dispatch_ts=window_end_ts,
                end_ts=window_end_ts + drain_s / 2,
                # Ten times the in-window figure: if the drain leaked into the window the
                # p95 would move, not merely wobble.
                ttfab_ms=ttfab_ms * 10,
                latency_ms=latency_ms * 10,
                outcome=outcome,
                run_id=run_id,
            )
        )

    return StepResult(
        run_id=run_id,
        step_index=step_index,
        concurrency=concurrency,
        model=MODEL,
        endpoint=ENDPOINT,
        started_ts=started_ts,
        ended_ts=ended_ts,
        events=events,
        samples=samples,
    )


def _measured(
    result: StepResult,
    *,
    run_index: int = 0,
    slo_ms: int = SLO_MS,
    hold_s: float = HOLD_S,
    window_s: float = WINDOW_S,
) -> StepSummary:
    """Summarize a rung the way ``run_ladder`` does, window anchoring included."""
    window_end = min(result.started_ts + hold_s, result.ended_ts)
    stats = summarize_window(result, start_ts=window_end - window_s, end_ts=window_end)
    return summarize_step(result, stats, run_index=run_index, slo_ms=slo_ms)


def _summary(**overrides: Any) -> StepSummary:
    """A passing, usable rung as the artifact carries it."""
    kwargs: dict[str, Any] = {
        "run_index": 0,
        "step_index": 0,
        "concurrency": 5,
        "achieved_rps": 12.0,
        "completed": 720,
        "ok": 720,
        "chars": 18_000,
        "ttfab_p95_ms": LADDER[5],
        "s_mean_s": 0.10986375146305409,
        "s_p95_s": 0.16457688123919073,
        "concurrency_mean": 5.0,
        "meets_slo": True,
        "saturated": False,
        "settled": True,
        "usable": True,
    }
    kwargs.update(overrides)
    return StepSummary(**kwargs)


def _rungs(
    ladder: dict[int, float],
    *,
    run_index: int = 0,
    slo_ms: int = SLO_MS,
    **overrides: Any,
) -> list[StepSummary]:
    """One pass's summaries from a ``{concurrency: p95_ms}`` table."""
    return [
        _summary(
            run_index=run_index,
            step_index=index,
            concurrency=rung,
            ttfab_p95_ms=p95,
            concurrency_mean=float(rung),
            meets_slo=p95 <= slo_ms,
            **overrides,
        )
        for index, (rung, p95) in enumerate(sorted(ladder.items()))
    ]


def _ladder(steps: Sequence[StepSummary], *, run_index: int = 0, **overrides: Any) -> LadderRun:
    return LadderRun(run_index=run_index, steps=list(steps), **overrides)


def _report(**overrides: Any) -> Any:
    """A report assembled through ``build_report``, so the real assembly is under test."""
    kwargs: dict[str, Any] = {
        "model": MODEL,
        "endpoint": ENDPOINT,
        "instance_type": "ml.g5.xlarge",
        "run_id": "qmax12345678",
        "slo_ms": SLO_MS,
        "frozen": True,
        "unbounded_queue": True,
        "hold_s": HOLD_S,
        "measure_window_s": WINDOW_S,
    }
    kwargs.setdefault("ladders", [_ladder(_rungs({**LADDER, **FAILING_RUNG}))])
    kwargs.update(overrides)
    return build_report(**kwargs)


# --------------------------------------------------------------------------- #
# Builders: AWS doubles
# --------------------------------------------------------------------------- #


def _variant(*, desired: int = 1, current: int = 1) -> dict[str, Any]:
    return {
        "EndpointName": ENDPOINT,
        "EndpointArn": f"arn:aws:sagemaker:us-east-1:1234:endpoint/{ENDPOINT}",
        "EndpointConfigName": f"{ENDPOINT}-config",
        "EndpointStatus": "InService",
        "CreationTime": NOW,
        "LastModifiedTime": NOW,
        "ProductionVariants": [
            {
                "VariantName": VARIANT,
                "DesiredInstanceCount": desired,
                "CurrentInstanceCount": current,
            }
        ],
    }


def _endpoint_config(*, instance_type: str = "ml.g5.xlarge") -> dict[str, Any]:
    return {
        "EndpointConfigName": f"{ENDPOINT}-config",
        "EndpointConfigArn": (
            f"arn:aws:sagemaker:us-east-1:1234:endpoint-config/{ENDPOINT}-config"
        ),
        "ProductionVariants": [
            {
                "VariantName": VARIANT,
                "ModelName": "kokoro-model",
                "InstanceType": instance_type,
                "InitialInstanceCount": 1,
            }
        ],
        "CreationTime": NOW,
    }


def _model(*, env: dict[str, str] | None = None) -> dict[str, Any]:
    return {
        "ModelName": "kokoro-model",
        "ModelArn": "arn:aws:sagemaker:us-east-1:1234:model/kokoro-model",
        "CreationTime": NOW,
        "PrimaryContainer": {
            "Image": "1234.dkr.ecr.us-east-1.amazonaws.com/cdk-assets:139b9068c5eb1f03",
            "Environment": {"MAX_REQUEST_AGE_S": "56"} if env is None else env,
        },
    }


def _scalable_target(*, suspended: bool) -> dict[str, Any]:
    return {
        "ServiceNamespace": SERVICE_NAMESPACE,
        "ResourceId": RID,
        "ScalableDimension": SCALABLE_DIMENSION,
        "MinCapacity": 1,
        "MaxCapacity": 4,
        "RoleARN": "arn:aws:iam::1234:role/aws-service-role/sagemaker.application-autoscaling",
        "SuspendedState": {
            "DynamicScalingInSuspended": suspended,
            "DynamicScalingOutSuspended": suspended,
            "ScheduledScalingSuspended": suspended,
        },
        "CreationTime": NOW,
    }


def _queue_fingerprint(
    sm_stub: Stubber,
    *,
    instance_type: str = "ml.g5.xlarge",
    env: dict[str, str] | None = None,
) -> None:
    """The three describes behind one configuration fingerprint, in order."""
    sm_stub.add_response("describe_endpoint", _variant())
    sm_stub.add_response("describe_endpoint_config", _endpoint_config(instance_type=instance_type))
    sm_stub.add_response("describe_model", _model(env=env))


def _queue_capture(aas_stub: Stubber, sm_stub: Stubber, *, suspended: bool) -> None:
    """The three calls one ``fixture.capture()`` makes, in order."""
    aas_stub.add_response(
        "describe_scalable_targets", {"ScalableTargets": [_scalable_target(suspended=suspended)]}
    )
    aas_stub.add_response(
        "describe_scaling_policies",
        {
            "ScalingPolicies": [
                {
                    "PolicyARN": f"arn:aws:autoscaling:us-east-1:1234:scalingPolicy:a:resource/{RID}",
                    "PolicyName": "TrackConcurrency",
                    "ServiceNamespace": SERVICE_NAMESPACE,
                    "ResourceId": RID,
                    "ScalableDimension": SCALABLE_DIMENSION,
                    "PolicyType": "TargetTrackingScaling",
                    "CreationTime": NOW,
                }
            ]
        },
    )
    sm_stub.add_response("describe_endpoint", _variant())


def _queue_suspend(aas_stub: Stubber, *, state: dict[str, bool]) -> None:
    aas_stub.add_response(
        "register_scalable_target",
        {},
        {
            "ServiceNamespace": SERVICE_NAMESPACE,
            "ResourceId": RID,
            "ScalableDimension": SCALABLE_DIMENSION,
            "SuspendedState": state,
        },
    )


def _queue_freeze(aas_stub: Stubber, sm_stub: Stubber, *, suspends: bool = True) -> None:
    """Queue ``freeze`` + ``require_frozen`` for an endpoint already at one instance.

    ``suspends=False`` leaves scale-out active after the suspend attempt, which is the
    state the guard exists to refuse.
    """
    _queue_capture(aas_stub, sm_stub, suspended=False)
    _queue_suspend(aas_stub, state=dict(SUSPEND_ALL))
    sm_stub.add_response("describe_endpoint", _variant())  # _pin_capacity: already pinned
    _queue_capture(aas_stub, sm_stub, suspended=suspends)  # freeze's own verification
    if suspends:
        _queue_capture(aas_stub, sm_stub, suspended=True)  # require_frozen, before any load


def _queue_thaw(aas_stub: Stubber, sm_stub: Stubber) -> None:
    """The restore: the captured state, not a blanket resume, and capacity left alone."""
    _queue_suspend(
        aas_stub,
        state={
            "DynamicScalingInSuspended": False,
            "DynamicScalingOutSuspended": False,
            "ScheduledScalingSuspended": False,
        },
    )
    sm_stub.add_response("describe_endpoint", _variant())


def _queue_window(
    stub: Stubber,
    *,
    concurrency_avg: Sequence[float] = (2.0, 3.0),
    concurrency_max: Sequence[float] = (9.0, 15.0),
    model_latency_p95_micros: float | None = 250_000.0,
    errors_5xx: float | None = 3.0,
    cpu: float | None = 0.0,
    gpu: float | None = 71.0,
) -> None:
    """Queue the ten ``get_metric_statistics`` replies one ``fetch_window`` makes.

    Ten, in ``INVOCATION_METRICS + UTILIZATION_METRICS`` order — so a spec added to
    ``observe.py`` without a reply here fails as a pending-response error rather than
    silently reading the next rung's window.

    ``concurrency_avg`` and ``concurrency_max`` are separate sequences on purpose: they
    are the two statistics of one metric, and the gap between them is what the deployed
    threshold's units depend on.
    """
    stub.add_response(
        "get_metric_statistics",
        {
            "Datapoints": [
                {"Timestamp": NOW + timedelta(seconds=10 * i), "Average": avg, "Maximum": mx}
                for i, (avg, mx) in enumerate(
                    zip(concurrency_avg, concurrency_max, strict=True)
                )
            ]
        },
    )
    stub.add_response(
        "get_metric_statistics",
        {
            "Datapoints": (
                []
                if model_latency_p95_micros is None
                else [
                    {
                        "Timestamp": NOW,
                        "Average": model_latency_p95_micros,
                        "ExtendedStatistics": {
                            "p95": model_latency_p95_micros,
                            "p99": model_latency_p95_micros,
                        },
                    }
                ]
            )
        },
    )
    stub.add_response("get_metric_statistics", {"Datapoints": []})  # OverheadLatency
    stub.add_response("get_metric_statistics", {"Datapoints": [{"Timestamp": NOW, "Sum": 600.0}]})
    stub.add_response(
        "get_metric_statistics",
        {"Datapoints": [] if errors_5xx is None else [{"Timestamp": NOW, "Sum": errors_5xx}]},
    )
    stub.add_response("get_metric_statistics", {"Datapoints": []})  # Invocation4XXErrors
    stub.add_response(
        "get_metric_statistics",
        {"Datapoints": [] if cpu is None else [{"Timestamp": NOW, "Average": cpu, "Maximum": cpu}]},
    )
    stub.add_response("get_metric_statistics", {"Datapoints": []})  # MemoryUtilization
    stub.add_response(
        "get_metric_statistics",
        {"Datapoints": [] if gpu is None else [{"Timestamp": NOW, "Average": gpu, "Maximum": gpu}]},
    )
    stub.add_response("get_metric_statistics", {"Datapoints": []})  # GPUMemoryUtilization


def _record_metric_calls(client: Any, calls: list[dict[str, Any]]) -> None:
    """Record the parameters of every ``GetMetricStatistics`` call.

    The Stubber still validates each request and response; this only captures the
    *windows asked for*, which is the claim two of these tests make and which a queue of
    responses cannot express.
    """
    client.meta.events.register(
        "before-parameter-build.cloudwatch.GetMetricStatistics",
        lambda params, **_: calls.append(dict(params)),
    )


class RecordingClock:
    """A ``Clock`` that records sleeps instead of taking them."""

    def __init__(self) -> None:
        self.slept: list[float] = []

    def clock(self) -> Clock:
        return Clock(monotonic=time.monotonic, sleep=self.slept.append, time=time.time)


class _NoInvocations:
    """A runtime client that fails the test if load is ever sent through it."""

    def __init__(self) -> None:
        self.calls = 0

    def invoke_endpoint_with_response_stream(self, **kwargs: Any) -> None:
        self.calls += 1
        raise AssertionError("load was sent despite the guard")


class FakeServer:
    """A serial server with a fixed service time, standing in for kokoro's lock.

    ``slots=1`` means only one request is ever *executing*, so anything else the client
    holds outstanding is queued — the ``Q + E = N`` split the ladder measures.
    """

    def __init__(self, *, service_time_s: float = 0.02, slots: int = 1) -> None:
        self._sem = threading.Semaphore(slots)
        self._service_time_s = service_time_s
        self._lock = threading.Lock()
        self._active = 0
        self.completed = 0
        self.max_observed_concurrency = 0

    def __call__(
        self, client: Any, endpoint: str, text: str, voice: str, *, deadline_ts: float | None = None
    ) -> InvokeResult:
        dispatch_ts = time.time()
        with self._sem:
            with self._lock:
                self._active += 1
                self.max_observed_concurrency = max(self.max_observed_concurrency, self._active)
            time.sleep(self._service_time_s)
            with self._lock:
                self._active -= 1
                self.completed += 1
        end_ts = time.time()
        return InvokeResult(
            outcome=InvokeOutcome.OK,
            dispatch_ts=dispatch_ts,
            end_ts=end_ts,
            latency_ms=(end_ts - dispatch_ts) * 1000.0,
            first_byte_ts=dispatch_ts + self._service_time_s / 2,
            ttfab_ms=self._service_time_s * 500.0,
            chars=len(text),
            audio_bytes=4800,
            audio_duration_s=0.2,
            http_status=200,
        )


class _RecordingRunner:
    """Records how each rung was launched and returns a scripted result."""

    def __init__(
        self,
        *,
        saturated_at: Sequence[int] = (),
        window_s: float = WINDOW_S,
        hold_s: float = HOLD_S,
    ) -> None:
        self._saturated_at = set(saturated_at)
        self._window_s = window_s
        self._hold_s = hold_s
        self.calls: list[dict[str, Any]] = []

    def __call__(self, client: Any, **kwargs: Any) -> StepResult:
        self.calls.append(kwargs)
        index = kwargs["step_index"]
        return _result(
            concurrency=kwargs["concurrency"],
            step_index=index,
            window_end_ts=T0_TS + index * 1000.0,
            window_s=self._window_s,
            hold_s=self._hold_s,
            outcome=(
                InvokeOutcome.SATURATED_503
                if index in self._saturated_at
                else InvokeOutcome.OK
            ),
            run_id=kwargs["run_id"],
        )


def _run_ladder(runner: _RecordingRunner, *, clock: Clock | None = None, **kwargs: Any) -> LadderRun:
    defaults: dict[str, Any] = {
        "model": MODEL,
        "endpoint": ENDPOINT,
        "voice": "af_heart",
        "texts": TEXTS,
        "slo_ms": SLO_MS,
        "concurrencies": (1, 5, 10),
        "hold_s": HOLD_S,
        "measure_window_s": WINDOW_S,
        "settle_between_steps_s": 0.0,
        "step_runner": runner,
    }
    defaults.update(kwargs)
    if clock is not None:
        defaults["clock"] = clock
    return run_ladder(None, **defaults)


def _measure(**kwargs: Any) -> Any:
    defaults: dict[str, Any] = {
        "model": MODEL,
        "texts": TEXTS,
        "slo_ms": SLO_MS,
        "concurrencies": (1, 5, 10),
        "cloudwatch_join": False,
        "require_frozen": False,
        "runtime_client": _NoInvocations(),
        "clock": SYSTEM_CLOCK,
    }
    defaults.update(kwargs)
    return measure(**defaults)


def _patch_ladder(monkeypatch: pytest.MonkeyPatch, *, raises: BaseException | None = None) -> list:
    """Replace the ladder so ``measure``'s orchestration can be tested on its own."""
    recorded: list[dict[str, Any]] = []

    def fake_run_ladder(client: Any, **kwargs: Any) -> LadderRun:
        recorded.append(kwargs)
        if raises is not None:
            raise raises
        return _ladder(_rungs(LADDER, run_index=kwargs["run_index"]), run_index=kwargs["run_index"])

    monkeypatch.setattr(qmax_mod, "run_ladder", fake_run_ladder)
    return recorded


# --------------------------------------------------------------------------- #
# Sizing
# --------------------------------------------------------------------------- #


class TestPoolSize:
    def test_clears_the_highest_rung_with_headroom(self) -> None:
        # A pool at exactly N lets urllib3 queue a connection, and a rung measured
        # against a queued connection looks exactly like a real SLO crossing.
        assert pool_size(60) == 60 + POOL_HEADROOM
        assert pool_size(60) > 60

    def test_needs_no_overshoot_allowance(self) -> None:
        # Unlike the open-loop version: closed-loop in-flight cannot exceed N, so the
        # size follows from the ladder rather than from an estimate of overshoot.
        assert pool_size(1) == 1 + POOL_HEADROOM

    @pytest.mark.parametrize("bad", [0, -1])
    def test_rejects_a_ladder_with_no_rung_to_size_against(self, bad: int) -> None:
        with pytest.raises(ValueError, match="max_concurrency must be >= 1"):
            pool_size(bad)


class TestTotalDuration:
    def test_settles_between_rungs_and_not_after_the_last(self) -> None:
        assert total_duration_s(
            concurrencies=(1, 5, 10), hold_s=100.0, settle_between_steps_s=10.0
        ) == pytest.approx(320.0)

    def test_a_single_rung_needs_no_drain(self) -> None:
        assert total_duration_s(
            concurrencies=(5,), hold_s=100.0, settle_between_steps_s=10.0
        ) == pytest.approx(100.0)

    def test_repeated_rungs_are_one_rung(self) -> None:
        # The ladder dedupes, so an estimate that counted them twice would over-state
        # the cost of the run people are deciding whether to commit to.
        assert total_duration_s(
            concurrencies=(5, 5, 10), hold_s=100.0, settle_between_steps_s=0.0
        ) == pytest.approx(200.0)

    def test_runs_multiply_the_whole_ladder(self) -> None:
        one = total_duration_s(concurrencies=(1, 5), hold_s=100.0, settle_between_steps_s=10.0)
        assert total_duration_s(
            concurrencies=(1, 5), hold_s=100.0, settle_between_steps_s=10.0, runs=3
        ) == pytest.approx(one * 3)


class TestDryRunPlan:
    def test_one_row_per_rung_per_run(self) -> None:
        plan = dry_run_plan(concurrencies=(1, 5, 10), hold_s=240.0, runs=2)
        assert [(int(r["run_index"]), int(r["concurrency"])) for r in plan] == [
            (0, 1),
            (0, 5),
            (0, 10),
            (1, 1),
            (1, 5),
            (1, 10),
        ]

    def test_rungs_are_sorted_and_deduplicated_as_the_ladder_will_walk_them(self) -> None:
        plan = dry_run_plan(concurrencies=(10, 1, 10, 5), hold_s=240.0)
        assert [int(r["step_index"]) for r in plan] == [0, 1, 2]
        assert [int(r["concurrency"]) for r in plan] == [1, 5, 10]

    def test_expected_requests_is_absent_without_a_service_time(self) -> None:
        # The difference from the open-loop version: a wrong S mis-predicts the count
        # without moving a single rung, so the forecast is optional and the rung is not.
        assert all("expected_requests" not in row for row in dry_run_plan(concurrencies=(5,)))

    def test_expected_requests_follows_from_the_service_time(self) -> None:
        # N/S completions per second, so a rung of N held for hold_s: a forecast derived
        # from the closed loop rather than a schedule driving it.
        row = dry_run_plan(concurrencies=(10,), hold_s=240.0, s_mean_s=0.25)[0]
        assert row["expected_requests"] == pytest.approx(10 * 240.0 / 0.25)

    @pytest.mark.parametrize("s_mean_s", [0.0, -0.25])
    def test_a_non_positive_service_time_forecasts_nothing(self, s_mean_s: float) -> None:
        # Rather than dividing by zero on the one path whose whole purpose is to be
        # runnable before any measurement exists.
        assert "expected_requests" not in dry_run_plan(concurrencies=(5,), s_mean_s=s_mean_s)[0]


# --------------------------------------------------------------------------- #
# Which rungs may inform Q_max
# --------------------------------------------------------------------------- #


class TestUnusableRungsAreExcludedNotFlagged:
    """Each of these looks like a healthy rung from the latency percentiles alone.

    So exclusion happens at ``summarize_step``, where ``meets_slo`` is decided, rather
    than being left to a reader of the artifact: a rung that cannot be trusted as a pass
    must not be able to *become* ``Q_max``.
    """

    def test_a_clean_rung_is_usable(self) -> None:
        step = _measured(_result())
        assert step.usable, step.unusable_reason
        assert step.unusable_reason is None
        assert step.meets_slo

    def test_a_mid_step_resize_is_not_a_per_instance_measurement(self) -> None:
        # N outstanding requests spread over two instances is a different measurement,
        # and it reads as *better* latency rather than as an error.
        step = _measured(_result(instance_counts=(1, 2)))
        assert not step.usable
        assert step.capacity_changed
        assert "instance count changed mid-step" in (step.unusable_reason or "")
        assert not step.meets_slo

    def test_a_window_with_no_completions_measured_nothing(self) -> None:
        step = _measured(_result(per_worker=0))
        assert not step.usable
        assert "no requests completed" in (step.unusable_reason or "")

    def test_a_worker_that_overlapped_itself_falsifies_the_concurrency(self) -> None:
        # The driver's one structural guarantee. If a worker dispatched before its
        # previous request finished, outstanding requests exceeded N and the rung's
        # concurrency is not the number it claims — which is the number being measured.
        step = _measured(_result(concurrency=5, overlapping=True))
        assert not step.usable
        assert "overlapping request(s) on a single worker" in (step.unusable_reason or "")
        assert "concurrency of 5" in (step.unusable_reason or "")

    def test_a_client_limited_rung_names_the_shortfall(self) -> None:
        # The closed-loop replacement for dispatch_skipped: our own dispatch overhead,
        # not the server's queue, set the latency. Excluded because it looks like a pass.
        step = _measured(_result(concurrency=5, in_flight=(1,) * 12))
        assert not step.usable
        assert step.client_bound
        assert "80% below N=5" in (step.unusable_reason or "")

    def test_a_rung_that_never_settled_is_a_moving_number(self) -> None:
        # On a serial server a queue still filling grows latency without bound, so the
        # window's p95 is a snapshot rather than the SLO verdict at this concurrency.
        step = _measured(_result(ttfab_ms=100.0, ttfab_slope_ms_per_s=16.0))
        assert not step.settled
        assert not step.usable
        assert "still climbing" in (step.unusable_reason or "")

    def test_the_reason_is_the_one_the_operator_can_act_on(self) -> None:
        # Ordering, not a set: a rung that both resized and shed work is a capacity
        # problem, and reporting the shedding would send the operator to the container.
        step = _measured(_result(instance_counts=(1, 2), outcome=InvokeOutcome.SATURATED_503))
        assert "instance count changed mid-step" in (step.unusable_reason or "")


class TestRejectionsCountAsSaturation:
    """Regression lock on the saturation rule that shipped.

    ``saturated`` was ``achieved_rps < 0.95 * offered_rps`` while ``achieved_rps`` counted
    any event carrying an ``end_ts`` — so a 1 ms 503 counted as a completion and *raised*
    measured throughput. A fully-shedding container therefore read as a healthy one, at
    an excellent p95, because the rejections never entered the latency percentiles.

    This has to hold before any queue bound is deployed. Enforcing ``Q_max`` at the
    instance is the point of measuring it, and the moment a bound exists every rung above
    it sheds: if shedding still reads as health, the ladder measures the bound and reports
    it as the concurrency at which the SLO breaks.
    """

    def test_a_rung_of_rejections_is_saturated(self) -> None:
        step = _measured(_result(concurrency=20, outcome=InvokeOutcome.SATURATED_503))
        assert step.saturated
        assert step.rejected == step.completed
        assert step.achieved_rps > 0

    def test_it_is_excluded_as_a_measurement(self) -> None:
        step = _measured(_result(concurrency=20, outcome=InvokeOutcome.SATURATED_503))
        assert not step.usable
        assert not step.meets_slo
        assert "the queue was supposed to be unbounded" in (step.unusable_reason or "")

    def test_the_rejections_are_named_in_the_outcome_table(self) -> None:
        # Named, not summed into an error count: a queue rejection and a model crash
        # call for opposite responses, and only the outcome tells them apart.
        step = _measured(_result(concurrency=20, per_worker=3, outcome=InvokeOutcome.SATURATED_503))
        assert step.outcome_counts == {"saturated_503": 60}
        assert step.ok == 0

    def test_no_p95_survives_a_rung_that_only_shed(self) -> None:
        # Percentiles are over successes only. Mixing a fast rejection in would pull p95
        # *down* under overload, turning saturation into an apparent improvement.
        step = _measured(_result(outcome=InvokeOutcome.SATURATED_503))
        assert step.ttfab_p95_ms is None

    def test_the_report_names_the_rung_and_logs_it_as_an_error(
        self, errors: list[str], logged: list[str]
    ) -> None:
        # ERROR rather than WARNING: against an unbounded queue nothing should be able to
        # refuse work, so this is a broken precondition and not a caveat on the answer.
        shed = _measured(_result(concurrency=20, outcome=InvokeOutcome.SATURATED_503))
        report = _report(ladders=[_ladder([*_rungs(LADDER), shed])])
        assert report.saturated_rungs == [20]
        assert any("saw the server refuse work" in m for m in errors)
        assert any("MAX_QUEUE_DEPTH" in m for m in logged)

    def test_a_shedding_rung_cannot_become_q_max(self) -> None:
        # The failure the old rule allowed: an excellent p95 over a shed filter, at the
        # highest rung on the ladder, adopted as the admission bound.
        shed = _summary(
            concurrency=60,
            step_index=5,
            ttfab_p95_ms=41.0,
            saturated=True,
            usable=False,
            meets_slo=False,
            unusable_reason="completions were rejections",
        )
        q_max, _ = find_q_max([*_rungs(LADDER), shed])
        assert q_max == 50


class TestSummarizeStep:
    def test_the_verdict_is_against_the_slo_it_was_judged_on(self) -> None:
        # Decided here rather than downstream, so the artifact carries the verdict and
        # the line together and cannot be re-read against a different promise.
        result = _result(ttfab_ms=2000.0)
        assert _measured(result, slo_ms=3000).meets_slo
        assert not _measured(result, slo_ms=1000).meets_slo

    def test_a_rung_exactly_on_the_line_passes(self) -> None:
        # `<=`, not `<`. A p95 equal to the SLO met the promise.
        assert _measured(_result(ttfab_ms=float(SLO_MS)), slo_ms=SLO_MS).meets_slo

    def test_an_unusable_rung_never_meets_the_slo(self) -> None:
        # Whatever its p95 reads: Q_max is the highest rung that both passed *and* was a
        # valid measurement, and folding that into one flag is what stops the two being
        # confused by a later reader.
        step = _measured(_result(ttfab_ms=50.0, in_flight=(1,) * 12))
        assert step.ttfab_p95_ms == pytest.approx(50.0)
        assert not step.meets_slo

    def test_it_stamps_the_run_it_belongs_to(self) -> None:
        assert _measured(_result(), run_index=2).run_index == 2

    def test_it_carries_both_concurrency_statistics(self) -> None:
        # The mean is what the ladder judges; the peak is recorded beside it because the
        # deployed alarm reads a *Maximum*, and the two diverge by up to 9.8x.
        step = _measured(_result(concurrency=10, in_flight=(10, 10, 10, 14)))
        assert step.concurrency_mean == pytest.approx(11.0)
        assert step.concurrency_peak == 14

    def test_it_carries_the_drift_the_settling_verdict_came_from(self) -> None:
        step = _measured(_result(ttfab_slope_ms_per_s=16.0))
        assert step.ttfab_drift_ms is not None
        assert step.ttfab_drift_ms > 0
        assert step.instance_counts == (1,)


# --------------------------------------------------------------------------- #
# The ladder
# --------------------------------------------------------------------------- #


class TestFindQMax:
    def test_the_highest_passing_rung_is_the_answer(self) -> None:
        q_max, _ = find_q_max(_rungs({**LADDER, **FAILING_RUNG}))
        assert q_max == 50

    def test_a_rung_above_that_failed_brackets_it(self) -> None:
        _, bracketed = find_q_max(_rungs({**LADDER, **FAILING_RUNG}))
        assert bracketed

    def test_a_ladder_that_ran_out_while_passing_is_only_a_lower_bound(self) -> None:
        # The distinction the artifact exists to carry: 50 is where we stopped looking,
        # not where the endpoint stopped meeting the SLO.
        q_max, bracketed = find_q_max(_rungs(LADDER))
        assert q_max == 50
        assert not bracketed

    def test_one_noisy_rung_does_not_truncate_the_ladder(self) -> None:
        # The highest passing rung, not the one below the first failure. A single rung
        # over the line mid-ladder is noise; the crossing is where it stays over.
        q_max, bracketed = find_q_max(_rungs({1: 92.0, 5: 3400.0, 10: 667.0}))
        assert q_max == 10
        assert not bracketed

    def test_nothing_passing_is_reported_as_nothing(self) -> None:
        # The honest answer for a ladder whose lowest rung is already past the line, and
        # not something to substitute a default for.
        assert find_q_max(_rungs({1: 4000.0, 5: 5000.0})) == (None, False)

    def test_an_excluded_rung_past_the_slo_still_brackets(self) -> None:
        # Exclusion says the rung cannot be trusted as a *pass*. A p95 well past the SLO
        # there is still a fact about the endpoint, and dropping it is what makes a
        # ladder that hit the wall report its answer as a lower bound.
        over = _summary(
            concurrency=60, step_index=5, ttfab_p95_ms=9000.0, usable=False, meets_slo=False
        )
        q_max, bracketed = find_q_max([*_rungs(LADDER), over])
        assert q_max == 50
        assert bracketed

    def test_an_unmeasured_rung_above_cannot_bracket(self) -> None:
        # No p95 means nothing was learned there, so it cannot stand in for a measured
        # SLO miss. This is the path a truncated ladder takes.
        blank = _summary(
            concurrency=60, step_index=5, ttfab_p95_ms=None, usable=False, meets_slo=False
        )
        assert find_q_max([*_rungs(LADDER), blank]) == (50, False)

    @pytest.mark.xfail(
        reason=(
            "DEFECT qmax.py:235-241: bracketing tests `not meets_slo`, which is true for "
            "any *excluded* rung whatever its p95, so a rung above Q_max that was thrown "
            "out as a measurement while comfortably inside the SLO sets "
            "q_max_bracketed=True. The lower-bound warning (qmax.py:678) and the CLI's "
            "'NOTE: LOWER BOUND' are then both suppressed for a ladder that never "
            "measured a crossing, and both derived thresholds inherit the bound "
            "silently. The predicate needs the p95 to be past the SLO, which means "
            "find_q_max needs slo_ms — StepSummary does not carry it."
        ),
        strict=True,
    )
    def test_an_excluded_rung_inside_the_slo_does_not_bracket(self) -> None:
        client_bound = _summary(
            concurrency=60,
            step_index=5,
            ttfab_p95_ms=400.0,
            concurrency_mean=12.0,
            client_bound=True,
            usable=False,
            meets_slo=False,
            unusable_reason="mean in-flight was 80% below N=60",
        )
        _, bracketed = find_q_max([*_rungs(LADDER), client_bound])
        assert not bracketed


class TestServiceTimeAtLowestRung:
    def test_it_comes_from_the_lowest_rung_not_from_q_max(self) -> None:
        # Service time at Q_max already contains queueing, and W_max = SLO - S_p95 would
        # then subtract a wait it had already counted.
        steps = [
            _summary(concurrency=1, s_mean_s=0.11, s_p95_s=0.16),
            _summary(concurrency=50, step_index=1, s_mean_s=2.9, s_p95_s=3.1),
        ]
        assert service_time_at_lowest_rung(steps) == (pytest.approx(0.11), pytest.approx(0.16))

    def test_an_excluded_lowest_rung_is_skipped(self) -> None:
        steps = [
            _summary(concurrency=1, s_mean_s=0.02, s_p95_s=0.03, usable=False),
            _summary(concurrency=5, step_index=1, s_mean_s=0.11, s_p95_s=0.16),
        ]
        assert service_time_at_lowest_rung(steps) == (pytest.approx(0.11), pytest.approx(0.16))

    def test_the_percentile_is_never_below_the_mean(self) -> None:
        # QMaxReport refuses s_p95_s < s_mean_s, so a rung whose percentiles disagree
        # would otherwise lose the whole run at serialization time.
        steps = [_summary(concurrency=1, s_mean_s=0.11, s_p95_s=0.05)]
        assert service_time_at_lowest_rung(steps) == (pytest.approx(0.11), pytest.approx(0.11))

    def test_no_rung_with_a_service_time_is_none(self) -> None:
        assert service_time_at_lowest_rung([_summary(s_mean_s=None)]) is None
        assert service_time_at_lowest_rung([]) is None


class TestRunLadder:
    def test_rejects_a_window_larger_than_the_hold(self) -> None:
        # The window is the trailing part of a rung, not an addition to it.
        with pytest.raises(ValueError, match="must not exceed hold_s"):
            _run_ladder(_RecordingRunner(), hold_s=60.0, measure_window_s=120.0)

    def test_rejects_an_empty_ladder(self) -> None:
        with pytest.raises(ValueError, match="concurrencies must not be empty"):
            _run_ladder(_RecordingRunner(), concurrencies=())

    @pytest.mark.parametrize("bad", [0, -5])
    def test_rejects_a_rung_that_is_not_a_count_of_requests(self, bad: int) -> None:
        with pytest.raises(ValueError, match="every concurrency must be >= 1"):
            _run_ladder(_RecordingRunner(), concurrencies=(1, bad))

    def test_holds_each_rung_at_the_concurrency_it_was_given(self) -> None:
        # No conversion: N is the independent variable, so the value asked for is the
        # value held. Every units defect on this path came from converting it to a rate.
        runner = _RecordingRunner()
        _run_ladder(runner, concurrencies=(1, 5, 10))
        assert [call["concurrency"] for call in runner.calls] == [1, 5, 10]
        assert all("offered_rps" not in call for call in runner.calls)

    def test_walks_upward_whatever_order_it_was_given(self) -> None:
        # Ascending so each rung inherits at most the previous rung's backlog, never a
        # heavier one's.
        runner = _RecordingRunner()
        _run_ladder(runner, concurrencies=(10, 1, 5))
        assert [call["concurrency"] for call in runner.calls] == [1, 5, 10]

    def test_a_repeated_rung_is_run_once(self) -> None:
        runner = _RecordingRunner()
        _run_ladder(runner, concurrencies=(5, 5, 10))
        assert [call["concurrency"] for call in runner.calls] == [5, 10]

    def test_each_rung_is_held_for_the_whole_hold(self) -> None:
        runner = _RecordingRunner()
        _run_ladder(runner, hold_s=240.0)
        assert {call["duration_s"] for call in runner.calls} == {240.0}

    def test_it_forwards_the_transport_and_the_pool_to_every_rung(self) -> None:
        # The ladder is what carries the transport down; binding it at the call site
        # would hide a break in that forwarding, and a Q_max does not transfer between
        # the two protocols.
        runner = _RecordingRunner()
        sentinel = object()
        _run_ladder(runner, invoke=sentinel, texts=TEXTS, voice="af_bella")
        assert {call["invoke"] for call in runner.calls} == {sentinel}
        assert {call["voice"] for call in runner.calls} == {"af_bella"}
        assert all(call["texts"] == TEXTS for call in runner.calls)

    def test_it_gives_every_rung_the_same_run_id(self) -> None:
        runner = _RecordingRunner()
        _run_ladder(runner, run_id="abc123")
        assert {call["run_id"] for call in runner.calls} == {"abc123"}

    def test_it_stamps_the_run_index_on_every_summary(self) -> None:
        ladder = _run_ladder(_RecordingRunner(), run_index=2)
        assert [step.run_index for step in ladder.steps] == [2, 2, 2]
        assert ladder.run_index == 2

    def test_it_measures_only_the_trailing_window(self) -> None:
        # Anchored off started_ts, not ended_ts: run_step lets in-flight requests finish
        # past the duration, so a window anchored at the end of the rung reaches into a
        # drain where in-flight falls toward zero — and then reports a mean concurrency
        # below N for a rung that held N perfectly.
        runner = _RecordingRunner()

        def draining(client: Any, **kwargs: Any) -> StepResult:
            runner.calls.append(kwargs)
            return _result(
                concurrency=kwargs["concurrency"],
                step_index=kwargs["step_index"],
                hold_s=kwargs["duration_s"],
                drain_s=100.0,
                run_id=kwargs["run_id"],
            )

        ladder = _run_ladder(runner, step_runner=draining, concurrencies=(10,))
        step = ladder.steps[0]
        assert step.usable, step.unusable_reason
        assert step.concurrency_mean == pytest.approx(10.0)
        assert step.ttfab_p95_ms == pytest.approx(100.0)

    def test_it_drains_between_rungs_but_not_after_the_last(self) -> None:
        # Without the drain a rung inherits the previous rung's backlog and crosses the
        # SLO for the wrong reason. After the last one it would only cost wall clock.
        recorder = RecordingClock()
        _run_ladder(
            _RecordingRunner(),
            clock=recorder.clock(),
            concurrencies=(1, 5, 10),
            settle_between_steps_s=7.0,
        )
        assert recorder.slept == [7.0, 7.0]

    def test_it_does_not_stop_at_the_first_slo_miss(self) -> None:
        # One rung past the crossing is what brackets Q_max from above.
        runner = _RecordingRunner()

        def over_slo(client: Any, **kwargs: Any) -> StepResult:
            runner.calls.append(kwargs)
            return _result(
                concurrency=kwargs["concurrency"],
                step_index=kwargs["step_index"],
                ttfab_ms=4000.0 if kwargs["concurrency"] >= 5 else 92.0,
                run_id=kwargs["run_id"],
            )

        ladder = _run_ladder(runner, step_runner=over_slo, concurrencies=(1, 5, 10))
        assert [step.meets_slo for step in ladder.steps] == [True, False, False]
        assert ladder.truncated_at is None

    def test_it_stops_after_two_consecutive_saturated_rungs(self) -> None:
        # Against an unbounded queue nothing should be refusing work, so higher rungs
        # would only re-measure whatever is shedding — at hold_s a rung.
        ladder = _run_ladder(
            _RecordingRunner(saturated_at=(1, 2)), concurrencies=(1, 5, 10, 20)
        )
        assert ladder.truncated_at == 2
        assert [step.concurrency for step in ladder.steps] == [1, 5, 10]

    def test_a_single_saturated_rung_does_not_stop_the_ladder(self) -> None:
        ladder = _run_ladder(_RecordingRunner(saturated_at=(1,)), concurrencies=(1, 5, 10, 20))
        assert ladder.truncated_at is None
        assert len(ladder.steps) == 4

    def test_a_truncated_ladder_does_not_drain_on_the_way_out(self) -> None:
        recorder = RecordingClock()
        ladder = _run_ladder(
            _RecordingRunner(saturated_at=(0, 1)),
            clock=recorder.clock(),
            concurrencies=(1, 5, 10),
            settle_between_steps_s=7.0,
        )
        assert ladder.truncated_at == 1
        assert recorder.slept == [7.0]

    def test_it_keeps_the_raw_results_beside_the_summaries(self) -> None:
        # The CloudWatch join needs the timestamps, which only the raw result carries.
        ladder = _run_ladder(_RecordingRunner())
        assert len(ladder.results) == len(ladder.steps) == 3
        assert ladder.instance_counts == (1,)

    def test_an_unusable_rung_is_said_out_loud_while_the_ladder_continues(
        self, logged: list[str]
    ) -> None:
        _run_ladder(_RecordingRunner(saturated_at=(0,)), concurrencies=(1, 5))
        assert any("the queue was supposed to be unbounded" in m for m in logged)


class TestRunLadderAgainstAFakeServer:
    """One pass through the real ``run_step``, to keep the two wired together.

    Asserts shape and the closed-loop split rather than rates: the numbers a half-second
    rung produces are timing-dependent, and the ``Q + E = N`` property they would be
    testing is already covered in ``test_loadgen.py``.
    """

    def test_it_drives_the_real_step_runner(self) -> None:
        server = FakeServer(service_time_s=0.02, slots=1)
        ladder = run_ladder(
            None,
            model=MODEL,
            endpoint=ENDPOINT,
            voice="af_heart",
            texts=TEXTS,
            slo_ms=SLO_MS,
            concurrencies=(2,),
            hold_s=0.4,
            measure_window_s=0.4,
            settle_between_steps_s=0.0,
            # `invoke` goes through run_ladder's own parameter rather than the partial:
            # the ladder forwards the transport to every rung, and binding it here would
            # hide a break in that forwarding.
            invoke=server,
            step_runner=functools.partial(run_step, monitor_interval_s=0.01),
        )

        assert len(ladder.steps) == 1
        step = ladder.steps[0]
        assert step.concurrency == 2
        assert step.completed >= 1
        assert step.concurrency_mean is not None
        assert server.completed >= 1
        # Two outstanding, one executing: the queue is the gap between the client's view
        # and the server's, which is the quantity this ladder walks.
        assert server.max_observed_concurrency == 1
        assert ladder.results[0].worker_overlaps == 0


# --------------------------------------------------------------------------- #
# CloudWatch join
# --------------------------------------------------------------------------- #


def _joinable(
    *, window_end_ts: float, run_index: int = 0, step_index: int = 0, concurrency: int = 5
) -> tuple[StepSummary, tuple[int, StepResult]]:
    """One rung ready to join, as ``(summary, (run_index, result))``."""
    result = _result(
        concurrency=concurrency,
        step_index=step_index,
        window_end_ts=window_end_ts,
        run_id=f"run-r{run_index}",
    )
    return _measured(result, run_index=run_index), (run_index, result)


def _join(client: Any, steps: Sequence[StepSummary], results: Sequence, **kwargs: Any) -> list:
    defaults: dict[str, Any] = {
        "endpoint": ENDPOINT,
        "variant": VARIANT,
        "measure_window_s": WINDOW_S,
        "hold_s": HOLD_S,
        "settle_delay_s": 0.0,
        "sleep": lambda _: None,
        "now": lambda: NOW,
    }
    defaults.update(kwargs)
    return join_cloudwatch(client, steps=list(steps), results=list(results), **defaults)


class TestTheJoinIsKeyedByRunAndRung:
    """Regression lock. This shipped, and it was silent.

    ``results`` was keyed by ``step_index`` alone, so with ``--runs 3`` each run's window
    overwrote the previous one and the survivor was applied to all three. It surfaced as
    byte-identical server metrics for runs 35 minutes apart — which reads as a beautifully
    repeatable endpoint rather than as a bug, and it corrupts exactly the field the
    deployed threshold's units come from.
    """

    def test_each_run_gets_its_own_window(self, cloudwatch: tuple[Any, Stubber]) -> None:
        client, stub = cloudwatch
        older, older_result = _joinable(window_end_ts=T0_TS - 600.0, run_index=0)
        newer, newer_result = _joinable(window_end_ts=T0_TS - 240.0, run_index=1)
        # Newest first: the settle wait belongs to the window closest to now.
        _queue_window(stub, concurrency_avg=(4.0,), concurrency_max=(11.0,))
        _queue_window(stub, concurrency_avg=(2.0,), concurrency_max=(9.0,))

        joined = _join(client, [older, newer], [older_result, newer_result])

        assert [step.run_index for step in joined] == [0, 1]
        assert joined[0].server_concurrency_peak == pytest.approx(9.0)
        assert joined[1].server_concurrency_peak == pytest.approx(11.0)
        stub.assert_no_pending_responses()

    def test_the_two_runs_asked_about_two_different_windows(
        self, cloudwatch: tuple[Any, Stubber]
    ) -> None:
        # Without this the test above could pass on identical windows, which is the very
        # state the defect produced.
        client, stub = cloudwatch
        calls: list[dict[str, Any]] = []
        _record_metric_calls(client, calls)
        older, older_result = _joinable(window_end_ts=T0_TS - 600.0, run_index=0)
        newer, newer_result = _joinable(window_end_ts=T0_TS - 240.0, run_index=1)
        _queue_window(stub)
        _queue_window(stub)

        _join(client, [older, newer], [older_result, newer_result])

        ends = {call["EndTime"] for call in calls}
        assert ends == {
            datetime.fromtimestamp(T0_TS - 600.0, tz=UTC),
            datetime.fromtimestamp(T0_TS - 240.0, tz=UTC),
        }

    def test_a_rung_with_no_window_of_its_own_stays_unjoined(
        self, cloudwatch: tuple[Any, Stubber]
    ) -> None:
        # The other half of the defect: rather than borrowing another run's numbers, a
        # rung whose fetch produced nothing carries no server-side numbers at all.
        client, stub = cloudwatch
        joined_step, result = _joinable(window_end_ts=T0_TS - 600.0, run_index=0)
        orphan, _ = _joinable(window_end_ts=T0_TS - 240.0, run_index=1)
        _queue_window(stub)

        joined = _join(client, [joined_step, orphan], [result])

        assert joined[0].server_concurrency_peak is not None
        assert joined[1].server_concurrency_peak is None


class TestTheJoinRecordsBothConcurrencyUnits:
    """``ConcurrentRequestsPerModel``/*Maximum* is the statistic the alarm reads.

    Nothing read it before, so the deployed threshold was derived from client occupancy
    and compared against a server peak — which is how ``0.713``, a value no positive
    arrival rate satisfies, reached a live endpoint. High-resolution datapoints retain
    three hours, so this join cannot be backfilled.
    """

    def test_the_peak_and_the_mean_are_both_recorded_and_differ(
        self, cloudwatch: tuple[Any, Stubber]
    ) -> None:
        client, stub = cloudwatch
        step, result = _joinable(window_end_ts=T0_TS - 240.0)
        # A bursty rung: two 10s periods whose averages and maxima disagree, which is the
        # ordinary case on a serial server and the reason the two are different numbers.
        _queue_window(stub, concurrency_avg=(2.0, 3.0), concurrency_max=(9.0, 15.0))

        joined = _join(client, [step], [result])

        assert joined[0].server_concurrency_mean == pytest.approx(2.5)
        assert joined[0].server_concurrency_peak == pytest.approx(15.0)

    def test_the_peak_is_the_maximum_across_periods_not_of_the_averages(
        self, cloudwatch: tuple[Any, Stubber]
    ) -> None:
        client, stub = cloudwatch
        step, result = _joinable(window_end_ts=T0_TS - 240.0)
        _queue_window(stub, concurrency_avg=(1.0, 1.0, 1.0), concurrency_max=(2.0, 20.0, 3.0))
        joined = _join(client, [step], [result])
        assert joined[0].server_concurrency_peak == pytest.approx(20.0)

    def test_model_latency_is_converted_from_microseconds(
        self, cloudwatch: tuple[Any, Stubber]
    ) -> None:
        client, stub = cloudwatch
        step, result = _joinable(window_end_ts=T0_TS - 240.0)
        _queue_window(stub, model_latency_p95_micros=250_000.0)
        joined = _join(client, [step], [result])
        assert joined[0].server_model_latency_p95_ms == pytest.approx(250.0)

    def test_an_idle_cpu_is_zero_not_missing(self, cloudwatch: tuple[Any, Stubber]) -> None:
        # 0% CPU on a GPU-bound rung is data. Reporting it as absent would make the
        # "what actually saturated" question unanswerable.
        client, stub = cloudwatch
        step, result = _joinable(window_end_ts=T0_TS - 240.0)
        _queue_window(stub, cpu=0.0)
        joined = _join(client, [step], [result])
        assert joined[0].cpu_utilization_mean == pytest.approx(0.0)
        assert joined[0].gpu_utilization_mean == pytest.approx(71.0)

    def test_the_5xx_count_comes_from_the_server(self, cloudwatch: tuple[Any, Stubber]) -> None:
        client, stub = cloudwatch
        step, result = _joinable(window_end_ts=T0_TS - 240.0)
        _queue_window(stub, errors_5xx=3.0)
        joined = _join(client, [step], [result])
        assert joined[0].server_5xx_total == pytest.approx(3.0)


class TestJoinCloudwatch:
    def test_it_fetches_the_measure_window_not_the_whole_rung(
        self, cloudwatch: tuple[Any, Stubber]
    ) -> None:
        client, stub = cloudwatch
        calls: list[dict[str, Any]] = []
        _record_metric_calls(client, calls)
        step, result = _joinable(window_end_ts=T0_TS - 240.0)
        _queue_window(stub)

        _join(client, [step], [result])

        assert len(calls) == 10
        for call in calls:
            assert (call["EndTime"] - call["StartTime"]).total_seconds() == WINDOW_S
            assert call["Period"] == HIGH_RES_PERIOD_S

    def test_the_window_ends_where_the_hold_ended_not_where_the_step_did(
        self, cloudwatch: tuple[Any, Stubber]
    ) -> None:
        # The same anchoring the client used, for the same reason: the drain past hold_s
        # is not part of the rung, and the two sides have to describe one window or the
        # cross-check compares different traffic.
        client, stub = cloudwatch
        result = _result(window_end_ts=T0_TS - 240.0, hold_s=HOLD_S, drain_s=100.0)
        calls: list[dict[str, Any]] = []
        _record_metric_calls(client, calls)
        _queue_window(stub)

        _join(client, [_measured(result)], [(0, result)])

        assert calls[0]["EndTime"] == datetime.fromtimestamp(T0_TS - 240.0, tz=UTC)

    def test_it_takes_the_settle_wait_on_the_newest_window(
        self, cloudwatch: tuple[Any, Stubber]
    ) -> None:
        # Waiting on the oldest returns instantly — its window is already older than the
        # delay — and the newest rung is then read before CloudWatch has aggregated it,
        # which surfaces as missing data rather than as a skipped wait.
        client, stub = cloudwatch
        older, older_result = _joinable(window_end_ts=(NOW - timedelta(seconds=300)).timestamp())
        newer, newer_result = _joinable(
            window_end_ts=(NOW - timedelta(seconds=30)).timestamp(), step_index=1
        )
        _queue_window(stub)
        _queue_window(stub)
        slept: list[float] = []

        _join(
            client,
            [older, newer],
            [older_result, newer_result],
            settle_delay_s=120.0,
            sleep=slept.append,
        )

        assert slept == [pytest.approx(90.0)]

    def test_no_results_returns_the_rungs_untouched(
        self, cloudwatch: tuple[Any, Stubber]
    ) -> None:
        client, stub = cloudwatch
        calls: list[dict[str, Any]] = []
        _record_metric_calls(client, calls)
        steps = _rungs(LADDER)
        assert _join(client, steps, []) == steps
        assert calls == []
        stub.assert_no_pending_responses()

    def test_a_failed_fetch_keeps_the_measured_ladder(
        self, cloudwatch: tuple[Any, Stubber], logged: list[str]
    ) -> None:
        # Telemetry must never lose a 40-minute measurement. The client-side numbers are
        # the ladder; the server cross-check and the unit conversion are the extras.
        client, stub = cloudwatch
        step, result = _joinable(window_end_ts=T0_TS - 240.0)
        stub.add_client_error("get_metric_statistics", service_error_code="AccessDenied")

        joined = _join(client, [step], [result], period_s=7)

        assert len(joined) == 1
        assert joined[0].ttfab_p95_ms == step.ttfab_p95_ms
        assert joined[0].server_concurrency_peak is None
        assert any("Could not join CloudWatch" in m for m in logged)

    def test_it_records_whether_the_client_and_server_agree(
        self, cloudwatch: tuple[Any, Stubber]
    ) -> None:
        client, stub = cloudwatch
        step, result = _joinable(window_end_ts=T0_TS - 240.0, concurrency=5)
        _queue_window(stub, concurrency_avg=(5.0,), concurrency_max=(9.0,))
        joined = _join(client, [step], [result])
        assert joined[0].concurrency_agreement == "client and server concurrency agree"

    def test_a_bottleneck_before_the_endpoint_is_named(
        self, cloudwatch: tuple[Any, Stubber]
    ) -> None:
        # The failure that motivated invoke.py: with a small connection pool a driver
        # asking for N in flight silently serializes and measures its own client. Every
        # client-side counter looks healthy, because the requests really are outstanding.
        client, stub = cloudwatch
        step, result = _joinable(window_end_ts=T0_TS - 240.0, concurrency=20)
        _queue_window(stub, concurrency_avg=(1.0,), concurrency_max=(2.0,))
        joined = _join(client, [step], [result])
        assert "bottlenecked before it reaches the endpoint" in (
            joined[0].concurrency_agreement or ""
        )


# --------------------------------------------------------------------------- #
# The report
# --------------------------------------------------------------------------- #


class TestBuildReport:
    def test_it_carries_the_slo_the_ladder_was_judged_on(self) -> None:
        # Q_max is *defined by* the SLO, so a plan built against a different line can be
        # refused rather than silently re-reading this ladder.
        report = _report(slo_ms=2500)
        assert report.slo_ms == 2500
        assert report.q_max == 50

    def test_the_answer_records_whether_it_was_bracketed(self) -> None:
        assert _report().q_max_bracketed
        assert not _report(ladders=[_ladder(_rungs(LADDER))]).q_max_bracketed

    def test_the_p95_at_q_max_says_how_much_slo_was_left(self) -> None:
        # A rung that passed at 2900ms against a 3000ms SLO is on the edge; one that
        # passed at 400ms means the ladder stopped short.
        assert _report().ttfab_p95_at_q_max_ms == pytest.approx(LADDER[50])

    def test_the_cross_run_answer_is_the_minimum(self) -> None:
        # Not the median: the median of two rungs is a concurrency no run tested, while
        # the minimum is both a rung that really ran and the conservative choice for a
        # number that sets an admission bound.
        report = _report(
            ladders=[
                _ladder(_rungs({**LADDER, **FAILING_RUNG}), run_index=0),
                _ladder(_rungs({1: 92.0, 5: 379.0, 10: 667.0, 20: 3400.0}), run_index=1),
            ]
        )
        assert report.q_max_per_run == (50, 10)
        assert report.q_max == 10
        assert report.runs == 2

    def test_every_run_keeps_its_own_answer(self) -> None:
        # Carried so the disagreement stays visible instead of being averaged away.
        report = _report(
            ladders=[
                _ladder(_rungs(LADDER), run_index=0),
                _ladder(_rungs({1: 92.0, 5: 379.0}), run_index=1),
            ]
        )
        assert report.q_max_per_run == (50, 5)
        assert report.q_max_spread == pytest.approx(9.0)

    def test_a_run_that_produced_nothing_does_not_contribute(self) -> None:
        report = _report(
            ladders=[
                _ladder(_rungs(LADDER), run_index=0),
                _ladder(_rungs({1: 9000.0}), run_index=1),
            ]
        )
        assert report.q_max_per_run == (50,)
        assert report.runs_contributing == 1
        assert report.runs == 2

    def test_q_max_comes_from_one_pass_not_from_the_pooled_rungs(self) -> None:
        # Pooling first would let the luckiest run stand in for all of them: rung 20
        # passed in run 0 and failed in run 1, and the answer has to be a crossing one
        # pass actually found.
        report = _report(
            ladders=[
                _ladder(_rungs({1: 92.0, 10: 667.0, 20: 1243.0}), run_index=0),
                _ladder(_rungs({1: 92.0, 10: 667.0, 20: 3400.0}), run_index=1),
            ]
        )
        assert report.q_max_per_run == (20, 10)
        assert report.q_max == 10

    def test_the_instance_counts_seen_are_carried(self) -> None:
        result = _result(instance_counts=(1, 2))
        ladder = _ladder(_rungs(LADDER))
        ladder.results = [result]
        report = _report(ladders=[ladder], frozen=True)
        assert report.instance_counts_observed == (1, 2)
        # Frozen but resized: not safe to read as per-instance, and the flag says so.
        assert not report.trustworthy

    def test_a_truncated_ladder_says_where_it_stopped(self) -> None:
        # So the artifact is never read as covering rungs that were never offered.
        report = _report(ladders=[_ladder(_rungs(LADDER), truncated_at=4)])
        assert report.ladder_truncated_at == 4

    def test_the_transport_is_on_the_artifact(self) -> None:
        report = _report(transport=Transport.BIDI)
        assert report.transport == "bidi"
        assert "bidi" in (report.provenance.note or "")

    def test_it_is_marked_measured(self) -> None:
        report = _report()
        assert report.provenance.origin is Origin.MEASURED
        assert report.provenance.run_id == "qmax12345678"

    def test_a_run_without_the_freeze_says_so_in_its_provenance(self) -> None:
        # A fleet-wide number has to stay identifiable as one: nothing else in the value
        # marks it, since a larger fleet reads as better latency.
        note = _report(frozen=False).provenance.note or ""
        assert "WITHOUT the autoscaling freeze" in note
        assert "N x Q_max" in note

    def test_the_service_time_is_the_lowest_rungs(self) -> None:
        report = _report()
        assert report.s_mean_s == pytest.approx(0.10986375146305409)
        assert report.s_p95_s >= report.s_mean_s

    def test_no_rung_meeting_the_slo_is_a_refusal(self) -> None:
        # A real finding, but not a Q_max — so it fails rather than serializing a number
        # nothing measured.
        with pytest.raises(QMaxError, match="no ladder rung met the 3000ms p95 TTFAB SLO"):
            _report(ladders=[_ladder(_rungs({1: 4000.0, 5: 5000.0}))])

    def test_the_refusal_quotes_the_best_p95_it_saw(self) -> None:
        # The number that distinguishes an unhealthy endpoint from one whose service time
        # alone exceeds the SLO — where the fix is a faster configuration, not a lower
        # Q_max.
        with pytest.raises(QMaxError, match=r"best p95 4000ms"):
            _report(ladders=[_ladder(_rungs({1: 4000.0, 5: 5000.0}))])

    def test_a_ladder_that_was_entirely_excluded_says_why(self) -> None:
        # Otherwise "no rung met the SLO" sends the operator to the endpoint when the
        # problem was the benchmark's own preconditions.
        shed = _measured(_result(concurrency=20, outcome=InvokeOutcome.SATURATED_503))
        with pytest.raises(QMaxError, match="Every rung was excluded as a measurement"):
            _report(ladders=[_ladder([shed])])

    def test_no_service_time_is_a_refusal(self) -> None:
        # W_max = SLO - S_p95 cannot be computed, so the artifact would carry a Q_max
        # with no latency behind it.
        steps = _rungs(LADDER, s_mean_s=None, s_p95_s=None)
        with pytest.raises(QMaxError, match="no ladder rung produced a service time"):
            _report(ladders=[_ladder(steps)])


class TestTheReportWarnsAboutWhatWeakensIt:
    def test_an_unbracketed_answer_is_called_a_lower_bound(self, logged: list[str]) -> None:
        # Both scaling thresholds are fractions of Q_max, so an unbracketed value makes
        # the policy add instances earlier than needed — which looks like working.
        _report(ladders=[_ladder(_rungs(LADDER))])
        assert any("is a LOWER bound" in m for m in logged)

    def test_a_bracketed_answer_is_not(self, logged: list[str]) -> None:
        _report()
        assert not any("LOWER bound" in m for m in logged)

    def test_a_wide_run_to_run_spread_is_noise_not_a_crossing(self, logged: list[str]) -> None:
        report = _report(
            ladders=[
                _ladder(_rungs({**LADDER, **FAILING_RUNG}), run_index=0),
                _ladder(_rungs({1: 92.0, 5: 379.0, 10: 3400.0}), run_index=1),
            ]
        )
        assert report.q_max_spread > SPREAD_WARN_THRESHOLD
        assert any("spread is" in m and "resolved noise" in m for m in logged)

    def test_one_run_out_of_two_is_not_a_repeatability_claim(self, logged: list[str]) -> None:
        # q_max_spread reads 0.0 from a single contributing run, which is not agreement.
        _report(
            ladders=[
                _ladder(_rungs({**LADDER, **FAILING_RUNG}), run_index=0),
                _ladder(_rungs({1: 9000.0}), run_index=1),
            ]
        )
        assert any("run(s) produced a Q_max" in m for m in logged)

    def test_client_limited_rungs_are_named(self, logged: list[str]) -> None:
        client_bound = _measured(_result(concurrency=20, in_flight=(1,) * 12))
        report = _report(ladders=[_ladder([*_rungs(LADDER), client_bound])])
        assert report.client_bound_rungs == [20]
        assert any("client-limited and excluded" in m for m in logged)

    def test_a_ladder_without_the_recovery_pair_is_useless_to_ttotal(
        self, logged: list[str]
    ) -> None:
        # ttotal holds a probe at 10 and declares recovery when p95 reaches the 5 value,
        # so a ladder missing either rung has to be re-run — and finding that out later
        # costs another 40 minutes and another freeze.
        _report(ladders=[_ladder(_rungs({1: 92.0, 20: 1243.0, 60: 3400.0}))])
        assert any(f"Rungs {list(RECOVERY_RUNGS)} produced no usable p95" in m for m in logged)

    def test_an_excluded_recovery_rung_counts_as_missing(self, logged: list[str]) -> None:
        # Offered is not measured: ladder_p95_ms filters on `usable`, so a rung the
        # ladder itself disowned must read as absent rather than as a level to compare
        # a probe against.
        steps = _rungs(LADDER)
        steps[1] = steps[1].model_copy(update={"usable": False, "meets_slo": False})
        report = _report(ladders=[_ladder(steps)])
        assert report.ttfab_p95_at(5) is None
        assert any("produced no usable p95" in m for m in logged)

    def test_no_n1_rung_leaves_the_alarm_without_a_threshold(self, logged: list[str]) -> None:
        # FirstChunkLatencyP95 watches an instance already serving, where the request has
        # spent none of its queue allowance, so the 3000ms SLO cannot stand in for it.
        report = _report(ladders=[_ladder(_rungs({5: 379.0, 10: 667.0, 60: 3400.0}))])
        assert report.ttfab_p95_at_c1_ms is None
        assert any("FirstChunkLatencyP95 alarm has no measured threshold" in m for m in logged)

    def test_an_unchecked_queue_bound_is_reported_as_unchecked(self, logged: list[str]) -> None:
        # None is not a pass. A container that sheds at a fixed depth reports that depth
        # as Q_max, and nothing else in the number would say so.
        report = _report(unbounded_queue=None)
        assert report.unbounded_queue is None
        assert any("unbounded-queue precondition was not checked" in m for m in logged)

    def test_a_checked_queue_is_silent(self, logged: list[str]) -> None:
        _report(unbounded_queue=True)
        assert not any("unbounded-queue precondition" in m for m in logged)

    def test_an_unfrozen_run_is_an_error_not_a_caveat(self, errors: list[str]) -> None:
        # ERROR because the number is not the quantity it claims to be: without the
        # freeze this may be N x Q_max, and a per-instance bound derived from it
        # over-admits by the size of the fleet.
        report = _report(frozen=False)
        assert not report.trustworthy
        assert any("NOT safe to read as per-instance" in m for m in errors)


class TestTheLadderTablesDownstreamReads:
    def test_the_p95_table_skips_rungs_that_measured_nothing(self) -> None:
        # Concurrency 60 must not already be a rung on this ladder, or `blank` would be
        # a second, distinct step at a concurrency `ladder_p95_ms` already has a real
        # (usable) measurement for -- which would pass even if the None-skip were
        # broken, because the real measurement is what the dict picks up.
        blank = _summary(concurrency=60, step_index=5, ttfab_p95_ms=None, meets_slo=False)
        report = _report(ladders=[_ladder([*_rungs(LADDER), blank])])
        assert 60 not in report.ladder_p95_ms
        assert set(report.ladder_p95_ms) == {1, 5, 10, 20, 50}

    def test_the_p95_table_skips_excluded_rungs(self) -> None:
        # 41ms at a shedding rung is what instant rejections look like. Reading it would
        # tell ttotal to expect recovery at a latency no served request hits. Concurrency
        # 60 must not already be a rung on this ladder for the same reason as above.
        shed = _summary(
            concurrency=60,
            step_index=5,
            ttfab_p95_ms=41.0,
            saturated=True,
            usable=False,
            meets_slo=False,
        )
        report = _report(ladders=[_ladder([*_rungs(LADDER), shed])])
        assert 60 not in report.ladder_p95_ms

    def test_a_rung_is_looked_up_exactly(self) -> None:
        # Not nearest: answering c=15 with the c=20 rung would silently compare a probe
        # against a concurrency it never ran at.
        report = _report()
        assert report.ttfab_p95_at(10) == pytest.approx(LADDER[10])
        assert report.ttfab_p95_at(15) is None

    def test_the_table_medians_across_runs(self) -> None:
        report = _report(
            ladders=[
                _ladder(_rungs({1: 90.0, 5: 300.0, 60: 3400.0}), run_index=0),
                _ladder(_rungs({1: 94.0, 5: 400.0, 60: 3400.0}), run_index=1),
                _ladder(_rungs({1: 92.0, 5: 500.0, 60: 3400.0}), run_index=2),
            ]
        )
        assert report.ttfab_p95_at(5) == pytest.approx(400.0)

    def test_the_unit_ratio_is_measured_per_rung(self, cloudwatch: tuple[Any, Stubber]) -> None:
        # Through the real join, because the conversion is only as good as the statistic
        # it reads: server *Maximum* over client mean in-flight, per rung.
        client, stub = cloudwatch
        rungs = [1, 50]
        steps: list[StepSummary] = []
        results: list[tuple[int, StepResult]] = []
        for index, rung in enumerate(rungs):
            step, pair = _joinable(
                window_end_ts=T0_TS - 600.0 + index * 300.0, step_index=index, concurrency=rung
            )
            steps.append(step)
            results.append(pair)
        # Queued newest-first, so the c=50 rung's window is fetched before the c=1 one's.
        for rung in reversed(rungs):
            peak = RATIOS[rung] * rung
            _queue_window(stub, concurrency_avg=(float(rung),), concurrency_max=(peak,))

        joined = _join(client, steps, results)
        report = _report(
            ladders=[_ladder(_rungs({**LADDER, **FAILING_RUNG}))], joined_steps=joined
        )

        ratios = report.cw_units_ratio_by_rung
        assert ratios[1] == pytest.approx(RATIOS[1])
        assert ratios[50] == pytest.approx(RATIOS[50])

    def test_the_unit_ratio_is_not_a_constant(self, cloudwatch: tuple[Any, Stubber]) -> None:
        # 9.8x at the bottom of one live kokoro ladder and 1.35x at the top. So the
        # planner interpolates on this table; a single fitted number deployed as a
        # threshold is the 0.713 defect, satisfiable by no positive arrival rate.
        client, stub = cloudwatch
        low, low_pair = _joinable(window_end_ts=T0_TS - 600.0, step_index=0, concurrency=1)
        high, high_pair = _joinable(window_end_ts=T0_TS - 300.0, step_index=1, concurrency=50)
        _queue_window(stub, concurrency_avg=(50.0,), concurrency_max=(RATIOS[50] * 50,))
        _queue_window(stub, concurrency_avg=(1.0,), concurrency_max=(RATIOS[1] * 1,))

        joined = _join(client, [low, high], [low_pair, high_pair])
        report = _report(
            ladders=[_ladder(_rungs({**LADDER, **FAILING_RUNG}))], joined_steps=joined
        )

        assert report.cw_units_ratio_by_rung[1] > 9.0
        assert report.cw_units_ratio_by_rung[50] < 1.5

    def test_a_rung_without_server_metrics_has_no_conversion(self) -> None:
        # Not 1:1 and not zero: the rung simply has no ratio, and the planner
        # interpolates from the rungs that do.
        assert _report().cw_units_ratio_by_rung == {}

    def test_the_joined_steps_are_what_the_artifact_carries(
        self, cloudwatch: tuple[Any, Stubber]
    ) -> None:
        client, stub = cloudwatch
        step, pair = _joinable(window_end_ts=T0_TS - 240.0, concurrency=5)
        _queue_window(stub)
        joined = _join(client, [step], [pair])
        report = _report(ladders=[_ladder(_rungs(LADDER))], joined_steps=joined)
        assert [s.server_concurrency_peak for s in report.steps] == [pytest.approx(15.0)]


# --------------------------------------------------------------------------- #
# measure(): orchestration and preconditions
# --------------------------------------------------------------------------- #


class TestMeasureRefusesBeforeItSendsLoad:
    def test_a_fleet_that_can_still_grow_stops_the_run(
        self,
        monkeypatch: pytest.MonkeyPatch,
        appscaling: tuple[Any, Stubber],
        sagemaker: tuple[Any, Stubber],
    ) -> None:
        # The enforcement point. A warning would be ignored and the resulting Q_max would
        # look entirely normal, because a larger fleet reads as better latency.
        aas, aas_stub = appscaling
        sm, sm_stub = sagemaker
        ladders = _patch_ladder(monkeypatch)
        client = _NoInvocations()
        _queue_fingerprint(sm_stub)
        _queue_freeze(aas_stub, sm_stub, suspends=False)

        with pytest.raises(FixtureError, match="scale-out still active"):
            _measure(
                require_frozen=True, runtime_client=client, appscaling=aas, sagemaker=sm
            )

        assert client.calls == 0
        assert ladders == []
        aas_stub.assert_no_pending_responses()
        sm_stub.assert_no_pending_responses()

    def test_a_bounded_queue_stops_the_run_before_anything_is_frozen(
        self,
        monkeypatch: pytest.MonkeyPatch,
        appscaling: tuple[Any, Stubber],
        sagemaker: tuple[Any, Stubber],
    ) -> None:
        # Checked before the freeze: a queue bound makes the whole run pointless, and
        # finding out after suspending production autoscaling costs a thaw for nothing.
        # No autoscaling response is queued, so any freeze call fails this test.
        aas, aas_stub = appscaling
        sm, sm_stub = sagemaker
        ladders = _patch_ladder(monkeypatch)
        _queue_fingerprint(sm_stub, env={"MAX_QUEUE_DEPTH": "24"})

        with pytest.raises(FixtureError, match="bounds its admission queue"):
            _measure(require_frozen=True, appscaling=aas, sagemaker=sm)

        assert ladders == []
        aas_stub.assert_no_pending_responses()
        sm_stub.assert_no_pending_responses()

    def test_the_bound_that_was_found_is_named(self, sagemaker: tuple[Any, Stubber]) -> None:
        sm, sm_stub = sagemaker
        _queue_fingerprint(sm_stub, env={"MAX_QUEUE_DEPTH": "24"})
        with pytest.raises(FixtureError, match="MAX_QUEUE_DEPTH=24"):
            _measure(sagemaker=sm)

    def test_a_bounded_queue_can_be_recorded_instead_of_refused(
        self,
        monkeypatch: pytest.MonkeyPatch,
        sagemaker: tuple[Any, Stubber],
        errors: list[str],
    ) -> None:
        # The override exists so a number can still be taken against a bounded
        # container, but the artifact has to stay identifiable as that measurement.
        sm, sm_stub = sagemaker
        _patch_ladder(monkeypatch)
        _queue_fingerprint(sm_stub, env={"MAX_QUEUE_DEPTH": "24"})

        report = _measure(require_unbounded_queue=False, sagemaker=sm)

        assert report.unbounded_queue is False
        assert any("--no-require-unbounded-queue was passed" in m for m in errors)

    @pytest.mark.parametrize("raw", ["0", None])
    def test_an_unset_or_zero_bound_proceeds(
        self, monkeypatch: pytest.MonkeyPatch, sagemaker: tuple[Any, Stubber], raw: str | None
    ) -> None:
        # A container reading 0 as "no bound" is the convention in streaming_proxy.py,
        # and refusing it would block a valid run.
        sm, sm_stub = sagemaker
        _patch_ladder(monkeypatch)
        env = {} if raw is None else {"MAX_QUEUE_DEPTH": raw}
        _queue_fingerprint(sm_stub, env=env)

        assert _measure(sagemaker=sm).unbounded_queue is True

    def test_an_unparseable_bound_is_not_a_pass(self, sagemaker: tuple[Any, Stubber]) -> None:
        # The container's own parse may succeed where ours did not, and guessing which
        # way it went is how a bounded queue gets measured as an unbounded one.
        sm, sm_stub = sagemaker
        _queue_fingerprint(sm_stub, env={"MAX_QUEUE_DEPTH": "twenty"})
        with pytest.raises(FixtureError, match="unparseable"):
            _measure(sagemaker=sm)

    def test_a_model_with_no_endpoint_is_refused_before_any_read(
        self, sagemaker: tuple[Any, Stubber]
    ) -> None:
        # Capacity planning applies to self-hosted endpoints only. No response is queued,
        # so any AWS call fails this test.
        sm, sm_stub = sagemaker
        with pytest.raises(ValueError, match="has no SageMaker endpoint"):
            _measure(model="polly-neural", sagemaker=sm)
        sm_stub.assert_no_pending_responses()


class TestMeasureOrchestration:
    def test_it_freezes_for_the_whole_ladder_and_thaws_after(
        self,
        monkeypatch: pytest.MonkeyPatch,
        appscaling: tuple[Any, Stubber],
        sagemaker: tuple[Any, Stubber],
    ) -> None:
        aas, aas_stub = appscaling
        sm, sm_stub = sagemaker
        ladders = _patch_ladder(monkeypatch)
        _queue_fingerprint(sm_stub)
        _queue_freeze(aas_stub, sm_stub)
        _queue_thaw(aas_stub, sm_stub)

        report = _measure(require_frozen=True, appscaling=aas, sagemaker=sm)

        assert report.frozen
        assert report.trustworthy
        assert len(ladders) == 1
        aas_stub.assert_no_pending_responses()
        sm_stub.assert_no_pending_responses()

    def test_it_thaws_when_the_ladder_raises(
        self,
        monkeypatch: pytest.MonkeyPatch,
        appscaling: tuple[Any, Stubber],
        sagemaker: tuple[Any, Stubber],
    ) -> None:
        # A benchmark that dies mid-rung must not leave production frozen.
        aas, aas_stub = appscaling
        sm, sm_stub = sagemaker
        _patch_ladder(monkeypatch, raises=RuntimeError("the transport gave up"))
        _queue_fingerprint(sm_stub)
        _queue_freeze(aas_stub, sm_stub)
        _queue_thaw(aas_stub, sm_stub)

        with pytest.raises(RuntimeError, match="the transport gave up"):
            _measure(require_frozen=True, appscaling=aas, sagemaker=sm)

        aas_stub.assert_no_pending_responses()
        sm_stub.assert_no_pending_responses()

    def test_running_unfrozen_is_warned_and_recorded(
        self,
        monkeypatch: pytest.MonkeyPatch,
        sagemaker: tuple[Any, Stubber],
        logged: list[str],
    ) -> None:
        sm, sm_stub = sagemaker
        _patch_ladder(monkeypatch)
        _queue_fingerprint(sm_stub)

        report = _measure(require_frozen=False, sagemaker=sm)

        assert not report.frozen
        assert any("WITHOUT the autoscaling freeze" in m for m in logged)

    def test_each_run_gets_its_own_run_id_under_one_report_id(
        self, monkeypatch: pytest.MonkeyPatch, sagemaker: tuple[Any, Stubber]
    ) -> None:
        # So per-request events can be traced back to the pass they came from, which is
        # what makes the (run_index, step_index) join checkable from the artifact alone.
        sm, sm_stub = sagemaker
        ladders = _patch_ladder(monkeypatch)
        _queue_fingerprint(sm_stub)

        report = _measure(runs=2, sagemaker=sm)

        assert [call["run_id"] for call in ladders] == [
            f"{report.run_id}-r0",
            f"{report.run_id}-r1",
        ]
        assert [call["run_index"] for call in ladders] == [0, 1]

    def test_the_ladder_gets_the_slo_and_the_window(
        self, monkeypatch: pytest.MonkeyPatch, sagemaker: tuple[Any, Stubber]
    ) -> None:
        sm, sm_stub = sagemaker
        ladders = _patch_ladder(monkeypatch)
        _queue_fingerprint(sm_stub)

        _measure(slo_ms=1500, hold_s=90.0, measure_window_s=30.0, sagemaker=sm)

        assert ladders[0]["slo_ms"] == 1500
        assert ladders[0]["hold_s"] == 90.0
        assert ladders[0]["measure_window_s"] == 30.0

    def test_the_ladder_is_deduplicated_and_sorted_before_it_runs(
        self, monkeypatch: pytest.MonkeyPatch, sagemaker: tuple[Any, Stubber]
    ) -> None:
        sm, sm_stub = sagemaker
        ladders = _patch_ladder(monkeypatch)
        _queue_fingerprint(sm_stub)

        _measure(concurrencies=(10, 1, 10, 5), sagemaker=sm)

        assert ladders[0]["concurrencies"] == [1, 5, 10]

    def test_the_text_pool_is_shuffled_once_and_shared(
        self, monkeypatch: pytest.MonkeyPatch, sagemaker: tuple[Any, Stubber]
    ) -> None:
        # Every rung walks the same order from index 0, so two rungs of equal length
        # synthesize the same characters — without it a rung could look worse merely for
        # having drawn longer texts.
        sm, sm_stub = sagemaker
        ladders = _patch_ladder(monkeypatch)
        _queue_fingerprint(sm_stub)

        _measure(runs=2, seed=7, sagemaker=sm)

        assert sorted(ladders[0]["texts"]) == sorted(TEXTS)
        assert ladders[0]["texts"] == ladders[1]["texts"]

    def test_skipping_the_cloudwatch_join_says_what_is_lost(
        self,
        monkeypatch: pytest.MonkeyPatch,
        sagemaker: tuple[Any, Stubber],
        logged: list[str],
    ) -> None:
        # High-resolution datapoints retain three hours, so this cannot be backfilled
        # after the run — and without it `plan` has no unit conversion at all.
        sm, sm_stub = sagemaker
        _patch_ladder(monkeypatch)
        _queue_fingerprint(sm_stub)

        report = _measure(cloudwatch_join=False, sagemaker=sm)

        assert all(step.server_concurrency_peak is None for step in report.steps)
        assert any("Skipping the CloudWatch join" in m for m in logged)

    def test_it_joins_every_rung_of_every_run(
        self,
        monkeypatch: pytest.MonkeyPatch,
        sagemaker: tuple[Any, Stubber],
        cloudwatch: tuple[Any, Stubber],
    ) -> None:
        # The (run_index, step_index) pairing again, this time through the seam measure
        # actually uses: two runs of one rung must produce two joined summaries.
        sm, sm_stub = sagemaker
        cw, cw_stub = cloudwatch
        recorded: list[dict[str, Any]] = []

        def fake_run_ladder(client: Any, **kwargs: Any) -> LadderRun:
            recorded.append(kwargs)
            run_index = kwargs["run_index"]
            # Windows long past, so the settle wait is already satisfied: measure calls
            # join_cloudwatch with no sleep seam, and a rung stamped "just now" would sit
            # through the real two-minute delay.
            #
            # step_index=5 is outside LADDER's 0-4 range on purpose: `_result` defaults
            # to step_index=0, which is the same key `_rungs(LADDER, ...)` gives its own
            # concurrency=1 rung, so both summaries would join against one CloudWatch
            # window and double every peak this test is asserting on.
            result = _result(
                concurrency=5,
                step_index=5,
                window_end_ts=datetime(2026, 1, 1, tzinfo=UTC).timestamp() + run_index * 600.0,
                run_id=kwargs["run_id"],
            )
            ladder = _ladder(_rungs(LADDER, run_index=run_index), run_index=run_index)
            ladder.results = [result]
            ladder.steps = [_measured(result, run_index=run_index), *ladder.steps]
            return ladder

        monkeypatch.setattr(qmax_mod, "run_ladder", fake_run_ladder)
        _queue_fingerprint(sm_stub)
        _queue_window(cw_stub, concurrency_avg=(4.0,), concurrency_max=(11.0,))
        _queue_window(cw_stub, concurrency_avg=(2.0,), concurrency_max=(9.0,))

        report = _measure(runs=2, cloudwatch_join=True, sagemaker=sm, cloudwatch=cw)

        peaks = [s.server_concurrency_peak for s in report.steps if s.server_concurrency_peak]
        assert sorted(peaks) == [pytest.approx(9.0), pytest.approx(11.0)]
        cw_stub.assert_no_pending_responses()


class TestMeasureRecordsTheConfiguration:
    def test_the_fingerprint_comes_off_the_endpoint(
        self, monkeypatch: pytest.MonkeyPatch, sagemaker: tuple[Any, Stubber]
    ) -> None:
        # Read from the endpoint rather than from the registry, which can be stale in
        # exactly the situation this harness supports: redeploy, re-measure, and a static
        # dict would stamp the fresh artifact with the old instance type.
        sm, sm_stub = sagemaker
        _patch_ladder(monkeypatch)
        _queue_fingerprint(sm_stub, instance_type="ml.g5.xlarge")

        report = _measure(sagemaker=sm)

        assert report.instance_type == "ml.g5.xlarge"
        assert report.deployed_config["image_digest"] == "139b9068c5eb1f03"
        assert report.deployed_config["container_env"] == {"MAX_REQUEST_AGE_S": "56"}

    def test_the_deployed_type_beats_the_registry_and_the_divergence_is_an_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
        sagemaker: tuple[Any, Stubber],
        errors: list[str],
    ) -> None:
        # A stale registry also makes `drift` and the cost model wrong, not just this
        # artifact, which is why it is louder than a warning on the measurement.
        sm, sm_stub = sagemaker
        _patch_ladder(monkeypatch)
        _queue_fingerprint(sm_stub, instance_type="ml.g6.12xlarge")

        report = _measure(sagemaker=sm)

        assert report.instance_type == "ml.g6.12xlarge"
        assert any("MODEL_INSTANCE_TYPES says" in m for m in errors)

    def test_an_unreadable_endpoint_falls_back_without_losing_the_run(
        self,
        monkeypatch: pytest.MonkeyPatch,
        sagemaker: tuple[Any, Stubber],
        logged: list[str],
    ) -> None:
        # A fallback fingerprint has no image digest, so it can never compare equal to
        # one measured against a known configuration.
        sm, sm_stub = sagemaker
        _patch_ladder(monkeypatch)
        sm_stub.add_client_error("describe_endpoint", service_error_code="ValidationException")

        report = _measure(sagemaker=sm)

        assert report.instance_type == "ml.g5.xlarge"
        assert report.deployed_config["image_digest"] is None
        assert report.config_slug == "g5xlarge-nodigest"
        assert any("Falling back to the registry type" in m for m in logged)

    def test_the_slug_names_the_configuration(
        self, monkeypatch: pytest.MonkeyPatch, sagemaker: tuple[Any, Stubber]
    ) -> None:
        # What keeps two configurations' ladders under two filenames: they are two
        # measurements, not two attempts at one.
        sm, sm_stub = sagemaker
        _patch_ladder(monkeypatch)
        _queue_fingerprint(sm_stub, instance_type="ml.g6.12xlarge")
        assert _measure(sagemaker=sm).config_slug == "g612xlarge-139b9068"

    def test_the_artifact_round_trips(
        self, monkeypatch: pytest.MonkeyPatch, sagemaker: tuple[Any, Stubber]
    ) -> None:
        from tts_bench.types import QMaxReport

        sm, sm_stub = sagemaker
        _patch_ladder(monkeypatch)
        _queue_fingerprint(sm_stub)

        report = _measure(sagemaker=sm)
        restored = QMaxReport.model_validate_json(report.model_dump_json())

        assert restored.q_max == report.q_max
        assert restored.deployed_config == report.deployed_config
        assert restored.slo_ms == SLO_MS

    def test_it_pins_to_the_count_it_was_asked_for(
        self, monkeypatch: pytest.MonkeyPatch, sagemaker: tuple[Any, Stubber]
    ) -> None:
        # pin_to reaches fixture.frozen and require_frozen together, so a run pinned at
        # one instance cannot verify against a different expectation.
        sm, sm_stub = sagemaker
        _patch_ladder(monkeypatch)
        seen: list[dict[str, Any]] = []

        class _Frozen:
            def __init__(self, endpoint: str, **kwargs: Any) -> None:
                seen.append({"call": "frozen", **kwargs})

            def __enter__(self) -> None:
                return None

            def __exit__(self, *exc: Any) -> bool:
                return False

        monkeypatch.setattr(fixture_mod, "frozen", _Frozen)
        monkeypatch.setattr(
            fixture_mod,
            "require_frozen",
            lambda endpoint, **kwargs: seen.append({"call": "require_frozen", **kwargs}),
        )
        _queue_fingerprint(sm_stub)

        _measure(require_frozen=True, pin_to=1, sagemaker=sm)

        assert [entry["call"] for entry in seen] == ["frozen", "require_frozen"]
        assert seen[0]["pin_to"] == 1
        assert seen[1]["expect_instances"] == 1


class TestDefaultLadder:
    def test_the_default_rungs_cover_every_downstream_consumer(self) -> None:
        # 1 is the FirstChunkLatencyP95 alarm's service-time reference; 5 and 10 are the
        # pair ttotal watches a p95 halve between. A ladder missing any of them forces
        # that consumer to refuse the artifact, and re-running costs another freeze.
        assert 1 in DEFAULT_CONCURRENCIES
        assert set(RECOVERY_RUNGS) <= set(DEFAULT_CONCURRENCIES)

    def test_the_recovery_pair_is_a_halving(self) -> None:
        # A second instance splits a concurrency-10 probe 5/5, so the lower rung has to
        # be exactly half the higher one for the comparison to mean anything.
        low, high = RECOVERY_RUNGS
        assert high == low * 2

    def test_the_rungs_are_ascending_and_distinct(self) -> None:
        assert list(DEFAULT_CONCURRENCIES) == sorted(set(DEFAULT_CONCURRENCIES))
