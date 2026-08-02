"""Tests for the step-and-hold ``C_max`` harness.

The whole point of ``cmax.py`` is to avoid producing a *plausible* number for the
wrong quantity, so these tests concentrate on the places where that can happen
silently:

- **The knee is judged against the knee we reported.** ``find_knee`` walks the
  whole ladder rather than stopping at the first failure, which means a
  pass/fail/pass ladder can look bracketed when the bracket sits *below* the
  reported knee. ``TestFindKnee.test_pass_fail_pass_is_not_bracketed`` is the
  regression for that.
- **``S`` must be uncontended.** Service time at the knee already includes
  queueing, and ``C_slo_cap = W_max / S`` would then bound a wait it had already
  counted. ``TestUncontendedServiceTime`` pins the source to the lowest step.
- **The CloudWatch settle wait belongs to the newest window.** Waiting on the
  oldest returns instantly, and the newest step then reads before aggregation —
  surfacing as missing data rather than as a skipped wait.
- **A step where the client was the limit is excluded, not flagged.** A knee
  measured against an exhausted thread pool looks exactly like a real one.

Synthetic ``StepResult``s drive the ladder logic so saturation and settling are
exact rather than timing-dependent; ``run_step`` itself is covered by
``test_loadgen.py``, including the open-loop property. One test does run the real
``run_step`` against a ``FakeServer`` to keep the two wired together.
"""

from __future__ import annotations

import functools
import threading
import time
from collections.abc import Iterator, Sequence
from datetime import UTC, datetime, timedelta

import pytest
from loguru import logger

from shared.capacity import lambda_cap_per_instance
from tts_bench import cmax, fixture
from tts_bench.bidi import Transport, invoke_bidi
from tts_bench.cmax import (
    SPREAD_WARN_THRESHOLD,
    CMaxError,
    LadderRun,
    ProbeResult,
    build_report,
    dry_run_plan,
    find_knee,
    find_throughput_ceiling,
    join_cloudwatch,
    measure,
    median_ceiling,
    median_curve,
    probe_service_time,
    rps_for_concurrency,
    run_ladder,
    summarize_step,
    total_duration_s,
    uncontended_service_time,
    worker_count,
)
from tts_bench.fixture import SCALABLE_DIMENSION, SERVICE_NAMESPACE, FixtureError
from tts_bench.invoke import InvokeOutcome, InvokeResult, invoke_stream
from tts_bench.loadgen import (
    SYSTEM_CLOCK,
    Clock,
    ConcurrencySample,
    LoadEvent,
    StepResult,
    run_step,
    summarize_window,
)
from tts_bench.types import CMaxReport, Origin, StepSummary
from tts_inference.types import TTSModelName

MODEL = "kokoro-82m"
ENDPOINT = "speech-kokoro-82m"
RID = f"endpoint/{ENDPOINT}/variant/primary"
TEXTS = ["Let me check that for you.", "Your appointment is confirmed.", "One moment please."]

NOW = datetime(2026, 7, 29, 12, 30, 0, tzinfo=UTC)
T0_TS = NOW.timestamp()
WINDOW_S = 60.0

#: A window that closed long enough ago that the CloudWatch settle wait is
#: already satisfied. ``measure`` calls ``join_cloudwatch`` without a sleep seam,
#: so a step stamped "just now" would make the test sit through the real 120s
#: settle delay.
SETTLED_TS = datetime(2026, 1, 1, tzinfo=UTC).timestamp()


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #


def _invoke_result(
    *,
    latency_ms: float = 250.0,
    outcome: InvokeOutcome = InvokeOutcome.OK,
) -> InvokeResult:
    return InvokeResult(
        outcome=outcome,
        dispatch_ts=T0_TS,
        end_ts=T0_TS + latency_ms / 1000.0,
        latency_ms=latency_ms,
        ttfab_ms=latency_ms / 2.0,
        chars=26,
        audio_bytes=4800,
        audio_duration_s=0.2,
        http_status=200 if outcome is InvokeOutcome.OK else 503,
        error_message=None if outcome is InvokeOutcome.OK else "boom",
    )


def _load_event(
    *,
    seq: int,
    step_index: int,
    offered_rps: float,
    scheduled_ts: float,
    end_ts: float | None,
    ttfab_ms: float | None,
    latency_ms: float | None,
    outcome: InvokeOutcome = InvokeOutcome.OK,
) -> LoadEvent:
    return LoadEvent(
        run_id="run",
        step_index=step_index,
        seq=seq,
        offered_rps=offered_rps,
        model=MODEL,
        endpoint=ENDPOINT,
        scheduled_ts=scheduled_ts,
        dispatch_ts=scheduled_ts,
        first_byte_ts=None,
        end_ts=end_ts,
        dispatch_delay_ms=0.0,
        ttfab_ms=ttfab_ms,
        latency_ms=latency_ms,
        outcome=outcome.value,
        http_status=200,
        error_class=None,
        error_message=None,
        chars=26,
        audio_bytes=4800,
        audio_duration_s=0.2,
        rtf=0.5,
        in_flight_at_dispatch=1,
        instance_count=1,
    )


def _step(
    *,
    step_index: int = 0,
    offered_rps: float = 10.0,
    achieved_rps: float | None = None,
    ttfab_ms: float | None = 100.0,
    latency_ms: float = 200.0,
    ended_ts: float = T0_TS,
    window_s: float = WINDOW_S,
    in_flight: Sequence[int] = (2,) * 10,
    instance_counts: Sequence[int] = (1,),
    skipped: int = 0,
) -> StepResult:
    """A ``StepResult`` whose trailing ``window_s`` yields chosen statistics.

    ``achieved_rps`` controls how many completions land in the window, which is
    what makes a step saturated (``achieved < 0.95 x offered``). ``in_flight``
    both sets ``concurrency_mean`` and, through its trend, whether the step
    settled — a ramp is what an over-capacity rate actually looks like.
    """
    start = ended_ts - window_s
    achieved = offered_rps if achieved_rps is None else achieved_rps
    count = int(round(achieved * window_s))

    events = [
        _load_event(
            seq=i,
            step_index=step_index,
            offered_rps=offered_rps,
            scheduled_ts=start + (i + 0.5) * window_s / max(count, 1) - latency_ms / 1000.0,
            end_ts=start + (i + 0.5) * window_s / max(count, 1),
            ttfab_ms=ttfab_ms,
            latency_ms=latency_ms,
        )
        for i in range(count)
    ]
    events += [
        _load_event(
            seq=count + i,
            step_index=step_index,
            offered_rps=offered_rps,
            scheduled_ts=start + (i + 0.5) * window_s / max(skipped, 1),
            end_ts=None,
            ttfab_ms=None,
            latency_ms=None,
            outcome=InvokeOutcome.DISPATCH_SKIPPED,
        )
        for i in range(skipped)
    ]

    samples = [
        ConcurrencySample(
            ts=start + (i + 0.5) * window_s / len(in_flight),
            in_flight=value,
            instance_count=instance_counts[
                min(i * len(instance_counts) // len(in_flight), len(instance_counts) - 1)
            ],
        )
        for i, value in enumerate(in_flight)
    ]

    return StepResult(
        run_id="run",
        step_index=step_index,
        offered_rps=offered_rps,
        model=MODEL,
        endpoint=ENDPOINT,
        arrival_process="poisson",
        seed=1234,
        started_ts=start - 180.0,
        ended_ts=ended_ts,
        scheduled_count=len(events),
        events=events,
        samples=samples,
    )


def _summarize(
    result: StepResult, *, window_s: float = WINDOW_S, run_index: int = 0
) -> StepSummary:
    stats = summarize_window(result, start_ts=result.ended_ts - window_s, end_ts=result.ended_ts)
    return summarize_step(
        result,
        stats,
        run_index=run_index,
        target_concurrency=result.offered_rps * 0.2,
    )


def _summary(
    *,
    step_index: int = 0,
    target_concurrency: float = 1.0,
    offered_rps: float = 5.0,
    achieved_rps: float | None = None,
    ttfab_p95_ms: float | None = 100.0,
    concurrency_mean: float | None = None,
    s_mean_s: float | None = 0.2,
    s_p95_s: float | None = 0.3,
    saturated: bool = False,
    settled: bool = True,
    usable: bool = True,
    skipped: int = 0,
    capacity_changed: bool = False,
    completed: int | None = None,
    run_index: int = 0,
) -> StepSummary:
    """A ``StepSummary`` built directly, for the pure knee/curve functions.

    ``achieved_rps`` defaults to ``offered_rps`` — the server kept up. The throughput
    ceiling reads it directly, so a test about the ceiling has to set it rather than rely
    on ``saturated``: the two are independent fields here even though the real
    ``WindowStats.saturated`` derives one from the other.
    """
    achieved = offered_rps if achieved_rps is None else achieved_rps
    return StepSummary(
        run_index=run_index,
        step_index=step_index,
        target_concurrency=target_concurrency,
        offered_rps=offered_rps,
        achieved_rps=achieved,
        completed=int(achieved * WINDOW_S) if completed is None else completed,
        ok=int(achieved * WINDOW_S) if completed is None else completed,
        ttfab_p95_ms=ttfab_p95_ms,
        s_mean_s=s_mean_s,
        s_p95_s=s_p95_s,
        concurrency_mean=target_concurrency if concurrency_mean is None else concurrency_mean,
        saturated=saturated,
        settled=settled,
        usable=usable,
        dispatch_skipped=skipped,
        capacity_changed=capacity_changed,
    )


def _probe(*, s_mean_s: float = 0.25, s_p95_s: float = 0.4) -> ProbeResult:
    return ProbeResult(s_mean_s=s_mean_s, s_p95_s=s_p95_s, samples=5, failures=0)


def _ladder(steps: Sequence[StepSummary], *, run_index: int = 0, **kwargs) -> LadderRun:
    return LadderRun(run_index=run_index, steps=list(steps), **kwargs)


class FakeServer:
    """Fixed service time with a concurrency limit. Mirrors ``test_loadgen.py``."""

    def __init__(self, *, service_time_s: float = 0.05, slots: int = 4) -> None:
        self._sem = threading.Semaphore(slots)
        self._service_time_s = service_time_s
        self._lock = threading.Lock()
        self.completed = 0

    def __call__(self, client, endpoint, text, voice, *, deadline_ts=None) -> InvokeResult:
        dispatch_ts = time.time()
        with self._sem:
            time.sleep(self._service_time_s)
            with self._lock:
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
            audio_duration_s=0.1,
            sample_rate=24000,
            http_status=200,
            chunks=1,
        )


class FakeCloudWatch:
    """Returns one datapoint per metric, or raises to exercise degradation."""

    #: Values chosen so each joined field is distinguishable in an assertion.
    #: ``ModelLatency`` is in MICROseconds, as AWS publishes it.
    VALUES = {
        "ConcurrentRequestsPerModel": 2.5,
        "ModelLatency": 250_000.0,
        "Invocation5XXErrors": 3.0,
        "GPUUtilization": 71.0,
        # Present but zero: an idle CPU is data, not a missing metric.
        "CPUUtilization": 0.0,
    }

    def __init__(self, *, raises: bool = False) -> None:
        self._raises = raises
        self.calls: list[dict] = []

    def get_metric_statistics(self, **kwargs) -> dict:
        self.calls.append(kwargs)
        if self._raises:
            raise RuntimeError("CloudWatch is having a day")
        value = self.VALUES.get(kwargs["MetricName"])
        if value is None:
            return {"Datapoints": []}
        return {
            "Datapoints": [
                {
                    "Timestamp": kwargs["StartTime"],
                    "Average": value,
                    "Maximum": value,
                    "Sum": value,
                    "ExtendedStatistics": {"p95": value, "p99": value},
                }
            ]
        }


class FakeAppScaling:
    """Minimal ``application-autoscaling`` double for the freeze path."""

    def __init__(self, *, scale_out_suspended: bool = True, has_target: bool = True) -> None:
        self._suspended = scale_out_suspended
        self._has_target = has_target
        self.registered: list[dict] = []

    def describe_scalable_targets(self, **kwargs) -> dict:
        if not self._has_target:
            return {"ScalableTargets": []}
        return {
            "ScalableTargets": [
                {
                    "ServiceNamespace": SERVICE_NAMESPACE,
                    "ResourceId": RID,
                    "ScalableDimension": SCALABLE_DIMENSION,
                    "MinCapacity": 1,
                    "MaxCapacity": 4,
                    "SuspendedState": {
                        "DynamicScalingInSuspended": self._suspended,
                        "DynamicScalingOutSuspended": self._suspended,
                        "ScheduledScalingSuspended": self._suspended,
                    },
                }
            ]
        }

    def describe_scaling_policies(self, **kwargs) -> dict:
        return {"ScalingPolicies": [{"PolicyName": "TrackConcurrency"}]}

    def register_scalable_target(self, **kwargs) -> dict:
        self.registered.append(kwargs)
        return {}


class FakeSageMaker:
    """Minimal ``sagemaker`` double: already at one instance, so no pin is needed.

    Also answers the three calls behind the configuration fingerprint. ``instance_type``
    defaults to the registry's value for kokoro so the common case is a *match*, which
    keeps the disagreement warning out of tests that are not about it.
    """

    def __init__(
        self,
        *,
        desired: int = 1,
        current: int = 1,
        instance_type: str = "ml.g5.xlarge",
        image: str | None = "repo/asset:abc123def456",
        env: dict[str, str] | None = None,
    ) -> None:
        self.desired = desired
        self.current = current
        self.instance_type = instance_type
        self.image = image
        self.env = {"MAX_REQUEST_AGE_S": "56"} if env is None else env
        self.updates: list[dict] = []

    def describe_endpoint(self, EndpointName: str) -> dict:  # noqa: N803 - boto3 API
        return {
            "EndpointConfigName": f"{EndpointName}-config",
            "ProductionVariants": [
                {
                    "VariantName": "primary",
                    "DesiredInstanceCount": self.desired,
                    "CurrentInstanceCount": self.current,
                }
            ],
        }

    def describe_endpoint_config(self, EndpointConfigName: str) -> dict:  # noqa: N803 - boto3 API
        return {
            "ProductionVariants": [
                {
                    "VariantName": "primary",
                    "InstanceType": self.instance_type,
                    "ModelName": "fake-model",
                }
            ]
        }

    def describe_model(self, ModelName: str) -> dict:  # noqa: N803 - boto3 API
        return {"PrimaryContainer": {"Image": self.image, "Environment": dict(self.env)}}

    def update_endpoint_weights_and_capacities(self, **kwargs) -> dict:
        self.updates.append(kwargs)
        return {}


class RecordingClock:
    """A ``Clock`` that records sleeps instead of taking them."""

    def __init__(self) -> None:
        self.slept: list[float] = []

    def clock(self) -> Clock:
        return Clock(monotonic=time.monotonic, sleep=self.slept.append, time=time.time)


@pytest.fixture
def logged() -> Iterator[list[str]]:
    """Captured loguru messages.

    ``caplog`` does not see these: loguru writes to its own sinks and does not
    propagate to the stdlib ``logging`` tree, so a ``caplog`` assertion here
    would pass vacuously whether or not the warning was ever emitted.
    """
    records: list[str] = []
    sink_id = logger.add(lambda message: records.append(message.record["message"]), level="WARNING")
    try:
        yield records
    finally:
        logger.remove(sink_id)


# --------------------------------------------------------------------------- #
# Little's Law and sizing
# --------------------------------------------------------------------------- #


class TestRpsForConcurrency:
    def test_converts_concurrency_to_rate(self) -> None:
        # L = lambda x W, so a 250ms service time needs 4 rps per unit in flight.
        assert rps_for_concurrency(1.0, 0.25) == pytest.approx(4.0)
        assert rps_for_concurrency(8.0, 0.25) == pytest.approx(32.0)

    def test_a_slow_model_needs_a_lower_rate_for_the_same_concurrency(self) -> None:
        # Why the ladder is in concurrency, not rate: 8 rps means something
        # entirely different to a 40ms model than to a 4s one.
        assert rps_for_concurrency(1.0, 4.0) == pytest.approx(0.25)
        assert rps_for_concurrency(1.0, 0.04) == pytest.approx(25.0)

    @pytest.mark.parametrize("target", [0.0, -1.0])
    def test_rejects_a_non_positive_target(self, target: float) -> None:
        with pytest.raises(ValueError, match="target_concurrency must be positive"):
            rps_for_concurrency(target, 0.25)

    @pytest.mark.parametrize("service_time", [0.0, -0.25])
    def test_rejects_a_non_positive_service_time(self, service_time: float) -> None:
        with pytest.raises(ValueError, match="s_mean_s must be positive"):
            rps_for_concurrency(1.0, service_time)


class TestWorkerCount:
    def test_leaves_headroom_over_the_peak(self) -> None:
        # Without headroom the client's pool saturates first and the measurement
        # describes the benchmark rather than the model.
        assert worker_count(16.0) > 16
        assert worker_count(16.0) == 40

    def test_never_drops_below_two(self) -> None:
        # A 0.5-concurrency bottom step still needs somewhere to dispatch.
        assert worker_count(0.5) >= 2

    def test_grows_with_the_ladder(self) -> None:
        assert worker_count(4.0) < worker_count(16.0)

    def test_an_override_raises_the_pool(self) -> None:
        # The whole reason the flag exists: bidi's in-flight overshoots the target by the
        # server's backlog, so no derived multiple of the target covers it, and a ceiling
        # measured against an exhausted pool is the benchmark's rather than the server's.
        assert worker_count(3.0) == 14
        assert worker_count(3.0, 64) == 64

    def test_an_override_cannot_shrink_the_pool(self, logged: list[str]) -> None:
        # Lowering it would recreate the one failure worker_count exists to prevent — a
        # *passing* step that was silently client-limited. Refused and said out loud.
        assert worker_count(3.0, 4) == 14
        assert any("below the 14 workers" in m for m in logged)

    def test_no_override_is_the_derived_size(self) -> None:
        assert worker_count(3.0, None) == worker_count(3.0)


# --------------------------------------------------------------------------- #
# Probe
# --------------------------------------------------------------------------- #


class TestProbeServiceTime:
    def test_measures_mean_and_p95_over_successes(self) -> None:
        latencies = [200.0, 220.0, 240.0, 260.0, 280.0]
        calls: list[str] = []

        def invoke(client, endpoint, text, voice, **kwargs) -> InvokeResult:
            calls.append(text)
            return _invoke_result(latency_ms=latencies[len(calls) - 1])

        probe = probe_service_time(
            None, endpoint=ENDPOINT, voice="af_heart", texts=TEXTS, requests=5, invoke=invoke
        )
        assert probe.s_mean_s == pytest.approx(0.24)
        assert probe.s_p95_s == pytest.approx(0.276)
        assert probe.samples == 5
        assert probe.failures == 0
        assert probe.usable

    def test_cycles_the_text_pool(self) -> None:
        # Three texts, five probes: the pool wraps rather than indexing off the end.
        seen: list[str] = []

        def invoke(client, endpoint, text, voice, **kwargs) -> InvokeResult:
            seen.append(text)
            return _invoke_result()

        probe_service_time(
            None, endpoint=ENDPOINT, voice="af_heart", texts=TEXTS, requests=5, invoke=invoke
        )
        assert seen == [TEXTS[0], TEXTS[1], TEXTS[2], TEXTS[0], TEXTS[1]]

    def test_counts_failures_but_keeps_the_successes(self) -> None:
        results = [
            _invoke_result(latency_ms=200.0),
            _invoke_result(outcome=InvokeOutcome.SERVER_5XX),
            _invoke_result(latency_ms=300.0),
        ]
        calls = iter(results)

        probe = probe_service_time(
            None,
            endpoint=ENDPOINT,
            voice="af_heart",
            texts=TEXTS,
            requests=3,
            invoke=lambda *a, **k: next(calls),
        )
        assert probe.samples == 2
        assert probe.failures == 1
        assert probe.s_mean_s == pytest.approx(0.25)

    def test_raises_when_every_probe_fails(self) -> None:
        # A ladder built on a guessed S would be arbitrary at every step, so this
        # fails loudly rather than defaulting.
        with pytest.raises(CMaxError, match="probe request"):
            probe_service_time(
                None,
                endpoint=ENDPOINT,
                voice="af_heart",
                texts=TEXTS,
                requests=3,
                invoke=lambda *a, **k: _invoke_result(outcome=InvokeOutcome.SERVER_5XX),
            )

    def test_failure_names_the_endpoint(self) -> None:
        with pytest.raises(CMaxError, match=ENDPOINT):
            probe_service_time(
                None,
                endpoint=ENDPOINT,
                voice="af_heart",
                texts=TEXTS,
                requests=1,
                invoke=lambda *a, **k: _invoke_result(outcome=InvokeOutcome.MODEL_ERROR),
            )


# --------------------------------------------------------------------------- #
# Step projection
# --------------------------------------------------------------------------- #


class TestSummarizeStep:
    def test_a_clean_step_is_usable(self) -> None:
        summary = _summarize(_step())
        assert summary.usable
        assert summary.unusable_reason is None
        assert summary.concurrency_mean == pytest.approx(2.0)
        assert summary.ttfab_p95_ms == pytest.approx(100.0)

    def test_a_capacity_change_makes_the_step_unusable(self) -> None:
        # C_max is per instance. A fleet that grew mid-step raises achieved
        # throughput for a reason unrelated to the knee.
        summary = _summarize(_step(instance_counts=(1, 2)))
        assert not summary.usable
        assert summary.capacity_changed
        assert "instance count changed" in (summary.unusable_reason or "")

    def test_an_empty_window_is_unusable(self) -> None:
        summary = _summarize(_step(achieved_rps=0.0))
        assert not summary.usable
        assert "no requests completed" in (summary.unusable_reason or "")

    def test_skipped_dispatches_make_the_step_unusable(self) -> None:
        # The client, not the server, was the limit — and a knee measured under
        # that condition looks exactly like a real one.
        summary = _summarize(_step(skipped=4))
        assert not summary.usable
        assert summary.dispatch_skipped == 4
        assert "measures the benchmark" in (summary.unusable_reason or "")

    def test_saturation_is_recorded_rather_than_hidden(self) -> None:
        summary = _summarize(_step(offered_rps=10.0, achieved_rps=5.0))
        assert summary.saturated
        # Saturation is an outcome, not an error: the step stays usable data.
        assert summary.usable

    def test_a_growing_queue_is_not_settled(self) -> None:
        summary = _summarize(_step(in_flight=tuple(range(1, 11))))
        assert not summary.settled

    def test_carries_the_target_and_the_offered_rate_separately(self) -> None:
        summary = _summarize(_step(offered_rps=8.0))
        assert summary.offered_rps == pytest.approx(8.0)
        assert summary.target_concurrency == pytest.approx(1.6)


# --------------------------------------------------------------------------- #
# Knee
# --------------------------------------------------------------------------- #


class TestFindKnee:
    def test_takes_the_highest_passing_step(self) -> None:
        steps = [
            _summary(step_index=0, target_concurrency=1.0, ttfab_p95_ms=80.0),
            _summary(step_index=1, target_concurrency=2.0, ttfab_p95_ms=120.0),
            _summary(step_index=2, target_concurrency=4.0, ttfab_p95_ms=900.0),
        ]
        knee = find_knee(steps, 300)
        assert knee is not None
        assert knee.step_index == 1
        assert knee.bracketed

    def test_reports_measured_concurrency_not_the_offered_rate(self) -> None:
        # At the knee the two differ, and the scaling policy sees the measured
        # quantity — so that is what C_max has to be expressed in.
        steps = [_summary(target_concurrency=2.0, concurrency_mean=3.4)]
        knee = find_knee(steps, 300)
        assert knee is not None
        assert knee.concurrency == pytest.approx(3.4)

    def test_pass_fail_pass_is_not_bracketed(self) -> None:
        # The regression this function was rewritten for. Walking upward and
        # accumulating a "something failed" flag would mark this bracketed on the
        # strength of the failure at 2.0 — which sits *below* the reported knee at
        # 3.0, so it brackets nothing.
        steps = [
            _summary(step_index=0, target_concurrency=1.0, ttfab_p95_ms=80.0),
            _summary(step_index=1, target_concurrency=2.0, ttfab_p95_ms=900.0),
            _summary(step_index=2, target_concurrency=3.0, ttfab_p95_ms=120.0),
        ]
        knee = find_knee(steps, 300)
        assert knee is not None
        assert knee.step_index == 2
        assert not knee.bracketed
        assert knee.is_lower_bound

    def test_still_passing_at_the_top_is_a_lower_bound(self) -> None:
        steps = [
            _summary(step_index=0, target_concurrency=1.0),
            _summary(step_index=1, target_concurrency=16.0),
        ]
        knee = find_knee(steps, 300)
        assert knee is not None
        assert knee.step_index == 1
        assert knee.is_lower_bound

    def test_returns_none_when_nothing_passes(self) -> None:
        steps = [_summary(ttfab_p95_ms=900.0), _summary(step_index=1, ttfab_p95_ms=1200.0)]
        assert find_knee(steps, 300) is None

    def test_a_saturated_step_cannot_host_the_knee(self) -> None:
        # Latency percentiles read fine while requests are being shed; achieved
        # throughput is what gives it away.
        steps = [
            _summary(step_index=0, target_concurrency=1.0, ttfab_p95_ms=80.0),
            _summary(step_index=1, target_concurrency=2.0, ttfab_p95_ms=90.0, saturated=True),
        ]
        knee = find_knee(steps, 300)
        assert knee is not None
        assert knee.step_index == 0
        assert knee.bracketed

    def test_an_unsettled_step_cannot_host_the_knee(self) -> None:
        # A step whose queue is still growing at the end of the window *is* the
        # measurement, whatever its percentiles read.
        steps = [
            _summary(step_index=0, target_concurrency=1.0),
            _summary(step_index=1, target_concurrency=2.0, settled=False),
        ]
        knee = find_knee(steps, 300)
        assert knee is not None
        assert knee.step_index == 0

    def test_unusable_steps_cannot_host_the_knee(self) -> None:
        steps = [
            _summary(step_index=0, target_concurrency=1.0),
            _summary(step_index=1, target_concurrency=8.0, usable=False),
        ]
        knee = find_knee(steps, 300)
        assert knee is not None
        assert knee.step_index == 0
        # This unusable step met the budget (100ms), so it is not evidence the
        # knee was bracketed — see test_an_unusable_step_that_blew_the_budget_brackets
        # for the case where it is.
        assert not knee.bracketed

    def test_an_unusable_step_that_blew_the_budget_brackets(self) -> None:
        # The live bidi ladder against speech-kokoro-82m: steps 2 and 3 were
        # excluded because the client ran out of workers, so the knee at step 1
        # was reported as a lower bound with "extend --target-concurrency" — on a
        # ladder that had already offered 3x the knee's rate and watched p95 go
        # from 276ms to 1927ms. Extending it would have measured nothing new.
        #
        # A client-limited step still measures real latency on the requests it
        # dispatched: TTFAB is timed from dispatch inside `invoke`, and the
        # dispatcher skips rather than queues, so a full pool cannot inflate it.
        steps = [
            _summary(step_index=1, target_concurrency=1.63, ttfab_p95_ms=276.0),
            _summary(step_index=2, target_concurrency=2.0, ttfab_p95_ms=1927.0, usable=False),
        ]
        knee = find_knee(steps, 300)
        assert knee is not None
        assert knee.step_index == 1
        assert knee.bracketed
        assert not knee.is_lower_bound

    def test_an_unusable_step_with_no_latency_does_not_bracket(self) -> None:
        # Nothing completed, so there is no p95 to judge and no evidence either
        # way. Bracketing on "did not pass" alone would silently upgrade a
        # lower-bound knee into a firm one.
        steps = [
            _summary(step_index=0, target_concurrency=1.0),
            _summary(step_index=1, target_concurrency=8.0, ttfab_p95_ms=None, usable=False),
        ]
        knee = find_knee(steps, 300)
        assert knee is not None
        assert knee.is_lower_bound

    def test_an_unusable_step_does_not_bracket_on_saturation_alone(self) -> None:
        # `saturated` compares achieved against *offered* rps, and a skipped
        # dispatch lowers achieved without the server ever seeing the request —
        # so a client-limited step reads as saturated even when the endpoint kept
        # up. Only its measured p95 may be trusted here.
        steps = [
            _summary(step_index=0, target_concurrency=1.0),
            _summary(
                step_index=1,
                target_concurrency=8.0,
                ttfab_p95_ms=90.0,
                saturated=True,
                usable=False,
            ),
        ]
        knee = find_knee(steps, 300)
        assert knee is not None
        assert knee.step_index == 0
        assert knee.is_lower_bound

    def test_a_step_with_no_ttfab_samples_cannot_pass(self) -> None:
        assert find_knee([_summary(ttfab_p95_ms=None)], 300) is None

    def test_falls_back_to_the_target_when_the_monitor_produced_nothing(self) -> None:
        # Only fires when the 1Hz monitor recorded no samples at all, which would
        # otherwise discard a perfectly good step.
        knee = find_knee([_summary(target_concurrency=2.0, concurrency_mean=None)], 300)
        assert knee is not None
        assert knee.concurrency == pytest.approx(2.0)

    def test_the_knee_moves_with_the_budget(self) -> None:
        # The property that makes one ladder answer four SLO questions.
        steps = [
            _summary(step_index=0, target_concurrency=1.0, ttfab_p95_ms=40.0),
            _summary(step_index=1, target_concurrency=2.0, ttfab_p95_ms=120.0),
            _summary(step_index=2, target_concurrency=4.0, ttfab_p95_ms=280.0),
            _summary(step_index=3, target_concurrency=8.0, ttfab_p95_ms=1400.0),
        ]
        assert find_knee(steps, 50).concurrency == pytest.approx(1.0)
        assert find_knee(steps, 150).concurrency == pytest.approx(2.0)
        assert find_knee(steps, 300).concurrency == pytest.approx(4.0)


class TestFindThroughputCeiling:
    def test_takes_the_highest_sustained_rate(self) -> None:
        steps = [
            _summary(step_index=0, target_concurrency=1.0, offered_rps=5.0),
            _summary(step_index=1, target_concurrency=2.0, offered_rps=10.0),
        ]
        ceiling = find_throughput_ceiling(steps, 0.2)
        assert ceiling is not None
        assert ceiling.max_sustained_rps == pytest.approx(10.0)
        assert ceiling.step_index == 1

    def test_converts_to_concurrency_through_littles_law(self) -> None:
        # C = lambda x S, so 10 rps at 200ms is 2.0 of *useful* concurrency. This is the
        # unit the planner compares against a knee, so getting the conversion wrong would
        # misprice every fleet built on the ceiling.
        ceiling = find_throughput_ceiling([_summary(offered_rps=10.0)], 0.2)
        assert ceiling is not None
        assert ceiling.concurrency == pytest.approx(2.0)

    def test_the_concurrency_round_trips_back_to_the_measured_rate(self) -> None:
        # The invariant that makes this number safe to plan on: feeding it to
        # lambda_cap_per_instance must return the rate we actually observed, not one
        # inferred from it. If these ever diverge the plan permits a rate nothing measured.
        ceiling = find_throughput_ceiling([_summary(offered_rps=13.03)], 0.10602401316328536)
        assert ceiling is not None
        assert lambda_cap_per_instance(ceiling.concurrency, 0.10602401316328536) == pytest.approx(
            ceiling.max_sustained_rps
        )

    def test_uses_the_uncontended_s_not_the_steps_own(self) -> None:
        # At the ceiling the step's own s_mean_s already contains queueing. Using it would
        # inflate the concurrency by exactly the wait this bound exists to exclude — here
        # 4x, since the contended figure is 4x the uncontended one.
        step = _summary(offered_rps=10.0, s_mean_s=0.8, s_p95_s=1.2)
        ceiling = find_throughput_ceiling([step], 0.2)
        assert ceiling is not None
        assert ceiling.concurrency == pytest.approx(2.0)

    def test_records_observed_concurrency_beside_the_useful_figure(self) -> None:
        # Not the same unit, and the gap is the point: ConcurrentRequestsPerModel — what
        # the deployed scaling policy tracks — reports the observed number.
        ceiling = find_throughput_ceiling(
            [_summary(offered_rps=10.0, concurrency_mean=4.2)],
            0.2,
        )
        assert ceiling is not None
        assert ceiling.concurrency == pytest.approx(2.0)
        assert ceiling.observed_concurrency == pytest.approx(4.2)
        assert ceiling.queueing_multiple == pytest.approx(2.1)

    def test_a_saturated_step_cannot_be_the_ceiling(self) -> None:
        # The retraction this function exists for. A saturated step's *measured*
        # concurrency is accumulated backlog, and reading it as capacity is how a ladder
        # that peaked at 13 rps came to look like it held 11 concurrent requests.
        steps = [
            _summary(step_index=0, target_concurrency=2.0, offered_rps=12.89, achieved_rps=13.03),
            _summary(
                step_index=1,
                target_concurrency=3.0,
                offered_rps=19.33,
                achieved_rps=16.88,
                saturated=True,
                concurrency_mean=11.15,
            ),
        ]
        ceiling = find_throughput_ceiling(steps, 0.10602401316328536)
        assert ceiling is not None
        assert ceiling.step_index == 0
        assert ceiling.max_sustained_rps == pytest.approx(13.03)
        # Would have been ~1.79 read off the saturated step's rate, and 11.15 read off its
        # concurrency. Both overstate an instance that tops out near 1.38.
        assert ceiling.concurrency == pytest.approx(1.3813, abs=1e-3)

    def test_an_unsettled_step_cannot_be_the_ceiling(self) -> None:
        # Achieved can match offered while the queue is still filling; the window just
        # ended before the backlog surfaced. Same reasoning as the knee's version.
        steps = [
            _summary(step_index=0, target_concurrency=1.0, offered_rps=5.0),
            _summary(step_index=1, target_concurrency=2.0, offered_rps=10.0, settled=False),
        ]
        ceiling = find_throughput_ceiling(steps, 0.2)
        assert ceiling is not None
        assert ceiling.step_index == 0

    def test_a_client_limited_step_can_still_be_the_ceiling(self) -> None:
        # Where this deliberately parts company with find_knee. `usable=False` means the
        # step cannot host a *latency* knee, but the rate it delivered is a rate the server
        # delivered, and discarding it is what hid kokoro's real limit.
        steps = [
            _summary(step_index=0, target_concurrency=1.0, offered_rps=5.0),
            _summary(step_index=1, target_concurrency=2.0, offered_rps=10.0, usable=False),
        ]
        ceiling = find_throughput_ceiling(steps, 0.2)
        assert ceiling is not None
        assert ceiling.step_index == 1

    def test_a_capacity_change_disqualifies_a_step(self) -> None:
        # Throughput that rose because a second instance appeared is not a per-instance
        # ceiling — the same reason it cannot host a knee.
        steps = [
            _summary(step_index=0, target_concurrency=1.0, offered_rps=5.0),
            _summary(step_index=1, target_concurrency=2.0, offered_rps=10.0, capacity_changed=True),
        ]
        ceiling = find_throughput_ceiling(steps, 0.2)
        assert ceiling is not None
        assert ceiling.step_index == 0

    def test_a_step_with_no_completions_is_not_a_ceiling(self) -> None:
        assert find_throughput_ceiling([_summary(offered_rps=5.0, completed=0)], 0.2) is None

    def test_returns_none_when_every_step_saturated(self) -> None:
        # The honest answer for a ladder whose lowest rate is already past capacity: no
        # ceiling was found, rather than a fabricated one from the least-bad step.
        steps = [
            _summary(step_index=0, saturated=True),
            _summary(step_index=1, target_concurrency=2.0, saturated=True),
        ]
        assert find_throughput_ceiling(steps, 0.2) is None

    def test_a_saturated_step_above_brackets_the_ceiling(self) -> None:
        steps = [
            _summary(step_index=0, target_concurrency=1.0, offered_rps=10.0),
            _summary(step_index=1, target_concurrency=2.0, offered_rps=20.0, saturated=True),
        ]
        ceiling = find_throughput_ceiling(steps, 0.2)
        assert ceiling is not None
        assert ceiling.bracketed

    def test_running_out_of_ladder_makes_it_a_lower_bound(self) -> None:
        ceiling = find_throughput_ceiling([_summary(target_concurrency=8.0, offered_rps=40.0)], 0.2)
        assert ceiling is not None
        assert not ceiling.bracketed
        assert ceiling.is_lower_bound

    def test_skipped_dispatches_make_it_a_lower_bound_too(self) -> None:
        # A second, independent reason to distrust the number, and it wants the opposite
        # fix: --max-workers, not a longer ladder. Bracketed and still a lower bound.
        steps = [
            _summary(step_index=0, target_concurrency=1.0, offered_rps=10.0, skipped=29),
            _summary(step_index=1, target_concurrency=2.0, offered_rps=20.0, saturated=True),
        ]
        ceiling = find_throughput_ceiling(steps, 0.2)
        assert ceiling is not None
        assert ceiling.bracketed
        assert ceiling.dispatch_skipped == 29
        assert ceiling.is_lower_bound

    def test_a_tie_on_rate_prefers_the_lower_target(self) -> None:
        # The same rate reached from a lower target is the less contended measurement of
        # it, so its latency and observed concurrency are the ones worth recording.
        steps = [
            _summary(step_index=0, target_concurrency=1.0, offered_rps=10.0, ttfab_p95_ms=150.0),
            _summary(step_index=1, target_concurrency=4.0, offered_rps=10.0, ttfab_p95_ms=900.0),
        ]
        ceiling = find_throughput_ceiling(steps, 0.2)
        assert ceiling is not None
        assert ceiling.step_index == 0
        assert ceiling.p95_ttfab_ms == pytest.approx(150.0)

    def test_records_the_latency_at_the_ceiling(self) -> None:
        # A ceiling reached with latency well inside budget is the finding: it proves the
        # limit was throughput and not the SLO.
        ceiling = find_throughput_ceiling([_summary(offered_rps=10.0, ttfab_p95_ms=551.0)], 0.2)
        assert ceiling is not None
        assert ceiling.p95_ttfab_ms == pytest.approx(551.0)

    def test_rejects_a_non_positive_service_time(self) -> None:
        with pytest.raises(ValueError, match="s_mean_s must be positive"):
            find_throughput_ceiling([_summary()], 0.0)

    def test_no_steps_at_all(self) -> None:
        assert find_throughput_ceiling([], 0.2) is None


class TestMedianCeiling:
    def _ceiling(self, rps: float, **kwargs):
        return find_throughput_ceiling([_summary(offered_rps=rps, **kwargs)], 0.2)

    def test_takes_the_median_rate_across_runs(self) -> None:
        merged = median_ceiling([self._ceiling(r) for r in (10.0, 11.0, 40.0)])
        assert merged is not None
        assert merged.max_sustained_rps == pytest.approx(11.0)

    def test_returns_a_run_that_really_happened(self) -> None:
        # Not a synthetic average: step_index, p95 and dispatch_skipped have to describe
        # one real step, or the artifact reports a step nobody ran.
        merged = median_ceiling(
            [
                self._ceiling(10.0, ttfab_p95_ms=100.0, skipped=1),
                self._ceiling(11.0, ttfab_p95_ms=200.0, skipped=2),
                self._ceiling(40.0, ttfab_p95_ms=900.0, skipped=3),
            ]
        )
        assert merged is not None
        assert (merged.p95_ttfab_ms, merged.dispatch_skipped) == (200.0, 2)

    def test_an_even_count_prefers_the_lower_middle(self) -> None:
        # The conservative choice for a capacity number: understating the fleet's per-
        # instance ceiling oversizes the fleet, which is the safe direction to be wrong.
        merged = median_ceiling([self._ceiling(r) for r in (10.0, 20.0)])
        assert merged is not None
        assert merged.max_sustained_rps == pytest.approx(10.0)

    def test_one_bracketing_run_is_enough(self) -> None:
        # A ceiling is a lower bound unless *some* run bracketed it. Requiring every run
        # to have done so would throw away evidence we paid 45 minutes for.
        bracketed = find_throughput_ceiling(
            [
                _summary(step_index=0, target_concurrency=1.0, offered_rps=10.0),
                _summary(step_index=1, target_concurrency=2.0, offered_rps=20.0, saturated=True),
            ],
            0.2,
        )
        merged = median_ceiling([self._ceiling(10.0), bracketed, self._ceiling(10.0)])
        assert merged is not None
        assert merged.bracketed

    def test_reports_how_many_runs_contributed(self) -> None:
        # The denominator that keeps a 0% spread from reading as agreement.
        merged = median_ceiling([self._ceiling(10.0), None, None])
        assert merged is not None
        assert merged.runs_contributing == 1
        assert merged.spread == pytest.approx(0.0)

    def test_reports_relative_spread(self) -> None:
        merged = median_ceiling([self._ceiling(r) for r in (9.0, 10.0, 11.0)])
        assert merged is not None
        assert merged.spread == pytest.approx(0.2)

    def test_no_run_found_a_ceiling(self) -> None:
        assert median_ceiling([None, None]) is None
        assert median_ceiling([]) is None


class TestTheCeilingAgainstTheRealLadder:
    """The committed kokoro artifact, checked offline.

    The whole reason this measurement exists: reading the ladder through the latency lens
    alone reported a knee, and reading its *rate* columns showed the server had already
    stopped keeping up. These pin what the real data says so a future change to either
    function has to confront it.
    """

    def _report(self):
        import json
        from pathlib import Path

        path = (
            Path(__file__).resolve().parents[3]
            / "artifacts"
            / "cmax-kokoro-82m-bidi-g5xlarge-139b9068.json"
        )
        if not path.exists():
            pytest.skip(f"{path.name} not present")
        return CMaxReport.model_validate(json.loads(path.read_text()))

    def test_every_run_agrees_the_ceiling_is_about_thirteen_rps(self) -> None:
        report = self._report()
        rates = [
            c.max_sustained_rps
            for i in range(report.runs)
            if (
                c := find_throughput_ceiling(
                    [s for s in report.steps if s.run_index == i], report.s_mean_s
                )
            )
            is not None
        ]
        assert len(rates) == 3
        assert all(12.5 < r < 13.5 for r in rates), rates

    def test_the_ceiling_lands_near_one_point_four_not_eleven(self) -> None:
        # 11.15 was the measured concurrency at target 3.0 — a saturated step, so that
        # figure is backlog. 1.79 would be its rate read as capacity. Both are wrong.
        report = self._report()
        merged = median_ceiling(
            [
                find_throughput_ceiling(
                    [s for s in report.steps if s.run_index == i], report.s_mean_s
                )
                for i in range(report.runs)
            ]
        )
        assert merged is not None
        assert merged.concurrency == pytest.approx(1.38, abs=0.02)
        assert merged.spread < 0.05

    def test_the_ceiling_sits_above_this_ladders_latency_knees(self) -> None:
        # Kokoro is throughput-bound in general, but *this* ladder never resolved it: its
        # top two steps were unusable, so no knee was ever reported above 1.26. The
        # planner therefore still picks the latency knee here, and it takes the denser
        # re-run to show the ceiling binding. Pinning the honest outcome rather than the
        # predicted one.
        report = self._report()
        merged = median_ceiling(
            [
                find_throughput_ceiling(
                    [s for s in report.steps if s.run_index == i], report.s_mean_s
                )
                for i in range(report.runs)
            ]
        )
        assert merged is not None
        assert max(report.c_max_curve.values()) < merged.concurrency
        assert (
            report.model_copy(update={"throughput_ceiling": merged}).throughput_bound_budgets == []
        )

    def test_most_of_the_residence_at_the_ceiling_is_already_queueing(self) -> None:
        # 2.1x. The instance is not idle at its throughput limit — it is backed up, which
        # is why the deployed scaling_target_value (which tracks the observed figure) and
        # this C_max are not the same number.
        report = self._report()
        merged = median_ceiling(
            [
                find_throughput_ceiling(
                    [s for s in report.steps if s.run_index == i], report.s_mean_s
                )
                for i in range(report.runs)
            ]
        )
        assert merged is not None
        assert merged.queueing_multiple == pytest.approx(2.1, abs=0.1)

    def test_the_ceiling_is_a_lower_bound_because_the_client_throttled(self) -> None:
        # Bracketed — target 3.0 did saturate — yet still a lower bound, because the pool
        # skipped dispatches at the ceiling step. This is the case --max-workers exists
        # for, and the reason the re-run raises it to 64.
        report = self._report()
        merged = median_ceiling(
            [
                find_throughput_ceiling(
                    [s for s in report.steps if s.run_index == i], report.s_mean_s
                )
                for i in range(report.runs)
            ]
        )
        assert merged is not None
        assert merged.bracketed
        assert merged.dispatch_skipped > 0
        assert merged.is_lower_bound


class TestMedianCurve:
    def test_takes_the_median_across_runs(self) -> None:
        # Median, not mean: with runs=3 one bad run must not move the answer, and
        # the derate is not sized to absorb an outlier.
        runs = [[find_knee([_summary(target_concurrency=c)], 300)] for c in (1.0, 1.1, 4.0)]
        curve, _ = median_curve(runs, [300])
        assert curve[300] == pytest.approx(1.1)

    def test_reports_relative_spread(self) -> None:
        runs = [[find_knee([_summary(target_concurrency=c)], 300)] for c in (0.9, 1.0, 1.1)]
        _, spread = median_curve(runs, [300])
        assert spread[300] == pytest.approx(0.2)
        assert spread[300] > SPREAD_WARN_THRESHOLD - 1e-9

    def test_a_single_run_has_no_spread(self) -> None:
        runs = [[find_knee([_summary(target_concurrency=1.0)], 300)]]
        _, spread = median_curve(runs, [300])
        assert spread[300] == pytest.approx(0.0)

    def test_a_budget_no_run_found_is_absent_not_zero(self) -> None:
        # Absent means "not measured"; zero would be read as "capacity is zero",
        # and the Measured validator rejects a zero anyway.
        runs = [[find_knee([_summary(ttfab_p95_ms=200.0)], 300)]]
        curve, spread = median_curve(runs, [50, 300])
        assert 50 not in curve
        assert 50 not in spread
        assert 300 in curve

    def test_no_runs_at_all_is_an_empty_curve(self) -> None:
        assert median_curve([], [300]) == ({}, {})


class TestUncontendedServiceTime:
    def test_prefers_the_lowest_usable_step(self) -> None:
        # Not the knee step: service time there already includes queueing, and
        # C_slo_cap = W_max / S would then bound a wait it had already counted.
        steps = [
            _summary(step_index=1, target_concurrency=4.0, s_mean_s=0.9, s_p95_s=1.4),
            _summary(step_index=0, target_concurrency=0.5, s_mean_s=0.21, s_p95_s=0.3),
        ]
        assert uncontended_service_time(steps, _probe()) == (
            pytest.approx(0.21),
            pytest.approx(0.3),
        )

    def test_ignores_saturated_steps(self) -> None:
        steps = [
            _summary(step_index=0, target_concurrency=0.5, s_mean_s=5.0, saturated=True),
            _summary(step_index=1, target_concurrency=1.0, s_mean_s=0.22, s_p95_s=0.31),
        ]
        s_mean, _ = uncontended_service_time(steps, _probe())
        assert s_mean == pytest.approx(0.22)

    def test_ignores_unusable_steps(self) -> None:
        steps = [
            _summary(step_index=0, target_concurrency=0.5, s_mean_s=5.0, usable=False),
            _summary(step_index=1, target_concurrency=1.0, s_mean_s=0.22, s_p95_s=0.31),
        ]
        s_mean, _ = uncontended_service_time(steps, _probe())
        assert s_mean == pytest.approx(0.22)

    def test_falls_back_to_the_probe(self) -> None:
        # The probe is uncontended by construction, so it is the more
        # conservative source rather than a worse one.
        assert uncontended_service_time([], _probe(s_mean_s=0.3, s_p95_s=0.5)) == (
            pytest.approx(0.3),
            pytest.approx(0.5),
        )

    def test_never_returns_a_p95_below_the_mean(self) -> None:
        # CMaxReport rejects contradictory percentiles, so a degenerate probe
        # must not be able to produce an unconstructable report.
        s_mean, s_p95 = uncontended_service_time([], _probe(s_mean_s=0.4, s_p95_s=0.1))
        assert s_p95 >= s_mean

    def test_a_step_missing_service_time_is_skipped(self) -> None:
        steps = [
            _summary(step_index=0, target_concurrency=0.5, s_mean_s=None),
            _summary(step_index=1, target_concurrency=1.0, s_mean_s=0.22, s_p95_s=0.31),
        ]
        s_mean, _ = uncontended_service_time(steps, _probe())
        assert s_mean == pytest.approx(0.22)


# --------------------------------------------------------------------------- #
# Ladder
# --------------------------------------------------------------------------- #


class _RecordingRunner:
    """Records how each step was launched and returns a scripted result."""

    def __init__(self, *, saturated_at: Sequence[int] = (), window_s: float = WINDOW_S) -> None:
        self._saturated_at = set(saturated_at)
        self._window_s = window_s
        self.calls: list[dict] = []

    def __call__(self, client, **kwargs) -> StepResult:
        self.calls.append(kwargs)
        index = kwargs["step_index"]
        offered = kwargs["offered_rps"]
        return _step(
            step_index=index,
            offered_rps=offered,
            achieved_rps=offered * 0.5 if index in self._saturated_at else offered,
            ended_ts=T0_TS + index * 1000.0,
            window_s=self._window_s,
        )


def _run_ladder(runner: _RecordingRunner, *, clock: Clock | None = None, **kwargs) -> LadderRun:
    defaults = {
        "model": MODEL,
        "endpoint": ENDPOINT,
        "voice": "af_heart",
        "texts": TEXTS,
        "s_mean_s": 0.25,
        "target_concurrencies": (1.0, 2.0, 4.0),
        "hold_s": 240.0,
        "measure_window_s": WINDOW_S,
        "settle_between_steps_s": 0.0,
        "seed": 1234,
        "step_runner": runner,
    }
    defaults.update(kwargs)
    if clock is not None:
        defaults["clock"] = clock
    return run_ladder(None, **defaults)


class TestRunLadder:
    def test_rejects_a_window_larger_than_the_hold(self) -> None:
        # The window is the trailing part of a step, not an addition to it.
        with pytest.raises(ValueError, match="must not exceed hold_s"):
            _run_ladder(_RecordingRunner(), hold_s=60.0, measure_window_s=120.0)

    def test_converts_each_target_into_a_rate_through_S(self) -> None:
        runner = _RecordingRunner()
        _run_ladder(runner, s_mean_s=0.25, target_concurrencies=(0.5, 1.0, 2.0))
        assert [c["offered_rps"] for c in runner.calls] == [
            pytest.approx(2.0),
            pytest.approx(4.0),
            pytest.approx(8.0),
        ]

    def test_walks_the_ladder_upward_whatever_order_it_was_given(self) -> None:
        runner = _RecordingRunner()
        _run_ladder(runner, target_concurrencies=(4.0, 1.0, 2.0))
        assert [c["offered_rps"] for c in runner.calls] == sorted(
            c["offered_rps"] for c in runner.calls
        )

    def test_holds_for_hold_s_and_measures_only_the_window(self) -> None:
        runner = _RecordingRunner(window_s=WINDOW_S)
        ladder = _run_ladder(runner, hold_s=240.0, measure_window_s=WINDOW_S)
        assert all(c["duration_s"] == 240.0 for c in runner.calls)
        # 10 rps over the 60s window, not over the 240s hold.
        assert ladder.steps[1].achieved_rps == pytest.approx(
            runner.calls[1]["offered_rps"], rel=0.1
        )

    def test_sizes_the_pool_from_the_top_of_the_ladder(self) -> None:
        # Every step gets the same pool: sizing per step would make the bottom
        # steps' dispatch behaviour differ from the top's for no good reason.
        runner = _RecordingRunner()
        _run_ladder(runner, target_concurrencies=(1.0, 16.0))
        assert {c["max_workers"] for c in runner.calls} == {worker_count(16.0)}

    def test_the_override_reaches_every_step(self) -> None:
        # A raised pool that only reached some steps would leave the others client-limited
        # while the artifact recorded one --max-workers, which is worse than not raising it.
        runner = _RecordingRunner()
        _run_ladder(runner, target_concurrencies=(1.0, 3.0), max_workers=64)
        assert {c["max_workers"] for c in runner.calls} == {64}

    def test_varies_the_seed_per_step_and_per_run(self) -> None:
        # Each step stays individually reproducible without every step drawing
        # the same arrival pattern.
        first = _RecordingRunner()
        _run_ladder(first, seed=1234, run_index=0)
        second = _RecordingRunner()
        _run_ladder(second, seed=1234, run_index=1)

        # seed + step_index + 1000 * run_index: the run stride is wide enough
        # that no two runs can ever land on the same per-step seed.
        assert [c["seed"] for c in first.calls] == [1234, 1235, 1236]
        assert [c["seed"] for c in second.calls] == [2234, 2235, 2236]

    def test_an_unseeded_run_stays_unseeded(self) -> None:
        runner = _RecordingRunner()
        _run_ladder(runner, seed=None)
        assert all(c["seed"] is None for c in runner.calls)

    def test_stops_after_two_consecutive_saturated_steps(self) -> None:
        # Past the knee, higher rates cost money and measure only how badly the
        # endpoint fails.
        runner = _RecordingRunner(saturated_at=(1, 2))
        ladder = _run_ladder(runner, target_concurrencies=(1.0, 2.0, 4.0, 8.0, 16.0))
        assert len(runner.calls) == 3
        assert ladder.truncated_at == 2

    def test_a_good_step_resets_the_saturation_counter(self) -> None:
        # One noisy step must not truncate a ladder that recovers.
        runner = _RecordingRunner(saturated_at=(0, 2, 3))
        ladder = _run_ladder(runner, target_concurrencies=(1.0, 2.0, 4.0, 8.0, 16.0))
        assert len(runner.calls) == 4
        assert ladder.truncated_at == 3

    def test_a_complete_ladder_is_not_marked_truncated(self) -> None:
        ladder = _run_ladder(_RecordingRunner())
        assert ladder.truncated_at is None

    def test_drains_between_steps_but_not_after_the_last(self) -> None:
        # Without the drain a step inherits the previous queue and its knee lands
        # low for the wrong reason.
        recorder = RecordingClock()
        _run_ladder(
            _RecordingRunner(),
            settle_between_steps_s=7.0,
            clock=recorder.clock(),
            target_concurrencies=(1.0, 2.0, 4.0),
        )
        assert recorder.slept == [7.0, 7.0]

    def test_a_truncated_ladder_does_not_drain_on_the_way_out(self) -> None:
        recorder = RecordingClock()
        _run_ladder(
            _RecordingRunner(saturated_at=(0, 1)),
            settle_between_steps_s=7.0,
            clock=recorder.clock(),
            target_concurrencies=(1.0, 2.0, 4.0),
        )
        assert recorder.slept == [7.0]

    def test_records_the_instance_counts_it_saw(self) -> None:
        ladder = _run_ladder(_RecordingRunner())
        assert ladder.instance_counts == (1,)

    def test_stamps_the_run_index_on_every_summary(self) -> None:
        ladder = _run_ladder(_RecordingRunner(), run_index=2)
        assert {s.run_index for s in ladder.steps} == {2}

    def test_passes_the_tripwire_and_sink_through_to_the_step(self) -> None:
        runner = _RecordingRunner()
        sink: list[LoadEvent] = []
        _run_ladder(runner, instance_count_fetch=lambda: 1, event_sink=sink.append)
        assert all(c["instance_count_fetch"] is not None for c in runner.calls)
        assert all(c["event_sink"] is not None for c in runner.calls)


class TestRunLadderAgainstAFakeServer:
    """One pass through the real ``run_step``, to keep the two wired together.

    Deliberately asserts shape rather than rates: the numbers a half-second step
    produces are timing-dependent, and the open-loop property they would be
    testing is already covered in ``test_loadgen.py``.
    """

    def test_produces_a_usable_summary(self) -> None:
        server = FakeServer(service_time_s=0.05, slots=4)
        ladder = run_ladder(
            None,
            model=MODEL,
            endpoint=ENDPOINT,
            voice="af_heart",
            texts=TEXTS,
            s_mean_s=0.05,
            target_concurrencies=(0.5,),
            hold_s=0.5,
            measure_window_s=0.5,
            settle_between_steps_s=0.0,
            arrival="fixed",
            seed=7,
            # ``invoke`` goes through run_ladder's own parameter rather than the
            # partial: the ladder forwards the transport to every step, and
            # binding it here would hide a break in that forwarding.
            invoke=server,
            step_runner=functools.partial(run_step, monitor_interval_s=0.05),
        )

        assert len(ladder.steps) == 1
        step = ladder.steps[0]
        assert step.offered_rps == pytest.approx(10.0)
        assert step.usable, step.unusable_reason
        assert step.completed >= 1
        assert step.concurrency_mean is not None
        # The service time the fake actually served, recovered end to end.
        assert step.s_mean_s == pytest.approx(0.05, rel=0.5)
        assert server.completed >= 1


# --------------------------------------------------------------------------- #
# CloudWatch join
# --------------------------------------------------------------------------- #


def _joinable(count: int = 2, *, window_s: float = WINDOW_S) -> tuple[list, list]:
    """``(steps, results)`` for ``count`` steps ending at spaced-out times.

    Step 0 ends five minutes ago and step ``count-1`` thirty seconds ago, so the
    oldest window is already past a 120s settle delay and the newest is not.
    """
    offsets = [300.0, 30.0] if count == 2 else [300.0 - i * 60.0 for i in range(count)]
    results = [
        _step(
            step_index=i,
            ended_ts=(NOW - timedelta(seconds=offsets[i])).timestamp(),
            window_s=window_s,
        )
        for i in range(count)
    ]
    steps = [_summarize(r, window_s=window_s) for r in results]
    return steps, results


class TestJoinCloudwatch:
    def test_takes_the_settle_wait_on_the_newest_window(self) -> None:
        # Waiting on the oldest returns instantly — its window is already older
        # than the delay — and the newest step is then read before CloudWatch has
        # aggregated it, which surfaces as missing data rather than a skipped wait.
        steps, results = _joinable()
        slept: list[float] = []

        join_cloudwatch(
            FakeCloudWatch(),
            endpoint=ENDPOINT,
            variant="primary",
            steps=steps,
            results=results,
            measure_window_s=WINDOW_S,
            settle_delay_s=120.0,
            sleep=slept.append,
            now=lambda: NOW,
        )
        assert slept == [pytest.approx(90.0)]

    def test_joins_server_side_metrics_onto_each_step(self) -> None:
        steps, results = _joinable()
        joined = join_cloudwatch(
            FakeCloudWatch(),
            endpoint=ENDPOINT,
            variant="primary",
            steps=steps,
            results=results,
            measure_window_s=WINDOW_S,
            settle_delay_s=0.0,
            sleep=lambda _: None,
            now=lambda: NOW,
        )
        assert len(joined) == 2
        for step in joined:
            assert step.server_concurrency_mean == pytest.approx(2.5)
            # ModelLatency is published in microseconds; reporting it raw would
            # inflate every latency 1000x and move the knee off the ladder.
            assert step.server_model_latency_p95_ms == pytest.approx(250.0)
            assert step.server_5xx_total == pytest.approx(3.0)
            assert step.gpu_utilization_mean == pytest.approx(71.0)

    def test_an_idle_cpu_is_zero_not_missing(self) -> None:
        steps, results = _joinable(1)
        joined = join_cloudwatch(
            FakeCloudWatch(),
            endpoint=ENDPOINT,
            variant="primary",
            steps=steps,
            results=results,
            measure_window_s=WINDOW_S,
            settle_delay_s=0.0,
            sleep=lambda _: None,
            now=lambda: NOW,
        )
        assert joined[0].cpu_utilization_mean == pytest.approx(0.0)

    def test_records_the_client_server_agreement(self) -> None:
        # The cross-check that catches a dispatcher or connection-pool
        # bottleneck, which from the client side looks like server saturation.
        steps, results = _joinable(1)
        joined = join_cloudwatch(
            FakeCloudWatch(),
            endpoint=ENDPOINT,
            variant="primary",
            steps=steps,
            results=results,
            measure_window_s=WINDOW_S,
            settle_delay_s=0.0,
            sleep=lambda _: None,
            now=lambda: NOW,
        )
        assert joined[0].concurrency_agreement is not None

    def test_a_broken_cloudwatch_does_not_lose_the_curve(self) -> None:
        # Telemetry degrades the report; it must never cost a measured curve.
        steps, results = _joinable()
        joined = join_cloudwatch(
            FakeCloudWatch(raises=True),
            endpoint=ENDPOINT,
            variant="primary",
            steps=steps,
            results=results,
            measure_window_s=WINDOW_S,
            settle_delay_s=0.0,
            sleep=lambda _: None,
            now=lambda: NOW,
        )
        assert len(joined) == len(steps)
        assert [s.ttfab_p95_ms for s in joined] == [s.ttfab_p95_ms for s in steps]
        assert all(s.server_concurrency_mean is None for s in joined)

    def test_no_results_returns_the_steps_untouched(self) -> None:
        steps, _ = _joinable()
        cw = FakeCloudWatch()
        assert join_cloudwatch(
            cw,
            endpoint=ENDPOINT,
            variant="primary",
            steps=steps,
            results=[],
            measure_window_s=WINDOW_S,
        ) == list(steps)
        assert cw.calls == []

    def test_fetches_the_measure_window_not_the_whole_step(self) -> None:
        steps, results = _joinable(1)
        cw = FakeCloudWatch()
        join_cloudwatch(
            cw,
            endpoint=ENDPOINT,
            variant="primary",
            steps=steps,
            results=results,
            measure_window_s=WINDOW_S,
            settle_delay_s=0.0,
            sleep=lambda _: None,
            now=lambda: NOW,
        )
        call = cw.calls[0]
        assert (call["EndTime"] - call["StartTime"]).total_seconds() == pytest.approx(WINDOW_S)
        assert call["Period"] == cmax.observe.HIGH_RES_PERIOD_S


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #


def _build(**kwargs) -> CMaxReport:
    defaults = {
        "model": MODEL,
        "endpoint": ENDPOINT,
        "instance_type": "ml.g5.xlarge",
        "run_id": "abc123",
        "ladders": [
            _ladder(
                [
                    _summary(step_index=0, target_concurrency=1.0, ttfab_p95_ms=80.0),
                    _summary(step_index=1, target_concurrency=2.0, ttfab_p95_ms=280.0),
                    _summary(step_index=2, target_concurrency=4.0, ttfab_p95_ms=900.0),
                ]
            )
        ],
        "probe": _probe(),
        "budgets": (150, 300),
        "frozen": True,
    }
    defaults.update(kwargs)
    return build_report(**defaults)


class TestBuildReport:
    def test_reports_a_curve_per_budget(self) -> None:
        report = _build()
        assert report.c_max_curve == {150: pytest.approx(1.0), 300: pytest.approx(2.0)}
        assert {k.ttfab_budget_ms for k in report.knees} == {150, 300}

    def test_raises_when_no_step_met_any_budget(self) -> None:
        # A real finding — the lowest rate is already past capacity — but it is
        # not a curve, so it fails rather than serializing an empty one.
        with pytest.raises(CMaxError, match="no ladder step met any TTFAB budget"):
            _build(ladders=[_ladder([_summary(ttfab_p95_ms=5000.0)])])

    def test_the_error_says_what_to_change(self) -> None:
        with pytest.raises(CMaxError, match="--target-concurrency"):
            _build(ladders=[_ladder([_summary(ttfab_p95_ms=5000.0)])])

    def test_records_the_derate_without_applying_it(self) -> None:
        # shared.capacity.c_target applies it once. Applying it here as well
        # would shrink every planned fleet by a second 0.875.
        report = _build(derate=0.5)
        assert report.derate == pytest.approx(0.5)
        assert report.c_max_curve[300] == pytest.approx(2.0)

    def test_takes_S_from_the_lowest_step_not_the_knee(self) -> None:
        report = _build(
            ladders=[
                _ladder(
                    [
                        _summary(step_index=0, target_concurrency=1.0, s_mean_s=0.2, s_p95_s=0.25),
                        _summary(step_index=1, target_concurrency=8.0, s_mean_s=1.9, s_p95_s=2.4),
                    ]
                )
            ]
        )
        assert report.s_mean_s == pytest.approx(0.2)

    def test_medians_the_curve_across_runs(self) -> None:
        ladders = [
            _ladder([_summary(target_concurrency=c, ttfab_p95_ms=280.0)], run_index=i)
            for i, c in enumerate((1.0, 1.1, 4.0))
        ]
        report = _build(ladders=ladders, budgets=(300,))
        assert report.runs == 3
        assert report.c_max_curve[300] == pytest.approx(1.1)
        assert report.curve_spread[300] > SPREAD_WARN_THRESHOLD

    def test_knees_come_from_the_last_run(self) -> None:
        ladders = [
            _ladder([_summary(step_index=0, target_concurrency=1.0, ttfab_p95_ms=280.0)]),
            _ladder(
                [_summary(step_index=5, target_concurrency=1.0, ttfab_p95_ms=280.0)], run_index=1
            ),
        ]
        report = _build(ladders=ladders, budgets=(300,))
        assert [k.step_index for k in report.knees] == [5]

    def test_carries_the_truncation_point(self) -> None:
        # So a curve is never read as covering rates that were never offered.
        report = _build(
            ladders=[
                _ladder(
                    [_summary(step_index=0, target_concurrency=1.0, ttfab_p95_ms=280.0)],
                    truncated_at=4,
                )
            ]
        )
        assert report.ladder_truncated_at == 4

    def test_flags_a_curve_that_is_not_per_instance(self) -> None:
        report = _build(frozen=False)
        assert not report.trustworthy
        assert "WITHOUT the autoscaling freeze" in (report.provenance.note or "")

    def test_a_frozen_run_is_trustworthy(self) -> None:
        assert _build(frozen=True).trustworthy

    def test_a_mid_run_capacity_change_defeats_the_freeze(self) -> None:
        ladder = _ladder([_summary(target_concurrency=1.0, ttfab_p95_ms=280.0)])
        ladder.results = [_step(instance_counts=(1, 2))]
        report = _build(ladders=[ladder])
        assert report.instance_counts_observed == (1, 2)
        assert not report.trustworthy

    def test_reports_which_budgets_are_lower_bounds(self) -> None:
        report = _build(
            ladders=[_ladder([_summary(target_concurrency=16.0, ttfab_p95_ms=40.0)])],
            budgets=(150, 300),
        )
        assert report.unbracketed_budgets == [150, 300]

    def test_a_knee_at_the_top_of_the_ladder_is_exhausted_not_inconclusive(self) -> None:
        # Extending the ladder is the right advice here: it genuinely ran out
        # while still passing.
        report = _build(
            ladders=[_ladder([_summary(target_concurrency=16.0, ttfab_p95_ms=40.0)])],
            budgets=(150, 300),
        )
        assert report.exhausted_budgets == [150, 300]
        assert report.inconclusive_budgets == []

    def test_a_knee_below_unusable_higher_steps_is_inconclusive(self) -> None:
        # Higher rates *were* offered; they just measured nothing judgeable. A
        # longer ladder cannot help, so the two notes must not be interchangeable.
        report = _build(
            ladders=[
                _ladder(
                    [
                        _summary(step_index=0, target_concurrency=1.0, ttfab_p95_ms=40.0),
                        _summary(
                            step_index=1,
                            target_concurrency=8.0,
                            ttfab_p95_ms=None,
                            usable=False,
                        ),
                    ]
                )
            ],
            budgets=(150,),
        )
        assert report.inconclusive_budgets == [150]
        assert report.exhausted_budgets == []

    def test_the_top_step_is_scoped_to_the_run_that_produced_the_knees(self) -> None:
        # `steps` holds every run and they can truncate at different points. A
        # global max would call this knee inconclusive on the strength of a step
        # that only run 0 ever reached.
        report = _build(
            ladders=[
                _ladder(
                    [
                        _summary(step_index=0, target_concurrency=1.0, ttfab_p95_ms=40.0),
                        _summary(step_index=1, target_concurrency=8.0, ttfab_p95_ms=40.0),
                    ],
                    run_index=0,
                ),
                _ladder(
                    [
                        _summary(
                            step_index=0, target_concurrency=1.0, ttfab_p95_ms=40.0, run_index=1
                        )
                    ],
                    run_index=1,
                ),
            ],
            budgets=(150,),
        )
        assert report.exhausted_budgets == [150]
        assert report.inconclusive_budgets == []

    def test_provenance_is_measured_and_names_the_run(self) -> None:
        report = _build(run_id="deadbeef", measured_at="2026-07-29T12:00:00+00:00")
        assert report.provenance.origin is Origin.MEASURED
        assert report.provenance.run_id == "deadbeef"
        assert report.provenance.measured_at == "2026-07-29T12:00:00+00:00"

    def test_prefers_joined_steps_when_given(self) -> None:
        joined = [
            _summary(step_index=0, target_concurrency=1.0, ttfab_p95_ms=80.0).model_copy(
                update={"server_concurrency_mean": 1.4}
            )
        ]
        report = _build(joined_steps=joined)
        assert [s.server_concurrency_mean for s in report.steps] == [pytest.approx(1.4)]

    def test_warns_loudly_on_a_wide_spread(self, logged: list[str]) -> None:
        # Above ~20% the ladder resolved noise, and no derate fixes that.
        ladders = [
            _ladder([_summary(target_concurrency=c, ttfab_p95_ms=280.0)], run_index=i)
            for i, c in enumerate((1.0, 2.0, 4.0))
        ]
        report = _build(ladders=ladders, budgets=(300,))
        assert report.curve_spread[300] > SPREAD_WARN_THRESHOLD
        assert any("provisional" in message for message in logged)

    def test_a_tight_spread_is_not_warned_about(self, logged: list[str]) -> None:
        # The warning has to be worth reading, so it must stay quiet when the
        # runs agree.
        ladders = [
            _ladder([_summary(target_concurrency=c, ttfab_p95_ms=280.0)], run_index=i)
            for i, c in enumerate((2.0, 2.05, 2.1))
        ]
        _build(ladders=ladders, budgets=(300,))
        assert not any("provisional" in message for message in logged)

    def test_an_unknown_model_is_rejected_by_the_registry(self) -> None:
        with pytest.raises(ValueError):
            _build(model="not-a-model")


def _rising_ladder(**kwargs) -> LadderRun:
    """A ladder whose offered rate rises with the target, as a real one does.

    ``_build``'s default holds ``offered_rps`` flat at 5.0 across three targets, which is
    fine for the knee (it reads concurrency) and meaningless for the ceiling (it reads
    rate). Rate and target move together here: 0.2s service time, so target C is 5C rps.
    """
    return _ladder(
        [
            _summary(step_index=0, target_concurrency=1.0, offered_rps=5.0, ttfab_p95_ms=80.0),
            _summary(step_index=1, target_concurrency=2.0, offered_rps=10.0, ttfab_p95_ms=280.0),
            _summary(step_index=2, target_concurrency=4.0, offered_rps=20.0, ttfab_p95_ms=900.0),
        ],
        **kwargs,
    )


class TestBuildReportCarriesTheCeiling:
    def test_the_report_carries_a_throughput_ceiling(self) -> None:
        report = _build(ladders=[_rising_ladder()])
        assert report.throughput_ceiling is not None
        # Highest sustained step is target 4.0 at 20 rps; S from the lowest step is 0.2s.
        assert report.throughput_ceiling.max_sustained_rps == pytest.approx(20.0)
        assert report.throughput_ceiling.concurrency == pytest.approx(4.0)

    def test_a_curve_always_comes_with_a_ceiling(self) -> None:
        # Structural, not incidental: hosting a latency knee requires `usable` (completed,
        # no capacity change) plus not-saturated and settled, which is strictly stronger
        # than _sustained. So build_report cannot emit a curve with no ceiling beside it,
        # and a None ceiling on a report that has a curve means the artifact predates the
        # field rather than that the measurement failed.
        report = _build(ladders=[_rising_ladder()])
        assert report.c_max_curve
        assert report.throughput_ceiling is not None

    def test_the_ceiling_uses_the_reports_uncontended_s(self) -> None:
        # Not the probe's, and not the ceiling step's own. Whatever
        # uncontended_service_time settled on is what the conversion has to use, or the
        # artifact's S and its ceiling describe different service times.
        report = _build(
            ladders=[
                _ladder(
                    [
                        _summary(
                            step_index=0, target_concurrency=1.0, offered_rps=2.0, s_mean_s=0.5
                        ),
                        _summary(
                            step_index=1, target_concurrency=2.0, offered_rps=4.0, s_mean_s=1.9
                        ),
                    ]
                )
            ],
            probe=_probe(s_mean_s=0.05),
            budgets=(300,),
        )
        assert report.s_mean_s == pytest.approx(0.5)
        assert report.throughput_ceiling is not None
        assert report.throughput_ceiling.concurrency == pytest.approx(2.0)

    def test_the_ceiling_is_the_median_across_runs(self) -> None:
        ladders = [
            _ladder(
                [_summary(target_concurrency=1.0, offered_rps=r, ttfab_p95_ms=280.0)], run_index=i
            )
            for i, r in enumerate((10.0, 11.0, 40.0))
        ]
        report = _build(ladders=ladders, budgets=(300,))
        assert report.throughput_ceiling is not None
        assert report.throughput_ceiling.max_sustained_rps == pytest.approx(11.0)
        assert report.throughput_ceiling.runs_contributing == 3

    def test_the_ceiling_is_per_ladder_not_pooled(self) -> None:
        # Pooling the steps first would let the luckiest run stand in for the median:
        # three runs peaking at 10, 11 and 40 rps must report 11, not 40.
        ladders = [
            _ladder(
                [
                    _summary(step_index=0, target_concurrency=1.0, offered_rps=5.0, run_index=i),
                    _summary(
                        step_index=1,
                        target_concurrency=2.0,
                        offered_rps=r,
                        ttfab_p95_ms=280.0,
                        run_index=i,
                    ),
                ],
                run_index=i,
            )
            for i, r in enumerate((10.0, 11.0, 40.0))
        ]
        report = _build(ladders=ladders, budgets=(300,))
        assert report.throughput_ceiling is not None
        assert report.throughput_ceiling.max_sustained_rps == pytest.approx(11.0)

    def test_no_sustained_step_means_no_ceiling_not_a_fabricated_one(self) -> None:
        report = _build(
            ladders=[
                _ladder(
                    [
                        _summary(step_index=0, target_concurrency=1.0, ttfab_p95_ms=80.0),
                        _summary(
                            step_index=1,
                            target_concurrency=2.0,
                            ttfab_p95_ms=280.0,
                            saturated=True,
                        ),
                    ]
                )
            ],
            budgets=(150,),
        )
        # Step 0 sustained, so there *is* a ceiling here; the saturated step is excluded.
        assert report.throughput_ceiling is not None
        assert report.throughput_ceiling.step_index == 0

    def test_every_step_saturated_fails_before_a_ceiling_is_reported(self) -> None:
        # The other side of the invariant above: with nothing sustained there is also no
        # curve, so build_report raises rather than returning a report whose ceiling is
        # None. A caller never sees the two disagree.
        with pytest.raises(CMaxError, match="no ladder step met any TTFAB budget"):
            _build(
                ladders=[
                    _ladder([_summary(target_concurrency=1.0, ttfab_p95_ms=80.0, saturated=True)])
                ],
                budgets=(150,),
            )

    def test_warns_when_a_knee_sits_above_the_ceiling(self, logged: list[str]) -> None:
        # The finding a latency-only reading cannot state. The 900ms step sustained its
        # rate and hosts the 1000ms knee at concurrency 4.0; the ceiling is 2.0 because
        # the faster step saturated. Nothing failed, and the knee is still wrong.
        report = _build(
            ladders=[
                _ladder(
                    [
                        _summary(
                            step_index=0,
                            target_concurrency=4.0,
                            offered_rps=10.0,
                            ttfab_p95_ms=900.0,
                            concurrency_mean=4.0,
                        ),
                        _summary(
                            step_index=1,
                            target_concurrency=8.0,
                            offered_rps=20.0,
                            achieved_rps=12.0,
                            ttfab_p95_ms=950.0,
                            saturated=True,
                        ),
                    ]
                )
            ],
            budgets=(1000,),
        )
        assert report.c_max_curve[1000] == pytest.approx(4.0)
        assert report.throughput_ceiling is not None
        assert report.throughput_ceiling.concurrency == pytest.approx(2.0)
        assert report.throughput_bound_budgets == [1000]
        assert any("backlog rather than capacity" in m for m in logged)

    def test_stays_quiet_when_the_knee_is_inside_the_ceiling(self, logged: list[str]) -> None:
        # The ordinary case, and the one that makes the warning worth reading. Both knees
        # (1.0 and 2.0) sit under the 4.0 ceiling.
        report = _build(ladders=[_rising_ladder()])
        assert max(report.c_max_curve.values()) < report.throughput_ceiling.concurrency
        assert report.throughput_bound_budgets == []
        assert not any("backlog rather than capacity" in m for m in logged)

    def test_warns_when_the_ceiling_is_a_lower_bound(self, logged: list[str]) -> None:
        _build(ladders=[_rising_ladder()])  # nothing saturated above: unbracketed
        assert any("LOWER bound" in m for m in logged)

    def test_the_lower_bound_warning_names_the_worker_pool_when_that_is_the_cause(
        self, logged: list[str]
    ) -> None:
        # Two causes, opposite fixes: extend the ladder, or raise --max-workers.
        _build(
            ladders=[
                _ladder(
                    [
                        _summary(
                            step_index=0,
                            target_concurrency=1.0,
                            offered_rps=10.0,
                            ttfab_p95_ms=80.0,
                            skipped=29,
                        ),
                        _summary(
                            step_index=1,
                            target_concurrency=2.0,
                            offered_rps=20.0,
                            achieved_rps=12.0,
                            ttfab_p95_ms=280.0,
                            saturated=True,
                        ),
                    ]
                )
            ],
            budgets=(150,),
        )
        assert any("--max-workers" in m for m in logged)

    def test_warns_when_the_instance_is_mostly_queueing_at_the_ceiling(
        self, logged: list[str]
    ) -> None:
        _build(
            ladders=[
                _ladder(
                    [
                        _summary(
                            step_index=0,
                            target_concurrency=1.0,
                            offered_rps=10.0,
                            ttfab_p95_ms=80.0,
                            concurrency_mean=4.5,
                        )
                    ]
                )
            ],
            budgets=(150,),
        )
        # 10 rps x 0.2s = 2.0 useful against 4.5 observed: 2.25x.
        assert any("residence is queueing" in m for m in logged)

    def test_the_no_curve_error_distinguishes_a_latency_limit_from_a_capacity_one(self) -> None:
        # "Lower --target-concurrency" is wrong advice for an endpoint that sustained every
        # rate it was offered and merely did so slowly. That one wants a looser budget.
        with pytest.raises(CMaxError, match="The limit here is latency, not throughput"):
            _build(
                ladders=[_ladder([_summary(offered_rps=10.0, ttfab_p95_ms=5000.0)])],
                budgets=(150,),
            )

    def test_the_no_curve_error_keeps_the_old_advice_when_nothing_sustained(self) -> None:
        with pytest.raises(CMaxError, match="--target-concurrency"):
            _build(
                ladders=[
                    _ladder([_summary(offered_rps=10.0, ttfab_p95_ms=5000.0, saturated=True)])
                ],
                budgets=(150,),
            )


class TestDryRunPlan:
    def test_one_row_per_step_per_run(self) -> None:
        plan = dry_run_plan(s_mean_s=0.25, target_concurrencies=(1.0, 2.0), runs=3)
        assert len(plan) == 6
        assert [int(r["run_index"]) for r in plan] == [0, 0, 1, 1, 2, 2]

    def test_shows_the_request_count_a_step_will_send(self) -> None:
        # Sizing a run is the question this answers: 10 steps x 240s x 3 runs is
        # over two hours, and that is worth seeing before committing to it.
        plan = dry_run_plan(s_mean_s=0.25, target_concurrencies=(1.0,), hold_s=240.0)
        assert plan[0]["offered_rps"] == pytest.approx(4.0)
        assert plan[0]["expected_requests"] == pytest.approx(960.0)

    def test_orders_the_ladder_upward(self) -> None:
        plan = dry_run_plan(s_mean_s=0.25, target_concurrencies=(4.0, 1.0))
        assert [r["target_concurrency"] for r in plan] == [1.0, 4.0]


class TestTotalDuration:
    def test_counts_holds_and_drains(self) -> None:
        assert total_duration_s(
            target_concurrencies=(1.0, 2.0, 4.0),
            hold_s=240.0,
            settle_between_steps_s=30.0,
        ) == pytest.approx(3 * 240.0 + 2 * 30.0)

    def test_scales_with_runs(self) -> None:
        one = total_duration_s(target_concurrencies=(1.0, 2.0), hold_s=60.0, runs=1)
        assert total_duration_s(target_concurrencies=(1.0, 2.0), hold_s=60.0, runs=3) == (
            pytest.approx(3 * one)
        )

    def test_a_single_step_needs_no_drain(self) -> None:
        assert total_duration_s(
            target_concurrencies=(1.0,), hold_s=60.0, settle_between_steps_s=30.0
        ) == pytest.approx(60.0)


# --------------------------------------------------------------------------- #
# End-to-end orchestration
# --------------------------------------------------------------------------- #


class _NoInvocations:
    """A runtime client that fails the test if it is ever used."""

    def __init__(self) -> None:
        self.calls = 0

    def invoke_endpoint_with_response_stream(self, **kwargs):
        self.calls += 1
        raise AssertionError("load was sent despite the guard")


def _patch_load(monkeypatch, *, ladder: LadderRun | None = None) -> list[dict]:
    """Replace probe and ladder so ``measure``'s orchestration can be tested alone."""
    recorded: list[dict] = []
    monkeypatch.setattr(cmax, "probe_service_time", lambda *a, **k: _probe())

    def fake_run_ladder(client, **kwargs) -> LadderRun:
        recorded.append(kwargs)
        return ladder or _ladder(
            [
                _summary(step_index=0, target_concurrency=1.0, ttfab_p95_ms=80.0),
                _summary(step_index=1, target_concurrency=2.0, ttfab_p95_ms=280.0),
            ],
            run_index=kwargs["run_index"],
        )

    monkeypatch.setattr(cmax, "run_ladder", fake_run_ladder)
    return recorded


def _patch_ladder_only(monkeypatch) -> list[dict]:
    """Replace only the ladder, leaving the caller's own probe patch in place."""
    recorded: list[dict] = []

    def fake_run_ladder(client, **kwargs) -> LadderRun:
        recorded.append(kwargs)
        return _ladder(
            [
                _summary(step_index=0, target_concurrency=1.0, ttfab_p95_ms=80.0),
                _summary(step_index=1, target_concurrency=2.0, ttfab_p95_ms=280.0),
            ],
            run_index=kwargs["run_index"],
        )

    monkeypatch.setattr(cmax, "run_ladder", fake_run_ladder)
    return recorded


def _measure(**kwargs) -> CMaxReport:
    defaults = {
        "model": MODEL,
        "texts": TEXTS,
        "budgets": (150, 300),
        "target_concurrencies": (1.0, 2.0),
        "cloudwatch_join": False,
        "runtime_client": _NoInvocations(),
        "clock": SYSTEM_CLOCK,
    }
    defaults.update(kwargs)
    return measure(**defaults)


class TestMeasureGuard:
    def test_refuses_to_send_load_when_the_fleet_can_still_grow(self) -> None:
        # The plan's mandated guard. A warning would be ignored and the resulting
        # C_max would look entirely normal.
        client = _NoInvocations()
        aas = FakeAppScaling(scale_out_suspended=False)

        with pytest.raises(FixtureError):
            _measure(runtime_client=client, appscaling=aas, sagemaker=FakeSageMaker())

        assert client.calls == 0

    def test_the_guard_runs_before_the_probe(self, monkeypatch) -> None:
        # The probe is inside the freeze too: no part of the measurement may run
        # against a fleet that is free to change.
        probes: list[int] = []
        monkeypatch.setattr(
            cmax, "probe_service_time", lambda *a, **k: probes.append(1) or _probe()
        )

        with pytest.raises(FixtureError):
            _measure(
                appscaling=FakeAppScaling(scale_out_suspended=False), sagemaker=FakeSageMaker()
            )
        assert probes == []


class TestMeasureOrchestration:
    def test_freezes_thaws_and_marks_the_artifact_frozen(self, monkeypatch) -> None:
        _patch_load(monkeypatch)
        aas = FakeAppScaling(scale_out_suspended=True)

        report = _measure(appscaling=aas, sagemaker=FakeSageMaker())

        assert report.frozen
        assert report.trustworthy
        # Suspend on the way in, restore on the way out.
        assert len(aas.registered) == 2
        assert aas.registered[0]["SuspendedState"]["DynamicScalingOutSuspended"] is True
        assert aas.registered[1]["SuspendedState"]["DynamicScalingOutSuspended"] is True

    def test_thaws_when_the_ladder_raises(self, monkeypatch) -> None:
        # An aborted benchmark must not leave production frozen.
        monkeypatch.setattr(cmax, "probe_service_time", lambda *a, **k: _probe())

        def boom(client, **kwargs):
            raise RuntimeError("step died")

        monkeypatch.setattr(cmax, "run_ladder", boom)
        aas = FakeAppScaling(scale_out_suspended=True)

        with pytest.raises(RuntimeError, match="step died"):
            _measure(appscaling=aas, sagemaker=FakeSageMaker())
        assert len(aas.registered) == 2

    def test_without_the_freeze_it_still_arms_the_tripwire(self, monkeypatch) -> None:
        # It matters *most* on this path: this is the run whose fleet is actually
        # free to change.
        recorded = _patch_load(monkeypatch)
        aas = FakeAppScaling(scale_out_suspended=False)

        report = _measure(require_frozen=False, appscaling=aas, sagemaker=FakeSageMaker())

        assert not report.frozen
        assert not report.trustworthy
        assert recorded[0]["instance_count_fetch"] is not None
        # Nothing was suspended, and nothing needed restoring.
        assert aas.registered == []

    def test_runs_the_ladder_once_per_run(self, monkeypatch) -> None:
        recorded = _patch_load(monkeypatch)
        report = _measure(
            runs=3, appscaling=FakeAppScaling(), sagemaker=FakeSageMaker(), derate=0.9
        )
        assert [k["run_index"] for k in recorded] == [0, 1, 2]
        assert report.runs == 3
        assert report.derate == pytest.approx(0.9)

    def test_gives_every_run_a_distinct_run_id_under_one_report_id(self, monkeypatch) -> None:
        recorded = _patch_load(monkeypatch)
        report = _measure(runs=2, appscaling=FakeAppScaling(), sagemaker=FakeSageMaker())
        ids = [k["run_id"] for k in recorded]
        assert ids == [f"{report.run_id}-r0", f"{report.run_id}-r1"]

    def test_resolves_the_endpoint_voice_and_instance_type(self, monkeypatch) -> None:
        recorded = _patch_load(monkeypatch)
        report = _measure(appscaling=FakeAppScaling(), sagemaker=FakeSageMaker())
        assert report.endpoint == ENDPOINT
        assert report.instance_type == "ml.g5.xlarge"
        assert recorded[0]["endpoint"] == ENDPOINT
        assert recorded[0]["voice"]

    def test_joins_cloudwatch_when_asked(self, monkeypatch) -> None:
        ladder = _ladder([_summary(step_index=0, target_concurrency=1.0, ttfab_p95_ms=80.0)])
        ladder.results = [_step(step_index=0, ended_ts=SETTLED_TS)]
        _patch_load(monkeypatch, ladder=ladder)
        cw = FakeCloudWatch()

        report = _measure(
            cloudwatch_join=True,
            cloudwatch=cw,
            appscaling=FakeAppScaling(),
            sagemaker=FakeSageMaker(),
        )
        assert cw.calls
        assert report.steps[0].server_concurrency_mean == pytest.approx(2.5)

    def test_records_the_arrival_process_and_seed(self, monkeypatch) -> None:
        _patch_load(monkeypatch)
        report = _measure(
            arrival="fixed", seed=99, appscaling=FakeAppScaling(), sagemaker=FakeSageMaker()
        )
        assert report.arrival_process == "fixed"
        assert report.seed == 99

    def test_a_model_with_no_endpoint_is_refused(self) -> None:
        # Polly is managed: there is no instance to size.
        with pytest.raises(ValueError, match="no SageMaker endpoint"):
            _measure(model="polly-neural")


class TestMeasureRecordsTheConfiguration:
    """What the artifact says it was measured against, and where that comes from.

    The endpoint, not ``cost.MODEL_INSTANCE_TYPES``. A registry states what *should* be
    deployed; a benchmark has to record what *is*. Getting this backwards is silent in
    exactly the workflow the harness exists for — redeploy on new hardware, re-measure,
    and the static dict stamps the fresh curve with the old type.
    """

    def test_the_fingerprint_comes_off_the_endpoint(self, monkeypatch) -> None:
        _patch_load(monkeypatch)
        sm = FakeSageMaker(image="repo/asset:deadbeefcafe", env={"MAX_REQUEST_AGE_S": "56"})

        report = _measure(appscaling=FakeAppScaling(), sagemaker=sm)

        assert report.deployed_config["instance_type"] == "ml.g5.xlarge"
        assert report.deployed_config["image_digest"] == "deadbeefcafe"
        assert report.deployed_config["container_env"] == {"MAX_REQUEST_AGE_S": "56"}

    def test_the_live_type_wins_over_the_registry(self, monkeypatch, logged: list[str]) -> None:
        # The stale-registry bug, made loud. ERROR rather than WARNING because the same
        # divergence also makes `drift` and the cost model wrong, not just this artifact.
        # A retype that landed on the endpoint but not in the registry is the realistic
        # shape of this: the measurement has to follow the hardware, not the dict.
        _patch_load(monkeypatch)
        sm = FakeSageMaker(instance_type="ml.g6.xlarge")

        report = _measure(appscaling=FakeAppScaling(), sagemaker=sm)

        assert report.instance_type == "ml.g6.xlarge"
        assert report.deployed_config["instance_type"] == "ml.g6.xlarge"
        assert any("MODEL_INSTANCE_TYPES says ml.g5.xlarge" in m for m in logged)

    def test_an_agreeing_registry_is_not_complained_about(
        self, monkeypatch, logged: list[str]
    ) -> None:
        _patch_load(monkeypatch)
        _measure(appscaling=FakeAppScaling(), sagemaker=FakeSageMaker())
        assert not any("MODEL_INSTANCE_TYPES" in m for m in logged)

    def test_an_unreadable_endpoint_falls_back_to_the_registry(
        self, monkeypatch, logged: list[str]
    ) -> None:
        # A failed describe must not cost a whole ladder run. The fallback is still
        # honest: no image digest, so it can never compare equal to a real fingerprint.
        _patch_load(monkeypatch)

        def boom(*_args, **_kwargs):
            raise FixtureError("endpoint is gone")

        monkeypatch.setattr(fixture, "describe_deployed_config", boom)

        report = _measure(appscaling=FakeAppScaling(), sagemaker=FakeSageMaker())

        assert report.instance_type == "ml.g5.xlarge"
        assert report.deployed_config["image_digest"] is None
        assert any("Falling back to the registry type" in m for m in logged)

    def test_the_slug_names_the_configuration(self, monkeypatch) -> None:
        # What defaults one artifact filename per configuration, so two of them cannot
        # overwrite each other by forgetting --output.
        _patch_load(monkeypatch)
        sm = FakeSageMaker(instance_type="ml.g6.12xlarge", image="repo/asset:139b9068c5eb1f03")

        report = _measure(appscaling=FakeAppScaling(), sagemaker=sm)

        assert report.config_slug == "g612xlarge-139b9068"

    def test_the_fingerprint_survives_the_join_into_measured(self, monkeypatch) -> None:
        # `plan` reads a Measured, so losing it here would leave the read-side check
        # with nothing to compare.
        _patch_load(monkeypatch)
        report = _measure(appscaling=FakeAppScaling(), sagemaker=FakeSageMaker())

        joined = report.to_measured(t_total_s=180.0)

        assert joined.deployed_config == report.deployed_config

    def test_the_fingerprint_round_trips_through_json(self, monkeypatch) -> None:
        _patch_load(monkeypatch)
        report = _measure(appscaling=FakeAppScaling(), sagemaker=FakeSageMaker())

        restored = CMaxReport.model_validate_json(report.model_dump_json())

        assert restored.deployed_config == report.deployed_config
        assert restored.config_slug == report.config_slug

    def test_an_artifact_without_one_still_has_a_slug(self) -> None:
        # Every committed artifact predates fingerprinting. `config_slug` is used to name
        # files, so it must not raise on them -- the read-side check is what refuses them.
        report = _build()
        assert report.deployed_config == {}
        assert report.config_slug == "g5xlarge-nodigest"


class TestJoinWithTTotal:
    """``CMaxReport.to_measured`` is the single place Phase 2 meets Phase 3."""

    def test_carries_the_freeze_through_the_join(self) -> None:
        # A curve measured without the freeze must stay identifiable as such
        # after it becomes a planner input.
        report = _build(frozen=False)
        joined = report.to_measured(t_total_s=180.0)
        assert not joined.frozen
        assert not joined.trustworthy

    def test_carries_the_curve_and_service_time(self) -> None:
        report = _build()
        joined = report.to_measured(t_total_s=180.0)
        assert joined.c_max_curve == report.c_max_curve
        assert joined.s_mean_s == pytest.approx(report.s_mean_s)
        assert joined.t_total_s == pytest.approx(180.0)

    def test_round_trips_through_json_with_integer_budgets(self) -> None:
        # cmax writes an artifact; plan reads it back. JSON stringifies dict
        # keys, so a lossy round trip would make every c_max_for lookup miss.
        report = _build()
        restored = CMaxReport.model_validate_json(report.model_dump_json())
        assert restored.c_max_curve == report.c_max_curve
        assert all(isinstance(k, int) for k in restored.c_max_curve)
        assert restored.model_name is TTSModelName.KOKORO_82M

    def test_carries_the_throughput_ceiling_into_the_planner_input(self) -> None:
        report = _build(ladders=[_rising_ladder()])
        joined = report.to_measured(t_total_s=180.0)
        assert joined.c_max_throughput == pytest.approx(4.0)

    def test_an_unbracketed_ceiling_arrives_flagged(self) -> None:
        # The planner takes the *minimum* of the two C_max kinds, so a lower-bound minimum
        # understates the fleet and the report has to say so rather than print a bare
        # number. `is_lower_bound`, not `bracketed`: a client-throttled ceiling is just as
        # untrustworthy as an unbracketed one.
        joined = _build(ladders=[_rising_ladder()]).to_measured(t_total_s=180.0)
        assert not joined.c_max_throughput_bracketed

    def test_a_bracketed_ceiling_arrives_trusted(self) -> None:
        report = _build(
            ladders=[
                _ladder(
                    [
                        _summary(
                            step_index=0,
                            target_concurrency=1.0,
                            offered_rps=10.0,
                            ttfab_p95_ms=80.0,
                        ),
                        _summary(
                            step_index=1,
                            target_concurrency=2.0,
                            offered_rps=20.0,
                            achieved_rps=12.0,
                            ttfab_p95_ms=280.0,
                            saturated=True,
                        ),
                    ]
                )
            ],
            budgets=(150,),
        )
        joined = report.to_measured(t_total_s=180.0)
        assert joined.c_max_throughput_bracketed

    def test_the_binding_c_max_takes_the_lower_of_the_two(self) -> None:
        # Both are real per-instance limits and an instance is bound by whichever it
        # reaches first, so planning on the higher one sizes a fleet for capacity that does
        # not exist. Here the knee at the 900ms budget (4.0) is above the 2.0 ceiling.
        report = _build(
            ladders=[
                _ladder(
                    [
                        _summary(
                            step_index=0,
                            target_concurrency=4.0,
                            offered_rps=10.0,
                            ttfab_p95_ms=900.0,
                            concurrency_mean=4.0,
                        ),
                        _summary(
                            step_index=1,
                            target_concurrency=8.0,
                            offered_rps=20.0,
                            achieved_rps=12.0,
                            ttfab_p95_ms=950.0,
                            saturated=True,
                        ),
                    ]
                )
            ],
            budgets=(1000,),
        )
        joined = report.to_measured(t_total_s=180.0)
        assert joined.binding_c_max(1000) == (pytest.approx(2.0), "throughput_ceiling")

    def test_the_binding_c_max_keeps_the_knee_when_it_is_lower(self) -> None:
        joined = _build(ladders=[_rising_ladder()]).to_measured(t_total_s=180.0)
        c_max, source = joined.binding_c_max(300)
        assert (c_max, source) == (pytest.approx(2.0), "latency_knee")

    def test_an_artifact_with_no_ceiling_says_the_comparison_never_happened(self) -> None:
        # Pre-existing artifacts have no ceiling to report, and "not measured" is a
        # different claim from "no ceiling found" — for a throughput-bound model that is
        # the difference between a checked answer and an unchecked one.
        joined = _build(ladders=[_rising_ladder()]).to_measured(t_total_s=180.0)
        stripped = joined.model_copy(update={"c_max_throughput": None})
        assert stripped.binding_c_max(300) == (pytest.approx(2.0), "latency_knee_only")

    def test_the_ceiling_round_trips_through_json(self) -> None:
        report = _build(ladders=[_rising_ladder()])
        restored = CMaxReport.model_validate_json(report.model_dump_json())
        assert restored.throughput_ceiling == report.throughput_ceiling

    def test_an_artifact_written_before_the_field_existed_still_loads(self) -> None:
        # The pre-fingerprint precedent: absent must read as "not measured" rather than
        # fail the load, or every artifact on disk becomes unreadable.
        payload = _build(ladders=[_rising_ladder()]).model_dump(mode="json")
        del payload["throughput_ceiling"]
        restored = CMaxReport.model_validate(payload)
        assert restored.throughput_ceiling is None
        assert restored.throughput_bound_budgets == []


# --------------------------------------------------------------------------- #
# Transport
# --------------------------------------------------------------------------- #


class TestTransportIsRecorded:
    """A ``C_max`` without its transport is not interpretable.

    The containers hold their inference lock differently per protocol — kokoro
    holds it across an entire bidi session but per-generator on response-stream —
    so the number does not transfer. The transport therefore has to survive all
    the way into the artifact and then into the planner input.
    """

    def test_defaults_to_response_stream(self) -> None:
        # Every C_max measured before the bidi transport existed came from this
        # path, so the default has to name it rather than be blank.
        assert _build().transport == "response-stream"

    def test_records_the_transport_it_was_given(self) -> None:
        assert _build(transport=Transport.BIDI).transport == "bidi"

    def test_accepts_the_raw_cli_string(self) -> None:
        # The CLI hands its --transport value straight through.
        assert _build(transport="bidi").transport == "bidi"

    def test_the_provenance_note_names_the_transport(self) -> None:
        # The note is what a reader sees first; a C_max attributed to the wrong
        # protocol would size a fleet for traffic it cannot serve.
        assert "bidi" in (_build(transport="bidi").provenance.note or "")

    def test_the_transport_survives_a_run_without_the_freeze(self) -> None:
        # Both caveats have to coexist: the note carries the freeze warning and
        # still has to say which protocol produced the number.
        note = _build(transport="bidi", frozen=False).provenance.note or ""
        assert "bidi" in note
        assert "WITHOUT the autoscaling freeze" in note

    def test_the_transport_survives_the_join_into_measured(self) -> None:
        # to_measured is where a C_max becomes a planner input; losing the
        # transport there means the plan configures a fleet from a capacity
        # number measured on a protocol production does not use.
        joined = _build(transport="bidi").to_measured(t_total_s=180.0)
        assert joined.transport == "bidi"

    def test_the_transport_round_trips_through_json(self) -> None:
        report = _build(transport="bidi")
        restored = CMaxReport.model_validate_json(report.model_dump_json())
        assert restored.transport == "bidi"


class TestMeasureUsesOneTransport:
    def test_the_probe_and_the_ladder_share_the_transport(self, monkeypatch) -> None:
        # S is what converts every concurrency target into a rate, so probing on
        # one protocol and laddering on another misprices every step.
        probes: list[dict] = []
        monkeypatch.setattr(
            cmax,
            "probe_service_time",
            lambda *a, **k: probes.append(k) or _probe(),
        )
        recorded = _patch_ladder_only(monkeypatch)

        _measure(
            transport="bidi",
            runtime_client=object(),
            appscaling=FakeAppScaling(),
            sagemaker=FakeSageMaker(),
        )

        assert probes[0]["invoke"] is invoke_bidi
        assert all(k["invoke"] is invoke_bidi for k in recorded)

    def test_response_stream_stays_the_default(self, monkeypatch) -> None:
        recorded = _patch_load(monkeypatch)
        report = _measure(appscaling=FakeAppScaling(), sagemaker=FakeSageMaker())
        assert report.transport == "response-stream"
        assert all(k["invoke"] is invoke_stream for k in recorded)

    def test_the_transport_reaches_the_artifact(self, monkeypatch) -> None:
        _patch_load(monkeypatch)
        report = _measure(
            transport="bidi",
            runtime_client=object(),
            appscaling=FakeAppScaling(),
            sagemaker=FakeSageMaker(),
        )
        assert report.transport == "bidi"

    def test_an_unknown_transport_is_refused_before_any_load(self) -> None:
        client = _NoInvocations()
        with pytest.raises(ValueError):
            _measure(transport="grpc", runtime_client=client)
        assert client.calls == 0

    def test_builds_the_matching_client_when_none_is_supplied(self, monkeypatch) -> None:
        # measure() owns client construction, and the client and the invoke
        # function have to be a matched pair: a boto3 client handed to
        # invoke_bidi fails on the first request of every step.
        built: list[tuple] = []
        monkeypatch.setattr(
            cmax,
            "make_client_for",
            lambda transport, region, **kw: built.append((transport, region, kw)) or object(),
        )
        _patch_load(monkeypatch)

        _measure(
            transport="bidi",
            runtime_client=None,
            appscaling=FakeAppScaling(),
            sagemaker=FakeSageMaker(),
        )
        assert built[0][0] is Transport.BIDI

    def test_the_connection_pool_matches_the_worker_pool(self, monkeypatch) -> None:
        # Two pools bound the same client, and the smaller one wins. Raising only the
        # threads would move the bottleneck from the executor to the HTTP pool and the
        # artifact would still record a client-limited ceiling under --max-workers 64.
        built: list[dict] = []
        monkeypatch.setattr(
            cmax,
            "make_client_for",
            lambda transport, region, **kw: built.append(kw) or object(),
        )
        recorded = _patch_load(monkeypatch)

        _measure(
            runtime_client=None,
            max_workers=64,
            appscaling=FakeAppScaling(),
            sagemaker=FakeSageMaker(),
        )
        assert built[0]["max_pool"] == 64
        assert {k["max_workers"] for k in recorded} == {64}

    def test_without_an_override_the_pools_are_the_derived_size(self, monkeypatch) -> None:
        built: list[dict] = []
        monkeypatch.setattr(
            cmax,
            "make_client_for",
            lambda transport, region, **kw: built.append(kw) or object(),
        )
        recorded = _patch_load(monkeypatch)

        _measure(
            runtime_client=None,
            target_concurrencies=(1.0, 2.0),
            appscaling=FakeAppScaling(),
            sagemaker=FakeSageMaker(),
        )
        assert built[0]["max_pool"] == worker_count(2.0)
        # run_ladder derives its own from the same targets, so it takes the override
        # itself rather than the resolved number.
        assert {k["max_workers"] for k in recorded} == {None}
