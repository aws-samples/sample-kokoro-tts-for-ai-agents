"""Tests for the closed-loop load generator.

The load generator is the one component whose correctness cannot be checked by
reading its output: a driver that holds the wrong concurrency still produces
plausible-looking latency percentiles. ``TestClosedLoopProperty`` is therefore
the load-bearing test in this file — it fails if outstanding requests ever
depart from the ``N`` the step claims, whatever the code looks like.

``Q_max`` is *defined* as a concurrency, so ``N`` is the independent variable and
holding it exactly is the whole contract. The previous open-loop driver reached a
concurrency only by choosing an arrival rate and hoping, via ``lambda = C / S``;
these tests pin the property that replaced that conversion.

Real threads with sub-second service times, matching the existing pattern in
``test_scalability.py``. ``VirtualClock`` is used where a step's *duration* would
otherwise dominate, since a closed loop never sleeps — time only passes while a
request is being served, so the stub transport is what advances it.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import replace

import pytest

from tts_bench.invoke import InvokeOutcome, InvokeResult
from tts_bench.loadgen import (
    CONCURRENCY_SHORTFALL_RATIO,
    LATENCY_DRIFT_RATIO,
    REJECTION_OUTCOMES,
    REJECTION_RATIO,
    SYSTEM_CLOCK,
    Clock,
    ConcurrencySample,
    JsonlWriter,
    LoadEvent,
    StepResult,
    build_text_pool,
    make_instance_count_fetcher,
    run_step,
    summarize_window,
)

TEXTS = ["Let me check that for you.", "Your appointment is confirmed.", "One moment please."]


class FakeServer:
    """A server with a fixed service time and a concurrency limit of ``slots``.

    Stands in for Kokoro's whole-stream lock: with ``slots=1`` only one request is
    ever *executing*, so any excess the client holds outstanding is queued — which
    is exactly the ``Q + E = N`` split the ladder measures.

    ``max_observed_concurrency`` counts requests past the semaphore, so it reports
    what the server executed. The client's own view is in ``StepResult.samples``,
    and the gap between the two is the queue.
    """

    def __init__(
        self,
        *,
        service_time_s: float = 0.05,
        slots: int = 1,
        fail_with: InvokeOutcome | None = None,
        fail_after: int | None = None,
    ) -> None:
        self._sem = threading.Semaphore(slots)
        self._service_time_s = service_time_s
        self._fail_with = fail_with
        self._fail_after = fail_after
        self._lock = threading.Lock()
        self.started = 0
        self.completed = 0
        self.max_observed_concurrency = 0
        self._active = 0

    def __call__(self, client, endpoint, text, voice, *, deadline_ts=None) -> InvokeResult:
        dispatch_ts = time.time()
        with self._lock:
            self.started += 1
            n = self.started
        with self._sem:
            with self._lock:
                self._active += 1
                self.max_observed_concurrency = max(self.max_observed_concurrency, self._active)
            time.sleep(self._service_time_s)
            with self._lock:
                self._active -= 1
                self.completed += 1

        end_ts = time.time()
        if self._fail_with is not None and (self._fail_after is None or n > self._fail_after):
            return InvokeResult(
                outcome=self._fail_with,
                dispatch_ts=dispatch_ts,
                end_ts=end_ts,
                latency_ms=(end_ts - dispatch_ts) * 1000.0,
                chars=len(text),
                http_status=503,
                error_class="ClientError",
            )
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


class VirtualClock:
    """A clock that only moves when a request is served.

    Lets a 240s step run in milliseconds. Advanced by the ``invoke`` stub rather
    than by ``sleep``, because a closed-loop driver never sleeps: it is blocked in
    the transport for the whole step, so service time *is* elapsed time. Locked
    because ``N`` workers advance it concurrently.
    """

    def __init__(self, *, epoch: float = 1_700_000_000.0) -> None:
        self._lock = threading.Lock()
        self._t = 0.0
        self._epoch = epoch

    def advance(self, seconds: float) -> None:
        with self._lock:
            self._t += seconds

    @property
    def elapsed(self) -> float:
        with self._lock:
            return self._t

    def clock(self) -> Clock:
        return Clock(
            monotonic=lambda: self.elapsed,
            sleep=self.advance,
            time=lambda: self._epoch + self.elapsed,
        )


def virtual_transport(vclock: VirtualClock, service_s: float):
    """An ``invoke`` stub that consumes ``service_s`` of *virtual* time per request."""
    clock = vclock.clock()

    def invoke(client, endpoint, text, voice, *, deadline_ts=None) -> InvokeResult:
        dispatch_ts = clock.time()
        vclock.advance(service_s)
        end_ts = clock.time()
        return InvokeResult(
            outcome=InvokeOutcome.OK,
            dispatch_ts=dispatch_ts,
            end_ts=end_ts,
            latency_ms=service_s * 1000.0,
            first_byte_ts=dispatch_ts,
            ttfab_ms=service_s * 1000.0,
            chars=len(text),
            audio_duration_s=0.1,
        )

    return invoke


def _event(
    *,
    seq: int = 0,
    worker_index: int = 0,
    concurrency: int = 5,
    outcome: InvokeOutcome = InvokeOutcome.OK,
    dispatch_ts: float | None = 0.0,
    end_ts: float | None = 1.0,
    latency_ms: float | None = 100.0,
    ttfab_ms: float | None = 50.0,
    chars: int = 20,
    rtf: float | None = 0.5,
) -> LoadEvent:
    return LoadEvent(
        run_id="run",
        step_index=0,
        seq=seq,
        worker_index=worker_index,
        concurrency=concurrency,
        model="kokoro-82m",
        endpoint="speech-kokoro-82m",
        dispatch_ts=dispatch_ts,
        first_byte_ts=None,
        end_ts=end_ts,
        ttfab_ms=ttfab_ms,
        latency_ms=latency_ms,
        outcome=outcome.value,
        http_status=200,
        error_class=None,
        error_message=None,
        chars=chars,
        audio_bytes=100,
        audio_duration_s=0.2,
        rtf=rtf,
        in_flight_at_dispatch=1,
        instance_count=1,
    )


def _step(
    events: list[LoadEvent],
    samples: list[ConcurrencySample] | None = None,
    *,
    concurrency: int = 5,
) -> StepResult:
    return StepResult(
        run_id="run",
        step_index=0,
        concurrency=concurrency,
        model="kokoro-82m",
        endpoint="speech-kokoro-82m",
        started_ts=0.0,
        ended_ts=10.0,
        events=events,
        samples=samples or [],
    )


class TestClosedLoopProperty:
    """The regression test that protects the point of this module.

    ``N`` outstanding requests, exactly, for the whole step — regardless of how the
    server behaves. Thresholds are expressed against the fake server's known
    service time rather than as bare numbers, so the assertions state the property
    instead of encoding a measurement.
    """

    SERVICE_S = 0.05
    DURATION_S = 1.0

    def _held_at(self, concurrency: int, *, slots: int = 32) -> tuple[FakeServer, StepResult]:
        server = FakeServer(service_time_s=self.SERVICE_S, slots=slots)
        result = run_step(
            client=None,
            model="kokoro-82m",
            endpoint="speech-kokoro-82m",
            voice="af_heart",
            texts=TEXTS,
            concurrency=concurrency,
            duration_s=self.DURATION_S,
            monitor_interval_s=0.02,
            invoke=server,
        )
        return server, result

    @staticmethod
    def _dispatch_window(result: StepResult, duration_s: float):
        """Summarize only while the loop was dispatching.

        A step ends with ``N`` requests in flight and lets them finish, so its
        wall-clock runs past ``duration_s`` and in-flight falls to zero over that
        tail. Averaging the tail in understates the concurrency that was actually
        held — which is why the caller (``qmax.py``) windows rather than reading the
        whole step, and why these assertions do the same.
        """
        return summarize_window(
            result, start_ts=result.started_ts, end_ts=result.started_ts + duration_s
        )

    def test_concurrency_five_is_held_at_exactly_five(self) -> None:
        server, result = self._held_at(5)

        # The structural guarantee: N workers, one request each, so the server can
        # never see more than N. This is the direction that would invalidate a
        # Q_max measurement, so it is asserted as an equality on both sides.
        assert server.max_observed_concurrency == 5

        stats = self._dispatch_window(result, self.DURATION_S)
        assert stats.concurrency == 5
        assert stats.concurrency_peak == 5
        assert stats.concurrency_mean == pytest.approx(5.0, rel=0.05)
        assert not stats.client_bound

    def test_achieved_throughput_is_n_over_service_time(self) -> None:
        # At fixed N, throughput is an *output*: N/latency. It is how the ladder
        # derives a rate, never a target the server is compared against.
        _, result = self._held_at(5)
        stats = self._dispatch_window(result, self.DURATION_S)
        expected = 5 / self.SERVICE_S
        assert stats.achieved_rps == pytest.approx(expected, rel=0.35)

    def test_concurrency_one_never_exceeds_one(self) -> None:
        # N=1 is both the probe and the ladder's c1 rung, which yields the alarm's
        # service-time reference. A second in-flight request there would inflate it.
        server, result = self._held_at(1)
        assert server.max_observed_concurrency == 1
        assert max(s.in_flight for s in result.samples) <= 1
        assert result.worker_overlaps == 0

    def test_no_worker_ever_overlaps_its_own_request(self) -> None:
        # Checkable from the artifact alone, which is the point: `Q + E = N` stops
        # being a claim about the code and becomes a property of the data.
        _, result = self._held_at(8)
        assert result.worker_overlaps == 0
        assert len({e.worker_index for e in result.events}) == 8
        assert all(e.concurrency == 8 for e in result.events)

    def test_a_slow_server_lowers_throughput_not_concurrency(self) -> None:
        # The defining difference from the open-loop driver, which held the *rate*
        # and let concurrency grow. Here concurrency is pinned and the rate moves,
        # so a serialized server shows up as queueing rather than as backlog.
        def step(service_time_s: float, slots: int) -> StepResult:
            return run_step(
                client=None,
                model="m",
                endpoint="e",
                voice="v",
                texts=TEXTS,
                concurrency=4,
                duration_s=0.5,
                monitor_interval_s=0.02,
                invoke=FakeServer(service_time_s=service_time_s, slots=slots),
            )

        fast = step(0.005, 32)
        slow = step(0.1, 1)

        for result in (fast, slow):
            assert max(s.in_flight for s in result.samples) == 4
            assert result.worker_overlaps == 0

        fast_stats = self._dispatch_window(fast, 0.5)
        slow_stats = self._dispatch_window(slow, 0.5)
        assert fast_stats.achieved_rps > 5 * slow_stats.achieved_rps

    def test_queueing_shows_up_as_latency_not_as_extra_concurrency(self) -> None:
        # slots=1 against N=4: one request executes and three wait, so the client
        # holds 4 while the server executes 1. That gap *is* the queue, and it is
        # what makes the SLO measurable end-to-end.
        server = FakeServer(service_time_s=0.05, slots=1)
        result = run_step(
            client=None,
            model="m",
            endpoint="e",
            voice="v",
            texts=TEXTS,
            concurrency=4,
            duration_s=0.6,
            monitor_interval_s=0.02,
            invoke=server,
        )
        assert server.max_observed_concurrency == 1
        stats = self._dispatch_window(result, 0.6)
        assert stats.concurrency_mean == pytest.approx(4.0, rel=0.1)
        # Serialized: four outstanding against one server cannot beat 1/service.
        assert stats.achieved_rps < 1.5 / 0.05


class TestRunStep:
    def test_every_dispatch_produces_exactly_one_event(self) -> None:
        # Reconciliation: a run whose totals do not account for every request it
        # held a worker for cannot support a capacity conclusion.
        result = run_step(
            client=None,
            model="m",
            endpoint="e",
            voice="v",
            texts=TEXTS,
            concurrency=4,
            duration_s=0.4,
            monitor_interval_s=0.05,
            invoke=FakeServer(service_time_s=0.01, slots=16),
        )
        assert result.events
        assert [e.seq for e in result.events] == list(range(len(result.events)))

    def test_failures_are_classified_not_swallowed(self) -> None:
        # scalability.py swallows exceptions, so a 503 becomes a retry spin and
        # total_requests counts only successes. Here every outcome is recorded.
        result = run_step(
            client=None,
            model="m",
            endpoint="e",
            voice="v",
            texts=TEXTS,
            concurrency=4,
            duration_s=0.3,
            monitor_interval_s=0.05,
            invoke=FakeServer(service_time_s=0.01, slots=8, fail_with=InvokeOutcome.SATURATED_503),
        )
        assert {e.outcome for e in result.events} == {InvokeOutcome.SATURATED_503.value}

    def test_a_raising_transport_is_recorded_and_the_worker_continues(self) -> None:
        # An unrecorded request is one the step held a worker for and cannot
        # account for; a *dead* worker silently drops concurrency below N, which
        # invalidates the only thing this driver guarantees.
        calls = {"n": 0}

        def flaky(client, endpoint, text, voice, *, deadline_ts=None) -> InvokeResult:
            calls["n"] += 1
            if calls["n"] % 2 == 0:
                raise RuntimeError("connection reset")
            now = time.time()
            return InvokeResult(
                outcome=InvokeOutcome.OK, dispatch_ts=now, end_ts=now, latency_ms=1.0
            )

        result = run_step(
            client=None,
            model="m",
            endpoint="e",
            voice="v",
            texts=TEXTS,
            concurrency=1,
            duration_s=0.2,
            monitor_interval_s=0.05,
            invoke=flaky,
        )
        errors = [e for e in result.events if e.outcome == InvokeOutcome.ERROR.value]
        assert errors
        assert all(e.error_class == "RuntimeError" for e in errors)
        # Every event still has a completion time, so none vanish from the totals.
        assert all(e.end_ts is not None for e in result.events)
        # The worker survived: it kept issuing after the raise.
        assert len(result.events) > 2 * len(errors) - 1

    def test_deadline_is_measured_from_dispatch(self) -> None:
        # In a closed loop dispatch *is* arrival — there is no schedule to be late
        # against, which is one of the things this design removes.
        vclock = VirtualClock()
        clock = vclock.clock()
        seen: list[float] = []

        def spy(client, endpoint, text, voice, *, deadline_ts=None) -> InvokeResult:
            seen.append(deadline_ts)
            dispatch_ts = clock.time()
            vclock.advance(1.0)
            return InvokeResult(
                outcome=InvokeOutcome.OK,
                dispatch_ts=dispatch_ts,
                end_ts=clock.time(),
                latency_ms=1000.0,
            )

        result = run_step(
            client=None,
            model="m",
            endpoint="e",
            voice="v",
            texts=TEXTS,
            concurrency=1,
            duration_s=5.0,
            request_deadline_s=60.0,
            monitor_interval_s=1.0,
            clock=clock,
            invoke=spy,
        )
        assert seen
        for event, deadline in zip(result.events, seen, strict=True):
            assert deadline == pytest.approx((event.dispatch_ts or 0.0) + 60.0, abs=1e-6)

    def test_events_stream_to_the_sink_as_they_complete(self) -> None:
        sink_calls: list[LoadEvent] = []
        result = run_step(
            client=None,
            model="m",
            endpoint="e",
            voice="v",
            texts=TEXTS,
            concurrency=8,
            duration_s=0.3,
            monitor_interval_s=0.05,
            event_sink=sink_calls.append,
            invoke=FakeServer(service_time_s=0.01, slots=8),
        )
        assert len(sink_calls) == len(result.events)

    def test_a_broken_sink_does_not_lose_the_run(self) -> None:
        def explode(event: LoadEvent) -> None:
            raise OSError("disk full")

        result = run_step(
            client=None,
            model="m",
            endpoint="e",
            voice="v",
            texts=TEXTS,
            concurrency=8,
            duration_s=0.3,
            monitor_interval_s=0.05,
            event_sink=explode,
            invoke=FakeServer(service_time_s=0.01, slots=8),
        )
        assert result.events

    def test_abort_event_stops_the_loop_early(self) -> None:
        abort = threading.Event()

        def stop_immediately(client, endpoint, text, voice, *, deadline_ts=None) -> InvokeResult:
            now = time.time()
            abort.set()
            return InvokeResult(
                outcome=InvokeOutcome.OK, dispatch_ts=now, end_ts=now, latency_ms=1.0
            )

        started = time.monotonic()
        result = run_step(
            client=None,
            model="m",
            endpoint="e",
            voice="v",
            texts=TEXTS,
            concurrency=2,
            duration_s=30.0,
            monitor_interval_s=0.05,
            stop_event=abort,
            invoke=stop_immediately,
        )
        # It returned long before its 30s deadline, and every worker stopped.
        assert time.monotonic() - started < 5.0
        assert len(result.events) <= 4

    def test_callers_stop_event_survives_a_completed_step(self) -> None:
        # A ladder shares one abort event across steps. If finishing a step set
        # it, every later step would abort immediately and the ladder would
        # silently stop after one rung.
        abort = threading.Event()
        for step in range(2):
            result = run_step(
                client=None,
                model="m",
                endpoint="e",
                voice="v",
                texts=TEXTS,
                concurrency=4,
                duration_s=0.2,
                step_index=step,
                monitor_interval_s=0.05,
                stop_event=abort,
                invoke=FakeServer(service_time_s=0.005, slots=4),
            )
            assert not abort.is_set()
            assert result.events

    def test_monitor_samples_instance_count(self) -> None:
        result = run_step(
            client=None,
            model="m",
            endpoint="e",
            voice="v",
            texts=TEXTS,
            concurrency=4,
            duration_s=0.3,
            monitor_interval_s=0.05,
            instance_count_fetch=lambda: 1,
            invoke=FakeServer(service_time_s=0.01, slots=4),
        )
        assert result.samples
        assert result.instance_counts == (1,)
        assert not result.capacity_changed

    def test_capacity_change_mid_step_is_detected(self) -> None:
        # The mid-run tripwire: suspension is not proof, so a fleet that grows
        # anyway must invalidate the step. N outstanding across two instances is
        # a different measurement from N on one.
        counts = iter([1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])

        def fetch() -> int:
            return next(counts, 2)

        result = run_step(
            client=None,
            model="m",
            endpoint="e",
            voice="v",
            texts=TEXTS,
            concurrency=4,
            duration_s=0.5,
            monitor_interval_s=0.05,
            instance_count_fetch=fetch,
            instance_count_ttl_s=0.0,
            invoke=FakeServer(service_time_s=0.01, slots=4),
        )
        assert result.capacity_changed
        assert result.instance_counts == (1, 2)

    def test_instance_count_fetch_failure_does_not_fake_a_capacity_change(self) -> None:
        # A throttled describe_endpoint must not look like a resize; that would
        # discard a perfectly good step.
        calls = {"n": 0}

        def flaky() -> int:
            calls["n"] += 1
            if calls["n"] % 2 == 0:
                raise RuntimeError("Rate exceeded")
            return 1

        result = run_step(
            client=None,
            model="m",
            endpoint="e",
            voice="v",
            texts=TEXTS,
            concurrency=4,
            duration_s=0.4,
            monitor_interval_s=0.05,
            instance_count_fetch=flaky,
            instance_count_ttl_s=0.0,
            invoke=FakeServer(service_time_s=0.01, slots=4),
        )
        assert result.instance_counts == (1,)
        assert not result.capacity_changed

    def test_instance_count_is_cached_between_samples(self) -> None:
        # describe_endpoint is rate-limited; sampling it at 1Hz for a 240s step
        # would be both wasteful and throttle-prone.
        calls = {"n": 0}

        def counting() -> int:
            calls["n"] += 1
            return 1

        run_step(
            client=None,
            model="m",
            endpoint="e",
            voice="v",
            texts=TEXTS,
            concurrency=8,
            duration_s=0.5,
            monitor_interval_s=0.02,
            instance_count_fetch=counting,
            instance_count_ttl_s=10.0,
            invoke=FakeServer(service_time_s=0.005, slots=8),
        )
        # One fetch for the whole step, not one per event or per sample.
        assert calls["n"] == 1

    def test_texts_cycle_so_steps_are_comparable(self) -> None:
        # Two steps of equal length must synthesize the same characters, or a step
        # could show a worse latency merely for having drawn longer texts. The
        # sequence counter is global across workers, so the *set* of texts a step
        # consumes depends only on how many requests it made.
        seen: list[str] = []

        def spy(client, endpoint, text, voice, *, deadline_ts=None) -> InvokeResult:
            seen.append(text)
            now = time.time()
            return InvokeResult(
                outcome=InvokeOutcome.OK, dispatch_ts=now, end_ts=now, latency_ms=1.0
            )

        run_step(
            client=None,
            model="m",
            endpoint="e",
            voice="v",
            texts=TEXTS,
            concurrency=1,
            duration_s=0.2,
            monitor_interval_s=0.05,
            invoke=spy,
        )
        assert seen[: len(TEXTS)] == TEXTS

    def test_a_long_step_runs_instantly_under_an_injected_clock(self) -> None:
        # A 240s step, in milliseconds. The clock is injected precisely so the
        # loop is testable without waiting for it, and a closed loop makes the
        # arithmetic exact: one worker at 1s service over 240s is 240 requests.
        vclock = VirtualClock()
        started = time.monotonic()
        result = run_step(
            client=None,
            model="m",
            endpoint="e",
            voice="v",
            texts=TEXTS,
            concurrency=1,
            duration_s=240.0,
            monitor_interval_s=1.0,
            clock=vclock.clock(),
            invoke=virtual_transport(vclock, 1.0),
        )
        assert time.monotonic() - started < 10.0
        assert len(result.events) == 240
        assert result.ended_ts - result.started_ts == pytest.approx(240.0)

    def test_rejects_zero_concurrency(self) -> None:
        with pytest.raises(ValueError, match="concurrency must be >= 1"):
            run_step(
                client=None,
                model="m",
                endpoint="e",
                voice="v",
                texts=TEXTS,
                concurrency=0,
                duration_s=1.0,
            )

    def test_rejects_negative_duration(self) -> None:
        with pytest.raises(ValueError, match="duration_s must be non-negative"):
            run_step(
                client=None,
                model="m",
                endpoint="e",
                voice="v",
                texts=TEXTS,
                concurrency=1,
                duration_s=-1.0,
            )

    def test_rejects_empty_text_pool(self) -> None:
        with pytest.raises(ValueError, match="texts must not be empty"):
            run_step(
                client=None,
                model="m",
                endpoint="e",
                voice="v",
                texts=[],
                concurrency=1,
                duration_s=1.0,
            )

    def test_a_zero_duration_step_returns_empty_rather_than_hanging(self) -> None:
        result = run_step(
            client=None,
            model="m",
            endpoint="e",
            voice="v",
            texts=TEXTS,
            concurrency=4,
            duration_s=0.0,
            monitor_interval_s=0.05,
            invoke=FakeServer(service_time_s=0.01, slots=4),
        )
        assert result.events == []


class TestWorkerOverlaps:
    def test_a_worker_reusing_itself_before_completion_is_counted(self) -> None:
        # The falsifiable form of the concurrency guarantee. Hand-built because a
        # correct driver cannot produce it — which is the point of checking.
        overlapping = [
            _event(seq=0, worker_index=0, dispatch_ts=0.0, end_ts=2.0),
            _event(seq=1, worker_index=0, dispatch_ts=1.0, end_ts=3.0),
        ]
        assert _step(overlapping).worker_overlaps == 1

    def test_sequential_requests_on_one_worker_do_not_overlap(self) -> None:
        clean = [
            _event(seq=0, worker_index=0, dispatch_ts=0.0, end_ts=1.0),
            _event(seq=1, worker_index=0, dispatch_ts=1.0, end_ts=2.0),
        ]
        assert _step(clean).worker_overlaps == 0

    def test_different_workers_are_expected_to_overlap(self) -> None:
        # N concurrent requests means N workers busy at once; counting those as
        # overlaps would flag every healthy step.
        concurrent = [
            _event(seq=0, worker_index=0, dispatch_ts=0.0, end_ts=2.0),
            _event(seq=1, worker_index=1, dispatch_ts=0.0, end_ts=2.0),
        ]
        assert _step(concurrent).worker_overlaps == 0

    def test_events_with_no_completion_time_are_skipped(self) -> None:
        partial = [
            _event(seq=0, worker_index=0, dispatch_ts=0.0, end_ts=None),
            _event(seq=1, worker_index=0, dispatch_ts=1.0, end_ts=2.0),
        ]
        assert _step(partial).worker_overlaps == 0


class TestTransportAgnosticism:
    """``run_step`` must not care which transport it is driving.

    With ``invoke=invoke_bidi`` the client is a ``SageMakerRuntimeHTTP2Client``,
    which shares no base class with botocore's. The driver never calls a method
    on it — it only hands it to ``invoke`` — so client and transport stay a
    matched pair chosen by the caller.
    """

    def test_the_client_is_passed_through_untouched(self) -> None:
        # Not merely "an object arrives": the *same* object, unwrapped. A driver
        # that adapted the client would silently break whichever transport it
        # was not written for.
        sentinel = object()
        seen: list[object] = []

        def spy(client, endpoint, text, voice, *, deadline_ts=None) -> InvokeResult:
            seen.append(client)
            now = time.time()
            return InvokeResult(
                outcome=InvokeOutcome.OK, dispatch_ts=now, end_ts=now, latency_ms=1.0
            )

        run_step(
            client=sentinel,
            model="m",
            endpoint="e",
            voice="v",
            texts=TEXTS,
            concurrency=4,
            duration_s=0.3,
            monitor_interval_s=0.05,
            invoke=spy,
        )
        assert seen
        assert all(c is sentinel for c in seen)

    def test_a_non_botocore_client_is_accepted(self) -> None:
        # The bidi client's type is the point: run_step's annotation was widened
        # to Any precisely so this is not a type error waiting to surprise a
        # ladder 40 minutes in.
        class NotABaseClient:
            """Has none of botocore's interface."""

        result = run_step(
            client=NotABaseClient(),
            model="m",
            endpoint="e",
            voice="v",
            texts=TEXTS,
            concurrency=4,
            duration_s=0.3,
            monitor_interval_s=0.05,
            invoke=lambda client, endpoint, text, voice, *, deadline_ts=None: InvokeResult(
                outcome=InvokeOutcome.OK,
                dispatch_ts=time.time(),
                end_ts=time.time(),
                latency_ms=1.0,
                audio_duration_s=0.1,
            ),
        )
        assert result.events
        assert all(e.outcome == InvokeOutcome.OK.value for e in result.events)

    def test_drives_the_real_bidi_transport_end_to_end(self) -> None:
        # The bidi transport wraps each session in asyncio.run inside the worker
        # thread it was called on. run_step gives each worker its own thread, so
        # this is the test that the two actually compose — a shared or missing
        # event loop fails here rather than mid-ladder.
        from tests.test_bidi import PCM_100MS, FakeBidiClient, FakeStream, _payload_event
        from tts_bench.bidi import invoke_bidi

        class PerRequestClient:
            """A fresh scripted session per request, as a real endpoint gives."""

            def __init__(self) -> None:
                self.sessions = 0
                self._lock = threading.Lock()

            async def invoke_endpoint_with_bidirectional_stream(self, input_):
                with self._lock:
                    self.sessions += 1
                delegate = FakeBidiClient(FakeStream([_payload_event(PCM_100MS)]))
                return await delegate.invoke_endpoint_with_bidirectional_stream(input_)

        client = PerRequestClient()
        result = run_step(
            client=client,
            model="kokoro-82m",
            endpoint="speech-kokoro-82m",
            voice="af_heart",
            texts=TEXTS,
            concurrency=4,
            duration_s=0.3,
            monitor_interval_s=0.05,
            invoke=invoke_bidi,
        )

        completed = [e for e in result.events if e.end_ts is not None]
        assert completed
        assert all(e.outcome == InvokeOutcome.OK.value for e in completed)
        # The accounting the ladder reads is populated on this transport too.
        assert all(e.audio_duration_s > 0 for e in completed)
        assert all(e.ttfab_ms is not None for e in completed)
        assert client.sessions == len(completed)
        assert result.worker_overlaps == 0


class TestSummarizeWindow:
    def test_achieved_rps_counts_completions_in_the_window(self) -> None:
        events = [_event(seq=i, end_ts=float(i)) for i in range(10)]
        stats = summarize_window(_step(events), start_ts=0.0, end_ts=10.0)
        assert stats.completed == 10
        assert stats.achieved_rps == pytest.approx(1.0)

    def test_events_outside_the_window_are_excluded(self) -> None:
        # Warm-up discard depends on this: only the measure window counts.
        events = [_event(seq=i, end_ts=float(i)) for i in range(10)]
        stats = summarize_window(_step(events), start_ts=5.0, end_ts=10.0)
        assert stats.completed == 5

    def test_the_step_concurrency_is_carried_through(self) -> None:
        # The x-axis of the ladder. It is the step's input, not something derived
        # from the events, so it cannot drift from what was actually held.
        stats = summarize_window(_step([_event(seq=0)], concurrency=30), start_ts=0.0, end_ts=2.0)
        assert stats.concurrency == 30

    def test_percentiles_use_successes_only(self) -> None:
        # A fast rejection must not pull p95 down and make overload look better.
        good = [_event(seq=i, end_ts=i / 10.0, latency_ms=1000.0) for i in range(5)]
        rejects = [
            _event(
                seq=5 + i,
                end_ts=(5 + i) / 10.0,
                latency_ms=1.0,
                ttfab_ms=None,
                outcome=InvokeOutcome.SATURATED_503,
            )
            for i in range(50)
        ]
        stats = summarize_window(_step(good + rejects), start_ts=0.0, end_ts=6.0)
        assert stats.latency_p95_ms == pytest.approx(1000.0)
        assert stats.ok == 5
        assert stats.outcome_counts[InvokeOutcome.SATURATED_503.value] == 50

    def test_events_with_no_completion_are_placed_by_dispatch(self) -> None:
        # A transport that raised before the driver could stamp an end time still
        # has to appear in the outcome table, or the totals stop reconciling.
        events = [
            _event(seq=0, end_ts=1.0),
            _event(
                seq=1,
                dispatch_ts=2.0,
                end_ts=None,
                latency_ms=None,
                ttfab_ms=None,
                outcome=InvokeOutcome.CLIENT_TIMEOUT,
            ),
        ]
        stats = summarize_window(_step(events), start_ts=0.0, end_ts=5.0)
        assert stats.outcome_counts[InvokeOutcome.CLIENT_TIMEOUT.value] == 1
        assert stats.completed == 1  # not a completion

    def test_service_time_is_reported_in_seconds(self) -> None:
        # planner.py consumes S in seconds; a millisecond leak here would be a
        # 1000x error in every downstream instance count.
        events = [_event(seq=i, end_ts=float(i), latency_ms=120.0) for i in range(4)]
        stats = summarize_window(_step(events), start_ts=0.0, end_ts=4.0)
        assert stats.s_mean_s == pytest.approx(0.12)
        assert stats.s_p95_s == pytest.approx(0.12)

    def test_service_time_is_none_without_successes(self) -> None:
        events = [
            _event(seq=i, end_ts=float(i), outcome=InvokeOutcome.SATURATED_503) for i in range(3)
        ]
        stats = summarize_window(_step(events), start_ts=0.0, end_ts=3.0)
        assert stats.s_mean_s is None
        assert stats.s_p95_s is None
        assert stats.ttfab_p95_ms is None

    def test_capacity_change_makes_a_window_unusable_for_the_ladder(self) -> None:
        samples = [
            ConcurrencySample(ts=0.0, in_flight=1, instance_count=1),
            ConcurrencySample(ts=1.0, in_flight=1, instance_count=2),
        ]
        events = [_event(seq=0, end_ts=0.5)]
        stats = summarize_window(_step(events, samples), start_ts=0.0, end_ts=2.0)
        assert stats.capacity_changed
        assert not stats.usable_for_ladder

    def test_empty_window_is_unusable_rather_than_a_zero_rung(self) -> None:
        stats = summarize_window(_step([]), start_ts=0.0, end_ts=1.0)
        assert stats.completed == 0
        assert not stats.usable_for_ladder

    def test_rejects_inverted_window(self) -> None:
        with pytest.raises(ValueError, match="window must be non-empty"):
            summarize_window(_step([]), start_ts=5.0, end_ts=5.0)


class TestSaturation:
    """Saturation is the server refusing work, not a rate comparison.

    A closed-loop driver cannot outrun the server, so "achieved below offered" has
    no meaning here. Reading ``outcome_counts`` also closes a real hole in the
    open-loop version, where a 1ms 503 counted as a completion and so *raised*
    achieved throughput — a fully-rejecting endpoint looked healthy.
    """

    def test_rejection_outcomes_are_the_three_refusals(self) -> None:
        assert REJECTION_OUTCOMES == {
            InvokeOutcome.SATURATED_503.value,
            InvokeOutcome.STALE_408.value,
            InvokeOutcome.THROTTLED_429.value,
        }
        # A model crash is not a refusal: it says nothing about queue depth.
        assert InvokeOutcome.MODEL_ERROR.value not in REJECTION_OUTCOMES
        assert InvokeOutcome.SERVER_5XX.value not in REJECTION_OUTCOMES

    def test_a_fully_rejecting_step_is_saturated(self) -> None:
        # The regression: these all have an end_ts, so the old rate-based rule
        # counted them as completions and read the step as keeping up.
        events = [
            _event(seq=i, end_ts=i / 100.0, latency_ms=1.0, outcome=InvokeOutcome.SATURATED_503)
            for i in range(100)
        ]
        stats = summarize_window(_step(events), start_ts=0.0, end_ts=1.0)
        assert stats.achieved_rps == pytest.approx(100.0)
        assert stats.rejected == 100
        assert stats.saturated
        assert stats.outcome_counts["saturated_503"] == 100

    def test_a_fully_rejecting_step_run_end_to_end_is_saturated(self) -> None:
        # Same conclusion through the driver rather than hand-built events, since
        # this is the precondition check a Q_max ladder relies on. Reads the whole
        # step: rejections come back fast, so there is no drain tail to exclude.
        result = run_step(
            client=None,
            model="m",
            endpoint="e",
            voice="v",
            texts=TEXTS,
            concurrency=4,
            duration_s=0.3,
            monitor_interval_s=0.05,
            invoke=FakeServer(service_time_s=0.01, slots=8, fail_with=InvokeOutcome.SATURATED_503),
        )
        stats = summarize_window(result, start_ts=result.started_ts, end_ts=result.ended_ts)
        assert stats.saturated
        assert stats.outcome_counts["saturated_503"] == stats.completed
        assert stats.ok == 0

    def test_an_isolated_throttle_is_not_saturation(self) -> None:
        # SageMaker throttles occasionally. Discarding a whole hold for one 429
        # would make a long ladder unfinishable.
        assert REJECTION_RATIO == 0.01
        events = [_event(seq=i, end_ts=i / 200.0) for i in range(199)]
        events.append(
            _event(seq=199, end_ts=0.995, outcome=InvokeOutcome.THROTTLED_429, ttfab_ms=None)
        )
        stats = summarize_window(_step(events), start_ts=0.0, end_ts=1.0)
        assert stats.rejected == 1
        assert not stats.saturated

    def test_a_healthy_step_is_not_saturated(self) -> None:
        events = [_event(seq=i, end_ts=i / 10.0) for i in range(10)]
        stats = summarize_window(_step(events), start_ts=0.0, end_ts=1.0)
        assert stats.rejected == 0
        assert not stats.saturated

    def test_an_empty_window_is_not_saturated(self) -> None:
        # No evidence either way. `usable_for_ladder` is what excludes it.
        stats = summarize_window(_step([]), start_ts=0.0, end_ts=1.0)
        assert not stats.saturated
        assert not stats.usable_for_ladder


class TestConcurrencyShortfall:
    """Mean in-flight below ``N`` means the *client* was the limit.

    The closed-loop replacement for ``dispatch_skipped``: there is no schedule to
    skip against, but there is still a number the driver promised to hold.
    """

    def test_holding_n_exactly_has_no_shortfall(self) -> None:
        samples = [ConcurrencySample(ts=float(i), in_flight=5, instance_count=1) for i in range(10)]
        stats = summarize_window(
            _step([_event()], samples, concurrency=5), start_ts=0.0, end_ts=10.0
        )
        assert stats.concurrency_shortfall == pytest.approx(0.0)
        assert not stats.client_bound

    def test_a_large_shortfall_reads_as_client_bound(self) -> None:
        # Half the requested concurrency never went out: whatever this step
        # measured, it was not the server at N.
        samples = [ConcurrencySample(ts=float(i), in_flight=5, instance_count=1) for i in range(10)]
        stats = summarize_window(
            _step([_event()], samples, concurrency=10), start_ts=0.0, end_ts=10.0
        )
        assert stats.concurrency_shortfall == pytest.approx(0.5)
        assert stats.client_bound

    def test_ordinary_turnaround_overhead_is_tolerated(self) -> None:
        # A worker spends a little time between completing one request and
        # dispatching the next, so the shortfall is never exactly zero.
        assert CONCURRENCY_SHORTFALL_RATIO == 0.05
        samples = [ConcurrencySample(ts=float(i), in_flight=39, instance_count=1) for i in range(9)]
        samples.append(ConcurrencySample(ts=9.0, in_flight=40, instance_count=1))
        stats = summarize_window(
            _step([_event()], samples, concurrency=40), start_ts=0.0, end_ts=10.0
        )
        assert stats.concurrency_shortfall < CONCURRENCY_SHORTFALL_RATIO
        assert not stats.client_bound

    def test_exceeding_n_is_reported_as_no_shortfall_not_a_negative_one(self) -> None:
        # Sampling can catch a release-then-acquire boundary; a negative fraction
        # would read as "better than requested", which is not a thing.
        samples = [ConcurrencySample(ts=float(i), in_flight=6, instance_count=1) for i in range(4)]
        stats = summarize_window(
            _step([_event()], samples, concurrency=5), start_ts=0.0, end_ts=4.0
        )
        assert stats.concurrency_shortfall == 0.0
        assert not stats.client_bound

    def test_shortfall_is_none_without_samples(self) -> None:
        stats = summarize_window(_step([_event()]), start_ts=0.0, end_ts=2.0)
        assert stats.concurrency_mean is None
        assert stats.concurrency_shortfall is None
        assert not stats.client_bound

    def test_peak_is_reported_separately_from_the_mean(self) -> None:
        # The CloudWatch threshold is compared against `Maximum`, not `Average`,
        # so the peak is not a curiosity — it is the deployed unit.
        samples = [ConcurrencySample(ts=float(i), in_flight=1, instance_count=1) for i in range(9)]
        samples.append(ConcurrencySample(ts=9.0, in_flight=20, instance_count=1))
        stats = summarize_window(
            _step([_event()], samples, concurrency=20), start_ts=0.0, end_ts=10.0
        )
        assert stats.concurrency_peak == 20
        assert stats.concurrency_mean == pytest.approx(2.9)


class TestSettled:
    """Steady state is judged on latency drift, not on an in-flight trend.

    In-flight is pinned at ``N`` by construction here, so a slope on it would
    always read zero — a check that silently always passes is worse than no check.
    """

    def test_flat_latency_is_settled(self) -> None:
        events = [_event(seq=i, end_ts=float(i), ttfab_ms=500.0) for i in range(10)]
        stats = summarize_window(_step(events), start_ts=0.0, end_ts=10.0)
        assert stats.ttfab_drift_ms == pytest.approx(0.0, abs=1e-6)
        assert stats.settled

    def test_climbing_latency_is_not_settled(self) -> None:
        # On a serial server a queue that is still filling grows latency without
        # bound, so whatever p95 the window reports is a moving number.
        events = [_event(seq=i, end_ts=float(i), ttfab_ms=100.0 + 50.0 * i) for i in range(10)]
        stats = summarize_window(_step(events), start_ts=0.0, end_ts=10.0)
        assert stats.ttfab_drift_ms == pytest.approx(500.0)
        assert not stats.settled

    def test_drift_is_reported_across_the_window_not_per_second(self) -> None:
        # A slope is hard to judge without knowing the window length; "latency
        # rose 500ms over this window" is not.
        events = [
            _event(seq=i, end_ts=float(i) / 2.0, ttfab_ms=100.0 + 50.0 * i) for i in range(10)
        ]
        stats = summarize_window(_step(events), start_ts=0.0, end_ts=5.0)
        # Same 50ms per event, but twice the rate, so the same total drift.
        assert stats.ttfab_drift_ms == pytest.approx(500.0)

    def test_drift_within_the_tolerance_is_settled(self) -> None:
        assert LATENCY_DRIFT_RATIO == 0.25
        events = [_event(seq=i, end_ts=float(i), ttfab_ms=1000.0 + 2.0 * i) for i in range(10)]
        stats = summarize_window(_step(events), start_ts=0.0, end_ts=10.0)
        assert stats.ttfab_drift_ms == pytest.approx(20.0)
        assert stats.settled

    def test_falling_latency_is_settled(self) -> None:
        # A warm-up tail draining is not an unsettled queue, and rejecting it
        # would discard exactly the steps that recovered.
        events = [_event(seq=i, end_ts=float(i), ttfab_ms=1000.0 - 50.0 * i) for i in range(10)]
        stats = summarize_window(_step(events), start_ts=0.0, end_ts=10.0)
        assert stats.ttfab_drift_ms is not None
        assert stats.ttfab_drift_ms < 0
        assert stats.settled

    def test_too_few_points_to_fit_is_settled_with_no_drift(self) -> None:
        events = [_event(seq=i, end_ts=float(i)) for i in range(2)]
        stats = summarize_window(_step(events), start_ts=0.0, end_ts=3.0)
        assert stats.ttfab_drift_ms is None
        assert stats.settled

    def test_simultaneous_completions_do_not_produce_a_fit(self) -> None:
        # Zero spread on x is a vertical line, not an infinite slope.
        events = [_event(seq=i, end_ts=1.0, ttfab_ms=100.0 * i) for i in range(5)]
        stats = summarize_window(_step(events), start_ts=0.0, end_ts=2.0)
        assert stats.ttfab_drift_ms is None
        assert stats.settled


class TestJsonlWriter:
    def test_writes_one_flushed_line_per_event(self, tmp_path) -> None:
        path = tmp_path / "events.jsonl"
        with JsonlWriter(path) as writer:
            writer(_event(seq=0))
            writer(_event(seq=1))
            # Flushed as it goes: a run killed here must keep what it measured.
            assert len(path.read_text().splitlines()) == 2

        rows = [json.loads(line) for line in path.read_text().splitlines()]
        assert [r["seq"] for r in rows] == [0, 1]
        assert rows[0]["outcome"] == "ok"

    def test_the_concurrency_and_worker_are_on_every_line(self) -> None:
        # `worker_overlaps` is checkable from the artifact alone only if both
        # reach it, and `concurrency` is the ladder's x-axis.
        row = json.loads(_event(seq=0, worker_index=3, concurrency=30).to_json())
        assert row["worker_index"] == 3
        assert row["concurrency"] == 30

    def test_creates_parent_directories(self, tmp_path) -> None:
        path = tmp_path / "nested" / "deeper" / "events.jsonl"
        with JsonlWriter(path) as writer:
            writer(_event())
        assert path.exists()

    def test_appends_across_steps(self, tmp_path) -> None:
        path = tmp_path / "events.jsonl"
        for step in range(2):
            with JsonlWriter(path) as writer:
                writer(_event(seq=step))
        assert len(path.read_text().splitlines()) == 2

    def test_refuses_use_outside_its_context(self, tmp_path) -> None:
        writer = JsonlWriter(tmp_path / "events.jsonl")
        with pytest.raises(RuntimeError, match="outside its context manager"):
            writer(_event())

    def test_is_usable_as_a_run_step_sink(self, tmp_path) -> None:
        path = tmp_path / "events.jsonl"
        with JsonlWriter(path) as writer:
            result = run_step(
                client=None,
                model="m",
                endpoint="e",
                voice="v",
                texts=TEXTS,
                concurrency=8,
                duration_s=0.3,
                monitor_interval_s=0.05,
                event_sink=writer,
                invoke=FakeServer(service_time_s=0.01, slots=8),
            )
        assert len(path.read_text().splitlines()) == len(result.events)


class TestLoadEvent:
    def test_round_trips_through_json(self) -> None:
        event = _event(seq=3)
        assert json.loads(event.to_json())["seq"] == 3

    def test_ok_reflects_the_outcome_string(self) -> None:
        assert _event().ok
        assert not _event(outcome=InvokeOutcome.SATURATED_503).ok

    def test_is_immutable(self) -> None:
        event = _event()
        with pytest.raises(Exception):  # noqa: B017 - FrozenInstanceError
            event.seq = 9  # type: ignore[misc]
        assert replace(event, seq=9).seq == 9


class TestBuildTextPool:
    def test_is_deterministic_for_a_seed(self) -> None:
        # Two steps must synthesize the same characters, or a rung could look
        # worse merely for having drawn longer texts.
        assert build_text_pool(TEXTS, seed=5) == build_text_pool(TEXTS, seed=5)

    def test_preserves_every_text(self) -> None:
        assert sorted(build_text_pool(TEXTS, seed=1)) == sorted(TEXTS)

    def test_rejects_an_empty_pool(self) -> None:
        with pytest.raises(ValueError, match="text pool is empty"):
            build_text_pool([])


class TestInstanceCountFetcher:
    def test_reads_current_not_desired(self) -> None:
        # Desired changes the moment a scaling action fires; only Current means
        # an instance is actually there.
        class FakeSageMaker:
            def describe_endpoint(self, EndpointName: str) -> dict:  # noqa: N803 - boto3 API
                return {
                    "ProductionVariants": [
                        {
                            "VariantName": "primary",
                            "CurrentInstanceCount": 2,
                            "DesiredInstanceCount": 4,
                        }
                    ]
                }

        assert make_instance_count_fetcher(FakeSageMaker(), "speech-kokoro-82m")() == 2

    def test_unknown_variant_reads_zero(self) -> None:
        class FakeSageMaker:
            def describe_endpoint(self, EndpointName: str) -> dict:  # noqa: N803 - boto3 API
                return {"ProductionVariants": [{"VariantName": "other"}]}

        assert make_instance_count_fetcher(FakeSageMaker(), "e")() == 0


class TestSystemClock:
    def test_uses_monotonic_for_the_deadline_and_wall_time_for_stamps(self) -> None:
        # A monotonic step deadline survives an NTP step; wall-clock stamps are
        # what correlate events with CloudWatch datapoints and container logs.
        assert SYSTEM_CLOCK.monotonic() < SYSTEM_CLOCK.time()
