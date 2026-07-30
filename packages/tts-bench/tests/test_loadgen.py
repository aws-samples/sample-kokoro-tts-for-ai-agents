"""Tests for the open-loop load generator.

The load generator is the one component whose correctness cannot be checked by
reading its output: a closed-loop driver produces plausible-looking numbers that
are simply wrong. ``TestOpenLoopProperty`` is therefore the load-bearing test in
this file — it fails if anyone reintroduces response-gated dispatch, whatever
the code looks like.

Real threads with sub-second service times, matching the existing pattern in
``test_scalability.py``. The clock is injected only where waiting would dominate.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import replace

import pytest

from tts_bench.invoke import InvokeOutcome, InvokeResult
from tts_bench.loadgen import (
    SATURATION_RATIO,
    SYSTEM_CLOCK,
    ArrivalProcess,
    Clock,
    ConcurrencySample,
    JsonlWriter,
    LoadEvent,
    StepResult,
    arrival_offsets,
    build_text_pool,
    make_instance_count_fetcher,
    run_step,
    summarize_window,
)

TEXTS = ["Let me check that for you.", "Your appointment is confirmed.", "One moment please."]


class FakeServer:
    """A server with a fixed service time and a concurrency limit of ``slots``.

    Stands in for Kokoro's whole-stream lock: with ``slots=1`` throughput can
    never exceed ``1/service_time`` no matter how fast requests arrive, which is
    the shape the open-loop property test needs.
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


def _event(
    *,
    seq: int = 0,
    outcome: InvokeOutcome = InvokeOutcome.OK,
    scheduled_ts: float = 0.0,
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
        offered_rps=10.0,
        model="kokoro-82m",
        endpoint="speech-kokoro-82m",
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
    offered_rps: float = 10.0,
) -> StepResult:
    return StepResult(
        run_id="run",
        step_index=0,
        offered_rps=offered_rps,
        model="kokoro-82m",
        endpoint="speech-kokoro-82m",
        arrival_process="poisson",
        seed=7,
        started_ts=0.0,
        ended_ts=10.0,
        scheduled_count=len(events),
        events=events,
        samples=samples or [],
    )


class TestArrivalOffsets:
    def test_fixed_is_evenly_spaced(self) -> None:
        offsets = arrival_offsets(10.0, 1.0, process=ArrivalProcess.FIXED)
        assert len(offsets) == 10
        assert offsets[0] == 0.0
        gaps = [b - a for a, b in zip(offsets, offsets[1:], strict=False)]
        assert all(g == pytest.approx(0.1) for g in gaps)

    def test_poisson_mean_rate_matches_target(self) -> None:
        # Poisson is the default because fixed spacing is the best case and
        # understates queueing; the count should still track lambda*T.
        offsets = arrival_offsets(50.0, 20.0, seed=42)
        assert len(offsets) == pytest.approx(1000, rel=0.1)

    def test_poisson_is_not_evenly_spaced(self) -> None:
        # If gaps were uniform, the generator would be silently producing the
        # best case while claiming Poisson.
        offsets = arrival_offsets(50.0, 10.0, seed=1)
        gaps = [b - a for a, b in zip(offsets, offsets[1:], strict=False)]
        assert max(gaps) > 3 * min(gaps)

    def test_seed_makes_schedules_reproducible(self) -> None:
        assert arrival_offsets(20.0, 5.0, seed=99) == arrival_offsets(20.0, 5.0, seed=99)

    def test_different_seeds_differ(self) -> None:
        assert arrival_offsets(20.0, 5.0, seed=1) != arrival_offsets(20.0, 5.0, seed=2)

    def test_offsets_are_ascending_and_bounded(self) -> None:
        offsets = arrival_offsets(30.0, 4.0, seed=3)
        assert offsets == sorted(offsets)
        assert all(0.0 <= o < 4.0 for o in offsets)

    def test_zero_rate_yields_no_arrivals(self) -> None:
        assert arrival_offsets(0.0, 10.0) == []
        assert arrival_offsets(10.0, 0.0) == []

    def test_offsets_are_absolute_not_cumulative_sleeps(self) -> None:
        # Absolute offsets are what stop a slow dispatch from pushing every
        # later arrival late; a drifting schedule silently lowers offered load.
        offsets = arrival_offsets(4.0, 2.5, process="fixed")
        assert offsets == [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25]

    @pytest.mark.parametrize(("rate", "duration"), [(-1.0, 1.0), (1.0, -1.0)])
    def test_rejects_negative_inputs(self, rate: float, duration: float) -> None:
        with pytest.raises(ValueError):
            arrival_offsets(rate, duration)


class TestOpenLoopProperty:
    """The regression test that protects the point of this module.

    A closed-loop driver cannot produce these signals: it sends slower when the
    server slows, so offered rate collapses to achieved rate and in-flight count
    stays pinned at the thread count.

    Thresholds are expressed relative to the fake server's known capacity
    (``1/service_time`` at ``slots=1``) rather than as bare numbers, so the
    assertions state the property instead of encoding a measurement.
    """

    OVERLOAD_S = 0.05
    """Service time. Capacity is 20 req/s serialized."""

    OVERLOAD_RPS = 60.0
    """Offered rate: 3x capacity, so the queue must grow."""

    DURATION_S = 1.2

    def _overloaded(self) -> tuple[FakeServer, StepResult]:
        server = FakeServer(service_time_s=self.OVERLOAD_S, slots=1)
        result = run_step(
            client=None,
            model="kokoro-82m",
            endpoint="speech-kokoro-82m",
            voice="af_heart",
            texts=TEXTS,
            offered_rps=self.OVERLOAD_RPS,
            duration_s=self.DURATION_S,
            max_workers=200,
            arrival=ArrivalProcess.FIXED,
            monitor_interval_s=0.05,
            invoke=server,
        )
        return server, result

    def test_offered_rate_holds_while_achieved_pins_at_capacity(self) -> None:
        server, result = self._overloaded()
        capacity_rps = 1.0 / self.OVERLOAD_S

        # The schedule was followed, not negotiated with the server.
        assert result.scheduled_count == int(self.OVERLOAD_RPS * self.DURATION_S)
        assert len(result.events) == result.scheduled_count

        achieved = server.completed / (result.ended_ts - result.started_ts)
        # Achieved throughput pinned at the server's ceiling...
        assert achieved < 1.5 * capacity_rps
        # ...while the offered rate stayed far above it. This gap is the
        # saturation signal a closed-loop driver destroys by construction.
        assert result.offered_rps > 2 * achieved

    def test_in_flight_count_grows_under_overload(self) -> None:
        _, result = self._overloaded()

        in_flight = [s.in_flight for s in result.samples]
        assert max(in_flight) > 10, f"in-flight never built up: {in_flight}"

        # Only the dispatch window. Sampling continues while the pool drains
        # after the last arrival, and that tail falls monotonically by
        # construction — including it would test the drain, not the backlog.
        dispatch_end = result.started_ts + self.DURATION_S
        dispatching = [s.in_flight for s in result.samples if s.ts < dispatch_end]
        half = len(dispatching) // 2
        assert max(dispatching[half:]) > max(dispatching[:half])

        # Growth is a sustained trend, not a spike. Backlog accumulates at
        # (offered - capacity) req/s, so allow generous slack for scheduling
        # jitter but require the sign and rough magnitude.
        expected_slope = self.OVERLOAD_RPS - 1.0 / self.OVERLOAD_S
        stats = summarize_window(result, start_ts=result.started_ts, end_ts=dispatch_end)
        assert stats.concurrency_slope_per_s is not None
        assert stats.concurrency_slope_per_s > expected_slope / 4
        assert not stats.settled

    def test_slow_server_does_not_reduce_requests_sent(self) -> None:
        # The defining difference from measure_scalability: a 200x slower server
        # gets exactly the same schedule.
        def step(service_time_s: float, slots: int) -> StepResult:
            return run_step(
                client=None,
                model="m",
                endpoint="e",
                voice="v",
                texts=TEXTS,
                offered_rps=40.0,
                duration_s=0.5,
                max_workers=200,
                arrival=ArrivalProcess.FIXED,
                monitor_interval_s=0.1,
                invoke=FakeServer(service_time_s=service_time_s, slots=slots),
            )

        fast = step(0.001, 8)
        slow = step(0.2, 1)
        assert fast.scheduled_count == slow.scheduled_count == 20
        assert len(fast.events) == len(slow.events)


class TestRunStep:
    def test_every_scheduled_arrival_produces_exactly_one_event(self) -> None:
        # Reconciliation: a run whose totals do not account for every scheduled
        # request cannot support a capacity conclusion.
        result = run_step(
            client=None,
            model="m",
            endpoint="e",
            voice="v",
            texts=TEXTS,
            offered_rps=30.0,
            duration_s=0.5,
            max_workers=16,
            arrival=ArrivalProcess.FIXED,
            monitor_interval_s=0.05,
            invoke=FakeServer(service_time_s=0.01, slots=16),
        )
        assert len(result.events) == result.scheduled_count
        assert [e.seq for e in result.events] == list(range(result.scheduled_count))

    def test_dispatch_skipped_is_recorded_not_blocked(self) -> None:
        # With one worker and a fast arrival rate, most arrivals cannot be sent.
        # They must be counted, because they mean the *client* ran out of room.
        server = FakeServer(service_time_s=0.2, slots=1)
        started = time.monotonic()
        result = run_step(
            client=None,
            model="m",
            endpoint="e",
            voice="v",
            texts=TEXTS,
            offered_rps=50.0,
            duration_s=0.6,
            max_workers=1,
            arrival=ArrivalProcess.FIXED,
            monitor_interval_s=0.05,
            invoke=server,
        )
        elapsed = time.monotonic() - started

        assert result.dispatch_skipped > 0
        assert len(result.events) == result.scheduled_count
        # The dispatcher walked the whole schedule in roughly the step duration
        # rather than stretching to fit the server: it never blocked.
        assert elapsed < 0.6 + 3 * 0.2

    def test_skipped_events_carry_no_completion_time(self) -> None:
        result = run_step(
            client=None,
            model="m",
            endpoint="e",
            voice="v",
            texts=TEXTS,
            offered_rps=50.0,
            duration_s=0.4,
            max_workers=1,
            arrival=ArrivalProcess.FIXED,
            monitor_interval_s=0.05,
            invoke=FakeServer(service_time_s=0.2, slots=1),
        )
        skipped = [e for e in result.events if e.outcome == InvokeOutcome.DISPATCH_SKIPPED.value]
        assert skipped
        for event in skipped:
            assert event.end_ts is None
            assert event.latency_ms is None
            assert event.chars > 0  # still accounted for in the outcome table

    def test_failures_are_classified_not_swallowed(self) -> None:
        # scalability.py swallows exceptions, so a 503 becomes a retry spin and
        # total_requests counts only successes. Here every outcome is recorded.
        result = run_step(
            client=None,
            model="m",
            endpoint="e",
            voice="v",
            texts=TEXTS,
            offered_rps=20.0,
            duration_s=0.5,
            max_workers=8,
            arrival=ArrivalProcess.FIXED,
            monitor_interval_s=0.05,
            invoke=FakeServer(service_time_s=0.01, slots=8, fail_with=InvokeOutcome.SATURATED_503),
        )
        outcomes = {e.outcome for e in result.events}
        assert outcomes == {InvokeOutcome.SATURATED_503.value}
        assert len(result.events) == result.scheduled_count

    def test_deadline_is_measured_from_scheduled_arrival(self) -> None:
        # A user's clock starts when they speak, not when a worker frees up, so
        # dispatch delay must eat into the deadline rather than hide inside it.
        seen: list[float] = []

        def spy(client, endpoint, text, voice, *, deadline_ts=None) -> InvokeResult:
            seen.append(deadline_ts)
            now = time.time()
            return InvokeResult(
                outcome=InvokeOutcome.OK,
                dispatch_ts=now,
                end_ts=now,
                latency_ms=1.0,
                chars=len(text),
                audio_duration_s=0.1,
            )

        result = run_step(
            client=None,
            model="m",
            endpoint="e",
            voice="v",
            texts=TEXTS,
            offered_rps=10.0,
            duration_s=0.4,
            max_workers=4,
            arrival=ArrivalProcess.FIXED,
            request_deadline_s=60.0,
            monitor_interval_s=0.05,
            invoke=spy,
        )
        assert seen
        for event, deadline in zip(result.events, seen, strict=False):
            assert deadline == pytest.approx(event.scheduled_ts + 60.0, abs=1e-6)

    def test_events_stream_to_the_sink_as_they_complete(self) -> None:
        sink_calls: list[LoadEvent] = []
        result = run_step(
            client=None,
            model="m",
            endpoint="e",
            voice="v",
            texts=TEXTS,
            offered_rps=20.0,
            duration_s=0.4,
            max_workers=8,
            arrival=ArrivalProcess.FIXED,
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
            offered_rps=20.0,
            duration_s=0.3,
            max_workers=8,
            arrival=ArrivalProcess.FIXED,
            monitor_interval_s=0.05,
            event_sink=explode,
            invoke=FakeServer(service_time_s=0.01, slots=8),
        )
        assert len(result.events) == result.scheduled_count

    def test_abort_event_stops_dispatch_early(self) -> None:
        abort = threading.Event()

        def stop_after_five(client, endpoint, text, voice, *, deadline_ts=None) -> InvokeResult:
            now = time.time()
            if not abort.is_set():
                abort.set()
            return InvokeResult(
                outcome=InvokeOutcome.OK, dispatch_ts=now, end_ts=now, latency_ms=1.0
            )

        result = run_step(
            client=None,
            model="m",
            endpoint="e",
            voice="v",
            texts=TEXTS,
            offered_rps=100.0,
            duration_s=2.0,
            max_workers=8,
            arrival=ArrivalProcess.FIXED,
            monitor_interval_s=0.05,
            stop_event=abort,
            invoke=stop_after_five,
        )
        assert len(result.events) < result.scheduled_count

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
                offered_rps=10.0,
                duration_s=0.2,
                max_workers=4,
                step_index=step,
                arrival=ArrivalProcess.FIXED,
                monitor_interval_s=0.05,
                stop_event=abort,
                invoke=FakeServer(service_time_s=0.005, slots=4),
            )
            assert not abort.is_set()
            assert len(result.events) == result.scheduled_count

    def test_monitor_samples_instance_count(self) -> None:
        result = run_step(
            client=None,
            model="m",
            endpoint="e",
            voice="v",
            texts=TEXTS,
            offered_rps=10.0,
            duration_s=0.3,
            max_workers=4,
            arrival=ArrivalProcess.FIXED,
            monitor_interval_s=0.05,
            instance_count_fetch=lambda: 1,
            invoke=FakeServer(service_time_s=0.01, slots=4),
        )
        assert result.samples
        assert result.instance_counts == (1,)
        assert not result.capacity_changed

    def test_capacity_change_mid_step_is_detected(self) -> None:
        # The mid-run tripwire: suspension is not proof, so a fleet that grows
        # anyway must invalidate the step rather than inflate C_max.
        counts = iter([1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])

        def fetch() -> int:
            return next(counts, 2)

        result = run_step(
            client=None,
            model="m",
            endpoint="e",
            voice="v",
            texts=TEXTS,
            offered_rps=10.0,
            duration_s=0.5,
            max_workers=4,
            arrival=ArrivalProcess.FIXED,
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
            offered_rps=10.0,
            duration_s=0.4,
            max_workers=4,
            arrival=ArrivalProcess.FIXED,
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
            offered_rps=40.0,
            duration_s=0.5,
            max_workers=8,
            arrival=ArrivalProcess.FIXED,
            monitor_interval_s=0.02,
            instance_count_fetch=counting,
            instance_count_ttl_s=10.0,
            invoke=FakeServer(service_time_s=0.005, slots=8),
        )
        # One fetch for the whole step, not one per event or per sample.
        assert calls["n"] == 1

    def test_in_flight_never_exceeds_max_workers(self) -> None:
        server = FakeServer(service_time_s=0.05, slots=32)
        run_step(
            client=None,
            model="m",
            endpoint="e",
            voice="v",
            texts=TEXTS,
            offered_rps=200.0,
            duration_s=0.6,
            max_workers=4,
            arrival=ArrivalProcess.FIXED,
            monitor_interval_s=0.02,
            invoke=server,
        )
        assert server.max_observed_concurrency <= 4

    def test_texts_cycle_so_steps_are_comparable(self) -> None:
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
            offered_rps=20.0,
            duration_s=0.4,
            max_workers=8,
            arrival=ArrivalProcess.FIXED,
            monitor_interval_s=0.05,
            invoke=spy,
        )
        assert seen[: len(TEXTS)] == TEXTS

    def test_rejects_zero_workers(self) -> None:
        with pytest.raises(ValueError, match="max_workers must be >= 1"):
            run_step(
                client=None,
                model="m",
                endpoint="e",
                voice="v",
                texts=TEXTS,
                offered_rps=1.0,
                duration_s=1.0,
                max_workers=0,
            )

    def test_rejects_empty_text_pool(self) -> None:
        with pytest.raises(ValueError, match="texts must not be empty"):
            run_step(
                client=None,
                model="m",
                endpoint="e",
                voice="v",
                texts=[],
                offered_rps=1.0,
                duration_s=1.0,
                max_workers=1,
            )

    def test_schedule_adherence_under_an_injected_clock(self) -> None:
        # A 240s step, instantly: the clock is injected precisely so the
        # scheduling logic is testable without waiting for it.
        virtual = {"t": 0.0}
        real_sleeps: list[float] = []

        def fake_sleep(seconds: float) -> None:
            real_sleeps.append(seconds)
            virtual["t"] += seconds

        clock = Clock(
            monotonic=lambda: virtual["t"],
            sleep=fake_sleep,
            time=lambda: 1_700_000_000.0 + virtual["t"],
        )

        def instant(client, endpoint, text, voice, *, deadline_ts=None) -> InvokeResult:
            now = clock.time()
            return InvokeResult(
                outcome=InvokeOutcome.OK, dispatch_ts=now, end_ts=now, latency_ms=0.0
            )

        started = time.monotonic()
        result = run_step(
            client=None,
            model="m",
            endpoint="e",
            voice="v",
            texts=TEXTS,
            offered_rps=4.0,
            duration_s=240.0,
            max_workers=8,
            arrival=ArrivalProcess.FIXED,
            monitor_interval_s=1.0,
            clock=clock,
            invoke=instant,
        )
        assert time.monotonic() - started < 10.0
        assert result.scheduled_count == 960
        # Each dispatch waited for its own absolute slot, so total virtual time
        # equals the step duration rather than accumulating drift.
        assert sum(real_sleeps) == pytest.approx(239.75, abs=0.01)


class TestSummarizeWindow:
    def test_achieved_rps_counts_completions_in_the_window(self) -> None:
        events = [_event(seq=i, end_ts=float(i)) for i in range(10)]
        stats = summarize_window(_step(events, offered_rps=1.0), start_ts=0.0, end_ts=10.0)
        assert stats.completed == 10
        assert stats.achieved_rps == pytest.approx(1.0)

    def test_events_outside_the_window_are_excluded(self) -> None:
        # Warm-up discard depends on this: only the measure window counts.
        events = [_event(seq=i, end_ts=float(i)) for i in range(10)]
        stats = summarize_window(_step(events), start_ts=5.0, end_ts=10.0)
        assert stats.completed == 5

    def test_saturation_is_achieved_below_offered(self) -> None:
        events = [_event(seq=i, end_ts=float(i) / 10.0) for i in range(5)]
        stats = summarize_window(_step(events, offered_rps=12.0), start_ts=0.0, end_ts=1.0)
        assert stats.achieved_rps == pytest.approx(5.0)
        assert stats.saturated

    def test_keeping_up_is_not_saturation(self) -> None:
        events = [_event(seq=i, end_ts=i / 10.0) for i in range(10)]
        stats = summarize_window(_step(events, offered_rps=10.0), start_ts=0.0, end_ts=1.0)
        assert not stats.saturated

    def test_saturation_ratio_tolerates_the_window_edge(self) -> None:
        # A step always ends with requests in flight, so demanding achieved ==
        # offered would report saturation on every healthy step.
        assert SATURATION_RATIO == 0.95
        events = [_event(seq=i, end_ts=i / 10.0) for i in range(10)]
        stats = summarize_window(_step(events, offered_rps=10.4), start_ts=0.0, end_ts=1.0)
        assert not stats.saturated

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

    def test_skipped_dispatches_are_counted_by_scheduled_time(self) -> None:
        # They have no completion time, and dropping them would break
        # reconciliation against the schedule.
        events = [
            _event(seq=0, end_ts=1.0),
            _event(
                seq=1,
                scheduled_ts=2.0,
                end_ts=None,
                latency_ms=None,
                ttfab_ms=None,
                outcome=InvokeOutcome.DISPATCH_SKIPPED,
            ),
        ]
        stats = summarize_window(_step(events), start_ts=0.0, end_ts=5.0)
        assert stats.outcome_counts[InvokeOutcome.DISPATCH_SKIPPED.value] == 1
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

    def test_concurrency_slope_is_zero_for_a_settled_step(self) -> None:
        samples = [ConcurrencySample(ts=float(i), in_flight=4, instance_count=1) for i in range(10)]
        stats = summarize_window(_step([], samples), start_ts=0.0, end_ts=10.0)
        assert stats.concurrency_slope_per_s == pytest.approx(0.0, abs=1e-9)
        assert stats.settled

    def test_growing_concurrency_is_not_settled(self) -> None:
        samples = [ConcurrencySample(ts=float(i), in_flight=i, instance_count=1) for i in range(10)]
        stats = summarize_window(_step([], samples), start_ts=0.0, end_ts=10.0)
        assert stats.concurrency_slope_per_s == pytest.approx(1.0)
        assert not stats.settled

    def test_capacity_change_makes_a_window_unusable_for_the_knee(self) -> None:
        samples = [
            ConcurrencySample(ts=0.0, in_flight=1, instance_count=1),
            ConcurrencySample(ts=1.0, in_flight=1, instance_count=2),
        ]
        events = [_event(seq=0, end_ts=0.5)]
        stats = summarize_window(_step(events, samples), start_ts=0.0, end_ts=2.0)
        assert stats.capacity_changed
        assert not stats.usable_for_knee

    def test_empty_window_is_unusable_rather_than_a_zero_knee(self) -> None:
        stats = summarize_window(_step([]), start_ts=0.0, end_ts=1.0)
        assert stats.completed == 0
        assert not stats.usable_for_knee

    def test_chars_per_hour_feeds_the_cost_model(self) -> None:
        events = [_event(seq=i, end_ts=float(i), chars=100) for i in range(10)]
        stats = summarize_window(_step(events), start_ts=0.0, end_ts=10.0)
        assert stats.chars_per_hour == pytest.approx(100 * 10 / 10 * 3600)

    def test_rejects_inverted_window(self) -> None:
        with pytest.raises(ValueError, match="window must be non-empty"):
            summarize_window(_step([]), start_ts=5.0, end_ts=5.0)


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
                offered_rps=20.0,
                duration_s=0.3,
                max_workers=8,
                arrival=ArrivalProcess.FIXED,
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
        # Two steps must synthesize the same characters, or a step could show a
        # worse knee merely for having drawn longer texts.
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
    def test_uses_monotonic_for_scheduling_and_wall_time_for_stamps(self) -> None:
        # Monotonic scheduling survives an NTP step; wall-clock stamps are what
        # correlate events with CloudWatch datapoints and container logs.
        assert SYSTEM_CLOCK.monotonic() < SYSTEM_CLOCK.time()
