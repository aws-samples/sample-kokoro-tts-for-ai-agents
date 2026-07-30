"""Open-loop load generator: arrival rate is set by us, never by the server.

This is the technical core of the capacity work, and the one place where the
existing benchmark is not merely incomplete but wrong. ``measure_scalability``
(``scalability.py:68``) runs N threads that each send the next request only
after the previous one returns, so offered load is ``N / mean_latency`` — when
the server slows down, the client sends *slower*. That is coordinated omission:
the queueing delay that a real user would experience never gets generated, so
the knee cannot be found. On Kokoro it produces a concrete wrong answer
(throughput pins at ~8.3 req/s for every level, the plateau break trips, and it
reports a concurrency of 4 for a model whose capacity is 1).

Here the schedule is computed up front from ``t0`` and followed regardless of
what the server does:

* **Absolute timestamps**, not accumulated sleeps, so a slow step cannot make
  later arrivals drift late.
* **Poisson inter-arrivals by default.** Fixed-interval arrival is the best case
  and understates queueing; real traffic clumps.
* **The dispatcher never blocks.** With no worker free it records
  ``dispatch_skipped`` and moves on. Blocking there would be the same
  self-throttling in a new place, and a skipped dispatch is a statement about
  the *client*, so it must be visible rather than silently absorbed.
* **A 1 Hz monitor** samples in-flight count and instance count. A fleet that
  grows mid-step invalidates a per-instance measurement, so the change is
  recorded on the step rather than averaged into it.

Every scheduled arrival produces exactly one :class:`LoadEvent`, so a step's
totals always reconcile: ``len(events) == len(schedule)``.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, TextIO

import numpy as np
from botocore.client import BaseClient
from loguru import logger

from tts_bench.invoke import InvokeOutcome, InvokeResult, invoke_stream

#: In-flight sampling interval. Fast enough to see a queue build inside a 60s
#: measure window, slow enough that the monitor thread costs nothing.
DEFAULT_MONITOR_INTERVAL_S = 1.0

#: How long a cached ``CurrentInstanceCount`` is reused. SageMaker will not
#: change capacity faster than this, and ``describe_endpoint`` is rate-limited,
#: so polling it per sample would be both wasteful and throttle-prone.
DEFAULT_INSTANCE_COUNT_TTL_S = 10.0

#: Client-side deadline per request, at SageMaker's hard invocation ceiling.
DEFAULT_REQUEST_DEADLINE_S = 60.0

#: Achieved/offered below this means the server is not keeping up. Not 1.0:
#: a step always ends with requests still in flight, so a small shortfall is
#: an artifact of the window, not of saturation.
SATURATION_RATIO = 0.95


class ArrivalProcess(StrEnum):
    """How inter-arrival times are drawn."""

    POISSON = "poisson"
    """Exponential gaps. The default: matches independent arrivals and, unlike
    fixed spacing, produces the bursts that actually build a queue."""

    FIXED = "fixed"
    """Uniform spacing. Useful for a reproducible floor, but it flatters the
    server — never use it alone to justify a capacity number."""


@dataclass(frozen=True, slots=True)
class Clock:
    """Injectable time source.

    Exists so the schedule-adherence tests can run without real waiting. The
    load generator reads *both* a monotonic clock (for scheduling, immune to NTP
    steps) and wall time (for event timestamps that correlate with CloudWatch).
    """

    monotonic: Callable[[], float]
    sleep: Callable[[float], None]
    time: Callable[[], float]


SYSTEM_CLOCK = Clock(monotonic=time.monotonic, sleep=time.sleep, time=time.time)


def arrival_offsets(
    rate_rps: float,
    duration_s: float,
    *,
    process: ArrivalProcess | str = ArrivalProcess.POISSON,
    seed: int | None = None,
) -> list[float]:
    """Arrival times as offsets from ``t0``, in seconds, ascending.

    Args:
        rate_rps: Target arrival rate. The schedule's *mean* rate for Poisson.
        duration_s: Schedule length. Arrivals at or beyond this are dropped.
        process: :class:`ArrivalProcess` member.
        seed: RNG seed. Supply one — an unseeded Poisson schedule makes two runs
            of the same step incomparable.

    Returns:
        Offsets in ``[0, duration_s)``. Empty when ``rate_rps`` is 0.

    Raises:
        ValueError: If ``rate_rps`` or ``duration_s`` is negative.
    """
    if rate_rps < 0:
        raise ValueError(f"rate_rps must be non-negative, got {rate_rps}")
    if duration_s < 0:
        raise ValueError(f"duration_s must be non-negative, got {duration_s}")
    if rate_rps == 0 or duration_s == 0:
        return []

    process = ArrivalProcess(process)
    if process is ArrivalProcess.FIXED:
        gap = 1.0 / rate_rps
        count = int(duration_s / gap)
        return [i * gap for i in range(count)]

    rng = np.random.default_rng(seed)
    offsets: list[float] = []
    t = 0.0
    # Draw in blocks: one exponential per call would dominate the loop at high
    # rates, and the schedule is built before any load is applied.
    block = max(16, int(rate_rps * duration_s * 1.2))
    while True:
        for gap in rng.exponential(1.0 / rate_rps, size=block):
            t += float(gap)
            if t >= duration_s:
                return offsets
            offsets.append(t)


@dataclass(frozen=True, slots=True)
class LoadEvent:
    """One scheduled arrival, whatever became of it.

    Serialized as one JSONL line. Written as events complete rather than at the
    end of the step, so a run killed mid-flight keeps everything it measured.
    """

    run_id: str
    step_index: int
    seq: int
    offered_rps: float
    model: str
    endpoint: str
    scheduled_ts: float
    dispatch_ts: float | None
    first_byte_ts: float | None
    end_ts: float | None
    dispatch_delay_ms: float | None
    ttfab_ms: float | None
    latency_ms: float | None
    outcome: str
    http_status: int | None
    error_class: str | None
    error_message: str | None
    chars: int
    audio_bytes: int
    audio_duration_s: float
    rtf: float | None
    in_flight_at_dispatch: int
    instance_count: int | None
    sample_id: str | None = None

    @property
    def ok(self) -> bool:
        return self.outcome == InvokeOutcome.OK.value

    def to_json(self) -> str:
        return json.dumps(asdict(self))


@dataclass(frozen=True, slots=True)
class ConcurrencySample:
    """One monitor tick.

    ``in_flight`` is measured client-side. It is cross-checked against AWS's
    ``ConcurrentRequestsPerModel`` afterwards: if ours is lower, the client is
    the bottleneck and the step says nothing about the server.
    """

    ts: float
    in_flight: int
    instance_count: int | None


@dataclass(slots=True)
class StepResult:
    """Everything one constant-rate step produced.

    Deliberately raw. Windowing and knee-fitting are the caller's job
    (``cmax.py``), because the warm-up to discard depends on the model.
    """

    run_id: str
    step_index: int
    offered_rps: float
    model: str
    endpoint: str
    arrival_process: str
    seed: int | None
    started_ts: float
    ended_ts: float
    scheduled_count: int
    events: list[LoadEvent] = field(default_factory=list)
    samples: list[ConcurrencySample] = field(default_factory=list)

    @property
    def instance_counts(self) -> tuple[int, ...]:
        """Distinct instance counts seen during the step, in order of first sight."""
        seen: list[int] = []
        for sample in self.samples:
            if sample.instance_count is not None and sample.instance_count not in seen:
                seen.append(sample.instance_count)
        return tuple(seen)

    @property
    def capacity_changed(self) -> bool:
        """True if the fleet resized mid-step.

        When true the step must be excluded from knee-fitting: achieved
        throughput rose for a reason unrelated to the latency knee, so any
        ``C_max`` derived from it is really ``N x C_max``.
        """
        return len(self.instance_counts) > 1

    @property
    def dispatch_skipped(self) -> int:
        return sum(1 for e in self.events if e.outcome == InvokeOutcome.DISPATCH_SKIPPED.value)


@dataclass(frozen=True, slots=True)
class WindowStats:
    """Aggregates over one time window of a step.

    ``achieved_rps`` counts *completions* in the window, so it is directly
    comparable with ``offered_rps``; that comparison is the saturation test.
    """

    offered_rps: float
    achieved_rps: float
    window_start_ts: float
    window_end_ts: float
    completed: int
    ok: int
    outcome_counts: dict[str, int]
    ttfab_p50_ms: float | None
    ttfab_p95_ms: float | None
    ttfab_p99_ms: float | None
    latency_p50_ms: float | None
    latency_p95_ms: float | None
    latency_p99_ms: float | None
    s_mean_s: float | None
    s_p95_s: float | None
    rtf_mean: float | None
    concurrency_mean: float | None
    concurrency_p95: float | None
    concurrency_slope_per_s: float | None
    chars_per_hour: float
    capacity_changed: bool
    instance_counts: tuple[int, ...]

    @property
    def saturated(self) -> bool:
        """Server could not keep up with the offered rate."""
        if self.offered_rps <= 0:
            return False
        return self.achieved_rps < SATURATION_RATIO * self.offered_rps

    @property
    def settled(self) -> bool:
        """In-flight count is not trending upward.

        An unsettled step *is* the measurement — a queue still growing at the
        end of the window means this rate is already past capacity, whatever the
        latency percentiles happen to read.
        """
        if self.concurrency_slope_per_s is None:
            return True
        return self.concurrency_slope_per_s <= 0.05

    @property
    def usable_for_knee(self) -> bool:
        """Whether this window may inform a per-instance ``C_max``."""
        return not self.capacity_changed and self.completed > 0


def _percentile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(values, q))


def summarize_window(
    result: StepResult,
    *,
    start_ts: float,
    end_ts: float,
) -> WindowStats:
    """Aggregate the part of a step that falls inside a window.

    Requests are attributed by ``end_ts`` (completion), which is what makes
    ``achieved_rps`` a throughput. ``dispatch_skipped`` events have no
    completion time and are counted by ``scheduled_ts`` instead, so they are
    never silently dropped from the outcome table.

    Raises:
        ValueError: If the window is empty or inverted.
    """
    if end_ts <= start_ts:
        raise ValueError(f"window must be non-empty, got [{start_ts}, {end_ts}]")

    duration = end_ts - start_ts
    completed: list[LoadEvent] = []
    outcome_counts: dict[str, int] = {}

    for event in result.events:
        if event.end_ts is None:
            # Never sent: place it by when it should have arrived.
            if start_ts <= event.scheduled_ts < end_ts:
                outcome_counts[event.outcome] = outcome_counts.get(event.outcome, 0) + 1
            continue
        if start_ts <= event.end_ts < end_ts:
            completed.append(event)
            outcome_counts[event.outcome] = outcome_counts.get(event.outcome, 0) + 1

    # Percentiles over successes only. Mixing in a fast rejection would pull p95
    # *down* under overload, turning saturation into an apparent improvement;
    # the rejections are accounted separately in ``outcome_counts``.
    oks = [e for e in completed if e.ok]
    ttfabs = [e.ttfab_ms for e in oks if e.ttfab_ms is not None]
    latencies = [e.latency_ms for e in oks if e.latency_ms is not None]
    rtfs = [e.rtf for e in oks if e.rtf is not None]
    latency_p95 = _percentile(latencies, 95)

    in_window = [s for s in result.samples if start_ts <= s.ts < end_ts]
    in_flight = [float(s.in_flight) for s in in_window]
    slope: float | None = None
    if len(in_window) >= 3:
        # Least-squares trend in requests per second. Positive means the queue
        # is still growing at the end of the window.
        ts = np.array([s.ts - start_ts for s in in_window], dtype=float)
        slope = float(np.polyfit(ts, np.array(in_flight, dtype=float), 1)[0])

    return WindowStats(
        offered_rps=result.offered_rps,
        achieved_rps=len(completed) / duration,
        window_start_ts=start_ts,
        window_end_ts=end_ts,
        completed=len(completed),
        ok=len(oks),
        outcome_counts=outcome_counts,
        ttfab_p50_ms=_percentile(ttfabs, 50),
        ttfab_p95_ms=_percentile(ttfabs, 95),
        ttfab_p99_ms=_percentile(ttfabs, 99),
        latency_p50_ms=_percentile(latencies, 50),
        latency_p95_ms=latency_p95,
        latency_p99_ms=_percentile(latencies, 99),
        s_mean_s=(sum(latencies) / len(latencies) / 1000.0) if latencies else None,
        s_p95_s=(latency_p95 / 1000.0) if latency_p95 is not None else None,
        rtf_mean=(sum(rtfs) / len(rtfs)) if rtfs else None,
        concurrency_mean=(sum(in_flight) / len(in_flight)) if in_flight else None,
        concurrency_p95=_percentile(in_flight, 95),
        concurrency_slope_per_s=slope,
        chars_per_hour=sum(e.chars for e in oks) / duration * 3600.0,
        capacity_changed=result.capacity_changed,
        instance_counts=result.instance_counts,
    )


class JsonlWriter:
    """Thread-safe append-only JSONL sink.

    Flushes every event: a benchmark that dies at the interesting moment must
    still have written the interesting moment.
    """

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._handle: TextIO | None = None

    def __enter__(self) -> JsonlWriter:
        self._handle = self._path.open("a", encoding="utf-8")
        return self

    def __exit__(self, *exc: Any) -> bool:
        if self._handle is not None:
            self._handle.close()
            self._handle = None
        return False

    def __call__(self, event: LoadEvent) -> None:
        if self._handle is None:
            raise RuntimeError("JsonlWriter used outside its context manager")
        line = event.to_json()
        with self._lock:
            self._handle.write(line + "\n")
            self._handle.flush()


class _InFlight:
    """Request counter, incremented at dispatch and decremented at completion.

    Only the dispatcher increments and only workers decrement, so a value read
    by the dispatcher can be stale in one direction only — too high, never too
    low. That makes the ``>= max_workers`` check conservative: it may skip a
    dispatch that would have fit, but it can never oversubscribe the pool.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._value = 0

    def acquire(self) -> int:
        """Increment and return the value *before* the increment."""
        with self._lock:
            before = self._value
            self._value += 1
            return before

    def release(self) -> None:
        with self._lock:
            self._value -= 1

    @property
    def value(self) -> int:
        with self._lock:
            return self._value


class _InstanceCountCache:
    """Caches ``CurrentInstanceCount`` so the monitor can sample it at 1 Hz.

    A miss returns the previous value rather than ``None``: a throttled
    ``describe_endpoint`` should not look like a capacity change.
    """

    def __init__(
        self,
        fetch: Callable[[], int] | None,
        *,
        clock: Clock,
        ttl_s: float = DEFAULT_INSTANCE_COUNT_TTL_S,
    ) -> None:
        self._fetch = fetch
        self._clock = clock
        self._ttl_s = ttl_s
        self._value: int | None = None
        self._fetched_at = float("-inf")

    def get(self) -> int | None:
        if self._fetch is None:
            return None
        now = self._clock.monotonic()
        if now - self._fetched_at < self._ttl_s:
            return self._value
        try:
            self._value = int(self._fetch())
        except Exception as exc:  # noqa: BLE001 - never let telemetry kill a run
            logger.warning("Could not read instance count: {}", exc)
        self._fetched_at = now
        return self._value


def make_instance_count_fetcher(
    sagemaker: BaseClient,
    endpoint_name: str,
    variant: str = "primary",
) -> Callable[[], int]:
    """Build the ``CurrentInstanceCount`` reader for the mid-run tripwire."""

    def fetch() -> int:
        described = sagemaker.describe_endpoint(EndpointName=endpoint_name)
        for summary in described.get("ProductionVariants", []):
            if summary.get("VariantName") == variant:
                return int(summary.get("CurrentInstanceCount", 0))
        return 0

    return fetch


def build_text_pool(
    texts: Iterable[str],
    *,
    seed: int | None = None,
) -> list[str]:
    """Shuffle a text pool once, deterministically.

    Every step then walks the same order from index 0, so two steps of equal
    length synthesize the *same* characters. Without that, a step could show a
    worse knee merely for having drawn longer texts.

    Raises:
        ValueError: If the pool is empty.
    """
    pool = list(texts)
    if not pool:
        raise ValueError("text pool is empty")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(pool))
    return [pool[i] for i in order]


def run_step(
    # Not `BaseClient`: with `invoke=invoke_bidi` this is a
    # `SageMakerRuntimeHTTP2Client`, which shares no base class with botocore's.
    # The driver never calls a method on it — it only hands it to `invoke` — so
    # the transport and its client stay a matched pair chosen by the caller.
    client: Any,
    *,
    model: str,
    endpoint: str,
    voice: str,
    texts: Sequence[str],
    offered_rps: float,
    duration_s: float,
    max_workers: int,
    step_index: int = 0,
    run_id: str | None = None,
    arrival: ArrivalProcess | str = ArrivalProcess.POISSON,
    seed: int | None = None,
    request_deadline_s: float = DEFAULT_REQUEST_DEADLINE_S,
    monitor_interval_s: float = DEFAULT_MONITOR_INTERVAL_S,
    instance_count_fetch: Callable[[], int] | None = None,
    instance_count_ttl_s: float = DEFAULT_INSTANCE_COUNT_TTL_S,
    event_sink: Callable[[LoadEvent], None] | None = None,
    stop_event: threading.Event | None = None,
    clock: Clock = SYSTEM_CLOCK,
    invoke: Callable[..., InvokeResult] = invoke_stream,
) -> StepResult:
    """Drive one constant-rate step and return everything it produced.

    Args:
        max_workers: Concurrency cap. Must exceed the highest in-flight count
            the step will reach, or the *client* becomes the bottleneck — which
            shows up as ``dispatch_skipped``, not as saturation.
        offered_rps: Arrival rate to hold regardless of server behaviour.
        duration_s: Step length. The caller measures a sub-window of it.
        request_deadline_s: Client-side per-request deadline, measured from the
            **scheduled** arrival, not from dispatch: that is what a user
            experiences, and it means dispatch delay eats into the budget
            instead of being hidden.
        instance_count_fetch: Optional reader for the mid-run capacity tripwire.
        invoke: The transport. :func:`invoke_stream` or
            :func:`tts_bench.bidi.invoke_bidi` — see
            :func:`tts_bench.bidi.invoke_for` — and injected by tests. ``client``
            is passed through untouched, so each transport receives the client
            type it built.

    Returns:
        A :class:`StepResult` with exactly one event per scheduled arrival.

    Raises:
        ValueError: If ``max_workers`` < 1 or ``texts`` is empty.
    """
    if max_workers < 1:
        raise ValueError(f"max_workers must be >= 1, got {max_workers}")
    if not texts:
        raise ValueError("texts must not be empty")

    run_id = run_id or uuid.uuid4().hex[:12]
    offsets = arrival_offsets(offered_rps, duration_s, process=arrival, seed=seed)
    # The caller's stop_event aborts the run; this step's own event also stops
    # the monitor at the end of the step. Setting the caller's event here would
    # abort every *later* step in a ladder as a side effect of finishing this one.
    abort = stop_event or threading.Event()
    done = threading.Event()
    in_flight = _InFlight()
    instance_counts = _InstanceCountCache(
        instance_count_fetch, clock=clock, ttl_s=instance_count_ttl_s
    )

    events: list[LoadEvent] = []
    events_lock = threading.Lock()
    samples: list[ConcurrencySample] = []
    samples_lock = threading.Lock()

    t0_mono = clock.monotonic()
    t0_wall = clock.time()

    def _wall_for(offset: float) -> float:
        """Wall-clock timestamp for a monotonic offset, without re-reading time."""
        return t0_wall + offset

    def _record(event: LoadEvent) -> None:
        with events_lock:
            events.append(event)
        if event_sink is not None:
            try:
                event_sink(event)
            except Exception as exc:  # noqa: BLE001 - a bad sink must not lose the run
                logger.error("event_sink failed for seq={}: {}", event.seq, exc)

    def _monitor() -> None:
        # Sampled on its own thread so a slow describe_endpoint delays telemetry
        # rather than the arrival schedule.
        #
        # Waits on the event in real time rather than through `clock`: sampling
        # is inherently wall-clock, and waiting on the event also means shutdown
        # is immediate instead of one interval late. Only the arrival schedule
        # goes through the injectable clock.
        while not (done.is_set() or abort.is_set()):
            with samples_lock:
                samples.append(
                    ConcurrencySample(
                        ts=clock.time(),
                        in_flight=in_flight.value,
                        instance_count=instance_counts.get(),
                    )
                )
            done.wait(monitor_interval_s)

    def _worker(seq: int, text: str, scheduled_wall: float, before: int) -> None:
        try:
            dispatch_wall = clock.time()
            result = invoke(
                client,
                endpoint,
                text,
                voice,
                deadline_ts=scheduled_wall + request_deadline_s,
            )
            _record(
                LoadEvent(
                    run_id=run_id,
                    step_index=step_index,
                    seq=seq,
                    offered_rps=offered_rps,
                    model=model,
                    endpoint=endpoint,
                    scheduled_ts=scheduled_wall,
                    dispatch_ts=result.dispatch_ts,
                    first_byte_ts=result.first_byte_ts,
                    end_ts=result.end_ts,
                    dispatch_delay_ms=(dispatch_wall - scheduled_wall) * 1000.0,
                    ttfab_ms=result.ttfab_ms,
                    latency_ms=result.latency_ms,
                    outcome=result.outcome.value,
                    http_status=result.http_status,
                    error_class=result.error_class,
                    error_message=result.error_message,
                    chars=result.chars,
                    audio_bytes=result.audio_bytes,
                    audio_duration_s=result.audio_duration_s,
                    rtf=result.rtf,
                    in_flight_at_dispatch=before,
                    instance_count=instance_counts.get(),
                )
            )
        finally:
            in_flight.release()

    monitor = threading.Thread(target=_monitor, name=f"loadgen-monitor-{step_index}", daemon=True)
    monitor.start()

    logger.info(
        "Step {}: {} arrivals at {:.2f} rps over {:.0f}s ({} arrivals)",
        step_index,
        len(offsets),
        offered_rps,
        duration_s,
        arrival,
    )

    try:
        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="loadgen") as pool:
            for seq, offset in enumerate(offsets):
                if abort.is_set():
                    break
                # Sleep against the absolute target, so a late dispatch does not
                # push every later arrival later too.
                remaining = (t0_mono + offset) - clock.monotonic()
                if remaining > 0:
                    clock.sleep(remaining)

                scheduled_wall = _wall_for(offset)
                text = texts[seq % len(texts)]

                if in_flight.value >= max_workers:
                    # Never block here: waiting for a worker is exactly the
                    # response-gated dispatch this module exists to avoid.
                    _record(
                        LoadEvent(
                            run_id=run_id,
                            step_index=step_index,
                            seq=seq,
                            offered_rps=offered_rps,
                            model=model,
                            endpoint=endpoint,
                            scheduled_ts=scheduled_wall,
                            dispatch_ts=None,
                            first_byte_ts=None,
                            end_ts=None,
                            dispatch_delay_ms=None,
                            ttfab_ms=None,
                            latency_ms=None,
                            outcome=InvokeOutcome.DISPATCH_SKIPPED.value,
                            http_status=None,
                            error_class=None,
                            error_message=f"no free worker of {max_workers}",
                            chars=len(text),
                            audio_bytes=0,
                            audio_duration_s=0.0,
                            rtf=None,
                            in_flight_at_dispatch=in_flight.value,
                            instance_count=instance_counts.get(),
                        )
                    )
                    continue

                before = in_flight.acquire()
                pool.submit(_worker, seq, text, scheduled_wall, before)
            # Exiting the `with` joins the pool: in-flight requests are allowed
            # to finish so their latencies are not truncated.
    finally:
        done.set()
        monitor.join(timeout=max(monitor_interval_s * 3, 1.0))

    ended_wall = clock.time()
    result = StepResult(
        run_id=run_id,
        step_index=step_index,
        offered_rps=offered_rps,
        model=model,
        endpoint=endpoint,
        arrival_process=str(ArrivalProcess(arrival)),
        seed=seed,
        started_ts=t0_wall,
        ended_ts=ended_wall,
        scheduled_count=len(offsets),
        events=sorted(events, key=lambda e: e.seq),
        samples=list(samples),
    )

    if result.capacity_changed:
        logger.warning(
            "Step {}: instance count changed mid-run {} — excluded from knee fitting",
            step_index,
            result.instance_counts,
        )
    if result.dispatch_skipped:
        logger.warning(
            "Step {}: {}/{} dispatches skipped (max_workers={}); the client, not the "
            "server, was the limit",
            step_index,
            result.dispatch_skipped,
            result.scheduled_count,
            max_workers,
        )
    return result
