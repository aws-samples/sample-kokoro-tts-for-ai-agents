"""Closed-loop load generator: concurrency is set by us, and held exactly.

``N`` workers each issue their next request only when their previous one completes,
so outstanding requests — queued **plus** executing — stay pinned at ``N``. That is
the user's own definition of the quantity being measured: "at 5 there should always
be Q (queue) + E (executing) = 5".

**This is deliberately the design the module was originally written to avoid, and the
reason is that the question changed.** The previous version drove an open-loop Poisson
arrival schedule and argued, correctly, that response-gated dispatch is *coordinated
omission*: when the server slows down the client sends slower, so the queueing delay a
real user would feel never gets generated. That argument is about measuring **throughput
at a given offered rate**, where the arrival rate is the independent variable and
concurrency is the outcome. It does not apply when the roles are reversed.

``Q_max`` is defined as a concurrency: the largest number of simultaneously outstanding
requests that still meets the SLO. Concurrency is therefore the independent variable, and
closed-loop sets it *directly and exactly*. Open-loop can only reach a concurrency by
choosing a rate and hoping — via ``lambda = C / S``, a conversion that needs a service
time we are also trying to measure, and which every units defect found on this endpoint
came through. Set ``N`` and count, and those defects stop existing rather than get fixed.

Coordinated omission has not been reintroduced, because nothing here is being read as a
rate promise: a step reports the latency *at* ``N`` outstanding, and ``achieved_rps`` is
an observed consequence of that latency, never a target that the server failed to meet.

What the design does cost, and how each cost is recovered:

* **In-flight count can no longer detect a growing queue** — it is pinned at ``N`` by
  construction, so a slope on it would always read zero. The signal moves to *latency*:
  :attr:`WindowStats.settled` fits TTFAB against time and reports whether the step had
  reached steady state. A check that silently always passes is worse than no check.
* **There is no ``dispatch_skipped``** to reveal a client-side bottleneck. The
  replacement is stronger: mean in-flight *should* equal ``N``, so any shortfall is
  client turnaround, measured directly by :attr:`WindowStats.concurrency_shortfall`.
* **Saturation is no longer a rate comparison.** A closed-loop driver cannot outrun the
  server, so "achieved below offered" has no meaning here. What still means saturation is
  the server *refusing* work, so :attr:`WindowStats.saturated` reads rejection outcomes.

A **1 Hz monitor** samples in-flight and instance count. A fleet that grows mid-step
invalidates a per-instance measurement, so the change is recorded on the step rather
than averaged into it.

Every dispatched request produces exactly one :class:`LoadEvent`, including one whose
transport raised — a worker that died silently would drop concurrency below ``N`` and
quietly invalidate the only thing this driver guarantees.
"""

from __future__ import annotations

import itertools
import json
import threading
import time
import uuid
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, field
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

#: Outcomes where the server refused the work rather than performed it. These are
#: the closed-loop saturation signal: a driver that cannot outrun the server can
#: still be told by the server to go away.
REJECTION_OUTCOMES = frozenset(
    {
        InvokeOutcome.SATURATED_503.value,
        InvokeOutcome.STALE_408.value,
        InvokeOutcome.THROTTLED_429.value,
    }
)

#: Fraction of completions that must be rejections before a window reads as
#: saturated. Not zero: a single 429 in a ten-minute step is SageMaker's own
#: throttling and should not discard the step. Not high either — a ``Q_max``
#: ladder is supposed to run against an *unbounded* queue, so sustained
#: rejections mean the run's preconditions were violated and the latency
#: percentiles describe something other than what was intended.
REJECTION_RATIO = 0.01

#: Largest fitted TTFAB drift across a window, as a fraction of that window's p95,
#: that still counts as steady state. 25% is loose enough to tolerate percentile
#: noise on a short hold and tight enough that a queue genuinely still filling —
#: which on a serial server grows latency without bound — fails it.
LATENCY_DRIFT_RATIO = 0.25

#: How far mean in-flight may fall below ``N`` before the *client* is the limit.
#: Each worker spends a little time between completing one request and dispatching
#: the next, so the shortfall is never exactly zero; at 5% of ``N`` it is the
#: closed-loop replacement for ``dispatch_skipped``.
CONCURRENCY_SHORTFALL_RATIO = 0.05


@dataclass(frozen=True, slots=True)
class Clock:
    """Injectable time source.

    Exists so the driver's tests can run without real waiting. The load generator
    reads *both* a monotonic clock (for the step deadline, immune to NTP steps) and
    wall time (for event timestamps that correlate with CloudWatch).
    """

    monotonic: Callable[[], float]
    sleep: Callable[[float], None]
    time: Callable[[], float]


SYSTEM_CLOCK = Clock(monotonic=time.monotonic, sleep=time.sleep, time=time.time)


@dataclass(frozen=True, slots=True)
class LoadEvent:
    """One dispatched request, whatever became of it.

    Serialized as one JSONL line. Written as events complete rather than at the
    end of the step, so a run killed mid-flight keeps everything it measured.
    """

    run_id: str
    step_index: int
    seq: int
    worker_index: int
    """Which worker issued it. A worker holds exactly one outstanding request at a
    time, so two events sharing a ``worker_index`` can never overlap — that is what
    makes ``Q + E = N`` true by construction rather than by sampling, and it is
    checkable after the fact from the artifact alone."""
    concurrency: int
    """The step's ``N``. Held exactly, unlike the open-loop ``offered_rps`` it
    replaced, which was a request the server could decline."""
    model: str
    endpoint: str
    dispatch_ts: float | None
    first_byte_ts: float | None
    end_ts: float | None
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
    ``ConcurrentRequestsPerModel`` afterwards: the two are the same quantity in
    different units, and the deployed threshold is compared against AWS's version.
    """

    ts: float
    in_flight: int
    instance_count: int | None


@dataclass(slots=True)
class StepResult:
    """Everything one constant-concurrency step produced.

    Deliberately raw. Windowing and ladder-fitting are the caller's job
    (``qmax.py``), because the warm-up to discard depends on the model.
    """

    run_id: str
    step_index: int
    concurrency: int
    model: str
    endpoint: str
    started_ts: float
    ended_ts: float
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

        When true the step must be excluded from the ladder: ``N`` outstanding
        requests spread over two instances is a different measurement from ``N`` on
        one, so the latency it reports belongs to no single per-instance ``Q_max``.
        """
        return len(self.instance_counts) > 1

    @property
    def worker_overlaps(self) -> int:
        """Events where one worker's request overlapped its own next one.

        Must be zero. A non-zero count means a worker dispatched before its previous
        request completed, so outstanding requests exceeded ``N`` and the step's
        concurrency is not the number it claims. Cheap to check and it falsifies the
        driver's one structural guarantee, so it is checked rather than assumed.
        """
        overlaps = 0
        by_worker: dict[int, list[LoadEvent]] = {}
        for event in self.events:
            by_worker.setdefault(event.worker_index, []).append(event)
        for events in by_worker.values():
            timed = sorted(
                (e for e in events if e.dispatch_ts is not None and e.end_ts is not None),
                key=lambda e: e.dispatch_ts or 0.0,
            )
            for earlier, later in zip(timed, timed[1:], strict=False):
                if (later.dispatch_ts or 0.0) < (earlier.end_ts or 0.0):
                    overlaps += 1
        return overlaps


@dataclass(frozen=True, slots=True)
class WindowStats:
    """Aggregates over one time window of a step.

    ``achieved_rps`` counts completions in the window. Unlike the open-loop version
    it is an *output*, not something to compare against a target: at fixed ``N`` it
    is ``N / latency``, so it is how throughput is derived from the ladder rather
    than how saturation is detected.
    """

    concurrency: int
    achieved_rps: float
    window_start_ts: float
    window_end_ts: float
    completed: int
    ok: int
    chars: int
    """Characters synthesized by the completions in the window. A raw count, kept here
    rather than turned into a rate: the planner needs chars-per-*request* to price a
    fleet in $/M chars, and a count over a count needs no units conversion — which is
    where the open-loop ``chars_per_hour`` went wrong."""
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
    concurrency_peak: int | None
    ttfab_drift_ms: float | None
    """Fitted change in TTFAB across the whole window, milliseconds. Positive means
    latency was still climbing when the window ended."""
    capacity_changed: bool
    instance_counts: tuple[int, ...]

    @property
    def rejected(self) -> int:
        """Completions where the server refused the work."""
        return sum(self.outcome_counts.get(name, 0) for name in REJECTION_OUTCOMES)

    @property
    def saturated(self) -> bool:
        """Server refused a meaningful share of the offered work.

        The closed-loop meaning of saturation. A rate comparison cannot serve here —
        the driver never offers more than the server accepts — so the signal is the
        server's own rejections. Reading ``outcome_counts`` also closes a real hole
        in the open-loop version, where a 1ms 503 counted as a completion and so
        *raised* achieved throughput: a fully-rejecting endpoint looked healthy.
        """
        if self.completed <= 0:
            return False
        return self.rejected / self.completed > REJECTION_RATIO

    @property
    def concurrency_shortfall(self) -> float | None:
        """How far mean in-flight fell below ``N``, as a fraction of ``N``.

        Zero would mean the client turned a completed request into its next dispatch
        instantly. Small values are ordinary overhead; a large one means the client
        was the bottleneck and the step measured our own latency, not the server's.
        """
        if self.concurrency_mean is None or self.concurrency <= 0:
            return None
        return max(0.0, (self.concurrency - self.concurrency_mean) / self.concurrency)

    @property
    def client_bound(self) -> bool:
        """Whether the client, not the server, limited this step."""
        shortfall = self.concurrency_shortfall
        return shortfall is not None and shortfall > CONCURRENCY_SHORTFALL_RATIO

    @property
    def settled(self) -> bool:
        """Whether latency had stopped climbing by the end of the window.

        The closed-loop replacement for an in-flight trend, which is pinned at ``N``
        here and would always read as settled. An unsettled step *is* a
        measurement — on a serial server a queue that is still filling grows latency
        without bound, so whatever p95 the window reports is a snapshot of a moving
        number rather than the SLO verdict at this concurrency.
        """
        if self.ttfab_drift_ms is None or self.ttfab_p95_ms is None or self.ttfab_p95_ms <= 0:
            return True
        return self.ttfab_drift_ms <= LATENCY_DRIFT_RATIO * self.ttfab_p95_ms

    @property
    def usable_for_ladder(self) -> bool:
        """Whether this window may inform a per-instance ``Q_max``."""
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
    ``achieved_rps`` a throughput. An event with no completion time — a transport
    that raised outright — is placed by its dispatch instead, so it is never
    silently dropped from the outcome table.

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
            placed_at = event.dispatch_ts
            if placed_at is not None and start_ts <= placed_at < end_ts:
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
    ttfab_p95 = _percentile(ttfabs, 95)

    # TTFAB against completion time, fitted in ms per second and reported as the
    # drift across the whole window: a slope is hard to judge without knowing the
    # window length, while "latency rose 900ms over this window" is not.
    drift: float | None = None
    timed = [(e.end_ts, e.ttfab_ms) for e in oks if e.end_ts is not None and e.ttfab_ms is not None]
    if len(timed) >= 3:
        xs = np.array([t - start_ts for t, _ in timed], dtype=float)
        ys = np.array([v for _, v in timed], dtype=float)
        if float(xs.max() - xs.min()) > 0:
            slope = float(np.polyfit(xs, ys, 1)[0])
            drift = slope * duration

    in_window = [s for s in result.samples if start_ts <= s.ts < end_ts]
    in_flight = [float(s.in_flight) for s in in_window]

    return WindowStats(
        concurrency=result.concurrency,
        achieved_rps=len(completed) / duration,
        window_start_ts=start_ts,
        window_end_ts=end_ts,
        completed=len(completed),
        ok=len(oks),
        # Over every completion, not just the successes: a rejected request still had
        # text attached, and the planner's chars-per-request is the mean size of what
        # the benchmark asked for rather than of what came back.
        chars=sum(e.chars for e in completed),
        outcome_counts=outcome_counts,
        ttfab_p50_ms=_percentile(ttfabs, 50),
        ttfab_p95_ms=ttfab_p95,
        ttfab_p99_ms=_percentile(ttfabs, 99),
        latency_p50_ms=_percentile(latencies, 50),
        latency_p95_ms=latency_p95,
        latency_p99_ms=_percentile(latencies, 99),
        s_mean_s=(sum(latencies) / len(latencies) / 1000.0) if latencies else None,
        s_p95_s=(latency_p95 / 1000.0) if latency_p95 is not None else None,
        rtf_mean=(sum(rtfs) / len(rtfs)) if rtfs else None,
        concurrency_mean=(sum(in_flight) / len(in_flight)) if in_flight else None,
        concurrency_peak=int(max(in_flight)) if in_flight else None,
        ttfab_drift_ms=drift,
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
    """Outstanding-request counter, incremented at dispatch and decremented at completion.

    Reads are only telemetry here — the closed loop bounds concurrency structurally,
    by giving each worker one request at a time, rather than by consulting a counter.
    That is why the count can be sampled freely without the sampling affecting what
    is dispatched.
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
    worse latency merely for having drawn longer texts.

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
    concurrency: int,
    duration_s: float,
    step_index: int = 0,
    run_id: str | None = None,
    request_deadline_s: float = DEFAULT_REQUEST_DEADLINE_S,
    monitor_interval_s: float = DEFAULT_MONITOR_INTERVAL_S,
    instance_count_fetch: Callable[[], int] | None = None,
    instance_count_ttl_s: float = DEFAULT_INSTANCE_COUNT_TTL_S,
    event_sink: Callable[[LoadEvent], None] | None = None,
    stop_event: threading.Event | None = None,
    clock: Clock = SYSTEM_CLOCK,
    invoke: Callable[..., InvokeResult] = invoke_stream,
) -> StepResult:
    """Hold ``concurrency`` requests outstanding for ``duration_s`` and return the result.

    Args:
        concurrency: ``N``. Exactly this many requests are outstanding at any moment,
            because exactly this many workers exist and each holds one. This is the
            independent variable of a ``Q_max`` ladder, which is why it is set rather
            than approached via an arrival rate.
        duration_s: How long to keep the loop running. A request already in flight
            when the deadline passes is allowed to finish rather than be truncated,
            so the wall-clock is slightly longer than this by design — a cut-off
            latency is a wrong latency, not a shorter step.
        request_deadline_s: Client-side per-request deadline, from dispatch. In a
            closed loop dispatch *is* arrival: there is no schedule to be late
            against, which is one of the things the design removes.
        instance_count_fetch: Optional reader for the mid-run capacity tripwire.
        invoke: The transport. :func:`invoke_stream` or
            :func:`tts_bench.bidi.invoke_bidi` — see
            :func:`tts_bench.bidi.invoke_for` — and injected by tests. ``client``
            is passed through untouched, so each transport receives the client
            type it built.

    Returns:
        A :class:`StepResult` with exactly one event per dispatched request.

    Raises:
        ValueError: If ``concurrency`` < 1, ``duration_s`` < 0, or ``texts`` is empty.
    """
    if concurrency < 1:
        raise ValueError(f"concurrency must be >= 1, got {concurrency}")
    if duration_s < 0:
        raise ValueError(f"duration_s must be non-negative, got {duration_s}")
    if not texts:
        raise ValueError("texts must not be empty")

    run_id = run_id or uuid.uuid4().hex[:12]
    # The caller's stop_event aborts the run; this step's own event also stops the
    # monitor at the end of the step. Setting the caller's event here would abort
    # every *later* step in a ladder as a side effect of finishing this one.
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
    # Global across workers so `seq` orders the step's requests, not one worker's.
    # itertools.count is atomic under the GIL for a single next() call.
    sequence = itertools.count()

    t0_mono = clock.monotonic()
    t0_wall = clock.time()
    deadline_mono = t0_mono + duration_s

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
        # rather than a worker's next dispatch.
        #
        # Waits on the event in real time rather than through `clock`: sampling is
        # inherently wall-clock, and waiting on the event also means shutdown is
        # immediate instead of one interval late.
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

    def _worker(worker_index: int) -> None:
        # One request at a time, for the whole step. The serial loop *is* the
        # concurrency bound: with N workers, outstanding requests cannot exceed N
        # no matter how the server behaves, so nothing has to be checked or capped.
        while not abort.is_set() and clock.monotonic() < deadline_mono:
            seq = next(sequence)
            text = texts[seq % len(texts)]
            before = in_flight.acquire()
            dispatch_wall = clock.time()
            try:
                result = invoke(
                    client,
                    endpoint,
                    text,
                    voice,
                    deadline_ts=dispatch_wall + request_deadline_s,
                )
                _record(
                    LoadEvent(
                        run_id=run_id,
                        step_index=step_index,
                        seq=seq,
                        worker_index=worker_index,
                        concurrency=concurrency,
                        model=model,
                        endpoint=endpoint,
                        dispatch_ts=result.dispatch_ts,
                        first_byte_ts=result.first_byte_ts,
                        end_ts=result.end_ts,
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
            except Exception as exc:  # noqa: BLE001 - a raising transport must not vanish
                # The transport classifies its own failures, so reaching here means
                # something outside that contract broke. Recorded rather than
                # swallowed: an unrecorded request is one the step held a worker for
                # and cannot account for, which breaks the reconciliation that makes
                # `Q + E = N` checkable from the artifact.
                logger.error("Worker {} seq={} raised: {}", worker_index, seq, exc)
                _record(
                    LoadEvent(
                        run_id=run_id,
                        step_index=step_index,
                        seq=seq,
                        worker_index=worker_index,
                        concurrency=concurrency,
                        model=model,
                        endpoint=endpoint,
                        dispatch_ts=dispatch_wall,
                        first_byte_ts=None,
                        end_ts=clock.time(),
                        ttfab_ms=None,
                        latency_ms=None,
                        outcome=InvokeOutcome.ERROR.value,
                        http_status=None,
                        error_class=type(exc).__name__,
                        error_message=str(exc),
                        chars=len(text),
                        audio_bytes=0,
                        audio_duration_s=0.0,
                        rtf=None,
                        in_flight_at_dispatch=before,
                        instance_count=instance_counts.get(),
                    )
                )
            finally:
                in_flight.release()

    monitor = threading.Thread(target=_monitor, name=f"loadgen-monitor-{step_index}", daemon=True)
    monitor.start()

    logger.info(
        "Step {}: holding {} outstanding request(s) for {:.0f}s",
        step_index,
        concurrency,
        duration_s,
    )

    # Plain threads rather than a pool: each worker runs one long serial loop, so
    # there is nothing for a pool's queue to schedule, and `N` threads is exactly
    # `N` outstanding requests with no bookkeeping in between. All N start together,
    # which is a burst the caller's warm-up window is there to discard.
    workers = [
        threading.Thread(target=_worker, args=(i,), name=f"loadgen-{step_index}-{i}", daemon=True)
        for i in range(concurrency)
    ]
    try:
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join()
    finally:
        done.set()
        monitor.join(timeout=max(monitor_interval_s * 3, 1.0))

    ended_wall = clock.time()
    result = StepResult(
        run_id=run_id,
        step_index=step_index,
        concurrency=concurrency,
        model=model,
        endpoint=endpoint,
        started_ts=t0_wall,
        ended_ts=ended_wall,
        events=sorted(events, key=lambda e: e.seq),
        samples=list(samples),
    )

    if result.capacity_changed:
        logger.warning(
            "Step {}: instance count changed mid-run {} — excluded from the ladder",
            step_index,
            result.instance_counts,
        )
    overlaps = result.worker_overlaps
    if overlaps:
        logger.error(
            "Step {}: {} overlapping request(s) on a single worker; outstanding "
            "requests exceeded the requested concurrency of {}",
            step_index,
            overlaps,
            concurrency,
        )
    return result
