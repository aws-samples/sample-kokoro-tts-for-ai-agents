"""Measure ``T_total``: the lag from forcing a scale-out to the new instance serving.

``T_total`` is one of the two measured variables in the scaling model. It sets how much
headroom the derived thresholds have to reserve — ``max_scaling_per_T_total`` is a surge
ratio *per ``T_total``*, so the whole policy is scaled by this number. A plan built on a
guessed ``T_total`` is a guess.

A single number is not actionable, so this decomposes the lag into stages, each bounded
by a timestamp from a different API:

=========================  ==========================================================
Stage                      Source
=========================  ==========================================================
``load_applied``           the probe's own start time (before the trigger)
``desired_set``            our ``UpdateEndpointWeightsAndCapacities`` call
``instance_logging``       the new log stream's ``firstEventTimestamp``
``container_start`` ...    ``=== STAGE ... ===`` markers from that stream
  ... ``ready``
``in_service``             ``DescribeEndpoint`` ``CurrentInstanceCount``
``traffic_recovered``      the probe's p95 halving — **bounded, not measured**
=========================  ==========================================================

**The clock starts at ``desired_set``, not at ``load_applied``.** The probe is warmed up
first, so how long it ran beforehand is a choice this module makes rather than lag the
fleet has to absorb. ``load_applied`` is still on the timeline because the probe has to
be *established* before the trigger for the recovery test to mean anything.

**One trigger: ``DesiredInstanceCount`` set directly, under a freeze.** The deployed
policy is suspended for the run, so no instance arrives from a second cause mid-
measurement. What that omits — the policy's own detection lag — is bounded arithmetically
instead (:data:`POLICY_LAG_BOUND_S`) and reported beside the measurement rather than
folded into it.

**Recovery is a halving, not a threshold crossing.** SageMaker does not reveal which
instance served a request, so "the new instance took its first real traffic" cannot be
read directly. It is inferred from the client side: the endpoint routes
``LEAST_OUTSTANDING_REQUESTS``, and against a serial server outstanding *is* queue depth,
so a probe held at :data:`PROBE_CONCURRENCY` splits in half when a second instance takes
traffic and its p95 halves with it. Both halves of that comparison come from the
``Q_max`` ladder, which measured p95 at both concurrencies. A probe at concurrency 1
cannot do this: p95 is already healthy before the scale-out, so any threshold inside the
SLO is met immediately and ``traffic_recovered`` collapses onto ``in_service``.

Every inferred stage carries ``bounded=True`` and the reason, because a plan is allowed
to build on a bound but not to mistake one for a measurement.

A missing stage degrades to a gap in the timeline; it never raises. Containers built
before the stage markers existed, and streams whose startup lines aged out of
CloudWatch, are both normal — and stage *order* legitimately differs by image, which is
why the timeline is sorted by observed time rather than by this module's enum.
"""

from __future__ import annotations

import threading
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import TYPE_CHECKING, Any

import numpy as np
from botocore.client import BaseClient
from loguru import logger

from tts_bench.observe import (
    DEFAULT_VARIANT,
    LogStream,
    list_log_streams,
    log_group_name,
    new_streams_since,
    stage_markers,
)

if TYPE_CHECKING:
    from shared.stages import StageMarker
    from tts_bench.loadgen import LoadEvent
    from tts_bench.types import Provenance

DEFAULT_MAX_WAIT_S = 1500.0
"""Give up waiting for a scale event. Deliberately generous: a cold container pulling a
multi-GB image can take ten minutes, and calling that a failure would hide the very
number this command exists to measure."""

DEFAULT_POLL_INTERVAL_S = 10.0
"""Capacity poll interval. Bounds the error on ``in_service`` — the fleet is seen at its
new count up to one interval late — so it is short enough not to dominate the container
stages it sits next to."""

SLOW_PROVISION_SUSPICION_S = 900.0
"""Past this, an in-flight activity is more likely blocked on instance capacity than on a
slow image pull. Set from the observed extremes: cold container starts in this repo run 5-10
minutes, while a run that never got its instance at all sat InProgress for 34. The two need
opposite responses — wait longer, versus stop waiting — so the message changes here."""

DEFAULT_RESTORE_WAIT_S = 900.0
"""How long the restore waits out an ``Updating`` endpoint. Longer than it sounds because
this is the tail of a run that already timed out: SageMaker has been observed holding
``Updating`` for half an hour while it tries to place instances, and the cost of giving up
early is a fleet left at the raised count."""

DEFAULT_SETTLE_S = 180.0
"""Probe held past the scale event, so the recovery bound has traffic on the far side of
it. Without this the run ends at ``in_service`` and ``T_total`` stops short of the thing
it is supposed to measure."""

DEFAULT_WARMUP_S = 30.0
"""Probe held *before* the trigger, so the pre-scale p95 is a steady state rather than a
cold start. The recovery test is a comparison against that steady state, so a probe that
has not settled would make the halving unreadable in whichever direction the warm-up
happened to bias it."""

PROBE_CONCURRENCY = 10
"""Concurrency the probe holds, in queued-plus-executing. Two constraints pick it, and
they are the reason it is not 1 and not ``Q_max``:

* It must be **saturating enough to observe the halving**. The endpoint routes
  ``LEAST_OUTSTANDING_REQUESTS``, so against a serial server a second instance splits the
  probe 5/5 and p95 drops toward the ladder's own c=5 value. At concurrency 1 there is
  nothing to split and p95 was already healthy, so recovery would read as instantaneous.
* It must stay **far under ``Q_max``** (measured 50 on kokoro/g5), so no request breaches
  the SLO merely to time a scale-out.

It is also a *rung on the ladder*, along with its half — see :data:`RECOVERY_RUNGS` in
``qmax`` — because both sides of the comparison have to be measured values."""

RECOVERY_WINDOW_S = 60.0
"""How long p95 must hold at the recovered level to count as recovered. One good request
proves nothing; a minute of them is a recovery."""

RECOVERY_MIN_SAMPLES = 5
"""Completions required in a recovery window, so a lull does not read as a recovery."""

RECOVERY_TOLERANCE = 1.25
"""Multiple of the ladder's half-concurrency p95 that still counts as recovered. Above 1
because the ladder measured that rung on a settled endpoint over a 60s window, while the
probe reads it seconds after a router started splitting traffic — demanding equality
would report "never recovered" for a scale-out that plainly worked."""

TRIGGER_FORCE_DESIRED = "force-desired"
"""The only trigger. ``DesiredInstanceCount`` is raised directly, under a freeze.

Driving load past the deployed policy instead was removed deliberately. It measured the
policy's detection lag, but it also let the policy add instances *during* the measurement:
the live alarm reads ``ConcurrentRequestsPerModel``/``Maximum`` against a target of 0.713,
which any probe traffic clears by a factor of ten, and proportional target tracking then
requests ``max_capacity`` in one jump. Instances arriving from two independent causes
inside one measurement is the best explanation for the 1->4 jump observed on 2026-07-31.
What the removal costs is bounded arithmetically — see :data:`POLICY_LAG_BOUND_S`."""

POLICY_LAG_BOUND_S = 60.0
"""Upper bound on the policy half this trigger skips, from the deployed policy's own
configuration rather than from a measurement: 10s metric period x 3 evaluation periods,
plus the 30s scale-out cooldown. Reported beside ``t_total_s`` rather than added to it —
a bound and a measurement do not belong in the same number."""


class TimelineStage(StrEnum):
    """Stages of the scaling lag, in the order they normally occur.

    Distinct from :class:`shared.stages.Stage`, which names *container* startup stages
    only. This enum spans the whole lag, of which those are the middle third.
    """

    LOAD_APPLIED = "load_applied"
    """The probe began holding :data:`PROBE_CONCURRENCY` outstanding requests.

    Before the trigger, not at it: the probe has to be established for the recovery
    comparison to have a pre-scale steady state. So this is *not* where the clock starts —
    see :attr:`TTotalReport.t_total_s`, which spans from :attr:`DESIRED_SET`.
    """

    DESIRED_SET = "desired_set"
    """We raised ``DesiredInstanceCount``. The clock starts here.

    Timestamped from our own call rather than from ``DescribeScalingActivities``: with the
    policy suspended there is no scaling activity to read, and this is the exact instant
    the request left the client.
    """

    INSTANCE_LOGGING = "instance_logging"
    """The new instance's log stream opened — the first external sign it exists.

    Bounds provisioning plus image pull from outside. A container cannot time its own
    pull, so this is the only place that cost is visible.
    """

    CONTAINER_STARTED = "container_started"
    WEIGHTS_FETCHED = "weights_fetched"
    FRAMEWORK_INIT = "framework_init"
    WEIGHTS_READY = "weights_ready"
    WARMUP_DONE = "warmup_done"

    READY = "ready"
    """The container says it is accepting traffic."""

    IN_SERVICE = "in_service"
    """SageMaker reports ``CurrentInstanceCount`` at the new value."""

    TRAFFIC_RECOVERED = "traffic_recovered"
    """The probe's p95 halved into the ladder's half-concurrency value. Bounded.

    The end of ``T_total``, and the only stage that answers "is the new instance actually
    serving": ``in_service`` says SageMaker counted it, which is strictly earlier.
    """


#: Container stage-marker names mapped onto this module's timeline. A name absent from
#: this map is still reported under its own name — an unrecognized marker is data, not an
#: error, and a container image may be ahead of this module.
_CONTAINER_STAGE_MAP: dict[str, TimelineStage] = {
    "container_start": TimelineStage.CONTAINER_STARTED,
    "weights_fetched": TimelineStage.WEIGHTS_FETCHED,
    "framework_init": TimelineStage.FRAMEWORK_INIT,
    "weights_ready": TimelineStage.WEIGHTS_READY,
    "warmup_done": TimelineStage.WARMUP_DONE,
    "ready": TimelineStage.READY,
}


class TTotalError(RuntimeError):
    """The measurement could not be set up, or no scale event occurred."""


@dataclass(frozen=True, slots=True)
class StageTime:
    """One stage boundary on the timeline."""

    stage: str
    at: datetime | None

    bounded: bool = False
    """True when ``at`` is inferred rather than read from an API that timestamps the
    event directly. Carried into the report so a bound is never read as a measurement."""

    source: str | None = None
    """Which API or file the timestamp came from, for auditability."""

    note: str | None = None
    """Why this stage is bounded, absent, or otherwise not a plain reading."""

    @property
    def observed(self) -> bool:
        return self.at is not None


@dataclass(frozen=True, slots=True)
class StageDuration:
    """Elapsed time between two consecutive observed stages."""

    from_stage: str
    to_stage: str
    seconds: float
    bounded: bool = False


@dataclass(slots=True)
class TTotalReport:
    """A measured scaling lag, decomposed.

    :attr:`t_total_s` spans ``desired_set`` to ``traffic_recovered`` — the *span*, not the
    sum of the stages, which would silently swallow any gap between two APIs' clocks. It
    starts at the trigger rather than at ``load_applied`` because the probe's warm-up is
    this module's choice, not lag the fleet has to absorb.

    What it omits is the policy's own detection lag, which this trigger bypasses by design.
    :attr:`t_total_with_policy_bound_s` adds :data:`POLICY_LAG_BOUND_S` for planning, and
    keeps the two figures separate so a bound is never read as a measurement.
    """

    model_name: str
    endpoint: str
    run_id: str
    trigger: str

    deployed_config: dict[str, Any] = field(default_factory=dict)
    """Fingerprint of the configuration this lag was measured against, read from the
    endpoint. See ``fixture.DeployedConfig``.

    Carried for the same reason ``QMaxReport`` carries one, and for one more: the
    planner consumes a ``Q_max`` ladder and a ``T_total`` lag *together*, so without a
    fingerprint on both sides there is nothing to compare and a g5 ladder can be paired
    with a g6 lag silently. Container start dominates this measurement and is a
    property of the image, which is exactly what the digest pins."""

    timeline: list[StageTime] = field(default_factory=list)
    instance_id: str | None = None
    from_instances: int = 0
    to_instances: int = 0

    probe_concurrency: int = PROBE_CONCURRENCY
    """Concurrency the probe held. Recorded because the recovery test is only valid
    against the ladder rungs for this value and its half."""

    p95_expected_before_ms: float | None = None
    """The ladder's p95 at :attr:`probe_concurrency` — what the probe *should* read before
    the scale-out. Carried so a probe that never reached it is diagnosable as a bad probe
    rather than as a failed recovery: those need opposite responses."""

    p95_recovered_target_ms: float | None = None
    """The ladder's p95 at half :attr:`probe_concurrency`, times
    :data:`RECOVERY_TOLERANCE`. The line ``traffic_recovered`` is declared at."""

    p95_before_ms: float | None = None
    """Probe p95 TTFAB between ``desired_set`` and ``in_service`` — the queued state being
    recovered from. Without it, a recovery figure has nothing to be a recovery *from*."""

    p95_after_ms: float | None = None

    requests_before: int = 0
    requests_after: int = 0

    notes: list[str] = field(default_factory=list)

    def at(self, stage: TimelineStage | str) -> datetime | None:
        name = str(stage)
        return next((e.at for e in self.timeline if e.stage == name and e.at is not None), None)

    def entry(self, stage: TimelineStage | str) -> StageTime | None:
        name = str(stage)
        return next((e for e in self.timeline if e.stage == name), None)

    @property
    def config_slug(self) -> str:
        """Short identifier of the configuration measured, for an artifact filename.

        Mirrors :attr:`tts_bench.types.QMaxReport.config_slug` so the two artifact
        families are named on the same axis and a matching pair is recognisable by
        filename before anything opens them.
        """
        from tts_bench.fixture import DeployedConfig

        return DeployedConfig.from_dict(self.deployed_config).slug

    @property
    def observed_stages(self) -> list[StageTime]:
        """Stages with a timestamp, in chronological order.

        Sorted by time rather than by enum order because container stages legitimately
        differ in sequence between images: the S3-syncing containers fetch weights in the
        shell before the framework starts, while the kokoro images load them inside the
        Python lifespan.
        """
        observed = [e for e in self.timeline if e.at is not None]
        return sorted(observed, key=lambda e: (e.at, e.stage))  # type: ignore[arg-type,return-value]

    @property
    def missing_stages(self) -> list[str]:
        return [e.stage for e in self.timeline if e.at is None]

    @property
    def durations(self) -> list[StageDuration]:
        """Elapsed times between consecutive observed stages."""
        observed = self.observed_stages
        out: list[StageDuration] = []
        for prev, nxt in zip(observed, observed[1:], strict=False):
            assert prev.at is not None and nxt.at is not None  # observed_stages drops None
            out.append(
                StageDuration(
                    from_stage=prev.stage,
                    to_stage=nxt.stage,
                    seconds=(nxt.at - prev.at).total_seconds(),
                    bounded=prev.bounded or nxt.bounded,
                )
            )
        return out

    def _span_from(self, start_stage: TimelineStage) -> float | None:
        start = self.at(start_stage)
        if start is None:
            return None
        end = self.at(TimelineStage.TRAFFIC_RECOVERED) or self.at(TimelineStage.IN_SERVICE)
        if end is None:
            observed = self.observed_stages
            end = observed[-1].at if observed else None
        if end is None or end < start:
            return None
        return (end - start).total_seconds()

    @property
    def t_total_s(self) -> float | None:
        """Trigger through to traffic recovered, in seconds.

        From ``desired_set``, so the probe's warm-up is excluded: the fleet's lag begins
        when something asks for capacity, and how long the probe ran before that is this
        module's choice.
        """
        span = self._span_from(TimelineStage.DESIRED_SET)
        if span is not None:
            return span
        # Without a desired_set stage, fall back to the full observed span rather than
        # reporting nothing — a report rebuilt from a past window still has stages.
        observed = self.observed_stages
        if len(observed) < 2:
            return None
        first, last = observed[0].at, observed[-1].at
        assert first is not None and last is not None
        return (last - first).total_seconds()

    @property
    def t_total_with_policy_bound_s(self) -> float | None:
        """:attr:`t_total_s` plus the policy detection lag this trigger skips.

        The figure to *plan* on, since production scales out via the policy rather than
        via a direct capacity call. Kept separate from :attr:`t_total_s` because one term
        is measured and the other is arithmetic from the deployed policy's configuration —
        adding them into a single field would make the sum unfalsifiable.
        """
        total = self.t_total_s
        return None if total is None else total + POLICY_LAG_BOUND_S

    @property
    def t_total_bounded(self) -> bool:
        """Whether ``t_total_s`` stops short of recovery and so *under*-reports the lag.

        True when ``traffic_recovered`` has no timestamp, which makes :attr:`t_total_s`
        fall back to ``in_service`` — strictly earlier than the instance serving, and
        under-reporting is the dangerous direction for a capacity plan.

        Deliberately **not** ``recovered.bounded``. Every recovery is an inference —
        SageMaker does not say which instance served a request, so the p95 halving is the
        only evidence there is, and :func:`assemble_timeline` marks that stage bounded on
        every run. Reading the flag off it made this property a constant ``True``, so both
        the report and the planner said "recovery inferred, so this is a floor" about runs
        whose recovery was observed cleanly. The two facts are separate: the ``~`` in the
        stage breakdown still marks the timestamp as inferred, while this marks the span
        as having no recovery endpoint at all.
        """
        recovered = self.entry(TimelineStage.TRAFFIC_RECOVERED)
        return recovered is None or recovered.at is None

    @property
    def dominant_stage(self) -> StageDuration | None:
        """The single longest stage — where attacking ``T_total`` actually pays off."""
        durations = self.durations
        return max(durations, key=lambda d: d.seconds) if durations else None

    @property
    def aws_share_s(self) -> float | None:
        """Lag before the container ran: instance provisioning and image pull.

        Split out because it is the half no container change can shorten. Ends at
        ``container_started`` when the container reported it, otherwise at the log stream
        opening, which is a looser bound but better than nothing.
        """
        start = self.at(TimelineStage.DESIRED_SET) or self.at(TimelineStage.LOAD_APPLIED)
        end = self.at(TimelineStage.CONTAINER_STARTED) or self.at(TimelineStage.INSTANCE_LOGGING)
        if start is None or end is None or end < start:
            return None
        return (end - start).total_seconds()

    @property
    def container_share_s(self) -> float | None:
        """Lag inside the container: weights, framework init, warm-up.

        The half a container change *can* shorten — baking weights into the image,
        skipping a warm-up, shrinking a layer.
        """
        start = self.at(TimelineStage.CONTAINER_STARTED) or self.at(TimelineStage.INSTANCE_LOGGING)
        end = self.at(TimelineStage.READY) or self.at(TimelineStage.IN_SERVICE)
        if start is None or end is None or end < start:
            return None
        return (end - start).total_seconds()

    def provenance(self) -> Provenance:
        """Provenance for feeding ``t_total_s`` into :class:`tts_bench.types.Measured`.

        Measured, not derived: it is an observation of a live scale event. The note records
        that the policy half is excluded, because a ``force-desired`` number is the
        container half only and must not be mistaken for a full ``T_total``.
        """
        from tts_bench.types import Origin, Provenance

        bits = [f"trigger={self.trigger}"]
        if self.trigger == TRIGGER_FORCE_DESIRED:
            bits.append(
                f"capacity half only; the policy's detection lag is bounded at "
                f"{POLICY_LAG_BOUND_S:.0f}s rather than measured"
            )
        if self.t_total_bounded:
            bits.append("recovery never bounded, so this stops at in_service and is a floor")
        if self.missing_stages:
            bits.append(f"{len(self.missing_stages)} stage(s) not observed")
        triggered_at = self.at(TimelineStage.DESIRED_SET) or self.at(TimelineStage.LOAD_APPLIED)
        return Provenance(
            origin=Origin.MEASURED,
            run_id=self.run_id,
            measured_at=triggered_at.isoformat() if triggered_at else None,
            endpoint=self.endpoint,
            note="; ".join(bits),
        )

    def to_dict(self) -> dict[str, Any]:
        total = self.t_total_s
        return {
            "model_name": self.model_name,
            "endpoint": self.endpoint,
            "run_id": self.run_id,
            "trigger": self.trigger,
            "deployed_config": dict(self.deployed_config),
            "config_slug": self.config_slug,
            "t_total_s": total,
            "t_total_with_policy_bound_s": self.t_total_with_policy_bound_s,
            "policy_lag_bound_s": POLICY_LAG_BOUND_S,
            "t_total_bounded": self.t_total_bounded,
            "aws_share_s": self.aws_share_s,
            "container_share_s": self.container_share_s,
            "instance_id": self.instance_id,
            "from_instances": self.from_instances,
            "to_instances": self.to_instances,
            "probe_concurrency": self.probe_concurrency,
            "p95_expected_before_ms": self.p95_expected_before_ms,
            "p95_recovered_target_ms": self.p95_recovered_target_ms,
            "p95_before_ms": self.p95_before_ms,
            "p95_after_ms": self.p95_after_ms,
            "requests_before": self.requests_before,
            "requests_after": self.requests_after,
            "timeline": [
                {
                    "stage": e.stage,
                    "at": e.at.isoformat() if e.at else None,
                    "bounded": e.bounded,
                    "source": e.source,
                    "note": e.note,
                }
                for e in self.timeline
            ],
            "durations": [
                {
                    "from": d.from_stage,
                    "to": d.to_stage,
                    "seconds": round(d.seconds, 3),
                    "bounded": d.bounded,
                    "share": round(d.seconds / total, 4) if total else None,
                }
                for d in self.durations
            ],
            "dominant_stage": (
                {
                    "from": self.dominant_stage.from_stage,
                    "to": self.dominant_stage.to_stage,
                    "seconds": round(self.dominant_stage.seconds, 3),
                }
                if self.dominant_stage
                else None
            ),
            "missing_stages": self.missing_stages,
            "notes": self.notes,
        }


# --------------------------------------------------------------------------- #
# Timeline assembly — pure, so the stage rules are testable without AWS
# --------------------------------------------------------------------------- #


def container_timeline(
    markers: Sequence[StageMarker],
    *,
    stream_opened_at: datetime | None = None,
) -> list[StageTime]:
    """Map a container's stage markers onto timeline entries.

    ``container_start`` is usually absent: the base image prints its own banner before our
    entrypoint runs, and CloudWatch may have aged out the earliest lines. When it is
    missing it is reconstructed by subtracting the earliest marker's ``elapsed_s`` from
    that marker's own timestamp — which is exactly why the emitters carry ``elapsed_s``
    rather than leaving consumers to difference millisecond timestamps.

    Args:
        markers: :class:`shared.stages.StageMarker` instances, in emitted order.
        stream_opened_at: The stream's ``firstEventTimestamp``, used only to sanity-check
            a reconstructed container start.
    """
    entries: list[StageTime] = []
    for marker in markers:
        stage = _CONTAINER_STAGE_MAP.get(marker.name)
        entries.append(
            StageTime(
                stage=str(stage) if stage else marker.name,
                at=marker.at,
                source=f"container log marker (elapsed_s={marker.elapsed_s:.3f})",
                note=None if stage else "stage name not known to this version of ttotal",
            )
        )

    if markers and not any(e.stage == str(TimelineStage.CONTAINER_STARTED) for e in entries):
        earliest = min(markers, key=lambda m: m.at)
        reconstructed = earliest.at - timedelta(seconds=earliest.elapsed_s)
        note = (
            f"reconstructed as {earliest.name} minus its elapsed_s="
            f"{earliest.elapsed_s:.3f}s; no container_start line was in the stream"
        )
        if stream_opened_at is not None and reconstructed < stream_opened_at:
            # The container claims it started before its stream had any events.
            # Legitimate when early lines aged out — worth stating, not correcting.
            note += " (predates the stream's first event, so earlier lines were dropped)"
        entries.append(
            StageTime(
                stage=str(TimelineStage.CONTAINER_STARTED),
                at=reconstructed,
                bounded=True,
                source="derived from elapsed_s",
                note=note,
            )
        )
    return entries


def _ttfab_samples(events: Sequence[LoadEvent]) -> list[tuple[datetime, float]]:
    """``(first-byte time, TTFAB ms)`` for events that produced audio, oldest first.

    Keyed on first byte rather than on stream completion: TTFAB is *observed* at the
    first byte, so that is when the sample exists. Using the end of the stream would push
    every sample later by a whole synthesis and blur the recovery boundary.
    """
    samples = [
        (datetime.fromtimestamp(e.first_byte_ts, tz=UTC), float(e.ttfab_ms))
        for e in events
        if e.ttfab_ms is not None and e.first_byte_ts is not None
    ]
    samples.sort(key=lambda pair: pair[0])
    return samples


def recovery_references(
    ladder_p95_ms: dict[int, float],
    *,
    probe_concurrency: int = PROBE_CONCURRENCY,
    tolerance: float = RECOVERY_TOLERANCE,
) -> tuple[float, float]:
    """``(expected_before_ms, recovered_target_ms)`` from a ``Q_max`` ladder's rungs.

    The two sides of the halving test, both measured on the same configuration by the same
    ladder. ``expected_before_ms`` is the probe's rung: what p95 should read while one
    instance serves the whole probe. ``recovered_target_ms`` is the half-concurrency rung
    times ``tolerance``: where p95 lands once a second instance splits the probe.

    Refuses rather than interpolating. A ladder without both rungs cannot answer this, and
    a nearest-rung fallback would compare a probe at 10 against a rung at 20 while
    reporting a number that looks measured.

    Raises:
        TTotalError: If either rung is missing, naming which — that is a re-run of the
            ladder with the rung added, not something this command can work around.
    """
    if probe_concurrency < 2 or probe_concurrency % 2:
        raise TTotalError(
            f"probe concurrency must be an even number of at least 2 to halve, got "
            f"{probe_concurrency}"
        )
    half = probe_concurrency // 2
    missing = [rung for rung in (probe_concurrency, half) if ladder_p95_ms.get(rung) is None]
    if missing:
        raise TTotalError(
            f"the Q_max artifact has no usable p95 at concurrency {missing} (measured rungs: "
            f"{sorted(ladder_p95_ms)}). Recovery is judged by the probe's p95 at "
            f"{probe_concurrency} halving into its value at {half}, so both must be rungs on "
            f"the ladder. Re-run `tts-bench qmax` with --concurrency including "
            f"{','.join(str(r) for r in sorted({half, probe_concurrency}))}."
        )
    return ladder_p95_ms[probe_concurrency], ladder_p95_ms[half] * tolerance


def recovery_bound(
    events: Sequence[LoadEvent],
    *,
    in_service_at: datetime,
    budget_ms: float,
    window_s: float = RECOVERY_WINDOW_S,
    min_samples: int = RECOVERY_MIN_SAMPLES,
) -> tuple[datetime | None, str]:
    """Earliest time after ``in_service_at`` at which p95 held at or under ``budget_ms``.

    The bound on ``traffic_recovered``. SageMaker does not say which instance served a
    request, so "when did the new instance take real traffic" is answered from the client
    side instead: when the probe's tail latency dropped and stayed down. That is an upper
    bound on the new instance being useful, and a lower bound on nothing — stated as such
    in the returned explanation, which the report carries verbatim.

    Args:
        budget_ms: The recovered level, from :func:`recovery_references` — the ladder's p95
            at *half* the probe's concurrency, plus tolerance. Deliberately not the SLO: at
            3000 ms an overloaded probe is already "inside budget", so this would return the
            first sample after ``in_service`` and measure nothing.

    Returns:
        ``(timestamp_or_none, explanation)``. The explanation is kept whether or not a
        bound was found, because "never recovered" is the more important result.
    """
    samples = [pair for pair in _ttfab_samples(events) if pair[0] >= in_service_at]
    if len(samples) < min_samples:
        return None, (
            f"only {len(samples)} completion(s) with a TTFAB after the new instance came "
            f"into service; {min_samples} are needed to bound recovery. Hold the probe "
            "longer past the scale event with --settle."
        )

    times = [t for t, _ in samples]
    values = [v for _, v in samples]

    for i in range(len(samples)):
        window = [
            values[j]
            for j in range(i, len(samples))
            if (times[j] - times[i]).total_seconds() <= window_s
        ]
        if len(window) < min_samples:
            # Too sparse to judge, and every later start is sparser still.
            break
        if float(np.percentile(window, 95)) <= budget_ms:
            return times[i], (
                f"probe p95 TTFAB held at or under {budget_ms:.0f}ms across {len(window)} "
                f"completions in the following {window_s:.0f}s — the router had split the "
                "probe, so a second instance was serving"
            )

    return None, (
        f"probe p95 TTFAB never held under {budget_ms:.0f}ms for {window_s:.0f}s after the "
        "new instance came into service, so recovery is not bounded from above and T_total "
        "is a floor. Either the added instance never took traffic, or the probe's own p95 "
        "was not what the ladder measured — compare p95_before_ms against "
        "p95_expected_before_ms before reading this as a scaling failure."
    )


def assemble_timeline(
    *,
    load_applied_at: datetime | None,
    desired_set_at: datetime | None,
    stream: LogStream | None,
    container_stages: Sequence[StageTime],
    in_service_at: datetime | None,
    recovered_at: datetime | None,
    recovery_note: str,
) -> list[StageTime]:
    """Build the ordered timeline from independently-fetched boundaries.

    Pure: every argument is already-fetched data, so the assembly and bounding rules are
    testable without AWS. Container-reported stages are appended where the container put
    them — see :attr:`TTotalReport.observed_stages` for why order is not imposed here.

    ``load_applied`` is on the timeline but is not the clock's start: it records that the
    probe was established *before* the trigger, so a `T_total` measured from ``desired_set``
    is known to have been measured under load rather than against an idle endpoint.
    """
    timeline: list[StageTime] = [
        StageTime(
            stage=str(TimelineStage.LOAD_APPLIED),
            at=load_applied_at,
            source="probe start",
            note=(
                "probe established before the trigger; not the start of the clock"
                if load_applied_at
                else "no probe ran, so recovery cannot be observed"
            ),
        ),
        StageTime(
            stage=str(TimelineStage.DESIRED_SET),
            at=desired_set_at,
            # Our own API call, so this timestamp is exact rather than polled — the one
            # boundary on the timeline that is neither bounded nor inferred.
            source="UpdateEndpointWeightsAndCapacities call",
            note=(
                "the clock starts here"
                if desired_set_at
                else "the capacity change was never requested"
            ),
        ),
        StageTime(
            stage=str(TimelineStage.INSTANCE_LOGGING),
            at=stream.first_event_at if stream else None,
            # An upper bound on provisioning: the instance existed at or before its first
            # log line, never after it.
            bounded=True,
            source="log stream firstEventTimestamp",
            note=(
                "upper bound on provisioning plus image pull, which a container cannot time itself"
                if stream
                else "no new log stream was seen for this event"
            ),
        ),
    ]
    timeline.extend(container_stages)
    if not any(e.stage == str(TimelineStage.READY) for e in container_stages):
        # Every container emits `ready`; the other container stages are legitimately
        # image-specific. So its absence is a stated gap rather than a silent omission —
        # otherwise a report with no container half at all looks complete.
        timeline.append(
            StageTime(
                stage=str(TimelineStage.READY),
                at=None,
                source="container log marker",
                note="no container reported becoming ready; see notes for why",
            )
        )
    timeline.append(
        StageTime(
            stage=str(TimelineStage.IN_SERVICE),
            at=in_service_at,
            source="DescribeEndpoint CurrentInstanceCount",
            note=None if in_service_at else "the fleet never reached the new count",
        )
    )
    timeline.append(
        StageTime(
            stage=str(TimelineStage.TRAFFIC_RECOVERED),
            at=recovered_at,
            bounded=True,
            source="client-side p95 recovery",
            note=recovery_note,
        )
    )
    return timeline


# --------------------------------------------------------------------------- #
# Live measurement
# --------------------------------------------------------------------------- #


def read_capacity(sagemaker: BaseClient, endpoint: str, variant: str) -> tuple[int, int]:
    """``(desired, current)`` instance counts for one variant.

    Raises:
        TTotalError: If the endpoint has no such variant.
    """
    response = sagemaker.describe_endpoint(EndpointName=endpoint)
    for v in response.get("ProductionVariants", []):
        if v.get("VariantName") == variant:
            return int(v.get("DesiredInstanceCount", 0)), int(v.get("CurrentInstanceCount", 0))
    raise TTotalError(f"{endpoint} has no variant {variant}")


@dataclass(slots=True)
class ScaleEvent:
    """A capacity increase, as observed while waiting for it."""

    from_instances: int
    to_instances: int
    desired_changed_at: datetime | None
    in_service_at: datetime | None

    @property
    def occurred(self) -> bool:
        return self.in_service_at is not None and self.to_instances > self.from_instances


def wait_for_scale_out(
    sagemaker: BaseClient,
    *,
    endpoint: str,
    variant: str = DEFAULT_VARIANT,
    from_instances: int,
    max_wait_s: float = DEFAULT_MAX_WAIT_S,
    poll_interval_s: float = DEFAULT_POLL_INTERVAL_S,
    stop_event: threading.Event | None = None,
    now: Callable[[], datetime] | None = None,
    sleep: Callable[[float], None] | None = None,
) -> ScaleEvent:
    """Poll until ``CurrentInstanceCount`` exceeds ``from_instances``.

    Records ``DesiredInstanceCount`` changing separately from the fleet reaching the new
    count: the gap between them *is* provisioning plus container startup, and collapsing
    the two would hide the stages this command exists to attribute.

    ``now`` and ``sleep`` are injectable so tests can drive the loop without waiting.

    Returns:
        A :class:`ScaleEvent` whose ``occurred`` is False on timeout — a real result
        meaning the policy did not act, not an error.
    """
    now = now or (lambda: datetime.now(UTC))
    sleep = sleep or time.sleep

    deadline = now() + timedelta(seconds=max_wait_s)
    desired_changed_at: datetime | None = None
    latest = from_instances

    while now() < deadline:
        if stop_event is not None and stop_event.is_set():
            logger.warning("Stopped waiting for scale-out on {}", endpoint)
            break
        desired, current = read_capacity(sagemaker, endpoint, variant)
        if desired > from_instances and desired_changed_at is None:
            desired_changed_at = now()
            logger.info("{}: DesiredInstanceCount rose to {}", endpoint, desired)
        if current > from_instances:
            logger.info("{}: reached {} instance(s)", endpoint, current)
            return ScaleEvent(
                from_instances=from_instances,
                to_instances=current,
                desired_changed_at=desired_changed_at,
                in_service_at=now(),
            )
        latest = current
        sleep(poll_interval_s)

    return ScaleEvent(
        from_instances=from_instances,
        to_instances=latest,
        desired_changed_at=desired_changed_at,
        in_service_at=None,
    )


def _endpoint_status(sagemaker: BaseClient, endpoint: str) -> str | None:
    """``EndpointStatus``, or ``None`` if it cannot be read.

    Only used to explain a timeout, so a failure here must not replace the diagnosis with
    its own traceback.
    """
    try:
        return str(sagemaker.describe_endpoint(EndpointName=endpoint).get("EndpointStatus", ""))
    except Exception:  # noqa: BLE001 - a diagnosis is worth more than this detail
        return None


def set_desired_count(
    sagemaker: BaseClient,
    *,
    endpoint: str,
    variant: str = DEFAULT_VARIANT,
    to_instances: int,
) -> None:
    """Set ``DesiredInstanceCount`` directly, bypassing any scaling policy."""
    logger.info("Setting {} variant {} to {} instance(s)", endpoint, variant, to_instances)
    sagemaker.update_endpoint_weights_and_capacities(
        EndpointName=endpoint,
        DesiredWeightsAndCapacities=[
            {"VariantName": variant, "DesiredInstanceCount": to_instances}
        ],
    )


def restore_desired_count(
    sagemaker: BaseClient,
    *,
    endpoint: str,
    variant: str = DEFAULT_VARIANT,
    to_instances: int,
    max_wait_s: float = DEFAULT_RESTORE_WAIT_S,
    poll_interval_s: float = DEFAULT_POLL_INTERVAL_S,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Put ``DesiredInstanceCount`` back. Best-effort, never raises.

    Runs on every exit path including Ctrl-C: a measurement must not leave an endpoint
    parked at four instances. A failure here is logged rather than raised, because raising
    from a ``finally`` would mask whatever actually went wrong.

    Waits out ``Updating`` first. A run that times out mid-scale-out leaves the endpoint
    exactly there, and SageMaker answers a capacity change in that state with
    ``ValidationException: Cannot update in-progress endpoint`` — so the naive restore
    fails at the one moment it is most needed, and the fleet keeps billing at the raised
    count. On giving up it says what to run by hand, since nothing else will shrink it:
    ``DesiredInstanceCount`` is not what a scale-in policy reads.
    """
    deadline = time.monotonic() + max_wait_s
    try:
        while True:
            described = sagemaker.describe_endpoint(EndpointName=endpoint)
            status = described.get("EndpointStatus", "")
            desired = next(
                (
                    int(v.get("DesiredInstanceCount", 0))
                    for v in described.get("ProductionVariants", [])
                    if v.get("VariantName") == variant
                ),
                None,
            )
            if desired == to_instances:
                return
            if status != "Updating":
                break
            if time.monotonic() >= deadline:
                raise TTotalError(
                    f"{endpoint} was still Updating after {max_wait_s:.0f}s, so its capacity "
                    f"could not be changed back"
                )
            logger.info(
                "{} is Updating; waiting {:.0f}s to restore {} instance(s)",
                endpoint,
                poll_interval_s,
                to_instances,
            )
            sleep(poll_interval_s)
        set_desired_count(sagemaker, endpoint=endpoint, variant=variant, to_instances=to_instances)
    except Exception as exc:  # noqa: BLE001 - restoration must not mask a real failure
        logger.error(
            "Could not restore {} to {} instance(s): {}. It is still billing at the raised "
            "count and no policy will shrink it — run: aws sagemaker "
            "update-endpoint-weights-and-capacities --endpoint-name {} "
            "--desired-weights-and-capacities VariantName={},DesiredInstanceCount={}",
            endpoint,
            to_instances,
            exc,
            endpoint,
            variant,
            to_instances,
        )


def collect_timeline(
    *,
    logs: BaseClient,
    endpoint: str,
    variant: str,
    model_name: str,
    run_id: str,
    trigger: str,
    recovered_budget_ms: float,
    p95_expected_before_ms: float | None,
    probe_concurrency: int,
    load_applied_at: datetime | None,
    desired_set_at: datetime | None,
    window_start: datetime,
    window_end: datetime,
    streams_before: Sequence[LogStream],
    event: ScaleEvent,
    load_events: Sequence[LoadEvent],
    deployed_config: dict[str, Any] | None = None,
) -> TTotalReport:
    """Fetch every stage boundary for one observed scale event and assemble it.

    Split from the driving of load so a report can be rebuilt from a window that has
    already happened — which is also what makes it testable with stub clients.

    Reads logs only. The metric, alarm and scaling-activity fetches this used to make were
    all attribution for a policy-driven trigger; with ``force-desired`` the policy is
    suspended for the whole run, so those APIs have nothing to say and querying them would
    invite reading meaning into their silence.

    Args:
        recovered_budget_ms: p95 that counts as recovered, from :func:`recovery_references`.
        p95_expected_before_ms: The ladder's p95 at ``probe_concurrency``, carried so a
            probe that never reached its own expectation is distinguishable from a
            scale-out that did not help. ``None`` when no ladder was supplied.
        deployed_config: Configuration fingerprint to stamp on the report, from
            :func:`tts_bench.fixture.fingerprint_or_registry`. Optional so a report can
            still be rebuilt from a historical window whose endpoint has since changed;
            the resulting slug then says ``unknown-nodigest``, which cannot compare equal
            to a real fingerprint and so fails the planner's pairing check rather than
            passing it by accident.
    """
    report = TTotalReport(
        model_name=model_name,
        endpoint=endpoint,
        run_id=run_id,
        trigger=trigger,
        deployed_config=dict(deployed_config or {}),
        from_instances=event.from_instances,
        to_instances=event.to_instances,
        probe_concurrency=probe_concurrency,
        p95_expected_before_ms=p95_expected_before_ms,
        p95_recovered_target_ms=recovered_budget_ms,
    )
    report.notes.append(
        f"trigger was --{TRIGGER_FORCE_DESIRED} under a scaling freeze, so the policy's own "
        f"detection lag is not in this number. It is bounded separately at "
        f"{POLICY_LAG_BOUND_S:.0f}s and reported as t_total_with_policy_bound_s; that field, "
        "not t_total_s, is what the planner sizes headroom against."
    )

    # The new instance is identified by its log stream: SageMaker does not report which
    # instance it added, but each one opens a stream of its own.
    streams_after = list_log_streams(logs, endpoint=endpoint, variant=variant)
    fresh = new_streams_since(streams_before, streams_after)
    stream = fresh[0] if fresh else None
    if stream is None:
        report.notes.append(
            f"no new log stream appeared under {log_group_name(endpoint)}, so container "
            "stages could not be read. Logs can lag the endpoint by a minute or two; the "
            "container half is a single gap in this report."
        )
    else:
        report.instance_id = stream.instance_id
        if len(fresh) > 1:
            report.notes.append(
                f"{len(fresh)} new streams appeared; attributing container stages to "
                f"{stream.name}, the most recently active. More than one instance was "
                "added, so this measures whichever one logged last."
            )

    container_stages: list[StageTime] = []
    if stream is not None:
        markers = stage_markers(logs, endpoint=endpoint, stream=stream.name)
        if markers:
            container_stages = container_timeline(markers, stream_opened_at=stream.first_event_at)
        else:
            report.notes.append(
                f"stream {stream.name} carried no STAGE markers, so container stages are a "
                "single gap. The image predates the markers, or its startup lines aged out."
            )

    recovered_at: datetime | None = None
    recovery_note = "not evaluated: the fleet never reached the new instance count"
    if event.in_service_at is not None:
        recovered_at, recovery_note = recovery_bound(
            load_events, in_service_at=event.in_service_at, budget_ms=recovered_budget_ms
        )
        samples = _ttfab_samples(load_events)
        # Anchored at the trigger, not at the probe's start: the warm-up leading up to it
        # includes all N workers arriving at once, and that burst's p95 is not the settled
        # level the ladder measured.
        floor = desired_set_at or load_applied_at or window_start
        before = [v for t, v in samples if floor <= t < event.in_service_at]
        after = [v for t, v in samples if t >= event.in_service_at]
        report.requests_before = len(before)
        report.requests_after = len(after)
        if before:
            report.p95_before_ms = float(np.percentile(before, 95))
        if after:
            report.p95_after_ms = float(np.percentile(after, 95))

    if (
        report.p95_before_ms is not None
        and p95_expected_before_ms is not None
        and report.p95_before_ms < recovered_budget_ms
    ):
        # The probe was supposed to be saturating. If its pre-scale p95 was already at the
        # recovered level, the halving has nothing to detect and `traffic_recovered` will
        # land on the first sample after in_service — a tautology. Said here rather than
        # left to be inferred from two numbers on the artifact.
        report.notes.append(
            f"the probe's pre-scale p95 was {report.p95_before_ms:.0f}ms, already at or under "
            f"the {recovered_budget_ms:.0f}ms recovered level, against an expected "
            f"{p95_expected_before_ms:.0f}ms at N={probe_concurrency}. Recovery cannot be "
            "detected from a probe that was never saturating: raise --probe-concurrency (and "
            "add the matching rungs to the Q_max ladder) rather than reading this T_total."
        )

    report.timeline = assemble_timeline(
        load_applied_at=load_applied_at,
        desired_set_at=desired_set_at,
        stream=stream,
        container_stages=container_stages,
        in_service_at=event.in_service_at,
        recovered_at=recovered_at,
        recovery_note=recovery_note,
    )

    if report.t_total_bounded:
        report.notes.append(
            "recovery was never bounded, so T_total spans only as far as in_service — "
            "earlier than the new instance serving. It under-reports rather than "
            "over-reports: treat it as a floor when sizing headroom."
        )
    return report


def measure(
    *,
    model_name: str,
    endpoint: str,
    variant: str = DEFAULT_VARIANT,
    region: str = "us-east-1",
    ladder_p95_ms: dict[int, float],
    texts: Sequence[str],
    voice: str,
    trigger: str = TRIGGER_FORCE_DESIRED,
    probe_concurrency: int = PROBE_CONCURRENCY,
    warmup_s: float = DEFAULT_WARMUP_S,
    max_wait_s: float = DEFAULT_MAX_WAIT_S,
    settle_s: float = DEFAULT_SETTLE_S,
    poll_interval_s: float = DEFAULT_POLL_INTERVAL_S,
    run_id: str | None = None,
    transport: str = "bidi",
    seed: int | None = 1234,
    event_sink: Callable[[LoadEvent], None] | None = None,
    appscaling: BaseClient | None = None,
    sagemaker: BaseClient | None = None,
    logs: BaseClient | None = None,
) -> TTotalReport:
    """Trigger one scale-out under a saturating probe and measure the lag, stage by stage.

    The order is fixed and each step exists because of a specific way the measurement fails
    without it:

    1. **Freeze** at one instance. Application Auto Scaling suspension does not block our
       own ``UpdateEndpointWeightsAndCapacities``, so we keep the trigger while the deployed
       policy loses its. Without this the probe's own load fires the live policy, and
       instances arrive from two causes inside one measurement.
    2. **Start the probe** and let it run ``warmup_s``, so the queue is established and p95
       has settled at its pre-scale level before anything changes.
    3. **Set desired = current + 1.** This call's timestamp starts the clock.
    4. **Wait**, then hold the probe ``settle_s`` longer so recovery has data past the event.
    5. **Restore** the starting desired count and thaw, on every exit path including Ctrl-C.

    Args:
        ladder_p95_ms: ``{concurrency: p95_ms}`` from a ``Q_max`` run on this same
            configuration, i.e. :attr:`tts_bench.types.QMaxReport.ladder_p95_ms`. Both
            ``probe_concurrency`` and half of it must be rungs in it — see
            :func:`recovery_references`.
        probe_concurrency: Closed-loop concurrency held throughout. Must be saturating: at
            ``N=1`` p95 is already healthy before the scale-out, so recovery would be
            declared at the first sample after ``in_service`` and measure nothing.
        warmup_s: Probe time before the trigger. Not part of ``T_total`` — the clock starts
            at ``desired_set``, so this is only long enough to establish the queue.

    Raises:
        TTotalError: If no scale event occurred within ``max_wait_s``, or the ladder lacks
            the rungs the halving test needs.
        tts_bench.fixture.FixtureError: If the freeze cannot be established.
        ValueError: If ``trigger`` is not a supported mode.
    """
    import boto3

    from tts_bench import fixture

    if trigger != TRIGGER_FORCE_DESIRED:
        raise ValueError(f"trigger must be {TRIGGER_FORCE_DESIRED!r}, got {trigger!r}")

    # Before anything is frozen or any load is sent: a ladder missing a rung is a re-run of
    # `qmax`, and finding that out after the probe has run costs the whole measurement.
    p95_expected_before_ms, recovered_budget_ms = recovery_references(
        ladder_p95_ms, probe_concurrency=probe_concurrency
    )

    run_id = run_id or uuid.uuid4().hex[:12]
    appscaling = appscaling or boto3.client("application-autoscaling", region_name=region)
    sagemaker = sagemaker or boto3.client("sagemaker", region_name=region)
    logs = logs or boto3.client("logs", region_name=region)

    # Read before the run rather than after: most of this measurement is container start,
    # so if a redeploy lands mid-run the lag belongs to the image that was serving when it
    # started, not to whatever replaced it.
    deployed = fixture.fingerprint_or_registry(
        model_name, endpoint=endpoint, region=region, variant=variant, sagemaker=sagemaker
    )

    load_events: list[LoadEvent] = []
    stop_event = threading.Event()
    driver: threading.Thread | None = None
    load_applied_at: datetime | None = None
    desired_set_at: datetime | None = None
    diagnosis: str | None = None
    timed_out = False

    # The freeze wraps everything, including the restore: thaw runs from `frozen.__exit__`,
    # which fires for KeyboardInterrupt too. `restore_capacity=False` because the inner
    # `finally` puts the count back itself — it has to wait out `Updating` first, which
    # thaw's unconditional restore does not.
    with fixture.frozen(
        endpoint,
        region=region,
        variant=variant,
        pin_to=1,
        appscaling=appscaling,
        sagemaker=sagemaker,
        restore_capacity=False,
    ):
        fixture.require_frozen(
            endpoint,
            region=region,
            variant=variant,
            expect_instances=1,
            appscaling=appscaling,
            sagemaker=sagemaker,
        )

        # After the freeze: freeze() pins to 1, so reading capacity before it would record
        # whatever the fleet happened to be at and restore to that instead.
        desired_before, current_before = read_capacity(sagemaker, endpoint, variant)
        # Captured before anything changes: the set difference against this is what names
        # the new instance afterwards.
        streams_before = list_log_streams(logs, endpoint=endpoint, variant=variant)
        logger.info(
            "T_total run {} on {}: {} instance(s), {} known log stream(s), probe N={}",
            run_id,
            endpoint,
            current_before,
            len(streams_before),
            probe_concurrency,
        )

        window_start = datetime.now(UTC)
        # Bound before the try so the finally can read it on any exit path.
        event = ScaleEvent(
            from_instances=current_before,
            to_instances=current_before,
            desired_changed_at=None,
            in_service_at=None,
        )

        try:
            driver, load_applied_at = _start_probe(
                model_name=model_name,
                endpoint=endpoint,
                region=region,
                variant=variant,
                voice=voice,
                texts=texts,
                transport=transport,
                concurrency=probe_concurrency,
                duration_s=warmup_s + max_wait_s + settle_s,
                run_id=run_id,
                seed=seed,
                sink=load_events.append,
                extra_sink=event_sink,
                stop_event=stop_event,
                sagemaker=sagemaker,
            )
            # Warm-up before the trigger, so p95 is at its pre-scale level when the clock
            # starts. Measuring from load_applied would fold this wait into T_total, and it
            # is this module's choice rather than anything AWS did.
            logger.info(
                "Warming the probe {:.0f}s at N={} before forcing capacity",
                warmup_s,
                probe_concurrency,
            )
            time.sleep(warmup_s)

            desired_set_at = datetime.now(UTC)
            set_desired_count(
                sagemaker, endpoint=endpoint, variant=variant, to_instances=current_before + 1
            )

            event = wait_for_scale_out(
                sagemaker,
                endpoint=endpoint,
                variant=variant,
                from_instances=current_before,
                max_wait_s=max_wait_s,
                poll_interval_s=poll_interval_s,
            )
            timed_out = not event.occurred
            if event.occurred:
                # Keep the probe running past the event so the recovery bound has data on
                # the far side of it. Without this, T_total stops at in_service.
                logger.info("Holding the probe {:.0f}s to bound recovery", settle_s)
                time.sleep(settle_s)
        finally:
            stop_event.set()
            if driver is not None:
                driver.join(timeout=120.0)
                if driver.is_alive():
                    logger.warning(
                        "Probe still running after 120s; requests may still be arriving at {}",
                        endpoint,
                    )
            window_end = datetime.now(UTC)
            # Only on a timeout: a Ctrl-C never sets this, so an interrupted run restores
            # immediately instead of spending API calls explaining a scale-out nobody
            # waited for.
            if timed_out:
                diagnosis = _explain_no_scale_out(
                    endpoint=endpoint,
                    from_instances=current_before,
                    max_wait_s=max_wait_s,
                    endpoint_status=_endpoint_status(sagemaker, endpoint),
                )
            restore_desired_count(
                sagemaker, endpoint=endpoint, variant=variant, to_instances=desired_before
            )

    if diagnosis is not None:
        raise TTotalError(diagnosis)

    return collect_timeline(
        logs=logs,
        endpoint=endpoint,
        variant=variant,
        model_name=model_name,
        run_id=run_id,
        trigger=trigger,
        recovered_budget_ms=recovered_budget_ms,
        p95_expected_before_ms=p95_expected_before_ms,
        probe_concurrency=probe_concurrency,
        load_applied_at=load_applied_at,
        desired_set_at=desired_set_at,
        window_start=window_start,
        window_end=window_end,
        streams_before=streams_before,
        event=event,
        load_events=load_events,
        deployed_config=deployed.to_dict(),
    )


def _explain_no_scale_out(
    *,
    endpoint: str,
    from_instances: int,
    max_wait_s: float,
    endpoint_status: str | None = None,
) -> str:
    """Why no scale-out happened, from the endpoint's own status.

    No policy is in the path — capacity was set directly, under a freeze — so the scaling
    activity log is silent by design and reading anything into its silence would be wrong.
    That leaves two outcomes, and they cost very different things to fix: ``Updating`` means
    SageMaker took the change and cannot place the instance, while a return to ``InService``
    at the old count means it gave up without recording a failure anywhere.
    """
    head = (
        f"{endpoint} did not reach more than {from_instances} instance(s) within "
        f"{max_wait_s:.0f}s. "
    )
    if endpoint_status == "Updating":
        # That this can take so long with the quota free is what makes the mode worth
        # running — it isolates provisioning from everything upstream of it.
        return head + (
            "DesiredInstanceCount was set directly and the endpoint is still Updating, so "
            "SageMaker accepted the change and has not placed the instance. With quota free, "
            "that points at instance capacity for the type rather than at anything in the "
            "scaling config. Check `aws sagemaker describe-endpoint` for a FailureReason, "
            "and consider another instance type or region."
        )
    return head + (
        f"DesiredInstanceCount was set directly, so no policy was involved and the scaling "
        f"activity log is silent by design. The endpoint reports "
        f"{endpoint_status or 'an unknown status'}: if it is back to InService at the old "
        "count, SageMaker abandoned the change without recording a failure."
    )


def _start_probe(
    *,
    model_name: str,
    endpoint: str,
    region: str,
    variant: str,
    voice: str,
    texts: Sequence[str],
    transport: str,
    concurrency: int,
    duration_s: float,
    run_id: str,
    seed: int | None,
    sink: Callable[[LoadEvent], None],
    extra_sink: Callable[[LoadEvent], None] | None,
    stop_event: threading.Event,
    sagemaker: BaseClient,
) -> tuple[threading.Thread, datetime]:
    """Start the closed-loop probe on a background thread; return it with its start time.

    Background, because the capacity poll has to run concurrently: the whole point is to
    watch AWS react *while* load is being offered. ``run_step`` blocks for its full
    duration, so the alternative would be measuring a scale-out that already finished.

    Closed-loop at exactly ``concurrency``, which is why nothing here converts to a rate:
    each of N workers issues its next request when its previous one returns, so queued plus
    executing is N by construction and the comparison against the ladder is like-for-like.
    """
    from tts_bench.bidi import invoke_for, make_client_for
    from tts_bench.loadgen import build_text_pool, make_instance_count_fetcher, run_step

    logger.info(
        "Holding N={} concurrent for up to {:.0f}s on {}", concurrency, duration_s, transport
    )
    # One connection per worker plus slack: a pool smaller than N would queue in the client
    # and show up as endpoint latency.
    client = make_client_for(transport, region, max_pool=concurrency + 4)
    invoke = invoke_for(transport)
    pool = build_text_pool(texts, seed=seed)

    def _sink(event: LoadEvent) -> None:
        sink(event)
        if extra_sink is not None:
            extra_sink(event)

    def _drive() -> None:
        try:
            run_step(
                client,
                model=model_name,
                endpoint=endpoint,
                voice=voice,
                texts=pool,
                concurrency=concurrency,
                duration_s=duration_s,
                run_id=run_id,
                # The tripwire `qmax` uses to *invalidate* a run is the signal here:
                # T_total is about capacity changing, so every event records the count.
                instance_count_fetch=make_instance_count_fetcher(sagemaker, endpoint, variant),
                event_sink=_sink,
                stop_event=stop_event,
                invoke=invoke,
            )
        except Exception:
            # Logged, not raised: this runs on a daemon thread where an exception would
            # otherwise vanish, and the events already collected are still worth reporting.
            logger.exception("Probe failed; the report will cover only what it wrote")

    driver = threading.Thread(target=_drive, name="ttotal-probe", daemon=True)
    started_at = datetime.now(UTC)
    driver.start()
    return driver, started_at


def render_text(report: TTotalReport) -> str:
    """Human-readable timeline, for the CLI."""
    total = report.t_total_s
    lines = [
        f"T_total for {report.endpoint} ({report.model_name}), run {report.run_id}:",
        (
            f"  {total:.1f}s from capacity requested to traffic recovered"
            if total is not None
            else "  not measurable from the stages observed"
        ),
    ]
    if report.t_total_bounded:
        lines.append(
            "  FLOOR: recovery was never bounded, so this spans only to in_service. See notes."
        )
    with_bound = report.t_total_with_policy_bound_s
    if with_bound is not None:
        # Printed as its own line, and labelled a bound rather than a measurement: this is
        # the number the planner sizes headroom against, and it is part measured, part
        # assumed. Folding the two together is how a bound comes to be quoted as data.
        lines.append(
            f"  {with_bound:.1f}s including a {POLICY_LAG_BOUND_S:.0f}s BOUND on the "
            "policy's detection lag (not measured — the policy was suspended)"
        )
    lines.append(
        f"  instances {report.from_instances} -> {report.to_instances}"
        + (f", new instance {report.instance_id}" if report.instance_id else "")
        + f", trigger {report.trigger}, probe N={report.probe_concurrency}"
    )

    aws_share, container_share = report.aws_share_s, report.container_share_s
    if aws_share is not None or container_share is not None:
        aws_txt = f"{aws_share:.0f}s" if aws_share is not None else "n/a"
        container_txt = f"{container_share:.0f}s" if container_share is not None else "n/a"
        lines.append(f"  AWS half {aws_txt}, container half {container_txt}")

    if report.p95_before_ms is not None or report.p95_after_ms is not None:
        before = f"{report.p95_before_ms:.0f}ms" if report.p95_before_ms is not None else "n/a"
        after = f"{report.p95_after_ms:.0f}ms" if report.p95_after_ms is not None else "n/a"
        lines.append(
            f"  probe p95 TTFAB {before} ({report.requests_before} reqs) -> "
            f"{after} ({report.requests_after} reqs)"
        )
        # The ladder's own values for the same two concurrencies, so the halving is checkable
        # rather than asserted: before should sit near the N rung, after near the N/2 one.
        expected = report.p95_expected_before_ms
        target = report.p95_recovered_target_ms
        if expected is not None and target is not None:
            lines.append(
                f"  ladder expected {expected:.0f}ms at N={report.probe_concurrency}, "
                f"recovered at or under {target:.0f}ms (its N={report.probe_concurrency // 2} "
                f"rung x {RECOVERY_TOLERANCE:.2f})"
            )

    lines.extend(["", "  stage breakdown:"])
    for duration in report.durations:
        share = f"{duration.seconds / total:5.1%}" if total else "    -"
        mark = "~" if duration.bounded else " "
        lines.append(
            f"   {share} {mark} {duration.seconds:7.1f}s  "
            f"{duration.from_stage} -> {duration.to_stage}"
        )
    if not report.durations:
        lines.append("   (fewer than two stages were observed)")

    dominant = report.dominant_stage
    if dominant is not None:
        lines.extend(
            [
                "",
                f"  dominant stage: {dominant.from_stage} -> {dominant.to_stage} "
                f"({dominant.seconds:.1f}s) — attack this to shrink T_total",
            ]
        )

    if report.missing_stages:
        lines.extend(["", f"  not observed: {', '.join(report.missing_stages)}"])
    if report.notes:
        lines.append("")
        lines.extend(f"  note: {note}" for note in report.notes)

    lines.extend(["", "  ~ marks a stage bounded by inference rather than timestamped directly."])
    return "\n".join(lines)
