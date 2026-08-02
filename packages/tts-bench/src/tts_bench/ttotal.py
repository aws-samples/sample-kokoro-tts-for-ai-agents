"""Measure ``T_total``: the lag from load arriving to new capacity serving it.

``T_total`` is the number the whole capacity plan is most sensitive to. It sets how
much standing headroom a ``k``-fold surge needs (:func:`shared.capacity.c_target`), how
much of that surge a queue can absorb (:func:`shared.capacity.queue_covers_surge`), and
therefore what the fleet costs. A plan built on a guessed ``T_total`` is a guess.

A single number is not actionable, so this decomposes the lag into stages, each bounded
by a timestamp from a different API:

=========================  ==========================================================
Stage                      Source
=========================  ==========================================================
``load_applied``           the load generator's own start time
``metric_published``       ``ConcurrentRequestsPerModel`` first reaching the target
``alarm_fired``            ``DescribeAlarmHistory`` transition to ALARM
``activity_started``       ``DescribeScalingActivities`` ``StartTime``
``instance_logging``       the new log stream's ``firstEventTimestamp``
``container_start`` ...    ``=== STAGE ... ===`` markers from that stream
  ... ``ready``
``in_service``             ``DescribeEndpoint`` ``CurrentInstanceCount``
``traffic_recovered``      client-side p95 recovery — **bounded, not measured**
=========================  ==========================================================

**What is not observable, and is reported as such.** SageMaker does not reveal which
instance served a request, so "the new instance took its first real traffic" cannot be
read directly. It is bounded by subtraction from the client side instead: when sustained
p95 came back inside budget. Every such stage carries ``bounded=True`` and the reason,
because a plan is allowed to build on a bound but not to mistake one for a measurement.

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
    HIGH_RES_PERIOD_S,
    SAGEMAKER_NAMESPACE,
    SCALABLE_DIMENSION,
    SERVICE_NAMESPACE,
    LogStream,
    MetricSpec,
    Stat,
    alarm_transitions,
    fetch_metric,
    first_datapoint_at_or_above,
    list_log_streams,
    log_group_name,
    new_streams_since,
    resource_id,
    scaling_activities,
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
"""Load held past the scale event, so the recovery bound has traffic on the far side of
it. Without this the run ends at ``in_service`` and ``T_total`` stops short of the thing
it is supposed to measure."""

DEFAULT_LOAD_MULTIPLE = 3.0
"""Offered concurrency as a multiple of ``C_target``. Above 1 by enough that the alarm
crosses on its first evaluation period instead of hovering at the threshold — and, for a
model whose ``C_target`` is a fraction of ``C_max``, enough that latency actually
degrades, so that its recovery is visible at all."""

RECOVERY_WINDOW_S = 60.0
"""How long p95 must hold inside budget to count as recovered. One good request proves
nothing; a minute of them is a recovery."""

RECOVERY_MIN_SAMPLES = 5
"""Completions required in a recovery window, so a lull does not read as a recovery."""

TRIGGER_DRIVE_LOAD = "drive-load"
TRIGGER_FORCE_DESIRED = "force-desired"


class TimelineStage(StrEnum):
    """Stages of the scaling lag, in the order they normally occur.

    Distinct from :class:`shared.stages.Stage`, which names *container* startup stages
    only. This enum spans the whole lag, of which those are the middle third.
    """

    LOAD_APPLIED = "load_applied"
    """The client began offering the raised rate. The clock starts here."""

    METRIC_PUBLISHED = "metric_published"
    """CloudWatch first shows concurrency at or above the scaling target."""

    ALARM_FIRED = "alarm_fired"
    """The policy's alarm went to ALARM, so the policy has decided."""

    ACTIVITY_STARTED = "activity_started"
    """Application Auto Scaling began changing ``DesiredInstanceCount``."""

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
    """Client-side p95 back inside budget. Bounded, not measured."""


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

CONCURRENCY_METRIC = MetricSpec(
    SAGEMAKER_NAMESPACE,
    "ConcurrentRequestsPerModel",
    stats=(Stat.MAXIMUM, Stat.AVERAGE),
)
"""The metric the scale-out policy tracks, at the resolution the policy sees it.
Measuring the same quantity the policy acts on is what makes ``metric_published`` the
policy's own detection lag rather than an unrelated observation."""


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

    Two totals, because the two plausible definitions differ by a stage that is real lag
    the fleet has to absorb:

    * :attr:`t_total_s` — load applied through to traffic recovered. What the planner
      should consume: a surge starts when load arrives, not when CloudWatch notices it.
    * :attr:`t_total_from_metric_s` — metric publication through to traffic recovered.
      The narrower reading, useful for attributing lag to AWS versus to the client's own
      ramp.

    Both are the *span*, not the sum of the stages, which would silently swallow any gap
    between two APIs' clocks.
    """

    model_name: str
    endpoint: str
    run_id: str
    trigger: str

    deployed_config: dict[str, Any] = field(default_factory=dict)
    """Fingerprint of the configuration this lag was measured against, read from the
    endpoint. See ``fixture.DeployedConfig``.

    Carried for the same reason ``CMaxReport`` carries one, and for one more: the
    planner consumes a ``C_max`` curve and a ``T_total`` lag *together*, so without a
    fingerprint on both sides there is nothing to compare and a g5 curve can be paired
    with a g6 lag silently. Container start dominates this measurement and is a
    property of the image, which is exactly what the digest pins."""

    timeline: list[StageTime] = field(default_factory=list)
    instance_id: str | None = None
    from_instances: int = 0
    to_instances: int = 0

    p95_before_ms: float | None = None
    """Client p95 TTFAB while the surge was unserved — the degradation being recovered
    from. Without it, a recovery figure has nothing to be a recovery *from*."""

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

        Mirrors :attr:`tts_bench.types.CMaxReport.config_slug` so the two artifact
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
        """Load applied through to traffic recovered, in seconds."""
        span = self._span_from(TimelineStage.LOAD_APPLIED)
        if span is not None:
            return span
        # Without a load_applied stage, fall back to the full observed span rather than
        # reporting nothing — a report rebuilt from a past window still has stages.
        observed = self.observed_stages
        if len(observed) < 2:
            return None
        first, last = observed[0].at, observed[-1].at
        assert first is not None and last is not None
        return (last - first).total_seconds()

    @property
    def t_total_from_metric_s(self) -> float | None:
        """Metric publication through to traffic recovered, in seconds."""
        return self._span_from(TimelineStage.METRIC_PUBLISHED)

    @property
    def t_total_bounded(self) -> bool:
        """Whether ``t_total_s`` rests on an inferred endpoint.

        True when recovery was never observed, or was itself only bounded. A bounded
        ``T_total`` is usable — but a bound that stops at ``in_service`` *under*-reports
        the lag, which is the dangerous direction for a capacity plan, so it is flagged.
        """
        recovered = self.entry(TimelineStage.TRAFFIC_RECOVERED)
        if recovered is None or recovered.at is None:
            return True
        return recovered.bounded

    @property
    def dominant_stage(self) -> StageDuration | None:
        """The single longest stage — where attacking ``T_total`` actually pays off."""
        durations = self.durations
        return max(durations, key=lambda d: d.seconds) if durations else None

    @property
    def aws_share_s(self) -> float | None:
        """Lag before the container ran: detection, alarm, activity, provision, pull.

        Split out because it is the half no container change can shorten. Ends at
        ``container_started`` when the container reported it, otherwise at the log stream
        opening, which is a looser bound but better than nothing.
        """
        start = self.at(TimelineStage.LOAD_APPLIED) or self.at(TimelineStage.METRIC_PUBLISHED)
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

        Measured, not derived: it is an observation of a live scale event. The note
        records the trigger and whether the endpoint was bounded, because a
        ``--force-desired`` number omits the AWS half entirely and must not be mistaken
        for a full ``T_total``.
        """
        from tts_bench.types import Origin, Provenance

        bits = [f"trigger={self.trigger}"]
        if self.trigger == TRIGGER_FORCE_DESIRED:
            bits.append("container half only, no policy involved")
        if self.t_total_bounded:
            bits.append("recovery inferred, so this is a floor")
        if self.missing_stages:
            bits.append(f"{len(self.missing_stages)} stage(s) not observed")
        applied_at = self.at(TimelineStage.LOAD_APPLIED)
        return Provenance(
            origin=Origin.MEASURED,
            run_id=self.run_id,
            measured_at=applied_at.isoformat() if applied_at else None,
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
            "t_total_from_metric_s": self.t_total_from_metric_s,
            "t_total_bounded": self.t_total_bounded,
            "aws_share_s": self.aws_share_s,
            "container_share_s": self.container_share_s,
            "instance_id": self.instance_id,
            "from_instances": self.from_instances,
            "to_instances": self.to_instances,
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


def recovery_bound(
    events: Sequence[LoadEvent],
    *,
    in_service_at: datetime,
    budget_ms: float,
    window_s: float = RECOVERY_WINDOW_S,
    min_samples: int = RECOVERY_MIN_SAMPLES,
) -> tuple[datetime | None, str]:
    """Earliest time after ``in_service_at`` at which p95 held inside budget.

    The bound on ``traffic_recovered``. SageMaker does not say which instance served a
    request, so "when did the new instance take real traffic" is answered from the client
    side: when the fleet's tail latency came back and stayed back. That is an upper bound
    on the new instance being useful, and a lower bound on nothing — stated as such in
    the returned explanation, which the report carries verbatim.

    Returns:
        ``(timestamp_or_none, explanation)``. The explanation is kept whether or not a
        bound was found, because "never recovered" is the more important result.
    """
    samples = [pair for pair in _ttfab_samples(events) if pair[0] >= in_service_at]
    if len(samples) < min_samples:
        return None, (
            f"only {len(samples)} completion(s) with a TTFAB after the new instance came "
            f"into service; {min_samples} are needed to bound recovery. Hold load longer "
            "past the scale event."
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
                f"p95 TTFAB held at or under {budget_ms:.0f}ms across {len(window)} "
                f"completions in the following {window_s:.0f}s"
            )

    return None, (
        f"p95 TTFAB never held under {budget_ms:.0f}ms for {window_s:.0f}s after the new "
        "instance came into service, so recovery is not bounded from above and T_total is "
        "a floor. Either the added instance did not relieve the surge, or the offered rate "
        "exceeded what even the grown fleet can serve."
    )


def assemble_timeline(
    *,
    load_applied_at: datetime | None,
    metric_published_at: datetime | None,
    metric_note: str | None,
    alarm_fired_at: datetime | None,
    alarm_note: str | None,
    activity_started_at: datetime | None,
    activity_note: str | None,
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
    """
    timeline: list[StageTime] = [
        StageTime(
            stage=str(TimelineStage.LOAD_APPLIED),
            at=load_applied_at,
            source="load generator start",
        ),
        StageTime(
            stage=str(TimelineStage.METRIC_PUBLISHED),
            at=metric_published_at,
            source=f"{CONCURRENCY_METRIC.name} at {HIGH_RES_PERIOD_S}s resolution",
            note=metric_note,
        ),
        StageTime(
            stage=str(TimelineStage.ALARM_FIRED),
            at=alarm_fired_at,
            source="DescribeAlarmHistory",
            note=alarm_note,
        ),
        StageTime(
            stage=str(TimelineStage.ACTIVITY_STARTED),
            at=activity_started_at,
            source="DescribeScalingActivities",
            note=activity_note,
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


def policy_alarm_names(
    appscaling: BaseClient,
    *,
    endpoint: str,
    variant: str = DEFAULT_VARIANT,
) -> list[str]:
    """Alarm names the live scaling policies own.

    Target tracking creates its alarms implicitly with generated names, so they cannot be
    predicted from config — they have to be read back off the policy.
    """
    try:
        policies = appscaling.describe_scaling_policies(
            ServiceNamespace=SERVICE_NAMESPACE,
            ResourceId=resource_id(endpoint, variant),
            ScalableDimension=SCALABLE_DIMENSION,
        ).get("ScalingPolicies", [])
    except Exception as exc:  # noqa: BLE001 - alarm names are for attribution only
        logger.warning("Could not read scaling policies for {}: {}", endpoint, exc)
        return []

    return [
        name
        for policy in policies
        for alarm in policy.get("Alarms", [])
        if (name := alarm.get("AlarmName"))
    ]


def deployed_target_value(
    appscaling: BaseClient,
    *,
    endpoint: str,
    variant: str = DEFAULT_VARIANT,
) -> float | None:
    """The ``TargetValue`` the live target-tracking policy is actually using.

    Read off the deployment rather than taken from ``speech_infra.config``, because the
    figure this measurement needs is the one the policy will act on. A config that has
    moved ahead of the last ``cdk deploy`` would otherwise put ``metric_published`` at
    the wrong threshold and mis-attribute detection lag.

    Returns:
        ``None`` when no target-tracking policy exists, which is a normal state under
        ``--trigger force-desired``.
    """
    try:
        policies = appscaling.describe_scaling_policies(
            ServiceNamespace=SERVICE_NAMESPACE,
            ResourceId=resource_id(endpoint, variant),
            ScalableDimension=SCALABLE_DIMENSION,
        ).get("ScalingPolicies", [])
    except Exception as exc:  # noqa: BLE001 - the caller can still supply a target
        logger.warning("Could not read scaling policies for {}: {}", endpoint, exc)
        return None

    for policy in policies:
        config = policy.get("TargetTrackingScalingPolicyConfiguration")
        if config and config.get("TargetValue") is not None:
            return float(config["TargetValue"])
    return None


def collect_timeline(
    *,
    cloudwatch: BaseClient,
    appscaling: BaseClient,
    logs: BaseClient,
    endpoint: str,
    variant: str,
    model_name: str,
    run_id: str,
    trigger: str,
    scaling_target: float,
    ttfab_budget_ms: float,
    load_applied_at: datetime | None,
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

    Args:
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
    )
    policy_driven = trigger != TRIGGER_FORCE_DESIRED

    metric_published_at: datetime | None = None
    metric_note: str | None = None
    if policy_driven:
        series = fetch_metric(
            cloudwatch,
            CONCURRENCY_METRIC,
            endpoint=endpoint,
            variant=variant,
            start=window_start,
            end=window_end,
            period_s=HIGH_RES_PERIOD_S,
        )
        metric_published_at = first_datapoint_at_or_above(series, scaling_target)
        if metric_published_at is None:
            metric_note = (
                f"concurrency never reached the scaling target of {scaling_target:.3f} in "
                "CloudWatch"
            )
            report.notes.append(
                f"{metric_note}, so the detection stage is unattributed even though capacity "
                "did change. The offered rate may have been too low, or the "
                f"{HIGH_RES_PERIOD_S}s periods may have averaged the surge away."
            )
    else:
        metric_note = "not applicable: capacity was set directly"
        report.notes.append(
            f"trigger was --{TRIGGER_FORCE_DESIRED}, so no metric, alarm, or policy stage "
            "exists. The AWS detection half of T_total is NOT measured here, and this "
            "number must not be fed to the planner as a full T_total."
        )

    alarm_fired_at: datetime | None = None
    alarm_note: str | None = None
    if policy_driven:
        names = policy_alarm_names(appscaling, endpoint=endpoint, variant=variant)
        to_alarm = [
            t
            for t in alarm_transitions(cloudwatch, names, start=window_start, end=window_end)
            if t.to_alarm
        ]
        if to_alarm:
            alarm_fired_at = to_alarm[0].timestamp
            alarm_note = f"alarm {to_alarm[0].alarm_name}"
        elif names:
            alarm_note = f"none of {len(names)} policy alarm(s) went to ALARM in the window"
            report.notes.append(f"{alarm_note}, so alarm evaluation is unattributed")
        else:
            alarm_note = "no policy alarms found to read"
    else:
        alarm_note = "not applicable: no policy was involved"

    activity_started_at: datetime | None = None
    activity_note: str | None = None
    activities = scaling_activities(
        appscaling, endpoint=endpoint, variant=variant, start=window_start, end=window_end
    )
    if activities:
        # The first *successful* activity, falling back to the first of any kind. A
        # policy blocked by an account quota records a Failed activity every ten
        # seconds, so taking activities[0] unconditionally would time the attempt that
        # was rejected rather than the one that added capacity.
        succeeded = [a for a in activities if a.succeeded]
        chosen = succeeded[0] if succeeded else activities[0]
        activity_started_at = chosen.start_time
        # The description is carried verbatim rather than pattern-matched: AWS words it
        # differently across capacity changes, and a filter that guessed wrong would
        # silently drop the only record of when the change began.
        activity_note = f"{chosen.description} ({chosen.status_code})"

        failed = [a for a in activities if a.failed]
        if failed:
            # The failure mode is silent from the endpoint's side: it stays InService at
            # its old count while the policy retries. The StatusMessage is the only place
            # the reason appears, so it is quoted rather than summarized.
            report.notes.append(
                f"{len(failed)} of {len(activities)} scaling activities FAILED. First "
                f"reason: {failed[0].status_message or failed[0].description!r}. The policy "
                "acted and SageMaker refused; nothing about this is visible on the endpoint, "
                "which stays InService at its old count."
            )
        if not succeeded:
            # An in-flight activity is neither: AWS took the change and is applying it.
            # Reporting that as a failure would blame the account for a slow image pull.
            in_flight = [a for a in activities if a.in_flight]
            activity_note += (
                f" — still {in_flight[0].status_code} when read"
                if in_flight
                else " — no activity succeeded"
            )
    elif policy_driven:
        activity_note = "no scaling activity recorded in the window"
        report.notes.append(
            f"{activity_note}; capacity changed without Application Auto Scaling recording "
            "it, which is worth investigating on its own"
        )
    else:
        activity_note = "not applicable: capacity was set directly"

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
            load_events, in_service_at=event.in_service_at, budget_ms=ttfab_budget_ms
        )
        samples = _ttfab_samples(load_events)
        floor = load_applied_at or window_start
        before = [v for t, v in samples if floor <= t < event.in_service_at]
        after = [v for t, v in samples if t >= event.in_service_at]
        report.requests_before = len(before)
        report.requests_after = len(after)
        if before:
            report.p95_before_ms = float(np.percentile(before, 95))
        if after:
            report.p95_after_ms = float(np.percentile(after, 95))

    report.timeline = assemble_timeline(
        load_applied_at=load_applied_at,
        metric_published_at=metric_published_at,
        metric_note=metric_note,
        alarm_fired_at=alarm_fired_at,
        alarm_note=alarm_note,
        activity_started_at=activity_started_at,
        activity_note=activity_note,
        stream=stream,
        container_stages=container_stages,
        in_service_at=event.in_service_at,
        recovered_at=recovered_at,
        recovery_note=recovery_note,
    )

    if report.t_total_bounded:
        report.notes.append(
            "T_total rests on an inferred endpoint rather than an observed recovery, so it "
            "under-reports rather than over-reports. Treat it as a floor when sizing "
            "headroom."
        )
    return report


def measure(
    *,
    model_name: str,
    endpoint: str,
    variant: str = DEFAULT_VARIANT,
    region: str = "us-east-1",
    scaling_target: float,
    ttfab_budget_ms: float,
    s_mean_s: float,
    texts: Sequence[str],
    voice: str,
    trigger: str = TRIGGER_DRIVE_LOAD,
    load_multiple: float = DEFAULT_LOAD_MULTIPLE,
    max_wait_s: float = DEFAULT_MAX_WAIT_S,
    settle_s: float = DEFAULT_SETTLE_S,
    poll_interval_s: float = DEFAULT_POLL_INTERVAL_S,
    run_id: str | None = None,
    transport: str = "bidi",
    seed: int | None = 1234,
    event_sink: Callable[[LoadEvent], None] | None = None,
    cloudwatch: BaseClient | None = None,
    appscaling: BaseClient | None = None,
    sagemaker: BaseClient | None = None,
    logs: BaseClient | None = None,
    quotas: BaseClient | None = None,
) -> TTotalReport:
    """Trigger one scale-out and measure the lag, stage by stage.

    Guards with :func:`tts_bench.fixture.require_scalable` first, which fails in seconds if
    no scale event is possible — rather than after twenty minutes of load. Restores the
    starting desired count on every exit path, Ctrl-C included.

    Args:
        trigger: ``"drive-load"`` offers ``load_multiple x scaling_target`` concurrency so
            the deployed policy fires, measuring both halves. ``"force-desired"`` raises
            ``DesiredInstanceCount`` directly, needs no policy, and measures the container
            half only — the fast path for iterating on the log parser.
        scaling_target: The deployed ``C_target``. Used both to size the offered load and to
            detect when CloudWatch first showed the crossing, so ``metric_published`` is
            the policy's own detection lag rather than an unrelated observation.
        s_mean_s: Measured mean service time, which converts a target concurrency into an
            arrival rate. The load generator is open-loop, so it needs a rate.

    Raises:
        TTotalError: If no scale event occurred within ``max_wait_s``.
        tts_bench.fixture.FixtureError: If scaling could not have fired.
        ValueError: If ``trigger`` is not one of the two supported modes.
    """
    import boto3

    from tts_bench import fixture

    if trigger not in (TRIGGER_DRIVE_LOAD, TRIGGER_FORCE_DESIRED):
        raise ValueError(
            f"trigger must be {TRIGGER_DRIVE_LOAD!r} or {TRIGGER_FORCE_DESIRED!r}, got {trigger!r}"
        )

    run_id = run_id or uuid.uuid4().hex[:12]
    cloudwatch = cloudwatch or boto3.client("cloudwatch", region_name=region)
    appscaling = appscaling or boto3.client("application-autoscaling", region_name=region)
    sagemaker = sagemaker or boto3.client("sagemaker", region_name=region)
    logs = logs or boto3.client("logs", region_name=region)

    if trigger == TRIGGER_DRIVE_LOAD:
        # Cheap precondition: a target exists, is not suspended, can reach at least two
        # instances, has policies attached, and the account has quota room for the jump
        # the policy will request. Raises FixtureError naming the fix. Seconds, against a
        # run that otherwise costs max_wait_s of load to discover the same thing.
        fixture.require_scalable(
            endpoint,
            region=region,
            variant=variant,
            appscaling=appscaling,
            sagemaker=sagemaker,
            quotas=quotas,
        )

    # Read before the run rather than after: most of this measurement is container start,
    # so if a redeploy lands mid-run the lag belongs to the image that was serving when it
    # started, not to whatever replaced it.
    deployed = fixture.fingerprint_or_registry(
        model_name, endpoint=endpoint, region=region, variant=variant, sagemaker=sagemaker
    )

    desired_before, current_before = read_capacity(sagemaker, endpoint, variant)
    # Captured before anything changes: the set difference against this is what names the
    # new instance afterwards.
    streams_before = list_log_streams(logs, endpoint=endpoint, variant=variant)
    logger.info(
        "T_total run {} on {}: {} instance(s), {} known log stream(s), trigger={}",
        run_id,
        endpoint,
        current_before,
        len(streams_before),
        trigger,
    )

    window_start = datetime.now(UTC)
    load_applied_at: datetime | None = None
    load_events: list[LoadEvent] = []
    stop_event = threading.Event()
    driver: threading.Thread | None = None
    # Bound before the try so the finally can read them on any exit path, Ctrl-C included.
    event = ScaleEvent(
        from_instances=current_before,
        to_instances=current_before,
        desired_changed_at=None,
        in_service_at=None,
    )
    timed_out = False
    diagnosis: str | None = None

    try:
        if trigger == TRIGGER_FORCE_DESIRED:
            load_applied_at = datetime.now(UTC)
            set_desired_count(
                sagemaker, endpoint=endpoint, variant=variant, to_instances=current_before + 1
            )
        else:
            driver, load_applied_at = _start_load(
                model_name=model_name,
                endpoint=endpoint,
                region=region,
                variant=variant,
                voice=voice,
                texts=texts,
                transport=transport,
                scaling_target=scaling_target,
                load_multiple=load_multiple,
                s_mean_s=s_mean_s,
                duration_s=max_wait_s + settle_s,
                run_id=run_id,
                seed=seed,
                sink=load_events.append,
                extra_sink=event_sink,
                stop_event=stop_event,
                sagemaker=sagemaker,
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
        if event.occurred and driver is not None:
            # Keep offering load past the event so the recovery bound has data on the far
            # side of it. Without this, T_total stops at in_service.
            logger.info("Holding load {:.0f}s to bound recovery", settle_s)
            time.sleep(settle_s)
    finally:
        stop_event.set()
        if driver is not None:
            driver.join(timeout=120.0)
            if driver.is_alive():
                logger.warning(
                    "Load driver still running after 120s; requests may still be arriving at {}",
                    endpoint,
                )
        # Read the activities BEFORE restoring. Restoring is itself a capacity change, and
        # Application Auto Scaling marks a still-running activity Overridden when one
        # arrives — so a restore that works destroys the evidence of why the run timed out,
        # turning "InProgress for 34 minutes" into an ambiguous "Overridden".
        window_end = datetime.now(UTC)
        # Only on a timeout: a Ctrl-C never sets this, so an interrupted run restores
        # immediately instead of spending API calls explaining a scale-out nobody waited for.
        if timed_out:
            diagnosis = _explain_no_scale_out(
                appscaling,
                endpoint=endpoint,
                variant=variant,
                from_instances=current_before,
                max_wait_s=max_wait_s,
                scaling_target=scaling_target,
                window_start=window_start,
                window_end=window_end,
                trigger=trigger,
                endpoint_status=_endpoint_status(sagemaker, endpoint),
            )
        restore_desired_count(
            sagemaker, endpoint=endpoint, variant=variant, to_instances=desired_before
        )

    if diagnosis is not None:
        # A timeout has several very different causes — the policy never acted, it acted
        # and was refused, or AWS took the change and never delivered — and only the
        # activity log distinguishes them. Say which, rather than leaving the operator to
        # guess after a run that already cost real minutes of load.
        raise TTotalError(diagnosis)

    return collect_timeline(
        cloudwatch=cloudwatch,
        appscaling=appscaling,
        logs=logs,
        endpoint=endpoint,
        variant=variant,
        model_name=model_name,
        run_id=run_id,
        trigger=trigger,
        scaling_target=scaling_target,
        ttfab_budget_ms=ttfab_budget_ms,
        load_applied_at=load_applied_at,
        window_start=window_start,
        window_end=window_end,
        streams_before=streams_before,
        event=event,
        load_events=load_events,
        deployed_config=deployed.to_dict(),
    )


def _explain_no_scale_out(
    appscaling: BaseClient,
    *,
    endpoint: str,
    variant: str,
    from_instances: int,
    max_wait_s: float,
    scaling_target: float,
    window_start: datetime,
    window_end: datetime,
    trigger: str = TRIGGER_DRIVE_LOAD,
    endpoint_status: str | None = None,
) -> str:
    """Why no scale-out happened, distinguishing "never acted" from "was refused".

    Several outcomes, not two, and they are ordered by how much they cost to fix. The
    endpoint looks the same in all of them — ``InService`` or ``Updating`` at its old
    count — but a refusal needs the account changed, an in-flight activity needs only a
    longer wait, and no activity at all means the policy never decided. A ``Failed``
    activity's ``StatusMessage`` carries AWS's own reason, quoted rather than classified
    because the set of reasons is AWS's to extend.

    Under ``force-desired`` there is no policy in the path at all, so the activity log is
    silent by design and reading anything into its silence would be wrong. That mode is
    diagnosed from the endpoint's own status instead.
    """
    head = (
        f"{endpoint} did not reach more than {from_instances} instance(s) within "
        f"{max_wait_s:.0f}s. "
    )
    if trigger == TRIGGER_FORCE_DESIRED:
        # No policy was involved, so "no scaling activity" is expected rather than a
        # finding. Updating means SageMaker took the change and is trying to place the
        # instance; that it can take this long with the quota free is what makes the mode
        # worth running — it isolates provisioning from everything upstream of it.
        if endpoint_status == "Updating":
            return head + (
                "DesiredInstanceCount was set directly and the endpoint is still Updating, "
                "so SageMaker accepted the change and has not placed the instance. With "
                "quota free, that points at instance capacity for the type rather than at "
                "anything in the scaling config. Check `aws sagemaker describe-endpoint` "
                "for a FailureReason, and consider another instance type or region."
            )
        return head + (
            f"DesiredInstanceCount was set directly, so no policy was involved and the "
            f"scaling activity log is silent by design. The endpoint reports "
            f"{endpoint_status or 'an unknown status'}: if it is back to InService at the "
            "old count, SageMaker abandoned the change without recording a failure."
        )
    activities = scaling_activities(
        appscaling, endpoint=endpoint, variant=variant, start=window_start, end=window_end
    )
    failed = [a for a in activities if a.failed]
    if failed:
        return head + (
            f"The policy DID act — {len(failed)} of {len(activities)} scaling activities "
            f"failed, so this is not a detection problem. AWS gave: "
            f"{failed[0].status_message or failed[0].description}"
        )
    in_flight = [a for a in activities if a.in_flight]
    if in_flight:
        # Not a failure: AWS accepted the change and is still applying it. Pulling a
        # multi-GB image onto several instances at once routinely outlasts the default
        # wait, and calling that an error would send the operator after the wrong thing.
        #
        # Past a point, though, "still provisioning" stops being the likely story. A live
        # run sat InProgress for 34 minutes while EC2 could not place the instances, and
        # SageMaker neither failed the activity nor set a FailureReason — it simply
        # returned the endpoint to InService at the old count. Waiting longer would not
        # have helped, so beyond the threshold this says capacity, not patience.
        waited = (window_end - in_flight[0].start_time).total_seconds()
        aws_said = in_flight[0].status_message or in_flight[0].description
        if waited >= SLOW_PROVISION_SUSPICION_S:
            return head + (
                f"The change was ACCEPTED and has been {in_flight[0].status_code} for "
                f"{waited / 60:.0f}min — far longer than an image pull. AWS reserves the "
                f"quota when it accepts, so this reads as instance capacity being "
                f"unavailable for the type rather than a slow start, and it is reported "
                f"neither as a failed activity nor as an endpoint FailureReason. Check "
                f"whether CurrentInstanceCount ever moved; if not, try another instance "
                f"type or region rather than a longer --max-wait. AWS gave: {aws_said}"
            )
        return head + (
            f"The change was ACCEPTED and is still {in_flight[0].status_code} after "
            f"{waited:.0f}s — AWS is provisioning, not refusing. Retry with a longer "
            f"--max-wait, or lower max_capacity so fewer instances are pulled at once. "
            f"AWS gave: {aws_said}"
        )
    if activities:
        return head + (
            f"{len(activities)} scaling activities were recorded, none failed and none is "
            "still in flight, so capacity moved without the endpoint reaching the new "
            "count. Worth reading `describe-scaling-activities` directly."
        )
    return head + (
        f"No scaling activity was recorded at all, so the policy never acted. Check that "
        f"the offered concurrency exceeds C_target={scaling_target:.3f}, and that "
        "`tts-bench drift` shows the policy's alarm out of INSUFFICIENT_DATA."
    )


def _start_load(
    *,
    model_name: str,
    endpoint: str,
    region: str,
    variant: str,
    voice: str,
    texts: Sequence[str],
    transport: str,
    scaling_target: float,
    load_multiple: float,
    s_mean_s: float,
    duration_s: float,
    run_id: str,
    seed: int | None,
    sink: Callable[[LoadEvent], None],
    extra_sink: Callable[[LoadEvent], None] | None,
    stop_event: threading.Event,
    sagemaker: BaseClient,
) -> tuple[threading.Thread, datetime]:
    """Start the load generator on a background thread and return it with its start time.

    Background, because the capacity poll has to run concurrently: the whole point is to
    watch AWS react *while* load is being offered. ``run_step`` blocks for its full
    duration, so the alternative would be measuring a scale-out that already finished.
    """
    from tts_bench.bidi import invoke_for, make_client_for
    from tts_bench.loadgen import build_text_pool, make_instance_count_fetcher, run_step

    target_concurrency = scaling_target * load_multiple
    offered_rps = target_concurrency / s_mean_s
    # Headroom over the target so the *client* never becomes the bottleneck. A capped
    # dispatcher shows up as dispatch_skipped, which would look like the endpoint
    # absorbing load it never actually received.
    max_workers = max(int(target_concurrency * 4) + 8, 16)
    logger.info(
        "Offering {:.2f} concurrency ({:.2f} rps) against C_target {:.3f} for up to {:.0f}s",
        target_concurrency,
        offered_rps,
        scaling_target,
        duration_s,
    )

    client = make_client_for(transport, region, max_pool=max_workers)
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
                offered_rps=offered_rps,
                duration_s=duration_s,
                max_workers=max_workers,
                run_id=run_id,
                seed=seed,
                # The tripwire `cmax` uses to *invalidate* a run is the signal here:
                # T_total is about capacity changing, so every event records the count.
                instance_count_fetch=make_instance_count_fetcher(sagemaker, endpoint, variant),
                event_sink=_sink,
                stop_event=stop_event,
                invoke=invoke,
            )
        except Exception:
            # Logged, not raised: this runs on a daemon thread where an exception would
            # otherwise vanish, and the events already collected are still worth reporting.
            logger.exception("Load driver failed; the report will cover only what it wrote")

    driver = threading.Thread(target=_drive, name="ttotal-load", daemon=True)
    started_at = datetime.now(UTC)
    driver.start()
    return driver, started_at


def render_text(report: TTotalReport) -> str:
    """Human-readable timeline, for the CLI."""
    total = report.t_total_s
    lines = [
        f"T_total for {report.endpoint} ({report.model_name}), run {report.run_id}:",
        (
            f"  {total:.1f}s from load applied to traffic recovered"
            if total is not None
            else "  not measurable from the stages observed"
        ),
    ]
    if report.t_total_bounded:
        lines.append("  BOUNDED: recovery was inferred, so this is a floor. See notes below.")
    from_metric = report.t_total_from_metric_s
    if from_metric is not None:
        lines.append(f"  {from_metric:.1f}s from metric publication (the narrower reading)")
    lines.append(
        f"  instances {report.from_instances} -> {report.to_instances}"
        + (f", new instance {report.instance_id}" if report.instance_id else "")
        + f", trigger {report.trigger}"
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
            f"  client p95 TTFAB {before} ({report.requests_before} reqs) -> "
            f"{after} ({report.requests_after} reqs)"
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
