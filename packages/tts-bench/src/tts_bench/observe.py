"""Read-only observation of CloudWatch, Application Auto Scaling, and SageMaker.

Four jobs, all strictly read-only:

1. **Join server-side metrics to a load step** (:func:`fetch_window`). ``cmax``
   measures concurrency client-side; AWS publishes ``ConcurrentRequestsPerModel``
   independently. When the two disagree the bottleneck is in *our* dispatcher, not
   the server, and the resulting ``C_max`` describes the benchmark rather than the
   model — see :func:`concurrency_agreement`.
2. **Supply ``ttotal``'s timeline** (:func:`first_datapoint_at_or_above`,
   :func:`alarm_transitions`, :func:`scaling_activities`). Each stage boundary is
   a timestamp from a different API; this module fetches them, ``ttotal.py``
   assembles them.
3. **Audit deployed scaling config against source** (:func:`audit_scaling`).
4. **Read container startup logs** (:func:`list_log_streams`, :func:`stage_markers`).
   ``T_total``'s second half happens inside the container, where no metric reaches;
   the only externally visible record is what the container printed.

Nothing here mutates anything. Freeze/thaw lives in :mod:`tts_bench.fixture`.

:func:`audit_scaling` is deliberately pure — it takes already-fetched targets,
policies, and alarms — so the reconciliation rules are table-testable without
boto3. ``speech_infra.config`` is resolved by the *caller* and passed in as
:class:`ExpectedScaling`, keeping ``aws-cdk-lib`` out of this package's import
graph while still letting ``drift`` compare against what CDK describes.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import TYPE_CHECKING

import botocore.exceptions
import numpy as np
from botocore.client import BaseClient
from loguru import logger

if TYPE_CHECKING:
    from shared.stages import StageMarker

SAGEMAKER_NAMESPACE = "AWS/SageMaker"
"""Invocation metrics. Published by SageMaker itself, no instrumentation needed."""

ENDPOINT_NAMESPACE = "/aws/sagemaker/Endpoints"
"""Host utilization metrics. A separate namespace, which is easy to get wrong."""

SERVICE_NAMESPACE = "sagemaker"
SCALABLE_DIMENSION = "sagemaker:variant:DesiredInstanceCount"
DEFAULT_VARIANT = "primary"

DEFAULT_SETTLE_DELAY_S = 120.0
"""Wait before fetching. CloudWatch aggregation lags, and a metric fetched
immediately after a step returns partial datapoints for the final period —
silently biasing the window average toward whatever the step ended on."""

HIGH_RES_PERIOD_S = 10
"""``ConcurrentRequestsPerModel`` is published at 10s resolution, which is what
cuts scaling detection lag from ~90-150s to ~30s."""

DEFAULT_PERIOD_S = 60

_VALID_SUB_MINUTE_PERIODS = (1, 5, 10, 30)
_MAX_DATAPOINTS = 1440
"""GetMetricStatistics hard limit. Exceeding it returns an empty result rather
than an error, so it must be checked here or a long window looks like a dead
endpoint."""


def _validate_period(period_s: int) -> None:
    if period_s <= 0:
        raise ValueError(f"period_s must be positive, got {period_s}")
    if period_s < 60 and period_s not in _VALID_SUB_MINUTE_PERIODS:
        raise ValueError(
            f"sub-minute period must be one of {_VALID_SUB_MINUTE_PERIODS}, got {period_s}"
        )
    if period_s >= 60 and period_s % 60 != 0:
        raise ValueError(f"period_s >= 60 must be a multiple of 60, got {period_s}")


class Stat(StrEnum):
    """CloudWatch statistic. ``p*`` values go in ``ExtendedStatistics``."""

    AVERAGE = "Average"
    MAXIMUM = "Maximum"
    MINIMUM = "Minimum"
    SUM = "Sum"
    SAMPLE_COUNT = "SampleCount"
    P50 = "p50"
    P90 = "p90"
    P95 = "p95"
    P99 = "p99"

    @property
    def is_extended(self) -> bool:
        return self.value.startswith("p")


@dataclass(frozen=True, slots=True)
class MetricSpec:
    """A metric to fetch, and how to aggregate it."""

    namespace: str
    name: str
    stats: tuple[Stat, ...] = (Stat.AVERAGE, Stat.MAXIMUM)
    unit: str | None = None

    @property
    def key(self) -> str:
        return self.name


#: Server-side load and error metrics. `ConcurrentRequestsPerModel` is the one the
#: scaling policy uses, so measuring it here is what makes the plan's target value
#: refer to the same quantity the policy will see in production.
INVOCATION_METRICS: tuple[MetricSpec, ...] = (
    MetricSpec(SAGEMAKER_NAMESPACE, "ConcurrentRequestsPerModel"),
    MetricSpec(SAGEMAKER_NAMESPACE, "ModelLatency", (Stat.AVERAGE, Stat.P95, Stat.P99)),
    MetricSpec(SAGEMAKER_NAMESPACE, "OverheadLatency", (Stat.AVERAGE, Stat.P95)),
    MetricSpec(SAGEMAKER_NAMESPACE, "Invocations", (Stat.SUM,)),
    MetricSpec(SAGEMAKER_NAMESPACE, "Invocation5XXErrors", (Stat.SUM,)),
    MetricSpec(SAGEMAKER_NAMESPACE, "Invocation4XXErrors", (Stat.SUM,)),
)

#: Host utilization. Answers "what actually saturated" — a GPU-bound knee and a
#: lock-bound knee look identical from the client side but need opposite fixes.
UTILIZATION_METRICS: tuple[MetricSpec, ...] = (
    MetricSpec(ENDPOINT_NAMESPACE, "CPUUtilization"),
    MetricSpec(ENDPOINT_NAMESPACE, "MemoryUtilization"),
    MetricSpec(ENDPOINT_NAMESPACE, "GPUUtilization"),
    MetricSpec(ENDPOINT_NAMESPACE, "GPUMemoryUtilization"),
)


def endpoint_dimensions(endpoint: str, variant: str = DEFAULT_VARIANT) -> list[dict[str, str]]:
    """Dimensions for the variant-level aggregate of either namespace."""
    return [
        {"Name": "EndpointName", "Value": endpoint},
        {"Name": "VariantName", "Value": variant},
    ]


def resource_id(endpoint: str, variant: str = DEFAULT_VARIANT) -> str:
    """Application Auto Scaling resource id for an endpoint variant."""
    return f"endpoint/{endpoint}/variant/{variant}"


def parse_resource_id(rid: str) -> tuple[str, str] | None:
    """Split ``endpoint/<name>/variant/<variant>`` into its parts.

    Returns:
        ``(endpoint, variant)``, or ``None`` if this is not an endpoint-variant
        resource id — the ``sagemaker`` namespace also covers inference
        components, and treating one of those as an endpoint would report a
        phantom orphan.
    """
    parts = rid.split("/")
    if len(parts) == 4 and parts[0] == "endpoint" and parts[2] == "variant":
        return parts[1], parts[3]
    return None


@dataclass(frozen=True, slots=True)
class Datapoint:
    """One CloudWatch period, with whichever statistics were requested."""

    timestamp: datetime
    values: dict[str, float]

    def get(self, stat: Stat) -> float | None:
        return self.values.get(stat.value)


@dataclass(frozen=True, slots=True)
class MetricSeries:
    """Datapoints for one metric over one window, sorted by timestamp.

    An empty series is not an error: ``GPUUtilization`` on a CPU endpoint, or
    ``Invocation5XXErrors`` on a clean run, legitimately return nothing. Callers
    must distinguish "zero" from "absent", which is why :meth:`total` returns
    ``None`` rather than ``0.0`` for an empty series.
    """

    spec: MetricSpec
    datapoints: tuple[Datapoint, ...]

    @property
    def empty(self) -> bool:
        return not self.datapoints

    def series(self, stat: Stat) -> list[float]:
        """Non-null values for one statistic, in timestamp order."""
        return [v for dp in self.datapoints if (v := dp.get(stat)) is not None]

    def mean(self, stat: Stat = Stat.AVERAGE) -> float | None:
        """Unweighted mean across periods.

        Unweighted because ``GetMetricStatistics`` does not return per-period
        weights for every statistic. With uniform periods and steady load the
        difference is negligible; under a partial trailing period it is not, which
        is what :data:`DEFAULT_SETTLE_DELAY_S` exists to avoid.
        """
        values = self.series(stat)
        return float(np.mean(values)) if values else None

    def peak(self, stat: Stat = Stat.MAXIMUM) -> float | None:
        values = self.series(stat)
        return float(np.max(values)) if values else None

    def total(self, stat: Stat = Stat.SUM) -> float | None:
        values = self.series(stat)
        return float(np.sum(values)) if values else None

    def percentile_of_periods(self, q: float, stat: Stat = Stat.AVERAGE) -> float | None:
        """Percentile *across periods*, which is not a percentile across requests.

        Named awkwardly on purpose. The p95 of 60 per-minute averages is not the
        p95 request latency; for that, request ``Stat.P95`` and let CloudWatch
        compute it over raw observations.
        """
        values = self.series(stat)
        return float(np.percentile(values, q)) if values else None


def fetch_metric(
    cloudwatch: BaseClient,
    spec: MetricSpec,
    *,
    endpoint: str,
    variant: str = DEFAULT_VARIANT,
    start: datetime,
    end: datetime,
    period_s: int = DEFAULT_PERIOD_S,
) -> MetricSeries:
    """Fetch one metric for one endpoint variant over ``[start, end)``.

    Returns an empty series when the metric does not exist for this endpoint,
    rather than raising — see :class:`MetricSeries`.

    Raises:
        ValueError: If ``period_s`` is not a CloudWatch-legal period, the window
            is inverted, or the window would exceed 1440 datapoints.
    """
    _validate_period(period_s)
    if end <= start:
        raise ValueError(f"end ({end}) must be after start ({start})")
    expected_points = (end - start).total_seconds() / period_s
    if expected_points > _MAX_DATAPOINTS:
        raise ValueError(
            f"window of {(end - start).total_seconds():.0f}s at period {period_s}s needs "
            f"{expected_points:.0f} datapoints, over the CloudWatch limit of {_MAX_DATAPOINTS}. "
            "Use a coarser period or a shorter window."
        )

    simple = [s.value for s in spec.stats if not s.is_extended]
    extended = [s.value for s in spec.stats if s.is_extended]

    kwargs: dict[str, object] = {
        "Namespace": spec.namespace,
        "MetricName": spec.name,
        "Dimensions": endpoint_dimensions(endpoint, variant),
        "StartTime": start,
        "EndTime": end,
        "Period": period_s,
    }
    if simple:
        kwargs["Statistics"] = simple
    if extended:
        kwargs["ExtendedStatistics"] = extended
    if spec.unit:
        kwargs["Unit"] = spec.unit

    try:
        response = cloudwatch.get_metric_statistics(**kwargs)
    except botocore.exceptions.ClientError as exc:
        logger.warning("Could not fetch {}/{}: {}", spec.namespace, spec.name, exc)
        return MetricSeries(spec=spec, datapoints=())

    points = [
        Datapoint(
            timestamp=raw["Timestamp"],
            values={
                **{s: float(raw[s]) for s in simple if s in raw},
                **{
                    s: float(v)
                    for s, v in (raw.get("ExtendedStatistics") or {}).items()
                    if v is not None
                },
            },
        )
        for raw in response.get("Datapoints", [])
    ]
    points.sort(key=lambda dp: dp.timestamp)
    return MetricSeries(spec=spec, datapoints=tuple(points))


@dataclass(frozen=True, slots=True)
class WindowMetrics:
    """Every server-side metric for one measurement window, keyed by metric name."""

    endpoint: str
    variant: str
    start: datetime
    end: datetime
    period_s: int
    series: dict[str, MetricSeries] = field(default_factory=dict)

    def get(self, name: str) -> MetricSeries | None:
        return self.series.get(name)

    @property
    def concurrency_mean(self) -> float | None:
        s = self.series.get("ConcurrentRequestsPerModel")
        return s.mean(Stat.AVERAGE) if s else None

    @property
    def concurrency_peak(self) -> float | None:
        s = self.series.get("ConcurrentRequestsPerModel")
        return s.peak(Stat.MAXIMUM) if s else None

    @property
    def model_latency_p95_ms(self) -> float | None:
        s = self.series.get("ModelLatency")
        micros = s.mean(Stat.P95) if s else None
        # ModelLatency is published in MICROseconds. Reporting it as ms without
        # this divide would inflate every latency by 1000x and move the knee off
        # the end of the ladder.
        return micros / 1000.0 if micros is not None else None

    @property
    def error_5xx_total(self) -> float:
        s = self.series.get("Invocation5XXErrors")
        return s.total(Stat.SUM) or 0.0 if s else 0.0

    @property
    def invocations_total(self) -> float:
        s = self.series.get("Invocations")
        return s.total(Stat.SUM) or 0.0 if s else 0.0

    @property
    def gpu_utilization_mean(self) -> float | None:
        s = self.series.get("GPUUtilization")
        return s.mean(Stat.AVERAGE) if s else None

    @property
    def has_gpu_metrics(self) -> bool:
        s = self.series.get("GPUUtilization")
        return s is not None and not s.empty


def fetch_window(
    cloudwatch: BaseClient,
    *,
    endpoint: str,
    variant: str = DEFAULT_VARIANT,
    start: datetime,
    end: datetime,
    period_s: int = DEFAULT_PERIOD_S,
    specs: Sequence[MetricSpec] = INVOCATION_METRICS + UTILIZATION_METRICS,
) -> WindowMetrics:
    """Fetch every spec for one window. Missing metrics come back empty, not absent."""
    series = {
        spec.key: fetch_metric(
            cloudwatch,
            spec,
            endpoint=endpoint,
            variant=variant,
            start=start,
            end=end,
            period_s=period_s,
        )
        for spec in specs
    }
    return WindowMetrics(
        endpoint=endpoint,
        variant=variant,
        start=start,
        end=end,
        period_s=period_s,
        series=series,
    )


DEFAULT_AGREEMENT_TOLERANCE = 0.25


@dataclass(frozen=True, slots=True)
class Agreement:
    """Whether client-observed concurrency matches what AWS counted."""

    client_mean: float
    server_mean: float | None
    tolerance: float

    @property
    def unavailable(self) -> bool:
        return self.server_mean is None

    @property
    def relative_error(self) -> float | None:
        if self.server_mean is None or self.server_mean <= 0:
            return None
        return abs(self.client_mean - self.server_mean) / self.server_mean

    @property
    def agrees(self) -> bool:
        err = self.relative_error
        return err is not None and err <= self.tolerance

    @property
    def client_lower(self) -> bool:
        """Client saw *less* concurrency than the server counted.

        Usually benign: the server counts queued requests we have already
        dispatched, plus its own in-flight accounting granularity.
        """
        return self.server_mean is not None and self.client_mean < self.server_mean

    @property
    def diagnosis(self) -> str:
        if self.unavailable:
            return "no server-side concurrency datapoints; cannot cross-check"
        if self.agrees:
            return "client and server concurrency agree"
        if self.client_lower:
            return (
                f"server counted more concurrency ({self.server_mean:.2f}) than the client "
                f"observed ({self.client_mean:.2f}); requests are queueing server-side, which "
                "is expected once the in-container queue is admitting"
            )
        return (
            f"client held {self.client_mean:.2f} in flight but the server counted only "
            f"{self.server_mean:.2f}; load is bottlenecked before it reaches the endpoint "
            "(connection pool, dispatcher, or DNS) — this C_max would describe the benchmark"
        )


def concurrency_agreement(
    client_mean_in_flight: float,
    window: WindowMetrics,
    *,
    tolerance: float = DEFAULT_AGREEMENT_TOLERANCE,
) -> Agreement:
    """Cross-check our concurrency accounting against AWS's.

    The failure this catches is the one that motivated ``invoke.py``: with
    botocore's default ``max_pool_connections=10``, a driver asking for 40 in
    flight silently serializes and measures its own client. Client-side counters
    look healthy throughout, because the requests really are outstanding — they
    just are not at the server.
    """
    return Agreement(
        client_mean=client_mean_in_flight,
        server_mean=window.concurrency_mean,
        tolerance=tolerance,
    )


def first_datapoint_at_or_above(
    series: MetricSeries,
    threshold: float,
    *,
    stat: Stat = Stat.MAXIMUM,
) -> datetime | None:
    """Timestamp of the first period reaching ``threshold``.

    ``ttotal``'s metric-publication stage. Uses ``Maximum`` by default: the
    question is when the raised load first became *visible* to CloudWatch, and a
    period average dilutes a spike that arrived mid-period.

    Returns:
        ``None`` if the threshold is never reached in the window — a real result
        for ``ttotal``, meaning the load never registered, and it must not be
        confused with "reached at the window start".
    """
    for dp in series.datapoints:
        value = dp.get(stat)
        if value is not None and value >= threshold:
            return dp.timestamp
    return None


@dataclass(frozen=True, slots=True)
class AlarmTransition:
    """One alarm state change, from ``DescribeAlarmHistory``."""

    alarm_name: str
    timestamp: datetime
    summary: str

    @property
    def to_alarm(self) -> bool:
        return "to ALARM" in self.summary

    @property
    def to_insufficient_data(self) -> bool:
        return "to INSUFFICIENT_DATA" in self.summary


def alarm_transitions(
    cloudwatch: BaseClient,
    alarm_names: Iterable[str],
    *,
    start: datetime,
    end: datetime,
) -> list[AlarmTransition]:
    """State updates for the given alarms, oldest first.

    Alarm names come from ``describe_scaling_policies`` → ``Alarms[]``; target
    tracking creates them implicitly, so they cannot be predicted from config.
    """
    transitions: list[AlarmTransition] = []
    for name in alarm_names:
        try:
            paginator = cloudwatch.get_paginator("describe_alarm_history")
            pages = paginator.paginate(
                AlarmName=name,
                HistoryItemType="StateUpdate",
                StartDate=start,
                EndDate=end,
                ScanBy="TimestampAscending",
            )
            for page in pages:
                for item in page.get("AlarmHistoryItems", []):
                    transitions.append(
                        AlarmTransition(
                            alarm_name=item.get("AlarmName", name),
                            timestamp=item["Timestamp"],
                            summary=item.get("HistorySummary", ""),
                        )
                    )
        except botocore.exceptions.ClientError as exc:
            logger.warning("Could not read alarm history for {}: {}", name, exc)

    transitions.sort(key=lambda t: t.timestamp)
    return transitions


@dataclass(frozen=True, slots=True)
class ScalingActivity:
    """One Application Auto Scaling activity."""

    activity_id: str
    start_time: datetime
    end_time: datetime | None
    status_code: str
    description: str
    cause: str
    status_message: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.status_code == "Successful"

    @property
    def duration_s(self) -> float | None:
        if self.end_time is None:
            return None
        return (self.end_time - self.start_time).total_seconds()


def scaling_activities(
    appscaling: BaseClient,
    *,
    endpoint: str,
    variant: str = DEFAULT_VARIANT,
    start: datetime | None = None,
    end: datetime | None = None,
) -> list[ScalingActivity]:
    """Scaling activities for one variant, oldest first.

    The API has no time filter, so the window is applied client-side after
    fetching. ``start``/``end`` of ``None`` means unbounded on that side.
    """
    rid = resource_id(endpoint, variant)
    activities: list[ScalingActivity] = []
    try:
        paginator = appscaling.get_paginator("describe_scaling_activities")
        pages = paginator.paginate(
            ServiceNamespace=SERVICE_NAMESPACE,
            ResourceId=rid,
            ScalableDimension=SCALABLE_DIMENSION,
        )
        for page in pages:
            for raw in page.get("ScalingActivities", []):
                activities.append(
                    ScalingActivity(
                        activity_id=raw["ActivityId"],
                        start_time=raw["StartTime"],
                        end_time=raw.get("EndTime"),
                        status_code=raw.get("StatusCode", ""),
                        description=raw.get("Description", ""),
                        cause=raw.get("Cause", ""),
                        status_message=raw.get("StatusMessage"),
                    )
                )
    except botocore.exceptions.ClientError as exc:
        logger.warning("Could not read scaling activities for {}: {}", rid, exc)
        return []

    if start is not None:
        activities = [a for a in activities if a.start_time >= start]
    if end is not None:
        activities = [a for a in activities if a.start_time <= end]
    activities.sort(key=lambda a: a.start_time)
    return activities


def settle_and_fetch_window(
    cloudwatch: BaseClient,
    *,
    endpoint: str,
    variant: str = DEFAULT_VARIANT,
    start: datetime,
    end: datetime,
    period_s: int = DEFAULT_PERIOD_S,
    settle_delay_s: float = DEFAULT_SETTLE_DELAY_S,
    sleep=None,
    now=None,
) -> WindowMetrics:
    """Wait for CloudWatch aggregation to catch up, then fetch.

    Sleeps only for the remaining part of the settle delay — if the caller
    already spent time on later ladder steps, that time counts. ``sleep`` and
    ``now`` are injectable so tests do not wait two minutes.
    """
    import time as _time

    sleep = sleep or _time.sleep
    now = now or (lambda: datetime.now(UTC))

    elapsed = (now() - end).total_seconds()
    remaining = settle_delay_s - elapsed
    if remaining > 0:
        logger.info("Waiting {:.0f}s for CloudWatch to settle", remaining)
        sleep(remaining)
    return fetch_window(
        cloudwatch,
        endpoint=endpoint,
        variant=variant,
        start=start,
        end=end,
        period_s=period_s,
    )


# --------------------------------------------------------------------------- #
# Drift audit
# --------------------------------------------------------------------------- #


class DriftKind(StrEnum):
    """What kind of disagreement was found between deployed state and source."""

    ORPHANED_TARGET = "orphaned_target"
    """A scalable target exists for an endpoint no config describes."""

    ORPHANED_POLICY = "orphaned_policy"
    """A live policy whose endpoint's config has scaling disabled."""

    INERT_POLICY = "inert_policy"
    """Policy targets a custom metric whose namespace publishes nothing."""

    CAPACITY_MISMATCH = "capacity_mismatch"
    """Deployed min/max capacity disagrees with config."""

    MISSING_TARGET = "missing_target"
    """Config enables scaling but no scalable target is registered."""

    SUSPENDED = "suspended"
    """Scaling is suspended — usually a freeze that was not thawed."""

    ALARM_INSUFFICIENT_DATA = "alarm_insufficient_data"
    """Policy alarm has never had data, so the policy has never evaluated."""


class Severity(StrEnum):
    ERROR = "error"
    """Can move capacity in a way nothing describes, or cannot move it at all."""

    WARN = "warn"
    INFO = "info"


@dataclass(frozen=True, slots=True)
class DriftFinding:
    """One reconciliation finding. Read-only output; nothing acts on these."""

    kind: DriftKind
    severity: Severity
    resource_id: str
    detail: str
    remediation: str | None = None

    @property
    def endpoint(self) -> str | None:
        parsed = parse_resource_id(self.resource_id)
        return parsed[0] if parsed else None


@dataclass(frozen=True, slots=True)
class ExpectedScaling:
    """What source-of-truth config says an endpoint's scaling should be.

    Built by the caller from ``speech_infra.config``. Passed in rather than
    imported so this module stays free of ``aws-cdk-lib``.
    """

    endpoint: str
    min_instances: int
    max_instances: int
    scaling_enabled: bool

    @property
    def effective_min(self) -> int:
        """Config's ``min_instances=0`` is coerced to 1 by the CDK constructs."""
        return max(self.min_instances, 1)


@dataclass(frozen=True, slots=True)
class LivePolicy:
    """A deployed scaling policy, reduced to the fields the audit reads."""

    policy_name: str
    resource_id: str
    policy_type: str
    metric_namespace: str | None
    metric_name: str | None
    target_value: float | None
    disable_scale_in: bool | None
    alarm_names: tuple[str, ...] = ()

    @property
    def uses_custom_metric(self) -> bool:
        return self.metric_namespace is not None


@dataclass(frozen=True, slots=True)
class LiveTarget:
    """A deployed scalable target, reduced to the fields the audit reads."""

    resource_id: str
    min_capacity: int
    max_capacity: int
    suspended_state: dict[str, bool] = field(default_factory=dict)

    @property
    def scale_out_suspended(self) -> bool:
        return bool(self.suspended_state.get("DynamicScalingOutSuspended", False))

    @property
    def any_suspended(self) -> bool:
        return any(self.suspended_state.values())


def _policy_from_raw(raw: Mapping[str, object]) -> LivePolicy:
    tt = raw.get("TargetTrackingScalingPolicyConfiguration") or {}
    custom = tt.get("CustomizedMetricSpecification") or {} if isinstance(tt, Mapping) else {}
    predefined = tt.get("PredefinedMetricSpecification") or {} if isinstance(tt, Mapping) else {}
    alarms = raw.get("Alarms") or []
    return LivePolicy(
        policy_name=str(raw.get("PolicyName", "")),
        resource_id=str(raw.get("ResourceId", "")),
        policy_type=str(raw.get("PolicyType", "")),
        metric_namespace=custom.get("Namespace"),
        metric_name=custom.get("MetricName") or predefined.get("PredefinedMetricType"),
        target_value=tt.get("TargetValue") if isinstance(tt, Mapping) else None,
        disable_scale_in=tt.get("DisableScaleIn") if isinstance(tt, Mapping) else None,
        alarm_names=tuple(a["AlarmName"] for a in alarms if isinstance(a, Mapping)),
    )


def list_live_targets(appscaling: BaseClient) -> list[LiveTarget]:
    """Every ``sagemaker`` endpoint-variant scalable target in the account.

    Account-wide by design: the point of ``drift`` is to find targets that no
    config mentions, and a per-endpoint query cannot find those.
    """
    targets: list[LiveTarget] = []
    paginator = appscaling.get_paginator("describe_scalable_targets")
    for page in paginator.paginate(ServiceNamespace=SERVICE_NAMESPACE):
        for raw in page.get("ScalableTargets", []):
            rid = raw.get("ResourceId", "")
            if parse_resource_id(rid) is None:
                continue
            targets.append(
                LiveTarget(
                    resource_id=rid,
                    min_capacity=int(raw.get("MinCapacity", 0)),
                    max_capacity=int(raw.get("MaxCapacity", 0)),
                    suspended_state=dict(raw.get("SuspendedState") or {}),
                )
            )
    return targets


def list_live_policies(appscaling: BaseClient) -> list[LivePolicy]:
    """Every ``sagemaker`` endpoint-variant scaling policy in the account."""
    policies: list[LivePolicy] = []
    paginator = appscaling.get_paginator("describe_scaling_policies")
    for page in paginator.paginate(ServiceNamespace=SERVICE_NAMESPACE):
        for raw in page.get("ScalingPolicies", []):
            if parse_resource_id(str(raw.get("ResourceId", ""))) is None:
                continue
            policies.append(_policy_from_raw(raw))
    return policies


def namespace_is_publishing(cloudwatch: BaseClient, namespace: str) -> bool:
    """Whether a metric namespace has any metrics at all.

    This is what makes the two live orphaned policies *dormant*: they target
    ``Speech/vLLM``, which publishes nothing, so their alarms sit in
    ``INSUFFICIENT_DATA`` forever. Dormant is not off — the moment that namespace
    starts publishing, they act.
    """
    try:
        response = cloudwatch.list_metrics(Namespace=namespace)
    except botocore.exceptions.ClientError as exc:
        logger.warning("Could not list metrics in {}: {}", namespace, exc)
        return True  # Unknown; do not claim a policy is inert on a failed read.
    return bool(response.get("Metrics"))


def alarm_states(cloudwatch: BaseClient, alarm_names: Sequence[str]) -> dict[str, str]:
    """Current state of each named alarm. Missing alarms are simply absent."""
    if not alarm_names:
        return {}
    states: dict[str, str] = {}
    try:
        paginator = cloudwatch.get_paginator("describe_alarms")
        # describe_alarms caps AlarmNames at 100 per call; chunk to stay legal.
        for i in range(0, len(alarm_names), 100):
            for page in paginator.paginate(AlarmNames=list(alarm_names[i : i + 100])):
                for alarm in page.get("MetricAlarms", []):
                    states[alarm["AlarmName"]] = alarm.get("StateValue", "")
    except botocore.exceptions.ClientError as exc:
        logger.warning("Could not describe alarms: {}", exc)
    return states


def audit_scaling(
    *,
    targets: Sequence[LiveTarget],
    policies: Sequence[LivePolicy],
    expected: Mapping[str, ExpectedScaling],
    publishing_namespaces: Mapping[str, bool] | None = None,
    alarm_state: Mapping[str, str] | None = None,
) -> list[DriftFinding]:
    """Reconcile deployed scaling against config. Pure; no I/O.

    ``expected`` is keyed by endpoint name. An endpoint absent from ``expected``
    is treated as having no source — that is the orphan case, and it is reported
    as an error because such a target can move capacity that no template
    describes and no ``cdk deploy`` will ever remove.

    ``publishing_namespaces`` and ``alarm_state`` are optional: when omitted, the
    inert-policy and alarm checks are skipped rather than guessed at.
    """
    publishing = publishing_namespaces or {}
    alarms = alarm_state or {}
    findings: list[DriftFinding] = []

    by_endpoint: dict[str, LiveTarget] = {}
    for target in targets:
        parsed = parse_resource_id(target.resource_id)
        if parsed is None:
            continue
        by_endpoint[parsed[0]] = target

    for target in targets:
        parsed = parse_resource_id(target.resource_id)
        if parsed is None:
            continue
        endpoint, _variant = parsed
        config = expected.get(endpoint)

        if config is None:
            findings.append(
                DriftFinding(
                    kind=DriftKind.ORPHANED_TARGET,
                    severity=Severity.ERROR,
                    resource_id=target.resource_id,
                    detail=(
                        f"scalable target exists (min={target.min_capacity}, "
                        f"max={target.max_capacity}) but no config describes {endpoint}"
                    ),
                    remediation=(
                        "aws application-autoscaling deregister-scalable-target "
                        f"--service-namespace sagemaker --resource-id {target.resource_id} "
                        f"--scalable-dimension {SCALABLE_DIMENSION}"
                    ),
                )
            )
            continue

        if not config.scaling_enabled:
            findings.append(
                DriftFinding(
                    kind=DriftKind.ORPHANED_TARGET,
                    severity=Severity.ERROR,
                    resource_id=target.resource_id,
                    detail=(
                        f"config has scaling disabled (min={config.min_instances}, "
                        f"max={config.max_instances}) but a live target allows "
                        f"{target.min_capacity}-{target.max_capacity} instances; CDK no longer "
                        "synthesizes this, so cdk deploy will never remove it"
                    ),
                    remediation=(
                        "raise max_instances in config.py if scaling is intended, otherwise: "
                        "aws application-autoscaling deregister-scalable-target "
                        f"--service-namespace sagemaker --resource-id {target.resource_id} "
                        f"--scalable-dimension {SCALABLE_DIMENSION}"
                    ),
                )
            )
        elif (
            target.min_capacity != config.effective_min
            or target.max_capacity != config.max_instances
        ):
            findings.append(
                DriftFinding(
                    kind=DriftKind.CAPACITY_MISMATCH,
                    severity=Severity.WARN,
                    resource_id=target.resource_id,
                    detail=(
                        f"live capacity {target.min_capacity}-{target.max_capacity} but config "
                        f"says {config.effective_min}-{config.max_instances}"
                    ),
                    remediation=f"cdk deploy Speech-{endpoint.removeprefix('speech-')}",
                )
            )

        if target.any_suspended:
            findings.append(
                DriftFinding(
                    kind=DriftKind.SUSPENDED,
                    severity=Severity.ERROR if target.scale_out_suspended else Severity.WARN,
                    resource_id=target.resource_id,
                    detail=f"SuspendedState={target.suspended_state}",
                    remediation=f"tts-bench thaw --endpoint {endpoint}",
                )
            )

    for endpoint, config in sorted(expected.items()):
        if config.scaling_enabled and endpoint not in by_endpoint:
            findings.append(
                DriftFinding(
                    kind=DriftKind.MISSING_TARGET,
                    severity=Severity.WARN,
                    resource_id=resource_id(endpoint),
                    detail=(
                        f"config enables scaling ({config.effective_min}-"
                        f"{config.max_instances}) but no scalable target is registered"
                    ),
                    remediation=f"cdk deploy Speech-{endpoint.removeprefix('speech-')}",
                )
            )

    for policy in policies:
        parsed = parse_resource_id(policy.resource_id)
        if parsed is None:
            continue
        endpoint, _variant = parsed
        config = expected.get(endpoint)

        if config is None or not config.scaling_enabled:
            findings.append(
                DriftFinding(
                    kind=DriftKind.ORPHANED_POLICY,
                    severity=Severity.ERROR,
                    resource_id=policy.resource_id,
                    detail=(
                        f"policy {policy.policy_name!r} is live on "
                        f"{policy.metric_namespace or 'predefined'}:{policy.metric_name} "
                        f"(target {policy.target_value}) but no config synthesizes it"
                    ),
                    remediation=(
                        "aws application-autoscaling delete-scaling-policy "
                        f"--service-namespace sagemaker --policy-name {policy.policy_name} "
                        f"--resource-id {policy.resource_id} "
                        f"--scalable-dimension {SCALABLE_DIMENSION}"
                    ),
                )
            )

        if policy.uses_custom_metric and policy.metric_namespace in publishing:
            if not publishing[policy.metric_namespace]:
                findings.append(
                    DriftFinding(
                        kind=DriftKind.INERT_POLICY,
                        severity=Severity.WARN,
                        resource_id=policy.resource_id,
                        detail=(
                            f"policy {policy.policy_name!r} targets "
                            f"{policy.metric_namespace}:{policy.metric_name}, but that namespace "
                            "publishes no metrics — the policy can never fire"
                        ),
                        remediation=(
                            "switch to the native SageMakerVariantConcurrentRequestsPerModel "
                            "predefined metric, or instrument the container"
                        ),
                    )
                )

        for name in policy.alarm_names:
            if alarms.get(name) == "INSUFFICIENT_DATA":
                findings.append(
                    DriftFinding(
                        kind=DriftKind.ALARM_INSUFFICIENT_DATA,
                        severity=Severity.WARN,
                        resource_id=policy.resource_id,
                        detail=(
                            f"alarm {name!r} for policy {policy.policy_name!r} is "
                            "INSUFFICIENT_DATA, so this policy has never evaluated"
                        ),
                        remediation="confirm the metric publishes, and set TreatMissingData",
                    )
                )

    return findings


def check_drift(
    appscaling: BaseClient,
    cloudwatch: BaseClient,
    expected: Mapping[str, ExpectedScaling],
) -> list[DriftFinding]:
    """Fetch live scaling state and audit it. Read-only.

    Thin I/O wrapper over :func:`audit_scaling`; the rules live there.
    """
    targets = list_live_targets(appscaling)
    policies = list_live_policies(appscaling)

    namespaces = {p.metric_namespace for p in policies if p.metric_namespace}
    publishing = {ns: namespace_is_publishing(cloudwatch, ns) for ns in sorted(namespaces)}

    alarm_names = sorted({name for p in policies for name in p.alarm_names})
    states = alarm_states(cloudwatch, alarm_names)

    return audit_scaling(
        targets=targets,
        policies=policies,
        expected=expected,
        publishing_namespaces=publishing,
        alarm_state=states,
    )


def utc_window(end: datetime, duration_s: float) -> tuple[datetime, datetime]:
    """``(start, end)`` for a window of ``duration_s`` ending at ``end``."""
    return end - timedelta(seconds=duration_s), end


# --------------------------------------------------------------------------- #
# CloudWatch Logs — the container half of T_total
# --------------------------------------------------------------------------- #

LOG_GROUP_PREFIX = "/aws/sagemaker/Endpoints"
"""Log group root. Note this string also names a *metric* namespace
(:data:`ENDPOINT_NAMESPACE`); they are unrelated APIs that happen to share it."""


def log_group_name(endpoint: str) -> str:
    """Log group for one endpoint's containers."""
    return f"{LOG_GROUP_PREFIX}/{endpoint}"


@dataclass(frozen=True, slots=True)
class LogStream:
    """One container's log stream, named ``<variant>/i-<instance-id>``."""

    name: str
    first_event_at: datetime | None
    last_event_at: datetime | None

    @property
    def instance_id(self) -> str | None:
        """The ``i-...`` part, or None if the name does not follow the convention."""
        _, _, tail = self.name.partition("/")
        return tail if tail.startswith("i-") else None

    @property
    def variant(self) -> str | None:
        head, sep, _ = self.name.partition("/")
        return head if sep else None


def _epoch_ms_to_dt(value: object) -> datetime | None:
    if not isinstance(value, int | float):
        return None
    return datetime.fromtimestamp(value / 1000.0, tz=UTC)


def list_log_streams(
    logs: BaseClient,
    *,
    endpoint: str,
    variant: str | None = DEFAULT_VARIANT,
    since: datetime | None = None,
    limit: int = 50,
) -> list[LogStream]:
    """Streams for an endpoint, most recently active first.

    Args:
        variant: Restrict to one variant via the stream-name prefix. ``None``
            lists every variant.
        since: Drop streams whose last event predates this. Filters on
            ``lastEventTimestamp`` rather than ``firstEventTimestamp`` because a
            long-lived instance started days ago is still serving now.

    Returns:
        Empty list if the group does not exist — true for an endpoint that has
        never run, and not an error worth raising here.
    """
    group = log_group_name(endpoint)
    # orderBy=LastEventTime and logStreamNamePrefix are mutually exclusive in the
    # API, and the ordering is what makes `limit` mean "most recent", so the
    # variant prefix is applied client-side below.
    kwargs: dict[str, object] = {
        "logGroupName": group,
        "orderBy": "LastEventTime",
        "descending": True,
    }

    streams: list[LogStream] = []
    try:
        paginator = logs.get_paginator("describe_log_streams")
        for page in paginator.paginate(**kwargs):
            for raw in page.get("logStreams", []):
                streams.append(
                    LogStream(
                        name=raw.get("logStreamName", ""),
                        first_event_at=_epoch_ms_to_dt(raw.get("firstEventTimestamp")),
                        last_event_at=_epoch_ms_to_dt(raw.get("lastEventTimestamp")),
                    )
                )
            if len(streams) >= limit:
                break
    except botocore.exceptions.ClientError as exc:
        logger.warning("Could not list log streams for {}: {}", group, exc)
        return []

    if variant:
        streams = [s for s in streams if s.name.startswith(f"{variant}/")]
    if since is not None:
        streams = [s for s in streams if s.last_event_at is None or s.last_event_at >= since]
    return streams[:limit]


def new_streams_since(
    before: Iterable[LogStream],
    after: Iterable[LogStream],
) -> list[LogStream]:
    """Streams present in ``after`` but not ``before``.

    How a scale-out's new instance is identified. SageMaker does not report which
    instance was added, but each one opens its own log stream, so the set
    difference across the event names it.
    """
    known = {s.name for s in before}
    return [s for s in after if s.name not in known]


def read_log_lines(
    logs: BaseClient,
    *,
    endpoint: str,
    stream: str,
    limit: int = 1000,
) -> list[str]:
    """Read a stream from its beginning, oldest first.

    From the head deliberately: every stage marker is emitted during startup, so
    the tail of a long-running container holds nothing this needs.

    Returns:
        Empty list if the stream or group is gone. A stream that aged out is a
        bounded-estimate case for ``ttotal``, not a failure.
    """
    lines: list[str] = []
    token: str | None = None
    try:
        while len(lines) < limit:
            kwargs: dict[str, object] = {
                "logGroupName": log_group_name(endpoint),
                "logStreamName": stream,
                "startFromHead": True,
                "limit": min(limit - len(lines), 1000),
            }
            if token is not None:
                kwargs["nextToken"] = token
            response = logs.get_log_events(**kwargs)
            events = response.get("events", [])
            lines.extend(str(e.get("message", "")) for e in events)
            next_token = response.get("nextForwardToken")
            # GetLogEvents returns the same token at the end of a stream rather
            # than omitting it; without this check the loop never terminates.
            if not events or next_token == token or next_token is None:
                break
            token = next_token
    except botocore.exceptions.ClientError as exc:
        logger.warning("Could not read log stream {}/{}: {}", endpoint, stream, exc)
        return lines

    return lines


def stage_markers(
    logs: BaseClient,
    *,
    endpoint: str,
    stream: str,
    limit: int = 1000,
) -> list[StageMarker]:
    """Parse the startup stage markers a container emitted, in order.

    The format contract lives in :mod:`shared.stages`, which is the single
    definition shared by four separately-built container images.

    Returns:
        Empty list when the container predates the markers, or its startup lines
        aged out. ``ttotal`` degrades to a bounded estimate rather than failing.
    """
    from shared.stages import parse_stage_markers

    return parse_stage_markers(read_log_lines(logs, endpoint=endpoint, stream=stream, limit=limit))
