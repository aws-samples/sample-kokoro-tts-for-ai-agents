"""Tests for read-only CloudWatch / appscaling observation and drift audit.

Split by testability: :func:`audit_scaling` is pure, so its rules are table-driven
with hand-built `LiveTarget`/`LivePolicy`/`ExpectedScaling` values. The fetch paths
use botocore ``Stubber``, which validates responses against the real service model
— so a stubbed `ScalableTarget` missing `RoleARN`, or a `GetMetricStatistics` call
with an illegal `Period`, fails here rather than in production.

The scenario the drift tests are built around is the live one: two orphaned
policies on ``Speech/vLLM``, a namespace that publishes nothing, whose alarms sit
in INSUFFICIENT_DATA. Dormant, not off — ``speech-orpheus-3b`` still carries a
``max_capacity=4`` target that no template describes.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import boto3
import pytest
from botocore.stub import Stubber

from tts_bench.observe import (
    ENDPOINT_NAMESPACE,
    HIGH_RES_PERIOD_S,
    INVOCATION_METRICS,
    SAGEMAKER_NAMESPACE,
    SCALABLE_DIMENSION,
    UTILIZATION_METRICS,
    DriftKind,
    ExpectedScaling,
    LivePolicy,
    LiveTarget,
    MetricSeries,
    MetricSpec,
    Severity,
    Stat,
    alarm_states,
    alarm_transitions,
    audit_scaling,
    check_drift,
    concurrency_agreement,
    endpoint_dimensions,
    fetch_metric,
    fetch_window,
    first_datapoint_at_or_above,
    list_live_policies,
    list_live_targets,
    namespace_is_publishing,
    parse_resource_id,
    resource_id,
    scaling_activities,
    settle_and_fetch_window,
    utc_window,
)

ENDPOINT = "speech-kokoro-82m"
T0 = datetime(2026, 7, 29, 12, 0, 0, tzinfo=UTC)
T1 = T0 + timedelta(minutes=5)
ROLE_ARN = "arn:aws:iam::1234:role/aws-service-role/sagemaker.application-autoscaling"


@pytest.fixture
def cloudwatch():
    client = boto3.client("cloudwatch", region_name="us-east-1")
    stub = Stubber(client)
    stub.activate()
    yield client, stub
    stub.deactivate()


@pytest.fixture
def appscaling():
    client = boto3.client("application-autoscaling", region_name="us-east-1")
    stub = Stubber(client)
    stub.activate()
    yield client, stub
    stub.deactivate()


def _datapoints(*rows: tuple[int, float]) -> list[dict]:
    """``(minute_offset, average)`` rows as CloudWatch datapoints."""
    return [
        {
            "Timestamp": T0 + timedelta(minutes=offset),
            "Average": value,
            "Maximum": value,
            "Unit": "Count",
        }
        for offset, value in rows
    ]


def _series(*rows: tuple[int, float], name: str = "ConcurrentRequestsPerModel") -> MetricSeries:
    """Build a MetricSeries directly, bypassing the fetch path."""
    from tts_bench.observe import Datapoint

    spec = MetricSpec(SAGEMAKER_NAMESPACE, name)
    points = tuple(
        Datapoint(
            timestamp=T0 + timedelta(minutes=offset),
            values={"Average": value, "Maximum": value},
        )
        for offset, value in rows
    )
    return MetricSeries(spec=spec, datapoints=points)


def _target(
    endpoint: str = ENDPOINT,
    *,
    min_capacity: int = 1,
    max_capacity: int = 4,
    suspended: dict[str, bool] | None = None,
) -> LiveTarget:
    return LiveTarget(
        resource_id=resource_id(endpoint),
        min_capacity=min_capacity,
        max_capacity=max_capacity,
        suspended_state=suspended or {},
    )


def _policy(
    endpoint: str = ENDPOINT,
    *,
    name: str = "TrackRunningRequests",
    namespace: str | None = "Speech/vLLM",
    metric: str | None = "vllm:num_requests_running",
    target_value: float = 8.0,
    alarms: tuple[str, ...] = (),
) -> LivePolicy:
    return LivePolicy(
        policy_name=name,
        resource_id=resource_id(endpoint),
        policy_type="TargetTrackingScaling",
        metric_namespace=namespace,
        metric_name=metric,
        target_value=target_value,
        disable_scale_in=None,
        alarm_names=alarms,
    )


def _expected(
    endpoint: str = ENDPOINT,
    *,
    min_instances: int = 1,
    max_instances: int = 4,
) -> ExpectedScaling:
    return ExpectedScaling(
        endpoint=endpoint,
        min_instances=min_instances,
        max_instances=max_instances,
        scaling_enabled=max_instances > max(min_instances, 1),
    )


class TestResourceId:
    def test_round_trips(self) -> None:
        assert parse_resource_id(resource_id(ENDPOINT)) == (ENDPOINT, "primary")

    def test_custom_variant_survives(self) -> None:
        assert parse_resource_id(resource_id(ENDPOINT, "canary")) == (ENDPOINT, "canary")

    def test_inference_component_is_not_an_endpoint(self) -> None:
        # The sagemaker namespace also covers inference components. Treating one
        # as an endpoint would report a phantom orphan against config.
        assert parse_resource_id("inference-component/my-component") is None

    def test_malformed_id_is_rejected(self) -> None:
        assert parse_resource_id("endpoint/only-two-parts") is None
        assert parse_resource_id("endpoint/a/notvariant/b") is None

    def test_dimensions_name_both_endpoint_and_variant(self) -> None:
        dims = endpoint_dimensions(ENDPOINT)
        assert {d["Name"] for d in dims} == {"EndpointName", "VariantName"}


class TestUtcWindow:
    def test_window_ends_where_asked(self) -> None:
        start, end = utc_window(T1, 300)
        assert end == T1
        assert (end - start).total_seconds() == 300


class TestFetchMetric:
    def test_parses_datapoints_in_timestamp_order(self, cloudwatch) -> None:
        client, stub = cloudwatch
        # Deliberately out of order: CloudWatch does not guarantee ordering, and
        # first_datapoint_at_or_above depends entirely on it being sorted.
        stub.add_response(
            "get_metric_statistics",
            {"Label": "ConcurrentRequestsPerModel", "Datapoints": _datapoints((2, 3.0), (0, 1.0))},
        )
        series = fetch_metric(
            client,
            MetricSpec(SAGEMAKER_NAMESPACE, "ConcurrentRequestsPerModel"),
            endpoint=ENDPOINT,
            start=T0,
            end=T1,
        )
        assert series.series(Stat.AVERAGE) == [1.0, 3.0]

    def test_sends_the_exact_dimensions_and_period(self, cloudwatch) -> None:
        client, stub = cloudwatch
        stub.add_response(
            "get_metric_statistics",
            {"Datapoints": []},
            {
                "Namespace": SAGEMAKER_NAMESPACE,
                "MetricName": "ConcurrentRequestsPerModel",
                "Dimensions": endpoint_dimensions(ENDPOINT),
                "StartTime": T0,
                "EndTime": T1,
                "Period": HIGH_RES_PERIOD_S,
                "Statistics": ["Average", "Maximum"],
            },
        )
        fetch_metric(
            client,
            MetricSpec(SAGEMAKER_NAMESPACE, "ConcurrentRequestsPerModel"),
            endpoint=ENDPOINT,
            start=T0,
            end=T1,
            period_s=HIGH_RES_PERIOD_S,
        )
        stub.assert_no_pending_responses()

    def test_percentile_stats_go_in_extended_statistics(self, cloudwatch) -> None:
        client, stub = cloudwatch
        # p95 in Statistics is a ValidationError from CloudWatch. Getting this
        # wrong would make every latency fetch fail at runtime only.
        stub.add_response(
            "get_metric_statistics",
            {"Datapoints": [{"Timestamp": T0, "ExtendedStatistics": {"p95": 412000.0}}]},
            {
                "Namespace": SAGEMAKER_NAMESPACE,
                "MetricName": "ModelLatency",
                "Dimensions": endpoint_dimensions(ENDPOINT),
                "StartTime": T0,
                "EndTime": T1,
                "Period": 60,
                "Statistics": ["Average"],
                "ExtendedStatistics": ["p95"],
            },
        )
        series = fetch_metric(
            client,
            MetricSpec(SAGEMAKER_NAMESPACE, "ModelLatency", (Stat.AVERAGE, Stat.P95)),
            endpoint=ENDPOINT,
            start=T0,
            end=T1,
        )
        assert series.series(Stat.P95) == [412000.0]

    def test_missing_metric_returns_empty_not_error(self, cloudwatch) -> None:
        client, stub = cloudwatch
        # GPUUtilization on a CPU endpoint. Absent is a legitimate answer.
        stub.add_response("get_metric_statistics", {"Datapoints": []})
        series = fetch_metric(
            client,
            MetricSpec(ENDPOINT_NAMESPACE, "GPUUtilization"),
            endpoint=ENDPOINT,
            start=T0,
            end=T1,
        )
        assert series.empty

    def test_client_error_degrades_to_empty(self, cloudwatch) -> None:
        client, stub = cloudwatch
        stub.add_client_error("get_metric_statistics", service_error_code="Throttling")
        series = fetch_metric(
            client,
            MetricSpec(SAGEMAKER_NAMESPACE, "Invocations", (Stat.SUM,)),
            endpoint=ENDPOINT,
            start=T0,
            end=T1,
        )
        # A throttled utilization fetch must not abort a 45-minute load run.
        assert series.empty

    def test_inverted_window_is_rejected(self, cloudwatch) -> None:
        client, _ = cloudwatch
        with pytest.raises(ValueError, match="must be after start"):
            fetch_metric(
                client,
                MetricSpec(SAGEMAKER_NAMESPACE, "Invocations"),
                endpoint=ENDPOINT,
                start=T1,
                end=T0,
            )

    @pytest.mark.parametrize("period", [0, -10, 7, 45, 90])
    def test_illegal_periods_are_rejected(self, cloudwatch, period: int) -> None:
        client, _ = cloudwatch
        # CloudWatch accepts 1/5/10/30 below a minute, then multiples of 60.
        with pytest.raises(ValueError):
            fetch_metric(
                client,
                MetricSpec(SAGEMAKER_NAMESPACE, "Invocations"),
                endpoint=ENDPOINT,
                start=T0,
                end=T1,
                period_s=period,
            )

    @pytest.mark.parametrize("period", [1, 5, 10, 30, 60, 300])
    def test_legal_periods_are_accepted(self, cloudwatch, period: int) -> None:
        client, stub = cloudwatch
        stub.add_response("get_metric_statistics", {"Datapoints": []})
        fetch_metric(
            client,
            MetricSpec(SAGEMAKER_NAMESPACE, "Invocations"),
            endpoint=ENDPOINT,
            start=T0,
            end=T0 + timedelta(seconds=period * 10),
            period_s=period,
        )

    def test_window_over_the_datapoint_limit_is_rejected(self, cloudwatch) -> None:
        client, _ = cloudwatch
        # 1440 is a hard cap that returns EMPTY rather than erroring, so an
        # over-long window would look exactly like a dead endpoint.
        with pytest.raises(ValueError, match="over the CloudWatch limit"):
            fetch_metric(
                client,
                MetricSpec(SAGEMAKER_NAMESPACE, "ConcurrentRequestsPerModel"),
                endpoint=ENDPOINT,
                start=T0,
                end=T0 + timedelta(hours=6),
                period_s=HIGH_RES_PERIOD_S,
            )


class TestMetricSeries:
    def test_mean_over_periods(self) -> None:
        assert _series((0, 1.0), (1, 2.0), (2, 3.0)).mean() == pytest.approx(2.0)

    def test_peak_uses_maximum(self) -> None:
        assert _series((0, 1.0), (1, 9.0)).peak() == 9.0

    def test_empty_series_returns_none_not_zero(self) -> None:
        # 0.0 would read as "no concurrency observed", which is a measurement;
        # None says the metric was never published, which is a different fact.
        empty = MetricSeries(spec=MetricSpec(SAGEMAKER_NAMESPACE, "GPUUtilization"), datapoints=())
        assert empty.mean() is None
        assert empty.peak() is None
        assert empty.total() is None

    def test_series_skips_null_statistics(self) -> None:
        from tts_bench.observe import Datapoint

        series = MetricSeries(
            spec=MetricSpec(SAGEMAKER_NAMESPACE, "ModelLatency", (Stat.P95,)),
            datapoints=(
                Datapoint(timestamp=T0, values={"p95": 100.0}),
                Datapoint(timestamp=T1, values={}),
            ),
        )
        assert series.series(Stat.P95) == [100.0]

    def test_percentile_of_periods_is_named_for_what_it_is(self) -> None:
        # p95 across per-minute averages, not across requests. The distinction
        # matters: this understates tail latency badly.
        series = _series(*[(i, float(i)) for i in range(100)])
        assert series.percentile_of_periods(95) == pytest.approx(94.05, abs=0.5)


class TestFetchWindow:
    def test_collects_every_spec_keyed_by_metric_name(self, cloudwatch) -> None:
        client, stub = cloudwatch
        specs = (
            MetricSpec(SAGEMAKER_NAMESPACE, "ConcurrentRequestsPerModel"),
            MetricSpec(SAGEMAKER_NAMESPACE, "Invocations", (Stat.SUM,)),
        )
        stub.add_response("get_metric_statistics", {"Datapoints": _datapoints((0, 2.0))})
        stub.add_response(
            "get_metric_statistics",
            {"Datapoints": [{"Timestamp": T0, "Sum": 120.0}]},
        )
        window = fetch_window(client, endpoint=ENDPOINT, start=T0, end=T1, specs=specs)
        assert window.concurrency_mean == 2.0
        assert window.invocations_total == 120.0

    def test_model_latency_is_converted_from_microseconds(self, cloudwatch) -> None:
        client, stub = cloudwatch
        # AWS publishes ModelLatency in microseconds. A missing divide here is a
        # 1000x error that moves the knee off the end of the ladder entirely.
        stub.add_response(
            "get_metric_statistics",
            {"Datapoints": [{"Timestamp": T0, "ExtendedStatistics": {"p95": 412_000.0}}]},
        )
        window = fetch_window(
            client,
            endpoint=ENDPOINT,
            start=T0,
            end=T1,
            specs=(MetricSpec(SAGEMAKER_NAMESPACE, "ModelLatency", (Stat.P95,)),),
        )
        assert window.model_latency_p95_ms == pytest.approx(412.0)

    def test_absent_error_metric_totals_zero(self, cloudwatch) -> None:
        client, stub = cloudwatch
        # A clean run publishes no 5XX datapoints at all. For a *count* of
        # errors, absent and zero mean the same thing, unlike for a gauge.
        stub.add_response("get_metric_statistics", {"Datapoints": []})
        window = fetch_window(
            client,
            endpoint=ENDPOINT,
            start=T0,
            end=T1,
            specs=(MetricSpec(SAGEMAKER_NAMESPACE, "Invocation5XXErrors", (Stat.SUM,)),),
        )
        assert window.error_5xx_total == 0.0

    def test_gpu_metrics_absent_on_cpu_endpoint(self, cloudwatch) -> None:
        client, stub = cloudwatch
        stub.add_response("get_metric_statistics", {"Datapoints": []})
        window = fetch_window(
            client,
            endpoint="speech-kokoro-82m-cpu",
            start=T0,
            end=T1,
            specs=(MetricSpec(ENDPOINT_NAMESPACE, "GPUUtilization"),),
        )
        assert not window.has_gpu_metrics
        assert window.gpu_utilization_mean is None


class TestSettleAndFetch:
    def test_waits_only_the_remaining_delay(self, cloudwatch) -> None:
        client, stub = cloudwatch
        for _ in range(len(INVOCATION_METRICS) + len(UTILIZATION_METRICS)):
            stub.add_response("get_metric_statistics", {"Datapoints": []})
        slept: list[float] = []
        # 90s already elapsed since the window closed, so only 30 of the 120s
        # settle delay remain. Time spent on later ladder steps counts toward it,
        # which is what keeps a 10-step ladder from paying the delay ten times.
        settle_and_fetch_window(
            client,
            endpoint=ENDPOINT,
            start=T0,
            end=T1,
            settle_delay_s=120.0,
            sleep=slept.append,
            now=lambda: T1 + timedelta(seconds=90),
        )
        assert slept == [pytest.approx(30.0)]
        stub.assert_no_pending_responses()

    def test_does_not_sleep_when_the_delay_already_passed(self, cloudwatch) -> None:
        client, stub = cloudwatch
        for _ in range(len(INVOCATION_METRICS) + len(UTILIZATION_METRICS)):
            stub.add_response("get_metric_statistics", {"Datapoints": []})
        slept: list[float] = []
        settle_and_fetch_window(
            client,
            endpoint=ENDPOINT,
            start=T0,
            end=T1,
            settle_delay_s=120.0,
            sleep=slept.append,
            now=lambda: T1 + timedelta(seconds=600),
        )
        assert slept == []


class TestConcurrencyAgreement:
    def test_close_values_agree(self) -> None:
        window = _window_with_concurrency(4.0)
        assert concurrency_agreement(4.2, window).agrees

    def test_client_far_below_server_is_flagged_as_queueing(self) -> None:
        window = _window_with_concurrency(12.0)
        agreement = concurrency_agreement(4.0, window)
        assert not agreement.agrees
        assert agreement.client_lower
        assert "queueing server-side" in agreement.diagnosis

    def test_client_far_above_server_names_the_client_bottleneck(self) -> None:
        # The connection-pool failure: botocore's default max_pool_connections=10
        # means a driver asking for 40 in flight serializes and measures itself.
        # Client counters look healthy the whole time.
        window = _window_with_concurrency(9.5)
        agreement = concurrency_agreement(40.0, window)
        assert not agreement.agrees
        assert not agreement.client_lower
        assert "bottlenecked before it reaches the endpoint" in agreement.diagnosis

    def test_no_server_data_does_not_claim_agreement(self) -> None:
        window = _window_with_concurrency(None)
        agreement = concurrency_agreement(4.0, window)
        assert agreement.unavailable
        assert not agreement.agrees
        assert "cannot cross-check" in agreement.diagnosis

    def test_zero_server_concurrency_is_not_agreement(self) -> None:
        # Dividing by it would be a ZeroDivisionError; claiming agreement would
        # be worse.
        window = _window_with_concurrency(0.0)
        agreement = concurrency_agreement(4.0, window)
        assert agreement.relative_error is None
        assert not agreement.agrees

    def test_tolerance_is_configurable(self) -> None:
        window = _window_with_concurrency(10.0)
        assert not concurrency_agreement(13.0, window, tolerance=0.1).agrees
        assert concurrency_agreement(13.0, window, tolerance=0.5).agrees


def _window_with_concurrency(mean: float | None):
    from tts_bench.observe import WindowMetrics

    if mean is None:
        series = MetricSeries(
            spec=MetricSpec(SAGEMAKER_NAMESPACE, "ConcurrentRequestsPerModel"), datapoints=()
        )
    else:
        series = _series((0, mean))
    return WindowMetrics(
        endpoint=ENDPOINT,
        variant="primary",
        start=T0,
        end=T1,
        period_s=60,
        series={"ConcurrentRequestsPerModel": series},
    )


class TestFirstDatapointAtOrAbove:
    def test_finds_the_first_crossing(self) -> None:
        series = _series((0, 1.0), (1, 2.0), (2, 8.0), (3, 9.0))
        found = first_datapoint_at_or_above(series, 8.0)
        assert found == T0 + timedelta(minutes=2)

    def test_threshold_never_reached_returns_none(self) -> None:
        # A real ttotal result: the raised load never registered in CloudWatch.
        # Must not be confused with "crossed at the window start".
        series = _series((0, 1.0), (1, 2.0))
        assert first_datapoint_at_or_above(series, 8.0) is None

    def test_empty_series_returns_none(self) -> None:
        empty = MetricSeries(
            spec=MetricSpec(SAGEMAKER_NAMESPACE, "ConcurrentRequestsPerModel"), datapoints=()
        )
        assert first_datapoint_at_or_above(empty, 1.0) is None

    def test_uses_maximum_by_default(self) -> None:
        from tts_bench.observe import Datapoint

        # A spike arriving mid-period shows in Maximum but is diluted in Average.
        # The question is when load first became visible, so Maximum is right.
        series = MetricSeries(
            spec=MetricSpec(SAGEMAKER_NAMESPACE, "ConcurrentRequestsPerModel"),
            datapoints=(Datapoint(timestamp=T0, values={"Average": 2.0, "Maximum": 8.0}),),
        )
        assert first_datapoint_at_or_above(series, 8.0) == T0
        assert first_datapoint_at_or_above(series, 8.0, stat=Stat.AVERAGE) is None


class TestAlarmTransitions:
    def test_parses_and_sorts_across_alarms(self, cloudwatch) -> None:
        client, stub = cloudwatch
        stub.add_response(
            "describe_alarm_history",
            {
                "AlarmHistoryItems": [
                    {
                        "AlarmName": "alarm-b",
                        "Timestamp": T0 + timedelta(seconds=40),
                        "HistoryItemType": "StateUpdate",
                        "HistorySummary": "Alarm updated from OK to ALARM",
                    }
                ]
            },
        )
        stub.add_response(
            "describe_alarm_history",
            {
                "AlarmHistoryItems": [
                    {
                        "AlarmName": "alarm-a",
                        "Timestamp": T0 + timedelta(seconds=10),
                        "HistoryItemType": "StateUpdate",
                        "HistorySummary": "Alarm updated from INSUFFICIENT_DATA to OK",
                    }
                ]
            },
        )
        transitions = alarm_transitions(client, ["alarm-b", "alarm-a"], start=T0, end=T1)
        assert [t.alarm_name for t in transitions] == ["alarm-a", "alarm-b"]
        assert transitions[1].to_alarm
        assert not transitions[0].to_alarm

    def test_detects_insufficient_data_transition(self, cloudwatch) -> None:
        client, stub = cloudwatch
        stub.add_response(
            "describe_alarm_history",
            {
                "AlarmHistoryItems": [
                    {
                        "AlarmName": "a",
                        "Timestamp": T0,
                        "HistoryItemType": "StateUpdate",
                        "HistorySummary": "Alarm updated from OK to INSUFFICIENT_DATA",
                    }
                ]
            },
        )
        (transition,) = alarm_transitions(client, ["a"], start=T0, end=T1)
        assert transition.to_insufficient_data

    def test_one_unreadable_alarm_does_not_lose_the_others(self, cloudwatch) -> None:
        client, stub = cloudwatch
        stub.add_client_error("describe_alarm_history", service_error_code="AccessDenied")
        stub.add_response(
            "describe_alarm_history",
            {
                "AlarmHistoryItems": [
                    {
                        "AlarmName": "readable",
                        "Timestamp": T0,
                        "HistoryItemType": "StateUpdate",
                        "HistorySummary": "Alarm updated from OK to ALARM",
                    }
                ]
            },
        )
        transitions = alarm_transitions(client, ["denied", "readable"], start=T0, end=T1)
        assert [t.alarm_name for t in transitions] == ["readable"]

    def test_no_alarms_is_no_calls(self, cloudwatch) -> None:
        client, stub = cloudwatch
        assert alarm_transitions(client, [], start=T0, end=T1) == []
        stub.assert_no_pending_responses()


class TestScalingActivities:
    def test_filters_to_the_window_client_side(self, appscaling) -> None:
        client, stub = appscaling
        # DescribeScalingActivities has no time filter in the API, so the window
        # must be applied after fetching.
        stub.add_response(
            "describe_scaling_activities",
            {
                "ScalingActivities": [
                    _raw_activity("old", T0 - timedelta(hours=2)),
                    _raw_activity("inside", T0 + timedelta(minutes=1)),
                    _raw_activity("future", T1 + timedelta(hours=2)),
                ]
            },
        )
        activities = scaling_activities(client, endpoint=ENDPOINT, start=T0, end=T1)
        assert [a.activity_id for a in activities] == ["inside"]

    def test_unbounded_window_returns_everything_sorted(self, appscaling) -> None:
        client, stub = appscaling
        stub.add_response(
            "describe_scaling_activities",
            {
                "ScalingActivities": [
                    _raw_activity("second", T0 + timedelta(minutes=5)),
                    _raw_activity("first", T0),
                ]
            },
        )
        activities = scaling_activities(client, endpoint=ENDPOINT)
        assert [a.activity_id for a in activities] == ["first", "second"]

    def test_duration_needs_an_end_time(self, appscaling) -> None:
        client, stub = appscaling
        raw = _raw_activity("running", T0)
        raw.pop("EndTime")
        stub.add_response("describe_scaling_activities", {"ScalingActivities": [raw]})
        (activity,) = scaling_activities(client, endpoint=ENDPOINT)
        # An in-progress activity has no duration yet; 0.0 would look instant.
        assert activity.duration_s is None

    def test_duration_is_end_minus_start(self, appscaling) -> None:
        client, stub = appscaling
        stub.add_response(
            "describe_scaling_activities",
            {"ScalingActivities": [_raw_activity("done", T0, end=T0 + timedelta(seconds=210))]},
        )
        (activity,) = scaling_activities(client, endpoint=ENDPOINT)
        assert activity.duration_s == 210.0
        assert activity.succeeded

    def test_client_error_returns_empty(self, appscaling) -> None:
        client, stub = appscaling
        stub.add_client_error("describe_scaling_activities", service_error_code="AccessDenied")
        assert scaling_activities(client, endpoint=ENDPOINT) == []


class TestActivityStatusIsThreeWay:
    """``StatusCode`` has six values, so the predicates are not each other's negations.

    A live run read an ``InProgress`` activity — AWS pulling three images — and, treating
    ``not succeeded`` as failure, reported "1 of 1 scaling activities failed" while
    quoting *"Successfully set desired instance count to 4"*. Refusal and slow
    provisioning look identical from the endpoint and have opposite fixes.
    """

    @pytest.mark.parametrize("status", ["Pending", "InProgress"])
    def test_in_flight_is_neither_success_nor_failure(self, appscaling, status: str) -> None:
        client, stub = appscaling
        raw = _raw_activity("running", T0, status=status)
        raw.pop("EndTime")
        stub.add_response("describe_scaling_activities", {"ScalingActivities": [raw]})

        (activity,) = scaling_activities(client, endpoint=ENDPOINT)

        assert activity.in_flight
        assert not activity.failed
        assert not activity.succeeded

    @pytest.mark.parametrize("status", ["Failed", "Unfulfilled"])
    def test_terminal_refusals_are_failures(self, appscaling, status: str) -> None:
        # Unfulfilled included: AWS accepted the change and then could not deliver it,
        # which is a capacity problem for the operator either way.
        client, stub = appscaling
        stub.add_response(
            "describe_scaling_activities",
            {"ScalingActivities": [_raw_activity("refused", T0, status=status)]},
        )

        (activity,) = scaling_activities(client, endpoint=ENDPOINT)

        assert activity.failed
        assert not activity.in_flight
        assert not activity.succeeded

    def test_overridden_is_not_a_failure(self, appscaling) -> None:
        # A policy revising its own decision. Reporting it as a fault would make normal
        # target-tracking behaviour look like an error.
        client, stub = appscaling
        stub.add_response(
            "describe_scaling_activities",
            {"ScalingActivities": [_raw_activity("superseded", T0, status="Overridden")]},
        )

        (activity,) = scaling_activities(client, endpoint=ENDPOINT)

        assert not activity.failed
        assert not activity.in_flight
        assert not activity.succeeded

    def test_every_documented_status_is_classified_at_most_once(self, appscaling) -> None:
        # Guards against a future status landing in two buckets, or the enum growing a
        # value that quietly reads as success.
        from tts_bench.observe import ScalingActivity

        for status in [
            "Pending",
            "InProgress",
            "Successful",
            "Overridden",
            "Unfulfilled",
            "Failed",
        ]:
            activity = ScalingActivity(
                activity_id="a",
                start_time=T0,
                end_time=None,
                status_code=status,
                description="",
                cause="",
            )
            flags = [activity.succeeded, activity.failed, activity.in_flight]
            assert sum(flags) <= 1, f"{status} is in {sum(flags)} buckets"


def _raw_activity(
    activity_id: str,
    start: datetime,
    *,
    end: datetime | None = None,
    status: str = "Successful",
) -> dict:
    return {
        "ActivityId": activity_id,
        "ServiceNamespace": "sagemaker",
        "ResourceId": resource_id(ENDPOINT),
        "ScalableDimension": SCALABLE_DIMENSION,
        "Description": "Setting desired instance count to 2.",
        "Cause": "monitor alarm triggered a scaling activity",
        "StartTime": start,
        "EndTime": end or start + timedelta(seconds=180),
        "StatusCode": status,
    }


class TestListLive:
    def test_targets_skip_non_endpoint_resources(self, appscaling) -> None:
        client, stub = appscaling
        stub.add_response(
            "describe_scalable_targets",
            {
                "ScalableTargets": [
                    _raw_target(ENDPOINT),
                    {
                        **_raw_target(ENDPOINT),
                        "ResourceId": "inference-component/my-component",
                    },
                ]
            },
        )
        targets = list_live_targets(client)
        assert [t.resource_id for t in targets] == [resource_id(ENDPOINT)]

    def test_targets_carry_capacity_and_suspension(self, appscaling) -> None:
        client, stub = appscaling
        stub.add_response(
            "describe_scalable_targets",
            {
                "ScalableTargets": [
                    {
                        **_raw_target(ENDPOINT, min_capacity=2, max_capacity=6),
                        "SuspendedState": {"DynamicScalingOutSuspended": True},
                    }
                ]
            },
        )
        (target,) = list_live_targets(client)
        assert (target.min_capacity, target.max_capacity) == (2, 6)
        assert target.scale_out_suspended

    def test_policy_custom_metric_is_extracted(self, appscaling) -> None:
        client, stub = appscaling
        stub.add_response(
            "describe_scaling_policies",
            {"ScalingPolicies": [_raw_policy()]},
        )
        (policy,) = list_live_policies(client)
        assert policy.metric_namespace == "Speech/vLLM"
        assert policy.metric_name == "vllm:num_requests_running"
        assert policy.target_value == 8.0
        assert policy.uses_custom_metric
        assert policy.alarm_names == ("AlarmHigh-1", "AlarmLow-1")

    def test_predefined_metric_policy_has_no_namespace(self, appscaling) -> None:
        client, stub = appscaling
        raw = _raw_policy()
        raw["TargetTrackingScalingPolicyConfiguration"] = {
            "TargetValue": 0.44,
            "PredefinedMetricSpecification": {
                "PredefinedMetricType": ("SageMakerVariantConcurrentRequestsPerModelHighResolution")
            },
            "DisableScaleIn": True,
        }
        stub.add_response("describe_scaling_policies", {"ScalingPolicies": [raw]})
        (policy,) = list_live_policies(client)
        # This is what Phase 5 deploys. It must not be reported as inert, since
        # AWS publishes the metric natively.
        assert not policy.uses_custom_metric
        assert policy.metric_name == "SageMakerVariantConcurrentRequestsPerModelHighResolution"
        assert policy.disable_scale_in is True
        assert policy.target_value == pytest.approx(0.44)


def _raw_target(endpoint: str, *, min_capacity: int = 1, max_capacity: int = 4) -> dict:
    return {
        "ServiceNamespace": "sagemaker",
        "ResourceId": resource_id(endpoint),
        "ScalableDimension": SCALABLE_DIMENSION,
        "MinCapacity": min_capacity,
        "MaxCapacity": max_capacity,
        # Required by the ScalableTarget shape; Stubber validates responses.
        "RoleARN": ROLE_ARN,
        "CreationTime": T0,
    }


def _raw_policy(endpoint: str = ENDPOINT) -> dict:
    return {
        "PolicyARN": "arn:aws:autoscaling:us-east-1:1234:scalingPolicy:abc:resource/x:policyName/y",
        "PolicyName": "TrackRunningRequests",
        "ServiceNamespace": "sagemaker",
        "ResourceId": resource_id(endpoint),
        "ScalableDimension": SCALABLE_DIMENSION,
        "PolicyType": "TargetTrackingScaling",
        "TargetTrackingScalingPolicyConfiguration": {
            "TargetValue": 8.0,
            "CustomizedMetricSpecification": {
                "MetricName": "vllm:num_requests_running",
                "Namespace": "Speech/vLLM",
                "Statistic": "Average",
            },
        },
        "Alarms": [
            {"AlarmName": "AlarmHigh-1", "AlarmARN": "arn:aws:cloudwatch:us-east-1:1234:alarm:h"},
            {"AlarmName": "AlarmLow-1", "AlarmARN": "arn:aws:cloudwatch:us-east-1:1234:alarm:l"},
        ],
        "CreationTime": T0,
    }


class TestNamespaceIsPublishing:
    def test_namespace_with_metrics(self, cloudwatch) -> None:
        client, stub = cloudwatch
        stub.add_response(
            "list_metrics",
            {"Metrics": [{"Namespace": SAGEMAKER_NAMESPACE, "MetricName": "Invocations"}]},
        )
        assert namespace_is_publishing(client, SAGEMAKER_NAMESPACE)

    def test_empty_namespace_is_not_publishing(self, cloudwatch) -> None:
        client, stub = cloudwatch
        # This is the live state of Speech/vLLM: zero metrics, which is why the
        # two orphaned policies are dormant rather than active.
        stub.add_response("list_metrics", {"Metrics": []})
        assert not namespace_is_publishing(client, "Speech/vLLM")

    def test_failed_read_does_not_claim_inert(self, cloudwatch) -> None:
        client, stub = cloudwatch
        # Reporting "this policy can never fire" on a throttled read would be a
        # false all-clear about a policy that can move capacity.
        stub.add_client_error("list_metrics", service_error_code="Throttling")
        assert namespace_is_publishing(client, "Speech/vLLM")


class TestAlarmStates:
    def test_maps_names_to_states(self, cloudwatch) -> None:
        client, stub = cloudwatch
        stub.add_response(
            "describe_alarms",
            {
                "MetricAlarms": [
                    {"AlarmName": "a", "StateValue": "INSUFFICIENT_DATA"},
                    {"AlarmName": "b", "StateValue": "OK"},
                ]
            },
        )
        assert alarm_states(client, ["a", "b"]) == {"a": "INSUFFICIENT_DATA", "b": "OK"}

    def test_no_names_makes_no_call(self, cloudwatch) -> None:
        client, stub = cloudwatch
        assert alarm_states(client, []) == {}
        stub.assert_no_pending_responses()

    def test_chunks_above_the_hundred_name_limit(self, cloudwatch) -> None:
        client, stub = cloudwatch
        names = [f"alarm-{i}" for i in range(150)]
        # DescribeAlarms rejects more than 100 AlarmNames, so this must be two
        # calls. Stubber's param matching is what proves the chunk boundaries.
        stub.add_response(
            "describe_alarms",
            {"MetricAlarms": [{"AlarmName": n, "StateValue": "OK"} for n in names[:100]]},
            {"AlarmNames": names[:100]},
        )
        stub.add_response(
            "describe_alarms",
            {"MetricAlarms": [{"AlarmName": n, "StateValue": "OK"} for n in names[100:]]},
            {"AlarmNames": names[100:]},
        )
        assert len(alarm_states(client, names)) == 150
        stub.assert_no_pending_responses()


class TestAuditOrphans:
    def test_target_with_no_config_is_an_orphan_error(self) -> None:
        findings = audit_scaling(
            targets=[_target("speech-gone", max_capacity=4)],
            policies=[],
            expected={},
        )
        (finding,) = findings
        assert finding.kind is DriftKind.ORPHANED_TARGET
        assert finding.severity is Severity.ERROR
        assert finding.endpoint == "speech-gone"
        assert "deregister-scalable-target" in finding.remediation

    def test_target_whose_config_disables_scaling_is_an_orphan(self) -> None:
        # The live orpheus-3b case: config pins min=max=1 so CDK synthesizes
        # nothing, yet a max_capacity=4 target survives from an earlier deploy.
        findings = audit_scaling(
            targets=[_target("speech-orpheus-3b", min_capacity=1, max_capacity=4)],
            policies=[],
            expected={
                "speech-orpheus-3b": _expected(
                    "speech-orpheus-3b", min_instances=1, max_instances=1
                )
            },
        )
        (finding,) = findings
        assert finding.kind is DriftKind.ORPHANED_TARGET
        assert finding.severity is Severity.ERROR
        assert "cdk deploy will never remove it" in finding.detail
        # Phase 5 step 0 deregisters this exact target, so the remediation has to
        # be pasteable. Both branches of the orphan check reach that step; only
        # naming the alternative (raise max_instances) is not enough.
        assert "deregister-scalable-target" in finding.remediation
        assert finding.resource_id in finding.remediation

    def test_policy_without_a_synthesizing_config_is_an_orphan(self) -> None:
        findings = audit_scaling(
            targets=[],
            policies=[_policy("speech-orpheus-3b")],
            expected={
                "speech-orpheus-3b": _expected(
                    "speech-orpheus-3b", min_instances=1, max_instances=1
                )
            },
        )
        kinds = [f.kind for f in findings]
        assert DriftKind.ORPHANED_POLICY in kinds
        orphan = next(f for f in findings if f.kind is DriftKind.ORPHANED_POLICY)
        assert "delete-scaling-policy" in orphan.remediation

    def test_healthy_deployment_reports_nothing(self) -> None:
        # The post-Phase-5 state: drift must be silent, or it is useless as a
        # verification step.
        assert (
            audit_scaling(
                targets=[_target(min_capacity=1, max_capacity=4)],
                policies=[
                    _policy(namespace=None, metric="SageMakerVariantConcurrentRequestsPerModel")
                ],
                expected={ENDPOINT: _expected(min_instances=1, max_instances=4)},
                publishing_namespaces={},
            )
            == []
        )

    def test_inference_component_resources_are_ignored(self) -> None:
        target = LiveTarget(
            resource_id="inference-component/my-component", min_capacity=1, max_capacity=4
        )
        assert audit_scaling(targets=[target], policies=[], expected={}) == []


class TestAuditCapacityAndSuspension:
    def test_max_capacity_mismatch_is_a_warning(self) -> None:
        findings = audit_scaling(
            targets=[_target(max_capacity=4)],
            policies=[],
            expected={ENDPOINT: _expected(max_instances=8)},
        )
        (finding,) = findings
        assert finding.kind is DriftKind.CAPACITY_MISMATCH
        assert finding.severity is Severity.WARN
        assert "1-4" in finding.detail and "1-8" in finding.detail

    def test_config_min_zero_is_compared_as_one(self) -> None:
        # Both CDK constructs coerce min_instances=0 to 1, so a live min of 1
        # against a configured 0 is correct, not drift.
        findings = audit_scaling(
            targets=[_target(min_capacity=1, max_capacity=4)],
            policies=[],
            expected={ENDPOINT: _expected(min_instances=0, max_instances=4)},
        )
        assert findings == []

    def test_config_enabled_but_no_target_is_missing(self) -> None:
        findings = audit_scaling(
            targets=[],
            policies=[],
            expected={ENDPOINT: _expected(max_instances=4)},
        )
        (finding,) = findings
        assert finding.kind is DriftKind.MISSING_TARGET
        assert finding.severity is Severity.WARN

    def test_config_disabled_and_no_target_is_not_a_finding(self) -> None:
        # kokoro-82m and chatterbox today: max_instances=1, nothing registered.
        # Consistent, so silent.
        assert (
            audit_scaling(
                targets=[],
                policies=[],
                expected={ENDPOINT: _expected(min_instances=1, max_instances=1)},
            )
            == []
        )

    def test_suspended_scale_out_is_an_error(self) -> None:
        # Almost always an aborted freeze. Left in place it means the endpoint
        # cannot respond to load at all.
        findings = audit_scaling(
            targets=[_target(suspended={"DynamicScalingOutSuspended": True})],
            policies=[],
            expected={ENDPOINT: _expected(max_instances=4)},
        )
        suspended = next(f for f in findings if f.kind is DriftKind.SUSPENDED)
        assert suspended.severity is Severity.ERROR
        assert "tts-bench thaw" in suspended.remediation

    def test_suspended_scale_in_only_is_a_warning(self) -> None:
        findings = audit_scaling(
            targets=[_target(suspended={"DynamicScalingInSuspended": True})],
            policies=[],
            expected={ENDPOINT: _expected(max_instances=4)},
        )
        suspended = next(f for f in findings if f.kind is DriftKind.SUSPENDED)
        assert suspended.severity is Severity.WARN

    def test_all_flags_false_is_not_suspension(self) -> None:
        # The normal state. describe_scalable_targets returns all three as False
        # rather than omitting them, so a truthiness bug here would flag every
        # healthy endpoint.
        findings = audit_scaling(
            targets=[
                _target(
                    suspended={
                        "DynamicScalingInSuspended": False,
                        "DynamicScalingOutSuspended": False,
                        "ScheduledScalingSuspended": False,
                    }
                )
            ],
            policies=[],
            expected={ENDPOINT: _expected(max_instances=4)},
        )
        assert [f for f in findings if f.kind is DriftKind.SUSPENDED] == []


class TestAuditInertPolicies:
    def test_policy_on_a_silent_namespace_is_inert(self) -> None:
        findings = audit_scaling(
            targets=[_target(max_capacity=4)],
            policies=[_policy()],
            expected={ENDPOINT: _expected(max_instances=4)},
            publishing_namespaces={"Speech/vLLM": False},
        )
        inert = next(f for f in findings if f.kind is DriftKind.INERT_POLICY)
        assert "publishes no metrics" in inert.detail
        assert "SageMakerVariantConcurrentRequestsPerModel" in inert.remediation

    def test_policy_on_a_publishing_namespace_is_fine(self) -> None:
        findings = audit_scaling(
            targets=[_target(max_capacity=4)],
            policies=[_policy(namespace="Speech/Queue", metric="queue_depth")],
            expected={ENDPOINT: _expected(max_instances=4)},
            publishing_namespaces={"Speech/Queue": True},
        )
        assert [f for f in findings if f.kind is DriftKind.INERT_POLICY] == []

    def test_unknown_publishing_state_is_not_guessed(self) -> None:
        # Omitting the namespace means we did not check. Reporting inert anyway
        # would be an unverified claim about a policy that can move capacity.
        findings = audit_scaling(
            targets=[_target(max_capacity=4)],
            policies=[_policy()],
            expected={ENDPOINT: _expected(max_instances=4)},
            publishing_namespaces={},
        )
        assert [f for f in findings if f.kind is DriftKind.INERT_POLICY] == []

    def test_predefined_metric_policy_is_never_inert(self) -> None:
        findings = audit_scaling(
            targets=[_target(max_capacity=4)],
            policies=[_policy(namespace=None, metric="SageMakerVariantInvocationsPerInstance")],
            expected={ENDPOINT: _expected(max_instances=4)},
            publishing_namespaces={"Speech/vLLM": False},
        )
        assert [f for f in findings if f.kind is DriftKind.INERT_POLICY] == []


class TestAuditAlarms:
    def test_insufficient_data_alarm_is_reported(self) -> None:
        findings = audit_scaling(
            targets=[_target(max_capacity=4)],
            policies=[_policy(alarms=("AlarmHigh-1",))],
            expected={ENDPOINT: _expected(max_instances=4)},
            alarm_state={"AlarmHigh-1": "INSUFFICIENT_DATA"},
        )
        alarm = next(f for f in findings if f.kind is DriftKind.ALARM_INSUFFICIENT_DATA)
        assert "has never evaluated" in alarm.detail
        assert "TreatMissingData" in alarm.remediation

    def test_ok_alarm_is_not_reported(self) -> None:
        findings = audit_scaling(
            targets=[_target(max_capacity=4)],
            policies=[_policy(alarms=("AlarmHigh-1",))],
            expected={ENDPOINT: _expected(max_instances=4)},
            alarm_state={"AlarmHigh-1": "OK"},
        )
        assert [f for f in findings if f.kind is DriftKind.ALARM_INSUFFICIENT_DATA] == []

    def test_unknown_alarm_state_is_not_reported(self) -> None:
        findings = audit_scaling(
            targets=[_target(max_capacity=4)],
            policies=[_policy(alarms=("AlarmHigh-1",))],
            expected={ENDPOINT: _expected(max_instances=4)},
            alarm_state={},
        )
        assert [f for f in findings if f.kind is DriftKind.ALARM_INSUFFICIENT_DATA] == []


class TestCheckDrift:
    def test_reproduces_the_live_orphan_scenario(self, appscaling, cloudwatch) -> None:
        """The state of this account right now, end to end.

        Two orphaned policies on a namespace publishing zero metrics, whose
        alarms have never had data. Everything below is what the real APIs return
        today; the assertion is that `drift` names all three problems.
        """
        aas, aas_stub = appscaling
        cw, cw_stub = cloudwatch

        aas_stub.add_response(
            "describe_scalable_targets",
            {
                "ScalableTargets": [
                    _raw_target("speech-orpheus-3b", min_capacity=1, max_capacity=4),
                    _raw_target("speech-kokoro-82m-cpu", min_capacity=1, max_capacity=1),
                ]
            },
        )
        aas_stub.add_response(
            "describe_scaling_policies",
            {
                "ScalingPolicies": [
                    _raw_policy("speech-orpheus-3b"),
                    _raw_policy("speech-kokoro-82m-cpu"),
                ]
            },
        )
        cw_stub.add_response("list_metrics", {"Metrics": []})
        cw_stub.add_response(
            "describe_alarms",
            {
                "MetricAlarms": [
                    {"AlarmName": "AlarmHigh-1", "StateValue": "INSUFFICIENT_DATA"},
                    {"AlarmName": "AlarmLow-1", "StateValue": "INSUFFICIENT_DATA"},
                ]
            },
        )

        expected = {
            "speech-orpheus-3b": _expected("speech-orpheus-3b", min_instances=1, max_instances=1),
            "speech-kokoro-82m-cpu": _expected(
                "speech-kokoro-82m-cpu", min_instances=0, max_instances=1
            ),
        }
        findings = check_drift(aas, cw, expected)

        kinds = {f.kind for f in findings}
        assert DriftKind.ORPHANED_TARGET in kinds
        assert DriftKind.ORPHANED_POLICY in kinds
        assert DriftKind.INERT_POLICY in kinds
        assert DriftKind.ALARM_INSUFFICIENT_DATA in kinds

        # orpheus can reach 4 instances that no template describes: an error, not
        # a warning, because an unguarded benchmark would measure a fleet.
        orpheus = [f for f in findings if f.endpoint == "speech-orpheus-3b"]
        assert any(f.severity is Severity.ERROR for f in orpheus)

        aas_stub.assert_no_pending_responses()
        cw_stub.assert_no_pending_responses()

    def test_clean_account_reports_nothing(self, appscaling, cloudwatch) -> None:
        aas, aas_stub = appscaling
        cw, _ = cloudwatch
        aas_stub.add_response("describe_scalable_targets", {"ScalableTargets": []})
        aas_stub.add_response("describe_scaling_policies", {"ScalingPolicies": []})
        assert check_drift(aas, cw, {}) == []

    def test_only_checks_namespaces_it_found(self, appscaling, cloudwatch) -> None:
        aas, aas_stub = appscaling
        cw, cw_stub = cloudwatch
        aas_stub.add_response("describe_scalable_targets", {"ScalableTargets": []})
        aas_stub.add_response("describe_scaling_policies", {"ScalingPolicies": []})
        # No policies means no list_metrics and no describe_alarms calls at all.
        check_drift(aas, cw, {})
        cw_stub.assert_no_pending_responses()
