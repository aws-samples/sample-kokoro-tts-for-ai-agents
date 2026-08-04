"""Alarms and a dashboard for a deployed endpoint.

Diagnostic only, and deliberately so. Scaling runs off AWS's own predefined metric
inside Application Auto Scaling (see :mod:`speech_infra.constructs.scaling`), so
nothing here can affect capacity — a misconfigured alarm degrades what an operator
can see, never what the fleet does.

Every metric used is one this account has been confirmed to publish for a live
endpoint. Two namespaces are involved, which is easy to get wrong: invocation and
latency metrics are ``AWS/SageMaker`` at ``{EndpointName, VariantName}``, while
utilization metrics live in ``/aws/sagemaker/Endpoints``.

**The concurrency widgets read ``Average``; the scale-out policy reads ``Maximum``.**
Same metric name, different statistic, and the two are not interchangeable: on one kokoro
ladder ``Maximum`` ran from 9.8x the client's own mean in-flight down to 1.35x as load
rose. So a threshold line drawn here sits where the *policy's* statistic would cross it,
not where this graph's average will — the graph is for seeing the trend, and the scaling
decision belongs to the policy. Deploying a client-measured occupancy against ``Maximum``
without that conversion is how ``scaling_target_value=0.713`` — a value no positive
arrival rate satisfies — reached this endpoint.
"""

from __future__ import annotations

from pathlib import Path

import aws_cdk as cdk
import aws_cdk.aws_cloudwatch as cloudwatch
from constructs import Construct

from speech_infra import measurements
from speech_infra.config import ModelEndpointConfig

#: Namespace for invocation, latency and concurrency metrics.
SAGEMAKER_NAMESPACE = "AWS/SageMaker"

#: Namespace for per-instance utilization. A different namespace with a different
#: dimension set, not a different metric name in the same one.
ENDPOINT_NAMESPACE = "/aws/sagemaker/Endpoints"


class EndpointObservability(Construct):
    """Alarms plus a dashboard for one endpoint variant."""

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        model_config: ModelEndpointConfig,
        endpoint_name: str,
        variant_name: str = "primary",
        artifact_dir: Path | None = None,
    ) -> None:
        """Args:
        artifact_dir: Where to look for the ``qmax`` artifact the latency alarm's
            threshold comes from. ``None`` reads
            :data:`~speech_infra.measurements.ARTIFACT_DIR`, which is what the app
            does; tests pass a temporary directory so a synth assertion does not
            depend on whether a benchmark happens to have been run in this checkout.
        """
        super().__init__(scope, construct_id)

        dims = {"EndpointName": endpoint_name, "VariantName": variant_name}
        period = cdk.Duration.minutes(1)

        def sagemaker_metric(name: str, statistic: str) -> cloudwatch.Metric:
            return cloudwatch.Metric(
                namespace=SAGEMAKER_NAMESPACE,
                metric_name=name,
                dimensions_map=dims,
                period=period,
                statistic=statistic,
            )

        # p95 rather than average: the SLO is stated as a percentile, and an average
        # first-chunk latency can sit comfortably inside budget while the tail is
        # entirely outside it.
        self.first_chunk_latency = sagemaker_metric("FirstChunkLatency", "p95")
        self.model_latency = sagemaker_metric("ModelLatency", "p95")
        self.concurrent_requests = sagemaker_metric("ConcurrentRequestsPerModel", "Average")
        self.invocations = sagemaker_metric("Invocations", "Sum")
        self.errors_5xx = sagemaker_metric("Invocation5XXErrors", "Sum")
        self.errors_4xx = sagemaker_metric("Invocation4XXErrors", "Sum")

        self.gpu_utilization = cloudwatch.Metric(
            namespace=ENDPOINT_NAMESPACE,
            metric_name="GPUUtilization",
            dimensions_map={"EndpointName": endpoint_name, "VariantName": variant_name},
            period=period,
            statistic="Average",
        )

        # The service-degradation alarm. FirstChunkLatency is SageMaker's name for
        # what the benchmark calls TTFAB; CloudWatch reports it in microseconds.
        #
        # Its threshold is the Q_max ladder's own N=1 rung -- p95 first byte with one
        # request outstanding -- and *not* ttfab_slo_ms. The two differ by an order of
        # magnitude, and this metric is the tighter one's: it is measured on an instance
        # already serving, where an in-flight request has spent none of its queue
        # allowance. Threshold it at the end-to-end SLO and the alarm fires only once
        # service time alone is ~10x past where the model stops keeping up, by which
        # point the queue has been missing the promise for a long time. The SLO is held
        # by sizing the fleet and bounding the queue; this alarm is how we notice the
        # instance itself degrading.
        #
        # Measured rather than configured. It used to be a second hand-set config field
        # (a 300ms ttfab_budget_ms beside the 3000ms SLO) with nothing tying either to a
        # measurement -- two independent latency fields that could disagree with the
        # promise, and did. Reading the ladder means it re-measures on every rerun,
        # including on a new instance type.
        #
        # No measurement, no alarm. A threshold has to come from somewhere real, so an
        # unmeasured model gets no alarm rather than an invented one -- and a fresh
        # checkout with no artifacts/ still synths.
        c1_ms = measurements.ttfab_p95_at_c1_ms(model_config.model_name, artifact_dir=artifact_dir)
        self.ttfab_alarm: cloudwatch.Alarm | None = None
        if c1_ms is not None:
            self.ttfab_alarm = cloudwatch.Alarm(
                self,
                "FirstChunkLatencyP95",
                metric=self.first_chunk_latency,
                threshold=c1_ms * 1000,
                evaluation_periods=3,
                datapoints_to_alarm=2,
                comparison_operator=cloudwatch.ComparisonOperator.GREATER_THAN_THRESHOLD,
                # An idle endpoint publishes no latency at all. Treating that as missing
                # rather than breaching keeps the alarm quiet overnight instead of
                # training operators to ignore it.
                treat_missing_data=cloudwatch.TreatMissingData.NOT_BREACHING,
                alarm_description=(
                    f"p95 first-chunk latency on {endpoint_name} over {c1_ms:.0f}ms, the "
                    "service time measured at one outstanding request. Not the end-to-end "
                    f"{model_config.ttfab_slo_ms}ms SLO: a request in flight here has spent "
                    "none of its queue allowance."
                ),
            )

        self.errors_alarm = cloudwatch.Alarm(
            self,
            "ServerErrors",
            metric=self.errors_5xx,
            threshold=1,
            evaluation_periods=1,
            comparison_operator=(cloudwatch.ComparisonOperator.GREATER_THAN_OR_EQUAL_TO_THRESHOLD),
            treat_missing_data=cloudwatch.TreatMissingData.NOT_BREACHING,
            alarm_description=f"{endpoint_name} returned a 5XX",
        )

        # Sustained saturation. Target tracking should be adding instances well
        # before this fires, so if it does fire the interesting question is why
        # scaling did not — max_instances reached, or the policy not evaluating.
        self.saturation_alarm = cloudwatch.Alarm(
            self,
            "ConcurrencyOverTarget",
            metric=self.concurrent_requests,
            threshold=model_config.scaling_target_value,
            evaluation_periods=5,
            datapoints_to_alarm=5,
            comparison_operator=cloudwatch.ComparisonOperator.GREATER_THAN_THRESHOLD,
            treat_missing_data=cloudwatch.TreatMissingData.NOT_BREACHING,
            alarm_description=(
                f"concurrency above C_scale_max ({model_config.scaling_target_value}) for "
                "five minutes; scale-out should already have responded"
            ),
        )

        self.dashboard = cloudwatch.Dashboard(
            self,
            "Dashboard",
            dashboard_name=f"{endpoint_name}-scaling",
        )
        self.dashboard.add_widgets(
            cloudwatch.GraphWidget(
                title="Concurrency vs scaling thresholds",
                left=[self.concurrent_requests],
                # Both thresholds drawn on the same axes as the measurement, so the
                # scaling decision is legible without arithmetic. The band between them
                # is where the fleet is meant to sit; Q_max is not drawn because it is a
                # per-instance number and this metric is per-model.
                left_annotations=[
                    cloudwatch.HorizontalAnnotation(
                        value=model_config.scaling_target_value,
                        label="C_scale_max (scale out)",
                    ),
                    cloudwatch.HorizontalAnnotation(
                        value=model_config.scale_in_threshold,
                        label="C_scale_min (scale in)",
                    ),
                ],
                width=12,
            ),
            cloudwatch.GraphWidget(
                title="Latency p95",
                left=[self.first_chunk_latency, self.model_latency],
                # Both lines the latency has to stay under, drawn together because the
                # gap between them *is* the queue allowance. Only the SLO is guaranteed
                # to be there: the measured one is absent until a ladder has run.
                left_annotations=[
                    cloudwatch.HorizontalAnnotation(
                        value=model_config.ttfab_slo_ms * 1000,
                        label=f"end-to-end SLO {model_config.ttfab_slo_ms}ms (queue included)",
                    ),
                    *(
                        [
                            cloudwatch.HorizontalAnnotation(
                                value=c1_ms * 1000,
                                label=f"service time at N=1, measured: {c1_ms:.0f}ms",
                            )
                        ]
                        if c1_ms is not None
                        else []
                    ),
                ],
                width=12,
            ),
            cloudwatch.GraphWidget(
                title="Invocations and errors",
                left=[self.invocations],
                right=[self.errors_5xx, self.errors_4xx],
                width=12,
            ),
            cloudwatch.GraphWidget(
                title="GPU utilization",
                left=[self.gpu_utilization],
                width=12,
            ),
        )
