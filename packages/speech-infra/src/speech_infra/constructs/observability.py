"""Alarms and a dashboard for a deployed endpoint.

Diagnostic only, and deliberately so. Scaling runs off AWS's own predefined metric
inside Application Auto Scaling (see :mod:`speech_infra.constructs.scaling`), so
nothing here can affect capacity — a misconfigured alarm degrades what an operator
can see, never what the fleet does.

Every metric used is one this account has been confirmed to publish for a live
endpoint. Two namespaces are involved, which is easy to get wrong: invocation and
latency metrics are ``AWS/SageMaker`` at ``{EndpointName, VariantName}``, while
utilization metrics live in ``/aws/sagemaker/Endpoints``.
"""

from __future__ import annotations

import aws_cdk as cdk
import aws_cdk.aws_cloudwatch as cloudwatch
from constructs import Construct

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
    ) -> None:
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

        # The SLO alarm. FirstChunkLatency is SageMaker's name for what the
        # benchmark calls TTFAB, so this is the one alarm that watches the quantity
        # C_max was measured against. Reported in microseconds by CloudWatch.
        self.ttfab_alarm = cloudwatch.Alarm(
            self,
            "FirstChunkLatencyP95",
            metric=self.first_chunk_latency,
            threshold=model_config.ttfab_budget_ms * 1000,
            evaluation_periods=3,
            datapoints_to_alarm=2,
            comparison_operator=cloudwatch.ComparisonOperator.GREATER_THAN_THRESHOLD,
            # An idle endpoint publishes no latency at all. Treating that as missing
            # rather than breaching keeps the alarm quiet overnight instead of
            # training operators to ignore it.
            treat_missing_data=cloudwatch.TreatMissingData.NOT_BREACHING,
            alarm_description=(
                f"p95 first-chunk latency over the {model_config.ttfab_budget_ms}ms budget "
                f"that {endpoint_name}'s C_target was derived against"
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
                f"concurrency above C_target ({model_config.scaling_target_value}) for five "
                "minutes; scale-out should already have responded"
            ),
        )

        self.dashboard = cloudwatch.Dashboard(
            self,
            "Dashboard",
            dashboard_name=f"{endpoint_name}-scaling",
        )
        self.dashboard.add_widgets(
            cloudwatch.GraphWidget(
                title="Concurrency vs C_target",
                left=[self.concurrent_requests],
                # The target drawn on the same axes as the measurement, so the
                # scaling decision is legible without arithmetic.
                left_annotations=[
                    cloudwatch.HorizontalAnnotation(
                        value=model_config.scaling_target_value,
                        label="C_target",
                    ),
                    cloudwatch.HorizontalAnnotation(
                        value=model_config.scale_in_threshold,
                        label="scale-in threshold",
                    ),
                ],
                width=12,
            ),
            cloudwatch.GraphWidget(
                title="Latency p95",
                left=[self.first_chunk_latency, self.model_latency],
                left_annotations=[
                    cloudwatch.HorizontalAnnotation(
                        value=model_config.ttfab_budget_ms * 1000,
                        label=f"TTFAB budget {model_config.ttfab_budget_ms}ms",
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
