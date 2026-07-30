"""Autoscaling construct for SageMaker real-time endpoints.

Two policies on one scalable target, because scale-out and scale-in want opposite
behaviour and no single policy expresses both:

* **Scale out fast**, on AWS's own high-resolution concurrency metric, with a short
  cooldown. Being slow to add capacity costs latency the SLO cannot pay back.
* **Scale in slowly**, one instance at a time, on a long cooldown. Being wrong here
  drops capacity that takes a full ``T_total`` to recover.

Application Auto Scaling takes the *maximum* of scale-out recommendations, so the
two compose rather than conflict — but only because the target-tracking policy sets
``disable_scale_in``. Without that it would remove instances on its own schedule and
fight the step policy.

The metric is deliberately AWS-published. The previous version tracked
``Speech/vLLM:vllm:num_requests_running``, a namespace nothing ever published to, so
its alarms sat in ``INSUFFICIENT_DATA`` and the policy could never fire.
"""

from __future__ import annotations

import aws_cdk as cdk
import aws_cdk.aws_applicationautoscaling as appscaling
import aws_cdk.aws_cloudwatch as cloudwatch
from constructs import Construct

from speech_infra.config import ModelEndpointConfig


class EndpointAutoscaling(Construct):
    """Target-tracking scale-out plus step scale-in, parameterized by model config."""

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

        resource_id = f"endpoint/{endpoint_name}/variant/{variant_name}"

        target = appscaling.ScalableTarget(
            self,
            "ScalableTarget",
            service_namespace=appscaling.ServiceNamespace.SAGEMAKER,
            scalable_dimension="sagemaker:variant:DesiredInstanceCount",
            resource_id=resource_id,
            min_capacity=max(model_config.min_instances, 1),
            max_capacity=model_config.max_instances,
        )
        self.target = target

        # Scale out on the native per-model concurrency metric. The high-resolution
        # variant publishes every 10s instead of every minute, which is the whole
        # reason to prefer it: at a 60s period the metric alone would add most of a
        # minute to T_total before an alarm could even evaluate.
        #
        # No dimensions are passed. Application Auto Scaling derives them from the
        # resource id, and supplying them is not permitted for a predefined metric.
        target.scale_to_track_metric(
            "TrackConcurrentRequests",
            predefined_metric=(
                appscaling.PredefinedMetric.SAGEMAKER_VARIANT_CONCURRENT_REQUESTS_PER_MODEL_HIGH_RESOLUTION
            ),
            target_value=model_config.scaling_target_value,
            # Scale-in belongs to the step policy below. Leaving it enabled here
            # would have both policies removing instances on different schedules.
            disable_scale_in=True,
            scale_out_cooldown=cdk.Duration.seconds(model_config.scale_out_cooldown_s),
        )

        # Scale in with an explicit metric rather than the predefined one, because a
        # step policy needs a metric object to build its alarm from. Same underlying
        # measurement at the standard 60s period — scale-in has no reason to hurry,
        # and a slower period is less twitchy.
        scale_in_metric = cloudwatch.Metric(
            namespace=model_config.scaling_metric_namespace,
            metric_name=model_config.scaling_metric_name,
            dimensions_map={
                "EndpointName": endpoint_name,
                "VariantName": variant_name,
            },
            period=cdk.Duration.minutes(1),
            statistic="Average",
        )

        target.scale_on_metric(
            "ScaleInOneAtATime",
            metric=scale_in_metric,
            scaling_steps=[
                appscaling.ScalingInterval(
                    upper=model_config.scale_in_threshold,
                    change=-1,
                ),
                # A no-op interval covering everything above the threshold. CDK
                # rejects a single-interval step policy ("You must supply at least 2
                # intervals"), and a change of 0 renders as no step adjustment at
                # all, so this buys the shape the API wants without adding behaviour.
                appscaling.ScalingInterval(
                    lower=model_config.scale_in_threshold,
                    change=0,
                ),
            ],
            adjustment_type=appscaling.AdjustmentType.CHANGE_IN_CAPACITY,
            cooldown=cdk.Duration.seconds(model_config.scale_in_cooldown_s),
            # Three consecutive quiet minutes before removing an instance. A single
            # datapoint below the threshold is just a gap between requests.
            evaluation_periods=3,
            datapoints_to_alarm=3,
            metric_aggregation_type=appscaling.MetricAggregationType.AVERAGE,
        )

        if model_config.emergency_step_enabled:
            # For load that arrives faster than target tracking converges: it adds
            # roughly one step per cooldown, so a sudden multiple of C_target waits
            # several cooldowns for capacity it needed at once. Scale-out only — the
            # step policy above owns scale-in, and two policies removing instances
            # would race.
            target.scale_on_metric(
                "EmergencyScaleOut",
                metric=scale_in_metric,
                scaling_steps=[
                    appscaling.ScalingInterval(
                        upper=model_config.scaling_target_value * 2,
                        change=0,
                    ),
                    appscaling.ScalingInterval(
                        lower=model_config.scaling_target_value * 2,
                        change=+2,
                    ),
                    appscaling.ScalingInterval(
                        lower=model_config.scaling_target_value * 4,
                        change=+4,
                    ),
                ],
                adjustment_type=appscaling.AdjustmentType.CHANGE_IN_CAPACITY,
                cooldown=cdk.Duration.seconds(model_config.scale_out_cooldown_s),
                evaluation_periods=1,
                metric_aggregation_type=appscaling.MetricAggregationType.AVERAGE,
            )
