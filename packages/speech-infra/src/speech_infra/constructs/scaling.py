"""Autoscaling construct for SageMaker real-time endpoints."""

from __future__ import annotations

import aws_cdk as cdk
import aws_cdk.aws_applicationautoscaling as appscaling
import aws_cdk.aws_cloudwatch as cloudwatch
from constructs import Construct

from speech_infra.config import ModelEndpointConfig


class EndpointAutoscaling(Construct):
    """Target-tracking autoscaling parameterized by model config."""

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

        metric = cloudwatch.Metric(
            namespace=model_config.scaling_metric_namespace,
            metric_name=model_config.scaling_metric_name,
            dimensions_map={
                "EndpointName": endpoint_name,
                "ModelName": model_config.model_name,
            },
            period=cdk.Duration.minutes(1),
            statistic="Average",
        )

        target.scale_to_track_metric(
            "TrackRunningRequests",
            target_value=model_config.scaling_target_value,
            custom_metric=metric,
            scale_in_cooldown=cdk.Duration.minutes(30),
            scale_out_cooldown=cdk.Duration.minutes(2),
        )
