"""Per-model endpoint stack with container type dispatch."""

from __future__ import annotations

from pathlib import Path

import aws_cdk as cdk
import aws_cdk.aws_ecr_assets as ecr_assets
import aws_cdk.aws_iam as iam
from constructs import Construct

from speech_infra import measurements
from speech_infra.config import ModelEndpointConfig
from speech_infra.constructs.observability import EndpointObservability
from speech_infra.constructs.scaling import EndpointAutoscaling
from speech_infra.constructs.vllm_endpoint import VllmStreamingEndpoint


class SpeechEndpointStack(cdk.Stack):
    """Deploys a single model's endpoint with autoscaling."""

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        model_config: ModelEndpointConfig,
        execution_role: iam.IRole,
        container_dir: str,
        image_uri_override: str | None = None,
        model_bucket_name: str | None = None,
        artifact_dir: Path | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        if image_uri_override:
            image_uri = image_uri_override
        else:
            asset = ecr_assets.DockerImageAsset(
                self,
                "VllmImage",
                directory=container_dir,
            )
            image_uri = asset.image_uri

        env_overrides: dict[str, str] = dict(model_config.container_env)
        env_overrides["ENDPOINT_NAME"] = model_config.endpoint_name
        env_overrides["SM_MODEL_ID"] = model_config.model_name
        env_overrides["AWS_REGION"] = self.region
        env_overrides["AWS_DEFAULT_REGION"] = self.region

        if model_config.queue_max_depth > 0:
            env_overrides["MAX_QUEUE_DEPTH"] = str(model_config.queue_max_depth)

        if model_bucket_name:
            s3_uri = f"s3://{model_bucket_name}/models/{model_config.hf_model_id}/"
            env_overrides["MODEL_S3_URI"] = s3_uri

            if model_config.codec_model_ids:
                codec_id = model_config.codec_model_ids[0]
                env_overrides["SNAC_S3_URI"] = f"s3://{model_bucket_name}/models/{codec_id}/"

        endpoint = VllmStreamingEndpoint(
            self,
            "GpuEndpoint",
            model_config=model_config,
            execution_role=execution_role,
            image_uri=image_uri,
            env_overrides=env_overrides,
        )

        # scaling_enabled alone is a static config property (max_instances >
        # min_instances) — it says nothing about whether the thresholds inside are
        # measured. scaling_target_value is a plain float, so a hand-set number and a
        # `plan`-computed one are indistinguishable by type; that indistinguishability
        # is how 0.713 -- a client occupancy deployed against a server statistic,
        # satisfiable by no positive arrival rate -- reached this endpoint without
        # anything refusing to synth it. scaling_thresholds_measured reads the actual
        # `plan` artifact, mirroring the "no measurement, no alarm" rule
        # EndpointObservability's own ttfab_alarm already applies.
        if model_config.scaling_enabled and measurements.scaling_thresholds_measured(
            model_config.model_name, artifact_dir=artifact_dir
        ):
            autoscaling = EndpointAutoscaling(
                self,
                "Autoscaling",
                model_config=model_config,
                endpoint_name=model_config.endpoint_name,
            )
            autoscaling.node.add_dependency(endpoint)

            # Gated on the same condition as scaling, not added unconditionally: the
            # alarms are all about whether scaling is keeping up, and the dashboard
            # annotates the two scaling thresholds, which a non-scaling model has no
            # measured values for.
            observability = EndpointObservability(
                self,
                "Observability",
                model_config=model_config,
                endpoint_name=model_config.endpoint_name,
                artifact_dir=artifact_dir,
            )
            observability.node.add_dependency(endpoint)
