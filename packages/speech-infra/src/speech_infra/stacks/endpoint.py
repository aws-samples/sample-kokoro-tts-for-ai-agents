"""Per-model endpoint stack with container type dispatch."""

from __future__ import annotations

import aws_cdk as cdk
import aws_cdk.aws_ecr_assets as ecr_assets
import aws_cdk.aws_iam as iam
from constructs import Construct

from speech_infra.config import ModelEndpointConfig
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

        autoscaling = EndpointAutoscaling(
            self,
            "Autoscaling",
            model_config=model_config,
            endpoint_name=model_config.endpoint_name,
        )
        autoscaling.node.add_dependency(endpoint)
