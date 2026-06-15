"""Model cache stack: CodeBuild project that syncs HF models to S3."""

from __future__ import annotations

import aws_cdk as cdk
import aws_cdk.aws_s3 as s3
from constructs import Construct

from speech_infra.config import ModelEndpointConfig
from speech_infra.constructs.model_cache import ModelCache


class SpeechModelCacheStack(cdk.Stack):
    """Syncs HuggingFace models to S3 for fast endpoint cold starts."""

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        model_bucket: s3.IBucket,
        model_configs: list[ModelEndpointConfig],
        hf_token_secret_name: str = "hf-token-dev",
        **kwargs: object,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        self.cache = ModelCache(
            self,
            "Cache",
            bucket=model_bucket,
            model_configs=model_configs,
            hf_token_secret_name=hf_token_secret_name,
        )
