"""CDK App entrypoint for speech model infrastructure."""

from __future__ import annotations

from pathlib import Path

import aws_cdk as cdk

from speech_infra.config import (
    TTS_MODEL_CONFIGS,
    ContainerType,
)
from speech_infra.stacks.endpoint import SpeechEndpointStack
from speech_infra.stacks.foundation import SpeechFoundationStack
from speech_infra.stacks.model_cache import SpeechModelCacheStack

CONTAINER_DIR = str(Path(__file__).resolve().parent.parent.parent / "containers" / "vllm")


def create_app() -> cdk.App:
    """Create the CDK app with all stacks."""
    app = cdk.App()

    region = app.node.try_get_context("region") or "us-east-1"
    env = cdk.Environment(
        account=app.node.try_get_context("account"),
        region=region,
    )

    image_uri_override = app.node.try_get_context("image_uri") or None

    foundation = SpeechFoundationStack(app, "SpeechFoundation", env=env)

    hf_token_secret = app.node.try_get_context("hf_token_secret") or "hf-token-dev"

    vllm_configs = {
        name: cfg
        for name, cfg in TTS_MODEL_CONFIGS.items()
        if cfg.container_type == ContainerType.VLLM
    }

    model_cache = SpeechModelCacheStack(
        app,
        "SpeechModelCache",
        model_bucket=foundation.model_bucket,
        model_configs=list(vllm_configs.values()),
        hf_token_secret_name=hf_token_secret,
        env=env,
    )
    model_cache.add_dependency(foundation)

    for _model_name, model_config in vllm_configs.items():
        endpoint_stack = SpeechEndpointStack(
            app,
            model_config.stack_id,
            model_config=model_config,
            execution_role=foundation.execution_role,
            container_dir=CONTAINER_DIR,
            image_uri_override=image_uri_override,
            model_bucket_name=foundation.model_bucket.bucket_name,
            env=env,
        )
        endpoint_stack.add_dependency(model_cache)

    return app


if __name__ == "__main__":
    app = create_app()
    app.synth()
