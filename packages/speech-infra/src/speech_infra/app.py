"""CDK App entrypoint for speech model infrastructure."""

from __future__ import annotations

from pathlib import Path

import aws_cdk as cdk

from speech_infra.config import (
    TTS_MODEL_CONFIGS,
    ContainerType,
    ModelEndpointConfig,
)
from speech_infra.stacks.endpoint import SpeechEndpointStack
from speech_infra.stacks.foundation import SpeechFoundationStack
from speech_infra.stacks.model_cache import SpeechModelCacheStack

CONTAINERS_ROOT = Path(__file__).resolve().parent.parent.parent / "containers"

CONTAINER_DIR_MAP: dict[str, str] = {
    "orpheus-3b": str(CONTAINERS_ROOT / "vllm"),
    "maya-veena": str(CONTAINERS_ROOT / "vllm"),
    "kokoro-82m": str(CONTAINERS_ROOT / "kokoro"),
    "kokoro-82m-cpu": str(CONTAINERS_ROOT / "kokoro-cpu"),
    "chatterbox-turbo": str(CONTAINERS_ROOT / "chatterbox"),
}


def _get_container_dir(config: ModelEndpointConfig) -> str:
    if config.model_name in CONTAINER_DIR_MAP:
        return CONTAINER_DIR_MAP[config.model_name]
    if config.container_type == ContainerType.VLLM:
        return str(CONTAINERS_ROOT / "vllm")
    return str(CONTAINERS_ROOT / config.model_name)


def _apply_instance_type_override(app: cdk.App, config: ModelEndpointConfig) -> ModelEndpointConfig:
    """Re-type one model's endpoint from CDK context, for a measurement run.

    ``-c kokoro-82m:instance_type=ml.g6.12xlarge`` — scoped per model rather than
    global, so trying a candidate type cannot silently re-type every stack in the app
    at once. ``config.py`` remains the source of truth for what is deployed long-term;
    this exists so evaluating a type is a flag rather than an edit-commit-deploy cycle.

    What keeps the resulting measurements honest is the configuration fingerprint
    ``tts-bench`` reads back off the endpoint: an artifact records the type it was
    actually measured against, not the one this file declares.
    """
    override = app.node.try_get_context(f"{config.model_name}:instance_type")
    if not override:
        return config
    # Revalidated rather than `model_copy(update=...)`, which skips validators in
    # Pydantic v2 — the `ml.` prefix check is the whole reason a typo fails at synth
    # instead of sitting in an Updating endpoint with no FailureReason.
    return ModelEndpointConfig.model_validate(
        {**config.model_dump(), "instance_type": str(override)}
    )


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

    cacheable_configs = [cfg for cfg in TTS_MODEL_CONFIGS.values() if cfg.cache_model_weights]

    model_cache = SpeechModelCacheStack(
        app,
        "SpeechModelCache",
        model_bucket=foundation.model_bucket,
        model_configs=cacheable_configs,
        hf_token_secret_name=hf_token_secret,
        env=env,
    )
    model_cache.add_dependency(foundation)

    for _model_name, declared_config in TTS_MODEL_CONFIGS.items():
        model_config = _apply_instance_type_override(app, declared_config)
        container_dir = _get_container_dir(model_config)
        model_bucket_name = (
            foundation.model_bucket.bucket_name if model_config.cache_model_weights else None
        )

        endpoint_stack = SpeechEndpointStack(
            app,
            model_config.stack_id,
            model_config=model_config,
            execution_role=foundation.execution_role,
            container_dir=container_dir,
            image_uri_override=image_uri_override,
            model_bucket_name=model_bucket_name,
            env=env,
        )
        if model_config.cache_model_weights:
            endpoint_stack.add_dependency(model_cache)

    return app


if __name__ == "__main__":
    app = create_app()
    app.synth()
