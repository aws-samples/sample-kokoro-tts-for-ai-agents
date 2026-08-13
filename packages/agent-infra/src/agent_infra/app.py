"""CDK App entrypoint for the agent-infra demo."""

from __future__ import annotations

from pathlib import Path

import aws_cdk as cdk

from agent_infra.config import AgentRuntimeConfig
from agent_infra.stacks.foundation import AgentFoundationStack
from agent_infra.stacks.runtime import AgentRuntimeStack

REPO_ROOT = Path(__file__).resolve().parents[4]


def create_app() -> cdk.App:
    """Create the CDK app with the foundation and runtime stacks."""
    app = cdk.App()

    region = app.node.try_get_context("region") or "us-east-1"
    env = cdk.Environment(
        account=app.node.try_get_context("account"),
        region=region,
    )

    image_uri_override = app.node.try_get_context("image_uri") or None

    bedrock_model_id = app.node.try_get_context("agent:bedrock_model_id")
    if not bedrock_model_id:
        raise ValueError(
            "bedrock_model_id is required — pass -c agent:bedrock_model_id=<id>, "
            "confirmed via `aws bedrock list-foundation-models --by-provider anthropic`"
        )
    config = AgentRuntimeConfig(bedrock_model_id=str(bedrock_model_id))

    foundation = AgentFoundationStack(app, "AgentFoundation", config=config, env=env)

    runtime_stack = AgentRuntimeStack(
        app,
        "AgentRuntime",
        config=config,
        execution_role=foundation.execution_role,
        repo_root=REPO_ROOT,
        image_uri_override=image_uri_override,
        env=env,
    )
    runtime_stack.add_dependency(foundation)

    return app


if __name__ == "__main__":
    app = create_app()
    app.synth()
