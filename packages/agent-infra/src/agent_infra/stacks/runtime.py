# Copyright (c) 2026 Amazon.com
# This file is licensed under the MIT License.
# See the LICENSE file in the project root for full license information.

"""Runtime stack: the container image and the AgentCore Runtime it powers.

``aws_bedrockagentcore`` in this repo's installed aws-cdk-lib (2.253.1) exposes
only the L1 ``CfnRuntime``/``CfnRuntimeEndpoint`` for this resource — confirmed
via ``dir(aws_cdk.aws_bedrockagentcore)``, no L2 wrapper exists to prefer.
"""

from __future__ import annotations

from pathlib import Path

import aws_cdk as cdk
import aws_cdk.aws_bedrockagentcore as bedrockagentcore
import aws_cdk.aws_ecr_assets as ecr_assets
import aws_cdk.aws_iam as iam
from constructs import Construct

from agent_infra.config import AgentRuntimeConfig


class AgentRuntimeStack(cdk.Stack):
    """Builds the agent container and deploys it to AgentCore Runtime."""

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        config: AgentRuntimeConfig,
        execution_role: iam.IRole,
        repo_root: Path,
        image_uri_override: str | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        if image_uri_override:
            image_uri = image_uri_override
        else:
            # directory is the repo root, not container/: the Dockerfile's
            # `COPY packages/tts-client/` needs that directory in its build
            # context, since tts-client is installed straight from source.
            asset = ecr_assets.DockerImageAsset(
                self,
                "AgentImage",
                directory=str(repo_root),
                file="packages/agent-infra/container/Dockerfile",
                platform=ecr_assets.Platform.LINUX_ARM64,
                exclude=[
                    ".venv",
                    "**/.venv",
                    "**/__pycache__",
                    "node_modules",
                    ".git",
                    "packages/models",
                    # cdk.out lives inside packages/agent-infra/, which is
                    # inside this same copied tree -- without excluding it,
                    # CDK copies its own output directory into itself
                    # recursively (a real ENAMETOOLONG hit while testing this).
                    "**/cdk.out",
                    # A harness state directory that happens to live at repo
                    # root here, unrelated to the image and enormous (spans
                    # other unrelated projects' session history).
                    ".claude",
                ],
            )
            image_uri = asset.image_uri

        self.runtime = bedrockagentcore.CfnRuntime(
            self,
            "AgentRuntime",
            # Many Bedrock resource-name fields reject hyphens (alphanumeric +
            # underscore only); no CDK-level validator confirms this one either
            # way, so the substitution is defensive rather than a proven need.
            agent_runtime_name=config.agent_runtime_name.replace("-", "_"),
            agent_runtime_artifact=bedrockagentcore.CfnRuntime.AgentRuntimeArtifactProperty(
                container_configuration=bedrockagentcore.CfnRuntime.ContainerConfigurationProperty(
                    container_uri=image_uri,
                ),
            ),
            network_configuration=bedrockagentcore.CfnRuntime.NetworkConfigurationProperty(
                network_mode="PUBLIC",
            ),
            role_arn=execution_role.role_arn,
            environment_variables={
                "AGENT_TTS_ENDPOINT": config.tts_endpoint_name,
                "AGENT_TTS_VOICE": config.tts_voice,
                "AGENT_BEDROCK_MODEL_ID": config.bedrock_model_id,
                "AWS_REGION": self.region,
            },
            description="Strands agent streaming TTS via Kokoro — bidi vs. batch mode demo.",
        )

        self.runtime_endpoint = bedrockagentcore.CfnRuntimeEndpoint(
            self,
            "AgentRuntimeEndpoint",
            agent_runtime_id=self.runtime.attr_agent_runtime_id,
            name="default",
        )
        self.runtime_endpoint.add_dependency(self.runtime)

        cdk.CfnOutput(self, "AgentRuntimeArn", value=self.runtime.attr_agent_runtime_arn)
        cdk.CfnOutput(
            self,
            "AgentRuntimeEndpointArn",
            value=self.runtime_endpoint.attr_agent_runtime_endpoint_arn,
        )
