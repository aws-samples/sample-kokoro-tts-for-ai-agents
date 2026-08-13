"""Synth assertions for the foundation and runtime stacks.

Read the rendered CloudFormation rather than the Python constructs -- that's
the only way to catch CDK silently dropping a property it doesn't support in
the position given, the same reasoning speech-infra's own
``test_scaling_synth.py`` documents for the same pattern.

``image_uri_override`` keeps ``DockerImageAsset`` out of the synth, so these
run as ordinary unit tests with no Docker daemon and no AWS credentials.
"""

from __future__ import annotations

from pathlib import Path

import aws_cdk as cdk
from aws_cdk.assertions import Match, Template

from agent_infra.config import AgentRuntimeConfig
from agent_infra.stacks.foundation import AgentFoundationStack
from agent_infra.stacks.runtime import AgentRuntimeStack

FAKE_IMAGE_URI = "111111111111.dkr.ecr.us-east-1.amazonaws.com/fake:latest"
ACCOUNT = "111111111111"
REGION = "us-east-1"


def _config(**overrides: object) -> AgentRuntimeConfig:
    defaults = {"bedrock_model_id": "anthropic.claude-opus-5"}
    defaults.update(overrides)
    return AgentRuntimeConfig(**defaults)


def _foundation_template(config: AgentRuntimeConfig) -> Template:
    app = cdk.App()
    env = cdk.Environment(account=ACCOUNT, region=REGION)
    stack = AgentFoundationStack(app, "UnderTest", config=config, env=env)
    return Template.from_stack(stack)


def _runtime_template(config: AgentRuntimeConfig) -> Template:
    app = cdk.App()
    env = cdk.Environment(account=ACCOUNT, region=REGION)
    host = cdk.Stack(app, "Host", env=env)
    import aws_cdk.aws_iam as iam

    role = iam.Role(
        host, "ExecutionRole", assumed_by=iam.ServicePrincipal("bedrock-agentcore.amazonaws.com")
    )

    stack = AgentRuntimeStack(
        app,
        "UnderTest",
        config=config,
        execution_role=role,
        repo_root=Path("/tmp"),
        image_uri_override=FAKE_IMAGE_URI,
        env=env,
    )
    return Template.from_stack(stack)


class TestAgentFoundationStack:
    def test_trust_policy_is_scoped_to_bedrock_agentcore(self) -> None:
        template = _foundation_template(_config())
        template.has_resource_properties(
            "AWS::IAM::Role",
            {
                "AssumeRolePolicyDocument": {
                    "Statement": Match.array_with(
                        [
                            Match.object_like(
                                {
                                    "Principal": {"Service": "bedrock-agentcore.amazonaws.com"},
                                    "Action": "sts:AssumeRole",
                                }
                            )
                        ]
                    )
                }
            },
        )

    def test_grants_scoped_sagemaker_invoke_on_the_tts_endpoint(self) -> None:
        template = _foundation_template(_config(tts_endpoint_name="speech-kokoro-82m"))
        template.has_resource_properties(
            "AWS::IAM::Policy",
            {
                "PolicyDocument": {
                    "Statement": Match.array_with(
                        [
                            Match.object_like(
                                {
                                    "Sid": "InvokeTtsEndpoint",
                                    "Action": Match.array_with(
                                        [
                                            "sagemaker:InvokeEndpoint",
                                            "sagemaker:InvokeEndpointWithBidirectionalStream",
                                        ]
                                    ),
                                    # Rendered as an Fn::Join (region/account/partition
                                    # tokens), not a plain string -- the endpoint name
                                    # is the literal tail of its last part.
                                    "Resource": {
                                        "Fn::Join": [
                                            "",
                                            Match.array_with(
                                                [
                                                    Match.string_like_regexp(
                                                        "endpoint/speech-kokoro-82m$"
                                                    )
                                                ]
                                            ),
                                        ]
                                    },
                                }
                            )
                        ]
                    )
                }
            },
        )

    def test_grants_bedrock_model_invocation(self) -> None:
        template = _foundation_template(_config())
        template.has_resource_properties(
            "AWS::IAM::Policy",
            {
                "PolicyDocument": {
                    "Statement": Match.array_with(
                        [
                            Match.object_like(
                                {
                                    "Sid": "BedrockModelInvocation",
                                    "Action": Match.array_with(["bedrock:InvokeModel"]),
                                }
                            )
                        ]
                    )
                }
            },
        )


class TestAgentRuntimeStack:
    def test_runtime_resource_shape(self) -> None:
        template = _runtime_template(_config(bedrock_model_id="anthropic.claude-opus-5"))
        template.has_resource_properties(
            "AWS::BedrockAgentCore::Runtime",
            {
                "AgentRuntimeArtifact": {
                    "ContainerConfiguration": {"ContainerUri": FAKE_IMAGE_URI}
                },
                "NetworkConfiguration": {"NetworkMode": "PUBLIC"},
                "EnvironmentVariables": Match.object_like(
                    {
                        "AGENT_BEDROCK_MODEL_ID": "anthropic.claude-opus-5",
                        "AGENT_TTS_ENDPOINT": "speech-kokoro-82m",
                    }
                ),
            },
        )

    def test_endpoint_depends_on_runtime(self) -> None:
        template = _runtime_template(_config())
        template.resource_count_is("AWS::BedrockAgentCore::Runtime", 1)
        template.resource_count_is("AWS::BedrockAgentCore::RuntimeEndpoint", 1)
        template.has_resource(
            "AWS::BedrockAgentCore::RuntimeEndpoint",
            {
                "Properties": {"Name": "default"},
                "DependsOn": Match.array_with([Match.string_like_regexp("AgentRuntime")]),
            },
        )
