"""Foundation stack: the IAM execution role AgentCore Runtime assumes.

Policy statements are the AWS-published AgentCore Runtime execution role
(https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/runtime-permissions.html),
not a guess: logs/ECR/X-Ray/CloudWatch/workload-token/model-invocation actions
scoped exactly as that doc specifies. The one addition specific to this
project is the ``sagemaker:InvokeEndpoint*`` statement, scoped to the
already-deployed Kokoro TTS endpoint this agent calls.
"""

from __future__ import annotations

import aws_cdk as cdk
import aws_cdk.aws_iam as iam
from constructs import Construct

from agent_infra.config import AgentRuntimeConfig


class AgentFoundationStack(cdk.Stack):
    """The IAM role AgentCore Runtime assumes to run the demo agent."""

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        config: AgentRuntimeConfig,
        **kwargs: object,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        account = self.account
        region = self.region
        agent_name = config.agent_runtime_name

        self.execution_role = iam.Role(
            self,
            "AgentCoreExecutionRole",
            role_name="agent-infra-agentcore-execution",
            assumed_by=iam.ServicePrincipal(
                "bedrock-agentcore.amazonaws.com",
                conditions={
                    "StringEquals": {"aws:SourceAccount": account},
                    "ArnLike": {"aws:SourceArn": f"arn:aws:bedrock-agentcore:{region}:{account}:*"},
                },
            ),
        )

        runtimes_log_group = (
            f"arn:aws:logs:{region}:{account}:log-group:/aws/bedrock-agentcore/runtimes/*"
        )

        self.execution_role.add_to_policy(
            iam.PolicyStatement(
                actions=["logs:DescribeLogStreams", "logs:CreateLogGroup"],
                resources=[runtimes_log_group],
            )
        )
        self.execution_role.add_to_policy(
            iam.PolicyStatement(
                actions=["logs:PutResourcePolicy"],
                resources=[
                    f"arn:aws:logs:{region}:{account}:log-group:/aws/bedrock-agentcore/runtimes/{agent_name}-*"
                ],
            )
        )
        self.execution_role.add_to_policy(
            iam.PolicyStatement(
                actions=["logs:DescribeLogGroups"],
                resources=[f"arn:aws:logs:{region}:{account}:log-group:*"],
            )
        )
        self.execution_role.add_to_policy(
            iam.PolicyStatement(
                actions=["logs:CreateLogStream", "logs:PutLogEvents"],
                resources=[
                    f"arn:aws:logs:{region}:{account}:log-group:/aws/bedrock-agentcore/runtimes/*:log-stream:*"
                ],
            )
        )

        self.execution_role.add_to_policy(
            iam.PolicyStatement(
                sid="ECRImageAccess",
                actions=["ecr:BatchGetImage", "ecr:GetDownloadUrlForLayer"],
                resources=[f"arn:aws:ecr:{region}:{account}:repository/*"],
            )
        )
        self.execution_role.add_to_policy(
            iam.PolicyStatement(
                sid="ECRTokenAccess",
                actions=["ecr:GetAuthorizationToken"],
                resources=["*"],
            )
        )

        self.execution_role.add_to_policy(
            iam.PolicyStatement(
                actions=[
                    "xray:PutTraceSegments",
                    "xray:PutTelemetryRecords",
                    "xray:GetSamplingRules",
                    "xray:GetSamplingTargets",
                ],
                resources=["*"],
            )
        )
        self.execution_role.add_to_policy(
            iam.PolicyStatement(
                actions=["cloudwatch:PutMetricData"],
                resources=["*"],
                conditions={"StringEquals": {"cloudwatch:namespace": "bedrock-agentcore"}},
            )
        )

        self.execution_role.add_to_policy(
            iam.PolicyStatement(
                sid="GetAgentAccessToken",
                actions=[
                    "bedrock-agentcore:GetWorkloadAccessToken",
                    "bedrock-agentcore:GetWorkloadAccessTokenForJWT",
                    "bedrock-agentcore:GetWorkloadAccessTokenForUserId",
                ],
                resources=[
                    f"arn:aws:bedrock-agentcore:{region}:{account}:workload-identity-directory/default",
                    f"arn:aws:bedrock-agentcore:{region}:{account}:workload-identity-directory/default/workload-identity/{agent_name}-*",
                ],
            )
        )
        self.execution_role.add_to_policy(
            iam.PolicyStatement(
                sid="BedrockModelInvocation",
                actions=["bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream"],
                resources=[
                    "arn:aws:bedrock:*::foundation-model/*",
                    f"arn:aws:bedrock:{region}:{account}:*",
                ],
            )
        )

        # Project-specific: invoke the already-deployed Kokoro TTS endpoint, both
        # transports the agent's bidi/batch mode switch uses. Scoped to the one
        # endpoint, not `resource_name="*"`, mirroring speech_infra's own
        # invocation_role in stacks/foundation.py.
        self.execution_role.add_to_policy(
            iam.PolicyStatement(
                sid="InvokeTtsEndpoint",
                actions=[
                    "sagemaker:InvokeEndpoint",
                    "sagemaker:InvokeEndpointWithBidirectionalStream",
                ],
                resources=[
                    cdk.Arn.format(
                        cdk.ArnComponents(
                            service="sagemaker",
                            resource="endpoint",
                            resource_name=config.tts_endpoint_name,
                        ),
                        self,
                    )
                ],
            )
        )

        cdk.CfnOutput(self, "ExecutionRoleArn", value=self.execution_role.role_arn)
