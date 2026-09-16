# Copyright (c) 2026 Amazon.com
# This file is licensed under the MIT License.
# See the LICENSE file in the project root for full license information.

"""Foundation stack: IAM roles and shared resources."""

from __future__ import annotations

import aws_cdk as cdk
import aws_cdk.aws_iam as iam
import aws_cdk.aws_s3 as s3
from constructs import Construct

from speech_infra.config import ModelEndpointConfig


class SpeechFoundationStack(cdk.Stack):
    """Shared infrastructure for all speech model endpoints."""

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        model_configs: list[ModelEndpointConfig],
        **kwargs: object,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        self.model_bucket = s3.Bucket(
            self,
            "ModelWeightsBucket",
            bucket_name=f"speech-model-weights-{cdk.Aws.ACCOUNT_ID}-{cdk.Aws.REGION}",
            removal_policy=cdk.RemovalPolicy.RETAIN,
            encryption=s3.BucketEncryption.S3_MANAGED,
            auto_delete_objects=False,
            lifecycle_rules=[
                s3.LifecycleRule(
                    abort_incomplete_multipart_upload_after=cdk.Duration.days(7),
                ),
            ],
        )

        self.execution_role = iam.Role(
            self,
            "SageMakerExecutionRole",
            role_name="speech-sagemaker-execution",
            assumed_by=iam.ServicePrincipal("sagemaker.amazonaws.com"),
        )

        self.execution_role.add_to_policy(
            iam.PolicyStatement(
                actions=[
                    "logs:CreateLogGroup",
                    "logs:CreateLogStream",
                    "logs:PutLogEvents",
                ],
                resources=["*"],
            )
        )

        # Least-privilege finding: AmazonSageMakerFullAccess (removed above) also
        # granted training-job, notebook, feature-store, and model-registry actions
        # this endpoint never uses. SageMaker itself needs exactly these three ECR
        # read actions to pull the container image. Not scoped to the one known CDK
        # bootstrap asset repo -- endpoint.py also supports --image-uri
        # (image_uri_override), which can point at a different repository in this
        # account/region.
        self.execution_role.add_to_policy(
            iam.PolicyStatement(
                actions=[
                    "ecr:BatchCheckLayerAvailability",
                    "ecr:BatchGetImage",
                    "ecr:GetDownloadUrlForLayer",
                ],
                resources=[
                    cdk.Arn.format(
                        cdk.ArnComponents(
                            service="ecr",
                            resource="repository",
                            resource_name="*",
                        ),
                        self,
                    )
                ],
            )
        )

        # ecr:GetAuthorizationToken supports no resource-level permissions at all --
        # every AWS-published ECR pull policy grants it on resources=["*"] for
        # exactly this reason.
        self.execution_role.add_to_policy(
            iam.PolicyStatement(
                actions=["ecr:GetAuthorizationToken"],
                resources=["*"],
            )
        )

        self.model_bucket.grant_read(self.execution_role)

        self.invocation_role = iam.Role(
            self,
            "ClientInvocationRole",
            role_name="speech-client-invocation",
            assumed_by=iam.CompositePrincipal(
                iam.AccountRootPrincipal(),
                iam.ServicePrincipal("sagemaker.amazonaws.com"),
            ),
        )

        self.invocation_role.add_to_policy(
            iam.PolicyStatement(
                actions=[
                    "sagemaker:InvokeEndpoint",
                    "sagemaker:InvokeEndpointWithResponseStream",
                    "sagemaker:InvokeEndpointWithBidirectionalStream",
                ],
                # Scoped to this project's own configured endpoints, not
                # resource_name="*" (every endpoint in the account) -- a
                # threat-modeling finding (overbroad ClientInvocationRole).
                resources=[
                    cdk.Arn.format(
                        cdk.ArnComponents(
                            service="sagemaker",
                            resource="endpoint",
                            resource_name=config.endpoint_name,
                        ),
                        self,
                    )
                    for config in model_configs
                ],
            )
        )

        cdk.CfnOutput(self, "ExecutionRoleArn", value=self.execution_role.role_arn)
        cdk.CfnOutput(self, "InvocationRoleArn", value=self.invocation_role.role_arn)
        cdk.CfnOutput(self, "ModelBucketName", value=self.model_bucket.bucket_name)
