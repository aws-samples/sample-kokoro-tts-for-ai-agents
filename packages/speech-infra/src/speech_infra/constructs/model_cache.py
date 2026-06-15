"""CodeBuild-based model cache that syncs HuggingFace models to S3."""

from __future__ import annotations

import hashlib
import textwrap
from pathlib import Path

import aws_cdk as cdk
import aws_cdk.aws_codebuild as codebuild
import aws_cdk.aws_iam as iam
import aws_cdk.aws_lambda as lambda_
import aws_cdk.aws_logs as logs
import aws_cdk.aws_s3 as s3
from aws_cdk import custom_resources as cr
from aws_cdk.aws_s3_assets import Asset
from constructs import Construct

from speech_infra.config import ModelEndpointConfig

ON_EVENT_CODE = textwrap.dedent("""\
    import boto3
    import os

    codebuild_client = boto3.client("codebuild")
    PROJECT_NAME = os.environ["PROJECT_NAME"]


    def handler(event, context):
        request_type = event["RequestType"]
        if request_type == "Delete":
            return {"PhysicalResourceId": event["PhysicalResourceId"]}

        response = codebuild_client.start_build(projectName=PROJECT_NAME)
        build_id = response["build"]["id"]
        return {
            "PhysicalResourceId": event.get("PhysicalResourceId", build_id),
            "Data": {"BuildId": build_id},
        }
""")

IS_COMPLETE_CODE = textwrap.dedent("""\
    import boto3
    import os

    codebuild_client = boto3.client("codebuild")


    def handler(event, context):
        request_type = event["RequestType"]
        if request_type == "Delete":
            return {"IsComplete": True}

        build_id = event["Data"]["BuildId"]
        response = codebuild_client.batch_get_builds(ids=[build_id])
        build = response["builds"][0]
        status = build["buildStatus"]

        if status == "SUCCEEDED":
            return {"IsComplete": True}
        elif status == "IN_PROGRESS":
            return {"IsComplete": False}
        else:
            raise RuntimeError(
                f"CodeBuild failed with status: {status}. "
                f"Check CloudWatch logs for project: {os.environ['PROJECT_NAME']}"
            )
""")


class ModelCache(Construct):
    """Downloads HuggingFace models to S3 during deployment via CodeBuild."""

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        bucket: s3.IBucket,
        model_configs: list[ModelEndpointConfig],
        hf_token_secret_name: str = "hf-token-dev",
    ) -> None:
        super().__init__(scope, construct_id)

        stack = cdk.Stack.of(self)

        all_model_ids: list[str] = []
        for cfg in model_configs:
            all_model_ids.extend(cfg.all_model_ids)
        unique_model_ids = sorted(set(all_model_ids))

        codebuild_role = iam.Role(
            self,
            "CodeBuildRole",
            assumed_by=iam.ServicePrincipal("codebuild.amazonaws.com"),
        )

        bucket.grant_read_write(codebuild_role)

        codebuild_role.add_to_policy(
            iam.PolicyStatement(
                actions=["secretsmanager:GetSecretValue"],
                resources=[
                    f"arn:aws:secretsmanager:{stack.region}:{stack.account}:secret:{hf_token_secret_name}*"
                ],
            )
        )

        script_path = Path(__file__).parent.parent / "scripts" / "sync_models.py"
        script_asset = Asset(self, "SyncScript", path=str(script_path))
        script_asset.grant_read(codebuild_role)

        build_spec = codebuild.BuildSpec.from_object(
            {
                "version": "0.2",
                "env": {
                    "variables": {
                        "MODEL_IDS": ",".join(unique_model_ids),
                        "S3_BUCKET": bucket.bucket_name,
                        "HF_TOKEN_SECRET": hf_token_secret_name,
                    },
                },
                "phases": {
                    "install": {
                        "runtime-versions": {"python": "3.12"},
                        "commands": [
                            "pip install huggingface_hub hf_transfer --quiet",
                            "export HF_HUB_ENABLE_HF_TRANSFER=1",
                        ],
                    },
                    "build": {
                        "commands": [
                            f"aws s3 cp {script_asset.s3_object_url} sync_models.py",
                            "python sync_models.py",
                        ],
                    },
                },
            }
        )

        self.project = codebuild.Project(
            self,
            "SyncProject",
            project_name=f"speech-model-sync-{stack.stack_name}",
            description="Syncs HuggingFace speech models to S3 for fast cold starts",
            role=codebuild_role,
            build_spec=build_spec,
            environment=codebuild.BuildEnvironment(
                build_image=codebuild.LinuxBuildImage.STANDARD_7_0,
                compute_type=codebuild.ComputeType.LARGE,
                privileged=False,
            ),
            timeout=cdk.Duration.hours(2),
            logging=codebuild.LoggingOptions(
                cloud_watch=codebuild.CloudWatchLoggingOptions(
                    enabled=True,
                    log_group=logs.LogGroup(
                        self,
                        "BuildLogs",
                        retention=logs.RetentionDays.ONE_WEEK,
                    ),
                )
            ),
        )

        on_event_fn = lambda_.Function(
            self,
            "OnEventHandler",
            runtime=lambda_.Runtime.PYTHON_3_12,
            handler="index.handler",
            code=lambda_.Code.from_inline(ON_EVENT_CODE),
            timeout=cdk.Duration.minutes(1),
            environment={"PROJECT_NAME": self.project.project_name},
        )

        is_complete_fn = lambda_.Function(
            self,
            "IsCompleteHandler",
            runtime=lambda_.Runtime.PYTHON_3_12,
            handler="index.handler",
            code=lambda_.Code.from_inline(IS_COMPLETE_CODE),
            timeout=cdk.Duration.minutes(1),
            environment={"PROJECT_NAME": self.project.project_name},
        )

        on_event_fn.add_to_role_policy(
            iam.PolicyStatement(
                actions=["codebuild:StartBuild"],
                resources=[self.project.project_arn],
            )
        )
        is_complete_fn.add_to_role_policy(
            iam.PolicyStatement(
                actions=["codebuild:BatchGetBuilds"],
                resources=[self.project.project_arn],
            )
        )

        provider = cr.Provider(
            self,
            "Provider",
            on_event_handler=on_event_fn,
            is_complete_handler=is_complete_fn,
            query_interval=cdk.Duration.seconds(30),
            total_timeout=cdk.Duration.hours(2),
        )

        model_hash = hashlib.sha256(",".join(unique_model_ids).encode()).hexdigest()[:16]

        self.trigger = cdk.CustomResource(
            self,
            "Trigger",
            service_token=provider.service_token,
            properties={"ModelHash": model_hash},
        )

        cdk.CfnOutput(
            scope,
            "ModelSyncProjectName",
            value=self.project.project_name,
            description="CodeBuild project for manual model sync",
        )
