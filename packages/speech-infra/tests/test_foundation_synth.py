# Copyright (c) 2026 Amazon.com
# This file is licensed under the MIT License.
# See the LICENSE file in the project root for full license information.

"""Synth assertions for SpeechFoundationStack's ClientInvocationRole.

Threat-modeling finding (High severity): the role's ``sagemaker:InvokeEndpoint*``
grant used ``resource_name="*"``, letting anything that can assume it invoke every
SageMaker endpoint in the account -- not just the Kokoro endpoint it exists to
serve. These tests pin the fix (scoped to this project's own configured endpoints)
against the rendered CloudFormation, the same way test_scaling_synth.py pins the
scaling policies -- reading the Python construct alone would not catch a property
CDK silently dropped or rendered wider than intended.
"""

from __future__ import annotations

import json

import aws_cdk as cdk
from aws_cdk.assertions import Template

from speech_infra.config import ContainerType, ModelEndpointConfig
from speech_infra.stacks.foundation import SpeechFoundationStack

INVOKE_ACTIONS = {
    "sagemaker:InvokeEndpoint",
    "sagemaker:InvokeEndpointWithResponseStream",
    "sagemaker:InvokeEndpointWithBidirectionalStream",
}


def _config(model_name: str) -> ModelEndpointConfig:
    return ModelEndpointConfig(
        model_name=model_name,
        hf_model_id=f"org/{model_name}",
        instance_type="ml.g5.xlarge",
        container_type=ContainerType.PYTORCH_CUSTOM,
    )


def _template(model_configs: list[ModelEndpointConfig]) -> Template:
    app = cdk.App()
    env = cdk.Environment(account="111111111111", region="us-east-1")
    stack = SpeechFoundationStack(app, "UnderTest", model_configs=model_configs, env=env)
    return Template.from_stack(stack)


def _invoke_statements(template: Template) -> list[dict]:
    """All IAM policy statements granting an InvokeEndpoint* action, across any role."""
    policies = template.find_resources("AWS::IAM::Policy")
    statements = []
    for resource in policies.values():
        for statement in resource["Properties"]["PolicyDocument"]["Statement"]:
            if set(statement.get("Action", [])) & INVOKE_ACTIONS:
                statements.append(statement)
    return statements


class TestClientInvocationRoleScoping:
    def test_no_wildcard_resource(self) -> None:
        # The regression this file exists to prevent: resource_name="*" grants
        # every SageMaker endpoint in the account, not just this project's.
        (statement,) = _invoke_statements(_template([_config("kokoro-82m")]))
        assert "endpoint/*" not in json.dumps(statement["Resource"])

    def test_scoped_to_the_one_configured_endpoint(self) -> None:
        config = _config("kokoro-82m")
        (statement,) = _invoke_statements(_template([config]))
        assert f"endpoint/{config.endpoint_name}" in json.dumps(statement["Resource"])

    def test_multiple_models_each_get_their_own_arn(self) -> None:
        # A future second model must not fall back to a shared wildcard just
        # because the role now serves more than one endpoint.
        configs = [_config("kokoro-82m"), _config("other-model")]
        (statement,) = _invoke_statements(_template(configs))
        resource_str = json.dumps(statement["Resource"])
        for config in configs:
            assert f"endpoint/{config.endpoint_name}" in resource_str
        assert "endpoint/*" not in resource_str

    def test_all_three_invoke_actions_still_granted(self) -> None:
        # Scoping the resource must not have narrowed the action set -- bidi
        # streaming needs InvokeEndpointWithBidirectionalStream specifically.
        (statement,) = _invoke_statements(_template([_config("kokoro-82m")]))
        assert set(statement["Action"]) == INVOKE_ACTIONS

    def test_sagemaker_execution_role_is_unaffected(self) -> None:
        # The fix touches the client-facing role's resource grant only; the
        # SageMaker service's own execution role must keep its single-principal
        # trust policy (Service-only, no AccountRootPrincipal composite).
        template = _template([_config("kokoro-82m")])
        roles = template.find_resources("AWS::IAM::Role")
        trust_statements = [
            role["Properties"]["AssumeRolePolicyDocument"]["Statement"] for role in roles.values()
        ]
        service_only = [
            statements
            for statements in trust_statements
            if len(statements) == 1
            and statements[0]["Principal"] == {"Service": "sagemaker.amazonaws.com"}
        ]
        assert len(service_only) == 1


EXECUTION_ROLE_NAME = "speech-sagemaker-execution"

ECR_PULL_ACTIONS = {
    "ecr:BatchCheckLayerAvailability",
    "ecr:BatchGetImage",
    "ecr:GetDownloadUrlForLayer",
}

LOGS_ACTIONS = {"logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"}


def _all_statements(template: Template) -> list[dict]:
    """Every IAM policy statement in the stack, across all roles."""
    policies = template.find_resources("AWS::IAM::Policy")
    statements = []
    for resource in policies.values():
        statements.extend(resource["Properties"]["PolicyDocument"]["Statement"])
    return statements


def _action_set(statement: dict) -> set[str]:
    """Normalize Action to a set -- CDK renders a single action as a bare string,
    not a one-element list, so plain set() on it would iterate characters instead
    of treating it as one action name.
    """
    action = statement.get("Action", [])
    return {action} if isinstance(action, str) else set(action)


class TestExecutionRoleLeastPrivilege:
    """Least-privilege finding: the execution role carried AmazonSageMakerFullAccess
    (training jobs, notebooks, feature store, model registry, ...) though the
    endpoint only ever needs to pull its container image, write CloudWatch logs, and
    read model weights from S3. These tests pin the narrowed policy against the
    rendered CloudFormation, the same way TestClientInvocationRoleScoping pins the
    sibling role above -- reading the Python construct alone would not catch CDK
    silently reattaching a managed policy or rendering a wildcard resource.
    """

    def test_no_managed_policy_attached(self) -> None:
        template = _template([_config("kokoro-82m")])
        roles = template.find_resources("AWS::IAM::Role")
        (execution_role,) = (
            r for r in roles.values() if r["Properties"].get("RoleName") == EXECUTION_ROLE_NAME
        )
        assert not execution_role["Properties"].get("ManagedPolicyArns")

    def test_ecr_pull_actions_granted(self) -> None:
        statements = _all_statements(_template([_config("kokoro-82m")]))
        matches = [s for s in statements if ECR_PULL_ACTIONS <= _action_set(s)]
        assert len(matches) == 1

    def test_ecr_pull_actions_not_scoped_to_wildcard_resource(self) -> None:
        # The three pull actions support resource-level scoping -- unlike
        # GetAuthorizationToken below, they must not fall back to resources=["*"].
        statements = _all_statements(_template([_config("kokoro-82m")]))
        (statement,) = (s for s in statements if ECR_PULL_ACTIONS <= _action_set(s))
        assert statement["Resource"] != "*"
        assert "repository/*" in json.dumps(statement["Resource"])

    def test_ecr_get_authorization_token_scoped_to_wildcard_resource(self) -> None:
        # ecr:GetAuthorizationToken supports no resource-level permissions at all;
        # every AWS-published ECR pull policy grants it on resources=["*"].
        statements = _all_statements(_template([_config("kokoro-82m")]))
        (statement,) = (s for s in statements if "ecr:GetAuthorizationToken" in _action_set(s))
        assert statement["Resource"] == "*"

    def test_logs_statement_unaffected(self) -> None:
        # Regression guard: narrowing the managed policy away must not have
        # touched the pre-existing CloudWatch Logs grant.
        statements = _all_statements(_template([_config("kokoro-82m")]))
        matches = [s for s in statements if LOGS_ACTIONS <= _action_set(s)]
        assert len(matches) == 1

    def test_s3_model_bucket_read_unaffected(self) -> None:
        # Regression guard: narrowing the managed policy away must not have
        # touched the pre-existing model_bucket.grant_read() statement.
        statements = _all_statements(_template([_config("kokoro-82m")]))
        matches = [s for s in statements if "s3:GetObject*" in _action_set(s)]
        assert len(matches) == 1
