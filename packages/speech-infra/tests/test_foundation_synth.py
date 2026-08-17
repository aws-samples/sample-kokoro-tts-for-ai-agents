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
            if len(statements) == 1 and statements[0]["Principal"] == {"Service": "sagemaker.amazonaws.com"}
        ]
        assert len(service_only) == 1
