"""Synth assertions for the endpoint stack's scaling and observability resources.

These tests read the rendered CloudFormation rather than the Python objects, which is
the only way to catch the class of bug this file exists for: CDK silently dropping a
property it does not support in the position given. The previous scaling construct
passed ``period=`` to a target-tracking custom metric for months; it never appeared in
the template, and no unit test on the construct would have noticed.

Passing ``image_uri_override`` keeps ``DockerImageAsset`` out of the synth, so these
run as ordinary unit tests with no Docker daemon and no AWS credentials.
"""

from __future__ import annotations

import aws_cdk as cdk
import aws_cdk.aws_iam as iam
import pytest
from aws_cdk.assertions import Match, Template

from speech_infra.config import TTS_MODEL_CONFIGS, ModelEndpointConfig
from speech_infra.stacks.endpoint import SpeechEndpointStack

FAKE_IMAGE_URI = "111111111111.dkr.ecr.us-east-1.amazonaws.com/fake:latest"
SCALING_POLICY = "AWS::ApplicationAutoScaling::ScalingPolicy"
SCALABLE_TARGET = "AWS::ApplicationAutoScaling::ScalableTarget"


def _template(config: ModelEndpointConfig) -> Template:
    """Synthesize one endpoint stack and return its rendered template."""
    app = cdk.App()
    env = cdk.Environment(account="111111111111", region="us-east-1")
    # The execution role lives in a separate stack in the real app; a host stack
    # keeps it out of the template under assertion.
    host = cdk.Stack(app, "Host", env=env)
    role = iam.Role(
        host, "ExecutionRole", assumed_by=iam.ServicePrincipal("sagemaker.amazonaws.com")
    )

    stack = SpeechEndpointStack(
        app,
        "UnderTest",
        model_config=config,
        execution_role=role,
        container_dir="/tmp",
        image_uri_override=FAKE_IMAGE_URI,
        env=env,
    )
    return Template.from_stack(stack)


@pytest.fixture(scope="module")
def kokoro_template() -> Template:
    """kokoro-82m: the one model configured to scale."""
    return _template(TTS_MODEL_CONFIGS["kokoro-82m"])


@pytest.fixture(scope="module")
def kokoro_config() -> ModelEndpointConfig:
    return TTS_MODEL_CONFIGS["kokoro-82m"]


class TestScalableTarget:
    def test_capacity_matches_config(
        self, kokoro_template: Template, kokoro_config: ModelEndpointConfig
    ) -> None:
        kokoro_template.has_resource_properties(
            SCALABLE_TARGET,
            {
                "MinCapacity": kokoro_config.min_instances,
                "MaxCapacity": kokoro_config.max_instances,
                "ResourceId": f"endpoint/{kokoro_config.endpoint_name}/variant/primary",
                "ScalableDimension": "sagemaker:variant:DesiredInstanceCount",
                "ServiceNamespace": "sagemaker",
            },
        )

    def test_one_target_serves_both_policies(self, kokoro_template: Template) -> None:
        # Both policies must attach to the same target. Two targets on one resource
        # id is rejected by the API at deploy time, which is a slow way to find out.
        kokoro_template.resource_count_is(SCALABLE_TARGET, 1)
        kokoro_template.resource_count_is(SCALING_POLICY, 2)


class TestScaleOutPolicy:
    def test_tracks_the_native_high_resolution_metric(self, kokoro_template: Template) -> None:
        # The bug this replaces: the old policy tracked Speech/vLLM, a namespace
        # nothing published to, so its alarms sat in INSUFFICIENT_DATA forever.
        kokoro_template.has_resource_properties(
            SCALING_POLICY,
            {
                "PolicyType": "TargetTrackingScaling",
                "TargetTrackingScalingPolicyConfiguration": Match.object_like(
                    {
                        "PredefinedMetricSpecification": {
                            "PredefinedMetricType": (
                                "SageMakerVariantConcurrentRequestsPerModelHighResolution"
                            ),
                        },
                    }
                ),
            },
        )

    def test_target_value_is_the_fractional_c_target(
        self, kokoro_template: Template, kokoro_config: ModelEndpointConfig
    ) -> None:
        # The reason scaling_target_value had to become a float. An int field would
        # have rendered 0.713 as 0 and scaled out on every single request.
        assert isinstance(kokoro_config.scaling_target_value, float)
        assert kokoro_config.scaling_target_value < 1.0
        kokoro_template.has_resource_properties(
            SCALING_POLICY,
            {
                "TargetTrackingScalingPolicyConfiguration": Match.object_like(
                    {"TargetValue": kokoro_config.scaling_target_value},
                ),
            },
        )

    def test_scale_in_disabled_so_it_cannot_fight_the_step_policy(
        self, kokoro_template: Template, kokoro_config: ModelEndpointConfig
    ) -> None:
        kokoro_template.has_resource_properties(
            SCALING_POLICY,
            {
                "TargetTrackingScalingPolicyConfiguration": Match.object_like(
                    {
                        "DisableScaleIn": True,
                        "ScaleOutCooldown": kokoro_config.scale_out_cooldown_s,
                    },
                ),
            },
        )

    def test_no_scale_in_cooldown_on_a_scale_out_only_policy(
        self, kokoro_template: Template
    ) -> None:
        # With DisableScaleIn set, a ScaleInCooldown would be inert configuration
        # that reads as if it governs scale-in. It does not.
        policies = kokoro_template.find_resources(
            SCALING_POLICY,
            {"Properties": {"PolicyType": "TargetTrackingScaling"}},
        )
        (policy,) = policies.values()
        config = policy["Properties"]["TargetTrackingScalingPolicyConfiguration"]
        assert "ScaleInCooldown" not in config


class TestScaleInPolicy:
    def test_removes_one_instance_at_a_time(
        self, kokoro_template: Template, kokoro_config: ModelEndpointConfig
    ) -> None:
        # Target tracking cannot express "one at a time"; that asymmetry is the
        # entire reason for a second policy. The no-op interval CDK requires must
        # not render as a second adjustment.
        kokoro_template.has_resource_properties(
            SCALING_POLICY,
            {
                "PolicyType": "StepScaling",
                "StepScalingPolicyConfiguration": Match.object_like(
                    {
                        "AdjustmentType": "ChangeInCapacity",
                        "Cooldown": kokoro_config.scale_in_cooldown_s,
                        "MetricAggregationType": "Average",
                        "StepAdjustments": [
                            {"MetricIntervalUpperBound": 0, "ScalingAdjustment": -1},
                        ],
                    }
                ),
            },
        )

    def test_alarm_watches_the_published_sagemaker_metric(
        self, kokoro_template: Template, kokoro_config: ModelEndpointConfig
    ) -> None:
        kokoro_template.has_resource_properties(
            "AWS::CloudWatch::Alarm",
            {
                "ComparisonOperator": "LessThanOrEqualToThreshold",
                "Namespace": "AWS/SageMaker",
                "MetricName": "ConcurrentRequestsPerModel",
                "Threshold": kokoro_config.scale_in_threshold,
                "EvaluationPeriods": 3,
                "DatapointsToAlarm": 3,
                "Dimensions": [
                    {"Name": "EndpointName", "Value": kokoro_config.endpoint_name},
                    {"Name": "VariantName", "Value": "primary"},
                ],
            },
        )

    def test_scale_in_threshold_is_below_the_target(
        self, kokoro_config: ModelEndpointConfig
    ) -> None:
        # If these crossed, the two policies would oscillate: scale out at the
        # target, scale straight back in above it.
        assert kokoro_config.scale_in_threshold < kokoro_config.scaling_target_value


class TestEndpoint:
    def test_retains_variant_properties_against_the_autoscaling_race(
        self, kokoro_template: Template
    ) -> None:
        # Without this, a deploy re-asserts initial_instance_count and can scale a
        # busy endpoint back down. That race is why autoscaling was disabled in
        # 6997191 rather than fixed.
        kokoro_template.has_resource_properties(
            "AWS::SageMaker::Endpoint",
            {"RetainAllVariantProperties": True},
        )

    def test_startup_timeout_comes_from_config(
        self, kokoro_template: Template, kokoro_config: ModelEndpointConfig
    ) -> None:
        kokoro_template.has_resource_properties(
            "AWS::SageMaker::EndpointConfig",
            {
                "ProductionVariants": [
                    Match.object_like(
                        {
                            "ContainerStartupHealthCheckTimeoutInSeconds": (
                                kokoro_config.container_startup_health_check_timeout_s
                            ),
                            "InitialInstanceCount": kokoro_config.min_instances,
                            "InstanceType": kokoro_config.instance_type,
                        }
                    )
                ]
            },
        )


class TestObservability:
    def test_slo_alarm_uses_the_budget_c_max_was_measured_against(
        self, kokoro_template: Template, kokoro_config: ModelEndpointConfig
    ) -> None:
        # FirstChunkLatency is SageMaker's name for TTFAB, and CloudWatch reports it
        # in microseconds — the budget is stated in ms.
        kokoro_template.has_resource_properties(
            "AWS::CloudWatch::Alarm",
            {
                "MetricName": "FirstChunkLatency",
                "ExtendedStatistic": "p95",
                "Threshold": kokoro_config.ttfab_budget_ms * 1000,
                "ComparisonOperator": "GreaterThanThreshold",
                # An idle endpoint publishes no latency; breaching on that would
                # train operators to ignore the one alarm that tracks the SLO.
                "TreatMissingData": "notBreaching",
            },
        )

    def test_dashboard_is_created(self, kokoro_template: Template) -> None:
        kokoro_template.resource_count_is("AWS::CloudWatch::Dashboard", 1)


class TestNonScalingModels:
    @pytest.mark.parametrize(
        "model_name",
        ["kokoro-82m-cpu", "chatterbox-turbo", "orpheus-3b", "maya-veena"],
    )
    def test_no_scaling_resources_without_a_measured_c_max(self, model_name: str) -> None:
        # The rule this guards: no model gets scaling until its own C_max is
        # measured. maya-veena is here because inheriting the class defaults (1-4)
        # once made scaling_enabled true for a model with no endpoint at all, which
        # is what `tts-bench drift` reported as a missing scalable target.
        config = TTS_MODEL_CONFIGS[model_name]
        assert not config.scaling_enabled

        template = _template(config)
        template.resource_count_is(SCALING_POLICY, 0)
        template.resource_count_is(SCALABLE_TARGET, 0)
        template.resource_count_is("AWS::CloudWatch::Alarm", 0)
        template.resource_count_is("AWS::CloudWatch::Dashboard", 0)


class TestEmergencyStepPolicy:
    def test_off_by_default(self, kokoro_template: Template) -> None:
        # Two policies, not three: the emergency step-out is opt-in.
        kokoro_template.resource_count_is(SCALING_POLICY, 2)

    def test_adds_a_scale_out_only_step_policy_when_enabled(
        self, kokoro_config: ModelEndpointConfig
    ) -> None:
        config = kokoro_config.model_copy(update={"emergency_step_enabled": True})
        template = _template(config)
        template.resource_count_is(SCALING_POLICY, 3)

        # It must not remove instances — the scale-in policy owns that direction,
        # and two policies scaling in would race.
        step_policies = template.find_resources(
            SCALING_POLICY,
            {"Properties": {"PolicyType": "StepScaling"}},
        )
        emergency = [
            p
            for p in step_policies.values()
            if any(
                adj["ScalingAdjustment"] > 0
                for adj in p["Properties"]["StepScalingPolicyConfiguration"]["StepAdjustments"]
            )
        ]
        assert len(emergency) == 1
        adjustments = emergency[0]["Properties"]["StepScalingPolicyConfiguration"][
            "StepAdjustments"
        ]
        assert all(adj["ScalingAdjustment"] > 0 for adj in adjustments)
