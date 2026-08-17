# Copyright (c) 2026 Amazon.com
# This file is licensed under the MIT License.
# See the LICENSE file in the project root for full license information.

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

import json
import os
from pathlib import Path

import aws_cdk as cdk
import aws_cdk.aws_iam as iam
import pytest
from aws_cdk.assertions import Match, Template

from speech_infra.config import TTS_MODEL_CONFIGS, ContainerType, ModelEndpointConfig
from speech_infra.stacks.endpoint import SpeechEndpointStack

FAKE_IMAGE_URI = "111111111111.dkr.ecr.us-east-1.amazonaws.com/fake:latest"
SCALING_POLICY = "AWS::ApplicationAutoScaling::ScalingPolicy"
SCALABLE_TARGET = "AWS::ApplicationAutoScaling::ScalableTarget"


def _template(config: ModelEndpointConfig, *, artifact_dir: Path | None = None) -> Template:
    """Synthesize one endpoint stack and return its rendered template.

    ``artifact_dir`` defaults to a path that cannot exist rather than to the real
    ``artifacts/``, so no assertion here depends on whether a benchmark has been run in
    this checkout. Tests that care about the measured latency alarm pass a ``tmp_path``.
    """
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
        artifact_dir=artifact_dir or Path("/nonexistent-artifacts"),
        env=env,
    )
    return Template.from_stack(stack)


def _write_qmax_artifact(
    directory: Path,
    model_name: str,
    *,
    ttfab_p95_at_c1_ms: float,
    slug: str = "g5xlarge-139b9068",
) -> Path:
    """Write the minimum of a ``qmax`` artifact that synth reads.

    Only two fields matter here: ``measurements`` parses the JSON rather than validating
    it against ``QMaxReport`` (``speech-infra`` cannot import ``tts-bench``, the dependency
    runs the other way), so a partial document is a faithful stand-in for what it will
    encounter. ``test_types.py`` owns the round-trip that proves the real report actually
    serializes these keys, and ``test_measurements.py`` the parsing rules.
    """
    path = directory / f"qmax-{model_name}-bidi-{slug}.json"
    path.write_text(
        json.dumps({"model_name": model_name, "ttfab_p95_at_c1_ms": ttfab_p95_at_c1_ms})
    )
    return path


def _write_plan_artifact(
    directory: Path,
    model_name: str,
    *,
    c_scale_max_in_cw_units: float | None = 30.75,
    verdict: str = "ok",
) -> Path:
    """Write the minimum of a ``plan`` artifact ``scaling_thresholds_measured`` reads.

    Named by convention rather than by contract: ``plan --output`` leaves the filename
    free-form, so the gate tells this apart from a ``qmax``/``ttotal`` artifact by the
    nested ``{"plan": {...}}`` shape ``scale_report.plan_to_dict`` writes, not by name.
    Whether ``EndpointAutoscaling``/``EndpointObservability`` get built at all now
    depends on this file existing and being feasible with a CW-unit conversion — see
    ``endpoint.py``'s gate — so any test exercising those constructs' shape needs one of
    these in its ``artifact_dir`` alongside whatever ``qmax`` artifact it cares about.
    """
    path = directory / f"plan-{model_name}.json"
    path.write_text(
        json.dumps(
            {
                "plan": {
                    "model_name": model_name,
                    "c_scale_max_in_cw_units": c_scale_max_in_cw_units,
                },
                "verdict": verdict,
            }
        )
    )
    return path


@pytest.fixture(scope="module")
def kokoro_artifact_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A directory carrying a measured, feasible ``plan`` artifact for kokoro-82m.

    Every test exercising the *shape* of the scaling policies needs this: without a real
    ``plan`` artifact behind it, ``kokoro_template`` would take the same no-resources path
    as an unmeasured model (see ``TestUnmeasuredScalingModel`` below), and there would be
    nothing left to assert on.
    """
    directory = tmp_path_factory.mktemp("kokoro-plan")
    _write_plan_artifact(directory, "kokoro-82m")
    return directory


@pytest.fixture(scope="module")
def kokoro_template(kokoro_artifact_dir: Path) -> Template:
    """kokoro-82m: the one model configured to scale, with a measured plan behind it."""
    return _template(TTS_MODEL_CONFIGS["kokoro-82m"], artifact_dir=kokoro_artifact_dir)


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

    def test_target_value_is_c_scale_max_verbatim(
        self, kokoro_template: Template, kokoro_config: ModelEndpointConfig
    ) -> None:
        # C_scale_max is (1-h) x Q_max, which rarely lands on an integer, so the field
        # is a float and must render as one: an int would truncate and scale out early.
        assert isinstance(kokoro_config.scaling_target_value, float)
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
    """The latency alarm's threshold is measured, not configured.

    It used to be a hand-set ``ttfab_budget_ms`` sitting beside the 3000ms SLO with
    nothing tying either to the other or to a measurement, which is how the config came
    to declare a 20s queue allowance under a 300ms budget. Now it comes from the ``Q_max``
    ladder's ``N=1`` rung — service time on an unqueued instance, which is the quantity
    this metric actually reports — so it re-measures on every rerun.

    These tests synth with an explicit ``artifact_dir`` rather than the real one, so they
    do not depend on whether a benchmark has been run in this checkout.
    """

    def test_no_artifact_means_no_alarm(self, tmp_path: Path) -> None:
        # A fresh clone, CI, or an unmeasured model must still synth. An alarm
        # threshold has to come from somewhere real, so the absence of a measurement
        # omits the alarm rather than inventing a number. No plan artifact either, so
        # this hits the scaling gate before it would even reach the qmax one -- see
        # TestUnmeasuredScalingModel for that path asserted directly.
        template = _template(TTS_MODEL_CONFIGS["kokoro-82m"], artifact_dir=tmp_path)
        latency_alarms = template.find_resources(
            "AWS::CloudWatch::Alarm",
            {"Properties": {"MetricName": "FirstChunkLatency"}},
        )
        assert latency_alarms == {}

    def test_threshold_is_the_ladders_n1_rung(self, tmp_path: Path) -> None:
        # FirstChunkLatency is SageMaker's name for TTFAB, and CloudWatch reports it
        # in microseconds — the artifact states it in ms.
        _write_qmax_artifact(tmp_path, "kokoro-82m", ttfab_p95_at_c1_ms=164.5)
        _write_plan_artifact(tmp_path, "kokoro-82m")
        template = _template(TTS_MODEL_CONFIGS["kokoro-82m"], artifact_dir=tmp_path)

        template.has_resource_properties(
            "AWS::CloudWatch::Alarm",
            {
                "MetricName": "FirstChunkLatency",
                "ExtendedStatistic": "p95",
                "Threshold": 164.5 * 1000,
                "ComparisonOperator": "GreaterThanThreshold",
                # An idle endpoint publishes no latency; breaching on that would
                # train operators to ignore the one alarm that watches service time.
                "TreatMissingData": "notBreaching",
            },
        )

    def test_threshold_is_not_the_slo(self, tmp_path: Path) -> None:
        # The distinction the deleted field existed to express, now derived. At an
        # SLO-sized threshold this alarm fires only once service time alone is ~10x
        # past keeping up, because a request in flight here has spent none of its
        # queue allowance.
        config = TTS_MODEL_CONFIGS["kokoro-82m"]
        _write_qmax_artifact(tmp_path, config.model_name, ttfab_p95_at_c1_ms=164.5)
        _write_plan_artifact(tmp_path, config.model_name)
        template = _template(config, artifact_dir=tmp_path)

        (alarm,) = template.find_resources(
            "AWS::CloudWatch::Alarm",
            {"Properties": {"MetricName": "FirstChunkLatency"}},
        ).values()
        assert alarm["Properties"]["Threshold"] < config.ttfab_slo_ms * 1000

    def test_newest_artifact_wins(self, tmp_path: Path) -> None:
        # A model legitimately has several artifacts — one per configuration measured,
        # which is what stops a rerun overwriting the previous hardware's numbers. The
        # image digest is not knowable at synth time (CDK builds the asset during this
        # very synth), so the tie-break is mtime.
        old = _write_qmax_artifact(
            tmp_path, "kokoro-82m", ttfab_p95_at_c1_ms=500.0, slug="g5xlarge-aaaaaaaa"
        )
        new = _write_qmax_artifact(
            tmp_path, "kokoro-82m", ttfab_p95_at_c1_ms=164.5, slug="g5xlarge-bbbbbbbb"
        )
        os.utime(old, (1_000_000, 1_000_000))
        os.utime(new, (2_000_000, 2_000_000))
        _write_plan_artifact(tmp_path, "kokoro-82m")

        template = _template(TTS_MODEL_CONFIGS["kokoro-82m"], artifact_dir=tmp_path)
        template.has_resource_properties(
            "AWS::CloudWatch::Alarm",
            {"MetricName": "FirstChunkLatency", "Threshold": 164.5 * 1000},
        )

    def test_another_models_artifact_does_not_apply(self, tmp_path: Path) -> None:
        # Reading a different model's service time would threshold kokoro's alarm on
        # hardware and code it never ran on.
        _write_qmax_artifact(tmp_path, "other-model", ttfab_p95_at_c1_ms=164.5)
        _write_plan_artifact(tmp_path, "kokoro-82m")
        template = _template(TTS_MODEL_CONFIGS["kokoro-82m"], artifact_dir=tmp_path)
        assert (
            template.find_resources(
                "AWS::CloudWatch::Alarm",
                {"Properties": {"MetricName": "FirstChunkLatency"}},
            )
            == {}
        )

    def test_an_unreadable_artifact_does_not_break_synth(self, tmp_path: Path) -> None:
        # A killed benchmark run can leave a truncated JSON file. The failure mode is
        # "no alarm", not "no deploy".
        (tmp_path / "qmax-kokoro-82m-bidi-g5xlarge-139b9068.json").write_text("{not json")
        _write_plan_artifact(tmp_path, "kokoro-82m")
        template = _template(TTS_MODEL_CONFIGS["kokoro-82m"], artifact_dir=tmp_path)
        template.resource_count_is("AWS::CloudWatch::Dashboard", 1)

    def test_dashboard_is_created(self, kokoro_template: Template) -> None:
        kokoro_template.resource_count_is("AWS::CloudWatch::Dashboard", 1)


class TestNonScalingModels:
    def test_no_scaling_resources_without_measured_thresholds(self) -> None:
        # The rule this guards: no model gets scaling until its own Q_max and T_total
        # are measured. Constructed directly rather than read from TTS_MODEL_CONFIGS:
        # the point is the invariant on any model with min_instances == max_instances,
        # not a property of which specific models happen to be configured today.
        config = ModelEndpointConfig(
            model_name="non-scaling-model",
            hf_model_id="org/non-scaling-model",
            instance_type="ml.g5.xlarge",
            container_type=ContainerType.PYTORCH_CUSTOM,
            min_instances=1,
            max_instances=1,
        )
        assert not config.scaling_enabled

        template = _template(config)
        template.resource_count_is(SCALING_POLICY, 0)
        template.resource_count_is(SCALABLE_TARGET, 0)
        template.resource_count_is("AWS::CloudWatch::Alarm", 0)
        template.resource_count_is("AWS::CloudWatch::Dashboard", 0)


class TestUnmeasuredScalingModel:
    """A model with ``scaling_enabled=True`` but no ``plan`` artifact takes the same
    no-resources path as a model that is not configured to scale at all.

    This is the gate task #61 exists for: ``scaling_enabled`` is a static property of
    ``min_instances``/``max_instances`` and says nothing about whether the thresholds
    behind it are measured. Before this gate, kokoro-82m synthesized both scaling
    policies off whatever literal sat in ``config.py`` — measured or not, which is
    exactly how ``0.713``, a client occupancy deployed against a server statistic that
    no positive arrival rate satisfies, reached this endpoint. ``TestNonScalingModels``
    above proves the *config* path to no-resources; this proves the *measurement* path
    reaches the same place even when ``scaling_enabled`` is true.
    """

    def test_no_plan_artifact_means_no_scaling_resources(self, tmp_path: Path) -> None:
        config = TTS_MODEL_CONFIGS["kokoro-82m"]
        assert config.scaling_enabled

        template = _template(config, artifact_dir=tmp_path)
        template.resource_count_is(SCALING_POLICY, 0)
        template.resource_count_is(SCALABLE_TARGET, 0)
        template.resource_count_is("AWS::CloudWatch::Alarm", 0)
        template.resource_count_is("AWS::CloudWatch::Dashboard", 0)

    def test_an_infeasible_plan_means_no_scaling_resources(self, tmp_path: Path) -> None:
        # A plan that stopped on its own findings (e.g. the sub-1.0 refusal, task #56)
        # must not be read as "measured, deploy it" just because the file exists.
        config = TTS_MODEL_CONFIGS["kokoro-82m"]
        _write_plan_artifact(tmp_path, config.model_name, verdict="INFEASIBLE")

        template = _template(config, artifact_dir=tmp_path)
        template.resource_count_is(SCALING_POLICY, 0)
        template.resource_count_is(SCALABLE_TARGET, 0)

    def test_a_plan_without_the_cw_conversion_means_no_scaling_resources(
        self, tmp_path: Path
    ) -> None:
        # A plan run without --cloudwatch has the client-measured occupancy but no
        # conversion into the units the alarm reads. Deploying that unconverted number
        # is the exact defect this gate exists to close, so it is refused the same as a
        # missing artifact rather than deployed with a null-shaped threshold.
        config = TTS_MODEL_CONFIGS["kokoro-82m"]
        _write_plan_artifact(tmp_path, config.model_name, c_scale_max_in_cw_units=None)

        template = _template(config, artifact_dir=tmp_path)
        template.resource_count_is(SCALING_POLICY, 0)
        template.resource_count_is(SCALABLE_TARGET, 0)

    def test_a_measured_feasible_plan_enables_scaling(self, tmp_path: Path) -> None:
        # The positive case, spelled out beside the three negatives above: once a real
        # plan exists, is feasible, and carries the CW-unit conversion, the gate opens.
        config = TTS_MODEL_CONFIGS["kokoro-82m"]
        _write_plan_artifact(tmp_path, config.model_name)

        template = _template(config, artifact_dir=tmp_path)
        template.resource_count_is(SCALING_POLICY, 2)
        template.resource_count_is(SCALABLE_TARGET, 1)


class TestEmergencyStepPolicy:
    def test_off_by_default(self, kokoro_template: Template) -> None:
        # Two policies, not three: the emergency step-out is opt-in.
        kokoro_template.resource_count_is(SCALING_POLICY, 2)

    def test_adds_a_scale_out_only_step_policy_when_enabled(
        self, kokoro_config: ModelEndpointConfig, kokoro_artifact_dir: Path
    ) -> None:
        config = kokoro_config.model_copy(update={"emergency_step_enabled": True})
        template = _template(config, artifact_dir=kokoro_artifact_dir)
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
