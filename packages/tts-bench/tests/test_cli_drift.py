"""Tests for the `drift` and `thaw` CLI commands.

These are the operator-facing edge of Phase 1, so what matters is the parts an
operator relies on and cannot verify by reading the output: the exit code
(`drift` is meant for CI), the remediation text (it gets pasted into a shell),
and the fact that neither command mutates anything it was not asked to.

The AWS clients are patched at the `boto3.client` seam because both commands
construct their own — that is the correct shape for a CLI, and it means these
tests exercise the real wiring from `_expected_scaling()` through
`observe.check_drift` to the rendered output, rather than a mock of the thing
under test.

Baselines are *derived* from `speech_infra.config` rather than hardcoded. Phase 5
raises `max_instances` on every model, which flips `scaling_enabled` and so
changes which endpoints a clean account must have targets for; a literal list
here would start failing for the right reason at the wrong place.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from tts_bench.cli import _expected_scaling, main
from tts_bench.observe import SCALABLE_DIMENSION, resource_id

T0 = datetime(2026, 7, 29, 12, 0, 0, tzinfo=UTC)
ROLE_ARN = "arn:aws:iam::1234:role/aws-service-role/sagemaker.application-autoscaling"

#: An endpoint no config describes — the orphan case, and the live state of
#: `speech-orpheus-3b` today (max_capacity=4 with `max_instances=1` in config).
ORPHAN = "speech-orpheus-3b"


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def _paginator(pages: list[dict]) -> MagicMock:
    paginator = MagicMock()
    paginator.paginate.return_value = pages
    return paginator


def _fake_appscaling(*, targets: list[dict], policies: list[dict]) -> MagicMock:
    client = MagicMock()

    def _get_paginator(name: str) -> MagicMock:
        if name == "describe_scalable_targets":
            return _paginator([{"ScalableTargets": targets}])
        if name == "describe_scaling_policies":
            return _paginator([{"ScalingPolicies": policies}])
        raise AssertionError(f"unexpected paginator {name}")

    client.get_paginator.side_effect = _get_paginator
    return client


def _fake_cloudwatch(*, metrics: list[dict], alarms: list[dict]) -> MagicMock:
    client = MagicMock()
    client.list_metrics.return_value = {"Metrics": metrics}
    client.get_paginator.return_value = _paginator([{"MetricAlarms": alarms}])
    return client


def _raw_target(endpoint: str, *, min_capacity: int = 1, max_capacity: int = 4) -> dict:
    return {
        "ServiceNamespace": "sagemaker",
        "ResourceId": resource_id(endpoint),
        "ScalableDimension": SCALABLE_DIMENSION,
        "MinCapacity": min_capacity,
        "MaxCapacity": max_capacity,
        "RoleARN": ROLE_ARN,
        "CreationTime": T0,
    }


def _vllm_policy(endpoint: str, *, name: str = "TrackRunningRequests") -> dict:
    """A policy shaped like the two orphans live in the account right now."""
    return {
        "PolicyARN": "arn:aws:autoscaling:us-east-1:1234:scalingPolicy:a:resource/b:policyName/c",
        "PolicyName": name,
        "ServiceNamespace": "sagemaker",
        "ResourceId": resource_id(endpoint),
        "ScalableDimension": SCALABLE_DIMENSION,
        "PolicyType": "TargetTrackingScaling",
        "TargetTrackingScalingPolicyConfiguration": {
            "TargetValue": 8.0,
            "CustomizedMetricSpecification": {
                "MetricName": "vllm:num_requests_running",
                "Namespace": "Speech/vLLM",
                "Statistic": "Average",
            },
        },
        "Alarms": [{"AlarmName": "AlarmHigh-1", "AlarmARN": "arn:aws:cloudwatch:::alarm:h"}],
        "CreationTime": T0,
    }


def _baseline_targets() -> list[dict]:
    """One matching target per endpoint whose config enables scaling.

    This is what "no drift" looks like: every scalable target the config asks
    for, at exactly the capacity it asks for, and nothing else.
    """
    return [
        _raw_target(
            expected.endpoint,
            min_capacity=expected.effective_min,
            max_capacity=expected.max_instances,
        )
        for expected in _expected_scaling().values()
        if expected.scaling_enabled
    ]


def _run_drift(runner: CliRunner, appscaling: MagicMock, cloudwatch: MagicMock, *args: str):
    clients = {"application-autoscaling": appscaling, "cloudwatch": cloudwatch}
    with patch("boto3.client", side_effect=lambda name, **_: clients[name]):
        return runner.invoke(main, ["drift", *args])


class TestExpectedScaling:
    def test_keys_are_endpoint_names_not_model_names(self) -> None:
        # The audit matches on resource ids, which carry `speech-`-prefixed
        # endpoint names. Keying by model name would report every endpoint as an
        # orphan and every config as missing a target.
        assert all(name.startswith("speech-") for name in _expected_scaling())

    def test_covers_every_configured_tts_model(self) -> None:
        from speech_infra.config import TTS_MODEL_CONFIGS

        assert len(_expected_scaling()) == len(TTS_MODEL_CONFIGS)

    def test_carries_the_scaling_enabled_gate(self) -> None:
        # A model pinned to one instance gets no synthesized policy, which is
        # exactly what makes a live policy on it an orphan. Asserted against a
        # pinned model rather than a named one so it keeps testing the gate as
        # models gain measured C_max values and start scaling.
        expected = _expected_scaling()
        pinned = {name for name, e in expected.items() if e.max_instances <= e.effective_min}
        assert pinned, "no pinned model left to check the gate against"
        assert all(not expected[name].scaling_enabled for name in pinned)
        assert expected["speech-kokoro-82m"].scaling_enabled

    def test_coerces_min_zero_the_way_cdk_does(self) -> None:
        # Several configs say min_instances=0; both CDK constructs wrap it in
        # max(..., 1). Comparing against the raw 0 would report a capacity
        # mismatch on every one of those endpoints.
        from speech_infra.config import TTS_MODEL_CONFIGS

        zeroed = [c for c in TTS_MODEL_CONFIGS.values() if c.min_instances == 0]
        assert zeroed, "no min_instances=0 config left to check coercion against"
        expected = _expected_scaling()
        assert all(expected[c.endpoint_name].effective_min == 1 for c in zeroed)


class TestDriftCommand:
    def test_matching_account_exits_zero(self, runner: CliRunner) -> None:
        result = _run_drift(
            runner,
            _fake_appscaling(targets=_baseline_targets(), policies=[]),
            _fake_cloudwatch(metrics=[], alarms=[]),
        )
        assert result.exit_code == 0
        assert "No drift" in result.output

    def test_a_clean_account_points_at_the_measurement(self, runner: CliRunner) -> None:
        # drift is the preflight for qmax, and saying so is what makes the sequence
        # discoverable. Only on a clean account: with findings on screen, the next step
        # is to fix them, not to start a 45-minute measurement against them.
        result = _run_drift(
            runner,
            _fake_appscaling(targets=_baseline_targets(), policies=[]),
            _fake_cloudwatch(metrics=[], alarms=[]),
        )
        assert "Next: tts-bench qmax" in result.output
        assert "--require-frozen" in result.output

    def test_findings_do_not_point_at_the_measurement(self, runner: CliRunner) -> None:
        result = _run_drift(
            runner,
            _fake_appscaling(targets=[], policies=[]),
            _fake_cloudwatch(metrics=[], alarms=[]),
        )
        assert "Next: tts-bench cmax" not in result.output

    def test_missing_target_is_reported_but_does_not_fail(self, runner: CliRunner) -> None:
        # An endpoint whose config enables scaling but has no live target cannot
        # scale at all. That is worth saying, but it is a deploy gap rather than
        # capacity moving unsupervised, so it must not turn CI red.
        result = _run_drift(
            runner,
            _fake_appscaling(targets=[], policies=[]),
            _fake_cloudwatch(metrics=[], alarms=[]),
        )
        assert result.exit_code == 0
        assert "missing_target" in result.output

    def test_orphan_exits_non_zero_for_ci(self, runner: CliRunner) -> None:
        # The whole point of --fail-on-error: a CI job has to go red on an
        # orphan, because nobody reads output that passes.
        result = _run_drift(
            runner,
            _fake_appscaling(
                targets=[*_baseline_targets(), _raw_target(ORPHAN, max_capacity=4)],
                policies=[_vllm_policy(ORPHAN)],
            ),
            _fake_cloudwatch(
                metrics=[],
                alarms=[{"AlarmName": "AlarmHigh-1", "StateValue": "INSUFFICIENT_DATA"}],
            ),
        )
        assert result.exit_code == 1
        assert "orphaned_target" in result.output
        assert "orphaned_policy" in result.output

    def test_no_fail_on_error_still_reports(self, runner: CliRunner) -> None:
        result = _run_drift(
            runner,
            _fake_appscaling(
                targets=[*_baseline_targets(), _raw_target(ORPHAN, max_capacity=4)], policies=[]
            ),
            _fake_cloudwatch(metrics=[], alarms=[]),
            "--no-fail-on-error",
        )
        assert result.exit_code == 0
        assert "orphaned_target" in result.output

    def test_errors_are_printed_before_warnings(self, runner: CliRunner) -> None:
        # A finding list gets read top-down. Burying the orphan that can move
        # capacity under an inert-policy warning defeats the report.
        result = _run_drift(
            runner,
            _fake_appscaling(
                targets=[*_baseline_targets(), _raw_target(ORPHAN, max_capacity=4)],
                policies=[_vllm_policy(ORPHAN)],
            ),
            _fake_cloudwatch(metrics=[], alarms=[]),
            "--no-fail-on-error",
        )
        assert result.output.index("[ERROR]") < result.output.index("[WARN]")
        assert "inert_policy" in result.output

    def test_counts_errors_and_warnings_separately(self, runner: CliRunner) -> None:
        result = _run_drift(
            runner,
            _fake_appscaling(
                targets=[*_baseline_targets(), _raw_target(ORPHAN, max_capacity=4)],
                policies=[_vllm_policy(ORPHAN)],
            ),
            _fake_cloudwatch(metrics=[], alarms=[]),
            "--no-fail-on-error",
        )
        # Orphaned target + orphaned policy = 2 errors; inert policy = 1 warning.
        assert "2 error, 1 warning" in result.output

    def test_remediation_is_a_runnable_command(self, runner: CliRunner) -> None:
        # This text gets pasted into a shell at Phase 5 step 0, so the resource
        # id and the scalable dimension both have to be present and correct.
        result = _run_drift(
            runner,
            _fake_appscaling(
                targets=[*_baseline_targets(), _raw_target(ORPHAN, max_capacity=4)],
                policies=[_vllm_policy(ORPHAN)],
            ),
            _fake_cloudwatch(metrics=[], alarms=[]),
            "--no-fail-on-error",
        )
        assert "aws application-autoscaling delete-scaling-policy" in result.output
        assert "aws application-autoscaling deregister-scalable-target" in result.output
        assert f"--resource-id {resource_id(ORPHAN)}" in result.output
        assert f"--scalable-dimension {SCALABLE_DIMENSION}" in result.output

    def test_json_output_is_machine_readable(self, runner: CliRunner, tmp_path) -> None:
        out = tmp_path / "drift.json"
        result = _run_drift(
            runner,
            _fake_appscaling(
                targets=[*_baseline_targets(), _raw_target(ORPHAN, max_capacity=4)], policies=[]
            ),
            _fake_cloudwatch(metrics=[], alarms=[]),
            "--output",
            str(out),
            "--no-fail-on-error",
        )
        assert result.exit_code == 0
        findings = json.loads(out.read_text())
        assert [f for f in findings if f["kind"] == "orphaned_target"] == [
            {
                "kind": "orphaned_target",
                "severity": "error",
                "resource_id": resource_id(ORPHAN),
                "endpoint": ORPHAN,
                "detail": findings[0]["detail"],
                "remediation": findings[0]["remediation"],
            }
        ]

    def test_json_output_is_written_even_when_clean(self, runner: CliRunner, tmp_path) -> None:
        # An empty array is the verification artifact Phase 5 signs off against.
        # A missing file is indistinguishable from a command that never ran.
        out = tmp_path / "drift.json"
        _run_drift(
            runner,
            _fake_appscaling(targets=_baseline_targets(), policies=[]),
            _fake_cloudwatch(metrics=[], alarms=[]),
            "--output",
            str(out),
        )
        assert json.loads(out.read_text()) == []

    def test_makes_no_mutating_calls(self, runner: CliRunner) -> None:
        # drift is documented read-only. This is the assertion that keeps it so.
        appscaling = _fake_appscaling(
            targets=[*_baseline_targets(), _raw_target(ORPHAN, max_capacity=4)],
            policies=[_vllm_policy(ORPHAN)],
        )
        cloudwatch = _fake_cloudwatch(metrics=[], alarms=[])
        _run_drift(runner, appscaling, cloudwatch, "--no-fail-on-error")

        for client in (appscaling, cloudwatch):
            forbidden = {
                name
                for name, *_ in client.method_calls
                if name.startswith(("register_", "put_", "delete_", "deregister_", "update_"))
            }
            assert forbidden == set(), f"drift called mutating APIs: {forbidden}"

    def test_inference_component_targets_are_ignored(self, runner: CliRunner) -> None:
        # The `sagemaker` namespace also covers inference components. Reading one
        # as an endpoint variant would report a phantom orphan.
        component = _raw_target(ORPHAN)
        component["ResourceId"] = "inference-component/my-component"
        result = _run_drift(
            runner,
            _fake_appscaling(targets=[*_baseline_targets(), component], policies=[]),
            _fake_cloudwatch(metrics=[], alarms=[]),
        )
        assert result.exit_code == 0
        assert "No drift" in result.output

    def test_leftover_freeze_is_an_error_pointing_at_thaw(self, runner: CliRunner) -> None:
        # An aborted cmax run leaves scale-out suspended, which looks healthy in
        # the console and cannot scale under load. The fix is one command.
        frozen = _raw_target(ORPHAN, max_capacity=4)
        frozen["SuspendedState"] = {
            "DynamicScalingInSuspended": True,
            "DynamicScalingOutSuspended": True,
            "ScheduledScalingSuspended": True,
        }
        result = _run_drift(
            runner,
            _fake_appscaling(targets=[*_baseline_targets(), frozen], policies=[]),
            _fake_cloudwatch(metrics=[], alarms=[]),
            "--no-fail-on-error",
        )
        assert "suspended" in result.output
        assert f"tts-bench thaw --endpoint {ORPHAN}" in result.output


class TestThawCommand:
    ENDPOINT = "speech-kokoro-82m"

    def _target(self, suspended: dict) -> dict:
        return {
            "ServiceNamespace": "sagemaker",
            "ResourceId": resource_id(self.ENDPOINT),
            "ScalableDimension": SCALABLE_DIMENSION,
            "MinCapacity": 1,
            "MaxCapacity": 4,
            "RoleARN": ROLE_ARN,
            "CreationTime": T0,
            "SuspendedState": suspended,
        }

    def _fake_clients(
        self,
        *,
        before: dict | None,
        after: dict | None = ...,  # type: ignore[assignment]
        desired: int = 1,
        current: int = 1,
    ) -> tuple[MagicMock, MagicMock]:
        """Clients for a thaw run. ``None`` means no scalable target registered.

        ``thaw`` describes twice — once to report the state it found, once to
        verify the resume took — so both responses are scripted. ``after``
        defaults to ``before``, which is the "nothing changed" case.
        """
        if after is ...:
            after = before

        appscaling = MagicMock()
        appscaling.describe_scalable_targets.side_effect = [
            {"ScalableTargets": [] if before is None else [self._target(before)]},
            {"ScalableTargets": [] if after is None else [self._target(after)]},
        ]
        appscaling.describe_scaling_policies.return_value = {"ScalingPolicies": []}

        sagemaker = MagicMock()
        sagemaker.describe_endpoint.return_value = {
            "ProductionVariants": [
                {
                    "VariantName": "primary",
                    "DesiredInstanceCount": desired,
                    "CurrentInstanceCount": current,
                }
            ]
        }
        return appscaling, sagemaker

    def _run(self, runner: CliRunner, appscaling, sagemaker, *args: str):
        clients = {"application-autoscaling": appscaling, "sagemaker": sagemaker}
        with patch("boto3.client", side_effect=lambda name, **_: clients[name]):
            return runner.invoke(main, ["thaw", "--endpoint", self.ENDPOINT, *args])

    def test_resumes_all_three_suspension_flags(self, runner: CliRunner) -> None:
        # Scheduled scaling can move capacity too, so a partial resume leaves a
        # benchmark's freeze half in place.
        appscaling, sagemaker = self._fake_clients(
            before={
                "DynamicScalingInSuspended": True,
                "DynamicScalingOutSuspended": True,
                "ScheduledScalingSuspended": True,
            },
            after={
                "DynamicScalingInSuspended": False,
                "DynamicScalingOutSuspended": False,
                "ScheduledScalingSuspended": False,
            },
        )
        result = self._run(runner, appscaling, sagemaker)
        assert result.exit_code == 0
        assert appscaling.register_scalable_target.call_args.kwargs["SuspendedState"] == {
            "DynamicScalingInSuspended": False,
            "DynamicScalingOutSuspended": False,
            "ScheduledScalingSuspended": False,
        }

    def test_never_rewrites_capacity_limits(self, runner: CliRunner) -> None:
        # Same load-bearing property as freeze: min/max are optional on
        # RegisterScalableTarget, and sending them would let a recovery command
        # silently undo a deliberate capacity change.
        appscaling, sagemaker = self._fake_clients(
            before={"DynamicScalingOutSuspended": True},
            after={"DynamicScalingOutSuspended": False},
        )
        self._run(runner, appscaling, sagemaker)
        sent = appscaling.register_scalable_target.call_args.kwargs
        assert "MinCapacity" not in sent
        assert "MaxCapacity" not in sent

    def test_does_not_touch_capacity_by_default(self, runner: CliRunner) -> None:
        # After a hard kill the pre-freeze desired count is gone. Restoring a
        # guessed 1 would shrink a fleet that was legitimately larger.
        appscaling, sagemaker = self._fake_clients(
            before={"DynamicScalingOutSuspended": True},
            after={"DynamicScalingOutSuspended": False},
            desired=3,
            current=3,
        )
        self._run(runner, appscaling, sagemaker)
        sagemaker.update_endpoint_weights_and_capacities.assert_not_called()

    def test_restore_capacity_requires_an_explicit_desired(self, runner: CliRunner) -> None:
        appscaling, sagemaker = self._fake_clients(before={})
        result = self._run(runner, appscaling, sagemaker, "--restore-capacity")
        assert result.exit_code != 0
        assert "--restore-capacity requires --desired" in result.output
        sagemaker.update_endpoint_weights_and_capacities.assert_not_called()

    def test_restores_the_requested_desired_count(self, runner: CliRunner) -> None:
        appscaling, sagemaker = self._fake_clients(before={}, desired=1, current=1)
        self._run(runner, appscaling, sagemaker, "--restore-capacity", "--desired", "2")
        sent = sagemaker.update_endpoint_weights_and_capacities.call_args.kwargs
        assert sent["EndpointName"] == self.ENDPOINT
        assert sent["DesiredWeightsAndCapacities"] == [
            {"VariantName": "primary", "DesiredInstanceCount": 2}
        ]

    def test_no_scalable_target_is_a_clean_no_op(self, runner: CliRunner) -> None:
        # True for speech-kokoro-82m and speech-chatterbox-turbo today. Nothing
        # to resume is a success: the endpoint already cannot scale, and
        # creating a target here would leave config CDK does not describe.
        appscaling, sagemaker = self._fake_clients(before=None)
        result = self._run(runner, appscaling, sagemaker)
        assert result.exit_code == 0
        assert "nothing to resume" in result.output.lower()
        appscaling.register_scalable_target.assert_not_called()

    def test_is_idempotent_on_an_already_thawed_endpoint(self, runner: CliRunner) -> None:
        # Recovery from a hard kill means running this without knowing whether
        # the freeze ever landed, so a healthy endpoint must be a no-op.
        appscaling, sagemaker = self._fake_clients(
            before={
                "DynamicScalingInSuspended": False,
                "DynamicScalingOutSuspended": False,
                "ScheduledScalingSuspended": False,
            }
        )
        result = self._run(runner, appscaling, sagemaker)
        assert result.exit_code == 0

    def test_reports_failure_when_suspension_survives(self, runner: CliRunner) -> None:
        # Usually a missing IAM permission. Exiting 0 here would tell an
        # operator production is healthy while scale-out is still dead.
        appscaling, sagemaker = self._fake_clients(before={"DynamicScalingOutSuspended": True})
        result = self._run(runner, appscaling, sagemaker)
        assert result.exit_code == 1
        assert "still suspended" in result.output
