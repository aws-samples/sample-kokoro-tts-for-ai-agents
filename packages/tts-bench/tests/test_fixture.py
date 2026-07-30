"""Tests for the autoscaling freeze/thaw guard.

Uses botocore's ``Stubber`` (first use in this repo) rather than moto: these
tests assert on the *exact request parameters* sent, which is the property that
matters. The load-bearing assertion is that a freeze never sends
``MinCapacity``/``MaxCapacity`` — a benchmark must not be able to rewrite an
endpoint's scaling limits as a side effect.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import boto3
import pytest
from botocore.stub import Stubber

from tts_bench.fixture import (
    SCALABLE_DIMENSION,
    SERVICE_NAMESPACE,
    SUSPEND_ALL,
    EndpointFixture,
    FixtureError,
    capture,
    freeze,
    frozen,
    require_frozen,
    require_scalable,
    resource_id,
    thaw,
)

ENDPOINT = "speech-kokoro-82m"
RID = f"endpoint/{ENDPOINT}/variant/primary"


@pytest.fixture
def appscaling() -> Any:
    client = boto3.client("application-autoscaling", region_name="us-east-1")
    stub = Stubber(client)
    stub.activate()
    yield client, stub
    stub.deactivate()


@pytest.fixture
def sagemaker() -> Any:
    client = boto3.client("sagemaker", region_name="us-east-1")
    stub = Stubber(client)
    stub.activate()
    yield client, stub
    stub.deactivate()


def _target(
    *,
    min_capacity: int = 1,
    max_capacity: int = 4,
    scale_out_suspended: bool = False,
    scale_in_suspended: bool = False,
    scheduled_suspended: bool = False,
) -> dict:
    return {
        "ServiceNamespace": SERVICE_NAMESPACE,
        "ResourceId": RID,
        "ScalableDimension": SCALABLE_DIMENSION,
        "MinCapacity": min_capacity,
        "MaxCapacity": max_capacity,
        # Required by the ScalableTarget shape; Stubber validates responses.
        "RoleARN": "arn:aws:iam::1234:role/aws-service-role/sagemaker.application-autoscaling",
        "SuspendedState": {
            "DynamicScalingInSuspended": scale_in_suspended,
            "DynamicScalingOutSuspended": scale_out_suspended,
            "ScheduledScalingSuspended": scheduled_suspended,
        },
        "CreationTime": "2026-06-18T17:19:51Z",
    }


def _endpoint(*, desired: int = 1, current: int = 1, variant: str = "primary") -> dict:
    return {
        "EndpointName": ENDPOINT,
        "EndpointArn": f"arn:aws:sagemaker:us-east-1:1234:endpoint/{ENDPOINT}",
        "EndpointConfigName": f"{ENDPOINT}-config",
        "EndpointStatus": "InService",
        "CreationTime": "2026-06-18T17:19:51Z",
        "LastModifiedTime": "2026-06-18T17:19:51Z",
        "ProductionVariants": [
            {
                "VariantName": variant,
                "CurrentWeight": 1.0,
                "DesiredWeight": 1.0,
                "CurrentInstanceCount": current,
                "DesiredInstanceCount": desired,
            }
        ],
    }


def _stub_describe_targets(stub: Stubber, targets: list[dict]) -> None:
    stub.add_response(
        "describe_scalable_targets",
        {"ScalableTargets": targets},
        {
            "ServiceNamespace": SERVICE_NAMESPACE,
            "ResourceIds": [RID],
            "ScalableDimension": SCALABLE_DIMENSION,
        },
    )


def _stub_describe_policies(stub: Stubber, names: list[str]) -> None:
    stub.add_response(
        "describe_scaling_policies",
        {
            "ScalingPolicies": [
                {
                    "PolicyARN": f"arn:aws:autoscaling:::policy/{name}",
                    "PolicyName": name,
                    "ServiceNamespace": SERVICE_NAMESPACE,
                    "ResourceId": RID,
                    "ScalableDimension": SCALABLE_DIMENSION,
                    "PolicyType": "TargetTrackingScaling",
                    "CreationTime": "2026-06-18T17:19:51Z",
                }
                for name in names
            ]
        },
        {
            "ServiceNamespace": SERVICE_NAMESPACE,
            "ResourceId": RID,
            "ScalableDimension": SCALABLE_DIMENSION,
        },
    )


def _stub_capture(
    aas_stub: Stubber,
    sm_stub: Stubber,
    *,
    targets: list[dict] | None = None,
    policies: list[str] | None = None,
    desired: int = 1,
    current: int = 1,
) -> None:
    """Queue the three calls a capture() makes, in order."""
    targets = [] if targets is None else targets
    _stub_describe_targets(aas_stub, targets)
    if targets:
        _stub_describe_policies(aas_stub, policies if policies is not None else ["policy-1"])
    sm_stub.add_response(
        "describe_endpoint", _endpoint(desired=desired, current=current), {"EndpointName": ENDPOINT}
    )


class TestResourceId:
    def test_matches_application_autoscaling_format(self) -> None:
        assert resource_id("speech-kokoro-82m") == "endpoint/speech-kokoro-82m/variant/primary"
        assert resource_id("e", "v") == "endpoint/e/variant/v"


class TestCapture:
    def test_reads_state_without_changing_anything(self, appscaling, sagemaker) -> None:
        aas, aas_stub = appscaling
        sm, sm_stub = sagemaker
        _stub_capture(
            aas_stub, sm_stub, targets=[_target()], policies=["p1", "p2"], desired=2, current=2
        )

        state = capture(ENDPOINT, appscaling=aas, sagemaker=sm)

        assert state.has_scalable_target
        assert state.min_capacity == 1
        assert state.max_capacity == 4
        assert state.desired_instance_count == 2
        assert state.current_instance_count == 2
        assert state.policy_names == ("p1", "p2")
        # Stubber raises on any unstubbed call, so a mutation would have failed.
        aas_stub.assert_no_pending_responses()
        sm_stub.assert_no_pending_responses()

    def test_no_scalable_target_is_not_an_error(self, appscaling, sagemaker) -> None:
        # True for speech-kokoro-82m and speech-chatterbox-turbo today.
        aas, aas_stub = appscaling
        sm, sm_stub = sagemaker
        _stub_capture(aas_stub, sm_stub, targets=[])

        state = capture(ENDPOINT, appscaling=aas, sagemaker=sm)

        assert state.suspended_state is None
        assert not state.has_scalable_target
        assert state.policy_names == ()

    def test_missing_variant_raises(self, appscaling, sagemaker) -> None:
        aas, aas_stub = appscaling
        sm, sm_stub = sagemaker
        _stub_describe_targets(aas_stub, [])
        sm_stub.add_response(
            "describe_endpoint", _endpoint(variant="other"), {"EndpointName": ENDPOINT}
        )

        with pytest.raises(FixtureError, match="no variant named"):
            capture(ENDPOINT, appscaling=aas, sagemaker=sm)

    def test_policy_listing_failure_is_not_fatal(self, appscaling, sagemaker) -> None:
        # Policy names feed reporting, not the freeze itself.
        aas, aas_stub = appscaling
        sm, sm_stub = sagemaker
        _stub_describe_targets(aas_stub, [_target()])
        aas_stub.add_client_error("describe_scaling_policies", service_error_code="AccessDenied")
        sm_stub.add_response("describe_endpoint", _endpoint(), {"EndpointName": ENDPOINT})

        state = capture(ENDPOINT, appscaling=aas, sagemaker=sm)
        assert state.policy_names == ()
        assert state.has_scalable_target


class TestFreeze:
    def test_suspends_without_sending_capacity_limits(self, appscaling, sagemaker) -> None:
        # THE load-bearing assertion. Stubber matches expected params exactly, so
        # this fails if MinCapacity or MaxCapacity is ever added to the call.
        aas, aas_stub = appscaling
        sm, sm_stub = sagemaker

        _stub_capture(aas_stub, sm_stub, targets=[_target()], desired=1, current=1)
        aas_stub.add_response(
            "register_scalable_target",
            {},
            {
                "ServiceNamespace": SERVICE_NAMESPACE,
                "ResourceId": RID,
                "ScalableDimension": SCALABLE_DIMENSION,
                "SuspendedState": SUSPEND_ALL,
            },
        )
        sm_stub.add_response("describe_endpoint", _endpoint(), {"EndpointName": ENDPOINT})
        _stub_capture(
            aas_stub,
            sm_stub,
            targets=[
                _target(scale_out_suspended=True, scale_in_suspended=True, scheduled_suspended=True)
            ],
        )

        before = freeze(ENDPOINT, appscaling=aas, sagemaker=sm, pin_to=1)

        # Returns pre-change state so thaw can restore it.
        assert before.suspended_state == {
            "DynamicScalingInSuspended": False,
            "DynamicScalingOutSuspended": False,
            "ScheduledScalingSuspended": False,
        }
        assert before.max_capacity == 4
        aas_stub.assert_no_pending_responses()

    def test_suspends_all_three_flags(self) -> None:
        # Scheduled actions can move capacity too, so scale-out alone is not
        # enough to guarantee a stable fleet.
        assert SUSPEND_ALL == {
            "DynamicScalingInSuspended": True,
            "DynamicScalingOutSuspended": True,
            "ScheduledScalingSuspended": True,
        }

    def test_pins_capacity_when_fleet_is_too_large(self, appscaling, sagemaker) -> None:
        aas, aas_stub = appscaling
        sm, sm_stub = sagemaker

        _stub_capture(aas_stub, sm_stub, targets=[_target()], desired=3, current=3)
        aas_stub.add_response("register_scalable_target", {})
        sm_stub.add_response(
            "describe_endpoint", _endpoint(desired=3, current=3), {"EndpointName": ENDPOINT}
        )
        sm_stub.add_response(
            "update_endpoint_weights_and_capacities",
            {"EndpointArn": "arn:aws:sagemaker:us-east-1:1234:endpoint/x"},
            {
                "EndpointName": ENDPOINT,
                "DesiredWeightsAndCapacities": [
                    {"VariantName": "primary", "DesiredInstanceCount": 1}
                ],
            },
        )
        sm_stub.add_response(
            "describe_endpoint", _endpoint(desired=1, current=1), {"EndpointName": ENDPOINT}
        )
        _stub_capture(aas_stub, sm_stub, targets=[_target(scale_out_suspended=True)])

        before = freeze(ENDPOINT, appscaling=aas, sagemaker=sm, pin_to=1, poll_interval_s=0.0)
        assert before.desired_instance_count == 3
        sm_stub.assert_no_pending_responses()

    def test_no_scalable_target_skips_suspend_and_creates_nothing(
        self, appscaling, sagemaker
    ) -> None:
        # Creating a target would leave configuration behind that CDK does not
        # describe — the exact drift that produced the two live orphans.
        aas, aas_stub = appscaling
        sm, sm_stub = sagemaker

        _stub_capture(aas_stub, sm_stub, targets=[])
        sm_stub.add_response("describe_endpoint", _endpoint(), {"EndpointName": ENDPOINT})
        _stub_capture(aas_stub, sm_stub, targets=[])

        before = freeze(ENDPOINT, appscaling=aas, sagemaker=sm, pin_to=1)

        assert before.suspended_state is None
        # No register_scalable_target was stubbed; a call would have raised.
        aas_stub.assert_no_pending_responses()

    def test_object_not_found_on_suspend_is_tolerated(self, appscaling, sagemaker) -> None:
        # Race: the target disappears between capture and suspend.
        aas, aas_stub = appscaling
        sm, sm_stub = sagemaker

        _stub_capture(aas_stub, sm_stub, targets=[_target()])
        aas_stub.add_client_error(
            "register_scalable_target", service_error_code="ObjectNotFoundException"
        )
        sm_stub.add_response("describe_endpoint", _endpoint(), {"EndpointName": ENDPOINT})
        _stub_capture(aas_stub, sm_stub, targets=[])

        before = freeze(ENDPOINT, appscaling=aas, sagemaker=sm, pin_to=1)
        assert before.has_scalable_target

    def test_refuses_when_scale_out_survives_the_suspend(self, appscaling, sagemaker) -> None:
        # Verification is not optional: an unverified freeze yields a C_max that
        # looks identical to a good one.
        aas, aas_stub = appscaling
        sm, sm_stub = sagemaker

        _stub_capture(aas_stub, sm_stub, targets=[_target()])
        aas_stub.add_response("register_scalable_target", {})
        sm_stub.add_response("describe_endpoint", _endpoint(), {"EndpointName": ENDPOINT})
        _stub_capture(aas_stub, sm_stub, targets=[_target(scale_out_suspended=False)])

        with pytest.raises(FixtureError, match="scale-out still active"):
            freeze(ENDPOINT, appscaling=aas, sagemaker=sm, pin_to=1)

    def test_refuses_when_instance_count_is_wrong(self, appscaling, sagemaker) -> None:
        aas, aas_stub = appscaling
        sm, sm_stub = sagemaker

        _stub_capture(aas_stub, sm_stub, targets=[_target()], desired=1, current=1)
        aas_stub.add_response("register_scalable_target", {})
        sm_stub.add_response("describe_endpoint", _endpoint(), {"EndpointName": ENDPOINT})
        _stub_capture(
            aas_stub,
            sm_stub,
            targets=[_target(scale_out_suspended=True)],
            desired=2,
            current=2,
        )

        with pytest.raises(FixtureError, match="expected 1 instance"):
            freeze(ENDPOINT, appscaling=aas, sagemaker=sm, pin_to=1)

    def test_rejects_pin_to_below_one(self, appscaling, sagemaker) -> None:
        aas, _ = appscaling
        sm, _ = sagemaker
        with pytest.raises(ValueError, match="pin_to must be >= 1"):
            freeze(ENDPOINT, appscaling=aas, sagemaker=sm, pin_to=0)


class TestThaw:
    def test_restores_exact_captured_state(self, appscaling, sagemaker) -> None:
        aas, aas_stub = appscaling
        sm, sm_stub = sagemaker
        captured = EndpointFixture(
            endpoint_name=ENDPOINT,
            variant="primary",
            suspended_state={
                "DynamicScalingInSuspended": True,
                "DynamicScalingOutSuspended": False,
                "ScheduledScalingSuspended": False,
            },
            min_capacity=1,
            max_capacity=4,
            desired_instance_count=2,
            current_instance_count=2,
        )
        # Restores the captured mix, not a blanket resume: scale-in was already
        # suspended before the benchmark and must stay that way.
        aas_stub.add_response(
            "register_scalable_target",
            {},
            {
                "ServiceNamespace": SERVICE_NAMESPACE,
                "ResourceId": RID,
                "ScalableDimension": SCALABLE_DIMENSION,
                "SuspendedState": {
                    "DynamicScalingInSuspended": True,
                    "DynamicScalingOutSuspended": False,
                    "ScheduledScalingSuspended": False,
                },
            },
        )
        sm_stub.add_response(
            "describe_endpoint", _endpoint(desired=1, current=1), {"EndpointName": ENDPOINT}
        )
        sm_stub.add_response(
            "update_endpoint_weights_and_capacities",
            {"EndpointArn": f"arn:aws:sagemaker:us-east-1:1234:endpoint/{ENDPOINT}"},
            {
                "EndpointName": ENDPOINT,
                "DesiredWeightsAndCapacities": [
                    {"VariantName": "primary", "DesiredInstanceCount": 2}
                ],
            },
        )

        thaw(captured, appscaling=aas, sagemaker=sm)

        aas_stub.assert_no_pending_responses()
        sm_stub.assert_no_pending_responses()

    def test_is_idempotent_when_capacity_already_matches(self, appscaling, sagemaker) -> None:
        # Lets `tts-bench thaw` recover from a hard kill without side effects.
        aas, aas_stub = appscaling
        sm, sm_stub = sagemaker
        captured = EndpointFixture(
            endpoint_name=ENDPOINT,
            variant="primary",
            suspended_state=dict.fromkeys(
                (
                    "DynamicScalingInSuspended",
                    "DynamicScalingOutSuspended",
                    "ScheduledScalingSuspended",
                ),
                False,
            ),
            min_capacity=1,
            max_capacity=4,
            desired_instance_count=1,
            current_instance_count=1,
        )
        aas_stub.add_response("register_scalable_target", {})
        sm_stub.add_response(
            "describe_endpoint", _endpoint(desired=1, current=1), {"EndpointName": ENDPOINT}
        )

        thaw(captured, appscaling=aas, sagemaker=sm)
        # No update call was stubbed, so issuing one would have raised.
        sm_stub.assert_no_pending_responses()

    def test_skips_suspend_restore_when_no_target_existed(self, appscaling, sagemaker) -> None:
        aas, aas_stub = appscaling
        sm, sm_stub = sagemaker
        captured = EndpointFixture(
            endpoint_name=ENDPOINT,
            variant="primary",
            suspended_state=None,
            min_capacity=None,
            max_capacity=None,
            desired_instance_count=1,
            current_instance_count=1,
        )
        sm_stub.add_response(
            "describe_endpoint", _endpoint(desired=1, current=1), {"EndpointName": ENDPOINT}
        )

        thaw(captured, appscaling=aas, sagemaker=sm)
        aas_stub.assert_no_pending_responses()

    def test_never_raises_so_it_cannot_mask_the_original_failure(
        self, appscaling, sagemaker
    ) -> None:
        aas, aas_stub = appscaling
        sm, sm_stub = sagemaker
        captured = EndpointFixture(
            endpoint_name=ENDPOINT,
            variant="primary",
            suspended_state=dict.fromkeys(
                (
                    "DynamicScalingInSuspended",
                    "DynamicScalingOutSuspended",
                    "ScheduledScalingSuspended",
                ),
                False,
            ),
            min_capacity=1,
            max_capacity=4,
            desired_instance_count=1,
            current_instance_count=1,
        )
        aas_stub.add_client_error("register_scalable_target", service_error_code="AccessDenied")
        sm_stub.add_client_error("describe_endpoint", service_error_code="AccessDenied")

        thaw(captured, appscaling=aas, sagemaker=sm)  # logs, does not raise


class TestFrozenContextManager:
    def _stub_full_freeze(self, aas_stub: Stubber, sm_stub: Stubber) -> None:
        _stub_capture(aas_stub, sm_stub, targets=[_target()], desired=1, current=1)
        aas_stub.add_response("register_scalable_target", {})
        sm_stub.add_response("describe_endpoint", _endpoint(), {"EndpointName": ENDPOINT})
        _stub_capture(aas_stub, sm_stub, targets=[_target(scale_out_suspended=True)])

    def _stub_thaw(self, aas_stub: Stubber, sm_stub: Stubber) -> None:
        aas_stub.add_response("register_scalable_target", {})
        sm_stub.add_response(
            "describe_endpoint", _endpoint(desired=1, current=1), {"EndpointName": ENDPOINT}
        )

    def test_thaws_on_clean_exit(self, appscaling, sagemaker) -> None:
        aas, aas_stub = appscaling
        sm, sm_stub = sagemaker
        self._stub_full_freeze(aas_stub, sm_stub)
        self._stub_thaw(aas_stub, sm_stub)

        with frozen(ENDPOINT, appscaling=aas, sagemaker=sm) as state:
            assert state.endpoint_name == ENDPOINT

        aas_stub.assert_no_pending_responses()
        sm_stub.assert_no_pending_responses()

    def test_thaws_on_exception_and_propagates_it(self, appscaling, sagemaker) -> None:
        aas, aas_stub = appscaling
        sm, sm_stub = sagemaker
        self._stub_full_freeze(aas_stub, sm_stub)
        self._stub_thaw(aas_stub, sm_stub)

        with pytest.raises(RuntimeError, match="benchmark blew up"):
            with frozen(ENDPOINT, appscaling=aas, sagemaker=sm):
                raise RuntimeError("benchmark blew up")

        # Restored despite the failure, and the error was not swallowed.
        aas_stub.assert_no_pending_responses()
        sm_stub.assert_no_pending_responses()

    def test_thaws_on_keyboard_interrupt(self, appscaling, sagemaker) -> None:
        # Ctrl-C must not leave production frozen. This is why `frozen` is a
        # class rather than a @contextmanager generator.
        aas, aas_stub = appscaling
        sm, sm_stub = sagemaker
        self._stub_full_freeze(aas_stub, sm_stub)
        self._stub_thaw(aas_stub, sm_stub)

        with pytest.raises(KeyboardInterrupt):
            with frozen(ENDPOINT, appscaling=aas, sagemaker=sm):
                raise KeyboardInterrupt

        aas_stub.assert_no_pending_responses()
        sm_stub.assert_no_pending_responses()

    def test_does_not_thaw_when_freeze_itself_failed(self, appscaling, sagemaker) -> None:
        # Nothing was changed, so there is nothing to restore.
        aas, aas_stub = appscaling
        sm, sm_stub = sagemaker
        _stub_capture(aas_stub, sm_stub, targets=[_target()])
        aas_stub.add_response("register_scalable_target", {})
        sm_stub.add_response("describe_endpoint", _endpoint(), {"EndpointName": ENDPOINT})
        _stub_capture(aas_stub, sm_stub, targets=[_target(scale_out_suspended=False)])

        with pytest.raises(FixtureError):
            with frozen(ENDPOINT, appscaling=aas, sagemaker=sm):
                pytest.fail("body must not run")


class TestRequireFrozen:
    def test_passes_when_suspended_and_pinned(self, appscaling, sagemaker) -> None:
        aas, aas_stub = appscaling
        sm, sm_stub = sagemaker
        _stub_capture(
            aas_stub, sm_stub, targets=[_target(scale_out_suspended=True)], desired=1, current=1
        )

        state = require_frozen(ENDPOINT, appscaling=aas, sagemaker=sm)
        assert state.scale_out_suspended

    def test_raises_when_scale_out_is_live(self, appscaling, sagemaker) -> None:
        # The orpheus-3b case: max_capacity=4 with an active policy.
        aas, aas_stub = appscaling
        sm, sm_stub = sagemaker
        _stub_capture(aas_stub, sm_stub, targets=[_target(scale_out_suspended=False)])

        with pytest.raises(FixtureError, match="scale-out is not suspended"):
            require_frozen(ENDPOINT, appscaling=aas, sagemaker=sm)

    def test_raises_when_more_than_one_instance(self, appscaling, sagemaker) -> None:
        aas, aas_stub = appscaling
        sm, sm_stub = sagemaker
        _stub_capture(
            aas_stub, sm_stub, targets=[_target(scale_out_suspended=True)], desired=3, current=3
        )

        with pytest.raises(FixtureError, match="current instance count is 3"):
            require_frozen(ENDPOINT, appscaling=aas, sagemaker=sm)

    def test_passes_when_no_scalable_target_exists(self, appscaling, sagemaker) -> None:
        # Nothing can add instances, which is the property we need.
        aas, aas_stub = appscaling
        sm, sm_stub = sagemaker
        _stub_capture(aas_stub, sm_stub, targets=[], desired=1, current=1)

        assert require_frozen(ENDPOINT, appscaling=aas, sagemaker=sm).scale_out_suspended

    def test_error_mentions_the_override(self, appscaling, sagemaker) -> None:
        aas, aas_stub = appscaling
        sm, sm_stub = sagemaker
        _stub_capture(aas_stub, sm_stub, targets=[_target()])

        with pytest.raises(FixtureError, match="--no-require-frozen"):
            require_frozen(ENDPOINT, appscaling=aas, sagemaker=sm)


class TestRequireScalable:
    def test_passes_when_a_scale_event_is_possible(self, appscaling, sagemaker) -> None:
        aas, aas_stub = appscaling
        sm, sm_stub = sagemaker
        _stub_capture(aas_stub, sm_stub, targets=[_target(max_capacity=4)], policies=["p1"])

        state = require_scalable(ENDPOINT, appscaling=aas, sagemaker=sm)
        assert state.max_capacity == 4

    def test_raises_without_a_scalable_target(self, appscaling, sagemaker) -> None:
        aas, aas_stub = appscaling
        sm, sm_stub = sagemaker
        _stub_capture(aas_stub, sm_stub, targets=[])

        with pytest.raises(FixtureError, match="no scalable target"):
            require_scalable(ENDPOINT, appscaling=aas, sagemaker=sm)

    def test_raises_when_max_capacity_is_one(self, appscaling, sagemaker) -> None:
        # Fail in seconds rather than after twenty minutes of waiting for a
        # scale event that cannot happen.
        aas, aas_stub = appscaling
        sm, sm_stub = sagemaker
        _stub_capture(aas_stub, sm_stub, targets=[_target(max_capacity=1)], policies=["p1"])

        with pytest.raises(FixtureError, match="max_capacity=1"):
            require_scalable(ENDPOINT, appscaling=aas, sagemaker=sm)

    def test_raises_when_left_suspended_by_an_aborted_freeze(self, appscaling, sagemaker) -> None:
        aas, aas_stub = appscaling
        sm, sm_stub = sagemaker
        _stub_capture(
            aas_stub, sm_stub, targets=[_target(scale_out_suspended=True)], policies=["p1"]
        )

        with pytest.raises(FixtureError, match="scale-out is suspended"):
            require_scalable(ENDPOINT, appscaling=aas, sagemaker=sm)

    def test_raises_when_no_policy_is_attached(self, appscaling, sagemaker) -> None:
        # A target with no policy is the inert half of the current drift.
        aas, aas_stub = appscaling
        sm, sm_stub = sagemaker
        _stub_capture(aas_stub, sm_stub, targets=[_target(max_capacity=4)], policies=[])

        with pytest.raises(FixtureError, match="no scaling policies"):
            require_scalable(ENDPOINT, appscaling=aas, sagemaker=sm)


class TestEndpointFixtureProperties:
    def test_no_target_counts_as_scale_out_suspended(self) -> None:
        state = EndpointFixture(
            endpoint_name=ENDPOINT,
            variant="primary",
            suspended_state=None,
            min_capacity=None,
            max_capacity=None,
            desired_instance_count=1,
            current_instance_count=1,
        )
        assert state.scale_out_suspended
        assert not state.has_scalable_target

    def test_partial_suspend_is_not_enough(self) -> None:
        state = EndpointFixture(
            endpoint_name=ENDPOINT,
            variant="primary",
            suspended_state={"DynamicScalingInSuspended": True},
            min_capacity=1,
            max_capacity=4,
            desired_instance_count=1,
            current_instance_count=1,
        )
        assert not state.scale_out_suspended

    def test_is_immutable(self) -> None:
        state = EndpointFixture(
            endpoint_name=ENDPOINT,
            variant="primary",
            suspended_state=None,
            min_capacity=None,
            max_capacity=None,
            desired_instance_count=1,
            current_instance_count=1,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            state.desired_instance_count = 5  # type: ignore[misc]
