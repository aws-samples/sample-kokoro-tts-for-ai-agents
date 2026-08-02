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
from loguru import logger

from tts_bench.fixture import (
    ENDPOINT_USAGE_QUOTA_CODE,
    SCALABLE_DIMENSION,
    SERVICE_NAMESPACE,
    SUSPEND_ALL,
    DeployedConfig,
    EndpointFixture,
    FixtureError,
    QuotaHeadroom,
    capture,
    describe_deployed_config,
    endpoint_quota_headroom,
    fingerprint_or_registry,
    freeze,
    frozen,
    registry_instance_type,
    require_frozen,
    require_scalable,
    resource_id,
    thaw,
    variant_instance_type,
)

ENDPOINT = "speech-kokoro-82m"
RID = f"endpoint/{ENDPOINT}/variant/primary"
INSTANCE_TYPE = "ml.g5.xlarge"

#: A CDK container-asset tag, which is a content hash of the build context — the
#: property that makes it a usable fingerprint for "did the serving code change".
#: Shape copied from the live kokoro endpoint.
IMAGE_DIGEST = "139b9068c5eb1f03c8312c17391dc35838e43e2417ae80d61e628e6ffb3d6a28"
ECR_REPO = "1234.dkr.ecr.us-east-1.amazonaws.com/cdk-hnb659fds-container-assets-1234-us-east-1"
MODEL_NAME = "m"

#: The other holders of the ml.g5.xlarge quota when this guard was written. Named
#: because the point of the check is that endpoints we are not benchmarking, and in one
#: case do not own, consume the allowance this one scales into.
OTHER_ENDPOINTS = ("speech-orpheus-3b", "speech-chatterbox-turbo")


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


@pytest.fixture
def quotas() -> Any:
    client = boto3.client("service-quotas", region_name="us-east-1")
    stub = Stubber(client)
    stub.activate()
    yield client, stub
    stub.deactivate()


@pytest.fixture
def logged() -> Any:
    """Captured loguru warnings.

    ``caplog`` does not see these — loguru does not propagate to the stdlib logging
    tree — so an assertion against it would pass whether or not anything was emitted.
    """
    records: list[str] = []
    sink_id = logger.add(lambda m: records.append(m.record["message"]), level="WARNING")
    try:
        yield records
    finally:
        logger.remove(sink_id)


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


def _config(*, instance_type: str | None = INSTANCE_TYPE, variant: str = "primary") -> dict:
    """An endpoint config. The *only* place the instance type is readable."""
    entry: dict[str, Any] = {"VariantName": variant, "ModelName": "m"}
    if instance_type is not None:
        entry["InstanceType"] = instance_type
        entry["InitialInstanceCount"] = 1
    return {
        "EndpointConfigName": f"{ENDPOINT}-config",
        "EndpointConfigArn": f"arn:aws:sagemaker:us-east-1:1234:endpoint-config/{ENDPOINT}-config",
        "ProductionVariants": [entry],
        "CreationTime": "2026-06-18T17:19:51Z",
    }


def _stub_variant_instance_type(
    sm_stub: Stubber, *, instance_type: str | None = INSTANCE_TYPE
) -> None:
    """Queue the describe_endpoint + describe_endpoint_config pair."""
    sm_stub.add_response("describe_endpoint", _endpoint(), {"EndpointName": ENDPOINT})
    sm_stub.add_response(
        "describe_endpoint_config",
        _config(instance_type=instance_type),
        {"EndpointConfigName": f"{ENDPOINT}-config"},
    )


def _stub_deployed_config(
    sm_stub: Stubber,
    *,
    instance_type: str | None = INSTANCE_TYPE,
    image: str | None = None,
    env: dict[str, str] | None = None,
) -> None:
    """Queue the describe_endpoint + _config + describe_model trio."""
    sm_stub.add_response("describe_endpoint", _endpoint(), {"EndpointName": ENDPOINT})
    sm_stub.add_response(
        "describe_endpoint_config",
        _config(instance_type=instance_type),
        {"EndpointConfigName": f"{ENDPOINT}-config"},
    )
    sm_stub.add_response(
        "describe_model",
        {
            "ModelName": MODEL_NAME,
            "ModelArn": f"arn:aws:sagemaker:us-east-1:1234:model/{MODEL_NAME}",
            "CreationTime": "2026-06-18T17:19:51Z",
            "PrimaryContainer": {
                "Image": image if image is not None else f"{ECR_REPO}:{IMAGE_DIGEST}",
                "Environment": {"MAX_REQUEST_AGE_S": "56"} if env is None else env,
            },
        },
        {"ModelName": MODEL_NAME},
    )


def _deployed(
    *,
    instance_type: str | None = INSTANCE_TYPE,
    image_digest: str | None = IMAGE_DIGEST,
    env: dict[str, str] | None = None,
) -> DeployedConfig:
    return DeployedConfig(
        instance_type=instance_type,
        image_digest=image_digest,
        container_env={"MAX_REQUEST_AGE_S": "56"} if env is None else env,
    )


def _summary(name: str, *, status: str = "InService") -> dict:
    return {
        "EndpointName": name,
        "EndpointArn": f"arn:aws:sagemaker:us-east-1:1234:endpoint/{name}",
        "CreationTime": "2026-06-18T17:19:51Z",
        "LastModifiedTime": "2026-06-18T17:19:51Z",
        "EndpointStatus": status,
    }


def _stub_headroom(
    sm_stub: Stubber,
    quotas_stub: Stubber,
    *,
    limit: float | None = 4.0,
    others: tuple[str, ...] = OTHER_ENDPOINTS,
    self_current: int = 1,
    self_desired: int | None = None,
    self_type: str = INSTANCE_TYPE,
) -> None:
    """Queue one endpoint_quota_headroom() pass: the quota read, then every endpoint."""
    if limit is None:
        quotas_stub.add_client_error("list_service_quotas", service_error_code="AccessDenied")
    else:
        quotas_stub.add_response(
            "list_service_quotas",
            {
                "Quotas": [
                    {
                        "QuotaCode": ENDPOINT_USAGE_QUOTA_CODE,
                        "QuotaName": f"{INSTANCE_TYPE} for endpoint usage",
                        "Value": limit,
                    }
                ]
            },
            {"ServiceCode": "sagemaker", "QuotaCode": ENDPOINT_USAGE_QUOTA_CODE},
        )

    names = (ENDPOINT, *others)
    sm_stub.add_response("list_endpoints", {"Endpoints": [_summary(n) for n in names]})
    for name in names:
        is_self = name == ENDPOINT
        described = _endpoint(
            desired=self_desired if (is_self and self_desired is not None) else 1,
            current=self_current if is_self else 1,
        )
        described["EndpointName"] = name
        described["EndpointConfigName"] = f"{name}-config"
        sm_stub.add_response("describe_endpoint", described, {"EndpointName": name})
        config = _config(instance_type=self_type if is_self else INSTANCE_TYPE)
        config["EndpointConfigName"] = f"{name}-config"
        sm_stub.add_response(
            "describe_endpoint_config", config, {"EndpointConfigName": f"{name}-config"}
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
    def test_passes_when_a_scale_event_is_possible(self, appscaling, sagemaker, quotas) -> None:
        aas, aas_stub = appscaling
        sm, sm_stub = sagemaker
        q, q_stub = quotas
        _stub_capture(aas_stub, sm_stub, targets=[_target(max_capacity=4)], policies=["p1"])
        _stub_variant_instance_type(sm_stub)
        # Room for the full jump to max_capacity=4 from 1: limit 8, three others in use.
        _stub_headroom(sm_stub, q_stub, limit=8.0)

        state = require_scalable(ENDPOINT, appscaling=aas, sagemaker=sm, quotas=q)
        assert state.max_capacity == 4
        q_stub.assert_no_pending_responses()

    def test_the_quota_read_is_skipped_when_not_asked_for(
        self, appscaling, sagemaker, quotas
    ) -> None:
        # Stubber raises on any unstubbed call, so nothing being queued is the assertion.
        aas, aas_stub = appscaling
        sm, sm_stub = sagemaker
        q, q_stub = quotas
        _stub_capture(aas_stub, sm_stub, targets=[_target(max_capacity=4)], policies=["p1"])

        state = require_scalable(
            ENDPOINT, appscaling=aas, sagemaker=sm, quotas=q, require_quota_headroom=0
        )
        assert state.max_capacity == 4
        sm_stub.assert_no_pending_responses()

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


class TestQuotaHeadroomArithmetic:
    def test_available_is_the_remainder(self) -> None:
        h = QuotaHeadroom(instance_type=INSTANCE_TYPE, limit=4, in_use=3)
        assert h.available == 1
        assert h.room_for(1) is True
        assert h.room_for(2) is False

    def test_an_unknown_limit_is_not_headroom(self) -> None:
        # Absence of evidence. Returning True here would defeat the guard silently on
        # any account whose role lacks servicequotas:ListServiceQuotas.
        h = QuotaHeadroom(instance_type=INSTANCE_TYPE, limit=None, in_use=3)
        assert h.available is None
        assert h.room_for(1) is None

    def test_over_quota_does_not_report_negative_room(self) -> None:
        # Reachable: a quota can be lowered under running endpoints.
        h = QuotaHeadroom(instance_type=INSTANCE_TYPE, limit=2, in_use=5)
        assert h.available == 0
        assert h.room_for(1) is False


class TestVariantInstanceType:
    def test_reads_it_from_the_endpoint_config(self, sagemaker) -> None:
        # DescribeEndpoint omits InstanceType entirely, which is why this is two calls.
        sm, sm_stub = sagemaker
        _stub_variant_instance_type(sm_stub)

        assert variant_instance_type(ENDPOINT, sagemaker=sm) == INSTANCE_TYPE

    def test_a_serverless_variant_has_none(self, sagemaker) -> None:
        # No instances, so no instance quota to check.
        sm, sm_stub = sagemaker
        _stub_variant_instance_type(sm_stub, instance_type=None)

        assert variant_instance_type(ENDPOINT, sagemaker=sm) is None

    def test_an_unknown_variant_has_none(self, sagemaker) -> None:
        sm, sm_stub = sagemaker
        _stub_variant_instance_type(sm_stub)

        assert variant_instance_type(ENDPOINT, variant="other", sagemaker=sm) is None

    def test_a_read_failure_raises(self, sagemaker) -> None:
        sm, sm_stub = sagemaker
        sm_stub.add_client_error("describe_endpoint", service_error_code="ValidationException")

        with pytest.raises(FixtureError, match="could not read the endpoint config"):
            variant_instance_type(ENDPOINT, sagemaker=sm)


class TestDescribeDeployedConfig:
    """Reading what a benchmark is actually measuring against.

    Three API calls, because SageMaker splits the answer three ways. The point of
    the fingerprint is that a re-deployed endpoint produces a *different* one, so
    these tests care about which fields move and which deliberately do not.
    """

    def test_it_reads_type_image_and_env(self, sagemaker) -> None:
        sm, sm_stub = sagemaker
        _stub_deployed_config(sm_stub)

        cfg = describe_deployed_config(ENDPOINT, sagemaker=sm)

        assert cfg.instance_type == INSTANCE_TYPE
        assert cfg.image_digest == IMAGE_DIGEST
        assert cfg.container_env == {"MAX_REQUEST_AGE_S": "56"}

    def test_ambient_env_is_excluded(self, sagemaker) -> None:
        # ENDPOINT_NAME and SM_MODEL_ID are set from the endpoint's own identity, and
        # AWS_* by the runtime. Including them would make the fingerprint endpoint-
        # specific, so the same build measured on two endpoints would never compare.
        sm, sm_stub = sagemaker
        _stub_deployed_config(
            sm_stub,
            env={
                "AWS_REGION": "us-east-1",
                "AWS_DEFAULT_REGION": "us-east-1",
                "ENDPOINT_NAME": ENDPOINT,
                "SM_MODEL_ID": "kokoro-82m",
                "MAX_REQUEST_AGE_S": "56",
            },
        )

        cfg = describe_deployed_config(ENDPOINT, sagemaker=sm)

        assert cfg.container_env == {"MAX_REQUEST_AGE_S": "56"}

    def test_the_slug_is_short_and_filename_safe(self, sagemaker) -> None:
        sm, sm_stub = sagemaker
        _stub_deployed_config(sm_stub, instance_type="ml.g6.12xlarge")

        slug = describe_deployed_config(ENDPOINT, sagemaker=sm).slug

        assert slug == f"g612xlarge-{IMAGE_DIGEST[:8]}"
        assert "/" not in slug and ":" not in slug and "." not in slug

    def test_an_unknown_variant_raises(self, sagemaker) -> None:
        # Unlike variant_instance_type, which returns None: a fingerprint of nothing
        # would be recorded on the artifact as though it described the run.
        sm, sm_stub = sagemaker
        _stub_deployed_config(sm_stub)

        with pytest.raises(FixtureError, match="no variant named 'other'"):
            describe_deployed_config(ENDPOINT, variant="other", sagemaker=sm)

    def test_an_unreadable_model_raises(self, sagemaker) -> None:
        sm, sm_stub = sagemaker
        sm_stub.add_response("describe_endpoint", _endpoint(), {"EndpointName": ENDPOINT})
        sm_stub.add_response(
            "describe_endpoint_config",
            _config(),
            {"EndpointConfigName": f"{ENDPOINT}-config"},
        )
        sm_stub.add_client_error("describe_model", service_error_code="ValidationException")

        with pytest.raises(FixtureError, match="could not read model"):
            describe_deployed_config(ENDPOINT, sagemaker=sm)

    def test_a_digest_reference_beats_a_tag(self, sagemaker) -> None:
        sm, sm_stub = sagemaker
        _stub_deployed_config(sm_stub, image=f"{ECR_REPO}:latest@sha256:{'a' * 64}")

        assert describe_deployed_config(ENDPOINT, sagemaker=sm).image_digest == "a" * 64

    def test_an_untagged_image_has_no_digest(self, sagemaker) -> None:
        # Not an error: it still fingerprints on instance type, and a missing digest
        # is honestly reported rather than invented.
        sm, sm_stub = sagemaker
        _stub_deployed_config(sm_stub, image=ECR_REPO)

        cfg = describe_deployed_config(ENDPOINT, sagemaker=sm)

        assert cfg.image_digest is None
        assert cfg.slug.endswith("-nodigest")


class TestFingerprintOrRegistry:
    """The read every measurement command makes before it starts.

    Shared by `cmax` and `ttotal` rather than owned by either, because both produce
    per-configuration artifacts and the planner refuses to pair two that disagree.
    The behaviour worth pinning is the precedence: the endpoint wins over the
    registry, loudly, because a stale registry is silent in exactly the workflow this
    harness exists for — redeploy on new hardware, re-measure, and a static dict
    stamps the fresh artifact with the old type.
    """

    def test_the_endpoint_wins_and_the_divergence_is_loud(self, sagemaker, logged) -> None:
        # ERROR rather than WARNING: the same divergence also makes `drift` and the
        # cost model wrong, not just this one artifact.
        sm, sm_stub = sagemaker
        _stub_deployed_config(sm_stub, instance_type="ml.g6.xlarge")

        cfg = fingerprint_or_registry("kokoro-82m", endpoint=ENDPOINT, sagemaker=sm)

        assert cfg.instance_type == "ml.g6.xlarge"
        assert any("MODEL_INSTANCE_TYPES says ml.g5.xlarge" in m for m in logged)

    def test_an_agreeing_registry_is_silent(self, sagemaker, logged) -> None:
        sm, sm_stub = sagemaker
        _stub_deployed_config(sm_stub)

        cfg = fingerprint_or_registry("kokoro-82m", endpoint=ENDPOINT, sagemaker=sm)

        assert cfg.instance_type == INSTANCE_TYPE
        assert not any("MODEL_INSTANCE_TYPES" in m for m in logged)

    def test_an_unreadable_endpoint_falls_back_without_losing_the_run(
        self, sagemaker, logged
    ) -> None:
        # A failed describe must not cost a whole measurement. The fallback stays
        # honest: no image digest, so it can never compare equal to a real fingerprint.
        sm, sm_stub = sagemaker
        sm_stub.add_client_error("describe_endpoint", service_error_code="ValidationException")

        cfg = fingerprint_or_registry("kokoro-82m", endpoint=ENDPOINT, sagemaker=sm)

        assert cfg.instance_type == INSTANCE_TYPE
        assert cfg.image_digest is None
        assert cfg.slug.endswith("-nodigest")
        assert any("Falling back to the registry type" in m for m in logged)

    def test_an_unknown_model_is_costed_loudly_rather_than_crashing(
        self, sagemaker, logged
    ) -> None:
        sm, sm_stub = sagemaker
        _stub_deployed_config(sm_stub)

        cfg = fingerprint_or_registry("not-a-model", endpoint=ENDPOINT, sagemaker=sm)

        # The live read still succeeded, so the artifact is correct; the warning is
        # about the registry lookup that would have been the fallback.
        assert cfg.instance_type == INSTANCE_TYPE
        assert any("not in MODEL_INSTANCE_TYPES" in m for m in logged)


class TestRegistryInstanceType:
    def test_a_known_model_resolves(self) -> None:
        assert registry_instance_type("kokoro-82m") == INSTANCE_TYPE

    def test_an_unknown_model_warns_and_defaults(self, logged) -> None:
        from tts_bench.cost import DEFAULT_INSTANCE_TYPE

        assert registry_instance_type("not-a-model") == DEFAULT_INSTANCE_TYPE
        assert any("not in MODEL_INSTANCE_TYPES" in m for m in logged)


class TestDeployedConfigMatching:
    """The guard that stops a measurement being replayed on other hardware.

    A hard error, not a warning: C_target, fleet size, queue depth and cost are all
    derived from inputs that only hold for the configuration measured, and a warning
    scrolls past.
    """

    def test_identical_configurations_pass(self) -> None:
        _deployed().assert_matches(_deployed())

    def test_a_different_instance_type_is_refused(self) -> None:
        with pytest.raises(FixtureError, match="properties of the GPU"):
            _deployed().assert_matches(_deployed(instance_type="ml.g6.xlarge"))

    def test_a_different_image_is_refused(self) -> None:
        # The case a type-only check would miss: same GPU, different serving code.
        # An admission queue lands here, and it moves C_max.
        with pytest.raises(FixtureError, match="serving code differs"):
            _deployed().assert_matches(_deployed(image_digest="f" * 64))

    def test_a_different_env_is_refused(self) -> None:
        with pytest.raises(FixtureError, match="container_env"):
            _deployed().assert_matches(_deployed(env={"MAX_REQUEST_AGE_S": "30"}))

    def test_the_message_names_every_difference(self) -> None:
        with pytest.raises(FixtureError) as excinfo:
            _deployed().assert_matches(
                _deployed(instance_type="ml.g6.xlarge", image_digest="f" * 64, env={})
            )

        message = str(excinfo.value)
        assert "instance_type" in message
        assert "image_digest" in message
        assert "container_env" in message
        # And it says what to do about it, since re-measuring is the intended fix.
        assert "--allow-config-mismatch" in message

    def test_the_override_warns_instead(self, logged) -> None:
        _deployed().assert_matches(_deployed(instance_type="ml.g6.xlarge"), allow_mismatch=True)

        assert any("different configuration" in m for m in logged)

    def test_a_full_digest_is_trimmed_in_the_message(self) -> None:
        # 64-char hashes on both sides make the difference unreadable.
        with pytest.raises(FixtureError) as excinfo:
            _deployed().assert_matches(_deployed(image_digest="f" * 64))

        assert "f" * 64 not in str(excinfo.value)
        assert "f" * 12 in str(excinfo.value)

    def test_it_round_trips_through_an_artifact(self) -> None:
        original = _deployed()

        assert DeployedConfig.from_dict(original.to_dict()) == original

    def test_a_pre_fingerprint_artifact_does_not_match(self) -> None:
        # Artifacts written before fingerprinting have no configuration recorded.
        # Treating that as a match is exactly the hole this closes, so an empty
        # fingerprint must differ from any real one.
        empty = DeployedConfig.from_dict({})

        assert empty != _deployed()
        with pytest.raises(FixtureError):
            empty.assert_matches(_deployed())


class TestEndpointQuotaHeadroom:
    """The account-wide read.

    The quota is per instance type, per region, and counts *every* endpoint — so the
    total has to come from listing them all, not from the one under test.
    """

    def test_totals_every_endpoint_holding_the_type(self, sagemaker, quotas) -> None:
        sm, sm_stub = sagemaker
        q, q_stub = quotas
        _stub_headroom(sm_stub, q_stub, limit=4.0)

        h = endpoint_quota_headroom(INSTANCE_TYPE, quotas=q, sagemaker=sm)

        assert h.limit == 4
        assert h.in_use == 3
        assert h.endpoints_in_use == tuple(
            sorted(f"{n}/primary" for n in (ENDPOINT, *OTHER_ENDPOINTS))
        )
        assert h.available == 1

    def test_ignores_endpoints_on_other_instance_types(self, sagemaker, quotas) -> None:
        # The quota is per type: a g5 benchmark is not blocked by c5 endpoints.
        sm, sm_stub = sagemaker
        q, q_stub = quotas
        _stub_headroom(sm_stub, q_stub, limit=4.0, self_type="ml.c5.large")

        h = endpoint_quota_headroom(INSTANCE_TYPE, quotas=q, sagemaker=sm)

        assert h.in_use == 2
        assert all(ENDPOINT not in name for name in h.endpoints_in_use)

    def test_counts_a_scale_out_in_flight_at_its_desired_count(self, sagemaker, quotas) -> None:
        # Slots are claimed when desired rises, not when the instance appears. Counting
        # current only would report room that a pending activity has already taken.
        sm, sm_stub = sagemaker
        q, q_stub = quotas
        _stub_headroom(sm_stub, q_stub, limit=8.0, self_current=1, self_desired=4)

        assert endpoint_quota_headroom(INSTANCE_TYPE, quotas=q, sagemaker=sm).in_use == 6

    def test_a_failed_endpoint_holds_nothing(self, sagemaker, quotas) -> None:
        sm, sm_stub = sagemaker
        q, q_stub = quotas
        q_stub.add_response(
            "list_service_quotas",
            {
                "Quotas": [
                    {
                        "QuotaCode": ENDPOINT_USAGE_QUOTA_CODE,
                        "QuotaName": f"{INSTANCE_TYPE} for endpoint usage",
                        "Value": 4.0,
                    }
                ]
            },
        )
        sm_stub.add_response(
            "list_endpoints", {"Endpoints": [_summary("dead-endpoint", status="Failed")]}
        )

        h = endpoint_quota_headroom(INSTANCE_TYPE, quotas=q, sagemaker=sm)

        assert h.in_use == 0
        assert h.endpoints_in_use == ()

    def test_an_unreadable_quota_degrades_to_unknown(self, sagemaker, quotas, logged) -> None:
        # A missing servicequotas permission must not be the thing that stops a
        # measurement — but it must not read as headroom either.
        sm, sm_stub = sagemaker
        q, q_stub = quotas
        _stub_headroom(sm_stub, q_stub, limit=None)

        h = endpoint_quota_headroom(INSTANCE_TYPE, quotas=q, sagemaker=sm)

        assert h.limit is None
        assert h.in_use == 3
        assert any("quota" in m for m in logged)

    def test_an_unlistable_account_degrades_to_zero_in_use(self, sagemaker, quotas, logged) -> None:
        sm, sm_stub = sagemaker
        q, q_stub = quotas
        q_stub.add_response(
            "list_service_quotas",
            {
                "Quotas": [
                    {
                        "QuotaCode": ENDPOINT_USAGE_QUOTA_CODE,
                        "QuotaName": f"{INSTANCE_TYPE} for endpoint usage",
                        "Value": 4.0,
                    }
                ]
            },
        )
        sm_stub.add_client_error("list_endpoints", service_error_code="AccessDenied")

        h = endpoint_quota_headroom(INSTANCE_TYPE, quotas=q, sagemaker=sm)

        assert h.in_use == 0
        assert any("list endpoints" in m for m in logged)

    def test_a_differently_named_quota_is_not_matched(self, sagemaker, quotas) -> None:
        # One QuotaCode covers every instance type, so the name is the discriminator.
        sm, sm_stub = sagemaker
        q, q_stub = quotas
        q_stub.add_response(
            "list_service_quotas",
            {
                "Quotas": [
                    {
                        "QuotaCode": ENDPOINT_USAGE_QUOTA_CODE,
                        "QuotaName": "ml.p4d.24xlarge for endpoint usage",
                        "Value": 100.0,
                    }
                ]
            },
        )
        sm_stub.add_response("list_endpoints", {"Endpoints": []})

        assert endpoint_quota_headroom(INSTANCE_TYPE, quotas=q, sagemaker=sm).limit is None


class TestRequireScalableChecksQuota:
    """The guard that the first live ``ttotal`` run needed and did not have.

    That run drove load for five minutes while Application Auto Scaling logged a
    ``Failed`` activity every ten seconds: quota 4, three instances in use, and a
    policy asking for three more at once. Nothing surfaced on the endpoint, which
    stayed ``InService`` at one instance throughout.
    """

    def _scalable(self, aas_stub, sm_stub, *, max_capacity: int = 4, current: int = 1) -> None:
        _stub_capture(
            aas_stub,
            sm_stub,
            targets=[_target(max_capacity=max_capacity)],
            policies=["p1"],
            desired=current,
            current=current,
        )
        _stub_variant_instance_type(sm_stub)

    def test_raises_when_the_policys_jump_does_not_fit(self, appscaling, sagemaker, quotas) -> None:
        # The live case exactly: limit 4, three in use, max_capacity 4 from 1 instance.
        # A check for one instance would have passed this.
        aas, aas_stub = appscaling
        sm, sm_stub = sagemaker
        q, q_stub = quotas
        self._scalable(aas_stub, sm_stub)
        _stub_headroom(sm_stub, q_stub, limit=4.0)

        with pytest.raises(FixtureError, match="endpoint usage") as excinfo:
            require_scalable(ENDPOINT, appscaling=aas, sagemaker=sm, quotas=q)

        message = str(excinfo.value)
        assert "will ask for 3 more" in message
        assert ENDPOINT_USAGE_QUOTA_CODE in message
        # Names the holders, so the operator knows what to stop rather than guessing.
        for name in OTHER_ENDPOINTS:
            assert name in message

    def test_the_error_says_what_max_capacity_would_fit(
        self, appscaling, sagemaker, quotas
    ) -> None:
        # Raising the quota is one fix; lowering max_capacity is the other, and it is
        # the one an operator can apply without an AWS support ticket.
        aas, aas_stub = appscaling
        sm, sm_stub = sagemaker
        q, q_stub = quotas
        self._scalable(aas_stub, sm_stub)
        _stub_headroom(sm_stub, q_stub, limit=4.0)

        with pytest.raises(FixtureError, match="lower max_capacity to 2"):
            require_scalable(ENDPOINT, appscaling=aas, sagemaker=sm, quotas=q)

    def test_passes_when_the_whole_jump_fits(self, appscaling, sagemaker, quotas) -> None:
        aas, aas_stub = appscaling
        sm, sm_stub = sagemaker
        q, q_stub = quotas
        self._scalable(aas_stub, sm_stub)
        _stub_headroom(sm_stub, q_stub, limit=6.0)

        assert require_scalable(ENDPOINT, appscaling=aas, sagemaker=sm, quotas=q).max_capacity == 4

    def test_an_explicit_headroom_overrides_the_derived_jump(
        self, appscaling, sagemaker, quotas
    ) -> None:
        # A caller who knows a step policy adds one at a time asks for one.
        aas, aas_stub = appscaling
        sm, sm_stub = sagemaker
        q, q_stub = quotas
        self._scalable(aas_stub, sm_stub)
        _stub_headroom(sm_stub, q_stub, limit=4.0)

        state = require_scalable(
            ENDPOINT, appscaling=aas, sagemaker=sm, quotas=q, require_quota_headroom=1
        )
        assert state.max_capacity == 4

    def test_a_serverless_variant_skips_the_quota_read(self, appscaling, sagemaker, quotas) -> None:
        aas, aas_stub = appscaling
        sm, sm_stub = sagemaker
        q, q_stub = quotas
        _stub_capture(aas_stub, sm_stub, targets=[_target(max_capacity=4)], policies=["p1"])
        _stub_variant_instance_type(sm_stub, instance_type=None)

        require_scalable(ENDPOINT, appscaling=aas, sagemaker=sm, quotas=q)
        q_stub.assert_no_pending_responses()

    def test_an_unverifiable_quota_warns_and_proceeds(
        self, appscaling, sagemaker, quotas, logged
    ) -> None:
        # Refusing here would make a missing IAM permission fatal to a measurement that
        # would otherwise work. The warning names the command that shows the truth.
        aas, aas_stub = appscaling
        sm, sm_stub = sagemaker
        q, q_stub = quotas
        self._scalable(aas_stub, sm_stub)
        _stub_headroom(sm_stub, q_stub, limit=None)

        state = require_scalable(ENDPOINT, appscaling=aas, sagemaker=sm, quotas=q)

        assert state.max_capacity == 4
        assert any("describe-scaling-activities" in m for m in logged)

    def test_an_already_scaled_out_variant_needs_only_the_remainder(
        self, appscaling, sagemaker, quotas
    ) -> None:
        # At 3 of max 4, the policy can only ask for one more, so one slot is enough.
        aas, aas_stub = appscaling
        sm, sm_stub = sagemaker
        q, q_stub = quotas
        self._scalable(aas_stub, sm_stub, current=3)
        _stub_headroom(sm_stub, q_stub, limit=6.0, self_current=3)

        assert require_scalable(ENDPOINT, appscaling=aas, sagemaker=sm, quotas=q).max_capacity == 4


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
