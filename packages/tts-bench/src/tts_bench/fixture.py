"""Freeze autoscaling and pin instance count for the duration of a benchmark.

``Q_max`` is a *per-instance* quantity, and the ladder holds ``N`` requests
outstanding against whatever fleet is behind the endpoint. If the fleet grows
mid-run, those ``N`` requests spread across more instances, so each one queues
less and the ladder keeps passing the SLO at rungs a single instance could not
serve — the run reports a ``Q_max`` that is really ``N_instances x Q_max``, with
no error and no warning. Both scaling thresholds are fractions of that number,
and so is the container's admission bound.

Nothing in the measurement can detect it afterwards. Closed-loop holds ``N``
exactly by construction, which is what makes the ladder trustworthy, and is also
why a larger fleet shows up as *better latency* rather than as saturation. So
freezing is enforced rather than advised: see ``require_frozen``, and
``instance_counts_observed`` on the artifact as the after-the-fact check.

A live scaling policy in this account makes that reachable today
(``speech-kokoro-82m`` has ``max_capacity=9``).

The same freeze covers ``T_total``. That run forces a scale-out itself, and the
deployed policy would otherwise fire during it and add instances from a second,
untracked cause — suspension stops Application Auto Scaling but not our own
``UpdateEndpointWeightsAndCapacities``, which is exactly the arrangement wanted.

Freezing is deliberately non-destructive. ``RegisterScalableTarget`` requires
only ``ServiceNamespace``/``ResourceId``/``ScalableDimension``, so
``SuspendedState`` is set without sending ``MinCapacity``/``MaxCapacity`` — the
configured limits survive a freeze/thaw cycle untouched.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import botocore.exceptions
from botocore.client import BaseClient
from loguru import logger

SERVICE_NAMESPACE = "sagemaker"
SCALABLE_DIMENSION = "sagemaker:variant:DesiredInstanceCount"
DEFAULT_VARIANT = "primary"

#: All three flags, because scheduled actions can move capacity too.
SUSPEND_ALL = {
    "DynamicScalingInSuspended": True,
    "DynamicScalingOutSuspended": True,
    "ScheduledScalingSuspended": True,
}

_RESUME_ALL = {
    "DynamicScalingInSuspended": False,
    "DynamicScalingOutSuspended": False,
    "ScheduledScalingSuspended": False,
}

DEFAULT_PIN_TIMEOUT_S = 1800.0
DEFAULT_POLL_INTERVAL_S = 15.0


class FixtureError(RuntimeError):
    """A benchmark precondition could not be established or verified."""


def resource_id(endpoint_name: str, variant: str = DEFAULT_VARIANT) -> str:
    """Application Auto Scaling resource id for an endpoint variant."""
    return f"endpoint/{endpoint_name}/variant/{variant}"


@dataclass(frozen=True, slots=True)
class EndpointFixture:
    """Autoscaling and capacity state captured before a benchmark.

    Restoring from this is what makes a freeze safe to run against a live
    endpoint. ``suspended_state=None`` means no scalable target is registered at
    all -- true for any endpoint deployed with ``scaling_enabled=False`` -- a
    freeze then has nothing to suspend and must not create one, since that
    would leave configuration behind that CDK does not describe.
    """

    endpoint_name: str
    variant: str
    suspended_state: dict[str, bool] | None
    min_capacity: int | None
    max_capacity: int | None
    desired_instance_count: int
    current_instance_count: int
    policy_names: tuple[str, ...] = field(default=())

    @property
    def has_scalable_target(self) -> bool:
        return self.suspended_state is not None

    @property
    def scale_out_suspended(self) -> bool:
        """True when nothing can add instances — including "no target at all"."""
        if self.suspended_state is None:
            return True
        return bool(self.suspended_state.get("DynamicScalingOutSuspended", False))

    @property
    def resource_id(self) -> str:
        return resource_id(self.endpoint_name, self.variant)


def _make_clients(
    region: str,
    appscaling: BaseClient | None,
    sagemaker: BaseClient | None,
) -> tuple[BaseClient, BaseClient]:
    """Resolve clients, allowing injection for tests."""
    if appscaling is not None and sagemaker is not None:
        return appscaling, sagemaker
    import boto3

    return (
        appscaling or boto3.client("application-autoscaling", region_name=region),
        sagemaker or boto3.client("sagemaker", region_name=region),
    )


def capture(
    endpoint_name: str,
    *,
    region: str = "us-east-1",
    variant: str = DEFAULT_VARIANT,
    appscaling: BaseClient | None = None,
    sagemaker: BaseClient | None = None,
) -> EndpointFixture:
    """Read current autoscaling and capacity state. Makes no changes.

    Raises:
        FixtureError: If the endpoint does not exist or has no such variant.
    """
    aas, sm = _make_clients(region, appscaling, sagemaker)
    rid = resource_id(endpoint_name, variant)

    suspended: dict[str, bool] | None = None
    min_cap: int | None = None
    max_cap: int | None = None
    try:
        targets = aas.describe_scalable_targets(
            ServiceNamespace=SERVICE_NAMESPACE,
            ResourceIds=[rid],
            ScalableDimension=SCALABLE_DIMENSION,
        ).get("ScalableTargets", [])
    except botocore.exceptions.ClientError as exc:
        raise FixtureError(f"could not describe scalable targets for {rid}: {exc}") from exc

    if targets:
        target = targets[0]
        suspended = dict(target.get("SuspendedState") or {})
        min_cap = target.get("MinCapacity")
        max_cap = target.get("MaxCapacity")

    policy_names: tuple[str, ...] = ()
    if targets:
        try:
            policies = aas.describe_scaling_policies(
                ServiceNamespace=SERVICE_NAMESPACE,
                ResourceId=rid,
                ScalableDimension=SCALABLE_DIMENSION,
            ).get("ScalingPolicies", [])
            policy_names = tuple(p["PolicyName"] for p in policies)
        except botocore.exceptions.ClientError as exc:
            # Not fatal: policy names are used for reporting and for ttotal's
            # alarm lookup, not for the freeze itself.
            logger.warning("Could not list scaling policies for {}: {}", rid, exc)

    desired, current = _read_capacity(sm, endpoint_name, variant)

    return EndpointFixture(
        endpoint_name=endpoint_name,
        variant=variant,
        suspended_state=suspended,
        min_capacity=min_cap,
        max_capacity=max_cap,
        desired_instance_count=desired,
        current_instance_count=current,
        policy_names=policy_names,
    )


def _read_capacity(sagemaker: BaseClient, endpoint_name: str, variant: str) -> tuple[int, int]:
    """Return ``(desired, current)`` instance counts for a variant."""
    try:
        described = sagemaker.describe_endpoint(EndpointName=endpoint_name)
    except botocore.exceptions.ClientError as exc:
        raise FixtureError(f"could not describe endpoint {endpoint_name}: {exc}") from exc

    for summary in described.get("ProductionVariants", []):
        if summary.get("VariantName") == variant:
            return (
                int(summary.get("DesiredInstanceCount", 0)),
                int(summary.get("CurrentInstanceCount", 0)),
            )
    raise FixtureError(f"endpoint {endpoint_name} has no variant named {variant!r}")


def _set_suspended_state(
    appscaling: BaseClient,
    rid: str,
    state: dict[str, bool],
) -> bool:
    """Set ``SuspendedState`` without touching capacity limits.

    Deliberately omits ``MinCapacity``/``MaxCapacity``: they are optional on
    ``RegisterScalableTarget``, so leaving them out preserves whatever is
    configured. Sending them would risk a benchmark silently rewriting the
    endpoint's scaling limits.

    Returns:
        True if a target existed and was updated, False if none is registered.
    """
    try:
        appscaling.register_scalable_target(
            ServiceNamespace=SERVICE_NAMESPACE,
            ResourceId=rid,
            ScalableDimension=SCALABLE_DIMENSION,
            SuspendedState=state,
        )
        return True
    except botocore.exceptions.ClientError as exc:
        code = (exc.response.get("Error", {}) or {}).get("Code", "")
        if code in ("ObjectNotFoundException", "ValidationException"):
            # No scalable target for this variant: nothing can scale it, which
            # is the state we wanted anyway.
            logger.info("No scalable target on {} — nothing to suspend", rid)
            return False
        raise


def _pin_capacity(
    sagemaker: BaseClient,
    endpoint_name: str,
    variant: str,
    pin_to: int,
    *,
    timeout_s: float,
    poll_interval_s: float,
) -> None:
    """Set desired instance count and wait for the fleet to match.

    Waiting matters: a benchmark that starts while the endpoint is still
    resizing measures a fleet in transition.
    """
    desired, current = _read_capacity(sagemaker, endpoint_name, variant)
    if desired == pin_to and current == pin_to:
        return

    if desired != pin_to:
        logger.info(
            "Pinning {} variant {} from desired={} to {}",
            endpoint_name,
            variant,
            desired,
            pin_to,
        )
        try:
            sagemaker.update_endpoint_weights_and_capacities(
                EndpointName=endpoint_name,
                DesiredWeightsAndCapacities=[
                    {"VariantName": variant, "DesiredInstanceCount": pin_to}
                ],
            )
        except botocore.exceptions.ClientError as exc:
            raise FixtureError(
                f"could not pin {endpoint_name} variant {variant} to {pin_to}: {exc}"
            ) from exc

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        desired, current = _read_capacity(sagemaker, endpoint_name, variant)
        if desired == pin_to and current == pin_to:
            logger.info("{} is at {} instance(s)", endpoint_name, pin_to)
            return
        logger.debug(
            "Waiting for {}: desired={} current={} target={}",
            endpoint_name,
            desired,
            current,
            pin_to,
        )
        time.sleep(poll_interval_s)

    raise FixtureError(
        f"{endpoint_name} did not reach {pin_to} instance(s) within {timeout_s:.0f}s "
        f"(desired={desired}, current={current})"
    )


def freeze(
    endpoint_name: str,
    *,
    region: str = "us-east-1",
    variant: str = DEFAULT_VARIANT,
    pin_to: int = 1,
    timeout_s: float = DEFAULT_PIN_TIMEOUT_S,
    poll_interval_s: float = DEFAULT_POLL_INTERVAL_S,
    appscaling: BaseClient | None = None,
    sagemaker: BaseClient | None = None,
) -> EndpointFixture:
    """Suspend scaling, pin capacity, and verify. Returns state for :func:`thaw`.

    Args:
        pin_to: Instance count to hold for the run. ``1`` for ``Q_max``, since
            the measurement is per-instance, and ``1`` for ``T_total`` too — that
            run raises the count itself and needs a known starting point.

    Returns:
        The state captured *before* any change, suitable for :func:`thaw`.

    Raises:
        FixtureError: If capacity does not settle, or if verification finds
            scaling still able to add instances.
    """
    if pin_to < 1:
        raise ValueError(f"pin_to must be >= 1, got {pin_to}")

    aas, sm = _make_clients(region, appscaling, sagemaker)
    before = capture(endpoint_name, region=region, variant=variant, appscaling=aas, sagemaker=sm)

    if before.has_scalable_target:
        logger.info("Suspending autoscaling on {}", before.resource_id)
        _set_suspended_state(aas, before.resource_id, dict(SUSPEND_ALL))

    _pin_capacity(
        sm,
        endpoint_name,
        variant,
        pin_to,
        timeout_s=timeout_s,
        poll_interval_s=poll_interval_s,
    )

    after = capture(endpoint_name, region=region, variant=variant, appscaling=aas, sagemaker=sm)
    if not after.scale_out_suspended:
        raise FixtureError(
            f"{after.resource_id}: scale-out still active after suspend attempt; "
            "refusing to benchmark a fleet that can grow mid-run"
        )
    if after.current_instance_count != pin_to:
        raise FixtureError(
            f"{endpoint_name}: expected {pin_to} instance(s), found {after.current_instance_count}"
        )

    logger.info(
        "Frozen: {} at {} instance(s), scale-out suspended",
        endpoint_name,
        after.current_instance_count,
    )
    return before


def thaw(
    fixture: EndpointFixture,
    *,
    region: str = "us-east-1",
    appscaling: BaseClient | None = None,
    sagemaker: BaseClient | None = None,
    restore_capacity: bool = True,
) -> None:
    """Restore captured state. Idempotent, and never raises.

    Called from ``finally`` blocks and signal paths, so it swallows and logs
    rather than propagating: an exception here would mask the original failure
    and still leave the endpoint frozen. Idempotence means
    ``tts-bench thaw --endpoint ...`` can recover from a hard kill.
    """
    aas, sm = _make_clients(region, appscaling, sagemaker)

    if fixture.suspended_state is not None:
        target_state = {**_RESUME_ALL, **fixture.suspended_state}
        try:
            _set_suspended_state(aas, fixture.resource_id, target_state)
            logger.info("Restored SuspendedState on {}", fixture.resource_id)
        except botocore.exceptions.ClientError as exc:
            logger.error(
                "FAILED to restore autoscaling on {}: {}. Re-run "
                "`tts-bench thaw --endpoint {}` or restore manually.",
                fixture.resource_id,
                exc,
                fixture.endpoint_name,
            )

    if not restore_capacity:
        return

    try:
        desired, _ = _read_capacity(sm, fixture.endpoint_name, fixture.variant)
        if desired == fixture.desired_instance_count:
            return
        sm.update_endpoint_weights_and_capacities(
            EndpointName=fixture.endpoint_name,
            DesiredWeightsAndCapacities=[
                {
                    "VariantName": fixture.variant,
                    "DesiredInstanceCount": fixture.desired_instance_count,
                }
            ],
        )
        logger.info(
            "Restored {} to desired={} instance(s)",
            fixture.endpoint_name,
            fixture.desired_instance_count,
        )
    except (botocore.exceptions.ClientError, FixtureError) as exc:
        logger.error(
            "FAILED to restore capacity on {}: {}. Expected desired={}.",
            fixture.endpoint_name,
            exc,
            fixture.desired_instance_count,
        )


def require_frozen(
    endpoint_name: str,
    *,
    region: str = "us-east-1",
    variant: str = DEFAULT_VARIANT,
    expect_instances: int = 1,
    appscaling: BaseClient | None = None,
    sagemaker: BaseClient | None = None,
) -> EndpointFixture:
    """Assert the endpoint cannot grow mid-run. Raises rather than warns.

    A warning would be ignored and the resulting ``Q_max`` would look normal — a
    larger fleet reads as better latency, not as an error — so this is the
    enforcement point behind ``qmax --require-frozen``.

    Raises:
        FixtureError: If scale-out is active or the instance count is wrong.
    """
    state = capture(
        endpoint_name, region=region, variant=variant, appscaling=appscaling, sagemaker=sagemaker
    )
    problems: list[str] = []
    if not state.scale_out_suspended:
        problems.append("scale-out is not suspended")
    if state.current_instance_count != expect_instances:
        problems.append(
            f"current instance count is {state.current_instance_count}, expected {expect_instances}"
        )
    if state.desired_instance_count != expect_instances:
        problems.append(
            f"desired instance count is {state.desired_instance_count}, expected {expect_instances}"
        )
    if problems:
        raise FixtureError(
            f"{endpoint_name} is not safe for a per-instance measurement: "
            + "; ".join(problems)
            + ". Run with --no-require-frozen only if you accept a fleet-wide number."
        )
    return state


#: Container environment keys that bound how deep the admission queue may get. A
#: ``Q_max`` ladder is supposed to find the depth at which the *SLO* breaks, so any
#: of these being set means the container sheds first and the ladder measures the
#: bound instead. Named individually rather than pattern-matched: a preflight that
#: refuses on anything queue-shaped would block runs for knobs it does not
#: understand, and this list is checked against ``containers/*/serve.py``.
QUEUE_DEPTH_ENV_KEYS: tuple[str, ...] = ("MAX_QUEUE_DEPTH", "MAX_PENDING_REQUESTS")


def require_unbounded_queue(
    endpoint_name: str,
    *,
    region: str = "us-east-1",
    variant: str = DEFAULT_VARIANT,
    sagemaker: BaseClient | None = None,
    deployed: DeployedConfig | None = None,
) -> DeployedConfig:
    """Assert the container will queue rather than shed. Raises rather than warns.

    ``Q_max`` is *defined* as the concurrency at which p95 first-byte time crosses the
    SLO. A container with a depth bound stops accepting work before that point, so the
    ladder finds the bound and reports it as a capacity number — and it looks entirely
    normal, since the rejections land in ``outcome_counts`` rather than in the latency
    percentiles the pass/fail line reads.

    Machine-checked rather than remembered: ``DeployedConfig`` already reads
    ``container_env`` off ``DescribeModel``, so the same call that fingerprints the run
    answers this.

    This is the *measurement* precondition, and it is the opposite of what we want in
    production — enforcing ``Q_max`` at the instance is the point of measuring it. The
    order matters: measure unbounded, then deploy the bound.

    Args:
        deployed: A fingerprint already read for this run, to avoid a second round of
            three ``describe_*`` calls. Read fresh when omitted.

    Returns:
        The configuration checked, so a caller can record it.

    Raises:
        FixtureError: If any key in :data:`QUEUE_DEPTH_ENV_KEYS` is set to a non-zero
            value. ``0`` and unset both pass — a container reading ``0`` as "no bound"
            is the convention in ``streaming_proxy.py``, and refusing it would block a
            valid run.
    """
    config = deployed or describe_deployed_config(
        endpoint_name, region=region, variant=variant, sagemaker=sagemaker
    )

    bounds: list[str] = []
    for key in QUEUE_DEPTH_ENV_KEYS:
        raw = config.container_env.get(key)
        if raw is None:
            continue
        try:
            value = int(str(raw).strip())
        except ValueError:
            # Unparseable is not a pass. The container's own parse may well succeed
            # where ours did not, and guessing which way it went is how a bounded
            # queue gets measured as an unbounded one.
            bounds.append(f"{key}={raw!r} (unparseable, so its effect is unknown)")
            continue
        if value != 0:
            bounds.append(f"{key}={value}")

    if bounds:
        raise FixtureError(
            f"{endpoint_name} bounds its admission queue ({', '.join(bounds)}), so a "
            "Q_max ladder would measure that bound rather than the concurrency at which "
            "the SLO breaks: the container starts refusing work before latency ever "
            "crosses the line. Measure Q_max against an unbounded queue first, then "
            "deploy the bound. Run with --no-require-unbounded-queue to record the "
            "number anyway, marked as measured against a bounded queue."
        )
    return config


#: Service Quotas code for "ml.<type> for endpoint usage". One quota per instance
#: type, account-wide and per-region, counting *every* endpoint — so an unrelated
#: team's endpoint consumes the same allowance ours scales into.
ENDPOINT_USAGE_QUOTA_CODE = "L-1928E07B"


@dataclass(frozen=True, slots=True)
class QuotaHeadroom:
    """Account-level room to add instances of one type.

    Application Auto Scaling does not consult this before acting: it raises
    ``DesiredInstanceCount``, SageMaker rejects the change with
    ``ResourceLimitExceeded``, and the activity is recorded as ``Failed`` while the
    policy retries every ten seconds. Nothing surfaces on the endpoint, whose status
    stays ``InService`` at its old count — so a scale-out that cannot happen looks
    exactly like one that has not happened yet.
    """

    instance_type: str
    limit: int | None
    """``None`` when the quota could not be read — absence of evidence, so callers
    must not treat it as headroom."""

    in_use: int
    """Instances of this type across every endpoint in the account and region.

    Counts ``max(current, desired)`` per variant: a scale-out already in flight has
    claimed its slots even though the instances are not running yet."""

    endpoints_in_use: tuple[str, ...] = field(default=())
    """Which endpoints hold them, so the operator knows what to stop or move."""

    @property
    def available(self) -> int | None:
        return None if self.limit is None else max(self.limit - self.in_use, 0)

    def room_for(self, added: int) -> bool | None:
        """Whether ``added`` more instances would fit. ``None`` when unknown."""
        room = self.available
        return None if room is None else room >= added


def _endpoint_instance_types(sagemaker: BaseClient) -> dict[str, tuple[str, int]]:
    """``{endpoint/variant: (instance_type, instances_held)}`` per live variant.

    ``DescribeEndpoint`` does not return the instance type — only the config does —
    so this is two calls per endpoint. Worth it: the quota is account-wide, and
    without the other endpoints' consumption a headroom figure is meaningless.

    ``instances_held`` is ``max(current, desired)``, because a variant mid-scale-out
    has already claimed the slots its desired count names.
    """
    out: dict[str, tuple[str, int]] = {}
    try:
        paginator = sagemaker.get_paginator("list_endpoints")
        for page in paginator.paginate():
            for summary in page.get("Endpoints", []):
                name = summary.get("EndpointName")
                if not name or summary.get("EndpointStatus") == "Failed":
                    # A failed endpoint holds no instances.
                    continue
                try:
                    described = sagemaker.describe_endpoint(EndpointName=name)
                    config = sagemaker.describe_endpoint_config(
                        EndpointConfigName=described["EndpointConfigName"]
                    )
                except botocore.exceptions.ClientError as exc:
                    logger.warning("Could not read instance type for endpoint {}: {}", name, exc)
                    continue
                counts = {
                    v.get("VariantName"): max(
                        int(v.get("CurrentInstanceCount", 0)),
                        int(v.get("DesiredInstanceCount", 0)),
                    )
                    for v in described.get("ProductionVariants", [])
                }
                for variant in config.get("ProductionVariants", []):
                    instance_type = variant.get("InstanceType")
                    if not instance_type:
                        # Serverless or async variant: no instances, no quota use.
                        continue
                    out[f"{name}/{variant.get('VariantName')}"] = (
                        instance_type,
                        counts.get(variant.get("VariantName"), 0),
                    )
    except botocore.exceptions.ClientError as exc:
        logger.warning("Could not list endpoints to total quota usage: {}", exc)
    return out


def endpoint_quota_headroom(
    instance_type: str,
    *,
    region: str = "us-east-1",
    quotas: BaseClient | None = None,
    sagemaker: BaseClient | None = None,
) -> QuotaHeadroom:
    """Read the account's remaining room for instances of one type.

    Read-only. Both halves degrade to a warning rather than raising, because a
    missing ``servicequotas:ListServiceQuotas`` permission must not be the thing
    that stops a measurement — an unknown limit is reported as unknown.
    """
    import boto3

    quotas = quotas or boto3.client("service-quotas", region_name=region)
    sm = sagemaker or boto3.client("sagemaker", region_name=region)

    limit: int | None = None
    wanted = f"{instance_type} for endpoint usage"
    try:
        paginator = quotas.get_paginator("list_service_quotas")
        for page in paginator.paginate(
            ServiceCode="sagemaker", QuotaCode=ENDPOINT_USAGE_QUOTA_CODE
        ):
            for quota in page.get("Quotas", []):
                if quota.get("QuotaName") == wanted and quota.get("Value") is not None:
                    limit = int(quota["Value"])
    except botocore.exceptions.ClientError as exc:
        logger.warning("Could not read the {} quota: {}", wanted, exc)

    holders = {
        key: count
        for key, (found_type, count) in _endpoint_instance_types(sm).items()
        if found_type == instance_type and count > 0
    }
    return QuotaHeadroom(
        instance_type=instance_type,
        limit=limit,
        in_use=sum(holders.values()),
        endpoints_in_use=tuple(sorted(holders)),
    )


def variant_instance_type(
    endpoint_name: str,
    *,
    region: str = "us-east-1",
    variant: str = DEFAULT_VARIANT,
    sagemaker: BaseClient | None = None,
) -> str | None:
    """The instance type one variant runs on, or ``None`` if it has none.

    ``DescribeEndpoint`` omits it, so this reads the endpoint config. ``None`` means
    serverless or async, which consume no instance quota.
    """
    import boto3

    sm = sagemaker or boto3.client("sagemaker", region_name=region)
    try:
        described = sm.describe_endpoint(EndpointName=endpoint_name)
        config = sm.describe_endpoint_config(EndpointConfigName=described["EndpointConfigName"])
    except botocore.exceptions.ClientError as exc:
        raise FixtureError(
            f"could not read the endpoint config for {endpoint_name}: {exc}"
        ) from exc

    for entry in config.get("ProductionVariants", []):
        if entry.get("VariantName") == variant:
            instance_type: str | None = entry.get("InstanceType")
            return instance_type
    return None


#: Environment keys SageMaker or the CDK stack sets on every container, so they carry
#: no information about *this* configuration. Excluded from the fingerprint: including
#: them would make the endpoint name part of the identity, and then an artifact could
#: never be compared across two endpoints running the same build.
_AMBIENT_ENV_KEYS = frozenset({"ENDPOINT_NAME", "SM_MODEL_ID"})

#: Number of image-tag characters in :attr:`DeployedConfig.slug`. The CDK asset tag is a
#: 64-char SHA-256; 8 hex chars is 4 billion values, which is plenty to tell apart the
#: handful of builds one model ever has, and short enough to read in a filename.
_SLUG_DIGEST_CHARS = 8


@dataclass(frozen=True, slots=True)
class DeployedConfig:
    """What a benchmark was actually measured against.

    ``Q_max``, ``S`` and ``T_total`` are properties of a *configuration*, not of a
    model. Three things move them, and all three are read here:

    * ``instance_type`` — the GPU. Also the GPU *count*: a container driving four
      GPUs runs on a different instance type, so multi-GPU serving is covered by
      this field rather than needing one of its own.
    * ``image_digest`` — the serving code. CDK tags container assets by a content
      hash of the build context, so this changes exactly when the container does,
      which is what makes an admission queue or a batching change visible here.
    * ``container_env`` — the knobs. ``MAX_REQUEST_AGE_S`` and ``MAX_QUEUE_DEPTH``
      are set this way, and either would bound the wait without touching the image.

    Recorded on every artifact and checked before one is replayed; see
    :meth:`assert_matches`.
    """

    instance_type: str | None
    image_digest: str | None
    container_env: dict[str, str] = field(default_factory=dict)

    @property
    def slug(self) -> str:
        """Short, stable, filename-safe identifier, e.g. ``g6xl-139b9068``.

        Used to default one artifact path per configuration, so measuring a second
        configuration cannot overwrite the first by forgetting ``--output``.
        """
        family = (self.instance_type or "unknown").removeprefix("ml.").replace(".", "")
        digest = (self.image_digest or "nodigest")[:_SLUG_DIGEST_CHARS]
        return f"{family}-{digest}"

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready form, for embedding in an artifact."""
        return {
            "instance_type": self.instance_type,
            "image_digest": self.image_digest,
            "container_env": dict(self.container_env),
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> DeployedConfig:
        """Rebuild from an artifact. Missing keys become ``None``/empty, not errors."""
        env = raw.get("container_env")
        return cls(
            instance_type=raw.get("instance_type"),
            image_digest=raw.get("image_digest"),
            container_env=dict(env) if isinstance(env, dict) else {},
        )

    def differences(self, other: DeployedConfig) -> list[str]:
        """Human-readable field-by-field differences, empty when the two agree.

        ``self`` is the recorded configuration, ``other`` the live one.
        """
        out: list[str] = []
        if self.instance_type != other.instance_type:
            out.append(
                f"instance_type: artifact {self.instance_type!r}, deployed "
                f"{other.instance_type!r} — S and TTFAB are properties of the GPU, and "
                "Q_max is W_max/S, so every number moves"
            )
        if self.image_digest != other.image_digest:
            out.append(
                f"image_digest: artifact {_short(self.image_digest)}, deployed "
                f"{_short(other.image_digest)} — the serving code differs, which can move "
                "Q_max on identical hardware"
            )
        if self.container_env != other.container_env:
            out.append(
                f"container_env: artifact {self.container_env}, deployed "
                f"{other.container_env} — these knobs bound queueing and request age"
            )
        return out

    def assert_matches(
        self,
        other: DeployedConfig,
        *,
        allow_mismatch: bool = False,
        artifact_label: str = "the measured artifact",
    ) -> None:
        """Refuse to reuse a measurement taken against a different configuration.

        A hard error rather than a warning: every downstream number — the two scaling
        thresholds, the fleet size, the queue depth, the cost — is derived from inputs
        that only hold for the configuration they were measured on, and a warning
        scrolls past.

        Args:
            other: The live configuration, from :func:`describe_deployed_config`.
            allow_mismatch: Downgrade to a warning. For the case where the operator
                knows the difference is irrelevant to what they are measuring.
            artifact_label: Named in the message, so the operator knows which file.

        Raises:
            FixtureError: If any field differs and ``allow_mismatch`` is false.
        """
        diffs = self.differences(other)
        if not diffs:
            return
        detail = "; ".join(diffs)
        if allow_mismatch:
            logger.warning(
                "{} was measured on a different configuration ({}). Proceeding because "
                "the mismatch was explicitly allowed.",
                artifact_label,
                detail,
            )
            return
        raise FixtureError(
            f"{artifact_label} was measured on a different configuration than the one "
            f"deployed now. {detail}. Re-measure against the deployed configuration, or "
            "pass --allow-config-mismatch to proceed anyway."
        )


def _short(digest: str | None) -> str:
    """A digest trimmed for a message. Full hashes make the diff unreadable."""
    if not digest:
        return "none"
    return digest[:12] if len(digest) > 12 else digest


def describe_deployed_config(
    endpoint_name: str,
    *,
    region: str = "us-east-1",
    variant: str = DEFAULT_VARIANT,
    sagemaker: BaseClient | None = None,
) -> DeployedConfig:
    """Read the live configuration fingerprint of one endpoint variant.

    Three calls, because SageMaker splits the answer three ways: the endpoint names
    its config, the config names the instance type and the model, and the model names
    the image and its environment.

    Raises:
        FixtureError: If the endpoint, its config, or its model cannot be read. This
            is deliberately fatal — a run that cannot identify what it is measuring
            produces an artifact nobody can trust later.
    """
    import boto3

    sm = sagemaker or boto3.client("sagemaker", region_name=region)
    try:
        described = sm.describe_endpoint(EndpointName=endpoint_name)
        config = sm.describe_endpoint_config(EndpointConfigName=described["EndpointConfigName"])
    except botocore.exceptions.ClientError as exc:
        raise FixtureError(
            f"could not read the endpoint config for {endpoint_name}: {exc}"
        ) from exc

    entry: dict[str, Any] = {}
    for candidate in config.get("ProductionVariants", []):
        if candidate.get("VariantName") == variant:
            entry = candidate
            break
    else:
        raise FixtureError(
            f"{endpoint_name} has no variant named {variant!r}, so there is no "
            "configuration to fingerprint"
        )

    model_name = entry.get("ModelName")
    image, env = None, {}
    if model_name:
        try:
            model = sm.describe_model(ModelName=model_name)
        except botocore.exceptions.ClientError as exc:
            raise FixtureError(
                f"could not read model {model_name} behind {endpoint_name}: {exc}"
            ) from exc
        container = model.get("PrimaryContainer") or {}
        image = container.get("Image")
        env = {
            key: value
            for key, value in (container.get("Environment") or {}).items()
            # AWS_* are injected by the runtime, and the ambient keys are set from the
            # endpoint's own name — neither describes the configuration under test.
            if not key.startswith("AWS_") and key not in _AMBIENT_ENV_KEYS
        }

    return DeployedConfig(
        instance_type=entry.get("InstanceType"),
        image_digest=_image_digest(image),
        container_env=env,
    )


def registry_instance_type(model: str) -> str:
    """Instance type from the benchmark registry, not from ``speech_infra``.

    ``cost.MODEL_INSTANCE_TYPES`` is kept in step with ``TTS_MODEL_CONFIGS`` by a
    consistency test, so this avoids pulling ``aws-cdk-lib`` in for one lookup.

    Only a fallback for :func:`fingerprint_or_registry`. A registry is a statement of
    what *should* be deployed, and a benchmark has to record what *is*.
    """
    from tts_bench.cost import DEFAULT_INSTANCE_TYPE, MODEL_INSTANCE_TYPES

    instance_type = MODEL_INSTANCE_TYPES.get(str(model))
    if instance_type is None:
        logger.warning(
            "{} is not in MODEL_INSTANCE_TYPES; costing against {}",
            model,
            DEFAULT_INSTANCE_TYPE,
        )
        return DEFAULT_INSTANCE_TYPE
    return instance_type


def fingerprint_or_registry(
    model: str,
    *,
    endpoint: str,
    region: str = "us-east-1",
    variant: str = DEFAULT_VARIANT,
    sagemaker: BaseClient | None = None,
) -> DeployedConfig:
    """The live configuration under test, falling back to the registry.

    Read from the endpoint rather than from ``MODEL_INSTANCE_TYPES`` because the
    registry can be stale in exactly the situation this harness exists to support:
    redeploy on a new instance type, re-measure, and the static dict would stamp the
    fresh artifact with the *old* type. That artifact then passes every downstream
    check while describing hardware it was never measured on.

    A disagreement between the two is logged at ERROR — it means the registry and
    the account have diverged, which also makes ``tts-bench drift`` and the cost
    model wrong, not just this artifact.

    Falls back to a registry-only fingerprint if the read fails: an unreadable
    endpoint should not lose a whole measurement run, and the resulting artifact is
    still honest, since a fallback fingerprint has no image digest and so can never
    silently compare equal to a real one.

    Shared by ``qmax`` and ``ttotal`` rather than owned by either, because a ``Q_max``
    and a ``T_total`` are both properties of a configuration and the planner refuses to
    combine two artifacts that disagree about which one.
    """
    registry_type = registry_instance_type(model)
    try:
        deployed = describe_deployed_config(
            endpoint, region=region, variant=variant, sagemaker=sagemaker
        )
    except FixtureError as exc:
        logger.warning(
            "Could not read the deployed configuration of {}: {}. Falling back to the "
            "registry type {}; this artifact will record no image digest and will not "
            "compare equal to one measured against a known configuration.",
            endpoint,
            exc,
            registry_type,
        )
        return DeployedConfig(instance_type=registry_type, image_digest=None)

    if deployed.instance_type and deployed.instance_type != registry_type:
        logger.error(
            "{} is deployed on {} but cost.MODEL_INSTANCE_TYPES says {}. Measuring "
            "against the deployed type; update the registry, or drift and the cost "
            "model will keep pricing this model wrong.",
            endpoint,
            deployed.instance_type,
            registry_type,
        )
    return deployed


def _image_digest(image: str | None) -> str | None:
    """The tag or digest off an ECR image URI, without the repository path.

    CDK publishes container assets tagged with a content hash of the build context,
    so the tag alone identifies the build. Keeping only the tag means an artifact
    stays comparable after an ECR repository is renamed or the account changes.
    """
    if not image:
        return None
    # A digest reference (@sha256:...) wins over a tag when both could appear.
    if "@" in image:
        return image.rsplit("@", 1)[1].removeprefix("sha256:")
    tail = image.rsplit("/", 1)[-1]
    return tail.rsplit(":", 1)[1] if ":" in tail else None


def require_scalable(
    endpoint_name: str,
    *,
    region: str = "us-east-1",
    variant: str = DEFAULT_VARIANT,
    min_max_capacity: int = 2,
    require_quota_headroom: int | None = None,
    appscaling: BaseClient | None = None,
    sagemaker: BaseClient | None = None,
    quotas: BaseClient | None = None,
) -> EndpointFixture:
    """Assert a scale event is actually possible. The inverse of :func:`require_frozen`.

    **Currently has no caller.** ``ttotal`` was its only one and deliberately stopped:
    the ``force-desired`` trigger raises ``DesiredInstanceCount`` itself under
    :func:`freeze`, so a live policy is not a precondition for it but a *confound* — the
    deployed ``> 0.713`` alarm firing mid-run is the best explanation for the 1->4 jump
    on 2026-07-31. Every check above the quota block therefore demands the very thing
    that mode suppresses.

    The quota block below is the half that still earns its keep, and it is reachable on
    its own through :func:`endpoint_quota_headroom`: a forced 1->2 that SageMaker refuses
    with ``ResourceLimitExceeded`` leaves the endpoint ``InService`` at its old count and
    surfaces nothing, so a run without that preflight waits out its whole timeout for an
    instance that was never coming.

    Args:
        require_quota_headroom: Instances the account must be able to add. ``None``
            derives it from ``max_capacity - current`` — the jump a target-tracking
            policy actually requests when the metric overshoots, which it does by
            design here. Checking only for 1 would have passed the case that motivated
            this guard: quota 4, three in use, and a policy that asked for three more
            at once. ``0`` skips the read.

    Raises:
        FixtureError: If no scalable target or policy exists, ``max_capacity`` is
            too low, scale-out is suspended (e.g. left over from an aborted freeze),
            or the account has no room for the instances the policy will request.
    """
    state = capture(
        endpoint_name, region=region, variant=variant, appscaling=appscaling, sagemaker=sagemaker
    )
    if not state.has_scalable_target:
        raise FixtureError(
            f"{endpoint_name} has no scalable target; deploy autoscaling before measuring T_total"
        )
    if state.suspended_state and state.suspended_state.get("DynamicScalingOutSuspended"):
        raise FixtureError(
            f"{state.resource_id}: scale-out is suspended, so no scale event can occur. "
            f"Run `tts-bench thaw --endpoint {endpoint_name}` first."
        )
    if (state.max_capacity or 0) < min_max_capacity:
        raise FixtureError(
            f"{state.resource_id}: max_capacity={state.max_capacity}, need at least "
            f"{min_max_capacity} to observe a scale-out"
        )
    if not state.policy_names:
        raise FixtureError(
            f"{state.resource_id}: no scaling policies attached, so nothing will "
            "trigger a scale event"
        )

    # The delta a target-tracking policy asks for, not +1. Its activities read "Setting
    # desired instance count to <max_capacity>" when the metric overshoots the target,
    # and SageMaker refuses the *whole* request rather than granting part of it — so the
    # room needed is the jump, not one instance.
    needed = (
        max((state.max_capacity or 0) - state.current_instance_count, 1)
        if require_quota_headroom is None
        else require_quota_headroom
    )
    if needed > 0:
        # The failure this catches is invisible from the endpoint: the policy fires,
        # SageMaker refuses on quota, the activity is logged Failed, and the endpoint
        # stays InService at its old count while the policy retries every 10s. Load
        # would run to its timeout against a scale-out that cannot occur.
        instance_type = variant_instance_type(
            endpoint_name, region=region, variant=variant, sagemaker=sagemaker
        )
        if instance_type is not None:
            headroom = endpoint_quota_headroom(
                instance_type, region=region, quotas=quotas, sagemaker=sagemaker
            )
            fits = headroom.room_for(needed)
            if fits is False:
                raise FixtureError(
                    f"{state.resource_id}: the account-level quota '{instance_type} for "
                    f"endpoint usage' is {headroom.limit} in {region}, with "
                    f"{headroom.in_use} already in use by "
                    f"{', '.join(headroom.endpoints_in_use)}. The policy will ask for "
                    f"{needed} more (max_capacity={state.max_capacity} from "
                    f"{state.current_instance_count}) and SageMaker will reject the whole "
                    "request with ResourceLimitExceeded — the endpoint stays InService at "
                    "its current count and nothing surfaces there. Request an increase for "
                    f"quota {ENDPOINT_USAGE_QUOTA_CODE}, delete an endpoint holding one, or "
                    f"lower max_capacity to {state.current_instance_count + (headroom.available or 0)}."
                )
            if fits is None:
                logger.warning(
                    "Could not read the '{} for endpoint usage' quota, so headroom for {} "
                    "more instance(s) is unverified. If the policy fires and no instance "
                    "appears, check `aws application-autoscaling "
                    "describe-scaling-activities` for ResourceLimitExceeded.",
                    instance_type,
                    needed,
                )
    return state


class frozen:  # noqa: N801 - context manager used as `with frozen(...)`
    """Context manager that freezes on enter and always restores on exit.

    Written as a class rather than ``@contextmanager`` so ``__exit__`` runs for
    ``KeyboardInterrupt`` and ``SystemExit`` too — a benchmark aborted with
    Ctrl-C must not leave production frozen.

    Example:
        >>> with frozen("speech-kokoro-82m", region="us-east-1") as state:  # doctest: +SKIP
        ...     run_steps()
    """

    def __init__(
        self,
        endpoint_name: str,
        *,
        region: str = "us-east-1",
        variant: str = DEFAULT_VARIANT,
        pin_to: int = 1,
        timeout_s: float = DEFAULT_PIN_TIMEOUT_S,
        poll_interval_s: float = DEFAULT_POLL_INTERVAL_S,
        appscaling: BaseClient | None = None,
        sagemaker: BaseClient | None = None,
        restore_capacity: bool = True,
    ) -> None:
        self._endpoint_name = endpoint_name
        self._region = region
        self._variant = variant
        self._pin_to = pin_to
        self._timeout_s = timeout_s
        self._poll_interval_s = poll_interval_s
        self._appscaling = appscaling
        self._sagemaker = sagemaker
        self._restore_capacity = restore_capacity
        self._fixture: EndpointFixture | None = None

    def __enter__(self) -> EndpointFixture:
        self._fixture = freeze(
            self._endpoint_name,
            region=self._region,
            variant=self._variant,
            pin_to=self._pin_to,
            timeout_s=self._timeout_s,
            poll_interval_s=self._poll_interval_s,
            appscaling=self._appscaling,
            sagemaker=self._sagemaker,
        )
        return self._fixture

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        """Returns None rather than False so the type states it never suppresses."""
        if self._fixture is not None:
            thaw(
                self._fixture,
                region=self._region,
                appscaling=self._appscaling,
                sagemaker=self._sagemaker,
                restore_capacity=self._restore_capacity,
            )
            self._fixture = None
