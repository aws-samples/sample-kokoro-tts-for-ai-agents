"""Freeze autoscaling and pin instance count for the duration of a benchmark.

``C_max`` is a *per-instance* quantity. If the fleet grows mid-run, achieved
throughput rises for a reason unrelated to the latency knee, and the saturation
marker (achieved < 0.95 x offered) never trips — the run reports a ``C_max`` that
is really ``N x C_max``, with no error and no warning. That number then
propagates into ``C_target``, ``N_peak``, queue depth, and fleet cost.

Two live scaling policies in this account make that reachable today
(``speech-orpheus-3b`` has ``max_capacity=4``), so freezing is enforced rather
than advised: see ``require_frozen``.

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
    all (true for ``speech-kokoro-82m`` and ``speech-chatterbox-turbo`` today) —
    a freeze then has nothing to suspend and must not create one, since that
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
        pin_to: Instance count to hold for the run. ``1`` for ``C_max``, since
            the measurement is per-instance.

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

    A warning would be ignored and the resulting ``C_max`` would look normal, so
    this is the enforcement point behind ``cmax --require-frozen``.

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


def require_scalable(
    endpoint_name: str,
    *,
    region: str = "us-east-1",
    variant: str = DEFAULT_VARIANT,
    min_max_capacity: int = 2,
    appscaling: BaseClient | None = None,
    sagemaker: BaseClient | None = None,
) -> EndpointFixture:
    """Assert a scale event is actually possible. The inverse of :func:`require_frozen`.

    ``ttotal`` needs scaling live. Without this check it would drive load for
    twenty minutes waiting for an event that cannot happen.

    Raises:
        FixtureError: If no scalable target or policy exists, ``max_capacity`` is
            too low, or scale-out is suspended (e.g. left over from an aborted
            freeze).
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

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        if self._fixture is not None:
            thaw(
                self._fixture,
                region=self._region,
                appscaling=self._appscaling,
                sagemaker=self._sagemaker,
                restore_capacity=self._restore_capacity,
            )
            self._fixture = None
        return False  # never suppress
