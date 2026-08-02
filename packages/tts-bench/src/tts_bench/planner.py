"""Turn measurements into a scaling configuration.

The step that closes the loop. ``cmax`` measures ``C_max`` and ``S``, ``ttotal``
measures the scaling lag stage by stage, and this module composes them with a stated
scenario into the four numbers ``ModelEndpointConfig`` actually needs:
``scaling_target_value``, ``queue_max_depth``, ``min_instances``, ``max_instances``.

No new math lives here. Every equation is in :mod:`shared.capacity`, which is
modality-neutral and unit-tested without AWS in scope; this module's job is
composition, provenance, and refusing to compose things that should not be combined.

**The provision stage is swept, not assumed.** ``T_total`` is not one portable number.
Its container stages are properties of the image and transfer between accounts; its
EC2-provision stage is a property of *this* account's spare capacity, and the customer
account uses reserved capacity where placement is guaranteed and faster. So the planner
takes the measured stages, subtracts the provision stage, and re-adds a *stated* one —
once per assumption — producing a plan per assumed provision time. See
:func:`plan_sweep`. A single-number plan would be precisely wrong for the account it
is meant to configure.

**Two refusals, both hard.** A ``C_max`` curve and a ``T_total`` lag measured on
different configurations cannot be combined (:func:`assert_pairable`) — that is what
the fingerprint is for. And a queueing budget that does not fit inside SageMaker's 60s
invocation ceiling is ``INFEASIBLE`` rather than a warning, because the requests it
admits wait the full ``W_max`` and then fail anyway.

**``W_max`` is derived, never stated.** The SLO is end-to-end — a request must reach
first byte within ``ttfab_slo_ms`` *including* its time in the queue — so the queueing
budget is ``SLO - S_p95`` and nothing else. It used to be a hand-set ``Scenario`` field
sitting beside the latency budget with no relation between them, which is how kokoro
came to be deployed with a 20 s queue allowance under a 300 ms budget. Two independent
numbers can disagree with the promise; one derived number cannot.

**``C_max`` is whichever of two limits binds.** ``cmax`` measures a latency knee and a
throughput ceiling, and an instance is bound by the one it reaches first.
:meth:`Measured.binding_c_max` takes the lower and names it; the findings distinguish a
bracketed answer from a lower bound, because those want different follow-up runs.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from loguru import logger

from shared.capacity import (
    CLOUDWATCH_HIGH_RES_PERIOD_S,
    SAGEMAKER_INVOCATION_CEILING_S,
    c_slo_cap,
    c_target,
    effective_c_target,
    effective_headroom_lag_s,
    fits_invocation_ceiling,
    lambda_cap_per_instance,
    max_added_wait_under_ceiling,
    n_instances,
    n_instances_from_streams,
    q_per_instance,
    queue_covers_surge,
    slo_is_feasible,
    utilization_at_k,
    w_absorbed,
    w_max_for_slo,
)
from tts_bench.types import (
    Finding,
    Measured,
    Origin,
    Provenance,
    ScalingPlan,
    Scenario,
    Verdict,
)

if TYPE_CHECKING:
    from tts_bench.types import CMaxReport

#: Stages whose duration belongs to AWS provisioning an EC2 instance, from the
#: scaling activity starting to the container's log stream opening. The one stage
#: reserved capacity changes, and so the one this module sweeps rather than trusts.
PROVISION_FROM_STAGE = "activity_started"
PROVISION_TO_STAGE = "instance_logging"

#: Provision times to sweep when none are given, seconds. Spans "reserved capacity,
#: warm" through "on-demand, contended" — the range within which the answer for the
#: customer account is expected to fall, without claiming to know where.
DEFAULT_PROVISION_SWEEP_S: tuple[float, ...] = (60.0, 120.0, 300.0, 600.0)

#: Growth factors to sweep. ``k`` is the one input that cannot be measured without
#: production traffic, so showing the config's sensitivity to it is more honest than
#: printing one row and letting the reader assume it was derived.
DEFAULT_K_SWEEP: tuple[float, ...] = (1.0, 2.0, 3.0, 5.0)

#: Cost multiple above the k=1 fleet at which the surge reserve stops being the
#: cheapest answer. Past this, shortening T_total or pre-warming beats buying idle
#: instances, and the plan says so rather than quietly costing 4x.
FLEET_COST_WARN_MULTIPLE = 3.0

#: Curve spread above which the ladder is measuring noise. Matches the threshold
#: documented on ``CMaxReport.curve_spread``.
CURVE_SPREAD_WARN = 0.2


class PlannerError(RuntimeError):
    """Inputs cannot be combined into a plan."""


@dataclass(frozen=True, slots=True)
class TTotalStages:
    """A measured scaling lag, split into the part that transfers and the part that does not.

    Read from a ``ttotal`` artifact rather than recomputed, because the artifact is
    the record of what was observed — including which stages were never seen.

    ``provision_s`` is separated out for one reason: it is the only stage whose value
    in the customer's reserved-capacity account is knowably different from ours. Ours
    is a measurement of EC2 spare capacity in us-east-1 on one afternoon; theirs is a
    property of a contract. Everything else — detection, alarm, image pull, weights,
    framework init, warm-up, recovery — is a property of the configuration and carries
    across.
    """

    total_s: float | None
    """Load applied through to traffic recovered, as measured. ``None`` when the run
    never observed enough stages to span it."""

    provision_s: float | None
    """``activity_started`` -> ``instance_logging``. ``None`` when either boundary was
    not observed, in which case there is nothing to substitute and
    :meth:`with_provision_s` says so."""

    bounded: bool = False
    """Whether ``total_s`` rests on an inferred endpoint. A bounded total that stops at
    ``in_service`` *under*-reports the lag, which is the dangerous direction."""

    trigger: str = ""
    """``drive-load`` or ``force-desired``. A ``force-desired`` run skips the metric and
    alarm stages entirely, so its total is not a ``T_total`` at all."""

    config_slug: str = ""
    run_id: str = ""
    missing_stages: tuple[str, ...] = ()

    @property
    def transferable_s(self) -> float | None:
        """Measured lag with the provision stage removed, seconds.

        The portable half. ``None`` when the total is unknown; equal to the total when
        the provision stage was not observed, since removing an unknown is not a
        subtraction we can do.
        """
        if self.total_s is None:
            return None
        if self.provision_s is None:
            return self.total_s
        return max(0.0, self.total_s - self.provision_s)

    @property
    def provision_measured(self) -> bool:
        return self.provision_s is not None

    def with_provision_s(self, provision_s: float) -> float:
        """The lag to plan against, substituting a stated provision time.

        Raises:
            PlannerError: If no measured total exists to substitute into. Callers that
                have no ``T_total`` at all should sweep an assumed *total* instead —
                see :func:`plan_sweep`'s ``assume_total`` path.
        """
        if provision_s < 0:
            raise ValueError(f"provision_s must be non-negative, got {provision_s}")
        transferable = self.transferable_s
        if transferable is None:
            raise PlannerError(
                "no measured T_total to substitute a provision time into. The ttotal "
                "artifact observed too few stages to span a total; re-run `tts-bench "
                "ttotal`, or plan against an assumed total with --assume-t-total."
            )
        return transferable + provision_s

    @classmethod
    def from_artifact(cls, raw: dict[str, Any]) -> TTotalStages:
        """Read from a ``ttotal`` artifact's ``to_dict`` form.

        Tolerant of missing keys throughout: an artifact from a run that observed only
        half the timeline is still the best information available, and refusing to
        parse it would send the operator back to a measurement that may not be
        repeatable in this account.
        """
        durations = raw.get("durations")
        provision_s: float | None = None
        if isinstance(durations, list):
            for entry in durations:
                if not isinstance(entry, dict):
                    continue
                if (
                    entry.get("from") == PROVISION_FROM_STAGE
                    and entry.get("to") == PROVISION_TO_STAGE
                ):
                    seconds = entry.get("seconds")
                    if isinstance(seconds, int | float):
                        provision_s = float(seconds)
                    break

        total = raw.get("t_total_s")
        missing = raw.get("missing_stages")
        return cls(
            total_s=float(total) if isinstance(total, int | float) else None,
            provision_s=provision_s,
            bounded=bool(raw.get("t_total_bounded", False)),
            trigger=str(raw.get("trigger", "")),
            config_slug=str(raw.get("config_slug", "")),
            run_id=str(raw.get("run_id", "")),
            missing_stages=tuple(str(s) for s in missing) if isinstance(missing, list) else (),
        )


def assert_pairable(
    measured: CMaxReport | Measured,
    stages: TTotalStages,
    *,
    allow_mismatch: bool = False,
) -> None:
    """Refuse to pair a ``C_max`` curve with a ``T_total`` lag from another configuration.

    The whole point of the fingerprint. Pairing a g5 curve with a g6 lag produces a
    plan for a fleet that exists nowhere: the curve sizes instances of one type while
    the lag describes how fast a different type boots, and nothing about the resulting
    numbers looks wrong.

    An empty slug on either side counts as a mismatch rather than a pass. ``unknown``
    and ``nodigest`` are what a pre-fingerprint artifact renders as, and treating an
    absent fingerprint as agreement would defeat the check exactly when it matters —
    on the old artifacts.

    Args:
        measured: The ``C_max`` side, either report shape.
        stages: The ``T_total`` side.
        allow_mismatch: Downgrade to a warning, for an operator who knows the
            difference is irrelevant to what they are planning.

    Raises:
        PlannerError: If the two slugs differ and ``allow_mismatch`` is false.
    """
    cmax_slug = _slug_of(measured)
    ttotal_slug = stages.config_slug

    if cmax_slug and ttotal_slug and cmax_slug == ttotal_slug:
        return

    if not cmax_slug or not ttotal_slug:
        detail = (
            f"one artifact carries no configuration fingerprint (C_max {cmax_slug or 'none'!r}, "
            f"T_total {ttotal_slug or 'none'!r}); it predates fingerprinting, so there is "
            "nothing to check it against"
        )
    else:
        detail = (
            f"C_max was measured on {cmax_slug!r} but T_total on {ttotal_slug!r}; the curve "
            "sizes instances of one configuration while the lag describes how fast another "
            "one boots"
        )

    if allow_mismatch:
        logger.warning(
            "Pairing artifacts from different configurations: {}. Proceeding because the "
            "mismatch was explicitly allowed; the resulting plan describes no deployed "
            "configuration exactly.",
            detail,
        )
        return
    raise PlannerError(
        f"cannot pair these artifacts: {detail}. Re-measure both against one "
        "configuration, or pass --allow-config-mismatch to proceed anyway."
    )


def _slug_of(measured: CMaxReport | Measured) -> str:
    """Configuration slug of either report shape.

    ``Measured`` has no ``config_slug`` property — it is the joined planner input
    rather than an artifact — so it is derived from the fingerprint the same way.
    """
    slug = getattr(measured, "config_slug", None)
    if isinstance(slug, str):
        return slug
    from tts_bench.fixture import DeployedConfig

    config = dict(getattr(measured, "deployed_config", {}) or {})
    if not config:
        return ""
    return DeployedConfig.from_dict(config).slug


def plan_one(
    measured: Measured,
    scenario: Scenario,
    *,
    provision_s: float | None = None,
    ceiling_s: float = SAGEMAKER_INVOCATION_CEILING_S,
) -> ScalingPlan:
    """One scaling configuration, for one scenario at one assumed provision time.

    Composes :mod:`shared.capacity` in the order the numbers depend on each other: the
    binding ``C_max`` at the chosen budget, the queueing budget the SLO leaves over, the
    binding target, the fleet that target implies at peak and trough, the queue depth
    that target's rate can drain, then the findings that say whether any of it holds.

    ``W_max`` is computed here rather than read off the scenario — ``SLO - S_p95``,
    which is the only value consistent with an end-to-end promise. ``C_max`` is the
    lower of the latency knee and the throughput ceiling, since an instance is bound by
    whichever it reaches first.

    Args:
        measured: Joined ``C_max`` + ``T_total`` input. ``t_total_s`` here is the lag
            to plan against, already substituted — see :func:`plan_sweep`.
        scenario: The stated load and policy choices. Every field is an assumption.
        provision_s: Recorded in the findings when the lag was built by substitution,
            so a swept row is identifiable as such. Does not change the arithmetic.
        ceiling_s: The invocation ceiling to judge against. Overridable for testing
            and for a stricter internal SLO, never for making an infeasible plan pass.

    Raises:
        PlannerError: If the chosen TTFAB budget is below every measured budget, so
            there is no knee to plan against.
    """
    budget_ms = scenario.ttfab_budget_ms
    try:
        c_max, c_max_source = measured.binding_c_max(budget_ms)
    except ValueError as exc:
        raise PlannerError(
            f"no C_max measured at or below the {budget_ms}ms budget "
            f"(measured: {sorted(measured.c_max_curve)}). Re-run `tts-bench cmax` with "
            f"--ttfab-budgets including {budget_ms}, or plan against a measured budget."
        ) from exc

    k = scenario.growth_factor_k
    derate = scenario.derate
    s_mean = measured.s_mean_s
    slo_s = scenario.ttfab_slo_ms / 1000.0
    # Derived, not read: the SLO is queue plus service, so the queue gets what service
    # leaves. Clamps at 0 when the model's own tail already misses the promise, which
    # `_slo_finding` reports as INFEASIBLE rather than as "no queue configured".
    w_max = w_max_for_slo(slo_s, measured.s_p95_s)

    target, binding = effective_c_target(c_max, k, s_mean, w_max, derate)

    peak_n = _fleet_for(scenario.peak_rps, scenario.peak_streams, s_mean, target)
    trough_n = _fleet_for(scenario.trough_rps, scenario.trough_streams, s_mean, target)

    # min is the floor a reserved-capacity account pays for whatever the traffic does,
    # so it is the trough fleet -- not 1 -- once a trough is stated. max is the peak
    # fleet: below it the SLO cannot be held at peak, and above it we are reserving
    # capacity no stated scenario uses.
    min_instances = max(scenario.min_instances_floor, trough_n)
    max_instances = max(min_instances, peak_n)

    queue_depth = q_per_instance(c_max, s_mean, w_max)
    absorbed = w_absorbed(w_max, k)
    headroom_lag, floored = effective_headroom_lag_s(w_max, k, measured.t_total_s)

    # k=1 is the no-reserve baseline: same load, same knee, no headroom held back.
    # Dividing fleets rather than costs keeps it an instance-count ratio, which is
    # what the reader can check against the two rows.
    baseline_target = c_target(c_max, 1.0, derate)
    baseline_peak = _fleet_for(scenario.peak_rps, scenario.peak_streams, s_mean, baseline_target)
    relative_cost = peak_n / baseline_peak if baseline_peak else 1.0

    plan_cost = _peak_cost(measured, scenario, peak_n, target)

    findings = _findings(
        measured=measured,
        scenario=scenario,
        c_max=c_max,
        c_max_source=c_max_source,
        target=target,
        binding=binding,
        peak_n=peak_n,
        queue_depth=queue_depth,
        w_max=w_max,
        headroom_lag=headroom_lag,
        floored=floored,
        relative_cost=relative_cost,
        provision_s=provision_s,
        ceiling_s=ceiling_s,
    )

    return ScalingPlan(
        model_name=measured.model_name,
        endpoint=measured.endpoint,
        instance_type=measured.instance_type,
        c_max=c_max,
        c_max_source=c_max_source,
        c_max_is_lower_bound=measured.binding_is_lower_bound(budget_ms),
        c_target=target,
        binding_constraint=binding,
        w_max_s=w_max,
        min_instances=min_instances,
        max_instances=max_instances,
        peak_instances=peak_n,
        trough_instances=trough_n,
        queue_max_depth=queue_depth,
        scale_out_cooldown_s=_scale_out_cooldown_s(measured.t_total_s),
        scale_in_cooldown_s=_scale_in_cooldown_s(measured.t_total_s),
        utilization_at_target=utilization_at_k(k, derate),
        w_absorbed_s=absorbed,
        headroom_lag_s=headroom_lag,
        peak_cost_per_hour=plan_cost[0],
        peak_cost_per_m_chars=plan_cost[1],
        relative_fleet_cost_vs_k1=relative_cost,
        findings=findings,
        measured=measured,
        scenario=scenario,
    )


def _fleet_for(
    rps: float | None,
    streams: float | None,
    s_mean_s: float,
    target_concurrency: float,
) -> int:
    """Instances for whichever load shape the scenario stated.

    Streams win when both are given: a stated stream count is a direct concurrency
    observation, while ``lambda x S`` is that same quantity inferred, and for bidi
    traffic the inference is the weaker of the two.
    """
    if streams is not None:
        return n_instances_from_streams(streams, target_concurrency)
    if rps is not None:
        return n_instances(rps, s_mean_s, target_concurrency)
    return 0


def _peak_cost(
    measured: Measured,
    scenario: Scenario,
    peak_n: int,
    target_concurrency: float,
) -> tuple[float, float]:
    """Fleet cost per hour and per million characters at peak.

    On-demand rates, which is an **upper bound**: a reserved-capacity or committed-use
    account pays less, and we can verify a Pricing API number but not a contract rate.
    Reported that way rather than adjusted by a guessed discount.

    Characters per hour come from the planned throughput, not from a saturated
    measurement: a fleet running at ``derate / k`` produces well below what one
    saturated instance extrapolates to, and using the saturated figure would report
    the unit cost of a fleet nobody is running.
    """
    from tts_bench.cost import cost_per_m_chars, hourly_rate

    per_hour = hourly_rate(measured.instance_type) * peak_n

    served_rps = _served_rps(scenario, measured.s_mean_s, target_concurrency, peak_n)
    chars_per_hour = served_rps * measured.chars_per_request * 3600.0
    if chars_per_hour <= 0:
        # Either no load, or a curve that recorded no throughput. inf rather than 0:
        # an endpoint producing nothing has no meaningful unit cost, and 0 would make
        # a plan for a broken measurement look free.
        return per_hour, float("inf")
    return per_hour, cost_per_m_chars(chars_per_hour, measured.instance_type, peak_n)


def _served_rps(
    scenario: Scenario,
    s_mean_s: float,
    target_concurrency: float,
    peak_n: int,
) -> float:
    """Requests per second the planned fleet actually serves at peak.

    The stated peak when one was given. For a stream-shaped scenario there is no
    stated rate, so it is inferred from the concurrency the fleet is allowed to carry
    — which is the honest reading: ``N x C_target`` streams each completing every
    ``S``.
    """
    if scenario.peak_rps is not None:
        return scenario.peak_rps
    if s_mean_s <= 0:
        return 0.0
    return peak_n * target_concurrency / s_mean_s


def _scale_out_cooldown_s(t_total_s: float) -> int:
    """Seconds to wait after scaling out before scaling out again.

    Short by design, and *not* ``T_total``: target tracking adds one step at a time,
    so a cooldown as long as the lag means a surge needing three instances takes
    three lags to answer. Bounded below by the metric period, since a policy cannot
    react to data it has not received.
    """
    return int(max(CLOUDWATCH_HIGH_RES_PERIOD_S, min(30.0, t_total_s / 2)))


def _scale_in_cooldown_s(t_total_s: float) -> int:
    """Seconds to wait after scaling in before scaling in again.

    Deliberately asymmetric with scale-out and deliberately long: capacity removed
    takes a full ``T_total`` to get back, so scaling in early converts a saved
    dollar into a missed SLO. Several lags' worth, floored at 5 minutes so a fast
    ``T_total`` does not produce a twitchy policy.
    """
    return int(max(300.0, 3 * t_total_s))


def _findings(
    *,
    measured: Measured,
    scenario: Scenario,
    c_max: float,
    c_max_source: str,
    target: float,
    binding: str,
    peak_n: int,
    queue_depth: int,
    w_max: float,
    headroom_lag: float,
    floored: bool,
    relative_cost: float,
    provision_s: float | None,
    ceiling_s: float,
) -> list[Finding]:
    """Every feasibility statement, in the order a reader needs them.

    Ordered by what invalidates what: an untrustworthy measurement makes the rest
    moot, an infeasible SLO makes the fleet size irrelevant, and cost only matters
    once the plan holds.

    ``w_max`` is passed in rather than read off the scenario: it is *derived* from the
    SLO in :func:`plan_one`, and re-deriving it here would be a second place for the
    arithmetic to live. The findings below quote it against the SLO it came from.
    """
    out: list[Finding] = []
    k = scenario.growth_factor_k

    out.append(_trust_finding(measured))
    out.append(_repeatability_finding(measured, scenario.ttfab_budget_ms))
    out.append(_c_max_source_finding(measured, scenario.ttfab_budget_ms, c_max, c_max_source))
    out.append(_slo_finding(measured, scenario.ttfab_slo_ms, w_max))

    fits, deadline = fits_invocation_ceiling(w_max, measured.s_p95_s, ceiling_s)
    if fits:
        out.append(
            Finding(
                name="invocation_ceiling",
                verdict=Verdict.OK,
                detail=(
                    f"a queued request finishes in {deadline:.1f}s worst case "
                    f"(W_max {w_max:.1f}s + p95 service {measured.s_p95_s:.2f}s), inside "
                    f"the {ceiling_s:.0f}s SageMaker invocation ceiling"
                ),
            )
        )
    else:
        largest = max_added_wait_under_ceiling(measured.s_p95_s, ceiling_s)
        out.append(
            Finding(
                name="invocation_ceiling",
                verdict=Verdict.INFEASIBLE,
                detail=(
                    f"a queued request would take {deadline:.1f}s worst case "
                    f"(W_max {w_max:.1f}s + p95 service {measured.s_p95_s:.2f}s), past the "
                    f"{ceiling_s:.0f}s SageMaker invocation ceiling. Those requests wait "
                    "the full W_max and then fail anyway, which is worse than refusing "
                    "them at admission."
                ),
                # W_max is SLO - S_p95, so the deadline *is* the stated SLO whenever the
                # SLO is feasible — this branch is reached only by an SLO past the
                # platform ceiling, and the fix is to state one inside it. Where S_p95
                # alone busts the ceiling no SLO helps, because nothing under the model's
                # own tail is reachable.
                recommendation=(
                    f"--ttfab-slo-ms {int(ceiling_s * 1000)} or less; a longer SLO cannot "
                    "be served whatever the queue does"
                    if largest > 0
                    else (
                        f"p95 service time ({measured.s_p95_s:.1f}s) alone exceeds the "
                        "ceiling; no SLO or queue length fixes this. Shorten the request, "
                        "or serve it off a real-time endpoint."
                    )
                ),
            )
        )

    out.append(
        Finding(
            name="binding_constraint",
            verdict=Verdict.OK,
            detail=(
                f"C_target {target:.3f} is set by {binding}: surge headroom wants "
                f"{c_target(c_max, k, scenario.derate):.3f} (derate {scenario.derate} x "
                f"C_max {c_max:.2f} / k {k:g}), the wait budget wants "
                f"{c_slo_cap(w_max, measured.s_mean_s):.3f} (W_max {w_max:.1f}s / S "
                f"{measured.s_mean_s:.3f}s + 1)"
            ),
            recommendation=(
                "a surge-bound target wants a smaller k or a shorter T_total"
                if binding == "surge_headroom"
                else "an SLO-bound target wants a faster model or a looser wait budget"
            ),
        )
    )

    covers = queue_covers_surge(w_max, k, measured.t_total_s)
    if covers:
        out.append(
            Finding(
                name="queue_covers_surge",
                verdict=Verdict.OK,
                detail=(
                    # k=1 makes W_absorbed infinite, which is true but unreadable as a
                    # duration. Say why it is infinite instead of printing "inf s".
                    f"traffic is flat at k=1, so no backlog accumulates and the queue "
                    f"covers the whole {measured.t_total_s:.0f}s T_total"
                    if k == 1
                    else (
                        f"the queue absorbs {w_absorbed(w_max, k):.0f}s of lag at k={k:g}, past "
                        f"the {measured.t_total_s:.0f}s T_total — a surge of this size is "
                        "invisible to clients"
                    )
                ),
            )
        )
    else:
        out.append(
            Finding(
                name="queue_covers_surge",
                verdict=Verdict.WARN,
                detail=(
                    f"the queue absorbs {w_absorbed(w_max, k):.0f}s of a {measured.t_total_s:.0f}s "
                    f"T_total at k={k:g}, leaving {headroom_lag:.0f}s for standing headroom "
                    f"to cover — which is what holds C_target down to {target:.3f}"
                ),
                recommendation=(
                    "shorten T_total, loosen the SLO if the promise allows (W_max follows "
                    "it), or accept the idle instances the reserve costs"
                ),
            )
        )

    out.append(_headroom_finding(k, headroom_lag, floored))

    out.append(
        Finding(
            name="queue_depth",
            verdict=Verdict.OK if queue_depth > 0 else Verdict.WARN,
            detail=(
                f"Q_max {queue_depth} per instance = Lambda_cap "
                f"{lambda_cap_per_instance(c_max, measured.s_mean_s):.2f} rps x W_max "
                f"{w_max:.2f}s"
                if queue_depth > 0
                else (
                    f"Q_max rounds to 0: at Lambda_cap "
                    f"{lambda_cap_per_instance(c_max, measured.s_mean_s):.2f} rps the "
                    f"{w_max:.2f}s W_max the SLO leaves does not cover one request, so no "
                    "queue can help"
                )
            ),
            recommendation=(
                None
                if queue_depth > 0
                else "loosen --ttfab-slo-ms, use a faster model, or shed load"
            ),
        )
    )

    if relative_cost >= FLEET_COST_WARN_MULTIPLE:
        out.append(
            Finding(
                name="fleet_cost",
                verdict=Verdict.WARN,
                detail=(
                    f"the k={k:g} reserve costs {relative_cost:.1f}x the k=1 fleet "
                    f"({peak_n} instances against "
                    f"{max(1, round(peak_n / relative_cost))}) at "
                    f"{utilization_at_k(k, scenario.derate):.0%} utilization"
                ),
                recommendation=(
                    "at this multiple, shortening T_total or pre-warming is cheaper than "
                    "buying headroom"
                ),
            )
        )
    else:
        out.append(
            Finding(
                name="fleet_cost",
                verdict=Verdict.OK,
                detail=(
                    f"the k={k:g} reserve costs {relative_cost:.1f}x the k=1 fleet at "
                    f"{utilization_at_k(k, scenario.derate):.0%} utilization; on-demand "
                    "rates, so an upper bound"
                ),
            )
        )

    out.append(_provision_finding(provision_s, lag_measured=measured.t_total_measured))
    return out


def _repeatability_finding(measured: Measured, budget_ms: int) -> Finding:
    """Whether ``C_max`` at the planned budget is repeatable or a single sample.

    Separate from :func:`_trust_finding`, which asks whether the number is
    *per-instance*. This asks whether it is *stable*, and the two fail independently:
    a properly frozen run pinned to one instance can still produce a knee that only
    one ladder pass out of three found, and every fleet size below divides by it.

    ``SUPPRESSED`` rather than ``OK`` when the curve carries no spread: an artifact
    from before this was recorded has not passed the check, and marking it ``OK``
    would claim a repeatability nobody measured.
    """
    budget = budget_ms if budget_ms in measured.c_max_curve else None
    if budget is None:
        below = [b for b in measured.c_max_curve if b <= budget_ms]
        budget = max(below) if below else None

    spread = measured.curve_spread.get(budget) if budget is not None else None
    contributing = measured.runs_contributing.get(budget) if budget is not None else None
    total = measured.runs_total

    if spread is None:
        return Finding(
            name="curve_repeatability",
            verdict=Verdict.SUPPRESSED,
            detail=(
                "the curve carries no run-to-run spread, so whether C_max is repeatable "
                "was never established — the artifact predates the check, or the run was "
                "a single pass"
            ),
            recommendation="re-run `tts-bench cmax --runs 3` to get a spread",
        )

    if contributing is not None and total > 1 and contributing < 2:
        return Finding(
            name="curve_repeatability",
            verdict=Verdict.WARN,
            detail=(
                f"C_max at {budget}ms came from {contributing} of {total} ladder runs, so "
                f"its {spread:.0%} spread is not a run-to-run agreement — it is one sample "
                "with nothing to disagree with. Every fleet size below divides by it."
            ),
            recommendation=(
                "re-run `tts-bench cmax`; the other runs' steps at this budget were "
                "unusable or unsettled, which usually means the ladder needs more "
                "workers or a longer hold"
            ),
        )

    if spread > CURVE_SPREAD_WARN:
        return Finding(
            name="curve_repeatability",
            verdict=Verdict.WARN,
            detail=(
                f"C_max at {budget}ms varied {spread:.0%} across "
                f"{contributing if contributing is not None else total} runs, past the "
                f"{CURVE_SPREAD_WARN:.0%} threshold — the ladder is resolving noise, and "
                f"the {measured.model_name} derate is not sized to absorb it"
            ),
            recommendation="hold each step longer, or space the ladder more widely",
        )

    return Finding(
        name="curve_repeatability",
        verdict=Verdict.OK,
        detail=(
            f"C_max at {budget}ms varied {spread:.0%} across "
            f"{contributing if contributing is not None else total} runs, inside the "
            f"{CURVE_SPREAD_WARN:.0%} threshold"
        ),
    )


def _c_max_source_finding(
    measured: Measured,
    budget_ms: int,
    c_max: float,
    source: str,
) -> Finding:
    """Which of the two measured limits the fleet size was divided by, and how firmly.

    Three distinctions that the report used to collapse into one number, and each wants
    a different follow-up run:

    * **latency knee, bracketed** — a step above it failed the budget. The knee is the
      knee, and nothing more is owed.
    * **latency knee, lower bound** — the ladder ran out while still passing. Every
      fleet size here is an over-estimate; a longer ladder resolves it.
    * **throughput ceiling** — the server stopped keeping up *before* latency crossed
      the budget, so the knee's higher concurrency was accumulated backlog. This is
      kokoro's regime on bidi and it is invisible to a latency-only reading, which is
      the whole reason the ceiling is measured.

    A fourth state, ``latency_knee_only``, is never ``OK``: no ceiling was measured, so
    the comparison did not happen. That is not the same claim as a knee that won it.
    """
    knee = measured.c_max_for(budget_ms)
    ceiling = measured.c_max_throughput
    lower_bound = measured.binding_is_lower_bound(budget_ms)

    if source == "throughput_ceiling":
        what = (
            f"C_max {c_max:.2f} is the throughput ceiling, under the {knee:.2f} latency "
            f"knee at {budget_ms}ms: the server stopped keeping up before latency crossed "
            "the budget, so the knee's extra concurrency was queue backlog, not capacity"
        )
        bracketed_clause = ", and a saturated step above it confirms the ceiling"
        extend = (
            "re-run `tts-bench cmax` with a higher --target-concurrency, and "
            "--max-workers above the in-flight count the backlog reaches, so the client "
            "pool is not what capped the rate"
        )
    elif source == "latency_knee":
        what = (
            f"C_max {c_max:.2f} is the latency knee at {budget_ms}ms, at or under the "
            f"{ceiling:.2f} throughput ceiling: latency degrades before throughput does"
        )
        bracketed_clause = ", and a step above it failed the budget, so it is the knee"
        extend = (
            "re-run `tts-bench cmax` with a higher --target-concurrency so a step above "
            "the knee actually fails the budget"
        )
    else:
        what = (
            f"C_max {c_max:.2f} is the latency knee at {budget_ms}ms and nothing else — no "
            "throughput ceiling was measured, so whether the server was still keeping up "
            "at that concurrency was never checked. A model holding its inference lock "
            "for a whole session is usually bound by the ceiling first"
        )
        bracketed_clause = "; the knee itself was bracketed"
        extend = "re-run `tts-bench cmax`, which measures the ceiling alongside the knee"

    if lower_bound is None:
        return Finding(
            name="c_max_source",
            verdict=Verdict.SUPPRESSED,
            detail=(
                f"{what}. Whether it was bracketed is unrecorded — the artifact predates "
                "the check, which is not the same as passing it"
            ),
            recommendation=extend,
        )
    if lower_bound:
        return Finding(
            name="c_max_source",
            verdict=Verdict.WARN,
            detail=(
                f"{what}, and it is a LOWER BOUND: nothing above it was observed to give "
                "way, so the real limit is higher and every fleet size here over-estimates"
            ),
            recommendation=extend,
        )
    return Finding(
        name="c_max_source",
        verdict=Verdict.SUPPRESSED if source == "latency_knee_only" else Verdict.OK,
        detail=f"{what}{bracketed_clause}",
        recommendation=extend if source == "latency_knee_only" else None,
    )


def _slo_finding(measured: Measured, slo_ms: int, w_max: float) -> Finding:
    """Whether the end-to-end SLO leaves any room to queue, and how much.

    The finding that makes ``W_max`` visibly derived. It is stated here rather than
    inferred from the config block, because the failure it catches is silent: a model
    whose own p95 already misses the promise reports ``W_max = 0``, and a reader seeing
    ``queue_max_depth = 0`` would take that for a design choice.

    ``INFEASIBLE`` at ``S_p95 >= SLO`` because nothing in this module's output moves it.
    Queue depth, instance count, and policy all govern *waiting*, and at that point
    there is no wait left to govern.
    """
    slo_s = slo_ms / 1000.0
    if not slo_is_feasible(slo_s, measured.s_p95_s):
        return Finding(
            name="slo_budget",
            verdict=Verdict.INFEASIBLE,
            detail=(
                f"p95 service time {measured.s_p95_s:.3f}s already meets or exceeds the "
                f"{slo_s:.1f}s first-byte SLO before a request waits at all, so W_max is 0. "
                "No queue depth, instance count, or scaling policy rescues this: they "
                "govern waiting, and there is none left to govern."
            ),
            recommendation=(
                f"--ttfab-slo-ms above {int(measured.s_p95_s * 1000)}, a shorter request, "
                "or a faster model"
            ),
        )

    detail = (
        f"W_max {w_max:.2f}s = {slo_s:.1f}s SLO - {measured.s_p95_s:.3f}s p95 service — "
        "derived, not configured, so the queue allowance cannot drift away from the "
        "promise it is spending"
    )
    if w_max < measured.s_p95_s:
        return Finding(
            name="slo_budget",
            verdict=Verdict.WARN,
            detail=(
                f"{detail}. That leaves less waiting room than one request's own tail "
                f"takes to serve, so a single request queued ahead misses the SLO — the "
                "promise holds for an uncontended request and little more"
            ),
            recommendation=(
                f"--ttfab-slo-ms of {int(measured.s_p95_s * 2000)} or more buys room for "
                "one request's wait; below that, size the fleet so nothing queues"
            ),
        )
    return Finding(name="slo_budget", verdict=Verdict.OK, detail=detail)


def _headroom_finding(k: float, headroom_lag: float, floored: bool) -> Finding:
    """Whether the standing headroom in this plan was sized from anything observable.

    ``effective_headroom_lag_s`` floors the uncovered lag at the CloudWatch metric
    period, and reports ``floored`` for two very different reasons that must not share
    a verdict:

    * At ``k=1`` traffic is flat, there is no backlog to cover, and the floor is a
      formality — warning about it would tell the operator to fix a plan that has
      nothing wrong with it.
    * Above ``k=1`` the floor means real uncovered lag was rounded *up* to the metric
      period, so the plan is sized for growth faster than CloudWatch can report and
      the headroom is a guess.
    """
    if not floored:
        return Finding(
            name="headroom_lag",
            verdict=Verdict.OK,
            detail=(
                f"standing headroom covers {headroom_lag:.0f}s of lag the queue cannot, "
                "which is longer than the metric period — so the policy has data to act on"
            ),
        )
    if k == 1:
        return Finding(
            name="headroom_lag",
            verdict=Verdict.OK,
            detail=(
                "flat traffic at k=1 needs no standing headroom, so the uncovered lag is "
                f"the {CLOUDWATCH_HIGH_RES_PERIOD_S:.0f}s metric period by formality, not "
                "by rounding"
            ),
        )
    return Finding(
        name="headroom_lag",
        verdict=Verdict.WARN,
        detail=(
            f"the uncovered lag floored at the {CLOUDWATCH_HIGH_RES_PERIOD_S:.0f}s "
            f"CloudWatch metric period at k={k:g}, so this plan is sized for growth faster "
            "than we can observe in time to act on it — the headroom is a guess, not a "
            "measurement"
        ),
        recommendation="treat k as the real input here and sweep it (--sweep-k)",
    )


def _trust_finding(measured: Measured) -> Finding:
    """Whether the curve underneath this plan is safe to read as per-instance.

    First because it invalidates everything after it: a ``C_max`` measured while the
    fleet was resizing is some multiple of the per-instance number, and every fleet
    size here divides by it.
    """
    if measured.trustworthy:
        return Finding(
            name="measurement_trust",
            verdict=Verdict.OK,
            detail=(
                "C_max was measured with autoscaling suspended and capacity pinned to "
                f"{measured.instance_counts_observed or (1,)}, so it reads as per-instance"
            ),
        )
    reasons = []
    if not measured.frozen:
        reasons.append("autoscaling was not suspended")
    counts = set(measured.instance_counts_observed)
    if len(counts) > 1:
        reasons.append(f"the fleet resized mid-run ({sorted(counts)})")
    return Finding(
        name="measurement_trust",
        verdict=Verdict.WARN,
        detail=(
            f"C_max may not be per-instance: {', and '.join(reasons)}. Every fleet size "
            "below divides by it, so an N-instance C_max understates the fleet N-fold."
        ),
        recommendation="re-run `tts-bench cmax --require-frozen`",
    )


def _provision_finding(provision_s: float | None, lag_measured: bool = True) -> Finding:
    """What part of ``T_total`` is measured and what part is stated.

    Never suppressed: the deliverable is a plan for an account whose placement latency
    is not ours to measure, so which half is which is the single most important caveat
    on the whole output.

    Args:
        provision_s: Provision time substituted into the lag, or ``None`` when the
            lag was used whole.
        lag_measured: Whether the lag came from a ``ttotal`` run at all. A whole lag
            from ``--assume-t-total`` is not "used exactly as measured" — nothing about
            it was measured — and saying so would launder a command-line argument into
            an observation, which is the one thing this finding exists to prevent.
    """
    if provision_s is None and not lag_measured:
        return Finding(
            name="provision_stage",
            verdict=Verdict.WARN,
            detail=(
                "T_total was stated whole, not measured — there is no observed stage "
                "breakdown, so nothing here distinguishes the transferable container "
                "stages from this account's EC2 provisioning."
            ),
            recommendation=(
                "run `tts-bench ttotal` and pass --ttotal, so the provision stage can be "
                "swept instead of buried in one number"
            ),
        )
    if provision_s is None:
        return Finding(
            name="provision_stage",
            verdict=Verdict.OK,
            detail=(
                "T_total is used exactly as measured, provision stage included. That "
                "stage is a property of this account's spare EC2 capacity; a "
                "reserved-capacity account will differ."
            ),
            recommendation="sweep it with --provision-s to see the plan's sensitivity",
        )
    return Finding(
        name="provision_stage",
        verdict=Verdict.OK,
        detail=(
            f"T_total assumes a {provision_s:.0f}s EC2 provision stage, substituted for the "
            "measured one. The container stages are measured and transfer between "
            "accounts; this one is stated."
        ),
    )


@dataclass(frozen=True, slots=True)
class SweepRow:
    """One plan in a sweep, with the assumptions that produced it.

    Carried alongside the plan rather than dug back out of it, because
    ``provision_s`` is not a ``ScalingPlan`` field — the plan is what to deploy, and
    this is why.
    """

    provision_s: float | None
    k: float
    t_total_s: float
    plan: ScalingPlan

    @property
    def provision_assumed(self) -> bool:
        return self.provision_s is not None


def plan_sweep(
    measured: Measured,
    scenario: Scenario,
    stages: TTotalStages | None = None,
    *,
    provision_sweep_s: Sequence[float] | None = None,
    k_sweep: Sequence[float] | None = None,
    ceiling_s: float = SAGEMAKER_INVOCATION_CEILING_S,
) -> list[SweepRow]:
    """A plan per (provision time, k) pair — the deliverable.

    Two sweeps because there are two inputs we cannot measure for the account being
    configured: the EC2 provision stage (a property of a capacity contract) and the
    growth factor ``k`` (a property of production traffic we do not have). Everything
    else is measured, and holding those fixed while these vary is what makes the
    output honest.

    Args:
        measured: Joined planner input. Its ``t_total_s`` is used directly when no
            provision sweep is requested.
        scenario: Stated load and policy. Its ``growth_factor_k`` is used when no
            ``k_sweep`` is given.
        stages: The measured stage breakdown, needed to substitute a provision time.
            Without it, ``provision_sweep_s`` cannot be honoured.
        provision_sweep_s: Provision times to assume, seconds. ``None`` or empty means
            plan once against the measured ``T_total`` as-is.
        k_sweep: Growth factors to plan for. ``None`` or empty means the scenario's own.
        ceiling_s: Invocation ceiling to judge against.

    Raises:
        PlannerError: If a provision sweep was requested but ``stages`` cannot supply
            a transferable lag to substitute into.
    """
    provisions: list[float | None]
    if provision_sweep_s:
        if stages is None:
            raise PlannerError(
                "a provision sweep needs the measured stage breakdown; pass the ttotal "
                "artifact so the measured provision stage can be substituted out."
            )
        provisions = [float(p) for p in provision_sweep_s]
    else:
        provisions = [None]

    ks = [float(k) for k in k_sweep] if k_sweep else [scenario.growth_factor_k]

    rows: list[SweepRow] = []
    for provision in provisions:
        if provision is None:
            t_total = measured.t_total_s
            row_measured = measured
        else:
            assert stages is not None  # guarded above
            t_total = stages.with_provision_s(provision)
            # Amend on "a provision time was substituted", not on "the total changed".
            # Sweeping the value that was actually measured produces an identical
            # number from a different claim -- the row is labelled `provision_assumed`
            # and its finding says "assumes a Ns provision stage", so leaving the
            # provenance saying "measured" would have the same plan describe its own
            # input two ways.
            row_measured = _with_t_total(measured, t_total)
        for k in ks:
            row_scenario = scenario.model_copy(update={"growth_factor_k": k})
            rows.append(
                SweepRow(
                    provision_s=provision,
                    k=k,
                    t_total_s=t_total,
                    plan=plan_one(
                        row_measured,
                        row_scenario,
                        provision_s=provision,
                        ceiling_s=ceiling_s,
                    ),
                )
            )
    return rows


def _with_t_total(measured: Measured, t_total_s: float) -> Measured:
    """``measured`` with a substituted lag, and a provenance note saying so.

    The note is the point. ``Measured`` is tagged ``MEASURED``, and a substituted lag
    is partly an assumption; without amending the note a swept row would claim to
    have measured a provision time nobody observed. Callers decide *whether* a
    substitution happened — an unchanged total is not evidence that none did, since
    sweeping the measured provision time reproduces it exactly.
    """
    note = measured.provenance.note
    amended = "T_total provision stage substituted, not measured"
    return measured.model_copy(
        update={
            "t_total_s": t_total_s,
            "provenance": measured.provenance.model_copy(
                update={"note": f"{note}; {amended}" if note else amended},
            ),
        }
    )


def measured_from_artifacts(
    cmax_report: CMaxReport,
    stages: TTotalStages,
    *,
    allow_config_mismatch: bool = False,
    assume_t_total_s: float | None = None,
    require_pairing: bool = True,
) -> Measured:
    """Join a ``C_max`` curve and a ``T_total`` lag into one planner input.

    Thin on purpose: ``CMaxReport.to_measured`` already owns the join, and this adds
    the pairing refusal and the "no total measured" path around it.

    Args:
        cmax_report: The curve.
        stages: The lag, from :meth:`TTotalStages.from_artifact`.
        allow_config_mismatch: Pair artifacts from different configurations anyway.
        assume_t_total_s: Lag to use when the ``ttotal`` run never spanned a total.
            Tagged as an assumption in the provenance, since it is one.
        require_pairing: Whether there are two artifacts to pair at all. False when
            the lag was stated rather than read from a ``ttotal`` run: a stated number
            carries no fingerprint, so checking one against the curve's would report a
            mismatch where there is nothing to mismatch. The lag is still labelled an
            assumption in the provenance, which is the honest complaint to make about
            it.

    Raises:
        PlannerError: If the artifacts do not pair, or if no lag is available from
            either the measurement or an assumption.
    """
    if require_pairing:
        assert_pairable(cmax_report, stages, allow_mismatch=allow_config_mismatch)

    total = stages.total_s
    if total is None:
        if assume_t_total_s is None:
            raise PlannerError(
                "the ttotal artifact observed too few stages to span a T_total, and no "
                "--assume-t-total was given. A plan needs a lag: re-run `tts-bench "
                "ttotal`, or state one and have it labelled an assumption."
            )
        total = assume_t_total_s
        provenance = Provenance(
            origin=Origin.ASSUMPTION,
            note=(
                f"T_total {total:.0f}s stated, not measured — the ttotal run observed "
                f"{len(stages.missing_stages)} missing stage(s)"
            ),
        )
    else:
        provenance = Provenance(
            origin=Origin.MEASURED,
            run_id=stages.run_id or None,
            note=(
                f"T_total measured, trigger={stages.trigger}"
                + (", recovery inferred so this is a floor" if stages.bounded else "")
            ),
        )

    return cmax_report.to_measured(t_total_s=total, t_total_provenance=provenance)
