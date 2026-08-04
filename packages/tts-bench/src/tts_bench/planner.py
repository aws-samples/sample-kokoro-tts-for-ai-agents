"""Turn measurements into a scaling configuration.

The step that closes the loop. ``qmax`` measures ``Q_max`` against the SLO, ``ttotal``
measures the scaling lag stage by stage, and this module composes them with a stated
scenario into what ``ModelEndpointConfig`` actually needs: ``scaling_target_value``,
``scale_in_threshold``, ``queue_max_depth``, ``min_instances``, ``max_instances``.

No new math lives here. Every equation is in :mod:`shared.capacity`, which is
modality-neutral and unit-tested without AWS in scope; this module's job is
composition, provenance, and refusing to compose things that should not be combined.

**Six variables, and only two of them are measured.** ``SLO`` and
``max_scaling_per_T_total`` are chosen; ``Q_max`` and ``T_total`` are measured;
``C_scale_max`` and ``C_scale_min`` are arithmetic on the other four
(:func:`shared.capacity.scale_thresholds`). There is no ceiling to compare against and
no knee to pick a column from — one ladder, one SLO, one answer.

**Three refusals, all hard.** A ``Q_max`` ladder and a ``T_total`` lag measured on
different configurations cannot be combined (:func:`assert_pairable`) — that is what
the fingerprint is for. A ``Q_max`` measured against one SLO cannot be read against
another, because the SLO is the line that *defines* it. And a queueing budget that does
not fit inside SageMaker's 60s invocation ceiling is ``INFEASIBLE`` rather than a
warning, because the requests it admits wait the full ``W_max`` and then fail anyway.

**``W_max`` is derived, never stated.** The SLO is end-to-end — a request must reach
first byte within ``ttfab_slo_ms`` *including* its time in the queue — so the queueing
budget is ``SLO - S_p95`` and nothing else. It used to be a hand-set ``Scenario`` field
sitting beside a second latency budget with no relation between them, which is how
kokoro came to be deployed with a 20 s queue allowance under a 300 ms budget. Two
independent numbers can disagree with the promise; one derived number cannot.

**The thresholds ship in CloudWatch units, not client units.** ``C_scale_max`` is a
client-side occupancy; the deployed alarm reads ``ConcurrentRequestsPerModel`` /
*Maximum*. Those differed by 1.35x to 9.8x across one kokoro ladder, so the conversion
is measured per configuration on the same run and reported beside the raw figure.
Deploying the unconverted number is the defect that put 0.713 on the endpoint — a value
no positive arrival rate satisfies.

**Two known limits of the simple rule, computed rather than asserted.**
:func:`shared.capacity.shed_probability` says whether ``C_scale_max`` fires early
enough to survive one ``T_total`` (``surge_survival``), and
``ScaleThresholds.min_safe_instances`` says whether scale-in flaps at the planned fleet
size (``scale_in_safety``). Both are findings, so a fragile plan is visibly fragile.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from loguru import logger

from shared.capacity import (
    CLOUDWATCH_HIGH_RES_PERIOD_S,
    SAGEMAKER_INVOCATION_CEILING_S,
    fits_invocation_ceiling,
    max_added_wait_under_ceiling,
    n_instances,
    n_instances_from_streams,
    scale_thresholds,
    shed_probability,
    slo_is_feasible,
    utilization_for_occupancy,
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
    from tts_bench.types import QMaxReport

#: Stages whose duration belongs to AWS provisioning an EC2 instance, from the capacity
#: change being requested to the container's log stream opening. The one stage reserved
#: capacity changes, and so the one the plan labels as this account's rather than the
#: configuration's.
PROVISION_FROM_STAGE = "desired_set"
PROVISION_TO_STAGE = "instance_logging"

#: Run-to-run spread in ``Q_max`` above which the ladder is measuring noise. Both
#: thresholds are fractions of ``Q_max``, so its spread propagates to both.
Q_MAX_SPREAD_WARN = 0.2

#: P(shedding within one ``T_total``) above which ``C_scale_max`` is not early enough.
#: Not zero: a queue is a stochastic object and some tail risk is the price of running
#: it at all. 0.1 is one surge in ten reaching ``Q_max`` before help arrives.
SHED_PROBABILITY_WARN = 0.1


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
    property of a contract. Everything else — image pull, weights, framework init,
    warm-up, recovery — is a property of the configuration and carries across.
    """

    total_s: float | None
    """``desired_set`` through to traffic recovered, as measured. ``None`` when the run
    never observed enough stages to span it."""

    provision_s: float | None
    """``desired_set`` -> ``instance_logging``. ``None`` when either boundary was not
    observed, in which case there is nothing to attribute to this account."""

    policy_bound_s: float | None = None
    """Detection lag the ``force-desired`` trigger bypasses, bounded from the deployed
    policy's own configuration rather than measured. Carried separately so the sum
    stays decomposable: see :attr:`plan_total_s`."""

    bounded: bool = False
    """Whether ``total_s`` rests on an inferred endpoint. A bounded total that stops at
    ``in_service`` *under*-reports the lag, which is the dangerous direction."""

    trigger: str = ""
    """Always ``force-desired``. That trigger raises ``DesiredInstanceCount`` directly,
    so its total is the capacity half only — the policy half is the bound above."""

    config_slug: str = ""
    run_id: str = ""
    missing_stages: tuple[str, ...] = ()

    @property
    def plan_total_s(self) -> float | None:
        """The lag to plan against: the measured span plus the bounded policy lag.

        Production scales out through the policy, not through a capacity call, so the
        measured half alone under-states what a surge has to be absorbed across. The
        two terms stay separate on the artifact and are added exactly here, once.
        """
        if self.total_s is None:
            return None
        return self.total_s + (self.policy_bound_s or 0.0)

    @property
    def provision_measured(self) -> bool:
        return self.provision_s is not None

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
        # Read the bound off the artifact rather than importing the constant: the
        # deployed policy's periods and cooldown are what the number came from, and a
        # plan built later should use the bound that applied when the lag was measured.
        bound = raw.get("policy_lag_bound_s")
        missing = raw.get("missing_stages")
        return cls(
            total_s=float(total) if isinstance(total, int | float) else None,
            provision_s=provision_s,
            policy_bound_s=float(bound) if isinstance(bound, int | float) else None,
            bounded=bool(raw.get("t_total_bounded", False)),
            trigger=str(raw.get("trigger", "")),
            config_slug=str(raw.get("config_slug", "")),
            run_id=str(raw.get("run_id", "")),
            missing_stages=tuple(str(s) for s in missing) if isinstance(missing, list) else (),
        )


def assert_pairable(
    measured: QMaxReport | Measured,
    stages: TTotalStages,
    *,
    allow_mismatch: bool = False,
) -> None:
    """Refuse to pair a ``Q_max`` ladder with a ``T_total`` lag from another configuration.

    The whole point of the fingerprint. Pairing a g5 ladder with a g6 lag produces a
    plan for a fleet that exists nowhere: the ladder sizes instances of one type while
    the lag describes how fast a different type boots, and nothing about the resulting
    numbers looks wrong.

    An empty slug on either side counts as a mismatch rather than a pass. ``unknown``
    and ``nodigest`` are what a pre-fingerprint artifact renders as, and treating an
    absent fingerprint as agreement would defeat the check exactly when it matters —
    on the old artifacts.

    Args:
        measured: The ``Q_max`` side, either report shape.
        stages: The ``T_total`` side.
        allow_mismatch: Downgrade to a warning, for an operator who knows the
            difference is irrelevant to what they are planning.

    Raises:
        PlannerError: If the two slugs differ and ``allow_mismatch`` is false.
    """
    qmax_slug = _slug_of(measured)
    ttotal_slug = stages.config_slug

    if qmax_slug and ttotal_slug and qmax_slug == ttotal_slug:
        return

    if not qmax_slug or not ttotal_slug:
        detail = (
            f"one artifact carries no configuration fingerprint (Q_max {qmax_slug or 'none'!r}, "
            f"T_total {ttotal_slug or 'none'!r}); it predates fingerprinting, so there is "
            "nothing to check it against"
        )
    else:
        detail = (
            f"Q_max was measured on {qmax_slug!r} but T_total on {ttotal_slug!r}; the ladder "
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


def _slug_of(measured: QMaxReport | Measured) -> str:
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
    policy_bound_s: float | None = None,
    ceiling_s: float = SAGEMAKER_INVOCATION_CEILING_S,
) -> ScalingPlan:
    """The scaling configuration for one measurement and one scenario.

    Composes :mod:`shared.capacity` in the order the numbers depend on each other: the
    two thresholds from ``Q_max`` and the surge ratio, the queueing budget the SLO
    leaves over, the fleet those thresholds imply at peak and trough, the CloudWatch
    conversion the alarm needs, then the findings that say whether any of it holds.

    ``W_max`` is computed here rather than read off the scenario — ``SLO - S_p95``,
    which is the only value consistent with an end-to-end promise.

    Args:
        measured: Joined ``Q_max`` + ``T_total`` input. Its ``t_total_s`` is the lag to
            plan against, policy bound already included by
            :func:`measured_from_artifacts`.
        scenario: The stated load and the two chosen variables. Every field is an
            assumption.
        provision_s: The measured EC2 provision stage, for the ``provision_stage``
            finding. Does not change the arithmetic — it is a *label* on which part of
            the lag belongs to this account rather than to the configuration.
        policy_bound_s: The bounded policy-detection term inside ``t_total_s``, for the
            same finding. ``None`` when the lag was stated whole.
        ceiling_s: The invocation ceiling to judge against. Overridable for testing
            and for a stricter internal SLO, never for making an infeasible plan pass.

    Raises:
        PlannerError: If the scenario's SLO is not the one ``Q_max`` was measured
            against, or if the surge ratio admits no usable thresholds.
    """
    if measured.slo_ms != scenario.ttfab_slo_ms:
        raise PlannerError(
            f"Q_max was measured against a {measured.slo_ms}ms SLO but this scenario asks "
            f"for {scenario.ttfab_slo_ms}ms. Q_max is *defined* by the SLO — it is the "
            "highest concurrency whose p95 stayed inside that line — so the ladder says "
            f"nothing about the other one. Plan at --ttfab-slo-ms {measured.slo_ms}, or "
            f"re-run `tts-bench qmax --slo-ms {scenario.ttfab_slo_ms}`."
        )

    q_max = measured.q_max
    surge = scenario.max_scaling_per_t_total
    try:
        thresholds = scale_thresholds(q_max, surge)
    except ValueError as exc:
        raise PlannerError(
            f"cannot derive thresholds from Q_max {q_max} at a {surge:g}x surge ratio: {exc}"
        ) from exc

    slo_s = scenario.ttfab_slo_ms / 1000.0
    # Derived, not read: the SLO is queue plus service, so the queue gets what service
    # leaves. Clamps at 0 when the model's own tail already misses the promise, which
    # `_slo_finding` reports as INFEASIBLE rather than as "no queue configured".
    w_max = w_max_for_slo(slo_s, measured.s_p95_s)

    # The fleet is sized on the scale-out threshold, not on Q_max: Q_max is where the
    # SLO breaks, and sizing a fleet to sit there is sizing it to sit at the edge.
    peak_n = _fleet_for(
        scenario.peak_rps, scenario.peak_streams, measured.s_mean_s, thresholds.c_scale_max
    )
    trough_n = _fleet_for(
        scenario.trough_rps, scenario.trough_streams, measured.s_mean_s, thresholds.c_scale_max
    )

    # min is the floor a reserved-capacity account pays for whatever the traffic does,
    # so it is the trough fleet -- not 1 -- once a trough is stated. max is the peak
    # fleet: below it the SLO cannot be held at peak, and above it we are reserving
    # capacity no stated scenario uses.
    min_instances = max(scenario.min_instances_floor, trough_n)
    max_instances = max(min_instances, peak_n)

    cw = _cw_units(measured, thresholds.c_scale_max)
    utilization = utilization_for_occupancy(thresholds.c_scale_max)
    shed = _shed_probability(measured, thresholds.c_scale_max, q_max)

    # A no-headroom fleet is the baseline: same load, same Q_max, scaling out only when
    # the SLO is already at the line. Dividing fleets rather than costs keeps it an
    # instance-count ratio, which is what the reader can check against the two numbers.
    baseline_peak = _fleet_for(
        scenario.peak_rps, scenario.peak_streams, measured.s_mean_s, float(q_max)
    )
    relative_cost = peak_n / baseline_peak if baseline_peak else 1.0

    plan_cost = _peak_cost(measured, scenario, peak_n, thresholds.c_scale_max)

    findings = _findings(
        measured=measured,
        scenario=scenario,
        thresholds=thresholds,
        peak_n=peak_n,
        min_instances=min_instances,
        w_max=w_max,
        cw=cw,
        utilization=utilization,
        shed=shed,
        relative_cost=relative_cost,
        provision_s=provision_s,
        policy_bound_s=policy_bound_s,
        ceiling_s=ceiling_s,
    )

    return ScalingPlan(
        model_name=measured.model_name,
        endpoint=measured.endpoint,
        instance_type=measured.instance_type,
        q_max=q_max,
        q_max_is_lower_bound=not measured.q_max_bracketed,
        c_scale_max=thresholds.c_scale_max,
        c_scale_min=thresholds.c_scale_min,
        c_scale_max_in_cw_units=cw[0],
        cw_units_ratio=cw[1],
        min_safe_instances=thresholds.min_safe_instances,
        w_max_s=w_max,
        # The same `ceiling_s` the finding above was judged against, so the renderer can
        # name it instead of interpolating the constant. One source for the number: a
        # plan whose config block claimed 60s while the finding checked something else
        # would contradict itself in the block that gets pasted into config.py.
        ceiling_s=ceiling_s,
        min_instances=min_instances,
        max_instances=max_instances,
        peak_instances=peak_n,
        trough_instances=trough_n,
        # The admission bound *is* Q_max: past it a request cannot reach first byte
        # inside the SLO, so admitting it buys a late success instead of an honest 503.
        queue_max_depth=q_max,
        scale_out_cooldown_s=_scale_out_cooldown_s(measured.t_total_s),
        scale_in_cooldown_s=_scale_in_cooldown_s(measured.t_total_s),
        utilization_at_c_scale_max=utilization,
        shed_probability_at_c_scale_max=shed,
        peak_cost_per_hour=plan_cost[0],
        peak_cost_per_m_chars=plan_cost[1],
        findings=findings,
        measured=measured,
        scenario=scenario,
    )


def _cw_units(measured: Measured, c_scale_max: float) -> tuple[float | None, float | None]:
    """``c_scale_max`` in CloudWatch ``Maximum`` units, and the ratio that converted it.

    The threshold that ships is the converted number. ``ConcurrentRequestsPerModel`` /
    *Maximum* over a 10s period and a client's mean in-flight are different quantities:
    the first is a peak over a window, the second an average over a run, and across one
    kokoro ladder they ran from 9.8x apart to 1.35x apart. So the ratio is measured per
    configuration, at the rung nearest the threshold being converted, rather than fitted
    once and reused.

    Returns ``(None, None)`` when the ladder recorded no server-side statistic — the
    plan then says the conversion is unavailable instead of assuming it is 1:1, which
    is the assumption that put 0.713 on the endpoint.
    """
    table = measured.cw_units_ratio_by_rung
    if not table:
        return None, None
    # Nearest rung, ties to the *higher* one: the ratio shrinks as load rises, so at a
    # tie the higher rung gives the smaller multiplier and the tighter threshold.
    rung = min(table, key=lambda r: (abs(r - c_scale_max), -r))
    ratio = table[rung]
    if ratio <= 0:
        return None, None
    return c_scale_max * ratio, ratio


def _shed_probability(measured: Measured, c_scale_max: float, q_max: int) -> float | None:
    """P(the queue reaches ``Q_max``) while one ``T_total`` elapses from ``C_scale_max``.

    ``s_mean_s`` is the service rate the simulation drains at. It includes the client
    round trip, so it over-states service slightly and the result is conservative in
    the safe direction — a real instance drains a little faster than this says.

    ``None`` rather than a number when the simulation's own preconditions do not hold,
    since a fabricated probability is worse than a missing one.
    """
    try:
        # Annotated locals throughout this module: `shared` ships no py.typed, so under
        # --ignore-missing-imports every capacity helper resolves to Any and returning
        # one directly trips warn_return_any.
        probability: float = shed_probability(
            c_scale_max,
            float(q_max),
            measured.t_total_s,
            measured.s_mean_s,
        )
        return probability
    except ValueError as exc:
        logger.warning("Cannot simulate shedding at C_scale_max {}: {}", c_scale_max, exc)
        return None


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
        from_streams: int = n_instances_from_streams(streams, target_concurrency)
        return from_streams
    if rps is not None:
        from_rps: int = n_instances(rps, s_mean_s, target_concurrency)
        return from_rps
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
    measurement: a fleet held at ``C_scale_max`` produces below what one saturated
    instance extrapolates to, and using the saturated figure would report the unit cost
    of a fleet nobody is running.
    """
    from tts_bench.cost import cost_per_m_chars, hourly_rate

    per_hour = hourly_rate(measured.instance_type) * peak_n

    served_rps = _served_rps(scenario, measured.s_mean_s, target_concurrency, peak_n)
    chars_per_hour = served_rps * measured.chars_per_request * 3600.0
    if chars_per_hour <= 0:
        # Either no load, or a ladder that recorded no throughput. inf rather than 0:
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
    — which is the honest reading: ``N x C_scale_max`` streams each completing every
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
    ``T_total`` does not produce a twitchy policy — and it is also what damps the
    flap ``scale_in_safety`` warns about.
    """
    return int(max(300.0, 3 * t_total_s))


def _findings(
    *,
    measured: Measured,
    scenario: Scenario,
    thresholds: Any,
    peak_n: int,
    min_instances: int,
    w_max: float,
    cw: tuple[float | None, float | None],
    utilization: float,
    shed: float | None,
    relative_cost: float,
    provision_s: float | None,
    policy_bound_s: float | None,
    ceiling_s: float,
) -> list[Finding]:
    """Every feasibility statement, in the order a reader needs them.

    Ordered by what invalidates what: an untrustworthy measurement makes the rest
    moot, an infeasible SLO makes the fleet size irrelevant, a threshold in the wrong
    units makes the policy wrong whatever the numbers say, and cost only matters once
    the plan holds.

    ``w_max`` and ``thresholds`` are passed in rather than recomputed: they are derived
    in :func:`plan_one`, and re-deriving them here would be a second place for the
    arithmetic to live.
    """
    out: list[Finding] = []

    out.append(_trust_finding(measured))
    out.append(_repeatability_finding(measured))
    out.append(_slo_finding(measured, scenario.ttfab_slo_ms, w_max))
    out.append(_ceiling_finding(measured, w_max, ceiling_s))
    out.append(_queue_depth_finding(measured, w_max))
    out.append(_threshold_units_finding(thresholds.c_scale_max, cw))
    out.append(
        _surge_finding(
            measured=measured,
            scenario=scenario,
            c_scale_max=thresholds.c_scale_max,
            utilization=utilization,
            shed=shed,
        )
    )
    out.append(
        _scale_in_finding(
            thresholds=thresholds,
            scenario=scenario,
            min_instances=min_instances,
        )
    )
    out.append(_fleet_cost_finding(scenario, thresholds.c_scale_max, peak_n, relative_cost))
    out.append(
        _provision_finding(
            provision_s=provision_s,
            policy_bound_s=policy_bound_s,
            total_s=measured.t_total_s,
            lag_measured=measured.t_total_measured,
        )
    )
    return out


def _ceiling_finding(measured: Measured, w_max: float, ceiling_s: float) -> Finding:
    """Whether a request that waits the full ``W_max`` still returns before SageMaker cuts it.

    A separate question from the SLO. The SLO is a promise we chose; the ceiling is a
    platform limit, and a queue sized past it admits requests that wait their whole
    allowance and then fail anyway — worse than refusing them at admission, because the
    client paid the wait for nothing.
    """
    fits, deadline = fits_invocation_ceiling(w_max, measured.s_p95_s, ceiling_s)
    if fits:
        return Finding(
            name="invocation_ceiling",
            verdict=Verdict.OK,
            detail=(
                f"a queued request finishes in {deadline:.1f}s worst case "
                f"(W_max {w_max:.1f}s + p95 service {measured.s_p95_s:.2f}s), inside "
                f"the {ceiling_s:.0f}s SageMaker invocation ceiling"
            ),
        )

    largest = max_added_wait_under_ceiling(measured.s_p95_s, ceiling_s)
    return Finding(
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


def _queue_depth_finding(measured: Measured, w_max: float) -> Finding:
    """Whether the admission bound the plan ships is the one that was measured.

    ``queue_max_depth`` is ``Q_max`` itself, so there is no arithmetic to check here —
    the finding exists to say *that*, and to compare it against the wait the SLO
    affords. Those two agree by construction when the ladder bracketed its answer, and
    the size of any disagreement is the size of the extrapolation.
    """
    q_max = measured.q_max
    # What the SLO's own wait budget says the depth should be, at the measured service
    # rate. A cross-check on the ladder, not an input: if the ladder stopped early this
    # is larger, which is exactly the lower-bound case.
    implied = w_max / measured.s_mean_s if measured.s_mean_s > 0 else 0.0

    if not measured.q_max_bracketed:
        return Finding(
            name="queue_depth",
            verdict=Verdict.WARN,
            detail=(
                f"queue_max_depth {q_max} is Q_max, but the ladder never measured a rung "
                f"above it missing the SLO — so it is a LOWER bound. The SLO's own wait "
                f"budget implies room for about {implied:.0f} "
                f"(W_max {w_max:.2f}s / S {measured.s_mean_s:.3f}s), and admitting fewer "
                "than that sheds requests that would have been served in time"
            ),
            recommendation=(
                f"re-run `tts-bench qmax` with --concurrency rungs above {q_max} so the "
                "crossing is bracketed"
            ),
        )
    return Finding(
        name="queue_depth",
        verdict=Verdict.OK,
        detail=(
            f"queue_max_depth {q_max} is the measured Q_max, bracketed from above. The "
            f"SLO's wait budget independently implies about {implied:.0f} "
            f"(W_max {w_max:.2f}s / S {measured.s_mean_s:.3f}s); past the bound a request "
            "cannot reach first byte in time, so it is refused rather than served late"
        ),
    )


def _threshold_units_finding(
    c_scale_max: float,
    cw: tuple[float | None, float | None],
) -> Finding:
    """Whether the scale-out threshold can be stated in the units the alarm reads.

    The one finding that exists because of a shipped defect rather than a limit of the
    model. ``C_scale_max`` is a client-side occupancy; the deployed alarm compares
    ``ConcurrentRequestsPerModel`` / *Maximum* over 10s against its threshold. Those are
    different quantities, and deploying the unconverted figure is how 0.713 reached the
    endpoint — a threshold that inverts to a negative arrival rate, so no traffic
    satisfies it and target tracking asks for the whole fleet on one request.

    ``SUPPRESSED``, never ``OK``, when the ladder recorded no server statistic: the
    conversion did not happen, which is not the same as not needing one. The high-res
    datapoints retain 3 hours, so it also cannot be backfilled — the fix is another run.
    """
    converted, ratio = cw
    if converted is None or ratio is None:
        return Finding(
            name="threshold_units",
            verdict=Verdict.SUPPRESSED,
            detail=(
                f"C_scale_max {c_scale_max:.2f} is a client-measured occupancy and the "
                "ladder recorded no ConcurrentRequestsPerModel / Maximum beside it, so "
                "there is no measured conversion into the units the alarm reads. "
                "Deploying the raw number is the defect that produced the 0.713 threshold"
            ),
            recommendation=(
                "re-run `tts-bench qmax --cloudwatch`; 10s datapoints retain 3 hours, so "
                "this cannot be recovered from the earlier run"
            ),
        )
    return Finding(
        name="threshold_units",
        verdict=Verdict.OK,
        detail=(
            f"scaling_target_value {converted:.2f} = C_scale_max {c_scale_max:.2f} x "
            f"{ratio:.2f}, the measured ratio of ConcurrentRequestsPerModel / Maximum to "
            "client mean in-flight at the nearest rung. The converted figure is what "
            "deploys; the occupancy is what was measured"
        ),
    )


def _surge_finding(
    *,
    measured: Measured,
    scenario: Scenario,
    c_scale_max: float,
    utilization: float,
    shed: float | None,
) -> Finding:
    """Whether ``C_scale_max`` fires early enough to survive one ``T_total``.

    The falsifiable form of the first known limit of the simple rule. ``C_scale_max``
    reserves headroom in queue *slots*, a finite stock, while surviving a surge is a
    question about drain *rate*, a flow — and occupancy converts to utilization
    steeply, so three quarters of ``Q_max`` is 97% utilized rather than three quarters
    of the way to trouble. Simulated rather than argued, because the simulation has a
    number and the argument does not.

    ``SUPPRESSED`` when the simulation could not run: an unsimulated risk is not a
    cleared one.
    """
    surge = scenario.max_scaling_per_t_total
    basis = (
        f"holding C_scale_max {c_scale_max:.2f} of Q_max {measured.q_max} is "
        f"{utilization:.1%} utilization on a single-server queue, and a replacement takes "
        f"{measured.t_total_s:.0f}s to arrive"
    )
    if shed is None:
        return Finding(
            name="surge_survival",
            verdict=Verdict.SUPPRESSED,
            detail=f"{basis}, but P(shedding before it lands) could not be simulated",
            recommendation="check t_total_s and S on the measurement; both must be positive",
        )
    if shed > SHED_PROBABILITY_WARN:
        return Finding(
            name="surge_survival",
            verdict=Verdict.WARN,
            detail=(
                f"{basis} — so P(the queue reaches Q_max before then) is {shed:.0%}, past "
                f"the {SHED_PROBABILITY_WARN:.0%} line. The {surge:g}x surge ratio reserves "
                "queue slots, which is a stock; surviving a surge is about drain rate, "
                "which is a flow, and at this utilization there is very little of it left"
            ),
            recommendation=(
                "scale out earlier than the simple rule (a smaller share of Q_max), "
                "shorten T_total, or hold standing headroom in instances"
            ),
        )
    return Finding(
        name="surge_survival",
        verdict=Verdict.OK,
        detail=(
            f"{basis} — P(the queue reaches Q_max before then) is {shed:.0%}, inside the "
            f"{SHED_PROBABILITY_WARN:.0%} line"
        ),
    )


def _scale_in_finding(
    *,
    thresholds: Any,
    scenario: Scenario,
    min_instances: int,
) -> Finding:
    """Whether removing one instance at ``C_scale_min`` lands back under ``C_scale_max``.

    The second known limit of the simple rule. Scale-in redistributes rather than
    removes load: dropping one of ``N`` multiplies each survivor's concurrency by
    ``N/(N-1)``, so at ``N=2`` the survivor inherits *double*. At a 1.25 surge ratio
    the thresholds are 0.75 and 0.5 of ``Q_max``, so 2->1 lands exactly on ``Q_max``
    and breaches the SLO on the way down — and kokoro runs ``min_instances=1``, which
    makes 2->1 the common case rather than the corner one.
    """
    safe = thresholds.min_safe_instances
    ratio = thresholds.c_scale_max / thresholds.c_scale_min if thresholds.c_scale_min else None
    surge = scenario.max_scaling_per_t_total

    if safe is None:
        return Finding(
            name="scale_in_safety",
            verdict=Verdict.WARN,
            detail=(
                f"at a {surge:g}x surge ratio the two thresholds are "
                f"{thresholds.c_scale_max:.2f} and {thresholds.c_scale_min:.2f}, which "
                "leaves no fleet size where removing an instance keeps the survivors under "
                "the scale-out point — every scale-in scales straight back out"
            ),
            recommendation=(
                "widen the gap between the thresholds with a smaller "
                "--max-scaling-per-t-total, or disable scale-in and manage the floor"
            ),
        )
    if min_instances < safe:
        return Finding(
            name="scale_in_safety",
            verdict=Verdict.WARN,
            detail=(
                f"scale-in is only stable from {safe} instances up: removing one of N "
                f"multiplies each survivor's concurrency by N/(N-1), which must stay "
                f"under {ratio:.2f} (= C_scale_max {thresholds.c_scale_max:.2f} / "
                f"C_scale_min {thresholds.c_scale_min:.2f}). This plan's floor is "
                f"{min_instances}, so a {min_instances + 1}->{min_instances} scale-in "
                f"leaves the survivors at {thresholds.c_scale_min * (min_instances + 1) / min_instances:.2f}"  # noqa: E501
            ),
            recommendation=(
                f"raise --min-floor to {safe}, or accept the flap that the "
                "scale-in cooldown damps but does not remove"
            ),
        )
    return Finding(
        name="scale_in_safety",
        verdict=Verdict.OK,
        detail=(
            f"scale-in is stable from {safe} instances up and this plan's floor is "
            f"{min_instances}: removing one leaves the survivors at "
            f"{thresholds.c_scale_min * min_instances / max(1, min_instances - 1):.2f}, "
            f"under the {thresholds.c_scale_max:.2f} scale-out point"
        ),
    )


def _fleet_cost_finding(
    scenario: Scenario,
    c_scale_max: float,
    peak_n: int,
    relative_cost: float,
) -> Finding:
    """What the surge headroom costs, against a fleet that reserves none.

    The baseline is scaling out at ``Q_max`` itself — the cheapest possible policy and
    also the one that misses the SLO the moment anything arrives. Stated as an instance
    ratio so the reader can check it against the two fleet sizes rather than trusting a
    dollar figure derived from on-demand rates.

    Always ``OK``, and that is a property of the model rather than a missing check. Both
    fleets are the same demand over a different divisor, so the ratio cannot exceed
    ``Q_max / C_scale_max``, which is ``1 / (1 - h)``; :func:`scale_thresholds` refuses
    ``h >= 0.5``, so the continuous ceiling is under 2x and integer rounding reaches
    exactly 2x (one instance against two) and no further. There is no reachable multiple
    at which "shorten T_total instead of buying headroom" becomes the cheaper advice, so
    the finding reports the cost and does not warn about it. The old ``C_max`` model could
    reach 4x, because there the divisor moved with ``k`` without a bound.
    """
    surge = scenario.max_scaling_per_t_total
    return Finding(
        name="fleet_cost",
        verdict=Verdict.OK,
        detail=(
            f"reserving headroom for a {surge:g}x surge costs {relative_cost:.1f}x a fleet "
            f"scaled at Q_max ({peak_n} instances at C_scale_max {c_scale_max:.2f} against "
            f"{max(1, round(peak_n / relative_cost)) if relative_cost else peak_n} at Q_max); "
            "on-demand rates, so an upper bound"
        ),
    )


def _repeatability_finding(measured: Measured) -> Finding:
    """Whether ``Q_max`` is repeatable or a single sample.

    Separate from :func:`_trust_finding`, which asks whether the number is
    *per-instance*. This asks whether it is *stable*, and the two fail independently:
    a properly frozen run pinned to one instance can still land on a rung only one
    ladder pass out of three agreed with, and both thresholds are fractions of it.

    ``SUPPRESSED`` rather than ``OK`` when only one pass contributed: a single sample
    has nothing to disagree with, so its 0% spread is not an agreement.
    """
    spread = measured.q_max_spread
    contributing = measured.runs_contributing

    if contributing < 2:
        return Finding(
            name="curve_repeatability",
            verdict=Verdict.SUPPRESSED,
            detail=(
                f"Q_max {measured.q_max} came from a single ladder pass, so whether it is "
                "repeatable was never established — a 0% spread across one run is not an "
                "agreement"
            ),
            recommendation="re-run `tts-bench qmax --runs 2` or more to get a spread",
        )

    if spread > Q_MAX_SPREAD_WARN:
        return Finding(
            name="curve_repeatability",
            verdict=Verdict.WARN,
            detail=(
                f"Q_max varied {spread:.0%} across {contributing} ladder passes, past the "
                f"{Q_MAX_SPREAD_WARN:.0%} threshold — the ladder is resolving noise, and "
                "both thresholds are fractions of the value it settled on"
            ),
            recommendation="hold each rung longer, or space the ladder more widely",
        )

    return Finding(
        name="curve_repeatability",
        verdict=Verdict.OK,
        detail=(
            f"Q_max varied {spread:.0%} across {contributing} ladder passes, inside the "
            f"{Q_MAX_SPREAD_WARN:.0%} threshold; the plan uses the minimum, which is a rung "
            "that was actually measured"
        ),
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


def _trust_finding(measured: Measured) -> Finding:
    """Whether ``Q_max`` is safe to read as per-instance, and against an unbounded queue.

    First because it invalidates everything after it. Two independent ways it can fail:
    a ladder run while the fleet was resizing measures some multiple of the per-instance
    number, and a ladder run against a container that sheds measures that container's
    admission bound rather than the depth at which the SLO breaks.
    """
    reasons = []
    if not measured.frozen:
        reasons.append("autoscaling was not suspended")
    counts = set(measured.instance_counts_observed)
    if len(counts) > 1:
        reasons.append(f"the fleet resized mid-run ({sorted(counts)})")
    if measured.unbounded_queue is False:
        reasons.append("the container bounds its admission queue, so it sheds before the SLO does")
    elif measured.unbounded_queue is None:
        reasons.append("the container's queue bound was never checked, which is not a pass")

    if not reasons:
        return Finding(
            name="measurement_trust",
            verdict=Verdict.OK,
            detail=(
                "Q_max was measured with autoscaling suspended, capacity pinned to "
                f"{measured.instance_counts_observed or (1,)}, and no container queue "
                "bound, so it reads as one instance's own limit"
            ),
        )
    return Finding(
        name="measurement_trust",
        verdict=Verdict.WARN,
        detail=(
            f"Q_max may not be this instance's own limit: {', and '.join(reasons)}. Both "
            "thresholds are fractions of it and every fleet size below divides by it."
        ),
        recommendation="re-run `tts-bench qmax --require-frozen --require-unbounded-queue`",
    )


def _provision_finding(
    *,
    provision_s: float | None,
    policy_bound_s: float | None,
    total_s: float,
    lag_measured: bool = True,
) -> Finding:
    """Which parts of ``T_total`` are measured and which are stated.

    Never suppressed: the deliverable is a plan for an account whose placement latency
    is not ours to measure, so which part is which is the single most important caveat
    on the whole output. Two parts can be non-measurements, and they are separate
    claims — an EC2 provision stage that is measured *here* but contractual *there*,
    and a policy detection lag this trigger bypasses by design and so bounds instead.

    Args:
        provision_s: The measured ``desired_set`` -> ``instance_logging`` stage, or
            ``None`` when that boundary was never observed.
        policy_bound_s: The bounded policy term included in ``total_s``, or ``None``
            when the lag carries no such term.
        total_s: The lag the plan is built on, both terms included.
        lag_measured: Whether the lag came from a ``ttotal`` run at all. A whole lag
            from ``--assume-t-total`` is not "used exactly as measured" — nothing about
            it was measured — and saying so would launder a command-line argument into
            an observation, which is the one thing this finding exists to prevent.
    """
    if not lag_measured:
        return Finding(
            name="provision_stage",
            verdict=Verdict.WARN,
            detail=(
                f"T_total {total_s:.0f}s was stated whole, not measured — there is no "
                "observed stage breakdown, so nothing here distinguishes the transferable "
                "container stages from this account's EC2 provisioning."
            ),
            recommendation=(
                "run `tts-bench ttotal` and pass --ttotal, so the provision stage is "
                "labelled instead of buried in one number"
            ),
        )

    bound_clause = (
        ""
        if not policy_bound_s
        else (
            f" Of that, {policy_bound_s:.0f}s is the policy's detection lag, which the "
            "force-desired trigger bypasses and so is BOUNDED from the deployed alarm's "
            "periods and cooldown rather than measured."
        )
    )
    if provision_s is None:
        return Finding(
            name="provision_stage",
            verdict=Verdict.OK,
            detail=(
                f"T_total {total_s:.0f}s: the run never observed the boundary between EC2 "
                "provisioning and the container starting, so the plan cannot say which "
                "part is a property of this account's spare capacity and which of the "
                f"image.{bound_clause}"
            ),
            recommendation=(
                "check the ttotal artifact's missing_stages; the instance's log stream is "
                "what marks that boundary"
            ),
        )
    share = provision_s / total_s if total_s > 0 else 0.0
    return Finding(
        name="provision_stage",
        verdict=Verdict.OK,
        detail=(
            f"T_total {total_s:.0f}s, of which {provision_s:.0f}s ({share:.0%}) is EC2 "
            "provisioning and image pull — a property of this account's spare capacity, "
            "not of the configuration. A reserved-capacity account, where placement is "
            f"guaranteed, will differ on that part and only that part.{bound_clause}"
        ),
    )


def measured_from_artifacts(
    qmax_report: QMaxReport,
    stages: TTotalStages,
    *,
    allow_config_mismatch: bool = False,
    assume_t_total_s: float | None = None,
    require_pairing: bool = True,
) -> Measured:
    """Join a ``Q_max`` ladder and a ``T_total`` lag into one planner input.

    Thin on purpose: ``QMaxReport.to_measured`` already owns the join, and this adds
    the pairing refusal and the "no total measured" path around it.

    The lag it carries through is :attr:`TTotalStages.plan_total_s` — the measured span
    *plus* the bounded policy term — because production scales out through the policy
    rather than through a capacity call. The provenance note names both terms, so the
    sum stays decomposable by anyone reading the artifact later.

    Args:
        qmax_report: The ladder.
        stages: The lag, from :meth:`TTotalStages.from_artifact`.
        allow_config_mismatch: Pair artifacts from different configurations anyway.
        assume_t_total_s: Lag to use when the ``ttotal`` run never spanned a total.
            Tagged as an assumption in the provenance, since it is one.
        require_pairing: Whether there are two artifacts to pair at all. False when
            the lag was stated rather than read from a ``ttotal`` run: a stated number
            carries no fingerprint, so checking one against the ladder's would report a
            mismatch where there is nothing to mismatch. The lag is still labelled an
            assumption in the provenance, which is the honest complaint to make about
            it.

    Raises:
        PlannerError: If the artifacts do not pair, or if no lag is available from
            either the measurement or an assumption.
    """
    if require_pairing:
        assert_pairable(qmax_report, stages, allow_mismatch=allow_config_mismatch)

    total = stages.plan_total_s
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
        bits = [f"T_total measured, trigger={stages.trigger}"]
        if stages.policy_bound_s:
            bits.append(
                f"{stages.total_s:.0f}s measured + {stages.policy_bound_s:.0f}s policy lag "
                "bounded from the deployed alarm, not measured"
            )
        if stages.bounded:
            # "floor" is the operative word: the span stops at in_service, strictly earlier
            # than the instance serving traffic, so it under-reports — and under-reporting a
            # lag the plan has to absorb a surge across is the dangerous direction.
            bits.append(
                "recovery never bounded, so the measured half stops at in_service and is a floor"
            )
        provenance = Provenance(
            origin=Origin.MEASURED,
            run_id=stages.run_id or None,
            note="; ".join(bits),
        )

    return qmax_report.to_measured(t_total_s=total, t_total_provenance=provenance)
