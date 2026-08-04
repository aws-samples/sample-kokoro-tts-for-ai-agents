"""Render a scaling plan: the six variables, the findings, and a paste-ready config block.

Separated from :mod:`tts_bench.planner` so the arithmetic is testable without asserting
on strings, and so the string layout can change without touching a single equation.

The paste-ready block is the point of the whole tool chain. Before it, the path from a
measurement to a deployed policy ran through hand-arithmetic in a comment block in
``config.py`` — which is how ``scaling_target_value`` and ``queue_max_depth`` came to be
four numbers nobody could re-derive without redoing the algebra. :func:`render_config`
emits exactly the fields ``ModelEndpointConfig`` declares, with the derivation beside
each one, so the loop closes by copy-paste.

**The rendering keeps client units and CloudWatch units visibly apart.** Every threshold
appears twice: as the occupancy that was measured and as the ``Maximum``-statistic value
that deploys, with the measured ratio between them. Collapsing those two into one column
is what put 0.713 — a client mean read as a server peak — onto a live endpoint.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from tts_bench.types import Verdict

if TYPE_CHECKING:
    from tts_bench.planner import TTotalStages
    from tts_bench.types import ScalingPlan

#: Marker per verdict. ``SUPPRESSED`` is not a pass and is not marked as one — it means
#: the condition was never evaluated, which a reader must be able to tell from OK.
_VERDICT_MARK: dict[Verdict, str] = {
    Verdict.OK: "OK  ",
    Verdict.WARN: "WARN",
    Verdict.INFEASIBLE: "STOP",
    Verdict.SUPPRESSED: "----",
}


def render_inputs(plan: ScalingPlan, stages: TTotalStages | None = None) -> str:
    """The four inputs — two measured, two chosen — before any derived output.

    First because the reader's first question is "on what". A plan whose ``Q_max`` came
    from an unfrozen run, or from a container that sheds, is a different object from one
    built on a clean measurement, and the difference is not visible downstream.
    """
    measured, scenario = plan.measured, plan.scenario
    lines = [
        f"Plan for {measured.model_name} on {plan.instance_type} "
        f"({measured.endpoint}), via {measured.transport}",
        "",
        "  measured:",
        f"    Q_max            {plan.q_max} concurrent per instance"
        + (
            " — LOWER BOUND, nothing above it was seen to fail"
            if plan.q_max_is_lower_bound
            else f", bracketed against the {measured.slo_ms}ms SLO"
        ),
        f"    S                mean {measured.s_mean_s * 1000:.0f}ms, "
        f"p95 {measured.s_p95_s * 1000:.0f}ms (includes the client round trip)",
    ]
    if measured.chars_per_request > 0:
        lines.append(f"    chars/request    {measured.chars_per_request:.0f}")

    if stages is not None:
        lines.extend(_render_lag(plan, stages))
    else:
        lines.append(f"    T_total          {measured.t_total_s:.0f}s")

    config_slug = _config_slug(plan)
    if config_slug:
        lines.append(f"    configuration    {config_slug}")
    if not measured.trustworthy:
        lines.append("    TRUST            not per-instance — see findings")

    lines.extend(
        [
            "",
            "  chosen (these are inputs, not observations):",
            f"    SLO              {scenario.ttfab_slo_ms / 1000:.1f}s to first byte, "
            "queue included",
            f"    surge ratio      {scenario.max_scaling_per_t_total:g}x traffic growth "
            "within one T_total",
            f"    peak             {_load_str(scenario.peak_rps, scenario.peak_streams)}",
            f"    trough           {_load_str(scenario.trough_rps, scenario.trough_streams)}",
            "",
            "  derived (none of these can be set independently):",
            f"    C_scale_max      {plan.c_scale_max:.2f} concurrent — scale out here "
            f"({1 - (scenario.max_scaling_per_t_total - 1):.2f} x Q_max)",
            f"    C_scale_min      {plan.c_scale_min:.2f} concurrent — scale in here "
            f"({1 - 2 * (scenario.max_scaling_per_t_total - 1):.2f} x Q_max)",
            f"    W_max            {plan.w_max_s:.2f}s queueing budget = "
            f"{scenario.ttfab_slo_ms / 1000:.1f}s SLO - {measured.s_p95_s:.3f}s p95 service",
            f"    utilization      {plan.utilization_at_c_scale_max:.1%} at C_scale_max "
            "(L/(1+L) on one server — steeply non-linear, hence surge_survival)",
        ]
    )
    lines.append(_render_units_line(plan))
    return "\n".join(lines)


def _render_units_line(plan: ScalingPlan) -> str:
    """``C_scale_max`` in the units the deployed alarm reads, or why it is unavailable.

    Its own line, immediately under the occupancy it converts, because the two numbers
    are easy to mistake for a rounding difference and hard to mistake for one when the
    multiplier is printed between them.
    """
    if plan.c_scale_max_in_cw_units is None or plan.cw_units_ratio is None:
        return (
            "    in CW units      UNAVAILABLE — the ladder recorded no "
            "ConcurrentRequestsPerModel/Maximum, so the client occupancy above cannot be "
            "converted into what the alarm compares against"
        )
    return (
        f"    in CW units      {plan.c_scale_max_in_cw_units:.2f} = C_scale_max x "
        f"{plan.cw_units_ratio:.2f} (ConcurrentRequestsPerModel/Maximum per client mean "
        "in-flight, measured on the same ladder) — THIS is what deploys"
    )


def _render_lag(plan: ScalingPlan, stages: TTotalStages) -> list[str]:
    """``T_total`` and which of its parts were measured.

    The most important caveat on the output, so it sits inline with the number rather
    than in a footnote. Two parts can be non-measurements for different reasons: the EC2
    provision stage is measured here but contractual in a reserved-capacity account, and
    the policy detection lag is bypassed by the ``force-desired`` trigger and bounded
    from the deployed alarm's own configuration instead.
    """
    out: list[str] = []
    if stages.total_s is None:
        out.append(f"    T_total          {plan.measured.t_total_s:.0f}s STATED, not measured")
        out.append(
            "      no stage breakdown behind it — run `tts-bench ttotal` to get one worth "
            "planning on"
        )
        return out

    out.append(f"    T_total          {plan.measured.t_total_s:.0f}s, planned against")
    out.append(f"      measured        {stages.total_s:.0f}s capacity request -> traffic served")
    if stages.policy_bound_s:
        out.append(
            f"      + BOUND         {stages.policy_bound_s:.0f}s policy detection, from the "
            "alarm's periods and cooldown — arithmetic, not a measurement"
        )
    if stages.provision_measured:
        assert stages.provision_s is not None  # provision_measured
        out.append(
            f"      of which        {stages.provision_s:.0f}s EC2 provision + image pull "
            "(this account's spare capacity, not the configuration's)"
        )
    else:
        out.append(
            "      provision stage not observed, so the plan cannot say which part is this "
            "account's placement latency"
        )
    if stages.bounded:
        out.append(
            "      FLOOR           recovery was never bounded, so the measured half stops at "
            "in_service and is a floor"
        )
    return out


def render_findings(plan: ScalingPlan) -> str:
    """Every feasibility statement, worst verdict first.

    Sorted so an ``INFEASIBLE`` cannot scroll off the top behind six OK lines. That
    ordering is the whole reason the 60s ceiling is a verdict rather than a log line.
    """
    if not plan.findings:
        return "No findings."

    order = {Verdict.INFEASIBLE: 0, Verdict.WARN: 1, Verdict.SUPPRESSED: 2, Verdict.OK: 3}
    lines = ["Findings:"]
    for finding in sorted(plan.findings, key=lambda f: order[f.verdict]):
        lines.append(f"  [{_VERDICT_MARK[finding.verdict]}] {finding.name}")
        lines.append(f"         {finding.detail}")
        if finding.recommendation:
            lines.append(f"         fix: {finding.recommendation}")
    return "\n".join(lines)


def render_config(plan: ScalingPlan) -> str:
    """A ``ModelEndpointConfig`` block, ready to paste into ``config.py``.

    Every scaling field the config declares, each with the derivation that produced it
    as a trailing comment. The comments are not decoration: the numbers currently in
    ``config.py`` were derived by hand once, and without their derivations beside them
    nobody could tell later which measurement they came from or whether a new one
    invalidated them.

    ``scaling_target_value`` and ``scale_in_threshold`` are emitted in **CloudWatch
    ``Maximum`` units**, which is what the alarms compare against. When the ladder
    recorded no server statistic there is no conversion to make, so the field is
    commented out rather than filled with the client figure — an unconverted threshold
    is the 0.713 defect, and a config that fails to parse is a better outcome than one
    that deploys a number no traffic satisfies.
    """
    measured, scenario = plan.measured, plan.scenario
    surge = scenario.max_scaling_per_t_total
    lines = [
        f'    "{measured.model_name}": ModelEndpointConfig(',
        f'        model_name="{measured.model_name}",',
        f'        instance_type="{plan.instance_type}",',
        f"        min_instances={plan.min_instances},"
        f"  # trough {_load_str(scenario.trough_rps, scenario.trough_streams)}"
        + (
            f"; {plan.min_safe_instances} is the smallest safe for scale-in"
            if plan.min_safe_instances is not None and plan.min_instances < plan.min_safe_instances
            else ""
        ),
        f"        max_instances={plan.max_instances},"
        f"  # peak {_load_str(scenario.peak_rps, scenario.peak_streams)}",
    ]

    ratio = plan.cw_units_ratio
    if plan.c_scale_max_in_cw_units is None or ratio is None:
        lines.extend(
            [
                "        # scaling_target_value=?,  # NO CONVERSION MEASURED. C_scale_max is "
                f"{plan.c_scale_max:.2f}",
                "        #   client-measured concurrency, but the alarm reads "
                "ConcurrentRequestsPerModel/Maximum,",
                "        #   which ran 1.35x-9.8x higher on one kokoro ladder. Re-run "
                "`tts-bench qmax --cloudwatch`;",
                "        #   deploying the raw occupancy is how 0.713 reached this endpoint.",
                "        # scale_in_threshold=?,  # same conversion, same reason",
            ]
        )
    else:
        lines.extend(
            [
                f"        scaling_target_value={plan.c_scale_max_in_cw_units:.3f},"
                f"  # C_scale_max {plan.c_scale_max:.2f} x {ratio:.2f} CW units; "
                f"= (1-h) x Q_max {plan.q_max} at h={surge - 1:.2f}",
                f"        scale_in_threshold={plan.c_scale_min * ratio:.3f},"
                f"  # C_scale_min {plan.c_scale_min:.2f} x {ratio:.2f}; "
                f"= (1-2h) x Q_max, one surge of excess headroom",
            ]
        )

    lines.extend(
        [
            f"        ttfab_slo_ms={scenario.ttfab_slo_ms},"
            f"  # end-to-end promise; W_max {plan.w_max_s:.2f}s + p95 service "
            f"{measured.s_p95_s:.2f}s {_ceiling_clause(plan)}",
            f"        queue_max_depth={plan.queue_max_depth},"
            f"  # = Q_max: past it a request cannot reach first byte inside the "
            f"{scenario.ttfab_slo_ms / 1000:.1f}s SLO",
            f"        scale_out_cooldown_s={plan.scale_out_cooldown_s},"
            "  # short: target tracking adds one instance at a time",
            f"        scale_in_cooldown_s={plan.scale_in_cooldown_s},"
            f"  # long: removed capacity costs a full {measured.t_total_s:.0f}s to replace",
            "    ),",
        ]
    )
    return "\n".join(lines)


def _ceiling_clause(plan: ScalingPlan) -> str:
    """Whether the deadline fits the invocation ceiling, per the finding that judged it.

    Both halves come off the plan rather than being restated here. The *verdict* is read
    from ``invocation_ceiling``, so a plan the check STOPped cannot be described as
    fitting; the *number* is ``plan.ceiling_s``, the value that check was given, because
    the ceiling is not always SageMaker's — ``--ceiling-s`` moves it, and the constant is
    only the default. A block that interpolated the constant printed "fits the 60s
    invocation ceiling" for a run judged at 2s, naming a limit that was never tested.
    This block gets pasted into ``config.py``, so a comment contradicting the findings
    above it is worse than no comment.
    """
    finding = next((f for f in plan.findings if f.name == "invocation_ceiling"), None)
    if finding is not None and finding.verdict is Verdict.INFEASIBLE:
        return "EXCEEDS the invocation ceiling — see the STOP finding above; do not deploy this"
    return f"fits the {plan.ceiling_s:.0f}s invocation ceiling"


def render_alarm_threshold(plan: ScalingPlan) -> str:
    """The ``FirstChunkLatencyP95`` alarm's threshold, which is not the SLO.

    That alarm watches service time on an instance already serving, where the request
    has spent none of its queue allowance. At an SLO-sized 3000 ms it fires only once
    the endpoint is roughly 10x past keeping up, which is not an alarm. The ladder's
    ``N=1`` rung is the number it wants, and it is measured on every rerun rather than
    hand-set — which is what the deleted second latency field used to be.
    """
    c1 = plan.measured.ttfab_p95_at_c1_ms
    if c1 is None:
        return (
            "FirstChunkLatencyP95 alarm: NO THRESHOLD — the ladder had no N=1 rung, so there "
            "is no measured service time to alarm on. Re-run `tts-bench qmax --concurrency "
            "1,...`; an alarm threshold has to come from somewhere real."
        )
    return (
        f"FirstChunkLatencyP95 alarm: {c1:.0f}ms (p95 TTFAB at one outstanding request). "
        f"NOT the {plan.scenario.ttfab_slo_ms}ms SLO: this alarm watches service time on an "
        "instance already serving, where a request has spent none of its queue allowance."
    )


def _verdict_of(plan: ScalingPlan) -> str:
    """The worst verdict across a plan's findings, as a short marker."""
    if plan.infeasible:
        return "INFEASIBLE"
    if plan.warnings:
        return f"warn ({len(plan.warnings)})"
    return "ok"


def _config_slug(plan: ScalingPlan) -> str:
    """Configuration fingerprint of the plan's inputs, or empty when unknown."""
    from tts_bench.fixture import DeployedConfig

    config = dict(plan.measured.deployed_config or {})
    if not config:
        return ""
    return DeployedConfig.from_dict(config).slug


def _load_str(rps: float | None, streams: float | None) -> str:
    """A stated load, in whichever unit it was given.

    Streams first, matching the planner: a stated stream count is a concurrency
    observation while a rate is that quantity inferred through ``lambda x S``.
    """
    if streams is not None:
        return f"{streams:g} concurrent streams"
    if rps is not None:
        return f"{rps:g} rps"
    return "not stated"


def render_plan(plan: ScalingPlan, stages: TTotalStages | None = None) -> str:
    """The whole report: inputs, findings, the config block, then the alarm threshold."""
    return "\n".join(
        [
            render_inputs(plan, stages),
            "",
            render_findings(plan),
            "",
            render_config(plan),
            "",
            render_alarm_threshold(plan),
        ]
    )


def plan_to_dict(plan: ScalingPlan) -> dict[str, object]:
    """The plan as JSON, for an artifact.

    Carries the rendered config block alongside the structured plan: the block is the
    thing an operator pastes, and regenerating it later means re-running this module
    against a plan whose renderer may have changed in between.
    """
    return {
        # Not derivable from the plan alone: a lag that was stated whole reads identically
        # to one that was measured, and that is the distinction the provenance exists for.
        "t_total_measured": plan.measured.t_total_measured,
        "plan": plan.model_dump(mode="json"),
        "verdict": _verdict_of(plan),
        "config_block": render_config(plan),
        "alarm_threshold": render_alarm_threshold(plan),
    }
