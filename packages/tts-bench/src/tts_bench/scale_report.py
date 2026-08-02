"""Render a scaling plan: the sweep table, the findings, and a paste-ready config block.

Separated from :mod:`tts_bench.planner` so the arithmetic is testable without asserting
on strings, and so the string layout can change without touching a single equation.

The paste-ready block is the point of the whole tool chain. Before it, the path from a
measurement to a deployed policy ran through hand-arithmetic in a comment block in
``config.py`` — which is how ``scaling_target_value`` and ``queue_max_depth`` came to be
four numbers nobody could re-derive without redoing the algebra. :func:`render_config`
emits exactly the fields ``ModelEndpointConfig`` declares, with the derivation beside
each one, so the loop closes by copy-paste.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from shared.capacity import SAGEMAKER_INVOCATION_CEILING_S
from tts_bench.types import Verdict

if TYPE_CHECKING:
    from collections.abc import Sequence

    from tts_bench.planner import SweepRow, TTotalStages
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
    """The measured and assumed inputs, before any of the derived output.

    First because the reader's first question is "on what". A plan whose ``C_max``
    came from an unfrozen run or whose ``T_total`` came from a ``force-desired`` probe
    is a different object from one built on a clean measurement, and the difference is
    not visible in the derived numbers.
    """
    measured, scenario = plan.measured, plan.scenario
    lines = [
        f"Plan for {measured.model_name} on {plan.instance_type} "
        f"({measured.endpoint}), via {measured.transport}",
        "",
        "  measured:",
        f"    C_max            {plan.c_max:.2f} concurrent, {_c_max_basis(plan)}",
        f"    S                mean {measured.s_mean_s * 1000:.0f}ms, "
        f"p95 {measured.s_p95_s * 1000:.0f}ms",
        f"    Lambda_cap       {plan.c_max / measured.s_mean_s:.2f} rps per instance (C_max / S)",
    ]
    if measured.chars_per_request > 0:
        lines.append(f"    chars/request    {measured.chars_per_request:.0f}")

    # The measured total, not this row's. `plan.measured.t_total_s` has already had a
    # provision time substituted into it by the sweep, so printing it here under the
    # heading "measured" -- directly above a stage breakdown that sums to something
    # else -- states two different totals as if both were observed. The per-row totals
    # are the sweep table's job.
    if stages is not None and stages.total_s is not None:
        lines.append(f"    T_total          {stages.total_s:.0f}s as measured")
        lines.extend(_render_stage_provenance(stages))
    elif stages is not None:
        # State the number in use, not only that it is unmeasured. "not measured" alone
        # sends the reader hunting for the lag every row below was built from, and it
        # belongs under a heading that says stated rather than one that says measured.
        lines.append(f"    T_total          {measured.t_total_s:.0f}s STATED, not measured")
        lines.extend(_render_stage_provenance(stages))
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
            "  assumed (these are inputs, not observations):",
            f"    peak             {_load_str(scenario.peak_rps, scenario.peak_streams)}",
            f"    trough           {_load_str(scenario.trough_rps, scenario.trough_streams)}",
            f"    k                {scenario.growth_factor_k:g}x growth within one T_total",
            f"    SLO              {scenario.ttfab_slo_ms / 1000:.1f}s to first byte, "
            "queue included",
            f"    derate           {scenario.derate}",
            "",
            "  derived from the SLO (not an input — it cannot be set independently):",
            f"    W_max            {plan.w_max_s:.2f}s queueing budget = "
            f"{scenario.ttfab_slo_ms / 1000:.1f}s SLO - {measured.s_p95_s:.3f}s p95 service",
        ]
    )
    return "\n".join(lines)


def _c_max_basis(plan: ScalingPlan) -> str:
    """Which measurement ``C_max`` came from, and whether it is bounded.

    Printed beside the number rather than left to the findings, because ``C_max`` is
    the one input every fleet size below divides by: a reader who takes a throughput
    ceiling for a latency knee will go looking for the wrong lever, and one who takes a
    lower bound for a measurement will not know the fleet is over-sized.
    """
    budget_ms = plan.scenario.ttfab_budget_ms
    basis = {
        "throughput_ceiling": "at the throughput ceiling",
        "latency_knee": f"at the p95 TTFAB {budget_ms}ms latency knee",
    }.get(plan.c_max_source, f"at the p95 TTFAB {budget_ms}ms latency knee, ceiling unmeasured")
    if plan.c_max_is_lower_bound is None:
        return f"{basis} (bracketing unrecorded)"
    if plan.c_max_is_lower_bound:
        return f"{basis} — LOWER BOUND, so the fleet below is over-sized"
    return basis


def _render_stage_provenance(stages: TTotalStages) -> list[str]:
    """Which part of ``T_total`` was measured and which is stated.

    The most important caveat on the output, so it sits inline with the number rather
    than in a footnote: the deliverable is a plan for a reserved-capacity account whose
    placement latency nobody here can measure.
    """
    out: list[str] = []
    if stages.provision_measured:
        assert stages.provision_s is not None  # provision_measured
        out.append(
            f"      of which        {stages.provision_s:.0f}s EC2 provision (this account's "
            "spare capacity; swept below)"
        )
        transferable = stages.transferable_s
        if transferable is not None:
            out.append(
                f"      transferable    {transferable:.0f}s detection + pull + container "
                "(properties of the image)"
            )
    else:
        out.append(
            "      provision stage not observed, so nothing to substitute out — the whole "
            "lag is treated as transferable"
        )
    if stages.bounded:
        out.append("      BOUNDED         recovery was inferred, so T_total is a floor")
    if stages.trigger == "force-desired":
        out.append(
            "      HALF            trigger=force-desired skips detection and alarm, so "
            "this is not a full T_total"
        )
    return out


def render_sweep(rows: Sequence[SweepRow]) -> str:
    """One line per (provision time, k) pair — the sweep that is the deliverable.

    Columns are ordered by what a reader acts on: the two assumptions, then the lag
    they produce, then the four numbers that go into ``config.py``, then the cost of
    holding them.
    """
    if not rows:
        return "No plans: the sweep produced nothing."

    header = (
        f"{'provision':>10} {'k':>4} {'T_total':>8} {'C_target':>9} {'Q_max':>6} "
        f"{'min':>4} {'max':>4} {'util':>6} {'$/hr':>8} {'vs k=1':>7}  verdict"
    )
    lines = [header, "-" * len(header)]
    for row in rows:
        plan = row.plan
        # "stated" when the whole lag came from --assume-t-total: nothing about it was
        # measured, and this column is the reader's shorthand for how much of the row
        # to trust.
        provision = f"{row.provision_s:.0f}s" if row.provision_s is not None else _lag_label(row)
        lines.append(
            f"{provision:>10} {row.k:>4g} {row.t_total_s:>7.0f}s {plan.c_target:>9.3f} "
            f"{plan.queue_max_depth:>6} {plan.min_instances:>4} {plan.max_instances:>4} "
            f"{plan.utilization_at_target:>5.0%} {plan.peak_cost_per_hour:>8.2f} "
            f"{plan.relative_fleet_cost_vs_k1:>6.1f}x  {_verdict_of(plan)}"
        )

    lines.append("")
    if any(row.provision_s is None and not row.plan.measured.t_total_measured for row in rows):
        lines.append(
            "  provision 'stated' means the whole T_total came from --assume-t-total, so there "
            "is no"
        )
        lines.append(
            "  stage breakdown behind it — run `tts-bench ttotal` to get one worth sweeping."
        )
    else:
        lines.append(
            "  provision 'measured' means T_total was used as observed here. Every other row "
            "substitutes"
        )
        lines.append(
            "  a stated EC2 provision time for the measured one, holding the container stages "
            "fixed — that"
        )
        lines.append("  is the row to read for a reserved-capacity account.")
    lines.append(
        "  $/hr is the peak fleet at on-demand rates: an UPPER BOUND. A committed account "
        "pays less."
    )
    return "\n".join(lines)


def _lag_label(row: SweepRow) -> str:
    """How a row's unswept ``T_total`` was obtained, for the provision column.

    Two rows can both show no substituted provision time for opposite reasons: one was
    measured whole, the other stated whole. Printing "measured" for both puts a
    command-line argument in a column headed by the word measured.
    """
    return "measured" if row.plan.measured.t_total_measured else "stated"


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
    as a trailing comment. The comments are not decoration: the four numbers currently
    in ``config.py`` were derived by hand once, and without their derivations beside
    them nobody could tell later which measurement they came from or whether a new one
    invalidated them.
    """
    measured, scenario = plan.measured, plan.scenario
    lambda_cap = plan.c_max / measured.s_mean_s
    return "\n".join(
        [
            f'    "{measured.model_name}": ModelEndpointConfig(',
            f'        model_name="{measured.model_name}",',
            f'        instance_type="{plan.instance_type}",',
            f"        min_instances={plan.min_instances},"
            f"  # trough {_load_str(scenario.trough_rps, scenario.trough_streams)}",
            f"        max_instances={plan.max_instances},"
            f"  # peak {_load_str(scenario.peak_rps, scenario.peak_streams)}",
            f"        scaling_target_value={plan.c_target:.3f},"
            f"  # {plan.binding_constraint}: derate {scenario.derate} x C_max "
            f"{plan.c_max:.2f} / k {scenario.growth_factor_k:g}",
            f"        scale_in_threshold={_scale_in_threshold(plan):.3f},"
            "  # well under the target so the two policies do not oscillate",
            f"        ttfab_slo_ms={scenario.ttfab_slo_ms},"
            f"  # end-to-end promise; W_max {plan.w_max_s:.2f}s + p95 service "
            f"{measured.s_p95_s:.2f}s fits the "
            f"{SAGEMAKER_INVOCATION_CEILING_S:.0f}s invocation ceiling",
            f"        ttfab_budget_ms={scenario.ttfab_budget_ms},"
            "  # which measured budget C_max was read at",
            f"        queue_max_depth={plan.queue_max_depth},"
            f"  # Lambda_cap {lambda_cap:.2f} rps x W_max {plan.w_max_s:.2f}s "
            f"(= {scenario.ttfab_slo_ms / 1000:.1f}s SLO - {measured.s_p95_s:.3f}s p95)",
            f"        scale_out_cooldown_s={plan.scale_out_cooldown_s},"
            "  # short: target tracking adds one instance at a time",
            f"        scale_in_cooldown_s={plan.scale_in_cooldown_s},"
            f"  # long: removed capacity costs a full {measured.t_total_s:.0f}s to replace",
            "    ),",
        ]
    )


def _scale_in_threshold(plan: ScalingPlan) -> float:
    """Concurrency at or below which an instance is removed.

    A fraction of ``C_target`` rather than a constant, so it keeps its separation from
    the target when the target moves — a fixed 0.2 sits *above* a ``C_target`` of 0.15
    and the two policies then fight each other.
    """
    return round(plan.c_target * 0.3, 3)


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


def render_plan(
    rows: Sequence[SweepRow],
    stages: TTotalStages | None = None,
) -> str:
    """The whole report: inputs, sweep, then findings and a config block per row.

    Findings are printed per row rather than once because they are not row-invariant —
    the 60s ceiling verdict and the queue-covers-surge verdict both move with ``k``,
    which is exactly what a sweep exists to show.
    """
    if not rows:
        return "No plans to report."

    sections = [render_inputs(rows[0].plan, stages), "", render_sweep(rows), ""]

    for row in rows:
        label = (
            f"provision {row.provision_s:.0f}s, k={row.k:g}"
            if row.provision_s is not None
            else f"T_total as {_lag_label(row)}, k={row.k:g}"
        )
        sections.extend(
            [
                f"=== {label} " + "=" * max(0, 60 - len(label)),
                "",
                render_findings(row.plan),
                "",
                render_config(row.plan),
                "",
            ]
        )
    return "\n".join(sections)


def plan_to_dict(row: SweepRow) -> dict[str, object]:
    """One sweep row as JSON, for an artifact.

    Flattens the assumptions in beside the plan: a row read back without its
    ``provision_s`` cannot be told from a measured one, and that is the distinction
    the whole sweep exists to preserve.
    """
    plan = row.plan
    return {
        "provision_s": row.provision_s,
        "provision_assumed": row.provision_assumed,
        # Not the same question as `provision_assumed`: a row with no substituted
        # provision time may have had its whole lag stated instead, and the two read
        # identically without this.
        "t_total_measured": plan.measured.t_total_measured,
        "growth_factor_k": row.k,
        "t_total_s": row.t_total_s,
        "plan": plan.model_dump(mode="json"),
        "verdict": _verdict_of(plan),
        "config_block": render_config(plan),
    }
