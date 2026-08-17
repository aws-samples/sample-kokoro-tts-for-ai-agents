# Copyright (c) 2026 Amazon.com
# This file is licensed under the MIT License.
# See the LICENSE file in the project root for full license information.

"""Tests for the planner: composition, refusals, and the findings.

The planner owns no equations — every one lives in `shared.capacity` and is tested there
against the worked examples. So these tests are about the things that can go wrong
*between* the equations:

- **Composition.** The two derived thresholds, the queue bound, the wait budget and the
  fleet must come out where hand-derivation puts them. `Q_max = 50` at `k = 1.25` gives
  `(37.5, 25.0)` and a safe floor of 3; that triple is the regression anchor for the
  whole chain, because the numbers `config.py` shipped before it (`0.713`, `41`) were
  hand-derived once and this tool exists to stop that happening again.
- **The refusals.** A `Q_max` measured against one SLO planned against another, a
  ladder and a lag from different configurations, and a `W_max` past the 60s invocation
  ceiling. All three produce a plan for something that exists nowhere.
- **Provenance.** A stated lag must not pass for a measured one. `t_total_measured` is a
  separate flag from `provenance.origin` precisely because `Q_max` stays measured when
  the lag was not, and the planner's own finding is the thing that says which.
- **The two known limits of the simple rule.** `surge_survival` and `scale_in_safety`
  exist to make them falsifiable rather than argued, so the tests assert the numbers
  they report, not just their verdicts.

`caplog` is not used: loguru does not propagate to the stdlib logging tree, so an
assertion against it would pass whether or not anything was emitted. The `logged`
fixture adds a real sink.
"""

from __future__ import annotations

import math
from typing import Any

import pytest
from loguru import logger

from shared.capacity import (
    CLOUDWATCH_HIGH_RES_PERIOD_S,
    SAGEMAKER_INVOCATION_CEILING_S,
    scale_thresholds,
)
from tts_bench.planner import (
    PROVISION_FROM_STAGE,
    PROVISION_TO_STAGE,
    Q_MAX_SPREAD_WARN,
    SHED_PROBABILITY_WARN,
    PlannerError,
    TTotalStages,
    assert_pairable,
    measured_from_artifacts,
    plan_one,
)
from tts_bench.types import (
    Measured,
    Origin,
    Provenance,
    QMaxReport,
    Scenario,
    StepSummary,
    Verdict,
)

#: The measured kokoro numbers, unrounded. ``S`` is the conflated figure the ladder
#: records — 34ms client round trip plus ~59ms of service — which is the right quantity
#: for a client-observed SLO and a slight over-estimate anywhere it stands in for
#: server-side work.
S_MEAN_S = 0.10986375146305409
S_P95_S = 0.16457688123919073
SLUG = "g5xlarge-139b9068"
DIGEST = "139b9068c5eb1f03c8312c17391dc35838e43e2417ae80d61e628e6ffb3d6a28"

#: ``Q_max`` the default fixture measures, and the two thresholds it derives. Named
#: rather than inlined because nearly every test needs at least one of them, and a test
#: that re-derived them by hand would agree with a broken planner.
Q_MAX = 50
C_SCALE_MAX = 37.5
C_SCALE_MIN = 25.0

#: W_max the default scenario derives: 3.0s SLO - 0.1646s p95 service. Named because
#: several tests need the number the plan will actually use, and re-deriving it in each
#: one was how a test came to assert against a wait budget the plan had not chosen.
W_MAX_S = 3.0 - S_P95_S

#: A ladder shaped roughly like M/M/1 on kokoro's measured service time, with rungs at
#: the concurrencies the workflow depends on: 1 for the alarm threshold, 5 and 10 for
#: ttotal's halving test, and 50 sitting just inside the 3000ms SLO.
LADDER = {1: 92.0, 5: 379.0, 10: 667.0, 20: 1243.0, 50: 2910.0}


@pytest.fixture
def logged():
    """Captured loguru warnings.

    ``caplog`` does not see these — loguru does not propagate to the stdlib logging
    tree — so an assertion against it would pass whether or not anything was emitted.
    """
    records: list[str] = []
    sink_id = logger.add(lambda message: records.append(message.record["message"]), level="WARNING")
    try:
        yield records
    finally:
        logger.remove(sink_id)


def _measured(**overrides: Any) -> Measured:
    """A trustworthy joined measurement: frozen, one instance, unbounded queue.

    Trustworthy by default so that a test about something else does not have to read
    past a ``measurement_trust`` warning to find its own finding. The tests that care
    about the warnings turn each precondition off individually.
    """
    fields: dict[str, Any] = {
        "model_name": "kokoro-82m",
        "endpoint": "speech-kokoro-82m",
        "instance_type": "ml.g5.xlarge",
        "q_max": Q_MAX,
        "q_max_bracketed": True,
        "slo_ms": 3000,
        "ttfab_p95_at_c1_ms": LADDER[1],
        "s_mean_s": S_MEAN_S,
        "s_p95_s": S_P95_S,
        "t_total_s": 300.0,
        "chars_per_request": 25.0,
        "q_max_spread": 0.0,
        "runs_contributing": 2,
        "ladder_p95_ms": dict(LADDER),
        "frozen": True,
        "unbounded_queue": True,
        "instance_counts_observed": (1,),
        "transport": "bidi",
        "deployed_config": {
            "instance_type": "ml.g5.xlarge",
            "image_digest": DIGEST,
            "container_env": {},
        },
    }
    fields.update(overrides)
    return Measured(**fields)


def _scenario(**overrides: Any) -> Scenario:
    """The default scenario, at the SLO the default measurement was measured against.

    ``ttfab_slo_ms`` has to match ``_measured``'s ``slo_ms`` or every plan refuses — see
    ``TestSloMustMatchTheMeasurement``. A test overriding one generally overrides both,
    which :func:`_plan_at_slo` does in one place.
    """
    fields: dict[str, Any] = {
        "peak_rps": 450.0,
        "trough_rps": 30.0,
        "max_scaling_per_t_total": 1.25,
        "ttfab_slo_ms": 3000,
    }
    fields.update(overrides)
    return Scenario(**fields)


def _plan_at_slo(slo_ms: int, **measured_overrides: Any):
    """A plan at a different SLO, with the measurement re-labelled to match.

    Q_max is *defined by* the SLO, so moving one without the other is the mismatch the
    planner refuses. These tests are about what the SLO does to the wait budget and the
    ceiling, not about the refusal, so they move both.
    """
    return plan_one(
        _measured(slo_ms=slo_ms, **measured_overrides),
        _scenario(ttfab_slo_ms=slo_ms),
    )


def _slo_for_wait(wait_s: float, s_p95_s: float = S_P95_S) -> int:
    """The SLO that leaves ``wait_s`` of queueing budget, in ms.

    W_max is not settable, so a test that wants a particular queue allowance has to
    state the SLO that produces it. Inverting ``w_max_for_slo`` here rather than in each
    test keeps the relation in one place; the tests still assert against ``plan.w_max_s``,
    so a broken inversion shows up as a failure rather than as two wrongs agreeing.
    """
    return round((wait_s + s_p95_s) * 1000)


def _stages(**overrides: Any) -> TTotalStages:
    fields: dict[str, Any] = {
        "total_s": 420.0,
        "provision_s": 180.0,
        "trigger": "force-desired",
        "config_slug": SLUG,
        "run_id": "ttotal123",
    }
    fields.update(overrides)
    return TTotalStages(**fields)


def _finding(plan, name: str):
    for finding in plan.findings:
        if finding.name == name:
            return finding
    raise AssertionError(f"no finding named {name!r} in {[f.name for f in plan.findings]}")


class TestPlanOneComposition:
    def test_derives_both_thresholds_from_q_max_and_the_surge_ratio(self) -> None:
        # The regression anchor for the whole chain. h = 0.25, so C_scale_max is
        # (1-h) x 50 and C_scale_min is (1-2h) x 50 -- and both must arrive on the plan
        # itself, not just inside a finding, because the plan is what gets pasted into
        # config.py. These replace the hand-derived 0.713 and 41 that shipped before.
        plan = plan_one(_measured(), _scenario())
        assert plan.q_max == Q_MAX
        assert plan.c_scale_max == pytest.approx(C_SCALE_MAX)
        assert plan.c_scale_min == pytest.approx(C_SCALE_MIN)
        assert plan.min_safe_instances == 3
        assert not plan.q_max_is_lower_bound

    def test_agrees_with_shared_capacity_rather_than_reimplementing_it(self) -> None:
        # The planner composes; it must not carry a second copy of the arithmetic.
        # Asserted against the function rather than against 37.5 so that a change to
        # the rule fails here as a disagreement rather than passing in one of two places.
        plan = plan_one(_measured(), _scenario())
        expected = scale_thresholds(Q_MAX, 1.25)
        assert (plan.c_scale_max, plan.c_scale_min, plan.min_safe_instances) == (
            pytest.approx(expected.c_scale_max),
            pytest.approx(expected.c_scale_min),
            expected.min_safe_instances,
        )

    def test_the_queue_bound_is_q_max_itself(self) -> None:
        # Not derived from the wait budget: past Q_max a request cannot reach first byte
        # inside the SLO, so admitting it buys a late success instead of an honest 503.
        plan = plan_one(_measured(), _scenario())
        assert plan.queue_max_depth == plan.q_max == Q_MAX

    def test_the_wait_budget_is_the_slo_less_p95_service(self) -> None:
        plan = plan_one(_measured(), _scenario())
        assert plan.w_max_s == pytest.approx(W_MAX_S)

    def test_the_wait_budget_follows_the_slo_and_nothing_else(self) -> None:
        # The property the reframe buys: there is no second latency field, so nothing
        # can state a queue allowance that disagrees with the promise.
        tight = _plan_at_slo(1000)
        loose = _plan_at_slo(10_000)
        assert tight.w_max_s == pytest.approx(1.0 - S_P95_S)
        assert loose.w_max_s == pytest.approx(10.0 - S_P95_S)
        # And Q_max is unchanged by it, because Q_max is a measurement: what moves is
        # whether the ladder's answer still looks consistent -- see queue_depth.
        assert tight.queue_max_depth == loose.queue_max_depth == Q_MAX

    def test_a_looser_surge_ratio_scales_out_earlier_and_needs_more_instances(self) -> None:
        # The direction a reader relies on: reserving more headroom means firing sooner
        # and buying more. Asserted as a property because the exact fleet sizes are
        # n_instances' business.
        tight = plan_one(_measured(), _scenario(max_scaling_per_t_total=1.05))
        loose = plan_one(_measured(), _scenario(max_scaling_per_t_total=1.4))
        assert loose.c_scale_max < tight.c_scale_max
        assert loose.c_scale_min < tight.c_scale_min
        assert loose.peak_instances >= tight.peak_instances

    def test_a_flat_scenario_puts_both_thresholds_at_q_max(self) -> None:
        # k=1 reserves no headroom, so there is no gap to scale in through. The plan is
        # still coherent -- it is the "never scale, just hold the SLO" configuration --
        # and scale_in_safety is what says the flap has nowhere to land.
        plan = plan_one(_measured(), _scenario(max_scaling_per_t_total=1.0))
        assert plan.c_scale_max == plan.c_scale_min == pytest.approx(float(Q_MAX))
        assert plan.min_safe_instances is None

    def test_a_surge_ratio_with_no_usable_thresholds_is_refused(self) -> None:
        # 1.5 puts C_scale_min at zero, which deploys as "never scale in". Pydantic
        # rejects it on the Scenario, so the planner's own refusal is unreachable from
        # the CLI -- this asserts the guard exists rather than that it is the only one.
        with pytest.raises(ValueError, match="less than 1.5"):
            _scenario(max_scaling_per_t_total=1.5)

    def test_fleet_sizes_bracket_the_stated_load(self) -> None:
        # 450 rps x 0.11s = 49.4 concurrent, at 37.5 per instance -> 2. The trough's
        # 30 rps is 3.3 concurrent -> 1.
        plan = plan_one(_measured(), _scenario())
        assert plan.peak_instances == 2
        assert plan.trough_instances == 1
        assert plan.min_instances == 1
        assert plan.max_instances == 2

    def test_the_fleet_is_sized_on_the_threshold_not_on_q_max(self) -> None:
        # Sizing at Q_max sizes the fleet to sit exactly where the SLO breaks. The
        # difference is the whole cost of the headroom, and fleet_cost reports it.
        plan = plan_one(_measured(), _scenario())
        assert plan.peak_instances == 2
        assert (
            "2 instances at C_scale_max 37.50 against 1 at Q_max"
            in _finding(plan, "fleet_cost").detail
        )

    def test_min_is_the_trough_fleet_not_one(self) -> None:
        # A reserved-capacity account pays the floor whatever the traffic does, so a
        # stated trough that needs three instances must raise min -- defaulting to 1
        # would under-reserve exactly the capacity that was reserved on purpose.
        plan = plan_one(_measured(), _scenario(peak_rps=1000.0, trough_rps=700.0))
        assert plan.trough_instances == 3
        assert plan.min_instances == 3

    def test_max_never_falls_below_min(self) -> None:
        # A peak below the floor is a coherent scenario (over-reserved on purpose),
        # and max < min is not a deployable config.
        plan = plan_one(_measured(), _scenario(peak_rps=0.5, trough_rps=0.5, min_instances_floor=3))
        assert plan.max_instances >= plan.min_instances == 3

    def test_max_instances_is_not_capped_by_anything_but_the_scenario(self) -> None:
        # The decoupling result: thresholds are per-instance and do not move with fleet
        # size, so supporting 100x the traffic is a larger max_instances and nothing
        # else. Nothing in the plan may quietly clamp it to a quota.
        plan = plan_one(_measured(), _scenario(peak_rps=45_000.0, trough_rps=30.0))
        assert plan.max_instances == plan.peak_instances == 132
        assert plan.c_scale_max == pytest.approx(C_SCALE_MAX)
        assert plan.c_scale_min == pytest.approx(C_SCALE_MIN)

    def test_streams_win_over_a_rate(self) -> None:
        # A stated stream count is a direct concurrency observation; lambda x S is the
        # same quantity inferred. For bidi the inference is the weaker of the two.
        by_streams = plan_one(_measured(), _scenario(peak_rps=450.0, peak_streams=80.0))
        assert by_streams.peak_instances == 3  # ceil(80 / 37.5)

    def test_utilization_is_recorded_because_the_threshold_hides_it(self) -> None:
        # 37.5 in the queue is L/(1+L) = 97.4% utilized, not "three quarters of the way
        # to trouble". Recorded on the plan because that is not visible from 37.5.
        plan = plan_one(_measured(), _scenario())
        assert plan.utilization_at_c_scale_max == pytest.approx(0.974, abs=5e-4)

    def test_cooldowns_are_asymmetric(self) -> None:
        # Scale-out must not be gated on the full lag: target tracking adds one
        # instance at a time, so a surge needing three would take three lags.
        # Scale-in is deliberately long -- removed capacity costs a full T_total.
        plan = plan_one(_measured(t_total_s=300.0), _scenario())
        assert plan.scale_out_cooldown_s == 30
        assert plan.scale_in_cooldown_s == 900
        assert plan.scale_out_cooldown_s < plan.scale_in_cooldown_s

    def test_scale_out_cooldown_never_below_the_metric_period(self) -> None:
        # A policy cannot react to data it has not received.
        plan = plan_one(_measured(t_total_s=1.0), _scenario())
        assert plan.scale_out_cooldown_s >= CLOUDWATCH_HIGH_RES_PERIOD_S

    def test_scale_in_cooldown_is_floored_at_five_minutes(self) -> None:
        # A fast T_total must not produce a twitchy scale-in: the flap
        # scale_in_safety warns about is damped by this cooldown, not by the lag.
        assert plan_one(_measured(t_total_s=1.0), _scenario()).scale_in_cooldown_s == 300

    def test_cost_is_the_peak_fleet(self) -> None:
        plan = plan_one(_measured(), _scenario())
        assert plan.peak_cost_per_hour > 0
        assert plan.peak_cost_per_m_chars > 0

    def test_unknown_chars_per_request_gives_an_infinite_unit_cost(self) -> None:
        # inf rather than 0: an endpoint whose throughput was never recorded has no
        # meaningful unit cost, and 0 would make a broken measurement look free.
        plan = plan_one(_measured(chars_per_request=0.0), _scenario())
        assert plan.peak_cost_per_m_chars == math.inf
        assert plan.peak_cost_per_hour > 0

    def test_the_inputs_are_carried_on_the_plan(self) -> None:
        # The artifact is what a later reader has. A plan that dropped its inputs could
        # not be checked against them, and re-deriving the six variables from the
        # outputs is exactly the hand-arithmetic this tool replaces.
        measured, scenario = _measured(), _scenario()
        plan = plan_one(measured, scenario)
        assert plan.measured == measured
        assert plan.scenario == scenario


class TestSloMustMatchTheMeasurement:
    """One SLO, one number — enforced across the artifact boundary.

    ``Q_max`` is *defined by* the SLO: it is the highest rung whose p95 stayed inside
    that line. So a ladder measured at 3000ms says nothing about a 1000ms promise, and
    reading it against one is the mistake two independent latency fields used to permit.
    """

    def test_a_mismatched_slo_is_refused(self) -> None:
        with pytest.raises(PlannerError, match="measured against a 3000ms SLO"):
            plan_one(_measured(slo_ms=3000), _scenario(ttfab_slo_ms=1000))

    def test_the_refusal_names_both_values_and_both_fixes(self) -> None:
        # Either direction is a legitimate resolution -- plan at what was measured, or
        # measure at what you want to plan -- so the message has to offer both.
        with pytest.raises(PlannerError) as excinfo:
            plan_one(_measured(slo_ms=3000), _scenario(ttfab_slo_ms=1000))
        message = str(excinfo.value)
        assert "3000ms" in message
        assert "1000ms" in message
        assert "--ttfab-slo-ms 3000" in message
        assert "qmax --slo-ms 1000" in message

    def test_a_matching_slo_plans(self) -> None:
        assert plan_one(_measured(slo_ms=1500), _scenario(ttfab_slo_ms=1500)).q_max == Q_MAX

    def test_the_refusal_comes_before_any_arithmetic(self) -> None:
        # A measurement that is also infeasible must still report the mismatch: the
        # mismatch invalidates the input, so any finding computed from it would be a
        # statement about a ladder that was never run.
        with pytest.raises(PlannerError, match="Q_max was measured"):
            plan_one(_measured(slo_ms=3000, s_p95_s=61.0), _scenario(ttfab_slo_ms=100))


class TestInvocationCeiling:
    def test_a_wait_past_the_ceiling_is_infeasible_not_a_warning(self) -> None:
        # The hard constraint. Those requests wait the full W_max and then fail
        # anyway, which is worse than refusing them at admission. Reached by an SLO past
        # the platform ceiling rather than a hand-set wait: with W_max derived, the
        # deadline *is* the SLO, so a 70s promise cannot be kept whatever the queue does.
        plan = _plan_at_slo(70_000)
        assert _finding(plan, "invocation_ceiling").verdict is Verdict.INFEASIBLE
        assert plan.infeasible

    def test_an_infeasible_plan_says_what_would_work(self) -> None:
        plan = _plan_at_slo(70_000)
        recommendation = _finding(plan, "invocation_ceiling").recommendation or ""
        assert "--ttfab-slo-ms" in recommendation
        # And the number it names must itself pass, or it is advice to build another
        # infeasible plan.
        largest = int(recommendation.split("--ttfab-slo-ms")[1].split()[0])
        assert not _plan_at_slo(largest).infeasible

    def test_the_stated_slo_is_feasible(self) -> None:
        plan = plan_one(_measured(), _scenario())
        assert _finding(plan, "invocation_ceiling").verdict is Verdict.OK
        assert not plan.infeasible

    def test_judges_p95_service_not_mean(self) -> None:
        # A mean-sized deadline is missed by half the requests that reach it. This model
        # has a 0.5s p95 against a 0.05s mean, and at a 60s SLO the ceiling check sees
        # 59.5s of queue plus that 0.5s tail -- exactly 60s, and one step looser breaks.
        assert _plan_at_slo(60_001, s_mean_s=0.05, s_p95_s=0.5).infeasible
        assert not _plan_at_slo(60_000, s_mean_s=0.05, s_p95_s=0.5).infeasible

    def test_a_model_slower_than_the_ceiling_cannot_be_rescued(self) -> None:
        # p95 service of 61s exceeds the 60s ceiling on its own, so there is no SLO to
        # recommend -- an unqueued request already fails.
        plan = _plan_at_slo(70_000, s_mean_s=30.0, s_p95_s=61.0)
        recommendation = _finding(plan, "invocation_ceiling").recommendation or ""
        assert "no SLO or queue length fixes this" in recommendation

    def test_an_explicit_ceiling_is_honoured(self) -> None:
        plan = plan_one(_measured(slo_ms=20_000), _scenario(ttfab_slo_ms=20_000), ceiling_s=10.0)
        assert plan.infeasible

    def test_the_default_ceiling_is_sagemakers(self) -> None:
        assert SAGEMAKER_INVOCATION_CEILING_S == 60.0

    def test_the_plan_records_the_ceiling_the_finding_judged_against(self) -> None:
        # `scale_report` prints this number as a comment on ttfab_slo_ms, and that block
        # is pasted into config.py verbatim. Reading the constant there instead named
        # 60s on a run judged at 10 -- a comment contradicting the finding above it,
        # which is how a policy gets deployed against a limit nobody checked. Asserted
        # against the finding's own text so the field and the check cannot drift apart:
        # there is one ceiling per run, not two that happen to agree.
        plan = plan_one(_measured(slo_ms=20_000), _scenario(ttfab_slo_ms=20_000), ceiling_s=10.0)
        assert plan.ceiling_s == 10.0
        detail = _finding(plan, "invocation_ceiling").detail
        assert f"{plan.ceiling_s:.0f}s SageMaker invocation ceiling" in detail

    def test_an_unstated_ceiling_is_recorded_as_sagemakers(self) -> None:
        # The constant is the default, so an ordinary run is unchanged by carrying it.
        plan = plan_one(_measured(), _scenario())
        assert plan.ceiling_s == SAGEMAKER_INVOCATION_CEILING_S


class TestMeasurementTrust:
    """Whether ``Q_max`` reads as one instance's own limit.

    First finding because it invalidates the rest: both thresholds are fractions of
    ``Q_max`` and every fleet size divides by it, so a ladder that measured something
    else makes the whole plan a description of nothing.
    """

    def test_a_trustworthy_measurement_passes(self) -> None:
        finding = _finding(plan_one(_measured(), _scenario()), "measurement_trust")
        assert finding.verdict is Verdict.OK

    def test_an_unfrozen_ladder_warns(self) -> None:
        finding = _finding(plan_one(_measured(frozen=False), _scenario()), "measurement_trust")
        assert finding.verdict is Verdict.WARN
        assert "not suspended" in finding.detail

    def test_a_resized_fleet_warns(self) -> None:
        finding = _finding(
            plan_one(_measured(instance_counts_observed=(1, 2)), _scenario()), "measurement_trust"
        )
        assert finding.verdict is Verdict.WARN
        assert "resized mid-run" in finding.detail

    def test_a_bounded_container_queue_warns(self) -> None:
        # Q_max is the depth at which the SLO breaks. A container that sheds first
        # measures its own MAX_QUEUE_DEPTH instead, which is a different number that
        # looks exactly the same on the ladder.
        finding = _finding(
            plan_one(_measured(unbounded_queue=False), _scenario()), "measurement_trust"
        )
        assert finding.verdict is Verdict.WARN
        assert "bounds its admission queue" in finding.detail

    def test_an_unchecked_queue_bound_is_not_a_pass(self) -> None:
        # None, not False: the check never ran. Treating that as clear is how a bounded
        # container would go unnoticed, since the ladder cannot tell the two apart.
        finding = _finding(
            plan_one(_measured(unbounded_queue=None), _scenario()), "measurement_trust"
        )
        assert finding.verdict is Verdict.WARN
        assert "never checked" in finding.detail

    def test_every_failing_precondition_is_named(self) -> None:
        # Not the first one only: they have different fixes, and an operator who
        # re-runs with --require-frozen alone would get the same warning back.
        finding = _finding(
            plan_one(
                _measured(frozen=False, unbounded_queue=False, instance_counts_observed=(1, 4)),
                _scenario(),
            ),
            "measurement_trust",
        )
        assert "not suspended" in finding.detail
        assert "resized mid-run" in finding.detail
        assert "bounds its admission queue" in finding.detail
        assert "--require-frozen --require-unbounded-queue" in (finding.recommendation or "")


class TestRepeatability:
    """Whether ``Q_max`` is *stable*, which is a separate question from per-instance.

    Both thresholds are fractions of it, so a ladder that resolved noise produces
    confidently wrong thresholds — the failure looks identical to a good measurement
    from the plan alone.
    """

    def test_a_tight_spread_across_runs_passes(self) -> None:
        plan = plan_one(_measured(q_max_spread=0.0, runs_contributing=2), _scenario())
        finding = _finding(plan, "curve_repeatability")
        assert finding.verdict is Verdict.OK
        assert "the plan uses the minimum" in finding.detail

    def test_a_wide_spread_warns(self) -> None:
        plan = plan_one(_measured(q_max_spread=0.45, runs_contributing=2), _scenario())
        finding = _finding(plan, "curve_repeatability")
        assert finding.verdict is Verdict.WARN
        assert "45%" in finding.detail

    def test_a_single_run_is_suppressed_not_ok(self) -> None:
        # The defect this exists for: spread is 0% when only one pass contributed,
        # which otherwise reads exactly like two passes agreeing perfectly.
        plan = plan_one(_measured(q_max_spread=0.0, runs_contributing=1), _scenario())
        finding = _finding(plan, "curve_repeatability")
        assert finding.verdict is Verdict.SUPPRESSED
        assert "not an agreement" in finding.detail
        assert "--runs 2" in (finding.recommendation or "")

    def test_the_warn_threshold_is_twenty_percent(self) -> None:
        assert Q_MAX_SPREAD_WARN == 0.2
        # Either side of it, at the same number of contributing runs.
        assert (
            _finding(
                plan_one(_measured(q_max_spread=0.19, runs_contributing=2), _scenario()),
                "curve_repeatability",
            ).verdict
            is Verdict.OK
        )
        assert (
            _finding(
                plan_one(_measured(q_max_spread=0.21, runs_contributing=2), _scenario()),
                "curve_repeatability",
            ).verdict
            is Verdict.WARN
        )


class TestQueueDepth:
    """That the admission bound shipped is the one that was measured.

    There is no arithmetic to check — ``queue_max_depth`` *is* ``Q_max`` — so the
    finding exists to say that, and to cross-check it against the depth the SLO's own
    wait budget affords. The two agree by construction when the ladder bracketed its
    answer, and any disagreement is the size of the extrapolation.
    """

    def test_a_bracketed_ladder_passes_and_shows_the_cross_check(self) -> None:
        # W_max 2.835s / S 0.110s implies room for about 26, against a measured 50. The
        # gap is real and worth printing: S is conflated with a 34ms round trip, so the
        # implied figure runs low.
        finding = _finding(plan_one(_measured(), _scenario()), "queue_depth")
        assert finding.verdict is Verdict.OK
        assert "bracketed from above" in finding.detail
        assert "about 26" in finding.detail

    def test_an_unbracketed_ladder_warns_that_q_max_is_a_lower_bound(self) -> None:
        # The ladder ran out while still passing, so real capacity is at least this.
        # Conservative in the safe direction -- the policy adds instances sooner than it
        # needs to -- but it costs money, so it is a warning rather than silence.
        plan = plan_one(_measured(q_max_bracketed=False), _scenario())
        finding = _finding(plan, "queue_depth")
        assert finding.verdict is Verdict.WARN
        assert "LOWER bound" in finding.detail
        assert "--concurrency rungs above 50" in (finding.recommendation or "")

    def test_the_lower_bound_flag_reaches_the_plan_not_only_the_finding(self) -> None:
        # The artifact is what a later reader has; a warning printed once to a terminal
        # is not a record of it.
        assert plan_one(_measured(q_max_bracketed=False), _scenario()).q_max_is_lower_bound
        assert not plan_one(_measured(q_max_bracketed=True), _scenario()).q_max_is_lower_bound


class TestThresholdUnits:
    """The shipped defect this finding exists for.

    ``C_scale_max`` is a client-measured occupancy; the deployed alarm compares
    ``ConcurrentRequestsPerModel`` / *Maximum* over 10s against its threshold. Deploying
    the unconverted figure is how 0.713 reached the endpoint — a value that inverts to a
    negative arrival rate, so no traffic satisfies it and target tracking asks for the
    whole fleet on one request.
    """

    def test_a_measured_ratio_converts_the_threshold(self) -> None:
        plan = plan_one(_measured(cw_units_ratio_by_rung={50: 1.35}), _scenario())
        assert plan.cw_units_ratio == pytest.approx(1.35)
        assert plan.c_scale_max_in_cw_units == pytest.approx(37.5 * 1.35)
        finding = _finding(plan, "threshold_units")
        assert finding.verdict is Verdict.OK
        assert "50.62" in finding.detail

    def test_the_converted_number_is_what_deploys(self) -> None:
        # Both figures are on the plan, and they must not be confused: the occupancy is
        # what was measured, the conversion is what the alarm reads.
        plan = plan_one(_measured(cw_units_ratio_by_rung={50: 1.35}), _scenario())
        assert plan.c_scale_max_in_cw_units != pytest.approx(plan.c_scale_max)
        assert "The converted figure is what" in _finding(plan, "threshold_units").detail

    def test_the_rung_nearest_the_threshold_is_used(self) -> None:
        # The ratio is not a constant -- it ran 9.8x at low load to 1.35x at high load
        # across one kokoro ladder -- so it has to be read where the threshold sits, not
        # averaged across the ladder.
        plan = plan_one(_measured(cw_units_ratio_by_rung={5: 9.8, 10: 5.1, 50: 1.35}), _scenario())
        assert plan.cw_units_ratio == pytest.approx(1.35)

    def test_a_tie_between_rungs_takes_the_higher_one(self) -> None:
        # 37.5 is equidistant from 25 and 50. The ratio shrinks as load rises, so the
        # higher rung gives the smaller multiplier and the tighter threshold.
        plan = plan_one(_measured(cw_units_ratio_by_rung={25: 3.0, 50: 1.35}), _scenario())
        assert plan.cw_units_ratio == pytest.approx(1.35)

    def test_no_server_statistic_is_suppressed_not_ok(self) -> None:
        # The conversion did not happen, which is not the same as not needing one.
        # SUPPRESSED, and the recommendation says it cannot be backfilled: high-res
        # datapoints retain 3 hours.
        plan = plan_one(_measured(cw_units_ratio_by_rung={}), _scenario())
        assert plan.c_scale_max_in_cw_units is None
        assert plan.cw_units_ratio is None
        finding = _finding(plan, "threshold_units")
        assert finding.verdict is Verdict.SUPPRESSED
        assert "0.713" in finding.detail
        assert "retain 3 hours" in (finding.recommendation or "")


class TestSurgeSurvival:
    """The first known limit of the simple rule, made falsifiable.

    ``C_scale_max`` reserves headroom in queue *slots*, a finite stock, while surviving
    a surge is a question about drain *rate*, a flow. Occupancy converts to utilization
    steeply, so three quarters of ``Q_max`` is 97% utilized rather than three quarters of
    the way to trouble. Simulated rather than argued, because the simulation has a number.
    """

    def test_the_default_threshold_warns_with_the_simulated_probability(self) -> None:
        # 88% at the numbers this branch measured. The finding is the whole reason the
        # simple rule is shippable: it says out loud what the rule costs.
        plan = plan_one(_measured(t_total_s=231.0), _scenario())
        finding = _finding(plan, "surge_survival")
        assert finding.verdict is Verdict.WARN
        assert plan.shed_probability_at_c_scale_max == pytest.approx(0.855, abs=0.01)
        assert "86%" in finding.detail
        assert "97.4% utilization" in finding.detail
        assert "scale out earlier" in (finding.recommendation or "")

    def test_a_short_lag_survives(self) -> None:
        # The same threshold is fine when help arrives quickly -- which is why the
        # finding is about the pair rather than about C_scale_max alone.
        plan = plan_one(_measured(t_total_s=0.5), _scenario())
        finding = _finding(plan, "surge_survival")
        assert finding.verdict is Verdict.OK
        assert plan.shed_probability_at_c_scale_max is not None
        assert plan.shed_probability_at_c_scale_max <= SHED_PROBABILITY_WARN

    def test_the_probability_rises_with_the_lag(self) -> None:
        # The monotonicity a reader relies on. Asserted as a property because the
        # simulation's exact values are shared.capacity's business.
        probabilities = [
            plan_one(_measured(t_total_s=t), _scenario()).shed_probability_at_c_scale_max
            for t in (1.0, 10.0, 60.0, 300.0)
        ]
        assert probabilities == sorted(probabilities)

    def test_it_is_deterministic_across_runs(self) -> None:
        # A published number that moved between two invocations of the same command
        # would be indistinguishable from a changed measurement.
        first = plan_one(_measured(), _scenario()).shed_probability_at_c_scale_max
        second = plan_one(_measured(), _scenario()).shed_probability_at_c_scale_max
        assert first == second

    def test_the_warn_line_is_one_in_ten(self) -> None:
        # Not zero: a queue is a stochastic object and some tail risk is the price of
        # running one at all.
        assert SHED_PROBABILITY_WARN == 0.1


class TestScaleInSafety:
    """The second known limit: scale-in redistributes load rather than removing it.

    Dropping one of ``N`` multiplies each survivor's concurrency by ``N/(N-1)``, so at
    ``N=2`` the survivor inherits *double*. At a 1.25 ratio the thresholds are 0.75 and
    0.5 of ``Q_max``, so 2->1 lands exactly on ``Q_max`` — and kokoro runs
    ``min_instances=1``, which makes 2->1 the common case rather than the corner one.
    """

    def test_kokoros_floor_of_one_warns(self) -> None:
        plan = plan_one(_measured(), _scenario())
        finding = _finding(plan, "scale_in_safety")
        assert finding.verdict is Verdict.WARN
        assert plan.min_instances == 1
        assert "only stable from 3 instances up" in finding.detail
        assert "2->1" in finding.detail
        # The survivor inherits 25.0 x 2 / 1 = 50.0, which is Q_max exactly.
        assert "50.00" in finding.detail
        # Named as the flag that exists: `--min-floor` (dest min_instances_floor). Advice
        # naming a flag click would reject is advice nobody can follow.
        assert "raise --min-floor to 3" in (finding.recommendation or "")

    def test_a_floor_at_the_safe_size_passes(self) -> None:
        plan = plan_one(_measured(), _scenario(min_instances_floor=3))
        finding = _finding(plan, "scale_in_safety")
        assert finding.verdict is Verdict.OK
        assert plan.min_safe_instances == 3
        # 25.0 x 3 / 2 = 37.5, which is C_scale_max -- safe by a hair, and that is what
        # min_safe_instances means.
        assert "37.50" in finding.detail

    def test_the_ratio_test_is_stated_so_it_can_be_checked(self) -> None:
        # N/(N-1) <= C_scale_max / C_scale_min. Printing the ratio is what lets a
        # reader verify the floor rather than take it.
        finding = _finding(plan_one(_measured(), _scenario()), "scale_in_safety")
        assert "N/(N-1)" in finding.detail
        assert "1.50" in finding.detail  # 37.5 / 25.0

    def test_a_flat_scenario_has_no_safe_size_at_all(self) -> None:
        # At k=1 the thresholds coincide, so every scale-in scales straight back out.
        # Distinct from "the floor is too low": no floor helps.
        plan = plan_one(_measured(), _scenario(max_scaling_per_t_total=1.0))
        finding = _finding(plan, "scale_in_safety")
        assert finding.verdict is Verdict.WARN
        assert plan.min_safe_instances is None
        assert "no fleet size" in finding.detail
        assert "smaller --max-scaling-per-t-total" in (finding.recommendation or "")

    def test_the_safe_floor_falls_as_the_thresholds_separate(self) -> None:
        # A wider gap tolerates a smaller fleet, which is the trade the ratio buys.
        wide = plan_one(_measured(), _scenario(max_scaling_per_t_total=1.4))
        narrow = plan_one(_measured(), _scenario(max_scaling_per_t_total=1.1))
        assert wide.min_safe_instances is not None
        assert narrow.min_safe_instances is not None
        assert wide.min_safe_instances < narrow.min_safe_instances


class TestSloBudget:
    """That ``W_max`` is visibly derived, and what happens when there is none left.

    Stated as its own finding because the failure it catches is silent: a model whose p95
    already misses the promise reports ``W_max = 0``, and a reader seeing
    ``queue_max_depth`` beside it would take that for a design choice.
    """

    def test_shows_the_arithmetic_rather_than_the_result(self) -> None:
        finding = _finding(plan_one(_measured(), _scenario()), "slo_budget")
        assert finding.verdict is Verdict.OK
        assert "3.0s SLO - 0.165s p95 service" in finding.detail
        assert "derived, not configured" in finding.detail

    def test_a_model_whose_tail_misses_the_slo_is_infeasible(self) -> None:
        # p95 service is 165ms, so a 150ms promise is broken before a request waits at
        # all. INFEASIBLE rather than WARN because nothing this module emits moves it:
        # queue depth, instance count and policy all govern waiting, and there is none.
        plan = _plan_at_slo(150)
        finding = _finding(plan, "slo_budget")
        assert finding.verdict is Verdict.INFEASIBLE
        assert plan.infeasible
        assert plan.w_max_s == 0.0
        assert "No queue depth, instance count, or scaling policy rescues this" in finding.detail

    def test_the_infeasible_recommendation_names_an_slo_that_works(self) -> None:
        recommendation = _finding(_plan_at_slo(150), "slo_budget").recommendation or ""
        slo = int(recommendation.split("--ttfab-slo-ms above")[1].split(",")[0])
        assert _finding(_plan_at_slo(slo + 1), "slo_budget").verdict is not Verdict.INFEASIBLE

    def test_the_boundary_at_p95_service_falls_on_the_feasible_side(self) -> None:
        # W_max is 0 exactly at p95 service, and an SLO cannot be stated more finely than
        # a millisecond: 165ms is four tenths of a millisecond above kokoro's 164.577ms
        # p95, so the promise is technically keepable and this warns rather than refusing.
        # Pinned because "0.4ms of queue allowance" and "infeasible" are one rounding
        # apart, and a reader needs to know which side of the line the tool puts it on.
        plan = _plan_at_slo(round(S_P95_S * 1000))
        assert _finding(plan, "slo_budget").verdict is Verdict.WARN
        assert 0.0 < plan.w_max_s < 0.001
        # One millisecond tighter and there is no budget at all.
        assert _finding(_plan_at_slo(164), "slo_budget").verdict is Verdict.INFEASIBLE

    def test_a_wait_budget_under_one_service_time_warns(self) -> None:
        # Feasible but barely: a single request queued ahead already misses the SLO, so
        # the promise holds for an uncontended request and little more.
        finding = _finding(_plan_at_slo(_slo_for_wait(S_P95_S / 2)), "slo_budget")
        assert finding.verdict is Verdict.WARN
        assert "a single request queued ahead misses the SLO" in finding.detail

    def test_a_roomy_slo_passes(self) -> None:
        assert (
            _finding(_plan_at_slo(_slo_for_wait(S_P95_S * 3)), "slo_budget").verdict is Verdict.OK
        )


class TestFleetCost:
    """What the headroom costs, and why it never warns.

    Both fleets are the same demand over a different divisor, so the ratio is bounded by
    ``Q_max / C_scale_max`` = ``1 / (1 - h)``, and ``scale_thresholds`` refuses
    ``h >= 0.5``. So the continuous ceiling is under 2x and integer rounding reaches
    exactly 2x and no further — there is no reachable multiple at which "shorten T_total
    instead of buying headroom" becomes the cheaper advice. The finding reports the cost
    and leaves the judgement to the reader.
    """

    def test_cost_is_labelled_an_upper_bound(self) -> None:
        # On-demand rates: a reserved-capacity account pays less, and we can verify a
        # Pricing API number but not a contract rate.
        assert "upper bound" in _finding(plan_one(_measured(), _scenario()), "fleet_cost").detail

    def test_it_reports_both_fleet_sizes_so_the_ratio_can_be_checked(self) -> None:
        finding = _finding(plan_one(_measured(), _scenario()), "fleet_cost")
        assert "2.0x" in finding.detail
        assert "2 instances at C_scale_max 37.50 against 1 at Q_max" in finding.detail

    def test_a_flat_scenario_costs_nothing_extra(self) -> None:
        # At k=1 the two fleets are the same fleet: no headroom reserved, no premium.
        finding = _finding(
            plan_one(_measured(), _scenario(max_scaling_per_t_total=1.0)), "fleet_cost"
        )
        assert "1.0x" in finding.detail

    def test_it_never_warns_because_the_multiple_is_bounded(self) -> None:
        # The bound is structural, so this is a property over the reachable range rather
        # than a spot check: a WARN branch here would be dead code, and a reader seeing
        # only OK verdicts is entitled to know that is the model and not an unchecked
        # condition. The old C_max model could reach 4x, because its divisor moved with
        # k without a bound.
        for surge in (1.0, 1.05, 1.25, 1.4, 1.49):
            for peak in (0.5, 20.0, 345.0, 450.0, 1000.0, 45_000.0):
                plan = plan_one(
                    _measured(),
                    _scenario(peak_rps=peak, trough_rps=0.5, max_scaling_per_t_total=surge),
                )
                assert _finding(plan, "fleet_cost").verdict is Verdict.OK


class TestProvisionStage:
    """Which parts of ``T_total`` are measured and which are stated.

    Never suppressed: the deliverable is a plan for an account whose placement latency is
    not ours to measure, so which part is which is the single most important caveat on
    the whole output.
    """

    def test_a_measured_provision_stage_is_labelled_this_accounts(self) -> None:
        plan = plan_one(_measured(t_total_s=420.0), _scenario(), provision_s=180.0)
        finding = _finding(plan, "provision_stage")
        assert finding.verdict is Verdict.OK
        assert "180s (43%)" in finding.detail
        assert "not of the configuration" in finding.detail

    def test_an_unobserved_boundary_says_so_rather_than_guessing(self) -> None:
        finding = _finding(plan_one(_measured(), _scenario(), provision_s=None), "provision_stage")
        assert finding.verdict is Verdict.OK
        assert "never observed the boundary" in finding.detail
        assert "missing_stages" in (finding.recommendation or "")

    def test_the_bounded_policy_term_is_named_as_bounded(self) -> None:
        # force-desired raises DesiredInstanceCount directly, so the policy's own
        # detection lag is bypassed and bounded from the deployed alarm's periods and
        # cooldown rather than measured. Summing it silently would launder the bound.
        finding = _finding(
            plan_one(
                _measured(t_total_s=420.0), _scenario(), provision_s=180.0, policy_bound_s=60.0
            ),
            "provision_stage",
        )
        assert "60s is the policy's detection lag" in finding.detail
        assert "BOUNDED" in finding.detail

    def test_a_wholly_stated_lag_is_not_reported_as_measured(self) -> None:
        # `provision_s is None` happens for two opposite reasons: the lag was measured
        # whole, or it was stated whole via --assume-t-total. Reporting the second as a
        # measurement launders a command-line argument into an observation, which is the
        # one thing this finding exists to prevent.
        finding = _finding(
            plan_one(_measured(t_total_measured=False), _scenario()), "provision_stage"
        )
        assert finding.verdict is Verdict.WARN
        assert "stated whole, not measured" in finding.detail
        assert "tts-bench ttotal" in (finding.recommendation or "")


class TestFindingOrder:
    def test_findings_are_ordered_by_what_invalidates_what(self) -> None:
        # An untrustworthy measurement makes the rest moot, an infeasible SLO makes the
        # fleet size irrelevant, and a threshold in the wrong units makes the policy
        # wrong whatever the numbers say. Pinned because the report prints them in this
        # order and a reader stops at the first thing that matters.
        plan = plan_one(_measured(), _scenario())
        assert [f.name for f in plan.findings] == [
            "measurement_trust",
            "curve_repeatability",
            "slo_budget",
            "invocation_ceiling",
            "queue_depth",
            "threshold_units",
            "surge_survival",
            "scale_in_safety",
            "fleet_cost",
            "provision_stage",
        ]

    def test_every_finding_is_emitted_on_every_plan(self) -> None:
        # Silence and a pass are different claims. A finding that dropped out when its
        # inputs were missing would read as OK, which is what SUPPRESSED is for.
        bare = plan_one(
            _measured(
                frozen=False,
                unbounded_queue=None,
                runs_contributing=1,
                q_max_bracketed=False,
                cw_units_ratio_by_rung={},
                ladder_p95_ms={},
                t_total_measured=False,
            ),
            _scenario(),
        )
        assert len(bare.findings) == 10
        assert all(f.detail for f in bare.findings)


class TestTTotalStages:
    def test_the_planning_lag_adds_the_bounded_policy_term(self) -> None:
        # Production scales out through the policy, not through a capacity call, so the
        # measured half alone under-states what a surge has to be absorbed across. The
        # two stay separate on the artifact and are added exactly once, here.
        assert _stages(policy_bound_s=60.0).plan_total_s == pytest.approx(480.0)

    def test_no_policy_term_leaves_the_measured_total_alone(self) -> None:
        assert _stages().plan_total_s == pytest.approx(420.0)

    def test_no_measured_total_has_no_planning_lag(self) -> None:
        assert _stages(total_s=None, policy_bound_s=60.0).plan_total_s is None

    def test_reads_the_provision_stage_from_an_artifact(self) -> None:
        stages = TTotalStages.from_artifact(
            {
                "t_total_s": 420.0,
                "policy_lag_bound_s": 60.0,
                "trigger": "force-desired",
                "config_slug": SLUG,
                "run_id": "r1",
                "durations": [
                    {"from": "in_service", "to": "traffic_recovered", "seconds": 20.0},
                    {"from": PROVISION_FROM_STAGE, "to": PROVISION_TO_STAGE, "seconds": 180.0},
                ],
            }
        )
        assert stages.provision_s == pytest.approx(180.0)
        assert stages.plan_total_s == pytest.approx(480.0)
        assert stages.provision_measured
        assert stages.config_slug == SLUG
        assert stages.trigger == "force-desired"

    def test_the_bound_is_read_from_the_artifact_not_a_constant(self) -> None:
        # The deployed policy's periods and cooldown are what the number came from, so a
        # plan built later must use the bound that applied when the lag was measured --
        # importing today's constant would silently re-date the measurement.
        assert TTotalStages.from_artifact({"t_total_s": 100.0}).policy_bound_s is None
        assert TTotalStages.from_artifact(
            {"t_total_s": 100.0, "policy_lag_bound_s": 45.0}
        ).plan_total_s == pytest.approx(145.0)

    def test_a_duration_for_another_stage_pair_is_ignored(self) -> None:
        stages = TTotalStages.from_artifact(
            {
                "t_total_s": 420.0,
                "durations": [{"from": "in_service", "to": "traffic_recovered", "seconds": 20.0}],
            }
        )
        assert stages.provision_s is None
        assert not stages.provision_measured

    def test_an_artifact_missing_everything_still_parses(self) -> None:
        # A run that observed half the timeline is still the best information
        # available, and refusing to parse it would send the operator back to a
        # measurement that may not be repeatable in this account.
        stages = TTotalStages.from_artifact({})
        assert stages.total_s is None
        assert stages.provision_s is None
        assert stages.plan_total_s is None
        assert stages.missing_stages == ()

    def test_malformed_durations_do_not_raise(self) -> None:
        for durations in ("not a list", [None], [{"from": PROVISION_FROM_STAGE}], [[]]):
            stages = TTotalStages.from_artifact({"t_total_s": 1.0, "durations": durations})
            assert stages.provision_s is None

    def test_a_bounded_total_is_carried_through(self) -> None:
        assert TTotalStages.from_artifact({"t_total_s": 300.0, "t_total_bounded": True}).bounded

    def test_missing_stages_are_carried_through(self) -> None:
        # weights_fetched is never emitted by kokoro's serve.py, so this list is
        # non-empty on every real artifact and the provision finding cites it.
        stages = TTotalStages.from_artifact(
            {"t_total_s": 300.0, "missing_stages": ["weights_fetched"]}
        )
        assert stages.missing_stages == ("weights_fetched",)


class TestAssertPairable:
    def test_matching_slugs_pass(self) -> None:
        # Returns None and raises nothing; not raising is the assertion.
        assert_pairable(_measured(), _stages())

    def test_differing_slugs_are_refused(self) -> None:
        with pytest.raises(PlannerError, match="cannot pair"):
            assert_pairable(_measured(), _stages(config_slug="g6xlarge-abcd1234"))

    def test_an_absent_fingerprint_counts_as_a_mismatch(self) -> None:
        # The check must not pass on the old artifacts -- that is exactly when it
        # matters. Treating "no fingerprint" as agreement would defeat it.
        with pytest.raises(PlannerError, match="no configuration fingerprint"):
            assert_pairable(_measured(deployed_config={}), _stages())
        with pytest.raises(PlannerError, match="no configuration fingerprint"):
            assert_pairable(_measured(), _stages(config_slug=""))

    def test_allow_mismatch_downgrades_to_a_warning(self, logged: list[str]) -> None:
        assert_pairable(_measured(), _stages(config_slug="g6xlarge-abcd"), allow_mismatch=True)
        assert any("different configurations" in message for message in logged)

    def test_the_refusal_names_both_configurations(self) -> None:
        with pytest.raises(PlannerError) as excinfo:
            assert_pairable(_measured(), _stages(config_slug="g6xlarge-abcd1234"))
        assert SLUG in str(excinfo.value)
        assert "g6xlarge-abcd1234" in str(excinfo.value)

    def test_the_refusal_says_how_to_proceed(self) -> None:
        with pytest.raises(PlannerError, match="--allow-config-mismatch"):
            assert_pairable(_measured(), _stages(config_slug="g6xlarge-abcd1234"))


class TestMeasuredFromArtifacts:
    """Joining the two artifacts, which is the only place the measured variables meet."""

    def _step(self, **overrides: Any) -> StepSummary:
        fields: dict[str, Any] = {
            "run_index": 0,
            "step_index": 0,
            "concurrency": 1,
            "achieved_rps": 9.0,
            "completed": 100,
            "ok": 100,
            "chars": 2500,
            "ttfab_p95_ms": LADDER[1],
            "concurrency_mean": 1.0,
            "meets_slo": True,
            "saturated": False,
            "settled": True,
            "usable": True,
        }
        fields.update(overrides)
        return StepSummary(**fields)

    def _report(self, **overrides: Any) -> QMaxReport:
        fields: dict[str, Any] = {
            "model_name": "kokoro-82m",
            "endpoint": "speech-kokoro-82m",
            "instance_type": "ml.g5.xlarge",
            "run_id": "qmax123",
            "slo_ms": 3000,
            "q_max": Q_MAX,
            "q_max_per_run": (Q_MAX, Q_MAX),
            "q_max_bracketed": True,
            "ttfab_p95_at_q_max_ms": LADDER[Q_MAX],
            "s_mean_s": S_MEAN_S,
            "s_p95_s": S_P95_S,
            "frozen": True,
            "unbounded_queue": True,
            "instance_counts_observed": (1,),
            "transport": "bidi",
            "runs": 2,
            "hold_s": 120.0,
            "measure_window_s": 90.0,
            "deployed_config": {
                "instance_type": "ml.g5.xlarge",
                "image_digest": DIGEST,
                "container_env": {},
            },
            "steps": [
                self._step(step_index=index, concurrency=rung, ttfab_p95_ms=p95)
                for index, (rung, p95) in enumerate(LADDER.items())
            ],
        }
        fields.update(overrides)
        return QMaxReport(**fields)

    def test_joins_a_ladder_and_a_lag(self) -> None:
        measured = measured_from_artifacts(self._report(), _stages())
        assert measured.q_max == Q_MAX
        assert measured.slo_ms == 3000
        assert measured.t_total_s == pytest.approx(420.0)
        assert measured.provenance.origin is Origin.MEASURED
        assert measured.t_total_measured

    def test_the_bounded_policy_term_reaches_the_lag_the_plan_uses(self) -> None:
        # The join is where the two terms are summed, so this is what the plan's
        # cooldowns and shed probability are computed against.
        measured = measured_from_artifacts(self._report(), _stages(policy_bound_s=60.0))
        assert measured.t_total_s == pytest.approx(480.0)
        note = measured.provenance.note or ""
        assert "60s policy lag" in note
        assert "not measured" in note

    def test_the_ladder_survives_the_join(self) -> None:
        # ttotal's recovery test reads a *pair* of rungs off this, so a plan artifact
        # read back later must not have to re-run a 40-minute ladder to answer it.
        measured = measured_from_artifacts(self._report(), _stages())
        assert measured.ladder_p95_ms == LADDER
        assert measured.ttfab_p95_at_c1_ms == pytest.approx(LADDER[1])

    def test_carries_the_spread_through_to_the_planner(self) -> None:
        # Without this the planner cannot tell a repeatable Q_max from a single sample,
        # and both thresholds are fractions of Q_max.
        measured = measured_from_artifacts(
            self._report(q_max=40, q_max_per_run=(40, 50)), _stages()
        )
        assert measured.q_max == 40
        assert measured.q_max_spread == pytest.approx(0.25)
        assert measured.runs_contributing == 2

    def test_carries_chars_per_request_for_pricing(self) -> None:
        # 2500 chars across 100 completions -> 25 per request.
        assert measured_from_artifacts(
            self._report(), _stages()
        ).chars_per_request == pytest.approx(25.0)

    def test_refuses_a_mismatched_pair(self) -> None:
        with pytest.raises(PlannerError, match="cannot pair"):
            measured_from_artifacts(self._report(), _stages(config_slug="g6xlarge-abcd"))

    def test_allows_a_mismatch_when_asked(self, logged: list[str]) -> None:
        measured = measured_from_artifacts(
            self._report(), _stages(config_slug="g6xlarge-abcd"), allow_config_mismatch=True
        )
        assert measured.t_total_s == pytest.approx(420.0)
        assert logged

    def test_no_total_and_no_assumption_is_refused(self) -> None:
        with pytest.raises(PlannerError, match="no --assume-t-total"):
            measured_from_artifacts(self._report(), _stages(total_s=None))

    def test_an_assumed_total_is_labelled_an_assumption(self) -> None:
        measured = measured_from_artifacts(
            self._report(), _stages(total_s=None), assume_t_total_s=240.0
        )
        assert measured.t_total_s == pytest.approx(240.0)
        assert "stated, not measured" in (measured.provenance.note or "")
        assert not measured.t_total_measured

    def test_the_origin_stays_measured_when_only_the_lag_was_stated(self) -> None:
        # A flag of its own rather than `origin`, which describes the object as a whole:
        # Q_max and S were measured here, so flipping the origin to ASSUMPTION would
        # disown two real measurements to disclaim one stated number.
        measured = measured_from_artifacts(
            self._report(), _stages(total_s=None), assume_t_total_s=240.0
        )
        assert measured.provenance.origin is Origin.MEASURED
        assert not measured.t_total_measured

    def test_a_measured_total_records_the_trigger(self) -> None:
        measured = measured_from_artifacts(self._report(), _stages(trigger="force-desired"))
        assert "force-desired" in (measured.provenance.note or "")
        assert measured.t_total_measured

    def test_a_bounded_total_says_it_is_a_floor(self) -> None:
        # A bounded total stops at in_service, so it *under*-reports the lag -- the
        # dangerous direction, since the plan absorbs a surge across it.
        assert "floor" in (
            measured_from_artifacts(self._report(), _stages(bounded=True)).provenance.note or ""
        )

    def test_the_ladders_own_note_survives_the_join(self) -> None:
        # Both notes, not a summary of one. A `Measured` read back out of a plan
        # artifact is all a reader has, so an unfrozen ladder joined to an assumed lag
        # must still say the ladder was unfrozen -- that caveat cannot be reconstructed
        # from `origin`, and it is the one that invalidates Q_max outright.
        report = self._report(
            frozen=False,
            provenance=Provenance(origin=Origin.MEASURED, note="WITHOUT the autoscaling freeze"),
        )
        measured = measured_from_artifacts(report, _stages(total_s=None), assume_t_total_s=240.0)
        note = measured.provenance.note or ""
        assert "WITHOUT the autoscaling freeze" in note
        assert "stated, not measured" in note

    def test_the_preconditions_survive_the_join(self) -> None:
        # measurement_trust reads all three off the joined object, so a ladder run
        # without a precondition has to stay identifiable as such after the join.
        measured = measured_from_artifacts(
            self._report(frozen=False, unbounded_queue=False, instance_counts_observed=(1, 2)),
            _stages(),
        )
        assert not measured.frozen
        assert measured.unbounded_queue is False
        assert measured.instance_counts_observed == (1, 2)
        assert not measured.trustworthy

    def test_pairing_can_be_skipped_when_the_lag_was_stated(self) -> None:
        # A stated lag carries no fingerprint, so checking one against the ladder's
        # would report a mismatch where there is nothing to mismatch.
        measured = measured_from_artifacts(
            self._report(),
            TTotalStages.from_artifact({}),
            assume_t_total_s=300.0,
            require_pairing=False,
        )
        assert measured.t_total_s == pytest.approx(300.0)
        assert "stated, not measured" in (measured.provenance.note or "")

    def test_the_joined_measurement_plans(self) -> None:
        # End to end through the seam the CLI uses: two artifacts in, a plan out, at
        # the SLO the ladder recorded.
        measured = measured_from_artifacts(self._report(), _stages())
        plan = plan_one(measured, _scenario(ttfab_slo_ms=measured.slo_ms))
        assert plan.c_scale_max == pytest.approx(C_SCALE_MAX)
        assert plan.queue_max_depth == Q_MAX
