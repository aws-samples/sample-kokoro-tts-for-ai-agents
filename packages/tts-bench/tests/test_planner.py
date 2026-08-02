"""Tests for the planner: composition, refusals, and the two sweeps.

The planner owns no equations — every one lives in `shared.capacity` and is tested
there against the worked examples. So these tests are about the four things that can
go wrong *between* the equations:

- **Composition.** The numbers must come out the same as hand-derivation, which is
  checked against the values `config.py` currently ships: `C_target 0.713` and
  `Q_max 296` at the deployed C_max and k=2. That pairing is the regression test for
  the whole chain, because those four numbers were derived by hand once and the tool
  exists to stop that happening again.
- **The refusals.** Pairing a curve and a lag from different configurations, and a
  `W_max` past the 60s invocation ceiling. Both are hard by design: the first produces
  a plan for a fleet that exists nowhere, and the second admits requests that wait the
  full budget and then fail anyway.
- **Provenance.** A swept provision time is an assumption, and a plan built on one must
  say so. If a substituted lag can pass for a measured one the whole sweep is
  misleading rather than merely approximate.
- **Monotonicity across both sweeps.** Raising `k` must never raise `C_target`, and a
  longer provision time must never shorten `T_total`. Asserted as properties rather
  than as fixed values, because the direction is the invariant a reader relies on when
  choosing a row.

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
)
from tts_bench.planner import (
    CURVE_SPREAD_WARN,
    DEFAULT_K_SWEEP,
    DEFAULT_PROVISION_SWEEP_S,
    PROVISION_FROM_STAGE,
    PROVISION_TO_STAGE,
    PlannerError,
    TTotalStages,
    assert_pairable,
    measured_from_artifacts,
    plan_one,
    plan_sweep,
)
from tts_bench.types import (
    CMaxReport,
    KneePoint,
    Measured,
    Origin,
    Provenance,
    Scenario,
    StepSummary,
    Verdict,
)

#: The deployed Kokoro measurement, as `config.py` records it. Not rounded: the point
#: of pinning these is to reproduce `scaling_target_value` exactly.
C_MAX = 1.63
S_MEAN_S = 0.10986375146305409
S_P95_S = 0.16457688123919073
SLUG = "g5xlarge-139b9068"
DIGEST = "139b9068c5eb1f03c8312c17391dc35838e43e2417ae80d61e628e6ffb3d6a28"


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
    fields: dict[str, Any] = {
        "model_name": "kokoro-82m",
        "endpoint": "speech-kokoro-82m",
        "instance_type": "ml.g5.xlarge",
        "c_max_curve": {300: C_MAX, 500: 2.0},
        "s_mean_s": S_MEAN_S,
        "s_p95_s": S_P95_S,
        "t_total_s": 300.0,
        "chars_per_request": 25.0,
        "curve_spread": {300: 0.05, 500: 0.1},
        "runs_contributing": {300: 3, 500: 3},
        "runs_total": 3,
        "frozen": True,
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
    fields: dict[str, Any] = {
        "peak_rps": 20.0,
        "trough_rps": 2.0,
        "growth_factor_k": 2.0,
        "ttfab_budget_ms": 300,
        "ttfab_slo_ms": 3000,
    }
    fields.update(overrides)
    return Scenario(**fields)


#: W_max the default scenario derives: 3.0s SLO - 0.1646s p95 service. Named because
#: several tests need the number the plan will actually use, and re-deriving it in each
#: one was how a test came to assert against a wait budget the plan had not chosen.
W_MAX_S = 3.0 - S_P95_S


def _slo_for_wait(wait_s: float, s_p95_s: float = S_P95_S) -> int:
    """The SLO that leaves ``wait_s`` of queueing budget, in ms.

    W_max is no longer settable, so a test that wants a particular queue allowance has
    to state the SLO that produces it. Inverting ``w_max_for_slo`` here rather than in
    each test keeps the relation in one place; the tests still assert against
    ``plan.w_max_s``, so a broken inversion shows up as a failure rather than as two
    wrongs agreeing.
    """
    return round((wait_s + s_p95_s) * 1000)


def _stages(**overrides: Any) -> TTotalStages:
    fields: dict[str, Any] = {
        "total_s": 420.0,
        "provision_s": 180.0,
        "trigger": "drive-load",
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
    def test_reproduces_the_deployed_config_numbers(self) -> None:
        # The regression test for the whole chain, against the numbers config.py ships.
        # C_target: 0.875 x 1.63 / 2 = 0.713, unchanged by the SLO -- c_slo_cap at a
        # 2.84s W_max is 26.8, far above a sub-1 target.
        #
        # Q_max moved, and that is the fix rather than a regression: it was 296, from a
        # hand-set 20s W_max no SLO justified. Under a 3s end-to-end promise
        # W_max = 3.0 - 0.1646 = 2.835s and Lambda_cap 14.84 x 2.835 = 42.
        plan = plan_one(_measured(), _scenario())
        assert plan.c_target == pytest.approx(0.713, abs=5e-4)
        assert plan.queue_max_depth == 42
        assert plan.w_max_s == pytest.approx(W_MAX_S)

    def test_the_queue_depth_follows_the_slo_and_nothing_else(self) -> None:
        # The property the whole reframe buys: there is no way to state a queue that
        # disagrees with the promise, because the promise is the only input.
        tight = plan_one(_measured(), _scenario(ttfab_slo_ms=1000))
        loose = plan_one(_measured(), _scenario(ttfab_slo_ms=10_000))
        assert tight.queue_max_depth < loose.queue_max_depth
        assert tight.w_max_s == pytest.approx(1.0 - S_P95_S)
        assert loose.w_max_s == pytest.approx(10.0 - S_P95_S)

    def test_fleet_sizes_bracket_the_stated_load(self) -> None:
        # 20 rps x 0.11s = 2.2 concurrent at 0.713 per instance -> 4 instances.
        plan = plan_one(_measured(), _scenario())
        assert plan.peak_instances == 4
        assert plan.trough_instances == 1
        assert plan.min_instances == 1
        assert plan.max_instances == 4

    def test_min_is_the_trough_fleet_not_one(self) -> None:
        # A reserved-capacity account pays the floor whatever the traffic does, so a
        # stated trough that needs three instances must raise min -- defaulting to 1
        # would under-reserve exactly the capacity that was reserved on purpose.
        plan = plan_one(_measured(), _scenario(trough_rps=18.0))
        assert plan.trough_instances == 3
        assert plan.min_instances == 3

    def test_max_never_falls_below_min(self) -> None:
        # A peak below the floor is a coherent scenario (over-reserved on purpose),
        # and max < min is not a deployable config.
        plan = plan_one(_measured(), _scenario(peak_rps=0.5, trough_rps=0.5, min_instances_floor=3))
        assert plan.max_instances >= plan.min_instances == 3

    def test_streams_win_over_a_rate(self) -> None:
        # A stated stream count is a direct concurrency observation; lambda x S is the
        # same quantity inferred. For bidi the inference is the weaker of the two.
        by_streams = plan_one(_measured(), _scenario(peak_rps=20.0, peak_streams=8.0))
        assert by_streams.peak_instances == 12  # ceil(8 / 0.713)

    def test_utilization_is_the_derate_over_k(self) -> None:
        plan = plan_one(_measured(), _scenario(growth_factor_k=2.0))
        assert plan.utilization_at_target == pytest.approx(0.4375)

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

    def test_reads_the_knee_at_the_requested_budget(self) -> None:
        loose = plan_one(_measured(), _scenario(ttfab_budget_ms=500))
        tight = plan_one(_measured(), _scenario(ttfab_budget_ms=300))
        assert loose.c_max == 2.0
        assert tight.c_max == C_MAX
        # A looser SLO admits more concurrency per instance, so it needs fewer of them.
        assert loose.peak_instances <= tight.peak_instances

    def test_refuses_a_budget_below_every_measured_one(self) -> None:
        with pytest.raises(PlannerError, match="no C_max measured"):
            plan_one(_measured(), _scenario(ttfab_budget_ms=100))

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


class TestBindingConstraint:
    def test_surge_headroom_binds_for_a_fast_model(self) -> None:
        plan = plan_one(_measured(), _scenario())
        assert plan.binding_constraint == "surge_headroom"
        assert "surge_headroom" in _finding(plan, "binding_constraint").detail

    def test_slo_wait_budget_binds_for_a_high_capacity_model(self) -> None:
        # It takes a knee well above k for the wait budget to bind at all: c_slo_cap is
        # never below 1 (the running request always fits), so a sub-1 target like
        # Kokoro's is surge-bound at every SLO including one that leaves no wait at all.
        # Here the derated knee is 3.5, and an SLO 0.2s above p95 service permits 2.82.
        plan = plan_one(
            _measured(c_max_curve={300: 8.0}, curve_spread={300: 0.05}, runs_contributing={300: 3}),
            _scenario(ttfab_slo_ms=_slo_for_wait(0.2)),
        )
        assert plan.binding_constraint == "slo_wait_budget"
        assert plan.c_target == pytest.approx(plan.w_max_s / S_MEAN_S + 1)
        assert "faster model" in (_finding(plan, "binding_constraint").recommendation or "")

    def test_a_sub_one_target_is_surge_bound_at_every_slo(self) -> None:
        # Worth pinning: it explains why the deployed Kokoro config reports
        # surge_headroom even where the SLO leaves no queue at all, which otherwise
        # looks like a bug. The 3s SLO the tool is built around moves Q_max, not this.
        for wait in (0.0, 0.05, 0.2, 1.0, 20.0):
            plan = plan_one(_measured(), _scenario(ttfab_slo_ms=_slo_for_wait(wait)))
            assert plan.binding_constraint == "surge_headroom"
            assert plan.c_target == pytest.approx(0.875 * C_MAX / 2)

    def test_the_reported_target_is_always_the_smaller_of_the_two(self) -> None:
        # The property, over both regimes: whichever bound is named, the target must
        # not exceed either one. A target above the wait budget misses the SLO; a
        # target above the derated knee has no surge headroom left.
        for c_max in (C_MAX, 8.0):
            measured = _measured(
                c_max_curve={300: c_max},
                curve_spread={300: 0.05},
                runs_contributing={300: 3},
            )
            for wait in (0.0, 0.05, 0.2, 1.0, 20.0):
                plan = plan_one(measured, _scenario(ttfab_slo_ms=_slo_for_wait(wait)))
                assert plan.c_target <= 0.875 * c_max / 2 + 1e-9
                assert plan.c_target <= plan.w_max_s / S_MEAN_S + 1 + 1e-9


class TestInvocationCeiling:
    def test_a_wait_past_the_ceiling_is_infeasible_not_a_warning(self) -> None:
        # The hard constraint. Those requests wait the full W_max and then fail
        # anyway, which is worse than refusing them at admission. Reached now by an SLO
        # past the platform ceiling rather than a hand-set wait: with W_max derived, the
        # deadline *is* the SLO, so a 70s promise cannot be kept whatever the queue does.
        plan = plan_one(_measured(), _scenario(ttfab_slo_ms=70_000))
        finding = _finding(plan, "invocation_ceiling")
        assert finding.verdict is Verdict.INFEASIBLE
        assert plan.infeasible

    def test_an_infeasible_plan_says_what_would_work(self) -> None:
        plan = plan_one(_measured(), _scenario(ttfab_slo_ms=70_000))
        recommendation = _finding(plan, "invocation_ceiling").recommendation or ""
        assert "--ttfab-slo-ms" in recommendation
        # And the number it names must itself pass, or it is advice to build another
        # infeasible plan.
        largest = int(recommendation.split("--ttfab-slo-ms")[1].split()[0])
        assert not plan_one(_measured(), _scenario(ttfab_slo_ms=largest)).infeasible

    def test_the_stated_slo_is_feasible(self) -> None:
        plan = plan_one(_measured(), _scenario())
        assert _finding(plan, "invocation_ceiling").verdict is Verdict.OK
        assert not plan.infeasible

    def test_judges_p95_service_not_mean(self) -> None:
        # A mean-sized deadline is missed by half the requests that reach it. This model
        # has a 0.5s p95 against a 0.05s mean, and at a 60s SLO the ceiling check sees
        # 59.5s of queue plus that 0.5s tail -- exactly 60s, and one step looser breaks.
        plan = plan_one(_measured(s_mean_s=0.05, s_p95_s=0.5), _scenario(ttfab_slo_ms=60_001))
        assert plan.infeasible
        assert not plan_one(
            _measured(s_mean_s=0.05, s_p95_s=0.5), _scenario(ttfab_slo_ms=60_000)
        ).infeasible

    def test_a_model_slower_than_the_ceiling_cannot_be_rescued(self) -> None:
        # p95 service of 61s exceeds the 60s ceiling on its own, so there is no SLO to
        # recommend -- an unqueued request already fails.
        plan = plan_one(
            _measured(s_mean_s=30.0, s_p95_s=61.0),
            _scenario(ttfab_slo_ms=70_000),
        )
        recommendation = _finding(plan, "invocation_ceiling").recommendation or ""
        assert "no SLO or queue length fixes this" in recommendation

    def test_an_explicit_ceiling_is_honoured(self) -> None:
        plan = plan_one(_measured(), _scenario(ttfab_slo_ms=20_000), ceiling_s=10.0)
        assert plan.infeasible

    def test_the_default_ceiling_is_sagemakers(self) -> None:
        assert SAGEMAKER_INVOCATION_CEILING_S == 60.0


class TestMeasurementFindings:
    def test_an_unfrozen_curve_warns(self) -> None:
        plan = plan_one(_measured(frozen=False), _scenario())
        finding = _finding(plan, "measurement_trust")
        assert finding.verdict is Verdict.WARN
        assert "not suspended" in finding.detail

    def test_a_resized_fleet_warns(self) -> None:
        plan = plan_one(_measured(instance_counts_observed=(1, 2)), _scenario())
        finding = _finding(plan, "measurement_trust")
        assert finding.verdict is Verdict.WARN
        assert "resized mid-run" in finding.detail

    def test_a_single_contributing_run_is_not_agreement(self) -> None:
        # The defect this exists for: spread is 0% when only one run found a knee,
        # which otherwise reads exactly like three runs agreeing perfectly.
        plan = plan_one(
            _measured(curve_spread={300: 0.0, 500: 0.0}, runs_contributing={300: 1, 500: 3}),
            _scenario(),
        )
        finding = _finding(plan, "curve_repeatability")
        assert finding.verdict is Verdict.WARN
        assert "1 of 3" in finding.detail

    def test_a_wide_spread_warns(self) -> None:
        plan = plan_one(_measured(curve_spread={300: 0.45, 500: 0.1}), _scenario())
        assert _finding(plan, "curve_repeatability").verdict is Verdict.WARN

    def test_a_tight_spread_across_runs_passes(self) -> None:
        plan = plan_one(_measured(), _scenario())
        assert _finding(plan, "curve_repeatability").verdict is Verdict.OK

    def test_no_spread_recorded_is_suppressed_not_ok(self) -> None:
        # SUPPRESSED because the condition was never evaluated. Marking a
        # pre-fingerprint artifact OK would claim a repeatability nobody measured.
        plan = plan_one(_measured(curve_spread={}, runs_contributing={}), _scenario())
        assert _finding(plan, "curve_repeatability").verdict is Verdict.SUPPRESSED

    def test_spread_is_read_at_the_planned_budget(self) -> None:
        # The 500ms spread is fine and the 300ms one is not; planning at 300ms must
        # see the bad one.
        measured = _measured(curve_spread={300: 0.5, 500: 0.01})
        assert _finding(plan_one(measured, _scenario(ttfab_budget_ms=300)), "curve_repeatability")
        assert (
            _finding(
                plan_one(measured, _scenario(ttfab_budget_ms=300)), "curve_repeatability"
            ).verdict
            is Verdict.WARN
        )
        assert (
            _finding(
                plan_one(measured, _scenario(ttfab_budget_ms=500)), "curve_repeatability"
            ).verdict
            is Verdict.OK
        )

    def test_the_warn_threshold_is_twenty_percent(self) -> None:
        assert CURVE_SPREAD_WARN == 0.2


class TestQueueFindings:
    def test_flat_traffic_reads_as_covered_without_printing_infinity(self) -> None:
        # W_absorbed is infinite at k=1, which is true but unreadable as a duration.
        plan = plan_one(_measured(), _scenario(growth_factor_k=1.0))
        detail = _finding(plan, "queue_covers_surge").detail
        assert "inf" not in detail
        assert "flat at k=1" in detail

    def test_an_uncovered_surge_warns_with_the_remainder(self) -> None:
        plan = plan_one(_measured(t_total_s=300.0), _scenario(growth_factor_k=3.0))
        finding = _finding(plan, "queue_covers_surge")
        assert finding.verdict is Verdict.WARN
        assert "standing headroom" in finding.detail

    def test_flat_traffic_does_not_warn_about_headroom(self) -> None:
        # k=1 floors the uncovered lag at the metric period, but by formality --
        # flat traffic needs no standing headroom at all, so warning would tell the
        # operator to fix a plan with nothing wrong with it.
        plan = plan_one(_measured(), _scenario(growth_factor_k=1.0))
        assert _finding(plan, "headroom_lag").verdict is Verdict.OK

    def test_a_floored_lag_above_k_one_warns(self) -> None:
        # The 2.84s W_max the 3s SLO leaves absorbs 2.84s at k=2, so a 10s lag leaves
        # 7.2s -- under the 10s metric period, which floors it. The headroom is then
        # sized for growth CloudWatch cannot report in time to act on.
        plan = plan_one(_measured(t_total_s=10.0), _scenario(growth_factor_k=2.0))
        finding = _finding(plan, "headroom_lag")
        assert finding.verdict is Verdict.WARN
        assert "faster than" in finding.detail

    def test_a_queue_that_rounds_to_zero_warns(self) -> None:
        # An SLO that leaves no queue at all: exactly p95 service, so W_max is 0. The
        # SLO finding calls that infeasible; this one says the queue cannot help.
        plan = plan_one(_measured(), _scenario(ttfab_slo_ms=_slo_for_wait(0.0)))
        finding = _finding(plan, "queue_depth")
        assert finding.verdict is Verdict.WARN
        assert plan.queue_max_depth == 0
        assert "SLO leaves" in finding.detail


class TestWhichCMaxBinds:
    """Which of the two measured limits was divided by, and how firmly.

    The distinction the report used to collapse into a single number. All four states
    want a different next run, and three of them are not ``OK``: a knee that was never
    bracketed over-sizes the fleet, a throughput ceiling means the knee's concurrency was
    backlog, and an unmeasured ceiling means the comparison never happened at all.
    """

    def _bracketed(self, **overrides: Any) -> Measured:
        """A curve whose knees are recorded as bracketed, so only the source varies.

        Without this the default fixture records no ``c_max_bracketed`` at all, which is
        the SUPPRESSED "unrecorded" state and would mask every other verdict here.
        """
        fields: dict[str, Any] = {"c_max_bracketed": {300: True, 500: True}}
        fields.update(overrides)
        return _measured(**fields)

    def test_a_knee_under_the_ceiling_is_the_knee(self) -> None:
        # Latency degrades before throughput does, which is the regime the tool was
        # originally written for and the one kokoro on bidi is *not* in.
        plan = plan_one(self._bracketed(c_max_throughput=5.0), _scenario())
        finding = _finding(plan, "c_max_source")
        assert plan.c_max_source == "latency_knee"
        assert plan.c_max == C_MAX
        assert finding.verdict is Verdict.OK
        assert "latency degrades before throughput" in finding.detail

    def test_a_ceiling_under_the_knee_binds_and_says_the_knee_was_backlog(self) -> None:
        # Kokoro's regime, and the whole reason the ceiling is measured: the ladder's
        # higher concurrency was accumulated queue, so planning on the knee would size a
        # fleet for capacity the instance does not have.
        plan = plan_one(self._bracketed(c_max_throughput=1.0), _scenario())
        finding = _finding(plan, "c_max_source")
        assert plan.c_max_source == "throughput_ceiling"
        assert plan.c_max == 1.0
        assert finding.verdict is Verdict.OK
        assert "queue backlog, not capacity" in finding.detail
        # And the smaller C_max has to reach the fleet size, not just the finding.
        assert (
            plan.peak_instances
            > plan_one(self._bracketed(c_max_throughput=5.0), _scenario()).peak_instances
        )

    def test_an_unmeasured_ceiling_is_suppressed_not_ok(self) -> None:
        # The state every committed artifact is in. Absent must not read as "compared
        # and the knee won" -- for a model that holds its inference lock for a whole
        # session the ceiling is the likelier bound, so silence here is the dangerous
        # direction.
        plan = plan_one(self._bracketed(), _scenario())
        finding = _finding(plan, "c_max_source")
        assert plan.c_max_source == "latency_knee_only"
        assert finding.verdict is Verdict.SUPPRESSED
        assert "was never checked" in finding.detail
        assert "measures the ceiling alongside the knee" in (finding.recommendation or "")

    def test_an_unbracketed_knee_is_a_lower_bound_and_says_the_fleet_is_over_sized(
        self,
    ) -> None:
        plan = plan_one(
            self._bracketed(c_max_throughput=5.0, c_max_bracketed={300: False}), _scenario()
        )
        finding = _finding(plan, "c_max_source")
        assert plan.c_max_is_lower_bound
        assert finding.verdict is Verdict.WARN
        assert "LOWER BOUND" in finding.detail
        assert "--target-concurrency" in (finding.recommendation or "")

    def test_an_unbracketed_ceiling_advises_raising_the_worker_pool(self) -> None:
        # A different fix from the knee's: a ceiling can be a lower bound because the
        # client pool never handed the server the offered rate, and extending the ladder
        # would then measure the same client limit one step higher.
        plan = plan_one(
            self._bracketed(c_max_throughput=1.0, c_max_throughput_bracketed=False), _scenario()
        )
        finding = _finding(plan, "c_max_source")
        assert plan.c_max_source == "throughput_ceiling"
        assert plan.c_max_is_lower_bound
        assert finding.verdict is Verdict.WARN
        assert "--max-workers" in (finding.recommendation or "")

    def test_an_unrecorded_bracketing_is_suppressed_rather_than_passed(self) -> None:
        # An artifact predating the check. Not the same claim as a bracketed knee, and
        # the plan records None rather than False so a reader can tell which.
        plan = plan_one(_measured(c_max_throughput=5.0, c_max_bracketed={}), _scenario())
        finding = _finding(plan, "c_max_source")
        assert plan.c_max_is_lower_bound is None
        assert finding.verdict is Verdict.SUPPRESSED
        assert "unrecorded" in finding.detail

    def test_the_lower_bound_flag_reaches_the_plan_not_only_the_finding(self) -> None:
        # The artifact is what a later reader has; a warning printed once to a terminal
        # is not a record of it.
        assert (
            plan_one(self._bracketed(c_max_throughput=5.0), _scenario()).c_max_is_lower_bound
            is False
        )


class TestSloBudget:
    """That ``W_max`` is visibly derived, and what happens when there is none left.

    Stated as its own finding because the failure it catches is silent: a model whose p95
    already misses the promise reports ``W_max = 0``, and a reader seeing
    ``queue_max_depth = 0`` beside it would take that for a design choice.
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
        plan = plan_one(_measured(), _scenario(ttfab_slo_ms=150))
        finding = _finding(plan, "slo_budget")
        assert finding.verdict is Verdict.INFEASIBLE
        assert plan.infeasible
        assert plan.w_max_s == 0.0
        assert plan.queue_max_depth == 0
        assert "No queue depth, instance count, or scaling policy rescues this" in finding.detail

    def test_the_infeasible_recommendation_names_an_slo_that_works(self) -> None:
        plan = plan_one(_measured(), _scenario(ttfab_slo_ms=150))
        recommendation = _finding(plan, "slo_budget").recommendation or ""
        slo = int(recommendation.split("--ttfab-slo-ms above")[1].split(",")[0])
        assert (
            _finding(plan_one(_measured(), _scenario(ttfab_slo_ms=slo + 1)), "slo_budget").verdict
            is not Verdict.INFEASIBLE
        )

    def test_an_slo_exactly_at_p95_service_is_infeasible(self) -> None:
        # The boundary. W_max is 0 there, so the promise holds only for a request that
        # never waits -- which is not a promise a queued endpoint can keep.
        plan = plan_one(_measured(), _scenario(ttfab_slo_ms=round(S_P95_S * 1000)))
        # round() lands one tenth of a millisecond above p95, so this is feasible by a
        # hair and warns instead. Pinned to document which side of the line it falls on.
        assert _finding(plan, "slo_budget").verdict is Verdict.WARN
        assert plan.w_max_s < S_P95_S

    def test_a_wait_budget_under_one_service_time_warns(self) -> None:
        # Feasible but barely: a single request queued ahead already misses the SLO, so
        # the promise holds for an uncontended request and little more.
        plan = plan_one(_measured(), _scenario(ttfab_slo_ms=_slo_for_wait(S_P95_S / 2)))
        finding = _finding(plan, "slo_budget")
        assert finding.verdict is Verdict.WARN
        assert "a single request queued ahead misses the SLO" in finding.detail

    def test_a_roomy_slo_passes(self) -> None:
        plan = plan_one(_measured(), _scenario(ttfab_slo_ms=_slo_for_wait(S_P95_S * 3)))
        assert _finding(plan, "slo_budget").verdict is Verdict.OK


class TestFleetCost:
    def test_a_costly_reserve_warns(self) -> None:
        plan = plan_one(_measured(), _scenario(growth_factor_k=5.0, peak_rps=100.0))
        finding = _finding(plan, "fleet_cost")
        assert finding.verdict is Verdict.WARN
        assert "cheaper than" in (finding.recommendation or "")

    def test_k_one_costs_one_times_itself(self) -> None:
        plan = plan_one(_measured(), _scenario(growth_factor_k=1.0))
        assert plan.relative_fleet_cost_vs_k1 == pytest.approx(1.0)

    def test_cost_is_labelled_an_upper_bound(self) -> None:
        plan = plan_one(_measured(), _scenario())
        assert "upper bound" in _finding(plan, "fleet_cost").detail


class TestTTotalStages:
    def test_transferable_is_the_total_less_the_provision(self) -> None:
        assert _stages().transferable_s == pytest.approx(240.0)

    def test_an_unobserved_provision_leaves_the_total_whole(self) -> None:
        # Removing an unknown is not a subtraction we can do, so the whole lag is
        # treated as transferable rather than silently halved.
        stages = _stages(provision_s=None)
        assert stages.transferable_s == pytest.approx(420.0)
        assert not stages.provision_measured

    def test_substituting_a_provision_time(self) -> None:
        assert _stages().with_provision_s(60.0) == pytest.approx(300.0)
        assert _stages().with_provision_s(600.0) == pytest.approx(840.0)

    def test_substituting_into_no_total_is_refused(self) -> None:
        with pytest.raises(PlannerError, match="no measured T_total"):
            _stages(total_s=None).with_provision_s(60.0)

    def test_a_negative_provision_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            _stages().with_provision_s(-1.0)

    def test_reads_the_provision_stage_from_an_artifact(self) -> None:
        stages = TTotalStages.from_artifact(
            {
                "t_total_s": 420.0,
                "trigger": "drive-load",
                "config_slug": SLUG,
                "run_id": "r1",
                "durations": [
                    {"from": "load_applied", "to": "metric_published", "seconds": 20.0},
                    {"from": PROVISION_FROM_STAGE, "to": PROVISION_TO_STAGE, "seconds": 180.0},
                ],
            }
        )
        assert stages.provision_s == pytest.approx(180.0)
        assert stages.transferable_s == pytest.approx(240.0)
        assert stages.config_slug == SLUG

    def test_an_artifact_missing_everything_still_parses(self) -> None:
        # A run that observed half the timeline is still the best information
        # available, and refusing to parse it would send the operator back to a
        # measurement that may not be repeatable in this account.
        stages = TTotalStages.from_artifact({})
        assert stages.total_s is None
        assert stages.provision_s is None
        assert stages.transferable_s is None

    def test_a_bounded_total_is_carried_through(self) -> None:
        assert TTotalStages.from_artifact({"t_total_s": 300.0, "t_total_bounded": True}).bounded


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


class TestPlanSweep:
    def test_one_row_per_pair(self) -> None:
        rows = plan_sweep(
            _measured(),
            _scenario(),
            _stages(),
            provision_sweep_s=[60.0, 300.0],
            k_sweep=[1.0, 2.0, 3.0],
        )
        assert len(rows) == 6
        assert {(row.provision_s, row.k) for row in rows} == {
            (p, k) for p in (60.0, 300.0) for k in (1.0, 2.0, 3.0)
        }

    def test_no_sweep_plans_once_against_the_measured_lag(self) -> None:
        rows = plan_sweep(_measured(t_total_s=300.0), _scenario(), _stages())
        assert len(rows) == 1
        assert rows[0].provision_s is None
        assert not rows[0].provision_assumed
        assert rows[0].t_total_s == pytest.approx(300.0)

    def test_a_longer_provision_never_shortens_the_lag(self) -> None:
        rows = plan_sweep(
            _measured(), _scenario(), _stages(), provision_sweep_s=DEFAULT_PROVISION_SWEEP_S
        )
        lags = [row.t_total_s for row in rows]
        assert lags == sorted(lags)
        # And each is the transferable half plus what was stated.
        for row in rows:
            assert row.t_total_s == pytest.approx(240.0 + (row.provision_s or 0.0))

    def test_a_higher_k_never_raises_the_target(self) -> None:
        rows = plan_sweep(_measured(), _scenario(), _stages(), k_sweep=DEFAULT_K_SWEEP)
        targets = [row.plan.c_target for row in rows]
        assert targets == sorted(targets, reverse=True)

    def test_a_higher_k_never_shrinks_the_fleet(self) -> None:
        rows = plan_sweep(_measured(), _scenario(), _stages(), k_sweep=DEFAULT_K_SWEEP)
        fleets = [row.plan.peak_instances for row in rows]
        assert fleets == sorted(fleets)

    def test_the_swept_k_overrides_the_scenario(self) -> None:
        rows = plan_sweep(_measured(), _scenario(growth_factor_k=2.0), _stages(), k_sweep=[5.0])
        assert rows[0].plan.scenario.growth_factor_k == 5.0

    def test_a_substituted_lag_is_marked_an_assumption(self) -> None:
        # The whole point of the sweep: a row built on a stated provision time must
        # not pass for a measured one.
        rows = plan_sweep(_measured(), _scenario(), _stages(), provision_sweep_s=[60.0])
        note = rows[0].plan.measured.provenance.note
        assert "substituted, not measured" in note
        assert _finding(rows[0].plan, "provision_stage").detail.startswith("T_total assumes a 60s")

    def test_sweeping_the_measured_provision_time_is_still_an_assumption(self) -> None:
        # 180s is what was measured, so substituting it reproduces the total exactly.
        # The row is nonetheless built on a *stated* provision time -- it is labelled
        # `provision_assumed` and its finding says "assumes a 180s provision stage" --
        # so a provenance still reading "measured" would have one plan describe its own
        # input two ways. Numeric equality is not evidence that no claim changed.
        rows = plan_sweep(_measured(), _scenario(), _stages(), provision_sweep_s=[180.0])
        assert rows[0].t_total_s == pytest.approx(420.0)
        assert rows[0].provision_assumed
        assert "substituted, not measured" in (rows[0].plan.measured.provenance.note or "")

    def test_the_measured_row_says_the_lag_is_this_accounts(self) -> None:
        rows = plan_sweep(_measured(), _scenario(), _stages())
        finding = _finding(rows[0].plan, "provision_stage")
        assert finding.verdict is Verdict.OK
        assert "as measured" in finding.detail
        assert "--provision-s" in (finding.recommendation or "")

    def test_a_wholly_stated_lag_is_not_reported_as_measured(self) -> None:
        # `provision_s is None` means "nothing was substituted", which happens for two
        # opposite reasons: the lag was measured whole, or it was stated whole via
        # --assume-t-total. Reporting the second as "used exactly as measured" launders
        # a command-line argument into an observation.
        stated = _measured(t_total_measured=False)
        finding = _finding(plan_one(stated, _scenario()), "provision_stage")
        assert finding.verdict is Verdict.WARN
        assert "stated whole, not measured" in finding.detail
        assert "tts-bench ttotal" in (finding.recommendation or "")

    def test_a_swept_row_reports_the_substitution_whatever_the_lags_origin(self) -> None:
        # Once a provision time is substituted, that is the fact worth reporting -- the
        # stage breakdown it was substituted into came from a measurement either way.
        rows = plan_sweep(
            _measured(t_total_measured=False), _scenario(), _stages(), provision_sweep_s=[60.0]
        )
        finding = _finding(rows[0].plan, "provision_stage")
        assert finding.verdict is Verdict.OK
        assert "assumes a 60s" in finding.detail

    def test_a_sweep_without_stages_is_refused(self) -> None:
        with pytest.raises(PlannerError, match="needs the measured stage breakdown"):
            plan_sweep(_measured(), _scenario(), None, provision_sweep_s=[60.0])

    def test_a_sweep_with_no_measured_total_is_refused(self) -> None:
        with pytest.raises(PlannerError, match="no measured T_total"):
            plan_sweep(_measured(), _scenario(), _stages(total_s=None), provision_sweep_s=[60.0])

    def test_the_ceiling_verdict_holds_across_every_row(self) -> None:
        # Infeasibility comes from the SLO and p95 service, neither of which the sweep
        # varies -- so it must not appear to depend on the row.
        rows = plan_sweep(
            _measured(),
            _scenario(ttfab_slo_ms=70_000),
            _stages(),
            provision_sweep_s=[60.0, 600.0],
            k_sweep=[1.0, 3.0],
        )
        assert all(row.plan.infeasible for row in rows)


class TestMeasuredFromArtifacts:
    def _report(self, **overrides: Any) -> CMaxReport:
        fields: dict[str, Any] = {
            "model_name": "kokoro-82m",
            "endpoint": "speech-kokoro-82m",
            "instance_type": "ml.g5.xlarge",
            "run_id": "cmax123",
            "c_max_curve": {300: C_MAX},
            "curve_spread": {300: 0.05},
            "runs_contributing": {300: 3},
            "runs": 3,
            "hold_s": 240.0,
            "measure_window_s": 120.0,
            "knees": [
                KneePoint(
                    ttfab_budget_ms=300,
                    concurrency=C_MAX,
                    offered_rps=14.8,
                    p95_ttfab_ms=280.0,
                    step_index=2,
                    bracketed=True,
                )
            ],
            "s_mean_s": S_MEAN_S,
            "s_p95_s": S_P95_S,
            "frozen": True,
            "instance_counts_observed": (1,),
            "transport": "bidi",
            "deployed_config": {
                "instance_type": "ml.g5.xlarge",
                "image_digest": DIGEST,
                "container_env": {},
            },
            "steps": [
                StepSummary(
                    run_index=0,
                    step_index=0,
                    target_concurrency=1.0,
                    offered_rps=9.0,
                    achieved_rps=9.0,
                    completed=100,
                    ok=100,
                    chars_per_hour=810000.0,
                    saturated=False,
                    settled=True,
                    usable=True,
                )
            ],
        }
        fields.update(overrides)
        return CMaxReport(**fields)

    def test_joins_a_curve_and_a_lag(self) -> None:
        measured = measured_from_artifacts(self._report(), _stages())
        assert measured.t_total_s == pytest.approx(420.0)
        assert measured.c_max_curve == {300: C_MAX}
        assert measured.provenance.origin is Origin.MEASURED

    def test_carries_the_spread_through_to_the_planner(self) -> None:
        # Without this the planner cannot tell a repeatable C_max from a single
        # sample, and every fleet size divides by C_max.
        measured = measured_from_artifacts(self._report(), _stages())
        assert measured.curve_spread == {300: 0.05}
        assert measured.runs_contributing == {300: 3}
        assert measured.runs_total == 3

    def test_carries_chars_per_request_for_pricing(self) -> None:
        # 810000 chars/hour at 9 rps -> 25 chars/request.
        measured = measured_from_artifacts(self._report(), _stages())
        assert measured.chars_per_request == pytest.approx(25.0)

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
        # C_max and S were measured here, so flipping the origin to ASSUMPTION would
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
        measured = measured_from_artifacts(self._report(), _stages(bounded=True))
        assert "floor" in (measured.provenance.note or "")

    def test_the_curves_own_note_survives_the_join(self) -> None:
        # Both notes, not a summary of one. A `Measured` read back out of a plan
        # artifact is all a reader has, so an unfrozen curve joined to an assumed lag
        # must still say the curve was unfrozen -- that caveat cannot be reconstructed
        # from `origin`, and it is the one that invalidates C_max outright.
        report = self._report(
            frozen=False,
            provenance=Provenance(origin=Origin.MEASURED, note="WITHOUT the autoscaling freeze"),
        )
        measured = measured_from_artifacts(report, _stages(total_s=None), assume_t_total_s=240.0)
        note = measured.provenance.note or ""
        assert "WITHOUT the autoscaling freeze" in note
        assert "stated, not measured" in note

    def test_pairing_can_be_skipped_when_the_lag_was_stated(self) -> None:
        # A stated lag carries no fingerprint, so checking one against the curve's
        # would report a mismatch where there is nothing to mismatch.
        measured = measured_from_artifacts(
            self._report(),
            TTotalStages.from_artifact({}),
            assume_t_total_s=300.0,
            require_pairing=False,
        )
        assert measured.t_total_s == pytest.approx(300.0)
        assert "stated, not measured" in (measured.provenance.note or "")
