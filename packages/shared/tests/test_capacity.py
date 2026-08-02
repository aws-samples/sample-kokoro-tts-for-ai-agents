"""Tests for capacity planning math.

Seeded with the worked examples from the scaling methodology, so a regression
here means the plan output stopped matching the documented reasoning.
"""

from __future__ import annotations

import math

import pytest

from shared.capacity import (
    CLOUDWATCH_HIGH_RES_PERIOD_S,
    DEFAULT_DERATE,
    SAGEMAKER_INVOCATION_CEILING_S,
    c_slo_cap,
    c_target,
    effective_c_target,
    effective_headroom_lag_s,
    fits_invocation_ceiling,
    fits_slo,
    lambda_cap_per_instance,
    max_added_wait_under_ceiling,
    min_samples_for_k,
    n_instances,
    n_instances_from_streams,
    q_per_instance,
    queue_covers_surge,
    request_deadline_s,
    slo_is_feasible,
    utilization_at_k,
    w_absorbed,
    w_max_for_slo,
)

#: Kokoro-82M on ml.g5.xlarge, bidi transport, from
#: artifacts/cmax-kokoro-82m-bidi-g5xlarge-139b9068.json. The numbers the 3s SLO was
#: actually reasoned about, so a regression here means the worked example moved.
KOKORO_S_MEAN_S = 0.10602401316328536
KOKORO_S_P95_S = 0.1645768812391907


class TestCTarget:
    def test_k_of_one_is_just_the_derate(self) -> None:
        # Flat traffic needs no surge reserve, only the jitter margin.
        assert c_target(8.0, k=1.0, derate=1.0) == 8.0
        assert c_target(8.0, k=1.0) == pytest.approx(7.0)

    @pytest.mark.parametrize(
        ("k", "expected_fraction"),
        [(1.0, 1.0), (2.0, 0.5), (3.0, 1 / 3), (4.0, 0.25), (5.0, 0.2)],
    )
    def test_reserves_one_over_k(self, k: float, expected_fraction: float) -> None:
        # The headline result: doubling traffic within one T_total means running
        # each instance at half its ceiling.
        assert c_target(100.0, k=k, derate=1.0) == pytest.approx(100.0 * expected_fraction)

    def test_worked_example_from_methodology(self) -> None:
        # C_max = 8, k_p99 = 2, derate = 0.875 -> C_target = 3.5
        assert c_target(8.0, k=2.0) == pytest.approx(3.5)

    def test_kokoro_case_c_max_one(self) -> None:
        # C_max=1 with k=2 yields a fractional target, which is exactly why
        # ModelEndpointConfig.scaling_target_value must be a float, not an int.
        assert c_target(1.0, k=2.0) == pytest.approx(0.4375)

    @pytest.mark.parametrize(
        ("c_max", "k", "derate"),
        [
            (0.0, 2.0, 0.875),  # no measured ceiling
            (-1.0, 2.0, 0.875),
            (8.0, 0.5, 0.875),  # k<1 would target above the knee
            (8.0, 2.0, 0.0),  # derate of zero means never send traffic
            (8.0, 2.0, 1.5),  # derate above 1 targets past the knee
        ],
    )
    def test_rejects_nonsense_inputs(self, c_max: float, k: float, derate: float) -> None:
        with pytest.raises(ValueError):
            c_target(c_max, k, derate)


class TestCSloCap:
    def test_wait_budget_of_zero_allows_only_the_running_request(self) -> None:
        assert c_slo_cap(0.0, s_mean_s=0.5) == pytest.approx(1.0)

    def test_budget_of_four_service_times_allows_five_deep(self) -> None:
        assert c_slo_cap(2.0, s_mean_s=0.5) == pytest.approx(5.0)

    def test_rejects_nonpositive_service_time(self) -> None:
        with pytest.raises(ValueError):
            c_slo_cap(2.0, s_mean_s=0.0)

    def test_rejects_negative_wait_budget(self) -> None:
        with pytest.raises(ValueError):
            c_slo_cap(-1.0, s_mean_s=0.5)


class TestWMaxForSlo:
    """``W_max`` derived from the end-to-end SLO instead of chosen."""

    def test_queue_gets_what_service_does_not_spend(self) -> None:
        # The worked example: a 3s promise on kokoro's 165ms tail leaves 2.835s.
        assert w_max_for_slo(3.0, KOKORO_S_P95_S) == pytest.approx(2.835, abs=1e-3)

    def test_uses_the_tail_not_the_mean(self) -> None:
        # A mean-sized budget would hand out 59ms more than the tail can afford,
        # which is exactly the request that misses the promise.
        tail_budget = w_max_for_slo(3.0, KOKORO_S_P95_S)
        mean_budget = w_max_for_slo(3.0, KOKORO_S_MEAN_S)
        assert tail_budget < mean_budget
        assert mean_budget - tail_budget == pytest.approx(KOKORO_S_P95_S - KOKORO_S_MEAN_S)

    def test_a_tight_slo_leaves_almost_no_queue(self) -> None:
        # Why the old 300ms budget was never an end-to-end number: 135ms of slack
        # is barely one service time, so the queue is not a queue.
        assert w_max_for_slo(0.3, KOKORO_S_P95_S) == pytest.approx(0.135, abs=1e-3)

    def test_clamps_at_zero_rather_than_going_negative(self) -> None:
        # A negative budget would read as a queue depth of zero being sufficient,
        # when in fact the SLO is already missed before queueing.
        assert w_max_for_slo(0.1, s_p95_s=4.0) == 0.0

    @pytest.mark.parametrize(("slo", "s_p95"), [(0.0, 0.16), (-1.0, 0.16), (3.0, 0.0), (3.0, -1.0)])
    def test_rejects_nonsense_inputs(self, slo: float, s_p95: float) -> None:
        with pytest.raises(ValueError):
            w_max_for_slo(slo, s_p95)


class TestSloIsFeasible:
    def test_a_fast_model_under_a_generous_slo(self) -> None:
        assert slo_is_feasible(3.0, KOKORO_S_P95_S)

    def test_a_model_slower_than_its_own_slo_is_infeasible(self) -> None:
        # No queue depth, instance count, or policy rescues this: the unqueued
        # tail already misses.
        assert not slo_is_feasible(3.0, s_p95_s=3.5)

    def test_service_time_exactly_at_the_slo_is_infeasible(self) -> None:
        # Boundary: spending the entire promise on service leaves nothing, and a
        # queue of depth zero still has the request waiting to be scheduled.
        assert not slo_is_feasible(3.0, s_p95_s=3.0)


class TestFitsSlo:
    def test_a_derived_w_max_exactly_fits_by_construction(self) -> None:
        # w_max_for_slo is the inverse, so round-tripping must land on the SLO
        # rather than a hair over it.
        w = w_max_for_slo(3.0, KOKORO_S_P95_S)
        fits, deadline = fits_slo(w, KOKORO_S_P95_S, slo_s=3.0)
        assert fits
        assert deadline == pytest.approx(3.0)

    def test_the_deployed_kokoro_config_misses_the_three_second_slo(self) -> None:
        # The regression this function exists for. max_added_wait_s=20.0 shipped
        # beside ttfab_budget_ms=300 and nothing related them, so a request using
        # its full queue allowance took ~20.2s to first byte.
        fits, deadline = fits_slo(20.0, KOKORO_S_P95_S, slo_s=3.0)
        assert not fits
        assert deadline == pytest.approx(20.165, abs=1e-3)

    def test_fitting_the_60s_ceiling_does_not_imply_fitting_the_slo(self) -> None:
        # Both bounds apply and neither implies the other. This is the case that
        # passed every check the code had before: legal on the platform, and
        # silently 6.7x over the promise.
        ceiling_ok, _ = fits_invocation_ceiling(20.0, KOKORO_S_P95_S)
        slo_ok, _ = fits_slo(20.0, KOKORO_S_P95_S, slo_s=3.0)
        assert ceiling_ok
        assert not slo_ok

    def test_the_slo_can_also_be_looser_than_the_ceiling(self) -> None:
        # Nothing stops a stated SLO exceeding 60s; the platform still hangs up,
        # so the ceiling has to keep binding independently.
        slo_ok, _ = fits_slo(70.0, KOKORO_S_P95_S, slo_s=120.0)
        ceiling_ok, _ = fits_invocation_ceiling(70.0, KOKORO_S_P95_S)
        assert slo_ok
        assert not ceiling_ok

    def test_agrees_with_request_deadline_s(self) -> None:
        # One definition of "what a queued request occupies", not two.
        _, deadline = fits_slo(2.0, KOKORO_S_P95_S, slo_s=3.0)
        assert deadline == request_deadline_s(2.0, KOKORO_S_P95_S)

    def test_rejects_a_nonpositive_slo(self) -> None:
        with pytest.raises(ValueError):
            fits_slo(2.0, KOKORO_S_P95_S, slo_s=0.0)


class TestSloAndQueueDepthTogether:
    """The end-to-end consequence: what the derived budget does to ``Q_max``."""

    def test_the_three_second_slo_shrinks_the_queue_sevenfold(self) -> None:
        # Deployed queue_max_depth was 296, derived from the wrong W_max. At the
        # SLO-derived budget the same C_max gives ~30.
        w = w_max_for_slo(3.0, KOKORO_S_P95_S)
        derived = q_per_instance(1.153, KOKORO_S_MEAN_S, w)
        shipped = q_per_instance(1.153, KOKORO_S_MEAN_S, 20.0)
        assert derived == 30
        assert shipped == 217
        assert shipped > derived * 7

    def test_the_wait_budget_does_not_bind_c_target_at_three_seconds(self) -> None:
        # Worth pinning because it is counterintuitive: loosening the SLO from
        # 300ms to 3s does not move the tracked target at all. c_slo_cap lands at
        # ~27.7, far above any measured kokoro knee, so surge headroom still binds
        # and only the queue depth changes.
        w = w_max_for_slo(3.0, KOKORO_S_P95_S)
        assert c_slo_cap(w, KOKORO_S_MEAN_S) == pytest.approx(27.7, abs=0.1)
        for c_max in (1.153, 1.392, 1.790):
            _, binding = effective_c_target(
                c_max, k=2.0, s_mean_s=KOKORO_S_MEAN_S, max_added_wait_s=w
            )
            assert binding == "surge_headroom"


class TestEffectiveCTarget:
    def test_surge_headroom_binds_for_a_fast_model(self) -> None:
        # S=0.12s (Kokoro-ish): the wait budget permits ~17 deep, so the
        # k-derated knee is the tighter constraint.
        target, binding = effective_c_target(c_max=8.0, k=2.0, s_mean_s=0.12, max_added_wait_s=2.0)
        assert binding == "surge_headroom"
        assert target == pytest.approx(3.5)

    def test_slo_budget_binds_for_a_slow_model(self) -> None:
        # S=4s: two seconds of added wait cannot cover even one queued request,
        # so the wait budget binds regardless of how high the knee sits.
        target, binding = effective_c_target(c_max=16.0, k=2.0, s_mean_s=4.0, max_added_wait_s=2.0)
        assert binding == "slo_wait_budget"
        assert target == pytest.approx(1.5)

    def test_takes_the_smaller_of_the_two(self) -> None:
        for s_mean in (0.05, 0.5, 1.0, 4.0, 20.0):
            target, _ = effective_c_target(c_max=8.0, k=2.0, s_mean_s=s_mean, max_added_wait_s=2.0)
            assert target <= c_target(8.0, 2.0) + 1e-9
            assert target <= c_slo_cap(2.0, s_mean) + 1e-9


class TestNInstances:
    def test_littles_law_then_divide(self) -> None:
        # 450 rps x 0.12s = 54 concurrent; at 3.5 per instance -> 16 instances.
        assert n_instances(450.0, s_mean_s=0.12, target_concurrency=3.5) == 16

    def test_rounds_up_never_down(self) -> None:
        # 10 rps x 1s = 10 concurrent at 3 per instance = 3.33 -> 4.
        assert n_instances(10.0, s_mean_s=1.0, target_concurrency=3.0) == 4

    def test_trickle_load_still_needs_one_instance(self) -> None:
        assert n_instances(0.01, s_mean_s=0.12, target_concurrency=3.5) == 1

    def test_zero_load_needs_none(self) -> None:
        # Scale-to-zero is a deployment question, but the math should not claim
        # an instance is required for no traffic.
        assert n_instances(0.0, s_mean_s=0.12, target_concurrency=3.5) == 0

    def test_kokoro_fractional_target(self) -> None:
        # C_target=0.4375 with S=0.12s: each instance sustains ~3.6 rps.
        assert n_instances(30.0, s_mean_s=0.12, target_concurrency=0.4375) == 9

    @pytest.mark.parametrize(
        ("rate", "s_mean", "target"),
        [(-1.0, 0.12, 3.5), (10.0, 0.0, 3.5), (10.0, 0.12, 0.0), (10.0, 0.12, -1.0)],
    )
    def test_rejects_nonsense_inputs(self, rate: float, s_mean: float, target: float) -> None:
        with pytest.raises(ValueError):
            n_instances(rate, s_mean, target)


class TestNInstancesFromStreams:
    def test_divides_streams_directly(self) -> None:
        assert n_instances_from_streams(54.0, target_concurrency=3.5) == 16

    def test_agrees_with_rps_path_after_conversion(self) -> None:
        # The two entry points must not disagree: lambda x S concurrent streams
        # is the same fleet as lambda rps.
        assert n_instances_from_streams(450.0 * 0.12, 3.5) == n_instances(450.0, 0.12, 3.5)

    def test_zero_streams_needs_none(self) -> None:
        assert n_instances_from_streams(0.0, target_concurrency=3.5) == 0


class TestLambdaCap:
    def test_littles_law_rearranged(self) -> None:
        assert lambda_cap_per_instance(1.0, s_mean_s=0.12) == pytest.approx(8.333, abs=1e-3)

    def test_kokoro_measured_ceiling(self) -> None:
        # C_max=1 at S=0.12s is the ~8.3 rps that the closed-loop harness
        # mistook for a plateau at concurrency 4.
        assert lambda_cap_per_instance(1.0, 0.12) == pytest.approx(25.0 / 3.0)


class TestQPerInstance:
    def test_depth_is_a_time_budget(self) -> None:
        # 8.33 rps drain rate x 2s of slack = 16 slots.
        assert q_per_instance(1.0, s_mean_s=0.12, max_added_wait_s=2.0) == 16

    def test_rounds_down(self) -> None:
        # 10 rps x 0.55s = 5.5 -> 5. A slot that cannot drain in time is worse
        # than a rejection: the client waits and fails anyway.
        assert q_per_instance(1.0, s_mean_s=0.1, max_added_wait_s=0.55) == 5

    def test_zero_wait_budget_means_no_queue(self) -> None:
        assert q_per_instance(1.0, s_mean_s=0.12, max_added_wait_s=0.0) == 0

    def test_sagemaker_ceiling_case(self) -> None:
        # W_max=50s (under the 60s invocation limit) on Kokoro allows a deep
        # queue — this is why the queue can cover a doubling here.
        assert q_per_instance(1.0, s_mean_s=0.12, max_added_wait_s=50.0) == 416


class TestWAbsorbed:
    def test_flat_traffic_covers_any_lag(self) -> None:
        assert w_absorbed(2.0, k=1.0) == math.inf

    def test_doubling_consumes_slack_at_wall_clock_rate(self) -> None:
        # Worked example: W_max=2s, k=2 -> W_absorbed=2s.
        assert w_absorbed(2.0, k=2.0) == pytest.approx(2.0)

    def test_tripling_halves_the_absorbed_lag(self) -> None:
        assert w_absorbed(2.0, k=3.0) == pytest.approx(1.0)

    def test_sagemaker_ceiling_absorbs_a_doubling(self) -> None:
        # The load-bearing fact for our queue design: 50s of slack covers a
        # doubling for 50s, which exceeds a T_total we can plausibly reach.
        assert w_absorbed(50.0, k=2.0) == pytest.approx(50.0)

    def test_rejects_k_below_one(self) -> None:
        with pytest.raises(ValueError):
            w_absorbed(2.0, k=0.5)


class TestQueueCoversSurge:
    def test_true_when_slack_exceeds_lag(self) -> None:
        assert queue_covers_surge(max_added_wait_s=50.0, k=2.0, t_total_s=45.0)

    def test_false_when_lag_exceeds_slack(self) -> None:
        assert not queue_covers_surge(max_added_wait_s=50.0, k=2.0, t_total_s=180.0)

    def test_boundary_is_inclusive(self) -> None:
        assert queue_covers_surge(max_added_wait_s=50.0, k=2.0, t_total_s=50.0)

    def test_tighter_at_higher_k(self) -> None:
        # Same 50s budget, same 45s lag: covers k=2, fails k=3.
        assert queue_covers_surge(50.0, k=2.0, t_total_s=45.0)
        assert not queue_covers_surge(50.0, k=3.0, t_total_s=45.0)

    def test_matches_the_closed_form(self) -> None:
        # W_max >= (k-1) x T_total, stated directly.
        for k in (2.0, 3.0, 5.0):
            for t_total in (10.0, 45.0, 120.0):
                expected = 50.0 >= (k - 1) * t_total
                assert queue_covers_surge(50.0, k, t_total) is expected


class TestEffectiveHeadroomLag:
    def test_queue_covering_everything_floors_at_metric_period(self) -> None:
        lag, floored = effective_headroom_lag_s(50.0, k=2.0, t_total_s=45.0)
        assert floored is True
        assert lag == CLOUDWATCH_HIGH_RES_PERIOD_S

    def test_uncovered_remainder_is_reported(self) -> None:
        # 2s of slack at k=2 absorbs 2s of a 180s lag.
        lag, floored = effective_headroom_lag_s(2.0, k=2.0, t_total_s=180.0)
        assert floored is False
        assert lag == pytest.approx(178.0)

    def test_flat_traffic_needs_no_standing_headroom(self) -> None:
        lag, floored = effective_headroom_lag_s(2.0, k=1.0, t_total_s=180.0)
        assert floored is True
        assert lag == CLOUDWATCH_HIGH_RES_PERIOD_S

    def test_never_returns_below_the_floor(self) -> None:
        for t_total in (0.0, 1.0, 9.9, 10.0, 60.0):
            lag, _ = effective_headroom_lag_s(2.0, k=2.0, t_total_s=t_total)
            assert lag >= CLOUDWATCH_HIGH_RES_PERIOD_S


class TestUtilizationAtK:
    def test_headroom_is_the_cost_of_surge_tolerance(self) -> None:
        assert utilization_at_k(1.0, derate=1.0) == pytest.approx(1.0)
        assert utilization_at_k(2.0, derate=1.0) == pytest.approx(0.5)
        assert utilization_at_k(5.0, derate=1.0) == pytest.approx(0.2)

    def test_k_of_five_is_the_stop_and_rethink_signal(self) -> None:
        # ~17% utilization: six instances paid for per instance of load. This is
        # the number the report cites when recommending against autoscaling as
        # the primary tool.
        assert utilization_at_k(5.0) == pytest.approx(0.175)

    def test_is_the_inverse_of_the_c_target_derating(self) -> None:
        for k in (1.0, 2.0, 3.0, 4.0):
            assert utilization_at_k(k) == pytest.approx(c_target(1.0, k) / 1.0)


class TestMinSamplesForK:
    def test_counts_whole_windows(self) -> None:
        assert min_samples_for_k(3600.0, t_total_s=180.0) == 20

    def test_partial_window_does_not_count(self) -> None:
        assert min_samples_for_k(350.0, t_total_s=180.0) == 1

    def test_no_history_yields_no_samples(self) -> None:
        # Our current situation: this is what makes measuring k impossible and
        # forces the scenario-argument approach.
        assert min_samples_for_k(0.0, t_total_s=180.0) == 0

    def test_rejects_nonpositive_lag(self) -> None:
        with pytest.raises(ValueError):
            min_samples_for_k(3600.0, t_total_s=0.0)


class TestDefaultDerate:
    def test_matches_the_methodology(self) -> None:
        assert DEFAULT_DERATE == 0.875


class TestRequestDeadline:
    def test_is_the_wait_plus_the_service(self) -> None:
        assert request_deadline_s(20.0, s_p95_s=0.3) == pytest.approx(20.3)

    def test_uses_p95_not_mean(self) -> None:
        # Asserted as a property rather than assumed from the signature: a deadline
        # sized on the mean is missed by half the requests that reach it.
        on_mean = request_deadline_s(20.0, s_p95_s=0.11)
        on_p95 = request_deadline_s(20.0, s_p95_s=0.55)
        assert on_p95 > on_mean

    def test_a_zero_wait_is_still_the_service_time(self) -> None:
        assert request_deadline_s(0.0, s_p95_s=0.55) == pytest.approx(0.55)

    @pytest.mark.parametrize(("wait", "p95"), [(-1.0, 0.3), (20.0, 0.0), (20.0, -0.3)])
    def test_rejects_impossible_inputs(self, wait: float, p95: float) -> None:
        with pytest.raises(ValueError):
            request_deadline_s(wait, s_p95_s=p95)


class TestFitsInvocationCeiling:
    def test_the_deployed_kokoro_budget_fits(self) -> None:
        # W_max=20s at the measured p95 of ~0.55s: 20.55s against a 60s ceiling.
        # The number config.py ships, so a regression here means the deployed
        # endpoint stopped being feasible by this tool's own standard.
        fits, deadline = fits_invocation_ceiling(20.0, s_p95_s=0.55)
        assert fits
        assert deadline == pytest.approx(20.55)

    def test_a_wait_past_the_ceiling_does_not_fit(self) -> None:
        # The failure this exists to catch: a generous W_max chosen to absorb a long
        # T_total, which the platform then refuses to honour.
        fits, deadline = fits_invocation_ceiling(70.0, s_p95_s=0.55)
        assert not fits
        assert deadline == pytest.approx(70.55)

    def test_service_time_alone_can_break_it(self) -> None:
        # No queue at all, and still infeasible: a model slower than the ceiling
        # cannot be deployed behind a real-time endpoint at any W_max.
        fits, _ = fits_invocation_ceiling(0.0, s_p95_s=61.0)
        assert not fits

    def test_sitting_exactly_on_the_ceiling_fits(self) -> None:
        # Inclusive on purpose: the boundary is a valid plan, and excluding it would
        # report INFEASIBLE for a design that meets the stated constraint exactly.
        fits, _ = fits_invocation_ceiling(59.5, s_p95_s=0.5)
        assert fits

    def test_the_ceiling_is_sixty_seconds(self) -> None:
        assert SAGEMAKER_INVOCATION_CEILING_S == 60.0

    def test_an_explicit_ceiling_overrides_the_default(self) -> None:
        # Another platform, or a stricter internal SLO, without a second function.
        assert fits_invocation_ceiling(20.0, s_p95_s=0.55, ceiling_s=10.0)[0] is False


class TestMaxAddedWaitUnderCeiling:
    def test_is_the_ceiling_less_the_service_time(self) -> None:
        assert max_added_wait_under_ceiling(0.55) == pytest.approx(59.45)

    def test_round_trips_with_the_feasibility_check(self) -> None:
        # The property that makes it a usable recommendation: the largest budget it
        # reports must itself pass. Anything else would advise an infeasible plan.
        largest = max_added_wait_under_ceiling(0.55)
        assert fits_invocation_ceiling(largest, s_p95_s=0.55)[0]

    def test_a_model_slower_than_the_ceiling_gets_no_budget(self) -> None:
        # Clamped rather than negative: a negative budget reads as "shorten the
        # queue", but no queue length rescues a model this slow.
        assert max_added_wait_under_ceiling(75.0) == 0.0

    def test_rejects_nonpositive_service_time(self) -> None:
        with pytest.raises(ValueError):
            max_added_wait_under_ceiling(0.0)
