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
    c_slo_cap,
    c_target,
    effective_c_target,
    effective_headroom_lag_s,
    lambda_cap_per_instance,
    min_samples_for_k,
    n_instances,
    n_instances_from_streams,
    q_per_instance,
    queue_covers_surge,
    utilization_at_k,
    w_absorbed,
)


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
