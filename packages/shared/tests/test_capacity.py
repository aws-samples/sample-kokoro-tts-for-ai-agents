"""Tests for capacity planning math.

Seeded with the worked examples from the scaling methodology, so a regression
here means the plan output stopped matching the documented reasoning.
"""

from __future__ import annotations

import pytest

from shared.capacity import (
    SAGEMAKER_INVOCATION_CEILING_S,
    fits_invocation_ceiling,
    fits_slo,
    max_added_wait_under_ceiling,
    n_instances,
    n_instances_from_streams,
    request_deadline_s,
    scale_thresholds,
    shed_probability,
    slo_is_feasible,
    utilization_for_occupancy,
    w_max_for_slo,
)

#: Kokoro-82M on ml.g5.xlarge, bidi transport, from
#: artifacts/cmax-kokoro-82m-bidi-g5xlarge-139b9068.json. The numbers the 3s SLO was
#: actually reasoned about, so a regression here means the worked example moved.
KOKORO_S_MEAN_S = 0.10602401316328536
KOKORO_S_P95_S = 0.1645768812391907

#: Service time from the latency-vs-queue-position fit on the same artifact:
#: TTFAB(q) = 33.9ms + (q+1) x 57.5ms, R^2 0.9998 over 32,960 OK events. Distinct
#: from KOKORO_S_MEAN_S, which conflates 34ms of client RTT with 59ms of service --
#: the reason the queueing math takes this number and not that one.
KOKORO_SERVICE_S = 0.0575

#: Q_max = W_max / service at the 3s SLO, and T_total from the 07-31 activity
#: history (3m51s). The pair the worked examples below use.
KOKORO_Q_MAX = 50.0
KOKORO_T_TOTAL_S = 231.0


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
        # Why the deleted 300ms budget was never an end-to-end number: 135ms of
        # slack is barely one service time, so the queue is not a queue.
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
        # beside a 300ms budget and nothing related them, so a request using its
        # full queue allowance took ~20.2s to first byte.
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


class TestScaleThresholds:
    """The two derived numbers, from ``Q_max`` and the chosen surge ratio."""

    def test_the_worked_example(self) -> None:
        # The plan's headline: Q_max 50 at a 1.25 surge ratio gives 37.5 and 25.0,
        # and 2->1 scale-in is unsafe so the smallest safe fleet is 3.
        out = scale_thresholds(KOKORO_Q_MAX, 1.25)
        assert out.c_scale_max == pytest.approx(37.5)
        assert out.c_scale_min == pytest.approx(25.0)
        assert out.min_safe_instances == 3

    def test_unpacks_positionally_in_order(self) -> None:
        # Callers unpack this straight into a deployed policy, so the order is
        # part of the contract: out, then in, then the fleet floor.
        c_out, c_in, n = scale_thresholds(KOKORO_Q_MAX, 1.25)
        assert (c_out, c_in, n) == (37.5, 25.0, 3)

    def test_flat_traffic_needs_no_headroom_at_all(self) -> None:
        # k=1.0 is "no surge expected": both thresholds sit at Q_max, which is the
        # degenerate policy of running the queue right up to the SLO line.
        out = scale_thresholds(KOKORO_Q_MAX, 1.0)
        assert out.c_scale_max == pytest.approx(KOKORO_Q_MAX)
        assert out.c_scale_min == pytest.approx(KOKORO_Q_MAX)
        # Coincident thresholds mean every scale-in immediately re-triggers
        # scale-out, so there is no safe fleet size to report rather than a
        # large-but-finite one.
        assert out.min_safe_instances is None

    def test_scale_out_is_always_above_scale_in(self) -> None:
        # The property that keeps the policy from fighting itself. Asserted across
        # the whole legal range because a transposition here deploys as an endpoint
        # that scales in and out on the same datapoint.
        for k in (1.0, 1.05, 1.1, 1.25, 1.4, 1.49):
            out = scale_thresholds(KOKORO_Q_MAX, k)
            assert out.c_scale_max >= out.c_scale_min

    def test_both_thresholds_stay_inside_q_max(self) -> None:
        # Neither may exceed the measured SLO limit: a threshold above Q_max is a
        # policy that only reacts after the promise is already broken.
        for k in (1.0, 1.25, 1.49):
            out = scale_thresholds(KOKORO_Q_MAX, k)
            assert out.c_scale_max <= KOKORO_Q_MAX
            assert out.c_scale_min <= KOKORO_Q_MAX

    def test_a_bigger_surge_ratio_triggers_earlier(self) -> None:
        # Monotone in the chosen ratio, in both thresholds. Expecting more surge
        # can only make the policy more eager, never less.
        ks = [1.0, 1.1, 1.25, 1.4]
        outs = [scale_thresholds(KOKORO_Q_MAX, k) for k in ks]
        assert [o.c_scale_max for o in outs] == sorted((o.c_scale_max for o in outs), reverse=True)
        assert [o.c_scale_min for o in outs] == sorted((o.c_scale_min for o in outs), reverse=True)

    def test_scales_linearly_with_q_max(self) -> None:
        # Both are fractions of Q_max and nothing else, so a rerun that doubles
        # Q_max on a bigger instance doubles both. This is what makes the tooling
        # a rerun rather than a re-derivation.
        small = scale_thresholds(25.0, 1.25)
        large = scale_thresholds(50.0, 1.25)
        assert large.c_scale_max == pytest.approx(2 * small.c_scale_max)
        assert large.c_scale_min == pytest.approx(2 * small.c_scale_min)
        assert large.min_safe_instances == small.min_safe_instances

    def test_min_safe_instances_is_the_fleet_where_scale_in_stops_flapping(self) -> None:
        # Removing 1 of N multiplies survivors' concurrency by N/(N-1). The
        # reported N is the smallest where that lands strictly under the scale-out
        # threshold, and N-1 must genuinely fail -- otherwise the number is just
        # conservative rather than minimal.
        out = scale_thresholds(KOKORO_Q_MAX, 1.25)
        n = out.min_safe_instances
        assert n is not None
        assert out.c_scale_min * n / (n - 1) <= out.c_scale_max
        assert out.c_scale_min * (n - 1) / (n - 2) > out.c_scale_max

    def test_the_two_to_one_case_lands_exactly_on_q_max(self) -> None:
        # Limit #2 from the plan, pinned numerically: kokoro runs min_instances=1,
        # so 2->1 is the common case and it breaches the SLO on the way down.
        out = scale_thresholds(KOKORO_Q_MAX, 1.25)
        assert out.c_scale_min * 2 / 1 == pytest.approx(KOKORO_Q_MAX)

    def test_the_three_to_two_case_lands_exactly_on_the_scale_out_threshold(self) -> None:
        # The other half of limit #2: 3->2 does not breach the SLO but re-triggers
        # scale-out immediately, which is a flap rather than an outage.
        out = scale_thresholds(KOKORO_Q_MAX, 1.25)
        assert out.c_scale_min * 3 / 2 == pytest.approx(out.c_scale_max)

    def test_a_gentler_surge_ratio_needs_a_bigger_fleet_to_scale_in_safely(self) -> None:
        # Counterintuitive and worth pinning: expecting *less* surge makes scale-in
        # harder, not easier. Safety depends on the ratio between the thresholds,
        # and a small h puts them nearly on top of each other -- at h=0.05 they are
        # 47.5 and 45.0, a ratio of 1.056, so N/(N-1) only fits from N=19 up.
        assert scale_thresholds(KOKORO_Q_MAX, 1.05).min_safe_instances == 19
        assert scale_thresholds(KOKORO_Q_MAX, 1.25).min_safe_instances == 3

    def test_min_safe_instances_falls_as_the_surge_ratio_grows(self) -> None:
        # The same fact as monotonicity, so a plan that loosens the ratio to make
        # scale-in safe is doing something real rather than coincidental.
        fleets = [
            scale_thresholds(KOKORO_Q_MAX, k).min_safe_instances for k in (1.05, 1.1, 1.25, 1.4)
        ]
        assert fleets == [19, 9, 3, 2]

    def test_a_fleet_exactly_on_the_boundary_counts_as_safe(self) -> None:
        # N/(N-1) <= r is inclusive, and the exact cases are the ones float error
        # would silently round the wrong way. At h=0.1 the thresholds are 45 and 40,
        # so 9 -> 8 lands on 45.0 exactly; reporting 10 here would pad the fleet on
        # nothing but representation error.
        out = scale_thresholds(KOKORO_Q_MAX, 1.1)
        assert out.min_safe_instances == 9
        assert out.c_scale_min * 9 / 8 == pytest.approx(out.c_scale_max)

    def test_never_claims_a_single_instance_fleet_is_safe(self) -> None:
        # N=1 has no scale-in to be safe about (min_instances floors at 1 on
        # SageMaker), and N/(N-1) is undefined there. Floored at 2 so the number
        # is always a fleet you can actually remove an instance from.
        for k in (1.0001, 1.01, 1.05, 1.25, 1.49):
            out = scale_thresholds(KOKORO_Q_MAX, k)
            assert out.min_safe_instances is None or out.min_safe_instances >= 2

    def test_rejects_a_shrinking_surge(self) -> None:
        with pytest.raises(ValueError, match=">= 1"):
            scale_thresholds(KOKORO_Q_MAX, 0.9)

    def test_refuses_a_ratio_that_would_disable_scale_in(self) -> None:
        # At h=0.5 the scale-in threshold reaches zero and beyond it goes
        # negative, which deploys as a policy that never scales in -- refused
        # rather than clamped, since a clamped 0.0 looks like a real threshold.
        with pytest.raises(ValueError, match="never scales in"):
            scale_thresholds(KOKORO_Q_MAX, 1.5)
        with pytest.raises(ValueError, match="never scales in"):
            scale_thresholds(KOKORO_Q_MAX, 2.0)

    @pytest.mark.parametrize("q_max", [0.0, -1.0])
    def test_rejects_a_nonpositive_q_max(self, q_max: float) -> None:
        with pytest.raises(ValueError):
            scale_thresholds(q_max, 1.25)


class TestUtilizationForOccupancy:
    """Occupancy read as load, which is the step that makes ``Q_max`` interpretable."""

    def test_the_textbook_pairs(self) -> None:
        # rho = L/(1+L): occupancy 1 is half utilized, 4 is 80%, 9 is 90%.
        assert utilization_for_occupancy(1.0) == pytest.approx(0.5)
        assert utilization_for_occupancy(4.0) == pytest.approx(0.8)
        assert utilization_for_occupancy(9.0) == pytest.approx(0.9)

    def test_the_derived_scale_out_threshold_is_97_percent_utilized(self) -> None:
        # Limit #1 from the plan. "Three quarters of Q_max" sounds like three
        # quarters of the way to trouble; it is 0.974 utilization.
        c_scale_max = scale_thresholds(KOKORO_Q_MAX, 1.25).c_scale_max
        assert utilization_for_occupancy(c_scale_max) == pytest.approx(0.974, abs=1e-3)

    def test_eighty_percent_utilization_is_eight_percent_of_q_max(self) -> None:
        # The same fact from the other direction: rho=0.80 is occupancy 4, i.e.
        # 8% of a Q_max of 50. Depth is exponentially sensitive to utilization.
        assert utilization_for_occupancy(4.0) == pytest.approx(0.8)
        assert 4.0 / KOKORO_Q_MAX == pytest.approx(0.08)

    def test_an_idle_queue_is_unutilized(self) -> None:
        assert utilization_for_occupancy(0.0) == 0.0

    def test_approaches_but_never_reaches_saturation(self) -> None:
        # An observed occupancy cannot express rho >= 1: a queue at critical load
        # has no steady-state mean to have measured in the first place.
        assert utilization_for_occupancy(1e6) < 1.0
        assert utilization_for_occupancy(1e6) > 0.999

    def test_rejects_negative_occupancy(self) -> None:
        with pytest.raises(ValueError):
            utilization_for_occupancy(-1.0)


class TestShedProbability:
    """P(the queue breaches ``Q_max``) while waiting out one ``T_total``."""

    def test_reproducible_on_a_fixed_seed(self) -> None:
        # The number goes into a published plan beside the artifact it came from,
        # so it has to be recomputable from the seed alone.
        args = (20.0, KOKORO_Q_MAX, 60.0, KOKORO_SERVICE_S)
        assert shed_probability(*args, seed=7) == shed_probability(*args, seed=7)

    def test_a_different_seed_is_a_different_sample(self) -> None:
        # Guards against a simulation that silently ignores its randomness and
        # would report the same figure for every input.
        args = (20.0, KOKORO_Q_MAX, 60.0, KOKORO_SERVICE_S)
        samples = {shed_probability(*args, seed=s) for s in range(6)}
        assert len(samples) > 1

    def test_does_not_disturb_the_global_random_stream(self) -> None:
        # A private Random instance, so a caller that seeded the module-global
        # generator gets the same sequence back afterwards -- and, conversely,
        # cannot change a published number by reseeding.
        import random

        random.seed(99)
        expected = [random.random() for _ in range(3)]
        random.seed(99)
        shed_probability(20.0, KOKORO_Q_MAX, 60.0, KOKORO_SERVICE_S, seed=1)
        assert [random.random() for _ in range(3)] == expected

    def test_monotone_in_occupancy(self) -> None:
        # The load-bearing property: starting closer to Q_max cannot make a breach
        # less likely. Checked on one seed so the comparison is paired.
        probs = [
            shed_probability(occ, KOKORO_Q_MAX, KOKORO_T_TOTAL_S, KOKORO_SERVICE_S, seed=3)
            for occ in (1.0, 4.0, 10.0, 20.0, 30.0, 37.5)
        ]
        assert probs == sorted(probs)

    def test_an_idle_queue_never_sheds(self) -> None:
        assert shed_probability(0.0, KOKORO_Q_MAX, KOKORO_T_TOTAL_S, KOKORO_SERVICE_S) == 0.0

    def test_starting_at_or_above_q_max_has_already_shed(self) -> None:
        # No simulation needed, and reported as certainty rather than as a sample
        # that happened to hit on the first step.
        assert shed_probability(50.0, KOKORO_Q_MAX, 60.0, KOKORO_SERVICE_S) == 1.0
        assert shed_probability(80.0, KOKORO_Q_MAX, 60.0, KOKORO_SERVICE_S) == 1.0

    def test_no_time_to_wait_means_nothing_can_go_wrong(self) -> None:
        # T_total of zero is an instance that arrives instantly; the threshold
        # question disappears with it.
        assert shed_probability(37.5, KOKORO_Q_MAX, 0.0, KOKORO_SERVICE_S) == 0.0

    def test_monotone_in_t_total(self) -> None:
        # A slower scale-out cannot make a breach less likely, which is why
        # T_total has to be measured rather than assumed.
        probs = [
            shed_probability(30.0, KOKORO_Q_MAX, t, KOKORO_SERVICE_S, seed=5)
            for t in (1.0, 10.0, 60.0, 240.0)
        ]
        assert probs == sorted(probs)

    def test_the_derived_threshold_is_a_coin_flip_or_worse(self) -> None:
        # Limit #1, made falsifiable: scaling out at 0.75 x Q_max and then waiting
        # a measured 231s breaches the SLO more often than not. This is the number
        # the surge_survival finding reports.
        p = shed_probability(37.5, KOKORO_Q_MAX, KOKORO_T_TOTAL_S, KOKORO_SERVICE_S, seed=1234)
        assert p > 0.5

    def test_a_low_utilization_threshold_survives_the_same_lag(self) -> None:
        # The contrast that makes the finding actionable rather than fatalistic:
        # same Q_max, same T_total, trigger at rho=0.80 instead, and the breach
        # becomes rare. Not zero -- Poisson variance never gives a free pass.
        p = shed_probability(4.0, KOKORO_Q_MAX, KOKORO_T_TOTAL_S, KOKORO_SERVICE_S, seed=1234)
        assert p < 0.1

    def test_a_deeper_queue_absorbs_more(self) -> None:
        # Monotone in Q_max at fixed occupancy: more slots between the trigger and
        # the SLO line can only help.
        shallow = shed_probability(10.0, 20.0, 60.0, KOKORO_SERVICE_S, seed=11)
        deep = shed_probability(10.0, 200.0, 60.0, KOKORO_SERVICE_S, seed=11)
        assert deep <= shallow

    def test_trials_bound_the_resolution(self) -> None:
        # A probability is always a multiple of 1/trials, which is the honest
        # precision of the number and worth pinning so a caller reading four
        # decimal places knows better.
        p = shed_probability(20.0, KOKORO_Q_MAX, 60.0, KOKORO_SERVICE_S, seed=2, trials=10)
        assert p * 10 == pytest.approx(round(p * 10))

    @pytest.mark.parametrize(
        ("occ", "q_max", "t_total", "service", "trials"),
        [
            (-1.0, 50.0, 60.0, 0.0575, 200),  # negative occupancy
            (20.0, 0.0, 60.0, 0.0575, 200),  # no queue to breach
            (20.0, 50.0, -1.0, 0.0575, 200),  # negative lag
            (20.0, 50.0, 60.0, 0.0, 200),  # instantaneous service
            (20.0, 50.0, 60.0, 0.0575, 0),  # no trials to average
        ],
    )
    def test_rejects_nonsense_inputs(
        self, occ: float, q_max: float, t_total: float, service: float, trials: int
    ) -> None:
        with pytest.raises(ValueError):
            shed_probability(occ, q_max, t_total, service, trials=trials)


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

    def test_divides_by_the_derived_scale_out_threshold(self) -> None:
        # The target is C_scale_max now, not a C_max derivative: that is the
        # concurrency the deployed policy actually holds each instance at, so it
        # is what the fleet size has to be computed against.
        c_scale_max = scale_thresholds(KOKORO_Q_MAX, 1.25).c_scale_max
        assert n_instances(450.0, KOKORO_SERVICE_S, c_scale_max) == 1
        assert n_instances(45000.0, KOKORO_SERVICE_S, c_scale_max) == 69

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
