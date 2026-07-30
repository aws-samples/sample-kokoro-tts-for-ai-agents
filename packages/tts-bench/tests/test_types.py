"""Tests for the capacity-planning models in tts_bench.types.

These models are the contract between the measurement phases (``cmax``,
``ttotal``) and the planner. Their validators exist to stop a plan being built on
a measurement that is quietly invalid — an empty knee curve, a percentile that
contradicts the mean, a trough above the peak. Every one of those would produce a
plausible-looking ``C_target`` rather than an error, which is the failure mode
this whole change is built to prevent.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from tts_bench.types import (
    Finding,
    Measured,
    Origin,
    Provenance,
    ScalingPlan,
    Scenario,
    Verdict,
)
from tts_inference.types import TTSModelName

CURVE = {50: 0.6, 150: 0.9, 300: 1.2, 500: 1.8}


def _measured(**overrides) -> Measured:
    kwargs = {
        "model_name": TTSModelName.KOKORO_82M,
        "endpoint": "speech-kokoro-82m",
        "instance_type": "ml.g5.xlarge",
        "c_max_curve": dict(CURVE),
        "s_mean_s": 0.42,
        "s_p95_s": 0.61,
        "t_total_s": 180.0,
        "frozen": True,
        "instance_counts_observed": (1, 1, 1),
    }
    kwargs.update(overrides)
    return Measured(**kwargs)


def _scenario(**overrides) -> Scenario:
    kwargs = {"peak_rps": 450.0, "trough_rps": 30.0}
    kwargs.update(overrides)
    return Scenario(**kwargs)


def _plan(**overrides) -> ScalingPlan:
    kwargs = {
        "model_name": TTSModelName.KOKORO_82M,
        "endpoint": "speech-kokoro-82m",
        "instance_type": "ml.g5.xlarge",
        "c_max": 1.2,
        "c_target": 0.525,
        "binding_constraint": "surge_headroom",
        "min_instances": 24,
        "max_instances": 360,
        "peak_instances": 360,
        "trough_instances": 24,
        "queue_max_depth": 5,
        "scale_out_cooldown_s": 30,
        "scale_in_cooldown_s": 600,
        "utilization_at_target": 0.4375,
        "w_absorbed_s": 2.0,
        "headroom_lag_s": 178.0,
        "peak_cost_per_hour": 506.88,
        "peak_cost_per_m_chars": 4.2,
        "relative_fleet_cost_vs_k1": 2.0,
        "measured": _measured(),
        "scenario": _scenario(),
    }
    kwargs.update(overrides)
    return ScalingPlan(**kwargs)


class TestProvenance:
    def test_measured_origin_is_flagged(self) -> None:
        assert Provenance(origin=Origin.MEASURED).is_measured

    def test_assumption_is_not_measured(self) -> None:
        # The distinction the report header rests on: k is supplied, not observed.
        assert not Provenance(origin=Origin.ASSUMPTION).is_measured

    def test_derived_is_not_measured(self) -> None:
        # Derived values inherit no authority from their measured inputs.
        assert not Provenance(origin=Origin.DERIVED).is_measured


class TestMeasured:
    def test_round_trips_through_json(self) -> None:
        # cmax writes an artifact; plan reads it back. Integer dict keys are the
        # risk here — JSON stringifies them, so a lossy round trip would make
        # every c_max_for lookup miss.
        restored = Measured.model_validate_json(_measured().model_dump_json())
        assert restored.c_max_curve == CURVE
        assert all(isinstance(k, int) for k in restored.c_max_curve)

    def test_is_immutable(self) -> None:
        with pytest.raises(ValidationError):
            _measured().s_mean_s = 99.0

    def test_empty_curve_is_rejected(self) -> None:
        # An empty curve means the run found no step meeting any budget. Nothing
        # downstream can plan from that, so it must not deserialize.
        with pytest.raises(ValidationError, match="c_max_curve must not be empty"):
            _measured(c_max_curve={})

    def test_non_positive_budget_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="budget must be positive"):
            _measured(c_max_curve={0: 1.0})

    def test_non_positive_concurrency_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="concurrency at budget 300 must be positive"):
            _measured(c_max_curve={300: 0.0})

    def test_p95_below_mean_is_rejected(self) -> None:
        # Percentiles that disagree with the mean mean the two came from
        # different sample sets — a units or windowing bug upstream.
        with pytest.raises(ValidationError, match="percentiles disagree"):
            _measured(s_mean_s=0.9, s_p95_s=0.4)

    def test_equal_mean_and_p95_is_allowed(self) -> None:
        # Legitimate for a fake or single-sample run; not evidence of a bug.
        assert _measured(s_mean_s=0.5, s_p95_s=0.5).s_p95_s == 0.5

    def test_zero_service_time_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _measured(s_mean_s=0.0)

    def test_zero_t_total_is_rejected(self) -> None:
        # T_total = 0 would make every feasibility check pass trivially.
        with pytest.raises(ValidationError):
            _measured(t_total_s=0.0)


class TestMeasuredTrustworthy:
    def test_frozen_and_stable_capacity_is_trustworthy(self) -> None:
        assert _measured(frozen=True, instance_counts_observed=(1, 1)).trustworthy

    def test_unfrozen_is_not_trustworthy(self) -> None:
        # Without the freeze, the fleet may have grown and C_max is really
        # N x C_max with no marker saying so. A stable observed count does not
        # redeem it: the monitor samples at 1Hz and can miss a change entirely.
        assert not _measured(frozen=False, instance_counts_observed=(1,)).trustworthy

    def test_capacity_change_defeats_the_freeze(self) -> None:
        assert not _measured(frozen=True, instance_counts_observed=(1, 2)).trustworthy

    def test_no_observations_is_still_trustworthy_if_frozen(self) -> None:
        # Nothing observed is not the same as a change observed; the freeze plus
        # cmax's own pre-flight check is what carries the guarantee here.
        assert _measured(frozen=True, instance_counts_observed=()).trustworthy

    def test_repeated_same_count_is_not_a_change(self) -> None:
        assert _measured(frozen=True, instance_counts_observed=(2, 2, 2)).trustworthy


class TestCMaxFor:
    def test_exact_budget_hit(self) -> None:
        assert _measured().c_max_for(300) == 1.2

    def test_falls_back_to_nearest_budget_below(self) -> None:
        # 250ms was not measured. Answering with the 150ms knee is conservative;
        # answering with the 300ms knee would claim a knee we never observed.
        assert _measured().c_max_for(250) == 0.9

    def test_budget_above_the_curve_uses_the_highest_measured(self) -> None:
        assert _measured().c_max_for(5_000) == 1.8

    def test_budget_below_every_measurement_raises(self) -> None:
        # Extrapolating downward would invent a knee. A 10ms SLO against a
        # 50ms-minimum run is a re-run, not an interpolation.
        with pytest.raises(ValueError, match="no measured budget at or below 10ms"):
            _measured().c_max_for(10)

    def test_error_names_the_measured_budgets(self) -> None:
        with pytest.raises(ValueError, match=r"measured: \[50, 150, 300, 500\]"):
            _measured().c_max_for(10)


class TestScenario:
    def test_rps_only_is_valid(self) -> None:
        assert Scenario(peak_rps=450.0).peak_rps == 450.0

    def test_streams_only_is_valid(self) -> None:
        # The honest input for bidirectional streaming, where one session is not
        # one request and the lambda x S conversion does not apply.
        assert Scenario(peak_streams=120.0).peak_streams == 120.0

    def test_no_load_at_all_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="supply either peak_rps or peak_streams"):
            Scenario()

    def test_trough_above_peak_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="trough_rps .* > peak_rps"):
            Scenario(peak_rps=100.0, trough_rps=200.0)

    def test_trough_streams_above_peak_streams_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="trough_streams .* > peak_streams"):
            Scenario(peak_streams=10.0, trough_streams=20.0)

    def test_trough_equal_to_peak_is_flat_not_invalid(self) -> None:
        assert Scenario(peak_rps=100.0, trough_rps=100.0).trough_rps == 100.0

    def test_growth_below_one_is_rejected(self) -> None:
        # k < 1 is shrinking traffic; there is no surge to reserve for and the
        # C_target formula would hand back more than C_max.
        with pytest.raises(ValidationError):
            Scenario(peak_rps=10.0, growth_factor_k=0.5)

    def test_flat_growth_is_allowed(self) -> None:
        assert Scenario(peak_rps=10.0, growth_factor_k=1.0).growth_factor_k == 1.0

    def test_derate_above_one_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Scenario(peak_rps=10.0, derate=1.5)

    def test_zero_added_wait_is_allowed(self) -> None:
        # A no-queue configuration is a legitimate scenario to price.
        assert Scenario(peak_rps=10.0, max_added_wait_s=0.0).max_added_wait_s == 0.0

    def test_defaults_are_tagged_as_assumptions(self) -> None:
        scenario = _scenario()
        assert scenario.provenance.origin is Origin.ASSUMPTION
        assert not scenario.provenance.is_measured

    def test_is_immutable(self) -> None:
        with pytest.raises(ValidationError):
            _scenario().growth_factor_k = 5.0


class TestScalingPlan:
    def test_no_findings_means_feasible(self) -> None:
        assert not _plan().infeasible

    def test_infeasible_finding_is_surfaced(self) -> None:
        plan = _plan(
            findings=[
                Finding(
                    name="queue_covers_surge",
                    verdict=Verdict.INFEASIBLE,
                    detail="W_max 2.0s < (k-1) x T_total 180.0s",
                )
            ]
        )
        assert plan.infeasible

    def test_warnings_exclude_other_verdicts(self) -> None:
        plan = _plan(
            findings=[
                Finding(name="ok_one", verdict=Verdict.OK, detail="fine"),
                Finding(name="warn_one", verdict=Verdict.WARN, detail="fragile"),
                Finding(name="suppressed_one", verdict=Verdict.SUPPRESSED, detail="no history"),
            ]
        )
        assert [f.name for f in plan.warnings] == ["warn_one"]

    def test_suppressed_is_not_infeasible(self) -> None:
        # SUPPRESSED means unevaluated for want of data. It must not block a
        # plan, and it must not read as a pass either.
        plan = _plan(
            findings=[
                Finding(
                    name="sample_count_guard",
                    verdict=Verdict.SUPPRESSED,
                    detail="no production history",
                )
            ]
        )
        assert not plan.infeasible
        assert plan.warnings == []

    def test_fractional_c_target_survives(self) -> None:
        # The reason scaling_target_value must become a float in config.py:
        # Kokoro's C_max of 1 at k=2 gives 0.44, which an int silently floors to
        # 0 and rounds into a policy that never scales.
        assert _plan(c_target=0.4375).c_target == 0.4375

    def test_is_immutable(self) -> None:
        with pytest.raises(ValidationError):
            _plan().c_target = 9.0

    def test_carries_its_inputs_for_audit(self) -> None:
        # A plan detached from its measurement cannot be checked later.
        plan = _plan()
        assert plan.measured.frozen
        assert plan.scenario.provenance.origin is Origin.ASSUMPTION

    def test_round_trips_through_json(self) -> None:
        plan = _plan(findings=[Finding(name="k_high", verdict=Verdict.WARN, detail="k=3")])
        restored = ScalingPlan.model_validate_json(plan.model_dump_json())
        assert restored.warnings[0].name == "k_high"
        assert restored.measured.c_max_curve == CURVE

    def test_min_instances_below_one_is_rejected(self) -> None:
        # Zero would be a scale-to-zero plan, which realtime endpoints cannot do.
        with pytest.raises(ValidationError):
            _plan(min_instances=0)
