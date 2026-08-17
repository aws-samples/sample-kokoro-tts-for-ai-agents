# Copyright (c) 2026 Amazon.com
# This file is licensed under the MIT License.
# See the LICENSE file in the project root for full license information.

"""Tests for the capacity-planning models in tts_bench.types.

These models are the contract between the measurement phases (``qmax``, ``ttotal``) and
the planner. Their validators exist to stop a plan being built on a measurement that is
quietly invalid — a percentile that contradicts the mean, a trough above the peak, a
CloudWatch conversion factor that is not positive. Every one of those would produce a
plausible-looking threshold rather than an error, which is the failure mode this whole
change is built to prevent.

Two themes recur, and both are scars:

**One SLO, one field.** ``Scenario`` carries ``ttfab_slo_ms`` and nothing else about
latency. It used to carry ``W_max`` and a second ``ttfab_budget_ms`` alongside, and two
independent latency fields is how this model came to declare a 20s queue allowance under a
300ms budget. The tests below assert the absence.

**Units cross a boundary.** What the client measures and what the deployed alarm reads are
different statistics, so ``c_scale_max`` and ``c_scale_max_in_cw_units`` are separate
fields and the ratio between them is measured per rung, never assumed.
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from shared.capacity import SAGEMAKER_INVOCATION_CEILING_S
from tts_bench.types import (
    Finding,
    Measured,
    Origin,
    Provenance,
    QMaxReport,
    ScalingPlan,
    Scenario,
    StepSummary,
    Verdict,
)
from tts_inference.types import TTSModelName

#: The ladder a Q_max run produces: concurrency -> p95 first byte. Roughly M/M/1 on
#: kokoro's measured numbers (34ms RTT + ~57ms service), so c=10 is about twice c=5 —
#: which is the relationship ttotal's halving test depends on.
LADDER = {1: 92.0, 5: 379.0, 10: 667.0, 20: 1243.0, 50: 2910.0}


def _measured(**overrides) -> Measured:
    kwargs = {
        "model_name": TTSModelName.KOKORO_82M,
        "endpoint": "speech-kokoro-82m",
        "instance_type": "ml.g5.xlarge",
        "q_max": 50,
        "slo_ms": 3000,
        "ttfab_p95_at_c1_ms": 92.0,
        "s_mean_s": 0.42,
        "s_p95_s": 0.61,
        "t_total_s": 180.0,
        "frozen": True,
        "instance_counts_observed": (1, 1, 1),
        "ladder_p95_ms": dict(LADDER),
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
        "q_max": 50,
        "c_scale_max": 37.5,
        "c_scale_min": 25.0,
        "c_scale_max_in_cw_units": 50.6,
        "cw_units_ratio": 1.35,
        "min_safe_instances": 3,
        "w_max_s": 2.39,
        "ceiling_s": SAGEMAKER_INVOCATION_CEILING_S,
        "min_instances": 24,
        "max_instances": 360,
        "peak_instances": 360,
        "trough_instances": 24,
        "queue_max_depth": 50,
        "scale_out_cooldown_s": 30,
        "scale_in_cooldown_s": 600,
        "utilization_at_c_scale_max": 0.974,
        "shed_probability_at_c_scale_max": 0.88,
        "peak_cost_per_hour": 506.88,
        "peak_cost_per_m_chars": 4.2,
        "measured": _measured(),
        "scenario": _scenario(),
    }
    kwargs.update(overrides)
    return ScalingPlan(**kwargs)


def _step(**overrides) -> StepSummary:
    kwargs = {
        "run_index": 0,
        "step_index": 0,
        "concurrency": 1,
        "achieved_rps": 10.9,
        "completed": 1090,
        "ok": 1090,
        "ttfab_p95_ms": 92.0,
        "concurrency_mean": 1.0,
        "meets_slo": True,
        "saturated": False,
        "settled": True,
        "usable": True,
    }
    kwargs.update(overrides)
    return StepSummary(**kwargs)


def _report(**overrides) -> QMaxReport:
    kwargs = {
        "model_name": TTSModelName.KOKORO_82M,
        "endpoint": "speech-kokoro-82m",
        "instance_type": "ml.g5.xlarge",
        "run_id": "qmax-20260803-000000",
        "slo_ms": 3000,
        "q_max": 50,
        "q_max_bracketed": True,
        "ttfab_p95_at_q_max_ms": 2910.0,
        "s_mean_s": 0.092,
        "s_p95_s": 0.11,
        "frozen": True,
        "hold_s": 120.0,
        "measure_window_s": 90.0,
        "steps": [
            _step(step_index=i, concurrency=c, ttfab_p95_ms=p95, concurrency_mean=float(c))
            for i, (c, p95) in enumerate(LADDER.items())
        ],
    }
    kwargs.update(overrides)
    return QMaxReport(**kwargs)


class TestProvenance:
    def test_measured_origin_is_flagged(self) -> None:
        assert Provenance(origin=Origin.MEASURED).is_measured

    def test_assumption_is_not_measured(self) -> None:
        # The distinction the report header rests on: the surge ratio is supplied,
        # not observed.
        assert not Provenance(origin=Origin.ASSUMPTION).is_measured

    def test_derived_is_not_measured(self) -> None:
        # Derived values inherit no authority from their measured inputs.
        assert not Provenance(origin=Origin.DERIVED).is_measured


class TestMeasured:
    def test_round_trips_through_json(self) -> None:
        # qmax writes an artifact; plan reads it back. Integer dict keys are the risk
        # here — JSON stringifies them, so a lossy round trip would make every ladder
        # lookup miss, and ttotal's recovery test is an exact-rung lookup.
        restored = Measured.model_validate_json(_measured().model_dump_json())
        assert restored.ladder_p95_ms == LADDER
        assert all(isinstance(k, int) for k in restored.ladder_p95_ms)

    def test_is_immutable(self) -> None:
        with pytest.raises(ValidationError):
            _measured().s_mean_s = 99.0

    def test_q_max_must_be_positive(self) -> None:
        # Zero would mean no concurrency met the SLO, which is not a capacity number
        # to plan from — both thresholds are fractions of it, so they would both be 0.
        with pytest.raises(ValidationError):
            _measured(q_max=0)

    def test_the_slo_it_was_measured_against_is_carried(self) -> None:
        # Q_max is *defined by* the SLO, so the artifact has to say which one. This is
        # what lets `plan` refuse a scenario asking for a different promise instead of
        # silently re-reading one ladder against another line.
        assert _measured().slo_ms == 3000

    def test_a_non_positive_slo_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _measured(slo_ms=0)

    def test_p95_below_mean_is_rejected(self) -> None:
        # Percentiles that disagree with the mean mean the two came from different
        # sample sets — a units or windowing bug upstream.
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

    def test_an_unbracketed_q_max_is_recorded_as_such(self) -> None:
        # A ladder that ran out while still passing gives a lower bound, and the
        # thresholds derived from it fire earlier than necessary. Costs money, not SLO
        # — but only if the plan can tell, hence the field.
        assert _measured(q_max_bracketed=False).q_max_bracketed is False

    def test_an_unchecked_queue_bound_is_not_a_pass(self) -> None:
        # None, not False: "nobody looked" and "we looked and it was bounded" call for
        # different fixes, and the planner reports them differently.
        assert _measured().unbounded_queue is None


class TestMeasuredLadder:
    """The ladder's other rungs are not scaffolding.

    Two consumers read them and neither can use the 3s SLO: the CDK alarm needs service
    time on an unqueued instance, and ``ttotal`` needs a *pair* of rungs to watch one p95
    halve into the other.
    """

    def test_rungs_survive_the_json_round_trip_as_ints(self) -> None:
        restored = Measured.model_validate_json(_measured().model_dump_json())
        assert restored.ladder_p95_ms[10] == 667.0

    def test_a_non_positive_rung_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="ladder rung must be positive"):
            _measured(ladder_p95_ms={0: 92.0})

    def test_a_negative_p95_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="p95 at rung 5 must be non-negative"):
            _measured(ladder_p95_ms={5: -1.0})

    def test_an_empty_ladder_is_allowed(self) -> None:
        # An artifact predating the field, or a run whose rungs were all unusable. The
        # consumers that need a rung refuse by name; this model does not have to know
        # which rungs they will ask for.
        assert _measured(ladder_p95_ms={}).ladder_p95_ms == {}


class TestMeasuredCwUnits:
    """The conversion between client occupancy and the statistic the alarm reads."""

    def test_ratios_round_trip_as_ints(self) -> None:
        restored = Measured.model_validate_json(
            _measured(cw_units_ratio_by_rung={5: 5.1, 50: 1.35}).model_dump_json()
        )
        assert restored.cw_units_ratio_by_rung == {5: 5.1, 50: 1.35}

    def test_a_non_positive_ratio_is_rejected(self) -> None:
        # This is the 0.713 defect in model form. A non-positive conversion deploys a
        # threshold that no arrival rate satisfies, and target tracking then reads one
        # request in flight as a demand for ten instances.
        with pytest.raises(ValidationError, match="would deploy a threshold no traffic satisfies"):
            _measured(cw_units_ratio_by_rung={5: 0.0})

    def test_a_non_positive_rung_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="ratio rung must be positive"):
            _measured(cw_units_ratio_by_rung={-1: 1.35})

    def test_empty_means_the_ladder_ran_without_cloudwatch(self) -> None:
        # Not 1:1. High-resolution datapoints retain 3 hours, so this cannot be
        # backfilled — the plan has to report the conversion as unavailable.
        assert _measured().cw_units_ratio_by_rung == {}


class TestMeasuredTransport:
    """The transport has to reach the planner, not stop at the artifact.

    A ``Q_max`` measured on response-stream does not describe capacity for bidi traffic —
    the containers hold their inference lock differently per protocol — so a plan that
    configures a real fleet must carry the protocol its capacity number came from.
    """

    def test_defaults_to_response_stream(self) -> None:
        # Artifacts written before the bidi transport existed have no transport field,
        # and they all came from response-stream. The default is what keeps them
        # readable rather than un-loadable.
        assert _measured().transport == "response-stream"

    def test_records_the_transport(self) -> None:
        assert _measured(transport="bidi").transport == "bidi"

    def test_round_trips_through_json(self) -> None:
        restored = Measured.model_validate_json(_measured(transport="bidi").model_dump_json())
        assert restored.transport == "bidi"

    def test_an_older_artifact_without_the_field_still_loads(self) -> None:
        # Explicitly: the field was added mid-project, and a run recorded before it
        # must not become unreadable.
        payload = _measured().model_dump()
        payload.pop("transport")
        assert Measured.model_validate(payload).transport == "response-stream"


class TestMeasuredTrustworthy:
    def test_frozen_and_stable_capacity_is_trustworthy(self) -> None:
        assert _measured(frozen=True, instance_counts_observed=(1, 1)).trustworthy

    def test_unfrozen_is_not_trustworthy(self) -> None:
        # Without the freeze, the fleet may have grown and Q_max is really N x Q_max
        # with no marker saying so — and it reads as *better* latency, not as an error.
        # A stable observed count does not redeem it: the monitor samples at 1Hz and
        # can miss a change entirely.
        assert not _measured(frozen=False, instance_counts_observed=(1,)).trustworthy

    def test_capacity_change_defeats_the_freeze(self) -> None:
        assert not _measured(frozen=True, instance_counts_observed=(1, 2)).trustworthy

    def test_no_observations_is_still_trustworthy_if_frozen(self) -> None:
        # Nothing observed is not the same as a change observed; the freeze plus
        # qmax's own pre-flight check is what carries the guarantee here.
        assert _measured(frozen=True, instance_counts_observed=()).trustworthy

    def test_repeated_same_count_is_not_a_change(self) -> None:
        assert _measured(frozen=True, instance_counts_observed=(2, 2, 2)).trustworthy


class TestQMaxReportLadder:
    def test_ttfab_p95_at_is_an_exact_rung(self) -> None:
        assert _report().ttfab_p95_at(10) == 667.0

    def test_an_unmeasured_rung_is_none_not_the_nearest(self) -> None:
        # Deliberately not a nearest-rung fallback: ttotal compares a probe held at N
        # against this table, and answering c=15 with the c=20 rung would silently
        # compare against a concurrency the probe never ran at. Callers must ask for a
        # rung that exists and refuse when it does not.
        assert _report().ttfab_p95_at(15) is None

    def test_unusable_rungs_are_excluded(self) -> None:
        report = _report(
            steps=[
                _step(step_index=0, concurrency=5, ttfab_p95_ms=379.0),
                _step(step_index=1, concurrency=10, ttfab_p95_ms=41.0, usable=False),
            ]
        )
        # 41ms at c=10 is what a rung of instant rejections looks like. Reading it
        # would tell ttotal to expect recovery at a latency no served request hits.
        assert report.ttfab_p95_at(10) is None

    def test_the_median_across_runs_is_used(self) -> None:
        report = _report(
            steps=[
                _step(run_index=0, concurrency=5, ttfab_p95_ms=300.0),
                _step(run_index=1, concurrency=5, ttfab_p95_ms=400.0),
                _step(run_index=2, concurrency=5, ttfab_p95_ms=500.0),
            ]
        )
        # With --runs 2 one noisy pass should not move the threshold that decides when
        # a scale-out is declared complete.
        assert report.ttfab_p95_at(5) == 400.0

    def test_rungs_lists_what_was_offered(self) -> None:
        # Including rungs that failed the SLO: "is c=10 on this ladder" is the question
        # a caller needs answered, and truncation is reported by ladder_truncated_at.
        assert _report().rungs == [1, 5, 10, 20, 50]


class TestTtfabP95AtC1Ms:
    """The CDK alarm's threshold. A ``computed_field``, not a bare property.

    ``speech_infra.measurements`` reads this off the artifact JSON at synth time and
    cannot import this module — ``tts-bench`` depends on ``speech-infra``, not the other
    way round — so the whole seam rests on the key being *in the serialized document*.
    A bare ``@property`` would be silently dropped by ``model_dump_json`` and the alarm
    would vanish with nothing failing.
    """

    def test_reads_the_n1_rung(self) -> None:
        assert _report().ttfab_p95_at_c1_ms == 92.0

    def test_appears_in_the_serialized_artifact(self) -> None:
        # The assertion the CDK stack's behaviour depends on. Keep it as a raw-JSON
        # check rather than an attribute read: what matters is the key on disk.
        payload = json.loads(_report().model_dump_json())
        assert payload["ttfab_p95_at_c1_ms"] == 92.0
        assert payload["model_name"] == "kokoro-82m"

    def test_is_none_when_the_ladder_had_no_n1_rung(self) -> None:
        # An alarm threshold has to come from somewhere real, so the absence is
        # reported rather than substituted for. speech-infra omits the alarm.
        report = _report(steps=[_step(concurrency=5, ttfab_p95_ms=379.0)])
        assert report.ttfab_p95_at_c1_ms is None

    def test_serializes_as_null_when_absent(self) -> None:
        report = _report(steps=[_step(concurrency=5, ttfab_p95_ms=379.0)])
        payload = json.loads(report.model_dump_json())
        assert payload["ttfab_p95_at_c1_ms"] is None

    def test_is_ignored_on_the_way_in(self) -> None:
        # Derived on the way out and ignored on the way in, so it cannot disagree with
        # the ladder it summarizes. A hand-edited artifact claiming 5000ms still
        # reports what the c=1 rung measured.
        payload = _report().model_dump()
        payload["ttfab_p95_at_c1_ms"] = 5000.0
        assert QMaxReport.model_validate(payload).ttfab_p95_at_c1_ms == 92.0

    def test_it_reaches_measured_through_to_measured(self) -> None:
        measured = _report().to_measured(t_total_s=180.0)
        assert measured.ttfab_p95_at_c1_ms == 92.0


class TestQMaxReportCwUnitsRatio:
    def test_ratio_is_server_peak_over_client_mean(self) -> None:
        report = _report(
            steps=[
                _step(
                    concurrency=5,
                    concurrency_mean=2.932,
                    server_concurrency_peak=15.0,
                ),
            ]
        )
        assert report.cw_units_ratio_by_rung[5] == pytest.approx(15.0 / 2.932)

    def test_a_rung_without_server_metrics_drops_out(self) -> None:
        # Not zero and not 1:1 — the rung simply has no conversion, and the planner
        # interpolates from the rungs that do.
        assert _report().cw_units_ratio_by_rung == {}

    def test_the_ratio_is_not_a_constant(self) -> None:
        # The measured shape, and the reason this is a table rather than a fitted
        # number: a lightly loaded endpoint's 10s peak is many multiples of its
        # average, while a saturated one's is barely above it.
        report = _report(
            steps=[
                _step(concurrency=5, concurrency_mean=0.712, server_concurrency_peak=7.0),
                _step(
                    step_index=1,
                    concurrency=50,
                    concurrency_mean=11.15,
                    server_concurrency_peak=15.0,
                ),
            ]
        )
        assert report.cw_units_ratio_by_rung[5] > 9.0
        assert report.cw_units_ratio_by_rung[50] < 1.5


class TestQMaxReportInvariants:
    def test_q_max_must_be_a_rung_some_run_measured(self) -> None:
        # The median of two rungs is a concurrency no run tested. The minimum is both
        # a real rung and the conservative one, so that is what the cross-run answer
        # has to be.
        with pytest.raises(ValidationError, match="must be the minimum of q_max_per_run"):
            _report(q_max=45, q_max_per_run=(40, 50))

    def test_the_minimum_is_accepted(self) -> None:
        assert _report(q_max=40, q_max_per_run=(40, 50)).q_max == 40

    def test_a_non_positive_per_run_answer_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="per-run q_max must be positive"):
            _report(q_max=50, q_max_per_run=(50, 0))

    def test_spread_is_relative_to_the_minimum(self) -> None:
        # Relative to the value the plan actually uses, so it reads directly as "how
        # much capacity the other runs claimed on top of what we planned for".
        report = _report(q_max=40, q_max_per_run=(40, 50))
        assert report.q_max_spread == pytest.approx(0.25)
        assert report.runs_contributing == 2

    def test_a_single_run_has_nothing_to_disagree_with(self) -> None:
        # 0.0 here is not agreement, which is why runs_contributing exists beside it.
        report = _report(q_max=50, q_max_per_run=(50,))
        assert report.q_max_spread == 0.0
        assert report.runs_contributing == 1

    def test_trustworthiness_does_not_absorb_the_queue_check(self) -> None:
        # An unchecked queue bound is a separate warning with a separate fix, and
        # folding it in would make one re-run look like the other's.
        assert _report(frozen=True, unbounded_queue=None).trustworthy


class TestToMeasured:
    def test_carries_the_preconditions_through(self) -> None:
        # A ladder run without a precondition must stay identifiable as such after the
        # join, or the plan reports an untrustworthy number as a measurement.
        report = _report(frozen=False, unbounded_queue=False, instance_counts_observed=(1, 2))
        measured = report.to_measured(t_total_s=180.0)
        assert not measured.frozen
        assert measured.unbounded_queue is False
        assert measured.instance_counts_observed == (1, 2)
        assert not measured.trustworthy

    def test_a_measured_lag_is_the_default(self) -> None:
        assert _report().to_measured(t_total_s=180.0).t_total_measured

    def test_an_assumed_lag_is_flagged(self) -> None:
        # provenance.origin describes the object as a whole and stays MEASURED — Q_max
        # and S really were measured — so it cannot answer this question. Without the
        # separate field the planner reports a command-line argument as "used exactly
        # as measured", the one claim the provenance machinery exists to prevent.
        measured = _report().to_measured(
            t_total_s=231.0,
            t_total_provenance=Provenance(origin=Origin.ASSUMPTION, note="--assume-t-total"),
        )
        assert not measured.t_total_measured
        assert measured.provenance.origin is Origin.MEASURED

    def test_both_notes_survive(self) -> None:
        # The caller's note carries caveats that cannot be reconstructed from `origin`
        # — that recovery was inferred so the lag is a floor, say — and a Measured read
        # back from a plan artifact is all a later reader has.
        report = _report(
            provenance=Provenance(origin=Origin.MEASURED, note="ladder frozen at 1 instance")
        )
        measured = report.to_measured(
            t_total_s=231.0,
            t_total_provenance=Provenance(origin=Origin.MEASURED, note="container half only"),
        )
        assert measured.provenance.note is not None
        assert "ladder frozen at 1 instance" in measured.provenance.note
        assert "container half only" in measured.provenance.note

    def test_the_slo_travels_with_the_number_it_defines(self) -> None:
        assert _report(slo_ms=3000).to_measured(t_total_s=180.0).slo_ms == 3000

    def test_runs_contributing_is_never_zero(self) -> None:
        # It is the denominator for the spread and must be a count of passes; a report
        # with no per-run detail still had one run.
        assert _report().to_measured(t_total_s=180.0).runs_contributing == 1


class TestScenario:
    def test_rps_only_is_valid(self) -> None:
        assert Scenario(peak_rps=450.0).peak_rps == 450.0

    def test_streams_only_is_valid(self) -> None:
        # The honest input for bidirectional streaming, where one session is not one
        # request and the lambda x S conversion does not apply.
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

    def test_a_surge_ratio_below_one_is_rejected(self) -> None:
        # Below 1 is shrinking traffic: there is no surge to reserve for, and
        # C_scale_max = (1-h) x Q_max would come out *above* Q_max.
        with pytest.raises(ValidationError):
            Scenario(peak_rps=10.0, max_scaling_per_t_total=0.5)

    def test_a_flat_surge_ratio_is_allowed(self) -> None:
        # h = 0: both thresholds collapse onto Q_max. A legitimate configuration for
        # traffic that genuinely does not surge, and the arithmetic still holds.
        assert Scenario(peak_rps=10.0, max_scaling_per_t_total=1.0).max_scaling_per_t_total == 1.0

    def test_a_surge_ratio_at_or_above_one_and_a_half_is_rejected(self) -> None:
        # C_scale_min = (1-2h) x Q_max reaches zero at h=0.5, and a non-positive
        # scale-in threshold deploys as "never scale in" — silently, since the alarm
        # is well-formed and simply never fires.
        with pytest.raises(ValidationError):
            Scenario(peak_rps=10.0, max_scaling_per_t_total=1.5)

    def test_the_slo_is_the_only_latency_input(self) -> None:
        # W_max used to sit here as an independent field beside a second ttfab budget,
        # and the two could disagree without anything noticing -- which is how kokoro
        # shipped a 20s queue allowance under a 300ms budget, a request reaching first
        # byte at 20.2s while the config claimed 0.3s. Both are derived now.
        scenario = Scenario(peak_rps=10.0)
        assert scenario.ttfab_slo_ms == 3000
        assert not hasattr(scenario, "max_added_wait_s")
        assert not hasattr(scenario, "ttfab_budget_ms")
        assert not hasattr(scenario, "derate")

    def test_a_non_positive_slo_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Scenario(peak_rps=10.0, ttfab_slo_ms=0)

    def test_defaults_are_tagged_as_assumptions(self) -> None:
        scenario = _scenario()
        assert scenario.provenance.origin is Origin.ASSUMPTION
        assert not scenario.provenance.is_measured

    def test_is_immutable(self) -> None:
        with pytest.raises(ValidationError):
            _scenario().max_scaling_per_t_total = 1.4


class TestScalingPlan:
    def test_no_findings_means_feasible(self) -> None:
        assert not _plan().infeasible

    def test_infeasible_finding_is_surfaced(self) -> None:
        plan = _plan(
            findings=[
                Finding(
                    name="slo_budget",
                    verdict=Verdict.INFEASIBLE,
                    detail="S_p95 0.61s already exceeds the 0.5s SLO",
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
        # SUPPRESSED means unevaluated for want of data. It must not block a plan, and
        # it must not read as a pass either.
        plan = _plan(
            findings=[
                Finding(
                    name="surge_survival",
                    verdict=Verdict.SUPPRESSED,
                    detail="could not simulate",
                )
            ]
        )
        assert not plan.infeasible
        assert plan.warnings == []

    def test_the_thresholds_stay_fractional(self) -> None:
        # (1-h) x Q_max rarely lands on an integer, and this is why
        # scaling_target_value is a float in config.py: an int field truncates 37.5 to
        # 37 in the safe direction but 0.44 to 0, which deploys as "never scale".
        plan = _plan(c_scale_max=37.5, c_scale_min=25.0)
        assert plan.c_scale_max == 37.5
        assert plan.c_scale_min == 25.0

    def test_scale_in_may_be_zero(self) -> None:
        # (1-2h) x Q_max reaches zero at the top of the allowed surge range. Valid as
        # a number, and the scale_in_safety finding is what says it means "never".
        assert _plan(c_scale_min=0.0).c_scale_min == 0.0

    def test_is_immutable(self) -> None:
        with pytest.raises(ValidationError):
            _plan().c_scale_max = 9.0

    def test_carries_its_inputs_for_audit(self) -> None:
        # A plan detached from its measurement cannot be checked later.
        plan = _plan()
        assert plan.measured.frozen
        assert plan.scenario.provenance.origin is Origin.ASSUMPTION

    def test_round_trips_through_json(self) -> None:
        plan = _plan(
            findings=[Finding(name="surge_survival", verdict=Verdict.WARN, detail="P(shed) 0.88")]
        )
        restored = ScalingPlan.model_validate_json(plan.model_dump_json())
        assert restored.warnings[0].name == "surge_survival"
        assert restored.measured.ladder_p95_ms == LADDER
        assert restored.q_max == 50

    def test_min_instances_below_one_is_rejected(self) -> None:
        # Zero would be a scale-to-zero plan, which realtime endpoints cannot do.
        with pytest.raises(ValidationError):
            _plan(min_instances=0)


class TestTheJudgedCeilingIsCarried:
    """The plan records the invocation ceiling it was judged against, not the constant.

    ``scale_report`` prints that number as a comment on ``ttfab_slo_ms``, and the block is
    pasted into ``config.py`` verbatim. Re-reading ``SAGEMAKER_INVOCATION_CEILING_S`` there
    would name 60s on a run judged at some other value — a comment contradicting the
    ``invocation_ceiling`` finding above it, which is how a policy gets deployed against a
    limit nobody checked.
    """

    def test_the_ceiling_is_a_recorded_field_not_a_constant_lookup(self) -> None:
        assert _plan(ceiling_s=2.0).ceiling_s == 2.0

    def test_it_is_required_rather_than_defaulting_to_sixty(self) -> None:
        # A plan that failed to record its ceiling must not render as one judged at 60s:
        # the whole point of the field is that the constant cannot stand in for it.
        kwargs = {k: v for k, v in _plan().model_dump().items() if k != "ceiling_s"}
        with pytest.raises(ValidationError):
            ScalingPlan(**kwargs)

    def test_a_non_positive_ceiling_is_rejected(self) -> None:
        # shared.capacity refuses one too, so accepting it here would only let an
        # unjudgeable plan exist long enough to be rendered.
        with pytest.raises(ValidationError):
            _plan(ceiling_s=0.0)

    def test_it_survives_the_json_round_trip(self) -> None:
        # The artifact is what a later reader has, so the ceiling has to travel with the
        # plan rather than be re-derived by whatever renders it next.
        restored = ScalingPlan.model_validate_json(_plan(ceiling_s=2.0).model_dump_json())
        assert restored.ceiling_s == 2.0


class TestScalingPlanUnits:
    """The two occupancy numbers are separate fields because they are separate units."""

    def test_the_cw_threshold_is_recorded_beside_the_client_one(self) -> None:
        plan = _plan(c_scale_max=37.5, c_scale_max_in_cw_units=50.6, cw_units_ratio=1.35)
        assert plan.c_scale_max == 37.5
        assert plan.c_scale_max_in_cw_units == 50.6
        # Recorded so a reader can see the size of the correction rather than having to
        # trust it. Collapsing the two is what put 0.713 on the endpoint.
        assert plan.cw_units_ratio == 1.35

    def test_no_cloudwatch_means_no_converted_threshold(self) -> None:
        # None rather than a copy of the client figure. High-res datapoints retain 3
        # hours, so the conversion cannot be reconstructed later — the plan says
        # "unavailable" instead of assuming 1:1.
        plan = _plan(c_scale_max_in_cw_units=None, cw_units_ratio=None)
        assert plan.c_scale_max_in_cw_units is None

    def test_a_non_positive_ratio_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _plan(cw_units_ratio=0.0)


class TestScalingPlanSurgeArithmetic:
    """The two known limits of the simple threshold rule, in the fields that report them."""

    def test_utilization_is_recorded_because_it_is_not_obvious(self) -> None:
        # 0.75 x Q_max is not three quarters of the way to trouble: occupancy 37.5 on a
        # single-server queue is L/(1+L) = 97% utilized. The threshold alone does not
        # show that, which is why the number is on the plan.
        assert _plan().utilization_at_c_scale_max == 0.974

    def test_shed_probability_is_the_falsifiable_form(self) -> None:
        # "Is this threshold early enough" as a number rather than an opinion:
        # simulated P(queue reaches Q_max) while waiting out one T_total.
        assert _plan().shed_probability_at_c_scale_max == 0.88

    def test_shed_probability_is_a_probability(self) -> None:
        with pytest.raises(ValidationError):
            _plan(shed_probability_at_c_scale_max=1.5)

    def test_it_may_be_absent(self) -> None:
        assert _plan(shed_probability_at_c_scale_max=None).shed_probability_at_c_scale_max is None

    def test_min_safe_instances_may_be_none(self) -> None:
        # None means no fleet size is safe at this surge ratio — a real outcome, and
        # distinct from "3", so the scale_in_safety finding can say which.
        assert _plan(min_safe_instances=None).min_safe_instances is None

    def test_queue_depth_is_q_max(self) -> None:
        # Past Q_max a request cannot reach first byte inside the SLO, so admitting it
        # produces a late success instead of an honest rejection.
        assert _plan().queue_max_depth == _plan().q_max
