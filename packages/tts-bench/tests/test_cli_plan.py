"""Tests for the `plan` CLI command.

`plan` is the only command in this package that touches no AWS and sends no load: it
reads two artifacts and prints a configuration. So these tests are about the ways a
*pure* command can still mislead:

- **It refuses inputs that cannot be combined.** A `Q_max` ladder and a `T_total` lag
  from different configurations produce a plan for a fleet that exists nowhere, and no
  number in the output looks wrong. A ladder measured against one SLO read against
  another is the same class of error, since `Q_max` is *defined by* the SLO.
  `--allow-config-mismatch` is the only way past the first, and it must warn; there is
  no way past the second.
- **It refuses to guess a missing input.** No stated peak means no fleet size; no lag
  means no headroom sizing. Both are usage errors rather than defaults, because a
  defaulted peak would produce a plausible plan for load nobody expects.
- **What it prints distinguishes measured from chosen from derived.** Six variables, and
  the report is grouped by which is which. A stated `T_total` that reads as a measured
  one launders a command-line argument into an observation.
- **The threshold it prints is the one that deploys.** `C_scale_max` is a client
  occupancy; the alarm reads `ConcurrentRequestsPerModel`/`Maximum`. Emitting the
  unconverted figure is how `0.713` reached a live endpoint, so with no measured
  conversion the field is commented out rather than filled in.
- **Its exit status can gate a deploy.** An SLO that breaks the 60s invocation ceiling,
  or one the model's own tail already misses, exits non-zero — so the same command works
  in a pipeline as on a terminal.

Every test runs inside `CliRunner.isolated_filesystem()` — `--output` takes a relative
path, and the artifact fixtures are written next to it. `caplog` is not used: loguru
does not propagate to the stdlib logging tree, so an assertion against it would pass
whether or not anything was emitted.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import click
import pytest
from click.testing import CliRunner
from loguru import logger

from tts_bench.cli import main
from tts_bench.fixture import DeployedConfig

MODEL = "kokoro-82m"
ENDPOINT = "speech-kokoro-82m"

#: The configuration both fixtures record, so a default run pairs cleanly and only a
#: test that *asks* for a mismatch sees one.
DEPLOYED = DeployedConfig(
    instance_type="ml.g5.xlarge",
    image_digest="139b9068c5eb1f03c8312c17391dc35838e43e2417ae80d61e628e6ffb3d6a28",
    container_env={},
)
SLUG = DEPLOYED.slug

#: kokoro's measured service time, unrounded. ``S`` is conflated — 34ms client round
#: trip plus ~59ms of service — which is correct for a client-observed SLO.
S_MEAN_S = 0.10986375146305409
S_P95_S = 0.16457688123919073

#: The ladder the fixture writes. 1 is the alarm's rung, 5 and 10 are the pair `ttotal`
#: compares against, and 50 sits just inside the 3000ms SLO — so ``Q_max`` is 50 and the
#: derived thresholds are 37.5 / 25.0 at the default 1.25 surge ratio.
LADDER = {1: 92.0, 5: 379.0, 10: 667.0, 20: 1243.0, 50: 2910.0}
Q_MAX = 50

#: ``ConcurrentRequestsPerModel``/``Maximum`` per client mean in-flight, by rung. Falls
#: as load rises — 9.8x to 1.35x on one kokoro ladder — which is why the conversion is
#: read at the rung nearest the threshold rather than averaged.
RATIOS = {1: 9.8, 5: 5.1, 10: 3.0, 20: 2.0, 50: 1.35}


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


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


def _param(name: str):
    """The `plan` command's declared parameter, by its Python name."""
    for param in main.commands["plan"].params:
        if param.name == name:
            return param
    raise AssertionError(f"plan has no parameter {name}")


def _steps(*, with_cloudwatch: bool = True) -> list[dict[str, Any]]:
    """One ladder pass, as the step dicts a qmax artifact carries.

    ``server_concurrency_peak`` is what the units conversion is read from, so a step list
    without it is the shape of a ladder run without ``--cloudwatch`` — the case that must
    comment the threshold out rather than deploy the raw occupancy.
    """
    return [
        {
            "run_index": 0,
            "step_index": index,
            "concurrency": rung,
            "achieved_rps": 9.0,
            "completed": 100,
            "ok": 100,
            "chars": 2500,
            "ttfab_p95_ms": p95,
            "concurrency_mean": float(rung),
            "server_concurrency_peak": rung * RATIOS[rung] if with_cloudwatch else None,
            "meets_slo": True,
            "saturated": False,
            "settled": True,
            "usable": True,
        }
        for index, (rung, p95) in enumerate(LADDER.items())
    ]


def _write_qmax(**overrides: Any) -> str:
    """A `qmax` artifact shaped like a real ladder run, in the cwd.

    Written as a dict rather than through ``QMaxReport`` so a test can produce documents
    the model would reject — a pre-fingerprint artifact, a missing rung — which is
    exactly what `plan` has to survive reading. ``test_types.py`` owns the round-trip.
    """
    payload: dict[str, Any] = {
        "model_name": MODEL,
        "endpoint": ENDPOINT,
        "instance_type": "ml.g5.xlarge",
        "run_id": "qmax123",
        "slo_ms": 3000,
        "q_max": Q_MAX,
        "q_max_per_run": [Q_MAX, Q_MAX],
        "q_max_bracketed": True,
        "ttfab_p95_at_q_max_ms": LADDER[Q_MAX],
        "s_mean_s": S_MEAN_S,
        "s_p95_s": S_P95_S,
        "frozen": True,
        "unbounded_queue": True,
        "instance_counts_observed": [1],
        "transport": "bidi",
        "runs": 2,
        "hold_s": 120.0,
        "measure_window_s": 90.0,
        "deployed_config": DEPLOYED.to_dict(),
        "steps": _steps(),
    }
    payload.update(overrides)
    Path("qmax.json").write_text(json.dumps(payload))
    return "qmax.json"


def _write_ttotal(**overrides: Any) -> str:
    """A `ttotal` artifact with a provision stage and a bounded policy term, in the cwd."""
    payload: dict[str, Any] = {
        "model_name": MODEL,
        "endpoint": ENDPOINT,
        "run_id": "ttotal456",
        "trigger": "force-desired",
        "config_slug": SLUG,
        "t_total_s": 420.0,
        "policy_lag_bound_s": 60.0,
        "t_total_bounded": False,
        "durations": [
            {"from": "desired_set", "to": "instance_logging", "seconds": 180.0},
            {"from": "instance_logging", "to": "in_service", "seconds": 120.0},
            {"from": "in_service", "to": "traffic_recovered", "seconds": 120.0},
        ],
        "missing_stages": ["weights_fetched"],
    }
    payload.update(overrides)
    Path("ttotal.json").write_text(json.dumps(payload))
    return "ttotal.json"


def _run(runner: CliRunner, *args: str):
    """Invoke `plan` with both default artifacts written, in an isolated filesystem.

    No AWS is patched because none is reached: `plan` reads files and computes. That is
    the property worth having a test named after, so it is asserted below rather than
    only relied on here.
    """
    with runner.isolated_filesystem():
        qmax = _write_qmax()
        ttotal = _write_ttotal()
        return runner.invoke(
            main,
            ["plan", "--qmax", qmax, "--ttotal", ttotal, *args],
            catch_exceptions=False,
        )


def _run_with(runner: CliRunner, *args: str, qmax: dict[str, Any], ttotal: dict[str, Any] | None):
    """Invoke `plan` against overridden artifacts; ``ttotal=None`` writes no lag file."""
    with runner.isolated_filesystem():
        qmax_path = _write_qmax(**qmax)
        argv = ["plan", "--qmax", qmax_path]
        if ttotal is not None:
            argv += ["--ttotal", _write_ttotal(**ttotal)]
        return runner.invoke(main, [*argv, *args], catch_exceptions=False)


class TestRefusesToGuess:
    def test_a_missing_peak_is_a_usage_error(self, runner: CliRunner) -> None:
        # Not a default. A defaulted peak produces a plausible fleet size for load
        # nobody stated, which is the one output of this command nobody can sanity-check
        # by eye.
        result = _run(runner)
        assert result.exit_code == 2
        assert "one of --peak-rps or --peak-streams is required" in result.output

    def test_streams_alone_are_enough(self, runner: CliRunner) -> None:
        # The honest input for bidi, where one session is not one request.
        result = _run(runner, "--peak-streams", "40")
        assert result.exit_code == 0
        assert "40 concurrent streams" in result.output

    def test_a_missing_ttotal_is_a_usage_error(self, runner: CliRunner) -> None:
        result = _run_with(runner, "--peak-rps", "450", qmax={}, ttotal=None)
        assert result.exit_code == 2
        assert "--ttotal is required" in result.output
        assert "--assume-t-total" in result.output

    def test_a_stated_lag_stands_in_for_the_artifact(self, runner: CliRunner) -> None:
        # The case the plan anticipates: the second instance will not place in this
        # account, so there is no lag to read and one gets stated instead.
        result = _run_with(
            runner, "--peak-rps", "450", "--assume-t-total", "300", qmax={}, ttotal=None
        )
        assert result.exit_code == 0
        # The number in use, and that it was stated. "not measured" on its own sends the
        # reader hunting for the lag every number below was built from.
        assert "300s STATED, not measured" in result.output

    def test_a_stated_lag_is_never_labelled_measured(self, runner: CliRunner) -> None:
        # Two separate render paths said "measured" about a command-line argument: the
        # inputs header and the provision finding. Each is asserted.
        result = _run_with(
            runner, "--peak-rps", "450", "--assume-t-total", "300", qmax={}, ttotal=None
        )
        assert result.exit_code == 0
        assert "planned against" not in result.output
        assert "run `tts-bench ttotal` to get one worth planning on" in result.output
        # And the finding warns rather than reporting a stage split it never saw.
        assert "[WARN] provision_stage" in result.output
        assert "stated whole, not measured" in result.output

    def test_a_measured_lag_is_still_labelled_measured(self, runner: CliRunner) -> None:
        # The other half of the pair: the fix must not relabel a real measurement.
        result = _run(runner, "--peak-rps", "450")
        assert "planned against" in result.output
        assert "420s capacity request -> traffic served" in result.output
        assert "STATED" not in result.output

    def test_a_stated_lag_needs_no_fingerprint_to_pair(self, runner: CliRunner) -> None:
        # A stated number carries no fingerprint, so the pairing check must not run --
        # it would report a mismatch where there is nothing to mismatch.
        result = _run_with(
            runner, "--peak-rps", "450", "--assume-t-total", "300", qmax={}, ttotal=None
        )
        assert result.exit_code == 0
        assert "cannot pair" not in result.output

    def test_an_unreadable_qmax_artifact_says_which_file_and_which_command(
        self, runner: CliRunner
    ) -> None:
        with runner.isolated_filesystem():
            Path("junk.json").write_text("not json")
            _write_ttotal()
            result = runner.invoke(
                main,
                ["plan", "--qmax", "junk.json", "--ttotal", "ttotal.json", "--peak-rps", "450"],
                catch_exceptions=False,
            )
        assert result.exit_code == 1
        assert "junk.json" in result.output
        assert "tts-bench qmax" in result.output

    def test_a_missing_qmax_artifact_fails_at_parse_time(self, runner: CliRunner) -> None:
        # `exists=True` on the option, so this never reaches the planner.
        with runner.isolated_filesystem():
            result = runner.invoke(main, ["plan", "--qmax", "nope.json", "--peak-rps", "450"])
        assert result.exit_code == 2

    def test_an_unreadable_ttotal_artifact_says_which_file(self, runner: CliRunner) -> None:
        with runner.isolated_filesystem():
            qmax = _write_qmax()
            Path("junk.json").write_text("{[")
            result = runner.invoke(
                main,
                ["plan", "--qmax", qmax, "--ttotal", "junk.json", "--peak-rps", "450"],
                catch_exceptions=False,
            )
        assert result.exit_code == 1
        assert "junk.json" in result.output

    def test_a_trough_above_the_peak_is_rejected(self, runner: CliRunner) -> None:
        result = _run(runner, "--peak-rps", "2", "--trough-rps", "20")
        assert result.exit_code == 2
        assert "trough_rps" in result.output


class TestConfigMismatch:
    def test_differing_fingerprints_stop_the_run(self, runner: CliRunner) -> None:
        result = _run_with(
            runner, "--peak-rps", "450", qmax={}, ttotal={"config_slug": "g6xlarge-abcd1234"}
        )
        assert result.exit_code == 1
        assert "cannot pair" in result.output
        # Both named: which two configurations disagreed is the whole content of the
        # message, and an operator with several artifacts cannot act without it.
        assert SLUG in result.output
        assert "g6xlarge-abcd1234" in result.output
        # And what the mismatch means, not just that there is one.
        assert "the ladder sizes instances of one configuration" in result.output

    def test_a_pre_fingerprint_ladder_counts_as_a_mismatch(self, runner: CliRunner) -> None:
        # The old artifacts are exactly when this check matters, so an absent fingerprint
        # must not read as agreement. A qmax artifact still knows its instance type, so it
        # renders `nodigest` -- a mismatch against a real digest that says which half is
        # missing.
        result = _run_with(runner, "--peak-rps", "450", qmax={"deployed_config": {}}, ttotal={})
        assert result.exit_code == 1
        assert "cannot pair" in result.output
        assert "g5xlarge-nodigest" in result.output

    def test_a_ttotal_artifact_with_no_fingerprint_at_all_is_refused(
        self, runner: CliRunner
    ) -> None:
        # The other side has no instance type to fall back on, so it gets its own
        # message: there is not a second configuration to name.
        result = _run_with(runner, "--peak-rps", "450", qmax={}, ttotal={"config_slug": ""})
        assert result.exit_code == 1
        assert "no configuration fingerprint" in result.output
        assert "predates fingerprinting" in result.output

    def test_the_flag_lets_a_mismatch_through_with_a_warning(
        self, runner: CliRunner, logged: list[str]
    ) -> None:
        result = _run_with(
            runner,
            "--peak-rps",
            "450",
            "--allow-config-mismatch",
            qmax={},
            ttotal={"config_slug": "g6xlarge-abcd1234"},
        )
        assert result.exit_code == 0
        assert any("different configurations" in message for message in logged)

    def test_matching_fingerprints_need_no_flag(self, runner: CliRunner) -> None:
        result = _run(runner, "--peak-rps", "450")
        assert result.exit_code == 0
        assert SLUG in result.output


class TestTheSloComesFromTheLadder:
    """One SLO, and it is the one the ladder was judged against.

    `Q_max` is *defined by* the SLO — the highest rung whose p95 stayed inside it — so a
    ladder measured at 3000ms says nothing about a 1000ms promise. `--ttfab-slo-ms`
    exists to state the one the artifact already records, not to convert between them.
    """

    def test_the_slo_defaults_to_whatever_the_ladder_measured(self, runner: CliRunner) -> None:
        # Not to 3000: the artifact records the line Q_max was judged against, and
        # defaulting to a constant would refuse a good ladder measured at another one.
        assert _param("ttfab_slo_ms").default is None
        result = _run_with(runner, "--peak-rps", "450", qmax={"slo_ms": 2000}, ttotal={})
        assert result.exit_code == 0
        assert "bracketed against the 2000ms SLO" in result.output
        assert "SLO              2.0s to first byte, queue included" in result.output

    def test_a_disagreeing_slo_is_refused_naming_both(self, runner: CliRunner) -> None:
        result = _run(runner, "--peak-rps", "450", "--ttfab-slo-ms", "1000")
        assert result.exit_code == 1
        assert "measured against a 3000ms SLO but this scenario asks for 1000ms" in result.output
        # Either direction resolves it, so both fixes are offered.
        assert "--ttfab-slo-ms 3000" in result.output
        assert "tts-bench qmax --slo-ms 1000" in result.output

    def test_restating_the_measured_slo_is_accepted(self, runner: CliRunner) -> None:
        # The flag is not vestigial: passing it explicitly is how a pipeline asserts the
        # ladder it was handed is the one it expects.
        assert _run(runner, "--peak-rps", "450", "--ttfab-slo-ms", "3000").exit_code == 0

    def test_there_is_no_second_latency_input(self, runner: CliRunner) -> None:
        # `--ttfab-budget-ms` and `--max-added-wait` used to sit beside the SLO with
        # nothing relating them, which is how the config came to promise 300ms while
        # allowing a 20s queue. A rejected flag is a better outcome than an accepted one.
        result = runner.invoke(main, ["plan", "--help"], catch_exceptions=False)
        for retired in ("--ttfab-budget-ms", "--max-added-wait"):
            assert retired not in result.output

    def test_w_max_prints_as_derived_with_its_arithmetic(self, runner: CliRunner) -> None:
        # Shown as a subtraction rather than a number: a reader who takes W_max for an
        # input will go looking for the flag that sets it, and there isn't one.
        result = _run(runner, "--peak-rps", "450")
        assert "derived (none of these can be set independently):" in result.output
        assert "W_max            2.84s queueing budget = 3.0s SLO - 0.165s p95 service" in (
            result.output
        )

    def test_the_queue_bound_is_q_max_not_a_derivation_of_the_slo(self, runner: CliRunner) -> None:
        # It *is* the measured number, and the comment says so. Past it a request cannot
        # reach first byte in time, so it is refused rather than served late.
        result = _run(runner, "--peak-rps", "450")
        assert "queue_max_depth=50,  # = Q_max" in result.output
        assert "inside the 3.0s SLO" in result.output


class TestTheSixVariables:
    """That the report is grouped by measured / chosen / derived, and shows the workings.

    The grouping is the deliverable: a reader has to be able to tell which numbers came
    off an endpoint, which somebody picked, and which are arithmetic on the other four.
    Before this the path from measurement to policy ran through a comment block nobody
    could re-derive.
    """

    def test_the_two_measured_variables_are_labelled_measured(self, runner: CliRunner) -> None:
        result = _run(runner, "--peak-rps", "450")
        out = result.output
        assert "  measured:" in out
        assert "Q_max            50 concurrent per instance" in out
        assert "T_total          480s" in out

    def test_the_two_chosen_variables_are_labelled_inputs(self, runner: CliRunner) -> None:
        result = _run(runner, "--peak-rps", "450")
        assert "chosen (these are inputs, not observations):" in result.output
        assert "SLO              3.0s to first byte, queue included" in result.output
        assert "surge ratio      1.25x traffic growth within one T_total" in result.output

    def test_the_two_derived_thresholds_show_their_fraction_of_q_max(
        self, runner: CliRunner
    ) -> None:
        # (1-h) and (1-2h) at h=0.25. Printed as the multiplier so the reader can check
        # 37.5 against 50 rather than trusting it.
        result = _run(runner, "--peak-rps", "450")
        assert "C_scale_max      37.50 concurrent — scale out here (0.75 x Q_max)" in result.output
        assert "C_scale_min      25.00 concurrent — scale in here (0.50 x Q_max)" in result.output

    def test_the_surge_ratio_moves_both_thresholds(self, runner: CliRunner) -> None:
        # The one chosen input that changes the policy, so it must reach the output.
        result = _run(runner, "--peak-rps", "450", "--max-scaling-per-t-total", "1.4")
        assert result.exit_code == 0
        assert "C_scale_max      30.00 concurrent" in result.output
        assert "C_scale_min      10.00 concurrent" in result.output

    def test_a_surge_ratio_at_or_past_one_and_a_half_is_a_bad_parameter(
        self, runner: CliRunner
    ) -> None:
        # h >= 0.5 puts C_scale_min at or below zero, which deploys as "never scale in".
        result = _run(runner, "--peak-rps", "450", "--max-scaling-per-t-total", "1.5")
        assert result.exit_code == 2
        assert "less than 1.5" in result.output

    def test_utilization_is_printed_because_the_threshold_hides_it(self, runner: CliRunner) -> None:
        # 37.5 in the queue is 97.4% utilized, not three quarters of the way to trouble.
        result = _run(runner, "--peak-rps", "450")
        assert "utilization      97.4% at C_scale_max" in result.output
        assert "steeply non-linear, hence surge_survival" in result.output

    def test_the_lag_is_split_into_measured_bounded_and_this_accounts(
        self, runner: CliRunner
    ) -> None:
        # The most important caveat in the output. Three parts, three provenances: the
        # measured span, the policy term bounded from the alarm's own configuration, and
        # the EC2 provision stage that is this account's placement latency rather than
        # the configuration's.
        result = _run(runner, "--peak-rps", "450")
        out = result.output
        assert "measured        420s capacity request -> traffic served" in out
        assert "+ BOUND         60s policy detection" in out
        assert "arithmetic, not a measurement" in out
        assert "of which        180s EC2 provision + image pull" in out
        assert "not the configuration's" in out

    def test_the_bounded_policy_term_is_added_to_the_lag_the_plan_uses(
        self, runner: CliRunner
    ) -> None:
        # 420 measured + 60 bounded. Production scales out through the policy, so the
        # measured half alone under-states what a surge must be absorbed across.
        assert "T_total          480s, planned against" in _run(runner, "--peak-rps", "450").output

    def test_the_config_block_pastes(self, runner: CliRunner) -> None:
        result = _run(runner, "--peak-rps", "450", "--trough-rps", "30")
        assert result.exit_code == 0
        assert '"kokoro-82m": ModelEndpointConfig(' in result.output
        for field in (
            "min_instances=",
            "max_instances=",
            "scaling_target_value=",
            "scale_in_threshold=",
            "ttfab_slo_ms=",
            "queue_max_depth=",
            "scale_out_cooldown_s=",
            "scale_in_cooldown_s=",
        ):
            assert field in result.output

    def test_every_config_field_carries_its_derivation(self, runner: CliRunner) -> None:
        # The comments are the audit trail. This block gets pasted into config.py, where
        # the arithmetic that produced each number is no longer on screen -- which is
        # exactly how the shipped numbers became four figures nobody could re-derive.
        result = _run(runner, "--peak-rps", "450", "--trough-rps", "30")
        out = result.output
        assert "= (1-h) x Q_max 50 at h=0.25" in out
        assert "= (1-2h) x Q_max, one surge of excess headroom" in out
        assert "# = Q_max: past it a request cannot reach first byte" in out
        assert "# short: target tracking adds one instance at a time" in out
        assert "# long: removed capacity costs a full 480s to replace" in out

    def test_the_alarm_threshold_is_the_ladders_first_rung_not_the_slo(
        self, runner: CliRunner
    ) -> None:
        # The FirstChunkLatency alarm watches service time on an instance already
        # serving, where the request has spent none of its queue allowance. At an
        # SLO-sized threshold it fires only once the endpoint is ~10x past keeping up.
        result = _run(runner, "--peak-rps", "450")
        assert "FirstChunkLatencyP95 alarm: 92ms (p95 TTFAB at one outstanding request)" in (
            result.output
        )
        assert "NOT the 3000ms SLO" in result.output

    def test_a_ladder_without_a_c1_rung_has_no_alarm_threshold(self, runner: CliRunner) -> None:
        # An alarm threshold has to come from somewhere real, so a missing rung says so
        # rather than falling back on the SLO -- which is the number it must not be.
        result = _run_with(
            runner,
            "--peak-rps",
            "450",
            qmax={"steps": [s for s in _steps() if s["concurrency"] != 1]},
            ttotal={},
        )
        assert result.exit_code == 0
        assert "NO THRESHOLD — the ladder had no N=1 rung" in result.output
        assert "tts-bench qmax --concurrency 1,..." in result.output

    def test_costs_are_labelled_upper_bounds(self, runner: CliRunner) -> None:
        # A committed account pays less, and a number presented without that caveat will
        # be quoted back as the price.
        assert "upper bound" in _run(runner, "--peak-rps", "450").output


class TestThresholdUnits:
    """The shipped defect, at the CLI boundary.

    ``C_scale_max`` is a client occupancy; the alarm compares
    ``ConcurrentRequestsPerModel``/*Maximum* over 10s against its threshold. Those
    differed by 1.35x to 9.8x on one kokoro ladder, and deploying the unconverted figure
    is how ``0.713`` — a value no positive arrival rate satisfies — reached a live
    endpoint.
    """

    def test_both_numbers_are_printed_with_the_multiplier_between_them(
        self, runner: CliRunner
    ) -> None:
        # Adjacent lines, with the ratio shown: the two are easy to mistake for a
        # rounding difference and hard to mistake for one when the multiplier is between.
        result = _run(runner, "--peak-rps", "450")
        assert "C_scale_max      37.50 concurrent" in result.output
        assert "in CW units      50.62 = C_scale_max x 1.35" in result.output
        assert "THIS is what deploys" in result.output

    def test_the_converted_figure_is_what_the_config_block_carries(self, runner: CliRunner) -> None:
        # 37.5 x 1.35 for the target, 25.0 x 1.35 for the scale-in threshold. The
        # occupancy stays in the comment, where it cannot be deployed by accident.
        result = _run(runner, "--peak-rps", "450")
        assert "scaling_target_value=50.625," in result.output
        assert "scale_in_threshold=33.750," in result.output
        assert "C_scale_max 37.50 x 1.35 CW units" in result.output

    def test_the_conversion_is_read_at_the_rung_nearest_the_threshold(
        self, runner: CliRunner
    ) -> None:
        # The ratio is not a constant, so reading it at the wrong rung is its own units
        # bug. 37.5 is nearest 50, whose ratio is 1.35 -- not the 9.8 at c=1.
        result = _run(runner, "--peak-rps", "450")
        assert "x 1.35" in result.output
        assert "9.80" not in result.output

    def test_no_measured_conversion_comments_the_field_out(self, runner: CliRunner) -> None:
        # A config that fails to parse is a better outcome than one that deploys a
        # threshold no traffic satisfies. Note the ladder still yields everything else.
        result = _run_with(
            runner, "--peak-rps", "450", qmax={"steps": _steps(with_cloudwatch=False)}, ttotal={}
        )
        assert result.exit_code == 0
        assert "# scaling_target_value=?,  # NO CONVERSION MEASURED" in result.output
        assert "# scale_in_threshold=?,  # same conversion, same reason" in result.output
        assert "scaling_target_value=50" not in result.output

    def test_no_measured_conversion_names_the_rerun_that_fixes_it(self, runner: CliRunner) -> None:
        result = _run_with(
            runner, "--peak-rps", "450", qmax={"steps": _steps(with_cloudwatch=False)}, ttotal={}
        )
        assert "in CW units      UNAVAILABLE" in result.output
        assert "tts-bench qmax --cloudwatch" in result.output
        assert "0.713 reached this endpoint" in result.output

    def test_a_missing_conversion_is_suppressed_rather_than_passed(self, runner: CliRunner) -> None:
        # `----`, not OK: the conversion did not happen, which is not the same as not
        # needing one. High-res datapoints retain 3 hours, so it cannot be backfilled.
        result = _run_with(
            runner, "--peak-rps", "450", qmax={"steps": _steps(with_cloudwatch=False)}, ttotal={}
        )
        assert "[----] threshold_units" in result.output
        assert "[OK  ] threshold_units" not in result.output


class TestTheDeployedConfigIsCompared:
    """Whether the two stored numbers still match what a fresh measurement derives.

    ``queue_max_depth`` and ``scaling_target_value`` are the values ``config.py`` stores
    rather than computes, because both need a measurement — so both can go stale
    silently, and both did. This check is what makes the staleness visible, and the
    fixture is deliberately the numbers this branch measured against the ones deployed.
    """

    def test_a_stale_queue_depth_warns_with_both_numbers(self, runner: CliRunner) -> None:
        # 41 deployed against a measured Q_max of 50. Naming both, and what the deployed
        # value does wrong, is the whole content of the message.
        result = _run(runner, "--peak-rps", "450")
        assert result.exit_code == 0
        assert "queue_max_depth=41 deployed" in result.output
        assert "3000ms end-to-end SLO is 50" in result.output
        assert "admits requests it can only serve late" in result.output

    def test_a_stale_target_is_compared_in_cloudwatch_units(self, runner: CliRunner) -> None:
        # Both sides in the units the alarm reads. Comparing the deployed value against
        # the client occupancy would report a mismatch of exactly the size of the
        # conversion and call it drift.
        result = _run(runner, "--peak-rps", "450")
        assert "scaling_target_value=0.713 deployed" in result.output
        assert "this plan derives 50.625" in result.output
        assert "into ConcurrentRequestsPerModel/Maximum units" in result.output

    def test_it_warns_rather_than_failing_the_command_that_found_it(
        self, runner: CliRunner
    ) -> None:
        # The plan *is* the answer, and finding the deployed value wrong is why it was
        # run. Exiting non-zero would fail the command that just told you what to fix.
        assert _run(runner, "--peak-rps", "450").exit_code == 0

    def test_agreement_is_silent(self, runner: CliRunner) -> None:
        # A ladder that bracketed at 41 with the deployed conversion reproduces both
        # stored numbers, and then there is nothing to say.
        ladder = {rung: p95 for rung, p95 in LADDER.items() if rung != 50}
        steps = [
            {
                "run_index": 0,
                "step_index": index,
                "concurrency": rung,
                "achieved_rps": 9.0,
                "completed": 100,
                "ok": 100,
                "chars": 2500,
                "ttfab_p95_ms": p95,
                "concurrency_mean": float(rung),
                # 0.713 / 30.75 -- the conversion that reproduces the deployed target
                # from C_scale_max at Q_max 41.
                "server_concurrency_peak": rung * (0.713 / 30.75),
                "meets_slo": True,
                "saturated": False,
                "settled": True,
                "usable": True,
            }
            for index, (rung, p95) in enumerate(ladder.items())
        ]
        result = _run_with(
            runner,
            "--peak-rps",
            "450",
            qmax={
                "q_max": 41,
                "q_max_per_run": [41, 41],
                "ttfab_p95_at_q_max_ms": 2400.0,
                "steps": steps,
            },
            ttotal={},
        )
        assert result.exit_code == 0
        # Asserted on "WARNING:" rather than "deployed": the T_total breakdown above says
        # "the deployed alarm's periods", so the bare word is in every render.
        assert "WARNING:" not in result.output

    def test_an_unrelated_endpoint_is_not_compared(self, runner: CliRunner) -> None:
        # The lookup is by endpoint name. A ladder measured on an endpoint config.py does
        # not know must not be compared against some other model's stored numbers.
        result = _run_with(
            runner, "--peak-rps", "450", qmax={"endpoint": "speech-experiment"}, ttotal={}
        )
        assert result.exit_code == 0
        assert "WARNING:" not in result.output


class TestTheTwoKnownLimits:
    """`surge_survival` and `scale_in_safety` reach the CLI, with their numbers.

    Both are arithmetic consequences of the simple rule rather than preferences, and both
    are why the rule is shippable: the tool says out loud what it costs instead of
    leaving it to be argued.
    """

    def test_surge_survival_reports_the_simulated_probability(self, runner: CliRunner) -> None:
        result = _run(runner, "--peak-rps", "450")
        assert "[WARN] surge_survival" in result.output
        assert "97.4% utilization on a single-server queue" in result.output
        assert "P(the queue reaches Q_max before then) is 97%" in result.output
        assert "reserves queue slots, which is a stock" in result.output

    def test_surge_survival_names_the_three_things_that_would_fix_it(
        self, runner: CliRunner
    ) -> None:
        result = _run(runner, "--peak-rps", "450")
        assert "scale out earlier than the simple rule" in result.output
        assert "shorten T_total" in result.output
        assert "hold standing headroom in instances" in result.output

    def test_scale_in_safety_warns_at_the_deployed_floor_of_one(self, runner: CliRunner) -> None:
        # kokoro runs min_instances=1, so 2->1 is the common case rather than the corner
        # one, and the survivor inherits 25.0 x 2 = 50.0 -- Q_max exactly.
        result = _run(runner, "--peak-rps", "450")
        assert "[WARN] scale_in_safety" in result.output
        assert "only stable from 3 instances up" in result.output
        assert "a 2->1 scale-in leaves the survivors at 50.00" in result.output

    def test_the_safe_floor_reaches_the_config_block(self, runner: CliRunner) -> None:
        # A warning printed once into a terminal is not a record. The block is what gets
        # pasted, so the caveat has to travel with min_instances.
        result = _run(runner, "--peak-rps", "450", "--trough-rps", "30")
        assert "min_instances=1,  # trough 30 rps; 3 is the smallest safe for scale-in" in (
            result.output
        )

    def test_raising_the_floor_satisfies_it(self, runner: CliRunner) -> None:
        result = _run(runner, "--peak-rps", "450", "--min-floor", "3")
        assert result.exit_code == 0
        assert "[OK  ] scale_in_safety" in result.output
        assert "min_instances=3," in result.output


class TestExitStatusGatesADeploy:
    def test_an_slo_past_the_invocation_ceiling_exits_non_zero(self, runner: CliRunner) -> None:
        # W_max is SLO - p95 service, so the request deadline *is* the SLO: a 70s promise
        # cannot be served inside a 60s invocation ceiling whatever the queue does.
        result = _run_with(runner, "--peak-rps", "450", qmax={"slo_ms": 70_000}, ttotal={})
        assert result.exit_code == 1
        assert "[STOP] invocation_ceiling" in result.output
        assert "past the 60s SageMaker invocation ceiling" in result.output

    def test_an_slo_under_the_models_own_tail_exits_non_zero(self, runner: CliRunner) -> None:
        # The other infeasibility: p95 service is 165ms, so a 150ms promise is missed
        # before a request waits at all. No queue depth or fleet size rescues it.
        result = _run_with(runner, "--peak-rps", "450", qmax={"slo_ms": 150}, ttotal={})
        assert result.exit_code == 1
        assert "[STOP] slo_budget" in result.output
        assert "No queue depth, instance count, or scaling policy rescues this" in result.output

    def test_an_infeasible_config_block_does_not_claim_the_ceiling_fits(
        self, runner: CliRunner
    ) -> None:
        # The block is what gets pasted, so a comment contradicting the STOP above it is
        # worse than no comment. It once said "fits the 60s invocation ceiling" on a plan
        # that had just failed that very check.
        result = _run_with(runner, "--peak-rps", "450", qmax={"slo_ms": 70_000}, ttotal={})
        assert "EXCEEDS the invocation ceiling" in result.output
        assert "do not deploy this" in result.output
        assert "fits the 60s invocation ceiling" not in result.output

    def test_a_feasible_plan_exits_zero_even_with_warnings(self, runner: CliRunner) -> None:
        # WARN findings are the normal case -- both known limits of the simple rule fire
        # on kokoro's own numbers. Exiting non-zero on them would make the gate useless.
        result = _run(runner, "--peak-rps", "450")
        assert result.exit_code == 0
        assert "[WARN]" in result.output

    def test_an_explicit_ceiling_moves_the_gate(self, runner: CliRunner) -> None:
        # For a platform whose limit is not SageMaker's 60s.
        result = _run(runner, "--peak-rps", "450", "--ceiling-s", "2")
        assert result.exit_code == 1
        assert "past the 2s SageMaker invocation ceiling" in result.output

    def test_a_moved_ceiling_is_the_one_the_config_block_names(self, runner: CliRunner) -> None:
        # 30s still clears a 3.0s deadline, so this is the *passing* branch -- the one that
        # read the module constant and so asserted a fit against 60s, a limit this run
        # never tested. The block is pasted into config.py verbatim, so its comment has to
        # name the ceiling the invocation_ceiling finding above it actually judged.
        result = _run(runner, "--peak-rps", "450", "--ceiling-s", "30")
        assert result.exit_code == 0
        assert "fits the 30s invocation ceiling" in result.output
        assert "fits the 60s invocation ceiling" not in result.output

    def test_an_unmoved_ceiling_still_names_sixty(self, runner: CliRunner) -> None:
        # The constant is the default, so an ordinary run reads exactly as it did before
        # the ceiling started travelling on the plan.
        result = _run(runner, "--peak-rps", "450")
        assert result.exit_code == 0
        assert "fits the 60s invocation ceiling" in result.output

    def test_a_moved_ceiling_that_stops_the_plan_claims_no_fit_at_all(
        self, runner: CliRunner
    ) -> None:
        # Both halves at once, which is the shape of a real --ceiling-s run that fails:
        # the verdict comes off the finding so nothing claims a fit, and the 2s that was
        # judged does not turn into a printed 60s on the way out either.
        result = _run(runner, "--peak-rps", "450", "--ceiling-s", "2")
        assert result.exit_code == 1
        assert "EXCEEDS the invocation ceiling" in result.output
        assert "fits the 2s invocation ceiling" not in result.output
        assert "fits the 60s invocation ceiling" not in result.output

    def test_the_worst_verdict_is_printed_first(self, runner: CliRunner) -> None:
        # A STOP must not scroll off the top behind six OK lines -- that ordering is the
        # whole reason the ceiling is a verdict rather than a log line.
        result = _run_with(runner, "--peak-rps", "450", qmax={"slo_ms": 70_000}, ttotal={})
        findings = result.output.split("Findings:")[1]
        assert findings.index("[STOP]") < findings.index("[OK  ]")


class TestArtifact:
    def test_nothing_is_written_unless_asked(self, runner: CliRunner) -> None:
        # Unlike a measurement, this run is cheap to repeat -- so the default is the
        # opposite of `qmax`'s, where omitting --output would discard 40 minutes.
        with runner.isolated_filesystem():
            qmax, ttotal = _write_qmax(), _write_ttotal()
            result = runner.invoke(
                main,
                ["plan", "--qmax", qmax, "--ttotal", ttotal, "--peak-rps", "450"],
                catch_exceptions=False,
            )
            assert result.exit_code == 0
            assert sorted(Path(".").iterdir()) == [Path("qmax.json"), Path("ttotal.json")]

    def test_output_writes_the_plan_and_the_block_that_pastes(self, runner: CliRunner) -> None:
        with runner.isolated_filesystem():
            qmax, ttotal = _write_qmax(), _write_ttotal()
            result = runner.invoke(
                main,
                [
                    "plan",
                    "--qmax",
                    qmax,
                    "--ttotal",
                    ttotal,
                    "--peak-rps",
                    "450",
                    "--output",
                    "plan.json",
                ],
                catch_exceptions=False,
            )
            assert result.exit_code == 0
            document = json.loads(Path("plan.json").read_text())

        assert document["plan"]["q_max"] == Q_MAX
        assert document["plan"]["c_scale_max"] == pytest.approx(37.5)
        assert document["plan"]["c_scale_min"] == pytest.approx(25.0)
        assert document["verdict"].startswith("warn")
        # The rendered block travels with the structured plan: it is the thing an
        # operator pastes, and regenerating it later means re-running a renderer that
        # may have changed in between.
        assert "ModelEndpointConfig(" in document["config_block"]
        assert "FirstChunkLatencyP95 alarm" in document["alarm_threshold"]

    def test_a_stated_lag_is_recorded_as_stated_in_the_artifact(self, runner: CliRunner) -> None:
        # Not derivable from the plan alone: a lag stated whole reads identically to one
        # measured whole, and that is the distinction the flag exists for.
        with runner.isolated_filesystem():
            qmax = _write_qmax()
            result = runner.invoke(
                main,
                [
                    "plan",
                    "--qmax",
                    qmax,
                    "--peak-rps",
                    "450",
                    "--assume-t-total",
                    "300",
                    "--output",
                    "plan.json",
                ],
                catch_exceptions=False,
            )
            assert result.exit_code == 0
            document = json.loads(Path("plan.json").read_text())

        assert not document["t_total_measured"]
        assert document["plan"]["measured"]["t_total_s"] == pytest.approx(300.0)

    def test_a_measured_lag_is_recorded_as_measured(self, runner: CliRunner) -> None:
        with runner.isolated_filesystem():
            qmax, ttotal = _write_qmax(), _write_ttotal()
            runner.invoke(
                main,
                [
                    "plan",
                    "--qmax",
                    qmax,
                    "--ttotal",
                    ttotal,
                    "--peak-rps",
                    "450",
                    "--output",
                    "plan.json",
                ],
                catch_exceptions=False,
            )
            document = json.loads(Path("plan.json").read_text())

        assert document["t_total_measured"]

    def test_the_written_verdict_matches_the_exit_status(self, runner: CliRunner) -> None:
        with runner.isolated_filesystem():
            qmax = _write_qmax(slo_ms=70_000)
            ttotal = _write_ttotal()
            result = runner.invoke(
                main,
                [
                    "plan",
                    "--qmax",
                    qmax,
                    "--ttotal",
                    ttotal,
                    "--peak-rps",
                    "450",
                    "--output",
                    "plan.json",
                ],
                catch_exceptions=False,
            )
            document = json.loads(Path("plan.json").read_text())

        assert result.exit_code == 1
        # Written before the non-zero exit: an artifact that exists only for passing runs
        # is useless for diagnosing a failing gate.
        assert document["verdict"] == "INFEASIBLE"


class TestTouchesNoAws:
    def test_no_boto_client_is_constructed(self, runner: CliRunner, monkeypatch) -> None:
        # Worth pinning rather than assuming: `plan` is the command that can run on a
        # laptop with no credentials, and the cost lookup is the plausible place a future
        # change would reach for the Pricing API.
        import boto3

        def refuse(*args, **kwargs):
            raise AssertionError(f"plan constructed a boto3 client: {args} {kwargs}")

        monkeypatch.setattr(boto3, "client", refuse)
        assert _run(runner, "--peak-rps", "450").exit_code == 0

    def test_no_region_option_is_offered(self) -> None:
        # Every other command takes --region. Offering one here would imply an API call
        # this command does not make.
        assert not any(param.name == "region" for param in main.commands["plan"].params)


class TestHelp:
    def test_help_needs_no_heavy_imports(self, runner: CliRunner) -> None:
        # cli.py imports planner/scale_report lazily so numpy and botocore stay out of
        # `--help`. A module-level import would work but slow every invocation.
        result = runner.invoke(main, ["plan", "--help"], catch_exceptions=False)
        assert result.exit_code == 0
        # Whitespace-collapsed: click rewraps the docstring to the terminal width, so
        # asserting on a literal phrase would break with the wrap position.
        assert "touches no AWS" in " ".join(result.output.split())

    def test_every_option_is_documented(self) -> None:
        undocumented = [
            param.name
            for param in main.commands["plan"].params
            if isinstance(param, click.Option) and not param.help
        ]
        assert undocumented == []

    def test_the_surge_ratio_defaults_to_the_worked_example(self) -> None:
        # 1.25 is the ratio the docs and the config comment are written around, and the
        # one a reader will assume rather than pass.
        assert _param("max_scaling_per_t_total").default == 1.25

    def test_the_floor_defaults_to_one_not_to_the_safe_size(self) -> None:
        # Deliberate: raising it to 3 to satisfy scale_in_safety is a cost decision, so
        # the tool reports the flap and leaves the choice to the operator.
        assert _param("min_instances_floor").default == 1

    def test_the_help_explains_how_the_thresholds_are_derived(self, runner: CliRunner) -> None:
        # Someone reading --help is deciding what to pass for the surge ratio, and the
        # formula is the only thing that makes 1.25 mean something.
        result = runner.invoke(main, ["plan", "--help"], catch_exceptions=False)
        collapsed = " ".join(result.output.split())
        assert "C_scale_max = (1-h) x Q_max" in collapsed
        assert "C_scale_min = (1-2h) x Q_max" in collapsed
