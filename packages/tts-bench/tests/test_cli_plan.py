"""Tests for the `plan` CLI command.

`plan` is the only command in this package that touches no AWS and sends no load: it
reads two artifacts and prints a configuration. So these tests are about the four ways
a *pure* command can still mislead:

- **It refuses inputs that cannot be combined.** A `C_max` curve and a `T_total` lag
  from different configurations produce a plan for a fleet that exists nowhere, and no
  number in the output looks wrong. `--allow-config-mismatch` is the only way past, and
  it must warn.
- **It refuses to guess a missing input.** No stated peak means no fleet size; no lag
  means no headroom sizing. Both are usage errors rather than defaults, because a
  defaulted peak would produce a plausible plan for load nobody expects.
- **What it prints distinguishes measured from assumed.** Every swept row is built on a
  provision time somebody stated. If a row can pass for a measurement the sweep is
  worse than not running it.
- **Its exit status can gate a deploy.** A row that breaks the 60s invocation ceiling
  exits non-zero, so the same command works in a pipeline as on a terminal.

Every test runs inside `CliRunner.isolated_filesystem()` — `--output` takes a relative
path, and the artifact fixtures are written next to it. `caplog` is not used: loguru
does not propagate to the stdlib logging tree, so an assertion against it would pass
whether or not anything was emitted.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

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

S_MEAN_S = 0.10602401316328536
C_MAX_300 = 1.152542372881356


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


def _write_cmax(**overrides: Any) -> str:
    """A `cmax` artifact shaped like the committed one, in the cwd."""
    payload: dict[str, Any] = {
        "model_name": MODEL,
        "endpoint": ENDPOINT,
        "instance_type": "ml.g5.xlarge",
        "run_id": "db5bec78d522",
        "c_max_curve": {"300": C_MAX_300, "500": 1.2627118644067796},
        "curve_spread": {"300": 0.0, "500": 0.17449664429530191},
        "runs_contributing": {"300": 3, "500": 2},
        "runs": 3,
        "hold_s": 240.0,
        "measure_window_s": 60.0,
        "s_mean_s": S_MEAN_S,
        "s_p95_s": 0.1645768812391907,
        "frozen": True,
        "instance_counts_observed": [1],
        "transport": "bidi",
        "deployed_config": DEPLOYED.to_dict(),
        "knees": [
            {
                "ttfab_budget_ms": 300,
                "concurrency": C_MAX_300,
                "offered_rps": 9.66724828056195,
                "p95_ttfab_ms": 237.79399786144495,
                "step_index": 1,
                "bracketed": True,
            }
        ],
        "steps": [],
    }
    payload.update(overrides)
    Path("cmax.json").write_text(json.dumps(payload))
    return "cmax.json"


def _write_ttotal(**overrides: Any) -> str:
    """A `ttotal` artifact with a provision stage to substitute out, in the cwd."""
    payload: dict[str, Any] = {
        "model_name": MODEL,
        "endpoint": ENDPOINT,
        "run_id": "ttotal456",
        "trigger": "drive-load",
        "config_slug": SLUG,
        "t_total_s": 420.0,
        "t_total_bounded": False,
        "durations": [
            {"from": "load_applied", "to": "metric_published", "seconds": 20.0},
            {"from": "metric_published", "to": "alarm_fired", "seconds": 30.0},
            {"from": "alarm_fired", "to": "activity_started", "seconds": 10.0},
            {"from": "activity_started", "to": "instance_logging", "seconds": 180.0},
            {"from": "instance_logging", "to": "ready", "seconds": 120.0},
            {"from": "ready", "to": "traffic_recovered", "seconds": 60.0},
        ],
        "missing_stages": [],
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
        measured = _write_cmax()
        ttotal = _write_ttotal()
        return runner.invoke(
            main,
            ["plan", "--measured", measured, "--ttotal", ttotal, *args],
            catch_exceptions=False,
        )


class TestRefusesToGuess:
    def test_a_missing_peak_is_a_usage_error(self, runner: CliRunner) -> None:
        # Not a default. A defaulted peak produces a plausible fleet size for load
        # nobody stated, which is the one output of this command nobody can sanity-check
        # by eye.
        result = _run(runner)
        assert result.exit_code == 2
        assert "--peak-rps or --peak-streams is required" in result.output

    def test_streams_alone_are_enough(self, runner: CliRunner) -> None:
        result = _run(runner, "--peak-streams", "4")
        assert result.exit_code == 0

    def test_a_missing_ttotal_is_a_usage_error(self, runner: CliRunner) -> None:
        with runner.isolated_filesystem():
            measured = _write_cmax()
            result = runner.invoke(
                main, ["plan", "--measured", measured, "--peak-rps", "20"], catch_exceptions=False
            )
        assert result.exit_code == 2
        assert "--ttotal is required" in result.output
        assert "--assume-t-total" in result.output

    def test_a_stated_lag_stands_in_for_the_artifact(self, runner: CliRunner) -> None:
        # The case the plan anticipates: the second instance will not place in this
        # account, so there is no lag to read and one gets stated instead.
        with runner.isolated_filesystem():
            measured = _write_cmax()
            result = runner.invoke(
                main,
                [
                    "plan",
                    "--measured",
                    measured,
                    "--peak-rps",
                    "20",
                    "--assume-t-total",
                    "300",
                ],
                catch_exceptions=False,
            )
        assert result.exit_code == 0
        # The number in use, and that it was stated. "not measured" on its own sends
        # the reader hunting for the lag every row below was built from.
        assert "300s STATED, not measured" in result.output

    def test_a_stated_lag_is_never_labelled_measured(self, runner: CliRunner) -> None:
        # Three places said "measured" about a command-line argument: the inputs
        # header, the sweep's provision column, and the per-row section heading. Each
        # is a separate render path, so each is asserted.
        with runner.isolated_filesystem():
            measured = _write_cmax()
            result = runner.invoke(
                main,
                ["plan", "--measured", measured, "--peak-rps", "20", "--assume-t-total", "300"],
                catch_exceptions=False,
            )
        assert result.exit_code == 0
        assert "T_total as stated, k=2" in result.output
        assert "provision 'stated' means" in result.output
        assert "T_total as measured" not in result.output
        # And the finding warns rather than calling it "used exactly as measured".
        assert "stated whole, not measured" in result.output
        assert "[WARN] provision_stage" in result.output

    def test_a_measured_lag_is_still_labelled_measured(self, runner: CliRunner) -> None:
        # The other half of the pair: the fix must not relabel a real measurement.
        result = _run(runner, "--peak-rps", "20")
        assert "420s as measured" in result.output
        assert "T_total as measured, k=2" in result.output
        assert "STATED" not in result.output

    def test_a_stated_lag_needs_no_fingerprint_to_pair(self, runner: CliRunner) -> None:
        # A stated number carries no fingerprint, so the pairing check must not run --
        # it would report a mismatch where there is nothing to mismatch.
        with runner.isolated_filesystem():
            measured = _write_cmax()
            result = runner.invoke(
                main,
                ["plan", "--measured", measured, "--peak-rps", "20", "--assume-t-total", "300"],
                catch_exceptions=False,
            )
        assert result.exit_code == 0
        assert "cannot pair" not in result.output

    def test_an_unreadable_cmax_artifact_says_which_file(self, runner: CliRunner) -> None:
        with runner.isolated_filesystem():
            Path("junk.json").write_text("not json")
            _write_ttotal()
            result = runner.invoke(
                main,
                ["plan", "--measured", "junk.json", "--ttotal", "ttotal.json", "--peak-rps", "20"],
                catch_exceptions=False,
            )
        assert result.exit_code == 1
        assert "junk.json" in result.output
        assert "tts-bench cmax" in result.output

    def test_a_missing_cmax_artifact_fails_at_parse_time(self, runner: CliRunner) -> None:
        # `exists=True` on the option, so this never reaches the planner.
        with runner.isolated_filesystem():
            result = runner.invoke(main, ["plan", "--measured", "nope.json", "--peak-rps", "20"])
        assert result.exit_code == 2

    def test_an_unreadable_ttotal_artifact_says_which_file(self, runner: CliRunner) -> None:
        with runner.isolated_filesystem():
            measured = _write_cmax()
            Path("junk.json").write_text("{[")
            result = runner.invoke(
                main,
                ["plan", "--measured", measured, "--ttotal", "junk.json", "--peak-rps", "20"],
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
        with runner.isolated_filesystem():
            measured = _write_cmax()
            ttotal = _write_ttotal(config_slug="g6xlarge-abcd1234")
            result = runner.invoke(
                main,
                ["plan", "--measured", measured, "--ttotal", ttotal, "--peak-rps", "20"],
                catch_exceptions=False,
            )
        assert result.exit_code == 1
        assert "cannot pair" in result.output
        # Both named: which two configurations disagreed is the whole content of the
        # message, and an operator with several artifacts cannot act without it.
        assert SLUG in result.output
        assert "g6xlarge-abcd1234" in result.output

    def test_a_pre_fingerprint_curve_counts_as_a_mismatch(self, runner: CliRunner) -> None:
        # The old artifacts are exactly when this check matters, so an absent
        # fingerprint must not read as agreement. A cmax artifact still knows its
        # instance type, so it renders `nodigest` rather than nothing -- which is a
        # mismatch against a real digest, and says which half is missing.
        with runner.isolated_filesystem():
            measured = _write_cmax(deployed_config={})
            ttotal = _write_ttotal()
            result = runner.invoke(
                main,
                ["plan", "--measured", measured, "--ttotal", ttotal, "--peak-rps", "20"],
                catch_exceptions=False,
            )
        assert result.exit_code == 1
        assert "cannot pair" in result.output
        assert "g5xlarge-nodigest" in result.output

    def test_a_ttotal_artifact_with_no_fingerprint_at_all_is_refused(
        self, runner: CliRunner
    ) -> None:
        # The other side has no instance type to fall back on, so it renders as nothing
        # and gets its own message: there is not a second configuration to name.
        with runner.isolated_filesystem():
            measured = _write_cmax()
            ttotal = _write_ttotal(config_slug="")
            result = runner.invoke(
                main,
                ["plan", "--measured", measured, "--ttotal", ttotal, "--peak-rps", "20"],
                catch_exceptions=False,
            )
        assert result.exit_code == 1
        assert "no configuration fingerprint" in result.output
        assert "predates fingerprinting" in result.output

    def test_the_flag_lets_a_mismatch_through_with_a_warning(
        self, runner: CliRunner, logged: list[str]
    ) -> None:
        with runner.isolated_filesystem():
            measured = _write_cmax()
            ttotal = _write_ttotal(config_slug="g6xlarge-abcd1234")
            result = runner.invoke(
                main,
                [
                    "plan",
                    "--measured",
                    measured,
                    "--ttotal",
                    ttotal,
                    "--peak-rps",
                    "20",
                    "--allow-config-mismatch",
                ],
                catch_exceptions=False,
            )
        assert result.exit_code == 0
        assert any("different configurations" in message for message in logged)

    def test_matching_fingerprints_need_no_flag(self, runner: CliRunner) -> None:
        result = _run(runner, "--peak-rps", "20")
        assert result.exit_code == 0
        assert SLUG in result.output


class TestOutputShape:
    def test_prints_inputs_sweep_findings_and_a_config_block(self, runner: CliRunner) -> None:
        result = _run(runner, "--peak-rps", "20", "--trough-rps", "2")
        assert result.exit_code == 0
        out = result.output
        # Inputs first: the reader's first question is "on what".
        assert "measured:" in out
        assert "assumed (these are inputs, not observations):" in out
        assert "C_max" in out and "Lambda_cap" in out
        # Then the four numbers, in a form that pastes.
        assert "ModelEndpointConfig(" in out
        for field in (
            "min_instances=",
            "max_instances=",
            "scaling_target_value=",
            "queue_max_depth=",
            "scale_out_cooldown_s=",
            "scale_in_cooldown_s=",
        ):
            assert field in out

    def test_labels_the_measured_stage_split(self, runner: CliRunner) -> None:
        # The most important caveat in the output: which part of T_total transfers to
        # another account and which is ours alone.
        result = _run(runner, "--peak-rps", "20")
        assert "EC2 provision" in result.output
        assert "transferable" in result.output

    def test_the_default_budget_is_the_tightest_measured(self, runner: CliRunner) -> None:
        # Conservative by construction: the knee at a tight SLO is the smaller number,
        # so defaulting to it sizes the fleet up rather than down.
        result = _run(runner, "--peak-rps", "20")
        assert "ttfab_budget_ms=300" in result.output

    def test_an_explicit_budget_reads_the_other_knee(self, runner: CliRunner) -> None:
        result = _run(runner, "--peak-rps", "20", "--ttfab-budget-ms", "500")
        assert "ttfab_budget_ms=500" in result.output
        assert "1.26 concurrent" in result.output

    def test_an_unmeasured_budget_is_refused_by_name(self, runner: CliRunner) -> None:
        result = _run(runner, "--peak-rps", "20", "--ttfab-budget-ms", "100")
        assert result.exit_code == 1
        assert "no C_max measured" in result.output
        assert "[300, 500]" in result.output

    def test_reports_which_constraint_binds(self, runner: CliRunner) -> None:
        result = _run(runner, "--peak-rps", "20")
        assert "binding_constraint" in result.output
        assert "surge_headroom" in result.output

    def test_says_costs_are_upper_bounds(self, runner: CliRunner) -> None:
        # A committed account pays less, and a number presented without that caveat
        # will be quoted back as the price.
        result = _run(runner, "--peak-rps", "20")
        assert "upper bound" in result.output


class TestTheSloIsTheOnlyQueueingInput:
    """The end-to-end SLO in, W_max and Q_max out — and nowhere to state them directly.

    `--max-added-wait` used to sit beside `--ttfab-budget-ms` with nothing relating them,
    which is how the deployed config came to promise 300ms while allowing a 20s queue. The
    tests here are about the three ways that could come back: a second way to say it, a
    W_max that reads as an input, and a Q_max that does not follow the promise.
    """

    def test_the_retired_flag_is_gone_rather_than_aliased(self, runner: CliRunner) -> None:
        # Two ways to say it is how the two came to disagree. A rejected flag is a worse
        # error message than an accepted one and a better outcome.
        result = _run(runner, "--peak-rps", "20", "--max-added-wait", "2")
        assert result.exit_code == 2
        assert "no such option" in result.output.lower()

    def test_the_default_slo_is_the_stated_promise(self) -> None:
        # 3s to first byte, queue included. Defaulted because it is the promise this
        # tooling exists to profile against, not an arbitrary starting point.
        assert _param("ttfab_slo_ms").default == 3000

    def test_w_max_prints_as_derived_with_its_arithmetic(self, runner: CliRunner) -> None:
        # Shown as a subtraction rather than a number: a reader who takes W_max for an
        # input will go looking for the flag that sets it, and there isn't one.
        result = _run(runner, "--peak-rps", "20")
        assert "derived from the SLO (not an input" in result.output
        assert "W_max            2.84s queueing budget = 3.0s SLO - 0.165s p95 service" in (
            result.output
        )

    def test_the_slo_is_listed_as_an_input_and_w_max_is_not(self, runner: CliRunner) -> None:
        assert "SLO              3.0s to first byte, queue included" in (
            _run(runner, "--peak-rps", "20").output
        )

    def test_the_queue_depth_follows_the_slo(self, runner: CliRunner) -> None:
        # Lambda_cap 10.87 rps x W_max. At 3s that is 30; at 10s, 106. The number is not
        # settable, so this is the only way to move it.
        assert "queue_max_depth=30," in _run(runner, "--peak-rps", "20").output
        assert (
            "queue_max_depth=106,"
            in _run(runner, "--peak-rps", "20", "--ttfab-slo-ms", "10000").output
        )

    def test_the_config_block_traces_the_queue_back_to_the_slo(self, runner: CliRunner) -> None:
        # The comment is the audit trail: this block gets pasted into config.py, where
        # the arithmetic that produced the number is no longer on screen.
        result = _run(runner, "--peak-rps", "20")
        assert "ttfab_slo_ms=3000," in result.output
        assert "Lambda_cap 10.87 rps x W_max 2.84s (= 3.0s SLO - 0.165s p95)" in result.output

    def test_the_measurement_budget_stays_separate_from_the_promise(
        self, runner: CliRunner
    ) -> None:
        # Both fields are emitted, and they differ by an order of magnitude on purpose:
        # 300ms is the measured column C_max was read at, 3000ms is the promise. The
        # FirstChunkLatencyP95 alarm reads the tighter one.
        result = _run(runner, "--peak-rps", "20")
        assert "ttfab_budget_ms=300," in result.output
        assert "ttfab_slo_ms=3000," in result.output

    def test_the_budget_selector_does_not_move_the_queue(self, runner: CliRunner) -> None:
        # Reading the knee at a looser budget changes C_max and so Lambda_cap, but W_max
        # comes from the SLO alone. Worth pinning: before the reframe these were the same
        # knob wearing two hats.
        result = _run(runner, "--peak-rps", "20", "--ttfab-budget-ms", "500")
        assert "W_max            2.84s" in result.output


class TestWhichCMaxWasDividedBy:
    def test_names_the_measurement_beside_the_number(self, runner: CliRunner) -> None:
        # C_max is what every fleet size below divides by, and the fixture artifact has no
        # throughput ceiling -- so the basis has to say the comparison never happened
        # rather than let the knee read as the answer.
        result = _run(runner, "--peak-rps", "20")
        assert "C_max            1.15 concurrent, at the p95 TTFAB 300ms latency knee" in (
            result.output
        )
        assert "ceiling unmeasured" in result.output

    def test_a_lower_bound_says_the_fleet_is_over_sized(self, runner: CliRunner) -> None:
        # An unbracketed knee: the ladder ran out while still passing, so the real limit
        # is higher and every instance count here is an over-estimate.
        with runner.isolated_filesystem():
            measured = _write_cmax(
                knees=[
                    {
                        "ttfab_budget_ms": 300,
                        "concurrency": C_MAX_300,
                        "offered_rps": 9.66724828056195,
                        "p95_ttfab_ms": 237.79399786144495,
                        "step_index": 1,
                        "bracketed": False,
                    }
                ]
            )
            ttotal = _write_ttotal()
            result = runner.invoke(
                main,
                ["plan", "--measured", measured, "--ttotal", ttotal, "--peak-rps", "20"],
                catch_exceptions=False,
            )
        assert result.exit_code == 0
        assert "LOWER BOUND, so the fleet below is over-sized" in result.output
        assert "[WARN] c_max_source" in result.output

    def test_a_throughput_ceiling_binds_and_says_the_knee_was_backlog(
        self, runner: CliRunner
    ) -> None:
        # Kokoro's regime on bidi, and the state no committed artifact is in yet: the
        # server stopped keeping up before latency crossed the budget, so the knee's
        # higher concurrency was accumulated queue.
        with runner.isolated_filesystem():
            measured = _write_cmax(
                throughput_ceiling={
                    "max_sustained_rps": 9.0,
                    "concurrency": 0.95,
                    "observed_concurrency": 2.93,
                    "offered_rps": 9.66724828056195,
                    "p95_ttfab_ms": 551.0,
                    "step_index": 2,
                    "bracketed": True,
                }
            )
            ttotal = _write_ttotal()
            result = runner.invoke(
                main,
                ["plan", "--measured", measured, "--ttotal", ttotal, "--peak-rps", "20"],
                catch_exceptions=False,
            )
        assert result.exit_code == 0
        assert "C_max            0.95 concurrent, at the throughput ceiling" in result.output
        assert "queue backlog, not capacity" in result.output
        # And it has to reach the fleet size, not stop at the finding: 0.95 is under the
        # 1.15 knee, so the same peak needs more instances.
        assert "max_instances=6," in result.output


class TestTheDeployedQueueIsCompared:
    """Whether the stored ``queue_max_depth`` still matches what the SLO implies.

    It is the one derived number `config.py` stores rather than computes, because it needs
    a measured `Lambda_cap`. So it can go stale silently, and it did: 296 outlived the
    W_max it came from. This check is what makes the staleness visible.
    """

    def test_a_disagreement_warns_with_both_numbers(self, runner: CliRunner) -> None:
        # The deployed value is 41, from the real artifact's p95; the fixture's p95 is a
        # hair different and gives 30. Naming both is the whole content of the message.
        result = _run(runner, "--peak-rps", "20")
        assert result.exit_code == 0
        assert "WARNING" in result.output
        assert "queue_max_depth=41 deployed" in result.output
        assert "3000ms end-to-end SLO implies 30" in result.output
        assert "admits requests it can only serve late" in result.output

    def test_it_warns_rather_than_failing_the_command_that_found_it(
        self, runner: CliRunner
    ) -> None:
        # The plan *is* the answer, and finding the deployed value wrong is why it was
        # run. Exiting non-zero would fail the command that just told you what to fix.
        assert _run(runner, "--peak-rps", "20").exit_code == 0

    def test_agreement_is_silent(self, runner: CliRunner) -> None:
        # A 4s SLO puts Q_max at 41, which is what kokoro has deployed.
        result = _run(runner, "--peak-rps", "20", "--ttfab-slo-ms", "4000")
        assert result.exit_code == 0
        assert "queue_max_depth=41 deployed" not in result.output
        assert "WARNING" not in result.output

    def test_an_unrelated_endpoint_is_not_compared(self, runner: CliRunner) -> None:
        # The lookup is by endpoint name. A curve measured on an endpoint config.py does
        # not know must not be compared against some other model's stored depth.
        with runner.isolated_filesystem():
            measured = _write_cmax(endpoint="speech-experiment")
            result = runner.invoke(
                main,
                [
                    "plan",
                    "--measured",
                    measured,
                    "--peak-rps",
                    "20",
                    "--assume-t-total",
                    "300",
                ],
                catch_exceptions=False,
            )
        assert result.exit_code == 0
        assert "deployed, but a" not in result.output


class TestSweeps:
    def test_a_provision_sweep_emits_one_row_per_value(self, runner: CliRunner) -> None:
        result = _run(runner, "--peak-rps", "20", "--provision-s", "60,300,600")
        assert result.exit_code == 0
        for label in ("provision 60s", "provision 300s", "provision 600s"):
            assert label in result.output

    def test_a_swept_row_says_the_provision_time_was_stated(self, runner: CliRunner) -> None:
        result = _run(runner, "--peak-rps", "20", "--provision-s", "60")
        assert "assumes a 60s EC2 provision stage" in result.output

    def test_the_unswept_row_says_the_lag_is_as_measured(self, runner: CliRunner) -> None:
        result = _run(runner, "--peak-rps", "20")
        assert "T_total as measured" in result.output

    def test_a_k_sweep_emits_one_row_per_value(self, runner: CliRunner) -> None:
        result = _run(runner, "--peak-rps", "20", "--sweep-k", "1,2,3,5")
        assert result.exit_code == 0
        for k in (1, 2, 3, 5):
            assert f"k={k}" in result.output

    def test_both_sweeps_cross(self, runner: CliRunner) -> None:
        result = _run(runner, "--peak-rps", "20", "--provision-s", "60,300", "--sweep-k", "1,2")
        assert result.exit_code == 0
        for provision in (60, 300):
            for k in (1, 2):
                assert f"provision {provision}s, k={k}" in result.output

    def test_a_non_numeric_sweep_is_a_bad_parameter(self, runner: CliRunner) -> None:
        result = _run(runner, "--peak-rps", "20", "--provision-s", "60,soon")
        assert result.exit_code == 2
        assert "--provision-s must be comma-separated numbers" in result.output

    def test_a_non_positive_sweep_value_is_refused(self, runner: CliRunner) -> None:
        result = _run(runner, "--peak-rps", "20", "--sweep-k", "0,2")
        assert result.exit_code == 2
        assert "--sweep-k values must be positive" in result.output

    def test_a_provision_sweep_without_a_measured_total_is_refused(self, runner: CliRunner) -> None:
        # Nothing to substitute into. Better to say so than to sweep a number that
        # replaces a stage nobody observed.
        with runner.isolated_filesystem():
            measured = _write_cmax()
            result = runner.invoke(
                main,
                [
                    "plan",
                    "--measured",
                    measured,
                    "--peak-rps",
                    "20",
                    "--assume-t-total",
                    "300",
                    "--provision-s",
                    "60",
                ],
                catch_exceptions=False,
            )
        assert result.exit_code == 1
        assert "no measured T_total" in result.output


class TestExitStatusGatesADeploy:
    def test_an_slo_past_the_ceiling_exits_non_zero(self, runner: CliRunner) -> None:
        # W_max is SLO - p95 service, so the request deadline *is* the SLO: a 70s promise
        # cannot be served inside a 60s invocation ceiling whatever the queue does.
        result = _run(runner, "--peak-rps", "20", "--ttfab-slo-ms", "70000")
        assert result.exit_code == 1
        assert "STOP" in result.output
        assert "INFEASIBLE" in result.output

    def test_an_slo_under_the_models_own_tail_exits_non_zero(self, runner: CliRunner) -> None:
        # The other infeasibility, and the one the SLO reframe added: p95 service is
        # 165ms, so a 100ms promise is missed before a request waits at all. No queue
        # depth or fleet size rescues it.
        result = _run(runner, "--peak-rps", "20", "--ttfab-slo-ms", "100")
        assert result.exit_code == 1
        assert "slo_budget" in result.output
        assert "No queue depth, instance count, or scaling policy rescues this" in result.output

    def test_a_feasible_plan_exits_zero_even_with_warnings(self, runner: CliRunner) -> None:
        # WARN findings are the normal case -- an unbracketed knee, a thin sample, a
        # costly reserve. Exiting non-zero on them would make the gate useless.
        result = _run(runner, "--peak-rps", "20", "--sweep-k", "5")
        assert result.exit_code == 0
        assert "WARN" in result.output

    def test_one_infeasible_row_in_a_sweep_is_enough(self, runner: CliRunner) -> None:
        # A sweep where any assumed provision time breaks the SLO is a plan that
        # depends on an assumption nobody has verified.
        result = _run(
            runner, "--peak-rps", "20", "--provision-s", "60,600", "--ttfab-slo-ms", "70000"
        )
        assert result.exit_code == 1

    def test_an_explicit_ceiling_moves_the_gate(self, runner: CliRunner) -> None:
        result = _run(runner, "--peak-rps", "20", "--ttfab-slo-ms", "20000", "--ceiling-s", "10")
        assert result.exit_code == 1


class TestArtifact:
    def test_nothing_is_written_unless_asked(self, runner: CliRunner) -> None:
        # Unlike a measurement, this run is cheap to repeat -- so the default is the
        # opposite of `cmax`'s, where omitting --output would discard 45 minutes.
        with runner.isolated_filesystem():
            measured = _write_cmax()
            ttotal = _write_ttotal()
            result = runner.invoke(
                main,
                ["plan", "--measured", measured, "--ttotal", ttotal, "--peak-rps", "20"],
                catch_exceptions=False,
            )
            assert result.exit_code == 0
            assert not Path("artifacts").exists()

    def test_output_writes_one_json_object_per_row(self, runner: CliRunner) -> None:
        with runner.isolated_filesystem():
            measured = _write_cmax()
            ttotal = _write_ttotal()
            result = runner.invoke(
                main,
                [
                    "plan",
                    "--measured",
                    measured,
                    "--ttotal",
                    ttotal,
                    "--peak-rps",
                    "20",
                    "--provision-s",
                    "60,300",
                    "--output",
                    "plan.json",
                ],
                catch_exceptions=False,
            )
            assert result.exit_code == 0
            rows = json.loads(Path("plan.json").read_text())

        assert len(rows) == 2
        assert [row["provision_s"] for row in rows] == [60.0, 300.0]
        assert all(row["provision_assumed"] for row in rows)
        # The assumption travels with the row. A row read back without it cannot be
        # told from a measurement, which is the distinction the sweep exists for.
        assert [row["t_total_s"] for row in rows] == [300.0, 540.0]
        assert all(row["t_total_measured"] for row in rows)
        assert all("ModelEndpointConfig(" in row["config_block"] for row in rows)

    def test_a_stated_lag_is_recorded_as_stated_in_the_artifact(self, runner: CliRunner) -> None:
        # Not the same question as `provision_assumed`: this row substituted nothing,
        # which happens both when the lag was measured whole and when it was stated
        # whole. Without the flag the two are indistinguishable on replay.
        with runner.isolated_filesystem():
            measured = _write_cmax()
            result = runner.invoke(
                main,
                [
                    "plan",
                    "--measured",
                    measured,
                    "--peak-rps",
                    "20",
                    "--assume-t-total",
                    "300",
                    "--output",
                    "plan.json",
                ],
                catch_exceptions=False,
            )
            assert result.exit_code == 0
            rows = json.loads(Path("plan.json").read_text())

        assert not rows[0]["provision_assumed"]
        assert not rows[0]["t_total_measured"]

    def test_the_written_verdict_matches_the_exit_status(self, runner: CliRunner) -> None:
        with runner.isolated_filesystem():
            measured = _write_cmax()
            ttotal = _write_ttotal()
            result = runner.invoke(
                main,
                [
                    "plan",
                    "--measured",
                    measured,
                    "--ttotal",
                    ttotal,
                    "--peak-rps",
                    "20",
                    "--ttfab-slo-ms",
                    "70000",
                    "--output",
                    "plan.json",
                ],
                catch_exceptions=False,
            )
            rows = json.loads(Path("plan.json").read_text())

        assert result.exit_code == 1
        # Written before the non-zero exit: an artifact that exists only for passing
        # runs is useless for diagnosing a failing gate.
        assert rows[0]["verdict"] == "INFEASIBLE"


class TestTouchesNoAws:
    def test_no_boto_client_is_constructed(self, runner: CliRunner, monkeypatch) -> None:
        # Worth pinning rather than assuming: `plan` is the command that can run on a
        # laptop with no credentials, and the cost lookup is the plausible place a
        # future change would reach for the Pricing API.
        import boto3

        def refuse(*args, **kwargs):
            raise AssertionError(f"plan constructed a boto3 client: {args} {kwargs}")

        monkeypatch.setattr(boto3, "client", refuse)
        result = _run(runner, "--peak-rps", "20")
        assert result.exit_code == 0

    def test_no_region_option_is_offered(self) -> None:
        # Every other command takes --region. Offering one here would imply an API
        # call this command does not make.
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
            if isinstance(param, __import__("click").Option) and not param.help
        ]
        assert undocumented == []

    def test_the_defaults_match_the_deployed_policy(self) -> None:
        # These three are the ones a reader will assume rather than pass, so they
        # should reproduce the shipped config rather than something arbitrary.
        assert _param("derate").default == 0.875
        assert _param("growth_factor_k").default == 2.0
        assert _param("min_instances_floor").default == 1
