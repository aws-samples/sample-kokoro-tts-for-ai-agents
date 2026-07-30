"""Tests for the `ttotal` CLI command.

`ttotal` is the one command in this package that both *sends load* and *mutates an
endpoint's desired instance count*, so the tests here are mostly about what happens
before either of those. Three concerns:

- **Every required input is resolved or refused, never guessed.** `S` converts a target
  concurrency into an arrival rate and the TTFAB budget defines what "recovered" means;
  a wrong `S` offers the wrong load and a missing budget makes recovery undefinable. So
  each is read from a `cmax` artifact, or from config, or the run is refused with the
  reason — and `--load-multiple <= 1` is refused outright, because load that never
  crosses `C_target` cannot cause a scale-out.
- **`C_target` comes off the live policy.** Reading it from `speech_infra.config` would
  let a config that has moved ahead of the last `cdk deploy` drive load past the wrong
  threshold and mis-attribute the whole detection stage.
- **A `force-desired` result is labelled as half a measurement**, on stdout and in the
  artifact both. It skips the metric and alarm stages entirely, and the failure mode —
  planning a surge against a container-only figure — is silent.

`main.commands["ttotal"]` is inspected directly for the flag defaults, because `cli.py`
imports `ttotal.py` lazily (numpy and botocore stay out of `--help`) and so restates
the trigger strings as literals.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from tts_bench import ttotal as ttotal_mod
from tts_bench.cli import (
    TTOTAL_TRIGGER_DRIVE_LOAD,
    TTOTAL_TRIGGER_FORCE_DESIRED,
    _config_ttfab_budget_ms,
    main,
)
from tts_bench.fixture import FixtureError
from tts_bench.ttotal import StageTime, TimelineStage, TTotalError, TTotalReport

MODEL = "kokoro-82m"
ENDPOINT = "speech-kokoro-82m"
T0 = datetime(2026, 7, 30, 11, 0, 0, tzinfo=UTC)

#: Measured on this endpoint, so a smoke run against it needs no flags beyond --model.
DEPLOYED_TARGET = 0.713
S_MEAN_S = 0.10986375146305409


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def _param(name: str):
    """The `ttotal` command's declared parameter, by its Python name."""
    for param in main.commands["ttotal"].params:
        if param.name == name:
            return param
    raise AssertionError(f"ttotal has no parameter {name}")


def _report(**overrides) -> TTotalReport:
    report = TTotalReport(
        model_name=MODEL,
        endpoint=ENDPOINT,
        run_id="run123",
        trigger=TTOTAL_TRIGGER_DRIVE_LOAD,
        from_instances=1,
        to_instances=2,
        instance_id="i-0abc123def4567890",
        p95_before_ms=1200.0,
        p95_after_ms=140.0,
        requests_before=300,
        requests_after=120,
    )
    report.timeline = [
        StageTime(str(TimelineStage.LOAD_APPLIED), T0, source="load generator start"),
        StageTime(str(TimelineStage.METRIC_PUBLISHED), T0 + timedelta(seconds=20)),
        StageTime(str(TimelineStage.ALARM_FIRED), T0 + timedelta(seconds=50)),
        StageTime(str(TimelineStage.ACTIVITY_STARTED), T0 + timedelta(seconds=55)),
        StageTime(str(TimelineStage.INSTANCE_LOGGING), T0 + timedelta(seconds=180), bounded=True),
        StageTime(str(TimelineStage.READY), T0 + timedelta(seconds=190)),
        StageTime(str(TimelineStage.IN_SERVICE), T0 + timedelta(seconds=240)),
        StageTime(str(TimelineStage.TRAFFIC_RECOVERED), T0 + timedelta(seconds=300), bounded=True),
    ]
    for key, value in overrides.items():
        setattr(report, key, value)
    return report


def _artifact(tmp_path, **overrides) -> str:
    """A minimal `cmax` artifact, shaped like the committed one."""
    payload = {
        "model_name": MODEL,
        "endpoint": ENDPOINT,
        "s_mean_s": S_MEAN_S,
        "c_max_curve": {"300": 1.6302521008403361, "500": 1.6302521008403361},
        "transport": "bidi",
    }
    payload.update(overrides)
    path = tmp_path / "cmax.json"
    path.write_text(json.dumps(payload))
    return str(path)


def _run(
    runner: CliRunner,
    *args: str,
    model: str = MODEL,
    report: TTotalReport | None = None,
    raises=None,
    target_value: float | None = DEPLOYED_TARGET,
):
    """Invoke `ttotal` with `measure` and the policy read replaced.

    `boto3.client` is left real but never reached for credentials: `measure` is the only
    thing that would use the clients, and it is patched. `deployed_target_value` is
    patched because it is the one AWS read the command makes *itself*, before deciding
    whether a run is even possible.
    """
    calls: list[dict] = []

    def fake_measure(**kwargs) -> TTotalReport:
        calls.append(kwargs)
        if raises is not None:
            raise raises
        return report if report is not None else _report()

    with (
        patch.object(ttotal_mod, "measure", fake_measure),
        patch.object(ttotal_mod, "deployed_target_value", return_value=target_value),
        patch("boto3.client"),
    ):
        result = runner.invoke(main, ["ttotal", "--model", model, *args])
    return result, calls


class TestTriggerLiteralsMatchTheModule:
    """The duplicated trigger strings against `ttotal.py`'s own.

    `click.Choice` is evaluated at import time, so referencing the module in the
    decorator would defeat the lazy import that keeps numpy out of `--help`. This is
    what makes restating them safe: renaming a trigger in one place fails here.
    """

    def test_drive_load(self) -> None:
        assert TTOTAL_TRIGGER_DRIVE_LOAD == ttotal_mod.TRIGGER_DRIVE_LOAD

    def test_force_desired(self) -> None:
        assert TTOTAL_TRIGGER_FORCE_DESIRED == ttotal_mod.TRIGGER_FORCE_DESIRED

    def test_the_choices_are_exactly_the_two_modes(self) -> None:
        assert list(_param("trigger").type.choices) == [
            ttotal_mod.TRIGGER_DRIVE_LOAD,
            ttotal_mod.TRIGGER_FORCE_DESIRED,
        ]

    def test_drive_load_is_the_default(self) -> None:
        # The mode that measures both halves. force-desired has to be asked for.
        assert _param("trigger").default == ttotal_mod.TRIGGER_DRIVE_LOAD

    @pytest.mark.parametrize(
        ("cli_name", "measure_name"),
        [
            ("load_multiple", "load_multiple"),
            ("max_wait_s", "max_wait_s"),
            ("settle_s", "settle_s"),
            ("poll_interval_s", "poll_interval_s"),
            ("region", "region"),
            ("variant", "variant"),
        ],
    )
    def test_shared_defaults_match_measure(self, cli_name: str, measure_name: str) -> None:
        import inspect

        expected = inspect.signature(ttotal_mod.measure).parameters[measure_name].default
        assert _param(cli_name).default == expected

    def test_help_does_not_import_ttotals_dependencies(self, runner: CliRunner) -> None:
        # The lazy-import convention the literals exist to preserve.
        result = runner.invoke(main, ["ttotal", "--help"])
        assert result.exit_code == 0
        assert "T_total" in result.output


class TestResolvingS:
    def test_read_from_a_cmax_artifact(self, runner: CliRunner, tmp_path) -> None:
        result, calls = _run(runner, "--measured", _artifact(tmp_path))
        assert result.exit_code == 0
        assert calls[0]["s_mean_s"] == pytest.approx(S_MEAN_S)

    def test_an_explicit_flag_wins_over_the_artifact(self, runner: CliRunner, tmp_path) -> None:
        result, calls = _run(runner, "--measured", _artifact(tmp_path), "--s-mean", "0.25")
        assert result.exit_code == 0
        assert calls[0]["s_mean_s"] == pytest.approx(0.25)

    def test_refused_when_neither_is_supplied(self, runner: CliRunner) -> None:
        # Not defaulted: an S guessed wrong offers the wrong rate, and the run then
        # measures a scale-out that either never fires or fires for the wrong reason.
        result, calls = _run(runner)
        assert result.exit_code != 0
        assert "--s-mean is required" in result.output
        assert calls == []

    def test_a_nonpositive_s_is_rejected(self, runner: CliRunner) -> None:
        result, calls = _run(runner, "--s-mean", "0")
        assert result.exit_code != 0
        assert calls == []

    def test_an_artifact_without_s_still_refuses(self, runner: CliRunner, tmp_path) -> None:
        path = _artifact(tmp_path)
        payload = json.loads(open(path).read())
        del payload["s_mean_s"]
        (tmp_path / "cmax.json").write_text(json.dumps(payload))

        result, _ = _run(runner, "--measured", path)
        assert result.exit_code != 0
        assert "--s-mean is required" in result.output


class TestResolvingTheBudget:
    def test_takes_the_loosest_budget_in_the_curve(self, runner: CliRunner, tmp_path) -> None:
        # The loosest SLO the C_max run measured, so recovery is judged against a bound
        # the model was actually shown to meet.
        result, calls = _run(runner, "--measured", _artifact(tmp_path))
        assert result.exit_code == 0
        assert calls[0]["ttfab_budget_ms"] == pytest.approx(500.0)

    def test_an_explicit_flag_wins(self, runner: CliRunner, tmp_path) -> None:
        result, calls = _run(runner, "--measured", _artifact(tmp_path), "--ttfab-budget-ms", "250")
        assert result.exit_code == 0
        assert calls[0]["ttfab_budget_ms"] == pytest.approx(250.0)

    def test_falls_back_to_config(self, runner: CliRunner) -> None:
        result, calls = _run(runner, "--s-mean", "0.11")
        assert result.exit_code == 0
        assert calls[0]["ttfab_budget_ms"] == pytest.approx(300.0)

    def test_refused_when_nothing_supplies_one(self, runner: CliRunner) -> None:
        with patch("tts_bench.cli._config_ttfab_budget_ms", return_value=None):
            result, calls = _run(runner, "--s-mean", "0.11")
        assert result.exit_code != 0
        assert "no recovery without one" in result.output
        assert calls == []

    def test_config_lookup_is_by_endpoint(self) -> None:
        assert _config_ttfab_budget_ms(ENDPOINT) == pytest.approx(300.0)
        assert _config_ttfab_budget_ms("not-an-endpoint") is None


class TestResolvingCTarget:
    def test_read_off_the_deployed_policy(self, runner: CliRunner) -> None:
        # Not from config: a config ahead of the last deploy would drive load past a
        # threshold the policy is not using and mis-attribute the detection stage.
        result, calls = _run(runner, "--s-mean", "0.11")
        assert result.exit_code == 0
        assert calls[0]["scaling_target"] == pytest.approx(DEPLOYED_TARGET)

    def test_an_explicit_flag_skips_the_read(self, runner: CliRunner) -> None:
        with (
            patch.object(ttotal_mod, "measure", lambda **kw: _report()),
            patch.object(
                ttotal_mod, "deployed_target_value", side_effect=AssertionError("read anyway")
            ),
            patch("boto3.client"),
        ):
            result = runner.invoke(
                main,
                ["ttotal", "--model", MODEL, "--s-mean", "0.11", "--scaling-target", "2.0"],
            )
        assert result.exit_code == 0

    def test_drive_load_without_a_policy_is_refused(self, runner: CliRunner) -> None:
        # There is nothing to drive load *past*, so the run could only measure a
        # scale-out that never happens.
        result, calls = _run(runner, "--s-mean", "0.11", target_value=None)
        assert result.exit_code != 0
        assert "no target-tracking policy" in result.output
        assert calls == []

    def test_force_desired_without_a_policy_still_runs(self, runner: CliRunner) -> None:
        result, calls = _run(
            runner,
            "--s-mean",
            "0.11",
            "--trigger",
            TTOTAL_TRIGGER_FORCE_DESIRED,
            target_value=None,
        )
        assert result.exit_code == 0
        assert calls[0]["trigger"] == TTOTAL_TRIGGER_FORCE_DESIRED

    def test_the_echo_states_what_it_will_offer(self, runner: CliRunner) -> None:
        result, _ = _run(runner, "--s-mean", "0.11", "--load-multiple", "3")
        assert "C_target=0.713" in result.output
        assert "offering 2.14 concurrency" in result.output


class TestLoadMultiple:
    def test_at_or_below_one_is_refused(self, runner: CliRunner) -> None:
        # A multiple of 1 offers exactly C_target, which the policy is designed to hold
        # without scaling. The run would wait out its whole timeout for nothing.
        for value in ("1.0", "0.5"):
            result, calls = _run(runner, "--s-mean", "0.11", "--load-multiple", value)
            assert result.exit_code != 0, value
            assert "must exceed 1.0" in result.output
            assert calls == []

    def test_above_one_is_passed_through(self, runner: CliRunner) -> None:
        result, calls = _run(runner, "--s-mean", "0.11", "--load-multiple", "2.5")
        assert result.exit_code == 0
        assert calls[0]["load_multiple"] == pytest.approx(2.5)


class TestTransport:
    def test_defaults_to_bidi(self, runner: CliRunner) -> None:
        # The transport C_max was measured on, and the one production uses.
        result, calls = _run(runner, "--s-mean", "0.11")
        assert result.exit_code == 0
        assert calls[0]["transport"] == "bidi"

    def test_a_mismatched_artifact_transport_warns(self, runner: CliRunner, tmp_path) -> None:
        # S differs per transport, so the offered rate would be wrong — a warning rather
        # than a refusal, because the operator may be deliberately comparing the two.
        result, calls = _run(
            runner,
            "--measured",
            _artifact(tmp_path, transport="response-stream"),
            "--transport",
            "bidi",
        )
        assert result.exit_code == 0
        assert "WARNING" in result.output
        assert "response-stream" in result.output
        assert calls[0]["transport"] == "bidi"

    def test_a_matching_transport_does_not_warn(self, runner: CliRunner, tmp_path) -> None:
        result, _ = _run(runner, "--measured", _artifact(tmp_path), "--transport", "bidi")
        assert "WARNING" not in result.output

    def test_an_unknown_transport_is_rejected(self, runner: CliRunner) -> None:
        result, calls = _run(runner, "--s-mean", "0.11", "--transport", "grpc")
        assert result.exit_code != 0
        assert calls == []


class TestForceDesiredIsLabelled:
    def test_stdout_says_it_is_a_lower_bound(self, runner: CliRunner) -> None:
        result, _ = _run(runner, "--s-mean", "0.11", "--trigger", TTOTAL_TRIGGER_FORCE_DESIRED)
        assert "container half only" in result.output
        assert "not T_total" in result.output

    def test_drive_load_gets_no_such_note(self, runner: CliRunner) -> None:
        result, _ = _run(runner, "--s-mean", "0.11")
        assert "container half only" not in result.output

    def test_the_artifact_carries_the_caveat_too(self, runner: CliRunner, tmp_path) -> None:
        # stdout scrolls away; the JSON is what a later `plan` run reads.
        out = tmp_path / "ttotal.json"
        report = _report(trigger=TTOTAL_TRIGGER_FORCE_DESIRED)
        _run(
            runner,
            "--s-mean",
            "0.11",
            "--trigger",
            TTOTAL_TRIGGER_FORCE_DESIRED,
            "--output",
            str(out),
            report=report,
        )
        payload = json.loads(out.read_text())
        assert payload["trigger"] == TTOTAL_TRIGGER_FORCE_DESIRED


class TestOutput:
    def test_prints_the_timeline(self, runner: CliRunner) -> None:
        result, _ = _run(runner, "--s-mean", "0.11")
        assert "T_total for speech-kokoro-82m" in result.output
        assert "300.0s from load applied to traffic recovered" in result.output
        assert "stage breakdown" in result.output

    def test_names_the_new_instance(self, runner: CliRunner) -> None:
        result, _ = _run(runner, "--s-mean", "0.11")
        assert "i-0abc123def4567890" in result.output

    def test_writes_a_readable_artifact(self, runner: CliRunner, tmp_path) -> None:
        out = tmp_path / "ttotal.json"
        result, _ = _run(runner, "--s-mean", "0.11", "--output", str(out))
        assert result.exit_code == 0

        payload = json.loads(out.read_text())
        assert payload["t_total_s"] == pytest.approx(300.0)
        assert payload["aws_share_s"] == pytest.approx(180.0)
        assert payload["endpoint"] == ENDPOINT
        assert len(payload["timeline"]) == 8

    def test_points_at_the_next_command_with_the_measured_t_total(self, runner: CliRunner) -> None:
        result, _ = _run(runner, "--s-mean", "0.11")
        assert "tts-bench plan --model kokoro-82m --t-total 300" in result.output

    def test_a_report_with_no_total_still_prints(self, runner: CliRunner) -> None:
        # Every API came back empty. Worth seeing rather than crashing on a None format.
        result, _ = _run(runner, "--s-mean", "0.11", report=_report(timeline=[]))
        assert result.exit_code == 0
        assert "not measurable" in result.output

    def test_events_are_written_when_asked(self, runner: CliRunner, tmp_path) -> None:
        out = tmp_path / "events.jsonl"
        result, calls = _run(runner, "--s-mean", "0.11", "--events", str(out))
        assert result.exit_code == 0
        assert calls[0]["event_sink"] is not None
        assert out.exists()

    def test_no_event_sink_without_the_flag(self, runner: CliRunner) -> None:
        _, calls = _run(runner, "--s-mean", "0.11")
        assert calls[0]["event_sink"] is None

    def test_the_writer_closes_even_when_measure_raises(self, runner: CliRunner, tmp_path) -> None:
        out = tmp_path / "events.jsonl"
        result, _ = _run(
            runner, "--s-mean", "0.11", "--events", str(out), raises=TTotalError("no scale-out")
        )
        assert result.exit_code != 0
        assert out.exists()


class TestRefusals:
    def test_a_ttotal_error_is_a_click_exception(self, runner: CliRunner) -> None:
        # The tool declining to report a lag it did not observe. A traceback would read
        # as a bug rather than as the guard working.
        result, _ = _run(
            runner, "--s-mean", "0.11", raises=TTotalError("did not reach more than 1 instance")
        )
        assert result.exit_code != 0
        assert "did not reach more than 1 instance" in result.output
        assert "Traceback" not in result.output

    def test_a_fixture_error_is_a_click_exception(self, runner: CliRunner) -> None:
        result, _ = _run(
            runner, "--s-mean", "0.11", raises=FixtureError("max_capacity is 1, cannot scale")
        )
        assert result.exit_code != 0
        assert "cannot scale" in result.output
        assert "Traceback" not in result.output

    def test_an_unexpected_error_is_not_swallowed(self, runner: CliRunner) -> None:
        result, _ = _run(runner, "--s-mean", "0.11", raises=RuntimeError("boom"))
        assert result.exit_code != 0
        assert isinstance(result.exception, RuntimeError)

    def test_a_model_without_an_endpoint_is_refused_before_any_aws_read(
        self, runner: CliRunner
    ) -> None:
        with (
            patch.object(
                ttotal_mod, "deployed_target_value", side_effect=AssertionError("read anyway")
            ),
            patch("boto3.client", side_effect=AssertionError("client built anyway")),
        ):
            result = runner.invoke(main, ["ttotal", "--model", "nonexistent-model"])
        assert result.exit_code != 0
        assert "Traceback" not in result.output

    def test_model_is_required(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["ttotal"])
        assert result.exit_code != 0
        assert "--model" in result.output
