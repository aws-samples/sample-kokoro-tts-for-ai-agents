"""Tests for the `cmax` CLI command.

`cmax.py` imports lazily — `cli.py` would otherwise pull numpy and botocore into
`--help` — so the flag defaults are written as literals beside the module
constants they mirror. `TestDefaultsMatchTheModule` is what makes that
duplication safe: it compares every shared default against
`inspect.signature(cmax.measure)`, so raising `DEFAULT_HOLD_S` without touching
the decorator fails here rather than silently running a shorter ladder than the
module documents.

The other three concerns are the ones an operator cannot check by reading the
output:

- **`--dry-run` really is dry.** `boto3.client` is patched to raise, so a dry run
  that constructed any client fails. This is the flag people use to size a run
  before spending money on it, and it advertises "no AWS calls made".
- **Refusals exit cleanly.** `CMaxError` and `FixtureError` are the tool
  declining to produce a number it cannot stand behind. A traceback would read
  as a bug in the tool rather than as the guard doing its job.
- **A curve the harness marked untrustworthy says so on stdout.** The artifact
  carries `frozen` and `instance_counts_observed`, but nobody opens the JSON
  before reading the table.
"""

from __future__ import annotations

import inspect
import json
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from tts_bench import cmax as cmax_mod
from tts_bench.cli import (
    CMAX_BUDGETS_DEFAULT,
    CMAX_LADDER_DEFAULT,
    _load_texts,
    _parse_floats,
    _parse_ints,
    main,
)
from tts_bench.fixture import FixtureError
from tts_bench.types import CMaxReport, KneePoint
from tts_inference.types import TTSModelName

MODEL = "kokoro-82m"
ENDPOINT = "speech-kokoro-82m"

#: A ladder small enough that the dry-run table can be asserted line by line.
SMALL_LADDER = "1,2"


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def _param(name: str):
    """The `cmax` command's declared parameter, by its Python name."""
    command = main.commands["cmax"]
    for param in command.params:
        if param.name == name:
            return param
    raise AssertionError(f"cmax has no parameter {name}")


def _knee(budget: int, concurrency: float, *, bracketed: bool = True) -> KneePoint:
    return KneePoint(
        ttfab_budget_ms=budget,
        concurrency=concurrency,
        offered_rps=concurrency / 0.25,
        p95_ttfab_ms=budget - 20.0,
        step_index=0,
        bracketed=bracketed,
    )


def _report(**kwargs) -> CMaxReport:
    defaults = {
        "model_name": TTSModelName.KOKORO_82M,
        "endpoint": ENDPOINT,
        "instance_type": "ml.g5.xlarge",
        "run_id": "abc123def456",
        "c_max_curve": {150: 1.0, 300: 2.0},
        "knees": [_knee(150, 1.0), _knee(300, 2.0)],
        "curve_spread": {150: 0.04, 300: 0.09},
        "s_mean_s": 0.25,
        "s_p95_s": 0.41,
        "frozen": True,
        "instance_counts_observed": (1,),
        "hold_s": 240.0,
        "measure_window_s": 60.0,
    }
    defaults.update(kwargs)
    return CMaxReport(**defaults)


def _run(
    runner: CliRunner,
    *args: str,
    model: str = MODEL,
    report: CMaxReport | None = None,
    raises=None,
):
    """Invoke `cmax` with `measure` replaced and every AWS client poisoned.

    Patching `boto3.client` rather than trusting the stub is deliberate: it is
    the assertion that no code path reached for a client on its own.
    """
    calls: list[dict] = []

    def fake_measure(**kwargs) -> CMaxReport:
        calls.append(kwargs)
        if raises is not None:
            raise raises
        return report if report is not None else _report()

    def no_clients(name: str, **_):
        raise AssertionError(f"an AWS client was constructed: {name}")

    with (
        patch.object(cmax_mod, "measure", fake_measure),
        patch("boto3.client", side_effect=no_clients),
    ):
        result = runner.invoke(main, ["cmax", "--model", model, *args])
    return result, calls


class TestDefaultsMatchTheModule:
    """The literal decorator defaults against `cmax`'s own.

    Without this the two drift apart quietly: the CLI would keep advertising a
    240s hold in `--help` while the module documented something else, and the
    artifact would record whichever one happened to win.
    """

    #: CLI parameter name -> `cmax.measure` parameter name. Same name for all of
    #: them except the ladder, which reads better singular as a flag.
    SHARED = {
        "hold_s": "hold_s",
        "measure_window_s": "measure_window_s",
        "settle_between_steps_s": "settle_between_steps_s",
        "runs": "runs",
        "derate": "derate",
        "seed": "seed",
        "pin_to": "pin_to",
        "require_frozen": "require_frozen",
        "cloudwatch_join": "cloudwatch_join",
        "region": "region",
        "variant": "variant",
        "arrival": "arrival",
    }

    @pytest.mark.parametrize(("cli_name", "measure_name"), sorted(SHARED.items()))
    def test_shared_default(self, cli_name: str, measure_name: str) -> None:
        expected = inspect.signature(cmax_mod.measure).parameters[measure_name].default
        assert _param(cli_name).default == expected, (
            f"--{cli_name.replace('_', '-')} default drifted from cmax.measure({measure_name}=)"
        )

    def test_the_ladder_literal_parses_to_the_module_constant(self) -> None:
        assert (
            _parse_floats(CMAX_LADDER_DEFAULT, flag="--target-concurrency")
            == cmax_mod.DEFAULT_TARGET_CONCURRENCIES
        )

    def test_the_budget_literal_parses_to_the_module_constant(self) -> None:
        assert (
            _parse_ints(CMAX_BUDGETS_DEFAULT, flag="--ttfab-budgets")
            == cmax_mod.DEFAULT_TTFAB_BUDGETS
        )

    def test_the_hold_literal_is_the_module_constant(self) -> None:
        # Named separately from the signature sweep because these three are the
        # ones that decide how long a run takes and what it costs.
        assert _param("hold_s").default == cmax_mod.DEFAULT_HOLD_S
        assert _param("measure_window_s").default == cmax_mod.DEFAULT_MEASURE_WINDOW_S
        assert _param("settle_between_steps_s").default == cmax_mod.DEFAULT_SETTLE_BETWEEN_STEPS_S

    def test_help_does_not_import_cmax(self) -> None:
        # The reason the literals exist at all. `--help` must stay instant, and
        # this asserts the lazy import rather than the comment claiming it.
        source = inspect.getsource(main.commands["cmax"].callback)
        assert "from tts_bench import cmax as cmax_mod" in source


class TestDryRun:
    def test_makes_no_aws_calls(self, runner: CliRunner) -> None:
        # `boto3.client` raises in `_run`, so reaching exit code 0 *is* the
        # assertion. The flag's own output promises this.
        result, calls = _run(runner, "--dry-run", "--target-concurrency", SMALL_LADDER)
        assert result.exit_code == 0, result.output
        assert calls == []
        assert "no AWS calls made" in result.output

    def test_does_not_freeze_anything(self, runner: CliRunner) -> None:
        result, _ = _run(runner, "--dry-run", "--target-concurrency", SMALL_LADDER)
        assert "nothing frozen" in result.output

    def test_one_row_per_step_per_run(self, runner: CliRunner) -> None:
        result, _ = _run(runner, "--dry-run", "--target-concurrency", SMALL_LADDER, "--runs", "3")
        rows = [
            line
            for line in result.output.splitlines()
            if line.strip().startswith(("0 ", "1 ", "2 "))
        ]
        assert len(rows) == 6

    def test_shows_the_request_count_and_wall_clock(self, runner: CliRunner) -> None:
        # The question a dry run answers: is this run worth the money and the
        # 45 minutes. Both numbers have to be on screen.
        result, _ = _run(
            runner,
            "--dry-run",
            "--target-concurrency",
            "1",
            "--hold",
            "240",
            "--dry-run-s",
            "0.25",
        )
        # C=1 at S=0.25s is 4 rps, so 240s offers 960 requests.
        assert "960" in result.output
        assert "Estimated wall clock: 4 min" in result.output

    def test_names_the_endpoint_it_would_measure(self, runner: CliRunner) -> None:
        # The model name is an alias; the endpoint is the thing that gets frozen.
        # Seeing it here is how an operator confirms they are about to freeze the
        # endpoint they meant.
        result, _ = _run(runner, "--dry-run", "--target-concurrency", SMALL_LADDER)
        assert ENDPOINT in result.output

    def test_says_that_s_was_assumed(self, runner: CliRunner) -> None:
        # A dry run cannot measure service time, and the whole ladder is scaled
        # by it. An unlabelled S here would be the one assumption in the tool
        # that looks measured.
        result, _ = _run(runner, "--dry-run", "--dry-run-s", "0.5")
        assert "S=0.500s" in result.output
        assert "not measured" in result.output

    def test_lists_the_budgets_it_would_report(self, runner: CliRunner) -> None:
        result, _ = _run(runner, "--dry-run", "--ttfab-budgets", "200,400")
        assert "[200, 400]" in result.output

    def test_needs_no_sample_dataset(self, runner: CliRunner) -> None:
        # Texts are loaded after the dry-run return, so a dry run works on a
        # checkout with no data/ directory.
        with patch("tts_bench.cli._load_texts", side_effect=AssertionError("loaded texts")):
            result, _ = _run(runner, "--dry-run", "--target-concurrency", SMALL_LADDER)
        assert result.exit_code == 0, result.output


class TestArgumentValidation:
    def test_a_window_larger_than_the_hold_is_rejected(self, runner: CliRunner) -> None:
        # The window is the trailing part of a step, not an addition to it, so
        # this is a misunderstanding worth naming rather than silently clamping.
        result, calls = _run(runner, "--hold", "60", "--measure-window", "120")
        assert result.exit_code == 2
        assert "cannot exceed" in result.output
        assert calls == []

    def test_a_window_equal_to_the_hold_is_allowed(self, runner: CliRunner) -> None:
        # Degenerate but valid: it means "measure the whole step, no warm-up".
        result, calls = _run(runner, "--hold", "60", "--measure-window", "60")
        assert result.exit_code == 0, result.output
        assert calls[0]["measure_window_s"] == pytest.approx(60.0)

    def test_zero_runs_is_rejected(self, runner: CliRunner) -> None:
        result, calls = _run(runner, "--runs", "0")
        assert result.exit_code == 2
        assert "at least 1" in result.output
        assert calls == []

    @pytest.mark.parametrize("bad", ["", ",", "abc", "0,1", "-1,2", "1 2"])
    def test_a_bad_ladder_is_rejected(self, runner: CliRunner, bad: str) -> None:
        result, calls = _run(runner, "--target-concurrency", bad)
        assert result.exit_code == 2
        assert calls == []

    @pytest.mark.parametrize("sloppy", ["1,2,", " 1 , 2 ", "1,,2"])
    def test_stray_commas_and_spaces_are_tolerated(self, runner: CliRunner, sloppy: str) -> None:
        # An empty field carries no value, so skipping it drops nothing the
        # operator meant. Rejecting a trailing comma would only be pedantry.
        result, calls = _run(runner, "--target-concurrency", sloppy)
        assert result.exit_code == 0, result.output
        assert calls[0]["target_concurrencies"] == (1.0, 2.0)

    @pytest.mark.parametrize("bad", ["", "abc", "0,300", "-5", "1.5,300"])
    def test_a_bad_budget_list_is_rejected(self, runner: CliRunner, bad: str) -> None:
        # 1.5 included on purpose: budgets are integer ms, and a float here would
        # miss every `c_max_curve` lookup after the JSON round trip.
        result, calls = _run(runner, "--ttfab-budgets", bad)
        assert result.exit_code == 2
        assert calls == []

    def test_validation_runs_before_any_load(self, runner: CliRunner) -> None:
        # Both checks are cheap and the run they guard is 45 minutes long.
        result, calls = _run(runner, "--hold", "60", "--measure-window", "120", "--runs", "0")
        assert result.exit_code == 2
        assert calls == []

    def test_an_unknown_arrival_process_is_rejected(self, runner: CliRunner) -> None:
        result, calls = _run(runner, "--arrival", "gaussian")
        assert result.exit_code == 2
        assert calls == []


class TestOptionsReachMeasure:
    def test_parsed_ladder_and_budgets_are_passed_through(self, runner: CliRunner) -> None:
        result, calls = _run(
            runner, "--target-concurrency", "0.5,1,4", "--ttfab-budgets", "150,300"
        )
        assert result.exit_code == 0, result.output
        assert calls[0]["target_concurrencies"] == (0.5, 1.0, 4.0)
        assert calls[0]["budgets"] == (150, 300)

    def test_scalars_are_passed_through(self, runner: CliRunner) -> None:
        result, calls = _run(
            runner,
            "--region",
            "us-west-2",
            "--variant",
            "canary",
            "--voice",
            "af_bella",
            "--hold",
            "120",
            "--measure-window",
            "30",
            "--settle-between-steps",
            "5",
            "--runs",
            "2",
            "--derate",
            "0.9",
            "--arrival",
            "fixed",
            "--seed",
            "7",
            "--pin-to",
            "2",
        )
        assert result.exit_code == 0, result.output
        kwargs = calls[0]
        assert kwargs["region"] == "us-west-2"
        assert kwargs["variant"] == "canary"
        assert kwargs["voice"] == "af_bella"
        assert kwargs["hold_s"] == pytest.approx(120.0)
        assert kwargs["measure_window_s"] == pytest.approx(30.0)
        assert kwargs["settle_between_steps_s"] == pytest.approx(5.0)
        assert kwargs["runs"] == 2
        assert kwargs["derate"] == pytest.approx(0.9)
        assert kwargs["arrival"] == "fixed"
        assert kwargs["seed"] == 7
        assert kwargs["pin_to"] == 2

    def test_the_freeze_is_on_by_default(self, runner: CliRunner) -> None:
        # The guard the whole measurement rests on, so its default is worth an
        # explicit test rather than only the signature sweep.
        _, calls = _run(runner)
        assert calls[0]["require_frozen"] is True

    def test_no_require_frozen_warns_before_sending_load(self, runner: CliRunner) -> None:
        result, calls = _run(runner, "--no-require-frozen")
        assert calls[0]["require_frozen"] is False
        assert "WARNING" in result.output
        assert "N x C_max" in result.output

    def test_cloudwatch_join_can_be_turned_off(self, runner: CliRunner) -> None:
        _, calls = _run(runner, "--no-cloudwatch")
        assert calls[0]["cloudwatch_join"] is False

    def test_texts_are_loaded_and_capped(self, runner: CliRunner) -> None:
        _, calls = _run(runner, "--max-samples", "5")
        assert len(calls[0]["texts"]) == 5

    def test_no_event_sink_without_the_flag(self, runner: CliRunner) -> None:
        _, calls = _run(runner)
        assert calls[0]["event_sink"] is None


class TestTransportFlag:
    def test_defaults_to_response_stream(self, runner: CliRunner) -> None:
        # Every C_max measured so far came from this path; changing the default
        # would silently make new runs incomparable with the existing artifacts.
        _, calls = _run(runner)
        assert calls[0]["transport"] == "response-stream"

    def test_bidi_reaches_measure(self, runner: CliRunner) -> None:
        _, calls = _run(runner, "--transport", "bidi")
        assert calls[0]["transport"] == "bidi"

    def test_an_unknown_transport_is_rejected_before_any_load(self, runner: CliRunner) -> None:
        result, calls = _run(runner, "--transport", "websocket")
        assert result.exit_code == 2
        assert calls == []

    def test_the_choices_match_the_transport_enum(self) -> None:
        # The flag hands its raw string to Transport(); a member added to the
        # enum without the flag would be unreachable from the CLI.
        from tts_bench.bidi import Transport

        assert set(_param("transport").type.choices) == {t.value for t in Transport}

    def test_the_pre_run_echo_names_the_transport(self, runner: CliRunner) -> None:
        # Printed before the ladder starts: a 45-minute run on the wrong protocol
        # should be catchable in the first second, not from the artifact.
        result, _ = _run(runner, "--transport", "bidi")
        assert "transport=bidi" in result.output


class TestEventsAndArtifact:
    def test_events_flag_opens_a_writer_for_the_run(self, runner: CliRunner, tmp_path) -> None:
        path = tmp_path / "events.jsonl"
        result, calls = _run(runner, "--events", str(path))
        assert result.exit_code == 0, result.output
        # A live sink, not a path: the writer flushes per event so a run killed
        # mid-step has still written the interesting part.
        assert callable(calls[0]["event_sink"])
        assert path.exists()
        assert str(path) in result.output

    def test_the_writer_closes_even_when_measure_raises(self, runner: CliRunner, tmp_path) -> None:
        path = tmp_path / "events.jsonl"
        result, calls = _run(
            runner, "--events", str(path), raises=cmax_mod.CMaxError("probe failed")
        )
        assert result.exit_code == 1
        # A closed JsonlWriter has dropped its handle and refuses to write, so
        # the guard fires before the event is ever serialized — which is why any
        # object serves as the probe here.
        sink = calls[0]["event_sink"]
        with pytest.raises(RuntimeError, match="outside its context manager"):
            sink(object())

    def test_output_writes_a_readable_artifact(self, runner: CliRunner, tmp_path) -> None:
        path = tmp_path / "cmax.json"
        result, _ = _run(runner, "--output", str(path))
        assert result.exit_code == 0, result.output
        assert str(path) in result.output

        reloaded = CMaxReport.model_validate_json(path.read_text())
        # JSON stringifies dict keys, so a lossy round trip would make every
        # downstream `c_max_for` lookup miss.
        assert reloaded.c_max_curve == {150: 1.0, 300: 2.0}
        assert reloaded.frozen is True

    def test_the_artifact_records_the_provenance(self, runner: CliRunner, tmp_path) -> None:
        path = tmp_path / "cmax.json"
        _run(runner, "--output", str(path))
        raw = json.loads(path.read_text())
        assert raw["provenance"]["origin"] == "measured"
        assert raw["run_id"] == "abc123def456"

    def test_without_output_the_curve_is_still_printed(self, runner: CliRunner) -> None:
        # The artifact is optional; a run that only wants to see the knee should
        # not have to name a file for it.
        result, _ = _run(runner)
        assert result.exit_code == 0, result.output
        assert "C_max curve" in result.output
        assert "Artifact:" not in result.output


class TestRendering:
    def test_prints_one_curve_row_per_budget(self, runner: CliRunner) -> None:
        result, _ = _run(runner)
        assert "150" in result.output
        assert "300" in result.output
        assert "ml.g5.xlarge" in result.output

    def test_prints_the_spread_per_budget(self, runner: CliRunner) -> None:
        # The number that says whether the ladder resolved a knee or noise.
        result, _ = _run(runner)
        assert "4%" in result.output
        assert "9%" in result.output

    def test_a_budget_with_no_spread_reads_as_not_applicable(self, runner: CliRunner) -> None:
        # One run cannot have a spread, and printing 0% would claim agreement
        # that was never tested.
        result, _ = _run(runner, report=_report(curve_spread={}))
        assert "n/a" in result.output

    def test_prints_uncontended_service_time_in_ms(self, runner: CliRunner) -> None:
        result, _ = _run(runner)
        assert "mean 250ms" in result.output
        assert "p95 410ms" in result.output

    def test_names_unbracketed_budgets_as_lower_bounds(self, runner: CliRunner) -> None:
        # A knee the ladder never crossed understates required capacity, which
        # is the direction that hurts.
        report = _report(
            c_max_curve={300: 16.0},
            knees=[_knee(300, 16.0, bracketed=False)],
            curve_spread={},
        )
        result, _ = _run(runner, report=report)
        assert "[300]" in result.output
        assert "lower bounds" in result.output

    def test_says_nothing_about_lower_bounds_when_every_knee_is_bracketed(
        self, runner: CliRunner
    ) -> None:
        result, _ = _run(runner)
        assert "lower bounds" not in result.output

    def test_reports_a_truncated_ladder(self, runner: CliRunner) -> None:
        # Rates above the truncation point were never offered, so the curve must
        # not read as covering them.
        result, _ = _run(runner, report=_report(ladder_truncated_at=4))
        assert "stopped at step 4" in result.output

    def test_warns_when_the_curve_is_not_per_instance(self, runner: CliRunner) -> None:
        # The artifact carries `frozen` and the instance counts, but nobody opens
        # the JSON before reading the table.
        report = _report(frozen=False, instance_counts_observed=(1, 2))
        result, _ = _run(runner, report=report)
        assert "WARNING" in result.output
        assert "not safe to read as per-instance" in result.output
        assert "[1, 2]" in result.output

    def test_a_trustworthy_curve_gets_no_warning(self, runner: CliRunner) -> None:
        result, _ = _run(runner)
        assert "not safe to read as per-instance" not in result.output

    def test_points_at_the_next_command(self, runner: CliRunner) -> None:
        # C_max alone plans nothing: T_total is the other half of the input.
        result, _ = _run(runner)
        assert "tts-bench ttotal" in result.output
        assert "tts-bench plan" in result.output

    def test_the_curve_header_names_the_transport(self, runner: CliRunner) -> None:
        # Nobody opens the JSON before reading the table, and a curve read as
        # response-stream when it is bidi sizes the fleet from the wrong number.
        result, _ = _run(runner, report=_report(transport="bidi"))
        assert "via bidi" in result.output


class TestRefusalsExitCleanly:
    def test_a_cmax_error_is_a_click_exception(self, runner: CliRunner) -> None:
        result, _ = _run(
            runner, raises=cmax_mod.CMaxError("no step met any TTFAB budget without saturating")
        )
        assert result.exit_code == 1
        assert "no step met any TTFAB budget" in result.output
        assert result.exception is None or isinstance(result.exception, SystemExit)

    def test_a_fixture_error_is_a_click_exception(self, runner: CliRunner) -> None:
        # The freeze guard firing is the tool working, not crashing.
        result, _ = _run(runner, raises=FixtureError("scale-out is not suspended"))
        assert result.exit_code == 1
        assert "scale-out is not suspended" in result.output
        assert "Traceback" not in result.output

    def test_an_unexpected_error_is_not_swallowed(self, runner: CliRunner) -> None:
        # Only the two deliberate refusals are converted. A bug must still show
        # a traceback, or it gets mistaken for a guard.
        result, _ = _run(runner, raises=RuntimeError("something broke"))
        assert result.exit_code == 1
        assert isinstance(result.exception, RuntimeError)

    def test_a_model_without_an_endpoint_is_refused(self, runner: CliRunner) -> None:
        # Polly is managed; there is no instance to size. Checked against the
        # real `resolve_endpoint`, since that mapping is the actual authority.
        result, calls = _run(runner, model="polly-neural")
        assert result.exit_code == 2
        assert "no SageMaker endpoint" in result.output
        assert calls == []

    def test_a_model_without_an_endpoint_is_refused_on_a_dry_run_too(
        self, runner: CliRunner
    ) -> None:
        # The dry run is where a wrong --model should cost nothing to discover.
        # Printing a schedule for a model that can never be measured wastes the
        # one cheap chance to catch it.
        result, _ = _run(runner, "--dry-run", model="polly-neural")
        assert result.exit_code == 2
        assert "no SageMaker endpoint" in result.output

    def test_an_unknown_model_is_a_usage_error_not_a_traceback(self, runner: CliRunner) -> None:
        result, calls = _run(runner, model="not-a-model")
        assert result.exit_code == 2
        assert "not-a-model" in result.output
        assert "Traceback" not in result.output
        assert calls == []

    def test_model_is_required(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["cmax"])
        assert result.exit_code == 2
        assert "--model" in result.output


class TestLoadTexts:
    def test_defaults_to_the_conversational_clips(self) -> None:
        # Not harvard_sentences.json: those are a uniform 39 chars and put the
        # knee at a length no real caller sends.
        from shared.loader import get_data_dir, load_tts_samples

        clips = load_tts_samples(get_data_dir() / "voice_response_clips.json")
        assert _load_texts(None, 5) == [s.text for s in clips.samples[:5]]

    def test_the_pool_is_mixed_length(self) -> None:
        # The property that makes the default the right one: a spread of
        # payload sizes, so the knee is not an artifact of one text length.
        texts = _load_texts(None, 50)
        assert len({len(t) for t in texts}) > 5

    def test_max_samples_above_the_dataset_size_takes_everything(self) -> None:
        assert len(_load_texts(None, 10_000)) == len(_load_texts(None, 51))

    def test_an_explicit_path_overrides_the_default(self) -> None:
        from shared.loader import get_data_dir

        path = get_data_dir() / "harvard_sentences.json"
        texts = _load_texts(str(path), 3)
        assert len(texts) == 3
        assert texts != _load_texts(None, 3)
