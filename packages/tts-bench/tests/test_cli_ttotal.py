"""Tests for the `ttotal` CLI command.

`ttotal` is the one command in this package that both *sends load* and *mutates an
endpoint's desired instance count*, so the tests here are mostly about what happens
before either of those. Four concerns:

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
- **An artifact measured on another configuration is refused.** `S` and the budget are
  properties of a GPU and a container build, not of a model, so replaying a g5 artifact
  against a g6 endpoint would produce a plan for a fleet that does not exist. Unlike the
  transport check this stops the run, and `--allow-config-mismatch` is the only way past.

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
from loguru import logger

from tts_bench import ttotal as ttotal_mod
from tts_bench.cli import (
    TTOTAL_TRIGGER_DRIVE_LOAD,
    TTOTAL_TRIGGER_FORCE_DESIRED,
    _config_ttfab_budget_ms,
    main,
)
from tts_bench.fixture import DeployedConfig, FixtureError
from tts_bench.ttotal import StageTime, TimelineStage, TTotalError, TTotalReport

MODEL = "kokoro-82m"
ENDPOINT = "speech-kokoro-82m"
T0 = datetime(2026, 7, 30, 11, 0, 0, tzinfo=UTC)

#: Measured on this endpoint, so a smoke run against it needs no flags beyond --model.
DEPLOYED_TARGET = 0.713
S_MEAN_S = 0.10986375146305409

#: The configuration the fake endpoint reports, and the one `_artifact` records — so a
#: default artifact replays cleanly and only a test that *asks* for a mismatch sees one.
DEPLOYED = DeployedConfig(
    instance_type="ml.g5.xlarge",
    image_digest="139b9068c5eb1f03c8312c17391dc35838e43e2417ae80d61e628e6ffb3d6a28",
    container_env={"MAX_REQUEST_AGE_S": "56"},
)


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
        # The same fingerprint the fake endpoint reports, since `measure` reads it off
        # the endpoint. Without it the artifact filename would say `unknown-nodigest`
        # and the pairing check against a cmax curve could never pass.
        deployed_config=DEPLOYED.to_dict(),
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
    """A minimal `cmax` artifact, shaped like the committed one.

    Carries the same fingerprint the fake endpoint reports, so the configuration check
    passes by default. Pass ``deployed_config=`` to make it disagree, or to drop it and
    stand in for an artifact written before fingerprinting existed.
    """
    payload = {
        "model_name": MODEL,
        "endpoint": ENDPOINT,
        "s_mean_s": S_MEAN_S,
        "c_max_curve": {"300": 1.6302521008403361, "500": 1.6302521008403361},
        "transport": "bidi",
        "deployed_config": DEPLOYED.to_dict(),
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
    deployed: DeployedConfig | Exception = DEPLOYED,
):
    """Invoke `ttotal` with `measure`, the policy read, and the fingerprint replaced.

    `boto3.client` is left real but never reached for credentials: `measure` is the only
    thing that would use the clients, and it is patched. `deployed_target_value` and
    `describe_deployed_config` are patched because they are the two AWS reads the command
    makes *itself*, before deciding whether a run is even possible. Pass an exception as
    `deployed` to make the fingerprint read fail.

    Runs in an isolated filesystem because `--output` now defaults to a *relative*
    `artifacts/` path, so without this the suite would write real artifacts into the
    checkout, beside the measurements they are meant to protect.
    """
    calls: list[dict] = []

    def fake_measure(**kwargs) -> TTotalReport:
        calls.append(kwargs)
        if raises is not None:
            raise raises
        return report if report is not None else _report()

    def fake_describe(*_args, **_kwargs) -> DeployedConfig:
        if isinstance(deployed, Exception):
            raise deployed
        return deployed

    with (
        patch.object(ttotal_mod, "measure", fake_measure),
        patch.object(ttotal_mod, "deployed_target_value", return_value=target_value),
        patch("tts_bench.fixture.describe_deployed_config", fake_describe),
        patch("boto3.client"),
        runner.isolated_filesystem(),
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

    def test_force_desired_does_not_need_one(self, runner: CliRunner) -> None:
        # It starts no load driver, so there is no arrival rate to convert. Requiring S
        # here would make the cheapest probe available -- can this endpoint get a second
        # instance at all? -- wait on a C_max measurement it does not use.
        result, calls = _run(runner, "--trigger", TTOTAL_TRIGGER_FORCE_DESIRED)
        assert result.exit_code == 0
        assert calls[0]["s_mean_s"] == 0.0
        # And it must not print a service time it never had.
        assert "S=0ms" not in result.output
        assert "do not apply" in result.output


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

    def test_the_budget_is_not_the_end_to_end_slo(self) -> None:
        # `plan` takes --ttfab-slo-ms and derives W_max from it; this command deliberately
        # does not, and the reason is measurement resolution rather than tidiness.
        # Recovery is "p95 came back", so the threshold has to sit *between* the
        # overloaded p95 and the recovered one. Kokoro at 3x C_target reaches 818ms, which
        # is already inside a 3000ms SLO -- thresholded there, every run would report
        # recovery at the instant the instance came into service and measure nothing.
        assert not any(param.name == "ttfab_slo_ms" for param in main.commands["ttotal"].params)
        assert _config_ttfab_budget_ms(ENDPOINT) == pytest.approx(300.0)

    def test_the_config_budget_is_far_tighter_than_the_configs_own_slo(self) -> None:
        # The two live side by side in ModelEndpointConfig and differ by 10x on purpose.
        # Pinned here because reading the wrong one is a silent failure, not an error.
        from speech_infra.config import TTS_MODEL_CONFIGS

        config = TTS_MODEL_CONFIGS["kokoro-82m"]
        assert config.ttfab_budget_ms == 300
        assert config.ttfab_slo_ms == 3000


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
            # Isolated for the same reason `_run` is: the artifact path defaults to a
            # relative artifacts/, so an un-isolated run writes into the checkout.
            runner.isolated_filesystem(),
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


class TestConfigurationMatching:
    """The artifact's fingerprint against the endpoint it is about to be replayed on.

    Stricter than the transport check above, and deliberately so. A transport mismatch
    mis-sizes the offered rate, which shows up in the result; a configuration mismatch
    means `S` and the budget were measured on another GPU or another container build, so
    every derived number — `C_target`, the fleet size, the queue depth — describes a
    fleet that does not exist, and nothing in the output looks wrong.
    """

    def test_a_matching_fingerprint_runs(self, runner: CliRunner, tmp_path) -> None:
        result, calls = _run(runner, "--measured", _artifact(tmp_path))
        assert result.exit_code == 0
        assert calls != []

    def test_a_different_instance_type_stops_the_run(self, runner: CliRunner, tmp_path) -> None:
        # The g5-artifact-on-g6 case this check exists for.
        g6 = DeployedConfig(
            instance_type="ml.g6.xlarge",
            image_digest=DEPLOYED.image_digest,
            container_env=dict(DEPLOYED.container_env),
        )
        result, calls = _run(runner, "--measured", _artifact(tmp_path), deployed=g6)
        assert result.exit_code != 0
        assert "ml.g5.xlarge" in result.output
        assert "ml.g6.xlarge" in result.output
        # No load offered and no desired count touched: the refusal has to come first.
        assert calls == []

    def test_a_different_image_digest_stops_the_run(self, runner: CliRunner, tmp_path) -> None:
        # Same hardware, rebuilt container — an admission queue lands here.
        rebuilt = DeployedConfig(
            instance_type=DEPLOYED.instance_type,
            image_digest="0000000011112222333344445555666677778888999900001111222233334444",
            container_env=dict(DEPLOYED.container_env),
        )
        result, calls = _run(runner, "--measured", _artifact(tmp_path), deployed=rebuilt)
        assert result.exit_code != 0
        assert "serving code differs" in result.output
        assert calls == []

    def test_a_different_container_env_stops_the_run(self, runner: CliRunner, tmp_path) -> None:
        retuned = DeployedConfig(
            instance_type=DEPLOYED.instance_type,
            image_digest=DEPLOYED.image_digest,
            container_env={"MAX_REQUEST_AGE_S": "30"},
        )
        result, calls = _run(runner, "--measured", _artifact(tmp_path), deployed=retuned)
        assert result.exit_code != 0
        assert "container_env" in result.output
        assert calls == []

    def test_the_override_downgrades_it_to_a_warning(
        self, runner: CliRunner, tmp_path, logged
    ) -> None:
        # For the operator who knows the difference is irrelevant to what they are
        # measuring -- e.g. T_total, which depends on nothing S depends on.
        g6 = DeployedConfig(instance_type="ml.g6.xlarge", image_digest=DEPLOYED.image_digest)
        result, calls = _run(
            runner,
            "--measured",
            _artifact(tmp_path),
            "--allow-config-mismatch",
            deployed=g6,
        )
        assert result.exit_code == 0
        assert calls != []
        assert any("different configuration" in message for message in logged)

    def test_an_artifact_predating_fingerprinting_is_refused(
        self, runner: CliRunner, tmp_path
    ) -> None:
        # Both committed artifacts look like this. Accepting them silently is the exact
        # hole this check closes, so "no fingerprint" is a mismatch rather than a pass.
        result, calls = _run(runner, "--measured", _artifact(tmp_path, deployed_config={}))
        assert result.exit_code != 0
        assert "predates configuration fingerprinting" in result.output
        assert calls == []

    def test_a_pre_fingerprint_artifact_is_allowed_with_the_override(
        self, runner: CliRunner, tmp_path
    ) -> None:
        result, calls = _run(
            runner,
            "--measured",
            _artifact(tmp_path, deployed_config={}),
            "--allow-config-mismatch",
        )
        assert result.exit_code == 0
        assert "WARNING" in result.output
        assert calls != []

    def test_an_unreadable_endpoint_warns_but_runs(self, runner: CliRunner, tmp_path) -> None:
        # One extra describe call failing is not a reason to refuse a run that is
        # otherwise viable -- the endpoint is about to be invoked either way, which is a
        # far better test of whether it is reachable.
        result, calls = _run(
            runner,
            "--measured",
            _artifact(tmp_path),
            deployed=FixtureError("no such endpoint"),
        )
        assert result.exit_code == 0
        assert "could not verify the deployed configuration" in result.output
        assert calls != []

    def test_no_check_without_an_artifact(self, runner: CliRunner) -> None:
        # Nothing is being replayed, so there is nothing to compare. A run driven
        # entirely by flags measures whatever is deployed, which is self-consistent.
        result, calls = _run(
            runner, "--s-mean", "0.11", deployed=FixtureError("must not be called")
        )
        assert result.exit_code == 0
        assert calls != []

    def test_the_flag_defaults_off(self) -> None:
        assert _param("allow_config_mismatch").default is False


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

    def test_the_artifact_carries_its_configuration(self, runner: CliRunner, tmp_path) -> None:
        # `plan` pairs a C_max curve with a T_total lag and refuses to mix two
        # configurations. Without a fingerprint on this side there would be nothing to
        # compare, and a g5 curve would pair with a g6 lag silently.
        out = tmp_path / "ttotal.json"
        _run(runner, "--s-mean", "0.11", "--output", str(out))

        payload = json.loads(out.read_text())
        assert payload["deployed_config"]["instance_type"] == "ml.g5.xlarge"
        assert payload["config_slug"] == "g5xlarge-139b9068"

    def test_without_output_the_artifact_is_still_saved(self, runner: CliRunner) -> None:
        # A T_total run costs a real scale-out and, on a timeout, most of max_wait_s of
        # offered load. Keeping the result is the default.
        result, _ = _run(runner, "--s-mean", "0.11")
        assert result.exit_code == 0, result.output
        assert "Artifact: artifacts/ttotal-kokoro-82m-drive-load-g5xlarge-139b9068.json" in (
            result.output
        )

    def test_the_trigger_is_in_the_default_name(self, runner: CliRunner) -> None:
        # force-desired measures only the container half, so its result is a different
        # measurement of the same configuration -- it must not overwrite a driven run.
        result, _ = _run(runner, "--trigger", TTOTAL_TRIGGER_FORCE_DESIRED)
        assert result.exit_code == 0, result.output
        assert "artifacts/ttotal-kokoro-82m-force-desired-" in result.output

    def test_no_save_writes_nothing_and_says_so(self, runner: CliRunner) -> None:
        result, _ = _run(runner, "--s-mean", "0.11", "--no-save")
        assert result.exit_code == 0, result.output
        assert "nothing written" in result.output
        assert "Artifact:" not in result.output

    def test_points_at_the_next_command(self, runner: CliRunner) -> None:
        # Names both artifacts rather than a --t-total number, because `plan` consumes
        # the stage breakdown: the provision stage is swept, not assumed, so a single
        # total is not the input.
        result, _ = _run(runner, "--s-mean", "0.11")
        assert "tts-bench plan --measured" in result.output
        # The path this run just wrote, not a placeholder: the slug in it is not something
        # to retype from memory, and pairing the wrong two artifacts is the mistake the
        # fingerprint exists to catch.
        assert "--ttotal artifacts/ttotal-kokoro-82m-drive-load-g5xlarge-139b9068.json" in (
            result.output
        )
        assert "--peak-rps" in result.output

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
