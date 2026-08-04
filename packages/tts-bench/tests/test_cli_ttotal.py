"""Tests for the `ttotal` CLI command.

`ttotal` is the one command in this package that both *sends load* and *mutates an
endpoint's desired instance count*, so almost everything here is about what happens
before either of those. Five concerns:

- **Nothing that makes the run impossible is discovered late.** The model, the ladder
  file, its transport, its configuration and its rungs are all checked before the
  endpoint is frozen and before a single request is offered. A refusal after the freeze
  costs the whole measurement — a real scale-out, and a fleet briefly parked at a raised
  count — so every refusal below also asserts that `measure` was never called.
- **Both recovery references come off the ladder, and neither is guessed.** Recovery is a
  halving: p95 at the probe's concurrency dropping into the ladder's value at half of it.
  Both have to be measured rungs, so a ladder missing either one is a re-run of `qmax`
  rather than something to interpolate through.
- **A ladder from elsewhere cannot be replayed here.** Transport is a hard refusal with no
  override — the containers hold their inference lock differently per transport, so p95
  does not transfer and the halving would be judged against the wrong number. A
  configuration mismatch is also a refusal, with `--allow-config-mismatch` the only way
  past, because `Q_max` and its ladder are properties of a GPU and a container build.
- **What this trigger does not measure is said out loud.** The deployed policy is
  suspended for the run, so its detection lag is bounded arithmetically rather than
  observed; a bound quoted as a measurement is the failure mode, and it is silent.
- **The result is kept by default and named for the configuration.** The run is expensive
  and not repeatable on demand, so `--no-save` has to be asked for, and two
  configurations' lags land under two filenames rather than overwriting each other.

`main.commands["ttotal"]` is inspected directly for flag defaults, because `cli.py`
imports `ttotal.py` lazily (numpy and botocore stay out of `--help`) and so restates the
trigger string as a literal. `measure` is patched on the `tts_bench.ttotal` module object
for the same reason: the CLI's `ttotal_mod` is a lazy local import of exactly that module.

Every test runs inside `CliRunner.isolated_filesystem()` — the artifact path defaults to a
relative `artifacts/`, so an un-isolated run would write real files into the checkout,
beside the measurements they are meant to protect. `caplog` is not used: loguru does not
propagate to the stdlib logging tree, so an assertion against it would pass whether or not
anything was emitted.
"""

from __future__ import annotations

import json
from collections.abc import Collection
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, NamedTuple
from unittest.mock import patch

import pytest
from click.testing import CliRunner, Result
from loguru import logger

from tts_bench import ttotal as ttotal_mod
from tts_bench.cli import TTOTAL_TRIGGER_FORCE_DESIRED, _default_artifact_path, main
from tts_bench.fixture import DeployedConfig, FixtureError
from tts_bench.qmax import QMaxError
from tts_bench.ttotal import StageTime, TimelineStage, TTotalError, TTotalReport

MODEL = "kokoro-82m"
ENDPOINT = "speech-kokoro-82m"
TRIGGER = TTOTAL_TRIGGER_FORCE_DESIRED
T0 = datetime(2026, 7, 30, 11, 0, 0, tzinfo=UTC)

#: The configuration the fake endpoint reports, and the one `_write_qmax` records — so a
#: default ladder replays cleanly and only a test that *asks* for a mismatch sees one.
DEPLOYED = DeployedConfig(
    instance_type="ml.g5.xlarge",
    image_digest="139b9068c5eb1f03c8312c17391dc35838e43e2417ae80d61e628e6ffb3d6a28",
    container_env={"MAX_REQUEST_AGE_S": "56"},
)
SLUG = DEPLOYED.slug

#: kokoro's measured service time, unrounded. Carried on the ladder rather than used here:
#: this command converts nothing to an arrival rate — the probe is closed-loop at N.
S_MEAN_S = 0.10986375146305409
S_P95_S = 0.16457688123919073

#: The ladder the fixture writes. 5 and 10 are the pair this command reads: the probe holds
#: 10 and recovery is judged against 5, since a second instance splits the probe in half.
LADDER = {1: 92.0, 5: 379.0, 10: 667.0, 20: 1243.0, 50: 2910.0}
Q_MAX = 50

#: What the two rungs above become once `recovery_references` has applied the tolerance.
EXPECTED_MS = LADDER[10]
RECOVERED_MS = LADDER[5] * ttotal_mod.RECOVERY_TOLERANCE


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


def _steps(*, drop: Collection[int] = (), unusable: Collection[int] = ()) -> list[dict[str, Any]]:
    """One ladder pass, as the step dicts a qmax artifact carries.

    ``drop`` omits rungs entirely and ``unusable`` keeps them but marks them as not
    informing ``Q_max``. Both matter and they are not the same input: ``ladder_p95_ms``
    filters on ``usable``, so a rung the ladder itself disowned must read as missing here
    rather than as a measured level to compare a probe against.
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
            "meets_slo": True,
            "saturated": False,
            "settled": True,
            "usable": rung not in unusable,
        }
        for index, (rung, p95) in enumerate(LADDER.items())
        if rung not in drop
    ]


def _write_qmax(**overrides: Any) -> str:
    """A `qmax` artifact shaped like a real ladder run, in the cwd.

    Written as a dict rather than through ``QMaxReport`` so a schema change surfaces here
    as a failure instead of being papered over, and so a test can produce documents the
    model would reject — a pre-fingerprint artifact, a ladder with a rung missing — which
    is exactly what this command has to survive reading. ``test_types.py`` owns the
    round-trip.
    """
    payload: dict[str, Any] = {
        "model_name": MODEL,
        "endpoint": ENDPOINT,
        "instance_type": DEPLOYED.instance_type,
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


def _report(**overrides: Any) -> TTotalReport:
    """The report a successful `measure` returns, with every stage observed.

    Spans 300s from ``desired_set`` to ``traffic_recovered``, of which 180s is provisioning
    plus image pull. Carries the fingerprint the fake endpoint reports, because `measure`
    reads it off the endpoint — without it the artifact filename would say
    ``unknown-nodigest`` and could never pair with the ladder it was measured beside.
    """
    report = TTotalReport(
        model_name=MODEL,
        endpoint=ENDPOINT,
        run_id="run123",
        trigger=TRIGGER,
        from_instances=1,
        to_instances=2,
        instance_id="i-0abc123def4567890",
        probe_concurrency=10,
        p95_expected_before_ms=EXPECTED_MS,
        p95_recovered_target_ms=RECOVERED_MS,
        p95_before_ms=1200.0,
        p95_after_ms=140.0,
        requests_before=300,
        requests_after=120,
        deployed_config=DEPLOYED.to_dict(),
    )
    report.timeline = [
        # The probe is established 30s before the trigger, and the clock starts at the
        # trigger — so a fixture whose load_applied and desired_set coincide would let a
        # regression that measured from the wrong one still pass.
        StageTime(str(TimelineStage.LOAD_APPLIED), T0, source="probe start"),
        StageTime(str(TimelineStage.DESIRED_SET), T0 + timedelta(seconds=30)),
        StageTime(str(TimelineStage.INSTANCE_LOGGING), T0 + timedelta(seconds=205), bounded=True),
        StageTime(str(TimelineStage.CONTAINER_STARTED), T0 + timedelta(seconds=210)),
        StageTime(str(TimelineStage.READY), T0 + timedelta(seconds=280)),
        StageTime(str(TimelineStage.IN_SERVICE), T0 + timedelta(seconds=300)),
        StageTime(str(TimelineStage.TRAFFIC_RECOVERED), T0 + timedelta(seconds=330), bounded=True),
    ]
    for key, value in overrides.items():
        setattr(report, key, value)
    return report


class Invocation(NamedTuple):
    """One `ttotal` run, with everything a test needs to judge it.

    ``calls`` and ``describes`` are what makes refusal *ordering* assertable: a check that
    fires after the probe has started is not the same check, and only these two say which
    happened. ``files`` is captured before the isolated filesystem is torn down, so an
    artifact's contents can be read without the test writing outside its sandbox.
    """

    result: Result
    calls: list[dict[str, Any]]
    describes: list[str]
    files: dict[str, str]


def _run(
    runner: CliRunner,
    *args: str,
    model: str = MODEL,
    qmax: dict[str, Any] | None = None,
    raw_qmax: str | None = None,
    existing: dict[str, str] | None = None,
    report: TTotalReport | None = None,
    raises: BaseException | None = None,
    deployed: DeployedConfig | Exception = DEPLOYED,
) -> Invocation:
    """Invoke `ttotal` against a written ladder, with `measure` and the fingerprint faked.

    `boto3.client` is patched to *fail*, not to return a mock: `measure` and
    `describe_deployed_config` are the only two things that would build one and both are
    replaced here, so a client being constructed means a new AWS read crept into the
    pre-flight path — which is the thing this file is mostly about.

    Pass ``raw_qmax`` to write a document that is not a qmax artifact at all, ``existing``
    to pre-place files (an artifact about to be overwritten), and an exception as
    ``deployed`` to make the fingerprint read fail.
    """
    calls: list[dict[str, Any]] = []
    describes: list[str] = []

    def fake_measure(**kwargs: Any) -> TTotalReport:
        calls.append(kwargs)
        if raises is not None:
            raise raises
        return report if report is not None else _report()

    def fake_describe(endpoint: str, **_kwargs: Any) -> DeployedConfig:
        describes.append(endpoint)
        if isinstance(deployed, Exception):
            raise deployed
        return deployed

    with (
        patch.object(ttotal_mod, "measure", fake_measure),
        patch("tts_bench.fixture.describe_deployed_config", fake_describe),
        patch("boto3.client", side_effect=AssertionError("built a boto3 client")),
        runner.isolated_filesystem(),
    ):
        for name, text in (existing or {}).items():
            path = Path(name)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
        if raw_qmax is None:
            qmax_path = _write_qmax(**(qmax or {}))
        else:
            qmax_path = "qmax.json"
            Path(qmax_path).write_text(raw_qmax)

        result = runner.invoke(main, ["ttotal", "--model", model, "--qmax", qmax_path, *args])
        files = {str(p): p.read_text() for p in sorted(Path().rglob("*")) if p.is_file()}
    return Invocation(result, calls, describes, files)


class TestTriggerLiteralsMatchTheModule:
    """The duplicated trigger string, and the defaults, against `ttotal.py`'s own.

    `click.Choice` is evaluated at import time, so referencing the module in the decorator
    would defeat the lazy import that keeps numpy out of `--help`. This is what makes
    restating it safe: a rename in the module fails here rather than leaving the CLI
    accepting a string `measure()` rejects with a `ValueError`.
    """

    def test_the_only_choice_is_the_modules_own_trigger(self) -> None:
        assert list(_param("trigger").type.choices) == [ttotal_mod.TRIGGER_FORCE_DESIRED]

    @pytest.mark.parametrize(
        "name", ["max_wait_s", "settle_s", "poll_interval_s", "region", "variant"]
    )
    def test_shared_defaults_match_measure(self, name: str) -> None:
        import inspect

        expected = inspect.signature(ttotal_mod.measure).parameters[name].default
        assert _param(name).default == expected

    def test_the_probe_defaults_resolve_to_the_modules_constants(self, runner: CliRunner) -> None:
        # These two default to None at the CLI so the module owns the number, which means
        # the default cannot be read off the parameter — only off what reaches `measure`.
        run = _run(runner)
        assert run.result.exit_code == 0, run.result.output
        assert run.calls[0]["probe_concurrency"] == ttotal_mod.PROBE_CONCURRENCY
        assert run.calls[0]["warmup_s"] == pytest.approx(ttotal_mod.DEFAULT_WARMUP_S)

    def test_help_does_not_import_ttotals_dependencies(self, runner: CliRunner) -> None:
        # The lazy-import convention the literal exists to preserve.
        result = runner.invoke(main, ["ttotal", "--help"])
        assert result.exit_code == 0
        assert "T_total" in result.output


class TestQmaxIsRequiredAndValidated:
    """The ladder is an input, not an optional refinement.

    Recovery is judged against a measured p95 at half the probe's concurrency. There is no
    default for that and no way to derive it, so the artifact is required and has to parse
    before anything else happens.
    """

    def test_the_ladder_is_required(self, runner: CliRunner) -> None:
        with runner.isolated_filesystem():
            result = runner.invoke(main, ["ttotal", "--model", MODEL])
        assert result.exit_code == 2
        assert "--qmax" in result.output

    def test_a_missing_ladder_file_fails_at_parse_time(self, runner: CliRunner) -> None:
        # `exists=True` on the option, so this never reaches the command body.
        with runner.isolated_filesystem():
            result = runner.invoke(main, ["ttotal", "--model", MODEL, "--qmax", "nope.json"])
        assert result.exit_code == 2

    def test_a_file_that_is_not_a_ladder_is_a_bad_parameter_on_the_flag(
        self, runner: CliRunner
    ) -> None:
        # Almost always a ttotal or plan artifact passed by mistake. Naming the flag, and
        # not raising the pydantic error as a traceback, is the difference between a typo
        # and an apparent bug.
        run = _run(runner, raw_qmax=json.dumps({"model_name": MODEL, "t_total_s": 420.0}))
        assert run.result.exit_code == 2
        assert "is not a qmax artifact" in run.result.output
        assert "Traceback" not in run.result.output
        # And it is refused before the endpoint is described, let alone frozen.
        assert run.calls == []
        assert run.describes == []


class TestTransportMustMatchTheLadder:
    """A ladder measured on one wire protocol cannot supply the other's recovery levels.

    Unlike the configuration check there is no override, and deliberately: the containers
    hold their inference lock differently per transport — kokoro holds it across a whole
    bidi session but per-generator on response-stream — so the p95 at a rung is a different
    quantity, not a noisier estimate of the same one.
    """

    def test_it_defaults_to_the_transport_production_uses(self, runner: CliRunner) -> None:
        run = _run(runner)
        assert run.result.exit_code == 0, run.result.output
        assert run.calls[0]["transport"] == "bidi"

    def test_a_ladder_from_another_transport_is_a_usage_error(self, runner: CliRunner) -> None:
        run = _run(runner, qmax={"transport": "response-stream"})
        assert run.result.exit_code == 2
        assert "p95 does not transfer between transports" in run.result.output
        # The fix, spelled as the flag to pass: the operator has one usable ladder and the
        # question is which transport to re-run on, not which artifact is wrong.
        assert "--transport response-stream" in run.result.output
        assert run.calls == []
        # Checked before the endpoint is described, so a wrong-transport ladder costs no
        # API call at all.
        assert run.describes == []

    def test_an_unknown_transport_is_rejected_by_the_choice(self, runner: CliRunner) -> None:
        run = _run(runner, "--transport", "grpc")
        assert run.result.exit_code == 2
        assert run.calls == []


class TestRecoveryReferences:
    """The two levels the halving is judged between, both read off the ladder's rungs.

    The probe holds N and a second instance splits it, so p95 should fall from the ladder's
    N rung toward its N/2 one. Both sides have to be measured values: an interpolated
    level would report a number that looks measured while comparing a probe at 10 against
    a rung at 20, and a threshold inside the SLO would declare recovery at the first
    request after `in_service` and measure nothing.
    """

    def test_both_levels_come_off_the_ladders_own_rungs(self, runner: CliRunner) -> None:
        run = _run(runner)
        assert run.result.exit_code == 0, run.result.output
        # 667ms at the probe's rung, and 379ms x 1.25 at half of it. Echoed before the run
        # because they are what makes the result readable while it is still happening.
        assert f"probe N=10 ({EXPECTED_MS:.0f}ms expected)" in run.result.output
        assert f"recovered at <= {RECOVERED_MS:.0f}ms (N=5 rung x 1.25)" in run.result.output

    def test_the_ladder_reaches_measure_as_a_rung_table(self, runner: CliRunner) -> None:
        run = _run(runner)
        assert run.calls[0]["ladder_p95_ms"] == LADDER
        # And the file it came from is named back, since pairing the wrong ladder with a
        # run is the mistake the fingerprint check exists to catch.
        assert "Ladder: qmax.json (Q_max 50, 5 usable rung(s), g5xlarge-139b9068)" in (
            run.result.output
        )

    @pytest.mark.parametrize(("missing", "named"), [(5, "[5]"), (10, "[10]")])
    def test_a_ladder_without_either_rung_is_a_usage_error(
        self, runner: CliRunner, missing: int, named: str
    ) -> None:
        # The refusal comes from `recovery_references`, which raises a TTotalError — it has
        # to reach the user as a usage error naming the rung, because the fix is a `qmax`
        # re-run with that concurrency added and nothing this command can do.
        run = _run(runner, qmax={"steps": _steps(drop={missing})})
        assert run.result.exit_code == 2
        assert f"no usable p95 at concurrency {named}" in run.result.output
        assert "tts-bench qmax" in run.result.output
        assert "Traceback" not in run.result.output
        assert run.calls == []

    def test_a_rung_the_ladder_disowned_does_not_count_as_measured(self, runner: CliRunner) -> None:
        # The rung is present but `usable` is false — the fleet resized under it, or the
        # client was the limit. Reading its p95 anyway would judge recovery against a
        # number the ladder itself refuses to report.
        run = _run(runner, qmax={"steps": _steps(unusable={5})})
        assert run.result.exit_code == 2
        assert "no usable p95 at concurrency [5]" in run.result.output
        assert run.calls == []

    def test_an_odd_probe_concurrency_cannot_halve_and_is_refused(self, runner: CliRunner) -> None:
        # There is no N/2 rung to compare against, so the whole recovery test is undefined.
        run = _run(runner, "--probe-concurrency", "7")
        assert run.result.exit_code == 2
        assert "even number of at least 2 to halve" in run.result.output
        assert run.calls == []

    def test_another_rung_pair_is_allowed_when_the_ladder_has_both(self, runner: CliRunner) -> None:
        # 20 and 10 are both on this ladder, so a heavier probe is legal — the constraint
        # is the pair being measured, not the specific number.
        run = _run(runner, "--probe-concurrency", "20")
        assert run.result.exit_code == 0, run.result.output
        assert run.calls[0]["probe_concurrency"] == 20
        assert f"recovered at <= {LADDER[10] * 1.25:.0f}ms (N=10 rung x 1.25)" in run.result.output

    def test_a_probe_at_or_above_q_max_warns_rather_than_refusing(self, runner: CliRunner) -> None:
        # Legal but worth saying: at or past Q_max the probe's own requests breach the SLO
        # for the whole run, which is not needed to time a scale-out. A refusal would be
        # wrong — the operator may be measuring exactly that.
        run = _run(runner, qmax={"q_max": 10, "q_max_per_run": [10, 10]})
        assert run.result.exit_code == 0, run.result.output
        assert "WARNING: --probe-concurrency 10 is at or above the measured Q_max of 10" in (
            run.result.output
        )
        assert "only has to be saturating, not overloaded" in run.result.output
        assert run.calls != []


class TestWarmupAndSettle:
    """The two waits either side of the trigger, and why neither may be zero-by-accident.

    Both bound what the measurement can see rather than what it costs, so a bad value
    produces a plausible report of nothing: no warm-up and the pre-scale p95 is a cold
    start, no settle and the run ends at `in_service` with recovery never observed.
    """

    def test_a_negative_warmup_is_a_bad_parameter(self, runner: CliRunner) -> None:
        run = _run(runner, "--warmup", "-1")
        assert run.result.exit_code == 2
        assert "--warmup cannot be negative" in run.result.output
        assert run.calls == []

    @pytest.mark.parametrize("value", ["0", "-30"])
    def test_a_settle_that_holds_no_traffic_past_the_event_is_refused(
        self, runner: CliRunner, value: str
    ) -> None:
        # Recovery is a p95 read from requests served *after* the scale event. With none of
        # them the run still succeeds and still writes an artifact — it just stops at
        # in_service and under-reports T_total, which is the dangerous direction.
        run = _run(runner, "--settle", value)
        assert run.result.exit_code == 2
        assert "far side of the scale event" in run.result.output
        assert "stops at in_service" in run.result.output
        assert run.calls == []

    def test_both_waits_are_passed_through_when_valid(self, runner: CliRunner) -> None:
        run = _run(runner, "--warmup", "45", "--settle", "240")
        assert run.result.exit_code == 0, run.result.output
        assert run.calls[0]["warmup_s"] == pytest.approx(45.0)
        assert run.calls[0]["settle_s"] == pytest.approx(240.0)


class TestConfigurationMatching:
    """The ladder's fingerprint against the endpoint it is about to be replayed on.

    Stricter than it looks worth being. The ladder supplies both recovery levels, so a
    ladder measured on another GPU or another container build means the halving is judged
    against p95 values from hardware that is not under test — and the run still completes,
    still writes an artifact, and nothing in the output looks wrong.
    """

    def test_a_matching_fingerprint_runs(self, runner: CliRunner) -> None:
        run = _run(runner)
        assert run.result.exit_code == 0, run.result.output
        assert run.calls != []

    def test_a_different_instance_type_stops_the_run(self, runner: CliRunner) -> None:
        # The g5-ladder-on-g6 case this check exists for.
        g6 = DeployedConfig(
            instance_type="ml.g6.xlarge",
            image_digest=DEPLOYED.image_digest,
            container_env=dict(DEPLOYED.container_env),
        )
        run = _run(runner, deployed=g6)
        assert run.result.exit_code != 0
        assert "ml.g5.xlarge" in run.result.output
        assert "ml.g6.xlarge" in run.result.output
        # No load offered and no desired count touched: the refusal has to come first.
        assert run.calls == []

    def test_a_different_image_digest_stops_the_run(self, runner: CliRunner) -> None:
        # Same hardware, rebuilt container — an admission queue or a batching change lands
        # here, and either moves every rung on the ladder.
        rebuilt = DeployedConfig(
            instance_type=DEPLOYED.instance_type,
            image_digest="0000000011112222333344445555666677778888999900001111222233334444",
            container_env=dict(DEPLOYED.container_env),
        )
        run = _run(runner, deployed=rebuilt)
        assert run.result.exit_code != 0
        assert "serving code differs" in run.result.output
        assert run.calls == []

    def test_a_different_container_env_stops_the_run(self, runner: CliRunner) -> None:
        retuned = DeployedConfig(
            instance_type=DEPLOYED.instance_type,
            image_digest=DEPLOYED.image_digest,
            container_env={"MAX_REQUEST_AGE_S": "30"},
        )
        run = _run(runner, deployed=retuned)
        assert run.result.exit_code != 0
        assert "container_env" in run.result.output
        assert run.calls == []

    def test_the_override_downgrades_it_to_a_warning(
        self, runner: CliRunner, logged: list[str]
    ) -> None:
        # For the operator who knows the difference is irrelevant to what they are
        # measuring — most of T_total is container start, which the ladder's p95 does not
        # enter at all.
        g6 = DeployedConfig(instance_type="ml.g6.xlarge", image_digest=DEPLOYED.image_digest)
        run = _run(runner, "--allow-config-mismatch", deployed=g6)
        assert run.result.exit_code == 0, run.result.output
        assert run.calls != []
        assert any("different configuration" in message for message in logged)

    def test_a_ladder_predating_fingerprinting_is_refused(self, runner: CliRunner) -> None:
        # The committed artifacts look like this. Accepting them silently is the exact hole
        # this check closes, so "no fingerprint" is a mismatch rather than a pass.
        run = _run(runner, qmax={"deployed_config": {}})
        assert run.result.exit_code != 0
        assert "predates configuration fingerprinting" in run.result.output
        assert run.calls == []

    def test_a_pre_fingerprint_ladder_is_allowed_with_the_override(self, runner: CliRunner) -> None:
        run = _run(runner, "--allow-config-mismatch", qmax={"deployed_config": {}})
        assert run.result.exit_code == 0, run.result.output
        assert "WARNING" in run.result.output
        assert run.calls != []

    def test_an_unreadable_endpoint_warns_but_runs(self, runner: CliRunner) -> None:
        # One extra describe call failing is not a reason to refuse a run that is otherwise
        # viable — the endpoint is about to be invoked either way, which is a far better
        # test of whether it is reachable.
        run = _run(runner, deployed=FixtureError("no such endpoint"))
        assert run.result.exit_code == 0, run.result.output
        assert f"WARNING: could not verify the deployed configuration of {ENDPOINT}" in (
            run.result.output
        )
        assert run.calls != []

    def test_the_flag_defaults_off(self) -> None:
        assert _param("allow_config_mismatch").default is False


class TestThePolicyHalfIsLabelledAsBounded:
    """That the run says what it does *not* measure, on stdout and in the artifact both.

    The trigger suspends the deployed policy and raises `DesiredInstanceCount` itself, so
    the policy's own detection lag is arithmetic from its configuration rather than an
    observation. `plan` sizes headroom against the sum, so the two terms have to stay
    separately labelled — a bound folded into a measurement is unfalsifiable.
    """

    def test_stdout_says_the_policy_lag_is_bounded_not_measured(self, runner: CliRunner) -> None:
        run = _run(runner)
        assert "bounded at 60s rather than measured" in run.result.output
        assert "t_total_s alone is the capacity half" in run.result.output

    def test_it_states_the_capacity_change_it_is_about_to_make(self, runner: CliRunner) -> None:
        # This command mutates production capacity, briefly, on purpose. Said out loud
        # before it happens, with the wait it may cost.
        run = _run(runner)
        assert f"This will suspend autoscaling on {ENDPOINT}" in run.result.output
        assert "raise DesiredInstanceCount by 1" in run.result.output

    def test_the_artifact_keeps_the_two_terms_apart(self, runner: CliRunner) -> None:
        # stdout scrolls away; the JSON is what a later `plan` run reads, and it reads the
        # bounded figure. Both numbers and the bound itself are on the artifact so the sum
        # can be taken apart again.
        run = _run(runner, "--output", "ttotal.json")
        payload = json.loads(run.files["ttotal.json"])
        assert payload["t_total_s"] == pytest.approx(300.0)
        assert payload["policy_lag_bound_s"] == pytest.approx(60.0)
        assert payload["t_total_with_policy_bound_s"] == pytest.approx(360.0)
        assert payload["trigger"] == TRIGGER


class TestOutput:
    def test_prints_the_timeline(self, runner: CliRunner) -> None:
        run = _run(runner)
        assert f"T_total for {ENDPOINT} ({MODEL}), run run123:" in run.result.output
        assert "300.0s from capacity requested to traffic recovered" in run.result.output
        assert "stage breakdown" in run.result.output

    def test_names_the_new_instance(self, runner: CliRunner) -> None:
        run = _run(runner)
        assert "i-0abc123def4567890" in run.result.output

    def test_writes_a_readable_artifact(self, runner: CliRunner) -> None:
        run = _run(runner, "--output", "ttotal.json")
        assert run.result.exit_code == 0, run.result.output

        payload = json.loads(run.files["ttotal.json"])
        assert payload["endpoint"] == ENDPOINT
        # The clock starts at the trigger, not at the probe's start: 300s, not the 330s the
        # timeline spans from load_applied.
        assert payload["t_total_s"] == pytest.approx(300.0)
        assert payload["aws_share_s"] == pytest.approx(180.0)
        assert len(payload["timeline"]) == 7
        # The rungs the recovery test was valid against. Without them a later reader cannot
        # tell whether the halving was judged against the right pair.
        assert payload["probe_concurrency"] == 10
        assert payload["p95_expected_before_ms"] == pytest.approx(EXPECTED_MS)
        assert payload["p95_recovered_target_ms"] == pytest.approx(RECOVERED_MS)

    def test_the_artifact_carries_its_configuration(self, runner: CliRunner) -> None:
        # `plan` pairs a Q_max ladder with a T_total lag and refuses to mix two
        # configurations. Without a fingerprint on this side there would be nothing to
        # compare, and a g5 ladder would pair with a g6 lag silently.
        run = _run(runner, "--output", "ttotal.json")
        payload = json.loads(run.files["ttotal.json"])
        assert payload["deployed_config"]["instance_type"] == "ml.g5.xlarge"
        assert payload["config_slug"] == SLUG

    def test_without_output_it_lands_under_the_name_the_helper_builds(
        self, runner: CliRunner
    ) -> None:
        # A T_total run costs a real scale-out and, on a timeout, most of max_wait_s of
        # offered load, so keeping the result is the default. Asserted against
        # `_default_artifact_path` rather than a literal: the naming rule lives in one
        # place, and the trigger and slug in it are what stop two runs colliding.
        expected = _default_artifact_path("ttotal", MODEL, TRIGGER, SLUG)
        run = _run(runner)
        assert run.result.exit_code == 0, run.result.output
        assert str(expected) in run.result.output
        assert str(expected) in run.files
        assert TRIGGER in expected.name and SLUG in expected.name

    def test_replacing_another_configurations_artifact_warns(self, runner: CliRunner) -> None:
        # The realistic mistake: re-measure after a redeploy, reuse the previous --output,
        # and the old hardware's lag is gone. The previous configuration is named because
        # it is usually the comparison the operator wanted.
        g6 = DeployedConfig(instance_type="ml.g6.xlarge", image_digest="abcd1234" * 8)
        run = _run(
            runner,
            "--output",
            "ttotal.json",
            existing={"ttotal.json": json.dumps({"deployed_config": g6.to_dict()})},
        )
        assert run.result.exit_code == 0, run.result.output
        assert "holds a measurement of a different configuration" in run.result.output
        assert g6.slug in run.result.output

    def test_no_save_writes_nothing_and_says_so(self, runner: CliRunner) -> None:
        run = _run(runner, "--no-save")
        assert run.result.exit_code == 0, run.result.output
        assert "nothing written" in run.result.output
        assert "Artifact:" not in run.result.output
        assert sorted(run.files) == ["qmax.json"]

    def test_points_at_the_plan_command_with_both_artifacts(self, runner: CliRunner) -> None:
        # The loop from measurement to deployed config closes by copy-paste: `plan` needs
        # this exact pair of files, and the slug in the lag's name is not something to
        # retype from memory — pairing the wrong two is what the fingerprint check catches.
        destination = _default_artifact_path("ttotal", MODEL, TRIGGER, SLUG)
        run = _run(runner)
        assert f"Next: tts-bench plan --qmax qmax.json --ttotal {destination}" in run.result.output

    def test_a_report_with_no_measurable_total_still_prints(self, runner: CliRunner) -> None:
        # Every API came back empty. Worth seeing, and worth still writing, rather than
        # crashing on a None format after the endpoint has already been scaled.
        run = _run(runner, report=_report(timeline=[]))
        assert run.result.exit_code == 0, run.result.output
        assert "not measurable from the stages observed" in run.result.output

    def test_events_are_written_when_asked(self, runner: CliRunner) -> None:
        run = _run(runner, "--events", "events.jsonl")
        assert run.result.exit_code == 0, run.result.output
        assert run.calls[0]["event_sink"] is not None
        assert "events.jsonl" in run.files
        assert "Events: events.jsonl" in run.result.output

    def test_no_event_sink_without_the_flag(self, runner: CliRunner) -> None:
        run = _run(runner)
        assert run.calls[0]["event_sink"] is None

    def test_the_events_writer_is_closed_even_when_measure_raises(self, runner: CliRunner) -> None:
        # A run that dies mid-measurement is exactly when the per-request log is worth
        # having, so the writer's context manager has to wrap the failure too.
        run = _run(runner, "--events", "events.jsonl", raises=TTotalError("no scale-out"))
        assert run.result.exit_code != 0
        assert "events.jsonl" in run.files
        # Closed, not merely created: the writer refuses to be called outside its context
        # manager, which is the only externally visible difference. The argument is never
        # reached, so it need not be a real event.
        sink = run.calls[0]["event_sink"]
        with pytest.raises(RuntimeError, match="outside its context manager"):
            sink(None)


class TestRefusals:
    def test_a_ttotal_error_is_a_click_exception(self, runner: CliRunner) -> None:
        # The tool declining to report a lag it did not observe. A traceback would read as
        # a bug rather than as the guard working.
        run = _run(runner, raises=TTotalError("did not reach more than 1 instance"))
        assert run.result.exit_code != 0
        assert "did not reach more than 1 instance" in run.result.output
        assert "Traceback" not in run.result.output

    def test_a_qmax_error_is_a_click_exception(self, runner: CliRunner) -> None:
        # Raised from the shared load-generation path, so it surfaces here too.
        run = _run(runner, raises=QMaxError("the probe completed nothing"))
        assert run.result.exit_code != 0
        assert "the probe completed nothing" in run.result.output
        assert "Traceback" not in run.result.output

    def test_a_fixture_error_is_a_click_exception(self, runner: CliRunner) -> None:
        run = _run(runner, raises=FixtureError("max_capacity is 1, cannot scale"))
        assert run.result.exit_code != 0
        assert "cannot scale" in run.result.output
        assert "Traceback" not in run.result.output

    def test_an_unexpected_error_is_not_swallowed(self, runner: CliRunner) -> None:
        # Only the three refusals above are converted. Anything else is a bug in this
        # package and must not be dressed up as the endpoint declining to scale.
        boom = RuntimeError("boom")
        run = _run(runner, raises=boom)
        assert run.result.exit_code != 0
        assert run.result.exception is boom

    def test_an_unknown_model_is_refused_before_any_aws_read(self, runner: CliRunner) -> None:
        run = _run(runner, model="nonexistent-model")
        assert run.result.exit_code == 2
        assert "Traceback" not in run.result.output
        assert run.calls == []
        assert run.describes == []

    def test_a_managed_model_with_no_endpoint_is_refused(self, runner: CliRunner) -> None:
        # Polly is a real model name with no SageMaker endpoint behind it, so there is no
        # instance count to raise and nothing to freeze — a different failure from a typo,
        # and the message has to say which.
        run = _run(runner, model="polly-neural")
        assert run.result.exit_code == 2
        assert "no SageMaker endpoint" in run.result.output
        assert run.calls == []

    def test_the_model_is_resolved_before_the_ladder_is_read(self, runner: CliRunner) -> None:
        # Both inputs are wrong. The model is what names the endpoint every other check is
        # about, so its refusal comes first rather than reporting a ladder that would be
        # irrelevant anyway.
        run = _run(runner, model="nonexistent-model", raw_qmax="not json")
        assert run.result.exit_code == 2
        assert "--model" in run.result.output
        assert "is not a qmax artifact" not in run.result.output

    def test_model_is_required(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["ttotal"])
        assert result.exit_code == 2
        assert "--model" in result.output
