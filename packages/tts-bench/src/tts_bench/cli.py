"""CLI for TTS performance benchmarking."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import click
from loguru import logger

from tts_inference.types import TTSModelName

if TYPE_CHECKING:
    from tts_bench.observe import ExpectedScaling
    from tts_bench.types import ScalingPlan

ALL_MODELS = [m.value for m in TTSModelName]

#: Where a measurement lands when ``--output`` is not given. These runs cost 45+ minutes
#: and an endpoint freeze, so the default is to keep the result.
ARTIFACT_DIR = Path("artifacts")


def _default_artifact_path(kind: str, *parts: str) -> Path:
    """Name an artifact after the configuration it measured.

    Built at *write* time rather than declared as a click default, because the
    configuration slug is only known once the run has described the endpoint —
    which is the whole point of naming files this way. Two configurations produce
    two filenames, so re-running after a redeploy cannot silently overwrite the
    curve it should be compared against.
    """
    stem = "-".join([kind, *(p for p in parts if p)])
    return ARTIFACT_DIR / f"{stem}.json"


def _save_artifact(path: Path, payload: str) -> None:
    """Write an artifact, creating ``artifacts/`` if this is the first run in a checkout."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload)


@click.group()
def main() -> None:
    """TTS Performance Benchmarking Suite."""


@main.command()
@click.option(
    "--models",
    default="all",
    help="Comma-separated model names or 'all'. Plural: this command compares models.",
)
@click.option("--runs", default=5, type=int, help="Runs per sample")
@click.option("--max-samples", default=10, type=int, help="Number of text samples")
@click.option("--region", default="us-east-1", help="AWS region every client is built in")
@click.option("--output", default=None, type=click.Path(), help="Output JSON path")
def latency(models: str, runs: int, max_samples: int, region: str, output: str | None) -> None:
    """Measure latency (TTFAB) for each model."""
    from shared.loader import get_data_dir, load_tts_samples
    from tts_bench.latency import measure_latency

    model_list = ALL_MODELS if models == "all" else [m.strip() for m in models.split(",")]

    dataset = load_tts_samples(get_data_dir() / "harvard_sentences.json")
    texts = [s.text for s in dataset.samples[:max_samples]]

    results = {}
    for model in model_list:
        logger.info("Measuring latency for {} ({} samples x {} runs)", model, len(texts), runs)
        try:
            stats = measure_latency(model, texts, runs_per_text=runs, region=region)
            results[model] = stats.model_dump()
            click.echo(
                f"  {model}: P50={stats.p50_ms:.0f}ms P90={stats.p90_ms:.0f}ms "
                f"P99={stats.p99_ms:.0f}ms mean={stats.mean_ms:.0f}ms"
            )
        except Exception as e:
            click.echo(f"  {model}: FAILED - {e}")
            results[model] = {"error": str(e)}

    if output:
        Path(output).write_text(json.dumps(results, indent=2))
        click.echo(f"\nResults: {output}")
    else:
        click.echo(f"\n{json.dumps(results, indent=2)}")


@main.command()
@click.option(
    "--models",
    default="all",
    help="Comma-separated model names or 'all'. Plural: this command compares models.",
)
@click.option("--concurrency", default="5,10,20", help="Comma-separated concurrency levels")
@click.option("--window", default=15.0, type=float, help="Seconds per concurrency level")
@click.option("--region", default="us-east-1", help="AWS region every client is built in")
@click.option("--output", default=None, type=click.Path(), help="Output JSON path")
def scalability(
    models: str, concurrency: str, window: float, region: str, output: str | None
) -> None:
    """Test scalability under concurrent load."""
    from tts_bench.scalability import measure_scalability

    model_list = ALL_MODELS if models == "all" else [m.strip() for m in models.split(",")]
    levels = [int(c.strip()) for c in concurrency.split(",")]
    text = "The birch canoe slid on the smooth planks."

    all_results = []
    for model in model_list:
        logger.info("Scalability test for {}", model)
        try:
            results = measure_scalability(model, text, levels, region=region, window_s=window)
            all_results.extend(results)
            for r in results:
                click.echo(
                    f"  {model} @{r['concurrency']}: "
                    f"{r['throughput_chars_per_s']:.0f} chars/s "
                    f"P50={r['p50_ms']:.0f}ms P99={r['p99_ms']:.0f}ms"
                )
        except Exception as e:
            click.echo(f"  {model}: FAILED - {e}")

    if output:
        Path(output).write_text(json.dumps(all_results, indent=2))


@main.command()
@click.option(
    "--models",
    default="all",
    help="Comma-separated model names or 'all'. Plural: this command compares models.",
)
@click.option("--max-samples", default=20, type=int, help="Number of text samples for throughput")
@click.option("--region", default="us-east-1", help="AWS region every client is built in")
@click.option("--output", default=None, type=click.Path(), help="Output JSON path")
def cost(models: str, max_samples: int, region: str, output: str | None) -> None:
    """Calculate cost per million characters."""
    from shared.loader import get_data_dir, load_tts_samples
    from tts_bench.cost import calculate_cost

    model_list = ALL_MODELS if models == "all" else [m.strip() for m in models.split(",")]

    dataset = load_tts_samples(get_data_dir() / "harvard_sentences.json")
    texts = [s.text for s in dataset.samples[:max_samples]]

    results = []
    for model in model_list:
        logger.info("Calculating cost for {}", model)
        try:
            result = calculate_cost(model, texts, region=region)
            results.append(result)
            click.echo(
                f"  {model}: ${result['cost_per_m_chars']:.2f}/M chars "
                f"({result['chars_per_min']:.0f} chars/min on {result['instance_type']})"
            )
        except Exception as e:
            click.echo(f"  {model}: FAILED - {e}")

    if output:
        Path(output).write_text(json.dumps(results, indent=2))


def _parse_ints(raw: str, *, flag: str) -> tuple[int, ...]:
    """Parse a comma-separated integer list, rejecting anything non-positive."""
    try:
        values = tuple(int(part) for part in raw.split(",") if part.strip())
    except ValueError as exc:
        raise click.BadParameter(f"{flag} must be comma-separated integers: {raw}") from exc
    if not values:
        raise click.BadParameter(f"{flag} must not be empty")
    if any(v <= 0 for v in values):
        raise click.BadParameter(f"{flag} values must be positive: {raw}")
    return values


def _load_texts(samples: str | None, max_samples: int) -> list[str]:
    """Load the load-generator's text pool.

    Defaults to ``voice_response_clips.json`` rather than ``harvard_sentences.json``
    (which `latency` and `cost` use): the clips are 6-61 chars of conversational
    healthcare speech, the realistic mixed-length payload for a latency-sensitive
    voice agent. Harvard sentences are a uniform 39 chars, which yields a sharp
    knee at a length no real caller sends.
    """
    from shared.loader import get_data_dir, load_tts_samples

    path = Path(samples) if samples else get_data_dir() / "voice_response_clips.json"
    dataset = load_tts_samples(path)
    return [s.text for s in dataset.samples[:max_samples]]


# Literal defaults, mirroring `tts_bench.qmax`. Importing that module here would
# pull numpy and botocore into `--help`, and every other command in this file
# imports lazily for the same reason. `test_cli_qmax.py` asserts these match the
# module constants, so the duplication cannot drift silently.
QMAX_LADDER_DEFAULT = "1,5,10,20,30,40,50,60"

# Duplicated from `tts_bench.ttotal` for the same reason: a click.Choice is evaluated at
# import time, so referencing the module here would defeat the lazy import.
# `test_cli_ttotal.py` asserts these match the module constants.
TTOTAL_TRIGGER_FORCE_DESIRED = "force-desired"


@main.command()
@click.option(
    "--model",
    required=True,
    help=(
        "Model name, e.g. kokoro-82m. Singular and required: this measures one "
        "configuration, and it freezes that endpoint to do it."
    ),
)
@click.option("--region", default="us-east-1", help="AWS region every client is built in")
@click.option("--variant", default="primary", help="Production variant on the endpoint")
@click.option("--voice", default=None, help="Override the model's default voice")
@click.option(
    "--concurrency",
    "concurrency",
    default=QMAX_LADDER_DEFAULT,
    help=(
        "Ladder in concurrency (queued + executing), comma-separated. Held exactly at "
        "each rung, so these are the values Q_max can come back as -- not rates to be "
        "converted. Three rungs are consumed downstream: 1 gives the alarm its "
        "service-time reference, 5 and 10 are the pair `ttotal` compares a halving "
        "against."
    ),
)
@click.option(
    "--slo-ms",
    "slo_ms",
    default=3000,
    type=int,
    help=(
        "p95 first-byte SLO in ms, queue time included. The pass/fail line that "
        "*defines* Q_max, so it is recorded on the artifact and `plan` refuses to read "
        "this ladder against a different one."
    ),
)
@click.option("--hold", "hold_s", default=240.0, type=float, help="Seconds held per rung")
@click.option(
    "--measure-window",
    "measure_window_s",
    default=60.0,
    type=float,
    help="Trailing seconds of each rung that are measured; the rest is warm-up",
)
@click.option(
    "--settle-between-steps",
    "settle_between_steps_s",
    default=30.0,
    type=float,
    help="Idle seconds between rungs so the previous queue drains",
)
@click.option("--runs", default=1, type=int, help="Ladder passes; 2+ to see run-to-run spread")
@click.option(
    "--transport",
    default="response-stream",
    type=click.Choice(["response-stream", "bidi"]),
    help="Wire protocol to measure. Q_max does not transfer between the two.",
)
@click.option(
    "--seed", default=1234, type=int, help="Text-pool shuffle seed; keeps runs comparable"
)
@click.option("--max-samples", default=50, type=int, help="Texts drawn into the pool")
@click.option("--samples", default=None, type=click.Path(), help="Override the sample JSON path")
@click.option(
    "--require-frozen/--no-require-frozen",
    default=True,
    help="Refuse to run unless scale-out is suspended and capacity is pinned",
)
@click.option(
    "--require-unbounded-queue/--no-require-unbounded-queue",
    default=True,
    help=(
        "Refuse to run when the container bounds its admission queue. A bounded "
        "container sheds before the SLO breaks, so the ladder would measure "
        "MAX_QUEUE_DEPTH rather than Q_max."
    ),
)
@click.option("--pin-to", default=1, type=int, help="Instances to pin for the run")
@click.option(
    "--cloudwatch/--no-cloudwatch",
    "cloudwatch_join",
    default=True,
    help=(
        "Join server-side metrics after a settle delay (~2 min). This is where the "
        "deployed threshold's units come from; 10s datapoints retain 3 hours, so "
        "skipping it cannot be undone later."
    ),
)
@click.option(
    "--output",
    default=None,
    type=click.Path(),
    help="Write the artifact JSON here. Defaults to artifacts/, named for the configuration.",
)
@click.option(
    "--no-save",
    is_flag=True,
    default=False,
    help="Discard the artifact. For a smoke test you do not intend to keep.",
)
@click.option("--events", default=None, type=click.Path(), help="Write per-request JSONL here")
@click.option(
    "--dry-run",
    is_flag=True,
    help="Print the schedule and exit. Makes no AWS calls, so S is assumed, not measured.",
)
@click.option(
    "--dry-run-s",
    default=0.25,
    type=float,
    help="Service time assumed by --dry-run, in seconds",
)
def qmax(
    model: str,
    region: str,
    variant: str,
    voice: str | None,
    concurrency: str,
    slo_ms: int,
    hold_s: float,
    measure_window_s: float,
    settle_between_steps_s: float,
    runs: int,
    transport: str,
    seed: int,
    max_samples: int,
    samples: str | None,
    require_frozen: bool,
    require_unbounded_queue: bool,
    pin_to: int,
    cloudwatch_join: bool,
    output: str | None,
    no_save: bool,
    events: str | None,
    dry_run: bool,
    dry_run_s: float,
) -> None:
    """Measure Q_max: the highest concurrency that still meets the SLO.

    Walks a closed-loop ladder, holding queued-plus-executing at exactly N per rung
    -- each of N workers issues its next request only when its previous one returns.
    Q_max is the highest rung whose p95 first-byte time stayed inside ``--slo-ms``,
    so the answer is a concurrency that was actually run rather than one inferred
    from a rate. Only the trailing window of each rung is measured, which excludes
    warm-up and the drain behind it.

    Autoscaling is frozen and capacity pinned for the whole run, restored on exit
    including on Ctrl-C. Q_max is a *per-instance* number: if the fleet grows mid-run
    the result is silently N x Q_max.

    The ladder must also bracket its answer from above -- one rung past the crossing
    -- or Q_max is only a lower bound, and both derived thresholds inherit that.

    ``--transport bidi`` measures the protocol production is configured for. It is a
    separate measurement, not a refinement: the containers serialize differently on
    it, so expect a lower Q_max on kokoro, which holds its inference lock across a
    whole bidi session.
    """
    from tts_bench import qmax as qmax_mod
    from tts_bench.invoke import resolve_endpoint

    rungs = _parse_ints(concurrency, flag="--concurrency")

    # Resolve the endpoint before anything else. A typo or a managed model then
    # fails as a usage error rather than a traceback out of `measure`, and it
    # fails on --dry-run too — which is the run people use to check a command
    # before committing 40 minutes and an endpoint freeze to it.
    try:
        endpoint = resolve_endpoint(model)
    except ValueError as exc:
        raise click.BadParameter(str(exc), param_hint="--model") from exc

    if measure_window_s > hold_s:
        raise click.BadParameter(
            f"--measure-window ({measure_window_s}) cannot exceed --hold ({hold_s}); it is the "
            "trailing part of a rung, not an addition to it"
        )
    if runs < 1:
        raise click.BadParameter("--runs must be at least 1")
    if slo_ms <= 0:
        raise click.BadParameter("--slo-ms must be positive; it is the line Q_max is defined by")
    if any(rung <= 0 for rung in rungs):
        raise click.BadParameter(
            "--concurrency rungs must be positive; N is a count of outstanding requests",
            param_hint="--concurrency",
        )

    estimate_s = qmax_mod.total_duration_s(
        concurrencies=rungs,
        hold_s=hold_s,
        settle_between_steps_s=settle_between_steps_s,
        runs=runs,
    )

    # Warned here rather than only at report time: a missing rung cannot be recovered
    # without re-running the ladder, and by then the 40 minutes are already spent.
    missing = [rung for rung in (1, *qmax_mod.RECOVERY_RUNGS) if rung not in set(rungs)]
    if missing:
        click.echo(
            f"WARNING: --concurrency omits {missing}. Rung 1 is the FirstChunkLatencyP95 "
            f"alarm's service-time reference; rungs {list(qmax_mod.RECOVERY_RUNGS)} are the "
            "pair `ttotal` watches a p95 halve between. Whichever is missing, that consumer "
            "has to refuse the artifact."
        )

    if dry_run:
        click.echo(
            f"{model} ({endpoint}): {len(set(rungs))} rung(s) x {runs} run(s), "
            f"SLO p95 {slo_ms}ms, assuming S={dry_run_s:.3f}s (not measured)"
        )
        click.echo(f"{'run':>4} {'step':>5} {'conc':>6} {'hold_s':>8} {'requests':>9}")
        for row in qmax_mod.dry_run_plan(
            concurrencies=rungs, hold_s=hold_s, s_mean_s=dry_run_s, runs=runs
        ):
            click.echo(
                f"{int(row['run_index']):>4} {int(row['step_index']):>5} "
                f"{int(row['concurrency']):>6} {row['hold_s']:>8.0f} "
                f"{row.get('expected_requests', 0.0):>9.0f}"
            )
        click.echo(
            f"\nEstimated wall clock: {estimate_s / 60:.0f} min "
            f"(+~2 min CloudWatch settle). `requests` is a forecast from the assumed S, "
            "not a schedule: a closed loop completes N/S per second, so a wrong S "
            "mis-predicts the count without moving a single rung."
        )
        click.echo("Dry run: no AWS calls made, nothing frozen.")
        return

    texts = _load_texts(samples, max_samples)
    click.echo(
        f"{model} ({endpoint}): {len(set(rungs))} rung(s) x {runs} run(s), "
        f"~{estimate_s / 60:.0f} min, SLO p95 {slo_ms}ms, {len(texts)} texts, "
        f"transport={transport}, frozen={require_frozen}"
    )
    if not require_frozen:
        click.echo(
            "WARNING: --no-require-frozen. If the fleet grows mid-run this measures N x Q_max; "
            "the artifact will record frozen=false."
        )
    if not cloudwatch_join:
        click.echo(
            "WARNING: --no-cloudwatch. No rung will record ConcurrentRequestsPerModel / "
            "Maximum, which is the statistic the deployed threshold is compared against, so "
            "`plan` will have no unit conversion. 10s datapoints retain 3 hours — this "
            "cannot be backfilled after the run."
        )

    from contextlib import nullcontext

    from tts_bench.fixture import FixtureError
    from tts_bench.loadgen import JsonlWriter

    try:
        # JsonlWriter opens on enter and flushes every event, so a run killed at
        # the interesting moment has still written the interesting moment.
        with JsonlWriter(events) if events else nullcontext() as writer:
            report = qmax_mod.measure(
                model=model,
                texts=texts,
                slo_ms=slo_ms,
                region=region,
                variant=variant,
                voice=voice,
                concurrencies=rungs,
                hold_s=hold_s,
                measure_window_s=measure_window_s,
                settle_between_steps_s=settle_between_steps_s,
                runs=runs,
                seed=seed,
                transport=transport,
                require_frozen=require_frozen,
                require_unbounded_queue=require_unbounded_queue,
                pin_to=pin_to,
                cloudwatch_join=cloudwatch_join,
                event_sink=writer,
            )
    except (qmax_mod.QMaxError, FixtureError) as exc:
        # Both are the tool refusing to produce a number it cannot stand behind,
        # so they exit cleanly with the reason. A traceback would read as a bug
        # rather than as the guard doing its job.
        raise click.ClickException(str(exc)) from exc
    if events:
        click.echo(f"Events: {events}")

    click.echo(
        f"\nQ_max ladder for {report.model_name} on {report.instance_type} "
        f"via {report.transport}, against p95 first byte <= {report.slo_ms}ms:"
    )
    # `cw_max` is CloudWatch's Maximum statistic and `cl_mean` the client's own mean
    # in-flight. Both columns, side by side, because they are different quantities that
    # ran from 9.8x apart to 1.35x apart over one ladder — printing only one is how a
    # client-measured occupancy came to be deployed as a server-side threshold.
    click.echo(
        f"{'run':>4} {'conc':>6} {'p95_ms':>8} {'slo':>5} {'rps':>7} "
        f"{'cl_mean':>8} {'cw_max':>7}  note"
    )
    for step in report.steps:
        if step.usable:
            note = "" if step.meets_slo else "over SLO"
        else:
            note = step.unusable_reason or "unusable"
        # "n/a" rather than 0 in every derived column: a zero p95 reads as an
        # instantaneous response, when what happened is that the rung measured nothing.
        click.echo(
            f"{step.run_index:>4} {step.concurrency:>6} "
            f"{f'{step.ttfab_p95_ms:.0f}' if step.ttfab_p95_ms is not None else 'n/a':>8} "
            f"{('ok' if step.meets_slo else 'OVER') if step.usable else '-':>5} "
            f"{step.achieved_rps:>7.2f} "
            f"{f'{step.concurrency_mean:.2f}' if step.concurrency_mean is not None else 'n/a':>8} "
            f"{f'{step.server_concurrency_peak:.1f}' if step.server_concurrency_peak is not None else 'n/a':>7}"  # noqa: E501
            f"  {note}"
        )

    click.echo(
        f"\nQ_max: {report.q_max} concurrent per instance "
        f"(p95 {report.ttfab_p95_at_q_max_ms:.0f}ms there, "
        f"{report.slo_ms - report.ttfab_p95_at_q_max_ms:.0f}ms of SLO left over)"
    )
    if not report.q_max_bracketed:
        click.echo(
            "  NOTE: LOWER BOUND — no rung above it was measured to actually miss the SLO. "
            f"Extend --concurrency past {report.q_max} to bracket it. Both derived "
            "thresholds are fractions of Q_max, so an unbracketed value scales out earlier "
            "than necessary rather than later."
        )
    if report.runs_contributing < report.runs:
        click.echo(
            f"  NOTE: only {report.runs_contributing}/{report.runs} run(s) produced an "
            "answer, so the spread below is not a run-to-run agreement — check the notes "
            "in the ladder above."
        )
    if report.runs_contributing > 1:
        # The minimum, not the median: the median of two rungs is a concurrency no run
        # tested, while the minimum is both a real rung and the conservative one.
        click.echo(
            f"  per-run: {list(report.q_max_per_run)} (spread {report.q_max_spread:.0%}; "
            "Q_max is the minimum, which is a rung that was actually measured)"
        )

    c1 = report.ttfab_p95_at_c1_ms
    if c1 is None:
        click.echo(
            "\np95 at c=1: not measured — the FirstChunkLatencyP95 alarm has no service-time "
            "threshold to read, and an alarm threshold has to come from somewhere real."
        )
    else:
        click.echo(f"\np95 at c=1: {c1:.0f}ms (FirstChunkLatencyP95 alarm threshold)")
    pair = [(rung, report.ttfab_p95_at(rung)) for rung in qmax_mod.RECOVERY_RUNGS]
    if all(value is not None for _, value in pair):
        click.echo(
            "Recovery pair for ttotal: "
            + ", ".join(f"c={rung} -> {value:.0f}ms" for rung, value in pair)
            + " (a probe held at the higher rung halves into the lower one when a second "
            "instance takes traffic)"
        )
    else:
        click.echo(
            f"Recovery pair for ttotal: incomplete — "
            f"{[rung for rung, value in pair if value is None]} produced no usable p95, so "
            "`ttotal` will refuse this artifact."
        )
    click.echo(
        f"S (lowest rung): mean {report.s_mean_s * 1000:.0f}ms p95 {report.s_p95_s * 1000:.0f}ms "
        "— includes the client round trip, so it over-states server-side work slightly"
    )

    if report.ladder_truncated_at is not None:
        click.echo(
            f"NOTE: ladder stopped at step {report.ladder_truncated_at} after repeated "
            "saturation; higher rungs were never offered."
        )
    if report.unbounded_queue is not True:
        click.echo(
            "WARNING: the container's queue bound was "
            + ("not verified" if report.unbounded_queue is None else "found to be SET")
            + ". Q_max is the depth at which the SLO breaks, so a container that sheds "
            "first measures its own MAX_QUEUE_DEPTH instead."
        )
    if not report.trustworthy:
        click.echo(
            f"WARNING: not safe to read as per-instance — frozen={report.frozen}, "
            f"instance counts seen {list(report.instance_counts_observed)}"
        )

    if no_save:
        click.echo(f"\n--no-save: nothing written. Configuration measured: {report.config_slug}")
        click.echo(f"\nNext: tts-bench ttotal --model {model} --qmax <qmax artifact>")
    else:
        # Defaulted rather than optional: this run costs 40+ minutes and an endpoint
        # freeze, and printing a suggested filename after the fact does not bring the
        # measurement back. The slug in the name is what keeps two configurations'
        # ladders apart -- they are two different measurements, not two attempts at one.
        destination = (
            Path(output)
            if output
            else _default_artifact_path("qmax", model, transport, report.config_slug)
        )
        _warn_if_replacing_another_config(destination, report.config_slug)
        _save_artifact(destination, report.model_dump_json(indent=2))
        click.echo(f"\nArtifact: {destination}")
        click.echo(f"Configuration measured: {report.config_slug}")
        # Echo the path back rather than "<artifact>": the next command needs this exact
        # file, and the slug in it is not something to retype from memory.
        click.echo(f"\nNext: tts-bench ttotal --model {model} --qmax {destination}")
        click.echo(f"Then: tts-bench plan --qmax {destination} --ttotal <ttotal artifact>")


@main.command()
@click.option(
    "--model",
    required=True,
    help=(
        "Model name, e.g. kokoro-82m. Singular and required: this measures one "
        "configuration, and it freezes and then scales that endpoint to do it."
    ),
)
@click.option(
    "--qmax",
    "qmax_path",
    required=True,
    type=click.Path(exists=True),
    help=(
        "A qmax artifact from THIS configuration. Required: recovery is judged by the "
        "probe's p95 halving between two of its rungs, so without it there is no "
        "measured level to compare against."
    ),
)
@click.option("--region", default="us-east-1", help="AWS region every client is built in")
@click.option("--variant", default="primary", help="Production variant on the endpoint")
@click.option("--voice", default=None, help="Override the model's default voice")
@click.option(
    "--trigger",
    type=click.Choice([TTOTAL_TRIGGER_FORCE_DESIRED]),
    default=TTOTAL_TRIGGER_FORCE_DESIRED,
    help=(
        "How to cause the scale-out. Only force-desired: driving load past the deployed "
        "policy let the policy add instances mid-measurement."
    ),
)
@click.option(
    "--probe-concurrency",
    default=None,
    type=int,
    help=(
        "Concurrency the probe holds throughout. Must be an even rung on the ladder, and "
        "so must its half. Defaults to 10."
    ),
)
@click.option(
    "--warmup",
    "warmup_s",
    default=None,
    type=float,
    help=(
        "Seconds the probe runs before the trigger, so p95 has settled. Not part of "
        "T_total -- the clock starts when capacity is requested."
    ),
)
@click.option(
    "--allow-config-mismatch",
    is_flag=True,
    default=False,
    help="Proceed even if --qmax was taken on a different deployed configuration",
)
@click.option("--max-wait", "max_wait_s", default=1500.0, type=float, help="Scale-out timeout")
@click.option(
    "--settle",
    "settle_s",
    default=180.0,
    type=float,
    help="Seconds the probe is held past the event, so recovery can be bounded",
)
@click.option(
    "--poll-interval",
    "poll_interval_s",
    default=10.0,
    type=float,
    help="Seconds between endpoint polls; bounds how precisely a stage boundary is placed",
)
@click.option(
    "--transport",
    type=click.Choice(["response-stream", "bidi"]),
    default="bidi",
    help="Wire protocol the probe uses. Must match the --qmax ladder's.",
)
@click.option(
    "--seed", default=1234, type=int, help="Text-pool shuffle seed; keeps runs comparable"
)
@click.option("--max-samples", default=50, type=int, help="Texts drawn into the pool")
@click.option("--samples", default=None, type=click.Path(), help="Override the sample JSON path")
@click.option(
    "--output",
    default=None,
    type=click.Path(),
    help="Write the artifact JSON here. Defaults to artifacts/, named for the configuration.",
)
@click.option(
    "--no-save",
    is_flag=True,
    default=False,
    help="Discard the artifact. For a smoke test you do not intend to keep.",
)
@click.option("--events", default=None, type=click.Path(), help="Write per-request JSONL here")
def ttotal(
    model: str,
    qmax_path: str,
    region: str,
    variant: str,
    voice: str | None,
    trigger: str,
    probe_concurrency: int | None,
    warmup_s: float | None,
    allow_config_mismatch: bool,
    max_wait_s: float,
    settle_s: float,
    poll_interval_s: float,
    transport: str,
    seed: int,
    max_samples: int,
    samples: str | None,
    output: str | None,
    no_save: bool,
    events: str | None,
) -> None:
    """Measure T_total: the lag from requesting capacity to that capacity serving traffic.

    Freezes the endpoint at one instance, holds a saturating closed-loop probe, then raises
    DesiredInstanceCount directly and attributes the lag stage by stage — provisioning,
    container startup, in-service, and traffic recovery — each from the API that timestamps
    it. The clock starts at our own capacity call, so the probe's warm-up is not counted.

    T_total is the input the capacity plan is most sensitive to: it sets how much standing
    headroom a surge needs. It is also the number most often guessed.

    Recovery is a halving, not a threshold crossing. The endpoint routes
    LEAST_OUTSTANDING_REQUESTS, so a second instance splits the probe in half and its p95
    drops toward the ladder's own half-concurrency value — that is what "the new instance
    is serving" means from outside, since SageMaker never says which instance served a
    request. Both rungs must be on the --qmax ladder or this refuses to run.

    The deployed policy is suspended throughout, so its detection lag is NOT measured: it
    is bounded separately and reported as t_total_with_policy_bound_s. Autoscaling and the
    starting instance count are both restored on exit, Ctrl-C included.
    """
    from tts_bench import qmax as qmax_mod
    from tts_bench import ttotal as ttotal_mod
    from tts_bench.fixture import FixtureError
    from tts_bench.invoke import resolve_endpoint, resolve_voice
    from tts_bench.types import QMaxReport

    try:
        endpoint = resolve_endpoint(model)
    except ValueError as exc:
        raise click.BadParameter(str(exc), param_hint="--model") from exc

    if probe_concurrency is None:
        probe_concurrency = ttotal_mod.PROBE_CONCURRENCY
    if warmup_s is None:
        warmup_s = ttotal_mod.DEFAULT_WARMUP_S
    if warmup_s < 0:
        raise click.BadParameter("--warmup cannot be negative", param_hint="--warmup")
    if settle_s <= 0:
        raise click.BadParameter(
            "--settle must be positive: recovery is judged from traffic on the far side of "
            "the scale event, and with none the run stops at in_service",
            param_hint="--settle",
        )

    try:
        qmax_report = QMaxReport.model_validate_json(Path(qmax_path).read_text())
    except ValueError as exc:
        # A qmax artifact is a QMaxReport dump; anything else is almost always a ttotal or
        # plan artifact passed by mistake, and the pydantic error names the missing fields.
        raise click.BadParameter(
            f"{qmax_path} is not a qmax artifact: {exc}", param_hint="--qmax"
        ) from exc

    ladder = qmax_report.ladder_p95_ms
    if qmax_report.transport != transport:
        # Not a warning: the containers hold their inference lock differently per transport,
        # so a p95 measured on one is not the level a probe on the other should reach. The
        # halving would be judged against the wrong number in both directions.
        raise click.UsageError(
            f"--qmax was measured on transport {qmax_report.transport!r} but this run uses "
            f"{transport!r}. The recovery levels come from that ladder, and p95 does not "
            f"transfer between transports. Re-run with --transport {qmax_report.transport}."
        )
    # A configuration mismatch means the ladder describes other hardware or other serving
    # code, so both recovery levels are wrong and nothing in the output would look it.
    _require_matching_config(
        json.loads(Path(qmax_path).read_text()),
        endpoint=endpoint,
        region=region,
        variant=variant,
        artifact_path=qmax_path,
        flag="--qmax",
        allow_mismatch=allow_config_mismatch,
    )

    # Checked here as well as inside measure(): this is the difference between a usage
    # error now and a FixtureError after the freeze has already been established.
    try:
        expected_ms, recovered_ms = ttotal_mod.recovery_references(
            ladder, probe_concurrency=probe_concurrency
        )
    except ttotal_mod.TTotalError as exc:
        raise click.UsageError(str(exc)) from exc

    if qmax_report.q_max is not None and probe_concurrency >= qmax_report.q_max:
        click.echo(
            f"WARNING: --probe-concurrency {probe_concurrency} is at or above the measured "
            f"Q_max of {qmax_report.q_max}, so the probe's own requests will breach the "
            f"{qmax_report.slo_ms}ms SLO for the whole run. The probe only has to be "
            "saturating, not overloaded."
        )

    click.echo(
        f"{model} ({endpoint}): trigger={trigger}, probe N={probe_concurrency} "
        f"({expected_ms:.0f}ms expected), recovered at <= {recovered_ms:.0f}ms "
        f"(N={probe_concurrency // 2} rung x {ttotal_mod.RECOVERY_TOLERANCE:.2f}), "
        f"transport={transport}"
    )
    click.echo(
        f"Autoscaling is suspended for the run, so the policy's detection lag is bounded at "
        f"{ttotal_mod.POLICY_LAG_BOUND_S:.0f}s rather than measured. `plan` reads the bounded "
        "figure; t_total_s alone is the capacity half."
    )
    click.echo(
        f"Ladder: {qmax_path} (Q_max {qmax_report.q_max}, {len(ladder)} usable rung(s), "
        f"{qmax_report.config_slug})"
    )
    # The mutating call is one UpdateEndpointWeightsAndCapacities, and it is reversed in the
    # same finally that thaws. Said out loud because this command changes production
    # capacity, briefly, on purpose.
    click.echo(
        f"This will suspend autoscaling on {endpoint}, raise DesiredInstanceCount by 1, wait "
        f"up to {max_wait_s / 60:.0f} min, then restore both."
    )

    from contextlib import nullcontext

    from tts_bench.loadgen import JsonlWriter

    try:
        with JsonlWriter(events) if events else nullcontext() as writer:
            report = ttotal_mod.measure(
                model_name=model,
                endpoint=endpoint,
                variant=variant,
                region=region,
                ladder_p95_ms=ladder,
                texts=_load_texts(samples, max_samples),
                voice=resolve_voice(model, voice),
                trigger=trigger,
                probe_concurrency=probe_concurrency,
                warmup_s=warmup_s,
                max_wait_s=max_wait_s,
                settle_s=settle_s,
                poll_interval_s=poll_interval_s,
                transport=transport,
                seed=seed,
                event_sink=writer,
            )
    except (ttotal_mod.TTotalError, qmax_mod.QMaxError, FixtureError) as exc:
        # All three are the tool refusing to report a lag it did not observe. A traceback
        # would read as a bug rather than as the guard doing its job.
        raise click.ClickException(str(exc)) from exc

    click.echo("")
    click.echo(ttotal_mod.render_text(report))

    if events:
        click.echo(f"\nEvents: {events}")

    if no_save:
        click.echo(f"\n--no-save: nothing written. Configuration measured: {report.config_slug}")
        click.echo(f"\nNext: tts-bench plan --qmax {qmax_path} --ttotal <ttotal artifact>")
    else:
        # Defaulted for the same reason as qmax: this run costs a real scale-out and, on a
        # timeout, most of max_wait_s of offered load. The trigger is in the name so a
        # future mode lands beside this one rather than overwriting it.
        destination = (
            Path(output)
            if output
            else _default_artifact_path("ttotal", model, trigger, report.config_slug)
        )
        _warn_if_replacing_another_config(destination, report.config_slug)
        _save_artifact(destination, json.dumps(report.to_dict(), indent=2))
        click.echo(f"\nArtifact: {destination}")
        click.echo(f"Configuration measured: {report.config_slug}")
        click.echo(f"\nNext: tts-bench plan --qmax {qmax_path} --ttotal {destination}")


@main.command()
@click.option(
    "--qmax",
    "qmax_path",
    required=True,
    type=click.Path(exists=True),
    help=(
        "A qmax artifact: the measured Q_max, the SLO it was measured against, and S. "
        "Both scaling thresholds are fractions of the Q_max in here."
    ),
)
@click.option(
    "--ttotal",
    "ttotal_path",
    default=None,
    type=click.Path(exists=True),
    help=(
        "A ttotal artifact: the scaling lag, by stage. Omit only with --assume-t-total, "
        "since a plan with no lag cannot say whether the scale-out threshold fires early "
        "enough to survive one."
    ),
)
@click.option(
    "--peak-rps",
    default=None,
    type=float,
    help="Expected peak arrival rate. Give this or --peak-streams.",
)
@click.option(
    "--trough-rps",
    default=None,
    type=float,
    help="Expected quiet-hours rate; sets min_instances, which reserved capacity pays for",
)
@click.option(
    "--peak-streams",
    default=None,
    type=float,
    help=(
        "Expected peak concurrent sessions. The honest input for bidi traffic, where "
        "one session is not one request; bypasses the lambda x S conversion."
    ),
)
@click.option("--trough-streams", default=None, type=float, help="Expected quiet-hours sessions")
@click.option(
    "--max-scaling-per-t-total",
    default=1.25,
    type=float,
    help=(
        "Surge ratio to survive within one T_total: 1.25 means traffic may grow 25% while "
        "a replacement instance arrives. Both thresholds derive from it — with h = ratio-1, "
        "C_scale_max = (1-h) x Q_max and C_scale_min = (1-2h) x Q_max. Not measurable "
        "without production history, so it is chosen."
    ),
)
@click.option(
    "--ttfab-slo-ms",
    default=None,
    type=int,
    help=(
        "End-to-end first-byte SLO: queue wait plus service. Defaults to whatever the "
        "qmax ladder was measured against, which is the only value it can be read at — "
        "Q_max is *defined* by the SLO, so a mismatch is refused rather than converted."
    ),
)
@click.option(
    "--min-floor",
    "min_instances_floor",
    default=1,
    type=int,
    help="Never plan below this, whatever the trough says",
)
@click.option(
    "--assume-t-total",
    "assume_t_total_s",
    default=None,
    type=float,
    help=(
        "Plan against a stated T_total when the ttotal run never spanned one. "
        "Labelled an assumption in the output, because it is one."
    ),
)
@click.option(
    "--allow-config-mismatch",
    is_flag=True,
    default=False,
    help="Pair a Q_max ladder and a T_total lag measured on different configurations",
)
@click.option(
    "--ceiling-s",
    default=None,
    type=float,
    help="Invocation ceiling to judge W_max against. Defaults to SageMaker's 60s.",
)
@click.option(
    "--output",
    default=None,
    type=click.Path(),
    help=(
        "Write the plan as JSON here. Nothing is written unless asked — unlike a "
        "measurement, this run is cheap to repeat."
    ),
)
def plan(
    qmax_path: str,
    ttotal_path: str | None,
    peak_rps: float | None,
    trough_rps: float | None,
    peak_streams: float | None,
    trough_streams: float | None,
    max_scaling_per_t_total: float,
    ttfab_slo_ms: int | None,
    min_instances_floor: int,
    assume_t_total_s: float | None,
    allow_config_mismatch: bool,
    ceiling_s: float | None,
    output: str | None,
) -> None:
    """Turn Q_max and T_total into a scaling configuration. Reads artifacts, touches no AWS.

    Composes the measured Q_max and T_total with a chosen SLO and surge ratio into the
    numbers ModelEndpointConfig needs — scaling_target_value, scale_in_threshold,
    queue_max_depth, min_instances, max_instances — and prints them as a paste-ready
    block. That closes the loop the tool chain exists for: before this, the path from a
    measurement to a deployed policy ran through hand-arithmetic in a comment.

    Six variables in total. SLO and --max-scaling-per-t-total are chosen; Q_max and
    T_total are measured; C_scale_max and C_scale_min are arithmetic on those four. With
    h = ratio - 1: C_scale_max = (1-h) x Q_max, C_scale_min = (1-2h) x Q_max.

    The thresholds print twice — as the client occupancy that was measured, and in the
    CloudWatch Maximum units the deployed alarm actually reads, with the measured ratio
    between them. Those differed by 1.35x to 9.8x on one kokoro ladder, and deploying
    the unconverted figure is how a threshold no traffic can satisfy reached the endpoint.

    Two known limits of the simple rule are computed rather than argued: surge_survival
    simulates P(the queue reaches Q_max) while one T_total elapses from C_scale_max, and
    scale_in_safety says whether removing an instance at C_scale_min lands the survivors
    back over C_scale_max.

    Refuses three things outright: pairing artifacts whose configuration fingerprints
    differ, reading a Q_max ladder against an SLO other than the one it was measured
    against, and an SLO that does not fit inside SageMaker's 60s invocation ceiling.
    """
    from shared.capacity import SAGEMAKER_INVOCATION_CEILING_S
    from tts_bench import scale_report
    from tts_bench.planner import (
        PlannerError,
        TTotalStages,
        measured_from_artifacts,
        plan_one,
    )
    from tts_bench.types import QMaxReport, Scenario

    if peak_rps is None and peak_streams is None:
        raise click.UsageError(
            "one of --peak-rps or --peak-streams is required. The fleet size is what a "
            "plan is for, and there is nothing to size it from without a stated peak."
        )

    try:
        qmax_report = QMaxReport.model_validate_json(Path(qmax_path).read_text())
    except (OSError, ValueError) as exc:
        raise click.ClickException(
            f"could not read {qmax_path} as a qmax artifact: {exc}. It must be the JSON "
            "written by `tts-bench qmax`."
        ) from exc

    # A ttotal artifact is the normal path; --assume-t-total covers the case the plan
    # anticipates, where the second instance would not place in this account at all.
    if ttotal_path:
        try:
            stages = TTotalStages.from_artifact(json.loads(Path(ttotal_path).read_text()))
        except (OSError, ValueError) as exc:
            raise click.ClickException(
                f"could not read {ttotal_path} as a ttotal artifact: {exc}"
            ) from exc
    elif assume_t_total_s is None:
        raise click.UsageError(
            "--ttotal is required, or --assume-t-total to plan against a stated lag. "
            "T_total is what decides whether the scale-out threshold fires early enough "
            "to survive a surge, so there is no plan without it."
        )
    else:
        # No artifact means no fingerprint to check and no stages to read, so the assumed
        # total is used whole. from_artifact({}) is the honest shape for that: every stage
        # missing, nothing measured.
        stages = TTotalStages.from_artifact({})

    try:
        planner_input = measured_from_artifacts(
            qmax_report,
            stages,
            allow_config_mismatch=allow_config_mismatch,
            assume_t_total_s=assume_t_total_s,
            require_pairing=bool(ttotal_path),
        )
    except PlannerError as exc:
        raise click.ClickException(str(exc)) from exc

    # Default to the ladder's own SLO rather than to 3000: the artifact records the line
    # Q_max was judged against, and defaulting to a constant would refuse a perfectly
    # good ladder measured at a different one. An explicit --ttfab-slo-ms that disagrees
    # still hits plan_one's refusal, which is the point of passing it explicitly.
    if ttfab_slo_ms is None:
        ttfab_slo_ms = qmax_report.slo_ms

    try:
        scenario = Scenario(
            peak_rps=peak_rps,
            trough_rps=trough_rps,
            peak_streams=peak_streams,
            trough_streams=trough_streams,
            max_scaling_per_t_total=max_scaling_per_t_total,
            ttfab_slo_ms=ttfab_slo_ms,
            min_instances_floor=min_instances_floor,
        )
    except ValueError as exc:
        raise click.BadParameter(str(exc)) from exc

    try:
        result = plan_one(
            planner_input,
            scenario,
            provision_s=stages.provision_s,
            policy_bound_s=stages.policy_bound_s,
            ceiling_s=ceiling_s if ceiling_s is not None else SAGEMAKER_INVOCATION_CEILING_S,
        )
    except PlannerError as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo(scale_report.render_plan(result, stages))
    _warn_if_config_disagrees(result)

    if output:
        _save_artifact(Path(output), json.dumps(scale_report.plan_to_dict(result), indent=2))
        click.echo(f"Plan: {output}")

    # Exit non-zero on an infeasible plan so this can gate a deploy.
    if result.infeasible:
        raise SystemExit(1)


def _warn_if_config_disagrees(plan: ScalingPlan) -> None:
    """Say so when the deployed scaling fields are not what this plan derives.

    ``queue_max_depth`` and ``scaling_target_value`` are the two derived numbers
    ``config.py`` stores rather than computes, because both need a measurement. So they
    can go stale silently, and both did: 296 was carried from a 20s W_max no SLO
    justified, and 0.713 was a client occupancy deployed against a server-peak statistic.

    A warning rather than a refusal, and printed after the plan. The plan *is* the
    answer, and the whole point of running it is to find out the deployed values are
    wrong; exiting non-zero here would fail the command that just told you what to fix.
    The paste-ready config block above already carries the right numbers.
    """
    try:
        from speech_infra.config import TTS_MODEL_CONFIGS
    except ImportError:  # pragma: no cover - depends on install layout
        return

    config = next((c for c in TTS_MODEL_CONFIGS.values() if c.endpoint_name == plan.endpoint), None)
    if config is None:
        return

    if config.queue_max_depth != plan.queue_max_depth:
        if config.queue_max_depth == 0:
            click.echo(
                f"\nNOTE: {config.model_name} has queue_max_depth=0 (unbounded) deployed; "
                f"the measured Q_max is {plan.queue_max_depth}. Paste the block above into "
                "config.py to bound it."
            )
        else:
            click.echo(
                f"\nWARNING: {config.model_name} has queue_max_depth="
                f"{config.queue_max_depth} deployed, but Q_max measured against a "
                f"{plan.measured.slo_ms}ms end-to-end SLO is {plan.queue_max_depth}. The "
                "deployed value admits requests it can only serve late. Paste the block "
                "above into config.py."
            )

    # Compared in CloudWatch units on both sides: the deployed value is what the alarm
    # reads, so comparing it against the client occupancy would report a mismatch of
    # exactly the size of the conversion and call it drift.
    target = plan.c_scale_max_in_cw_units
    if target is not None and abs(config.scaling_target_value - target) > 0.01:
        click.echo(
            f"\nWARNING: {config.model_name} has scaling_target_value="
            f"{config.scaling_target_value} deployed; this plan derives {target:.3f} "
            f"(C_scale_max {plan.c_scale_max:.2f} x {plan.cw_units_ratio:.2f} into "
            "ConcurrentRequestsPerModel/Maximum units). Paste the block above into config.py."
        )


def _warn_if_replacing_another_config(destination: Path, slug: str) -> None:
    """Say so when an artifact from a *different* configuration is about to be replaced.

    The realistic mistake in a per-configuration workflow: re-run after a redeploy,
    reuse the previous ``--output``, and the old hardware's curve is gone. Cheap to
    warn, and the previous configuration is worth naming because it is usually the
    comparison the operator wanted.

    Takes the slug rather than a report so ``cmax`` and ``ttotal`` share one guard —
    both write per-configuration artifacts and both can clobber the previous one.
    """
    if not destination.exists():
        return
    try:
        previous = json.loads(destination.read_text()).get("deployed_config")
    except (OSError, json.JSONDecodeError):
        return
    if not isinstance(previous, dict) or not previous:
        return

    from tts_bench.fixture import DeployedConfig

    old = DeployedConfig.from_dict(previous)
    if old.slug == slug:
        return
    click.echo(
        f"WARNING: {destination} holds a measurement of a different configuration "
        f"({old.slug}, {old.instance_type}) and is being overwritten with "
        f"{slug}. Keep the old one under its own name if you want to "
        "compare the two."
    )


def _require_matching_config(
    artifact: dict[str, object],
    *,
    endpoint: str,
    region: str,
    variant: str,
    artifact_path: str | None,
    flag: str = "--qmax",
    allow_mismatch: bool,
) -> None:
    """Refuse to reuse a measurement taken against a different configuration.

    ``Q_max`` and ``T_total`` are properties of a configuration — the GPU, the serving
    code, the container knobs — not of a model. Replaying an artifact from one
    configuration onto another produces a plan for a fleet that does not exist, and
    nothing about the output would look wrong.

    An artifact with no fingerprint predates this check and is treated as a mismatch:
    accepting it silently is the hole this closes.
    """
    from tts_bench.fixture import DeployedConfig, FixtureError, describe_deployed_config

    label = f"{flag} {artifact_path}" if artifact_path else "the measured artifact"
    raw = artifact.get("deployed_config")
    recorded = DeployedConfig.from_dict(raw if isinstance(raw, dict) else {})

    try:
        live = describe_deployed_config(endpoint, region=region, variant=variant)
    except FixtureError as exc:
        # Not fatal: the run itself may still be viable, and refusing it because one
        # extra describe call failed would be its own kind of wrong.
        click.echo(f"WARNING: could not verify the deployed configuration of {endpoint}: {exc}")
        return

    if not raw:
        message = (
            f"{label} records no deployed configuration, so there is no way to tell "
            f"whether it describes the {live.instance_type} endpoint deployed now. It "
            "predates configuration fingerprinting — re-measure it, or pass "
            "--allow-config-mismatch to proceed anyway."
        )
        if not allow_mismatch:
            raise click.ClickException(message)
        click.echo(f"WARNING: {message}")
        return

    try:
        recorded.assert_matches(live, allow_mismatch=allow_mismatch, artifact_label=label)
    except FixtureError as exc:
        raise click.ClickException(str(exc)) from exc


def _expected_scaling() -> dict[str, ExpectedScaling]:
    """Build the drift audit's source of truth from the CDK config registry.

    Imported lazily and only here. ``speech_infra.config`` pulls in
    ``aws-cdk-lib``, so it is deliberately not a ``tts-bench`` dependency —
    resolving it at the CLI boundary keeps ``tts_bench.observe`` importable
    without it, and keeps the measurement path (``cmax``, ``ttotal``) free of a
    CDK import it has no use for.

    ``drift`` is the one command that cannot work without it: comparing deployed
    capacity to intended capacity requires the intent, and only ``config.py``
    records it.

    Raises:
        click.ClickException: If ``speech_infra`` is not installed.
    """
    try:
        from speech_infra.config import TTS_MODEL_CONFIGS
    except ImportError as exc:  # pragma: no cover - depends on install layout
        raise click.ClickException(
            "drift compares deployed scaling against speech_infra.config, which is not "
            "installed. Run from the workspace root (`uv run tts-bench drift`), or install "
            "the speech-infra package."
        ) from exc

    from tts_bench.observe import ExpectedScaling

    return {
        config.endpoint_name: ExpectedScaling(
            endpoint=config.endpoint_name,
            min_instances=config.min_instances,
            max_instances=config.max_instances,
            scaling_enabled=config.scaling_enabled,
        )
        for config in TTS_MODEL_CONFIGS.values()
    }


@main.command()
@click.option("--region", default="us-east-1", help="AWS region every client is built in")
@click.option("--output", default=None, type=click.Path(), help="Write findings as JSON")
@click.option(
    "--fail-on-error/--no-fail-on-error",
    default=True,
    help="Exit non-zero when an ERROR finding is present (for CI)",
)
def drift(region: str, output: str | None, fail_on_error: bool) -> None:
    """Reconcile deployed autoscaling against config. Read-only, changes nothing.

    Flags orphaned targets and policies (live in the account, no longer
    synthesized by CDK), inert policies (metric namespace publishing nothing),
    capacity mismatches, and leftover suspensions from an aborted benchmark.

    An orphan is an ERROR rather than a warning: it can move capacity that no
    template describes, `cdk deploy` will never remove it, and an unguarded
    Q_max run against such an endpoint measures a fleet instead of an instance.
    """
    import boto3

    from tts_bench.observe import Severity, check_drift

    appscaling = boto3.client("application-autoscaling", region_name=region)
    cloudwatch = boto3.client("cloudwatch", region_name=region)

    findings = check_drift(appscaling, cloudwatch, _expected_scaling())

    if not findings:
        click.echo("No drift: deployed autoscaling matches config.")
        # Where drift sits in the sequence: it is the preflight for a measurement, because
        # an orphaned policy moving capacity mid-run is what makes a Q_max ladder read as a
        # fleet. Clean means the freeze in `qmax --require-frozen` has nothing to fight.
        click.echo("\nNext: tts-bench qmax --model <model> --require-frozen")
    else:
        order = {Severity.ERROR: 0, Severity.WARN: 1, Severity.INFO: 2}
        for finding in sorted(findings, key=lambda f: (order[f.severity], f.resource_id)):
            marker = "ERROR" if finding.severity is Severity.ERROR else finding.severity.upper()
            click.echo(f"[{marker}] {finding.kind.value} — {finding.resource_id}")
            click.echo(f"        {finding.detail}")
            if finding.remediation:
                click.echo(f"        fix: {finding.remediation}")

        errors = sum(1 for f in findings if f.severity is Severity.ERROR)
        warns = sum(1 for f in findings if f.severity is Severity.WARN)
        click.echo(f"\n{len(findings)} finding(s): {errors} error, {warns} warning")

    if output:
        Path(output).write_text(
            json.dumps(
                [
                    {
                        "kind": f.kind.value,
                        "severity": f.severity.value,
                        "resource_id": f.resource_id,
                        "endpoint": f.endpoint,
                        "detail": f.detail,
                        "remediation": f.remediation,
                    }
                    for f in findings
                ],
                indent=2,
            )
        )
        click.echo(f"Findings: {output}")

    if fail_on_error and any(f.severity is Severity.ERROR for f in findings):
        raise SystemExit(1)


@main.command()
@click.option("--endpoint", required=True, help="Endpoint name, e.g. speech-kokoro-82m")
@click.option("--variant", default="primary", help="Production variant on the endpoint")
@click.option("--region", default="us-east-1", help="AWS region every client is built in")
@click.option(
    "--restore-capacity/--no-restore-capacity",
    default=False,
    help="Also restore desired instance count (needs --desired)",
)
@click.option(
    "--desired",
    default=None,
    type=int,
    help="Desired instance count to restore. Required with --restore-capacity.",
)
def thaw(
    endpoint: str, variant: str, region: str, restore_capacity: bool, desired: int | None
) -> None:
    """Resume autoscaling on an endpoint after an aborted benchmark.

    `cmax` restores state itself, including on Ctrl-C. This exists for a hard
    kill, where nothing got to run — the operation is idempotent, so running it
    on a healthy endpoint is a no-op.

    Capacity is *not* restored by default: after a hard kill we do not know what
    the desired count was before the freeze, and guessing 1 could shrink a fleet
    that was legitimately larger. Pass --desired explicitly to set it.
    """
    from tts_bench.fixture import EndpointFixture, capture
    from tts_bench.fixture import thaw as do_thaw

    if restore_capacity and desired is None:
        raise click.UsageError("--restore-capacity requires --desired")

    current = capture(endpoint, region=region, variant=variant)
    click.echo(
        f"{endpoint}: suspended_state={current.suspended_state} "
        f"desired={current.desired_instance_count} current={current.current_instance_count}"
    )

    if not current.has_scalable_target:
        click.echo("No scalable target registered — nothing to resume.")
        if not restore_capacity:
            return

    # Reconstruct a fixture describing the state we want, rather than the state
    # we found: thaw() restores what it is given, and after a hard kill the
    # captured "before" state is gone.
    target = EndpointFixture(
        endpoint_name=endpoint,
        variant=variant,
        suspended_state={} if current.has_scalable_target else None,
        min_capacity=current.min_capacity,
        max_capacity=current.max_capacity,
        desired_instance_count=desired if desired is not None else current.desired_instance_count,
        current_instance_count=current.current_instance_count,
        policy_names=current.policy_names,
    )
    do_thaw(target, region=region, restore_capacity=restore_capacity)

    after = capture(endpoint, region=region, variant=variant)
    click.echo(f"{endpoint}: suspended_state={after.suspended_state}")
    if after.scale_out_suspended and after.has_scalable_target:
        click.echo("WARNING: scale-out still suspended. Check IAM permissions.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
