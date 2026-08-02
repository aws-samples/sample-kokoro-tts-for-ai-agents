"""CLI for TTS performance benchmarking."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import click
from loguru import logger

from tts_inference.types import TTSModelName

if TYPE_CHECKING:
    from collections.abc import Sequence

    from tts_bench.observe import ExpectedScaling
    from tts_bench.planner import SweepRow

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


def _parse_floats(raw: str, *, flag: str) -> tuple[float, ...]:
    """Parse a comma-separated numeric list, rejecting anything non-positive."""
    try:
        values = tuple(float(part) for part in raw.split(",") if part.strip())
    except ValueError as exc:
        raise click.BadParameter(f"{flag} must be comma-separated numbers: {raw}") from exc
    if not values:
        raise click.BadParameter(f"{flag} must not be empty")
    if any(v <= 0 for v in values):
        raise click.BadParameter(f"{flag} values must be positive: {raw}")
    return values


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


# Literal defaults, mirroring `tts_bench.cmax`. Importing that module here would
# pull numpy and botocore into `--help`, and every other command in this file
# imports lazily for the same reason. `test_cli_cmax.py` asserts these match the
# module constants, so the duplication cannot drift silently.
CMAX_LADDER_DEFAULT = "0.5,1,1.5,2,3,4,6,8,12,16"
CMAX_BUDGETS_DEFAULT = "50,150,300,500"

# Duplicated from `tts_bench.ttotal` for the same reason: a click.Choice is evaluated at
# import time, so referencing the module here would defeat the lazy import.
# `test_cli_ttotal.py` asserts these match the module constants.
TTOTAL_TRIGGER_DRIVE_LOAD = "drive-load"
TTOTAL_TRIGGER_FORCE_DESIRED = "force-desired"


@main.command()
@click.option(
    "--model",
    required=True,
    help=(
        "Model name, e.g. kokoro-82m. Singular and required: this measures one "
        "configuration, and it freezes or scales that endpoint to do it."
    ),
)
@click.option("--region", default="us-east-1", help="AWS region every client is built in")
@click.option("--variant", default="primary", help="Production variant on the endpoint")
@click.option("--voice", default=None, help="Override the model's default voice")
@click.option(
    "--target-concurrency",
    "target_concurrency",
    default=CMAX_LADDER_DEFAULT,
    help="Ladder in expected concurrency (lambda x S), comma-separated",
)
@click.option(
    "--ttfab-budgets",
    default=CMAX_BUDGETS_DEFAULT,
    help="p95 TTFAB budgets in ms; the knee is reported for each",
)
@click.option("--hold", "hold_s", default=240.0, type=float, help="Seconds held per step")
@click.option(
    "--measure-window",
    "measure_window_s",
    default=60.0,
    type=float,
    help="Trailing seconds of each step that are measured; the rest is warm-up",
)
@click.option(
    "--settle-between-steps",
    "settle_between_steps_s",
    default=30.0,
    type=float,
    help="Idle seconds between steps so the previous queue drains",
)
@click.option("--runs", default=1, type=int, help="Ladder passes; 3 to see run-to-run spread")
@click.option("--derate", default=0.875, type=float, help="Recorded in the artifact, not applied")
@click.option(
    "--transport",
    default="response-stream",
    type=click.Choice(["response-stream", "bidi"]),
    help="Wire protocol to measure. C_max does not transfer between the two.",
)
@click.option(
    "--arrival",
    default="poisson",
    type=click.Choice(["poisson", "fixed"]),
    help="Arrival process. Poisson is the realistic one; fixed isolates the harness.",
)
@click.option("--seed", default=1234, type=int, help="Arrival-schedule seed; keeps runs comparable")
@click.option("--max-samples", default=50, type=int, help="Texts drawn into the pool")
@click.option("--samples", default=None, type=click.Path(), help="Override the sample JSON path")
@click.option(
    "--require-frozen/--no-require-frozen",
    default=True,
    help="Refuse to run unless scale-out is suspended and capacity is pinned",
)
@click.option("--pin-to", default=1, type=int, help="Instances to pin for the run")
@click.option(
    "--max-workers",
    default=None,
    type=int,
    help=(
        "Raise the client thread/connection pool above the derived size. Only raises. "
        "Needed when a transport holds its lock per session (bidi), where in-flight "
        "overshoots the target by the server's backlog and the derived pool runs out."
    ),
)
@click.option(
    "--cloudwatch/--no-cloudwatch",
    "cloudwatch_join",
    default=True,
    help="Join server-side metrics after a settle delay (~2 min)",
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
def cmax(
    model: str,
    region: str,
    variant: str,
    voice: str | None,
    target_concurrency: str,
    ttfab_budgets: str,
    hold_s: float,
    measure_window_s: float,
    settle_between_steps_s: float,
    runs: int,
    derate: float,
    transport: str,
    arrival: str,
    seed: int,
    max_samples: int,
    samples: str | None,
    require_frozen: bool,
    pin_to: int,
    max_workers: int | None,
    cloudwatch_join: bool,
    output: str | None,
    no_save: bool,
    events: str | None,
    dry_run: bool,
    dry_run_s: float,
) -> None:
    """Measure per-instance concurrency at the latency knee (C_max).

    Holds a fixed arrival rate per step and measures only the trailing window, so
    warm-up and queue drain are excluded. The knee is reported for every TTFAB
    budget from one ladder, which turns choosing an SLO into a table lookup
    rather than another 45-minute run.

    Autoscaling is frozen and capacity pinned for the whole run, restored on exit
    including on Ctrl-C. C_max is a *per-instance* number: if the fleet grows
    mid-run, throughput rises for a reason unrelated to the knee and the result
    is silently N x C_max.

    ``--transport bidi`` measures the protocol production is configured for. It
    is a separate measurement, not a refinement: the containers serialize
    differently on it, so expect a lower C_max on kokoro, which holds its
    inference lock across a whole bidi session.
    """
    from tts_bench import cmax as cmax_mod
    from tts_bench.invoke import resolve_endpoint

    targets = _parse_floats(target_concurrency, flag="--target-concurrency")
    budgets = _parse_ints(ttfab_budgets, flag="--ttfab-budgets")

    # Resolve the endpoint before anything else. A typo or a managed model then
    # fails as a usage error rather than a traceback out of `measure`, and it
    # fails on --dry-run too — which is the run people use to check a command
    # before committing 45 minutes and an endpoint freeze to it.
    try:
        endpoint = resolve_endpoint(model)
    except ValueError as exc:
        raise click.BadParameter(str(exc), param_hint="--model") from exc

    if measure_window_s > hold_s:
        raise click.BadParameter(
            f"--measure-window ({measure_window_s}) cannot exceed --hold ({hold_s}); it is the "
            "trailing part of a step, not an addition to it"
        )
    if runs < 1:
        raise click.BadParameter("--runs must be at least 1")

    estimate_s = cmax_mod.total_duration_s(
        target_concurrencies=targets,
        hold_s=hold_s,
        settle_between_steps_s=settle_between_steps_s,
        runs=runs,
    )

    if dry_run:
        click.echo(
            f"{model} ({endpoint}): {len(targets)} step(s) x {runs} run(s), "
            f"assuming S={dry_run_s:.3f}s (not measured)"
        )
        click.echo(f"{'run':>4} {'step':>5} {'target':>8} {'rps':>9} {'hold_s':>8} {'requests':>9}")
        for row in cmax_mod.dry_run_plan(
            s_mean_s=dry_run_s, target_concurrencies=targets, hold_s=hold_s, runs=runs
        ):
            click.echo(
                f"{int(row['run_index']):>4} {int(row['step_index']):>5} "
                f"{row['target_concurrency']:>8.2f} {row['offered_rps']:>9.2f} "
                f"{row['hold_s']:>8.0f} {row['expected_requests']:>9.0f}"
            )
        click.echo(
            f"\nEstimated wall clock: {estimate_s / 60:.0f} min "
            f"(+~2 min CloudWatch settle). Budgets: {list(budgets)}"
        )
        click.echo("Dry run: no AWS calls made, nothing frozen.")
        return

    texts = _load_texts(samples, max_samples)
    click.echo(
        f"{model} ({endpoint}): {len(targets)} step(s) x {runs} run(s), "
        f"~{estimate_s / 60:.0f} min, {len(texts)} texts, transport={transport}, "
        f"frozen={require_frozen}"
    )
    if not require_frozen:
        click.echo(
            "WARNING: --no-require-frozen. If the fleet grows mid-run this measures N x C_max; "
            "the artifact will record frozen=false."
        )

    from contextlib import nullcontext

    from tts_bench.fixture import FixtureError
    from tts_bench.loadgen import JsonlWriter

    try:
        # JsonlWriter opens on enter and flushes every event, so a run killed at
        # the interesting moment has still written the interesting moment.
        with JsonlWriter(events) if events else nullcontext() as writer:
            report = cmax_mod.measure(
                model=model,
                texts=texts,
                region=region,
                variant=variant,
                voice=voice,
                target_concurrencies=targets,
                budgets=budgets,
                hold_s=hold_s,
                measure_window_s=measure_window_s,
                settle_between_steps_s=settle_between_steps_s,
                runs=runs,
                derate=derate,
                arrival=arrival,
                seed=seed,
                transport=transport,
                require_frozen=require_frozen,
                pin_to=pin_to,
                max_workers=max_workers,
                cloudwatch_join=cloudwatch_join,
                event_sink=writer,
            )
    except (cmax_mod.CMaxError, FixtureError) as exc:
        # Both are the tool refusing to produce a number it cannot stand behind,
        # so they exit cleanly with the reason. A traceback would read as a bug
        # rather than as the guard doing its job.
        raise click.ClickException(str(exc)) from exc
    if events:
        click.echo(f"Events: {events}")

    click.echo(
        f"\nC_max curve for {report.model_name} on {report.instance_type} via {report.transport}:"
    )
    click.echo(
        f"{'budget_ms':>10} {'C_max':>8} {'rps':>8} {'p95_ttfab':>10} {'spread':>8} {'runs':>6}"
    )
    knees = {k.ttfab_budget_ms: k for k in report.knees}
    for budget in sorted(report.c_max_curve):
        knee = knees.get(budget)
        spread = report.curve_spread.get(budget)
        contributing = report.runs_contributing.get(budget)
        # A missing knee prints as "n/a", never as 0.00. Zeros in a rate and a latency
        # column read as a measurement that came back empty, when what happened is that
        # the detail for this budget is absent while the concurrency beside it is real.
        click.echo(
            f"{budget:>10} {report.c_max_curve[budget]:>8.2f} "
            f"{f'{knee.offered_rps:.2f}' if knee else 'n/a':>8} "
            f"{f'{knee.p95_ttfab_ms:.0f}' if knee else 'n/a':>10} "
            f"{f'{spread:.0%}' if spread is not None else 'n/a':>8} "
            f"{f'{contributing}/{report.runs}' if contributing is not None else 'n/a':>6}"
        )

    # A curve built from one pass of a --runs 3 ladder is a single sample. Saying so
    # here matters because `spread` is 0% in exactly that case, which otherwise reads
    # as three runs agreeing.
    thin = sorted(b for b, n in report.runs_contributing.items() if n < report.runs)
    if thin and report.runs > 1:
        click.echo(
            f"NOTE: budgets {thin} had fewer than {report.runs} runs find a knee, so their "
            "spread is not a run-to-run agreement. The other runs' steps were unusable or "
            "unsettled at that budget — check the ladder above."
        )

    # The second C_max, printed as its own block rather than a row in the curve: it has no
    # budget to key it by, and the whole point is that it is not a latency measurement.
    ceiling = report.throughput_ceiling
    if ceiling is None:
        click.echo(
            "\nThroughput ceiling: not measured — no step sustained its offered rate. "
            "Every rate on this ladder was already past capacity, so C_max is below the "
            "lowest step; re-run with a lower --target-concurrency."
        )
    else:
        click.echo(
            f"\nThroughput ceiling: {ceiling.max_sustained_rps:.2f} rps sustained "
            f"-> C_max {ceiling.concurrency:.2f} "
            f"(useful concurrency; p95 TTFAB {ceiling.p95_ttfab_ms:.0f}ms at step "
            f"{ceiling.step_index}, {ceiling.runs_contributing}/{report.runs} runs, "
            f"spread {ceiling.spread:.0%})"
        )
        if ceiling.observed_concurrency is not None:
            multiple = ceiling.queueing_multiple
            click.echo(
                f"  observed in-flight there was {ceiling.observed_concurrency:.2f}"
                + (
                    f" — {multiple:.1f}x the useful figure, i.e. that much of each "
                    "request's residence is queueing. ConcurrentRequestsPerModel (what "
                    "the scaling policy tracks) reports the observed number, so the two "
                    "are not interchangeable."
                    if multiple is not None and multiple > 1.5
                    else f" ({multiple:.1f}x useful)"
                    if multiple is not None
                    else ""
                )
            )
        if report.throughput_bound_budgets:
            click.echo(
                f"  WARNING: budgets {report.throughput_bound_budgets} report a knee ABOVE this "
                "ceiling. Nothing failed at those steps — latency stayed inside budget — but "
                "the server had stopped keeping up, so their concurrency is backlog, not "
                "capacity. Plan on the ceiling."
            )
        if ceiling.is_lower_bound:
            reason = (
                f"{ceiling.dispatch_skipped} dispatch(es) were skipped at that step, so the "
                "server never saw the full offered rate — raise --max-workers"
                if ceiling.dispatch_skipped
                else "no saturated step was observed above it — extend --target-concurrency"
            )
            click.echo(f"  NOTE: this is a LOWER bound ({reason}).")

    click.echo(
        f"\nS (uncontended): mean {report.s_mean_s * 1000:.0f}ms p95 {report.s_p95_s * 1000:.0f}ms"
    )
    if report.exhausted_budgets:
        click.echo(
            f"NOTE: budgets {report.exhausted_budgets} still passed at the top of the ladder, "
            "so those are lower bounds. Extend --target-concurrency to bracket them."
        )
    if report.inconclusive_budgets:
        click.echo(
            f"NOTE: budgets {report.inconclusive_budgets} are lower bounds, but higher rates "
            "*were* offered and produced no usable latency — a longer ladder will not help. "
            "Check the unusable_reason on the steps above the knee."
        )
    if report.ladder_truncated_at is not None:
        click.echo(
            f"NOTE: ladder stopped at step {report.ladder_truncated_at} after repeated "
            "saturation; higher rates were never offered."
        )
    if not report.trustworthy:
        click.echo(
            f"WARNING: not safe to read as per-instance — frozen={report.frozen}, "
            f"instance counts seen {list(report.instance_counts_observed)}"
        )

    if no_save:
        click.echo(f"\n--no-save: nothing written. Configuration measured: {report.config_slug}")
        click.echo(f"\nNext: tts-bench ttotal --model {model} --measured <cmax artifact>")
    else:
        # Defaulted rather than optional: this run costs 45+ minutes and an endpoint
        # freeze, and printing a suggested filename after the fact does not bring the
        # measurement back. The slug in the name is what keeps two configurations'
        # curves apart -- they are two different measurements, not two attempts at one.
        destination = (
            Path(output)
            if output
            else _default_artifact_path("cmax", model, transport, report.config_slug)
        )
        _warn_if_replacing_another_config(destination, report.config_slug)
        _save_artifact(destination, report.model_dump_json(indent=2))
        click.echo(f"\nArtifact: {destination}")
        click.echo(f"Configuration measured: {report.config_slug}")
        # Echo the path back rather than "<artifact>": the next command needs this exact
        # file, and the slug in it is not something to retype from memory.
        click.echo(f"\nNext: tts-bench ttotal --model {model} --measured {destination}")
        click.echo(f"Then: tts-bench plan --measured {destination} --ttotal <ttotal artifact>")


@main.command()
@click.option(
    "--model",
    required=True,
    help=(
        "Model name, e.g. kokoro-82m. Singular and required: this measures one "
        "configuration, and it freezes or scales that endpoint to do it."
    ),
)
@click.option("--region", default="us-east-1", help="AWS region every client is built in")
@click.option("--variant", default="primary", help="Production variant on the endpoint")
@click.option("--voice", default=None, help="Override the model's default voice")
@click.option(
    "--trigger",
    type=click.Choice([TTOTAL_TRIGGER_DRIVE_LOAD, TTOTAL_TRIGGER_FORCE_DESIRED]),
    default=TTOTAL_TRIGGER_DRIVE_LOAD,
    help="How to cause the scale-out. force-desired skips the policy entirely.",
)
@click.option(
    "--scaling-target",
    default=None,
    type=float,
    help=(
        "C_target to drive past. Defaults to the deployed policy's TargetValue; with "
        "--trigger drive-load and no policy deployed, this is an error rather than a guess."
    ),
)
@click.option(
    "--load-multiple",
    default=3.0,
    type=float,
    help="Offered concurrency as a multiple of C_target",
)
@click.option(
    "--s-mean",
    "s_mean_s",
    default=None,
    type=float,
    help="Mean service time in seconds, converting target concurrency to a rate",
)
@click.option(
    "--measured",
    default=None,
    type=click.Path(exists=True),
    help="A cmax artifact to read S and the TTFAB budget from",
)
@click.option(
    "--ttfab-budget-ms",
    default=None,
    type=float,
    help="Budget the recovery bound is judged against. Defaults from --measured or config.",
)
@click.option(
    "--allow-config-mismatch",
    is_flag=True,
    default=False,
    help="Proceed even if --measured was taken on a different deployed configuration",
)
@click.option("--max-wait", "max_wait_s", default=1500.0, type=float, help="Scale-out timeout")
@click.option(
    "--settle",
    "settle_s",
    default=180.0,
    type=float,
    help="Seconds of load held past the event, so recovery can be bounded",
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
    help="Wire protocol for the offered load. Match the --measured curve.",
)
@click.option(
    "--arrival-seed",
    "seed",
    default=1234,
    type=int,
    help="Arrival-schedule seed; keeps runs comparable",
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
    region: str,
    variant: str,
    voice: str | None,
    trigger: str,
    scaling_target: float | None,
    load_multiple: float,
    s_mean_s: float | None,
    measured: str | None,
    ttfab_budget_ms: float | None,
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
    """Measure T_total: the lag from load arriving to new capacity serving it.

    Drives load past C_target so the deployed policy fires, then attributes the lag
    stage by stage — metric publication, alarm evaluation, scaling activity, instance
    provisioning, container startup, and traffic recovery — each from the API that
    timestamps it.

    T_total is the input the capacity plan is most sensitive to: it sets how much
    standing headroom a surge needs and how much of one a queue can absorb. It is also
    the number most often guessed.

    `--trigger force-desired` sets DesiredInstanceCount directly. That needs no policy
    and measures only the container half, so its result is NOT a full T_total — it exists
    for iterating quickly on the container stages.

    Restores the starting desired instance count on exit, Ctrl-C included.
    """
    import boto3

    from tts_bench import ttotal as ttotal_mod
    from tts_bench.fixture import FixtureError
    from tts_bench.invoke import resolve_endpoint, resolve_voice

    try:
        endpoint = resolve_endpoint(model)
    except ValueError as exc:
        raise click.BadParameter(str(exc), param_hint="--model") from exc

    # S and the TTFAB budget both come from a cmax artifact when there is one: they are
    # measured properties of this model on this transport, and re-deriving them here
    # would let a T_total run silently disagree with the C_max it will be planned with.
    artifact: dict[str, object] | None = None
    if measured:
        artifact = json.loads(Path(measured).read_text())
        if s_mean_s is None and isinstance(artifact.get("s_mean_s"), int | float):
            s_mean_s = float(artifact["s_mean_s"])  # type: ignore[arg-type]
        if ttfab_budget_ms is None:
            curve = artifact.get("c_max_curve")
            if isinstance(curve, dict) and curve:
                # The largest budget in the curve: the loosest SLO the C_max run
                # measured, so recovery is judged against a bound the model can meet.
                ttfab_budget_ms = float(max(int(k) for k in curve))
        if artifact.get("transport") and artifact["transport"] != transport:
            click.echo(
                f"WARNING: --measured was taken on transport {artifact['transport']!r} but "
                f"this run uses {transport!r}. S differs per transport, so the offered rate "
                "may not reach C_target."
            )
        # A transport mismatch only mis-sizes the offered rate, so it warns. A
        # configuration mismatch means S itself was measured on other hardware or other
        # serving code, which makes the whole plan describe a fleet that does not exist —
        # so it stops the run.
        _require_matching_config(
            artifact,
            endpoint=endpoint,
            region=region,
            variant=variant,
            artifact_path=measured,
            allow_mismatch=allow_config_mismatch,
        )

    if s_mean_s is None:
        if trigger == TTOTAL_TRIGGER_DRIVE_LOAD:
            raise click.UsageError(
                "--s-mean is required (or pass --measured <cmax artifact> to read it). It "
                "converts a target concurrency into the arrival rate the open-loop driver "
                "needs, and guessing it would offer the wrong load."
            )
        # force-desired starts no load driver, so there is no rate to convert. Requiring a
        # service time here would make the cheapest probe available -- can this endpoint
        # get another instance at all? -- depend on having already measured C_max.
        s_mean_s = 0.0
    elif s_mean_s <= 0:
        raise click.BadParameter("--s-mean must be positive")
    if load_multiple <= 1.0:
        raise click.BadParameter(
            "--load-multiple must exceed 1.0, or the offered load never crosses C_target "
            "and no scale-out can occur"
        )

    appscaling = boto3.client("application-autoscaling", region_name=region)
    if scaling_target is None:
        scaling_target = ttotal_mod.deployed_target_value(
            appscaling, endpoint=endpoint, variant=variant
        )
    if scaling_target is None:
        if trigger == TTOTAL_TRIGGER_DRIVE_LOAD:
            raise click.UsageError(
                f"{endpoint} has no target-tracking policy to read a TargetValue from, so "
                "there is nothing to drive load past. Deploy scaling first, or pass "
                "--scaling-target explicitly."
            )
        # force-desired never reads the metric, so any value is inert here.
        scaling_target = 0.0

    # Deliberately the measured budget and not the end-to-end SLO, which is the one place
    # in this package where those two come apart. Recovery is "p95 came back", and it is
    # judged by *discrimination*: the threshold has to sit between the overloaded p95 and
    # the recovered one. Kokoro overloaded at 3x C_target reaches a p95 TTFAB of 818ms,
    # which is already inside a 3000ms SLO -- threshold it there and the run reports
    # recovery at the instant the instance came into service, having measured nothing. The
    # SLO is what the *plan* promises; this is what the *measurement* can resolve.
    if ttfab_budget_ms is None:
        ttfab_budget_ms = _config_ttfab_budget_ms(endpoint)
    if ttfab_budget_ms is None:
        raise click.UsageError(
            "--ttfab-budget-ms is required (or pass --measured, or configure "
            "ttfab_budget_ms for this model). Recovery is defined as p95 back inside a "
            "budget, so there is no recovery without one."
        )

    if trigger == TTOTAL_TRIGGER_FORCE_DESIRED:
        click.echo(
            f"{model} ({endpoint}): trigger={trigger}, budget={ttfab_budget_ms:.0f}ms. "
            "No load is offered, so C_target and S do not apply."
        )
        click.echo(
            "NOTE: --trigger force-desired measures the container half only. The result is "
            "a lower bound on T_total, not T_total."
        )
    else:
        click.echo(
            f"{model} ({endpoint}): trigger={trigger}, C_target={scaling_target:.3f}, "
            f"offering {scaling_target * load_multiple:.2f} concurrency, "
            f"S={s_mean_s * 1000:.0f}ms, budget={ttfab_budget_ms:.0f}ms, transport={transport}"
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
                scaling_target=scaling_target,
                ttfab_budget_ms=ttfab_budget_ms,
                s_mean_s=s_mean_s,
                texts=_load_texts(samples, max_samples),
                voice=resolve_voice(model, voice),
                trigger=trigger,
                load_multiple=load_multiple,
                max_wait_s=max_wait_s,
                settle_s=settle_s,
                poll_interval_s=poll_interval_s,
                transport=transport,
                seed=seed,
                event_sink=writer,
                appscaling=appscaling,
            )
    except (ttotal_mod.TTotalError, FixtureError) as exc:
        # Both are the tool refusing to report a lag it did not observe. A traceback
        # would read as a bug rather than as the guard doing its job.
        raise click.ClickException(str(exc)) from exc

    click.echo("")
    click.echo(ttotal_mod.render_text(report))

    if events:
        click.echo(f"\nEvents: {events}")

    if no_save:
        click.echo(f"\n--no-save: nothing written. Configuration measured: {report.config_slug}")
        click.echo(
            "\nNext: tts-bench plan --measured <cmax artifact> --ttotal <ttotal artifact> "
            "--peak-rps <N>"
        )
    else:
        # Defaulted for the same reason as cmax: this run costs a real scale-out and, on a
        # timeout, most of max_wait_s of offered load. The trigger is in the name because
        # force-desired measures only the container half -- the two are different
        # measurements of the same configuration, not two attempts at one.
        destination = (
            Path(output)
            if output
            else _default_artifact_path("ttotal", model, trigger, report.config_slug)
        )
        _warn_if_replacing_another_config(destination, report.config_slug)
        _save_artifact(destination, json.dumps(report.to_dict(), indent=2))
        click.echo(f"\nArtifact: {destination}")
        click.echo(f"Configuration measured: {report.config_slug}")
        click.echo(
            f"\nNext: tts-bench plan --measured <cmax artifact> --ttotal {destination} "
            "--peak-rps <N>"
        )


@main.command()
@click.option(
    "--measured",
    required=True,
    type=click.Path(exists=True),
    help="A cmax artifact: the C_max curve and S this plan is built on",
)
@click.option(
    "--ttotal",
    "ttotal_path",
    default=None,
    type=click.Path(exists=True),
    help=(
        "A ttotal artifact: the scaling lag, by stage. Omit only with --assume-t-total, "
        "since a plan with no lag has nothing to size headroom against."
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
    "--growth-factor-k",
    default=2.0,
    type=float,
    help="Traffic growth within one T_total. Not measurable without production traffic.",
)
@click.option(
    "--sweep-k",
    default=None,
    help=(
        "Comma-separated growth factors to plan for, e.g. 1,2,3,5. Overrides "
        "--growth-factor-k; the config's sensitivity to k is the point."
    ),
)
@click.option(
    "--ttfab-slo-ms",
    default=3000,
    type=int,
    help=(
        "End-to-end first-byte SLO: queue wait plus service. W_max = SLO - p95 service "
        "is derived from it, so this is the one field that sets the queueing budget."
    ),
)
@click.option(
    "--ttfab-budget-ms",
    default=None,
    type=int,
    help=(
        "Which measured budget to read the knee at — a column selector, not the SLO. "
        "Defaults to the tightest budget in the curve."
    ),
)
@click.option(
    "--derate",
    default=0.875,
    type=float,
    help="Fraction of the measured knee to target, for jitter margin",
)
@click.option(
    "--min-floor",
    "min_instances_floor",
    default=1,
    type=int,
    help="Never plan below this, whatever the trough says",
)
@click.option(
    "--provision-s",
    default=None,
    help=(
        "Comma-separated EC2 provision times to assume, e.g. 60,120,300,600. One plan "
        "per value, holding the measured container stages fixed — this is the stage a "
        "reserved-capacity account changes, so it is swept rather than trusted."
    ),
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
    help="Pair a C_max curve and a T_total lag measured on different configurations",
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
        "Write the sweep as JSON here. Nothing is written unless asked — unlike a "
        "measurement, this run is cheap to repeat."
    ),
)
def plan(
    measured: str,
    ttotal_path: str | None,
    peak_rps: float | None,
    trough_rps: float | None,
    peak_streams: float | None,
    trough_streams: float | None,
    growth_factor_k: float,
    sweep_k: str | None,
    ttfab_slo_ms: int,
    ttfab_budget_ms: int | None,
    derate: float,
    min_instances_floor: int,
    provision_s: str | None,
    assume_t_total_s: float | None,
    allow_config_mismatch: bool,
    ceiling_s: float | None,
    output: str | None,
) -> None:
    """Turn measurements into a scaling configuration. Reads artifacts, touches no AWS.

    Composes the measured C_max curve, S, and T_total with a stated load into the four
    numbers ModelEndpointConfig needs — scaling_target_value, queue_max_depth,
    min_instances, max_instances — and prints them as a paste-ready block. That closes
    the loop the tool chain exists for: before this, the path from a measurement to a
    deployed policy ran through hand-arithmetic in a comment.

    Two inputs cannot be measured for the account being configured, so both are swept
    rather than assumed. --provision-s replaces the EC2 provision stage, which is a
    property of a capacity contract rather than of the image. --sweep-k varies the
    growth factor, which needs production traffic to observe.

    The SLO is end-to-end: --ttfab-slo-ms is the whole promise, queue included, and
    W_max and queue_max_depth are derived from it rather than set beside it. Two
    independent fields could disagree with the promise; one derived field cannot.

    Refuses two things outright: pairing artifacts whose configuration fingerprints
    differ, and an SLO that does not fit inside SageMaker's 60s invocation ceiling.
    """
    from shared.capacity import SAGEMAKER_INVOCATION_CEILING_S
    from tts_bench import scale_report
    from tts_bench.planner import (
        PlannerError,
        TTotalStages,
        measured_from_artifacts,
        plan_sweep,
    )
    from tts_bench.types import CMaxReport, Scenario

    if peak_rps is None and peak_streams is None:
        raise click.UsageError(
            "one of --peak-rps or --peak-streams is required. The fleet size is what a "
            "plan is for, and there is nothing to size it from without a stated peak."
        )

    try:
        cmax_report = CMaxReport.model_validate_json(Path(measured).read_text())
    except (OSError, ValueError) as exc:
        raise click.ClickException(
            f"could not read {measured} as a cmax artifact: {exc}. It must be the JSON "
            "written by `tts-bench cmax`."
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
            "T_total sets how much standing headroom a surge needs and how much of one "
            "the queue can absorb, so there is no plan without it."
        )
    else:
        # No artifact means no fingerprint to check and no stages to substitute into, so
        # the assumed total is used whole. from_artifact({}) is the honest shape for
        # that: every stage missing, nothing measured.
        stages = TTotalStages.from_artifact({})

    try:
        planner_input = measured_from_artifacts(
            cmax_report,
            stages,
            allow_config_mismatch=allow_config_mismatch,
            assume_t_total_s=assume_t_total_s,
            require_pairing=bool(ttotal_path),
        )
    except PlannerError as exc:
        raise click.ClickException(str(exc)) from exc

    if ttfab_budget_ms is None:
        # The tightest measured budget, not the loosest: the knee at a tight SLO is the
        # smaller number, so defaulting this way sizes the fleet conservatively. `ttotal`
        # defaults the other way for a different reason -- recovery has to be judged
        # against a bound the model can actually meet.
        ttfab_budget_ms = min(planner_input.c_max_curve)

    try:
        scenario = Scenario(
            peak_rps=peak_rps,
            trough_rps=trough_rps,
            peak_streams=peak_streams,
            trough_streams=trough_streams,
            growth_factor_k=growth_factor_k,
            ttfab_budget_ms=ttfab_budget_ms,
            ttfab_slo_ms=ttfab_slo_ms,
            derate=derate,
            min_instances_floor=min_instances_floor,
        )
    except ValueError as exc:
        raise click.BadParameter(str(exc)) from exc

    provisions = _parse_floats(provision_s, flag="--provision-s") if provision_s else None
    ks = _parse_floats(sweep_k, flag="--sweep-k") if sweep_k else None

    try:
        rows = plan_sweep(
            planner_input,
            scenario,
            stages,
            provision_sweep_s=provisions,
            k_sweep=ks,
            ceiling_s=ceiling_s if ceiling_s is not None else SAGEMAKER_INVOCATION_CEILING_S,
        )
    except PlannerError as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo(scale_report.render_plan(rows, stages))
    _warn_if_config_queue_disagrees(rows)

    if output:
        _save_artifact(
            Path(output),
            json.dumps([scale_report.plan_to_dict(row) for row in rows], indent=2),
        )
        click.echo(f"Plan: {output}")

    # Exit non-zero on an infeasible row so this can gate a deploy. Any row, not all:
    # a sweep where one assumed provision time breaks the SLO is a plan that depends on
    # an assumption nobody has verified, which is exactly what should stop a pipeline.
    if any(row.plan.infeasible for row in rows):
        raise SystemExit(1)


def _warn_if_config_queue_disagrees(rows: Sequence[SweepRow]) -> None:
    """Say so when the deployed ``queue_max_depth`` is not what this SLO implies.

    ``queue_max_depth`` is the one derived number ``config.py`` stores rather than
    computes, because it needs ``Lambda_cap`` and that is a measurement. So it can go
    stale silently, and it did: 296 was carried from a 20s W_max no SLO justified, and
    nothing compared it to anything until this check existed.

    A warning rather than a refusal, and printed after the plan. The plan *is* the
    answer, and the whole point of running it is to find out that the deployed value is
    wrong; exiting non-zero here would fail the command that just told you what to fix.
    The paste-ready config block above already carries the right number.

    Reads whichever row's ``k`` matches the deployed target most closely — Q_max does
    not vary with ``k`` or provision time at all (it is ``Lambda_cap x W_max``), so any
    row's value is the same and the first is enough.
    """
    if not rows:
        return
    plan = rows[0].plan
    try:
        from speech_infra.config import TTS_MODEL_CONFIGS
    except ImportError:  # pragma: no cover - depends on install layout
        return

    for config in TTS_MODEL_CONFIGS.values():
        if config.endpoint_name != plan.endpoint:
            continue
        if config.queue_max_depth == plan.queue_max_depth:
            return
        if config.queue_max_depth == 0:
            click.echo(
                f"\nNOTE: {config.model_name} has queue_max_depth=0 (unbounded) deployed; "
                f"this SLO implies {plan.queue_max_depth}. Paste the block above into "
                "config.py to bound it."
            )
            return
        click.echo(
            f"\nWARNING: {config.model_name} has queue_max_depth="
            f"{config.queue_max_depth} deployed, but a {plan.scenario.ttfab_slo_ms}ms "
            f"end-to-end SLO implies {plan.queue_max_depth} "
            f"(Lambda_cap {plan.c_max / plan.measured.s_mean_s:.2f} rps x W_max "
            f"{plan.w_max_s:.2f}s). The deployed value admits requests it can only serve "
            "late. Paste the block above into config.py."
        )
        return


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
    allow_mismatch: bool,
) -> None:
    """Refuse to reuse a measurement taken against a different configuration.

    ``C_max``, ``S`` and ``T_total`` are properties of a configuration — the GPU, the
    serving code, the container knobs — not of a model. Replaying an artifact from one
    configuration onto another produces a plan for a fleet that does not exist, and
    nothing about the output would look wrong.

    An artifact with no fingerprint predates this check and is treated as a mismatch:
    accepting it silently is the hole this closes.
    """
    from tts_bench.fixture import DeployedConfig, FixtureError, describe_deployed_config

    label = f"--measured {artifact_path}" if artifact_path else "the measured artifact"
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


def _config_ttfab_budget_ms(endpoint: str) -> float | None:
    """The TTFAB budget configured for an endpoint, if speech_infra is importable.

    A fallback for `ttotal` run without a cmax artifact. Lazy and forgiving for the same
    reason as :func:`_expected_scaling`: `speech_infra` pulls in aws-cdk-lib, which the
    measurement path has no other use for.
    """
    try:
        from speech_infra.config import TTS_MODEL_CONFIGS
    except ImportError:  # pragma: no cover - depends on install layout
        return None

    for config in TTS_MODEL_CONFIGS.values():
        if config.endpoint_name == endpoint:
            return float(config.ttfab_budget_ms)
    return None


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
    C_max run against such an endpoint measures a fleet instead of an instance.
    """
    import boto3

    from tts_bench.observe import Severity, check_drift

    appscaling = boto3.client("application-autoscaling", region_name=region)
    cloudwatch = boto3.client("cloudwatch", region_name=region)

    findings = check_drift(appscaling, cloudwatch, _expected_scaling())

    if not findings:
        click.echo("No drift: deployed autoscaling matches config.")
        # Where drift sits in the sequence: it is the preflight for a measurement, because
        # an orphaned policy moving capacity mid-run is what makes a C_max curve read as a
        # fleet. Clean means the freeze in `cmax --require-frozen` has nothing to fight.
        click.echo("\nNext: tts-bench cmax --model <model> --require-frozen")
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
