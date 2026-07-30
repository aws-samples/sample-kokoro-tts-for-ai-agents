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

ALL_MODELS = [m.value for m in TTSModelName]


@click.group()
def main() -> None:
    """TTS Performance Benchmarking Suite."""


@main.command()
@click.option("--models", default="all", help="Comma-separated model names or 'all'")
@click.option("--runs", default=5, type=int, help="Runs per sample")
@click.option("--max-samples", default=10, type=int, help="Number of text samples")
@click.option("--region", default="us-east-1")
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
@click.option("--models", default="all", help="Comma-separated model names or 'all'")
@click.option("--concurrency", default="5,10,20", help="Comma-separated concurrency levels")
@click.option("--window", default=15.0, type=float, help="Seconds per concurrency level")
@click.option("--region", default="us-east-1")
@click.option("--output", default=None, type=click.Path())
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
@click.option("--models", default="all", help="Comma-separated model names or 'all'")
@click.option("--max-samples", default=20, type=int, help="Number of text samples for throughput")
@click.option("--region", default="us-east-1")
@click.option("--output", default=None, type=click.Path())
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


@main.command()
@click.option("--model", required=True, help="Model name, e.g. kokoro-82m")
@click.option("--region", default="us-east-1")
@click.option("--variant", default="primary")
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
@click.option("--arrival", default="poisson", type=click.Choice(["poisson", "fixed"]))
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
    "--cloudwatch/--no-cloudwatch",
    "cloudwatch_join",
    default=True,
    help="Join server-side metrics after a settle delay (~2 min)",
)
@click.option("--output", default=None, type=click.Path(), help="Write the artifact JSON here")
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
    cloudwatch_join: bool,
    output: str | None,
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
    click.echo(f"{'budget_ms':>10} {'C_max':>8} {'rps':>8} {'p95_ttfab':>10} {'spread':>8}")
    knees = {k.ttfab_budget_ms: k for k in report.knees}
    for budget in sorted(report.c_max_curve):
        knee = knees.get(budget)
        spread = report.curve_spread.get(budget)
        click.echo(
            f"{budget:>10} {report.c_max_curve[budget]:>8.2f} "
            f"{knee.offered_rps if knee else 0:>8.2f} "
            f"{knee.p95_ttfab_ms if knee else 0:>10.0f} "
            f"{f'{spread:.0%}' if spread is not None else 'n/a':>8}"
        )

    click.echo(
        f"\nS (uncontended): mean {report.s_mean_s * 1000:.0f}ms p95 {report.s_p95_s * 1000:.0f}ms"
    )
    if report.unbracketed_budgets:
        click.echo(
            f"NOTE: budgets {report.unbracketed_budgets} still passed at the top of the ladder, "
            "so those are lower bounds. Extend --target-concurrency to bracket them."
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

    if output:
        Path(output).write_text(report.model_dump_json(indent=2))
        click.echo(f"\nArtifact: {output}")
    click.echo("Next: tts-bench ttotal, then tts-bench plan --measured <artifact>")


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
@click.option("--region", default="us-east-1")
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
@click.option("--variant", default="primary")
@click.option("--region", default="us-east-1")
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
