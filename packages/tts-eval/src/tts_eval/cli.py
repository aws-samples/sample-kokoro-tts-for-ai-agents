# Copyright (c) 2026 Amazon.com
# This file is licensed under the MIT License.
# See the LICENSE file in the project root for full license information.

"""CLI for TTS evaluation."""

from __future__ import annotations

import time
from pathlib import Path

import click
from loguru import logger

from tts_inference.types import TTSModelName

ALL_MODELS = [m.value for m in TTSModelName]


@click.group()
def main() -> None:
    """TTS Model Evaluation Suite."""


@main.command()
@click.option(
    "--models",
    default="all",
    help="Comma-separated model names or 'all'",
)
@click.option(
    "--samples",
    default=None,
    type=click.Path(exists=True),
    help="Path to samples JSON (default: data/harvard_sentences.json)",
)
@click.option(
    "--output-dir",
    default=None,
    type=click.Path(),
    help="Output directory (default: results/{timestamp})",
)
@click.option("--max-samples", default=None, type=int, help="Limit number of samples")
@click.option("--skip-wer", is_flag=True, help="Skip WER scoring (faster)")
@click.option("--skip-bench", is_flag=True, help="Skip cost/scalability benchmarks")
@click.option("--concurrency", default="1,10,50,100", help="Scalability concurrency levels")
@click.option("--workers", default=10, type=int, help="Parallel workers per model for synthesis")
@click.option(
    "--streaming-mode",
    type=click.Choice(["response-stream", "bidirectional"]),
    default="response-stream",
    help="Streaming mode: response-stream (default) or bidirectional (HTTP/2)",
)
@click.option("--region", default="us-east-1")
def run(
    models: str,
    samples: str | None,
    output_dir: str | None,
    max_samples: int | None,
    skip_wer: bool,
    skip_bench: bool,
    concurrency: str,
    workers: int,
    streaming_mode: str,
    region: str,
) -> None:
    """Run full evaluation (quality + benchmarks) across models."""
    from shared.loader import get_data_dir, load_tts_samples
    from tts_eval.report import generate_report
    from tts_eval.runner import EvalRunner

    model_list = ALL_MODELS if models == "all" else [m.strip() for m in models.split(",")]

    if samples is None:
        samples_path = get_data_dir() / "harvard_sentences.json"
    else:
        samples_path = Path(samples)

    dataset = load_tts_samples(samples_path)
    sample_list = dataset.samples
    if max_samples:
        sample_list = sample_list[:max_samples]

    if output_dir is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        out = Path("results") / timestamp
    else:
        out = Path(output_dir)

    logger.info("Starting evaluation: {} models x {} samples", len(model_list), len(sample_list))

    runner = EvalRunner(
        models=model_list,
        samples=sample_list,
        output_dir=out,
        region=region,
        skip_wer=skip_wer,
        max_workers=workers,
        streaming_mode=streaming_mode,
    )
    results = runner.run()

    bench_results = None
    if not skip_bench:
        bench_results = _run_benchmarks(
            model_list, [s.text for s in sample_list], concurrency, region
        )

    json_path, md_path = generate_report(results, out, bench_results=bench_results)
    logger.info("Results written to {}", out)
    click.echo(f"\nReport: {md_path}")
    click.echo(f"Data:   {json_path}")
    click.echo(f"Audio:  {out / 'samples'}/")


def _run_benchmarks(
    model_list: list[str],
    texts: list[str],
    concurrency: str,
    region: str,
) -> dict:
    """Run cost and scalability benchmarks for all models."""
    from tts_bench.cost import calculate_cost
    from tts_bench.scalability import measure_scalability

    levels = [int(c.strip()) for c in concurrency.split(",")]
    bench_text = texts[0] if texts else "The birch canoe slid on the smooth planks."

    cost_results = []
    scalability_results = []

    for model in model_list:
        logger.info("Running cost benchmark for {}", model)
        try:
            cost = calculate_cost(model, texts[:20], region=region)
            cost_results.append(cost)
        except Exception as e:
            logger.warning("Cost benchmark failed for {}: {}", model, e)

        logger.info("Running scalability benchmark for {}", model)
        try:
            scale = measure_scalability(model, bench_text, levels, region=region)
            scalability_results.extend(scale)
        except Exception as e:
            logger.warning("Scalability benchmark failed for {}: {}", model, e)

    bench: dict = {}
    if cost_results:
        bench["cost"] = cost_results
    if scalability_results:
        bench["scalability"] = scalability_results
    return bench


@main.group("human-panel")
def human_panel() -> None:
    """Human panel evaluation tools."""


@human_panel.command("generate")
@click.option("--samples-dir", required=True, type=click.Path(exists=True))
@click.option("--output-dir", default="results/latest", type=click.Path())
@click.option("--num-samples", default=None, type=int)
@click.option("--seed", default=42, type=int)
def panel_generate(
    samples_dir: str,
    output_dir: str,
    num_samples: int | None,
    seed: int,
) -> None:
    """Generate randomized blind listening test from synthesized samples."""
    from tts_eval.human_panel import generate_listening_test

    test_dir = generate_listening_test(
        samples_dir=Path(samples_dir),
        output_dir=Path(output_dir),
        num_samples=num_samples,
        seed=seed,
    )
    click.echo(f"Listening test generated: {test_dir}")
    click.echo(f"  Audio: {test_dir / 'listening_test'}/")
    click.echo(f"  Scoresheet: {test_dir / 'scoresheet.csv'}")


@human_panel.command("score")
@click.option("--scoresheet", required=True, type=click.Path(exists=True))
@click.option("--mapping", required=True, type=click.Path(exists=True))
def panel_score(scoresheet: str, mapping: str) -> None:
    """Ingest completed scoresheets and compute results."""
    import json

    from tts_eval.human_panel import ingest_scores

    results = ingest_scores(Path(scoresheet), Path(mapping))
    click.echo(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
