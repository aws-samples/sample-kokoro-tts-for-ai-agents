"""CLI for TTS performance benchmarking."""

from __future__ import annotations

import json
import time
from pathlib import Path

import click
from loguru import logger

from tts_inference.types import TTSModelName

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
@click.option("--concurrency", default="10,50,100", help="Comma-separated concurrency levels")
@click.option("--region", default="us-east-1")
@click.option("--output", default=None, type=click.Path())
def scalability(models: str, concurrency: str, region: str, output: str | None) -> None:
    """Test scalability under concurrent load."""
    from tts_bench.scalability import measure_scalability

    model_list = ALL_MODELS if models == "all" else [m.strip() for m in models.split(",")]
    levels = [int(c.strip()) for c in concurrency.split(",")]
    text = "The birch canoe slid on the smooth planks."

    all_results = []
    for model in model_list:
        logger.info("Scalability test for {}", model)
        try:
            results = measure_scalability(model, text, levels, region=region)
            all_results.extend(results)
            for r in results:
                click.echo(
                    f"  {model} @{r['concurrency']}: "
                    f"success={r['success_rate']*100:.0f}% "
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


if __name__ == "__main__":
    main()
