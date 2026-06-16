"""Scalability benchmarking - concurrent request load testing."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from loguru import logger

from tts_eval.synthesize import SynthesisClient
from tts_inference.types import TTSModelName


def measure_scalability(
    model: str | TTSModelName,
    text: str,
    concurrency_levels: list[int],
    region: str = "us-east-1",
) -> list[dict]:
    """Test model under concurrent load.

    Args:
        model: Model to benchmark.
        text: Text to synthesize (same for all requests).
        concurrency_levels: List of concurrency levels to test (e.g. [10, 50, 100]).
        region: AWS region.

    Returns:
        List of dicts with results per concurrency level.
    """
    client = SynthesisClient(region=region)
    model = TTSModelName(model)
    results = []

    for concurrency in concurrency_levels:
        logger.info("Testing {} at concurrency={}", model.value, concurrency)
        level_result = _run_concurrent(client, model, text, concurrency)
        level_result["model"] = model.value
        level_result["concurrency"] = concurrency
        results.append(level_result)

    return results


def _run_concurrent(
    client: SynthesisClient,
    model: TTSModelName,
    text: str,
    concurrency: int,
) -> dict:
    """Run N concurrent requests and collect stats."""
    latencies: list[float] = []
    errors = 0

    def _invoke() -> float | None:
        try:
            result = client.synthesize(model, text)
            return result["latency_ms"]
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(_invoke) for _ in range(concurrency)]
        for future in as_completed(futures):
            latency = future.result()
            if latency is not None:
                latencies.append(latency)
            else:
                errors += 1

    if not latencies:
        return {
            "success_rate": 0.0,
            "p50_ms": 0,
            "p90_ms": 0,
            "p99_ms": 0,
            "mean_ms": 0,
            "errors": errors,
        }

    arr = np.array(latencies)
    return {
        "success_rate": round(len(latencies) / concurrency, 3),
        "p50_ms": round(float(np.percentile(arr, 50)), 1),
        "p90_ms": round(float(np.percentile(arr, 90)), 1),
        "p99_ms": round(float(np.percentile(arr, 99)), 1),
        "mean_ms": round(float(arr.mean()), 1),
        "errors": errors,
    }
