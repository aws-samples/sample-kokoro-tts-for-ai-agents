"""Latency benchmarking - measures time-to-first-audio-byte."""

from __future__ import annotations

import numpy as np
from loguru import logger

from tts_bench.types import LatencyStats
from tts_eval.synthesize import SynthesisClient
from tts_inference.types import TTSModelName


def measure_latency(
    model: str | TTSModelName,
    texts: list[str],
    runs_per_text: int = 3,
    region: str = "us-east-1",
) -> LatencyStats:
    """Measure TTFAB latency for a model across samples.

    Args:
        model: Model to benchmark.
        texts: List of text inputs to synthesize.
        runs_per_text: Number of times to synthesize each text.
        region: AWS region.

    Returns:
        LatencyStats with percentile breakdowns.
    """
    client = SynthesisClient(region=region)
    model = TTSModelName(model)
    latencies: list[float] = []

    total = len(texts) * runs_per_text
    for i, text in enumerate(texts):
        for run in range(runs_per_text):
            try:
                result = client.synthesize(model, text)
                latencies.append(result["latency_ms"])
            except Exception as e:
                logger.warning(
                    "Latency run failed ({}/{}): {}", i * runs_per_text + run + 1, total, e
                )

    if not latencies:
        raise RuntimeError(f"All latency measurements failed for {model}")

    arr = np.array(latencies)
    return LatencyStats(
        p50_ms=round(float(np.percentile(arr, 50)), 1),
        p90_ms=round(float(np.percentile(arr, 90)), 1),
        p95_ms=round(float(np.percentile(arr, 95)), 1),
        p99_ms=round(float(np.percentile(arr, 99)), 1),
        mean_ms=round(float(arr.mean()), 1),
        min_ms=round(float(arr.min()), 1),
        max_ms=round(float(arr.max()), 1),
    )
