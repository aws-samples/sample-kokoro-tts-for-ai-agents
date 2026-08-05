"""Latency benchmarking - measures time-to-first-audio-byte."""

from __future__ import annotations

import numpy as np
from loguru import logger

from tts_bench.invoke import resolve_endpoint, resolve_voice
from tts_bench.types import LatencyStats
from tts_client.client import TTSClient
from tts_client.polly import PollyClient
from tts_client.types import SynthesisRequest
from tts_eval.synthesize import POLLY_VOICES
from tts_inference.types import TTSModelName

_POLLY_MODELS = (
    TTSModelName.POLLY_STANDARD,
    TTSModelName.POLLY_NEURAL,
    TTSModelName.POLLY_GENERATIVE,
)


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
    model = TTSModelName(model)
    total = len(texts) * runs_per_text
    latencies: list[float] = []

    # Polly is a managed API, not a SageMaker endpoint -- resolve_endpoint has
    # nothing to resolve for it, so it keeps going through PollyClient.
    # Split into two loops (rather than a client: TTSClient | PollyClient
    # union used from one) because the two clients' synthesize() signatures
    # are unrelated -- (voice_id, engine, text) -> result vs (endpoint,
    # request) -> result -- and mypy cannot narrow which one applies from a
    # runtime bool.
    if model in _POLLY_MODELS:
        polly_client = PollyClient(region=region)
        voice_config = POLLY_VOICES[model]
        for i, text in enumerate(texts):
            for run in range(runs_per_text):
                try:
                    result = polly_client.synthesize(
                        voice_id=voice_config["voice_id"],
                        engine=voice_config["engine"],
                        text=text,
                    )
                    latencies.append(result.latency_ms)
                except Exception as e:
                    logger.warning(
                        "Latency run failed ({}/{}): {}", i * runs_per_text + run + 1, total, e
                    )
    else:
        client = TTSClient(region=region)
        endpoint = resolve_endpoint(model)
        voice = resolve_voice(model)
        for i, text in enumerate(texts):
            for run in range(runs_per_text):
                try:
                    request = SynthesisRequest(text=text, voice=voice)
                    latencies.append(client.synthesize(endpoint, request).latency_ms)
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
