"""Scalability benchmarking - time-windowed concurrent load testing via streaming."""

from __future__ import annotations

import threading
import time
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
    window_s: float = 15.0,
) -> list[dict]:
    """Test model under concurrent load using time-windowed streaming invoke.

    For each concurrency level, workers continuously send requests for
    `window_s` seconds. Measures throughput and latency distributions.

    Args:
        model: Model to benchmark.
        text: Text to synthesize (same for all requests).
        concurrency_levels: List of concurrency levels to test (e.g. [5, 10, 20]).
        region: AWS region.
        window_s: Duration of each concurrency window in seconds.

    Returns:
        List of dicts with results per concurrency level.

    .. warning::
        Closed-loop, so not usable for capacity planning; use
        ``tts_bench.loadgen`` / ``tts_bench.cmax`` for that. Each of the N
        workers sends its next request only after the previous one returns
        (``_run_concurrent`` below), which makes the offered rate
        ``N / mean_latency`` — the *server* sets the arrival rate, and a slower
        server is sent less work. The queueing delay a real user would
        experience is therefore never generated, so no latency knee appears.
        Errors are also swallowed rather than classified, so a 503 becomes a hot
        retry loop and ``total_requests`` counts successes only.

        Kept as-is because ``tts_eval.cli._run_benchmarks`` wires its untyped
        dict contract into the eval report.
    """
    client = SynthesisClient(region=region)
    model = TTSModelName(model)
    results = []

    for concurrency in concurrency_levels:
        logger.info("Testing {} at concurrency={} for {:.0f}s", model.value, concurrency, window_s)
        level_result = _run_concurrent(client, model, text, concurrency, window_s)
        level_result["model"] = model.value
        level_result["concurrency"] = concurrency
        results.append(level_result)

    return results


def _run_concurrent(
    client: SynthesisClient,
    model: TTSModelName,
    text: str,
    concurrency: int,
    window_s: float,
) -> dict:
    """Run sustained concurrent load for a time window and collect stats."""
    stop_event = threading.Event()
    lock = threading.Lock()
    latencies: list[float] = []
    ttfab_values: list[float] = []
    total_chars = 0

    def _worker() -> None:
        nonlocal total_chars
        while not stop_event.is_set():
            try:
                result = client.synthesize_stream(model, text)
                lat = float(result["latency_ms"])
                ttfab = float(result["ttfab_ms"])
                with lock:
                    latencies.append(lat)
                    ttfab_values.append(ttfab)
                    total_chars += result["chars"]
            except Exception:
                if stop_event.is_set():
                    break

    t0 = time.perf_counter()

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(_worker) for _ in range(concurrency)]
        stop_event.wait(timeout=window_s)
        stop_event.set()
        for future in as_completed(futures):
            future.result()

    wall_time = time.perf_counter() - t0

    if not latencies:
        return {
            "throughput_chars_per_s": 0.0,
            "p50_ms": 0.0,
            "p90_ms": 0.0,
            "p99_ms": 0.0,
            "mean_ms": 0.0,
            "ttfab_p50_ms": 0.0,
            "ttfab_p99_ms": 0.0,
            "total_requests": 0,
            "window_s": round(wall_time, 2),
        }

    arr = np.array(latencies)
    ttfab_arr = np.array(ttfab_values)
    throughput = total_chars / wall_time if wall_time > 0 else 0.0

    logger.info(
        "Concurrency={}: {} requests in {:.1f}s, {:.0f} chars/s, P50={:.0f}ms P99={:.0f}ms",
        concurrency,
        len(latencies),
        wall_time,
        throughput,
        float(np.percentile(arr, 50)),
        float(np.percentile(arr, 99)),
    )

    return {
        "throughput_chars_per_s": round(throughput, 1),
        "p50_ms": round(float(np.percentile(arr, 50)), 1),
        "p90_ms": round(float(np.percentile(arr, 90)), 1),
        "p99_ms": round(float(np.percentile(arr, 99)), 1),
        "mean_ms": round(float(arr.mean()), 1),
        "ttfab_p50_ms": round(float(np.percentile(ttfab_arr, 50)), 1),
        "ttfab_p99_ms": round(float(np.percentile(ttfab_arr, 99)), 1),
        "total_requests": len(latencies),
        "window_s": round(wall_time, 2),
    }
