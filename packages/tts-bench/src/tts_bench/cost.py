"""Cost model - calculates $/M chars based on saturated concurrent throughput."""

from __future__ import annotations

import itertools
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from loguru import logger

from tts_eval.synthesize import SynthesisClient
from tts_inference.types import TTSModelName

INSTANCE_COST_PER_HOUR: dict[str, float] = {
    "ml.g5.xlarge": 1.408,
    "ml.g5.2xlarge": 2.816,
    "ml.g5.4xlarge": 5.632,
    "ml.g4dn.xlarge": 0.736,
    "ml.g4dn.2xlarge": 1.120,
    "ml.p3.2xlarge": 4.284,
    "ml.c5.xlarge": 0.238,
    "ml.c5.2xlarge": 0.476,
}

MODEL_INSTANCE_TYPES: dict[str, str] = {
    TTSModelName.ORPHEUS_3B: "ml.g5.xlarge",
    TTSModelName.KOKORO_82M: "ml.g5.xlarge",
    TTSModelName.KOKORO_82M_CPU: "ml.c5.2xlarge",
    TTSModelName.CHATTERBOX_TURBO: "ml.g5.xlarge",
}

POLLY_COST_PER_M_CHARS: dict[str, float] = {
    TTSModelName.POLLY_STANDARD: 4.00,
    TTSModelName.POLLY_NEURAL: 16.00,
    TTSModelName.POLLY_GENERATIVE: 30.00,
}

SATURATION_LEVELS = [2, 4, 8, 16, 32]


def find_saturation_concurrency(
    client: SynthesisClient,
    model: TTSModelName,
    text: str,
    max_concurrency: int = 32,
) -> int:
    """Find the concurrency level that maximizes throughput.

    Tests doubling levels (2, 4, 8, 16, 32) by sending a burst of N
    concurrent streaming requests. Measures throughput (chars/sec) at
    each level. Returns the level where throughput plateaus — adding
    more concurrency yields < 20% improvement.
    """
    prev_throughput = 0.0
    best_level = 1
    chars = len(text)

    for level in SATURATION_LEVELS:
        if level > max_concurrency:
            break

        logger.info("Testing saturation at concurrency={}", level)
        t0 = time.perf_counter()
        successes = 0
        errors = 0

        with ThreadPoolExecutor(max_workers=level) as executor:
            futures = [executor.submit(client.synthesize_stream, model, text) for _ in range(level)]
            for future in as_completed(futures):
                try:
                    future.result()
                    successes += 1
                except Exception:
                    errors += 1

        elapsed = time.perf_counter() - t0
        throughput = (successes * chars) / elapsed if elapsed > 0 else 0

        logger.info(
            "Concurrency={}: {}/{} success, {:.0f} chars/s ({:.1f}s elapsed)",
            level,
            successes,
            level,
            throughput,
            elapsed,
        )

        if successes == 0:
            break

        if errors > level / 2:
            break

        improvement = (
            (throughput - prev_throughput) / prev_throughput
            if prev_throughput > 0
            else float("inf")
        )
        best_level = level

        if improvement < 0.2 and prev_throughput > 0:
            logger.info("Throughput plateaued at concurrency={}", level)
            break

        prev_throughput = throughput

    logger.info("Saturation concurrency: {}", best_level)
    return best_level


def measure_sustained_throughput(
    client: SynthesisClient,
    model: TTSModelName,
    texts: list[str],
    concurrency: int,
    window_s: float = 60.0,
) -> dict:
    """Run sustained load at given concurrency for a time window.

    Each worker continuously sends streaming requests until the window expires.
    Only fully completed requests are counted.
    """
    stop_event = threading.Event()
    lock = threading.Lock()
    counters = {"total_chars": 0, "total_requests": 0}
    text_iter = itertools.cycle(texts)

    def _worker() -> None:
        while not stop_event.is_set():
            with lock:
                text = next(text_iter)
            try:
                result = client.synthesize_stream(model, text)
                with lock:
                    counters["total_chars"] += result["chars"]
                    counters["total_requests"] += 1
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

    wall_time_s = time.perf_counter() - t0
    chars_per_hr = (counters["total_chars"] / wall_time_s) * 3600 if wall_time_s > 0 else 0

    logger.info(
        "Sustained throughput: {} requests, {} chars in {:.1f}s ({:.0f} chars/hr)",
        counters["total_requests"],
        counters["total_chars"],
        wall_time_s,
        chars_per_hr,
    )

    return {
        "total_chars": counters["total_chars"],
        "total_requests": counters["total_requests"],
        "wall_time_s": round(wall_time_s, 2),
        "chars_per_hr": round(chars_per_hr, 0),
    }


def calculate_cost(
    model: str | TTSModelName,
    texts: list[str],
    region: str = "us-east-1",
    window_s: float = 60.0,
    max_concurrency: int = 32,
) -> dict:
    """Calculate cost per million characters based on saturated throughput.

    1. Finds saturation concurrency (highest level with 100% success)
    2. Runs sustained load at that concurrency for window_s seconds
    3. Calculates: $/M chars = (instance_cost_per_hr / chars_per_hr) * 1_000_000
    """
    client = SynthesisClient(region=region)
    model = TTSModelName(model)

    if model in POLLY_COST_PER_M_CHARS:
        return {
            "model": model.value,
            "instance_type": "managed",
            "instance_cost_per_hr": 0,
            "saturation_concurrency": "N/A",
            "chars_per_hr": 0,
            "chars_per_min": 0,
            "cost_per_m_chars": POLLY_COST_PER_M_CHARS[model],
            "total_requests": 0,
            "window_s": 0,
        }

    instance_type = MODEL_INSTANCE_TYPES.get(model, "ml.g5.xlarge")
    instance_cost = INSTANCE_COST_PER_HOUR.get(instance_type, 1.408)

    probe_text = texts[0] if texts else "The birch canoe slid on the smooth planks."
    saturation = find_saturation_concurrency(
        client, model, probe_text, max_concurrency=max_concurrency
    )

    throughput = measure_sustained_throughput(
        client, model, texts, concurrency=saturation, window_s=window_s
    )

    chars_per_hr = throughput["chars_per_hr"]
    cost_per_m_chars = (
        (instance_cost / chars_per_hr) * 1_000_000 if chars_per_hr > 0 else float("inf")
    )

    return {
        "model": model.value,
        "instance_type": instance_type,
        "instance_cost_per_hr": instance_cost,
        "saturation_concurrency": saturation,
        "chars_per_hr": round(chars_per_hr, 0),
        "chars_per_min": round(chars_per_hr / 60, 1),
        "cost_per_m_chars": round(cost_per_m_chars, 2),
        "total_requests": throughput["total_requests"],
        "window_s": throughput["wall_time_s"],
    }
