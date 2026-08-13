"""Cost model - calculates $/M chars based on saturated concurrent throughput."""

from __future__ import annotations

import itertools
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from loguru import logger

from tts_bench.invoke import resolve_endpoint, resolve_voice
from tts_client.client import TTSClient
from tts_client.types import SynthesisRequest
from tts_inference.types import TTSModelName

#: SageMaker real-time inference, us-east-1, on-demand. The g5 and g6 rows were read
#: from the Pricing API (``USE1-Host`` usagetype) on 2026-07-30; the rest predate that.
#:
#: Note the per-GPU arithmetic, because it is the whole question behind a multi-GPU
#: container: ``ml.g6.12xlarge`` has four L4s at $1.438/GPU-hr against $1.1267 for one
#: on an ``ml.g6.xlarge`` — 5.1x the price of a single-GPU box. So a container driving
#: four GPUs has to serve more than 5x the throughput of a single-GPU one just to break
#: even on unit cost. What it buys instead is one ``T_total`` per four GPUs of capacity
#: rather than four, and a fleet that steps in units of four.
INSTANCE_COST_PER_HOUR: dict[str, float] = {
    "ml.g5.xlarge": 1.408,
    "ml.g5.2xlarge": 2.816,
    "ml.g5.4xlarge": 5.632,
    "ml.g6.xlarge": 1.1267,
    "ml.g6.12xlarge": 5.752,
    "ml.g4dn.xlarge": 0.736,
    "ml.g4dn.2xlarge": 1.120,
    "ml.p3.2xlarge": 4.284,
    "ml.c5.xlarge": 0.238,
    "ml.c5.2xlarge": 0.476,
}

#: Fallback only. `cmax` reads the type off the endpoint and warns when it disagrees with
#: this dict, because a registry states what *should* be deployed and a measurement has to
#: record what *is*. Keep it in step with `speech_infra.config` all the same: `cost_per_m_chars`
#: has no endpoint to ask, so a stale row here silently misprices.
MODEL_INSTANCE_TYPES: dict[str, str] = {
    TTSModelName.KOKORO_82M: "ml.g5.xlarge",
}

POLLY_COST_PER_M_CHARS: dict[str, float] = {
    TTSModelName.POLLY_STANDARD: 4.00,
    TTSModelName.POLLY_NEURAL: 16.00,
    TTSModelName.POLLY_GENERATIVE: 30.00,
}

SATURATION_LEVELS = [2, 4, 8, 16, 32]

DEFAULT_INSTANCE_TYPE = "ml.g5.xlarge"


def hourly_rate(instance_type: str) -> float:
    """The hourly rate for an instance type, warning loudly when it is a guess.

    Falls back to ``DEFAULT_INSTANCE_TYPE`` rather than raising, because a missing
    price should not lose a completed measurement — but it warns, because sweeping
    instance types is now a normal activity and the error is unbounded in the wrong
    direction. Pricing an ``ml.g6.12xlarge`` fleet at ``ml.g5.xlarge`` rates
    understates cost by 4x, and nothing about the resulting figure looks wrong.
    """
    hourly = INSTANCE_COST_PER_HOUR.get(instance_type)
    if hourly is None:
        fallback = INSTANCE_COST_PER_HOUR[DEFAULT_INSTANCE_TYPE]
        logger.warning(
            "No price for {}; costing it at {} rates (${:.4f}/hr). This figure is not "
            "trustworthy — add the type to INSTANCE_COST_PER_HOUR from the Pricing API.",
            instance_type,
            DEFAULT_INSTANCE_TYPE,
            fallback,
        )
        return fallback
    return hourly


def cost_per_m_chars(
    chars_per_hr: float,
    instance_type: str,
    instance_count: int = 1,
) -> float:
    """Dollars per million characters.

    ``chars_per_hr`` is the throughput of the **whole fleet**, not of one
    instance, and ``instance_count`` scales only the cost side. Deliberately no
    linear-scaling assumption: a planned fleet is sized to scale out at
    ``C_scale_max``, below ``Q_max``, so it runs with queue headroom rather than
    saturated and its useful throughput is well below
    ``instance_count x saturated_per_instance``. Passing per-instance throughput
    with ``instance_count=N`` would divide by an ``N`` that never appears in the
    numerator's reality and report the saturated unit cost for an idle fleet.

    ``calculate_cost`` measures one instance and passes ``instance_count=1``;
    the planner passes the fleet throughput at ``C_scale_max`` alongside
    ``N_peak``, which is why reserving surge headroom shows up as a higher unit
    cost.

    Returns:
        ``inf`` when throughput is zero — an endpoint that produces nothing has
        no meaningful cost per character, and returning 0.0 would make a broken
        model look free.

    Raises:
        ValueError: If ``instance_count`` < 1.
    """
    if instance_count < 1:
        raise ValueError(f"instance_count must be >= 1, got {instance_count}")
    if chars_per_hr <= 0:
        return float("inf")
    return (hourly_rate(instance_type) * instance_count / chars_per_hr) * 1_000_000


def find_saturation_concurrency(
    client: TTSClient,
    endpoint: str,
    voice: str,
    text: str,
    max_concurrency: int = 32,
) -> int:
    """Find the concurrency level that maximizes throughput.

    Tests doubling levels (2, 4, 8, 16, 32) by sending a burst of N
    concurrent streaming requests. Measures throughput (chars/sec) at
    each level. Returns the level where throughput plateaus — adding
    more concurrency yields < 20% improvement.

    .. warning::
        Not suitable for capacity planning; use ``tts_bench.qmax`` instead. It
        answers the wrong question: throughput plateaus at the point the server
        is saturated, whereas what bounds the SLO is the *wait*, which keeps
        growing long after throughput has flattened. On Kokoro the plateau trips
        at level 4 while requests still reach first byte in 300ms — nowhere near
        the concurrency where the 3s promise breaks. ``qmax`` steps concurrency
        against the SLO itself, which is the quantity a scaling policy needs.
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

        request = SynthesisRequest(text=text, voice=voice)
        with ThreadPoolExecutor(max_workers=level) as executor:
            futures = [executor.submit(client.synthesize, endpoint, request) for _ in range(level)]
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
    client: TTSClient,
    endpoint: str,
    voice: str,
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
                result = client.synthesize(endpoint, SynthesisRequest(text=text, voice=voice))
                with lock:
                    counters["total_chars"] += result.chars
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

    client = TTSClient(region=region)
    endpoint = resolve_endpoint(model)
    voice = resolve_voice(model)

    instance_type = MODEL_INSTANCE_TYPES.get(model, DEFAULT_INSTANCE_TYPE)
    instance_cost = hourly_rate(instance_type)

    probe_text = texts[0] if texts else "The birch canoe slid on the smooth planks."
    saturation = find_saturation_concurrency(
        client, endpoint, voice, probe_text, max_concurrency=max_concurrency
    )

    throughput = measure_sustained_throughput(
        client, endpoint, voice, texts, concurrency=saturation, window_s=window_s
    )

    chars_per_hr = throughput["chars_per_hr"]
    # Single instance here; the planner calls cost_per_m_chars with N_peak.
    unit_cost = cost_per_m_chars(chars_per_hr, instance_type, instance_count=1)

    return {
        "model": model.value,
        "instance_type": instance_type,
        "instance_cost_per_hr": instance_cost,
        "saturation_concurrency": saturation,
        "chars_per_hr": round(chars_per_hr, 0),
        "chars_per_min": round(chars_per_hr / 60, 1),
        "cost_per_m_chars": round(unit_cost, 2),
        "total_requests": throughput["total_requests"],
        "window_s": throughput["wall_time_s"],
    }
