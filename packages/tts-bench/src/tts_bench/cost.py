"""Cost model - calculates $/M chars based on throughput and instance pricing."""

from __future__ import annotations

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
}

MODEL_INSTANCE_TYPES: dict[str, str] = {
    TTSModelName.ORPHEUS_3B: "ml.g5.xlarge",
    TTSModelName.KOKORO_82M: "ml.g5.xlarge",
    TTSModelName.CHATTERBOX_TURBO: "ml.g5.xlarge",
}


def calculate_cost(
    model: str | TTSModelName,
    texts: list[str],
    region: str = "us-east-1",
) -> dict:
    """Calculate cost per million characters based on measured throughput.

    Runs synthesis on provided texts, measures total chars / total time,
    then calculates: $/M chars = (instance_cost_per_hr / chars_per_hr) * 1_000_000

    Args:
        model: Model to benchmark.
        texts: Representative text inputs to measure throughput.
        region: AWS region.

    Returns:
        Dict with model, instance_type, chars_per_min, cost_per_m_chars.
    """
    client = SynthesisClient(region=region)
    model = TTSModelName(model)
    instance_type = MODEL_INSTANCE_TYPES.get(model, "ml.g5.xlarge")
    instance_cost = INSTANCE_COST_PER_HOUR.get(instance_type, 1.408)

    total_chars = 0
    total_time_ms = 0.0

    for text in texts:
        try:
            result = client.synthesize(model, text)
            total_chars += result["chars"]
            total_time_ms += result["latency_ms"]
        except Exception as e:
            logger.warning("Cost measurement failed for {}: {}", model.value, e)

    if total_chars == 0 or total_time_ms == 0:
        raise RuntimeError(f"No successful synthesis for cost calculation ({model.value})")

    chars_per_min = total_chars / (total_time_ms / 1000 / 60)
    chars_per_hr = chars_per_min * 60
    cost_per_m_chars = (instance_cost / chars_per_hr) * 1_000_000

    return {
        "model": model.value,
        "instance_type": instance_type,
        "instance_cost_per_hr": instance_cost,
        "chars_per_min": round(chars_per_min, 1),
        "chars_per_hr": round(chars_per_hr, 0),
        "cost_per_m_chars": round(cost_per_m_chars, 2),
        "total_chars_measured": total_chars,
        "total_time_s": round(total_time_ms / 1000, 2),
    }
