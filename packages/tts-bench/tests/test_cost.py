"""Tests for cost model calculations."""

from __future__ import annotations

from unittest.mock import MagicMock

from tts_bench.cost import (
    INSTANCE_COST_PER_HOUR,
    MODEL_INSTANCE_TYPES,
    SATURATION_LEVELS,
    find_saturation_concurrency,
    measure_sustained_throughput,
)
from tts_inference.types import TTSModelName


class TestCostConfig:
    def test_deployed_models_have_instance_types(self) -> None:
        deployed = [
            TTSModelName.ORPHEUS_3B,
            TTSModelName.KOKORO_82M,
            TTSModelName.CHATTERBOX_TURBO,
        ]
        for model in deployed:
            assert model in MODEL_INSTANCE_TYPES

    def test_all_instance_types_have_costs(self) -> None:
        for instance_type in MODEL_INSTANCE_TYPES.values():
            assert instance_type in INSTANCE_COST_PER_HOUR

    def test_instance_costs_are_positive(self) -> None:
        for instance_type, cost in INSTANCE_COST_PER_HOUR.items():
            assert cost > 0, f"{instance_type} has non-positive cost: {cost}"

    def test_saturation_levels_are_doubling(self) -> None:
        for i in range(1, len(SATURATION_LEVELS)):
            assert SATURATION_LEVELS[i] == SATURATION_LEVELS[i - 1] * 2


class TestFindSaturationConcurrency:
    def test_returns_last_good_level(self) -> None:
        client = MagicMock()
        client.synthesize_stream.return_value = {"chars": 40, "latency_ms": 1000}

        result = find_saturation_concurrency(
            client, TTSModelName.KOKORO_82M, "test", max_concurrency=8
        )
        assert result in SATURATION_LEVELS

    def test_handles_failures_at_high_concurrency(self) -> None:
        client = MagicMock()
        call_count = {"n": 0}

        def _mock_stream(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] > 4:
                raise RuntimeError("overloaded")
            return {"chars": 40, "latency_ms": 1000}

        client.synthesize_stream.side_effect = _mock_stream

        result = find_saturation_concurrency(
            client, TTSModelName.KOKORO_82M, "test", max_concurrency=8
        )
        assert result >= 1


class TestMeasureSustainedThroughput:
    def test_counts_completed_requests(self) -> None:
        client = MagicMock()
        client.synthesize_stream.return_value = {"chars": 40, "latency_ms": 100}

        result = measure_sustained_throughput(
            client,
            TTSModelName.KOKORO_82M,
            ["hello world"],
            concurrency=2,
            window_s=1.0,
        )

        assert result["total_requests"] > 0
        assert result["total_chars"] > 0
        assert result["chars_per_hr"] > 0
        assert result["wall_time_s"] >= 1.0
