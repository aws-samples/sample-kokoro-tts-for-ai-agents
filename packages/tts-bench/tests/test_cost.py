"""Tests for cost model calculations."""

from __future__ import annotations

from tts_bench.cost import INSTANCE_COST_PER_HOUR, MODEL_INSTANCE_TYPES
from tts_inference.types import TTSModelName


class TestCostModel:
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
