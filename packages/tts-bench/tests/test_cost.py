"""Tests for cost model calculations."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from speech_infra.config import TTS_MODEL_CONFIGS
from tts_bench.cost import (
    DEFAULT_INSTANCE_TYPE,
    INSTANCE_COST_PER_HOUR,
    MODEL_INSTANCE_TYPES,
    POLLY_COST_PER_M_CHARS,
    SATURATION_LEVELS,
    calculate_cost,
    cost_per_m_chars,
    find_saturation_concurrency,
    measure_sustained_throughput,
)
from tts_eval.synthesize import ENDPOINT_MAP
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

    def test_legacy_saturation_ladder_is_geometric(self) -> None:
        # Correct about the old code, and the reason it is legacy: a doubling
        # ladder starts at 2, so it cannot resolve a C_max of 1 from 2 — on
        # Kokoro (capacity 1) throughput pins at every level, the 20% plateau
        # test trips immediately, and find_saturation_concurrency returns 4.
        # tts_bench.cmax uses a fine-grained ladder in lambda for that reason.
        for i in range(1, len(SATURATION_LEVELS)):
            assert SATURATION_LEVELS[i] == SATURATION_LEVELS[i - 1] * 2


class TestRegistryConsistency:
    """`MODEL_INSTANCE_TYPES` and `ENDPOINT_MAP` duplicate `TTS_MODEL_CONFIGS`.

    The duplication is deliberate — importing `speech_infra.config` at runtime
    would drag `aws-cdk-lib` into a benchmarking package for two dict lookups —
    but the copies must not drift, or the planner prices a model against the
    wrong instance type and `cmax` freezes the wrong endpoint.
    """

    def test_instance_types_match_deployment_config(self) -> None:
        for model, instance_type in MODEL_INSTANCE_TYPES.items():
            config = TTS_MODEL_CONFIGS.get(str(model))
            assert config is not None, f"{model} has a cost entry but no deployment config"
            assert instance_type == config.instance_type, (
                f"{model}: cost.py says {instance_type}, config.py deploys {config.instance_type}"
            )

    def test_endpoint_names_match_deployment_config(self) -> None:
        for model, endpoint in ENDPOINT_MAP.items():
            config = TTS_MODEL_CONFIGS.get(str(model))
            assert config is not None, f"{model} has an endpoint mapping but no deployment config"
            assert endpoint == config.endpoint_name

    def test_deployed_pytorch_models_are_all_priced(self) -> None:
        # maya-veena is deliberately absent from both benchmark registries: it
        # is deployed but has no eval/bench wiring. Asserting the exact gap
        # means adding it to config.py without adding it here fails here,
        # rather than silently falling back to DEFAULT_INSTANCE_TYPE.
        unpriced = {name for name in TTS_MODEL_CONFIGS if name not in MODEL_INSTANCE_TYPES}
        assert unpriced == {"maya-veena"}


class TestCostPerMChars:
    def test_known_instance_type(self) -> None:
        # 1.408 $/hr over 1M chars/hr is 1.408 $/M chars.
        assert cost_per_m_chars(1_000_000, "ml.g5.xlarge") == pytest.approx(1.408)

    def test_fleet_cost_scales_with_instance_count(self) -> None:
        # chars_per_hr is fleet-wide, so count scales cost only. A fleet held at
        # 50% for surge headroom costs twice per character what a saturated
        # single instance does — that is the number the planner is reporting.
        one = cost_per_m_chars(1_000_000, "ml.g5.xlarge", instance_count=1)
        two = cost_per_m_chars(1_000_000, "ml.g5.xlarge", instance_count=2)
        assert two == pytest.approx(one * 2)

    def test_unknown_instance_type_falls_back_to_default(self) -> None:
        fallback = cost_per_m_chars(1_000_000, "ml.does.not.exist")
        assert fallback == pytest.approx(INSTANCE_COST_PER_HOUR[DEFAULT_INSTANCE_TYPE])

    def test_zero_throughput_is_infinite_not_free(self) -> None:
        # 0.0 would sort a dead endpoint to the top of a cheapest-first table.
        assert cost_per_m_chars(0.0, "ml.g5.xlarge") == float("inf")

    def test_negative_throughput_is_infinite(self) -> None:
        assert cost_per_m_chars(-5.0, "ml.g5.xlarge") == float("inf")

    def test_rejects_fleet_smaller_than_one_instance(self) -> None:
        with pytest.raises(ValueError, match="instance_count must be >= 1"):
            cost_per_m_chars(1_000_000, "ml.g5.xlarge", instance_count=0)

    def test_cheaper_instance_costs_less_per_char_at_equal_throughput(self) -> None:
        gpu = cost_per_m_chars(500_000, "ml.g5.xlarge")
        cpu = cost_per_m_chars(500_000, "ml.c5.2xlarge")
        assert cpu < gpu


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


class TestPollyCost:
    def test_polly_models_have_fixed_pricing(self) -> None:
        assert TTSModelName.POLLY_STANDARD in POLLY_COST_PER_M_CHARS
        assert TTSModelName.POLLY_NEURAL in POLLY_COST_PER_M_CHARS
        assert TTSModelName.POLLY_GENERATIVE in POLLY_COST_PER_M_CHARS

    def test_polly_cost_values(self) -> None:
        assert POLLY_COST_PER_M_CHARS[TTSModelName.POLLY_STANDARD] == 4.00
        assert POLLY_COST_PER_M_CHARS[TTSModelName.POLLY_NEURAL] == 16.00
        assert POLLY_COST_PER_M_CHARS[TTSModelName.POLLY_GENERATIVE] == 30.00

    def test_calculate_cost_short_circuits_for_polly(self) -> None:
        result = calculate_cost("polly-neural", texts=["test"], region="us-east-1")
        assert result["cost_per_m_chars"] == 16.00
        assert result["instance_type"] == "managed"
        assert result["total_requests"] == 0


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
