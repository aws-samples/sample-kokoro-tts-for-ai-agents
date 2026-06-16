"""Tests for scalability benchmarking logic."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from tts_bench.scalability import _run_concurrent


class TestRunConcurrent:
    def test_all_success(self) -> None:
        client = MagicMock()
        client.synthesize.return_value = {"latency_ms": 100.0}

        result = _run_concurrent(client, "kokoro-82m", "test text", concurrency=5)

        assert result["success_rate"] == 1.0
        assert result["errors"] == 0
        assert result["p50_ms"] > 0
        assert client.synthesize.call_count == 5

    def test_all_failures(self) -> None:
        client = MagicMock()
        client.synthesize.side_effect = RuntimeError("endpoint down")

        result = _run_concurrent(client, "kokoro-82m", "test text", concurrency=3)

        assert result["success_rate"] == 0.0
        assert result["errors"] == 3

    def test_partial_failures(self) -> None:
        client = MagicMock()
        call_count = {"n": 0}

        def _side_effect(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] % 2 == 0:
                raise RuntimeError("fail")
            return {"latency_ms": 150.0}

        client.synthesize.side_effect = _side_effect

        result = _run_concurrent(client, "kokoro-82m", "test text", concurrency=4)

        assert 0 < result["success_rate"] < 1.0
        assert result["errors"] > 0
