"""Tests for time-windowed scalability benchmarking logic."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from tts_bench.scalability import _run_concurrent, measure_scalability


class TestRunConcurrent:
    def test_measures_throughput(self) -> None:
        client = MagicMock()
        client.synthesize_stream.return_value = {
            "latency_ms": 100.0,
            "ttfab_ms": 50.0,
            "chars": 42,
        }

        result = _run_concurrent(
            client, "kokoro-82m", "test text of forty two chars length!!", 2, 0.5
        )

        assert result["throughput_chars_per_s"] > 0
        assert result["p50_ms"] > 0
        assert result["p99_ms"] > 0
        assert result["mean_ms"] > 0
        assert result["ttfab_p50_ms"] > 0
        assert result["ttfab_p99_ms"] > 0
        assert result["total_requests"] > 0
        assert result["window_s"] > 0
        assert client.synthesize_stream.call_count >= 2

    def test_handles_all_errors(self) -> None:
        client = MagicMock()
        client.synthesize_stream.side_effect = RuntimeError("endpoint down")

        result = _run_concurrent(client, "kokoro-82m", "test text", 2, 0.3)

        assert result["throughput_chars_per_s"] == 0.0
        assert result["total_requests"] == 0
        assert result["p50_ms"] == 0.0

    def test_handles_partial_errors(self) -> None:
        client = MagicMock()
        call_count = {"n": 0}

        def _side_effect(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] % 3 == 0:
                raise RuntimeError("fail")
            return {"latency_ms": 150.0, "ttfab_ms": 80.0, "chars": 10}

        client.synthesize_stream.side_effect = _side_effect

        result = _run_concurrent(client, "kokoro-82m", "test text", 2, 0.5)

        assert result["total_requests"] > 0
        assert result["throughput_chars_per_s"] > 0


class TestMeasureScalability:
    @patch("tts_bench.scalability._run_concurrent")
    def test_runs_all_levels(self, mock_run) -> None:
        def _make_result(*args, **kwargs):
            return {
                "throughput_chars_per_s": 100.0,
                "p50_ms": 50.0,
                "p90_ms": 80.0,
                "p99_ms": 120.0,
                "mean_ms": 60.0,
                "ttfab_p50_ms": 30.0,
                "ttfab_p99_ms": 70.0,
                "total_requests": 20,
                "window_s": 15.0,
            }

        mock_run.side_effect = _make_result

        results = measure_scalability("kokoro-82m", "test", [2, 4, 8], window_s=1.0)

        assert len(results) == 3
        assert results[0]["concurrency"] == 2
        assert results[1]["concurrency"] == 4
        assert results[2]["concurrency"] == 8
        assert all(r["model"] == "kokoro-82m" for r in results)
        assert mock_run.call_count == 3
