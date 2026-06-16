"""Tests for EvalResult serialization."""

from __future__ import annotations

from tts_eval.runner import EvalResult


class TestEvalResult:
    def test_to_dict_basic(self) -> None:
        result = EvalResult(
            model="kokoro-82m",
            sample_id="harvard-001",
            text="Hello world",
            utmos=3.5,
            wer=0.05,
            transcript="hello world",
            latency_ms=1200.0,
            audio_duration_s=1.5,
        )
        d = result.to_dict()
        assert d["model"] == "kokoro-82m"
        assert d["sample_id"] == "harvard-001"
        assert d["utmos"] == 3.5
        assert d["wer"] == 0.05
        assert d["latency_ms"] == 1200.0
        assert "error" not in d

    def test_to_dict_with_error(self) -> None:
        result = EvalResult(
            model="orpheus-3b",
            sample_id="s1",
            text="Test",
            error="Synthesis failed: timeout",
        )
        d = result.to_dict()
        assert d["error"] == "Synthesis failed: timeout"
        assert d["utmos"] is None
        assert d["wer"] is None

    def test_to_dict_partial_scores(self) -> None:
        result = EvalResult(
            model="chatterbox-turbo",
            sample_id="s1",
            text="Test",
            utmos=3.2,
            wer=None,
            latency_ms=3000.0,
            audio_duration_s=5.0,
        )
        d = result.to_dict()
        assert d["utmos"] == 3.2
        assert d["wer"] is None
        assert d["latency_ms"] == 3000.0
