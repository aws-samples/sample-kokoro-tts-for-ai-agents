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
            model="kokoro-82m",
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
            model="kokoro-82m",
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

    def test_to_dict_includes_rtf_and_ttfab(self) -> None:
        result = EvalResult(
            model="kokoro-82m",
            sample_id="s1",
            text="Test",
            latency_ms=500.0,
            audio_duration_s=2.0,
            rtf=0.25,
            ttfab_ms=120.0,
        )
        d = result.to_dict()
        assert d["rtf"] == 0.25
        assert d["ttfab_ms"] == 120.0

    def test_to_dict_rtf_and_ttfab_default_none(self) -> None:
        result = EvalResult(
            model="kokoro-82m",
            sample_id="s1",
            text="Test",
        )
        d = result.to_dict()
        assert d["rtf"] is None
        assert d["ttfab_ms"] is None
