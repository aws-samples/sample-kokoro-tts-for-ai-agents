"""Tests for report generation."""

from __future__ import annotations

import json
from pathlib import Path

from tts_eval.report import generate_report
from tts_eval.runner import EvalResult


class TestGenerateReport:
    def test_produces_json_and_markdown(self, tmp_path: Path) -> None:
        results = [
            EvalResult(
                model="kokoro-82m",
                sample_id="s1",
                text="Hello world",
                utmos=3.5,
                wer=0.1,
                transcript="hello world",
                latency_ms=1200.0,
                audio_duration_s=1.5,
            ),
            EvalResult(
                model="kokoro-82m",
                sample_id="s2",
                text="Test sentence",
                utmos=3.8,
                wer=0.0,
                transcript="test sentence",
                latency_ms=1100.0,
                audio_duration_s=1.2,
            ),
        ]

        json_path, md_path = generate_report(results, tmp_path)

        assert json_path.exists()
        assert md_path.exists()
        assert json_path.suffix == ".json"
        assert md_path.suffix == ".md"

    def test_json_structure(self, tmp_path: Path) -> None:
        results = [
            EvalResult(
                model="orpheus-3b",
                sample_id="s1",
                text="Test",
                utmos=3.2,
                latency_ms=2000.0,
                audio_duration_s=2.0,
            ),
        ]

        json_path, _ = generate_report(results, tmp_path)
        data = json.loads(json_path.read_text())

        assert "timestamp" in data
        assert "models" in data
        assert "orpheus-3b" in data["models"]
        assert "summary" in data

    def test_markdown_contains_model_table(self, tmp_path: Path) -> None:
        results = [
            EvalResult(
                model="kokoro-82m",
                sample_id="s1",
                text="Test",
                utmos=3.5,
                latency_ms=1000.0,
                audio_duration_s=1.0,
            ),
            EvalResult(
                model="orpheus-3b",
                sample_id="s1",
                text="Test",
                utmos=3.3,
                latency_ms=2000.0,
                audio_duration_s=1.5,
            ),
        ]

        _, md_path = generate_report(results, tmp_path)
        content = md_path.read_text()

        assert "kokoro-82m" in content
        assert "orpheus-3b" in content
        assert "UTMOS" in content or "MOS" in content

    def test_handles_failed_results(self, tmp_path: Path) -> None:
        results = [
            EvalResult(
                model="kokoro-82m",
                sample_id="s1",
                text="Test",
                error="Connection refused",
            ),
        ]

        json_path, md_path = generate_report(results, tmp_path)
        assert json_path.exists()
        assert md_path.exists()
