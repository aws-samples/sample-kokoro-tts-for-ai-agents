"""Types for TTS performance benchmarking."""

from __future__ import annotations

from pydantic import BaseModel, Field

from tts_inference.types import TTSModelName


class LatencyStats(BaseModel):
    """Latency statistics for a benchmark run."""

    p50_ms: float
    p90_ms: float
    p95_ms: float
    p99_ms: float
    mean_ms: float
    min_ms: float
    max_ms: float


class ThroughputResult(BaseModel):
    """Throughput benchmark result."""

    model_name: TTSModelName
    chars_per_second: float = Field(description="Characters synthesized per wall-clock second")
    realtime_factor: float = Field(description="RTF: processing_time / audio_duration")
    latency: LatencyStats
    num_samples: int
    total_chars: int
    total_audio_seconds: float
    total_elapsed_seconds: float


class BenchmarkReport(BaseModel):
    """Aggregate benchmark report."""

    model_name: TTSModelName
    throughput: ThroughputResult | None = None
    latency: LatencyStats | None = None
