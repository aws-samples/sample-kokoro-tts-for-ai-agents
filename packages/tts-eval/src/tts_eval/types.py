# Copyright (c) 2026 Amazon.com
# This file is licensed under the MIT License.
# See the LICENSE file in the project root for full license information.

"""Types for TTS evaluation results."""

from __future__ import annotations

from pydantic import BaseModel, Field

from tts_inference.types import TTSModelName


class NaturalnessScore(BaseModel):
    """Naturalness evaluation for a single sample."""

    model_name: TTSModelName
    source_text: str
    mos: float = Field(ge=1.0, le=5.0, description="Mean Opinion Score (1-5)")
    pesq: float | None = Field(default=None, description="PESQ score (-0.5 to 4.5)")
    stoi: float | None = Field(default=None, ge=0.0, le=1.0, description="STOI (0-1)")


class SpeakerSimilarityScore(BaseModel):
    """Speaker similarity evaluation for voice cloning."""

    model_name: TTSModelName
    source_text: str
    cosine_similarity: float = Field(ge=-1.0, le=1.0)
    reference_audio: str


class TTSEvalReport(BaseModel):
    """Aggregate TTS evaluation report."""

    model_name: TTSModelName
    num_samples: int
    mean_mos: float
    mean_pesq: float | None = None
    mean_stoi: float | None = None
    naturalness_scores: list[NaturalnessScore]
    speaker_scores: list[SpeakerSimilarityScore] | None = None
