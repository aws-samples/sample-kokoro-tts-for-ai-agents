"""Types for STT evaluation results."""

from __future__ import annotations

from pydantic import BaseModel, Field

from stt_inference.types import STTModelName


class WERResult(BaseModel):
    """Word Error Rate evaluation result for a single sample."""

    model_name: STTModelName
    source: str
    wer: float = Field(ge=0.0, description="Word Error Rate (0.0 = perfect)")
    mer: float = Field(ge=0.0, description="Match Error Rate")
    wil: float = Field(ge=0.0, description="Word Information Lost")
    substitutions: int = 0
    deletions: int = 0
    insertions: int = 0
    reference_length: int = 0


class CERResult(BaseModel):
    """Character Error Rate evaluation result for a single sample."""

    model_name: STTModelName
    source: str
    cer: float = Field(ge=0.0, description="Character Error Rate (0.0 = perfect)")
    reference_length: int = 0


class EvalReport(BaseModel):
    """Aggregate evaluation report across samples."""

    model_name: STTModelName
    num_samples: int
    mean_wer: float
    mean_cer: float
    median_wer: float
    median_cer: float
    wer_results: list[WERResult]
    cer_results: list[CERResult]
