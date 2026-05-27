"""STT inference package -- local GPU and SageMaker backends."""

from stt_inference.models.base import get_transcriber, list_available_models
from stt_inference.types import (
    AudioFormat,
    AudioInput,
    ExecutionMode,
    STTModelName,
    TranscriptionResult,
    TranscriptionSegment,
    WordSegment,
)

__all__ = [
    "AudioFormat",
    "AudioInput",
    "ExecutionMode",
    "STTModelName",
    "TranscriptionResult",
    "TranscriptionSegment",
    "WordSegment",
    "get_transcriber",
    "list_available_models",
]
