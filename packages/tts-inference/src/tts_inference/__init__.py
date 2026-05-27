"""TTS inference package -- local GPU and SageMaker backends."""

from tts_inference.models.base import get_synthesizer, list_available_models
from tts_inference.types import (
    AudioEncoding,
    ExecutionMode,
    SynthesisRequest,
    SynthesisResult,
    TTSModelName,
    VoiceConfig,
)

__all__ = [
    "AudioEncoding",
    "ExecutionMode",
    "SynthesisRequest",
    "SynthesisResult",
    "TTSModelName",
    "VoiceConfig",
    "get_synthesizer",
    "list_available_models",
]
