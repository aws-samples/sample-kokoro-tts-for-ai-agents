"""Base protocol for model-specific STT handlers.

Handlers contain the core inference logic shared between
SageMaker endpoints and local extractors.
"""

from typing import Any, Protocol

import numpy as np
import soundfile as sf

from stt_inference.types import TranscriptionResult


class STTHandler(Protocol):
    """Protocol defining the interface for STT handlers."""

    def model_fn(self, model_dir: str) -> dict[str, Any]:
        """Load model and processor at startup.

        Args:
            model_dir: Directory containing model artifacts or HuggingFace model ID.

        Returns:
            Dictionary containing model, processor, device, and any other artifacts.
        """
        ...

    def predict_fn(
        self,
        audio_path: str,
        model_artifacts: dict[str, Any],
    ) -> TranscriptionResult:
        """Run STT on an audio file.

        Args:
            audio_path: Path to audio file.
            model_artifacts: Artifacts from model_fn().

        Returns:
            TranscriptionResult with text, segments, and timing.
        """
        ...


def load_audio(path: str, target_sr: int = 16000) -> tuple[np.ndarray, int]:
    """Load audio file and resample to target sample rate.

    Args:
        path: Path to audio file.
        target_sr: Target sample rate in Hz.

    Returns:
        Tuple of (audio array, sample rate).
    """
    audio, sr = sf.read(path, dtype="float32")
    if sr != target_sr:
        try:
            import librosa

            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        except ImportError as err:
            raise RuntimeError(
                f"Audio sample rate {sr} != {target_sr} and librosa is not installed "
                "for resampling. Install librosa or provide audio at the target sample rate."
            ) from err
        sr = target_sr
    return audio, sr
