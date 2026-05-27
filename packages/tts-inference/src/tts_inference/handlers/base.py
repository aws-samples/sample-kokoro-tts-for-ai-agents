"""Base protocol for model-specific TTS handlers.

Handlers contain the core inference logic shared between
SageMaker endpoints and local synthesizers.
"""

from typing import Any, Protocol

from tts_inference.types import VoiceConfig


class TTSHandler(Protocol):
    """Protocol defining the interface for TTS handlers."""

    def model_fn(self, model_dir: str) -> dict[str, Any]:
        """Load model and vocoder at startup.

        Args:
            model_dir: Directory containing model artifacts or HuggingFace model ID.

        Returns:
            Dictionary containing model, vocoder, device, and any other artifacts.
        """
        ...

    def predict_fn(
        self,
        text: str,
        model_artifacts: dict[str, Any],
        voice_config: VoiceConfig | None = None,
    ) -> tuple[bytes, int, float]:
        """Synthesize speech from text.

        Args:
            text: Input text to synthesize.
            model_artifacts: Artifacts from model_fn().
            voice_config: Optional voice configuration.

        Returns:
            Tuple of (wav_bytes, sample_rate, duration_seconds).
        """
        ...
