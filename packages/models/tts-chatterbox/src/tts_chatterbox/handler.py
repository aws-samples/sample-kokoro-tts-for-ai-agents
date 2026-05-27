"""Chatterbox-Turbo TTS model handler -- load and synthesize."""

from io import BytesIO
from typing import Any

import torch
import torchaudio

from tts_inference.types import VoiceConfig

SAMPLE_RATE = 24000


class ChatterboxHandler:
    """Handler for Chatterbox-Turbo TTS with voice cloning."""

    def model_fn(self, model_dir: str) -> dict[str, Any]:
        from chatterbox.tts import ChatterboxTTS

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = ChatterboxTTS.from_pretrained(device=device)

        return {"model": model, "device": device}

    def predict_fn(
        self,
        text: str,
        model_artifacts: dict[str, Any],
        voice_config: VoiceConfig | None = None,
    ) -> tuple[bytes, int, float]:
        model = model_artifacts["model"]

        reference_audio = None
        if voice_config and voice_config.reference_audio_path:
            reference_audio = voice_config.reference_audio_path

        if reference_audio is None:
            raise ValueError(
                "Chatterbox-Turbo requires a reference audio path for voice cloning. "
                "Set voice_config.reference_audio_path."
            )

        wav = model.generate(text, audio_prompt_path=reference_audio)

        audio_np = wav.squeeze().cpu().numpy()
        duration = len(audio_np) / SAMPLE_RATE

        buffer = BytesIO()
        torchaudio.save(buffer, wav.cpu(), SAMPLE_RATE, format="wav")
        wav_bytes = buffer.getvalue()

        return wav_bytes, SAMPLE_RATE, duration
