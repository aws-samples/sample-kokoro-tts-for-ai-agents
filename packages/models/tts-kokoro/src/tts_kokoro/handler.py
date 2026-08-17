# Copyright (c) 2026 Amazon.com
# This file is licensed under the MIT License.
# See the LICENSE file in the project root for full license information.

"""Kokoro-82M model handler -- load and synthesize."""

from io import BytesIO
from typing import Any

import numpy as np
import soundfile as sf
from kokoro import KPipeline

from tts_inference.types import VoiceConfig

SAMPLE_RATE = 24000
DEFAULT_VOICE = "af_heart"


class KokoroHandler:
    """Handler for Kokoro-82M TTS."""

    def model_fn(self, model_dir: str) -> dict[str, Any]:
        pipeline = KPipeline(lang_code="a")
        return {"pipeline": pipeline}

    def predict_fn(
        self,
        text: str,
        model_artifacts: dict[str, Any],
        voice_config: VoiceConfig | None = None,
    ) -> tuple[bytes, int, float]:
        pipeline = model_artifacts["pipeline"]

        voice = DEFAULT_VOICE
        speed = 1.0
        if voice_config:
            if voice_config.voice_id:
                voice = voice_config.voice_id
            speed = voice_config.speed

        audio_chunks: list[np.ndarray] = []
        for _graphemes, _phonemes, audio in pipeline(text, voice=voice, speed=speed):
            if audio is not None:
                audio_chunks.append(audio)

        if not audio_chunks:
            return b"", SAMPLE_RATE, 0.0

        full_audio = np.concatenate(audio_chunks)
        duration = len(full_audio) / SAMPLE_RATE

        buffer = BytesIO()
        sf.write(buffer, full_audio, SAMPLE_RATE, format="WAV")
        wav_bytes = buffer.getvalue()

        return wav_bytes, SAMPLE_RATE, duration
