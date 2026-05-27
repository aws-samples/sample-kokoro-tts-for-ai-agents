"""Qwen3-ASR-1.7B model handler -- load and predict."""

from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoProcessor

from stt_inference.handlers.base import load_audio
from stt_inference.types import (
    STTModelName,
    TranscriptionResult,
    TranscriptionSegment,
)

MODEL_ID = "Qwen/Qwen3-ASR-1.7B"


class Qwen3ASRHandler:
    """Handler for Qwen3-ASR-1.7B."""

    def model_fn(self, model_dir: str) -> dict[str, Any]:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.bfloat16 if device == "cuda" else torch.float32

        processor = AutoProcessor.from_pretrained(
            model_dir or MODEL_ID,
            trust_remote_code=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_dir or MODEL_ID,
            torch_dtype=torch_dtype,
            device_map=device,
            trust_remote_code=True,
        )

        return {
            "model": model,
            "processor": processor,
            "device": device,
            "torch_dtype": torch_dtype,
        }

    def predict_fn(
        self,
        audio_path: str,
        model_artifacts: dict[str, Any],
    ) -> TranscriptionResult:
        import time

        model = model_artifacts["model"]
        processor = model_artifacts["processor"]

        audio, sr = load_audio(audio_path, target_sr=16000)
        audio_duration = len(audio) / sr

        start = time.perf_counter()

        inputs = processor(
            audios=[audio],
            sampling_rate=sr,
            return_tensors="pt",
            padding=True,
        )
        inputs = inputs.to(model.device)

        generated_ids = model.generate(
            **inputs,
            max_new_tokens=256,
        )

        transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        elapsed = time.perf_counter() - start

        segments = [
            TranscriptionSegment(
                text=transcription.strip(),
                start=0.0,
                end=audio_duration,
            )
        ]

        return TranscriptionResult(
            source=audio_path,
            model_name=STTModelName.QWEN3_ASR,
            text=transcription.strip(),
            segments=segments,
            duration_seconds=audio_duration,
            elapsed_seconds=elapsed,
        )
