"""Whisper Large V3 model handler -- load and predict."""

from typing import Any

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

from stt_inference.handlers.base import load_audio
from stt_inference.types import (
    STTModelName,
    TranscriptionResult,
    TranscriptionSegment,
    WordSegment,
)

MODEL_ID = "openai/whisper-large-v3"


class WhisperHandler:
    """Handler for OpenAI Whisper Large V3."""

    def model_fn(self, model_dir: str) -> dict[str, Any]:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if device == "cuda" else torch.float32

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_dir or MODEL_ID,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )
        model.to(device)

        processor = AutoProcessor.from_pretrained(model_dir or MODEL_ID)

        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=torch_dtype,
            device=device,
        )

        return {"pipeline": pipe, "device": device}

    def predict_fn(
        self,
        audio_path: str,
        model_artifacts: dict[str, Any],
    ) -> TranscriptionResult:
        import time

        pipe = model_artifacts["pipeline"]

        audio, sr = load_audio(audio_path, target_sr=16000)
        audio_duration = len(audio) / sr

        start = time.perf_counter()
        result = pipe(
            {"array": audio, "sampling_rate": sr},
            return_timestamps="word",
            generate_kwargs={"language": None, "task": "transcribe"},
        )
        elapsed = time.perf_counter() - start

        segments: list[TranscriptionSegment] = []
        if "chunks" in result:
            current_words: list[WordSegment] = []
            for chunk in result["chunks"]:
                ts = chunk.get("timestamp", (0.0, 0.0))
                current_words.append(
                    WordSegment(
                        word=chunk["text"].strip(),
                        start=ts[0] if ts[0] is not None else 0.0,
                        end=ts[1] if ts[1] is not None else 0.0,
                    )
                )

            if current_words:
                segments.append(
                    TranscriptionSegment(
                        text=result["text"].strip(),
                        start=current_words[0].start,
                        end=current_words[-1].end,
                        words=current_words,
                    )
                )
        else:
            segments.append(
                TranscriptionSegment(
                    text=result["text"].strip(),
                    start=0.0,
                    end=audio_duration,
                )
            )

        return TranscriptionResult(
            source=audio_path,
            model_name=STTModelName.WHISPER_LARGE_V3,
            text=result["text"].strip(),
            segments=segments,
            duration_seconds=audio_duration,
            elapsed_seconds=elapsed,
        )
