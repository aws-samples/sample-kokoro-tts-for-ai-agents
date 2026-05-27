"""Maya1/Veena TTS model handler -- load and synthesize."""

from io import BytesIO
from typing import Any

import soundfile as sf
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from tts_inference.types import VoiceConfig

MODEL_ID = "maya-research/veena-tts"
SNAC_MODEL_ID = "hubertsiuzdak/snac_24khz"
SAMPLE_RATE = 24000
DEFAULT_VOICE = "kavya"


class MayaHandler:
    """Handler for Maya1/Veena TTS."""

    def model_fn(self, model_dir: str) -> dict[str, Any]:
        from snac import SNAC

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_dir or MODEL_ID,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
        )

        tokenizer = AutoTokenizer.from_pretrained(
            model_dir or MODEL_ID,
            trust_remote_code=True,
        )

        snac_model = SNAC.from_pretrained(SNAC_MODEL_ID).eval().cuda()

        return {
            "model": model,
            "tokenizer": tokenizer,
            "snac_model": snac_model,
        }

    def predict_fn(
        self,
        text: str,
        model_artifacts: dict[str, Any],
        voice_config: VoiceConfig | None = None,
    ) -> tuple[bytes, int, float]:
        model = model_artifacts["model"]
        tokenizer = model_artifacts["tokenizer"]
        snac_model = model_artifacts["snac_model"]

        voice = DEFAULT_VOICE
        if voice_config and voice_config.voice_id:
            voice = voice_config.voice_id

        prompt = f"<|voice|>{voice}<|text|>{text}<|speech|>"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=2048,
                do_sample=True,
                temperature=0.7,
                top_p=0.95,
            )

        speech_tokens = generated[0][inputs["input_ids"].shape[1] :]

        with torch.no_grad():
            audio = snac_model.decode(speech_tokens.unsqueeze(0))

        audio_np = audio.squeeze().cpu().numpy()
        duration = len(audio_np) / SAMPLE_RATE

        buffer = BytesIO()
        sf.write(buffer, audio_np, SAMPLE_RATE, format="WAV")
        wav_bytes = buffer.getvalue()

        return wav_bytes, SAMPLE_RATE, duration
