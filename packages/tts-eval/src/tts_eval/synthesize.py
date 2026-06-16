"""SageMaker endpoint synthesis client for TTS evaluation.

Provides a unified interface to invoke any TTS model endpoint and get
back WAV audio bytes with timing information.
"""

from __future__ import annotations

import json
import struct
import time

import boto3
from loguru import logger

from tts_inference.types import TTSModelName

ENDPOINT_MAP: dict[str, str] = {
    TTSModelName.ORPHEUS_3B: "speech-orpheus-3b",
    TTSModelName.KOKORO_82M: "speech-kokoro-82m",
    TTSModelName.CHATTERBOX_TURBO: "speech-chatterbox-turbo",
}

DEFAULT_VOICES: dict[str, str] = {
    TTSModelName.ORPHEUS_3B: "tara",
    TTSModelName.KOKORO_82M: "af_heart",
    TTSModelName.CHATTERBOX_TURBO: "female_shadowheart4",
}


def wav_duration(data: bytes) -> float:
    """Calculate WAV audio duration from RIFF header."""
    if len(data) < 44 or data[:4] != b"RIFF":
        return 0.0
    sr: int = struct.unpack_from("<I", data, 24)[0]
    bits: int = struct.unpack_from("<H", data, 34)[0]
    channels: int = struct.unpack_from("<H", data, 22)[0]
    data_size = len(data) - 44
    return float(data_size / (sr * channels * (bits // 8)))


class SynthesisClient:
    """Client for invoking TTS SageMaker endpoints."""

    def __init__(self, region: str = "us-east-1") -> None:
        self._client = boto3.client("sagemaker-runtime", region_name=region)
        self._region = region

    def synthesize(
        self,
        model: str | TTSModelName,
        text: str,
        voice: str | None = None,
    ) -> dict:
        """Synthesize text to audio via SageMaker endpoint.

        Returns:
            Dict with keys: audio_bytes, duration_s, latency_ms, chars,
            sample_rate, voice, model.
        """
        model = TTSModelName(model)
        endpoint = ENDPOINT_MAP[model]
        voice = voice or DEFAULT_VOICES[model]

        payload = json.dumps({"text": text, "voice": voice})

        t0 = time.perf_counter()
        resp = self._client.invoke_endpoint(
            EndpointName=endpoint,
            ContentType="application/json",
            Accept="audio/wav",
            Body=payload.encode("utf-8"),
        )
        latency_ms = (time.perf_counter() - t0) * 1000

        audio_bytes = resp["Body"].read()
        duration = wav_duration(audio_bytes)

        if not audio_bytes or audio_bytes[:4] != b"RIFF":
            logger.warning("Invalid WAV response from {} ({}B)", endpoint, len(audio_bytes))

        return {
            "audio_bytes": audio_bytes,
            "duration_s": duration,
            "latency_ms": latency_ms,
            "chars": len(text),
            "sample_rate": struct.unpack_from("<I", audio_bytes, 24)[0]
            if len(audio_bytes) >= 28
            else 24000,
            "voice": voice,
            "model": model,
        }
