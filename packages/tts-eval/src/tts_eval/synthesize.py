"""SageMaker endpoint synthesis client for TTS evaluation.

Provides a unified interface to invoke any TTS model endpoint and get
back WAV audio bytes with timing information.
"""

from __future__ import annotations

import json
import struct
import threading
import time

import boto3
from loguru import logger

from tts_inference.types import TTSModelName

ENDPOINT_MAP: dict[str, str] = {
    TTSModelName.ORPHEUS_3B: "speech-orpheus-3b",
    TTSModelName.KOKORO_82M: "speech-kokoro-82m",
    TTSModelName.KOKORO_82M_CPU: "speech-kokoro-82m-cpu",
    TTSModelName.CHATTERBOX_TURBO: "speech-chatterbox-turbo",
}

DEFAULT_VOICES: dict[str, str] = {
    TTSModelName.ORPHEUS_3B: "tara",
    TTSModelName.KOKORO_82M: "af_heart",
    TTSModelName.KOKORO_82M_CPU: "af_heart",
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

    _thread_local = threading.local()

    def __init__(self, region: str = "us-east-1") -> None:
        self._client = boto3.client("sagemaker-runtime", region_name=region)
        self._region = region

    def _get_thread_client(self):
        """Get a thread-local boto3 client for concurrent use."""
        if not hasattr(self._thread_local, "client"):
            self._thread_local.client = boto3.client("sagemaker-runtime", region_name=self._region)
        return self._thread_local.client

    def synthesize(
        self,
        model: str | TTSModelName,
        text: str,
        voice: str | None = None,
    ) -> dict:
        """Synthesize text to audio via SageMaker endpoint (synchronous HTTP).

        Returns:
            Dict with keys: audio_bytes, duration_s, latency_ms, chars,
            sample_rate, voice, model.
        """
        model = TTSModelName(model)
        endpoint = ENDPOINT_MAP[model]
        voice = voice or DEFAULT_VOICES[model]

        payload = json.dumps(
            {
                "text": text,
                "voice": voice,
                "request_timestamp": time.time(),
            }
        )

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

    def synthesize_stream(
        self,
        model: str | TTSModelName,
        text: str,
        voice: str | None = None,
    ) -> dict:
        """Streaming synthesis via invoke_endpoint_with_response_stream.

        No fixed timeout — stream stays open until server finishes.
        Thread-safe: uses thread-local boto3 clients for concurrent usage.

        Returns:
            Dict with keys: audio_bytes, duration_s, ttfab_ms, latency_ms,
            chars, sample_rate, voice, model.
        """
        model = TTSModelName(model)
        endpoint = ENDPOINT_MAP[model]
        voice = voice or DEFAULT_VOICES[model]

        payload = json.dumps(
            {
                "text": text,
                "voice": voice,
                "request_timestamp": time.time(),
            }
        )

        client = self._get_thread_client()
        t0 = time.perf_counter()
        resp = client.invoke_endpoint_with_response_stream(
            EndpointName=endpoint,
            ContentType="application/json",
            Accept="audio/wav",
            Body=payload.encode("utf-8"),
        )

        chunks: list[bytes] = []
        ttfab_ms: float | None = None
        for event in resp["Body"]:
            if "PayloadPart" in event:
                chunk = event["PayloadPart"]["Bytes"]
                if ttfab_ms is None:
                    ttfab_ms = (time.perf_counter() - t0) * 1000
                chunks.append(chunk)

        latency_ms = (time.perf_counter() - t0) * 1000
        audio_bytes = b"".join(chunks)
        duration = wav_duration(audio_bytes)

        if not audio_bytes or audio_bytes[:4] != b"RIFF":
            logger.warning("Invalid WAV response from {} ({}B)", endpoint, len(audio_bytes))

        return {
            "audio_bytes": audio_bytes,
            "duration_s": duration,
            "ttfab_ms": ttfab_ms or latency_ms,
            "latency_ms": latency_ms,
            "chars": len(text),
            "sample_rate": struct.unpack_from("<I", audio_bytes, 24)[0]
            if len(audio_bytes) >= 28
            else 24000,
            "voice": voice,
            "model": model,
        }
