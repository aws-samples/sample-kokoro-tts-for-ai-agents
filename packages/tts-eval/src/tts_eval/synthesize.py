"""SageMaker endpoint synthesis client for TTS evaluation.

Provides a unified interface to invoke any TTS model endpoint and get
back WAV or MP3 audio bytes with timing information.
"""

from __future__ import annotations

import base64
import io
import json
import struct
import threading
import time

import boto3
import librosa
from loguru import logger

from tts_inference.types import TTSModelName

ENDPOINT_MAP: dict[str, str] = {
    TTSModelName.ORPHEUS_3B: "speech-orpheus-3b",
    TTSModelName.KOKORO_82M: "speech-kokoro-82m",
    TTSModelName.KOKORO_82M_CPU: "speech-kokoro-82m-cpu",
    TTSModelName.CHATTERBOX_TURBO: "speech-chatterbox-turbo",
}

POLLY_VOICES: dict[str, dict[str, str]] = {
    TTSModelName.POLLY_STANDARD: {"engine": "standard", "voice_id": "Salli"},
    TTSModelName.POLLY_NEURAL: {"engine": "neural", "voice_id": "Joanna"},
    TTSModelName.POLLY_GENERATIVE: {"engine": "generative", "voice_id": "Ruth"},
}

DEFAULT_VOICES: dict[str, str] = {
    TTSModelName.ORPHEUS_3B: "tara",
    TTSModelName.KOKORO_82M: "af_heart",
    TTSModelName.KOKORO_82M_CPU: "af_heart",
    TTSModelName.CHATTERBOX_TURBO: "ENG_US_F_KimW",
    TTSModelName.POLLY_STANDARD: "Salli",
    TTSModelName.POLLY_NEURAL: "Joanna",
    TTSModelName.POLLY_GENERATIVE: "Ruth",
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


def _parse_sse_frame(frame: bytes) -> tuple[str | None, dict]:
    """Parse one SSE frame into (event_name, payload).

    Returns (None, {}) for keep-alive comments and blank frames.
    """
    name: str | None = None
    data: dict = {}
    for line in frame.decode("utf-8").splitlines():
        if line.startswith("event: "):
            name = line[len("event: ") :]
        elif line.startswith("data: "):
            data = json.loads(line[len("data: ") :])
    return name, data


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
        if model in POLLY_VOICES:
            return self._synthesize_polly(model, text)
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
        if model in POLLY_VOICES:
            return self._synthesize_polly(model, text)
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

    def synthesize_sse(
        self,
        model: str | TTSModelName,
        text: str,
        voice: str | None = None,
    ) -> dict:
        """Streaming synthesis over SSE with base64 MP3 chunks.

        This is the transport the AgentCore relay uses, so benchmarking it needs
        its own method: `synthesize_stream` requests raw chunked WAV and asserts
        RIFF, which the eval baseline depends on.

        Returns:
            Dict with keys: audio_bytes (decoded MP3), audio_format, duration_s,
            ttfab_ms, latency_ms, chars, sample_rate, voice, model, total_chunks.
        """
        model = TTSModelName(model)
        endpoint = ENDPOINT_MAP[model]
        voice = voice or DEFAULT_VOICES[model]

        payload = json.dumps(
            {
                "text": text,
                "voice": voice,
                "transport": "sse",
                "format": "mp3",
                "request_timestamp": time.time(),
            }
        )

        client = self._get_thread_client()
        t0 = time.perf_counter()
        resp = client.invoke_endpoint_with_response_stream(
            EndpointName=endpoint,
            ContentType="application/json",
            Body=payload.encode("utf-8"),
        )

        chunks: list[bytes] = []
        ttfab_ms: float | None = None
        sample_rate = 24000
        duration_s = 0.0
        buffer = b""

        for event in resp["Body"]:
            if "PayloadPart" not in event:
                continue
            buffer += event["PayloadPart"]["Bytes"]
            # Frames straddle PayloadPart boundaries: a measured 4-frame response
            # arrived as 6 parts with 2 ending mid-frame. Split on the blank line
            # and keep the remainder rather than parsing parts individually.
            while b"\n\n" in buffer:
                frame, buffer = buffer.split(b"\n\n", 1)
                name, data = _parse_sse_frame(frame)
                if name is None:
                    continue
                if name == "audio_chunk":
                    if ttfab_ms is None:
                        # Stamped here, not on audio_stream_start: that frame is
                        # emitted before inference begins and would report ~0ms.
                        ttfab_ms = (time.perf_counter() - t0) * 1000
                    chunks.append(base64.b64decode(data["data"]))
                elif name == "audio_stream_start":
                    sample_rate = data.get("sample_rate", sample_rate)
                elif name == "audio_stream_end":
                    duration_s = data.get("duration_s", 0.0)
                elif name == "error":
                    raise RuntimeError(
                        f"{endpoint} SSE synthesis failed: {data.get('message', 'unknown')}"
                    )

        latency_ms = (time.perf_counter() - t0) * 1000
        audio_bytes = b"".join(chunks)

        if not audio_bytes:
            logger.warning("No audio chunks in SSE response from {}", endpoint)

        return {
            "audio_bytes": audio_bytes,
            "audio_format": "mp3",
            "duration_s": duration_s,
            "ttfab_ms": ttfab_ms or latency_ms,
            "latency_ms": latency_ms,
            "chars": len(text),
            "sample_rate": sample_rate,
            "voice": voice,
            "model": model,
            "total_chunks": len(chunks),
        }

    def _get_polly_client(self):
        """Get a thread-local boto3 Polly client."""
        if not hasattr(self._thread_local, "polly_client"):
            self._thread_local.polly_client = boto3.client("polly", region_name=self._region)
        return self._thread_local.polly_client

    def _synthesize_polly(self, model: TTSModelName, text: str) -> dict:
        """Synthesize via Amazon Polly API. Returns MP3 at 24kHz."""
        config = POLLY_VOICES[model]
        voice_id = config["voice_id"]
        engine = config["engine"]
        sample_rate = 24000

        polly = self._get_polly_client()
        t0 = time.perf_counter()
        response = polly.synthesize_speech(
            Text=text,
            Engine=engine,
            VoiceId=voice_id,
            OutputFormat="mp3",
            SampleRate=str(sample_rate),
        )
        stream = response["AudioStream"]
        first_chunk = stream.read(1024)
        ttfab_ms = (time.perf_counter() - t0) * 1000
        rest = stream.read()
        audio_bytes = first_chunk + rest
        latency_ms = (time.perf_counter() - t0) * 1000

        y, sr_actual = librosa.load(io.BytesIO(audio_bytes), sr=None)
        duration = len(y) / sr_actual

        return {
            "audio_bytes": audio_bytes,
            "audio_format": "mp3",
            "duration_s": duration,
            "ttfab_ms": ttfab_ms,
            "latency_ms": latency_ms,
            "chars": len(text),
            "sample_rate": sample_rate,
            "voice": voice_id,
            "model": model,
        }


def synthesize_bidirectional(
    model: str | TTSModelName,
    text: str,
    voice: str | None = None,
    region: str = "us-east-1",
) -> dict:
    """Bidirectional streaming synthesis via SageMaker HTTP/2.

    Convenience re-export from bidi_client module.
    Returns same dict shape as SynthesisClient.synthesize_stream().
    """
    from tts_eval.bidi_client import synthesize_bidirectional as _bidi

    return _bidi(model, text, voice, region)


def _pcm_to_wav(pcm_bytes: bytes, sample_rate: int) -> bytes:
    """Wrap raw 16-bit mono PCM in a WAV header."""
    data_size = len(pcm_bytes)
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + data_size,
        b"WAVE",
        b"fmt ",
        16,
        1,
        1,
        sample_rate,
        sample_rate * 2,
        2,
        16,
        b"data",
        data_size,
    )
    return header + pcm_bytes
