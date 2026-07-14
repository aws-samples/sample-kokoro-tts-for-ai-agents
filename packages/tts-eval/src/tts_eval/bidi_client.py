"""Bidirectional streaming TTS client via SageMaker HTTP/2.

Uses the experimental aws-sdk-sagemaker-runtime-http2 SDK to establish
a persistent bidirectional connection to a SageMaker endpoint. Audio
chunks arrive incrementally as the model generates them.

This is an additive parallel path alongside the existing boto3-based
synthesize_stream() — it does not replace it.
"""

from __future__ import annotations

import asyncio
import json
import struct
import time
import uuid

from loguru import logger

from tts_eval.synthesize import DEFAULT_VOICES, ENDPOINT_MAP, POLLY_VOICES
from tts_inference.types import TTSModelName

SAMPLE_RATE = 24000


def _pcm_to_wav(pcm_bytes: bytes, sample_rate: int = SAMPLE_RATE) -> bytes:
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


class BidirectionalTTSClient:
    """Async client for SageMaker bidirectional streaming TTS.

    Establishes an HTTP/2 connection to the SageMaker runtime endpoint
    on port 8443, sends text via the input stream, and receives audio
    chunks via the output stream.
    """

    def __init__(self, region: str = "us-east-1") -> None:
        self._region = region
        self._client = None

    def _get_client(self):
        if self._client is None:
            from aws_sdk_sagemaker_runtime_http2.client import SageMakerRuntimeHTTP2Client
            from aws_sdk_sagemaker_runtime_http2.config import Config, HTTPAuthSchemeResolver
            from smithy_aws_core.auth.sigv4 import SigV4AuthScheme
            from smithy_aws_core.identity import EnvironmentCredentialsResolver

            config = Config(
                endpoint_uri=f"https://runtime.sagemaker.{self._region}.amazonaws.com:8443",
                region=self._region,
                aws_credentials_identity_resolver=EnvironmentCredentialsResolver(),
                auth_scheme_resolver=HTTPAuthSchemeResolver(),
                auth_schemes={"aws.auth#sigv4": SigV4AuthScheme(service="sagemaker")},
            )
            self._client = SageMakerRuntimeHTTP2Client(config=config)
        return self._client

    async def synthesize(
        self,
        model: str | TTSModelName,
        text: str,
        voice: str | None = None,
    ) -> dict:
        """Send text and receive streaming audio via bidirectional connection.

        Returns same dict shape as SynthesisClient.synthesize_stream():
            audio_bytes, duration_s, ttfab_ms, latency_ms, chars,
            sample_rate, voice, model
        """
        from aws_sdk_sagemaker_runtime_http2.models import (
            InvokeEndpointWithBidirectionalStreamInput,
            RequestPayloadPart,
            RequestStreamEventPayloadPart,
        )

        model = TTSModelName(model)
        if model in POLLY_VOICES:
            raise ValueError(f"Polly models don't support bidirectional streaming: {model}")

        endpoint = ENDPOINT_MAP[model]
        voice = voice or DEFAULT_VOICES[model]
        request_id = f"bidi-{uuid.uuid4().hex[:8]}"

        client = self._get_client()

        t0 = time.perf_counter()

        stream = await client.invoke_endpoint_with_bidirectional_stream(
            InvokeEndpointWithBidirectionalStreamInput(endpoint_name=endpoint)
        )

        message = json.dumps(
            {
                "text": text,
                "voice": voice,
                "request_id": request_id,
            }
        )
        payload = RequestPayloadPart(bytes_=message.encode("utf-8"))
        event = RequestStreamEventPayloadPart(value=payload)
        await stream.input_stream.send(event)
        await stream.input_stream.close()

        audio_chunks: list[bytes] = []
        ttfab_ms: float | None = None

        output = await stream.await_output()
        output_stream = output[1]

        while True:
            result = await output_stream.receive()
            if result is None:
                break

            if result.value and result.value.bytes_:
                chunk = result.value.bytes_
                if chunk[0:1] == b"{":
                    data = json.loads(chunk.decode("utf-8"))
                    msg_type = data.get("type")
                    if msg_type == "error":
                        logger.warning(
                            "Bidirectional stream error from {}: {}",
                            model.value,
                            data.get("message"),
                        )
                        break
                else:
                    if ttfab_ms is None:
                        ttfab_ms = (time.perf_counter() - t0) * 1000
                    audio_chunks.append(chunk)

        latency_ms = (time.perf_counter() - t0) * 1000
        pcm_bytes = b"".join(audio_chunks)
        audio_bytes = _pcm_to_wav(pcm_bytes) if pcm_bytes else b""
        duration = len(pcm_bytes) / (SAMPLE_RATE * 2) if pcm_bytes else 0.0

        return {
            "audio_bytes": audio_bytes,
            "duration_s": duration,
            "ttfab_ms": ttfab_ms or latency_ms,
            "latency_ms": latency_ms,
            "chars": len(text),
            "sample_rate": SAMPLE_RATE,
            "voice": voice,
            "model": model,
        }


def synthesize_bidirectional(
    model: str | TTSModelName,
    text: str,
    voice: str | None = None,
    region: str = "us-east-1",
) -> dict:
    """Synchronous wrapper for bidirectional streaming synthesis.

    Runs the async client in a new event loop. Thread-safe.
    Returns same dict shape as SynthesisClient.synthesize_stream().
    """
    client = BidirectionalTTSClient(region=region)
    return asyncio.run(client.synthesize(model, text, voice))
