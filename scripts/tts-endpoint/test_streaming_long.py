"""Test streaming TTS with longer text passages.

Sends longer text to the endpoint and measures streaming performance:
- Time to first audio chunk
- Per-segment timing
- Total generation time
- Real-time factor (RTF)
- Combined audio output (single WAV file)

Usage (run from repo root):
    uv run python scripts/tts-endpoint/test_streaming_long.py
    uv run python scripts/tts-endpoint/test_streaming_long.py --mode sync
    uv run python scripts/tts-endpoint/test_streaming_long.py --mode stream
"""

from __future__ import annotations

import argparse
import json
import struct
import time
from pathlib import Path

import boto3
from botocore.config import Config

ENDPOINT = "speech-orpheus-3b"
REGION = "us-east-1"
OUTPUT_DIR = Path("outputs/tts-endpoint/long")
SAMPLE_RATE = 24000

LONG_TEXTS = [
    {
        "id": "paragraph-narrative",
        "voice": "tara",
        "text": (
            "The old lighthouse keeper had spent forty years watching the storms roll in "
            "from the Atlantic. Each one was different, he'd say, like a fingerprint pressed "
            "against the sky. Some came with fury, rattling the windows and shaking the very "
            "foundations of the tower. Others crept in silently, wrapping the coast in a thick "
            "gray blanket that swallowed sound and light alike."
        ),
    },
    {
        "id": "paragraph-expressive",
        "voice": "tara",
        "text": (
            "Oh my goodness, you won't believe what happened today! So I was walking down "
            "the street, minding my own business, when this enormous dog comes bounding "
            "towards me. I mean, this thing was the size of a small horse! And the owner is "
            "just standing there, laughing, as I'm trying to figure out whether to run or "
            "just accept my fate. Turns out the dog just wanted to lick my face."
        ),
    },
    {
        "id": "technical-explanation",
        "voice": "leo",
        "text": (
            "The architecture uses a three-layer hierarchical quantization scheme. "
            "The coarse layer captures the fundamental frequency and prosody at a low "
            "temporal resolution. The mid layer adds harmonic detail, doubling the frame "
            "rate. And the fine layer provides the spectral texture that makes the output "
            "sound natural, running at four times the coarse rate. Together, these three "
            "layers reconstruct a twenty-four kilohertz waveform from just seven codes "
            "per frame."
        ),
    },
]


def _pcm_to_wav(pcm_bytes: bytes, sample_rate: int) -> bytes:
    """Wrap raw PCM int16 mono data in a WAV header."""
    data_size = len(pcm_bytes)
    channels = 1
    bits_per_sample = 16
    byte_rate = sample_rate * channels * (bits_per_sample // 8)
    block_align = channels * (bits_per_sample // 8)

    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + data_size,
        b"WAVE",
        b"fmt ",
        16,
        1,  # PCM format
        channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        b"data",
        data_size,
    )
    return header + pcm_bytes


def test_sync(client: object, sample: dict) -> dict:
    """Synchronous invoke: full WAV response."""
    payload = json.dumps({"text": sample["text"], "voice": sample["voice"]})

    t0 = time.perf_counter()
    response = client.invoke_endpoint(
        EndpointName=ENDPOINT,
        ContentType="application/json",
        Accept="audio/wav",
        Body=payload.encode(),
    )
    elapsed = time.perf_counter() - t0

    body = response["Body"].read()
    out_path = OUTPUT_DIR / "sync" / f"{sample['id']}.wav"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(body)

    audio_duration = _wav_duration(body)
    print(f"  [{sample['id']}] {len(sample['text'])} chars")
    print(f"    Latency: {elapsed*1000:.0f}ms | Audio: {audio_duration:.2f}s | RTF: {elapsed/audio_duration:.2f}x")
    print(f"    Saved: {out_path}")

    return {
        "id": sample["id"],
        "chars": len(sample["text"]),
        "latency_ms": round(elapsed * 1000),
        "audio_duration_s": round(audio_duration, 2),
        "rtf": round(elapsed / audio_duration, 2) if audio_duration > 0 else None,
        "audio_bytes": len(body),
    }


def test_response_stream(client: object, sample: dict) -> dict:
    """Response streaming: chunked audio via InvokeEndpointWithResponseStream."""
    payload = json.dumps({"text": sample["text"], "voice": sample["voice"], "stream": True})

    t0 = time.perf_counter()
    response = client.invoke_endpoint_with_response_stream(
        EndpointName=ENDPOINT,
        ContentType="application/json",
        Accept="audio/wav",
        Body=payload.encode(),
    )

    pcm_chunks: list[bytes] = []
    first_chunk_time: float | None = None

    for event in response["Body"]:
        if "PayloadPart" in event:
            chunk = event["PayloadPart"]["Bytes"]
            if first_chunk_time is None:
                first_chunk_time = time.perf_counter() - t0
            pcm_chunks.append(chunk)
        elif "ModelStreamError" in event:
            print(f"    Stream error: {event['ModelStreamError'].get('Message')}")
            break

    elapsed = time.perf_counter() - t0
    full_pcm = b"".join(pcm_chunks)

    # Combine into single WAV
    wav_data = _pcm_to_wav(full_pcm, SAMPLE_RATE)
    out_path = OUTPUT_DIR / "stream" / f"{sample['id']}.wav"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(wav_data)

    audio_duration = len(full_pcm) / (SAMPLE_RATE * 2)
    print(f"  [{sample['id']}] {len(sample['text'])} chars")
    print(
        f"    First chunk: {first_chunk_time*1000:.0f}ms | Total: {elapsed*1000:.0f}ms | "
        f"Audio: {audio_duration:.2f}s | Chunks: {len(pcm_chunks)} | RTF: {elapsed/audio_duration:.2f}x"
    )
    print(f"    Saved: {out_path}")

    return {
        "id": sample["id"],
        "chars": len(sample["text"]),
        "first_chunk_ms": round(first_chunk_time * 1000) if first_chunk_time else None,
        "total_ms": round(elapsed * 1000),
        "audio_duration_s": round(audio_duration, 2),
        "chunks": len(pcm_chunks),
        "rtf": round(elapsed / audio_duration, 2) if audio_duration > 0 else None,
    }


def _wav_duration(data: bytes) -> float:
    if len(data) < 44 or data[:4] != b"RIFF":
        return len(data) / (SAMPLE_RATE * 2)
    try:
        sample_rate = struct.unpack_from("<I", data, 24)[0]
        bits = struct.unpack_from("<H", data, 34)[0]
        channels = struct.unpack_from("<H", data, 22)[0]
        data_size = len(data) - 44
        return data_size / (sample_rate * channels * (bits // 8))
    except (struct.error, ZeroDivisionError):
        return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["sync", "stream", "both"], default="both")
    args = parser.parse_args()

    config = Config(region_name=REGION, read_timeout=120, retries={"max_attempts": 0})
    client = boto3.client("sagemaker-runtime", config=config)

    print(f"Endpoint: {ENDPOINT}")
    print(f"Mode: {args.mode}")
    print(f"Samples: {len(LONG_TEXTS)}")
    print()

    results = []

    if args.mode in ("sync", "both"):
        print("=== Sync Mode ===")
        for sample in LONG_TEXTS:
            try:
                r = test_sync(client, sample)
                results.append({"mode": "sync", **r})
            except Exception as e:
                print(f"  [{sample['id']}] FAILED: {e}")
        print()

    if args.mode in ("stream", "both"):
        print("=== Response Stream Mode ===")
        for sample in LONG_TEXTS:
            try:
                r = test_response_stream(client, sample)
                results.append({"mode": "stream", **r})
            except Exception as e:
                print(f"  [{sample['id']}] FAILED: {e}")
        print()

    # Summary
    print("=== Summary ===")
    for mode in ("sync", "stream"):
        mode_results = [r for r in results if r["mode"] == mode]
        if not mode_results:
            continue
        avg_rtf = sum(r["rtf"] for r in mode_results if r.get("rtf")) / len(mode_results)
        total_audio = sum(r["audio_duration_s"] for r in mode_results)
        print(f"  {mode.upper()}: avg RTF={avg_rtf:.2f}x, total audio={total_audio:.1f}s")

    out_file = OUTPUT_DIR / "long_results.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(results, indent=2))
    print(f"\nResults: {out_file}")


if __name__ == "__main__":
    main()
