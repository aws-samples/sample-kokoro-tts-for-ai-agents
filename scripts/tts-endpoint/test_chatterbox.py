"""Test Chatterbox-Turbo TTS endpoint on SageMaker.

Tests synchronous invocation with voice cloning.

Usage:
    uv run python scripts/tts-endpoint/test_chatterbox.py
    uv run python scripts/tts-endpoint/test_chatterbox.py --voice female_shadowheart4
    uv run python scripts/tts-endpoint/test_chatterbox.py --endpoint speech-chatterbox-turbo
"""

from __future__ import annotations

import argparse
import json
import struct
import time
from pathlib import Path

import boto3

ENDPOINT = "speech-chatterbox-turbo"
REGION = "us-east-1"
DEFAULT_VOICE = "female_shadowheart4"

TEST_CASES = [
    ("short", "Hello, world!"),
    ("medium", "The quick brown fox jumps over the lazy dog. This sentence tests naturalness."),
    (
        "long",
        "In the beginning, there was silence. Then came the machines that could speak, "
        "turning written words into flowing speech with remarkable clarity and naturalness. "
        "Each generation improved upon the last.",
    ),
    ("expressive", "Wait... really? Yes! That's incredible — absolutely incredible."),
    (
        "conversational",
        "So I was thinking, maybe we could try a different approach? What do you think?",
    ),
]


def wav_duration(data: bytes) -> float:
    if len(data) < 44 or data[:4] != b"RIFF":
        return 0.0
    sr = struct.unpack_from("<I", data, 24)[0]
    bits = struct.unpack_from("<H", data, 34)[0]
    channels = struct.unpack_from("<H", data, 22)[0]
    data_size = len(data) - 44
    return data_size / (sr * channels * (bits // 8))


def test_invocation(
    client: object,
    endpoint: str,
    text: str,
    voice: str,
) -> dict:
    payload = json.dumps({"text": text, "voice": voice})
    t0 = time.perf_counter()
    resp = client.invoke_endpoint(
        EndpointName=endpoint,
        ContentType="application/json",
        Accept="audio/wav",
        Body=payload.encode("utf-8"),
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000
    body = resp["Body"].read()
    duration = wav_duration(body)
    return {
        "text_chars": len(text),
        "latency_ms": round(elapsed_ms, 1),
        "audio_bytes": len(body),
        "audio_duration_s": round(duration, 2),
        "rtf": round(elapsed_ms / 1000 / duration, 3) if duration > 0 else None,
        "is_wav": body[:4] == b"RIFF",
        "body": body,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Test Chatterbox-Turbo endpoint")
    parser.add_argument("--endpoint", default=ENDPOINT)
    parser.add_argument("--region", default=REGION)
    parser.add_argument("--voice", default=DEFAULT_VOICE)
    parser.add_argument("--output-dir", default="outputs/chatterbox")
    args = parser.parse_args()

    client = boto3.client("sagemaker-runtime", region_name=args.region)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Endpoint: {args.endpoint}")
    print(f"Voice: {args.voice}")
    print(f"Test cases: {len(TEST_CASES)}")
    print()

    results: list[dict] = []

    for label, text in TEST_CASES:
        try:
            r = test_invocation(client, args.endpoint, text, args.voice)
            results.append({"label": label, **{k: v for k, v in r.items() if k != "body"}})
            print(
                f"  {label:14s} | {r['text_chars']:3d} chars | "
                f"{r['latency_ms']:7.0f}ms | {r['audio_duration_s']:.2f}s | "
                f"RTF {r['rtf']:.3f} | {'WAV' if r['is_wav'] else 'ERR'}"
            )

            wav_path = output_dir / f"{args.voice}_{label}.wav"
            wav_path.write_bytes(r["body"])
        except Exception as e:
            print(f"  {label:14s} | FAILED: {e}")
            results.append({"label": label, "error": str(e)})
    print()

    successful = [r for r in results if "error" not in r]
    if successful:
        avg_rtf = sum(r["rtf"] for r in successful) / len(successful)
        avg_latency = sum(r["latency_ms"] for r in successful) / len(successful)
        print(f"Summary: {len(successful)}/{len(results)} passed")
        print(f"  Avg latency: {avg_latency:.0f}ms")
        print(f"  Avg RTF: {avg_rtf:.3f}")
        print(f"  Output: {output_dir}/")
    else:
        print("All tests failed!")


if __name__ == "__main__":
    main()
