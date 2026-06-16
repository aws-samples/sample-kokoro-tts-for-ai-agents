"""Test Kokoro-82M TTS endpoint on SageMaker.

Tests synchronous invocation and validates WAV output quality.

Usage:
    uv run python scripts/tts-endpoint/test_kokoro.py
    uv run python scripts/tts-endpoint/test_kokoro.py --voice am_adam
    uv run python scripts/tts-endpoint/test_kokoro.py --endpoint speech-kokoro-82m --region us-east-1
"""

from __future__ import annotations

import argparse
import json
import struct
import time
from pathlib import Path

import boto3

ENDPOINT = "speech-kokoro-82m"
REGION = "us-east-1"
VOICES = ["af_heart", "af_nova", "am_adam", "af_sarah", "am_michael"]

TEST_CASES = [
    ("short", "Hello, world!"),
    ("medium", "The quick brown fox jumps over the lazy dog. This sentence tests naturalness."),
    (
        "long",
        "In the beginning, there was silence. Then came the machines that could speak, "
        "turning written words into flowing speech with remarkable clarity and naturalness. "
        "Each generation improved upon the last, bringing us closer to voices indistinguishable "
        "from human speakers.",
    ),
    ("punctuation", "Wait... really? Yes! That's incredible — absolutely incredible."),
    ("numbers", "The meeting is at 3:45 PM on January 1st, 2026. There will be 12 attendees."),
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
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Test Kokoro-82M endpoint")
    parser.add_argument("--endpoint", default=ENDPOINT)
    parser.add_argument("--region", default=REGION)
    parser.add_argument("--voice", default="af_heart", choices=VOICES)
    parser.add_argument("--output-dir", default="outputs/kokoro")
    parser.add_argument("--all-voices", action="store_true", help="Test all available voices")
    args = parser.parse_args()

    client = boto3.client("sagemaker-runtime", region_name=args.region)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    voices_to_test = VOICES if args.all_voices else [args.voice]

    print(f"Endpoint: {args.endpoint}")
    print(f"Voices: {voices_to_test}")
    print(f"Test cases: {len(TEST_CASES)}")
    print()

    results: list[dict] = []

    for voice in voices_to_test:
        print(f"--- Voice: {voice} ---")
        for label, text in TEST_CASES:
            try:
                r = test_invocation(client, args.endpoint, text, voice)
                results.append({"voice": voice, "label": label, **r})
                print(
                    f"  {label:12s} | {r['text_chars']:3d} chars | "
                    f"{r['latency_ms']:7.0f}ms | {r['audio_duration_s']:.2f}s | "
                    f"RTF {r['rtf']:.3f} | {'WAV' if r['is_wav'] else 'ERR'}"
                )

                wav_path = output_dir / f"{voice}_{label}.wav"
                payload = json.dumps({"text": text, "voice": voice})
                resp = client.invoke_endpoint(
                    EndpointName=args.endpoint,
                    ContentType="application/json",
                    Accept="audio/wav",
                    Body=payload.encode("utf-8"),
                )
                wav_path.write_bytes(resp["Body"].read())
            except Exception as e:
                print(f"  {label:12s} | FAILED: {e}")
                results.append({"voice": voice, "label": label, "error": str(e)})
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
