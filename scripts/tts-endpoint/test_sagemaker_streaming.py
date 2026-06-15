"""Test SageMaker bidirectional streaming TTS endpoint.

Tests the deployed Orpheus-3B endpoint in three modes:
1. Synchronous invoke (POST /invocations -> full WAV)
2. Response streaming (InvokeEndpointWithResponseStream -> chunked audio)
3. Local proxy WebSocket (direct WS to streaming proxy for dev/debug)

Usage (run from repo root):
    uv run python scripts/tts-endpoint/test_sagemaker_streaming.py
    uv run python scripts/tts-endpoint/test_sagemaker_streaming.py --endpoint speech-orpheus-3b
    uv run python scripts/tts-endpoint/test_sagemaker_streaming.py --mode stream
    uv run python scripts/tts-endpoint/test_sagemaker_streaming.py --mode local --proxy-url http://localhost:8080
    uv run python scripts/tts-endpoint/test_sagemaker_streaming.py --samples 3 --category short
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
import time
from io import BytesIO
from pathlib import Path

import boto3
from botocore.config import Config

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from packages.shared.src.shared.loader import load_tts_samples
from packages.shared.src.shared.types import TTSSample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test SageMaker streaming TTS")
    parser.add_argument(
        "--endpoint",
        default="speech-orpheus-3b",
        help="SageMaker endpoint name",
    )
    parser.add_argument("--region", default="us-east-1", help="AWS region")
    parser.add_argument(
        "--mode",
        choices=["sync", "stream", "local", "all"],
        default="all",
        help="Test mode: sync, stream, local (WebSocket), or all",
    )
    parser.add_argument(
        "--proxy-url",
        default="http://localhost:8080",
        help="Local proxy URL for 'local' mode",
    )
    parser.add_argument("--samples", type=int, default=3, help="Number of samples to test")
    parser.add_argument("--category", default=None, help="Filter samples by category")
    parser.add_argument("--voice", default="tara", help="Voice ID for synthesis")
    parser.add_argument(
        "--output-dir",
        default="outputs/tts-endpoint",
        help="Directory to save audio outputs",
    )
    return parser.parse_args()


def get_samples(num_samples: int, category: str | None) -> list[TTSSample]:
    dataset = load_tts_samples()
    samples = dataset.samples
    if category:
        samples = [s for s in samples if s.category == category]
    return samples[:num_samples]


def test_sync_invoke(
    client: object,
    endpoint_name: str,
    text: str,
    voice: str,
    sample_id: str,
    output_dir: Path,
) -> dict:
    """Test synchronous invocation (POST /invocations -> full WAV)."""
    payload = json.dumps({"text": text, "voice": voice})

    t0 = time.perf_counter()
    response = client.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/json",
        Accept="audio/wav",
        Body=payload.encode(),
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000

    body = response["Body"].read()
    content_type = response.get("ContentType", "unknown")

    out_path = output_dir / "sync" / f"{sample_id}.wav"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(body)

    audio_duration = _wav_duration(body)

    return {
        "mode": "sync",
        "sample_id": sample_id,
        "text_chars": len(text),
        "latency_ms": round(elapsed_ms, 1),
        "audio_bytes": len(body),
        "audio_duration_s": round(audio_duration, 2),
        "content_type": content_type,
        "rtf": round(elapsed_ms / 1000 / audio_duration, 3) if audio_duration > 0 else None,
    }


def test_response_stream(
    client: object,
    endpoint_name: str,
    text: str,
    voice: str,
    sample_id: str,
    output_dir: Path,
) -> dict:
    """Test response streaming (InvokeEndpointWithResponseStream -> chunked audio)."""
    payload = json.dumps({"text": text, "voice": voice, "stream": True})

    t0 = time.perf_counter()
    response = client.invoke_endpoint_with_response_stream(
        EndpointName=endpoint_name,
        ContentType="application/json",
        Accept="audio/wav",
        Body=payload.encode(),
    )

    chunks: list[bytes] = []
    first_chunk_ms: float | None = None
    chunk_count = 0

    event_stream = response["Body"]
    for event in event_stream:
        if "PayloadPart" in event:
            chunk = event["PayloadPart"]["Bytes"]
            if first_chunk_ms is None:
                first_chunk_ms = (time.perf_counter() - t0) * 1000
            chunks.append(chunk)
            chunk_count += 1
        elif "ModelStreamError" in event:
            error = event["ModelStreamError"]
            print(f"  Stream error: {error.get('Message', 'unknown')}")
            break
        elif "InternalStreamFailure" in event:
            error = event["InternalStreamFailure"]
            print(f"  Internal failure: {error.get('Message', 'unknown')}")
            break

    total_elapsed_ms = (time.perf_counter() - t0) * 1000
    full_audio = b"".join(chunks)

    out_path = output_dir / "stream" / f"{sample_id}.wav"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(full_audio)

    audio_duration = _wav_duration(full_audio) if full_audio else 0

    return {
        "mode": "stream",
        "sample_id": sample_id,
        "text_chars": len(text),
        "first_chunk_ms": round(first_chunk_ms, 1) if first_chunk_ms else None,
        "total_latency_ms": round(total_elapsed_ms, 1),
        "chunk_count": chunk_count,
        "audio_bytes": len(full_audio),
        "audio_duration_s": round(audio_duration, 2),
        "rtf": round(total_elapsed_ms / 1000 / audio_duration, 3) if audio_duration > 0 else None,
    }


def test_local_websocket(
    proxy_url: str,
    text: str,
    voice: str,
    sample_id: str,
    output_dir: Path,
) -> dict:
    """Test WebSocket bidirectional streaming against local proxy."""
    import websockets.sync.client as ws_client

    ws_url = proxy_url.replace("http://", "ws://").replace("https://", "wss://")
    ws_url = f"{ws_url}/invocations-bidirectional-stream"

    request_id = f"test-{sample_id}"
    msg = json.dumps({"text": text, "voice": voice, "request_id": request_id})

    audio_chunks: list[bytes] = []
    first_audio_ms: float | None = None
    metadata: dict = {}

    t0 = time.perf_counter()
    with ws_client.connect(ws_url) as ws:
        ws.send(msg)

        while True:
            response = ws.recv()
            if isinstance(response, bytes):
                if first_audio_ms is None:
                    first_audio_ms = (time.perf_counter() - t0) * 1000
                audio_chunks.append(response)
            else:
                data = json.loads(response)
                msg_type = data.get("type")
                if msg_type == "synthesis_complete":
                    metadata = data
                    break
                elif msg_type == "error":
                    print(f"  WS error: {data.get('message')}")
                    break

        ws.send(json.dumps({"type": "close"}))

    total_elapsed_ms = (time.perf_counter() - t0) * 1000
    full_audio = b"".join(audio_chunks)

    out_path = output_dir / "websocket" / f"{sample_id}.raw"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(full_audio)

    audio_duration = len(full_audio) / (24000 * 2) if full_audio else 0

    return {
        "mode": "websocket",
        "sample_id": sample_id,
        "text_chars": len(text),
        "first_audio_ms": round(first_audio_ms, 1) if first_audio_ms else None,
        "total_latency_ms": round(total_elapsed_ms, 1),
        "chunk_count": len(audio_chunks),
        "audio_bytes": len(full_audio),
        "audio_duration_s": round(audio_duration, 2),
        "server_duration_s": metadata.get("duration_seconds"),
        "server_elapsed_s": metadata.get("elapsed_seconds"),
        "rtf": round(total_elapsed_ms / 1000 / audio_duration, 3) if audio_duration > 0 else None,
    }


def _wav_duration(data: bytes) -> float:
    """Extract duration from WAV header, or estimate from raw size."""
    if len(data) < 44:
        return 0
    if data[:4] == b"RIFF":
        try:
            sample_rate = struct.unpack_from("<I", data, 24)[0]
            data_size = len(data) - 44
            bits_per_sample = struct.unpack_from("<H", data, 34)[0]
            channels = struct.unpack_from("<H", data, 22)[0]
            bytes_per_sample = bits_per_sample // 8
            return data_size / (sample_rate * channels * bytes_per_sample)
        except (struct.error, ZeroDivisionError):
            pass
    return len(data) / (24000 * 2)


def print_results(results: list[dict]) -> None:
    """Print a formatted summary table."""
    if not results:
        print("No results to display.")
        return

    print("\n" + "=" * 90)
    print(f"{'Mode':<10} {'Sample':<15} {'Chars':<6} {'1st Chunk':<10} "
          f"{'Total ms':<10} {'Chunks':<8} {'Audio s':<8} {'RTF':<6}")
    print("-" * 90)

    for r in results:
        first_chunk = r.get("first_chunk_ms") or r.get("first_audio_ms") or "-"
        if isinstance(first_chunk, float):
            first_chunk = f"{first_chunk:.0f}"
        total = r.get("total_latency_ms") or r.get("latency_ms", "-")
        if isinstance(total, float):
            total = f"{total:.0f}"
        chunks = r.get("chunk_count", 1)
        audio_s = r.get("audio_duration_s", "-")
        rtf = r.get("rtf", "-")
        if isinstance(rtf, float):
            rtf = f"{rtf:.2f}"

        print(
            f"{r['mode']:<10} {r['sample_id']:<15} {r['text_chars']:<6} "
            f"{str(first_chunk):<10} {str(total):<10} {str(chunks):<8} "
            f"{str(audio_s):<8} {str(rtf):<6}"
        )

    print("=" * 90)

    for mode in ("sync", "stream", "websocket"):
        mode_results = [r for r in results if r["mode"] == mode]
        if not mode_results:
            continue
        latencies = [
            r.get("total_latency_ms") or r.get("latency_ms", 0) for r in mode_results
        ]
        first_chunks = [
            r.get("first_chunk_ms") or r.get("first_audio_ms") for r in mode_results
        ]
        first_chunks = [f for f in first_chunks if f is not None]
        avg_latency = sum(latencies) / len(latencies) if latencies else 0
        avg_first = sum(first_chunks) / len(first_chunks) if first_chunks else None
        print(
            f"\n  {mode.upper()} - avg total: {avg_latency:.0f}ms"
            + (f", avg first chunk: {avg_first:.0f}ms" if avg_first else "")
            + f" ({len(mode_results)} samples)"
        )

    print()


def main() -> None:
    args = parse_args()
    samples = get_samples(args.samples, args.category)

    if not samples:
        print("No samples found. Check data/tts_samples.json.")
        sys.exit(1)

    print(f"Testing endpoint: {args.endpoint}")
    print(f"Region: {args.region}")
    print(f"Voice: {args.voice}")
    print(f"Samples: {len(samples)}")
    print(f"Mode: {args.mode}")
    print()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []

    modes_to_test = []
    if args.mode in ("sync", "all"):
        modes_to_test.append("sync")
    if args.mode in ("stream", "all"):
        modes_to_test.append("stream")
    if args.mode in ("local", "all"):
        modes_to_test.append("local")

    client = None
    if "sync" in modes_to_test or "stream" in modes_to_test:
        config = Config(
            region_name=args.region,
            read_timeout=120,
            retries={"max_attempts": 0},
        )
        client = boto3.client("sagemaker-runtime", config=config)

    for sample in samples:
        sample_id = sample.id
        text = sample.text
        print(f"  [{sample_id}] ({len(text)} chars) {text[:60]}...")

        if "sync" in modes_to_test and client:
            try:
                result = test_sync_invoke(
                    client, args.endpoint, text, args.voice, sample_id, output_dir
                )
                results.append(result)
                print(f"    sync: {result['latency_ms']:.0f}ms, "
                      f"{result['audio_duration_s']}s audio")
            except Exception as e:
                print(f"    sync FAILED: {e}")

        if "stream" in modes_to_test and client:
            try:
                result = test_response_stream(
                    client, args.endpoint, text, args.voice, sample_id, output_dir
                )
                results.append(result)
                fc = result.get("first_chunk_ms", "?")
                print(f"    stream: first_chunk={fc}ms, total={result['total_latency_ms']:.0f}ms, "
                      f"{result['chunk_count']} chunks, {result['audio_duration_s']}s audio")
            except Exception as e:
                print(f"    stream FAILED: {e}")

        if "local" in modes_to_test:
            try:
                result = test_local_websocket(
                    args.proxy_url, text, args.voice, sample_id, output_dir
                )
                results.append(result)
                fa = result.get("first_audio_ms", "?")
                print(f"    websocket: first_audio={fa}ms, total={result['total_latency_ms']:.0f}ms, "
                      f"{result['chunk_count']} chunks, {result['audio_duration_s']}s audio")
            except Exception as e:
                print(f"    websocket FAILED: {e}")

    print_results(results)

    results_path = output_dir / "results.json"
    results_path.write_text(json.dumps(results, indent=2))
    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
