# Copyright (c) 2026 Amazon.com
# This file is licensed under the MIT License.
# See the LICENSE file in the project root for full license information.

"""Compare time-to-first-audio between bidi and batch TTS delivery modes.

Runs the same prompt through the agent server twice, once per ``mode``, and
measures from each mode's own event stream:

- time to the first ``text_delta`` event (the agent's own time-to-first-token)
- time to the first ``audio_chunk`` event (TTFA -- the headline number)
- time to ``agent_end`` (total)

Each event arrives as one ``data: <json>`` line (``BedrockAgentCoreApp``'s
real wire format -- confirmed by reading its source, not the ``event:``/
``data:`` pair the deleted SageMaker-layer SSE relay used). Both ``httpx``
and boto3's streaming response reconstruct lines regardless of how the
underlying transport chunks bytes, so plain line iteration is enough --
no manual buffer-and-split-on-blank-line parser is needed here.

Targets a local container (``http://...``) or a deployed AgentCore Runtime
(an ``arn:aws:bedrock-agentcore:...`` ARN, invoked via boto3's
``invoke_agent_runtime`` -- there is no bare HTTPS URL for a deployed
Runtime).
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass

import httpx

MODES = ("bidi", "batch")


@dataclass
class ModeTiming:
    mode: str
    ttft_ms: float | None = None
    ttfa_ms: float | None = None
    total_ms: float | None = None
    chunk_count: int = 0
    error: str | None = None


def _record_event(timing: ModeTiming, event: dict, now_ms: float) -> None:
    event_type = event.get("type")
    if event_type == "text_delta" and timing.ttft_ms is None:
        timing.ttft_ms = now_ms
    elif event_type == "audio_chunk":
        timing.chunk_count += 1
        if timing.ttfa_ms is None:
            timing.ttfa_ms = now_ms
    elif event_type == "agent_end":
        timing.total_ms = now_ms
    elif event_type == "error":
        timing.error = event.get("message")


def _run_mode_local(base_url: str, prompt: str, voice: str, mode: str) -> ModeTiming:
    timing = ModeTiming(mode=mode)
    t0 = time.perf_counter()
    with httpx.stream(
        "POST",
        f"{base_url}/invocations",
        json={"prompt": prompt, "mode": mode, "voice": voice},
        timeout=120.0,
    ) as response:
        for line in response.iter_lines():
            if not line.startswith("data: "):
                continue
            event = json.loads(line[len("data: ") :])
            _record_event(timing, event, (time.perf_counter() - t0) * 1000.0)
    return timing


def _run_mode_deployed(agent_runtime_arn: str, prompt: str, voice: str, mode: str) -> ModeTiming:
    import boto3

    timing = ModeTiming(mode=mode)
    client = boto3.client("bedrock-agentcore")
    payload = json.dumps({"prompt": prompt, "mode": mode, "voice": voice}).encode()

    t0 = time.perf_counter()
    response = client.invoke_agent_runtime(
        agentRuntimeArn=agent_runtime_arn,
        runtimeSessionId=str(uuid.uuid4()),
        payload=payload,
    )
    if "text/event-stream" not in response.get("contentType", ""):
        timing.error = f"unexpected contentType: {response.get('contentType')!r}"
        return timing

    for line in response["response"].iter_lines():
        if not line:
            continue
        decoded = line.decode("utf-8") if isinstance(line, bytes) else line
        if not decoded.startswith("data: "):
            continue
        event = json.loads(decoded[len("data: ") :])
        _record_event(timing, event, (time.perf_counter() - t0) * 1000.0)
    return timing


def _run_mode(target: str, prompt: str, voice: str, mode: str) -> ModeTiming:
    if target.startswith("arn:"):
        return _run_mode_deployed(target, prompt, voice, mode)
    return _run_mode_local(target, prompt, voice, mode)


def run_comparison(target: str, prompt: str, voice: str) -> list[ModeTiming]:
    """Run both modes against ``target`` and print a comparison table.

    ``target`` is either a local server base URL (``http://localhost:8080``)
    or a deployed AgentCore Runtime ARN.
    """
    results = [_run_mode(target, prompt, voice, mode) for mode in MODES]
    _print_table(results)
    return results


def _print_table(results: list[ModeTiming]) -> None:
    print(f"{'Mode':<8}{'TTF-Text (ms)':<16}{'TTFA (ms)':<14}{'Total (ms)':<14}{'Chunks':<8}")
    for r in results:
        if r.error:
            print(f"{r.mode:<8}error: {r.error}")
            continue
        print(
            f"{r.mode:<8}{r.ttft_ms or 0:<16.0f}{r.ttfa_ms or 0:<14.0f}"
            f"{r.total_ms or 0:<14.0f}{r.chunk_count:<8}"
        )

    bidi = next((r for r in results if r.mode == "bidi"), None)
    batch = next((r for r in results if r.mode == "batch"), None)
    if bidi and batch and bidi.ttfa_ms and batch.ttfa_ms:
        speedup = batch.ttfa_ms / bidi.ttfa_ms
        print(f"\nbidi TTFA is {speedup:.1f}x faster than batch TTFA.")


def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target", default="http://localhost:8080", help="Base URL or Runtime ARN."
    )
    parser.add_argument("--prompt", default="Tell me a short story about a robot learning to sing.")
    parser.add_argument("--voice", default="af_heart")
    args = parser.parse_args()
    run_comparison(args.target, args.prompt, args.voice)


if __name__ == "__main__":
    _main()
