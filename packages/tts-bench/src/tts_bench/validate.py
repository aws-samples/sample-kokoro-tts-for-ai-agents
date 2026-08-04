"""Closed-loop confirmation that a deployed autoscaling policy behaves as planned.

Two checks, in order:

1. ``scale_out`` — drive the fleet just over its per-instance C_scale_max threshold and
   confirm DesiredInstanceCount rises within T_total seconds.

2. ``slo_under_load`` — after the new instance is serving, hold at 80% of C_scale_max
   per instance for 60 s and confirm p95 TTFAB stays inside the SLO.

Usage::

    uv run tts-bench validate --model kokoro-82m \\
        --plan artifacts/plan-kokoro-82m.json \\
        --transport response-stream
"""

from __future__ import annotations

import json
import math
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


@dataclass(frozen=True, slots=True)
class ValidateFinding:
    check: str
    verdict: str  # "ok" | "warn" | "fail"
    detail: str


def load_plan(path: str | Path) -> dict[str, Any]:
    """Read a plan artifact and return the inner ``plan`` dict."""
    raw = json.loads(Path(path).read_text())
    return raw["plan"]


def _check_scale_out(
    *,
    client: Any,
    endpoint: str,
    model: str,
    voice: str,
    texts: Sequence[str],
    transport: str,
    c_scale_max: float,
    t_total_s: float,
    region: str,
    variant: str,
) -> tuple[ValidateFinding, int]:
    """Drive just over C_scale_max and wait for DesiredInstanceCount to rise.

    Returns the finding and the new desired count so the next check can use it.
    """
    import boto3

    from tts_bench.bidi import invoke_for
    from tts_bench.fixture import DEFAULT_VARIANT, _read_capacity
    from tts_bench.loadgen import run_step

    sagemaker = boto3.client("sagemaker", region_name=region)
    desired_before, _ = _read_capacity(sagemaker, endpoint, variant)
    fleet_concurrency = math.ceil(desired_before * c_scale_max) + 2

    stop = threading.Event()
    step_result_holder: list[Any] = []

    def _run() -> None:
        result = run_step(
            client,
            model=model,
            endpoint=endpoint,
            voice=voice,
            texts=texts,
            concurrency=fleet_concurrency,
            duration_s=t_total_s * 1.5 + 30,
            stop_event=stop,
            invoke=invoke_for(transport),
        )
        step_result_holder.append(result)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    deadline = time.monotonic() + t_total_s * 1.5
    poll_interval_s = 10.0
    trigger_time: float | None = None
    desired_after = desired_before

    while time.monotonic() < deadline:
        time.sleep(poll_interval_s)
        desired_now, _ = _read_capacity(sagemaker, endpoint, variant)
        if desired_now > desired_before:
            trigger_time = time.monotonic()
            desired_after = desired_now
            break

    stop.set()
    thread.join(timeout=30)

    if trigger_time is None:
        return (
            ValidateFinding(
                check="scale_out",
                verdict="fail",
                detail=(
                    f"DesiredInstanceCount did not rise from {desired_before} within "
                    f"{t_total_s * 1.5:.0f}s (1.5 × T_total={t_total_s:.0f}s) "
                    f"at fleet concurrency {fleet_concurrency} "
                    f"(C_scale_max={c_scale_max} × {desired_before} instances + 2)"
                ),
            ),
            desired_before,
        )

    # We can't know the exact moment the alarm fired, so we measure from when
    # the thread started. This overstates the observed lag slightly (includes
    # time to reach steady state) but is conservative in the right direction.
    observed_s = t_total_s * 1.5 - (deadline - trigger_time)
    if observed_s <= t_total_s:
        verdict = "ok"
    else:
        verdict = "warn"

    return (
        ValidateFinding(
            check="scale_out",
            verdict=verdict,
            detail=(
                f"fleet {desired_before}→{desired_after} within "
                f"~{observed_s:.0f}s  (T_total budget {t_total_s:.0f}s)"
            ),
        ),
        desired_after,
    )


def _wait_for_current(
    *,
    sagemaker: Any,
    endpoint: str,
    variant: str,
    desired: int,
    t_total_s: float,
) -> bool:
    """Poll until current instance count reaches desired, or T_total elapses.

    Returns True if current caught up within the budget.
    """
    from tts_bench.fixture import _read_capacity

    deadline = time.monotonic() + t_total_s
    while time.monotonic() < deadline:
        _, current = _read_capacity(sagemaker, endpoint, variant)
        if current >= desired:
            return True
        time.sleep(15.0)
    return False


def _check_slo_under_load(
    *,
    client: Any,
    endpoint: str,
    model: str,
    voice: str,
    texts: Sequence[str],
    transport: str,
    c_scale_max: float,
    ttfab_slo_ms: float,
    n_instances: int,
    region: str,
    variant: str,
) -> ValidateFinding:
    """Hold at 80% of C_scale_max per instance for 60 s and check p95 TTFAB."""
    from tts_bench.bidi import invoke_for
    from tts_bench.loadgen import run_step, summarize_window

    hold_concurrency = max(1, math.floor(n_instances * c_scale_max * 0.8))
    hold_s = 70.0  # 10 s warmup discarded below + 60 s measured window
    warmup_s = 10.0

    result = run_step(
        client,
        model=model,
        endpoint=endpoint,
        voice=voice,
        texts=texts,
        concurrency=hold_concurrency,
        duration_s=hold_s,
        invoke=invoke_for(transport),
    )

    window = summarize_window(
        result,
        start_ts=result.started_ts + warmup_s,
        end_ts=result.ended_ts,
    )

    p95 = window.ttfab_p95_ms
    if p95 is None:
        return ValidateFinding(
            check="slo_under_load",
            verdict="fail",
            detail=(
                f"no completed requests in the measurement window "
                f"(concurrency {hold_concurrency}, {window.completed} total completions)"
            ),
        )

    if p95 < ttfab_slo_ms:
        verdict = "ok"
    elif p95 < ttfab_slo_ms * 1.5:
        verdict = "warn"
    else:
        verdict = "fail"

    return ValidateFinding(
        check="slo_under_load",
        verdict=verdict,
        detail=(
            f"p95 TTFAB {p95:.0f}ms  (SLO {ttfab_slo_ms:.0f}ms, "
            f"fleet {n_instances} instances, concurrency {hold_concurrency})"
        ),
    )


def run(
    plan_path: str | Path,
    transport: str,
    *,
    voice: str | None = None,
    texts: list[str],
    region: str = "us-east-1",
    dry_run: bool = False,
) -> list[ValidateFinding]:
    """Run all validate checks and return findings in order."""
    plan = load_plan(plan_path)

    endpoint = plan["endpoint"]
    model_name = plan["model_name"]
    instance_type = plan["instance_type"]
    c_scale_max: float = plan["c_scale_max"]
    t_total_s: float = plan["measured"]["t_total_s"]
    ttfab_slo_ms: float = float(plan["scenario"]["ttfab_slo_ms"])

    from tts_bench.invoke import resolve_voice
    from tts_bench.types import TTSModelName

    resolved_voice = resolve_voice(TTSModelName(model_name), voice)

    if dry_run:
        import sys

        print(
            f"Validate {model_name}  ({endpoint}  {instance_type})\n"
            f"  C_scale_max={c_scale_max}  T_total={t_total_s:.0f}s  SLO={ttfab_slo_ms:.0f}ms\n"
            f"  transport={transport}  voice={resolved_voice}  texts={len(texts)}\n"
            "\n"
            "  check 1 — scale_out:\n"
            "    read DesiredInstanceCount, start load at ceil(n × C_scale_max) + 2,\n"
            f"    poll every 10 s until desired rises or {t_total_s * 1.5:.0f}s elapses\n"
            "\n"
            "  check 2 — slo_under_load:\n"
            "    wait for current instances to match new desired (up to T_total),\n"
            f"    hold at floor(n × C_scale_max × 0.8) for 70 s, read p95 TTFAB vs {ttfab_slo_ms:.0f}ms\n"
            "\n"
            "Dry run: no AWS calls made.",
            file=sys.stderr,
        )
        return []

    from tts_bench.bidi import make_client_for
    from tts_bench.fixture import DEFAULT_VARIANT, _read_capacity

    import boto3

    pool_size = max(64, math.ceil(c_scale_max) * 4)
    client = make_client_for(transport, region, max_pool=pool_size)
    variant = DEFAULT_VARIANT
    sagemaker = boto3.client("sagemaker", region_name=region)

    findings: list[ValidateFinding] = []

    scale_out_finding, desired_after = _check_scale_out(
        client=client,
        endpoint=endpoint,
        model=model_name,
        voice=resolved_voice,
        texts=texts,
        transport=transport,
        c_scale_max=c_scale_max,
        t_total_s=t_total_s,
        region=region,
        variant=variant,
    )
    findings.append(scale_out_finding)

    # Wait for the new instance to be in service before measuring SLO.
    _wait_for_current(
        sagemaker=sagemaker,
        endpoint=endpoint,
        variant=variant,
        desired=desired_after,
        t_total_s=t_total_s,
    )

    slo_finding = _check_slo_under_load(
        client=client,
        endpoint=endpoint,
        model=model_name,
        voice=resolved_voice,
        texts=texts,
        transport=transport,
        c_scale_max=c_scale_max,
        ttfab_slo_ms=ttfab_slo_ms,
        n_instances=desired_after,
        region=region,
        variant=variant,
    )
    findings.append(slo_finding)

    return findings
