"""Closed-loop confirmation that a deployed autoscaling policy behaves as planned.

Four checks, in order:

1. ``scale_out`` — drive the fleet just over its per-instance C_scale_max threshold and
   confirm DesiredInstanceCount rises within T_total seconds.

2. ``slo_under_load`` — after the new instance is serving, hold at 80% of C_scale_max
   per instance for 60 s and confirm p95 TTFAB stays inside the SLO.

3. ``rejection_level`` — run two steps: one at Q_max-10, one at Q_max+20. Confirm
   zero 503s below the bound and non-zero above it. The asymmetric margins account
   for SageMaker round-robin routing variance across instances.

4. ``surge_absorption`` — baseline at 80% C_scale_max, then surge to
   ``surge_ratio × C_scale_max`` and hold for 1.2 × T_total seconds. Confirm the
   fleet grows and p95 (measured after the scale-out window) stays inside 1.5 × SLO.

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
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


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
    from tts_bench.fixture import _read_capacity
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


def _check_rejection_level(
    *,
    client: Any,
    endpoint: str,
    model: str,
    voice: str,
    texts: Sequence[str],
    transport: str,
    q_max: int,
    n_instances: int,
) -> ValidateFinding:
    """Two steps: one below Q_max, one above. Confirm the admission bound is enforced."""
    from tts_bench.bidi import invoke_for
    from tts_bench.loadgen import run_step, summarize_window

    invoke = invoke_for(transport)
    warmup_s = 10.0
    hold_s = 70.0

    below_concurrency = max(1, math.floor(n_instances * (q_max - 10)))
    above_concurrency = math.ceil(n_instances * (q_max + 20))

    below_result = run_step(
        client,
        model=model,
        endpoint=endpoint,
        voice=voice,
        texts=texts,
        concurrency=below_concurrency,
        duration_s=hold_s,
        invoke=invoke,
    )
    below_window = summarize_window(
        below_result,
        start_ts=below_result.started_ts + warmup_s,
        end_ts=below_result.ended_ts,
    )

    above_result = run_step(
        client,
        model=model,
        endpoint=endpoint,
        voice=voice,
        texts=texts,
        concurrency=above_concurrency,
        duration_s=hold_s,
        invoke=invoke,
    )
    above_window = summarize_window(
        above_result,
        start_ts=above_result.started_ts + warmup_s,
        end_ts=above_result.ended_ts,
    )

    below_rejected = below_window.rejected
    above_rejected = above_window.rejected
    above_pct = (above_rejected / above_window.completed * 100) if above_window.completed else 0.0

    if below_rejected == 0 and above_rejected > 0:
        verdict = "ok"
    elif below_rejected > 0 and above_rejected == 0:
        verdict = "fail"
    else:
        verdict = "warn"

    return ValidateFinding(
        check="rejection_level",
        verdict=verdict,
        detail=(
            f"{below_rejected} rejections at conc {below_concurrency} (below Q_max={q_max}); "
            f"{above_rejected} ({above_pct:.1f}%) at conc {above_concurrency} (above Q_max)"
        ),
    )


def _check_surge_absorption(
    *,
    client: Any,
    endpoint: str,
    model: str,
    voice: str,
    texts: Sequence[str],
    transport: str,
    c_scale_max: float,
    t_total_s: float,
    ttfab_slo_ms: float,
    surge_ratio: float,
    n_instances: int,
    region: str,
    variant: str,
) -> ValidateFinding:
    """Baseline at 80% C_scale_max, then surge to surge_ratio × C_scale_max.

    Holds for 1.2 × T_total to give the fleet time to scale out. Measures p95 and
    rejection rate over the second half of the surge window.
    """
    import boto3

    from tts_bench.bidi import invoke_for
    from tts_bench.fixture import _read_capacity
    from tts_bench.loadgen import run_step, summarize_window

    invoke = invoke_for(transport)
    sagemaker = boto3.client("sagemaker", region_name=region)

    # Baseline: below threshold, 60 s.
    baseline_concurrency = max(1, math.floor(n_instances * c_scale_max * 0.8))
    run_step(
        client,
        model=model,
        endpoint=endpoint,
        voice=voice,
        texts=texts,
        concurrency=baseline_concurrency,
        duration_s=60.0,
        invoke=invoke,
    )

    desired_before, _ = _read_capacity(sagemaker, endpoint, variant)
    surge_concurrency = math.ceil(n_instances * c_scale_max * surge_ratio)
    surge_hold_s = t_total_s * 1.2

    surge_result = run_step(
        client,
        model=model,
        endpoint=endpoint,
        voice=voice,
        texts=texts,
        concurrency=surge_concurrency,
        duration_s=surge_hold_s,
        invoke=invoke,
    )

    desired_after, _ = _read_capacity(sagemaker, endpoint, variant)
    fleet_grew = desired_after > desired_before

    # Measure the second half only — the first half is before the new instance serves.
    measure_start = surge_result.started_ts + surge_hold_s * 0.5
    window = summarize_window(
        surge_result,
        start_ts=measure_start,
        end_ts=surge_result.ended_ts,
    )

    p95 = window.ttfab_p95_ms
    rejection_pct = (window.rejected / window.completed * 100) if window.completed else 0.0
    rejection_threshold = 1.0

    if not fleet_grew:
        verdict = "fail"
    elif p95 is not None and p95 >= ttfab_slo_ms * 1.5:
        verdict = "fail"
    elif rejection_pct >= rejection_threshold:
        verdict = "warn"
    else:
        verdict = "ok"

    p95_str = f"{p95:.0f}ms" if p95 is not None else "n/a"
    return ValidateFinding(
        check="surge_absorption",
        verdict=verdict,
        detail=(
            f"fleet {desired_before}→{desired_after}; "
            f"p95 {p95_str} in surge window; "
            f"{rejection_pct:.1f}% shed  "
            f"(concurrency {surge_concurrency}, ratio {surge_ratio:.2f}×)"
        ),
    )


def run(
    plan_path: str | Path,
    transport: str,
    *,
    voice: str | None = None,
    texts: list[str],
    region: str = "us-east-1",
    surge_ratio: float | None = None,
    dry_run: bool = False,
) -> list[ValidateFinding]:
    """Run all validate checks and return findings in order."""
    plan = load_plan(plan_path)

    endpoint = plan["endpoint"]
    model_name = plan["model_name"]
    instance_type = plan["instance_type"]
    c_scale_max: float = plan["c_scale_max"]
    q_max: int = int(plan["q_max"])
    t_total_s: float = plan["measured"]["t_total_s"]
    ttfab_slo_ms: float = float(plan["scenario"]["ttfab_slo_ms"])
    effective_surge_ratio: float = surge_ratio or float(plan["scenario"]["max_scaling_per_t_total"])

    from tts_bench.invoke import resolve_voice
    from tts_bench.types import TTSModelName

    resolved_voice = resolve_voice(TTSModelName(model_name), voice)

    if dry_run:
        import sys

        print(
            f"Validate {model_name}  ({endpoint}  {instance_type})\n"
            f"  C_scale_max={c_scale_max}  Q_max={q_max}  T_total={t_total_s:.0f}s  SLO={ttfab_slo_ms:.0f}ms\n"
            f"  transport={transport}  voice={resolved_voice}  texts={len(texts)}\n"
            f"  surge_ratio={effective_surge_ratio}\n"
            "\n"
            "  check 1 — scale_out:\n"
            "    read DesiredInstanceCount, start load at ceil(n × C_scale_max) + 2,\n"
            f"    poll every 10 s until desired rises or {t_total_s * 1.5:.0f}s elapses\n"
            "\n"
            "  check 2 — slo_under_load:\n"
            "    wait for current instances to match new desired (up to T_total),\n"
            f"    hold at floor(n × C_scale_max × 0.8) for 70 s, read p95 TTFAB vs {ttfab_slo_ms:.0f}ms\n"
            "\n"
            "  check 3 — rejection_level:\n"
            f"    two steps: conc floor(n × {q_max - 10}) then ceil(n × {q_max + 20}),\n"
            "    expect 0 rejections below Q_max, non-zero above\n"
            "\n"
            "  check 4 — surge_absorption:\n"
            f"    baseline 60 s at 0.8 × C_scale_max, then surge to {effective_surge_ratio:.2f}× C_scale_max\n"
            f"    for {t_total_s * 1.2:.0f}s; confirm fleet grows and p95 < {ttfab_slo_ms * 1.5:.0f}ms\n"
            "\n"
            "Dry run: no AWS calls made.",
            file=sys.stderr,
        )
        return []

    import boto3

    from tts_bench.bidi import make_client_for
    from tts_bench.fixture import DEFAULT_VARIANT

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

    # Wait for the new instance to be in service before proceeding.
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

    rejection_finding = _check_rejection_level(
        client=client,
        endpoint=endpoint,
        model=model_name,
        voice=resolved_voice,
        texts=texts,
        transport=transport,
        q_max=q_max,
        n_instances=desired_after,
    )
    findings.append(rejection_finding)

    surge_finding = _check_surge_absorption(
        client=client,
        endpoint=endpoint,
        model=model_name,
        voice=resolved_voice,
        texts=texts,
        transport=transport,
        c_scale_max=c_scale_max,
        t_total_s=t_total_s,
        ttfab_slo_ms=ttfab_slo_ms,
        surge_ratio=effective_surge_ratio,
        n_instances=desired_after,
        region=region,
        variant=variant,
    )
    findings.append(surge_finding)

    return findings
