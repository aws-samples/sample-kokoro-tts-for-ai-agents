"""Report generation - produces JSON (machine-readable) and Markdown (human-readable)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from tts_eval.runner import EvalResult


def generate_report(
    results: list[EvalResult],
    output_dir: Path,
    bench_results: dict | None = None,
    human_scores: dict | None = None,
) -> tuple[Path, Path]:
    """Generate both JSON and Markdown reports.

    Args:
        results: List of evaluation results from EvalRunner.
        output_dir: Directory to write reports to.
        bench_results: Optional benchmark results (latency/cost/scalability).
        human_scores: Optional human panel scores.

    Returns:
        Tuple of (json_path, markdown_path).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    report_data = _build_report_data(results, bench_results, human_scores)

    json_path = output_dir / "results.json"
    json_path.write_text(json.dumps(report_data, indent=2, default=str))

    md_path = output_dir / "report.md"
    md_path.write_text(_render_markdown(report_data))

    return json_path, md_path


def _build_report_data(
    results: list[EvalResult],
    bench_results: dict | None = None,
    human_scores: dict | None = None,
) -> dict:
    """Build structured report data."""
    models = sorted({r.model for r in results})

    per_model: dict[str, dict] = {}
    for model in models:
        model_results = [r for r in results if r.model == model]
        successful = [r for r in model_results if not r.error]

        utmos_scores = [r.utmos for r in successful if r.utmos is not None]
        wer_scores = [r.wer for r in successful if r.wer is not None]

        rtf_values = [r.rtf for r in successful if r.rtf is not None]
        ttfab_values = [r.ttfab_ms for r in successful if r.ttfab_ms is not None]

        per_model[model] = {
            "total_samples": len(model_results),
            "successful": len(successful),
            "failed": len(model_results) - len(successful),
            "utmos": {
                "mean": _mean(utmos_scores),
                "min": min(utmos_scores) if utmos_scores else None,
                "max": max(utmos_scores) if utmos_scores else None,
                "count": len(utmos_scores),
            },
            "wer": {
                "mean": _mean(wer_scores),
                "min": min(wer_scores) if wer_scores else None,
                "max": max(wer_scores) if wer_scores else None,
                "count": len(wer_scores),
            },
            "latency_ms": {
                "mean": _mean([r.latency_ms for r in successful if r.latency_ms]),
            },
            "rtf": {
                "mean": _mean(rtf_values),
                "min": min(rtf_values) if rtf_values else None,
                "max": max(rtf_values) if rtf_values else None,
            },
            "ttfab_ms": {
                "mean": _mean(ttfab_values),
                "p50": _percentile(ttfab_values, 50),
                "p99": _percentile(ttfab_values, 99),
            },
            "samples": [r.to_dict() for r in model_results],
        }

    report = {
        "timestamp": datetime.now(UTC).isoformat(),
        "models": models,
        "summary": per_model,
    }

    if bench_results:
        report["benchmarks"] = bench_results
    if human_scores:
        report["human_panel"] = human_scores

    return report


VOICE_CUSTOMIZATION: dict[str, dict[str, str]] = {
    "kokoro-82m": {
        "method": "Pre-trained voices",
        "effort": "None",
        "notes": "30+ built-in voices, no cloning support",
    },
    "polly-standard": {
        "method": "Managed service",
        "effort": "None",
        "notes": "AWS Polly Standard, Salli voice (en-US)",
    },
    "polly-neural": {
        "method": "Managed service",
        "effort": "None",
        "notes": "AWS Polly Neural, Joanna voice (en-US)",
    },
    "polly-generative": {
        "method": "Managed service",
        "effort": "None",
        "notes": "AWS Polly Generative, Ruth voice (en-US)",
    },
}


def _render_markdown(data: dict) -> str:
    """Render report data as Markdown."""
    lines = [
        "# TTS Model Evaluation Report",
        "",
        f"**Generated:** {data['timestamp']}",
        f"**Models:** {', '.join(data['models'])}",
        "",
        "## Audio Quality (Automated)",
        "",
        "| Model | UTMOS (1-5) | WER | Samples |",
        "|-------|-------------|-----|---------|",
    ]

    for model in data["models"]:
        s = data["summary"][model]
        utmos = f"{s['utmos']['mean']:.2f}" if s["utmos"]["mean"] is not None else "N/A"
        wer_val = f"{s['wer']['mean']:.3f}" if s["wer"]["mean"] is not None else "N/A"
        lines.append(f"| {model} | {utmos} | {wer_val} | {s['successful']}/{s['total_samples']} |")

    lines.extend(
        [
            "",
            "## Latency",
            "",
            "| Model | Mean Latency (ms) | RTF | TTFAB P50 (ms) | TTFAB P99 (ms) |",
            "|-------|-------------------|-----|----------------|----------------|",
        ]
    )
    for model in data["models"]:
        s = data["summary"][model]
        lat = f"{s['latency_ms']['mean']:.0f}" if s["latency_ms"]["mean"] is not None else "N/A"
        rtf_val = f"{s['rtf']['mean']:.2f}" if s["rtf"]["mean"] is not None else "N/A"
        ttfab_p50 = f"{s['ttfab_ms']['p50']:.0f}" if s["ttfab_ms"]["p50"] is not None else "N/A"
        ttfab_p99 = f"{s['ttfab_ms']['p99']:.0f}" if s["ttfab_ms"]["p99"] is not None else "N/A"
        if ttfab_p50 == "N/A" and "benchmarks" in data and "scalability" in data["benchmarks"]:
            baseline = [
                e
                for e in data["benchmarks"]["scalability"]
                if e["model"] == model and e.get("concurrency") == 1
            ]
            if baseline:
                ttfab_p50 = f"{baseline[0]['ttfab_p50_ms']:.0f}"
                ttfab_p99 = f"{baseline[0]['ttfab_p99_ms']:.0f}"
        lines.append(f"| {model} | {lat} | {rtf_val} | {ttfab_p50} | {ttfab_p99} |")

    if "benchmarks" in data:
        bench = data["benchmarks"]

        if "cost" in bench:
            lines.extend(
                [
                    "",
                    "## Cost",
                    "",
                    "| Model | $/M chars | Throughput (chars/min) | Sat. C | Instance | $/hr |",
                    "|-------|-----------|-----------------------|--------|----------|------|",
                ]
            )
            for entry in bench["cost"]:
                cost = entry["cost_per_m_chars"]
                cpm = entry["chars_per_min"]
                sat_c = entry.get("saturation_concurrency", "N/A")
                inst = entry["instance_type"]
                cost_hr = entry.get("instance_cost_per_hr", "N/A")
                cost_hr_str = f"${cost_hr:.2f}" if isinstance(cost_hr, int | float) else cost_hr
                lines.append(
                    f"| {entry['model']} | ${cost:.2f} | {cpm:.0f} | {sat_c} | {inst} | {cost_hr_str} |"
                )

        if "scalability" in bench:
            lines.extend(
                [
                    "",
                    "## Scalability",
                    "",
                    "| Model | Concurrency | Throughput (chars/s) | Success | P50 (ms) | P99 (ms) | TTFAB P50 |",
                    "|-------|-------------|----------------------|---------|----------|----------|-----------|",
                ]
            )
            for entry in bench["scalability"]:
                ttfab = entry.get("ttfab_p50_ms", "N/A")
                ttfab_str = f"{ttfab:.0f}" if isinstance(ttfab, int | float) else ttfab
                throughput = entry.get("throughput_chars_per_s", 0)
                total_req = entry.get("total_requests", 0)
                success_rate = "100%" if total_req > 0 else "0%"
                lines.append(
                    f"| {entry['model']} | {entry['concurrency']} | "
                    f"{throughput:.0f} | {success_rate} | "
                    f"{entry['p50_ms']:.0f} | {entry['p99_ms']:.0f} | {ttfab_str} |"
                )

    lines.extend(
        [
            "",
            "## Voice Customization",
            "",
            "| Model | Cloning Method | Effort | Notes |",
            "|-------|----------------|--------|-------|",
        ]
    )
    for model in data["models"]:
        vc = VOICE_CUSTOMIZATION.get(model, {"method": "Unknown", "effort": "N/A", "notes": ""})
        lines.append(f"| {model} | {vc['method']} | {vc['effort']} | {vc['notes']} |")

    if "human_panel" in data:
        lines.extend(
            [
                "",
                "## Human Panel Scores (3-5 listeners, blind)",
                "",
                "| Model | Naturalness | Clarity | Pacing | Consistency | Overall |",
                "|-------|-------------|---------|--------|-------------|---------|",
            ]
        )
        hp = data["human_panel"].get("model_scores", {})
        for model, scores in hp.items():
            nat = (
                f"{scores['naturalness']['mean']:.1f}"
                if scores.get("naturalness", {}).get("mean")
                else "N/A"
            )
            cla = (
                f"{scores['clarity']['mean']:.1f}"
                if scores.get("clarity", {}).get("mean")
                else "N/A"
            )
            pac = (
                f"{scores['pacing']['mean']:.1f}" if scores.get("pacing", {}).get("mean") else "N/A"
            )
            con = (
                f"{scores['consistency']['mean']:.1f}"
                if scores.get("consistency", {}).get("mean")
                else "N/A"
            )
            ovr = (
                f"{scores['overall']['mean']:.1f}"
                if scores.get("overall", {}).get("mean")
                else "N/A"
            )
            lines.append(f"| {model} | {nat} | {cla} | {pac} | {con} | {ovr} |")
    else:
        lines.extend(
            [
                "",
                "## Human Panel Scores (3-5 listeners, blind)",
                "",
                "*Pending: Generate listening test with `tts-eval human-panel generate` "
                "and collect scores from 3-5 listeners.*",
                "",
                "Dimensions: Naturalness (1-5), Clarity (1-5), Pacing (1-5), "
                "Consistency (1-5), Overall (1-5)",
            ]
        )

    lines.extend(
        [
            "",
            "## Voice Snippets (Side-by-Side)",
            "",
        ]
    )
    models = data["models"]
    if len(models) > 0:
        header = "| Sample |" + " | ".join(models) + " |"
        sep = "|--------|" + " | ".join(["---"] * len(models)) + " |"
        lines.extend([header, sep])

        sample_ids: list[str] = []
        for model in models:
            for sample in data["summary"][model].get("samples", []):
                sid = sample.get("sample_id", "")
                if sid and sid not in sample_ids:
                    sample_ids.append(sid)

        for sid in sample_ids[:10]:
            row = f"| {sid} |"
            for model in models:
                row += f" `samples/{model}/{sid}.wav` |"
            lines.append(row)

    lines.extend(["", "---", "*Report generated by tts-eval*", ""])
    return "\n".join(lines)


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 4)


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    idx = (pct / 100) * (len(s) - 1)
    lo = int(idx)
    hi = lo + 1
    if hi >= len(s):
        return s[-1]
    frac = idx - lo
    return s[lo] + frac * (s[hi] - s[lo])
