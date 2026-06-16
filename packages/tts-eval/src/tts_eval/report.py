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
        utmos = f"{s['utmos']['mean']:.2f}" if s["utmos"]["mean"] else "N/A"
        wer_val = f"{s['wer']['mean']:.3f}" if s["wer"]["mean"] else "N/A"
        lines.append(f"| {model} | {utmos} | {wer_val} | {s['successful']}/{s['total_samples']} |")

    lines.extend(
        ["", "## Latency", "", "| Model | Mean Latency (ms) |", "|-------|-------------------|"]
    )
    for model in data["models"]:
        s = data["summary"][model]
        lat = f"{s['latency_ms']['mean']:.0f}" if s["latency_ms"]["mean"] else "N/A"
        lines.append(f"| {model} | {lat} |")

    if "benchmarks" in data:
        lines.extend(["", "## Performance Benchmarks", ""])
        bench = data["benchmarks"]

        if "cost" in bench:
            lines.extend(
                [
                    "### Cost per Million Characters",
                    "",
                    "| Model | $/M chars | Throughput (chars/min) | Instance |",
                    "|-------|-----------|-----------------------|----------|",
                ]
            )
            for entry in bench["cost"]:
                cost = entry["cost_per_m_chars"]
                cpm = entry["chars_per_min"]
                inst = entry["instance_type"]
                lines.append(f"| {entry['model']} | ${cost:.2f} | {cpm:.0f} | {inst} |")

        if "scalability" in bench:
            lines.extend(
                [
                    "",
                    "### Scalability",
                    "",
                    "| Model | Concurrency | Success Rate | P50 (ms) | P99 (ms) |",
                    "|-------|-------------|--------------|----------|----------|",
                ]
            )
            for entry in bench["scalability"]:
                lines.append(
                    f"| {entry['model']} | {entry['concurrency']} | "
                    f"{entry['success_rate']*100:.0f}% | "
                    f"{entry['p50_ms']:.0f} | {entry['p99_ms']:.0f} |"
                )

    if "human_panel" in data:
        lines.extend(
            [
                "",
                "## Human Panel Scores",
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

    lines.extend(["", "---", "*Report generated by tts-eval*", ""])
    return "\n".join(lines)


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 4)
