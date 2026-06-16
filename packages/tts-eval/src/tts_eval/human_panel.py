"""Human panel evaluation support.

Generates randomized blind listening tests and ingests scored results.
Models are anonymized (Model A/B/C) with randomized presentation order.
"""

from __future__ import annotations

import csv
import json
import random
import shutil
from pathlib import Path

from loguru import logger

from tts_inference.types import TTSModelName


def generate_listening_test(
    samples_dir: Path,
    output_dir: Path,
    num_samples: int | None = None,
    seed: int = 42,
) -> Path:
    """Generate a blind listening test package from synthesized samples.

    Args:
        samples_dir: Directory containing model subdirectories with WAV files.
            Expected structure: samples_dir/{model_name}/{sample_id}.wav
        output_dir: Where to write the listening test package.
        num_samples: Number of samples to include (None = all available).
        seed: Random seed for reproducible randomization.

    Returns:
        Path to the generated listening test directory.
    """
    rng = random.Random(seed)

    models = sorted(
        [d.name for d in samples_dir.iterdir() if d.is_dir()],
    )
    if not models:
        raise ValueError(f"No model directories found in {samples_dir}")

    all_samples = sorted(
        [f.stem for f in (samples_dir / models[0]).glob("*.wav")],
    )
    if num_samples and num_samples < len(all_samples):
        all_samples = rng.sample(all_samples, num_samples)

    model_labels = {}
    shuffled_models = list(models)
    rng.shuffle(shuffled_models)
    for i, model in enumerate(shuffled_models):
        model_labels[model] = chr(ord("A") + i)

    test_dir = output_dir / "human_panel"
    audio_dir = test_dir / "listening_test"
    audio_dir.mkdir(parents=True, exist_ok=True)

    items = []
    for sample_id in all_samples:
        for model in models:
            src = samples_dir / model / f"{sample_id}.wav"
            if not src.exists():
                logger.warning("Missing: {}", src)
                continue

            label = model_labels[model]
            dest_name = f"{sample_id}_model-{label}.wav"
            shutil.copy2(src, audio_dir / dest_name)
            items.append({
                "sample_id": sample_id,
                "model_label": label,
                "filename": dest_name,
            })

    rng.shuffle(items)

    _write_scoresheet(test_dir, items)
    _write_mapping(test_dir, model_labels)

    logger.info(
        "Generated listening test: {} items, {} models, {} samples",
        len(items),
        len(models),
        len(all_samples),
    )
    return test_dir


def _write_scoresheet(test_dir: Path, items: list[dict]) -> None:
    """Write blank CSV scoresheet for human listeners."""
    scoresheet = test_dir / "scoresheet.csv"
    with scoresheet.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "filename",
            "sample_id",
            "model_label",
            "naturalness_1_5",
            "clarity_1_5",
            "pacing_1_5",
            "consistency_1_5",
            "overall_1_5",
            "notes",
        ])
        for item in items:
            writer.writerow([
                item["filename"],
                item["sample_id"],
                item["model_label"],
                "",
                "",
                "",
                "",
                "",
                "",
            ])


def _write_mapping(test_dir: Path, model_labels: dict[str, str]) -> None:
    """Write model identity mapping (kept separate from scoresheet)."""
    mapping = test_dir / "mapping.json"
    mapping.write_text(
        json.dumps(
            {"model_labels": model_labels, "label_to_model": {v: k for k, v in model_labels.items()}},
            indent=2,
        )
    )


def ingest_scores(
    scoresheet_path: Path,
    mapping_path: Path,
) -> dict:
    """Ingest completed scoresheets and compute aggregated results.

    Args:
        scoresheet_path: Path to completed CSV scoresheet.
        mapping_path: Path to mapping.json to decode model identities.

    Returns:
        Dict with per-model aggregated scores and inter-rater stats.
    """
    mapping = json.loads(mapping_path.read_text())
    label_to_model = mapping["label_to_model"]

    dimensions = ["naturalness", "clarity", "pacing", "consistency", "overall"]
    model_scores: dict[str, dict[str, list[float]]] = {}

    with scoresheet_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = row["model_label"]
            model = label_to_model.get(label, label)

            if model not in model_scores:
                model_scores[model] = {d: [] for d in dimensions}

            for dim in dimensions:
                val = row.get(f"{dim}_1_5", "").strip()
                if val:
                    model_scores[model][dim].append(float(val))

    results = {}
    for model, scores in model_scores.items():
        results[model] = {}
        for dim, values in scores.items():
            if values:
                results[model][dim] = {
                    "mean": round(sum(values) / len(values), 2),
                    "count": len(values),
                    "min": min(values),
                    "max": max(values),
                }
            else:
                results[model][dim] = {"mean": None, "count": 0}

    return {"model_scores": results, "mapping": label_to_model}
