"""Data loader for sample datasets."""

from __future__ import annotations

import json
from pathlib import Path

from loguru import logger

from shared.types import TTSSampleDataset


def _find_project_root() -> Path:
    """Walk up from this file to find the project root (contains pyproject.toml + data/)."""
    current = Path(__file__).resolve()
    candidate = current.parents[4]
    if (candidate / "pyproject.toml").exists() and (candidate / "data").is_dir():
        return candidate
    for parent in current.parents:
        if (parent / "pyproject.toml").exists() and (parent / "data").is_dir():
            return parent
    raise FileNotFoundError(
        "Cannot locate project root (expected pyproject.toml + data/ directory)"
    )


def get_data_dir() -> Path:
    """Return the absolute path to the project's data/ directory."""
    return _find_project_root() / "data"


def load_tts_samples(path: Path | str | None = None) -> TTSSampleDataset:
    """Load TTS sample texts from JSON.

    Args:
        path: Optional explicit path to the JSON file. If None, uses
              the default location at data/tts_samples.json.

    Returns:
        Parsed and validated TTSSampleDataset.
    """
    if path is None:
        path = get_data_dir() / "tts_samples.json"
    else:
        path = Path(path)

    logger.debug("Loading TTS samples from {}", path)

    if not path.exists():
        raise FileNotFoundError(f"TTS samples file not found: {path}")

    raw = json.loads(path.read_text(encoding="utf-8"))
    dataset = TTSSampleDataset.model_validate(raw)
    logger.info("Loaded {} TTS samples from {}", len(dataset.samples), path.name)
    return dataset
