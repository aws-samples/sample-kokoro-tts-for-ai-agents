# Copyright (c) 2026 Amazon.com
# This file is licensed under the MIT License.
# See the LICENSE file in the project root for full license information.

"""Tests for shared data loader."""

import json
from pathlib import Path

import pytest

from shared.loader import get_data_dir, load_tts_samples
from shared.types import SampleCategory, TTSSampleDataset


class TestGetDataDir:
    def test_returns_existing_path(self) -> None:
        data_dir = get_data_dir()
        assert data_dir.exists()
        assert data_dir.is_dir()

    def test_contains_expected_subdirs(self) -> None:
        data_dir = get_data_dir()
        assert (data_dir / "audio").exists()
        assert (data_dir / "transcripts").exists()


class TestLoadTtsSamples:
    def test_default_path(self) -> None:
        dataset = load_tts_samples()
        assert isinstance(dataset, TTSSampleDataset)
        assert dataset.version == "1.0.0"
        assert len(dataset.samples) >= 15

    def test_all_samples_have_required_fields(self) -> None:
        dataset = load_tts_samples()
        for sample in dataset.samples:
            assert sample.id
            assert sample.text
            assert sample.category in SampleCategory
            assert sample.language == "en"

    def test_categories_coverage(self) -> None:
        dataset = load_tts_samples()
        categories = {s.category for s in dataset.samples}
        assert SampleCategory.SHORT in categories
        assert SampleCategory.MEDIUM in categories
        assert SampleCategory.LONG in categories

    def test_explicit_path(self, tmp_path: Path) -> None:
        data = {
            "version": "0.1.0",
            "description": "test",
            "samples": [
                {
                    "id": "tmp-01",
                    "text": "Hello",
                    "category": "short",
                    "style": "conversational",
                    "linguistic_features": [],
                    "language": "en",
                }
            ],
        }
        json_file = tmp_path / "test_samples.json"
        json_file.write_text(json.dumps(data))

        dataset = load_tts_samples(json_file)
        assert len(dataset.samples) == 1
        assert dataset.samples[0].id == "tmp-01"

    def test_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_tts_samples(tmp_path / "nonexistent.json")
