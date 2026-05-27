"""Tests for shared types."""

from shared.types import (
    LinguisticFeature,
    SampleCategory,
    SpeakingStyle,
    TTSSample,
    TTSSampleDataset,
)


class TestSampleCategory:
    def test_enum_values(self) -> None:
        assert SampleCategory.SHORT == "short"
        assert SampleCategory.MEDIUM == "medium"
        assert SampleCategory.LONG == "long"
        assert len(SampleCategory) == 3


class TestSpeakingStyle:
    def test_enum_values(self) -> None:
        assert SpeakingStyle.CONVERSATIONAL == "conversational"
        assert SpeakingStyle.FORMAL == "formal"
        assert SpeakingStyle.TECHNICAL == "technical"
        assert SpeakingStyle.NARRATIVE == "narrative"
        assert SpeakingStyle.INSTRUCTIONAL == "instructional"
        assert len(SpeakingStyle) == 5


class TestLinguisticFeature:
    def test_enum_values(self) -> None:
        assert LinguisticFeature.QUESTION == "question"
        assert LinguisticFeature.PHONETICALLY_TRICKY == "phonetically_tricky"
        assert len(LinguisticFeature) == 12


class TestTTSSample:
    def test_construction(self) -> None:
        sample = TTSSample(
            id="test-01",
            text="Hello world",
            category=SampleCategory.SHORT,
            style=SpeakingStyle.CONVERSATIONAL,
        )
        assert sample.id == "test-01"
        assert sample.text == "Hello world"
        assert sample.linguistic_features == []
        assert sample.expected_duration_range_seconds is None
        assert sample.language == "en"
        assert sample.notes is None

    def test_with_all_fields(self) -> None:
        sample = TTSSample(
            id="test-02",
            text="Testing 1 2 3",
            category=SampleCategory.MEDIUM,
            style=SpeakingStyle.TECHNICAL,
            linguistic_features=[LinguisticFeature.NUMBERS, LinguisticFeature.TECHNICAL_TERMS],
            expected_duration_range_seconds=(2.0, 4.0),
            language="en",
            notes="A test sample",
        )
        assert sample.expected_duration_range_seconds == (2.0, 4.0)
        assert len(sample.linguistic_features) == 2

    def test_json_roundtrip(self) -> None:
        sample = TTSSample(
            id="rt-01",
            text="Round trip test",
            category=SampleCategory.SHORT,
            style=SpeakingStyle.FORMAL,
            linguistic_features=[LinguisticFeature.PROPER_NOUNS],
            expected_duration_range_seconds=(1.0, 2.5),
        )
        json_str = sample.model_dump_json()
        restored = TTSSample.model_validate_json(json_str)
        assert restored.id == "rt-01"
        assert restored.expected_duration_range_seconds == (1.0, 2.5)
        assert restored.linguistic_features == [LinguisticFeature.PROPER_NOUNS]


class TestTTSSampleDataset:
    def _make_dataset(self) -> TTSSampleDataset:
        return TTSSampleDataset(
            version="1.0.0",
            description="Test dataset",
            samples=[
                TTSSample(
                    id="s1",
                    text="Short one",
                    category=SampleCategory.SHORT,
                    style=SpeakingStyle.CONVERSATIONAL,
                    linguistic_features=[LinguisticFeature.QUESTION],
                ),
                TTSSample(
                    id="s2",
                    text="Medium tech",
                    category=SampleCategory.MEDIUM,
                    style=SpeakingStyle.TECHNICAL,
                    linguistic_features=[LinguisticFeature.NUMBERS, LinguisticFeature.ACRONYMS],
                ),
                TTSSample(
                    id="s3",
                    text="Long narrative",
                    category=SampleCategory.LONG,
                    style=SpeakingStyle.NARRATIVE,
                    linguistic_features=[LinguisticFeature.NUMBERS],
                ),
            ],
        )

    def test_get_by_id_found(self) -> None:
        ds = self._make_dataset()
        sample = ds.get_by_id("s2")
        assert sample is not None
        assert sample.text == "Medium tech"

    def test_get_by_id_not_found(self) -> None:
        ds = self._make_dataset()
        assert ds.get_by_id("nonexistent") is None

    def test_filter_by_category(self) -> None:
        ds = self._make_dataset()
        short = ds.filter_by_category(SampleCategory.SHORT)
        assert len(short) == 1
        assert short[0].id == "s1"

    def test_filter_by_style(self) -> None:
        ds = self._make_dataset()
        tech = ds.filter_by_style(SpeakingStyle.TECHNICAL)
        assert len(tech) == 1
        assert tech[0].id == "s2"

    def test_filter_by_feature(self) -> None:
        ds = self._make_dataset()
        with_numbers = ds.filter_by_feature(LinguisticFeature.NUMBERS)
        assert len(with_numbers) == 2
        assert {s.id for s in with_numbers} == {"s2", "s3"}
