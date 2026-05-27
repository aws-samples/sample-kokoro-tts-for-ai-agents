"""Types for shared data loading and sample datasets."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class SampleCategory(StrEnum):
    """Length category for TTS text samples."""

    SHORT = "short"
    MEDIUM = "medium"
    LONG = "long"


class SpeakingStyle(StrEnum):
    """Speaking style for TTS text samples."""

    CONVERSATIONAL = "conversational"
    FORMAL = "formal"
    TECHNICAL = "technical"
    NARRATIVE = "narrative"
    INSTRUCTIONAL = "instructional"


class LinguisticFeature(StrEnum):
    """Linguistic features present in a text sample."""

    QUESTION = "question"
    EXCLAMATION = "exclamation"
    NUMBERS = "numbers"
    ABBREVIATIONS = "abbreviations"
    PROPER_NOUNS = "proper_nouns"
    FOREIGN_WORDS = "foreign_words"
    TECHNICAL_TERMS = "technical_terms"
    MIXED_PUNCTUATION = "mixed_punctuation"
    DIALOGUE = "dialogue"
    DATES = "dates"
    ACRONYMS = "acronyms"
    PHONETICALLY_TRICKY = "phonetically_tricky"


class TTSSample(BaseModel):
    """A single TTS text sample with metadata."""

    id: str = Field(description="Unique sample identifier (kebab-case)")
    text: str = Field(description="Text content to synthesize")
    category: SampleCategory
    style: SpeakingStyle
    linguistic_features: list[LinguisticFeature] = Field(default_factory=list)
    expected_duration_range_seconds: tuple[float, float] | None = Field(
        default=None,
        description="Rough expected audio duration range [min, max] in seconds",
    )
    language: str = Field(default="en", description="BCP-47 language tag")
    notes: str | None = None


class TTSSampleDataset(BaseModel):
    """Top-level container for TTS sample text data."""

    version: str = Field(description="Schema version")
    description: str = Field(default="")
    samples: list[TTSSample]

    def get_by_id(self, sample_id: str) -> TTSSample | None:
        for sample in self.samples:
            if sample.id == sample_id:
                return sample
        return None

    def filter_by_category(self, category: SampleCategory) -> list[TTSSample]:
        return [s for s in self.samples if s.category == category]

    def filter_by_style(self, style: SpeakingStyle) -> list[TTSSample]:
        return [s for s in self.samples if s.style == style]

    def filter_by_feature(self, feature: LinguisticFeature) -> list[TTSSample]:
        return [s for s in self.samples if feature in s.linguistic_features]
