"""Structured-output schema of the translations (test split and training split)."""

from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, ConfigDict, Field

from ..schemas import (
    TranslationNote,
    TranslationWarning,
)


class ChoiceInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str = Field(
        ..., description="Original answer-choice label, e.g. A, B, C, D, and so on."
    )
    text: str = Field(..., description="Original answer-choice text.")


class MedicalMCQTranslationInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target_language: Literal["Swedish"] = Field(
        default="Swedish", description="Target language for translation."
    )
    question: str = Field(..., description="Source question text.")
    choices: list[ChoiceInput] = Field(
        ..., min_length=1, description="Original answer choices in source order."
    )


class TranslatedChoice(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str = Field(
        ..., description="Original answer-choice label copied exactly from input."
    )
    text: str = Field(..., description="Translated Swedish answer-choice text.")


class MedicalMCQTranslationOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_language: str = Field(..., description="Detected source language.")
    target_language: Literal["Swedish"] = Field(
        ..., description="Target language. Must be Swedish."
    )
    question: str = Field(..., description="Translated Swedish question.")
    choices: list[TranslatedChoice] = Field(
        ..., min_length=1, description="Translated choices in the same order as input."
    )
    translation_notes: list[TranslationNote] = Field(
        ...,
        description=(
            "Important terminology decisions or context-dependent translation choices. "
            "Return an empty array if there are no noteworthy terminology decisions."
        ),
    )
    warnings: list[TranslationWarning] = Field(
        ...,
        description=(
            "Potential ambiguity, awkward source wording, missing context, or answer-change risk. "
            "Return an empty array if there are no warnings."
        ),
    )


__all__ = [
    "ChoiceInput",
    "MedicalMCQTranslationInput",
    "MedicalMCQTranslationOutput",
    "TranslatedChoice",
    "TranslationNote",
    "TranslationWarning",
]
