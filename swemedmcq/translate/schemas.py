"""Translation notes and warnings, shared by the test- and training-split translation schemas."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class TranslationNote(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_term: str = Field(..., description="Original source term or phrase.")
    translated_term: str = Field(
        ..., description="Swedish rendering used in the translation."
    )
    category: Literal[
        "standard_swedish_medical_term",
        "adapted_translation",
        "abbreviation_expanded",
        "retained_as_standard_usage",
        "context_dependent_translation",
    ] = Field(..., description="Type of terminology decision.")
    rationale: str = Field(
        ..., description="Brief explanation of the translation decision."
    )


class TranslationWarning(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal[
        "ambiguous_source",
        "awkward_or_incomplete_source",
        "image_or_missing_context_required",
        "possible_domain_specific_term",
        "possible_answer_change_risk",
    ] = Field(..., description="Type of issue that may require human review.")
    severity: Literal["low", "medium", "high"] = Field(
        ..., description="Severity of the warning."
    )
    message: str = Field(..., description="Concise explanation of the issue.")


__all__ = ["TranslationNote", "TranslationWarning"]
