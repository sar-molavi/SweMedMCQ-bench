"""Structured-output schemas of the benchmark answers.

The JSON schema of each class (``model_json_schema()``, including the class name
and docstring) is what vLLM's grammar receives,
so names, docstrings and field descriptions are part of the protocol.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ResponseOutput(BaseModel):
    """The model's structured response to a multiple-choice medical question."""
    model_config = ConfigDict(extra="forbid")

    explanation: str = Field(
        description="The medical reasoning explaining why this answer is correct and others are not."
    )
    answer_index: str = Field(
        description=(
            "The letter corresponding to the correct answer option "
            "(e.g., 'A', 'B', 'C', 'D', 'E', etc.), depending on how many options are provided."
        )
    )


class SweResponseOutput(BaseModel):
    """Modellens strukturerade svar på en flervalsfråga inom medicin."""
    model_config = ConfigDict(extra="forbid")

    explanation: str = Field(
        description="Den medicinska förklaringen till varför detta svar är korrekt och varför de andra alternativen inte är det."
    )
    answer_index: str = Field(
        description=(
            "Bokstaven som motsvarar det korrekta svarsalternativet "
            "(t.ex. 'A', 'B', 'C', 'D', 'E' osv.), beroende på hur många alternativ som anges."
        )
    )


class AnswerLevelResponseOutput(BaseModel):
    """Structured output for answer-only MCQ prediction."""

    model_config = ConfigDict(extra="forbid")

    answer_index: str = Field(
        description=(
            "The letter corresponding to the correct answer option "
            "(e.g., 'A', 'B', 'C', 'D', 'E', etc.)."
        )
    )


__all__ = ["AnswerLevelResponseOutput", "ResponseOutput", "SweResponseOutput"]
