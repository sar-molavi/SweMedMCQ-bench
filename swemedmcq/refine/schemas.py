"""JSON schemas for every LLM step (enforced by constrained decoding).

The schemas are small on purpose: they are not pasted into prompts; the prompt
describes the fields in prose and decoding enforces the structure.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

FIELD_PATTERN = r"^(question|option:[A-Z])$"

CategoryOut = Literal[
    "accuracy/mistranslation",
    "accuracy/omission",
    "accuracy/addition",
    "terminology",
    "fluency/grammar",
    "fluency/spelling",
    "source_issue",
]


class _S(BaseModel):
    model_config = ConfigDict(extra="forbid")


# --- DETECT --------------------------------------------------------------------
class DetectedError(_S):
    field: str = Field(pattern=FIELD_PATTERN)
    sv_quote: str = Field(min_length=1, max_length=300)
    en_quote: str = Field(max_length=300)
    category: CategoryOut
    severity: Literal["minor", "major", "critical"]
    explanation: str = Field(max_length=400)


class DetectOutput(_S):
    errors: list[DetectedError] = Field(max_length=12)


# specialised detectors (one per pass): the category set is narrowed by the schema
class MeaningError(_S):
    field: str = Field(pattern=FIELD_PATTERN)
    sv_quote: str = Field(min_length=1, max_length=300)
    en_quote: str = Field(max_length=300)
    category: Literal["accuracy/mistranslation", "accuracy/omission", "accuracy/addition", "source_issue"]
    severity: Literal["minor", "major", "critical"]
    explanation: str = Field(max_length=400)


class DetectMeaningOutput(_S):
    errors: list[MeaningError] = Field(max_length=12)


class TerminologyError(_S):
    field: str = Field(pattern=FIELD_PATTERN)
    sv_quote: str = Field(min_length=1, max_length=300)
    en_quote: str = Field(max_length=300)
    category: Literal["terminology"]
    severity: Literal["minor", "major", "critical"]
    explanation: str = Field(max_length=400)


class DetectTerminologyOutput(_S):
    errors: list[TerminologyError] = Field(max_length=12)


class LanguageError(_S):
    field: str = Field(pattern=FIELD_PATTERN)
    sv_quote: str = Field(min_length=1, max_length=300)
    en_quote: str = Field(max_length=300)
    category: Literal["fluency/grammar", "fluency/spelling"]
    severity: Literal["major", "critical"]  # minor language issues are not asked for
    explanation: str = Field(max_length=400)


class DetectLanguageOutput(_S):
    errors: list[LanguageError] = Field(max_length=12)


# --- VERIFY --------------------------------------------------------------------
class VerifyDecision(_S):
    id: str
    verdict: Literal["confirm", "dismiss"]
    reason: str = Field(max_length=300)
    certainty: Literal["low", "medium", "high"]


class VerifyOutput(_S):
    decisions: list[VerifyDecision]


# --- REPAIR --------------------------------------------------------------------
class RepairEdit(_S):
    finding_ids: list[str] = Field(min_length=1)
    field: str = Field(pattern=FIELD_PATTERN)
    old: str = Field(min_length=1, max_length=400)
    new: str = Field(max_length=600)


class NoChange(_S):
    finding_id: str
    reason: str = Field(max_length=300)


class RepairOutput(_S):
    edits: list[RepairEdit]
    no_change_needed: list[NoChange] = Field(default_factory=list)


# --- INTEGRATE (DRUG / TERM-A) ----------------------------------------------------
class IntegrateEdit(_S):
    old: str = Field(min_length=1, max_length=400)
    new: str = Field(max_length=600)


class IntegrateOutput(_S):
    reason: str = Field(max_length=300)
    edits: list[IntegrateEdit] = Field(max_length=8)


# --- RETRANSLATE -----------------------------------------------------------------
class RetranslateOutput(_S):
    text: str = Field(min_length=1)


# --- JUDGES --------------------------------------------------------------------
# `reason` first (it also serves as the analysis field of Gemma's schema-thinking fallback),
# then the self-reported `certainty` (a low-certainty vote counts as "equal"), then the verdict.
class PairwiseOutput(_S):
    reason: str = Field(max_length=500)
    certainty: Literal["low", "medium", "high"]
    preferred: Literal["A", "B", "equal", "both_wrong"]


class NonInferiorityOutput(_S):
    reason: str = Field(max_length=500)
    certainty: Literal["low", "medium", "high"]
    introduces_error: bool


SCHEMAS: dict[str, type[BaseModel]] = {
    "detect": DetectOutput,
    "detect_meaning": DetectMeaningOutput,
    "detect_terminology": DetectTerminologyOutput,
    "detect_language": DetectLanguageOutput,
    "integrate": IntegrateOutput,
    "verify": VerifyOutput,
    "repair": RepairOutput,
    "retranslate": RetranslateOutput,
    "pairwise": PairwiseOutput,
    "noninferiority": NonInferiorityOutput,
}
