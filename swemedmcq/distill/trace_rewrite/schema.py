"""Teacher output schema for Swedish trace rewrite.

The first rewrite scenario asks the teacher for only one field: the replacement
Swedish reasoning trace. The existing SFT row keeps its original answer JSON,
so answer labels and explanations can be validated/combined outside this schema.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ConservativeTraceRewriteOutput(BaseModel):
    """Strict teacher JSON for the conservative Swedish rewrite variant.

    Extra fields are forbidden so the teacher cannot silently return an answer
    label or explanation in this phase. The dataset-combine step can insert this
    reasoning back into the original row.
    """

    model_config = ConfigDict(extra="forbid")

    reasoning: str = Field(
        min_length=1,
        description=(
            "A natural Swedish reasoning trace grounded in the Swedish question, "
            "the source English reasoning, and the existing Swedish explanation."
        ),
    )

    @field_validator("reasoning")
    @classmethod
    def _strip_non_empty(cls, value: str) -> str:
        """Normalize whitespace and reject blank reasoning strings."""

        stripped = value.strip()
        if not stripped:
            raise ValueError("field must not be empty")
        return stripped


TeacherTraceRewriteOutput = ConservativeTraceRewriteOutput


__all__ = [
    "ConservativeTraceRewriteOutput",
    "TeacherTraceRewriteOutput",
]
