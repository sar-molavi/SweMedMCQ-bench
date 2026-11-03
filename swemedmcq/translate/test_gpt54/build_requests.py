"""Prompt set -> OpenAI ``/v1/responses`` batch requests with the translation schema.

Same flags as ``swemedmcq.translate.openai_batch.prepare_batch_requests`` (``--help``).
"""

from __future__ import annotations

from .schema import MedicalMCQTranslationOutput

from ..openai_batch.prepare_batch_requests import main, parse_args


if __name__ == "__main__":
    config = parse_args()
    config.response_output = MedicalMCQTranslationOutput
    main(config)
