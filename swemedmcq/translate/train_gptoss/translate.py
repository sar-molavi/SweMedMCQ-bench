"""vLLM translation of the training split (GPT-OSS-120B).

The vLLM runner (``swemedmcq.common.vllm_runner``, same flags) with
``MedicalMCQTranslationOutput`` as JSON schema.
"""

from ..test_gpt54.schema import MedicalMCQTranslationOutput
from ...common.vllm_runner import parse_args, main

if __name__ == "__main__":
    config = parse_args()
    config.structured_output = MedicalMCQTranslationOutput
    main(config)
