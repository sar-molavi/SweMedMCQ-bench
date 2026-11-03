"""vLLM generation with the answer-only prompt (either language): JSON schema ``AnswerLevelResponseOutput``.

Same flags as ``swemedmcq.common.vllm_runner`` (``--help``).
"""

from .schemas import AnswerLevelResponseOutput
from .vllm_runner import main, parse_args

if __name__ == "__main__":
    config = parse_args()
    config.structured_output = AnswerLevelResponseOutput
    main(config)
