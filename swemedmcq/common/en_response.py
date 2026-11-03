"""vLLM generation with the English explanation (trace) prompt: JSON schema ``ResponseOutput``.

Same flags as ``swemedmcq.common.vllm_runner`` (``--help``).
"""

from .schemas import ResponseOutput
from .vllm_runner import main, parse_args

if __name__ == "__main__":
    config = parse_args()
    config.structured_output = ResponseOutput
    main(config)
