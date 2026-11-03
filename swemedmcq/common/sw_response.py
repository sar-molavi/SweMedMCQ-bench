"""vLLM generation with the Swedish explanation (trace) prompt: JSON schema ``SweResponseOutput``.

Same flags as ``swemedmcq.common.vllm_runner`` (``--help``).
"""

from .schemas import SweResponseOutput
from .vllm_runner import main, parse_args

if __name__ == "__main__":
    config = parse_args()
    config.structured_output = SweResponseOutput
    main(config)
