"""vLLM wrapper for trace-rewrite generation."""

from __future__ import annotations

from ...common.vllm_runner import main, parse_args
from .schema import ConservativeTraceRewriteOutput


if __name__ == "__main__":
    config = parse_args()
    config.structured_output = ConservativeTraceRewriteOutput
    main(config)
