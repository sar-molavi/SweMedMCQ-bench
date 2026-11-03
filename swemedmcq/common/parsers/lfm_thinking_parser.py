"""vLLM reasoning-parser plugin for the LFM2.5 students (``<think> ... </think>`` opened by the chat template)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from vllm.entrypoints.openai.engine.protocol import DeltaMessage
from vllm.reasoning import ReasoningParserManager
from vllm.reasoning.basic_parsers import BaseThinkingReasoningParser

if TYPE_CHECKING:
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
    from vllm.tokenizers import TokenizerLike


class LFMThinkingParser(BaseThinkingReasoningParser):
    """Reasoning parser for LFM checkpoints with a custom <think> chat template."""

    def __init__(self, tokenizer: "TokenizerLike", *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)
        chat_kwargs = kwargs.get("chat_template_kwargs", {}) or {}
        self.thinking_enabled = chat_kwargs.get("enable_thinking", True)

    @property
    def start_token(self) -> str:
        return "<think>"

    @property
    def end_token(self) -> str:
        return "</think>"

    def extract_reasoning(
        self,
        model_output: str,
        request: "ChatCompletionRequest | ResponsesRequest",
    ) -> tuple[str | None, str | None]:
        model_output_parts = model_output.partition(self.start_token)
        model_output = (
            model_output_parts[2] if model_output_parts[1] else model_output_parts[0]
        )

        if self.end_token not in model_output:
            if not self.thinking_enabled:
                return None, model_output
            return model_output, None

        reasoning, _, content = model_output.partition(self.end_token)
        return reasoning, content or None

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
        if self.start_token_id in delta_token_ids:
            start_idx = delta_text.find(self.start_token)
            if start_idx >= 0:
                delta_text = delta_text[start_idx + len(self.start_token) :]

        if self.end_token_id in delta_token_ids:
            end_idx = delta_text.find(self.end_token)
            if end_idx >= 0:
                reasoning = delta_text[:end_idx]
                content = delta_text[end_idx + len(self.end_token) :]
                if not reasoning and not content:
                    return None
                return DeltaMessage(
                    reasoning=reasoning or None,
                    content=content or None,
                )
            return None

        if not delta_text:
            return None
        if self.end_token_id in previous_token_ids:
            return DeltaMessage(content=delta_text)
        return DeltaMessage(reasoning=delta_text)


ReasoningParserManager.register_module(
    ["lfm_thinking"],
    module=LFMThinkingParser,
)
