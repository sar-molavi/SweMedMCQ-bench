"""vLLM reasoning-parser plugin for the EuroLLM students: the text up to ``</think>`` is reasoning, the JSON answer follows."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from vllm.entrypoints.openai.engine.protocol import DeltaMessage
from vllm.reasoning import ReasoningParser, ReasoningParserManager


class EuroLLMThinkingParser(ReasoningParser):
    """Parse ordinary-text thinking spans inside EuroLLM assistant turns.

    The native template ends assistant turns with <|im_end|>. The SFT
    preparer inserts <think> and </think> as text, not vocabulary additions.
    Decode generated IDs rather than assuming single-token delimiters.

    Training text is ``<think>\n{reasoning}\n</think>\n\n{answer}`` with the
    opening ``<think>\n`` in the (masked) prompt, so run with
    ``--assistant-prefix '<think>\n'``. The reasoning ends after the blank line
    that follows ``</think>``: starting the JSON grammar right at ``</think>``
    forces ``{`` where the model expects ``\n\n`` and can send it into an
    endless run of newlines inside the JSON.
    """

    start_token = "<think>"
    end_token = "</think>"
    reasoning_end = "</think>\n\n"
    _MARKER_WINDOW = 16  # tokens; reasoning_end is 10 characters

    def __init__(self, tokenizer, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)
        self.thinking_enabled = (kwargs.get("chat_template_kwargs") or {}).get(
            "enable_thinking", True
        )

    def _decode(self, ids: Sequence[int]) -> str:
        return self.model_tokenizer.decode(list(ids), skip_special_tokens=False)

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        return not self.thinking_enabled or self.reasoning_end in self._decode(input_ids)

    def is_reasoning_end_streaming(
        self, input_ids: Sequence[int], delta_ids: Iterable[int]
    ) -> bool:
        # input_ids is the whole sequence (prompt + output, ending with delta_ids).
        # Only the new tokens plus a few before them can complete the marker, so
        # decode a short tail instead of the whole sequence on every step.
        if not self.thinking_enabled:
            return True
        tail = len(list(delta_ids)) + self._MARKER_WINDOW
        return self.reasoning_end in self._decode(input_ids[-tail:])

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        if not self.thinking_enabled:
            return input_ids
        text = self._decode(input_ids)
        end = text.find(self.reasoning_end)
        if end < 0:
            return []
        boundary = end + len(self.reasoning_end)
        # Keep original IDs: re-encoding could change the JSON grammar prefix.
        for index in range(1, len(input_ids) + 1):
            if len(self._decode(input_ids[:index])) >= boundary:
                return input_ids[index:]
        return []

    def count_reasoning_tokens(self, token_ids: Sequence[int]) -> int:
        if not self.thinking_enabled:
            return 0
        return len(token_ids) - len(self.extract_content_ids(list(token_ids)))

    def extract_reasoning(self, model_output: str, request) -> tuple[str | None, str | None]:
        if not self.thinking_enabled:
            return None, model_output or None
        body = model_output.partition(self.start_token)
        body = body[2] if body[1] else body[0]
        reasoning, separator, content = body.partition(self.end_token)
        return reasoning or None, (content or None) if separator else None

    @staticmethod
    def _hold_partial_marker(text: str, marker: str) -> str:
        for size in range(min(len(text), len(marker) - 1), 0, -1):
            if text.endswith(marker[:size]):
                return text[:-size]
        return text

    def _visible_parts(self, text: str) -> tuple[str, str]:
        if not self.thinking_enabled:
            return "", text
        if self.start_token.startswith(text) and text != self.start_token:
            return "", ""
        body = text.partition(self.start_token)
        body = body[2] if body[1] else body[0]
        reasoning, separator, content = body.partition(self.end_token)
        if separator:
            return reasoning, content
        return self._hold_partial_marker(body, self.end_token), ""

    def extract_reasoning_streaming(
        self, previous_text: str, current_text: str, delta_text: str,
        previous_token_ids: Sequence[int], current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
        previous_reasoning, previous_content = self._visible_parts(previous_text)
        reasoning, content = self._visible_parts(current_text)
        reasoning_delta = reasoning[len(previous_reasoning):]
        content_delta = content[len(previous_content):]
        if not reasoning_delta and not content_delta:
            return None
        return DeltaMessage(reasoning=reasoning_delta or None, content=content_delta or None)

    # Compatibility with vLLM releases using the older method names.
    def extract_reasoning_content(self, model_output, request):
        return self.extract_reasoning(model_output, request)

    def extract_reasoning_content_streaming(self, *args, **kwargs):
        return self.extract_reasoning_streaming(*args, **kwargs)


ReasoningParserManager.register_module(["eurollm_thinking"], module=EuroLLMThinkingParser)
