"""GPT-OSS (Harmony) reasoning parser for JSON output after the analysis channel.

vLLM 0.28's built-in ``openai_gptoss`` parser reports the reasoning as ended
from the first token (its chat server instead wraps the JSON schema in a
Harmony structural tag). With a plain JSON schema, as the batch runner sends,
that forces JSON at the first token and suppresses the analysis channel.

This parser restores the behaviour of earlier vLLM releases (used for the first
benchmark runs): the reasoning ends at the final-channel header
``<|channel|>final[ <|constrain|>json]<|message|>``, so the analysis channel is
free text and only the final message is JSON. A prompt that already ends with
that header (the "reasoning off" token sets) starts with JSON directly.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from vllm.reasoning import ReasoningParser, ReasoningParserManager

# Longest header we accept between <|channel|> and <|message|>
# ("final <|constrain|>json" is 4 tokens).
_MAX_HEADER = 16


class GptOssFinalChannelParser(ReasoningParser):
    def __init__(self, tokenizer, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)
        self.channel_id = self.vocab["<|channel|>"]
        self.message_id = self.vocab["<|message|>"]
        self.final_ids = self.model_tokenizer.encode("final", add_special_tokens=False)

    def _final_header_end(self, ids: Sequence[int]) -> int | None:
        """Index of the <|message|> closing the last channel header if it is the final channel."""
        for start in range(len(ids) - 1, -1, -1):
            if ids[start] != self.channel_id:
                continue
            header = list(ids[start + 1 : start + 1 + _MAX_HEADER])
            if header[: len(self.final_ids)] != self.final_ids or self.message_id not in header:
                return None
            return start + 1 + header.index(self.message_id)
        return None

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        return self._final_header_end(input_ids) is not None

    def is_reasoning_end_streaming(self, input_ids: Sequence[int], delta_ids: Iterable[int]) -> bool:
        # Only a new <|message|> can end the reasoning; look at a short tail, not the whole sequence.
        delta = list(delta_ids)
        if self.message_id not in delta:
            return False
        tail = list(input_ids[-(_MAX_HEADER + 1 + len(delta)) :])
        return self._final_header_end(tail) is not None

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        end = self._final_header_end(input_ids)
        return [] if end is None else input_ids[end + 1 :]

    def extract_reasoning(self, model_output: str, request) -> tuple[str | None, str | None]:
        reasoning, sep, rest = model_output.rpartition("<|channel|>final")
        if not sep:
            return model_output or None, None
        content = rest.partition("<|message|>")[2]
        return reasoning or None, content or None

    def extract_reasoning_streaming(self, *args, **kwargs):
        raise NotImplementedError("Only used for structured-output gating in batch runs.")


ReasoningParserManager.register_module(["gptoss_final"], module=GptOssFinalChannelParser)
