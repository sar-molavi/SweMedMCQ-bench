"""vLLM reasoning-parser plugin ``gptoss_structured`` for the refinement loop (JSON only in GPT-OSS's final channel)."""

# Loaded by vLLM as a reasoning-parser plugin
# (structured_outputs_config.reasoning_parser_plugin); it registers the parser
# "gptoss_structured": the JSON grammar is enabled only after
# "<|channel|>final<|message|>", so GPT-OSS can reason freely in the analysis
# channel first.
from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

from vllm.reasoning import ReasoningParser, ReasoningParserManager

if TYPE_CHECKING:
    from vllm.entrypoints.generate.base.protocol import DeltaMessage
    from vllm.entrypoints.openai.chat_completion.protocol import (
        ChatCompletionRequest,
    )
    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
    from vllm.tokenizers import TokenizerLike


PARSER_NAME = "gptoss_structured"

ASSISTANT_START_MARKER = "<|start|>assistant"
FINAL_CHANNEL_MARKER = "<|channel|>final<|message|>"


class GptOssStructuredReasoningParser(ReasoningParser):
    """Detect the GPT-OSS Harmony reasoning-to-final boundary.

    This parser is intended for vLLM structured decoding with GPT-OSS models.

    GPT-OSS uses the Harmony message format, where assistant reasoning and final
    output are separated by channels rather than conventional reasoning tokens
    such as ``<think>`` and ``</think>``.

    A typical generated assistant turn has the form::

        <|start|>assistant<|channel|>analysis<|message|>
        ...
        <|end|>
        <|start|>assistant<|channel|>final<|message|>
        ...
        <|return|>

    For structured generation, the desired behavior is:

    - During ``analysis``: structured-output grammar is disabled.
    - After ``<|channel|>final<|message|>``: structured-output grammar is
      enabled.

    This class only detects that transition. It does not parse GPT-OSS
    reasoning, final content, commentary, or tool calls. Those remain the
    responsibility of vLLM's Harmony response parser.
    """

    def __init__(
        self,
        tokenizer: "TokenizerLike",
        *args,
        **kwargs,
    ) -> None:
        """Initialize the structured-output boundary parser.

        Args:
            tokenizer: Tokenizer used by the GPT-OSS model.
            *args: Additional positional arguments forwarded to
                :class:`ReasoningParser`.
            **kwargs: Additional keyword arguments forwarded to
                :class:`ReasoningParser`.

        Raises:
            ValueError: If one of the required Harmony markers cannot be
                tokenized.
        """
        super().__init__(tokenizer, *args, **kwargs)

        self._assistant_start_ids = self._encode_marker(
            ASSISTANT_START_MARKER
        )
        self._final_marker_ids = self._encode_marker(
            FINAL_CHANNEL_MARKER
        )

    def _encode_marker(self, marker: str) -> tuple[int, ...]:
        """Encode a Harmony marker into its model token-ID sequence.

        Harmony markers may span multiple tokenizer tokens, so boundary
        detection must operate on token subsequences rather than assuming a
        single token ID.

        Args:
            marker: Harmony marker string to encode.

        Returns:
            Token IDs representing ``marker``.

        Raises:
            ValueError: If the tokenizer produces no token IDs.
        """
        token_ids = tuple(
            self.model_tokenizer.encode(
                marker,
                add_special_tokens=False,
            )
        )

        if not token_ids:
            raise ValueError(
                "Tokenizer produced no token IDs for "
                f"Harmony marker {marker!r}."
            )

        return token_ids

    @staticmethod
    def _find_subsequence(
        sequence: Sequence[int],
        subsequence: Sequence[int],
        *,
        start: int = 0,
    ) -> int:
        """Find the first occurrence of a token subsequence.

        Args:
            sequence: Token sequence to search.
            subsequence: Token sequence to locate.
            start: Minimum index at which a match may begin.

        Returns:
            Index of the first matching subsequence at or after ``start``, or
            ``-1`` when no match exists.
        """
        size = len(subsequence)

        if size == 0:
            return max(0, start)

        last_start = len(sequence) - size

        for idx in range(max(0, start), last_start + 1):
            if all(
                sequence[idx + offset] == token_id
                for offset, token_id in enumerate(subsequence)
            ):
                return idx

        return -1

    @staticmethod
    def _rfind_subsequence(
        sequence: Sequence[int],
        subsequence: Sequence[int],
    ) -> int:
        """Find the last occurrence of a token subsequence.

        Args:
            sequence: Token sequence to search.
            subsequence: Token sequence to locate.

        Returns:
            Index of the final matching subsequence, or ``-1`` when no match
            exists.
        """
        size = len(subsequence)

        if size == 0:
            return len(sequence)

        for idx in range(len(sequence) - size, -1, -1):
            if all(
                sequence[idx + offset] == token_id
                for offset, token_id in enumerate(subsequence)
            ):
                return idx

        return -1

    def is_reasoning_end(
        self,
        input_ids: Sequence[int],
    ) -> bool:
        """Determine whether the current assistant turn has entered ``final``.

        This check is used when initializing structured decoding.

        The full prompt may contain completed assistant turns from earlier
        conversation history, including previous ``final`` channel markers.
        Searching the entire prompt for any final marker would therefore
        produce false positives.

        Instead, only tokens following the most recent
        ``<|start|>assistant`` marker are inspected.

        Args:
            input_ids: Prompt and/or generated token IDs available when the
                reasoning state is initialized.

        Returns:
            ``True`` if the latest assistant turn already contains
            ``<|channel|>final<|message|>``; otherwise ``False``.
        """
        latest_assistant_start = self._rfind_subsequence(
            input_ids,
            self._assistant_start_ids,
        )

        if latest_assistant_start < 0:
            return False

        search_from = (
            latest_assistant_start
            + len(self._assistant_start_ids)
        )

        return (
            self._find_subsequence(
                input_ids,
                self._final_marker_ids,
                start=search_from,
            )
            >= 0
        )

    def is_reasoning_end_streaming(
        self,
        input_ids: Sequence[int],
        delta_ids: Iterable[int],
    ) -> bool:
        """Detect entry into the GPT-OSS ``final`` channel while streaming.

        The final-channel marker may span multiple decode iterations. The
        search therefore includes the current token delta plus enough tokens
        immediately preceding it to detect a marker split across iterations.

        Restricting the search to this small window also prevents final markers
        from earlier turns in the conversation from incorrectly enabling the
        structured-output grammar.

        Args:
            input_ids: Complete token sequence available after the current
                generation update.
            delta_ids: Token IDs generated during the current update.

        Returns:
            ``True`` if the current update completes or contains
            ``<|channel|>final<|message|>``; otherwise ``False``.
        """
        delta = tuple(delta_ids)

        if not delta:
            return False

        carry = len(self._final_marker_ids) - 1

        window_start = max(
            0,
            len(input_ids) - len(delta) - carry,
        )
        window = input_ids[window_start:]

        return (
            self._find_subsequence(
                window,
                self._final_marker_ids,
            )
            >= 0
        )

    def extract_content_ids(
        self,
        input_ids: list[int],
    ) -> list[int]:
        """Reject direct extraction of GPT-OSS content tokens.

        This parser only controls the structured-output reasoning boundary.
        GPT-OSS/Harmony response parsing must be performed by vLLM's Harmony
        parser.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(
            f"{PARSER_NAME} only detects the structured-output reasoning "
            "boundary. Use vLLM's Harmony parser for GPT-OSS response parsing."
        )

    def extract_reasoning(
        self,
        model_output: str,
        request: "ChatCompletionRequest | ResponsesRequest",
    ) -> tuple[str | None, str | None]:
        """Reject direct reasoning/content extraction.

        GPT-OSS reasoning and final output are represented by Harmony channels.
        This parser only detects when structured decoding should become active.

        Args:
            model_output: Raw decoded model output.
            request: Request associated with the generation.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(
            f"{PARSER_NAME} only detects the structured-output reasoning "
            "boundary. Use vLLM's Harmony parser for GPT-OSS response parsing."
        )

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> "DeltaMessage | None":
        """Reject direct streaming Harmony response parsing.

        This parser does not transform GPT-OSS streaming output into reasoning
        and content deltas. vLLM's Harmony response parser must perform that
        operation.

        Args:
            previous_text: Decoded output before the current update.
            current_text: Decoded output including the current update.
            delta_text: Text introduced by the current update.
            previous_token_ids: Token IDs generated before the update.
            current_token_ids: Token IDs generated through the update.
            delta_token_ids: Token IDs introduced by the update.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(
            f"{PARSER_NAME} only detects the structured-output reasoning "
            "boundary. Use vLLM's Harmony parser for GPT-OSS response parsing."
        )


ReasoningParserManager.register_module(
    name=PARSER_NAME,
    module=GptOssStructuredReasoningParser,
)