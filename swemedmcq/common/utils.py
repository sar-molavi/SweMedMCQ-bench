"""Shared helpers: prompt formatting and extraction of the reasoning and the final answer from model outputs."""

from __future__ import annotations

import re
import json

# "gemma4" (Gemma-4 thought channel) is only used when asked for; "auto" tries the other three.
SUPPORTED_RESPONSE_FORMATS = ("auto", "gpt_oss", "qwen", "medgemma", "gemma4")


class PartialFormatDict(dict):
    def __missing__(self, key):
        return "{" + key + "}"


def extract_final_block_gpt_oss(text: str) -> str | None:
    """
    Return the substring contained in the <|channel|>final...<|message|> ... <|return|> block.

    Tolerates optional tokens (like <|constrain|>JSON) between 'final' and '<|message|>'.
    """
    # Allow optional intermediate tokens between <|channel|>final and <|message|>
    pattern_complete = (
        r"<\|channel\|>final(?:\s*<\|.*?\|>.*?)*\s*<\|message\|>(.*?)<\|return\|>"
    )
    match = re.search(pattern_complete, text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Fallback if <|return|> is missing
    pattern_partial = r"<\|channel\|>final(?:\s*<\|.*?\|>.*?)*\s*<\|message\|>(.*)"
    match = re.search(pattern_partial, text, re.DOTALL)
    if match:
        return match.group(1).strip()

    return None


def extract_reason_block(text: str) -> str | None:
    """
    Return the substring contained in the <|channel|>final<|message|> ... <|return|> block.

    Parameters
    ----------
    text : str
        The full LLM output containing reasoning + final blocks.

    Returns
    -------
    str or None
        The captured text of the final block, or None if no match is found.
    """
    pattern = r"<\|channel\|>analysis<\|message\|>(.*?)<\|end\|>"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else None


def extract_reasoning_trace(text: str, response_format: str = "auto") -> str | None:
    """Extract a visible reasoning trace for supported response formats."""

    if response_format not in SUPPORTED_RESPONSE_FORMATS:
        raise ValueError(
            f"Unsupported response_format={response_format!r}. "
            f"Expected one of {SUPPORTED_RESPONSE_FORMATS}."
        )

    stripped = strip_generation_suffix_tokens(text)
    formats = (
        ("gpt_oss", "qwen", "medgemma")
        if response_format == "auto"
        else (response_format,)
    )

    for current_format in formats:
        if current_format == "gpt_oss":
            reasoning = extract_reason_block(stripped)
        elif current_format == "qwen":
            reasoning, _ = split_qwen_reasoning_content(stripped)
        elif current_format == "gemma4":
            reasoning, _ = split_gemma4_reasoning_content(stripped)
        else:
            reasoning = None

        if reasoning:
            return reasoning

    return None


def split_qwen_reasoning_content(text: str) -> tuple[str | None, str | None]:
    """Split Qwen3/Qwen3.5 output into reasoning and final content.

    Qwen3.5 typically receives ``<think>`` in the prompt and only emits
    ``</think>`` in the generated output, so the generated prefix before
    ``</think>`` is the reasoning trace.
    """

    cleaned = strip_generation_suffix_tokens(text)
    if "<think>" in cleaned:
        cleaned = cleaned.split("<think>", 1)[1]
    if "</think>" not in cleaned:
        return None, cleaned.strip() or None
    reasoning, content = cleaned.split("</think>", 1)
    reasoning = reasoning.strip() or None
    content = content.strip() or None
    return reasoning, content


GEMMA4_CHANNEL_START = "<|channel>"
GEMMA4_CHANNEL_END = "<channel|>"
GEMMA4_THOUGHT_LABEL = "thought\n"


def split_gemma4_reasoning_content(text: str) -> tuple[str | None, str | None]:
    """Split Gemma-4 output into the thought channel and the final content.

    Thinking on: ``<|channel>thought\n...reasoning...<channel|>{json}<turn|>``. When the
    prompt already opens the channel (``<|channel>thought\n`` prefilled), the output
    starts inside the thought and only ``<channel|>`` appears. Without ``<channel|>``
    there is no closed thought block: (None, whole text), as for Qwen.
    """

    cleaned = strip_generation_suffix_tokens(text)
    while cleaned.endswith("<turn|>"):
        cleaned = strip_generation_suffix_tokens(cleaned[: -len("<turn|>")])
    if GEMMA4_CHANNEL_END not in cleaned:
        return None, cleaned.strip() or None
    reasoning, content = cleaned.split(GEMMA4_CHANNEL_END, 1)
    reasoning = reasoning.split(GEMMA4_CHANNEL_START, 1)[-1].lstrip()
    if reasoning.startswith(GEMMA4_THOUGHT_LABEL):
        reasoning = reasoning[len(GEMMA4_THOUGHT_LABEL):]
    return reasoning.strip() or None, content.strip() or None


def strip_generation_suffix_tokens(text: str) -> str:
    """Remove common trailing generation sentinel tokens from model outputs."""

    suffixes = ("<|return|>", "<|im_end|>", "<eos>", "</s>")
    cleaned = text.strip()
    changed = True
    while changed:
        changed = False
        for suffix in suffixes:
            if cleaned.endswith(suffix):
                cleaned = cleaned[: -len(suffix)].rstrip()
                changed = True
    return cleaned


def extract_final_answer_text(
    decoded_text: str, response_format: str = "auto"
) -> str | None:
    """Extract final free-text content from a decoded model response."""
    if response_format not in SUPPORTED_RESPONSE_FORMATS:
        raise ValueError(
            f"Unsupported response_format={response_format!r}. "
            f"Expected one of {SUPPORTED_RESPONSE_FORMATS}."
        )

    stripped = strip_generation_suffix_tokens(decoded_text)
    formats = (
        ("gpt_oss", "qwen", "medgemma")
        if response_format == "auto"
        else (response_format,)
    )

    for current_format in formats:
        if current_format == "gpt_oss":
            final_answer = extract_final_block_gpt_oss(stripped)
            if final_answer:
                return final_answer
        elif current_format == "qwen":
            _, final_answer = split_qwen_reasoning_content(stripped)
            if final_answer:
                return final_answer
        elif current_format == "gemma4":
            _, final_answer = split_gemma4_reasoning_content(stripped)
            if final_answer:
                return final_answer
        elif stripped:
            return stripped

    return stripped or None


def extract_braced_json(text: str) -> str | None:
    """Return the substring from the first '{' to the last '}' when present."""

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    return text[start : end + 1].strip()


def extract_last_braced_json(text: str) -> str | None:
    """Return the last balanced brace-delimited block in a text string.

    This is useful when the model emits reasoning first and the final answer
    appears as the last JSON object in the response. The scan is brace-aware,
    so nested JSON objects are handled correctly.
    """

    last_result = None
    last_start = None
    depth = 0
    in_string = False
    escaped = False

    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
            continue

        if char == "{":
            if depth == 0:
                last_start = index
            depth += 1
            continue

        if char == "}" and depth > 0:
            depth -= 1
            if depth == 0 and last_start is not None:
                candidate = text[last_start : index + 1].strip()
                if candidate:
                    # Keep scanning so later top-level objects override earlier ones.
                    last_result = candidate
                last_start = None
                continue

    return last_result


def _append_candidate(
    candidates: list[str], seen: set[str], candidate: str | None
) -> None:
    """Add a candidate and useful normalized variants while preserving order."""

    if candidate is None:
        return

    cleaned = strip_generation_suffix_tokens(candidate.strip())
    if not cleaned:
        return

    queue = [cleaned]
    if '\\"' in cleaned:
        queue.append(cleaned.replace('\\"', '"'))

    while queue:
        current = queue.pop(0).strip()
        if not current or current in seen:
            continue

        seen.add(current)
        candidates.append(current)

        braced = extract_braced_json(current)
        if braced and braced not in seen:
            queue.append(braced)

        try:
            parsed = json.loads(current)
        except Exception:
            continue

        if isinstance(parsed, str):
            if parsed not in seen:
                queue.append(parsed)
        elif isinstance(parsed, (dict, list)):
            normalized = json.dumps(parsed, ensure_ascii=False)
            if normalized not in seen:
                queue.append(normalized)


def parse_json_payload(candidate: str, model) -> dict | None:
    """Parse a JSON candidate, unwrapping one nested JSON-string layer if needed."""

    try:
        parsed = json.loads(candidate)
    except Exception:
        return None

    if isinstance(parsed, str):
        try:
            parsed = json.loads(parsed)
        except Exception:
            return None

    try:
        return model.model_validate(parsed).model_dump()
    except Exception:
        return None


def build_structured_output_candidates(
    text: str, response_format: str = "auto"
) -> list[str]:
    """Build plausible JSON payload candidates for different model output styles."""

    if response_format not in SUPPORTED_RESPONSE_FORMATS:
        raise ValueError(
            f"Unsupported response_format={response_format!r}. "
            f"Expected one of {SUPPORTED_RESPONSE_FORMATS}."
        )

    candidates: list[str] = []
    seen: set[str] = set()
    stripped = strip_generation_suffix_tokens(text)

    formats = (
        ("gpt_oss", "qwen", "medgemma")
        if response_format == "auto"
        else (response_format,)
    )

    for current_format in formats:
        if current_format == "gpt_oss":
            final_block = extract_final_block_gpt_oss(stripped)
            _append_candidate(
                candidates, seen, extract_last_braced_json(final_block or stripped)
            )
            _append_candidate(candidates, seen, final_block)
            _append_candidate(candidates, seen, stripped)
        elif current_format == "qwen":
            _, qwen_content = split_qwen_reasoning_content(stripped)
            _append_candidate(candidates, seen, extract_last_braced_json(qwen_content))
            _append_candidate(candidates, seen, qwen_content)
            _append_candidate(candidates, seen, stripped)
        elif current_format == "gemma4":
            _, gemma_content = split_gemma4_reasoning_content(stripped)
            _append_candidate(candidates, seen, extract_last_braced_json(gemma_content or ""))
            _append_candidate(candidates, seen, gemma_content)
            _append_candidate(candidates, seen, stripped)
        elif current_format == "medgemma":
            _append_candidate(candidates, seen, extract_last_braced_json(stripped))
            _append_candidate(candidates, seen, stripped)

    return candidates
