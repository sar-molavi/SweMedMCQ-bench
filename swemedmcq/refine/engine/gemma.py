"""Gemma-4-31B-it (google/gemma-4-31B-it) backend (optional third judge).

* Engine: ``language_model_only=True`` (the checkpoint is a vision-language
  ``Gemma4ForConditionalGeneration``; only the text tower is loaded).
* Structured output: vLLM's *built-in* ``gemma4`` reasoning parser (no plugin
  file).  It keeps the JSON grammar off while the prompt+output tail is inside
  an open thought channel ``<|channel>`` and switches it on after the closing
  ``<channel|>``.  It reads ``enable_thinking`` from the per-request
  ``chat_template_kwargs`` (passed through ``reasoning_parser_kwargs``).

Two thinking modes (model config key ``thinking``):

``native`` (default)
    Rendering with ``enable_thinking=True`` puts ``<|think|>`` in the system
    turn; the generation prompt ends with ``<|turn>model\\n``.  We then append
    ``<|channel>thought\\n`` (ids 100, 45518, 107) ourselves, so that the
    thought block is *always* open: without that prefill the model may answer
    directly, the parser never sees ``<|channel>`` ... ``<channel|>`` and the
    grammar would never be switched on (the parser's backward scan stops at
    ``<|turn>`` and reports "still reasoning").  Generated tokens before
    ``<channel|>`` (101) are the reasoning, tokens after it the content.  If
    ``<channel|>`` never appears the output is reasoning only, the content is
    empty and the pool treats the attempt as failed (retry per policy).

``schema`` (fallback)
    ``enable_thinking=False``: the template itself closes an empty thought
    block (``<|channel>thought\\n<channel|>``), so the grammar is on from the
    first generated token and nothing is prefilled.  The model does not think;
    the *caller's* schema must therefore put a free-text field (``analysis``)
    first so that the model reasons inside the JSON before the verdict.  The
    backend reports that field as the reasoning when it is present.  Only the
    backend switches mode; prompts and schemas are the caller's business.

Sampling: model-card defaults (temperature 1.0, top_p 0.95, top_k 64) with a
seed derived from the request's ``seed_key``; attempt k uses seed + (k-1).
Gemma has no reasoning-effort knob; ``reasoning_effort`` in its config is
ignored.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Sequence

from .pool import Attempt, VLLMBackend

GEMMA4_PARSER = "gemma4"  # vLLM built-in reasoning parser (vllm/reasoning/__init__.py)

CHANNEL_START = "<|channel>"  # id 100 in the Gemma-4 vocabulary
CHANNEL_END = "<channel|>"  # id 101
THOUGHT_PREFILL = "<|channel>thought\n"  # ids [100, 45518, 107]
THOUGHT_LABEL = "thought\n"
# <eos>, <turn|>, <|tool_response>: generation_config eos_token_id [1, 106, 50]
STOP_TOKENS = ("<eos>", "<turn|>", "<|tool_response>")

THINKING_MODES = ("native", "schema")


def with_prefill(prompt_ids: Sequence[int], prefill_ids: Sequence[int]) -> list[int]:
    """``prompt_ids`` with ``prefill_ids`` appended (unless already at the end)."""
    ids = [int(t) for t in prompt_ids]
    pre = [int(t) for t in prefill_ids]
    if pre and ids[-len(pre):] == pre:
        return ids
    return ids + pre


def split_gemma(
    token_ids: Sequence[int],
    decode: Callable[[list[int], bool], str],
    *,
    start_id: int,
    end_id: int,
    label_ids: Sequence[int] = (),
    stop_ids: Sequence[int] = (),
    thinking: bool = True,
) -> tuple[str, str, int]:
    """(reasoning, content, reasoning_tokens) of a Gemma-4 completion.

    ``decode(ids, skip_special_tokens)`` turns ids into text.  The generated ids
    of a prefilled prompt start inside the thought block; a repeated
    ``<|channel>`` and/or the ``thought\\n`` label at the start are dropped.
    Everything before ``<channel|>`` is reasoning, everything after it is the
    content (trailing stop tokens removed, other special tokens kept so that a
    leaked channel marker is visible and fails JSON validation).

    No ``<channel|>``: with ``thinking`` the whole output is unfinished
    reasoning and the content is empty (=> the attempt fails); without
    ``thinking`` (schema mode, the template already closed the block) the
    whole output is content.
    """
    ids = [int(t) for t in token_ids]
    stops = {int(t) for t in stop_ids}
    while ids and ids[-1] in stops:
        ids.pop()
    if end_id in ids:
        cut = ids.index(end_id)
        head, tail = ids[:cut], ids[cut + 1 :]
    elif thinking:
        head, tail = ids, []
        cut = len(ids)
    else:
        return "", decode(ids, False).strip(), 0
    if head and head[0] == start_id:
        head = head[1:]
    label = [int(t) for t in label_ids]
    if label and head[: len(label)] == label:
        head = head[len(label) :]
    reasoning = decode(head, True)
    if reasoning.startswith(THOUGHT_LABEL):  # label tokenised differently from ``label_ids``
        reasoning = reasoning[len(THOUGHT_LABEL) :]
    reasoning = reasoning.strip()
    content = decode(tail, False).strip() if tail else ""
    return reasoning, content, cut


def analysis_from_content(content: str, field: str = "analysis") -> str:
    """The ``field`` string of a JSON object in ``content`` ('' if absent)."""
    try:
        obj = json.loads(content)
    except (TypeError, ValueError):
        return ""
    value = obj.get(field) if isinstance(obj, dict) else None
    return value.strip() if isinstance(value, str) else ""


class GemmaBackend(VLLMBackend):
    reasoning_parser = GEMMA4_PARSER

    def __init__(self, name: str, cfg: dict[str, Any], engine_cfg: dict[str, Any] | None = None) -> None:
        super().__init__(name, cfg, engine_cfg)
        self.thinking = str(self.cfg.get("thinking", "native"))
        if self.thinking not in THINKING_MODES:
            raise ValueError(f"gemma: thinking must be one of {THINKING_MODES}, got {self.thinking!r}")
        self.analysis_field = str(self.cfg.get("analysis_field", "analysis"))
        self._ids: dict[str, Any] = {}

    @property
    def enable_thinking(self) -> bool:
        return self.thinking == "native"

    def load_tokenizer(self) -> None:
        super().load_tokenizer()
        self.init_special_ids()

    def init_special_ids(self) -> None:
        """Channel / prefill / stop token ids from ``self.tokenizer``."""
        vocab = self.tokenizer.get_vocab()
        start, end = vocab.get(CHANNEL_START), vocab.get(CHANNEL_END)
        if start is None or end is None:
            raise ValueError(f"{self.path}: tokenizer has no {CHANNEL_START}/{CHANNEL_END} tokens")
        prefill = [int(t) for t in self.tokenizer.encode(THOUGHT_PREFILL, add_special_tokens=False)]
        if not prefill or prefill[0] != start:
            raise ValueError(f"{self.path}: {THOUGHT_PREFILL!r} does not tokenise to <|channel> + label: {prefill}")
        self._ids = {
            "start": int(start),
            "end": int(end),
            "prefill": prefill,
            "label": prefill[1:],
            "stop": [int(vocab[t]) for t in STOP_TOKENS if t in vocab],
        }

    def extra_engine_kwargs(self) -> dict[str, Any]:
        return {"language_model_only": bool(self.cfg.get("language_model_only", True))}

    def chat_template_kwargs(self) -> dict[str, Any]:
        return {"enable_thinking": self.enable_thinking}

    def render_params(self) -> dict[str, Any]:
        return {"enable_thinking": self.enable_thinking, "thinking": self.thinking}

    def _messages(self, system: str, user: str) -> list[dict[str, str]]:
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def render_text(self, system: str, user: str) -> str:
        text = self.tokenizer.apply_chat_template(
            self._messages(system, user), add_generation_prompt=True, tokenize=False, **self.chat_template_kwargs()
        )
        if self.enable_thinking and not text.endswith(THOUGHT_PREFILL):
            text += THOUGHT_PREFILL
        return text

    def _render(self, system: str, user: str) -> list[int]:
        ids = self.tokenizer.apply_chat_template(
            self._messages(system, user),
            add_generation_prompt=True,
            tokenize=True,
            return_dict=False,
            **self.chat_template_kwargs(),
        )
        if isinstance(ids, dict):
            ids = ids["input_ids"]
        ids = [int(t) for t in ids]
        return with_prefill(ids, self._ids["prefill"]) if self.enable_thinking else ids

    def attempt(self, index: int, base_seed: int, *, retry: dict[str, Any]) -> Attempt:
        return Attempt(
            model=self.name,
            index=index,
            temperature=float(self.cfg.get("temperature", 1.0)),
            top_p=float(self.cfg.get("top_p", 0.95)),
            top_k=int(self.cfg.get("top_k", 64)),
            seed=(base_seed + index - 1) % (2**32),
        )

    def _decode(self, ids: list[int], skip_special: bool) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=skip_special)

    def parse(self, token_ids: list[int]) -> tuple[str, str, int]:
        reasoning, content, n = split_gemma(
            token_ids,
            self._decode,
            start_id=self._ids["start"],
            end_id=self._ids["end"],
            label_ids=self._ids["label"],
            stop_ids=self._ids["stop"],
            thinking=self.enable_thinking,
        )
        if not self.enable_thinking and not reasoning:
            reasoning = analysis_from_content(content, self.analysis_field)
        return reasoning, content, n
