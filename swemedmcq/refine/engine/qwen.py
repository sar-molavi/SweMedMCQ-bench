"""Qwen3.8-27B backend (blind judge and GPT-OSS fallback).

* Engine: ``language_model_only=True`` (the checkpoint is a vision-language
  ``Qwen3_5ForConditionalGeneration``; only the text tower is loaded).
* Rendering: HF chat template with ``enable_thinking=True`` and
  ``reasoning_effort`` from the model config (default ``xhigh``, the
  template's own default; ``medium``/``low`` add an effort instruction).
  The effort is always passed explicitly so that it is part of the cache key.
  The generation prompt ends with ``<think>\\n``.
* Structured output: the ``qwen3_8_thinking`` plugin keeps the JSON grammar off
  until ``</think>``; ``chat_template_kwargs`` are passed per request through
  ``reasoning_parser_kwargs`` so the parser knows thinking is on.
* Parsing: generated tokens before ``</think>`` are the reasoning, after it
  the content.
* Sampling: model-card defaults (temperature 1.0, top_p 0.95, top_k 20) with a
  seed derived from the request's ``seed_key``; attempt k uses seed + (k-1).
"""

from __future__ import annotations

from typing import Any

from .parsers import QWEN3_8_THINKING
from .pool import Attempt, VLLMBackend

DEFAULT_REASONING_EFFORT = "xhigh"
THINK_START = "<think>"
THINK_END = "</think>"


def split_think(token_ids: list[int], tokenizer: Any, start_id: int | None, end_id: int | None) -> tuple[str, str, int]:
    """(reasoning, content, reasoning_tokens) of a thinking-mode completion."""
    ids = list(token_ids)
    if start_id is not None and ids and ids[0] == start_id:
        ids = ids[1:]
    if end_id is not None and end_id in ids:
        cut = ids.index(end_id)
        reasoning = tokenizer.decode(ids[:cut], skip_special_tokens=True)
        content = tokenizer.decode(ids[cut + 1 :], skip_special_tokens=True)
        return reasoning.strip(), content.strip(), cut
    # No </think>: everything is (unfinished) reasoning.
    return tokenizer.decode(ids, skip_special_tokens=True).strip(), "", len(ids)


class QwenBackend(VLLMBackend):
    reasoning_parser = QWEN3_8_THINKING

    def __init__(self, name: str, cfg: dict[str, Any], engine_cfg: dict[str, Any] | None = None) -> None:
        super().__init__(name, cfg, engine_cfg)
        self.reasoning_effort = str(self.cfg.get("reasoning_effort", DEFAULT_REASONING_EFFORT))
        self.enable_thinking = bool(self.cfg.get("enable_thinking", True))
        self._think_ids: tuple[int | None, int | None] = (None, None)

    def load_tokenizer(self) -> None:
        super().load_tokenizer()
        vocab = self.tokenizer.get_vocab()
        self._think_ids = (vocab.get(THINK_START), vocab.get(THINK_END))
        if self._think_ids[1] is None:
            raise ValueError(f"{self.path}: tokenizer has no {THINK_END} token")

    def extra_engine_kwargs(self) -> dict[str, Any]:
        return {"language_model_only": bool(self.cfg.get("language_model_only", True))}

    def chat_template_kwargs(self) -> dict[str, Any]:
        return {"enable_thinking": self.enable_thinking, "reasoning_effort": self.reasoning_effort}

    def render_params(self) -> dict[str, Any]:
        return dict(self.chat_template_kwargs())

    def render_text(self, system: str, user: str) -> str:
        return self.tokenizer.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            add_generation_prompt=True,
            tokenize=False,
            **self.chat_template_kwargs(),
        )

    def _render(self, system: str, user: str) -> list[int]:
        ids = self.tokenizer.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=False,
            **self.chat_template_kwargs(),
        )
        if isinstance(ids, dict):
            ids = ids["input_ids"]
        return [int(t) for t in ids]

    def attempt(self, index: int, base_seed: int, *, retry: dict[str, Any]) -> Attempt:
        return Attempt(
            model=self.name,
            index=index,
            temperature=float(self.cfg.get("temperature", 1.0)),
            top_p=float(self.cfg.get("top_p", 0.95)),
            top_k=int(self.cfg.get("top_k", 20)),
            seed=(base_seed + index - 1) % (2**32),
        )

    def parse(self, token_ids: list[int]) -> tuple[str, str, int]:
        return split_think(token_ids, self.tokenizer, *self._think_ids)
