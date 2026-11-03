"""GPT-OSS (openai/gpt-oss-120b) backend.

* Rendering: the model's own HF chat template (system -> developer
  "# Instructions", user), ``reasoning_effort`` from the model config
  (default ``medium``), and the
  template's ``strftime_now("%Y-%m-%d")`` replaced by the configured
  ``fixed_date`` so that prompts (and therefore cache keys) do not change from
  day to day.
* Structured output: the ``gptoss_structured`` reasoning-parser plugin enables
  the JSON grammar only after ``<|channel|>final<|message|>``; without it the
  grammar would force JSON from the first token and the model would not reason.
* Parsing: Harmony (``openai_harmony``) over the generated token ids; the
  ``analysis`` channel is the reasoning, the ``final`` channel the content.
* Sampling: greedy (temperature from the config, default 0); the retry attempt
  samples at ``retry.gptoss_retry_temperature`` with the request's seed.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from .parsers import GPTOSS_STRUCTURED
from .pool import Attempt, VLLMBackend

log = logging.getLogger(__name__)

DEFAULT_REASONING_EFFORT = "medium"
DATE_CALL = 'strftime_now("%Y-%m-%d")'
FINAL_MARKER = "<|channel|>final<|message|>"


def patch_chat_template(template: str, fixed_date: str | None) -> str:
    """Replace the template's ``strftime_now(...)`` date stamp by a literal."""
    if not fixed_date:
        return template
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", fixed_date):
        raise ValueError(f"fixed_date must be YYYY-MM-DD, got {fixed_date!r}")
    if DATE_CALL not in template:
        raise ValueError("GPT-OSS chat template has no strftime_now date stamp to patch")
    return template.replace(DATE_CALL, f'"{fixed_date}"')


def split_harmony(token_ids: list[int], encoding: Any, tokenizer: Any = None) -> tuple[str, str, int]:
    """(reasoning, content, reasoning_tokens) of a GPT-OSS completion.

    Uses Harmony's streamable parser so that truncated output (no ``<|return|>``)
    still yields the partial channels.  Falls back to text markers if Harmony
    rejects the token stream.
    """
    from openai_harmony import Role, StreamableParser

    reasoning_parts: list[str] = []
    final_parts: list[str] = []
    reasoning_tokens = 0
    try:
        parser = StreamableParser(encoding, role=Role.ASSISTANT)
        for i, tok in enumerate(token_ids):
            parser.process(tok)
            if parser.current_channel == "final" and reasoning_tokens == 0:
                reasoning_tokens = i + 1
        messages = list(parser.messages)
        for msg in messages:
            text = "".join(getattr(c, "text", "") for c in msg.content)
            if msg.channel == "final":
                final_parts.append(text)
            elif msg.channel == "analysis":
                reasoning_parts.append(text)
        # Partial (unterminated) message, e.g. finish_reason == "length".
        current = parser.current_content or ""
        if current:
            if parser.current_channel == "final":
                final_parts.append(current)
            elif parser.current_channel == "analysis":
                reasoning_parts.append(current)
        if reasoning_tokens == 0 and not final_parts:
            reasoning_tokens = len(token_ids)
        return "\n".join(reasoning_parts), (final_parts[-1] if final_parts else ""), reasoning_tokens
    except Exception as exc:  # malformed Harmony stream: decode text and split on markers
        if tokenizer is None:
            raise
        log.debug("Harmony parse failed (%s); falling back to text markers", exc)
        text = tokenizer.decode(token_ids, skip_special_tokens=False)
        head, sep, tail = text.partition(FINAL_MARKER)
        reasoning = re.sub(r"<\|[^|]*\|>", " ", head if sep else text)
        reasoning = re.sub(r"\banalysis\b", "", reasoning, count=1).strip()
        content = re.sub(r"<\|(return|end|call)\|>.*$", "", tail, flags=re.S) if sep else ""
        return reasoning, content, len(token_ids)


class GptOssBackend(VLLMBackend):
    reasoning_parser = GPTOSS_STRUCTURED

    def __init__(self, name: str, cfg: dict[str, Any], engine_cfg: dict[str, Any] | None = None) -> None:
        super().__init__(name, cfg, engine_cfg)
        self.reasoning_effort = str(self.cfg.get("reasoning_effort", DEFAULT_REASONING_EFFORT))
        self.fixed_date = self.cfg.get("fixed_date")
        self.chat_template: str | None = None
        self._harmony: Any = None

    def load_tokenizer(self) -> None:
        super().load_tokenizer()
        self.chat_template = patch_chat_template(self.tokenizer.chat_template, self.fixed_date)
        from openai_harmony import HarmonyEncodingName, load_harmony_encoding

        self._harmony = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

    def render_params(self) -> dict[str, Any]:
        return {"reasoning_effort": self.reasoning_effort, "fixed_date": self.fixed_date}

    def render_text(self, system: str, user: str) -> str:
        return self.tokenizer.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            add_generation_prompt=True,
            tokenize=False,
            chat_template=self.chat_template,
            reasoning_effort=self.reasoning_effort,
        )

    def _render(self, system: str, user: str) -> list[int]:
        ids = self.tokenizer.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=False,
            chat_template=self.chat_template,
            reasoning_effort=self.reasoning_effort,
        )
        if isinstance(ids, dict):
            ids = ids["input_ids"]
        return [int(t) for t in ids]

    def attempt(self, index: int, base_seed: int, *, retry: dict[str, Any]) -> Attempt:
        if index == 1:
            temperature = float(self.cfg.get("temperature", 0.0))
        else:
            temperature = float(retry.get("gptoss_retry_temperature", 0.6))
        # Greedy needs no seed; the sampled retry uses the request's seed.
        seed = None if temperature == 0.0 else (base_seed + max(0, index - 2)) % (2**32)
        top_p = self.cfg.get("top_p") if temperature > 0 else None
        return Attempt(model=self.name, index=index, temperature=temperature, top_p=top_p, seed=seed)

    def chat_template_kwargs(self) -> dict[str, Any]:
        return {"reasoning_effort": self.reasoning_effort}

    def parse(self, token_ids: list[int]) -> tuple[str, str, int]:
        return split_harmony(token_ids, self._harmony, self.tokenizer)
