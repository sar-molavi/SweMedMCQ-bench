#!/usr/bin/env python
"""Pre-tokenise benchmark prompts ("reasoning off" runs, and GPT-OSS with a fixed date).

Recipes:

* GPT-OSS: chat template with ``reasoning_effort="disabled"`` plus an empty
  analysis channel, so the model starts directly in the final channel.
* Qwen3-4B-Thinking-2507: chat template plus a prefilled ``\\n</think>\\n``.

* GPT-OSS reasoning on (``gpt_oss_on``): the plain chat template (default
  reasoning effort), pre-tokenised only to fix the date below.

Qwen3.5/3.6 do not need this; they use ``--no-reasoning`` (enable_thinking=False).

GPT-OSS's chat template writes today's date into its system prompt
("Current date: ..."). ``--date`` pins it, so that every run of a configuration
(and both languages) sees the same system prompt (``date`` in configs/models.yaml).

The output is a pickle {prompt_id: [token ids]} that ``swemedmcq.common.*_response``
accepts as ``--input-path``.
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
from pathlib import Path

from transformers import AutoTokenizer

RECIPES = {
    "gpt_oss": "openai/gpt-oss-20b",  # same tokenizer and template as gpt-oss-120b
    "gpt_oss_on": "openai/gpt-oss-20b",
    "qwen3_4b_thinking": "Qwen/Qwen3-4B-Thinking-2507",
}


def load_prompts(path: Path) -> dict[str, list[dict[str, str]]]:
    prompts = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                prompts[row["prompt_id"]] = row["prompt"]
    return prompts


def format_prompts(recipe: str, prompts: dict[str, list], date: str | None = None) -> list[str]:
    tokenizer = AutoTokenizer.from_pretrained(RECIPES[recipe])
    if recipe == "gpt_oss_on":
        formatted = tokenizer.apply_chat_template(list(prompts.values()), add_generation_prompt=True, tokenize=False)
        suffix = ""
    elif recipe == "gpt_oss":
        formatted = tokenizer.apply_chat_template(
            list(prompts.values()),
            add_generation_prompt=True,
            tokenize=False,
            reasoning_effort="disabled",
        )
        suffix = "<|channel|>analysis<|message|><|end|><|start|>assistant<|channel|>final<|message|>"
    else:
        formatted = tokenizer.apply_chat_template(
            list(prompts.values()), add_generation_prompt=True, tokenize=False
        )
        suffix = "\n</think>\n"
    if recipe.startswith("gpt_oss"):
        if not date:
            raise SystemExit("GPT-OSS recipes need --date (the system prompt contains the current date).")
        formatted = [re.sub(r"Current date: \d{4}-\d{2}-\d{2}", f"Current date: {date}", t, count=1) for t in formatted]
        if not all(f"Current date: {date}" in t for t in formatted):
            raise SystemExit("Could not set the date in the GPT-OSS system prompt.")
    return tokenizer, [text + suffix for text in formatted]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", choices=sorted(RECIPES), required=True)
    parser.add_argument("--input", type=Path, required=True, help="Prompt JSONL (prompt_id, prompt).")
    parser.add_argument("--output", type=Path, required=True, help="Output .pkl token set.")
    parser.add_argument("--show", type=int, default=0, help="Print N decoded examples for checking.")
    parser.add_argument("--date", help="YYYY-MM-DD for the GPT-OSS system prompt (required for gpt_oss*).")
    args = parser.parse_args()

    prompts = load_prompts(args.input)
    tokenizer, formatted = format_prompts(args.recipe, prompts, args.date)
    token_ids = tokenizer(formatted, add_special_tokens=False, return_attention_mask=False)["input_ids"]
    token_set = dict(zip(prompts.keys(), token_ids))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as handle:
        pickle.dump(token_set, handle)
    print(f"Wrote {len(token_set)} token sequences to {args.output}")
    for text in formatted[: args.show]:
        print("-" * 80)
        print(text[-400:])


if __name__ == "__main__":
    main()
