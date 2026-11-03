"""Build the final clustered dataset from valid Swedish trace rewrites."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .dataset import has_reasoning, iter_jsonl, write_json
from .rewrite_prompts import safe_prompt_id
from ..flatten_sft_dataset import DEFAULT_SOURCE_PREFERENCE, flatten_file


def load_rewrites(path: Path) -> dict[str, str]:
    rewrites = {}
    for line_number, row in iter_jsonl(path):
        prompt_id = row.get("prompt_id")
        data = row.get("data")
        reasoning = data.get("reasoning") if isinstance(data, dict) else None
        if not isinstance(prompt_id, str) or not isinstance(reasoning, str):
            raise ValueError(f"{path}:{line_number}: invalid extracted rewrite")
        if prompt_id in rewrites:
            raise ValueError(f"{path}:{line_number}: duplicate prompt_id {prompt_id}")
        rewrites[prompt_id] = reasoning.strip()
    return rewrites


def build_dataset(
    source_path: Path,
    rewrites_path: Path,
    output_path: Path,
    summary_path: Path,
    flat_output_path: Path | None = None,
) -> dict[str, Any]:
    rewrites = load_rewrites(rewrites_path)
    if not rewrites:
        raise ValueError(f"No valid rewrites found in {rewrites_path}")
    used = set()
    counts = {
        "source_prompts": 0,
        "source_responses": 0,
        "rewritten_responses": 0,
        "responses_without_reasoning": 0,
        "dropped_responses": 0,
        "output_prompts": 0,
        "output_responses": 0,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output:
        for line_number, row in iter_jsonl(source_path):
            counts["source_prompts"] += 1
            source_prompt_id = row.get("prompt_id")
            responses = row.get("responses")
            if not isinstance(source_prompt_id, str) or not isinstance(responses, list):
                raise ValueError(f"{source_path}:{line_number}: invalid prompt row")

            kept = []
            reasoning_index = 0
            for response in responses:
                counts["source_responses"] += 1
                if not isinstance(response, dict):
                    raise ValueError(f"{source_path}:{line_number}: invalid response")
                if not has_reasoning(response):
                    counts["responses_without_reasoning"] += 1
                    kept.append(response)
                    continue

                rewrite_id = (
                    f"{safe_prompt_id(source_prompt_id)}__response_{reasoning_index:03d}"
                )
                reasoning_index += 1
                rewritten_reasoning = rewrites.get(rewrite_id)
                if rewritten_reasoning is None:
                    counts["dropped_responses"] += 1
                    continue
                rewritten = dict(response)
                rewritten["reasoning"] = rewritten_reasoning
                kept.append(rewritten)
                used.add(rewrite_id)
                counts["rewritten_responses"] += 1

            if kept:
                final_row = dict(row)
                final_row["responses"] = kept
                output.write(
                    json.dumps(final_row, ensure_ascii=False, separators=(",", ":"))
                    + "\n"
                )
                counts["output_prompts"] += 1
                counts["output_responses"] += len(kept)

    unused = set(rewrites) - used
    if unused:
        raise ValueError(f"{len(unused)} valid rewrites did not match the source dataset")

    summary = {
        "source_path": str(source_path),
        "rewrites_path": str(rewrites_path),
        "output_path": str(output_path),
        **counts,
    }
    if flat_output_path is not None:
        flat_stats = flatten_file(
            output_path, flat_output_path, "all", DEFAULT_SOURCE_PREFERENCE
        )
        summary["flat_output_path"] = str(flat_output_path)
        summary["flat_output_examples"] = flat_stats.output_examples
    write_json(summary_path, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-path", type=Path, required=True)
    parser.add_argument("--rewrites-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--summary-path", type=Path, required=True)
    parser.add_argument("--flat-output-path", type=Path)
    args = parser.parse_args()
    print(json.dumps(build_dataset(**vars(args)), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
