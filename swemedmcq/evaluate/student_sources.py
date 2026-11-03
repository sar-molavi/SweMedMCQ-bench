#!/usr/bin/env python
"""Prepare metadata-rich source files for final test-set scoring."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterable

from swemedmcq import paths
from swemedmcq.evaluate.registry import data_dir


QUESTION_PATTERNS = {
    "en": re.compile(r"\*\*Question:\*\*\s*(.*?)\s*\*\*Options:\*\*", re.DOTALL),
    "sv": re.compile(r"\*\*Fråga:\*\*\s*(.*?)\s*\*\*Alternativ:\*\*", re.DOTALL),
}


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    """Yield non-empty JSON objects from a JSONL file."""
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            yield row


def load_rows_by_prompt_id(path: Path) -> dict[str, dict[str, Any]]:
    """Load a JSONL file into a dictionary keyed by ``prompt_id``."""
    rows: dict[str, dict[str, Any]] = {}
    for row in iter_jsonl(path):
        prompt_id = row.get("prompt_id")
        if not isinstance(prompt_id, str) or not prompt_id.strip():
            raise ValueError(f"{path}: row is missing a non-empty prompt_id")
        if prompt_id in rows:
            raise ValueError(f"{path}: duplicate prompt_id {prompt_id!r}")
        rows[prompt_id] = row
    if not rows:
        raise ValueError(f"{path}: no rows found")
    return rows


def dataset_name_from_prompt_id(prompt_id: str) -> str:
    """Infer the dataset prefix from a test prompt identifier."""
    return prompt_id.split("_", 1)[0]


def question_from_prompt(prompt: Any, language: str) -> str:
    """Extract the displayed question text from a chat prompt."""
    if not isinstance(prompt, list) or not prompt:
        return ""
    content = prompt[0].get("content") if isinstance(prompt[0], dict) else None
    if not isinstance(content, str):
        return ""
    match = QUESTION_PATTERNS[language].search(content)
    return match.group(1).strip() if match else content.strip()


def scoring_row(
    prompt_row: dict[str, Any],
    *,
    fallback_row: dict[str, Any] | None,
    language: str,
) -> dict[str, Any]:
    """Build the minimal source row required by ``extract_eval_predictions``."""
    prompt_id = prompt_row["prompt_id"]
    gold_source = prompt_row if prompt_row.get("gold_answer") is not None else fallback_row
    if gold_source is None or gold_source.get("gold_answer") is None:
        raise KeyError(f"Missing gold_answer for {prompt_id}")
    dataset_name = (
        prompt_row.get("dataset_name")
        or gold_source.get("dataset_name")
        or dataset_name_from_prompt_id(prompt_id)
    )
    question = prompt_row.get("question") or question_from_prompt(prompt_row.get("prompt"), language)
    if not question:
        question = gold_source.get("question") or ""
    if not question:
        raise ValueError(f"Missing question text for {prompt_id}")

    return {
        "prompt_id": prompt_id,
        "source_prompt_id": prompt_id,
        "dataset_name": dataset_name,
        "language": language,
        "question": question,
        "gold_answer": gold_source["gold_answer"],
        "allowed_choices": gold_source.get("allowed_choices"),
    }


def write_source(
    *,
    prompt_path: Path,
    output_path: Path,
    language: str,
    fallback_rows: dict[str, dict[str, Any]],
) -> int:
    """Write a scoring source JSONL from a prompt file and optional fallback metadata."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for prompt_row in iter_jsonl(prompt_path):
            prompt_id = prompt_row["prompt_id"]
            row = scoring_row(
                prompt_row,
                fallback_row=fallback_rows.get(prompt_id),
                language=language,
            )
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count


def parse_args() -> argparse.Namespace:
    """Parse command-line options for final test source preparation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--en-prompt-path", type=Path, default=data_dir() / "en_trace.jsonl")
    parser.add_argument("--sv-prompt-path", type=Path, default=data_dir() / "sv_L4a_trace.jsonl")
    parser.add_argument(
        "--metadata-fallback-path",
        type=Path,
        default=data_dir() / "en_answer_split.jsonl",
        help="Used only when a prompt row lacks gold_answer/dataset metadata.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=paths.path("results_dir") / "sources",
    )
    return parser.parse_args()


def main() -> None:
    """Create English and Swedish scoring source files."""
    args = parse_args()
    fallback_rows = load_rows_by_prompt_id(args.metadata_fallback_path)
    for language, prompt_path in {"en": args.en_prompt_path, "sv": args.sv_prompt_path}.items():
        output_path = args.output_dir / f"{language}_test_source.jsonl"
        count = write_source(
            prompt_path=prompt_path,
            output_path=output_path,
            language=language,
            fallback_rows=fallback_rows,
        )
        print(f"Wrote {count} {language} source rows to {output_path}")


if __name__ == "__main__":
    main()
