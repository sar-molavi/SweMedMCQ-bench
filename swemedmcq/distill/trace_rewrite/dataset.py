"""Prepare flat Swedish SFT rows for the trace-rewrite workflow.

The first step of the workflow is intentionally simple: split one flat Swedish
SFT JSONL file into rows that can be used as-is and rows whose English
reasoning should later be rewritten with a teacher model.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any

def iter_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    """Yield non-empty JSONL rows with one-based line numbers.

    The line number is used only for clear error messages in downstream code.
    Each row must be a JSON object because the SFT pipeline expects object
    records with fields such as ``question``, ``reasoning``, and ``answer``.
    """

    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            yield line_number, row


def has_reasoning(row: dict[str, Any]) -> bool:
    """Return whether a response has a non-empty reasoning trace.

    The Swedish train split (``splits/train.jsonl``) is prompt-clustered, so reasoning is
    stored per response rather than at the top level.
    """

    reasoning = row.get("reasoning")
    return isinstance(reasoning, str) and bool(reasoning.strip())


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    """Write rows as compact UTF-8 JSONL and return the number written."""

    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count


def write_json(path: Path, value: Any) -> None:
    """Write an indented UTF-8 JSON file, creating parent directories."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def split_by_reasoning(
    input_path: Path,
    no_reasoning_path: Path,
    with_reasoning_path: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    """Split prompt-clustered SFT JSONL by response-level reasoning traces.

    ``no_reasoning_path`` receives rows that should pass through unchanged.
    ``with_reasoning_path`` receives rows that will later be converted into
    vLLM teacher prompts for Swedish reasoning rewrite. The returned manifest
    records the output paths and row counts.
    """

    rows_without_reasoning: list[dict[str, Any]] = []
    rows_with_reasoning: list[dict[str, Any]] = []
    input_rows = 0
    input_responses = 0

    for line_number, row in iter_jsonl(input_path):
        input_rows += 1
        responses = row.get("responses")
        if not isinstance(responses, list):
            raise ValueError(f"{input_path}:{line_number}: responses must be a list")

        with_reasoning = [
            response
            for response in responses
            if isinstance(response, dict) and has_reasoning(response)
        ]
        without_reasoning = [
            response
            for response in responses
            if isinstance(response, dict) and not has_reasoning(response)
        ]
        input_responses += len(with_reasoning) + len(without_reasoning)

        if with_reasoning:
            with_reasoning_row = dict(row)
            with_reasoning_row["responses"] = with_reasoning
            rows_with_reasoning.append(with_reasoning_row)
        if without_reasoning:
            without_reasoning_row = dict(row)
            without_reasoning_row["responses"] = without_reasoning
            rows_without_reasoning.append(without_reasoning_row)

    no_reasoning_count = write_jsonl(no_reasoning_path, rows_without_reasoning)
    with_reasoning_count = write_jsonl(with_reasoning_path, rows_with_reasoning)
    no_reasoning_responses = sum(len(row["responses"]) for row in rows_without_reasoning)
    with_reasoning_responses = sum(len(row["responses"]) for row in rows_with_reasoning)
    manifest = {
        "input_path": str(input_path),
        "no_reasoning_path": str(no_reasoning_path),
        "with_reasoning_path": str(with_reasoning_path),
        "input_rows": input_rows,
        "input_responses": input_responses,
        "no_reasoning_rows": no_reasoning_count,
        "no_reasoning_responses": no_reasoning_responses,
        "with_reasoning_rows": with_reasoning_count,
        "with_reasoning_responses": with_reasoning_responses,
    }
    write_json(manifest_path, manifest)
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the JSONL split command."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--no-reasoning-path", type=Path)
    parser.add_argument("--with-reasoning-path", type=Path)
    parser.add_argument("--manifest-path", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Run the split command and print the manifest JSON to stdout."""

    args = build_arg_parser().parse_args(argv)
    no_reasoning_path = args.no_reasoning_path or args.output_dir / "train_no_reasoning.jsonl"
    with_reasoning_path = args.with_reasoning_path or args.output_dir / "train_with_reasoning.jsonl"
    manifest_path = args.manifest_path or args.output_dir / "split_manifest.json"
    manifest = split_by_reasoning(
        input_path=args.input_path,
        no_reasoning_path=no_reasoning_path,
        with_reasoning_path=with_reasoning_path,
        manifest_path=manifest_path,
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
