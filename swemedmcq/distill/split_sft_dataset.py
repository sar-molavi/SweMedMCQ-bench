"""Assign Swedish SFT prompts to the authoritative English SFT splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

from swemedmcq.distill.prompt_ids import read_prompt_ids


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    """Yield JSON objects from a non-empty JSONL file."""
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc


def load_prompt_ids(path: Path) -> set[str]:
    """English split membership: a prompt id file (``data/train/split``) or an English split JSONL."""
    return read_prompt_ids(path)


def split_rows_by_reference(
    rows: Iterable[dict[str, Any]],
    english_train_ids: set[str],
    english_validation_ids: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    """Assign Swedish rows from English membership and return unmatched IDs."""
    overlap = english_train_ids & english_validation_ids
    if overlap:
        preview = ", ".join(sorted(overlap)[:5])
        raise ValueError(
            "English reference train and validation splits overlap on "
            f"{len(overlap)} prompt IDs: {preview}"
        )

    train_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    unmatched_ids: list[str] = []
    seen_ids: set[str] = set()
    for row in rows:
        prompt_id = str(row.get("prompt_id") or "")
        if not prompt_id:
            raise ValueError("Swedish filtered input contains a row without prompt_id")
        if prompt_id in seen_ids:
            raise ValueError(f"Swedish filtered input has duplicate prompt_id: {prompt_id}")
        seen_ids.add(prompt_id)
        if prompt_id in english_train_ids:
            train_rows.append(row)
        elif prompt_id in english_validation_ids:
            validation_rows.append(row)
        else:
            unmatched_ids.append(prompt_id)
    return train_rows, validation_rows, sorted(unmatched_ids)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    """Write prompt-clustered SFT records as compact UTF-8 JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, separators=(",", ":")) + "\n")


def response_count(rows: Iterable[dict[str, Any]]) -> int:
    return sum(len(row.get("responses") or []) for row in rows)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--english-train-path",
        type=Path,
        default=Path("data/train/split/train_prompt_ids.txt.gz"),
        help="English train prompt ids (released split file or English split JSONL).",
    )
    parser.add_argument(
        "--english-validation-path",
        type=Path,
        default=Path("data/train/split/validation_prompt_ids.txt.gz"),
        help="English validation prompt ids (released split file or English split JSONL).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    english_train_ids = load_prompt_ids(args.english_train_path)
    english_validation_ids = load_prompt_ids(args.english_validation_path)
    input_rows = list(iter_jsonl(args.input_path))
    train_rows, validation_rows, unmatched_ids = split_rows_by_reference(
        input_rows, english_train_ids, english_validation_ids
    )

    train_path = args.output_dir / "train.jsonl"
    validation_path = args.output_dir / "validation.jsonl"
    write_jsonl(train_path, train_rows)
    write_jsonl(validation_path, validation_rows)
    manifest = {
        "input_path": str(args.input_path),
        "train_path": str(train_path),
        "validation_path": str(validation_path),
        "english_train_path": str(args.english_train_path),
        "english_validation_path": str(args.english_validation_path),
        "split_strategy": "authoritative English flat prompt_id membership",
        "input_prompt_count": len(input_rows),
        "train_prompt_count": len(train_rows),
        "validation_prompt_count": len(validation_rows),
        "excluded_prompt_count": len(unmatched_ids),
        "excluded_prompt_ids": unmatched_ids,
        "train_response_count": response_count(train_rows),
        "validation_response_count": response_count(validation_rows),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "split_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=True))


if __name__ == "__main__":
    main()
