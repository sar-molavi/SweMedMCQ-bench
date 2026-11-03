"""Check that bilingual train/eval prompt IDs are disjoint across splits."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence


DEFAULT_TRAIN_PATHS = (
    Path("work/sft/en/flat/train.jsonl"),
    Path("work/sft/sv/splits/flat/train.jsonl"),
)
DEFAULT_VALIDATION_PATHS = (
    Path("work/sft/en/flat/validation_train_time.jsonl"),
    Path("work/sft/sv/splits/flat/validation_train_time.jsonl"),
)


def iter_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            yield line_number, row


def load_prompt_id_counts(paths: Sequence[Path], split_name: str) -> Counter[str]:
    """Load prompt IDs, allowing repeated IDs within a split."""

    counts: Counter[str] = Counter()
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"{split_name} file does not exist: {path}")
        for line_number, row in iter_jsonl(path):
            prompt_id = row.get("prompt_id")
            if not isinstance(prompt_id, str) or not prompt_id.strip():
                raise ValueError(f"{path}:{line_number}: missing non-empty prompt_id")
            counts[prompt_id.strip()] += 1
    if not counts:
        raise ValueError(f"No prompt IDs found for {split_name}.")
    return counts


def check_disjoint_prompt_ids(
    train_paths: Sequence[Path],
    validation_paths: Sequence[Path],
) -> dict[str, Any]:
    """Return summary counts or raise if train and validation share prompt IDs."""

    train_counts = load_prompt_id_counts(train_paths, "train")
    validation_counts = load_prompt_id_counts(validation_paths, "validation")
    overlap = sorted(set(train_counts) & set(validation_counts))
    if overlap:
        preview = ", ".join(overlap[:20])
        suffix = "..." if len(overlap) > 20 else ""
        raise ValueError(
            "Train/validation prompt_id overlap detected: "
            f"{len(overlap)} unique prompt ID(s). Examples: {preview}{suffix}"
        )

    return {
        "train_paths": [str(path) for path in train_paths],
        "validation_paths": [str(path) for path in validation_paths],
        "train_unique_prompt_ids": len(train_counts),
        "train_rows": sum(train_counts.values()),
        "train_duplicate_rows": sum(count - 1 for count in train_counts.values()),
        "validation_unique_prompt_ids": len(validation_counts),
        "validation_rows": sum(validation_counts.values()),
        "validation_duplicate_rows": sum(
            count - 1 for count in validation_counts.values()
        ),
        "overlap_unique_prompt_ids": 0,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-path",
        action="append",
        type=Path,
        dest="train_paths",
        help="Train JSONL path. Repeat for EN/SV. Defaults to bilingual train flat files.",
    )
    parser.add_argument(
        "--validation-path",
        action="append",
        type=Path,
        dest="validation_paths",
        help=(
            "Validation JSONL path. Repeat for EN/SV. Defaults to bilingual "
            "train-time validation flat files."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    train_paths = tuple(args.train_paths or DEFAULT_TRAIN_PATHS)
    validation_paths = tuple(args.validation_paths or DEFAULT_VALIDATION_PATHS)
    summary = check_disjoint_prompt_ids(train_paths, validation_paths)
    print(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

