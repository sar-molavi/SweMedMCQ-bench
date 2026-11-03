"""Combine EN/SV answer-level split JSONL files."""

from __future__ import annotations

import argparse
import json
import random
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

DEFAULT_SHUFFLE_SEED = 2000


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc


def load_eval(path: Path, language: str) -> list[dict[str, Any]]:
    rows = []
    for row in iter_jsonl(path):
        result = dict(row)
        result["language"] = language
        rows.append(result)
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, separators=(",", ":")) + "\n")
            count += 1
    return count


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def combine_eval(
    *,
    english_path: Path,
    swedish_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    english_rows = load_eval(english_path, "en")
    swedish_rows = load_eval(swedish_path, "sv")
    combined = english_rows + swedish_rows
    output_count = write_jsonl(output_path, combined)
    return {
        "english_path": str(english_path),
        "swedish_path": str(swedish_path),
        "output_path": str(output_path),
        "english_count": len(english_rows),
        "swedish_count": len(swedish_rows),
        "output_count": output_count,
    }


def combine_split(
    *,
    english_path: Path,
    swedish_path: Path,
    output_path: Path,
    shuffle_seed: int | None,
) -> dict[str, Any]:
    english_rows = load_eval(english_path, "en")
    swedish_rows = load_eval(swedish_path, "sv")
    combined = english_rows + swedish_rows
    if shuffle_seed is not None:
        random.Random(shuffle_seed).shuffle(combined)
    output_count = write_jsonl(output_path, combined)
    return {
        "english_path": str(english_path),
        "swedish_path": str(swedish_path),
        "output_path": str(output_path),
        "english_count": len(english_rows),
        "swedish_count": len(swedish_rows),
        "output_count": output_count,
        "shuffle_seed": shuffle_seed,
    }


def copy_split(*, input_path: Path, output_path: Path, language: str) -> dict[str, Any]:
    rows = load_eval(input_path, language)
    output_count = write_jsonl(output_path, rows)
    return {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "language": language,
        "output_count": output_count,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--english-eval",
        type=Path,
        default=Path("work/sft/answer_level/en/splits/eval.jsonl"),
    )
    parser.add_argument(
        "--swedish-eval",
        type=Path,
        default=Path("work/sft/answer_level/sv/splits/eval.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("work/sft/answer_level/bilingual"),
    )
    parser.add_argument(
        "--english-train",
        type=Path,
        default=Path("work/sft/answer_level/en/splits/train.jsonl"),
    )
    parser.add_argument(
        "--swedish-train",
        type=Path,
        default=Path("work/sft/answer_level/sv/splits/train.jsonl"),
    )
    parser.add_argument(
        "--english-validation",
        type=Path,
        default=Path("work/sft/answer_level/en/splits/validation.jsonl"),
    )
    parser.add_argument(
        "--swedish-validation",
        type=Path,
        default=Path("work/sft/answer_level/sv/splits/validation.jsonl"),
    )
    parser.add_argument(
        "--mode",
        choices=("splits", "eval"),
        default="splits",
    )
    parser.add_argument("--shuffle-seed", type=int, default=DEFAULT_SHUFFLE_SEED)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    if args.mode == "eval":
        manifest = {
            "eval": combine_eval(
                english_path=args.english_eval,
                swedish_path=args.swedish_eval,
                output_path=args.output_dir / "splits" / "eval.jsonl",
            )
        }
    else:
        manifest = {
            "train": combine_split(
                english_path=args.english_train,
                swedish_path=args.swedish_train,
                output_path=args.output_dir / "splits" / "train.jsonl",
                shuffle_seed=args.shuffle_seed,
            ),
            "validation": combine_split(
                english_path=args.english_validation,
                swedish_path=args.swedish_validation,
                output_path=args.output_dir / "splits" / "validation.jsonl",
                shuffle_seed=None,
            ),
            "validation_en": copy_split(
                input_path=args.english_validation,
                output_path=args.output_dir / "splits" / "validation_en.jsonl",
                language="en",
            ),
            "validation_sv": copy_split(
                input_path=args.swedish_validation,
                output_path=args.output_dir / "splits" / "validation_sv.jsonl",
                language="sv",
            ),
        }
    manifest_path = args.output_dir / "manifest.json"
    write_json(manifest_path, manifest)
    print(json.dumps({**manifest, "manifest_path": str(manifest_path)}, ensure_ascii=True, sort_keys=True))


if __name__ == "__main__":
    main()
