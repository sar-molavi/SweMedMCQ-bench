"""Extract normalized Swedish answer-level rows from filtered translations."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from swemedmcq.paths import path as repo_path

from .build_answer_level_dataset import DEFAULT_DATASET_NAMES


DEFAULT_INPUT_PATH = repo_path("train_translation")


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc


def normalize_answer(value: Any, *, prompt_id: str) -> str:
    answer = str(value or "").strip().upper()
    if not answer:
        raise ValueError(f"{prompt_id}: missing gold answer")
    return answer


def load_answer_keys(
    data_dir: Path,
    file_name: str,
    datasets: Sequence[str],
) -> dict[str, str]:
    answer_keys: dict[str, str] = {}
    for dataset_name in datasets:
        path = data_dir / dataset_name / f"{file_name}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"Expected answer-key source not found: {path}")
        for row in iter_jsonl(path):
            if "realidx" not in row:
                raise ValueError(f"{path}: row is missing realidx")
            prompt_id = f"{dataset_name}_{row['realidx']}"
            answer_keys[prompt_id] = normalize_answer(
                row.get("answer_idx", row.get("gold_answer")),
                prompt_id=prompt_id,
            )
    return answer_keys


def translation_payload(row: Mapping[str, Any]) -> Mapping[str, Any]:
    output = row.get("translation_output")
    if isinstance(output, Mapping):
        return output
    return row


def normalize_choices(value: Any, *, prompt_id: str) -> list[dict[str, str]]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{prompt_id}: choices must be a non-empty list")
    choices: list[dict[str, str]] = []
    for index, choice in enumerate(value):
        if not isinstance(choice, Mapping):
            raise ValueError(f"{prompt_id}: choice {index} must be an object")
        label = str(choice.get("label") or "").strip().upper()
        text = str(choice.get("text") or "").strip()
        if not label or not text:
            raise ValueError(f"{prompt_id}: choice {index} is missing label/text")
        choices.append({"label": label, "text": text})
    return choices


def extract_row(
    row: Mapping[str, Any],
    answer_keys: Mapping[str, str],
) -> dict[str, Any]:
    prompt_id = str(row.get("prompt_id") or "").strip()
    if not prompt_id:
        raise ValueError("translated row is missing prompt_id")
    payload = translation_payload(row)
    question = str(payload.get("question") or row.get("question") or "").strip()
    if not question:
        raise ValueError(f"{prompt_id}: missing Swedish question")
    choices = normalize_choices(
        payload.get("choices", row.get("choices")),
        prompt_id=prompt_id,
    )
    gold_answer = row.get("gold_answer", row.get("answer_idx"))
    if gold_answer is None:
        gold_answer = answer_keys.get(prompt_id)
    return {
        "prompt_id": prompt_id,
        "dataset_name": str(row.get("dataset_name") or prompt_id.rsplit("_", 1)[0]),
        "realidx": str(row.get("realidx") or prompt_id.split("_", 1)[-1]),
        "question": question,
        "choices": choices,
        "gold_answer": normalize_answer(gold_answer, prompt_id=prompt_id),
    }


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


def extract_rows(
    *,
    input_path: Path,
    answer_key_data_dir: Path,
    answer_key_file_name: str,
    output_path: Path,
    report_path: Path,
    datasets: Sequence[str],
) -> dict[str, Any]:
    answer_keys = load_answer_keys(answer_key_data_dir, answer_key_file_name, datasets)
    rows_by_id: dict[str, dict[str, Any]] = {}
    duplicate_rows = 0
    for row in iter_jsonl(input_path):
        extracted = extract_row(row, answer_keys)
        prompt_id = extracted["prompt_id"]
        existing = rows_by_id.get(prompt_id)
        if existing is not None:
            duplicate_rows += 1
            if existing != extracted:
                raise ValueError(f"Conflicting duplicate prompt_id {prompt_id}")
            continue
        rows_by_id[prompt_id] = extracted
    output_count = write_jsonl(output_path, rows_by_id.values())
    missing_answer_keys = sorted(set(rows_by_id) - set(answer_keys))
    report = {
        "input_path": str(input_path),
        "answer_key_data_dir": str(answer_key_data_dir),
        "answer_key_file_name": answer_key_file_name,
        "output_path": str(output_path),
        "input_row_count": output_count + duplicate_rows,
        "output_row_count": output_count,
        "duplicate_rows": duplicate_rows,
        "answer_key_count": len(answer_keys),
        "missing_answer_key_count": len(missing_answer_keys),
        "missing_answer_key_preview": missing_answer_keys[:20],
    }
    write_json(report_path, report)
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument(
        "--answer-key-data-dir",
        type=Path,
        default=Path("data/source"),
        help="English data root used only for prompt_id -> gold answer lookup.",
    )
    parser.add_argument(
        "--answer-key-file-name",
        default="train",
        help="Split file name without .jsonl for the gold answer lookup.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("work/sft/answer_level/sv/source/train_trans_valid_filtered.jsonl"),
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("work/sft/answer_level/sv/source/extraction_manifest.json"),
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=list(DEFAULT_DATASET_NAMES),
        help="Datasets to scan for gold answers.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    report = extract_rows(
        input_path=args.input_path,
        answer_key_data_dir=args.answer_key_data_dir,
        answer_key_file_name=args.answer_key_file_name,
        output_path=args.output_path,
        report_path=args.report_path,
        datasets=args.datasets,
    )
    print(json.dumps(report, ensure_ascii=True, sort_keys=True))


if __name__ == "__main__":
    main()
