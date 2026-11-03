"""Build answer-only EN/SV MCQ JSONL files.

This module expects each language input to already contain MCQ rows in the
target language, then formats user prompts whose only valid model output is
``{"answer_index": "<LETTER>"}``. A validation prompt-id file can carve a
validation split out of a full training split while leaving the rest for
training.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from string import ascii_uppercase
from typing import Any

from ..prompt_ids import read_prompt_ids
from ...common.prompts import ANSWER_RESPONSE_PROMPT as RESPONSE_PROMPT
from ...common.prompts import ANSWER_SWE_RESPONSE_PROMPT as SWE_RESPONSE_PROMPT


DEFAULT_DATASET_NAMES: Sequence[str] = (
    "medexqa",
    "mmlu-pro",
    "afrimedqa",
    "mmlu",
    "medqa",
    "medxpertqa-u",
    "medmcqa",
    "medxpertqa-r",
    "pubmedqa",
    "medbullets",
)

PROMPT_TEMPLATES = {
    "en": RESPONSE_PROMPT,
    "sv": SWE_RESPONSE_PROMPT,
}


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc


def load_prompt_ids(path: Path) -> set[str]:
    return read_prompt_ids(path)


def normalize_answer(value: Any, *, prompt_id: str) -> str:
    answer = str(value or "").strip().upper()
    if not answer:
        raise ValueError(f"{prompt_id}: missing answer_idx/gold_answer")
    return answer


def normalize_choices(row: Mapping[str, Any], *, prompt_id: str) -> Mapping[str, Any] | Sequence[Any]:
    choices = row.get("options", row.get("choices"))
    if not choices:
        raise ValueError(f"{prompt_id}: missing options/choices")
    if not isinstance(choices, Mapping) and not isinstance(choices, Sequence):
        raise TypeError(f"{prompt_id}: options/choices must be a mapping or sequence")
    if isinstance(choices, (str, bytes)):
        raise TypeError(f"{prompt_id}: options/choices must not be a string")
    return choices


def choices_to_str(choices: Mapping[str, Any] | Sequence[Any]) -> tuple[str, list[str]]:
    if isinstance(choices, Mapping):
        pairs = [
            (str(key).strip().upper(), value)
            for key, value in choices.items()
            if value is not None
        ]
    else:
        if len(choices) > len(ascii_uppercase):
            raise ValueError("choices_to_str supports at most 26 answer options.")
        pairs = []
        for index, value in enumerate(choices):
            if value is None:
                continue
            if isinstance(value, Mapping):
                label = str(value.get("label") or ascii_uppercase[index]).strip().upper()
                text = value.get("text")
                if text is None:
                    text = value.get("value")
                if text is None:
                    raise ValueError(f"Choice {index} is missing text/value")
                pairs.append((label, text))
            else:
                pairs.append((ascii_uppercase[index], value))
    if not pairs:
        raise ValueError("Missing answer options")
    return "\n\n".join(f"{key}) {value}" for key, value in pairs), [key for key, _ in pairs]


def prompt_id_for_row(row: Mapping[str, Any], dataset_name: str | None) -> str:
    prompt_id = row.get("prompt_id")
    if prompt_id:
        return str(prompt_id)
    if dataset_name is None:
        raise ValueError("Rows without prompt_id must come from dataset subdirectories")
    if "realidx" not in row:
        raise ValueError(f"{dataset_name}: row is missing realidx")
    return f"{dataset_name}_{row['realidx']}"


def dataset_name_for_row(row: Mapping[str, Any], dataset_name: str | None, prompt_id: str) -> str:
    if row.get("dataset_name"):
        return str(row["dataset_name"])
    if dataset_name is not None:
        return dataset_name
    if "_" not in prompt_id:
        raise ValueError(f"{prompt_id}: missing dataset_name")
    return prompt_id.rsplit("_", 1)[0]


def make_record(row: Mapping[str, Any], *, language: str, dataset_name: str | None = None) -> dict[str, Any]:
    prompt_id = prompt_id_for_row(row, dataset_name)
    resolved_dataset = dataset_name_for_row(row, dataset_name, prompt_id)
    question = str(row.get("question") or "").strip()
    if not question:
        raise ValueError(f"{prompt_id}: missing question")
    options, allowed_choices = choices_to_str(normalize_choices(row, prompt_id=prompt_id))
    template = PROMPT_TEMPLATES[language]
    prompt = template.format(question=question, options=options)
    return {
        "prompt_id": prompt_id,
        "dataset_name": resolved_dataset,
        "language": language,
        "prompt": [{"role": "user", "content": prompt}],
        "gold_answer": normalize_answer(
            row.get("gold_answer", row.get("answer_idx")),
            prompt_id=prompt_id,
        ),
        "allowed_choices": allowed_choices,
    }


def input_files(input_path: Path, file_name: str, datasets: Sequence[str] | None) -> list[tuple[Path, str | None]]:
    if input_path.is_file():
        return [(input_path, None)]
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    names = list(datasets or DEFAULT_DATASET_NAMES)
    return [(input_path / name / f"{file_name}.jsonl", name) for name in names]


def load_records(
    *,
    input_path: Path,
    file_name: str,
    language: str,
    datasets: Sequence[str] | None,
    prompt_ids: set[str] | None = None,
) -> tuple[list[dict[str, Any]], int]:
    records_by_id: dict[str, dict[str, Any]] = {}
    duplicate_rows = 0
    for path, dataset_name in input_files(input_path, file_name, datasets):
        if not path.exists():
            raise FileNotFoundError(f"Expected evaluation file not found: {path}")
        for row in iter_jsonl(path):
            record = make_record(row, language=language, dataset_name=dataset_name)
            prompt_id = record["prompt_id"]
            existing = records_by_id.get(prompt_id)
            if existing is not None:
                duplicate_rows += 1
                if existing != record:
                    raise ValueError(f"Conflicting duplicate prompt_id {prompt_id}")
                continue
            if prompt_ids is not None and prompt_id not in prompt_ids:
                continue
            records_by_id[prompt_id] = record
    if prompt_ids is not None:
        missing = prompt_ids - set(records_by_id)
        if missing:
            preview = ", ".join(sorted(missing)[:10])
            raise ValueError(f"Requested prompt ID(s) were not found: {preview}")
    return list(records_by_id.values()), duplicate_rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, separators=(",", ":")) + "\n")
            count += 1
    return count


def prompt_only_rows(rows: Iterable[dict[str, Any]]) -> Iterable[dict[str, Any]]:
    for row in rows:
        yield {
            "prompt_id": row["prompt_id"],
            "prompt": row["prompt"],
        }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_split_manifest(path: Path, manifest: dict[str, Any]) -> None:
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        existing_languages = existing.get("languages")
        if isinstance(existing_languages, dict):
            merged_languages = dict(existing_languages)
            merged_languages.update(manifest.get("languages", {}))
            manifest = {**existing, **manifest, "languages": merged_languages}
    write_json(path, manifest)


def split_train_validation(
    records: Sequence[dict[str, Any]],
    validation_prompt_ids: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    for record in records:
        if record["prompt_id"] in validation_prompt_ids:
            validation_rows.append(record)
        else:
            train_rows.append(record)
    return train_rows, validation_rows


def build_language_eval(
    *,
    language: str,
    input_path: Path,
    file_name: str,
    output_name: str,
    output_dir: Path,
    datasets: Sequence[str] | None,
    prompt_ids_file: Path | None = None,
) -> dict[str, Any]:
    prompt_ids = load_prompt_ids(prompt_ids_file) if prompt_ids_file is not None else None
    records, duplicate_rows = load_records(
        input_path=input_path,
        file_name=file_name,
        language=language,
        datasets=datasets,
        prompt_ids=prompt_ids,
    )
    eval_path = output_dir / language / "splits" / f"{output_name}.jsonl"
    prompt_path = output_dir / language / "prompts" / f"{output_name}_prompt_set.jsonl"
    eval_count = write_jsonl(eval_path, records)
    prompt_count = write_jsonl(prompt_path, prompt_only_rows(records))
    return {
        "input_path": str(input_path),
        "file_name": file_name,
        "prompt_ids_file": str(prompt_ids_file) if prompt_ids_file is not None else None,
        "requested_prompt_count": len(prompt_ids) if prompt_ids is not None else None,
        "eval_path": str(eval_path),
        "prompt_path": str(prompt_path),
        "input_prompt_count": len(records),
        "duplicate_rows": duplicate_rows,
        "eval_prompt_count": eval_count,
        "prompt_set_count": prompt_count,
    }


def build_language_splits(
    *,
    language: str,
    input_path: Path,
    train_file_name: str,
    test_file_name: str | None,
    validation_prompt_ids_path: Path,
    output_dir: Path,
    datasets: Sequence[str] | None,
) -> dict[str, Any]:
    validation_prompt_ids = load_prompt_ids(validation_prompt_ids_path)
    train_source_records, train_duplicate_rows = load_records(
        input_path=input_path,
        file_name=train_file_name,
        language=language,
        datasets=datasets,
    )
    if test_file_name is None:
        test_records: list[dict[str, Any]] = []
        test_duplicate_rows = 0
    else:
        test_records, test_duplicate_rows = load_records(
            input_path=input_path,
            file_name=test_file_name,
            language=language,
            datasets=datasets,
        )
    train_rows, validation_rows = split_train_validation(
        train_source_records,
        validation_prompt_ids,
    )
    language_dir = output_dir / language
    split_dir = language_dir / "splits"
    prompt_dir = language_dir / "prompts"
    train_path = split_dir / "train.jsonl"
    validation_path = split_dir / "validation.jsonl"
    validation_prompt_path = prompt_dir / "validation_prompt_set.jsonl"
    train_count = write_jsonl(train_path, train_rows)
    validation_count = write_jsonl(validation_path, validation_rows)
    validation_prompt_count = write_jsonl(
        validation_prompt_path,
        prompt_only_rows(validation_rows),
    )
    if test_file_name is None:
        test_path = None
        test_prompt_path = None
        test_count = 0
        test_prompt_count = 0
    else:
        test_path = split_dir / "test.jsonl"
        test_prompt_path = prompt_dir / "test_prompt_set.jsonl"
        test_count = write_jsonl(test_path, test_records)
        test_prompt_count = write_jsonl(
            test_prompt_path,
            prompt_only_rows(test_records),
        )
    train_ids = {record["prompt_id"] for record in train_rows}
    validation_ids = {record["prompt_id"] for record in validation_rows}
    test_ids = {record["prompt_id"] for record in test_records}
    present_train_source_ids = {
        record["prompt_id"] for record in train_source_records
    }
    missing_validation_prompt_ids = sorted(
        validation_prompt_ids - present_train_source_ids
    )
    return {
        "input_path": str(input_path),
        "train_file_name": train_file_name,
        "test_file_name": test_file_name,
        "validation_prompt_ids_path": str(validation_prompt_ids_path),
        "validation_prompt_id_count": len(validation_prompt_ids),
        "train_path": str(train_path),
        "validation_path": str(validation_path),
        "test_path": str(test_path) if test_path is not None else None,
        "validation_prompt_path": str(validation_prompt_path),
        "test_prompt_path": str(test_prompt_path) if test_prompt_path is not None else None,
        "input_train_prompt_count": len(train_source_records),
        "input_test_prompt_count": len(test_records),
        "train_duplicate_rows": train_duplicate_rows,
        "test_duplicate_rows": test_duplicate_rows,
        "train_prompt_count": train_count,
        "validation_prompt_count": validation_count,
        "test_prompt_count": test_count,
        "validation_prompt_set_count": validation_prompt_count,
        "test_prompt_set_count": test_prompt_count,
        "missing_validation_prompt_count": len(
            missing_validation_prompt_ids
        ),
        "missing_validation_prompt_preview": missing_validation_prompt_ids[:20],
        "train_validation_overlap_count": len(train_ids & validation_ids),
        "train_test_overlap_count": len(train_ids & test_ids),
        "validation_test_overlap_count": len(validation_ids & test_ids),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--english-input",
        type=Path,
        required=True,
        help="English evaluation JSONL file or root with dataset subdirectories.",
    )
    parser.add_argument(
        "--swedish-input",
        type=Path,
        default=None,
        help="Swedish evaluation JSONL file or root with dataset subdirectories.",
    )
    parser.add_argument(
        "--language",
        choices=("en", "sv", "both"),
        default="both",
        help="Which language eval file(s) to build.",
    )
    parser.add_argument(
        "--file-name",
        default="test_hard",
        help="Split file name without .jsonl when an input is a dataset root.",
    )
    parser.add_argument(
        "--train-file-name",
        default="train",
        help="Training split file name without .jsonl for split generation.",
    )
    parser.add_argument(
        "--test-file-name",
        default="test",
        help="Test split file name without .jsonl for split generation.",
    )
    parser.add_argument(
        "--no-test",
        action="store_true",
        help="Build only train/validation splits from the input source.",
    )
    parser.add_argument(
        "--validation-prompt-ids",
        type=Path,
        default=None,
        help="Prompt IDs to carve out of the training split as validation.",
    )
    parser.add_argument(
        "--output-name",
        default="eval",
        help="Output file stem written under the language directory.",
    )
    parser.add_argument(
        "--prompt-ids-file",
        type=Path,
        default=None,
        help="Optional prompt-id file selecting a subset from the input split.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("work/sft/answer_level"),
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=list(DEFAULT_DATASET_NAMES),
        help="Dataset subdirectories to process when inputs are roots.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    languages: dict[str, Any] = {}
    if args.validation_prompt_ids is not None:
        if args.language in {"en", "both"}:
            languages["en"] = build_language_splits(
                language="en",
                input_path=args.english_input,
                train_file_name=args.train_file_name,
                test_file_name=None if args.no_test else args.test_file_name,
                validation_prompt_ids_path=args.validation_prompt_ids,
                output_dir=args.output_dir,
                datasets=args.datasets,
            )
        if args.language in {"sv", "both"}:
            if args.swedish_input is None:
                raise ValueError("--swedish-input is required when --language is sv or both")
            languages["sv"] = build_language_splits(
                language="sv",
                input_path=args.swedish_input,
                train_file_name=args.train_file_name,
                test_file_name=None if args.no_test else args.test_file_name,
                validation_prompt_ids_path=args.validation_prompt_ids,
                output_dir=args.output_dir,
                datasets=args.datasets,
            )
        manifest = {
            "objective": "answer_index_only_train_validation_test",
            "languages": languages,
        }
        manifest_path = args.output_dir / "manifest_splits.json"
    else:
        if args.language in {"en", "both"}:
            languages["en"] = build_language_eval(
                language="en",
                input_path=args.english_input,
                file_name=args.file_name,
                output_name=args.output_name,
                output_dir=args.output_dir,
                datasets=args.datasets,
                prompt_ids_file=args.prompt_ids_file,
            )
        if args.language in {"sv", "both"}:
            if args.swedish_input is None:
                raise ValueError("--swedish-input is required when --language is sv or both")
            languages["sv"] = build_language_eval(
                language="sv",
                input_path=args.swedish_input,
                file_name=args.file_name,
                output_name=args.output_name,
                output_dir=args.output_dir,
                datasets=args.datasets,
                prompt_ids_file=args.prompt_ids_file,
            )
        manifest = {
            "objective": "answer_index_only_eval",
            "languages": languages,
        }
        manifest_path = args.output_dir / f"manifest_{args.output_name}.json"
    if args.validation_prompt_ids is not None:
        write_split_manifest(manifest_path, manifest)
    else:
        write_json(manifest_path, manifest)
    print(json.dumps({**manifest, "manifest_path": str(manifest_path)}, ensure_ascii=True, sort_keys=True))


if __name__ == "__main__":
    main()
