"""Export prompt IDs and verified source provenance for SFT splits."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterator


DEFAULT_SPLIT_DIR = Path("work/sft/en/splits")
DEFAULT_DATA_DIR = Path("data/source")


def read_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    """Yield one-based line numbers and JSON objects from a JSONL file."""
    with path.open("rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON in {path}:{line_number}: {error}") from error
            if not isinstance(record, dict):
                raise ValueError(f"Expected a JSON object in {path}:{line_number}")
            yield line_number, record


def load_split(path: Path, split: str) -> list[dict[str, str]]:
    """Load and validate the unique prompt records in one split."""
    records: list[dict[str, str]] = []
    seen: dict[str, int] = {}
    for line_number, record in read_jsonl(path):
        prompt_id = record.get("prompt_id")
        dataset_name = record.get("dataset_name")
        if not isinstance(prompt_id, str) or not prompt_id:
            raise ValueError(f"Missing or invalid prompt_id in {path}:{line_number}")
        if not isinstance(dataset_name, str) or not dataset_name:
            raise ValueError(f"Missing or invalid dataset_name in {path}:{line_number}")
        if prompt_id in seen:
            raise ValueError(
                f"Duplicate prompt_id {prompt_id!r} in {path}:{seen[prompt_id]} "
                f"and {path}:{line_number}"
            )
        seen[prompt_id] = line_number
        records.append(
            {
                "prompt_id": prompt_id,
                "split": split,
                "dataset_name": dataset_name,
                "split_path": path.as_posix(),
                "split_line_number": str(line_number),
            }
        )
    return records


def build_provenance_index(
    data_dir: Path, dataset_names: set[str]
) -> dict[str, dict[str, str]]:
    """Index original records using the same ID rule as prompt generation."""
    index: dict[str, dict[str, str]] = {}
    for dataset_name in sorted(dataset_names):
        source_path = data_dir / dataset_name / "train.jsonl"
        if not source_path.is_file():
            raise FileNotFoundError(f"Original dataset split not found: {source_path}")
        for line_number, record in read_jsonl(source_path):
            realidx = record.get("realidx")
            if realidx is None or isinstance(realidx, (dict, list)):
                raise ValueError(
                    f"Missing or invalid realidx in {source_path}:{line_number}"
                )
            realidx_text = str(realidx)
            prompt_id = f"{dataset_name}_{realidx_text}"
            if prompt_id in index:
                previous = index[prompt_id]
                raise ValueError(
                    f"Duplicate reconstructed prompt_id {prompt_id!r}: "
                    f"{previous['source_path']}:{previous['source_line_number']} and "
                    f"{source_path}:{line_number}"
                )
            index[prompt_id] = {
                "source_path": source_path.as_posix(),
                "source_line_number": str(line_number),
                "realidx": realidx_text,
            }
    return index


def validate_manifest(
    manifest_path: Path, train_count: int, validation_count: int
) -> None:
    """Check exported counts against the split manifest when it is present."""
    if not manifest_path.is_file():
        return
    with manifest_path.open("rt", encoding="utf-8") as handle:
        manifest = json.load(handle)
    expected = {
        "train_prompt_count": train_count,
        "validation_prompt_count": validation_count,
        "prompt_count": train_count + validation_count,
    }
    for key, actual in expected.items():
        if manifest.get(key) != actual:
            raise ValueError(
                f"{manifest_path}: {key} is {manifest.get(key)!r}, expected {actual}"
            )


def write_lines(path: Path, values: Iterator[str]) -> None:
    with path.open("wt", encoding="utf-8") as handle:
        for value in values:
            handle.write(value)
            handle.write("\n")


def write_provenance(path: Path, records: list[dict[str, str]]) -> None:
    fieldnames = [
        "prompt_id",
        "split",
        "dataset_name",
        "realidx",
        "source_path",
        "source_line_number",
        "split_path",
        "split_line_number",
    ]
    with path.open("wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export unique train/validation prompt IDs and map each one back to "
            "data/<dataset>/train.jsonl."
        )
    )
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_SPLIT_DIR / "prompt_ids",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train = load_split(args.split_dir / "train.jsonl", "train")
    validation = load_split(args.split_dir / "validation.jsonl", "validation")

    train_ids = {record["prompt_id"] for record in train}
    validation_ids = {record["prompt_id"] for record in validation}
    overlap = train_ids & validation_ids
    if overlap:
        examples = ", ".join(sorted(overlap)[:5])
        raise ValueError(
            f"Train/validation prompt_id overlap ({len(overlap)} IDs), e.g. {examples}"
        )

    validate_manifest(
        args.split_dir / "split_manifest.json", len(train), len(validation)
    )
    all_records = train + validation
    dataset_names = {record["dataset_name"] for record in all_records}
    provenance = build_provenance_index(args.data_dir, dataset_names)

    missing: list[str] = []
    for record in all_records:
        source = provenance.get(record["prompt_id"])
        if source is None:
            missing.append(record["prompt_id"])
            continue
        expected_prefix = f"{record['dataset_name']}_"
        if not record["prompt_id"].startswith(expected_prefix):
            raise ValueError(
                f"Dataset mismatch for {record['prompt_id']!r}: "
                f"split says {record['dataset_name']!r}"
            )
        record.update(source)
    if missing:
        examples = ", ".join(missing[:5])
        raise ValueError(
            f"Could not locate {len(missing)} prompt IDs in original train files, "
            f"e.g. {examples}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_lines(
        args.output_dir / "train_prompt_ids.txt",
        (record["prompt_id"] for record in train),
    )
    write_lines(
        args.output_dir / "validation_prompt_ids.txt",
        (record["prompt_id"] for record in validation),
    )
    write_provenance(args.output_dir / "prompt_id_provenance.csv", all_records)

    counts = Counter(record["dataset_name"] for record in all_records)
    print(f"train prompt IDs:      {len(train):,}")
    print(f"validation prompt IDs: {len(validation):,}")
    print(f"overlap:               {len(overlap):,}")
    print("datasets:")
    for dataset_name, count in sorted(counts.items()):
        print(f"  {dataset_name}: {count:,}")
    print(f"wrote: {args.output_dir}")


if __name__ == "__main__":
    main()
