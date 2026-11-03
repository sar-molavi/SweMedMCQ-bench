"""Flat English SFT files of the trace students, and their train-time validation set.

Reads the prompt-level English splits (``split_english_sft``) and writes one row per
kept teacher response (``train.jsonl``, ``validation.jsonl``), plus a stratified sample of
at most ``--validation-cap`` validation prompts with one preferred response each
(``validation_train_time.jsonl``), which is evaluated during training.

    python -m swemedmcq.distill.build_flat_splits --train-input work/sft/en/splits/train.jsonl \\
        --validation-input work/sft/en/splits/validation.jsonl --output-dir work/sft/en/flat
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence


from .create_train_time_validation import (
    SampledPrompt,
    allocate_counts,
    stratum_key,
    write_flat_jsonl,
    write_prompt_jsonl,
)
from .flatten_sft_dataset import (
    flatten_file,
    parse_source_preference,
    selected_responses,
    write_json,
)



def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
    return rows


def write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def stratified_sample(
    rows: Sequence[dict[str, Any]],
    cap: int,
    seed: int,
) -> list[SampledPrompt]:
    strata: dict[str, list[SampledPrompt]] = defaultdict(list)
    for index, record in enumerate(rows):
        stratum = stratum_key(record)
        strata[stratum].append(SampledPrompt(index=index, record=record, stratum=stratum))

    allocations = allocate_counts(
        {stratum: len(items) for stratum, items in strata.items()},
        cap=cap,
    )
    rng = random.Random(seed)
    selected = []
    for stratum, items in strata.items():
        shuffled = list(items)
        rng.shuffle(shuffled)
        selected.extend(shuffled[: allocations.get(stratum, 0)])
    return sorted(selected, key=lambda item: item.index)


def one_response_record(
    record: dict[str, Any],
    source_preference: Sequence[str],
) -> dict[str, Any] | None:
    responses = selected_responses(
        record,
        selection="preferred-one",
        source_preference=source_preference,
    )
    if not responses:
        return None
    result = dict(record)
    result["responses"] = responses
    return result


def create_train_time_validation(
    validation_input: Path,
    prompt_output: Path,
    flat_output: Path,
    manifest_output: Path,
    source_preference: Sequence[str],
    cap: int,
    seed: int,
) -> dict[str, Any]:
    rows = load_jsonl(validation_input)
    sampled = stratified_sample(rows, cap=cap, seed=seed)
    prompt_records = []
    skipped_prompt_ids = []
    stratum_counts = Counter()
    for item in sampled:
        record = one_response_record(item.record, source_preference=source_preference)
        if record is None:
            skipped_prompt_ids.append(item.record.get("prompt_id"))
            continue
        prompt_records.append(record)
        stratum_counts[item.stratum] += 1

    write_prompt_jsonl(prompt_output, prompt_records)
    flat_stats = write_flat_jsonl(flat_output, prompt_records)

    manifest = {
        "input_path": str(validation_input),
        "prompt_output_path": str(prompt_output),
        "flat_output_path": str(flat_output),
        "cap": cap,
        "seed": seed,
        "selection": "preferred-one",
        "source_preference": list(source_preference),
        "input_prompts": len(rows),
        "sampled_prompts": len(sampled),
        "output_prompts": len(prompt_records),
        "skipped_prompt_ids": skipped_prompt_ids,
        "stratum_counts": dict(sorted(stratum_counts.items())),
        "flat": flat_stats.to_dict(),
    }
    write_json(manifest_output, manifest)
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--train-input", type=Path, default=Path("work/sft/en/splits/train.jsonl"))
    parser.add_argument(
        "--validation-input",
        type=Path,
        default=Path("work/sft/en/splits/validation.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("work/sft/en/flat"),
    )
    parser.add_argument("--work-dir", type=Path, default=Path("work/sft/en/flat/prompt"),
                        help="Folder for the prompt-level train-time validation file.")
    parser.add_argument(
        "--source-preference",
        action="append",
        help="Source run priority for train-time validation preferred-one selection.",
    )
    parser.add_argument("--validation-cap", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=13)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    source_preference = tuple(parse_source_preference(args.source_preference))
    train_prompt_path = args.train_input
    validation_prompt_path = args.validation_input

    train_output = args.output_dir / "train.jsonl"
    validation_output = args.output_dir / "validation.jsonl"
    train_stats = flatten_file(
        train_prompt_path,
        train_output,
        selection="all",
        source_preference=source_preference,
    )
    validation_stats = flatten_file(
        validation_prompt_path,
        validation_output,
        selection="all",
        source_preference=source_preference,
    )

    train_time_manifest = create_train_time_validation(
        validation_input=validation_prompt_path,
        prompt_output=args.work_dir / "validation_train_time.jsonl",
        flat_output=args.output_dir / "validation_train_time.jsonl",
        manifest_output=args.output_dir / "validation_train_time_manifest.json",
        source_preference=source_preference,
        cap=args.validation_cap,
        seed=args.seed,
    )

    manifest = {
        "selection": "all",
        "source_preference": list(source_preference),
        "train": {
            "input_path": str(args.train_input),
            "output_path": str(train_output),
            **train_stats.to_dict(),
        },
        "validation": {
            "input_path": str(args.validation_input),
            "output_path": str(validation_output),
            **validation_stats.to_dict(),
        },
        "validation_train_time": train_time_manifest,
    }
    write_json(args.output_dir / "manifest.json", manifest)

    print(f"Wrote {train_stats.output_examples} train examples to {train_output}")
    print(f"Wrote {validation_stats.output_examples} validation examples to {validation_output}")
    print(f"Wrote train-time validation to {args.output_dir / 'validation_train_time.jsonl'}")
    print(f"Wrote manifest to {args.output_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
