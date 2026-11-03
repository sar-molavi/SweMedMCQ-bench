"""Create a compact validation split for training-time SFT evaluation.

The full validation split is prompt-clustered but flattens to multiple response
targets per prompt. This utility samples a limited number of validation prompts
and keeps one preferred response per prompt, producing a stable, small loss proxy
for frequent Trainer evaluation.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .flatten_sft_dataset import (
    DEFAULT_SOURCE_PREFERENCE,
    ExportStats,
    make_example,
    parse_source_preference,
    response_is_usable,
    selected_responses,
    write_json,
)


@dataclass
class SampledPrompt:
    index: int
    record: dict[str, Any]
    stratum: str


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


def response_count_bucket(record: dict[str, Any]) -> str:
    count = len([response for response in record.get("responses", []) if response_is_usable(response)])
    if count <= 1:
        return "1"
    if count == 2:
        return "2"
    return "3+"


def has_reasoning(record: dict[str, Any]) -> bool:
    for response in record.get("responses", []):
        reasoning = response.get("reasoning")
        if isinstance(reasoning, str) and reasoning.strip():
            return True
    return False


def stratum_key(record: dict[str, Any]) -> str:
    return "|".join(
        [
            str(record.get("dataset_name") or ""),
            response_count_bucket(record),
            "has_reasoning" if has_reasoning(record) else "no_reasoning",
        ]
    )


def allocate_counts(stratum_sizes: dict[str, int], cap: int) -> dict[str, int]:
    total = sum(stratum_sizes.values())
    if cap >= total:
        return dict(stratum_sizes)
    if cap <= 0:
        raise ValueError(f"cap must be positive, got {cap}")

    quotas = {
        stratum: cap * size / total
        for stratum, size in stratum_sizes.items()
    }
    allocations = {
        stratum: min(size, int(quotas[stratum]))
        for stratum, size in stratum_sizes.items()
    }
    remaining = cap - sum(allocations.values())

    order = sorted(
        stratum_sizes,
        key=lambda stratum: (
            quotas[stratum] - int(quotas[stratum]),
            stratum_sizes[stratum],
            stratum,
        ),
        reverse=True,
    )
    while remaining > 0:
        progressed = False
        for stratum in order:
            if remaining == 0:
                break
            if allocations[stratum] >= stratum_sizes[stratum]:
                continue
            allocations[stratum] += 1
            remaining -= 1
            progressed = True
        if not progressed:
            break
    return allocations


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


def write_prompt_jsonl(path: Path, records: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def write_flat_jsonl(path: Path, records: Sequence[dict[str, Any]]) -> ExportStats:
    stats = ExportStats()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            stats.input_prompts += 1
            responses = record.get("responses", [])
            if not responses:
                stats.skipped_prompts += 1
                continue
            response = responses[0]
            example = make_example(record, response)
            handle.write(json.dumps(example, ensure_ascii=True) + "\n")
            stats.output_examples += 1
            if example["reasoning"] is not None:
                stats.examples_with_reasoning += 1
            stats.source_counts[str(example["source_run"])] += 1
            stats.dataset_counts[str(example["dataset_name"])] += 1
    return stats


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a capped, one-response-per-prompt validation split for training-time eval."
    )
    parser.add_argument(
        "--validation-input",
        type=Path,
        default=Path("work/sft/en/splits/validation.jsonl"),
        help="Prompt-clustered validation JSONL to sample from.",
    )
    parser.add_argument(
        "--prompt-output",
        type=Path,
        default=Path("work/sft/en/splits/validation_train_time.jsonl"),
        help="Prompt-clustered train-time validation output JSONL.",
    )
    parser.add_argument(
        "--flat-output",
        type=Path,
        default=Path("work/sft/en/splits/flat/validation_train_time.jsonl"),
        help="Flat train-time validation output JSONL for SFT preprocessing.",
    )
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=Path("work/sft/en/splits/validation_train_time_manifest.json"),
        help="Manifest path for sampling metadata and counts.",
    )
    parser.add_argument(
        "--cap",
        type=int,
        default=2048,
        help="Maximum number of validation prompts/examples to keep.",
    )
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--source-preference",
        action="append",
        help="Source run priority for preferred-one selection. Repeat in priority order.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    source_preference = parse_source_preference(args.source_preference)

    rows = load_jsonl(args.validation_input)
    sampled = stratified_sample(rows, cap=args.cap, seed=args.seed)
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

    write_prompt_jsonl(args.prompt_output, prompt_records)
    flat_stats = write_flat_jsonl(args.flat_output, prompt_records)

    manifest = {
        "input_path": str(args.validation_input),
        "prompt_output_path": str(args.prompt_output),
        "flat_output_path": str(args.flat_output),
        "cap": args.cap,
        "seed": args.seed,
        "selection": "preferred-one",
        "source_preference": list(source_preference),
        "input_prompts": len(rows),
        "sampled_prompts": len(sampled),
        "output_prompts": len(prompt_records),
        "skipped_prompt_ids": skipped_prompt_ids,
        "stratum_counts": dict(sorted(stratum_counts.items())),
        "flat": flat_stats.to_dict(),
    }
    write_json(args.manifest_output, manifest)

    print(f"Wrote {len(prompt_records)} prompt rows to {args.prompt_output}")
    print(f"Wrote {flat_stats.output_examples} flat examples to {args.flat_output}")
    print(f"Wrote manifest to {args.manifest_output}")


if __name__ == "__main__":
    main()
