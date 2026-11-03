"""Split the filtered English SFT pool into train and validation prompts.

Splitting is done per prompt (all responses of a prompt go to the same side).
Prompts are stratified by ``dataset_name | response_count_bucket | has_reasoning``;
strata with fewer than ``--min-stratum-size`` prompts fall back to ``dataset_name``.
Within each stratum (in sorted key order) the prompt ids are shuffled with one
``random.Random(seed)`` and ``round(n * fraction)`` of them (at least 1 for strata
of the minimum size, never all) go to validation. Prompts without kept responses
are left out.

The released split is in ``data/train/split/{train,validation}_prompt_ids.txt.gz``.
With ``--train-ids`` and ``--validation-ids`` the prompts are assigned from these files
instead of being drawn, which reproduces the released split exactly; prompts listed in
neither file are left out. The Swedish split follows the same ids (``split_sft_dataset``).

    python -m swemedmcq.distill.split_english_sft \\
        --input-path work/sft/en/sft_filtered_en_responses.jsonl --output-dir work/sft/en/splits \\
        --train-ids data/train/split/train_prompt_ids.txt.gz \\
        --validation-ids data/train/split/validation_prompt_ids.txt.gz
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from swemedmcq.distill.prompt_ids import read_prompt_ids


def load_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


def has_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def response_count_bucket(count: int) -> str:
    if count <= 1:
        return "1"
    if count == 2:
        return "2"
    return "3+"


def stratum_keys(rows: list[dict[str, Any]], min_stratum_size: int) -> list[str]:
    full = []
    for row in rows:
        responses = row.get("responses") or []
        has_reasoning = any(has_text(response.get("reasoning")) for response in responses)
        full.append(f"{row.get('dataset_name')}|{response_count_bucket(len(responses))}|{has_reasoning}")
    counts = Counter(full)
    return [key if counts[key] >= min_stratum_size else str(row.get("dataset_name"))
            for key, row in zip(full, rows)]


def validation_ids(rows: list[dict[str, Any]], *, seed: int, fraction: float, min_stratum_size: int) -> set[str]:
    groups: dict[str, list[str]] = defaultdict(list)
    for key, row in zip(stratum_keys(rows, min_stratum_size), rows):
        groups[key].append(row["prompt_id"])
    rng = random.Random(seed)
    chosen: set[str] = set()
    for key in sorted(groups):
        ids = list(groups[key])
        rng.shuffle(ids)
        n_val = round(len(ids) * fraction)
        if len(ids) >= min_stratum_size and fraction > 0:
            n_val = max(1, n_val)
        n_val = min(n_val, len(ids) - 1) if len(ids) > 1 else 0
        chosen.update(ids[:n_val])
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-path", type=Path, required=True, help="Filtered prompt-clustered pool (filter_sft_en_responses).")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--validation-fraction", type=float, default=0.05)
    parser.add_argument("--min-stratum-size", type=int, default=4)
    parser.add_argument("--train-ids", type=Path, help="Released train prompt ids (with --validation-ids).")
    parser.add_argument("--validation-ids", type=Path, help="Released validation prompt ids (with --train-ids).")
    args = parser.parse_args()
    if (args.train_ids is None) != (args.validation_ids is None):
        parser.error("--train-ids and --validation-ids go together")

    rows = [row for row in load_jsonl(args.input_path) if row.get("responses")]
    if args.train_ids:
        train_ids, val = read_prompt_ids(args.train_ids), read_prompt_ids(args.validation_ids)
        if train_ids & val:
            raise ValueError("a prompt id is in both the train and the validation file")
        rows = [row for row in rows if row["prompt_id"] in train_ids or row["prompt_id"] in val]
    else:
        val = validation_ids(rows, seed=args.seed, fraction=args.validation_fraction,
                             min_stratum_size=args.min_stratum_size)
    train_rows = [row for row in rows if row["prompt_id"] not in val]
    validation_rows = [row for row in rows if row["prompt_id"] in val]
    write_jsonl(args.output_dir / "train.jsonl", train_rows)
    write_jsonl(args.output_dir / "validation.jsonl", validation_rows)
    manifest = {
        "input_path": str(args.input_path),
        "train_path": str(args.output_dir / "train.jsonl"),
        "validation_path": str(args.output_dir / "validation.jsonl"),
        "seed": args.seed,
        "validation_fraction": args.validation_fraction,
        "min_stratum_size": args.min_stratum_size,
        "prompt_count": len(rows),
        "train_prompt_count": len(train_rows),
        "validation_prompt_count": len(validation_rows),
        "split": ("released prompt ids" if args.train_ids else
                  "stratified by dataset_name|response_count_bucket|has_reasoning, sparse strata fall back to dataset_name"),
    }
    (args.output_dir / "split_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                                                         encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("prompt_count", "train_prompt_count", "validation_prompt_count")}))


if __name__ == "__main__":
    main()
