"""Step 1 (CPU, once): build prepared items and shards.

    python -m swemedmcq.refine.cli.prepare --run-dir runs/refine/full --num-shards 8 \
        [--input data/test/layers/layer1_draft.jsonl] [--answers data/test/benchmark.jsonl]

Input: the GPT-5.4 draft, one JSON object per line with ``prompt_id``, ``question``,
``options``, ``sv_question`` and ``sv_options`` (options as JSON objects; option
dicts stored as Python-literal strings are accepted too). Answers: a JSON object
``{prompt_id: letter}`` or a JSONL file with ``prompt_id`` and ``answer`` (the
benchmark file); they are only used by the reports (gold-vs-distractor edits).

For every draft: detect corruption on the raw text, apply hygiene (L0), split the
MedXpertQA English answer-list tail and strip any Swedish tail (only when the
English tail matches the options), apply the Swedish number locale (L1), and
assign a shard by a stable hash of the prompt id.

Idempotent: if `prep/manifest.json` already matches the input hash and shard
count, nothing is rebuilt (use --force to rebuild).
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from collections import Counter
from pathlib import Path

from swemedmcq.paths import path as repo_path

from ..io import atomic_write_json, atomic_write_text, fields_from_row, read_jsonl, stable_hash
from ..locale import to_swedish_locale
from ..medxpertqa import split_en_tail, strip_sv_tail, tail_format, tail_matches_options
from ..normalise import hygiene, is_corrupt
from ..types import Change, Item



def as_options(value: dict | str) -> dict[str, str]:
    """Options as a dict; a Python-literal string (``"{'A': ...}"``) is parsed."""
    if isinstance(value, str):
        value = ast.literal_eval(value)
    if not isinstance(value, dict):
        raise ValueError(f"options must be a mapping, got {type(value).__name__}")
    return value


def read_drafts(path: Path) -> list[dict]:
    rows = read_jsonl(path)
    for row in rows:
        row["options"] = as_options(row["options"])
        row["sv_options"] = as_options(row["sv_options"])
    return rows


def read_answers(path: Path) -> dict[str, str]:
    """``{prompt_id: letter}`` from a JSON mapping or a JSONL file with ``answer`` (or ``gold_answer``)."""
    if not path.exists():
        return {}
    if path.suffix == ".jsonl":
        return {r["prompt_id"]: r.get("answer", r.get("gold_answer")) for r in read_jsonl(path)}
    return json.loads(path.read_text())


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def shard_of(prompt_id: str, num_shards: int) -> int:
    return int(stable_hash(prompt_id)[:8], 16) % num_shards


def prepare_item(row: dict, answer: str | None, num_shards: int) -> Item:
    pid = row["prompt_id"]
    en = fields_from_row(row["question"], row["options"])
    sv_draft = fields_from_row(row["sv_question"], row["sv_options"])
    corrupt = [k for k, v in sv_draft.items() if is_corrupt(v)]

    en_tail = fmt = None
    exception = False
    body, tail = split_en_tail(row["question"])
    if tail is not None:
        if tail_matches_options(tail, row["options"]):
            en_tail, fmt = tail, tail_format(tail)
            en["question"] = body
        else:
            exception = True  # keep both questions untouched

    changes_l0: list[Change] = []
    sv_l0: dict[str, str] = {}
    for key, text in sv_draft.items():
        clean, ch = hygiene(text, key)
        sv_l0[key] = clean
        changes_l0.extend(ch)
    if en_tail is not None:
        stripped, removed = strip_sv_tail(sv_l0["question"])
        if removed:
            changes_l0.append(Change(field="question", rule="tail_strip", start=len(stripped),
                                     before=removed, after=""))
            sv_l0["question"] = stripped

    changes_l1: list[Change] = []
    sv_l1: dict[str, str] = {}
    for key, text in sv_l0.items():
        loc, ch = to_swedish_locale(text, key, en_text=en.get(key))  # English reference: thousands/decimals
        sv_l1[key] = loc
        changes_l1.extend(ch)

    return Item(prompt_id=pid, dataset=pid.split("_", 1)[0], answer=answer, en=en, sv_draft=sv_draft,
                sv_l0=sv_l0, sv_l1=sv_l1, en_tail=en_tail, tail_format=fmt, tail_exception=exception,
                corrupt_fields=corrupt, changes_l0=changes_l0, changes_l1=changes_l1,
                shard=shard_of(pid, num_shards))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--input", type=Path, default=None, help="Draft JSONL (default: paths.yaml `draft`).")
    ap.add_argument("--answers", type=Path, default=None, help="Gold letters (default: paths.yaml `benchmark`).")
    ap.add_argument("--num-shards", type=int, required=True)
    ap.add_argument("--ids-file", type=Path, help="Optional subset of prompt ids (one per line), e.g. for smoke runs.")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    args.input = args.input or repo_path("draft")
    args.answers = args.answers or repo_path("benchmark")

    prep = args.run_dir / "prep"
    manifest_path = prep / "manifest.json"
    ids = None
    if args.ids_file:
        ids = [l.split()[0] for l in args.ids_file.read_text().splitlines() if l.strip() and not l.startswith("#")]
    manifest = {
        "input": str(args.input), "input_sha256": file_sha256(args.input),
        "num_shards": args.num_shards, "ids_sha256": stable_hash(ids) if ids else None,
    }
    if manifest_path.exists() and not args.force:
        old = json.loads(manifest_path.read_text())
        if all(old.get(k) == v for k, v in manifest.items()):
            print(f"prep up to date ({old['n_items']} items, {args.num_shards} shards); use --force to rebuild")
            return
        raise SystemExit(f"{manifest_path} exists with different inputs; use --force or a new --run-dir")

    answers = read_answers(args.answers)
    rows = read_drafts(args.input)
    if ids is not None:
        wanted = set(ids)
        rows = [r for r in rows if r["prompt_id"] in wanted]
    items = [prepare_item(r, answers.get(r["prompt_id"]), args.num_shards) for r in rows]

    by_shard: dict[int, list[str]] = {}
    for it in items:
        by_shard.setdefault(it.shard, []).append(it.model_dump_json())
    for k in range(args.num_shards):
        lines = by_shard.get(k, [])
        atomic_write_text(prep / f"shard_{k:03d}.jsonl", "\n".join(lines) + ("\n" if lines else ""))
    stats = {
        "n_items": len(items),
        "per_shard": {k: len(v) for k, v in sorted(by_shard.items())},
        "corrupt_items": sum(1 for it in items if it.corrupt_fields),
        "medxpertqa_tail_split": sum(1 for it in items if it.en_tail),
        "medxpertqa_tail_exceptions": [it.prompt_id for it in items if it.tail_exception],
        "items_changed_l0": sum(1 for it in items if it.changes_l0),
        "items_changed_l1": sum(1 for it in items if it.changes_l1),
        "l0_rules": dict(Counter(c.rule for it in items for c in it.changes_l0)),
        "l1_rules": dict(Counter(c.rule for it in items for c in it.changes_l1)),
    }
    atomic_write_json(prep / "stats.json", stats)
    atomic_write_json(manifest_path, {**manifest, "n_items": len(items)})
    print(json.dumps({k: v for k, v in stats.items() if k != "medxpertqa_tail_exceptions"}, indent=1))


if __name__ == "__main__":
    main()
