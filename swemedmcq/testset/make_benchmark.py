"""Build the benchmark file and the layer files from a refinement run.

Reads ``<run>/release/`` (written by ``swemedmcq.refine.cli.assemble``) and the gold
letters, and writes one JSON object per item and line, in the item order of the run:

  benchmark.jsonl               layer 4a: layer 3 + tier-1 (code-made) edits; the benchmark
  layers/layer1_draft.jsonl     Swedish text of the GPT-5.4 draft
  layers/layer2_hygiene.jsonl   draft + typography (L0)
  layers/layer3_locale.jsonl    draft + typography + Swedish number style (L1)
  layers/layer4_refined.jsonl   layer 3 + tier-1 and tier-2 edits

with the fields prompt_id, dataset, question, options, answer, sv_question, sv_options,
category, tier, core, expert_queue, hard. It also writes the edit history of every item
(edit_history.jsonl.gz, from provenance.jsonl), the run summary (refinement_report.json,
from report.json) and the queued items (expert_queue.jsonl).

``category`` is the refinement status of the item,
``tier`` the highest tier among its accepted edits (0: none, 1: code-made only,
2: at least one model-written edit), ``expert_queue`` whether it was queued for human
review, ``core`` the reported set: not disputed and not in the expert queue, and
``hard`` whether the item is in the hard subset of its source dataset
(``<source>/<dataset>/test_hard.jsonl`` of MedicalAgentsBench).

    python -m swemedmcq.testset.make_benchmark --run-dir runs/refine/full --answers <gold> --out-dir data/test
"""

from __future__ import annotations

import argparse
import gzip
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from swemedmcq import paths

LAYERS = {
    "benchmark.jsonl": "layer4a_auto.jsonl",
    "layers/layer1_draft.jsonl": "layer1_draft.jsonl",
    "layers/layer2_hygiene.jsonl": "layer2_hygiene.jsonl",
    "layers/layer3_locale.jsonl": "layer3_locale.jsonl",
    "layers/layer4_refined.jsonl": "layer4_refined.jsonl",
}


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def read_answers(path: Path) -> dict[str, str]:
    if path.suffix == ".jsonl":
        return {r["prompt_id"]: r.get("answer", r.get("gold_answer")) for r in read_jsonl(path)}
    return json.loads(path.read_text(encoding="utf-8"))


def annotations(release: Path) -> dict[str, dict[str, Any]]:
    """category, tier, expert_queue and core per item, from provenance.jsonl and expert_queue.jsonl."""
    queue = {row["prompt_id"] for row in read_jsonl(release / "expert_queue.jsonl")}
    out = {}
    for prov in read_jsonl(release / "provenance.jsonl"):
        pid, loop, status = prov["prompt_id"], prov.get("loop") or {}, prov["status"]
        tiers = set((loop.get("tiers") or {}).values())
        out[pid] = {
            "category": status,
            "tier": max(tiers) if tiers else 0,
            # Core rule: drop the disputed items and the items queued for human review.
            "core": not status.startswith("disputed") and pid not in queue,
            "expert_queue": pid in queue,
        }
    return out


def read_hard(source: Path, datasets: set[str]) -> set[str]:
    """prompt_ids of the MedicalAgentsBench hard subsets (``<dataset>_<realidx>``)."""
    hard = set()
    for dataset in sorted(datasets):
        path = source / dataset / "test_hard.jsonl"
        if not path.exists():
            raise SystemExit(f"missing {path}: --source must point to the MedicalAgentsBench data")
        hard |= {f"{dataset}_{row['realidx']}" for row in read_jsonl(path)}
    return hard


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True, help="Refinement run (with release/).")
    parser.add_argument("--answers", type=Path, required=True,
                        help="Gold letters: JSON {prompt_id: letter} or JSONL with prompt_id and answer.")
    parser.add_argument("--source", type=Path, default=None,
                        help="MedicalAgentsBench data with <dataset>/test_hard.jsonl (default: paths.yaml `source_benchmark`).")
    parser.add_argument("--out-dir", type=Path, default=None, help="Default: the folder of paths.yaml `benchmark`.")
    args = parser.parse_args()

    release = args.run_dir / "release"
    out_dir = args.out_dir or paths.path("benchmark").parent
    answers = read_answers(args.answers)
    notes = annotations(release)
    hard = read_hard(args.source or paths.path("source_benchmark"), {pid.split("_", 1)[0] for pid in notes})
    for name, layer in LAYERS.items():
        target = out_dir / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as handle:
            for row in read_jsonl(release / layer):
                pid = row["prompt_id"]
                handle.write(json.dumps({
                    "prompt_id": pid,
                    "dataset": pid.split("_", 1)[0],
                    "question": row["question"],
                    "options": row["options"],
                    "answer": answers[pid],
                    "sv_question": row["sv_question"],
                    "sv_options": row["sv_options"],
                    **notes[pid],
                    "hard": pid in hard,
                }, ensure_ascii=False) + "\n")
        print(f"wrote {target}")

    # Edit history: the provenance record of every item, without the internal run label.
    target = out_dir / "edit_history.jsonl.gz"
    with target.open("wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gz:
        for prov in read_jsonl(release / "provenance.jsonl"):
            (prov.get("loop") or {}).pop("loop_version", None)
            gz.write((json.dumps(prov, ensure_ascii=False) + "\n").encode("utf-8"))
    print(f"wrote {target}")
    shutil.copyfile(release / "report.json", out_dir / "refinement_report.json")
    shutil.copyfile(release / "expert_queue.jsonl", out_dir / "expert_queue.jsonl")
    print(json.dumps({"items": len(notes), "core": sum(n["core"] for n in notes.values()),
                      "hard": len(hard & set(notes)), "hard_core": sum(n["core"] for p, n in notes.items() if p in hard),
                      "category": dict(Counter(n["category"] for n in notes.values())),
                      "tier": dict(Counter(n["tier"] for n in notes.values()))}, indent=1))


if __name__ == "__main__":
    main()
