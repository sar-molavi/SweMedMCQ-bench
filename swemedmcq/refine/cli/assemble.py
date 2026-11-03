"""Step 3 (CPU, once): assemble the released layers, provenance and report.

    python -m swemedmcq.refine.cli.assemble --run-dir runs/refine/full [--input data/test/layers/layer1_draft.jsonl]

Outputs (`<run>/release/`), all in the schema of the draft file
(prompt_id, question, options, sv_question, sv_options):

  layer1_draft.jsonl     the GPT-5.4 drafts, byte-identical to the input file
  layer2_hygiene.jsonl   + L0 (typography, control characters, MedXpertQA Swedish tail rebuilt)
  layer3_locale.jsonl    + L1 (Swedish number locale)
  layer4a_auto.jsonl     + tier-1 edits only (code-made: secondary units, drug-name substitution)
  layer4_refined.jsonl   + all accepted refinement edits (tier 1 and the LLM-made tier 2)

plus `provenance.jsonl` (one full record per item), `expert_queue.jsonl` and
`report.json`.  Refuses to run unless every shard is DONE (or --allow-partial:
missing items keep their layer-3 text and are listed in the report).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..evaluation.report import build_report
from ..io import atomic_write_json, atomic_write_text, read_jsonl
from ..medxpertqa import build_sv_tail
from ..types import Item, field_label
from swemedmcq.paths import path as repo_path

from .prepare import as_options


def to_row(item: Item, raw: dict, sv: dict[str, str]) -> dict:
    options = {field_label(k): v for k, v in sv.items() if k.startswith("option:")}
    options = {k: options[k] for k in raw["sv_options"]}  # keep the original option order
    question = sv["question"]
    if item.en_tail is not None and not item.tail_exception:
        question = question.rstrip() + build_sv_tail(options, item.tail_format)
    return {"prompt_id": raw["prompt_id"], "question": raw["question"], "options": raw["options"],
            "sv_question": question, "sv_options": options}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--input", type=Path, default=None, help="Draft JSONL (default: paths.yaml `draft`).")
    ap.add_argument("--allow-partial", action="store_true")
    args = ap.parse_args(argv)
    args.input = args.input or repo_path("draft")

    prep = args.run_dir / "prep"
    manifest = json.loads((prep / "manifest.json").read_text())
    n_shards = manifest["num_shards"]
    not_done = [k for k in range(n_shards) if not (args.run_dir / "shards" / f"{k:03d}" / "DONE").exists()]
    if not_done and not args.allow_partial:
        raise SystemExit(f"shards not DONE: {not_done}; finish them or use --allow-partial")

    items: dict[str, Item] = {}
    for k in range(n_shards):
        for r in read_jsonl(prep / f"shard_{k:03d}.jsonl"):
            it = Item.model_validate(r)
            items[it.prompt_id] = it
    results: dict[str, dict] = {}
    for k in range(n_shards):
        for p in (args.run_dir / "shards" / f"{k:03d}" / "results").glob("*.json"):
            rec = json.loads(p.read_text())
            results[rec["prompt_id"]] = rec

    raw_lines = [l for l in args.input.read_text(encoding="utf-8").splitlines(keepends=True) if l.strip()]
    out = args.run_dir / "release"
    layers = {name: [] for name in ("layer2_hygiene", "layer3_locale", "layer4a_auto", "layer4_refined")}
    draft_lines, provenance, queue, missing = [], [], [], []
    for line in raw_lines:
        raw = json.loads(line)
        raw["options"], raw["sv_options"] = as_options(raw["options"]), as_options(raw["sv_options"])
        pid = raw["prompt_id"]
        if pid not in items:
            continue  # subset runs (smoke)
        draft_lines.append(line if line.endswith("\n") else line + "\n")
        it = items[pid]
        rec = results.get(pid)
        if rec is None:
            missing.append(pid)
        final = rec["sv_final"] if rec else it.sv_l1
        auto = (rec.get("sv_auto") or it.sv_l1) if rec else it.sv_l1  # records without tier-1 text: layer 3
        layers["layer2_hygiene"].append(to_row(it, raw, it.sv_l0))
        layers["layer3_locale"].append(to_row(it, raw, it.sv_l1))
        layers["layer4a_auto"].append(to_row(it, raw, auto))
        layers["layer4_refined"].append(to_row(it, raw, final))
        prov = {"prompt_id": pid, "status": rec["status"] if rec else "missing",
                "changes_l0": [c.model_dump() for c in it.changes_l0],
                "changes_l1": [c.model_dump() for c in it.changes_l1],
                "corrupt_fields": it.corrupt_fields, "tail_exception": it.tail_exception,
                "loop": rec}
        provenance.append(prov)
        if rec is None or rec["status"] in {"disputed_critical", "failed"} or it.corrupt_fields \
                or rec.get("residual_critical"):
            queue.append({"prompt_id": pid, "status": prov["status"], "corrupt_fields": it.corrupt_fields,
                          "residual_critical": bool(rec and rec.get("residual_critical"))})

    atomic_write_text(out / "layer1_draft.jsonl", "".join(draft_lines))
    for name, rows in layers.items():
        atomic_write_text(out / f"{name}.jsonl", "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    atomic_write_text(out / "provenance.jsonl",
                      "".join(json.dumps(p, ensure_ascii=False) + "\n" for p in provenance))
    atomic_write_text(out / "expert_queue.jsonl", "".join(json.dumps(q, ensure_ascii=False) + "\n" for q in queue))

    report = build_report(items, results)
    report["missing_items"] = missing
    report["shards_not_done"] = not_done
    atomic_write_json(out / "report.json", report)
    print(json.dumps({k: report[k] for k in ("n_items", "status_counts")}, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
