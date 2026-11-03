"""Per-edit audit of the accepted edits: build the blinded sample and score the labels.

Every accepted edit is shown on its own, next to the text of its field just before the
edit (after L0 and L1 and any earlier accepted edit of the same field; for MedXpertQA
without the answer list at the end of the question), with the English
field as reference; the rater picks the better version (A, B) or ``equal``. The sample is
stratified by edit class, and the per-class shares are weighted by the number of
accepted edits in each class.

    python -m swemedmcq.review.edit_audit make  --out runs/review/edit_audit
    python -m swemedmcq.review.edit_audit score --labels review/edit_audit/labels.csv

``make`` writes key.json (which version is the edit, the class and its population) and
three batches of items (batch0-2.json) for the rater. ``score`` reads a CSV with the
columns sample_id, stratum, stratum_population, confidence, edit_is and verdict and
prints the share of better, equal and worse edits per class and per tier.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

from swemedmcq import paths
from swemedmcq.refine.cli.prepare import shard_of
from swemedmcq.refine.medxpertqa import strip_sv_tail

SEED = 2026
NUM_SHARDS = 8  # shards of the refinement run; fixes the order in which edits are listed
# Edits sampled per class.
SAMPLE = {"T1_units": 6, "T1_drug": 14, "T2_drug_integr": 6, "T2_term_a": 22,
          "T2_accuracy": 22, "T2_leak": 22, "T2_det": 14, "T2_grammar": 30}
CLASS_NAMES = {
    "T1_drug": "Drug name -> INN", "T1_units": "Missing unit",
    "T2_grammar": "Grammar", "T2_leak": "English left in Swedish", "T2_accuracy": "Meaning",
    "T2_term_a": "Term-base terminology", "T2_drug_integr": "Drug name, reworded",
    "T2_det": "Spelling/gender",
}


def read_jsonl(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def field_text(row: dict, field: str, lang: str) -> str:
    prefix = "sv_" if lang == "sv" else ""
    if field == "question":
        return row[prefix + "question"]
    return row[prefix + "options"][field.split(":", 1)[1]]


def edit_class(cand: dict, findings: list[dict], tier: int | None) -> str:
    routes = {f.get("route") or "" for f in findings}
    kind = cand["kind"]
    if kind == "units" or (kind == "drug" and tier == 1):
        return "T1_" + kind
    if kind == "drug":
        return "T2_drug_integr"
    if kind == "term_a":
        return "T2_term_a"
    for route, name in (("ACCURACY", "T2_accuracy"), ("LEAK", "T2_leak"), ("DET", "T2_det"), ("GRAMMAR", "T2_grammar")):
        if route in routes:
            return name
    return "T2_other"


def accepted_edits(history: Path, normalised: Path) -> list[dict]:
    """Every accepted edit with the text before and after it, in the order of the run's result files."""
    layer3 = {row["prompt_id"]: row for row in read_jsonl(normalised)}
    records = [rec for rec in read_jsonl(history) if (rec.get("loop") or {}).get("accepted")]
    records.sort(key=lambda rec: (shard_of(rec["prompt_id"], NUM_SHARDS), rec["prompt_id"] + ".json"))
    rows, state = [], {}
    for rec in records:
        loop, pid = rec["loop"], rec["prompt_id"]
        findings = {f["id"]: f for f in loop["findings"]}
        cands = {c["id"]: c for c in loop["candidates"]}
        for cid in loop["accepted"]:
            cand = cands[cid]
            fs = [findings[i] for i in cand["finding_ids"] if i in findings]
            key = (pid, cand["field"])
            if key not in state:
                # The loop edits the question without the MedXpertQA answer list, which L0 rebuilds.
                start = field_text(layer3[pid], cand["field"], "sv")
                state[key] = strip_sv_tail(start)[0] if cand["field"] == "question" else start
            before = state[key]
            state[key] = cand["new_text"]
            rows.append({"stratum": edit_class(cand, fs, loop.get("tiers", {}).get(cid)), "pid": pid, "cid": cid,
                         "field": cand["field"], "level": (cand.get("confidence") or {}).get("level"),
                         "en": field_text(layer3[pid], cand["field"], "en"),
                         "before": before, "after": cand["new_text"]})
    return rows


def make(args: argparse.Namespace) -> None:
    rows = accepted_edits(args.history, args.normalised)
    pop = Counter(r["stratum"] for r in rows)
    rng = random.Random(SEED)
    sample = []
    for stratum, n in SAMPLE.items():
        pool = [r for r in rows if r["stratum"] == stratum]
        sample += rng.sample(pool, min(n, len(pool)))
    rng.shuffle(sample)
    key, items = {}, []
    for i, r in enumerate(sample):
        sid = f"S{i:03d}"
        flip = rng.random() < 0.5
        a, b = (r["after"], r["before"]) if flip else (r["before"], r["after"])
        key[sid] = {"stratum": r["stratum"], "prompt_id": r["pid"], "candidate": r["cid"], "field": r["field"],
                    "edit_is": "A" if flip else "B", "confidence": r["level"], "stratum_population": pop[r["stratum"]]}
        items.append({"sample_id": sid, "prompt_id": r["pid"], "field": r["field"], "en": r["en"], "A": a, "B": b})
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "key.json").write_text(json.dumps(key, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    for j in range(3):
        (args.out / f"batch{j}.json").write_text(json.dumps(items[j::3], ensure_ascii=False, indent=1) + "\n",
                                                encoding="utf-8")
    print(json.dumps({"accepted_edits": len(rows), "per_class": dict(pop), "sample": len(items)}, indent=1))


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return round(c - h, 3), round(c + h, 3)


def score(args: argparse.Namespace) -> None:
    with args.labels.open(encoding="utf-8") as handle:
        labels = list(csv.DictReader(handle))
    per: dict[str, Counter] = defaultdict(Counter)
    pop: dict[str, int] = {}
    by_level: dict[str, Counter] = defaultdict(Counter)
    for row in labels:
        verdict = row["verdict"].strip()
        outcome = "equal" if verdict.lower() == "equal" else "better" if verdict == row["edit_is"] else "worse"
        per[row["stratum"]][outcome] += 1
        pop[row["stratum"]] = int(row["stratum_population"])
        if row["stratum"].startswith("T2"):
            by_level[row["confidence"]][outcome] += 1
    out: dict = {"classes": {}, "tiers": {}}
    print(f"{'class':26s} {'edits':>5s} {'n':>3s} {'better%':>8s} {'equal%':>7s} {'worse%':>7s}  worse 95% CI")
    for stratum in sorted(per):
        c, n = per[stratum], sum(per[stratum].values())
        shares = {o: 100 * c[o] / n for o in ("better", "equal", "worse")}
        out["classes"][stratum] = {"name": CLASS_NAMES.get(stratum, stratum), "edits": pop[stratum], "n": n,
                                   **shares, "worse_wilson95": wilson(c["worse"], n)}
        print(f"{CLASS_NAMES.get(stratum, stratum):26s} {pop[stratum]:5d} {n:3d} {shares['better']:8.1f} "
              f"{shares['equal']:7.1f} {shares['worse']:7.1f}  {wilson(c['worse'], n)}")
    # Tier totals: class shares weighted by the number of accepted edits in the class.
    for tier in ("T1", "T2"):
        strata = [s for s in per if s.startswith(tier)]
        total = sum(pop[s] for s in strata)
        shares = {o: sum(pop[s] * per[s][o] / sum(per[s].values()) for s in strata) * 100 / total
                  for o in ("better", "equal", "worse")}
        out["tiers"][tier] = {"edits": total, "n": sum(sum(per[s].values()) for s in strata), **shares}
        print(f"{'tier ' + tier[1] + ' (weighted)':26s} {total:5d} {out['tiers'][tier]['n']:3d} {shares['better']:8.1f} "
              f"{shares['equal']:7.1f} {shares['worse']:7.1f}")
    out["tier2_by_confidence"] = {k: dict(v) for k, v in sorted(by_level.items())}
    print("tier 2 by confidence level:", out["tier2_by_confidence"])
    if args.out:
        args.out.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make", help="Build the blinded sample.")
    m.add_argument("--history", type=Path, default=paths.ROOT / "data/test/edit_history.jsonl.gz")
    m.add_argument("--normalised", type=Path, default=paths.ROOT / "data/test/layers/layer3_locale.jsonl")
    m.add_argument("--out", type=Path, required=True)
    s = sub.add_parser("score", help="Score a labelled sample.")
    s.add_argument("--labels", type=Path, default=paths.ROOT / "review/edit_audit/labels.csv")
    s.add_argument("--out", type=Path, help="Optional JSON output.")
    args = ap.parse_args(argv)
    make(args) if args.cmd == "make" else score(args)


if __name__ == "__main__":
    main()
