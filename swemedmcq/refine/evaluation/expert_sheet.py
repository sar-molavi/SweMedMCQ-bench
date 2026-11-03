"""Build the blinded expert-review sheets (CSV + XLSX) and score them.

    python -m swemedmcq.refine.evaluation.expert_sheet make  --run-dir runs/refine/full [--n-pairwise 200] [--exclude-ids ids.txt]
    python -m swemedmcq.refine.evaluation.expert_sheet score --run-dir runs/refine/full [--filled runs/refine/full/expert/expert_review.xlsx]

Sheets:
  1 pairwise  stratified sample of `refined` items; the draft (after the same
              mechanical layers L0+L1) and the refined item in random order (A/B).
              level strata (--strata level): the item's confidence level (lowest level of its accepted edits:
              high / medium / deterministic), optionally x edit class (--strata level_class);
              the score reports harm per level.
              tier strata (--strata tier, default): tier1 = items with code-made edits only
              (draft vs layer4a_auto); tier2 = items with LLM-made edits, shown against their
              tier-1 text (layer4a_auto vs layer4_refined), so the harm of the LLM edits is
              measured on its own.
  2 findings  accepted findings of the sampled items + a random sample of dismissed and vetoed findings.
  3 control   samples of `clean` and `disputed_*` items: does the Swedish contain an error?

The blinding key (which of A/B is refined), strata sizes and inclusion
probabilities go to `expert/key.json`, never into the sheets.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

from ..io import atomic_write_json, read_jsonl, stable_hash
from ..prompting import ordered_fields
from ..types import Item
from .xlsx import read_xlsx, write_xlsx

SEED = 20261006
PAIRWISE_COLS = ["sample_id", "dataset", "english_item", "swedish_A", "swedish_B",
                 "verdict (A better / B better / equal / both wrong)", "meaning_changed (y/n)", "comment"]
FINDING_COLS = ["finding_uid", "dataset", "field", "english_field", "swedish_field", "flagged_swedish_span",
                "claimed_problem", "label (real error / not an error)", "comment"]
CONTROL_COLS = ["control_id", "dataset", "english_item", "swedish_item",
                "has_error (y/n)", "which error (quote + short description)", "severity (minor/major/critical)",
                "comment"]


def item_block(fields: dict[str, str]) -> str:
    return "\n".join(f"[{k}] {fields[k]}" for k in ordered_fields(fields))


def load(run_dir: Path) -> tuple[dict[str, Item], dict[str, dict]]:
    prov = {p["prompt_id"]: p for p in read_jsonl(run_dir / "release" / "provenance.jsonl")}
    items = {}
    for path in sorted((run_dir / "prep").glob("shard_*.jsonl")):
        for r in read_jsonl(path):
            it = Item.model_validate(r)
            if it.prompt_id in prov:
                items[it.prompt_id] = it
    return items, prov


EDIT_CLASS_PRIORITY = ("retranslate", "llm_edit", "term_a", "drug", "termbase", "units")
LEVEL_ORDER = ("medium", "high", "deterministic")


def edit_class(rec: dict) -> str:
    """Most invasive kind among the item's accepted edits."""
    kinds = {c["kind"] for c in rec["candidates"] if c["id"] in set(rec["accepted"])}
    for k in EDIT_CLASS_PRIORITY:
        if k in kinds:
            return k
    return "other"


def confidence_level(rec: dict) -> str:
    """Item confidence = the lowest confidence level of its accepted edits ("v1" for records without levels)."""
    acc = set(rec["accepted"])
    levels = {(c.get("confidence") or {}).get("level") for c in rec["candidates"] if c["id"] in acc}
    if None in levels or not levels:
        return "v1"
    for lv in LEVEL_ORDER:
        if lv in levels:
            return lv
    return "other"


def item_tier(rec: dict) -> str:
    """tier1: only code-made edits accepted; tier2: at least one LLM-made edit ("v2" for records without tiers)."""
    t = rec.get("tiers")
    if not t:
        return "v2"
    return "tier2" if 2 in t.values() else "tier1"


def reference_text(it, rec: dict, mode: str) -> dict:
    """The text the refined item is compared with: the tier-1 text for tier-2 items in tier mode,
    otherwise the draft after L0+L1."""
    if mode == "tier" and item_tier(rec) == "tier2" and rec.get("sv_auto"):
        return rec["sv_auto"]
    return it.sv_l1


def stratum(rec: dict, mode: str = "level") -> str:
    """Sampling stratum of a refined item: its release tier ("tier"), confidence level ("level")
    or level and edit class ("level_class"); "class" stratifies by edit class only."""
    if mode == "tier":
        t = item_tier(rec)
        return t if t != "tier2" else f"tier2/{confidence_level(rec)}"
    if mode == "class":
        return edit_class(rec)
    if mode == "level_class":
        return f"{confidence_level(rec)}/{edit_class(rec)}"
    return confidence_level(rec)


def sample_stratified(ids_by_stratum: dict[str, list[str]], n: int, min_per: int, rng: random.Random):
    """Allocate n proportionally with a floor of min_per per stratum; returns (sample, inclusion probs)."""
    total = sum(len(v) for v in ids_by_stratum.values())
    alloc = {}
    for s, ids in ids_by_stratum.items():
        k = max(min_per, round(n * len(ids) / total)) if total else 0
        alloc[s] = min(k, len(ids))
    sample, prob = [], {}
    for s, ids in sorted(ids_by_stratum.items()):
        chosen = rng.sample(sorted(ids), alloc[s])
        sample += [(s, i) for i in chosen]
        prob[s] = alloc[s] / len(ids) if ids else 0.0
    return sample, prob


def make(args: argparse.Namespace) -> None:
    items, prov = load(args.run_dir)
    exclude = set()
    if args.exclude_ids:
        exclude = {l.split()[0] for l in args.exclude_ids.read_text().splitlines() if l.strip() and not l.startswith("#")}
    seed = args.seed
    rng = random.Random(seed)
    refined = defaultdict(list)
    for pid, p in prov.items():
        if pid in exclude or not p["loop"]:
            continue
        if p["status"] == "refined":
            refined[stratum(p["loop"], args.strata)].append(pid)
    sample, prob = sample_stratified(refined, args.n_pairwise, args.min_per_stratum, rng)

    out = args.run_dir / "expert"
    key: dict = {"seed": seed, "strata_mode": args.strata, "strata_sizes": {s: len(v) for s, v in refined.items()},
                 "inclusion_prob": prob, "pairwise": {}, "findings": {}, "control": {}}
    pair_rows = [PAIRWISE_COLS]
    for n, (s, pid) in enumerate(sample, start=1):
        it, rec = items[pid], prov[pid]["loop"]
        draft, refined_fields = item_block(reference_text(it, rec, args.strata)), item_block(rec["sv_final"])
        refined_is_a = int(stable_hash(seed, pid)[:2], 16) % 2 == 0
        a, b = (refined_fields, draft) if refined_is_a else (draft, refined_fields)
        sid = f"P{n:04d}"
        key["pairwise"][sid] = {"prompt_id": pid, "stratum": s, "refined": "A" if refined_is_a else "B",
                                "level": confidence_level(rec), "edit_class": edit_class(rec),
                                "tier": item_tier(rec),
                                "reference": "layer4a_auto" if reference_text(it, rec, args.strata) is not it.sv_l1 else "layer3"}
        pair_rows.append([sid, it.dataset, item_block(it.en), a, b, "", "", ""])

    find_rows = [FINDING_COLS]
    chosen = {pid for _, pid in sample}
    pool_dismissed = []
    for pid, p in prov.items():
        if pid in exclude or not p["loop"]:
            continue
        for f in p["loop"]["findings"]:
            if f["source"] in {"decoy", "corrupt"}:
                continue
            if pid in chosen and f["status"] == "resolved":
                _add_finding(find_rows, key, items[pid], f, "accepted")
            elif f["status"] in {"dismissed", "vetoed", "rejected_by_judges"}:
                pool_dismissed.append((f["status"], pid, f))
    rng.shuffle(pool_dismissed)
    for status, pid, f in pool_dismissed[: args.n_dismissed]:
        _add_finding(find_rows, key, items[pid], f, status)

    ctrl_rows = [CONTROL_COLS]
    clean = sorted(pid for pid, p in prov.items() if p["status"] in {"clean", "mechanical_only"} and pid not in exclude)
    disputed = sorted(pid for pid, p in prov.items() if p["status"].startswith("disputed") and pid not in exclude)
    picks = [("clean", pid) for pid in rng.sample(clean, min(args.n_clean, len(clean)))] + \
            [("disputed", pid) for pid in rng.sample(disputed, min(args.n_disputed, len(disputed)))]
    rng.shuffle(picks)
    key["control_strata"] = {"clean": len(clean), "disputed": len(disputed)}
    for n, (s, pid) in enumerate(picks, start=1):
        cid = f"C{n:04d}"
        key["control"][cid] = {"prompt_id": pid, "stratum": s}
        ctrl_rows.append([cid, items[pid].dataset, item_block(items[pid].en), item_block(items[pid].sv_l1), "", "", "", ""])

    out.mkdir(parents=True, exist_ok=True)
    sheets = {"1_pairwise": pair_rows, "2_findings": find_rows, "3_control": ctrl_rows}
    for name, rows in sheets.items():
        with (out / f"{name}.csv").open("w", encoding="utf-8", newline="") as fh:
            csv.writer(fh).writerows(rows)
    write_xlsx(out / "expert_review.xlsx", sheets,
               widths={"1_pairwise": [10, 12, 60, 60, 60, 22, 14, 30],
                       "2_findings": [12, 12, 10, 50, 50, 25, 50, 22, 30],
                       "3_control": [10, 12, 60, 60, 12, 40, 14, 30]})
    atomic_write_json(out / "key.json", key)
    print(f"pairwise {len(pair_rows) - 1}, findings {len(find_rows) - 1}, control {len(ctrl_rows) - 1} -> {out}")


def _add_finding(rows, key, it: Item, f: dict, status: str) -> None:
    uid = f"F{len(rows):05d}"
    key["findings"][uid] = {"prompt_id": it.prompt_id, "finding_id": f["id"], "status": status,
                            "source": f["source"], "category": f["category"], "severity": f["severity"],
                            "route": f.get("route", "")}
    claim = f'{f["category"]} ({f["severity"]}): {f["explanation"]}'
    rows.append([uid, it.dataset, f["field"], it.en[f["field"]], it.sv_l1[f["field"]],
                 f["sv_quote"] or f["en_quote"], claim, "", ""])


# ---------------------------------------------------------------------------- scoring
def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return round((c - h) / d, 4), round((c + h) / d, 4)


def _rows(path: Path, sheet: str) -> list[dict]:
    if path.suffix == ".xlsx":
        rows = read_xlsx(path)[sheet]
    else:
        with (path.parent / f"{sheet}.csv").open(encoding="utf-8") as fh:
            rows = list(csv.reader(fh))
    head = rows[0]
    return [dict(zip(head, r + [""] * (len(head) - len(r)))) for r in rows[1:] if any(r)]


def score(args: argparse.Namespace) -> None:
    out = args.run_dir / "expert"
    key = json.loads((out / "key.json").read_text())
    filled = args.filled or out / "expert_review.xlsx"
    res: dict = {}
    verdicts = Counter()
    by_stratum: dict[str, Counter] = defaultdict(Counter)
    by_level: dict[str, Counter] = defaultdict(Counter)
    by_level_class: dict[str, Counter] = defaultdict(Counter)
    by_tier: dict[str, Counter] = defaultdict(Counter)
    meaning = Counter()
    for r in _rows(filled, "1_pairwise"):
        v = r[PAIRWISE_COLS[5]].strip().lower()
        if not v:
            continue
        info = key["pairwise"][r["sample_id"]]
        if v.startswith(("a", "b")) and "better" in v:
            outcome = "refined_better" if v[0].upper() == info["refined"] else "draft_better"
        elif v.startswith("eq"):
            outcome = "equal"
        else:
            outcome = "both_wrong"
        verdicts[outcome] += 1
        by_stratum[info["stratum"]][outcome] += 1
        by_tier[info.get("tier", "v2")][outcome] += 1
        by_level[info.get("level", "v1")][outcome] += 1
        by_level_class[f'{info.get("level", "v1")}/{info.get("edit_class", info["stratum"])}'][outcome] += 1
        meaning[r[PAIRWISE_COLS[6]].strip().lower()[:1] or "?"] += 1
    n = sum(verdicts.values())
    harm = verdicts["draft_better"]
    res["pairwise"] = {"n": n, "counts": dict(verdicts), "harm_rate": round(harm / n, 4) if n else None,
                       "harm_wilson95": wilson(harm, n), "win_rate": round(verdicts["refined_better"] / n, 4) if n else None,
                       "meaning_changed": dict(meaning), "by_stratum": {k: dict(v) for k, v in by_stratum.items()},
                       # harm reported separately for high and medium confidence
                       "by_level": {k: {**dict(v), "harm_rate": round(v["draft_better"] / sum(v.values()), 4),
                                        "harm_wilson95": wilson(v["draft_better"], sum(v.values()))}
                                    for k, v in sorted(by_level.items())},
                       "by_level_class": {k: dict(v) for k, v in sorted(by_level_class.items())},
                       # tier 1 vs the draft; tier 2 vs its tier-1 text (harm of the LLM edits alone)
                       "by_tier": {k: {**dict(v), "harm_rate": round(v["draft_better"] / sum(v.values()), 4),
                                       "harm_wilson95": wilson(v["draft_better"], sum(v.values()))}
                                   for k, v in sorted(by_tier.items())}}
    # inverse-probability-weighted harm rate over the refined population
    w_harm = w_tot = 0.0
    for s, c in by_stratum.items():
        p = key["inclusion_prob"].get(s) or 0
        if p:
            w_harm += c["draft_better"] / p
            w_tot += sum(c.values()) / p
    res["pairwise"]["harm_rate_ipw"] = round(w_harm / w_tot, 4) if w_tot else None
    non_ties = verdicts["refined_better"] + verdicts["draft_better"]
    if non_ties:
        k = verdicts["refined_better"]
        p_sign = sum(math.comb(non_ties, i) for i in range(k, non_ties + 1)) / 2 ** non_ties
        res["pairwise"]["sign_test_one_sided_p"] = p_sign

    fstats: dict[str, Counter] = defaultdict(Counter)
    for r in _rows(filled, "2_findings"):
        lab = r[FINDING_COLS[7]].strip().lower()
        if not lab:
            continue
        info = key["findings"][r["finding_uid"]]
        fstats[info["status"]]["real" if lab.startswith("real") else "not"] += 1
    res["findings"] = {s: {**dict(c), "real_rate_wilson95": wilson(c["real"], c["real"] + c["not"])}
                       for s, c in fstats.items()}

    cstats: dict[str, Counter] = defaultdict(Counter)
    for r in _rows(filled, "3_control"):
        lab = r[CONTROL_COLS[4]].strip().lower()
        if not lab:
            continue
        cstats[key["control"][r["control_id"]]["stratum"]]["error" if lab.startswith("y") else "ok"] += 1
    res["control"] = {s: {**dict(c), "error_rate_wilson95": wilson(c["error"], c["error"] + c["ok"])}
                      for s, c in cstats.items()}
    atomic_write_json(out / "scores.json", res)
    print(json.dumps(res, indent=1))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make")
    m.add_argument("--run-dir", type=Path, required=True)
    m.add_argument("--n-pairwise", type=int, default=200)
    m.add_argument("--min-per-stratum", type=int, default=40)
    m.add_argument("--n-dismissed", type=int, default=100)
    m.add_argument("--n-clean", type=int, default=100)
    m.add_argument("--n-disputed", type=int, default=50)
    m.add_argument("--exclude-ids", type=Path)
    m.add_argument("--seed", type=int, default=SEED,
                   help=f"Sampling and A/B-order seed (default {SEED}, the released sheets).")
    m.add_argument("--strata", choices=("tier", "level", "level_class", "class"), default="tier",
                   help="Pairwise strata: release tier x level (default), confidence level, level x edit "
                        "class, or edit class.")
    s = sub.add_parser("score")
    s.add_argument("--run-dir", type=Path, required=True)
    s.add_argument("--filled", type=Path, help="Filled expert_review.xlsx (or any of the CSVs in the same folder).")
    args = ap.parse_args(argv)
    make(args) if args.cmd == "make" else score(args)


if __name__ == "__main__":
    main()
