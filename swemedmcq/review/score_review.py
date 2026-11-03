"""Assemble and score a set of rated review sheets.

Inputs (``--labels``): the raters' CSVs, named as in ``review/answers_template/``:
  1_pairwise_<batch>_<a|b>.csv, 1_pairwise_adjudicated.csv (disagreements only),
  2_findings_<batch>.csv, 3_control_<batch>.csv, 4_disputed_<batch>.csv, 5_source_1.csv
and the keys in ``--keys``: key.json (sheets 1-3), key_disputed.json, key_source.json
and disputed_strata_pop.json (sheets 4-5).

It fills the full sheets of ``--sheets`` (``review/sheets_full``) with the verdicts,
scores them with ``swemedmcq.refine.evaluation.expert_sheet score`` (harm rate per tier
and confidence level with Wilson intervals, inverse-probability-weighted harm, sign
test, precision of the findings, control error rates) and adds the agreement between
the two pairwise raters, the severity counts and the summaries of sheets 4 and 5.

    python -m swemedmcq.review.score_review --labels review/llm_labels --keys review/keys --out runs/review/llm

``--out`` receives run/expert/ (filled sheets and scores.json), summary.json and labels/.
Scoring ``review/llm_labels`` reproduces ``review/llm_labels/summary.json``.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import glob
import io
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from swemedmcq import paths
from swemedmcq.refine.evaluation import expert_sheet

SHEETS = paths.ROOT / "review" / "sheets_full"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def labels(pattern: str, key: str) -> dict[str, dict[str, str]]:
    rows = {}
    for path in sorted(glob.glob(pattern)):
        for row in read_csv(Path(path)):
            rows[row[key].strip()] = row
    return rows


def kappa(a: list[str], b: list[str]) -> float:
    cats = sorted(set(a) | set(b))
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in cats)
    return round((po - pe) / (1 - pe), 3) if pe < 1 else 1.0


def norm_verdict(v: str) -> str:
    v = v.strip().lower()
    return "A better" if v.startswith("a") else "B better" if v.startswith("b better") else \
        "equal" if v.startswith("eq") else "both wrong"


def fill_sheets(lab: Path, keys: Path, src: Path, run_dir: Path) -> dict:
    exp = run_dir / "expert"
    exp.mkdir(parents=True, exist_ok=True)
    shutil.copy(keys / "key.json", exp / "key.json")
    info: dict = {}

    # Sheet 1: two raters, adjudication on disagreement.
    a = labels(str(lab / "1_pairwise_*_a.csv"), "sample_id")
    b = labels(str(lab / "1_pairwise_*_b.csv"), "sample_id")
    adj = labels(str(lab / "1_pairwise_adjudicated.csv"), "sample_id")
    rows = read_csv(src / "1_pairwise.csv")
    head = list(rows[0].keys())
    verdict_col, meaning_col, comment_col = head[5], head[6], head[7]
    va, vb, missing, disagree = [], [], [], []
    for row in rows:
        sid = row["sample_id"]
        if sid not in a or sid not in b:
            missing.append(sid)
            continue
        x, y = norm_verdict(a[sid]["verdict"]), norm_verdict(b[sid]["verdict"])
        va.append(x), vb.append(y)
        if x == y:
            final, meaning = x, a[sid]["meaning_changed"]
        elif sid in adj:
            final, meaning = norm_verdict(adj[sid]["verdict"]), adj[sid]["meaning_changed"]
        else:
            disagree.append(sid)
            continue
        row[verdict_col], row[meaning_col] = final, meaning
        row[comment_col] = (a[sid]["comment"] if x == y else adj[sid]["comment"])
    info["pairwise"] = {"rated_by_both": len(va), "agreement": round(sum(x == y for x, y in zip(va, vb)) / max(1, len(va)), 3),
                        "kappa": kappa(va, vb) if va else None, "missing": missing,
                        "unadjudicated_disagreements": disagree}
    write_csv(exp / "1_pairwise.csv", head, rows)

    # Sheets 2 and 3: one rater.
    for sheet, key, col_map in (
        ("2_findings", "finding_uid", {"label": 7, "comment": 8}),
        ("3_control", "control_id", {"has_error": 4, "which_error": 5, "severity": 6, "comment": 7}),
    ):
        got = labels(str(lab / f"{sheet}_*.csv"), key)
        rows = read_csv(src / f"{sheet}.csv")
        head = list(rows[0].keys())
        for row in rows:
            r = got.get(row[key])
            if r:
                for name, idx in col_map.items():
                    row[head[idx]] = r.get(name, "")
        info[sheet] = {"rated": sum(row[key] in got for row in rows), "total": len(rows)}
        write_csv(exp / f"{sheet}.csv", head, rows)
    return info


def write_csv(path: Path, head: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=head)
        writer.writeheader()
        writer.writerows(rows)


def severity_tables(lab: Path, run_dir: Path) -> dict:
    """Pairwise severity of the harmful and helpful edits, and control error severities."""
    key = json.loads((run_dir / "expert/key.json").read_text())
    rows = read_csv(run_dir / "expert/1_pairwise.csv")
    sev = {**labels(str(lab / "1_pairwise_*_a.csv"), "sample_id"),
           **labels(str(lab / "1_pairwise_adjudicated.csv"), "sample_id")}
    out: dict = defaultdict(Counter)
    for row in rows:
        v = row[list(row)[5]]
        if not v:
            continue
        info = key["pairwise"][row["sample_id"]]
        outcome = "equal" if v == "equal" else "both wrong" if v == "both wrong" else \
            ("refined_better" if v[0] == info["refined"] else "draft_better")
        s = (sev.get(row["sample_id"]) or {}).get("severity", "").strip().lower() or "?"
        out[f"{info.get('tier')}/{info.get('level')}/{outcome}"][s] += 1
    ctrl_key = key["control"]
    ctrl = defaultdict(Counter)
    for row in read_csv(run_dir / "expert/3_control.csv"):
        has = row[list(row)[4]].strip().lower()
        if has:
            ctrl[ctrl_key[row["control_id"]]["stratum"]][row[list(row)[6]].strip().lower() or "none"] += 1
    return {"pairwise_severity": {k: dict(v) for k, v in sorted(out.items())},
            "control_severity": {k: dict(v) for k, v in ctrl.items()}}


def extra_sheets(lab: Path, keys: Path) -> dict:
    res: dict = {}
    kd = json.loads((keys / "key_disputed.json").read_text())
    got = labels(str(lab / "4_disputed_*.csv"), "item_id")
    # Stratum population sizes, as used when sampling (dominant open-finding route).
    strata_pop = json.loads((keys / "disputed_strata_pop.json").read_text()) if (keys / "disputed_strata_pop.json").exists() else {}
    by = defaultdict(Counter)
    for iid, info in kd.items():
        r = got.get(iid)
        if not r:
            continue
        ok = r["item_ok_for_benchmark"].strip().lower()[:1]
        by[info["stratum"]]["ok" if ok == "y" else "not_ok"] += 1
        by[info["stratum"]]["sev_" + (r["worst_severity"].strip().lower() or "none")] += 1
        fl = r["findings_labels"].lower()
        by[info["stratum"]]["flags_real"] += fl.count("real")
        by[info["stratum"]]["flags_not"] += fl.count(" not") + fl.count(")not")
    res["disputed_by_stratum"] = {k: dict(v) for k, v in sorted(by.items())}
    if strata_pop:
        num = den = 0.0
        for s, c in by.items():
            n = c["ok"] + c["not_ok"]
            if n:
                num += strata_pop[s] * c["not_ok"] / n
                den += strata_pop[s]
        res["disputed_not_ok_weighted"] = round(num / den, 3) if den else None
    src = labels(str(lab / "5_source_1.csv"), "item_id")
    res["source_issues"] = {
        "n": len(src),
        "issue_real": dict(Counter(r["issue_real"].strip().lower()[:1] for r in src.values())),
        "answer_still_unambiguous": dict(Counter(r["answer_still_unambiguous"].strip().lower()[:1] for r in src.values())),
    }
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", type=Path, default=paths.ROOT / "review/llm_labels", help="Folder with the rated CSVs (default: review/llm_labels).")
    ap.add_argument("--keys", type=Path, default=paths.ROOT / "review/keys", help="Folder with key.json and the keys of sheets 4-5 (default: review/keys).")
    ap.add_argument("--sheets", type=Path, default=SHEETS, help="Unrated full sheets (default: review/sheets_full).")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    run_dir = out / "run"
    info = fill_sheets(args.labels, args.keys, args.sheets, run_dir)
    with contextlib.redirect_stdout(io.StringIO()):
        expert_sheet.main(["score", "--run-dir", str(run_dir), "--filled", str(run_dir / "expert/1_pairwise.csv")])
    scores = json.loads((run_dir / "expert/scores.json").read_text())
    summary = {"coverage_and_agreement": info, "scores": scores,
               **severity_tables(args.labels, run_dir), **extra_sheets(args.labels, args.keys)}
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    (out / "labels").mkdir(exist_ok=True)
    for path in glob.glob(str(args.labels / "*.csv")):
        shutil.copy(path, out / "labels" / Path(path).name)
    print(json.dumps(summary, indent=1)[:6000])


if __name__ == "__main__":
    main()
