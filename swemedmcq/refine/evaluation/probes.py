"""Judge probes: measure the blind pairwise judges without human labels.

    python -m swemedmcq.refine.evaluation.probes --run-dir runs/refine/full --config configs/refine/with_gemma.yaml [--n 300]

Probe types (built deterministically from prepared items, layer L1 text):
  identical      A == B                              -> expected "equal"
  negation       drop "inte"/"ej"                     -> original better
  number         change one number                    -> original better
  laterality     swap höger/vänster                   -> original better
  english_inn    verified Swedish term -> its English source term (e.g. vankomycin -> vancomycin)
                                                      -> original better
  agreement      swap the article en/ett               -> original better
  silver         LLM-labelled grammar edits, read from --silver-labels (CSV with prompt_id,
                 field, en, draft, final, label; default <resources>/grammar_edit_sample_labels.csv,
                 skipped when absent): B -> edit better, W -> draft better, E -> equal

Each probe is judged by every judge of `loop.judge_models` in both orders (Qwen: k seeded samples
per order, `loop.qwen_judge_samples`); a low-certainty preference counts as "equal".  Output: `<run>/probes/results.json`
(accuracy and position consistency per judge and probe type) and `probes.jsonl` (all verdicts).
Resumable through the request cache.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import yaml

from .. import prompting as P
from ..io import resources_dir
from ..engine import LLMRequest, ModelPool
from ..io import atomic_write_json, atomic_write_text, read_jsonl
from ..loop.item_loop import LoopConfig
from ..loop.routing import ProtectedTerms, load_keep
from ..schemas import SCHEMAS
from ..termbase import TermBase
from ..types import Item

SEED = 7
PROT: ProtectedTerms | None = None  # protected terms shown to the judges (as in the loop)
SWAPS = {"höger": "vänster", "vänster": "höger", "högra": "vänstra", "vänstra": "högra",
         "högersidig": "vänstersidig", "vänstersidig": "högersidig"}


def _sub_once(pattern: str, repl, text: str) -> str | None:
    new, n = re.subn(pattern, repl, text, count=1)
    return new if n else None


def inject(kind: str, text: str, en: str, tb_pairs: list) -> str | None:
    if kind == "negation":
        return _sub_once(r"\s(inte|ej)\b", "", text)
    if kind == "number":
        def bump(m):
            return str(int(m.group(0)) + 1)
        return _sub_once(r"(?<![\d.,])\d{1,3}(?![\d.,])", bump, text)
    if kind == "laterality":
        return _sub_once(r"\b(" + "|".join(SWAPS) + r")\b", lambda m: SWAPS[m.group(1)], text)
    if kind == "agreement":
        return _sub_once(r"\b(en|ett)\b(?=\s+\w)", lambda m: "ett" if m.group(1) == "en" else "en", text)
    if kind == "english_inn":
        for p in tb_pairs:
            if p.sv_span in text and p.en_span.lower() != p.sv_span.lower():
                return text.replace(p.sv_span, p.en_span, 1)
    return None


def build_probes(items: list[Item], tb: TermBase, n: int) -> list[dict]:
    rng = random.Random(SEED)
    rng.shuffle(items)
    kinds = ["negation", "number", "laterality", "agreement", "english_inn"]
    quota = {"identical": n // 6, **{k: n // 6 for k in kinds}}
    probes: list[dict] = []
    for it in items:
        pairs = None
        for key in P.ordered_fields(it.sv_l1):
            text = it.sv_l1[key]
            if quota["identical"] > 0 and len(text.split()) >= 4:
                probes.append({"kind": "identical", "pid": it.prompt_id, "field": key, "orig": text, "mod": text,
                               "expected": "equal"})
                quota["identical"] -= 1
                continue
            for k in kinds:
                if quota[k] <= 0:
                    continue
                if k == "english_inn" and pairs is None:
                    pairs, _ = tb.ground(it.en, it.sv_l1)
                mod = inject(k, text, it.en[key], [p for p in (pairs or []) if p.field == key])
                if mod and mod != text:
                    probes.append({"kind": k, "pid": it.prompt_id, "field": key, "orig": text, "mod": mod,
                                   "expected": "orig"})
                    quota[k] -= 1
                    break
        if all(v <= 0 for v in quota.values()):
            break
    return probes


def silver_probes(items_by_id: dict[str, Item], path: Path | None = None) -> list[dict]:
    if path is None or not path.exists():
        return []
    out = []
    with path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            exp = {"B": "mod", "W": "orig", "E": "equal"}[row["label"]]
            out.append({"kind": "silver", "pid": row["prompt_id"], "field": row["field"], "orig": row["draft"],
                        "mod": row["final"], "expected": exp, "en_field": row["en"]})
    return out


async def judge_all(pool, probes: list[dict], items: dict[str, Item], judges=("qwen", "gptoss"),
                    qwen_samples: int = 1) -> list[dict]:
    """Every probe x judge x sample (Qwen: k seeded samples) x order.  ``got`` is the effective
    answer after the certainty rule (a low-certainty preference counts as "equal");
    ``got_raw`` is the stated answer."""
    async def one(i: int, pr: dict, judge: str, order: str, k: int) -> dict:
        it = items.get(pr["pid"])
        en = it.en if it else {pr["field"]: pr.get("en_field", "")}
        sv = it.sv_l1 if it else {pr["field"]: pr["orig"]}
        a, b = (pr["orig"], pr["mod"]) if order == "orig_first" else (pr["mod"], pr["orig"])
        system, user = P.pairwise_prompt(en, sv, pr["field"], a, b, protected=PROT.present(a, b) if PROT else [])
        res = await pool.generate(LLMRequest(model=judge, system=system, user=user, schema_name="pairwise",
                                             seed_key=["probe", str(i), pr["kind"], judge, order, f"k{k}"]))
        raw = (res.parsed or {}).get("preferred")
        if raw in ("A", "B"):
            mod_label = "B" if order == "orig_first" else "A"
            got = "mod" if raw == mod_label else "orig"
        else:
            got = raw  # equal / both_wrong / None
        cert = (res.parsed or {}).get("certainty")
        eff = "equal" if cert == "low" and got in ("mod", "orig", "both_wrong") else got
        return {**{k_: v for k_, v in pr.items() if k_ not in ("orig", "mod")}, "i": i, "judge": judge,
                "family": getattr(res, "model_used", judge), "order": order, "k": k, "got": eff, "got_raw": got,
                "certainty": cert, "reason": (res.parsed or {}).get("reason", ""), "status": res.status}

    tasks = [one(i, pr, j, o, k) for i, pr in enumerate(probes) for j in judges
             for k in range(qwen_samples if j == "qwen" else 1) for o in ("orig_first", "mod_first")]
    return await asyncio.gather(*tasks)


def summarise(rows: list[dict]) -> dict:
    acc: dict[str, Counter] = defaultdict(Counter)
    by_probe: dict[tuple, dict] = defaultdict(dict)
    for r in rows:
        k = f'{r["judge"]}/{r["kind"]}'
        acc[k]["n"] += 1
        acc[k]["correct"] += r["got"] == r["expected"]
        by_probe[(r["judge"], r["i"], r.get("k", 0))][r["order"]] = (r["got"], r["expected"], r["kind"])
    both: dict[str, Counter] = defaultdict(Counter)
    for (judge, _, _), d in by_probe.items():
        if len(d) == 2:
            (g1, e, kind), (g2, _, _) = d["orig_first"], d["mod_first"]
            k = f"{judge}/{kind}"
            both[k]["n"] += 1
            both[k]["both_orders_correct"] += g1 == e and g2 == e
            both[k]["position_consistent"] += g1 == g2
    low = Counter(r["judge"] for r in rows if r.get("certainty") == "low")
    return {"per_call_accuracy": {k: round(v["correct"] / v["n"], 3) for k, v in sorted(acc.items())},
            "per_probe": {k: {"n": v["n"], "both_orders_correct": round(v["both_orders_correct"] / v["n"], 3),
                              "position_consistent": round(v["position_consistent"] / v["n"], 3)}
                          for k, v in sorted(both.items())},
            "low_certainty_calls": dict(low)}


async def run(args) -> None:
    cfg = yaml.safe_load(args.config.read_text())
    items = {}
    for path in sorted((args.run_dir / "prep").glob("shard_*.jsonl")):
        for r in read_jsonl(path):
            it = Item.model_validate(r)
            items[it.prompt_id] = it
    tb = TermBase.load(resources_dir(cfg))
    silver = args.silver_labels or resources_dir(cfg) / "grammar_edit_sample_labels.csv"
    probes = build_probes(sorted(items.values(), key=lambda x: x.prompt_id), tb, args.n) + silver_probes(items, silver)
    out = args.run_dir / "probes"
    loop_cfg = LoopConfig.from_dict(cfg.get("loop"))
    judges = [j for j in loop_cfg.judge_models if j in cfg.get("models", {})]
    global PROT
    PROT = ProtectedTerms(load_keep(None))
    pool = await ModelPool.create(cfg, SCHEMAS, args.run_dir / "cache", models=judges)
    try:
        rows = await judge_all(pool, probes, items, judges, args.qwen_samples or loop_cfg.qwen_judge_samples)
    finally:
        await pool.close()
    atomic_write_text(out / "probes.jsonl", "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    summary = {"n_probes": len(probes), "by_kind": dict(Counter(p["kind"] for p in probes)), **summarise(rows)}
    atomic_write_json(out / "results.json", summary)
    print(json.dumps(summary, indent=1))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--silver-labels", type=Path, default=None,
                    help="CSV of labelled grammar edits (default: <resources>/grammar_edit_sample_labels.csv if present).")
    ap.add_argument("--qwen-samples", type=int, default=None,
                    help="Seeded Qwen samples per order (default: loop.qwen_judge_samples of the config).")
    asyncio.run(run(ap.parse_args(argv)))


if __name__ == "__main__":
    main()
