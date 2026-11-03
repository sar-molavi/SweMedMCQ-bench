"""Run report: automatic statistics that need no human labels."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from statistics import median

from ..spans import char_edit_distance
from ..types import Item


def _pct(a: int, b: int) -> float | None:
    return round(100 * a / b, 2) if b else None


def _chi2_2x2(a: int, b: int, c: int, d: int) -> dict:
    """Pearson chi-square (1 dof) for [[a, b], [c, d]] with p-value (erfc form)."""
    n = a + b + c + d
    rows, cols = (a + b, c + d), (a + c, b + d)
    if 0 in rows or 0 in cols:
        return {"chi2": None, "p": None}
    chi2 = n * (a * d - b * c) ** 2 / (rows[0] * rows[1] * cols[0] * cols[1])
    return {"chi2": round(chi2, 3), "p": math.erfc(math.sqrt(chi2 / 2))}


def _kappa(pairs: list[tuple[str, str]]) -> float | None:
    if not pairs:
        return None
    labels = sorted({x for p in pairs for x in p})
    n = len(pairs)
    po = sum(1 for a, b in pairs if a == b) / n
    ca, cb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    pe = sum(ca[l] * cb[l] for l in labels) / n**2
    return round((po - pe) / (1 - pe), 3) if pe < 1 else None


C_BINS = [i / 10 for i in range(11)]


def _c_histogram(scores: list[float]) -> dict[str, int]:
    out: dict[str, int] = {}
    for lo, hi in zip(C_BINS, C_BINS[1:]):
        label = f"[{lo:.1f},{hi:.1f}{']' if hi == 1.0 else ')'}"
        out[label] = sum(1 for s in scores if lo <= s < hi or (hi == 1.0 and s == 1.0))
    return out


def confidence_report(results: dict[str, dict]) -> dict:
    """Counts of accepted edits per confidence level (and kind), and the c-distribution of the
    rejected candidates."""
    accepted: Counter = Counter()
    reverted: Counter = Counter()
    by_kind: dict[str, Counter] = defaultdict(Counter)
    rejected_c: list[float] = []
    rejected_complete_c: list[float] = []
    tiers: Counter = Counter()
    items_by_tier: Counter = Counter()
    for r in results.values():
        acc = set(r["accepted"])
        t = r.get("tiers") or {}
        tiers.update(f"tier{v}" for v in t.values())
        if t:
            items_by_tier["tier2" if 2 in t.values() else "tier1_only"] += 1
        for c in r["candidates"]:
            conf = c.get("confidence")
            if not conf:
                continue
            if c["id"] in acc:
                accepted[conf["level"]] += 1
                by_kind[c["kind"]][conf["level"]] += 1
            elif conf.get("reverted_by_final_check"):
                reverted[conf["level"]] += 1
            elif conf["level"] == "rejected":
                rejected_c.append(conf["score"])
                if conf.get("votes", {}).get("complete"):
                    rejected_complete_c.append(conf["score"])
    return {
        "accepted_by_level": dict(accepted),
        "accepted_by_kind_level": {k: dict(v) for k, v in sorted(by_kind.items())},
        "reverted_by_final_check_by_level": dict(reverted),
        # release tiers: 1 = code-made edits (layer4a_auto), 2 = LLM-made edits (layer4_refined only)
        "accepted_by_tier": dict(tiers),
        "refined_items_by_tier": dict(items_by_tier),
        "rejected": {
            "n": len(rejected_c),
            "median_c": median(rejected_c) if rejected_c else None,
            "c_histogram": _c_histogram(rejected_c),
            # short-circuited candidates stop voting once rejection is certain; the complete ones
            # (the agreement sample) show the unbiased distribution
            "complete_n": len(rejected_complete_c),
            "complete_c_histogram": _c_histogram(rejected_complete_c),
        },
    }


def build_report(items: dict[str, Item], results: dict[str, dict]) -> dict:
    rep: dict = {"n_items": len(items), "n_results": len(results)}
    rep["status_counts"] = dict(Counter(r["status"] for r in results.values()))
    per_ds: dict[str, Counter] = defaultdict(Counter)
    for r in results.values():
        per_ds[r["dataset"]][r["status"]] += 1
    rep["status_by_dataset"] = {k: dict(v) for k, v in sorted(per_ds.items())}
    rep["stop_reasons"] = dict(Counter(r["stop"] for r in results.values()))
    rep["rounds_used"] = dict(Counter(len(r["rounds"]) for r in results.values()))

    # findings
    f_all = [f for r in results.values() for f in r["findings"]]
    rep["findings"] = {
        "total": len(f_all),
        "by_source_status": {s: dict(Counter(f["status"] for f in f_all if f["source"] == s))
                             for s in sorted({f["source"] for f in f_all})},
        "by_category_status": {c: dict(Counter(f["status"] for f in f_all if f["category"] == c))
                               for c in sorted({f["category"] for f in f_all})},
        "by_severity": dict(Counter(f["severity"] for f in f_all)),
    }
    llm = [f for f in f_all if f["source"] in {"gptoss", "qwen"}]
    rep["findings"]["llm_ungrounded_rate_pct"] = _pct(sum(f["status"] == "ungrounded" for f in llm), len(llm))
    verified = [f for f in f_all if f["status"] in {"open", "resolved", "dismissed", "rejected_by_judges"} and f["source"] != "corrupt"
                and not f["status_reason"].startswith(("dismissed in an earlier", "repair:"))]
    rep["findings"]["dismissal_rate_pct"] = _pct(
        sum(f["status"] == "dismissed" for f in verified), len(verified))

    # routing classes of the findings
    routed = [f for f in f_all if f.get("route")]
    rep["findings"]["by_route_status"] = {rt: dict(Counter(f["status"] for f in routed if f["route"] == rt))
                                          for rt in sorted({f["route"] for f in routed})}
    acc_conf = [f.get("confirmations", {}) for f in f_all if f.get("route") == "ACCURACY" and f.get("confirmations")]
    rep["findings"]["accuracy_confirmations"] = dict(Counter(
        f'gptoss={c.get("gptoss", "-")}|qwen={c.get("qwen", "-")}' for c in acc_conf))
    rep["findings"]["accuracy_support"] = dict(Counter(
        "+".join(sorted(f.get("support") or [])) or "-" for f in f_all if f.get("route") == "ACCURACY"))

    # decoys: verified-correct spans / false claims presented as errors; "confirm" = false confirmation
    decoys = [d for r in results.values() for d in r["decoys"] if d["verdict"]]
    rep["decoys"] = {"n": len(decoys),
                     "false_confirm_rate_pct": _pct(sum(d["verdict"] == "confirm" for d in decoys), len(decoys))}
    by_mk: dict[str, Counter] = defaultdict(Counter)
    for d in decoys:
        k = f'{d.get("model", "gptoss")}/{d.get("kind", "terminology")}'
        by_mk[k]["n"] += 1
        by_mk[k]["confirm"] += d["verdict"] == "confirm"
        by_mk[k]["confirm_not_low"] += d["verdict"] == "confirm" and d.get("certainty") != "low"
    rep["decoys"]["by_model_kind"] = {k: {**dict(v), "false_confirm_rate_pct": _pct(v["confirm_not_low"], v["n"])}
                                      for k, v in sorted(by_mk.items())}
    vlog = [v for r in results.values() for v in r.get("verify_log", [])]
    rep["verifier"] = {"calls": len(vlog), "reasked": sum(bool(v["reasked"]) for v in vlog),
                       "missing_after_reask": sum(v["missing_after_reask"] for v in vlog)}

    # candidates and acceptance
    cands = [(r, c) for r in results.values() for c in r["candidates"]]
    acc_ids = {(r["prompt_id"], cid) for r in results.values() for cid in r["accepted"]}
    by_kind: dict[str, Counter] = defaultdict(Counter)
    gate_reasons: Counter = Counter()
    dropped_reasons: Counter = Counter()
    for r, c in cands:
        k = f'{c["kind"]}/{c["producer"]}'
        by_kind[k]["total"] += 1
        by_kind[k]["gate_ok"] += bool(c["gate_ok"])
        by_kind[k]["accepted"] += (r["prompt_id"], c["id"]) in acc_ids
        for g in c["gate_reasons"]:
            gate_reasons[g.split(":")[0]] += 1
        for d in c.get("dropped_ops", []):
            for g in d.get("reasons", []):
                dropped_reasons[g.split(":")[0]] += 1
    rep["candidates"] = {"by_kind_producer": {k: dict(v) for k, v in by_kind.items()},
                         "gate_reasons": dict(gate_reasons.most_common()),
                         "per_op_gate_dropped_reasons": dict(dropped_reasons.most_common())}
    rep["confidence"] = confidence_report(results)

    # judges
    verdicts = [(r["prompt_id"], cid, v) for r in results.values() for cid, vs in r["verdicts"].items() for v in vs]
    pw = [v for _, _, v in verdicts if v["task"] == "pairwise"]
    judges = sorted({v["judge"] for _, _, v in verdicts}) or ["qwen", "gptoss"]
    rep["judges"] = {
        "pairwise_by_judge_order": {
            f'{j}/{o}': dict(Counter(v["preferred"] for v in pw if v["judge"] == j and v["order"] == o))
            for j in judges for o in ("current_first", "candidate_first")},
        "votes_by_family_task": {f'{fam}/{t}': dict(Counter(v.get("vote") for _, _, v in verdicts
                                                            if (v.get("family") or v["judge"]) == fam
                                                            and v["task"] == t))
                                 for fam in sorted({v.get("family") or v["judge"] for _, _, v in verdicts})
                                 for t in ("pairwise", "noninferiority")},
        "certainty": dict(Counter(f'{v["judge"]}/{v.get("certainty")}' for _, _, v in verdicts)),
        "noninferiority": dict(Counter(f'{v["judge"]}/introduces_error={v["introduces_error"]}'
                                       for _, _, v in verdicts if v["task"] == "noninferiority")),
    }
    per_cand: dict[tuple, dict] = defaultdict(dict)
    for pid, cid, v in verdicts:
        if v["task"] == "pairwise":
            per_cand[(pid, cid)][(v["judge"], v["order"], v.get("sample", 0))] = v["preferred"]
    pos_consistent = Counter()
    for d in per_cand.values():
        for (j, o, k), a in d.items():
            if o == "current_first" and (j, "candidate_first", k) in d and a and d[(j, "candidate_first", k)]:
                pos_consistent[(j, a == d[(j, "candidate_first", k)])] += 1
    rep["judges"]["position_consistency"] = {
        j: _pct(pos_consistent[(j, True)], pos_consistent[(j, True)] + pos_consistent[(j, False)]) for j in judges}
    pairs_needed = [("qwen", "current_first", 0), ("gptoss", "current_first", 0),
                    ("qwen", "candidate_first", 0), ("gptoss", "candidate_first", 0)]
    # An unanswered judge call is recorded as None; it is not a vote for κ.
    full = [d for d in per_cand.values() if all(d.get(p) is not None for p in pairs_needed)]
    rep["judges"]["kappa_qwen_vs_gptoss_on_full_sample"] = {
        "n": len(full),
        "current_first": _kappa([(d[("qwen", "current_first", 0)], d[("gptoss", "current_first", 0)]) for d in full]),
        "candidate_first": _kappa([(d[("qwen", "candidate_first", 0)], d[("gptoss", "candidate_first", 0)])
                                   for d in full]),
    }
    finals = [r["final_check"] for r in results.values() if r.get("final_check")]
    rep["item_final_check"] = {"n": len(finals), "failed": sum(not f["ok"] for f in finals),
                               "reverted_candidates": sum(len(f.get("reverted", [])) for f in finals)}

    # edits
    refined = [r for r in results.values() if r["status"] == "refined"]
    dists = []
    for r in refined:
        it = items[r["prompt_id"]]
        for k in r["changed_fields"]:
            dists.append(char_edit_distance(it.sv_l1[k], r["sv_final"][k]))
    rep["edits"] = {"refined_items": len(refined), "changed_fields": len(dists),
                    "median_char_edit_distance": median(dists) if dists else None}

    # gold-vs-distractor audit: are keyed options edited at a different rate than distractors?
    a = b = c = d = 0  # a: gold edited, b: gold unedited, c: distractor edited, d: distractor unedited
    for r in results.values():
        it = items[r["prompt_id"]]
        if not it.answer:
            continue
        for k in it.en:
            if not k.startswith("option:"):
                continue
            edited = k in r["changed_fields"]
            if k == f"option:{it.answer}":
                a, b = a + edited, b + (not edited)
            else:
                c, d = c + edited, d + (not edited)
    rep["gold_vs_distractor_edits"] = {"gold_edited": a, "gold_total": a + b, "gold_rate_pct": _pct(a, a + b),
                                       "distractor_edited": c, "distractor_total": c + d,
                                       "distractor_rate_pct": _pct(c, c + d), **_chi2_2x2(a, b, c, d)}

    # calls / cost
    calls = [c for r in results.values() for c in r["calls"]]
    per_step: dict[str, Counter] = defaultdict(Counter)
    for c in calls:
        s = per_step[f'{c["step"]}/{c["model"]}']
        s["calls"] += 1
        s["from_cache"] += bool(c["from_cache"])
        s["failed"] += c["status"] != "ok"
        s["fallback"] += c.get("model_used", c["model"]) != c["model"]
        s["reasoning_empty"] += bool(c["reasoning_empty"])
        s["prompt_tokens"] += c["prompt_tokens"] or 0
        s["completion_tokens"] += c["completion_tokens"] or 0
        s[f'finish_{c["finish_reason"]}'] += 1
    rep["calls"] = {k: dict(v) for k, v in sorted(per_step.items())}
    return rep
