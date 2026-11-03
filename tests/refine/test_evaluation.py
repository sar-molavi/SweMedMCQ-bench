"""Report / expert-sheet / probe summaries on loop records (fake engine, no GPU)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from swemedmcq.refine.evaluation.expert_sheet import confidence_level, edit_class, stratum  # noqa: E402
from swemedmcq.refine.evaluation.probes import summarise  # noqa: E402
from swemedmcq.refine.evaluation.report import build_report, confidence_report  # noqa: E402
from test_loop import Responder, make_item, run  # noqa: E402


def _records(tmp_path):
    it = make_item()
    good = run(it, Responder(), tmp_path / "a")

    def judge(model, a, b, k, order, req):
        return ("A" if " and " in a else "B"), "high"

    bad = run(it.model_copy(update={"prompt_id": "medqa_test_2"}), Responder(judge=judge), tmp_path / "b")
    return it, good, bad


def test_report_counts_confidence_levels_and_rejected_c(tmp_path):
    it, good, bad = _records(tmp_path)
    items = {good["prompt_id"]: it, bad["prompt_id"]: it.model_copy(update={"prompt_id": bad["prompt_id"]})}
    rep = build_report(items, {r["prompt_id"]: r for r in (good, bad)})
    assert rep["confidence"]["accepted_by_level"] == {"high": 1}
    assert rep["confidence"]["rejected"]["n"] >= 1
    assert sum(rep["confidence"]["rejected"]["c_histogram"].values()) == rep["confidence"]["rejected"]["n"]
    assert rep["findings"]["by_route_status"]["LEAK"]["resolved"] == 1
    assert rep["findings"]["by_route_status"]["LEAK"]["rejected_by_judges"] >= 1
    assert "qwen/pairwise" in rep["judges"]["votes_by_family_task"]
    assert confidence_report({"x": good})["accepted_by_kind_level"] == {"llm_edit": {"high": 1}}


def test_expert_strata_by_level_and_class(tmp_path):
    _, good, _ = _records(tmp_path)
    assert confidence_level(good) == "high" and edit_class(good) == "llm_edit"
    assert stratum(good) == "high" and stratum(good, "level_class") == "high/llm_edit"
    assert stratum(good, "class") == "llm_edit"
    v1 = {**good, "candidates": [{**c, "confidence": None} for c in good["candidates"]]}
    assert confidence_level(v1) == "v1"


def test_probe_summary_uses_samples():
    rows = []
    for k in range(3):
        for order, got in (("orig_first", "orig"), ("mod_first", "orig" if k else "mod")):
            rows.append({"judge": "qwen", "kind": "negation", "i": 0, "k": k, "order": order, "got": got,
                         "expected": "orig", "certainty": "high"})
    s = summarise(rows)
    assert s["per_probe"]["qwen/negation"]["n"] == 3
    assert s["per_probe"]["qwen/negation"]["both_orders_correct"] == round(2 / 3, 3)
