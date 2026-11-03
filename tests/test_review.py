from __future__ import annotations

import csv
import json
from argparse import Namespace
from pathlib import Path

from swemedmcq.review import edit_audit


def test_edit_class_routes() -> None:
    assert edit_audit.edit_class({"kind": "units"}, [], None) == "T1_units"
    assert edit_audit.edit_class({"kind": "drug"}, [], 1) == "T1_drug"
    assert edit_audit.edit_class({"kind": "drug"}, [], 2) == "T2_drug_integr"
    assert edit_audit.edit_class({"kind": "term_a"}, [], 2) == "T2_term_a"
    findings = [{"route": "GRAMMAR"}, {"route": "ACCURACY"}]
    assert edit_audit.edit_class({"kind": "llm_edit"}, findings, 2) == "T2_accuracy"
    assert edit_audit.edit_class({"kind": "llm_edit"}, [{"route": "LEAK"}], 2) == "T2_leak"


def test_score_weights_classes_by_population(tmp_path: Path) -> None:
    rows = [
        # T1: one class, all better.
        {"sample_id": "S0", "stratum": "T1_drug", "stratum_population": "10", "confidence": "high", "edit_is": "A", "verdict": "A"},
        # T2: class of 30 edits, 1 of 2 worse; class of 10 edits, 1 of 1 better.
        {"sample_id": "S1", "stratum": "T2_grammar", "stratum_population": "30", "confidence": "high", "edit_is": "B", "verdict": "A"},
        {"sample_id": "S2", "stratum": "T2_grammar", "stratum_population": "30", "confidence": "medium", "edit_is": "B", "verdict": "equal"},
        {"sample_id": "S3", "stratum": "T2_leak", "stratum_population": "10", "confidence": "high", "edit_is": "A", "verdict": "A"},
    ]
    labels = tmp_path / "labels.csv"
    with labels.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    out = tmp_path / "scores.json"
    edit_audit.score(Namespace(labels=labels, out=out))
    res = json.loads(out.read_text())
    assert res["tiers"]["T1"]["better"] == 100
    # worse: (30 * 0.5) / 40; better: (10 * 1.0) / 40
    assert abs(res["tiers"]["T2"]["worse"] - 37.5) < 1e-9
    assert abs(res["tiers"]["T2"]["better"] - 25.0) < 1e-9
    assert res["tier2_by_confidence"] == {"high": {"worse": 1, "better": 1}, "medium": {"equal": 1}}
