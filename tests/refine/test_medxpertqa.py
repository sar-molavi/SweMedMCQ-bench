"""medxpertqa: English tail split, format capture, Swedish tail strip and rebuild."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from swemedmcq.paths import path as repo_path
from swemedmcq.refine.medxpertqa import (
    build_sv_tail,
    split_en_tail,
    strip_sv_tail,
    tail_format,
    tail_matches_options,
)

# Eight MedXpertQA drafts used below; the full-file test reads the released draft.
FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "draft_medxpertqa.jsonl"
INPUT = repo_path("draft")

OPTS = {"A": "68 Gy", "B": "55 Gy", "C": "62 Gy"}
EN_Q = "What dose is recommended?\nAnswer Choices: (A) 68 Gy (B) 55 Gy (C) 62 Gy"


def test_split_en_tail_basic():
    body, tail = split_en_tail(EN_Q)
    assert body == "What dose is recommended?"
    assert tail == "\nAnswer Choices: (A) 68 Gy (B) 55 Gy (C) 62 Gy"
    assert body + tail == EN_Q
    assert tail_matches_options(tail, OPTS)
    assert not tail_matches_options(tail, {**OPTS, "C": "64 Gy"})


def test_split_en_tail_absent():
    assert split_en_tail("No list here?") == ("No list here?", None)


def test_newline_separated_format_is_preserved():
    q = "Q?\n\nAnswer Choices:\n(A) x\n(B) y\n(C) z"
    body, tail = split_en_tail(q)
    assert body == "Q?"
    fmt = tail_format(tail)
    assert tail_matches_options(tail, {"A": "x", "B": "y", "C": "z"})
    assert build_sv_tail({"A": "ä", "B": "ö", "C": "å"}, fmt) == "\n\nSvarsalternativ:\n(A) ä\n(B) ö\n(C) å"


def test_space_format_rebuild():
    _, tail = split_en_tail(EN_Q)
    fmt = tail_format(tail)
    assert build_sv_tail({"A": "68 Gy", "B": "55 Gy", "C": "62 Gy"}, fmt) == \
        "\nSvarsalternativ: (A) 68 Gy (B) 55 Gy (C) 62 Gy"


def test_table_options_with_inner_newlines():
    opts = {"A": "| d |\n| --- |\n| 2 µm |", "B": "| d |  \n| --- |  \n| 4 µm |"}
    q = "Which?\nAnswer Choices: (A) " + opts["A"] + " (B) " + opts["B"]
    _, tail = split_en_tail(q)
    assert tail_matches_options(tail, opts)
    assert json.loads(tail_format(tail))["sep"] == " "


@pytest.mark.parametrize(
    "sv, body",
    [
        ("Vilken dos?\nSvarsalternativ: (A) 68 Gy (B) 55 Gy", "Vilken dos?"),
        ("Vilken dos?\nAnswer Choices: (A) 68 Gy (B) 55 Gy", "Vilken dos?"),
        ("Vilken dos?\nSvarsalternativ:", "Vilken dos?"),
        ("Välj det bästa alternativet:\nSvarsalternativ:", "Välj det bästa alternativet:"),
        ("Vilken dos?\n(A) 68 Gy (B) 55 Gy (C) 62 Gy", "Vilken dos?"),
        ("Vilken dos? (A) 68 Gy (B) 55 Gy", "Vilken dos?"),
    ],
)
def test_strip_sv_tail_variants(sv, body):
    b, t = strip_sv_tail(sv)
    assert b == body and t is not None and b + t == sv


@pytest.mark.parametrize(
    "sv",
    [
        "Identifiera dem bland följande svarsalternativ:",  # natural sentence, lowercase
        "Vilket av följande svarsalternativ anger förhållandet?",
        "Patienten (A) har feber",  # a lone (A) is not a list
        "Ingen lista här.",
    ],
)
def test_strip_sv_tail_leaves_bodies_alone(sv):
    assert strip_sv_tail(sv) == (sv, None)


# ---------------------------------------------------------------- real data
def _medxpertqa_rows(path: Path) -> dict:
    rows = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if r["prompt_id"].startswith("medxpertqa"):
                rows[r["prompt_id"]] = r
    return rows


@pytest.fixture(scope="module")
def mx_rows():
    return _medxpertqa_rows(FIXTURE)


@pytest.fixture(scope="module")
def mx_all_rows():
    if not INPUT.exists():
        pytest.skip("released draft not available (data/test/layers/layer1_draft.jsonl)")
    return _medxpertqa_rows(INPUT)


@pytest.mark.parametrize("pid", ["medxpertqa-u_Text-6", "medxpertqa-u_Text-13", "medxpertqa-u_Text-150",
                                 "medxpertqa-u_Text-1827", "medxpertqa-r_Text-62"])
def test_real_items_roundtrip(mx_rows, pid):
    r = mx_rows[pid]
    body, tail = split_en_tail(r["question"])
    assert tail is not None and body + tail == r["question"]
    assert tail_matches_options(tail, r["options"])
    fmt = tail_format(tail)
    sv_body, _ = strip_sv_tail(r["sv_question"])
    assert "(A)" not in sv_body and "Svarsalternativ:" not in sv_body
    rebuilt = sv_body + build_sv_tail(r["sv_options"], fmt)
    assert rebuilt.endswith(f"(J) {r['sv_options']['J']}")
    assert "\nSvarsalternativ: (A) " in rebuilt


def test_real_item_with_existing_tail_rebuilds_identically(mx_rows):
    r = mx_rows["medxpertqa-u_Text-6"]  # draft already has the correct Swedish tail
    fmt = tail_format(split_en_tail(r["question"])[1])
    _body, old_tail = strip_sv_tail(r["sv_question"])
    assert old_tail == build_sv_tail(r["sv_options"], fmt)


def test_real_known_exceptions(mx_rows):
    for pid in ["medxpertqa-u_Text-2230", "medxpertqa-r_Text-2270"]:
        r = mx_rows[pid]
        _, tail = split_en_tail(r["question"])
        assert not tail_matches_options(tail, r["options"])


def test_real_lowercase_label_sentence_not_stripped(mx_rows):
    r = mx_rows["medxpertqa-u_Text-135"]
    assert strip_sv_tail(r["sv_question"])[1] is None


def test_all_real_items_split(mx_all_rows):
    n_match = sum(tail_matches_options(split_en_tail(r["question"])[1], r["options"]) for r in mx_all_rows.values())
    assert len(mx_all_rows) == 2450 and n_match >= 2440
    n_sv = sum(strip_sv_tail(r["sv_question"])[1] is not None for r in mx_all_rows.values())
    assert n_sv == 592
