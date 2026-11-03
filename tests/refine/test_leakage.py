"""leakage: candidate English tokens in Swedish text."""

from __future__ import annotations

from swemedmcq.refine.leakage import leakage_candidates

SV = {"patienten", "har", "och", "ges", "vid", "lateral", "av", "patella", "en", "med", "test"}
EN = {"release", "screening", "heave", "test", "lateral", "the", "and", "with", "patella", "dose"}


def is_sv(tok: str) -> bool:
    return tok.lower() in SV


def is_en(tok: str) -> bool:
    return tok.lower() in EN


def cands(text: str, whitelist: set[str] | None = None, en_text: str | None = None) -> list[str]:
    return [t for t, _, _ in leakage_candidates(text, "question", is_swedish=is_sv, is_english=is_en,
                                                whitelist=whitelist or set(), en_text=en_text)]


def test_basic_candidates_with_offsets():
    text = "Lateral release av patella och screening"
    out = leakage_candidates(text, "question", is_swedish=is_sv, is_english=is_en, whitelist=set())
    assert out == [("release", 8, 15), ("screening", 31, 40)]


def test_swedish_and_short_tokens_skipped():
    assert cands("test med en dose") == ["dose"]
    assert cands("the and") == ["the", "and"]  # length >= 3
    assert cands("of") == []


def test_all_caps_abbreviations_skipped():
    assert cands("HEAVE vid") == []  # <= 6 chars
    assert cands("RELEASE och SCREENING") == ["RELEASE", "SCREENING"]


def test_whitelist():
    assert cands("screening och release", whitelist={"screening"}) == ["release"]
    assert cands("Screening", whitelist={"screening"}) == []


def test_tokens_in_english_source_are_skipped_when_given():
    assert cands("lateral release", en_text="Lateral release of the patella") == []
    assert cands("lateral release", en_text=None) == ["release"]


def test_hyphenated_and_compound_tokens():
    assert cands("högerkammar-heave") == ["heave"]
    assert cands("högerkammarheave") == []  # compounds are not split here
