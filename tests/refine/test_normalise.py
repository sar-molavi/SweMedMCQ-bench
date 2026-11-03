"""normalise: L0 hygiene and corruption detection."""

from __future__ import annotations

import unicodedata

from swemedmcq.refine.locale import revert
from swemedmcq.refine.normalise import hygiene, is_corrupt


def test_typography_is_normalised_and_logged():
    raw = "beta\u2011blockerare\u00a0och\u202f5\u2009mg \u2010 ok"
    out, changes = hygiene(raw, "question")
    assert out == "beta-blockerare och 5 mg - ok"
    assert [c.rule for c in changes] == ["nbhyphen", "nbsp", "nnbsp", "thinsp", "hyphen"]
    assert all(c.field == "question" for c in changes)
    assert revert(out, changes) == raw


def test_control_chars_stripped_but_newline_and_tab_kept():
    raw = "rad1\r\nrad2\tx\x07y"
    out, changes = hygiene(raw, "option:A")
    assert out == "rad1\nrad2\txy"
    assert {c.rule for c in changes} == {"control_char"}
    assert revert(out, changes) == raw


def test_nul_is_removed_and_logged_as_nul():
    raw = "Patienten\x00 har feber"
    out, changes = hygiene(raw, "question")
    assert out == "Patienten har feber"
    assert [(c.rule, c.start, c.before, c.after) for c in changes] == [("nul", 9, "\x00", "")]


def test_nfc_and_invisible_chars():
    raw = unicodedata.normalize("NFD", "år") + "\u200b\ufeffx\u00ad"
    out, changes = hygiene(raw, "question")
    assert out == "årx"
    assert changes[0].rule == "nfc"
    assert revert(out, changes) == raw


def test_clean_text_is_untouched():
    text = "En 45-årig kvinna, 38,4 °C \u2013 inget avvikande.\nSvara."
    assert hygiene(text, "question") == (text, [])


def test_is_corrupt():
    assert is_corrupt("abc\x00def")
    assert is_corrupt("abc\x07")
    assert not is_corrupt("rad\nrad\ttab\r\n")
    assert not is_corrupt("vanlig text \u2013 38,4 °C")
