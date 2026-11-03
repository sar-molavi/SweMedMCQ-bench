"""spans: quote finding with offset mapping, span ops, diffs and edit windows."""

from __future__ import annotations

from swemedmcq.refine.spans import (
    apply_ops,
    char_edit_distance,
    clause_bounds,
    diff_spans,
    find_all_quotes,
    find_quote,
    norm_for_match,
    replace_all,
    within_window,
)
from swemedmcq.refine.types import SpanOp

NBH = "\u2011"  # non-breaking hyphen
NDASH = "\u2013"
MDASH = "\u2014"


def test_norm_for_match():
    assert norm_for_match("  Beta" + NBH + "Blockerare \n\t och  \u201cX\u201d ") == 'beta-blockerare och "x"'
    assert norm_for_match("a" + NDASH + "b" + MDASH + "c\u2010d") == "a-b-c-d"
    assert norm_for_match("\u2018x\u2019") == "'x'"


def test_find_quote_exact_first():
    text = "Patienten har feber och feber."
    assert find_quote(text, "feber") == (14, 19)


def test_find_quote_normalised_maps_back_to_original_offsets():
    text = "Ge beta" + NBH + "blockerare  i\n hög   dos."
    s, e = find_quote(text, "Beta-blockerare i hög dos")
    assert text[s:e] == "beta" + NBH + "blockerare  i\n hög   dos"
    s, e = find_quote("svar: \u201cja\u201d" + NDASH + "nej", '"ja"-nej')
    assert (s, e) == (6, 14)


def test_find_quote_whole_word():
    # 'mig' must not be grounded inside 'miglitol'
    text = "Hon tar miglitol. Ge mig svaret."
    assert find_quote(text, "mig") == (text.index(" mig ") + 1, text.index(" mig ") + 4)
    assert find_quote("miglitol ges", "mig") is None
    # quotes that start or end with punctuation are bounded by it
    t = "temp 37 °C (98,6 °F) och puls 80"
    s, e = find_quote(t, "(98,6 °F)")
    assert t[s:e] == "(98,6 °F)"
    # normalised fallback is whole-word too
    assert find_quote("beta" + NBH + "blockerare", "Beta-blockerare") == (0, 15)
    assert find_all_quotes("feber, feber och feberkramp", "feber") == [(0, 5), (7, 12)]


def test_find_quote_absent_or_empty():
    assert find_quote("abc", "xyz") is None
    assert find_quote("abc", "") is None
    assert find_quote("abc", "   ") is None


def test_apply_ops():
    text = "En 45-årig man med feber och hosta."
    out = apply_ops(text, [SpanOp(old="feber", new="hög feber"), SpanOp(old="hosta", new="torrhosta")])
    assert out == "En 45-årig man med hög feber och torrhosta."


def test_apply_ops_normalised_old():
    text = "beta" + NBH + "blockerare ges"
    assert apply_ops(text, [SpanOp(old="Beta-blockerare", new="betablockerare")]) == "betablockerare ges"


def test_apply_ops_rejects_missing_ambiguous_overlapping():
    text = "feber och feber"
    assert apply_ops(text, [SpanOp(old="hosta", new="x")]) is None
    # a single-word op applies to every occurrence
    assert apply_ops(text, [SpanOp(old="feber", new="x")]) == "x och x"
    # a multi-word op must still occur exactly once
    assert apply_ops("hög feber och hög feber", [SpanOp(old="hög feber", new="feber")]) is None
    assert apply_ops("abc def", [SpanOp(old="abc d", new="x"), SpanOp(old="c def", new="y")]) is None
    assert apply_ops("abc", [SpanOp(old="", new="x")]) is None


def test_apply_ops_merges_duplicate_ops_for_repeated_words():
    # a repeated word given as duplicate operations must still apply
    text = "Etorikoxib ges. Sedan etorikoxib igen, och etorikoxib en tredje gång."
    ops = [SpanOp(old="etorikoxib", new="etoricoxib")] * 3
    assert apply_ops(text, ops) == "Etoricoxib ges. Sedan etoricoxib igen, och etoricoxib en tredje gång."
    # same old, different new: conflict
    assert apply_ops(text, [SpanOp(old="etorikoxib", new="a"), SpanOp(old="etorikoxib", new="b")]) is None


def test_apply_ops_whole_words_only():
    # 'feber' in 'feberkramper' is not an occurrence
    assert apply_ops("feber och feberkramper", [SpanOp(old="feber", new="hög feber")]) == \
        "hög feber och feberkramper"
    # a sub-word edit is still possible when it is unique
    assert apply_ops("feberkramper", [SpanOp(old="kramper", new="kramp")]) == "feberkramp"
    assert apply_ops("Rubella och rubella", [SpanOp(old="Rubella", new="rubella")]) == "rubella och rubella"


def test_replace_all_word_boundary_and_case():
    text = "Vancomycin ges. Sedan vancomycin igen; vancomycinresistens kvarstår."
    out, n = replace_all(text, ["vancomycin"], "vankomycin")
    assert out == "Vankomycin ges. Sedan vankomycin igen; vancomycinresistens kvarstår."
    assert n == 2


def test_replace_all_longest_form_first_and_noop_not_counted():
    out, n = replace_all("metabolisk acidos och acidos", ["acidos", "metabolisk acidos"], "metabol acidos")
    assert out == "metabol acidos och metabol acidos" and n == 2
    out, n = replace_all("vankomycin", ["vankomycin"], "vankomycin")
    assert (out, n) == ("vankomycin", 0)


def test_replace_all_keeps_abbreviation_case():
    out, n = replace_all("ett ekg togs", ["ekg"], "EKG")
    assert out == "ett EKG togs" and n == 1


def test_diff_spans_and_distance():
    assert diff_spans("abc", "abc") == []
    assert diff_spans("abcdef", "abXdef") == [(2, 3, 2, 3)]
    assert char_edit_distance("kitten", "sitting") == 3
    assert char_edit_distance("", "abc") == 3
    assert char_edit_distance("same", "same") == 0
    assert char_edit_distance("prefix-a-suffix", "prefix-bb-suffix") == 2


def test_clause_bounds():
    text = "Hon har feber, hosta och 38,5 grader; ingen dyspné."
    s, e = clause_bounds(text, text.find("hosta"), text.find("hosta") + 5)
    assert text[s:e] == " hosta och 38,5 grader"
    s, e = clause_bounds(text, 0, 3)
    assert text[s:e] == "Hon har feber"


def test_within_window():
    old = "Hon har feber, hosta och ont i halsen; ingen dyspné, inga utslag."
    a = old.find("hosta")
    anchors = [(a, a + 5)]
    assert within_window(old, old.replace("hosta", "torrhosta"), anchors)
    assert within_window(old, old.replace("ont i halsen", "halsont"), anchors)
    assert not within_window(old, old.replace("utslag", "exantem"), anchors)
    # extra tokens reach just across the clause border
    assert within_window(old, old.replace("feber", "hög feber"), anchors, extra_tokens=3)
    assert not within_window(old, old.replace("feber", "hög feber"), anchors, extra_tokens=0)
    assert within_window(old, old, [])
    assert not within_window(old, old + "!", [])
