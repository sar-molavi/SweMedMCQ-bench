"""gate: every rejection reason, and clean acceptance."""

from __future__ import annotations

from typing import ClassVar

from swemedmcq.refine.gate import GateResult, gate

SV_WORDS = {
    "patienten", "har", "hög", "feber", "och", "hosta", "ges", "vankomycin", "intravenöst", "i", "dagar",
    "behandlas", "med", "en", "ett", "torrhosta", "lätt", "svår", "kraftig", "smärta", "buken", "dos", "mg",
    "ingen", "dyspné", "inga", "utslag", "exantem", "blodtryck", "lågt", "akut", "kronisk", "pankreatit",
    "gastrit", "kolecystit", "the", "of", "kvinna", "årig",
}
EN_WORDS = {"vancomycin", "release", "screening", "the", "of", "severe", "pain", "fever"}


class Lexicon:
    whitelist: ClassVar[set[str]] = {"ekg"}

    @staticmethod
    def is_swedish(tok: str) -> bool:
        return tok.lower() in SV_WORDS

    @staticmethod
    def is_english(tok: str) -> bool:
        return tok.lower() in EN_WORDS


LEX = Lexicon()
EN_Q = "The patient has a high fever and cough and receives 1 g vancomycin intravenously for 7 days."
CUR_Q = "Patienten har hög feber och hosta och ges 1 g vankomycin intravenöst i 7 dagar."


def run(candidate: str, *, current: str = CUR_Q, kind: str = "llm_edit", field: str = "question",
        en: str = EN_Q, anchors=None, verified=None, others=None, lexicon=LEX) -> GateResult:
    if anchors is None:
        a = current.find("hosta")
        anchors = [(a, a + 5)] if a >= 0 else [(0, len(current))]
    return gate(kind=kind, field=field, current=current, candidate=candidate, en_text=en,
                other_options=others or {}, anchors=anchors, verified_sv_spans=verified or [], lexicon=lexicon)


def test_clean_edit_passes():
    res = run(CUR_Q.replace("hosta", "torrhosta"), verified=["vankomycin"])
    assert res.ok and res.reasons == []


def test_noop():
    assert "noop" in run(CUR_Q).reasons
    assert "noop" in run(CUR_Q.replace("hög feber", "hög  feber")).reasons


def test_structure_empty_newline_labels():
    assert "structure" in run("   ").reasons
    assert "structure" in run(CUR_Q.replace("hosta", "hosta\n")).reasons
    assert "structure" in run(CUR_Q + " (A) feber").reasons
    assert "structure" in run(CUR_Q.replace("hosta", "hosta Svarsalternativ:")).reasons
    assert "structure" in run(CUR_Q.replace("hosta", "```hosta```")).reasons


def test_numbers():
    res = run(CUR_Q.replace("hosta och ges 1 g", "hosta och ges en g").replace("i 7 dagar", "i dagar"),
              anchors=[(0, len(CUR_Q))], kind="retranslate")
    assert "numbers" in res.reasons
    # no new misses relative to current: fine
    assert "numbers" not in run(CUR_Q.replace("hosta", "torrhosta")).reasons


def test_non_swedish_token():
    res = run(CUR_Q.replace("hosta", "cough"))
    assert "non_swedish_token:cough" not in res.reasons  # "cough" occurs in the English source
    res = run(CUR_Q.replace("hosta", "hustling"))
    assert "non_swedish_token:hustling" in res.reasons
    # exemptions: numbers/codes, ALL-CAPS abbreviations, whitelist
    assert run(CUR_Q.replace("hosta", "hosta CRP")).ok
    assert run(CUR_Q.replace("hosta", "hosta ekg")).ok
    assert run(CUR_Q.replace("hosta", "hosta HbA1c")).ok


def test_new_leakage_even_if_word_is_in_english_source():
    res = run(CUR_Q.replace("vankomycin", "vancomycin"), anchors=[(0, len(CUR_Q))])
    assert "new_leakage:vancomycin" in res.reasons
    assert all(not r.startswith("non_swedish_token") for r in res.reasons)


def test_existing_leakage_is_not_new():
    cur = CUR_Q.replace("hosta", "hosta (screening)")
    assert run(cur.replace("hög feber", "kraftig feber"), current=cur, anchors=[(0, len(cur))]).ok


def test_lost_verified_term():
    res = run(CUR_Q.replace("vankomycin", "Vankomycin"), verified=["vankomycin"], anchors=[(0, len(CUR_Q))])
    assert not any(r.startswith("lost_verified_term") for r in res.reasons)  # case-insensitive
    res = run(CUR_Q.replace("ges 1 g vankomycin", "ges 1 g medlet"), verified=["vankomycin"],
              anchors=[(0, len(CUR_Q))], lexicon=LEX)
    assert "lost_verified_term:vankomycin" in res.reasons


def test_outside_window_only_for_llm_edits():
    cur = "Patienten har hög feber, hosta och ges 1 g vankomycin intravenöst, i 7 dagar."
    cand = cur.replace("hög feber", "feber")
    a = cur.find("dagar")
    assert "outside_window" in run(cand, current=cur, anchors=[(a, a + 5)]).reasons
    assert "outside_window" not in run(cand, current=cur, anchors=[(a, a + 5)], kind="termbase").reasons
    assert "outside_window" not in run(cand, current=cur, anchors=[(a, a + 5)], kind="retranslate").reasons
    f = cur.find("feber")
    assert run(cand, current=cur, anchors=[(f, f + 5)]).ok


def test_options_not_parallel():
    others = {"option:A": "akut pankreatit", "option:B": "kronisk gastrit", "option:C": "akut kolecystit"}
    cur = "kronisk pankreatit"
    long = "kronisk pankreatit med svår smärta i buken och lågt blodtryck och ingen dyspné och inga utslag"
    res = run(long, current=cur, field="option:D", others=others, kind="retranslate", en="chronic pancreatitis")
    assert "options_not_parallel" in res.reasons
    res = run("akut pankreatit", current=cur, field="option:D", others=others, kind="retranslate",
              en="chronic pancreatitis")
    assert res.ok
    # already outside before: not a new violation
    res = run(long + " och feber", current=long, field="option:D", others=others, kind="retranslate",
              en="chronic pancreatitis")
    assert "options_not_parallel" not in res.reasons


def test_lexicon_without_whitelist_attribute():
    class Bare:
        is_swedish = staticmethod(Lexicon.is_swedish)
        is_english = staticmethod(Lexicon.is_english)

    res = run(CUR_Q.replace("hosta", "hosta ekg"), lexicon=Bare())
    assert "non_swedish_token:ekg" in res.reasons


# ---- protected terms, respellings, op lists --------------------------------------------------


class FormsLexicon(Lexicon):
    """Fake lexicon with inflected forms (as swemedmcq.refine.termbase.Lexicon.forms)."""

    @staticmethod
    def forms(term: str) -> set[str]:
        t = term.lower()
        return {t, t + "et", t + "en", t + "er"}


def test_lost_verified_term_compares_forms():
    lx = FormsLexicon()
    SV_WORDS.update({"nukleinsyraamplifieringstest", "nukleinsyraamplifieringstestet", "visade", "positivt", "ett"})
    cur = "Ett Nukleinsyraamplifieringstest visade positivt."
    cand = "Nukleinsyraamplifieringstestet visade positivt."
    res = run(cand, current=cur, kind="retranslate", anchors=[(0, len(cur))],
              verified=["Nukleinsyraamplifieringstest"], lexicon=lx, en="A nucleic acid amplification test was positive.")
    assert not any(r.startswith("lost_verified_term") for r in res.reasons)
    res = run("Ett test visade positivt.", current=cur, kind="retranslate", anchors=[(0, len(cur))],
              verified=["Nukleinsyraamplifieringstest"], lexicon=lx, en="A nucleic acid amplification test was positive.")
    assert "lost_verified_term:Nukleinsyraamplifieringstest" in res.reasons


def test_removed_protected_term():
    SV_WORDS.update({"hereditet", "familjehistoria", "för", "diabetes", "finns"})
    cur = "Hereditet för diabetes finns."
    en = "There is a family history of diabetes."
    res = gate(kind="retranslate", field="question", current=cur, candidate="Familjehistoria för diabetes finns.",
               en_text=en, other_options={}, anchors=[(0, len(cur))], verified_sv_spans=[], lexicon=LEX,
               protected={"hereditet"})
    assert "removed_protected_term:hereditet" in res.reasons
    # keeping it (inflected or not) is fine
    res = gate(kind="retranslate", field="question", current=cur, candidate="Hereditet för diabetes finns!",
               en_text=en, other_options={}, anchors=[(0, len(cur))], verified_sv_spans=[], lexicon=LEX,
               protected={"hereditet"})
    assert not any(r.startswith("removed_protected_term") for r in res.reasons)


def test_gate_ops_drops_failing_ops_and_regates():
    from swemedmcq.refine.gate import gate_ops
    from swemedmcq.refine.types import SpanOp

    ops = [SpanOp(old="hosta", new="torrhosta"),           # fine
           SpanOp(old="hög feber", new="hustling feber"),  # non-Swedish token
           SpanOp(old="saknas", new="x"),                  # cannot be applied
           SpanOp(old="hosta", new="torrhosta")]           # duplicate: merged
    kept, text, res, rejected = gate_ops(kind="llm_edit", field="question", current=CUR_Q, ops=ops, en_text=EN_Q,
                                         other_options={}, verified_sv_spans=[], lexicon=LEX)
    assert kept == [SpanOp(old="hosta", new="torrhosta")]
    assert text == CUR_Q.replace("hosta", "torrhosta") and res.ok
    reasons = {op.old: r for op, r in rejected}
    assert "non_swedish_token:hustling" in reasons["hög feber"]
    assert reasons["saknas"] == ["apply_failed"]
    # nothing survives
    kept, text, res, rejected = gate_ops(kind="llm_edit", field="question", current=CUR_Q,
                                         ops=[SpanOp(old="hosta", new="hustling")], en_text=EN_Q, other_options={},
                                         verified_sv_spans=[], lexicon=LEX)
    assert kept == [] and text is None and not res.ok and "non_swedish_token:hustling" in res.reasons


def test_gate_ops_protected_term():
    from swemedmcq.refine.gate import gate_ops
    from swemedmcq.refine.types import SpanOp

    SV_WORDS.update({"påverkad", "i", "nöd", "är", "patienten"})
    cur = "Patienten är påverkad."
    kept, text, _res, rejected = gate_ops(kind="llm_edit", field="question", current=cur,
                                          ops=[SpanOp(old="påverkad", new="i nöd")], en_text="The patient is in distress.",
                                         other_options={}, verified_sv_spans=[], lexicon=LEX, protected={"påverkad"})
    assert kept == [] and text is None
    assert rejected[0][1] == ["removed_protected_term:påverkad"]
