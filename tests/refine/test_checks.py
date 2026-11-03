"""checks: deterministic noun-gender and spelling checks (route DET)."""

from __future__ import annotations


import pytest

from swemedmcq.refine.io import resources_dir
from swemedmcq.refine.checks import gender_errors, spelling_suspects, suggestions
from swemedmcq.refine.termbase import Lexicon

BUILD = resources_dir()

MORPH = {
    "noun": {
        "kvot": ["u", {"sg indef nom": ["kvot"], "sg def nom": ["kvoten"], "pl indef nom": ["kvoter"]}],
        "värde": ["n", {"sg indef nom": ["värde"], "sg def nom": ["värdet"], "pl indef nom": ["värden"]}],
        "patient": ["u", {"sg indef nom": ["patient"], "sg def nom": ["patienten"]}],
        "hus": ["n", {"sg indef nom": ["hus"], "pl indef nom": ["hus"]}],
        "tumör": ["u", {"sg indef nom": ["tumör"]}],
        "stor": ["u", {"sg indef nom": ["stor"]}],
    },
    "adj": {
        "förhöjd": {"pos indef sg u nom": ["förhöjd"], "pos indef sg n nom": ["förhöjt"], "pos indef pl nom": ["förhöjda"]},
        "stor": {"pos indef sg u nom": ["stor"], "pos indef sg n nom": ["stort"], "pos indef pl nom": ["stora"]},
    },
    "pos": {"kvot": "nn", "värde": "nn", "patient": "nn", "hus": "nn", "tumör": "nn", "stor": "av,nn",
            "förhöjd": "av", "förhöjt": "av", "stort": "av", "kontroll": "nn", "en": "nn,pn,nl", "av": "pp"},
}


def _lex() -> Lexicon:
    words = {"kvot", "kvoten", "värde", "värdet", "patient", "patienten", "hus", "tumör", "stor", "stort", "förhöjd",
             "förhöjt", "prevalens", "en", "ett", "av", "på", "rapporterades", "lökskal", "utseende", "lök", "skal",
             "fluorescens", "immun", "immunfluorescensmikroskopi", "oblikt", "visade", "har", "tenar", "eminens",
             "torakal", "kontroll", "behandling"}
    return Lexicon(sv_words=words, en_words={"report", "the"}, compound_parts={"lök", "skal", "immun", "prevalens"},
                   lemma_forms={"lök": ["lök"], "skal": ["skal"], "immun": ["immun"], "prevalens": ["prevalens"]},
                   morph=MORPH, whitelist={"keep": ["hereditet"]})


@pytest.fixture(scope="module")
def lx() -> Lexicon:
    return _lex()


def test_gender_article(lx):
    g = gender_errors("Ett prevalenskvot på 2 rapporterades.", lx)
    assert g == [dict(start=0, end=17, span="Ett prevalenskvot", expected="En prevalenskvot")]
    assert gender_errors("en värde", lx)[0]["expected"] == "ett värde"
    assert gender_errors("Ett patient", lx)[0]["expected"] == "En patient"


def test_gender_adjective(lx):
    assert gender_errors("ett förhöjd värde", lx) == [
        dict(start=0, end=17, span="ett förhöjd värde", expected="ett förhöjt värde")]
    assert gender_errors("en stort hus", lx)[0]["expected"] == "ett stort hus"


def test_gender_no_false_alarms(lx):
    assert gender_errors("ett värde och en kvot", lx) == []
    assert gender_errors("en av dem", lx) == []          # not a noun
    assert gender_errors("ett stor tumör", lx) == []     # 'stor' is also an adjective: not certain
    assert gender_errors("en kvoten", lx) == []          # definite form: not an article error here
    assert gender_errors("ett hus", lx) == []            # sg = pl is fine
    assert gender_errors("en report", lx) == []          # English word
    # a duck-typed lexicon without SALDOM morphology reports nothing
    class Bare:
        pass
    assert gender_errors("ett patient", Bare()) == []


def test_spelling_link_letters(lx):
    assert suggestions("löksskalsutseende", lx) == ["lökskalsutseende"]
    assert suggestions("immunofluorescens", lx) == ["immunfluorescens"]
    s = spelling_suspects("Ett löksskalsutseende och immunofluorescens visade", lx)
    assert [(x["token"], x["suggestions"]) for x in s] == [
        ("löksskalsutseende", ["lökskalsutseende"]), ("immunofluorescens", ["immunfluorescens"])]
    assert s[0]["start"] == 4 and s[0]["end"] == 4 + len("löksskalsutseende")


def test_spelling_anglicisms_and_typos(lx):
    assert suggestions("obliqt", lx) == ["oblikt"]
    assert suggestions("torakal", lx) == []  # already Swedish
    assert suggestions("thorakal", lx) == ["torakal"]
    assert suggestions("kontorll", lx) == ["kontroll"]  # transposition


def test_spelling_skips(lx):
    # in the English source, capitalised inside a sentence (a name), protected, English, Swedish
    assert spelling_suspects("Ge immunofluorescens", lx, en="immunofluorescens test") == []
    assert spelling_suspects("enligt Thorakal i texten", lx) == []
    assert spelling_suspects("hereditet", Lexicon(sv_words={"heredit"}, en_words=(), whitelist={"keep": ["hereditet"]})) == []
    assert spelling_suspects("the report", lx) == []
    assert spelling_suspects("en kvot", lx) == []


needs_build = pytest.mark.skipif(not (BUILD / "sv_morph.json.gz").exists(), reason="term base not available (data/resources or REFINE_RESOURCES)")


@pytest.fixture(scope="module")
def built() -> Lexicon:
    return Lexicon.load(BUILD)


@needs_build
def test_built_gender(built):
    assert gender_errors("Ett prevalenskvot på 2,1 rapporterades.", built)[0]["expected"] == "En prevalenskvot"
    assert gender_errors("ett förhöjd värde", built)[0]["expected"] == "ett förhöjt värde"
    for ok in ("en biopsi", "en distal radiusfraktur", "ett vestibulärt fynd", "en kraftigt komminut fraktur",
               "ett international normalized ratio", "en stillasittande livsstil", "en lila färg", "ett barn"):
        assert gender_errors(ok, built) == [], ok


@needs_build
def test_built_spelling(built):
    sus = {x["token"]: x["suggestions"] for x in spelling_suspects(
        "Löksskalsutseende vid immunofluorescens, obliqt snitt och thorakalt.", built)}
    assert sus["Löksskalsutseende"][0] == "lökskalsutseende"
    assert sus["immunofluorescens"][0] == "immunfluorescens"
    assert sus["obliqt"][0] == "oblikt"
    assert sus["thorakalt"][0] == "torakalt"
    # valid words that a naive spelling check would flag: no suspects
    text = ("mikrocytär tromboembolisk bulbouretrala sinusoidalt pterygoideus extenderad adducerade "
            "vikthanteringsprogram erytrocythemolysatet calcaneusfraktur genitourinärt")
    assert spelling_suspects(text, built) == []
