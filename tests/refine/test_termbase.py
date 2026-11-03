"""Tests for swemedmcq.refine.termbase (offline; tiny fixtures) plus checks on the built term base."""

from __future__ import annotations


import pytest

from swemedmcq.refine.io import resources_dir
from swemedmcq.refine.termbase import Lexicon, TermBase, en_key, en_tokens, is_acronym, singularise

BUILD = resources_dir()

LEMMA_FORMS = {
    "acidos": ["acidos", "acidosen", "acidoser", "acidoserna"],
    "metabolisk": ["metabolisk", "metaboliska", "metaboliskt"],
    "resistens": ["resistens", "resistensen"],
    "resistent": ["resistent", "resistenta"],
    "enterokock": ["enterokock", "enterokocken", "enterokocker", "enterokockerna"],
    "tumör": ["tumör", "tumören", "tumörer", "tumörerna"],
    "skalle": ["skalle", "skallen", "skallar"],
    "form": ["form", "formen", "former"],
    "behandling": ["behandling", "behandlingen", "behandlingar"],
    "hjärtsvikt": ["hjärtsvikt", "hjärtsvikten"],
    "patient": ["patient", "patienten", "patienter"],
    "ha": ["ha", "har", "hade"],
    "ge": ["ge", "ger", "gav"],
    "intravenös": ["intravenös", "intravenöst", "intravenösa"],
    "respiratorisk": ["respiratorisk", "respiratoriska", "respiratoriskt"],
    "depression": ["depression", "depressionen"],
    "metabol": ["metabol", "metabola", "metabolt"],
}


def _lexicon() -> Lexicon:
    words = set()
    for forms in LEMMA_FORMS.values():
        words.update(forms)
    words.update({"och", "med", "kokain", "bakterie"})
    return Lexicon(
        sv_words=words,
        en_words={"blood", "patient", "depression", "will", "heart"},
        lemma_forms=LEMMA_FORMS,
        compound_parts={"hjärt", "skall"},
        whitelist={"abbreviations": ["EKG", "CRP"], "latin_anatomy": ["musculus"], "ambiguous_en": ["depression"]},
    )


ROWS = [
    # source npl_atc: Wikidata EN INN x NPL Swedish name
    dict(entry_id="npl_atc:J01XA01:A:1", en_term="vancomycin", en_key="vancomycin", sv_forms=["vankomycin"],
         level="A", single_sense=True, obligation_ok=True, source="npl_atc", concept_id="J01XA01"),
    dict(entry_id="swemesh:D014640:A:1", en_term="Vancomycin", en_key="vancomycin", sv_forms=["Vankomycin"],
         level="A", single_sense=True, obligation_ok=True, source="swemesh", concept_id="D014640"),
    dict(entry_id="npl_atc:J05AB01:A:1", en_term="acyclovir", en_key="acyclovir", sv_forms=["aciklovir"],
         level="A", single_sense=True, obligation_ok=True, source="npl_atc", concept_id="J05AB01"),
    dict(entry_id="swemesh:D000138:A:1", en_term="metabolic acidosis", en_key="metabolic acidosis",
         sv_forms=["metabolisk acidos"], level="A", single_sense=True, obligation_ok=True, source="swemesh",
         concept_id="D000138"),
    dict(entry_id="swemesh:D000138:B:1", en_term="metabolic acidosis", en_key="metabolic acidosis",
         sv_forms=["acidos"], level="B", single_sense=True, obligation_ok=False, source="swemesh",
         concept_id="D000138"),
    dict(entry_id="swemesh:D065507:A:1", en_term="Vancomycin-Resistant Enterococci",
         en_key=en_key("Vancomycin-Resistant Enterococci"), sv_forms=["vankomycinresistenta enterokocker"],
         level="A", single_sense=True, obligation_ok=True, source="swemesh", concept_id="D065507"),
    dict(entry_id="swemesh:D006333:A:1", en_term="heart failure", en_key="heart failure", sv_forms=["hjärtsvikt"],
         level="A", single_sense=True, obligation_ok=False, source="swemesh", concept_id="D006333"),
    dict(entry_id="swemesh:D003863:A:1", en_term="Depression", en_key="depression", sv_forms=["depression"],
         level="A", single_sense=False, obligation_ok=False, source="swemesh", concept_id="D003863"),
    dict(entry_id="swemesh:D009369:A:1", en_term="tumors", en_key=en_key("tumors"), sv_forms=["Tumörer"],
         level="A", single_sense=True, obligation_ok=True, source="swemesh", concept_id="D009369"),
    dict(entry_id="swemesh:D012886:A:1", en_term="skull", en_key="skull", sv_forms=["skalle"],
         level="A", single_sense=True, obligation_ok=True, source="swemesh", concept_id="D012886"),
    dict(entry_id="swemesh:D000163:A:1", en_term="AIDS", en_key="aid", sv_forms=["aids"],
         level="A", single_sense=False, obligation_ok=False, source="swemesh", concept_id="D000163"),
]


@pytest.fixture(scope="module")
def tb() -> TermBase:
    return TermBase(ROWS, _lexicon())


# ---- English normalisation ---------------------------------------------------------------


def test_en_normalisation():
    assert singularise("arteries") == "artery"
    assert singularise("enterococci") == "enterococci"
    assert singularise("diabetes") == "diabete"  # consistent on both sides
    assert singularise("pus") == "pus" and singularise("analysis") == "analysis"
    assert en_key("Alzheimer's Diseases") == "alzheimer disease"
    assert en_key("Vancomycin-Resistant Enterococci") == "vancomycin resistant enterococci"
    assert [t[0] for t in en_tokens("type-2 diabetes")] == ["type", "2", "diabete"]
    assert is_acronym("AIDS") and is_acronym("NIDDM") and not is_acronym("tPA") and not is_acronym("Aids")


# ---- spot ---------------------------------------------------------------------------------


def test_spot_longest_match_and_merge(tb):
    text = "Vancomycin-resistant enterococci were treated with IV vancomycin."
    hits = tb.spot(text, "question")
    assert [h.en_span for h in hits] == ["Vancomycin-resistant enterococci", "vancomycin"]
    h = hits[1]
    assert (h.start, h.end) == (text.rindex("vancomycin"), text.rindex("vancomycin") + len("vancomycin"))
    assert h.level == "A" and h.single_sense
    assert h.sv_forms == ["vankomycin"]  # merged across npl_atc + swemesh, case-insensitively deduped
    assert h.entry_id.startswith("npl_atc:")  # highest-priority source
    assert h.source == "npl_atc+swemesh"


def test_spot_plural_and_acronyms(tb):
    assert [h.en_span for h in tb.spot("benign tumor", "option:A")] == ["tumor"]
    assert [h.en_span for h in tb.spot("Tumors of the skull", "question")] == ["Tumors", "skull"]
    # acronyms are case-sensitive: 'aids' (verb) must not match AIDS
    assert tb.spot("hearing aids", "question") == []
    assert [h.en_span for h in tb.spot("a patient with AIDS", "question")] == ["AIDS"]


# ---- Lexicon ------------------------------------------------------------------------------


def test_lexicon_validity_and_compounds():
    lx = _lexicon()
    assert lx.is_swedish("Acidosen")
    assert lx.is_swedish("hjärtsvikten")
    assert lx.is_swedish("kokainresistens")  # two SALDO words
    assert lx.is_swedish("tumörsresistens")  # linking s
    assert lx.is_swedish("hjärtbehandling")  # compound part + word
    assert not lx.is_swedish("vancomycin")
    assert lx.is_english("Blood") and not lx.is_english("acidos")
    assert {"EKG", "ekg", "CRP", "musculus"} <= lx.whitelist
    assert "depression" in lx.ambiguous_en


def test_lexicon_forms_inflect_last_word():
    lx = _lexicon()
    f = lx.forms("metabolisk acidos")
    assert {"metabolisk acidos", "metabolisk acidosen", "metaboliska acidoser"} <= f
    assert {"tumör", "tumörerna"} <= lx.forms("Tumörer")  # plural lemma form -> lemma
    # unknown word: heuristic endings
    assert {"vankomycin", "vankomycinet", "vankomycinen"} <= lx.forms("vankomycin")


# ---- match_sv -----------------------------------------------------------------------------


def test_match_sv_inflection_case_and_boundaries(tb):
    m = tb.match_sv("Patienten har Metaboliska acidoser.", ["metabolisk acidos"])
    assert [x[2] for x in m] == ["Metaboliska acidoser"]
    assert tb.match_sv("acidosgrad", ["acidos"]) == []  # remainder not Swedish
    assert tb.match_sv("kokain", ["kokainresistens"]) == []


def test_match_sv_compound_rule(tb):
    # prefix: form + Swedish remainder (stem >= 5)
    assert [x[2] for x in tb.match_sv("vankomycinresistens påvisades", ["vankomycin"])] == ["vankomycinresistens"]
    # suffix: Swedish word + form (linking s allowed)
    assert [x[2] for x in tb.match_sv("Kokainsacidos", ["acidos"])] == ["Kokainsacidos"]
    # compound-initial stem of a form ending in a vowel: skalle -> skallform
    assert [x[2] for x in tb.match_sv("Vilken skallform ses?", ["skalle"])] == ["skallform"]
    # hyphen variation
    assert tb.match_sv("vankomycin-resistens", ["vankomycin"])


# ---- ground / is_accepted_rendering --------------------------------------------------------


def test_ground_verified_and_obligations(tb):
    en = {"question": "Start IV vancomycin for metabolic acidosis.", "option:A": "Acyclovir", "option:B": "Heart failure"}
    sv = {"question": "Starta intravenöst vancomycin vid metabol acidos.", "option:A": "Aciklovir", "option:B": "Hjärtinsufficiens"}
    verified, obligations = tb.ground(en, sv)
    v = {(x.field, x.en_span): x for x in verified}
    assert v[("option:A", "Acyclovir")].sv_span == "Aciklovir" and v[("option:A", "Acyclovir")].level == "A"
    # 'metabol acidos' is not the A form, but the B form 'acidos' verifies it (level B)
    assert v[("question", "metabolic acidosis")].level == "B"
    o = {(x.field, x.en_span) for x in obligations}
    assert ("question", "vancomycin") in o  # English INN left in the Swedish text
    assert ("option:B", "Heart failure") not in o  # obligation_ok=False (non-cognate translation)
    for x in obligations:
        assert x.level == "A" and x.sv_forms


def test_ground_respects_alignment_and_single_sense(tb):
    en = {"question": "Depression and vancomycin", "option:A": "vancomycin"}
    sv = {"question": "Nedstämdhet och vankomycin", "option:A": "kokain"}
    verified, obligations = tb.ground(en, sv)
    assert [(x.field, x.en_span) for x in obligations] == [("option:A", "vancomycin")]
    assert all(x.entry_id != "swemesh:D003863:A:1" for x in obligations)  # not single-sense


def test_is_accepted_rendering(tb):
    assert tb.is_accepted_rendering("vancomycin", "vankomycin")
    assert tb.is_accepted_rendering("vancomycin", "vankomycinet")
    assert tb.is_accepted_rendering("acyclovir", "Aciklovir")
    assert tb.is_accepted_rendering("metabolic acidosis", "metabolisk acidos")
    # cognate modifier variant of the head: 'metabol acidos'
    assert tb.is_accepted_rendering("metabolic acidosis", "metabol acidos")
    # the modifier must be a cognate of an English word inside the term
    assert not tb.is_accepted_rendering("metabolic acidosis", "respiratorisk acidos")
    # extra content in the Swedish quote that is not covered -> no veto
    assert not tb.is_accepted_rendering("IV vancomycin", "intravenöst vankomycin")
    # wrong rendering / no term / not single-sense / B-only -> no veto
    assert not tb.is_accepted_rendering("vancomycin", "vancomycin")
    assert not tb.is_accepted_rendering("the patient", "patienten")
    assert not tb.is_accepted_rendering("depression", "depression")
    assert not tb.is_accepted_rendering("", "vankomycin")


# ---- built artefacts (skipped when the build is absent) ------------------------------------

needs_build = pytest.mark.skipif(not (BUILD / "termbase.parquet").exists(), reason="term base not available (data/resources or REFINE_RESOURCES)")


@pytest.fixture(scope="module")
def built():
    lx = Lexicon.load(BUILD)
    return lx, TermBase.load(BUILD, lexicon=lx)


@needs_build
def test_built_lexicon(built):
    lx, _ = built
    assert lx.is_swedish("hjärtinfarkt") and lx.is_swedish("acidosen") and lx.is_swedish("sternum")
    assert lx.is_english("blood") and lx.is_english("kidney")
    assert not lx.is_english("bilateral")  # also Swedish -> removed from the English list
    assert "EKG" in lx.whitelist and "ekg" in lx.whitelist


@needs_build
def test_built_npl_join(built):
    import json

    _, tb = built
    manifest = json.loads((BUILD / "MANIFEST.json").read_text())
    if "wikidata_atc" not in manifest["sources"]:
        pytest.skip("Wikidata download was not available at build time")
    for en, sv in [("vancomycin", "vankomycin"), ("acyclovir", "aciklovir"), ("vincristine", "vinkristin"), ("alteplase", "alteplas")]:
        hits = tb.spot(en, "question")
        assert hits and hits[0].level == "A", en
        assert sv in [f.lower() for f in hits[0].sv_forms], (en, hits[0].sv_forms)
        assert tb.is_accepted_rendering(en, sv)
    assert any(tb.entries[h.entry_id]["source"] == "npl_atc" for h in tb.spot("vancomycin", "question"))


@needs_build
def test_built_examples(built):
    _, tb = built
    assert tb.is_accepted_rendering("metabolic acidosis", "metabol acidos")
    assert tb.is_accepted_rendering("type 2 diabetes mellitus", "typ 2-diabetes")
    assert tb.is_accepted_rendering("acetaminophen", "paracetamol")
    _, obl = tb.ground({"option:A": "Vancomycin"}, {"option:A": "Vancomycin"})
    assert [o.en_span for o in obl] == ["Vancomycin"]
    ver, obl = tb.ground({"option:A": "Vancomycin"}, {"option:A": "Vankomycin"})
    assert not obl and ver[0].sv_span == "Vankomycin"


# ---- keep list, drugs, substitution ----------------------------------------------------------

from swemedmcq.refine.termbase import is_transliteration, translit_key  # noqa: E402


def test_keep_list_loaded_lowercased():
    lx = Lexicon(sv_words=(), en_words=(), whitelist={"keep": ["Hereditet", "persisterande  ductus arteriosus"]})
    assert lx.keep == {"hereditet", "persisterande ductus arteriosus"}
    assert Lexicon(sv_words=(), en_words=()).keep == set()


def test_keep_list_prefers_source_whitelist(tmp_path):
    build = tmp_path / "build"
    build.mkdir()
    (build / "whitelist.yaml").write_text("keep: [alt]\n", encoding="utf-8")
    (tmp_path / "whitelist.yaml").write_text("keep: [Lungröntgen, påverkad]\n", encoding="utf-8")
    from swemedmcq.refine.termbase import _load_keep

    assert _load_keep(build) == ["Lungröntgen", "påverkad"]
    (tmp_path / "whitelist.yaml").write_text("abbreviations: [EKG]\n", encoding="utf-8")
    assert _load_keep(build) == ["alt"]  # fallback to the build copy


@pytest.mark.parametrize(
    "en, sv",
    [("Phenytoin", "fenytoin"), ("clopidogrel", "klopidogrel"), ("canagliflozin", "kanagliflozin"),
     ("methyldopa", "metyldopa"), ("demeclocycline", "demeklocyklin"), ("primaquine", "primakin"),
     ("methicillin", "meticillin"), ("vincristine", "vinkristin"), ("carvedilol", "karvedilol"),
     ("ethambutol", "etambutol"), ("vancomycin", "vankomycin"), ("etoricoxib", "etoricoxib")],
)
def test_transliteration(en, sv):
    assert is_transliteration(en, sv)


@pytest.mark.parametrize("en, sv", [("hydroxyurea", "hydroxikarbamid"), ("acetaminophen", "paracetamol"),
                                    ("tromethamine", "trometamol"), ("vancomycin", "vankomycin resistens")])
def test_not_transliteration(en, sv):
    assert not is_transliteration(en, sv)
    assert translit_key("Phenytoin") == translit_key("fenytoin")


DRUG_ROWS = ROWS + [
    dict(entry_id="npl_atc:L01XX05:A:1", en_term="hydroxycarbamide", en_key="hydroxycarbamide",
         sv_forms=["hydroxikarbamid"], level="A", single_sense=True, obligation_ok=True, source="npl_atc",
         concept_id="L01XX05", drug=True),
    dict(entry_id="npl_atc:L01XX05:A:2", en_term="hydroxyurea", en_key="hydroxyurea",
         sv_forms=["hydroxikarbamid"], level="A", single_sense=True, obligation_ok=True, source="npl_atc",
         concept_id="L01XX05", drug=True),
    dict(entry_id="swemesh:D006918:A:1", en_term="Hydroxyurea", en_key="hydroxyurea", sv_forms=["hydroxiurea"],
         level="A", single_sense=True, obligation_ok=True, source="swemesh", concept_id="D006918", drug=True),
    dict(entry_id="swemesh:D008712:A:1", en_term="Methicillin", en_key="methicillin", sv_forms=["meticillin"],
         level="A", single_sense=True, obligation_ok=True, source="swemesh", concept_id="D008712", drug=True),
    dict(entry_id="swemesh:D007822:A:1", en_term="Lacerations", en_key="laceration",
         sv_forms=["laceration", "lacerationer"], level="A", single_sense=True, obligation_ok=True, source="swemesh",
         concept_id="D007822", drug=False),
]


@pytest.fixture(scope="module")
def dtb() -> TermBase:
    lx = _lexicon()
    lx.sv_words |= {"mandibularkanalen", "kanal", "underkäkskanal", "mandibular", "en", "fick", "flera", "styr",
                    "levern", "i", "gavs", "igen", "ges"}
    lx.en_words |= {"nuclear", "factor", "laceration", "lacerations", "hepatocyte"}
    lx._morph = {"noun": {"tumör": ["u", {"sg indef nom": ["tumör"], "sg def nom": ["tumören"],
                                          "pl indef nom": ["tumörer"], "pl def nom": ["tumörerna"]}]},
                 "adj": {}, "pos": {"tumör": "nn", "tumören": "nn", "tumörer": "nn", "tumörerna": "nn"}}
    return TermBase(DRUG_ROWS, lx, npl_substances=["vankomycin", "hydroxikarbamid", "aciklovir"])


def test_drug_inn(dtb):
    assert dtb.drug_inn("vancomycin") == "vankomycin"
    assert dtb.drug_inn("Vancomycin") == "vankomycin"
    assert dtb.drug_inn_evidence("vancomycin") == ("vankomycin", "npl")
    assert dtb.drug_inn("acyclovir") == "aciklovir"  # y->i, c->k
    # MeSH drug descriptor when NPL has no name (meticillin is not marketed)
    assert dtb.drug_inn_evidence("methicillin") == ("meticillin", "swemesh")
    # NPL name is not a transliteration: never fall back to the MeSH transliteration 'hydroxiurea'
    assert dtb.drug_inn("hydroxyurea") is None
    # not a drug / not in the term base / multi-word
    assert dtb.drug_inn("laceration") is None
    assert dtb.drug_inn("skull") is None
    assert dtb.drug_inn("metabolic acidosis") is None


def test_substitution_inflection_and_drugs(dtb):
    # transliterated drug: every occurrence, case and Swedish suffix carried over
    out = dtb.substitution("Vancomycinet gavs; vancomycin igen.", "vancomycin", "vankomycin")
    assert out == ("Vankomycinet gavs; vankomycin igen.", 2)
    # English plural -> Swedish plural (heuristic -er for a Latin -ion noun)
    out = dtb.substitution("Hon fick flera lacerations.", "lacerations", "laceration")
    assert out == ("Hon fick flera lacerationer.", 1)
    # SALDOM inflection of the Swedish lemma ('tumörer' for an English plural)
    assert dtb.substitution("flera tumors", "tumors", "tumör") == ("flera tumörer", 1)
    assert dtb.substitution("tumoren växer", "tumor", "tumör") == ("tumören växer", 1)  # SALDOM sg def
    assert dtb.substitution("tumorerna växer", "tumor", "tumör") == ("tumörerna växer", 1)
    # unknown Swedish lemma and a definite suffix: inflection unknown -> None
    assert dtb.substitution("lacerationen", "laceration", "lacerering") is None


def test_substitution_refuses_swedish_targets_and_names(dtb):
    # Swedish words are never replaced by a term-base form (e.g. Mandibularkanalen -> underkäkskanal)
    assert dtb.substitution("Mandibularkanalen syns.", "Mandibularkanalen", "underkäkskanal") is None
    # inside an English gene/protein name: left alone
    assert dtb.substitution("Hepatocyte nuclear factor styr levern.", "Hepatocyte", "hepatocyt") is None
    out = dtb.substitution("Hepatocyte nuclear factor styr; en hepatocyte i levern.", "hepatocyte", "hepatocyt")
    assert out == ("Hepatocyte nuclear factor styr; en hepatocyt i levern.", 1)
    # nothing to replace
    assert dtb.substitution("inget här", "vancomycin", "vankomycin") == ("inget här", 0)


# ---- Swedish validity (built lexicon) ---------------------------------------------------------

V2_VALID = ("bukkalyta interokklusalt farmakoterapier triacylglycerider Mesangiell retromammär sideroblaster "
            "iliolumbala njurbikarbonatutsöndring arterierna miktion kostovertebral ascenderande mikrosomala "
            "patellär esofagusatres plasmareninaktivitet papillarmuskeln skafoidbenet sakroiliakaleden hyperekoisk "
            "impakterad ulnarnerven adnexala faryngeala mandibulära paravertebrala intraorbitala dermala "
            "underbukssmärta lökskalsutseende immunfluorescens prevalenskvot cyanotisk").split()
V2_INVALID = ("näsmekatus gjutariarbetare tendonreflexerna dactylit löksskalsutseende immunofluorescens obliqt "
              "relieving deliveries patterns ligands").split()


@needs_build
@pytest.mark.parametrize("word", V2_VALID)
def test_built_is_swedish_regressions(built, word):
    lx, _ = built
    assert lx.is_swedish(word), word


@needs_build
@pytest.mark.parametrize("word", V2_INVALID)
def test_built_not_swedish(built, word):
    lx, _ = built
    assert not lx.is_swedish(word), word


@needs_build
def test_built_drug_inn(built):
    _, tb = built
    for en, sv in [("Phenytoin", "fenytoin"), ("clopidogrel", "klopidogrel"), ("canagliflozin", "kanagliflozin"),
                   ("methyldopa", "metyldopa"), ("demeclocycline", "demeklocyklin"), ("primaquine", "primakin"),
                   ("methicillin", "meticillin"), ("vincristine", "vinkristin"), ("carvedilol", "karvedilol"),
                   ("ethambutol", "etambutol")]:
        assert tb.drug_inn(en) == sv, en
    assert tb.drug_inn("hydroxyurea") is None


@needs_build
def test_built_npl_priority_and_singular_forms(built):
    _, tb = built
    if not tb.npl_substances:
        pytest.skip("build without npl_substances.txt.gz")
    assert tb.spot("hydroxyurea", "q")[0].sv_forms[0] == "hydroxikarbamid"
    assert tb.spot("tromethamine", "q")[0].sv_forms[0] == "trometamol"
    for en, sg in [("laceration", "laceration"), ("immunoglobulin", "immunglobulin"), ("bacteriophage", "bakteriofag"),
                   ("hepatocyte", "hepatocyt")]:
        assert tb.spot(en, "q")[0].sv_forms[0].lower() == sg, en


@needs_build
def test_built_substitution(built):
    _, tb = built
    assert tb.substitution("en laceration och flera lacerations", "lacerations", "laceration") == \
        ("en laceration och flera lacerationer", 1)
    assert tb.substitution("Mandibularkanalen syns", "Mandibularkanalen", "underkäkskanal") is None
    assert tb.substitution("ulnarisnerven skadades", "ulnarisnerven", "ulnarnerv") is None
    assert tb.substitution("Hepatocyte nuclear factor-1 (HNF-1)", "Hepatocyte", "hepatocyt") is None
    assert tb.substitution("Maxilla", "maxilla", "maxill") is None  # whitelisted Latin anatomy
