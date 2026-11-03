"""numbers: missing-number detection."""

from __future__ import annotations

import pytest

from swemedmcq.refine.numbers import NumberMiss, missing_numbers


def miss(en: str, sv: str) -> list[str]:
    return [m.token for m in missing_numbers(en, sv, "question")]


@pytest.mark.parametrize(
    "en, sv",
    [
        ("38.4 °C", "38,4 °C"),
        ("0.5 mg", "0,5 mg"),
        (".5 µm", "0,5 µm"),
        ("2.50 mg", "2,5 mg"),
        ("21,000 cells", "21 000 celler"),
        ("21,000 cells", "21000 celler"),
        ("21,000 cells", "21\u00a0000 celler"),
        ("1,250,000 people", "1 250 000 personer"),
        ("count 3,000/mm3", "3,000/mm3"),
        ("45%", "45 %"),
        ("45 %", "45%"),
        ("45% of patients", "45 procent av patienterna"),
        ("3 patients", "tre patienter"),
        ("1 tablet", "en tablett"),
        ("one of 2 sons", "en av två söner"),
        ("the 1st dose", "den första dosen"),
        ("every 2nd day", "vartannat dygn"),
        ("a 3-year-old boy", "en treårig pojke"),
        ("a 3-year-old", "en treåring"),
        ("1/2 tablet", "en halv tablett"),
        ("type 2 diabetes", "typ II-diabetes"),
        ("phase 3 trial", "fas III-studie"),
        ("factor 8 deficiency", "brist på faktor VIII"),
        ("grade 4 tumour", "grad IV-tumör"),
        ("stage 2b", "stadium IIb"),
        ("NYHA class 3", "NYHA-klass III"),
        ("Le Fort 2 fracture", "Le Fort II-fraktur"),
        ("2,3-BPG", "2,3-BPG"),
        ("chromosomes 1,5", "kromosom 1 och 5"),
        ("at 10:30", "kl. 10.30"),
        ("at 2:00 AM", "kl. 02.00"),
        ("at 10 pm", "kl. 22"),
        ("over 24 hours", "under det senaste dygnet"),
        ("on 3/12/2020", "den 3 december 2020"),
        ("100.4°F", "38 °C"),
        ("154 lb", "70 kg"),
        ("E11.9", "E11.9"),
        ("5 and 15", "15 och 5"),
        ("CD4 count 200", "CD4-tal 200"),
    ],
)
def test_found(en, sv):
    assert miss(en, sv) == []


def test_no_substring_hits():
    assert miss("5 mg and 15 mg", "15 mg") == ["5 mg"]
    assert miss("1.5", "11,5") == ["1.5"]


def test_multiset_counts():
    assert miss("5 then 5", "5") == ["5"]
    assert miss("5 then 5", "5 och 5") == []


def test_real_omissions_are_reported():
    assert miss("leukocyte count of 14,000/mm3", "leukocyter 1400/mm3") == ["14,000"]
    assert miss("temperature 37.6°C (99.7°F)", "temperatur 37,6 °C") == ["99.7°F"]


def test_kinds_and_tokens():
    out = missing_numbers("45% had 5 mg and 7 episodes", "inget", "option:C")
    assert out == [
        NumberMiss(token="45%", field="option:C", kind="percent"),
        NumberMiss(token="5 mg", field="option:C", kind="unit"),
        NumberMiss(token="7", field="option:C", kind="number"),
    ]


def test_ambiguous_swedish_run_prefers_the_matching_reading():
    # "3,5" read as decimal for the English 3.5, as a list for "3 and 5"
    assert miss("Grade I (3.5%) and 3 more", "Grad I (3,5 %) och 3 till") == []
    assert miss("items 3,5", "punkterna 3,5") == []
    assert miss("platelets 85,000/mm3 and 3,000/mm3", "85 000/mm3 och 3,000/mm3") == []


def test_unambiguous_evidence_is_used_first():
    # "3,000" must not steal the "3" of "mm3"
    assert miss("Leukocyte count 3,000/mm3", "Leukocyter 3,000/mm3") == []


def test_no_numbers():
    assert missing_numbers("No digits here", "", "question") == []


def test_number_words_do_not_match_inside_words():
    # "sex" is not a compound head ("sexuell"), "tio" not in "station"
    assert miss("6 partners", "sexuella partner") == ["6"]
    assert miss("10 stations", "stationer") == ["10"]


# ---- word / unit equivalences ---------------------------------------------------------------------


def test_equivalences():
    assert miss("cervix 100% effaced", "livmoderhalsen helt utplånad") == []
    assert miss("the 2nd most common cause", "den näst vanligaste orsaken") == []
    assert miss("the 2nd dose", "den andra dosen") == []
    assert miss("Hb12a level 5", "nivå 5") == []  # digits inside a garbled token are ignored
    # still strict elsewhere
    assert miss("100 mg", "helt") == ["100 mg"]
    assert miss("2 tablets", "näst sista") == ["2"]
    assert miss("the 2nd most common", "den nästa vanliga") == ["2"]


# ---- secondary units -----------------------------------------------------------------------------

from swemedmcq.refine.numbers import secondary_unit_insertions  # noqa: E402


def _insert(sv: str, ins: list[tuple[int, str]]) -> str:
    for off, text in reversed(ins):
        sv = sv[:off] + text + sv[off:]
    return sv


@pytest.mark.parametrize(
    "en, sv, after",
    [
        ("Temperature is 37°C (98.6°F).", "Temperaturen är 37 °C.", "Temperaturen är 37 °C (98,6 °F)."),
        ("temp 37.6 °C (99.7 °F), weight 70 kg (154 lb), height 170 cm (5 ft 7 in)",
         "temp 37,6 °C, vikt 70 kg, längd 170 cm",
         "temp 37,6 °C (99,7 °F), vikt 70 kg (154 lb), längd 170 cm (5 ft 7 in)"),
        ("She weighs 1,250 kg (2,756 lb)", "Hon väger 1 250 kg", "Hon väger 1 250 kg (2756 lb)"),
        # imperial primary, metric kept: the imperial value goes after the metric one
        ("98.6°F (37°C)", "37 °C", "37 °C (98,6 °F)"),
        ("a distance of 5 km (3.1 mi)", "en sträcka på 5 km", "en sträcka på 5 km (3,1 mi)"),
    ],
)
def test_secondary_unit_insertions(en, sv, after):
    ins = secondary_unit_insertions(en, sv)
    assert _insert(sv, ins) == after
    assert secondary_unit_insertions(en, after) == []  # nothing left to insert


@pytest.mark.parametrize(
    "en, sv",
    [
        ("37°C (98.6°F)", "37 °C (98,6 °F)"),  # already there
        ("lost 6 pounds (2.7 kg)", "gått ned 6 pounds (2,7 kg)"),  # secondary present (English unit)
        ("37°C (98.6°F)", "38 °C"),  # primary value not in the Swedish
        ("37°C", "37 °C"),  # no pair
        ("5 mg (10 mL)", "5 mg"),  # not a unit-conversion pair
    ],
)
def test_secondary_unit_no_insertion(en, sv):
    assert secondary_unit_insertions(en, sv) == []
