"""locale: conservative decimal comma / thousands space, exclusions, revert, idempotence."""

from __future__ import annotations

import pytest

from swemedmcq.refine.locale import revert, to_swedish_locale


def loc(text: str) -> str:
    return to_swedish_locale(text, "question")[0]


@pytest.mark.parametrize(
    "before, after",
    [
        ("Temperatur 38.4 °C.", "Temperatur 38,4 °C."),
        ("Temperatur 38.4°C och 101.1°F.", "Temperatur 38,4°C och 101,1°F."),
        ("Ge 1.5 mg/kg iv.", "Ge 1,5 mg/kg iv."),
        ("Kalium 7.2 mmol/L", "Kalium 7,2 mmol/L"),
        ("Andelen var 12.5 % i gruppen", "Andelen var 12,5 % i gruppen"),
        ("Andelen var 12.5% i gruppen", "Andelen var 12,5% i gruppen"),
        ("Hb 12.5 g/dL", "Hb 12,5 g/dL"),
        ("dos 0.5-1.5 mg", "dos 0,5-1,5 mg"),
        ("dos 0.5 till 1.5 mg", "dos 0,5 till 1,5 mg"),
        ("en 2.5 x 3.1 cm stor resistens", "en 2,5 x 3,1 cm stor resistens"),
        ("kreatinin 1.7 mg/dL (referens 0.6-1.2 mg/dL)", "kreatinin 1,7 mg/dL (referens 0,6-1,2 mg/dL)"),
        ("tryck 22.5 mmHg", "tryck 22,5 mmHg"),
        ("vikt 70.3 kg", "vikt 70,3 kg"),
        ("T-score -2.5 SD", "T-score -2,5 SD"),
        ("LPK 4.5 × 10^9/L", "LPK 4,5 × 10^9/L"),
        ("i 2.5 år", "i 2,5 år"),
        ("Leukocyter 21,000/mm3", "Leukocyter 21 000/mm3"),
        ("Trombocyter 250,000/µL", "Trombocyter 250 000/µL"),
        ("totalt 1,250,000 invånare", "totalt 1 250 000 invånare"),
        ("Av 17,137 remisser", "Av 17 137 remisser"),
    ],
)
def test_conversions(before, after):
    out, changes = to_swedish_locale(before, "question")
    assert out == after
    assert changes and revert(out, changes) == before


@pytest.mark.parametrize(
    "text",
    [
        "Diagnoskod E11.9 och J18.9",
        "ICD-10 I21.4 mg",  # a code followed by something unit-like
        "2,3-BPG ökar",
        "1,25-dihydroxivitamin D",
        "1,25-(OH)2-vitamin D",
        "Påståenden 1,2,3 & 4 är korrekta",
        "p < 0.05",
        "p=0.001",
        "P = 0.0008 %",
        "kl. 10.30",
        "klockan 22.15",
        "år 2019",
        "avsnitt 2.3.1 mm",
        "version 1.2 m",
        "pH 7.35",  # no unit: left alone (conservative)
        "BMI 31.2",
        "OR 2.1 (95 % KI 1.2-3.4)",
        "Faktor 1,200 eller 1,500",  # 4-digit comma groups are ambiguous
        "konfidensintervall 2,684 till 11,719)",  # may already be Swedish decimals
    ],
)
def test_exclusions_unchanged(text):
    assert to_swedish_locale(text, "question") == (text, [])


def test_mixed_measurement_and_section():
    assert loc("13.5 mg enligt avsnitt 2.3.1") == "13,5 mg enligt avsnitt 2.3.1"


def test_p_value_next_to_percentage():
    assert loc("54.1% vs 20.7%, P = 0.0008") == "54,1% vs 20,7%, P = 0.0008"


def test_already_swedish_and_idempotent():
    text = "Temperatur 39,5 °C, leukocyter 12 000/mm3, 38.4°C"
    once = loc(text)
    assert once == "Temperatur 39,5 °C, leukocyter 12 000/mm3, 38,4°C"
    assert loc(once) == once


def test_change_log_offsets():
    text = "a 1.5 mg b 21,000/mm3"
    out, changes = to_swedish_locale(text, "option:B")
    assert [(c.rule, c.start, c.before, c.after, c.field) for c in changes] == [
        ("decimal_comma", 2, "1.5", "1,5", "option:B"),
        ("thousands_space", 11, "21,000", "21 000", "option:B"),
    ]
    assert len(out) == len(text)


def test_revert_detects_tampering():
    _out, changes = to_swedish_locale("1.5 mg", "question")
    with pytest.raises(ValueError):
        revert("9,9 mg", changes)


def test_word_starting_with_unit_letter_is_not_a_unit():
    assert loc("grupp 1.2 med") == "grupp 1.2 med"
    assert loc("steg 2.1 gäller") == "steg 2.1 gäller"
    assert loc("1.5 m lång") == "1,5 m lång"


# ---- the English source as reference ------------------------------------------------------


def loc_en(sv: str, en: str) -> str:
    return to_swedish_locale(sv, "question", en_text=en)[0]


@pytest.mark.parametrize(
    "sv, en, after",
    [
        # English thousands commas left in the drafts
        ("Leukocytantal: 6,800/mm^3", "Leukocyte count: 6,800/mm^3", "Leukocytantal: 6800/mm^3"),
        ("AST 4,702 U/L", "AST 4,702 U/L", "AST 4702 U/L"),
        ("(n = 1,250)", "(n = 1,250)", "(n = 1250)"),
        ("mer än 2,300 medelålders män", "more than 2,300 middle-aged men", "mer än 2300 medelålders män"),
        ("Trombocyter 150 000-350,000", "Platelets 150,000-350,000", "Trombocyter 150 000-350 000"),
        ("konfidensintervall 2,684 till 11,719)", "confidence interval 2,684 to 11,719)",
         "konfidensintervall 2684 till 11 719)"),
        ("totalt 1,250,000 invånare", "a total of 1,250,000 inhabitants", "totalt 1 250 000 invånare"),
        # decimal points that are verbatim in the English
        ("MIC var 0.75, 1, 1,5", "MIC was 0.75, 1, 1.5", "MIC var 0,75, 1, 1,5"),
        ("pH 7.35 och BMI 31.2", "pH 7.35 and BMI 31.2", "pH 7,35 och BMI 31,2"),
        ("OR 2.1 (95 % KI 1.2-3.4)", "OR 2.1 (95% CI 1.2-3.4)", "OR 2,1 (95 % KI 1,2-3,4)"),
    ],
)
def test_english_reference(sv, en, after):
    out, changes = to_swedish_locale(sv, "question", en_text=en)
    assert out == after
    assert revert(out, changes) == sv  # sequential offsets survive length changes ('4,702' -> '4702')
    assert loc_en(out, en) == out  # idempotent


@pytest.mark.parametrize(
    "sv, en",
    [
        ("p = 0.03", "p = 0.03"),
        ("p<0.001", "p<0.001"),
        ("P-värdet 0.04", "P value 0.04"),
        ("Diagnoskod E11.9", "Diagnosis code E11.9"),
        ("enligt avsnitt 2.1", "according to section 2.1"),
        ("version 1.2", "version 1.2"),
        ("avsnitt 2.3.1", "section 2.3.1"),
        ("2,3-BPG ökar", "2,3-BPG increases"),
        ("1,25-dihydroxivitamin D", "1,25-dihydroxyvitamin D"),
        ("kl. 10.30", "at 10.30"),
        ("10.30", "at 10.30 am"),
        ("IL-1,000", "IL-1,000"),
        ("Faktor 1,200", "Factor 1.200"),  # not verbatim: left alone
    ],
)
def test_english_reference_exclusions(sv, en):
    assert to_swedish_locale(sv, "question", en_text=en) == (sv, [])


def test_english_reference_rules_are_logged():
    _, changes = to_swedish_locale("AST 4,702 U/L, 0.75 och 21,000/mm3", "option:A",
                                   en_text="AST 4,702 U/L, 0.75 and 21,000/mm3")
    assert [(c.rule, c.before, c.after) for c in changes] == [
        ("thousands_en", "4,702", "4702"),
        ("decimal_comma_en", "0.75", "0,75"),
        ("thousands_en", "21,000", "21 000"),
    ]
    assert changes[1].start == 14  # 15 in the input, minus the comma removed by the first change


def test_without_english_reference_unchanged_behaviour():
    # 4-digit groups are ambiguous without the English text
    assert to_swedish_locale("AST 4,702 U/L", "question") == ("AST 4,702 U/L", [])
    # the hyphen look-behind is fixed for ranges in the conservative rule too
    assert loc("Trombocyter 150 000-350,000/µL") == "Trombocyter 150 000-350 000/µL"
