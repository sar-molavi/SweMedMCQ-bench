"""Numbers: which numbers of an English field are missing in its Swedish rendering.

* Whole-number matching on canonical values (no substring hits: "5" is not in "15").
* Decimal point == decimal comma; thousands grouping ("21,000" == "21 000" == "21000").
* Ambiguous Swedish runs ("2,3", "10.30", "3 100") have several readings
  (decimal / thousands / list); the reading that matches most English numbers wins.
* Swedish number words 0-20 (+ tens, ordinals, "vartannat", "halv"), month names,
  Roman numerals after type/phase/factor/grade/stage/class/Le Fort (Swedish forms).
* Unit conversions are accepted (°F -> °C, lb -> kg, in -> cm, a.m./p.m. -> 24 h).
* Multiset semantics: "5 ... 5" in English needs two 5s in Swedish.

Also: "100% effaced" ~ "helt utplånad", 2nd ~ "näst"/"andra", digits inside garbled alphanumeric tokens
("Hb12a") are ignored, and ``secondary_unit_insertions`` restores dropped "(98,6 °F)"-style units.

"""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation

from pydantic import BaseModel

from swemedmcq.refine.types import FieldKey

NUM_WORDS: dict[int, str] = {
    0: "noll", 1: "en|ett|första|enda|halv", 2: "två|andra|båda|bägge|par|vartannat|varannan",
    3: "tre|tredje", 4: "fyra|fjärde", 5: "fem|femte", 6: "sex|sjätte", 7: "sju|sjunde",
    8: "åtta|åttonde", 9: "nio|nionde", 10: "tio|tionde", 11: "elva|elfte", 12: "tolv|tolfte",
    13: "tretton|trettonde", 14: "fjorton|fjortonde", 15: "femton|femtonde", 16: "sexton|sextonde",
    17: "sjutton|sjuttonde", 18: "arton|artonde", 19: "nitton|nittonde", 20: "tjugo|tjugonde",
    30: "trettio", 40: "fyrtio", 50: "femtio", 100: "hundra", 1000: "tusen",
}
ROMAN: dict[int, str] = {1: "I", 2: "II", 3: "III", 4: "IV", 5: "V", 6: "VI", 7: "VII", 8: "VIII",
                         9: "IX", 10: "X", 11: "XI", 12: "XII"}
IMPERIAL = r"(?:°\s?F|lbs?\b|pounds?|ft\b|feet|foot|in\b|inch(?:es)?|oz\b|ounces?|miles?)"
MONTHS = ["januari", "februari", "mars", "april", "maj", "juni", "juli", "augusti", "september",
          "oktober", "november", "december"]

_ROMAN_VAL = {v: k for k, v in ROMAN.items()}
_WS = " \u00a0\u202f\u2009"

# Swedish word -> value (exact word match), plus cardinals 2..20 that may start a compound
_WORD_VALUE: dict[str, int] = {}
for _v, _alts in NUM_WORDS.items():
    for _w in _alts.split("|"):
        _WORD_VALUE[_w] = _v
_WORD_VALUE.update({"halva": 1, "halvt": 1, "hälften": 1, "båda": 2, "bägge": 2, "tvåa": 2, "trea": 3,
                    "fyra": 4, "femma": 5, "sexa": 6, "sjua": 7, "åtta": 8, "nia": 9, "tia": 10,
                    "dubbel": 2, "dubbla": 2, "dubbelt": 2, "trippel": 3, "ettan": 1, "tvåan": 2,
                    "trean": 3, "fyran": 4, "femman": 5})
_WORD_VALUE.update({m: i + 1 for i, m in enumerate(MONTHS)})
_HALF_WORDS = {"halv", "halva", "halvt", "hälften"}
# "100% effaced" ~ "helt utplånad": a Swedish word for "complete" stands for an English 100 %
_FULL_WORDS = {"helt", "hel", "hela", "fullständig", "fullständigt", "fullständiga", "total", "totalt", "totala",
               "komplett", "kompletta", "fullt", "full", "fulla"}
_FULL = "full"
# "2nd most common" ~ "näst vanligaste": näst- stands for an English ordinal 2nd/second
_NAST_RE = re.compile(r"(?i)(?<![^\W\d_])näst[\s-]?[^\W\d_]*?(?:sta|ste|st)(?![^\W\d_])")
_ORD2 = "ord2"
_EN_ORDINAL_RE = re.compile(r"(?:st|nd|rd|th)\b", re.IGNORECASE)
_DAY_WORDS = {"dygn", "dygnet", "dygnets", "dygns"}  # 24 hours
_COMPOUND_HEADS = {w: v for w, v in _WORD_VALUE.items()
                   if 1 <= v <= 1000 and len(w) >= 3
                   and w not in {"par", "sex", "båda", "bägge", "vartannat", "varannan", "enda", "halv"}
                   and w not in MONTHS}
_COMPOUND_TAILS = (r"(?:-?(?:årig\w*|åring\w*|års\w*|dagars|dygns|veckors|månaders|timmars|minuters|"
                   r"sekunders|faldig\w*|dubbl\w*|siffrig\w*|kärl\w*|del\w*|sidig\w*|gång\w*|stegs?\w*|"
                   r"faktor\w*|grads?\w*|tiden|talet|kammar\w*|ledad\w*|linje\w*|generation\w*|hand\w*|ordning\w*))")
_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)
_COMPOUND_RE = re.compile(r"(?i)(?<![^\W\d_])(" + "|".join(sorted(_COMPOUND_HEADS, key=len, reverse=True))
                          + r")" + _COMPOUND_TAILS + r"(?![^\W\d_])")
_ROMAN_CTX_RE = re.compile(
    r"(?i:typ|fas|faktor|grad|stadium|stadie|klass|kategori|le\s+fort|type|phase|factor|grade|stage|class)"
    r"[^\W\d_]{0,4}[\s\-]+(?-i:(XII|XI|IX|X|VIII|VII|VI|IV|V|III|II|I))(?:[a-c]\b|(?![A-Za-z]))"
)

_EN_RUN_RE = re.compile(r"(?<!\d)(?:\d+(?:[.,]\d+)*|(?<![\w.])\.\d+)")
_SV_RUN_RE = re.compile(
    rf"(?<!\d)(?:\d{{1,3}}(?:[{_WS}]\d{{3}})+(?![\d])(?:[.,]\d+)?|\d+(?:[.,]\d+)*|(?<![\w.])[.,]\d+)"
)
_UNIT_AFTER_RE = re.compile(
    r"[\s-]?(?:°\s?[CF]|°|mmHg|mm\s?Hg|cmH2O|kPa|bpm|mmol|µmol|μmol|nmol|pmol|mEq|mOsm|mg|µg|μg|mcg|ng|pg|kg|g|"
    r"ml|mL|dL|dl|L|l|cm|mm|µm|μm|nm|km|m|IU|U|Gy|cGy|mSv|Hz|kcal|kDa|min|h|hours?|hrs?|days?|weeks?|"
    r"months?|years?|lbs?|pounds?|ft|feet|inch(?:es)?|oz|miles?|breaths|beats)(?![A-Za-z])"
)
_PERCENT_AFTER_RE = re.compile(r"\s?%")


class NumberMiss(BaseModel):
    """An English number with no counterpart in the Swedish field."""

    token: str
    field: FieldKey
    kind: str  # "number" | "percent" | "unit"


def _canon(s: str) -> str | None:
    """Canonical decimal string ("2.50" -> "2.5", "007" -> "7", ".5" -> "0.5")."""
    try:
        d = Decimal(s)
    except InvalidOperation:
        return None
    out = format(d.normalize(), "f")
    return out if out != "-0" else "0"


def _readings(run: str, *, english: bool) -> list[list[str]]:
    """Possible numeric readings of a digit run, most natural first."""
    run = run.strip()
    if any(c in _WS for c in run):  # Swedish space-grouped thousands, e.g. "21 000" or "12 500,5"
        m = re.fullmatch(rf"([\d{_WS}]+?)(?:([.,])(\d+))?", run)
        groups = re.split(f"[{_WS}]", m.group(1)) if m else [run]
        frac = (m.group(3) or "") if m else ""
        tail = "." + frac if frac else ""
        whole = _canon("".join(groups) + tail)
        parts = [x for x in map(_canon, groups[:-1] + [groups[-1] + tail]) if x]
        return [r for r in ([whole] if whole else [], parts) if r]
    if run.startswith((".", ",")):
        return [[x for x in (_canon("0." + run[1:]),) if x]]
    seps = re.findall(r"[.,]", run)
    pieces = re.split(r"[.,]", run)
    if not seps:
        return [[x for x in (_canon(run),) if x]]
    as_list = [x for x in map(_canon, pieces) if x]
    grouped3 = all(len(p) == 3 for p in pieces[1:]) and 1 <= len(pieces[0]) <= 3
    readings: list[list[str]] = []
    if english:
        if set(seps) == {","} and grouped3:
            readings.append([_canon("".join(pieces)) or ""])
        elif len(seps) == 1 and seps[0] == ".":
            readings.append([_canon(run) or ""])
        elif "," in seps and "." == seps[-1] and all(len(p) == 3 for p in pieces[1:-1]) and \
                all(s == "," for s in seps[:-1]) and 1 <= len(pieces[0]) <= 3:
            readings.append([_canon("".join(pieces[:-1]) + "." + pieces[-1]) or ""])
        else:
            readings.append(as_list)
        return [r for r in readings if all(r)]
    # Swedish: decimal (either separator), thousands (3-digit groups), or a list
    if len(seps) == 1:
        readings.append([_canon(pieces[0] + "." + pieces[1]) or ""])
    if grouped3:
        readings.append([_canon("".join(pieces)) or ""])
    if len(seps) >= 2 and len(set(seps)) == 2 and all(len(p) == 3 for p in pieces[1:-1]):
        readings.append([_canon("".join(pieces[:-1]) + "." + pieces[-1]) or ""])
    readings.append(as_list)
    return [r for r in readings if all(r)]


def _f_to_c(v: Decimal) -> list[str]:
    c = (v - 32) * Decimal(5) / Decimal(9)
    return [_canon(str(round(c, n))) or "" for n in (0, 1)]


def _scaled(v: Decimal, factor: str) -> list[str]:
    x = v * Decimal(factor)
    return [_canon(str(round(x, n))) or "" for n in (0, 1)]


class _EnNum:
    """One English number: surface token, kind, primary value and accepted alternatives."""

    __slots__ = ("accept", "kind", "token", "value")

    def __init__(self, token: str, kind: str, value: str, accept: set[str]):
        self.token, self.kind, self.value, self.accept = token, kind, value, accept


def _english_numbers(text: str) -> list[_EnNum]:
    out: list[_EnNum] = []
    for m in _EN_RUN_RE.finditer(text):
        run = m.group(0)
        if run == "00" and text[max(0, m.start() - 1): m.start()] == ":":
            continue  # full hour "2:00": Swedish "kl. 2" / "02.00" carry no minutes
        before = text[m.start() - 1: m.start()] if m.start() else ""
        if before.isalpha() and text[m.end(): m.end() + 1].isalpha():
            continue  # digits inside a garbled / alphanumeric token ("Hb12a", "T2N1M0")
        after = text[m.end(): m.end() + 12]
        readings = _readings(run, english=True)
        if not readings:
            continue
        values = readings[0]
        pm = _PERCENT_AFTER_RE.match(after)
        um = _UNIT_AFTER_RE.match(after)
        if pm:
            kind, suffix = "percent", "%"
        elif um:
            kind, suffix = "unit", um.group(0)
        else:
            kind, suffix = "number", ""
        multi = len(values) > 1
        for i, val in enumerate(values):
            accept = {val}
            last = i == len(values) - 1
            if last and not multi:
                dv = Decimal(val)
                if re.match(r"[\s-]?°\s?F", after):
                    accept.update(_f_to_c(dv))
                elif re.match(r"[\s-]?(?:lbs?|pounds?)\b", after):
                    accept.update(_scaled(dv, "0.45359237"))
                elif re.match(r"[\s-]?(?:inch(?:es)?|in\b)", after):
                    accept.update(_scaled(dv, "2.54"))
                elif re.match(r"[\s-]?(?:miles?)\b", after):
                    accept.update(_scaled(dv, "1.609344"))
                else:
                    ampm = re.match(r"(?::\d\d)?\s?([ap])\.?m\b", after, re.IGNORECASE)
                    if ampm and dv == dv.to_integral_value() and 1 <= dv <= 12:
                        hour = int(dv) % 12 + (12 if ampm.group(1).lower() == "p" else 0)
                        accept.add(str(hour))
                if kind == "percent" and dv == 100:
                    accept.add(_FULL)
                if dv == 2 and _EN_ORDINAL_RE.match(after):
                    accept.add(_ORD2)
            token = (run + suffix) if (not multi and suffix) else (run if not multi else val)
            out.append(_EnNum(token.strip(), kind if (last or not multi) else "number", val, accept))
    return out


def _swedish_readings(text: str) -> list[list[list[str]]]:
    """Per Swedish evidence unit: its alternative readings (each a list of values)."""
    units: list[list[list[str]]] = []
    for m in _SV_RUN_RE.finditer(text):
        r = _readings(m.group(0), english=False)
        if r:
            units.append(r)
    for m in _WORD_RE.finditer(text):
        w = m.group(0).lower()
        if w in _FULL_WORDS:
            units.append([[_FULL]])
        if w in _WORD_VALUE:
            val = str(_WORD_VALUE[w])
            if w in _HALF_WORDS:
                units.append([["0.5"], ["1", "2"], ["1"]])
                continue
            units.append([[val]])
        elif w in _DAY_WORDS:
            units.append([["24"], ["1"]])
    for m in _COMPOUND_RE.finditer(text):
        w = m.group(1).lower()
        if m.group(0).lower() != w:  # bare words were handled above
            units.append([[str(_WORD_VALUE[w])]])
    for m in _ROMAN_CTX_RE.finditer(text):
        units.append([[str(_ROMAN_VAL[m.group(1)])]])
    for _ in _NAST_RE.finditer(text):
        units.append([[_ORD2]])
    return units


def missing_numbers(en_text: str, sv_text: str, field: FieldKey) -> list[NumberMiss]:
    """English numbers of ``en_text`` with no counterpart in ``sv_text`` (multiset)."""
    english = _english_numbers(en_text)
    if not english:
        return []
    open_idx = list(range(len(english)))

    def assign(values: list[str], commit: bool, convert: bool) -> int:
        used: list[int] = []
        for v in values:
            for i in open_idx:
                if i not in used and (v in english[i].accept if convert else v == english[i].value):
                    used.append(i)
                    break
        if commit:
            for i in used:
                open_idx.remove(i)
        return len(used)

    units = _swedish_readings(sv_text)
    # phase 0: unambiguous Swedish evidence; phase 1: the natural reading of an
    # ambiguous run, if it matches completely
    pending: list[list[list[str]]] = []
    for readings in [u for u in units if len(u) == 1] + [u for u in units if len(u) > 1]:
        if readings[0] and assign(readings[0], False, False) == len(readings[0]):
            assign(readings[0], True, False)
        else:
            pending.append(readings)
    # phase 2: best alternative reading (exact values), then 3: with unit conversions
    for convert in (False, True):
        rest: list[list[list[str]]] = []
        for readings in pending:
            if not open_idx:
                break
            scores = [assign(r, False, convert) for r in readings]
            best = max(range(len(readings)), key=lambda k: (scores[k], -k))
            if scores[best]:
                assign(readings[best], True, convert)
            else:
                rest.append(readings)
        pending = rest
    return [NumberMiss(token=english[i].token, field=field, kind=english[i].kind) for i in open_idx]


# ---------------------------------------------------------------------------
# secondary units: "37.6°C (99.7°F)" in English, "37,6 °C" in Swedish -> insert " (99,7 °F)"
# ---------------------------------------------------------------------------

_NUM_RE = r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:[.,]\d+)?"
_SV_NUM_RE = rf"\d{{1,3}}(?:[{_WS}]\d{{3}})+(?:,\d+)?|\d+(?:[.,]\d+)?"
# unit family -> (English unit regex, Swedish unit regex, canonical Swedish symbol)
_UNIT_DEFS: dict[str, tuple[str, str, str]] = {
    "C": (r"°\s?C\b|degrees?\s+(?:C\b|Celsius)", r"°\s?C\b|grader(?:\s+Celsius)?", "°C"),
    "F": (r"°\s?F\b|degrees?\s+(?:F\b|Fahrenheit)", r"°\s?F\b|grader\s+Fahrenheit", "°F"),
    "kg": (r"kg\b|kilograms?\b", r"kg\b|kilo(?:gram)?\b", "kg"),
    "lb": (r"lbs?\b|pounds?\b", r"lbs?\b|pund\b|pounds?\b", "lb"),
    "cm": (r"cm\b|centimet(?:er|re)s?\b", r"cm\b|centimeter\b", "cm"),
    "m": (r"m\b|met(?:er|re)s?\b", r"m\b|meter\b", "m"),
    "km": (r"km\b|kilomet(?:er|re)s?\b", r"km\b|kilometer\b", "km"),
    "mi": (r"mi\b|miles?\b", r"mi\b|engelska\s+mil\b|miles?\b", "mi"),
    "ftin": (r"(?:ft|feet|foot)\b(?:\s*,?\s*(?:" + _NUM_RE + r")\s*(?:in|inch(?:es)?)\b)?",
             r"(?:ft|fot|feet|foot)\b(?:\s*(?:" + _SV_NUM_RE + r")\s*(?:in|tum|inch(?:es)?)\b)?", "ft"),
    "in": (r"(?:in|inch(?:es)?)\b", r"in\b|tum\b", "in"),
}
_PAIRS = {("C", "F"), ("F", "C"), ("kg", "lb"), ("lb", "kg"), ("cm", "ftin"), ("ftin", "cm"), ("m", "ftin"),
          ("ftin", "m"), ("cm", "in"), ("in", "cm"), ("km", "mi"), ("mi", "km")}
_IMPERIAL = {"F", "lb", "ftin", "in", "mi"}


def _val(token: str, *, english: bool) -> Decimal | None:
    t = token.strip()
    if english:
        t = t.replace(",", "")
    else:
        t = re.sub(f"[{_WS}]", "", t).replace(",", ".")
    try:
        return Decimal(t)
    except InvalidOperation:
        return None


def _sv_number(token: str) -> str:
    """English number token in Swedish format ("98.6" -> "98,6", "1,250" -> "1 250")."""
    t = token.replace(",", "")
    whole, dot, frac = t.partition(".")
    if len(whole) >= 5:
        whole = f"{int(whole):,}".replace(",", " ")
    return whole + ("," + frac if dot else "")


def _en_quantities(en: str, fam: str) -> re.Pattern[str]:
    return re.compile(rf"(?<![\w.,])({_NUM_RE})\s?-?(?:{_UNIT_DEFS[fam][0]})", re.IGNORECASE)


def _quantity_text(num: str, fam: str, unit_text: str) -> str:
    text = _sv_number(num) + " " + _UNIT_DEFS[fam][2]
    if fam == "ftin":
        inch = re.search(rf"({_NUM_RE})\s*(?:in|inch(?:es)?)\b", unit_text, re.IGNORECASE)
        if inch:
            text += " " + _sv_number(inch.group(1)) + " in"
    return text


def _en_secondary_pairs(en: str) -> list[tuple[str, Decimal, str, str, Decimal, str]]:
    """(primary family, value, Swedish text, secondary family, value, Swedish text) per English pair."""
    out = []
    fams = list(_UNIT_DEFS)
    unit_alt = "|".join(f"(?:{_UNIT_DEFS[f][0]})" for f in fams)
    rx = re.compile(rf"(?<![\w.,])({_NUM_RE})\s?-?({unit_alt})\s*\(\s*(?:approximately\s+|about\s+|~\s*|≈\s*)?"
                    rf"({_NUM_RE})\s?-?({unit_alt})\s*\)", re.IGNORECASE)
    for m in rx.finditer(en):
        p_fam = next((f for f in fams if re.fullmatch(_UNIT_DEFS[f][0], m.group(2), re.IGNORECASE)), None)
        s_fam = next((f for f in fams if re.fullmatch(_UNIT_DEFS[f][0], m.group(4), re.IGNORECASE)), None)
        if p_fam is None or s_fam is None or (p_fam, s_fam) not in _PAIRS:
            continue
        pv, sv_ = _val(m.group(1), english=True), _val(m.group(3), english=True)
        if pv is None or sv_ is None:
            continue
        out.append((p_fam, pv, _quantity_text(m.group(1), p_fam, m.group(2)),
                    s_fam, sv_, _quantity_text(m.group(3), s_fam, m.group(4))))
    return out


def _sv_quantities(sv: str, fam: str) -> list[tuple[Decimal | None, int]]:
    rx = re.compile(rf"(?<![\w.,])({_SV_NUM_RE})\s?-?(?:{_UNIT_DEFS[fam][1]})", re.IGNORECASE)
    return [(_val(m.group(1), english=False), m.end()) for m in rx.finditer(sv)]


def secondary_unit_insertions(en: str, sv: str) -> list[tuple[int, str]]:
    """Secondary units of the English that the Swedish dropped.

    For every English ``<value unit> (<value unit>)`` pair (°C/°F, kg/lb, cm or m/ft-in, cm/in,
    km/mi) whose primary quantity occurs in ``sv`` while the secondary value does not, return
    ``(offset, text)``: insert ``text`` (e.g. ``" (99,5 °F)"``, Swedish number format) at ``offset``,
    right after the Swedish primary quantity.  When the English primary is imperial and the Swedish
    kept only the metric secondary (``98.6°F (37°C)`` -> ``37 °C``), the imperial value is inserted
    after the metric one.  Sorted by offset; apply in reverse order."""
    out: list[tuple[int, str]] = []
    used: set[int] = set()

    def insert_after(fam: str, value: Decimal, text: str) -> bool:
        for v, end in _sv_quantities(sv, fam):
            if v != value or end in used or re.match(r"\s*\(", sv[end:]):
                continue
            used.add(end)
            out.append((end, f" ({text})"))
            return True
        return False

    for p_fam, pv, p_text, s_fam, sv_val, s_text in _en_secondary_pairs(en):
        has_p = any(v == pv for v, _ in _sv_quantities(sv, p_fam))
        has_s = any(v == sv_val for v, _ in _sv_quantities(sv, s_fam))
        if has_p and not has_s:
            insert_after(p_fam, pv, s_text)
        elif has_s and not has_p and p_fam in _IMPERIAL:
            insert_after(s_fam, sv_val, p_text)
    return sorted(out)
