"""Layer L1: conservative Swedish number locale.

Rules, all logged (sequential offsets) so they can be reverted:

* ``decimal_comma``: ``38.4 °C`` -> ``38,4 °C``, but ONLY when the number (or a
  range/dimension chain of numbers, ``0.5-1.5 mg``, ``2.5 x 3.1 cm``) is directly
  followed by a unit, ``°``, ``%`` or ``× 10^``.
* ``thousands_space``: ``21,000`` -> ``21 000``, only for comma-grouped numbers
  with at least 5 digits and either several groups (``1,250,000``), a round last
  group (``21,000``, ``12,500``), a count denominator (``14,350/mm3``) or a
  following noun (``17,137 remisser``); ``(2,684 till 11,719)`` may already be
  Swedish decimals and is left alone.

With the English source as reference (``en_text``), a comma-grouped token that occurs verbatim in
the English becomes ``1234`` / ``12 345`` (``thousands_en``) and a dotted decimal that occurs
verbatim in the English gets a decimal comma (``decimal_comma_en``); see ``to_swedish_locale``.

Never touched: codes (``E11.9``), section/version numbers (``2.3.1``, ``v1.2``),
chemical locants (``2,3-BPG``), lists (``1,2,3 & 4``), p-values, clock times
(``kl. 10.30``) and years.  The function is idempotent.
"""

from __future__ import annotations

import re

from swemedmcq.refine.types import Change, FieldKey

_L = "A-Za-zÅÄÖåäöÉéÜüµμ"

_UNITS = [
    # pressure / vitals
    "mmHg", "mm Hg", "cmH2O", "cm H2O", "kPa", "Pa", "bpm", "slag/min", "andetag/min", "/min",
    # concentration / amount
    "mmol", "µmol", "μmol", "umol", "nmol", "pmol", "mol", "mekv", "mEq", "mOsm", "mosm", "osm",
    "mM", "µM", "μM", "nM", "pM",
    # mass
    "kg", "mg", "µg", "μg", "mcg", "ng", "pg", "g",
    # volume
    "ml", "mL", "dl", "dL", "cl", "µl", "μl", "µL", "μL", "fl", "fL", "l", "L",
    # length
    "km", "cm", "mm", "µm", "μm", "nm", "m",
    # enzyme / units
    "IU", "IE", "mU", "mIU", "U", "E/l", "E/L", "kat", "µkat", "μkat",
    # radiation / energy / electrical
    "Gy", "cGy", "mGy", "mSv", "Sv", "Bq", "MBq", "kcal", "cal", "kJ", "J", "W", "mW", "V", "mV",
    "µV", "μV", "mA", "Hz", "kHz", "MHz", "dB", "Ω", "ohm",
    # time
    "ms", "sek", "s", "min", "h", "tim", "timmar", "timme", "dygn", "dagar", "dag", "veckor",
    "vecka", "månader", "månad", "år",
    # molecular / misc
    "kDa", "Da", "SD", "gånger", "procent", "promille", "Fr",
]
_UNIT_ALT = "|".join(re.escape(u) for u in sorted(_UNITS, key=len, reverse=True))
# a unit must not run on into a word ("m" in "mycket"); "/" may follow ("mg/kg")
UNIT_RE = rf"(?:°|%|‰|[x×]\s?10\b|(?:{_UNIT_ALT})(?![{_L}0-9]))"

_NUM = r"\d+(?:[.,]\d+)?"
_JOIN = r"(?:\s*[-\u2013\u2014−]\s*|\s+(?:till|och|eller|to|and|or)\s+|\s*[x×]\s*(?=\d))"
_CHAIN_RE = re.compile(rf"(?<![\w.,:/]){_NUM}(?:{_JOIN}{_NUM})*(?![.,]\d)\s?(?={UNIT_RE})")
_DEC_RE = re.compile(r"(?<![\d.,])(\d+)\.(\d+)(?![\d.,])")
_EXCLUDE_BEFORE_RE = re.compile(
    r"(?:\b[pP]\s*[<>=≤≥]\s*|\b[pP][\s-]?(?:värde|value)t?\s*(?:[<>=≤≥]|var|är|blev|på|of|was|is)?\s*"
    r"|\bkl\.?\s*|\bklockan\s*|\bversion\s*|\bv\.?\s*|§\s*|\bavsnitt\s*|\bkapitel\s*|\btabell\s*"
    r"|\bfigur\s*|\bfig\.?\s*|\bpunkt\s*|\bsteg\s*|\bfas\s*|\bsektion\s*|\bDSM[\w-]*\s*|\bICD[\w-]*\s*)$"
)
_COUNT_AFTER_RE = re.compile(
    r"\s?/\s?(?:mm|µl|μl|ul|µL|μL|uL|ml|mL|mikroliter|l\b|L\b)"
    r"|\s+(?!(?:till|och|eller|vs|mot|respektive|to|and|or)\b)[^\W\d_]{3,}"  # "17,137 remisser"
)
# comma-grouped thousands; a hyphen before the number is fine after a digit or space
# ("150 000-350,000") but not after a letter ("IL-1,000" is not a range)
_THOUSANDS_RE = re.compile(r"(?<![\w.,:/])(?<![^\W\d_]-)(\d{1,3}(?:,\d{3})+)(?![\d]|[.,]\d|-?[A-Za-z])")
# a number with one decimal point, not part of a code (E11.9), a dotted chain (2.3.1) or a word
_EN_DECIMAL_RE = re.compile(r"(?<![\w.,:/])(?<![^\W\d_]-)(\d+\.\d+)(?![\d]|[.,]\d|[^\W\d_])")
_EN_TIME_AFTER_RE = re.compile(r"\s?(?:[ap]\.?m\b|o'clock|h\b|hrs?\b)", re.IGNORECASE)
_EN_TIME_BEFORE_RE = re.compile(r"\b(?:at|by|until|from|before|after)\s+$", re.IGNORECASE)
_EN_NUMBER_TOKEN_RE = re.compile(r"(?<![\w.,])\d+(?:[.,]\d+)*(?![\d\w])")


def _english_number_tokens(en_text: str | None) -> set[str]:
    """Number tokens of the English source, verbatim ("6,800", "0.75", "1,250,000")."""
    if not en_text:
        return set()
    out: set[str] = set()
    for m in _EN_NUMBER_TOKEN_RE.finditer(en_text):
        tok = m.group(0)
        if "." in tok and (_EN_TIME_AFTER_RE.match(en_text, m.end()) or _EN_TIME_BEFORE_RE.search(en_text[: m.start()])):
            continue  # "at 10.30", "10.30 am": a clock time, not a decimal
        out.add(tok)
    return out


def to_swedish_locale(text: str, field: FieldKey, en_text: str | None = None) -> tuple[str, list[Change]]:
    """Apply the decimal-comma and thousands rules (see module docstring).

    With ``en_text`` (the English source field) two reference rules are added:

    * ``thousands_en``: a comma-grouped token (``1,234``, ``12,345``, ``1,234,567``) that also occurs
      verbatim in the English is an English thousands grouping: ``1234`` (4 digits) or ``12 345``;
    * ``decimal_comma_en``: a number with a decimal point that occurs verbatim in the English gets a
      decimal comma (``MIC 0.75, 1, 1,5`` -> ``0,75``), except p-values, codes, versions, section numbers
      and chains (``2.3.1``).

    Change offsets have sequential semantics (``4,702`` -> ``4702`` shortens the text)."""
    en_nums = _english_number_tokens(en_text)
    edits: dict[int, tuple[str, str, str]] = {}  # start -> (before, after, rule)
    for chain in _CHAIN_RE.finditer(text):
        prefix = text[max(0, chain.start() - 12): chain.start()]
        if _EXCLUDE_BEFORE_RE.search(prefix):
            continue
        for d in _DEC_RE.finditer(chain.group(0)):
            start = chain.start() + d.start()
            before = d.group(0)
            edits[start] = (before, before.replace(".", ","), "decimal_comma")
    if en_nums:
        for d in _EN_DECIMAL_RE.finditer(text):
            tok = d.group(1)
            if tok not in en_nums or d.start() in edits:
                continue
            prefix = text[max(0, d.start() - 14): d.start()]
            if _EXCLUDE_BEFORE_RE.search(prefix):
                continue
            edits[d.start()] = (tok, tok.replace(".", ","), "decimal_comma_en")
    for m in _THOUSANDS_RE.finditer(text):
        token = m.group(1)
        prefix = text[max(0, m.start() - 14): m.start()]
        if _EXCLUDE_BEFORE_RE.search(prefix):
            continue
        digits = token.replace(",", "")
        if token in en_nums:
            after = digits if len(digits) == 4 else token.replace(",", " ")
            edits[m.start()] = (token, after, "thousands_en")
            continue
        if len(digits) < 5:
            continue
        # "11,719" may already be a Swedish decimal: require several groups, a
        # round last group ("21,000", "12,500"), a count denominator ("/mm3") or a
        # following noun ("17,137 remisser")
        groups = token.split(",")
        if not (len(groups) >= 3 or groups[-1].endswith("00") or _COUNT_AFTER_RE.match(text, m.end())):
            continue
        edits[m.start()] = (token, token.replace(",", " "), "thousands_space")
    changes: list[Change] = []
    out = text
    shift = 0
    for start, (before, after, rule) in sorted(edits.items()):
        pos = start + shift  # sequential semantics: offset in the text just before this change
        out = out[:pos] + after + out[pos + len(before):]
        changes.append(Change(field=field, rule=rule, start=pos, before=before, after=after))
        shift += len(after) - len(before)
    return out, changes


def revert(text: str, changes: list[Change]) -> str:
    """Undo ``changes`` (sequential semantics), last change first.

    Raises ``ValueError`` if the text no longer holds a change's ``after`` at its offset.
    """
    for ch in reversed(changes):
        end = ch.start + len(ch.after)
        if text[ch.start:end] != ch.after:
            raise ValueError(f"cannot revert {ch.rule} at {ch.start}: text changed")
        text = text[: ch.start] + ch.before + text[end:]
    return text
