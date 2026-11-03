"""Routing of confirmed findings and the protected-term (`keep`) list.

Every finding that survives grounding is put into exactly one class; the class decides how it
is repaired and how the edit is accepted:

    DRUG           English INN in the Swedish text, NPL Swedish INN differs only by transliteration
                   -> code substitution at every occurrence + GPT-OSS integration pass
                   -> gate + non-inferiority voting (evidence line)
    TERM-A         terminology finding with a single-sense level-A term-base form (non-drug)
                   -> GPT-OSS integration repair with the form as a hard constraint -> strict pairwise
    LEAK           English leftover without a term-base form -> GPT-OSS span repair -> strict pairwise
    ACCURACY       mistranslation / omission / addition (incl. missing numbers, corrupt fields)
                   -> confirmed by both models -> GPT-OSS span repair -> strict pairwise
    GRAMMAR        major/critical LLM grammar or spelling finding (P3) -> confirmed by both models
                   -> GPT-OSS span repair -> strict pairwise
    DET            deterministic: missing secondary unit (code insertion, gate only), noun gender and
                   spelling (GPT-OSS span repair, gate, strict pairwise)
    SV-NOEVIDENCE  terminology finding on a Swedish span without a level-A form -> reported only
    PROTECTED      span is on the `keep` list -> vetoed at grounding
    REPORTED       other policy-reported findings (source issues, minor LLM language findings,
                   minor leakage without a form)
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable

from ..io import resources_dir

DRUG, TERM_A, LEAK, ACCURACY, DET = "DRUG", "TERM-A", "LEAK", "ACCURACY", "DET"
GRAMMAR = "GRAMMAR"
SV_NOEVIDENCE, PROTECTED, REPORTED = "SV-NOEVIDENCE", "PROTECTED", "REPORTED"

ACCURACY_CATEGORIES = {"accuracy/mistranslation", "accuracy/omission", "accuracy/addition"}
FLUENCY_CATEGORIES = {"fluency/grammar", "fluency/spelling"}
LLM_SOURCES = {"gptoss", "qwen", "gemma"}
# deterministic findings that are confirmed by their own evidence (no LLM verifier)
SELF_CONFIRMED_SOURCES = {"corrupt", "units", "drug"}
VERIFY_ROUTES = {TERM_A, LEAK, ACCURACY, DET, GRAMMAR}
# routes that need confirmation by BOTH models (GPT-OSS verify + Qwen k-sample majority)
DUAL_CONFIRM_ROUTES = {ACCURACY, GRAMMAR}

_WORD_RE = re.compile(r"[^\W\d_]+(?:-[^\W\d_]+)*", re.UNICODE)
# Swedish inflection endings accepted after a protected term (påverkad -> påverkade, stomi -> stomin)
_SUFFIX = r"(?:e|a|t|n|s|en|et|na|ns|er|ar|or|na|ne|erna|arna|orna|ens|ets|es)?"
# words that may surround a protected term inside a finding quote without making it a different span
_FILLER = {
    "en", "ett", "den", "det", "de", "är", "var", "och", "i", "på", "med", "av", "som", "om", "till", "för",
    "har", "hade", "vid", "mycket", "lätt", "allmänt", "the", "a",
}


def load_keep(lexicon: Any = None, path: Path | None = None) -> set[str]:
    """Protected terms: the source whitelist's `keep:` list (original case, for the prompts) plus
    any further ``Lexicon.keep`` entries (lowercased there)."""
    terms: set[str] = set()
    path = path or resources_dir() / "whitelist.yaml"
    if path.exists():
        import yaml

        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        terms = {str(t) for t in (data.get("keep") or [])}
    lowered = {" ".join(t.lower().split()) for t in terms}
    for t in getattr(lexicon, "keep", None) or ():
        if " ".join(str(t).lower().split()) not in lowered:
            terms.add(str(t))
    return terms


def _participle_variants(term: str) -> list[str]:
    """The term plus the neuter/plural forms of a final participle or adjective, which the suffix
    rule cannot produce: medtagen -> medtaget/medtagna, påverkad -> påverkat/påverkade,
    opåverkad -> opåverkat."""
    head, _, last = term.rpartition(" ")
    pre = head + " " if head else ""
    out = [term]
    low = last.lower()
    if low.endswith("en") and len(low) > 4:
        out += [pre + last[:-2] + e for e in ("et", "na", "ne")]
    elif low.endswith("ad") and len(low) > 4:
        out += [pre + last[:-1] + e for e in ("t", "e")]
    elif low.endswith("d") and len(low) > 4 and low[-2] not in "aeiouyåäö":
        out += [pre + last[:-1] + "t", pre + last + "a", pre + last + "e"]
    return out


class ProtectedTerms:
    """Whole-word, case-insensitive matcher for the `keep` list (inflected forms included)."""

    def __init__(self, terms: Iterable[str], lexicon: Any = None) -> None:
        # Inflection is matched by Swedish endings after the term (påverkad -> påverkade, stomi ->
        # stomin), not by Lexicon.forms: SALDO lemmatisation maps participles to their verbs
        # (påverkad -> påverkar, utsättande -> utsatt) and would protect ordinary verbs.
        self.terms = sorted({t.strip() for t in terms if t and t.strip()}, key=lambda t: (-len(t), t.lower()))
        self._rx: list[tuple[str, re.Pattern[str]]] = []
        for t in self.terms:
            alts = [re.escape(v.lower()).replace(r"\ ", r"\s+") for v in _participle_variants(t)]
            alt = "(?:" + "|".join(sorted(set(alts), key=len, reverse=True)) + ")"
            self._rx.append((t, re.compile(r"(?<![\w-])" + alt + _SUFFIX + r"(?![\w])", re.IGNORECASE)))

    def __bool__(self) -> bool:
        return bool(self.terms)

    @property
    def lowered(self) -> set[str]:
        return {t.lower() for t in self.terms}

    def occurrences(self, text: str) -> list[tuple[int, int, str]]:
        out: list[tuple[int, int, str]] = []
        taken: list[tuple[int, int]] = []
        for term, rx in self._rx:  # longest terms first
            for m in rx.finditer(text or ""):
                s, e = m.span()
                if any(s < te and ts < e for ts, te in taken):
                    continue
                taken.append((s, e))
                out.append((s, e, term))
        return sorted(out)

    def present(self, *texts: str) -> list[str]:
        found: list[str] = []
        for text in texts:
            found.extend(t for _, _, t in self.occurrences(text))
        return list(dict.fromkeys(found))

    def counts(self, text: str) -> dict[str, int]:
        out: dict[str, int] = {}
        for _, _, t in self.occurrences(text):
            out[t] = out.get(t, 0) + 1
        return out

    def removed(self, current: str, candidate: str) -> list[str]:
        """Protected terms that occur fewer times in ``candidate`` than in ``current``."""
        before, after = self.counts(current), self.counts(candidate)
        return [t for t, n in before.items() if after.get(t, 0) < n]

    def covers(self, text: str, span: tuple[int, int] | None, quote: str) -> str | None:
        """The protected term a finding quote is on, or None.

        The quote is on a protected term when it lies inside an occurrence of the term, or when it
        consists of the term plus at most filler words (articles, copula, degree adverbs)."""
        occ = self.occurrences(text)
        if span is not None:
            s, e = span
            for os_, oe, term in occ:
                if os_ <= s and e <= oe:
                    return term
                if s <= os_ and oe <= e:
                    rest = (text[s:os_] + " " + text[oe:e]).lower()
                    if all(w in _FILLER for w in _WORD_RE.findall(rest)):
                        return term
            return None
        # quote not located in the text: test the quote on its own
        for os_, oe, term in self.occurrences(quote):
            rest = (quote[:os_] + " " + quote[oe:]).lower()
            if all(w in _FILLER for w in _WORD_RE.findall(rest)):
                return term
        return None


def words(text: str) -> list[str]:
    return _WORD_RE.findall(text or "")


def is_english_token(tok: str, lexicon: Any, en_text: str = "") -> bool:
    """An English token per the leakage lexicon (``is_english and not is_swedish``), or a token
    copied verbatim from the English source that is not a Swedish word."""
    if not tok or len(tok) < 3:
        return False
    whitelist = getattr(lexicon, "whitelist", None) or set()
    if tok in whitelist or tok.lower() in whitelist:
        return False
    if lexicon.is_english(tok) and not lexicon.is_swedish(tok):
        return True
    if en_text and tok.casefold() in {w.casefold() for w in words(en_text)} and not lexicon.is_swedish(tok):
        return not tok.isupper()  # abbreviations are kept as in the source
    return False


def span_is_english(quote: str, lexicon: Any, en_text: str = "") -> bool:
    """A finding span is 'English' when it contains an English token (see :func:`is_english_token`),
    or when a multi-word span is copied verbatim from the English source (``peer support``) and is
    not a whitelisted Latin/international phrase or an all-caps abbreviation."""
    toks = words(quote)
    if any(is_english_token(t, lexicon, en_text) for t in toks):
        return True
    q = " ".join(quote.split())
    if len(toks) >= 2 and en_text and not q.isupper():
        whitelist = getattr(lexicon, "whitelist", None) or set()
        if q in whitelist or q.lower() in whitelist:
            return False
        en_norm = " ".join(en_text.split()).casefold()
        if re.search(r"(?<!\w)" + re.escape(q.casefold()) + r"(?!\w)", en_norm) and any(len(t) >= 4 for t in toks):
            return True
    return False
