"""Deterministic language checks (route DET): noun gender and spelling suspects.

Both are precision-first: they only report cases that SALDOM / the lexicon make unambiguous, and the
findings are routed to an LLM repair whose result must pass the gate (new tokens in the lexicon).

* :func:`gender_errors` - an indefinite article (``en``/``ett``) or an indefinite adjective that
  disagrees with the gender of the following noun (``ett prevalenskvot`` -> ``en prevalenskvot``,
  ``ett förhöjd värde`` -> ``ett förhöjt värde``).  The noun (or the last part of a compound) must
  be a singular indefinite noun form with exactly one gender in SALDOM and no other part of speech.
* :func:`spelling_suspects` - alphabetic tokens of at least five letters that are not Swedish
  (``Lexicon.is_swedish``, compounds included), not English, not in the English source, not
  whitelisted or protected, and within edit distance 2 of a Swedish word (``löksskalsutseende`` ->
  ``lökskalsutseende``, ``immunofluorescens`` -> ``immunfluorescens``, ``obliqt`` -> ``oblikt``).

The lexicon is duck-typed (``swemedmcq.refine.termbase.Lexicon``); without SALDOM morphology (``analyses``,
``noun_gender``) :func:`gender_errors` reports nothing.
"""

from __future__ import annotations

import re
from typing import Any

_TOKEN_RE = re.compile(r"[^\W\d_]+", re.UNICODE)
_WORD_RE = re.compile(r"[^\W\d_]+(?:-[^\W\d_]+)*", re.UNICODE)
_ALPHABET = "abcdefghijklmnopqrstuvwxyzåäöé"
_ARTICLES = {"en": "u", "ett": "n"}
_ARTICLE_FOR = {"u": "en", "n": "ett"}
_MAX_SUGGESTIONS = 5


# ---------------------------------------------------------------------------
# gender
# ---------------------------------------------------------------------------


def _adj_genders(lexicon: Any, form: str) -> tuple[set[str], str | None]:
    """Genders an indefinite singular adjective form agrees with, and its lemma (one lemma only)."""
    an = [a for a in lexicon.analyses(form) if a[1] == "av"]
    if not an or {a[1] for a in lexicon.analyses(form)} != {"av"}:
        return set(), None
    lemmas = {a[0] for a in an}
    if len(lemmas) != 1:
        return set(), None
    genders = set()
    for _lem, _pos, _g, msd in an:
        if msd == "pos indef sg u nom":
            genders.add("u")
        elif msd == "pos indef sg n nom":
            genders.add("n")
        else:
            genders.add("other")
    return genders, lemmas.pop()


def _adj_form(lexicon: Any, lemma: str, gender: str) -> str | None:
    forms = lexicon.morph["adj"].get(lemma, {}).get(f"pos indef sg {gender} nom") or []
    return forms[0] if len(forms) == 1 else None


def _case_like(src: str, new: str) -> str:
    return new[:1].upper() + new[1:] if src[:1].isupper() else new


def gender_errors(sv: str, lexicon: Any) -> list[dict]:
    """Article / adjective gender disagreements with the head noun (see module docstring).

    Each item: ``start``, ``end`` (span in ``sv``), ``span`` (``sv[start:end]``) and ``expected``
    (the corrected span)."""
    if not hasattr(lexicon, "noun_gender") or not hasattr(lexicon, "analyses"):
        return []
    toks = [(m.group(0), m.start(), m.end()) for m in _WORD_RE.finditer(sv)]
    out: list[dict] = []
    for i, (art, s, e) in enumerate(toks):
        a = art.lower()
        if a not in _ARTICLES or i + 1 >= len(toks):
            continue
        # tokens must be adjacent words (only whitespace between them)
        def adjacent(j: int) -> bool:
            return j < len(toks) and sv[toks[j - 1][2]: toks[j][1]].strip() == ""

        if not adjacent(i + 1):
            continue
        w1, _s1, e1 = toks[i + 1]
        if lexicon.is_english(w1):
            continue
        g1 = lexicon.noun_gender(w1) if "-" not in w1 else None
        if g1 is not None:
            nxt = toks[i + 2][0] if adjacent(i + 2) else ""
            is_wl = getattr(lexicon, "is_whitelisted", lambda _t: False)
            if nxt and not lexicon.is_swedish(nxt) and not is_wl(nxt):
                continue  # inside a foreign phrase ('ett international normalized ratio')
            if g1 != _ARTICLES[a]:
                expected = _case_like(art, _ARTICLE_FOR[g1]) + sv[e:e1]
                out.append(dict(start=s, end=e1, span=sv[s:e1], expected=expected))
            continue
        # article + adjective + noun
        if not adjacent(i + 2):
            continue
        genders, lemma = _adj_genders(lexicon, w1.lower())
        if not genders or "other" in genders or len(genders) != 1:
            continue
        w2, _s2, e2 = toks[i + 2]
        g2 = lexicon.noun_gender(w2) if "-" not in w2 else None
        if g2 is None:
            continue
        adj_g = next(iter(genders))
        art_ok = g2 == _ARTICLES[a]
        adj_ok = adj_g == g2
        if art_ok and adj_ok:
            continue
        new_art = art if art_ok else _case_like(art, _ARTICLE_FOR[g2])
        new_adj = w1 if adj_ok else _adj_form(lexicon, lemma, g2)
        if new_adj is None:
            continue
        new_adj = _case_like(w1, new_adj)
        expected = new_art + sv[e:toks[i + 1][1]] + new_adj + sv[e1:e2]
        out.append(dict(start=s, end=e2, span=sv[s:e2], expected=expected))
    return out


# ---------------------------------------------------------------------------
# spelling
# ---------------------------------------------------------------------------


def _edits1(w: str) -> set[str]:
    splits = [(w[:i], w[i:]) for i in range(len(w) + 1)]
    deletes = {a + b[1:] for a, b in splits if b}
    transposes = {a + b[1] + b[0] + b[2:] for a, b in splits if len(b) > 1}
    replaces = {a + c + b[1:] for a, b in splits if b for c in _ALPHABET if c != b[0]}
    inserts = {a + c + b for a, b in splits for c in _ALPHABET}
    return deletes | transposes | replaces | inserts


def _within2(a: str, b: str) -> int | None:
    """Levenshtein distance if <= 2, else None (banded)."""
    la, lb = len(a), len(b)
    if abs(la - lb) > 2:
        return None
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [99] * lb
        lo, hi = max(1, i - 2), min(lb, i + 2)
        best = cur[0] if lo == 1 else 99
        for j in range(lo, hi + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] != b[j - 1]))
            best = min(best, cur[j])
        if best > 2:
            return None
        prev = cur
    return prev[lb] if prev[lb] <= 2 else None


class _Buckets:
    """Lexicon words grouped by (first two letters, length) for the distance-2 search."""

    def __init__(self, words: set[str]):
        self.by: dict[tuple[str, int], list[str]] = {}
        for w in words:
            if len(w) >= 4 and w.isalpha():
                self.by.setdefault((w[:2], len(w)), []).append(w)

    def near(self, tok: str) -> list[tuple[int, str]]:
        out = []
        for n in range(len(tok) - 2, len(tok) + 3):
            for w in self.by.get((tok[:2], n), ()):
                d = _within2(tok, w)
                if d is not None:
                    out.append((d, w))
        return out


_BUCKETS: dict[int, _Buckets] = {}


def _buckets(lexicon: Any) -> _Buckets:
    key = id(lexicon)
    b = _BUCKETS.get(key)
    if b is None:
        b = _Buckets(set(getattr(lexicon, "sv_words", ())))
        _BUCKETS.clear()
        _BUCKETS[key] = b
    return b


_INFL_ENDINGS = ("", "a", "e", "n", "t", "s", "r", "en", "et", "er", "ar", "or", "na", "ns", "ts", "es", "erna", "arna",
                 "orna", "ens", "ets", "ers", "ars", "ade", "at", "ad", "de", "te", "da", "ta")


def _inflectional_pair(a: str, b: str) -> bool:
    """``a`` and ``b`` differ only in their inflectional ending (spermatogonie / spermatogonier):
    both are probably valid, so neither is a spelling error."""
    i = 0
    while i < min(len(a), len(b)) and a[i] == b[i]:
        i += 1
    return i >= 4 and a[i:] in _INFL_ENDINGS and b[i:] in _INFL_ENDINGS


def _core(lexicon: Any, w: str) -> bool:
    """A suggestion the lexicon vouches for: a SALDOM form or a longer term-bank / hunspell word."""
    pos = getattr(lexicon, "pos", None)
    return bool(pos(w)) if callable(pos) and len(w) < 7 else w in getattr(lexicon, "sv_words", set())


_LATIN_END = re.compile(r"(?:us|um|is|ae|ii|ior|ium|ia|eus|ei|orum|arum|alis|aris|ica|icum|icus|ata|atum|osa|osum|ea|eum|ae)$")
_ANGLICISMS = (("th", "t"), ("ph", "f"), ("qu", "kv"), ("q", "k"), ("ca", "ka"), ("co", "ko"), ("cu", "ku"), ("cl", "kl"),
               ("cr", "kr"), ("ct", "kt"))


def _attested(lexicon: Any, v: str, min_prefix: int) -> bool:
    """``v`` is a lexicon word, the beginning of one, or a lexicon word of at least ``min_prefix``
    letters followed by a Swedish remainder (``lökskal|s|utseende``)."""
    words = getattr(lexicon, "sv_words", set())
    if v in words:
        return True
    pa = getattr(lexicon, "_prefix_attested_exact", None)
    if callable(pa) and pa(v):
        return True
    for k in range(max(min_prefix, 4), len(v) - 3):
        if v[:k] in words:
            rest = v[k:]
            rest = rest[1:] if rest[:1] == "s" and not lexicon.is_swedish(rest) else rest
            if len(rest) >= 4 and lexicon.is_swedish(rest):
                return True
    return False


def _english_or_boundary(t: str, lexicon: Any) -> bool:
    """The English-looking letters are an English word at the start ('thrust|teknik') or straddle a
    compound boundary ('vikt|hantering', 'erytrocyt|hemolysat'): not an anglicised spelling."""
    if lexicon.is_english(t) or any(lexicon.is_english(t[:k]) and lexicon.is_swedish(t[k:]) for k in range(6, len(t) - 4)):
        return True
    for m in re.finditer(r"(?=th|ph)", t):
        i = m.start() + 1
        head, tail = t[:i], t[i:]
        if len(head) >= 3 and len(tail) >= 4 and lexicon.is_swedish(head) and lexicon.is_swedish(tail):
            return True
    return False


def _typo_variants(t: str) -> set[str]:
    """Keyboard-typo corrections: adjacent transposition, a consonant doubled (a doubled letter is
    never removed: Latin loans keep theirs, 'adducera')."""
    out = {t[:i] + t[i + 1] + t[i] + t[i + 2:] for i in range(len(t) - 1) if t[i] != t[i + 1]}
    out |= {t[:i] + t[i] + t[i:] for i in range(1, len(t) - 1) if t[i] not in "aeiouyåäö"}
    return out


def suggestions(tok: str, lexicon: Any) -> list[str]:
    """Corrections of ``tok`` at edit distance <= 2 that the lexicon supports strongly, nearest first.

    Precision first, three kinds only:

    * a superfluous linking letter at a compound boundary (``immuno|fluorescens`` -> ``immunfluorescens``,
      ``löks|skal…`` -> ``lökskal…``), the corrected word checked with ``Lexicon.is_swedish``;
    * an English spelling of a Swedish word (``th``/``ph``/``c``/``qu`` -> ``t``/``f``/``k``/``kv``:
      ``thenareminenser`` -> ``tenareminenser``), not for Latin forms (``-us``, ``-alis``, ...);
    * a keyboard typo (transposition, doubled or undoubled letter) giving a lexicon word.

    General substitutions (``mikrocytär`` -> ``makrocytär``) are never suggested: in medical text
    they usually turn one valid word into another."""
    t = tok.lower()
    words = getattr(lexicon, "sv_words", set())
    found: dict[str, int] = {}
    for i in range(3, len(t) - 3):
        if t[i] in "so" and (t[i] == "o" or t[i - 1] == "s" or t[i + 1] == "s"):
            v = t[:i] + t[i + 1:]
            if lexicon.is_swedish(v) and _attested(lexicon, v, i + 3):
                found[v] = 1
    if not _LATIN_END.search(t) and not _english_or_boundary(t, lexicon):
        v = t
        for a, b in _ANGLICISMS[:4]:
            v = v.replace(a, b)
        w = v
        for a, b in _ANGLICISMS[4:]:
            w = w.replace(a, b)
        if v != t and (v in words or lexicon.is_swedish(v)):  # th/ph/qu: never Swedish spellings
            found.setdefault(v, 1)
        elif w != t and len(t) >= 7 and (w in words or _attested(lexicon, w, len(w))):  # c -> k: attested only
            found.setdefault(w, 1)
    if len(t) >= 7:
        pos = getattr(lexicon, "pos", None)
        for v in _typo_variants(t):
            if v in words and (not callable(pos) or pos(v)):  # SALDOM forms only (no hunspell noise)
                found.setdefault(v, 1)
    return [w for _, w in sorted((d, w) for w, d in found.items())][:_MAX_SUGGESTIONS]


def _protected(tok: str, lexicon: Any) -> bool:
    keep = getattr(lexicon, "keep", None) or ()
    t = tok.lower()
    return any(t == k or t in k.split() for k in keep)


def spelling_suspects(sv: str, lexicon: Any, en: str = "") -> list[dict]:
    """Likely misspellings (see module docstring).  Each item: ``start``, ``end``, ``token`` and
    ``suggestions`` (lexicon words at edit distance <= 2, nearest first, at most five)."""
    en_tokens = {t.casefold() for t in _TOKEN_RE.findall(en or "")}
    out: list[dict] = []
    seen: set[str] = set()
    for m in _TOKEN_RE.finditer(sv):
        tok = m.group(0)
        if len(tok) < 5 or tok.isupper() or any(c.isupper() for c in tok[1:]):
            continue
        if tok[:1].isupper():
            before = sv[: m.start()].rstrip()
            if before and before[-1] not in ".!?:":
                continue  # a capitalised word inside a sentence: a name
        key = tok.casefold()
        if key in seen or key in en_tokens:
            continue
        seen.add(key)
        is_wl = getattr(lexicon, "is_whitelisted", None)
        if (is_wl(tok) if callable(is_wl) else tok.lower() in getattr(lexicon, "whitelist", set())):
            continue
        if _protected(tok, lexicon) or lexicon.is_english(tok) or lexicon.is_swedish(tok):
            continue
        sug = suggestions(tok, lexicon)
        if not sug or len(sug) > 3:
            continue  # nothing close, or too many neighbours to say which word was meant
        if any(_inflectional_pair(key, w) for w in sug):
            continue  # an inflection the lexicon lacks, not a misspelling
        out.append(dict(start=m.start(), end=m.end(), token=tok, suggestions=sug))
    return out
