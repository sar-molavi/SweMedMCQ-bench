"""Span utilities: robust quote finding, span edits, diffs and edit windows."""

from __future__ import annotations

import difflib
import itertools
import re

from swemedmcq.refine.types import SpanOp

_CHAR_MAP: dict[str, str] = {}
for _c in "\u2018\u2019\u201a\u201b\u2032\u00b4`":
    _CHAR_MAP[_c] = "'"
for _c in "\u201c\u201d\u201e\u201f\u2033\u00ab\u00bb":
    _CHAR_MAP[_c] = '"'
for _c in "\u2010\u2011\u2012\u2013\u2014\u2015\u2212":
    _CHAR_MAP[_c] = "-"
_CHAR_MAP["\u2026"] = "..."
for _c in "\u00ad\u200b\ufeff":  # invisible
    _CHAR_MAP[_c] = ""
_CLAUSE_DELIMS = ",;:.?!()\n"


def _normalise_with_map(s: str) -> tuple[str, list[int]]:
    """Normalised string and, per normalised char, the index of its source char.

    Casefold, unify quotes/dashes, collapse whitespace runs to one space and
    strip leading/trailing whitespace."""
    out: list[str] = []
    idx: list[int] = []
    pending_space = -1
    for i, ch in enumerate(s):
        if ch.isspace():
            if pending_space < 0:
                pending_space = i
            continue
        if pending_space >= 0:
            if out:
                out.append(" ")
                idx.append(pending_space)
            pending_space = -1
        for c in _CHAR_MAP.get(ch, ch).casefold():
            out.append(c)
            idx.append(i)
    return "".join(out), idx


def norm_for_match(s: str) -> str:
    """Casefold, collapse whitespace, unify quotes and dashes."""
    return _normalise_with_map(s)[0]


def _find_all(text: str, quote: str) -> list[tuple[int, int]]:
    """All non-overlapping occurrences: exact if any, else normalised (original offsets)."""
    if not quote or not quote.strip():
        return []
    spans: list[tuple[int, int]] = []
    k = text.find(quote)
    while k >= 0:
        spans.append((k, k + len(quote)))
        k = text.find(quote, k + len(quote))
    if spans:
        return spans
    nt, mp = _normalise_with_map(text)
    nq = norm_for_match(quote)
    if not nq:
        return []
    k = nt.find(nq)
    while k >= 0:
        end = k + len(nq)
        spans.append((mp[k], mp[end - 1] + 1))
        k = nt.find(nq, end)
    return spans


def _is_word_char(ch: str) -> bool:
    return ch.isalnum() or ch == "_"


def _whole_word(text: str, s: int, e: int) -> bool:
    """``text[s:e]`` starts and ends at a word boundary (a quote that starts or ends with a
    non-word character, e.g. ``(98,6 °F)``, is bounded by that character)."""
    if s > 0 and _is_word_char(text[s]) and _is_word_char(text[s - 1]):
        return False
    return not (e < len(text) and _is_word_char(text[e - 1]) and _is_word_char(text[e]))


def _find_all_words(text: str, quote: str, exact_first: bool = True) -> list[tuple[int, int]]:
    """Whole-word occurrences: exact ones if any (``exact_first``), else normalised ones (case,
    quotes, dashes and whitespace unified).  A normalised match is also tried when every exact
    occurrence sits inside a longer word."""
    if not quote or not quote.strip():
        return []
    quote = quote.strip()
    if exact_first:
        spans = [sp for sp in _find_all(text, quote) if _whole_word(text, *sp)]
        if spans:
            return spans
    nt, mp = _normalise_with_map(text)
    nq = norm_for_match(quote)
    out: list[tuple[int, int]] = []
    k = nt.find(nq) if nq else -1
    while k >= 0:
        end = k + len(nq)
        sp = (mp[k], mp[end - 1] + 1)
        if _whole_word(text, *sp):
            out.append(sp)
        k = nt.find(nq, end)
    return out


def find_quote(text: str, quote: str) -> tuple[int, int] | None:
    """``(start, end)`` of the first whole-word occurrence of ``quote`` in ``text``: exact first,
    else a normalised match mapped back to original offsets; ``None`` if absent or empty.

    Whole-word: ``mig`` is not found inside ``miglitol``."""
    spans = _find_all_words(text, quote)
    return spans[0] if spans else None


def find_all_quotes(text: str, quote: str) -> list[tuple[int, int]]:
    """Every whole-word occurrence of ``quote`` (see :func:`find_quote`)."""
    return _find_all_words(text, quote)


def apply_ops(text: str, ops: list[SpanOp]) -> str | None:
    """Apply span replacements.

    * Duplicate operations (same ``old`` and ``new``) are merged first; the same ``old`` with
      different ``new`` values is a conflict (``None``).
    * ``old`` is matched as whole words.  A single-word ``old`` that occurs several times is
      replaced at EVERY occurrence (the case of the first letter follows each occurrence); a
      multi-word ``old`` must occur exactly once.  An ``old`` that only occurs inside a longer
      word is accepted when it occurs there exactly once (a sub-word edit).
    * Missing ``old`` or overlapping operations -> ``None``."""
    merged: dict[str, str] = {}
    for op in ops:
        old = op.old
        if not old or not old.strip():
            return None
        if old in merged and merged[old] != op.new:
            return None
        merged[old] = op.new
    located: list[tuple[int, int, str]] = []
    for old, new in merged.items():
        spans = _find_all_words(text, old, exact_first=False)
        single_word = len(old.split()) == 1
        if not spans:
            spans = _find_all(text, old)
            if len(spans) != 1:
                return None
        elif len(spans) > 1 and not single_word:
            return None
        for s, e in spans:
            found = text[s:e]
            located.append((s, e, new if found == old else _match_case(found, new)))
    located.sort()
    for (_, e1, _), (s2, _, _) in itertools.pairwise(located):
        if s2 < e1:
            return None
    out = text
    for s, e, new in reversed(located):
        out = out[:s] + new + out[e:]
    return out


def _match_case(found: str, new: str) -> str:
    if not new or not found:
        return new
    if found[0].isupper():
        return new[0].upper() + new[1:]
    if found[0].islower() and not new[1:2].isupper():
        return new[0].lower() + new[1:]
    return new


def replace_all(text: str, old_forms: list[str], new_form: str) -> tuple[str, int]:
    """Replace every word-bounded, case-insensitive occurrence of any of
    ``old_forms`` (longest first) by ``new_form``, preserving the case of the
    first letter.  Returns ``(new_text, n_replacements)``; a match that already
    reads as the replacement is left alone and not counted."""
    forms = sorted({f for f in old_forms if f and f.strip()}, key=len, reverse=True)
    if not forms:
        return text, 0
    pattern = re.compile(
        r"(?<![\w])(?:" + "|".join(re.escape(f) for f in forms) + r")(?![\w])", re.IGNORECASE
    )
    count = 0

    def sub(m: re.Match[str]) -> str:
        nonlocal count
        repl = _match_case(m.group(0), new_form)
        if repl != m.group(0):
            count += 1
        return repl

    return pattern.sub(sub, text), count


def diff_spans(old: str, new: str) -> list[tuple[int, int, int, int]]:
    """Changed ranges ``(i1, i2, j1, j2)`` (difflib opcodes other than ``equal``)."""
    sm = difflib.SequenceMatcher(None, old, new, autojunk=False)
    return [(i1, i2, j1, j2) for tag, i1, i2, j1, j2 in sm.get_opcodes() if tag != "equal"]


def char_edit_distance(a: str, b: str) -> int:
    """Levenshtein distance over characters."""
    p = 0
    while p < len(a) and p < len(b) and a[p] == b[p]:
        p += 1
    a, b = a[p:], b[p:]
    s = 0
    while s < len(a) and s < len(b) and a[-1 - s] == b[-1 - s]:
        s += 1
    if s:
        a, b = a[:-s], b[:-s]
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
        prev = cur
    return prev[-1]


def _is_delim(text: str, i: int) -> bool:
    ch = text[i]
    if ch not in _CLAUSE_DELIMS:
        return False
    # a "," or "." between digits is a decimal separator / thousands group
    return not (ch in ",." and 0 < i < len(text) - 1 and text[i - 1].isdigit() and text[i + 1].isdigit())


def clause_bounds(text: str, start: int, end: int) -> tuple[int, int]:
    """Clause around ``[start, end)``: from after the previous delimiter
    (``, ; : . ? ! ( )`` or newline) to the next one, or the field ends."""
    s = start
    while s > 0 and not _is_delim(text, s - 1):
        s -= 1
    e = max(end, start)
    while e < len(text) and not _is_delim(text, e):
        e += 1
    return s, e


def _expand(text: str, s: int, e: int, n: int) -> tuple[int, int]:
    """Widen ``[s, e)`` by ``n`` whitespace-separated tokens on each side."""
    for _ in range(n):
        while s > 0 and text[s - 1].isspace():
            s -= 1
        while s > 0 and not text[s - 1].isspace():
            s -= 1
    for _ in range(n):
        while e < len(text) and text[e].isspace():
            e += 1
        while e < len(text) and not text[e].isspace():
            e += 1
    return s, e


def within_window(old: str, new: str, anchors: list[tuple[int, int]], extra_tokens: int = 3) -> bool:
    """True if every changed region of ``old`` lies inside the clause of some
    anchor (a span in ``old``) widened by ``extra_tokens`` tokens."""
    changes = diff_spans(old, new)
    if not changes:
        return True
    windows = [_expand(old, *clause_bounds(old, a, b), extra_tokens) for a, b in anchors]
    for i1, i2, _, _ in changes:
        if not any(ws <= i1 and i2 <= we for ws, we in windows):
            return False
    return True
