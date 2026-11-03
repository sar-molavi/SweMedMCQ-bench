"""Candidate English leakage in a Swedish field (S1 hint; never auto-confirmed)."""

from __future__ import annotations

import re
from collections.abc import Callable

from swemedmcq.refine.types import FieldKey

_TOKEN_RE = re.compile(r"[^\W\d_]+", re.UNICODE)


def _in_whitelist(tok: str, whitelist: set[str]) -> bool:
    return tok in whitelist or tok.lower() in whitelist or tok.casefold() in whitelist


def leakage_candidates(
    sv_text: str,
    field: FieldKey,
    *,
    is_swedish: Callable[[str], bool],
    is_english: Callable[[str], bool],
    whitelist: set[str],
    en_text: str | None = None,
) -> list[tuple[str, int, int]]:
    """Alphabetic tokens (length >= 3) of ``sv_text`` that look English:
    ``is_english(tok)`` and not ``is_swedish(tok)``, not whitelisted, not an
    ALL-CAPS abbreviation of length <= 6, and (when ``en_text`` is given) not
    occurring in the English source field.  Returns ``(token, start, end)``."""
    en_tokens = {t.casefold() for t in _TOKEN_RE.findall(en_text)} if en_text else set()
    out: list[tuple[str, int, int]] = []
    for m in _TOKEN_RE.finditer(sv_text):
        tok = m.group(0)
        if len(tok) < 3:
            continue
        if tok.isupper() and len(tok) <= 6:
            continue
        if _in_whitelist(tok, whitelist) or tok.casefold() in en_tokens:
            continue
        if is_english(tok) and not is_swedish(tok):
            out.append((tok, m.start(), m.end()))
    return out
