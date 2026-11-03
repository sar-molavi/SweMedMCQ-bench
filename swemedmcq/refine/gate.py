"""Deterministic acceptance gate for a candidate field edit (loop step 6)."""

from __future__ import annotations

import difflib
import re
import statistics
from collections.abc import Iterable
from typing import Any

from pydantic import BaseModel

from swemedmcq.refine.leakage import leakage_candidates
from swemedmcq.refine.numbers import missing_numbers
from swemedmcq.refine.spans import apply_ops, char_edit_distance, find_all_quotes, norm_for_match, within_window
from swemedmcq.refine.types import CandidateKind, FieldKey, SpanOp

PARALLEL_RANGE = (0.33, 3.0)

_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)
# option labels, answer-list labels, field markers and markup the model must not inject
_LABEL_RE = re.compile(
    r"\([A-J]\)"
    r"|(?m:^\s*[A-J][.):]\s)"
    r"|Svarsalternativ\s*:|Answer Choices\s*:"
    r"|\b(?:option|alternativ|question|fråga)\s*[A-J]?\s*:"
    r"|```|</?[a-z]+>|\{\s*\""
    , re.IGNORECASE,
)


class GateResult(BaseModel):
    ok: bool
    reasons: list[str]


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text)


def _is_abbreviation(tok: str) -> bool:
    return tok.isupper()


def _ratio(text: str, median: float) -> float:
    return len(text) / median


def gate(
    *,
    kind: CandidateKind,
    field: FieldKey,
    current: str,
    candidate: str,
    en_text: str,
    other_options: dict[FieldKey, str],
    anchors: list[tuple[int, int]],
    verified_sv_spans: list[str],
    lexicon: Any,
    protected: Iterable[str] | Any = frozenset(),
) -> GateResult:
    """Check a candidate against the current field text; any reason => not ok.

    ``lexicon`` is duck-typed: ``is_swedish(tok)``, ``is_english(tok)`` and optional ``whitelist``
    (set) and ``forms(term)`` attributes.  ``protected`` (the `keep` list, or a
    ``swemedmcq.refine.loop.routing.ProtectedTerms``): a candidate that removes an occurrence of a protected
    term gets ``removed_protected_term:<term>``.  ``lost_verified_term`` compares inflected forms
    (``Lexicon.forms``), so ``-test`` -> ``-testet`` keeps the term."""
    reasons: list[str] = []
    whitelist: set[str] = set(getattr(lexicon, "whitelist", None) or ())

    # noop
    if " ".join(candidate.split()) == " ".join(current.split()):
        reasons.append("noop")

    # structure: empty, newline or label injection
    if not candidate.strip() or candidate.count("\n") > current.count("\n") or \
            len(_LABEL_RE.findall(candidate)) > len(_LABEL_RE.findall(current)):
        reasons.append("structure")

    # numbers
    if len(missing_numbers(en_text, candidate, field)) > len(missing_numbers(en_text, current, field)):
        reasons.append("numbers")

    # new non-Swedish tokens
    cur_tokens = {t.casefold() for t in _tokens(current)}
    en_tokens = {t.casefold() for t in _tokens(en_text)}
    seen: set[str] = set()
    for tok in _tokens(candidate):
        key = tok.casefold()
        if key in cur_tokens or key in seen:
            continue
        seen.add(key)
        if any(c.isdigit() for c in tok) or key in en_tokens or _is_abbreviation(tok):
            continue
        if tok in whitelist or tok.lower() in whitelist or key in whitelist:
            continue
        if not lexicon.is_swedish(tok):
            reasons.append(f"non_swedish_token:{tok}")

    # new leakage (English-source tokens are NOT exempt here: copying an English
    # word from the source into the Swedish text is exactly the drift to block)
    def leaks(text: str) -> list[str]:
        return [t for t, _, _ in leakage_candidates(text, field, is_swedish=lexicon.is_swedish,
                                                     is_english=lexicon.is_english, whitelist=whitelist)]

    old_leaks = {t.casefold() for t in leaks(current)}
    for tok in dict.fromkeys(leaks(candidate)):
        if tok.casefold() not in old_leaks:
            reasons.append(f"new_leakage:{tok}")

    # verified term-base renderings must survive (any inflected form, also as a compound's first part)
    n_cur, n_cand = norm_for_match(current), norm_for_match(candidate)
    for span in dict.fromkeys(verified_sv_spans):
        ns = norm_for_match(span)
        if not ns:
            continue
        rx = _forms_regex(ns, lexicon)
        if rx.search(n_cur) and not rx.search(n_cand):
            reasons.append(f"lost_verified_term:{span}")

    # protected (`keep`) terms must not be removed
    for term in _removed_protected(protected, current, candidate):
        reasons.append(f"removed_protected_term:{term}")

    # respelling of a valid Swedish word (LLM edits only): these were mostly wrong
    # "corrections" of established spellings (ocklusal -> okklusal, gendopning -> gendoping)
    if kind in ("llm_edit", "retranslate"):
        for old, new in respellings(current, candidate, en_text, lexicon):
            reasons.append(f"respelled_swedish_word:{old}->{new}")

    # edit window (LLM span edits only)
    if kind == "llm_edit" and not within_window(current, candidate, anchors):
        reasons.append("outside_window")

    # options stay parallel
    if field.startswith("option:") and other_options:
        lengths = [len(v) for k, v in other_options.items() if k != field]
        median = statistics.median(lengths) if lengths else 0
        if median > 0:
            lo, hi = PARALLEL_RANGE
            before_in = lo <= _ratio(current, median) <= hi
            after_in = lo <= _ratio(candidate, median) <= hi
            if before_in and not after_in:
                reasons.append("options_not_parallel")

    return GateResult(ok=not reasons, reasons=reasons)


INFLECTION_TAIL = 3  # a change confined to the last characters of both tokens is an inflection


def respellings(current: str, candidate: str, en_text: str, lexicon: Any) -> list[tuple[str, str]]:
    """One-to-one token replacements that only respell a valid Swedish word.

    A pair (old, new) is reported when ``old`` is Swedish and not English, does not occur in the
    English source (an English leftover may be fixed), the two differ by at most 2 characters, the
    difference is not confined to the ending (``förskjuten`` -> ``förskjutet`` is grammar), and
    ``new`` is not better attested than ``old`` (a listed word beats an only-analysable one; see
    ``_attestation``).  Case-only changes are not respellings."""
    a, b = _tokens(current), _tokens(candidate)
    en_tokens = {t.casefold() for t in _tokens(en_text)}
    out: list[tuple[str, str]] = []
    sm = difflib.SequenceMatcher(a=[t.casefold() for t in a], b=[t.casefold() for t in b], autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag != "replace" or i2 - i1 != j2 - j1:
            continue
        for old, new in zip(a[i1:i2], b[j1:j2]):
            o, n = old.casefold(), new.casefold()
            if o == n or len(o) < 4 or any(c.isdigit() for c in o + n) or o in en_tokens:
                continue
            if char_edit_distance(o, n) > 2:
                continue
            k = 0
            while k < min(len(o), len(n)) and o[k] == n[k]:
                k += 1
            if k >= len(o) - INFLECTION_TAIL and k >= len(n) - INFLECTION_TAIL:
                continue  # ending only: inflection or agreement
            if lexicon.is_swedish(old) and not lexicon.is_english(old) \
                    and _attestation(new, lexicon) <= _attestation(old, lexicon):
                out.append((old, new))
    return out


def _attestation(tok: str, lexicon: Any) -> int:
    """2 = listed word form, 1 = analysable as Swedish (compound / derivation), 0 = not Swedish."""
    fn = getattr(lexicon, "attested", None)
    if callable(fn) and fn(tok):
        return 2
    return 1 if lexicon.is_swedish(tok) else 0


def _forms_regex(ns: str, lexicon: Any) -> re.Pattern[str]:
    """Word-initial match of any inflected form of the (normalised) term."""
    variants = {ns}
    forms = getattr(lexicon, "forms", None)
    if callable(forms):
        try:
            variants |= {norm_for_match(f) for f in forms(ns)}
        except (AttributeError, KeyError, TypeError, ValueError):  # duck-typed lexicons without inflection data
            variants = {ns}
    alts = "|".join(re.escape(v) for v in sorted((v for v in variants if v), key=len, reverse=True))
    return re.compile(r"(?<!\w)(?:" + alts + r")")


def _removed_protected(protected: Any, current: str, candidate: str) -> list[str]:
    if not protected:
        return []
    if hasattr(protected, "removed"):
        return list(protected.removed(current, candidate))
    from swemedmcq.refine.loop.routing import ProtectedTerms  # lazy: avoids an import cycle

    return list(ProtectedTerms(protected).removed(current, candidate))


def gate_ops(
    *,
    kind: CandidateKind,
    field: FieldKey,
    current: str,
    ops: list[SpanOp],
    en_text: str,
    other_options: dict[FieldKey, str],
    verified_sv_spans: list[str],
    lexicon: Any,
    protected: Iterable[str] | Any = frozenset(),
    anchors: list[tuple[int, int]] | None = None,
) -> tuple[list[SpanOp], str | None, GateResult, list[tuple[SpanOp, list[str]]]]:
    """Gate every operation on its own, drop the failing ones, then gate the survivors together.

    Each op is applied alone to ``current`` (``apply_failed`` if it cannot be located) and gated with
    its own occurrences (plus ``anchors``) as the edit window.  Returns ``(kept_ops, new_text,
    final_result, rejected)``: ``new_text`` is ``current`` with the kept ops applied (None if none
    survive or the combination cannot be applied) and ``final_result`` the gate of that combined
    text; ``rejected`` lists each dropped op with its reasons."""
    if not isinstance(protected, (set, frozenset, list, tuple)) or not protected:
        prot = protected
    else:
        from swemedmcq.refine.loop.routing import ProtectedTerms  # lazy: avoids an import cycle

        prot = ProtectedTerms(protected)
    base_anchors = list(anchors or [])

    def op_anchors(op_list: list[SpanOp]) -> list[tuple[int, int]]:
        out = list(base_anchors)
        for op in op_list:
            out.extend(find_all_quotes(current, op.old))
        return out or [(0, len(current))]

    def run(op_list: list[SpanOp]) -> tuple[str | None, GateResult]:
        text = apply_ops(current, op_list)
        if text is None:
            return None, GateResult(ok=False, reasons=["apply_failed"])
        res = gate(kind=kind, field=field, current=current, candidate=text, en_text=en_text,
                   other_options=other_options, anchors=op_anchors(op_list), verified_sv_spans=verified_sv_spans,
                   lexicon=lexicon, protected=prot)
        return text, res

    kept: list[SpanOp] = []
    rejected: list[tuple[SpanOp, list[str]]] = []
    seen: set[tuple[str, str]] = set()
    for op in ops:
        if (op.old, op.new) in seen:
            continue  # duplicate op: merged
        seen.add((op.old, op.new))
        _, res = run([op])
        if res.ok:
            kept.append(op)
        else:
            rejected.append((op, list(res.reasons)))
    if not kept:
        reasons = sorted({r for _, rs in rejected for r in rs}) or ["noop"]
        return [], None, GateResult(ok=False, reasons=reasons), rejected
    text, res = run(kept)
    return kept, text if res.ok else None, res, rejected
