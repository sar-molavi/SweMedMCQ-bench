"""MedXpertQA answer-list tails.

MedXpertQA questions embed their options at the end of the English question::

    <body>\\nAnswer Choices: (A) ... (B) ... (J) ...

The loop works on the body only; the Swedish tail is stripped from the draft
and rebuilt at assembly from the final Swedish options, in the separator layout
of the English tail (captured by :func:`tail_format`).

Convention: ``body + tail == question`` for both split functions; the tail
starts with the whitespace that precedes the label.
"""

from __future__ import annotations

import json
import re
from collections import Counter

EN_LABEL = "Answer Choices:"
SV_LABEL = "Svarsalternativ:"

_EN_LABEL_RE = re.compile(r"\s*Answer Choices:")
_SV_LABEL_RE = re.compile(r"\s*(?:Svarsalternativen|Svarsalternativ|Answer [Cc]hoices)\s*:")
_FMT_HEAD_RE = re.compile(r"(\s*)(?:Answer Choices|Svarsalternativ)\s*:(\s*)(?=\(A\))")
_A_RE = re.compile(r"\(A\)\s")

_DEFAULT_FMT = {"pre": "\n", "after": " ", "sep": " "}


def _labels_after(text: str, pos: int) -> list[tuple[str, int]]:
    """Positions of ``(B)``, ``(C)``, ... found in order after ``pos``."""
    found: list[tuple[str, int]] = []
    for code in range(ord("B"), ord("Z") + 1):
        lab = f"({chr(code)})"
        k = text.find(lab, pos)
        if k < 0:
            break
        found.append((chr(code), k))
        pos = k + len(lab)
    return found


def split_en_tail(question: str) -> tuple[str, str | None]:
    """Split ``question`` into ``(body, tail)``; the tail includes "Answer Choices:"
    and its leading whitespace.  Returns ``(question, None)`` if there is no tail."""
    matches = list(_EN_LABEL_RE.finditer(question))
    if not matches:
        return question, None
    chosen = matches[-1]
    for m in reversed(matches):
        if re.match(r"\s*\(A\)", question[m.end():]):
            chosen = m
            break
    return question[: chosen.start()], question[chosen.start():]


def tail_format(tail: str) -> str:
    """Opaque format id (compact JSON) describing the separators of a tail:
    whitespace before the label, after the colon, and between entries."""
    fmt = dict(_DEFAULT_FMT)
    m = _FMT_HEAD_RE.match(tail)
    if not m:
        return json.dumps(fmt, separators=(",", ":"))
    fmt["pre"], fmt["after"] = m.group(1), m.group(2)
    runs: list[str] = []
    for _, k in _labels_after(tail, m.end() + 3):
        j = k
        while j > 0 and tail[j - 1].isspace():
            j -= 1
        runs.append(tail[j:k])
    if runs:
        counts = Counter(runs)
        best = max(counts.values())
        fmt["sep"] = min((r for r, c in counts.items() if c == best), key=len)
    return json.dumps(fmt, separators=(",", ":"))


def _render(options: dict[str, str], fmt: str, label: str) -> str:
    spec = {**_DEFAULT_FMT, **json.loads(fmt)}
    entries = spec["sep"].join(f"({lab}) {text}" for lab, text in options.items())
    return f"{spec['pre']}{label}{spec['after']}{entries}"


def tail_matches_options(tail: str, options: dict[str, str]) -> bool:
    """True if the tail lists exactly ``options`` (exact, else whitespace-insensitive)."""
    rebuilt = _render(options, tail_format(tail), EN_LABEL)
    if rebuilt == tail:
        return True
    return " ".join(rebuilt.split()) == " ".join(tail.split())


def strip_sv_tail(question: str) -> tuple[str, str | None]:
    """Remove a Swedish answer-list tail ("Svarsalternativ:" / "Answer Choices:"
    label, or a bare "(A) ... (B) ..." list at the end).  Returns ``(body, tail)``."""
    for m in reversed(list(_SV_LABEL_RE.finditer(question))):
        rest = question[m.end():]
        if re.match(r"\s*\(A\)", rest):
            return question[: m.start()], question[m.start():]
        label_at_line_start = m.start() == 0 or "\n" in m.group(0)[: m.group(0).find(m.group(0).strip())]
        if not rest.strip() and label_at_line_start:
            return question[: m.start()], question[m.start():]
    # bare list: "(A) x (B) y ..." at the end, after a line break or a sentence end
    starts = list(_A_RE.finditer(question))
    if starts:
        a = starts[-1].start()
        if _labels_after(question, a + 3):
            j = a
            while j > 0 and question[j - 1].isspace():
                j -= 1
            gap = question[j:a]
            prev = question[j - 1] if j > 0 else ""
            if j == 0 or "\n" in gap or (gap and prev in "?:."):
                return question[:j], question[j:]
    return question, None


def build_sv_tail(sv_options: dict[str, str], fmt: str) -> str:
    """Swedish tail, e.g. ``"\\nSvarsalternativ: (A) ... (B) ..."``, in the layout ``fmt``."""
    return _render(sv_options, fmt, SV_LABEL)
