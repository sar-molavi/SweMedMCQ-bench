"""Layer L0: Unicode hygiene of a Swedish draft field.

Every edit is logged as a :class:`~swemedmcq.refine.types.Change` with sequential
semantics (``start`` is the offset in the text just before that change), so
``swemedmcq.refine.locale.revert`` can undo the whole list.
"""

from __future__ import annotations

import unicodedata

from swemedmcq.refine.types import Change, FieldKey

# character -> (replacement, rule)
_CHAR_RULES: dict[str, tuple[str, str]] = {
    "\u2011": ("-", "nbhyphen"),
    "\u2010": ("-", "hyphen"),
    "\u00a0": (" ", "nbsp"),
    "\u202f": (" ", "nnbsp"),
    "\u2009": (" ", "thinsp"),
    "\u200a": (" ", "thinsp"),
    # invisible format characters that break token matching
    "\u200b": ("", "invisible"),
    "\ufeff": ("", "invisible"),
    "\u00ad": ("", "invisible"),
    "\x00": ("", "nul"),
}
_KEEP_CONTROL = {"\n", "\t"}
_BENIGN_CONTROL = {"\n", "\t", "\r"}


def _is_control(ch: str) -> bool:
    return unicodedata.category(ch) == "Cc"


def hygiene(text: str, field: FieldKey) -> tuple[str, list[Change]]:
    """NFC; U+2011/U+2010 -> "-"; U+202F/U+00A0/U+2009(/U+200A) -> " ";
    drop zero-width/soft-hyphen/BOM; strip control chars except ``\\n``/``\\t``
    (a NUL is logged with ``rule="nul"``)."""
    changes: list[Change] = []
    nfc = unicodedata.normalize("NFC", text)
    if nfc != text:
        changes.append(Change(field=field, rule="nfc", start=0, before=text, after=nfc))
    out: list[str] = []
    pos = 0  # length of the output so far == offset in the text before the next change
    for ch in nfc:
        if ch in _CHAR_RULES:
            repl, rule = _CHAR_RULES[ch]
        elif _is_control(ch) and ch not in _KEEP_CONTROL:
            repl, rule = "", "control_char"
        else:
            out.append(ch)
            pos += 1
            continue
        changes.append(Change(field=field, rule=rule, start=pos, before=ch, after=repl))
        out.append(repl)
        pos += len(repl)
    return "".join(out), changes


def is_corrupt(text: str) -> bool:
    """True if the RAW draft contains a NUL or another control character
    (``\\n``, ``\\t`` and ``\\r`` are benign)."""
    return any(_is_control(ch) and ch not in _BENIGN_CONTROL for ch in text)
