"""io: atomic writes, crash-safe JSONL append/read, stable hashing."""

from __future__ import annotations

import json
import os

import pytest

from swemedmcq.refine.io import (
    atomic_append_jsonl,
    atomic_write_json,
    atomic_write_text,
    fields_from_row,
    read_jsonl,
    stable_hash,
)
from swemedmcq.refine.types import Change


def test_atomic_write_text_creates_dirs_and_leaves_no_temp(tmp_path):
    p = tmp_path / "a" / "b" / "x.txt"
    atomic_write_text(p, "hej\nvärld")
    assert p.read_text(encoding="utf-8") == "hej\nvärld"
    atomic_write_text(p, "ny")
    assert p.read_text(encoding="utf-8") == "ny"
    assert os.listdir(p.parent) == ["x.txt"]


def test_atomic_write_json_roundtrip_with_models(tmp_path):
    p = tmp_path / "x.json"
    ch = Change(field="question", rule="nbsp", start=1, before="\u00a0", after=" ")
    atomic_write_json(p, {"c": ch, "s": {2, 1}, "å": "ö"})
    data = json.loads(p.read_text(encoding="utf-8"))
    assert data["c"]["rule"] == "nbsp" and data["s"] == [1, 2] and data["å"] == "ö"


def test_append_and_read_jsonl(tmp_path):
    p = tmp_path / "s" / "log.jsonl"
    assert read_jsonl(p) == []
    for i in range(3):
        atomic_append_jsonl(p, {"i": i, "t": "å"})
    assert [r["i"] for r in read_jsonl(p)] == [0, 1, 2]


def test_torn_tail_is_ignored_and_repaired(tmp_path):
    p = tmp_path / "log.jsonl"
    atomic_append_jsonl(p, {"i": 0})
    with open(p, "ab") as fh:  # simulate a crash in the middle of a write
        fh.write(b'{"i": 1, "x": "abc')
    assert read_jsonl(p) == [{"i": 0}]
    with pytest.raises(ValueError):
        read_jsonl(p, tolerate_torn_tail=False)
    atomic_append_jsonl(p, {"i": 2})
    assert read_jsonl(p) == [{"i": 0}, {"i": 2}]
    assert p.read_bytes().endswith(b'{"i": 2}\n')


def test_torn_only_line(tmp_path):
    p = tmp_path / "log.jsonl"
    p.write_bytes(b'{"i"')
    assert read_jsonl(p) == []
    atomic_append_jsonl(p, {"i": 1})
    assert read_jsonl(p) == [{"i": 1}]


def test_unterminated_but_complete_last_line_is_kept(tmp_path):
    p = tmp_path / "log.jsonl"
    p.write_text('{"i": 0}\n{"i": 1}', encoding="utf-8")
    assert read_jsonl(p) == [{"i": 0}, {"i": 1}]


def test_corrupt_middle_line_raises(tmp_path):
    p = tmp_path / "log.jsonl"
    p.write_text('{"i": 0}\n{bad\n{"i": 2}\n', encoding="utf-8")
    with pytest.raises(ValueError):
        read_jsonl(p)


def test_stable_hash_is_canonical():
    h = stable_hash({"b": 1, "a": [1, 2]}, "x")
    assert h == stable_hash({"a": [1, 2], "b": 1}, "x")
    assert h != stable_hash({"a": [1, 2], "b": 1}, "y")
    assert len(h) == 64 and int(h, 16) >= 0
    assert stable_hash("å") == stable_hash("å")


def test_fields_from_row():
    f = fields_from_row("Q?", {"A": "a", "B": "b"})
    assert f == {"question": "Q?", "option:A": "a", "option:B": "b"}
    assert list(f) == ["question", "option:A", "option:B"]
