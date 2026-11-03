"""Crash-safe file I/O helpers and small record utilities.

Every write goes through a temp file in the target directory, is flushed and
``fsync``-ed, and is then moved into place with ``os.replace``.  The JSONL
append is crash-safe in the sense that a torn (partially written) last line is
ignored by :func:`read_jsonl` and repaired by the next append.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from swemedmcq.paths import path as repo_path
from swemedmcq.paths import resolve
from swemedmcq.refine.types import FieldKey, option_key

PathLike = str | os.PathLike[str]


def resources_dir(cfg: dict | None = None) -> Path:
    """Term base, lexicons, whitelist and style guide: $REFINE_RESOURCES, else the config's
    ``paths.resources``, else ``resources`` of configs/paths.yaml (relative to the repository root)."""
    value = os.environ.get("REFINE_RESOURCES") or ((cfg or {}).get("paths") or {}).get("resources")
    return resolve(value) if value else repo_path("resources")


def _fsync_dir(directory: Path) -> None:
    """Best-effort fsync of a directory so that a rename is durable."""
    try:
        fd = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _json_default(obj: Any) -> Any:
    """JSON fallback for pydantic models, paths, sets and bytes."""
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    if isinstance(obj, os.PathLike):
        return os.fspath(obj)
    if isinstance(obj, (set, frozenset)):
        return sorted(obj, key=lambda x: json.dumps(x, sort_keys=True, default=str))
    if isinstance(obj, (bytes, bytearray)):
        return obj.hex()
    raise TypeError(f"not JSON serialisable: {type(obj).__name__}")


def atomic_write_text(path: PathLike, text: str) -> None:
    """Write ``text`` (UTF-8) to ``path`` atomically (temp + fsync + replace)."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, target)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    _fsync_dir(target.parent)


def atomic_write_json(path: PathLike, obj: Any) -> None:
    """Write ``obj`` as pretty-printed UTF-8 JSON, atomically."""
    text = json.dumps(obj, ensure_ascii=False, indent=2, default=_json_default) + "\n"
    atomic_write_text(path, text)


def atomic_append_jsonl(path: PathLike, obj: Any) -> None:
    """Append one JSON line to ``path`` and fsync it.

    If the file ends with a torn line (no trailing newline, left by a crash),
    that fragment is truncated first so the new record starts on a clean line.
    The record is written with a single ``write`` call on an ``O_APPEND`` fd.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    line = (json.dumps(obj, ensure_ascii=False, default=_json_default) + "\n").encode("utf-8")
    fd = os.open(target, os.O_RDWR | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        size = os.fstat(fd).st_size
        if size > 0:
            os.lseek(fd, size - 1, os.SEEK_SET)
            if os.read(fd, 1) != b"\n":
                _truncate_torn_tail(fd, size)
        view = memoryview(line)
        while view:
            written = os.write(fd, view)
            view = view[written:]
        os.fsync(fd)
    finally:
        os.close(fd)


def _truncate_torn_tail(fd: int, size: int) -> None:
    """Cut the file after its last newline (drop a partially written record)."""
    chunk = 1 << 16
    pos = size
    while pos > 0:
        start = max(0, pos - chunk)
        os.lseek(fd, start, os.SEEK_SET)
        data = os.read(fd, pos - start)
        k = data.rfind(b"\n")
        if k >= 0:
            os.ftruncate(fd, start + k + 1)
            return
        pos = start
    os.ftruncate(fd, 0)


def read_jsonl(path: PathLike, *, tolerate_torn_tail: bool = True) -> list[dict]:
    """Read a JSONL file; a missing file reads as ``[]``.

    A last line without a trailing newline that does not parse (a torn write)
    is ignored when ``tolerate_torn_tail``; any other bad line raises ``ValueError``.
    """
    target = Path(path)
    if not target.exists():
        return []
    raw = target.read_bytes().decode("utf-8")
    lines = raw.split("\n")
    complete_tail = raw.endswith("\n")
    if complete_tail:
        lines = lines[:-1]
    out: list[dict] = []
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        is_last_unterminated = i == len(lines) - 1 and not complete_tail
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError as exc:
            if is_last_unterminated and tolerate_torn_tail:
                break
            raise ValueError(f"{target}: bad JSON on line {i + 1}: {exc}") from exc
    return out


def stable_hash(*parts: Any) -> str:
    """sha256 hex digest of the canonical JSON encoding of ``parts``."""
    payload = json.dumps(
        list(parts), sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=_json_default
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def fields_from_row(question: str, options: dict[str, str]) -> dict[FieldKey, str]:
    """``{"question": q, "option:A": ..., ...}`` in the given option order."""
    fields: dict[FieldKey, str] = {"question": question}
    for label, text in options.items():
        fields[option_key(label)] = text
    return fields
