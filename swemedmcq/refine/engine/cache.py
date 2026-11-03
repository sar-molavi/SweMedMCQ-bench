"""Content-addressed, crash-safe cache of LLM request results.

One JSON file per *attempt* (one concrete vLLM request):
``<cache_dir>/<model>/<key[:2]>/<key>.json``.  The key is the sha256 of
everything that determines the model output (model + snapshot revision, vLLM
version, prompt token ids, sampling parameters, JSON schema, max_tokens), so a
restarted job replays completed requests without touching the GPU.

Writes are atomic (temp file + fsync + ``os.replace``) and happen *before* the
result is returned to the caller.

The helpers ``_stable_hash`` and ``_atomic_write_json`` mirror
``swemedmcq.refine.io.stable_hash`` / ``swemedmcq.refine.io.atomic_write_json``; they are private
copies so the engine has no dependency on the deterministic core.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


def _stable_hash(*parts: Any) -> str:
    """sha256 hex digest of the canonical JSON of ``parts``."""
    blob = json.dumps(list(parts), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _atomic_write_json(path: Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(obj, handle, ensure_ascii=False, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise
    # Make the rename durable too (best effort; not all filesystems allow it).
    try:
        dir_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except OSError:
        pass


def request_key(
    *,
    model_id: str,
    vllm_version: str,
    prompt_token_ids: list[int],
    sampling: dict[str, Any],
    schema_json: dict[str, Any],
    max_tokens: int,
) -> str:
    """Content address of one concrete request (see module docstring)."""
    return _stable_hash(
        "refine-llm-request-v1",
        model_id,
        vllm_version,
        prompt_token_ids,
        sampling,
        schema_json,
        max_tokens,
    )


class ResultCache:
    """Directory of cached attempt results keyed by :func:`request_key`."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    def path(self, model: str, key: str) -> Path:
        return self.root / model / key[:2] / f"{key}.json"

    def get(self, model: str, key: str) -> dict[str, Any] | None:
        """Return the stored record, or None when absent or unreadable."""
        path = self.path(model, key)
        try:
            with path.open("r", encoding="utf-8") as handle:
                record = json.load(handle)
        except FileNotFoundError:
            return None
        except (OSError, json.JSONDecodeError):
            # A torn file cannot exist with os.replace, but never let a bad
            # cache entry crash a run: treat it as a miss and overwrite it.
            return None
        return record if isinstance(record, dict) else None

    def put(self, model: str, key: str, record: dict[str, Any]) -> Path:
        path = self.path(model, key)
        _atomic_write_json(path, record)
        return path
