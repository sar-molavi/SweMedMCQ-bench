"""Repository root, configuration files and every data/output location.

All locations are listed in ``configs/paths.yaml``. Relative paths are resolved
against the repository root. Overrides, in this order:

* ``SWEMEDMCQ_ROOT``: another repository root (e.g. a copy of the data on a cluster);
* ``SWEMEDMCQ_<KEY>``: one location, e.g. ``SWEMEDMCQ_RESULTS_DIR=/scratch/eval``;
* ``SWEMEDMCQ_CONFIG_DIR``: another configuration folder.

Hugging Face models are read from the usual cache (``HF_HUB_CACHE`` / ``HF_HOME``);
nothing here changes it.

Shell scripts read locations with ``python -m swemedmcq.paths <key> [<key> ...]``.
"""

from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

PACKAGE_DIR = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("SWEMEDMCQ_ROOT") or PACKAGE_DIR.parent).resolve()
CONFIG_DIR = Path(os.environ.get("SWEMEDMCQ_CONFIG_DIR") or ROOT / "configs")


@lru_cache(maxsize=None)
def load_config(name: str) -> dict[str, Any]:
    """``configs/<name>.yaml`` as a dict (``name`` may contain a subfolder, e.g. ``refine/default``)."""
    with (CONFIG_DIR / f"{name}.yaml").open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def resolve(value: str | os.PathLike) -> Path:
    """A path relative to the repository root (absolute paths are kept)."""
    value = Path(value)
    return value if value.is_absolute() else ROOT / value


def path(key: str) -> Path:
    """A location from ``configs/paths.yaml``, overridable with ``SWEMEDMCQ_<KEY>``."""
    env = os.environ.get(f"SWEMEDMCQ_{key.upper()}")
    if env:
        return resolve(env)
    paths = load_config("paths")
    if key not in paths:
        raise KeyError(f"No location {key!r} in {CONFIG_DIR / 'paths.yaml'}")
    return resolve(paths[key])


def main(argv: list[str] | None = None) -> None:
    keys = sys.argv[1:] if argv is None else argv
    if keys and keys[0] in ("-h", "--help"):
        print(__doc__)
        print("usage: python -m swemedmcq.paths [KEY ...]   (no key: list every location)")
        return
    if not keys:
        for key in load_config("paths"):
            print(f"{key}\t{path(key)}")
        return
    for key in keys:
        print(path(key))


if __name__ == "__main__":
    main()
