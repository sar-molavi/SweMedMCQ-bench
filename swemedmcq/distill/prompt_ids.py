"""Read the released train/validation prompt ids (``data/train/split/*_prompt_ids.txt.gz``).

One prompt id per line, plain text or gzip. JSONL files (one row per response, with a
``prompt_id`` field) are also accepted, for split files written by the pipeline.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path


def read_prompt_ids(path: Path) -> set[str]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Prompt id file does not exist: {path}")
    opener = gzip.open if path.suffix == ".gz" else open
    is_jsonl = path.name.removesuffix(".gz").endswith(".jsonl")
    ids: set[str] = set()
    with opener(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            prompt_id = str(json.loads(line).get("prompt_id") or "") if is_jsonl else line.split("\t")[-1].strip()
            if not prompt_id:
                raise ValueError(f"{path}:{line_number}: no prompt_id")
            ids.add(prompt_id)
    if not ids:
        raise ValueError(f"Prompt id file contains no ids: {path}")
    return ids
