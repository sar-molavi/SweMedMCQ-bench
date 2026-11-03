"""Join the extracted GPT-5.4 translations with the English items into the draft file.

The draft (input of the refinement loop) has one row per item: ``prompt_id``,
``question``, ``options`` (English), ``sv_question``, ``sv_options`` (Swedish, keyed
by the original option labels). Rows follow the order of the English item file.
Several extraction files can be given (e.g. the main batch and a retry); a later
file wins for a prompt id that occurs twice.

    python -m swemedmcq.translate.test_gpt54.to_draft \\
        --english data/test/benchmark.jsonl \\
        --translations work/translate/test/main_results_valid.jsonl work/translate/test/retry_results_valid.jsonl \\
        -o data/test/layers/layer1_draft.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--english", type=Path, required=True,
                        help="JSONL with prompt_id, question and options (English), e.g. the benchmark file.")
    parser.add_argument("--translations", type=Path, nargs="+", required=True,
                        help="Valid extraction rows of swemedmcq.translate.test_gpt54.extract (task translation).")
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--allow-missing", action="store_true", help="Skip English items without a translation.")
    args = parser.parse_args()

    translated: dict[str, dict[str, Any]] = {}
    for path in args.translations:
        for row in read_jsonl(path):
            translated[row["prompt_id"]] = row
    rows, missing = [], []
    for item in read_jsonl(args.english):
        ext = translated.get(item["prompt_id"])
        if ext is None:
            missing.append(item["prompt_id"])
            continue
        rows.append({
            "prompt_id": item["prompt_id"],
            "question": item["question"],
            "options": item["options"],
            "sv_question": ext["question"],
            "sv_options": {choice["label"]: choice["text"] for choice in ext["choices"]},
        })
    if missing and not args.allow_missing:
        raise SystemExit(f"{len(missing)} items have no translation, e.g. {missing[:5]} (--allow-missing to skip)")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} rows to {args.output} ({len(missing)} without a translation)")


if __name__ == "__main__":
    main()
