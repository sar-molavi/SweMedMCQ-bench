"""Check the MedXpertQA tail functions on every MedXpertQA item of the drafts.

    python -m swemedmcq.refine.check_medxpertqa_tails [--input PATH] [--examples N]
"""

from __future__ import annotations

import argparse
import re
from collections import Counter

from swemedmcq.paths import path as repo_path
from swemedmcq.refine.io import read_jsonl
from swemedmcq.refine.medxpertqa import (
    build_sv_tail,
    split_en_tail,
    strip_sv_tail,
    tail_format,
    tail_matches_options,
)

DEFAULT_INPUT = str(repo_path("draft"))


def _label(tail: str) -> str:
    m = re.match(r"\s*([^\s(][^:(]*?)\s*:", tail)
    return m.group(1) if m else "(bare list)"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default=DEFAULT_INPUT)
    ap.add_argument("--examples", type=int, default=3)
    args = ap.parse_args()

    rows = [r for r in read_jsonl(args.input) if r["prompt_id"].startswith("medxpertqa")]
    n = len(rows)
    no_tail: list[str] = []
    exceptions: list[str] = []
    formats: Counter[str] = Counter()
    sv_labels: Counter[str] = Counter()
    sv_tail_ids: list[str] = []
    sv_body_with_list: list[str] = []
    examples: list[tuple[str, str]] = []
    roundtrip_bad: list[str] = []
    sv_empty: list[str] = []
    sv_mismatch: list[str] = []

    for r in rows:
        pid, q = r["prompt_id"], r["question"]
        body, tail = split_en_tail(q)
        if tail is None:
            no_tail.append(pid)
            continue
        if body + tail != q:
            roundtrip_bad.append(pid)
        if not tail_matches_options(tail, r["options"]):
            exceptions.append(pid)
        fmt = tail_format(tail)
        formats[fmt] += 1

        sv_body, sv_tail = strip_sv_tail(r["sv_question"])
        if sv_body + (sv_tail or "") != r["sv_question"]:
            roundtrip_bad.append(pid + " (sv)")
        if sv_tail is not None:
            sv_tail_ids.append(pid)
            sv_labels[_label(sv_tail)] += 1
            entries = sv_tail.split(":", 1)[1] if ":" in sv_tail.split("(A)", 1)[0] else sv_tail
            if not entries.strip():
                sv_empty.append(pid)
            elif " ".join(entries.split()) != " ".join(
                    " ".join(f"({k}) {v}" for k, v in r["sv_options"].items()).split()):
                sv_mismatch.append(pid)
        if "(A)" in sv_body and "(B)" in sv_body:
            sv_body_with_list.append(pid)
        rebuilt = sv_body.rstrip() + build_sv_tail(r["sv_options"], fmt)
        if len(examples) < args.examples and (sv_tail is None or len(examples) == 0):
            examples.append((pid, rebuilt))

    print(f"MedXpertQA items:                      {n}")
    print(f"English tails found:                   {n - len(no_tail)}  (missing: {len(no_tail)})")
    print(f"  tail matches options:                {n - len(no_tail) - len(exceptions)}")
    print(f"  exceptions (tail != options):        {len(exceptions)}")
    for pid in exceptions:
        print(f"    - {pid}")
    print(f"  tail formats:                        {dict(formats)}")
    print(f"Swedish drafts that had a tail:        {len(sv_tail_ids)}")
    for lab, c in sv_labels.most_common():
        print(f"    label {lab!r}: {c}")
    print(f"  label only, no entries:              {len(sv_empty)}")
    print(f"  entries differ from sv_options:      {len(sv_mismatch)} {sv_mismatch[:20]}")
    print(f"Swedish bodies still holding (A)/(B):  {len(sv_body_with_list)} {sv_body_with_list[:10]}")
    print(f"body + tail != question (must be 0):   {len(roundtrip_bad)} {roundtrip_bad[:10]}")
    if no_tail:
        print(f"items without an English tail: {no_tail[:20]}")
    print("\nExamples of rebuilt Swedish questions (last 400 chars):")
    for pid, text in examples:
        print(f"--- {pid}\n...{text[-400:]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
