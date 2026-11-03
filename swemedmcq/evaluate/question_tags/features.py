"""Simple per-item features of the core test items (no model calls).

Text features come from the English question of the benchmark file: number of words
and options, clinical vignette, numbers, lab units, likely calculation (regular
expressions, deliberately simple; the lab-unit pattern also fires on blood pressure in
mm Hg). The subject comes from the source dataset files where a dataset has one
(skipped if they are not available), and the number of open benchmark configurations
answering each item correctly per language from the per-item results of
``swemedmcq.evaluate.collect`` (if present).

    python -m swemedmcq.evaluate.question_tags.features

Writes <results_dir>/question_tags/features.parquet.
"""

from __future__ import annotations

import json
import re

import pandas as pd

from swemedmcq import paths

INPUT = paths.path("benchmark")
SOURCE = paths.path("source_benchmark")
ITEMS = paths.path("results_dir") / "items.parquet"
OUT = paths.path("results_dir") / "question_tags" / "features.parquet"
DATASETS = ("medexqa", "mmlu-pro", "afrimedqa", "mmlu", "medqa", "medxpertqa-u", "medmcqa", "medxpertqa-r",
            "pubmedqa", "medbullets")

AGE = r"\b\d{1,3}(?:\.\d)?\s*-?\s*(?:years?|yrs?|y|months?|mo|weeks?|wk|days?)\s*-?\s*old\b|\baged?\s+\d{1,3}\b"
PERSON = r"(?:man|woman|male|female|boy|girl|gentleman|lady|patient|child|infant|neonate|newborn|baby|mother|primigravida|multigravida|g\d\s*p\d|nurse|student|athlete)"
VIGNETTE = re.compile(rf"(?:{AGE})[^.?!]{{0,60}}?\b{PERSON}|\b{PERSON}[^.?!]{{0,30}}?(?:{AGE})", re.I)
NUMBER = re.compile(r"(?<![A-Za-z])\d+(?:[.,]\d+)?")
LAB_UNIT = re.compile(
    r"\b(?:m|µ|μ|u|n|p)?(?:g|mol|eq|osm|mEq|IU|U)\s*/\s*(?:d?L|dl|ml|mL|kg|24\s*h|day)\b"
    r"|\bmm\s*Hg\b|\bmmHg\b|/\s*(?:mm\^?3|mm³|µL|μL|uL|hpf|HPF)\b|\bx\s*10\^?\s*\d+\s*/\s*L\b"
    r"|\bmOsm\b|\bfL\b|\bU/L\b|\bIU/L\b|\bcells/|\bpH\s*[=:]?\s*\d",
    re.I,
)
CALC_WORDS = re.compile(
    r"\bcalculat|\bcompute|\bestimate[ds]?\b|\bhow (?:many|much|long)\b|\bwhat (?:is|was|would be) the (?:\w+ ){0,3}"
    r"(?:value|dose|rate|ratio|risk|probability|percentage|number|clearance|concentration|volume|half-life|gradient|gap)\b"
    r"|\bsensitivity\b|\bspecificity\b|\bpredictive value\b|\bodds ratio\b|\brelative risk\b|\bnumber needed\b"
    r"|\bhalf-life\b|\bvolume of distribution\b|\bloading dose\b|\bmaintenance dose\b|\binfusion rate\b"
    r"|\bcreatinine clearance\b|\banion gap\b|\bA-a gradient\b|\bfiltration fraction\b|\bfree water\b"
    r"|\bincidence\b|\bprevalence\b|\battributable\b|\blikelihood ratio\b",
    re.I,
)
NUMERIC_OPTION = re.compile(r"^\s*[-+~≈]?\s*\d+(?:[.,]\d+)?\s*(?:%|[a-zA-Zµμ/^\d\s.·]{0,14})?$")


def stem(question: str) -> str:
    """Question text without the option list that MedXpertQA repeats inside it."""
    return question.split("Answer Choices:")[0].rstrip()


def raw_subjects() -> dict[str, str | None]:
    """Subject label from the raw dataset files, where the dataset has one."""
    out: dict[str, str | None] = {}
    for dataset in DATASETS:
        path = SOURCE / dataset / "test.jsonl"
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                r = json.loads(line)
                subject = None
                if dataset == "mmlu":
                    subject = r.get("subject")
                elif dataset == "mmlu-pro":
                    subject = str(r.get("src", "")).removeprefix("ori_mmlu-") or None
                elif dataset == "medqa":
                    subject = r.get("meta_info")
                elif dataset == "medexqa":
                    subject = str(r["realidx"]).split("_", 1)[1]
                out[f"{dataset}_{r['realidx']}"] = subject
    return out


def likely_calculation(question: str, options: dict, n_numbers: int) -> bool:
    numeric_options = sum(bool(NUMERIC_OPTION.match(str(v))) for v in options.values())
    mostly_numeric = numeric_options >= max(2, 0.75 * len(options))
    if n_numbers < 2:
        return False
    return bool(CALC_WORDS.search(question)) and mostly_numeric or (mostly_numeric and n_numbers >= 3)


def text_features() -> pd.DataFrame:
    subjects = raw_subjects()
    rows = []
    with INPUT.open(encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            if not item["core"]:
                continue
            q = stem(item["question"])
            n_numbers = len(NUMBER.findall(q))
            pid = item["prompt_id"]
            rows.append(
                {
                    "prompt_id": pid,
                    "dataset": pid.split("_", 1)[0],
                    "is_hard": bool(item.get("hard", False)),
                    "subject": subjects.get(pid),
                    "n_words_question_en": len(q.split()),
                    "n_options": len(item["options"]),
                    "is_vignette": bool(VIGNETTE.search(q)),
                    "n_numbers": n_numbers,
                    "has_lab_values": bool(LAB_UNIT.search(q)),
                    "likely_calculation": likely_calculation(q, item["options"], n_numbers),
                }
            )
    df = pd.DataFrame(rows)
    assert df.dataset.isin(DATASETS).all()
    return df


def failure_counts() -> pd.DataFrame | None:
    """Number of the open benchmark configurations answering each item correctly, per language."""
    if not ITEMS.exists():
        return None
    b = pd.read_parquet(ITEMS, columns=["part", "config", "layer", "prompt_id", "is_correct"])
    b = b[(b.part == "benchmark") & b.layer.isin(["en", "L4a"])].rename(columns={"config": "model_dir"})
    b["lang"] = b.layer.map({"en": "en", "L4a": "sv"})
    if b.empty:
        return None
    b["is_correct"] = b.is_correct.fillna(False).astype(bool)
    counts = b.pivot_table(index="prompt_id", columns="lang", values="is_correct", aggfunc="sum")
    n_cfg = b.pivot_table(index="prompt_id", columns="lang", values="model_dir", aggfunc="nunique")
    out = pd.DataFrame(index=counts.index)
    for lang in ("en", "sv"):
        out[f"n_correct_{lang}"] = counts.get(lang).astype("Int64")
        out[f"n_configs_{lang}"] = n_cfg.get(lang).astype("Int64")
    return out.reset_index()


def build() -> pd.DataFrame:
    df = text_features()
    counts = failure_counts()
    if counts is None:
        print("  note: no per-item benchmark results; model-failure counts left empty")
        for c in ("n_correct_en", "n_correct_sv", "n_configs_en", "n_configs_sv"):
            df[c] = pd.array([pd.NA] * len(df), dtype="Int64")
    else:
        df = df.merge(counts, on="prompt_id", how="left")
    return df


def main() -> None:
    df = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)
    print(f"saved {OUT} ({len(df):,} items, {int(df.is_hard.sum())} hard)")
    print(df.groupby("is_hard")[["n_words_question_en", "is_vignette", "has_lab_values", "likely_calculation"]].mean())


if __name__ == "__main__":
    main()
