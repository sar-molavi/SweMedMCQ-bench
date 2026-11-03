"""Merge every per-item result into one table.

    python -m swemedmcq.evaluate.collect

Reads the scored jobs of this repository's results folder (paths.yaml ``results_dir``)
and writes ``<results_dir>/items.parquet`` with one row per (configuration, layer, item):

  part          benchmark | student
  config        configuration name (models.yaml name)
  model         display name (GPT-OSS-120B, LFM2.5-1.2B, ...)
  reasoning     on | off | single            (benchmark)
  student, regime, train_lang, prompt, decoding   (students; models.yaml `labels`)
  layer         en | L4a | L4
  prompt_id, dataset, is_correct, status, status_group, n_gen_tokens, n_reasoning_tokens,
  n_prompt_tokens (benchmark models only)
  core, category  from <testset_dir>/data/items.csv (core = the reported items)

Layer L4 = L4a with the items whose text differs replaced by their L4 (delta) results.
Also writes ``items_text.parquet`` (reasoning text and explanation per row) and
``collect_report.json`` (what was found and what is missing).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from swemedmcq import paths
from swemedmcq.evaluate.registry import Job, models

OUT = paths.path("results_dir")
STUDENT_LABEL = {"lfm": "LFM2.5-1.2B", "eurollm": "EuroLLM-1.7B"}
LAYER_OF_SET = {"en": "en", "sv_L4a": "L4a", "sv_L4_delta": "L4"}


def status_group(status: str) -> str:
    if status in ("correct", "wrong_answer"):
        return status
    if status in ("generation_not_finished", "not_finished"):
        return "not_finished"
    return "invalid_format"


# ---------------------------------------------------------------- benchmark
def benchmark() -> tuple[list[pd.DataFrame], list[pd.DataFrame], list[str]]:
    items, texts, missing = [], [], []
    root = OUT / "scored/bench_items"
    for m in models():
        if m.part != "benchmark":
            continue
        for set_ in m.sets:
            lang = "en" if set_ == "en" else "sv"
            shard = root / set_ / f"{m.name}__{lang}.parquet"
            if not shard.exists():
                missing.append(f"{set_}/{m.name}")
                continue
            df = pd.read_parquet(shard)
            ids = set(pd.read_json(Job(m, set_).prompts(), lines=True).prompt_id)
            df = df[df.prompt_id.isin(ids)]
            items.append(df.assign(layer=LAYER_OF_SET[set_], config=m.name))
            text = shard.with_name(shard.name.replace(".parquet", ".text.parquet"))
            if text.exists():
                t = pd.read_parquet(text)
                texts.append(t[t.prompt_id.isin(ids)].assign(layer=LAYER_OF_SET[set_], config=m.name))
    return items, texts, missing


def bench_frame(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "part": "benchmark", "config": df.config, "model": df.model, "reasoning": df.reasoning,
        "layer": df.layer, "prompt_id": df.prompt_id, "dataset": df.dataset,
        "is_correct": df.is_correct.astype(bool), "status": df.status,
        "status_group": df.status.map(status_group),
        "n_prompt_tokens": df.n_prompt_tokens, "n_gen_tokens": df.n_gen_tokens,
        "n_reasoning_tokens": df.n_reasoning_tokens.astype("Int64"),
    })


# ---------------------------------------------------------------- students
FIELDS = ("prompt_id", "dataset_name", "is_correct", "prediction_status", "num_generated_tokens",
          "reasoning", "explanation")


def clean(value):
    """Drop lone surrogates (invalid UTF-8) that some model completions contain."""
    if isinstance(value, str):
        return value.encode("utf-8", "replace").decode("utf-8")
    return value


def read_predictions(path: Path) -> pd.DataFrame:
    """Only the needed fields; the stdlib parser tolerates the lone surrogates some completions contain."""
    rows = []
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                rows.append({k: clean(row.get(k)) for k in FIELDS})
    return pd.DataFrame(rows, columns=FIELDS)


def student_rows(path: Path, m, layer: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = read_predictions(path)
    labels = m.labels
    meta = {"part": "student", "config": m.name, "model": STUDENT_LABEL[labels["student"]],
            "student": labels["student"], "regime": labels["regime"], "train_lang": labels["train_lang"],
            "prompt": m.prompt, "decoding": m.decoding, "layer": layer}
    items = pd.DataFrame({
        **meta, "prompt_id": df.prompt_id, "dataset": df.dataset_name,
        "is_correct": df.is_correct.fillna(False).astype(bool), "status": df.prediction_status,
        "status_group": df.prediction_status.map(status_group),
        "n_gen_tokens": df.num_generated_tokens,
    })
    text = pd.DataFrame({"config": m.name, "layer": layer, "prompt_id": df.prompt_id,
                         "reasoning_text": df.reasoning, "explanation": df.explanation})
    return items, text


def students() -> tuple[list[pd.DataFrame], list[pd.DataFrame], list[str]]:
    items, texts, missing = [], [], []
    for m in models():
        if m.part != "student":
            continue
        for set_ in m.sets:
            path = OUT / "scored" / set_ / m.name / "scored/predictions.jsonl"
            if path.exists():
                i, t = student_rows(path, m, LAYER_OF_SET[set_])
                items.append(i), texts.append(t)
            else:
                missing.append(f"{set_}/{m.name}")
    return items, texts, missing


# ---------------------------------------------------------------- assembly
def add_full_l4(df: pd.DataFrame) -> pd.DataFrame:
    """L4 rows exist only for the delta items; complete them with the L4a rows of the rest."""
    delta = df[df.layer == "L4"]
    if delta.empty:
        return df
    base = df[(df.layer == "L4a") & df.config.isin(delta.config.unique())]
    keep = base.merge(delta[["config", "prompt_id"]], on=["config", "prompt_id"], how="left", indicator=True)
    rest = base[(keep["_merge"] == "left_only").to_numpy()].assign(layer="L4")
    return pd.concat([df, rest], ignore_index=True)


def main() -> None:
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    items_csv = pd.read_csv(paths.path("testset_dir") / "data/items.csv")
    b_items, b_texts, b_missing = benchmark()
    s_items, s_texts, s_missing = students()
    frames = ([bench_frame(pd.concat(b_items, ignore_index=True))] if b_items else []) + s_items
    if not frames:
        raise SystemExit(f"No scored results under {OUT / 'scored'}")
    df = pd.concat(frames, ignore_index=True)
    df = add_full_l4(df)
    df = df.merge(items_csv[["prompt_id", "core", "status"]].rename(columns={"status": "category"}),
                  on="prompt_id", how="left")
    n_items = len(items_csv)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT / "items.parquet", index=False, compression="zstd")
    text_frames = b_texts + s_texts
    if text_frames:
        text = pd.concat(text_frames, ignore_index=True)[
            ["config", "layer", "prompt_id", "reasoning_text", "explanation"]].astype("string")
        text.to_parquet(OUT / "items_text.parquet", index=False, compression="zstd")

    counts = df.groupby(["part", "config", "layer"]).size().unstack(fill_value=0)
    report = {"rows": len(df), "configs_by_layer": {c: int((counts[c] > 0).sum()) for c in counts.columns},
              "missing_jobs": b_missing + s_missing,
              # Every layer has all items once complete (L4 after add_full_l4).
              "incomplete": [f"{c}/{l}: {n}" for c, row in counts.droplevel(0).iterrows()
                             for l, n in row.items() if 0 < n < n_items]}
    (OUT / "collect_report.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(counts.to_string())
    print(json.dumps({k: v for k, v in report.items() if k != "missing_jobs"}, indent=1))
    print(f"{len(report['missing_jobs'])} jobs not scored yet")


if __name__ == "__main__":
    main()
