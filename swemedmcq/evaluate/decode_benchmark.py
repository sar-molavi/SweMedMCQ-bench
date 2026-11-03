#!/usr/bin/env python
"""Decode the benchmark-model vLLM pickles into one row per (config, language, item).

Unpickling needs vLLM (inference environment). ``swemedmcq.evaluate.run`` calls this
for every benchmark job; by hand:

    python -m swemedmcq.evaluate.decode_benchmark --input-root DIR --gold GOLD.jsonl \
        --cache-dir OUT [--langs en sv] [--only REGEX]

Input layout:

    <input-root>/en_response/<model_dir>/result_<prompt_id>.pkl
    <input-root>/sv_response/<model_dir>/...     (or sw_response/)

The answer is parsed with the shared extractor (``swemedmcq.common.utils``
candidates + ``ResponseOutput`` schema validation) and scored: the upper-cased
``answer_index`` must equal the gold letter, and a missing or unparsable
prediction counts as wrong.

Gold letters come from ``--gold``: a JSONL file with ``prompt_id`` and ``gold_answer``
(the answer split of the test set) or ``answer`` (the benchmark file). Every item of
that file gets a row (``missing`` when it has no pickle). ``is_hard`` is taken from a
boolean ``hard`` field of the benchmark file (``--benchmark``) when it has one.

Outputs (in --cache-dir):

    <tag>/<model_dir>__<lang>.parquet        per-shard item rows
    <tag>/<model_dir>__<lang>.text.parquet   per-shard reasoning text and explanation
    <tag>.parquet                            merged item rows
    <tag>_text.parquet                       merged text rows
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path

from swemedmcq import paths

TOKENIZER_DIR = paths.path("results_dir") / "tokenizers"

ITEM_COLUMNS = ["model_dir", "model", "reasoning", "lang", "prompt_id", "dataset", "gold",
                "answer", "is_correct", "status", "n_prompt_tokens", "n_gen_tokens",
                "n_reasoning_tokens", "finish_reason", "is_hard"]
TEXT_COLUMNS = ["prompt_id", "model_dir", "lang", "reasoning_text", "explanation"]
TOKENIZER_PATTERNS = ["tokenizer*", "special_tokens*", "chat_template*", "vocab*", "merges*",
                      "added_tokens*", "config.json", "generation_config.json"]


@dataclass(frozen=True)
class Config:
    model: str        # display label
    reasoning: str    # "on" | "off" | "single"
    repo: str         # HF repo of the tokenizer
    fmt: str          # response format of the extractor: gpt_oss | qwen | medgemma | gemma4


CONFIGS = {
    "Qwen2.5-3B-Instruct": Config("Qwen2.5-3B", "single", "Qwen/Qwen2.5-3B-Instruct", "qwen"),
    "Qwen3-4B-Thinking-2507": Config("Qwen3-4B-Thinking", "on", "Qwen/Qwen3-4B-Thinking-2507", "qwen"),
    "Qwen3-4B-Thinking-2507-no-reasoning": Config("Qwen3-4B-Thinking", "off", "Qwen/Qwen3-4B-Thinking-2507", "qwen"),
    "Qwen3.5-35B-A3B": Config("Qwen3.5-35B-A3B", "on", "Qwen/Qwen3.5-35B-A3B", "qwen"),
    "Qwen3.5-35B-A3B-no-reasoning": Config("Qwen3.5-35B-A3B", "off", "Qwen/Qwen3.5-35B-A3B", "qwen"),
    "Qwen3.6-35B-A3B": Config("Qwen3.6-35B-A3B", "on", "Qwen/Qwen3.6-35B-A3B", "qwen"),
    "Qwen3.6-35B-A3B-no-reasoning": Config("Qwen3.6-35B-A3B", "off", "Qwen/Qwen3.6-35B-A3B", "qwen"),
    "gpt-oss-120b": Config("GPT-OSS-120B", "on", "openai/gpt-oss-120b", "gpt_oss"),
    "gpt-oss-120b-no-reasoning": Config("GPT-OSS-120B", "off", "openai/gpt-oss-120b", "gpt_oss"),
    "gpt-oss-20b": Config("GPT-OSS-20B", "on", "openai/gpt-oss-20b", "gpt_oss"),
    "gpt-oss-20b-no-reasoning": Config("GPT-OSS-20B", "off", "openai/gpt-oss-20b", "gpt_oss"),
    "medgemma-27b-text-it": Config("MedGemma-27B", "single", "google/medgemma-27b-text-it", "medgemma"),
    # Refinement-loop models (configs/models.yaml). Gemma-4 reasoning runs prefill
    # "<|channel>thought\n", so the output starts inside the thought and closes it with <channel|>.
    "Qwen3.8-27B": Config("Qwen3.8-27B", "on", "Qwen/Qwen3.8-27B", "qwen"),
    "Qwen3.8-27B-no-reasoning": Config("Qwen3.8-27B", "off", "Qwen/Qwen3.8-27B", "qwen"),
    "gemma-4-31B-it": Config("Gemma-4-31B", "on", "google/gemma-4-31B-it", "gemma4"),
    "gemma-4-31B-it-no-reasoning": Config("Gemma-4-31B", "off", "google/gemma-4-31B-it", "gemma4"),
}
LANG_DIRS = {"en": ["en_response"], "sv": ["sw_response", "sv_response"]}


# ---------------------------------------------------------------- tokenizers

def tokenizer_path(repo: str) -> Path:
    return TOKENIZER_DIR / repo.replace("/", "--")


def stage_tokenizer(repo: str) -> Path:
    """Copy tokenizer files into <results_dir>/tokenizers/ (HF cache first, else download)."""
    dest = tokenizer_path(repo)
    if (dest / "tokenizer_config.json").exists():
        return dest
    from huggingface_hub import snapshot_download

    try:
        src = Path(snapshot_download(repo, allow_patterns=TOKENIZER_PATTERNS, local_files_only=True))
        if not (src / "tokenizer_config.json").exists():
            raise FileNotFoundError(src)
        tmp = dest.with_name(dest.name + ".tmp")
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        for path in src.iterdir():
            if path.is_file() and any(path.match(p) for p in TOKENIZER_PATTERNS):
                shutil.copyfile(path.resolve(), tmp / path.name)
        tmp.rename(dest)
    except Exception:
        snapshot_download(repo, allow_patterns=TOKENIZER_PATTERNS, local_dir=str(dest))
    print(f"tokenizer {repo} -> {dest}", flush=True)
    return dest


# ---------------------------------------------------------------- reference data

def read_rows(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def load_gold(path: Path) -> dict[str, str]:
    gold = {}
    for row in read_rows(path):
        gold[row["prompt_id"]] = str(row.get("gold_answer", row.get("answer"))).strip().upper()
    return gold


def load_hard(path: Path | None) -> set[str]:
    if path is None or not path.exists():
        return set()
    return {row["prompt_id"] for row in read_rows(path) if row.get("hard") or row.get("is_hard")}


# ---------------------------------------------------------------- per item

def _find(ids: list[int], pattern: list[int]) -> int:
    n = len(pattern)
    first = pattern[0]
    for i, tok in enumerate(ids):
        if tok == first and ids[i:i + n] == pattern:
            return i
    return -1


class Markers:
    """Token-level markers that separate reasoning from the final answer."""

    def __init__(self, tokenizer, fmt: str):
        self.fmt = fmt
        self.final = self.analysis = self.think_end = self.channel_end = None
        if fmt == "gpt_oss":
            channel = tokenizer.convert_tokens_to_ids("<|channel|>")
            self.final = [channel] + tokenizer.encode("final", add_special_tokens=False)
            self.analysis = [channel] + tokenizer.encode("analysis", add_special_tokens=False)
        elif fmt == "qwen":
            tid = tokenizer.convert_tokens_to_ids("</think>")
            self.think_end = tid if isinstance(tid, int) and tid != tokenizer.unk_token_id else None
        elif fmt == "gemma4":
            self.channel_end = tokenizer.convert_tokens_to_ids("<channel|>")

    def reasoning_tokens(self, ids: list[int], mode: str) -> int | None:
        """Generated tokens before the final-answer segment (reasoning plus its delimiters).

        If the reasoning segment never closes (truncated), all generated tokens count.
        """
        if mode == "single":
            return None
        if self.fmt == "gpt_oss":
            pos = _find(ids, self.final)
            if pos >= 0:
                return pos
            return len(ids) if _find(ids, self.analysis) >= 0 else 0
        if self.fmt == "qwen" and self.think_end is not None:
            if self.think_end in ids:
                return ids.index(self.think_end) + 1
            return len(ids) if mode == "on" else 0
        if self.fmt == "gemma4":  # the thought channel is opened in the prompt; up to <channel|>
            if self.channel_end in ids:
                return ids.index(self.channel_end) + 1
            return len(ids) if mode == "on" else 0
        return 0


def reasoning_text_of(decoded: str, fmt: str, mode: str, extract_reasoning_trace) -> str | None:
    if mode == "single":
        return None
    text = extract_reasoning_trace(decoded, fmt)
    if text:
        return text
    # Truncated reasoning (no closing marker): keep the partial trace.
    if fmt == "gpt_oss":
        head = "<|channel|>analysis<|message|>"
        if head in decoded and "<|channel|>final" not in decoded:
            return decoded.split(head, 1)[1].strip() or None
    elif fmt == "qwen" and mode == "on" and "</think>" not in decoded:
        return decoded.replace("<think>", "").strip() or None
    elif fmt == "gemma4" and mode == "on" and "<channel|>" not in decoded:
        text = decoded.split("<|channel>", 1)[-1].lstrip()
        return text.removeprefix("thought\n").removesuffix("<turn|>").strip() or None
    return None


def process_shard(task: dict) -> dict:
    """Decode one (model_dir, lang) directory; write its two parquet shards."""
    import pandas as pd
    from transformers import AutoTokenizer

    from swemedmcq.common.result_io import parse_json_candidates
    from swemedmcq.common.schemas import ResponseOutput, SweResponseOutput
    from swemedmcq.common.utils import build_structured_output_candidates, extract_reasoning_trace

    import pickle

    model_dir, lang, cfg = task["model_dir"], task["lang"], CONFIGS[task["model_dir"]]
    fmt = task["fmt"] or cfg.fmt
    in_dir = Path(task["in_dir"])
    schema = SweResponseOutput if lang == "sv" else ResponseOutput
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path(cfg.repo)))
    markers = Markers(tokenizer, cfg.fmt)
    gold, hard = task["gold"], task["hard"]
    prompt_ids = sorted(gold)
    if task["limit"]:
        prompt_ids = prompt_ids[: task["limit"]]

    items, texts = [], []
    t0 = time.time()
    for k, prompt_id in enumerate(prompt_ids, 1):
        dataset = prompt_id.split("_", 1)[0]
        row = {"model_dir": model_dir, "model": cfg.model, "reasoning": cfg.reasoning, "lang": lang,
               "prompt_id": prompt_id, "dataset": dataset, "gold": gold[prompt_id], "answer": None,
               "is_correct": False, "status": "missing", "n_prompt_tokens": 0, "n_gen_tokens": 0,
               "n_reasoning_tokens": None, "finish_reason": "missing", "is_hard": prompt_id in hard}
        text = {"prompt_id": prompt_id, "model_dir": model_dir, "lang": lang,
                "reasoning_text": None, "explanation": None}
        path = in_dir / f"result_{prompt_id}.pkl"
        if path.exists():
            with path.open("rb") as handle:
                obj = pickle.load(handle)
            out = obj.outputs[0]
            ids = list(out.token_ids)
            decoded = tokenizer.decode(ids)
            payload = parse_json_candidates(build_structured_output_candidates(decoded, fmt), schema)
            answer = None
            if payload is not None:
                answer = str(payload.get("answer_index") or "").strip().upper() or None
                text["explanation"] = payload.get("explanation")
            finish = str(out.finish_reason)
            row.update(
                answer=answer,
                is_correct=answer is not None and answer == row["gold"],
                n_prompt_tokens=len(obj.prompt_token_ids or []),
                n_gen_tokens=len(ids),
                n_reasoning_tokens=markers.reasoning_tokens(ids, cfg.reasoning),
                finish_reason=finish,
            )
            if answer is not None:
                row["status"] = "correct" if row["is_correct"] else "wrong_answer"
            else:
                row["status"] = "not_finished" if finish == "length" else "invalid"
            text["reasoning_text"] = reasoning_text_of(decoded, fmt, cfg.reasoning, extract_reasoning_trace)
        items.append(row)
        texts.append(text)
        if k % 2000 == 0:
            print(f"  {model_dir}/{lang}: {k}/{len(prompt_ids)} ({time.time() - t0:.0f}s)", flush=True)

    df = pd.DataFrame(items, columns=ITEM_COLUMNS)
    df["n_reasoning_tokens"] = df["n_reasoning_tokens"].astype("Int64")
    df["answer"] = df["answer"].astype("string")
    tf = pd.DataFrame(texts, columns=TEXT_COLUMNS).astype("string")
    for frame, dest in ((tf, Path(task["text_path"])), (df, Path(task["item_path"]))):
        tmp = dest.with_name(dest.name + ".tmp")
        frame.to_parquet(tmp, index=False, compression="zstd")
        tmp.replace(dest)
    acc = df.is_correct.mean() * 100
    print(f"done {model_dir}/{lang}: {len(df)} rows, acc {acc:.2f}, "
          f"status {df.status.value_counts().to_dict()} ({time.time() - t0:.0f}s)", flush=True)
    return {"model_dir": model_dir, "lang": lang, "rows": len(df), "accuracy": acc}


# ---------------------------------------------------------------- driver

def discover(input_root: Path, langs: list[str], only: str | None) -> list[tuple[str, str, Path]]:
    import re

    found = []
    for lang in langs:
        lang_dir = next((input_root / d for d in LANG_DIRS[lang] if (input_root / d).is_dir()), None)
        if lang_dir is None:
            print(f"WARNING: no {LANG_DIRS[lang]} under {input_root}", flush=True)
            continue
        for model_dir in sorted(p.name for p in lang_dir.iterdir() if p.is_dir()):
            if model_dir not in CONFIGS:
                continue
            if only and not re.search(only, model_dir):
                continue
            found.append((model_dir, lang, lang_dir / model_dir))
    return found


def merge(shard_dir: Path, out_items: Path, out_text: Path) -> None:
    import pandas as pd
    import pyarrow.parquet as pq

    item_paths = sorted(p for p in shard_dir.glob("*.parquet") if not p.name.endswith(".text.parquet"))
    df = pd.concat([pd.read_parquet(p) for p in item_paths], ignore_index=True)
    df = df[ITEM_COLUMNS].sort_values(["model_dir", "lang", "prompt_id"]).reset_index(drop=True)
    tmp = out_items.with_name(out_items.name + ".tmp")
    df.to_parquet(tmp, index=False, compression="zstd")
    tmp.replace(out_items)
    print(f"merged {len(item_paths)} shards -> {out_items} ({len(df)} rows)", flush=True)

    import pyarrow as pa

    schema = pa.schema([(c, pa.large_string()) for c in TEXT_COLUMNS])
    text_paths = sorted(shard_dir.glob("*.text.parquet"))
    tmp = out_text.with_name(out_text.name + ".tmp")
    with pq.ParquetWriter(tmp, schema, compression="zstd") as writer:
        for p in text_paths:
            table = pq.read_table(p, columns=TEXT_COLUMNS)
            writer.write_table(pa.table([table[c].cast(pa.large_string()) for c in TEXT_COLUMNS], schema=schema))
    tmp.replace(out_text)
    print(f"merged {len(text_paths)} text shards -> {out_text}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-root", type=Path, required=True,
                        help="Root with en_response/ and sv_response/ (or sw_response/).")
    parser.add_argument("--gold", type=Path, required=True,
                        help="JSONL with prompt_id and gold_answer (answer split) or answer (benchmark file).")
    parser.add_argument("--benchmark", type=Path, default=paths.path("benchmark"),
                        help="Benchmark file; its boolean `hard` field (if any) fills is_hard.")
    parser.add_argument("--tag", default="benchmark_items",
                        help="Output name: cache/<tag>/ shards, cache/<tag>.parquet, cache/<tag>_text.parquet.")
    parser.add_argument("--langs", nargs="+", default=["en", "sv"], choices=["en", "sv"])
    parser.add_argument("--only", help="Regex on model_dir.")
    parser.add_argument("--limit", type=int, default=0, help="Smoke test: first N items per shard.")
    parser.add_argument("--workers", type=int, default=12, help="Parallel shards (max 16).")
    parser.add_argument("--response-format", choices=["gpt_oss", "qwen", "medgemma", "gemma4", "auto"],
                        help="Override the per-model extractor format.")
    parser.add_argument("--force", action="store_true", help="Recompute existing shards.")
    parser.add_argument("--no-merge", action="store_true")
    parser.add_argument("--cache-dir", type=Path, default=paths.path("results_dir") / "scored" / "bench_items",
                        help="Where <tag>/ shards and merged parquets go.")
    args = parser.parse_args()

    tag = args.tag if not args.limit else f"{args.tag}_limit{args.limit}"
    shard_dir = args.cache_dir / tag
    shard_dir.mkdir(parents=True, exist_ok=True)
    gold, hard = load_gold(args.gold), load_hard(args.benchmark)

    tasks = []
    for model_dir, lang, in_dir in discover(args.input_root, args.langs, args.only):
        item_path = shard_dir / f"{model_dir}__{lang}.parquet"
        text_path = shard_dir / f"{model_dir}__{lang}.text.parquet"
        if item_path.exists() and text_path.exists() and not args.force:
            print(f"skip {model_dir}/{lang} (shard exists)", flush=True)
            continue
        tasks.append({"model_dir": model_dir, "lang": lang, "in_dir": str(in_dir), "gold": gold,
                      "hard": hard, "limit": args.limit, "fmt": args.response_format,
                      "item_path": str(item_path), "text_path": str(text_path)})

    for repo in sorted({CONFIGS[t["model_dir"]].repo for t in tasks}):
        stage_tokenizer(repo)

    # Largest (reasoning-on) shards first.
    tasks.sort(key=lambda t: CONFIGS[t["model_dir"]].reasoning != "on")
    print(f"input root {args.input_root}; {len(tasks)} shards to decode -> {shard_dir}", flush=True)
    if tasks:
        workers = max(1, min(args.workers, 16, len(tasks)))
        with get_context("spawn").Pool(workers, maxtasksperchild=1) as pool:
            for result in pool.imap_unordered(process_shard, tasks):
                pass
    if not args.no_merge:
        merge(shard_dir, args.cache_dir / f"{tag}.parquet", args.cache_dir / f"{tag}_text.parquet")


if __name__ == "__main__":
    main()
