"""Label every English test item: question type (10), specialty (17) and three flags.

GPT-OSS-120B, greedy decoding, reasoning effort low, JSON output. The model sees the
question and its options (``prompt.md`` as the system message), never the answer.
It runs in-process with vLLM's AsyncLLMEngine; the model reasons first (Harmony
``analysis`` channel) and the JSON-schema grammar is switched on only after
``<|channel|>final<|message|>`` by the ``gptoss_structured`` reasoning-parser plugin of
the refinement engine. A failed or invalid output gets one sampled retry (T=0.6).

Resumable: every attempt is appended to a JSONL cache keyed by the sha256 of
(model id, rendered prompt, sampling parameters, schema). A re-run skips items whose
attempt is cached as ok and retries failed ones. Items with identical question and
options share one cache entry and get the same labels.

    python -m swemedmcq.evaluate.question_tags.tag_items --gpus 0,1 --limit 100   # smoke test
    python -m swemedmcq.evaluate.question_tags.tag_items --gpus 0,1               # all 10,363 items
    python -m swemedmcq.evaluate.question_tags.tag_items --export-only            # tags.parquet from the cache

Writes <results_dir>/question_tags/{tag_cache.jsonl,tags.parquet,spotcheck_50.csv}.
GPT-OSS needs the harmony vocab files on offline nodes (scripts/refine/fetch_offline.sh).
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import random
import time
from pathlib import Path
from typing import Literal

from swemedmcq import paths
from swemedmcq.evaluate.question_tags.features import stem

HERE = Path(__file__).resolve().parent
OUT_DIR = paths.path("results_dir") / "question_tags"
INPUT = paths.path("benchmark")  # English question and options of every test item
PARSER_PLUGIN = paths.ROOT / "swemedmcq/refine/engine/parsers/gptoss_structured.py"
FINAL_MARKER = "<|channel|>final<|message|>"
FIXED_DATE = "2026-10-08"  # replaces strftime_now() so prompts (and cache keys) are stable

QUESTION_TYPES = [
    "diagnosis", "next_step_management", "treatment_pharmacology", "mechanism_pathophysiology",
    "basic_science_anatomy", "test_interpretation", "calculation", "epidemiology_statistics_research",
    "ethics_communication", "other",
]
SPECIALTIES = [
    "internal_medicine", "cardiology", "infectious_disease", "pharmacology", "surgery",
    "obstetrics_gynecology", "pediatrics", "psychiatry_psychology", "neurology", "oncology_hematology",
    "microbiology_immunology", "biochemistry_genetics", "anatomy_physiology", "public_health_epidemiology",
    "eye_ent_skin", "dentistry_oral", "allied_health_other",
]
FLAGS = ["needs_calculation", "has_lab_values", "is_clinical_vignette"]


def schema_model():
    from pydantic import BaseModel, ConfigDict

    class Tags(BaseModel):
        model_config = ConfigDict(extra="forbid")
        question_type: Literal[tuple(QUESTION_TYPES)]  # type: ignore[valid-type]
        specialty: Literal[tuple(SPECIALTIES)]  # type: ignore[valid-type]
        needs_calculation: bool
        has_lab_values: bool
        is_clinical_vignette: bool

    return Tags


def user_message(item: dict) -> str:
    options = "\n".join(f"{k}. {v}" for k, v in item["options"].items())
    return f"Question:\n{stem(item['question']).strip()}\n\nOptions:\n{options}"


def load_items() -> list[dict]:
    with INPUT.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    return [{k: row[k] for k in ("prompt_id", "question", "options")} for row in rows]


def sha(*parts) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def read_cache(path: Path) -> dict[str, dict]:
    """Latest record per cache key."""
    records: dict[str, dict] = {}
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:  # torn last line after a crash
                    continue
                records[rec["key"]] = rec
    return records


def model_id(path: str) -> str:
    hub = Path(os.environ.get("HF_HUB_CACHE", ""))
    ref = hub / ("models--" + path.replace("/", "--")) / "refs/main"
    return f"{path}@{ref.read_text().strip()}" if ref.exists() else path


# --------------------------------------------------------------------------- generation


async def run(args, items: list[dict], cache_path: Path) -> None:
    from transformers import AutoTokenizer
    from vllm import SamplingParams, TokensPrompt
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.engine.async_llm_engine import AsyncLLMEngine
    from vllm.sampling_params import RequestOutputKind, StructuredOutputsParams

    Tags = schema_model()
    schema = Tags.model_json_schema()
    system = (HERE / "prompt.md").read_text(encoding="utf-8").strip()
    tok = AutoTokenizer.from_pretrained(args.model)
    template = tok.chat_template.replace('strftime_now("%Y-%m-%d")', f'"{FIXED_DATE}"')
    mid = model_id(args.model)

    def render(item: dict) -> list[int]:
        ids = tok.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": user_message(item)}],
            add_generation_prompt=True, tokenize=True, return_dict=False,
            chat_template=template, reasoning_effort=args.reasoning_effort,
        )
        return list(ids["input_ids"] if isinstance(ids, dict) else ids)

    def attempts(prompt_id: str) -> list[dict]:
        seed = int(sha(prompt_id)[:8], 16)
        return [{"temperature": 0.0, "seed": None}, {"temperature": 0.6, "top_p": 1.0, "seed": seed}]

    cache = read_cache(cache_path)
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpus)
    engine = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(
        model=args.model,
        tensor_parallel_size=len(gpus),
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=True,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        distributed_executor_backend="mp",
        disable_log_stats=True,
        structured_outputs_config={
            "reasoning_parser": "gptoss_structured",
            "reasoning_parser_plugin": str(PARSER_PLUGIN),
            "enable_in_reasoning": False,
        },
    ))
    handle = cache_path.open("a", encoding="utf-8")
    counts = {"cached": 0, "ok": 0, "failed": 0}
    sem = asyncio.Semaphore(args.concurrency)
    t0 = time.monotonic()

    async def one(item: dict) -> None:
        prompt = render(item)
        for i, sampling in enumerate(attempts(item["prompt_id"]), start=1):
            max_tokens = min(args.max_tokens, args.max_model_len - len(prompt))
            key = sha("hq-tags-v1", mid, prompt, sampling, schema, max_tokens, args.reasoning_effort)
            rec = cache.get(key)
            if rec is not None and rec["status"] == "ok":
                counts["cached"] += 1
                return
            if rec is not None and not str(rec["error"]).startswith("engine_error"):
                continue  # deterministic failure already recorded: go to the sampled retry
            async with sem:
                rec = await generate(item, prompt, sampling, max_tokens, key, i)
            cache[key] = rec
            handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
            handle.flush()
            if rec["status"] == "ok":
                counts["ok"] += 1
                break
        else:
            counts["failed"] += 1
        done = sum(counts.values())
        if done % 200 == 0:
            rate = (counts["ok"] + counts["failed"]) / max(time.monotonic() - t0, 1e-6)
            print(f"  {done:,}/{len(items):,} {counts} {rate:.2f} items/s", flush=True)

    async def generate(item, prompt, sampling, max_tokens, key, attempt) -> dict:
        rec = {"key": key, "prompt_id": item["prompt_id"], "model": mid, "attempt": attempt,
               "sampling": sampling, "reasoning_effort": args.reasoning_effort, "status": "failed",
               "parsed": None, "error": None, "reasoning": "", "content": "", "finish_reason": None,
               "n_prompt_tokens": len(prompt), "n_gen_tokens": 0}
        if max_tokens <= 0:
            rec["error"] = "over_context"
            return rec
        params = SamplingParams(
            temperature=sampling["temperature"], max_tokens=max_tokens, detokenize=False,
            output_kind=RequestOutputKind.FINAL_ONLY, structured_outputs=StructuredOutputsParams(json=schema),
            **{k: v for k, v in sampling.items() if k in ("top_p", "seed") and v is not None},
        )
        final = None
        try:
            async for out in engine.generate(TokensPrompt(prompt_token_ids=prompt), params, request_id=f"{item['prompt_id']}-{attempt}"):
                final = out
            output = final.outputs[0]
        except Exception as exc:  # engine error: recorded, retried on the next run
            rec["error"] = f"engine_error: {type(exc).__name__}: {exc}"
            return rec
        text = tok.decode(list(output.token_ids), skip_special_tokens=False)
        head, sep, tail = text.partition(FINAL_MARKER)
        content = tail.split("<|return|>")[0].split("<|end|>")[0].strip() if sep else ""
        reasoning = head.replace("<|channel|>analysis<|message|>", "").split("<|end|>")[0].strip()
        rec.update(reasoning=reasoning, content=content, finish_reason=output.finish_reason,
                   n_gen_tokens=len(output.token_ids))
        try:
            rec["parsed"] = Tags.model_validate_json(content).model_dump()
            rec["status"] = "ok"
        except Exception as exc:
            rec["error"] = f"invalid_json (finish_reason={output.finish_reason}): {str(exc)[:200]}"
        return rec

    try:
        await asyncio.gather(*(one(item) for item in items))
    finally:
        handle.close()
        engine.shutdown()
    print(f"done: {counts} in {time.monotonic() - t0:.0f}s")


# --------------------------------------------------------------------------- export


def export(items: list[dict], cache_path: Path, out: Path) -> None:
    import pandas as pd

    ok: dict[str, dict] = {}
    for rec in read_cache(cache_path).values():
        if rec["status"] == "ok":
            prev = ok.get(rec["prompt_id"])
            if prev is None or rec["attempt"] < prev["attempt"]:
                ok[rec["prompt_id"]] = rec
    # Items with identical question + options share one cache entry (same prompt,
    # same key), which records only one prompt_id: share its tags with the duplicates.
    by_text: dict[str, dict] = {}
    for item in items:
        if item["prompt_id"] in ok:
            by_text.setdefault(sha(user_message(item)), ok[item["prompt_id"]])
    rows = []
    for item in items:
        rec = ok.get(item["prompt_id"]) or by_text.get(sha(user_message(item)))
        if rec is None:
            continue
        rows.append({"prompt_id": item["prompt_id"], "dataset": item["prompt_id"].split("_", 1)[0],
                     **rec["parsed"], "tag_model": rec["model"].split("@")[0], "tag_attempt": rec["attempt"],
                     "tag_reasoning_effort": rec["reasoning_effort"], "tag_gen_tokens": rec["n_gen_tokens"]})
    df = pd.DataFrame(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"saved {out} ({len(df):,} of {len(items):,} items tagged)")
    if len(df) >= 50:
        by_id = {item["prompt_id"]: item for item in items}
        spot = df.sample(50, random_state=7).copy()
        spot["question"] = [by_id[p]["question"][:300].replace("\n", " ") for p in spot.prompt_id]
        spot["options"] = [" | ".join(f"{k}: {v}" for k, v in by_id[p]["options"].items())[:300] for p in spot.prompt_id]
        cols = ["prompt_id", "dataset", "question", "options", "question_type", "specialty", *FLAGS]
        spot = spot[cols]
        for c in ("question_type", "specialty", *FLAGS):
            spot[f"ok_{c}"] = ""
        spot["notes"] = ""
        spot.to_csv(out.parent / "spotcheck_50.csv", index=False)
        print(f"saved {out.parent / 'spotcheck_50.csv'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default="openai/gpt-oss-120b")
    ap.add_argument("--gpus", default="0,1", help="physical GPU ids; TP = number of GPUs")
    ap.add_argument("--limit", type=int, default=None, help="random sample of N items (seed 0) for smoke tests")
    ap.add_argument("--out", type=Path, default=OUT_DIR / "tags.parquet")
    ap.add_argument("--cache", type=Path, default=OUT_DIR / "tag_cache.jsonl")
    ap.add_argument("--reasoning-effort", default="low", choices=["low", "medium", "high"])
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--max-num-seqs", type=int, default=64)
    ap.add_argument("--max-num-batched-tokens", type=int, default=1024)
    ap.add_argument("--concurrency", type=int, default=256)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.955)
    ap.add_argument("--export-only", action="store_true", help="only rebuild tags.parquet from the cache")
    args = ap.parse_args()

    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

    items = load_items()
    selected = items
    if args.limit is not None:
        selected = random.Random(0).sample(items, min(args.limit, len(items)))
    args.cache.parent.mkdir(parents=True, exist_ok=True)
    if not args.export_only:
        asyncio.run(run(args, selected, args.cache))
    export(items, args.cache, args.out)


if __name__ == "__main__":
    main()
