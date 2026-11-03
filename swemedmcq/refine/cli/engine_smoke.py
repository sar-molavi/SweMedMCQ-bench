"""Engine smoke test: all models loaded at once, reasoning + valid JSON, batching, cache replay.

Usage (from the repository root):

    python -m swemedmcq.refine.cli.engine_smoke --config configs/refine/default.yaml
    python -m swemedmcq.refine.cli.engine_smoke --config configs/refine/with_gemma.yaml \
        --models gptoss,qwen,gemma --run-name engine_smoke --fresh

Checks (exit code 1 if any fails):
  1. Rendering: GPT-OSS is reproducible and carries the fixed date; Qwen opens
     a ``<think>`` block; Gemma (native thinking) has ``<|think|>`` in the
     system turn and ends with the prefilled ``<|channel>thought\n`` (ids
     100, 45518, 107), or (schema mode) with the template's closed empty block.
  2. Every engine's processes sit only on its configured GPUs (nvidia-smi;
     ``--placement warn`` reports a mismatch without failing).
  3. N short prompts per model with a tiny schema: status ok, schema-valid
     JSON, NON-EMPTY reasoning (guards the "forced to JSON from token 1" bug),
     no fallback needed; Gemma: no thought-channel leakage in the content.
  4. ~200 concurrent requests per model with the real ``pairwise`` schema
     (all models at once): throughput, failures, finish reasons, reasoning.
  5. A rerun of every request is served 100 % from the cache (no GPU attempt).
  6. The parent process holds no CUDA context / GPU memory.

Writes ``<out-dir>/summary.json`` (default ``$RUNS_DIR/<run-name>`` or
``<refine_runs>/<run-name>``, paths.yaml) with per-model reasoning token/char statistics and
three reasoning excerpts per model, and ``<out-dir>/reasoning_samples.jsonl``
with the full reasoning and content of the short prompts.  The cache lives in
``<out-dir>/cache`` unless ``--cache-dir`` is given.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import shutil
import statistics
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel

from swemedmcq.paths import CONFIG_DIR, path as repo_path


class SmokeAnswer(BaseModel):
    answer: str
    confidence: Literal["low", "medium", "high"]


class SmokeAnswerAnalysis(BaseModel):
    """Reason-then-answer variant for Gemma ``thinking: schema``."""

    analysis: str
    answer: str
    confidence: Literal["low", "medium", "high"]


class SmokePairwiseAnalysis(BaseModel):
    analysis: str
    reason: str
    preferred: Literal["A", "B", "equal", "both_wrong"]


SMOKE_SCHEMAS = {
    "smoke_answer": SmokeAnswer,
    "smoke_answer_analysis": SmokeAnswerAnalysis,
    "smoke_pairwise_analysis": SmokePairwiseAnalysis,
}
ALL_MODELS = ("gptoss", "qwen", "gemma")
# Thought-channel markers that must never reach Gemma's content.
GEMMA_LEAK_MARKERS = ("<|channel>", "<channel|>", "<|think|>", "<|turn>", "<turn|>")
GEMMA_PREFILL_IDS = [100, 45518, 107]


SYSTEM = (
    "You are a careful reviewer of Swedish medical translations. "
    "Answer with a JSON object with the keys 'answer' (a short Swedish or English answer) "
    "and 'confidence' (low, medium or high)."
)
SHORT_PROMPTS = [
    "Is 'vankomycin' the correct Swedish spelling of the antibiotic vancomycin? Answer yes or no.",
    "Translate 'metabolic acidosis' into Swedish medical terminology.",
    "Is 'Patienten har feber på 38,4 °C' a correct Swedish rendering of 'The patient has a fever of 38.4 °C'?",
    "What is the Swedish word for 'resistance' in the context of antibiotic resistance?",
    "Should 'myocardial infarction' be translated as 'hjärtinfarkt' or 'myokardinfarkt' in a Swedish exam question?",
]

PAIRWISE_SYSTEM = (
    "You compare two Swedish translations (A and B) of one English sentence from a medical exam. "
    "Prefer the translation that is more accurate and more idiomatic clinical Swedish. "
    "Return JSON with 'reason' (one or two sentences) and 'preferred' (A, B, equal or both_wrong)."
)
_SENTENCES = [
    ("The patient was started on intravenous vancomycin.", "Patienten sattes in på intravenöst vankomycin.",
     "Patienten startades på intravenös vancomycin."),
    ("Arterial blood gas shows metabolic acidosis.", "Arteriell blodgas visar metabol acidos.",
     "Arteriell blodgas visar metabolisk acidos."),
    ("The isolate showed resistance to methicillin.", "Isolatet visade resistens mot meticillin.",
     "Isolatet visade motstånd mot methicillin."),
    ("She presents with a 3-day history of fever and cough.", "Hon söker med feber och hosta sedan 3 dagar.",
     "Hon presenterar med en 3-dagars historia av feber och hosta."),
    ("Which of the following is the most likely diagnosis?", "Vilken av följande är den mest sannolika diagnosen?",
     "Vilket av det följande är mest troliga diagnos?"),
    ("The serum potassium is 6.8 mmol/L.", "Serumkalium är 6,8 mmol/l.", "Serum potassium är 6.8 mmol/L."),
    ("An ECG shows ST elevation in leads II, III and aVF.", "EKG visar ST-höjning i avledning II, III och aVF.",
     "Ett EKG visar ST-elevation i ledningar II, III och aVF."),
    ("He has a history of type 2 diabetes mellitus.", "Han har en anamnes på diabetes mellitus typ 2.",
     "Han har en historia av typ 2 diabetes mellitus."),
    ("Acyclovir was administered for suspected herpes encephalitis.",
     "Aciklovir gavs vid misstänkt herpesencefalit.", "Acyclovir administrerades för misstänkt herpes encefalit."),
    ("The child has a petechial rash on the lower extremities.", "Barnet har petekiala utslag på nedre extremiteterna.",
     "Barnet har ett petechialt utslag på de lägre extremiteterna."),
]


def pairwise_prompts(n: int) -> list[str]:
    out = []
    for i in range(n):
        en, good, bad = _SENTENCES[i % len(_SENTENCES)]
        a, b = (good, bad) if (i // len(_SENTENCES)) % 2 == 0 else (bad, good)
        out.append(f"Case {i + 1}.\nEnglish: {en}\nA: {a}\nB: {b}")
    return out


# --------------------------------------------------------------------------- #
# GPU inspection
# --------------------------------------------------------------------------- #


def _nvsmi(query: str, kind: str) -> list[list[str]]:
    out = subprocess.run(
        ["nvidia-smi", f"--query-{kind}={query}", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    ).stdout
    return [[c.strip() for c in line.split(",")] for line in out.strip().splitlines() if line.strip()]


def gpu_memory() -> dict[str, int]:
    try:
        return {idx: int(mem) for idx, mem in _nvsmi("index,memory.used", "gpu")}
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        logging.warning("nvidia-smi memory query failed: %s", exc)
        return {}


def gpu_apps() -> dict[int, list[tuple[str, int]]]:
    """pid -> [(gpu index, MiB)]"""
    try:
        uuid_to_idx = {uuid: idx for idx, uuid in _nvsmi("index,uuid", "gpu")}
        rows = _nvsmi("pid,gpu_uuid,used_memory", "compute-apps")
    except (OSError, subprocess.SubprocessError) as exc:
        logging.warning("nvidia-smi process query failed: %s", exc)
        return {}
    apps: dict[int, list[tuple[str, int]]] = {}
    for pid, uuid, mem in rows:
        if not pid.isdigit():
            continue
        apps.setdefault(int(pid), []).append((uuid_to_idx.get(uuid, uuid), int(mem) if mem.isdigit() else 0))
    return apps


def check_placement(pool, mem_before: dict[str, int]) -> tuple[bool, list[str]]:
    lines, ok = [], True
    apps = gpu_apps()
    mem_after = gpu_memory()
    claimed: set[str] = set()
    for backend in pool.backends.values():
        for rep in backend.replicas:
            want = set(rep.physical_gpus)
            seen: dict[str, int] = {}
            for pid in rep.pids:
                for idx, mib in apps.get(pid, []):
                    seen[idx] = seen.get(idx, 0) + mib
            got = set(seen)
            good = got == want
            ok &= good
            claimed |= got
            lines.append(
                f"  {rep.name}: wanted GPUs {sorted(want)}, engine pids on GPUs "
                f"{ {k: f'{v} MiB' for k, v in sorted(seen.items())} } -> {'OK' if good else 'MISMATCH'}"
            )
    for idx in sorted(mem_after):
        delta = mem_after[idx] - mem_before.get(idx, 0)
        lines.append(f"  GPU {idx}: {mem_before.get(idx, 0):>6} -> {mem_after[idx]:>6} MiB (delta {delta:+})"
                     + ("  [ours]" if idx in claimed else ""))
    # No process of ours on GPUs we did not ask for (e.g. the busy GPU 1).
    ours = {pid for b in pool.backends.values() for r in b.replicas for pid in r.pids} | {os.getpid()}
    stray = {idx for pid, uses in apps.items() if pid in ours for idx, _ in uses} - claimed
    if stray:
        ok = False
        lines.append(f"  STRAY: our processes also on GPUs {sorted(stray)}")
    return ok, lines


# --------------------------------------------------------------------------- #


async def run_batch(pool, reqs):
    t0 = time.monotonic()
    results = await asyncio.gather(*[pool.generate(r) for r in reqs])
    return results, time.monotonic() - t0


def summarise(name: str, results, wall: float) -> dict:
    ok = [r for r in results if r.status == "ok"]
    comp = sum(r.completion_tokens for r in results)
    lat = [r.latency_s for r in results if r.latency_s]
    return {
        "name": name,
        "n": len(results),
        "ok": len(ok),
        "failed": len(results) - len(ok),
        "reasoning_empty": sum(r.reasoning_empty for r in results),
        "finish_reasons": dict(Counter(r.finish_reason for r in results)),
        "wall_s": round(wall, 1),
        "completion_tokens": comp,
        "tok_per_s": round(comp / wall, 1) if wall else None,
        "req_per_s": round(len(results) / wall, 2) if wall else None,
        "latency_median_s": round(statistics.median(lat), 1) if lat else None,
        "reasoning_tokens_median": statistics.median([r.reasoning_tokens for r in results]) if results else None,
        "from_cache": sum(r.from_cache for r in results),
    }


def _dist(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    xs = sorted(values)
    return {
        "n": len(xs),
        "min": xs[0],
        "median": statistics.median(xs),
        "mean": round(statistics.fmean(xs), 1),
        "p90": xs[min(len(xs) - 1, int(0.9 * len(xs)))],
        "max": xs[-1],
    }


def reasoning_stats(results) -> dict:
    """Per-model reasoning statistics over a list of LLMResults."""
    comp = sum(r.completion_tokens for r in results)
    rtok = sum(r.reasoning_tokens for r in results)
    return {
        "n": len(results),
        "ok": sum(r.status == "ok" for r in results),
        "retried": sum(r.attempts > 1 for r in results),  # e.g. Gemma never closed its thought block
        "reasoning_empty": sum(r.reasoning_empty for r in results),
        "reasoning_tokens": _dist([r.reasoning_tokens for r in results]),
        "reasoning_chars": _dist([len(r.reasoning) for r in results]),
        "content_chars": _dist([len(r.content) for r in results]),
        "completion_tokens": _dist([r.completion_tokens for r in results]),
        "reasoning_token_share": round(rtok / comp, 3) if comp else None,
        "finish_reasons": dict(Counter(str(r.finish_reason) for r in results)),
    }


def excerpts(results, k: int = 3, chars: int = 800) -> list[dict]:
    """Up to ``k`` reasoning excerpts (ok results with reasoning first)."""
    ranked = sorted(results, key=lambda r: (r.status != "ok", r.reasoning_empty))
    return [
        {
            "status": r.status,
            "reasoning_tokens": r.reasoning_tokens,
            "reasoning": r.reasoning[:chars] + (" [...]" if len(r.reasoning) > chars else ""),
            "content": r.content[:chars],
        }
        for r in ranked[:k]
    ]


def gemma_leaks(content: str, reasoning: str = "") -> list[str]:
    """Thought-channel leakage in a Gemma result (empty list = clean)."""
    problems = [f"content contains {m!r}" for m in GEMMA_LEAK_MARKERS if m in content]
    if content.lstrip().lower().startswith("thought"):
        problems.append("content starts with 'thought'")
    if reasoning.startswith("thought\n"):
        problems.append("reasoning still starts with the 'thought' label")
    return problems


def parse_models(values: list[str] | None, configured: list[str]) -> list[str]:
    """``--models gptoss,qwen gemma`` -> ['gptoss', 'qwen', 'gemma'] (default: all configured)."""
    if not values:
        return list(configured)
    models = [m.strip() for v in values for m in v.split(",") if m.strip()]
    bad = [m for m in models if m not in ALL_MODELS]
    if bad:
        raise SystemExit(f"--models: unknown model(s) {bad}; expected {list(ALL_MODELS)}")
    missing = [m for m in models if m not in configured]
    if missing:
        raise SystemExit(f"--models: {missing} not configured in the config's models: ({configured})")
    return list(dict.fromkeys(models))


def default_out_dir(run_name: str) -> Path:
    runs = os.environ.get("RUNS_DIR")
    return (Path(runs) if runs else repo_path("refine_runs")) / run_name


async def main_async(args) -> int:
    from swemedmcq.refine.engine import LLMRequest, ModelPool
    from swemedmcq.refine.schemas import SCHEMAS

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    models = parse_models(args.models, list(cfg["models"]))
    out_dir = Path(args.out_dir) if args.out_dir else default_out_dir(args.run_name)
    cache_dir = Path(args.cache_dir) if args.cache_dir else out_dir / "cache"
    if args.fresh and cache_dir.exists():
        shutil.rmtree(cache_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    schemas = {**SCHEMAS, **SMOKE_SCHEMAS}
    gemma_mode = str((cfg["models"].get("gemma") or {}).get("thinking", "native"))
    # Gemma's schema fallback reasons inside the JSON: its schemas start with "analysis".
    schema_mode = {m: (m == "gemma" and gemma_mode == "schema") for m in models}
    short_schema = {m: "smoke_answer_analysis" if schema_mode[m] else "smoke_answer" for m in models}
    tp_schema = {m: "smoke_pairwise_analysis" if schema_mode[m] else "pairwise" for m in models}
    print(f"config={args.config} models={models} out_dir={out_dir} gemma.thinking={gemma_mode}")
    failures: list[str] = []
    warnings: list[str] = []

    mem_before = gpu_memory()
    print(f"GPU memory before load (MiB): {mem_before}")
    t0 = time.monotonic()
    pool = await ModelPool.create(cfg, schemas, cache_dir, models=models)
    print(f"engines loaded in {time.monotonic() - t0:.0f}s")
    try:
        # 1. rendering ---------------------------------------------------------
        if "gptoss" in pool.backends:
            b = pool.backends["gptoss"]
            t1, t2 = b.render_text("S", "U"), b.render_text("S", "U")
            date = cfg["models"]["gptoss"].get("fixed_date")
            date_line = next((ln for ln in t1.splitlines() if ln.startswith("Current date:")), None)
            print(f"[render] gptoss date line: {date_line!r}; reasoning line: "
                  f"{next((ln for ln in t1.splitlines() if ln.startswith('Reasoning:')), None)!r}")
            if t1 != t2 or date_line != f"Current date: {date}":
                failures.append("gptoss rendering not reproducible / fixed date missing")
        if "qwen" in pool.backends:
            qb = pool.backends["qwen"]
            q = qb.render_text("S", "U")
            print(f"[render] qwen reasoning_effort={qb.reasoning_effort!r}; prompt tail: {q[-60:]!r}")
            if not q.rstrip().endswith("<think>"):
                failures.append("qwen prompt does not open a <think> block")
        if "gemma" in pool.backends:
            gb = pool.backends["gemma"]
            g = gb.render_text("S", "U")
            ids = gb.render("S", "U")
            print(f"[render] gemma thinking={gb.thinking}; prompt head: {g[:60]!r}; tail: {g[-60:]!r}; "
                  f"last ids: {ids[-6:]}")
            if gb.thinking == "native":
                if ids[-3:] != GEMMA_PREFILL_IDS or not g.endswith("<|turn>model\n<|channel>thought\n"):
                    failures.append(f"gemma prompt does not end with the thought prefill (ids {ids[-3:]})")
                if "<|think|>" not in g:
                    failures.append("gemma prompt has no <|think|> in the system turn (enable_thinking ignored?)")
            elif not g.endswith("<|channel>thought\n<channel|>"):
                failures.append("gemma schema mode: template did not close an empty thought block")

        # 2. placement -----------------------------------------------------------
        if args.placement != "skip":
            placed, lines = check_placement(pool, mem_before)
            print("[gpus] placement:")
            print("\n".join(lines))
            if not placed:
                (failures if args.placement == "check" else warnings).append("engines not on the configured GPUs")

        # 3. short prompts -----------------------------------------------------------
        short = {
            m: [LLMRequest(model=m, system=SYSTEM, user=p, schema_name=short_schema[m],
                           seed_key=["smoke", "short", str(i)]) for i, p in enumerate(SHORT_PROMPTS[: args.n_short])]
            for m in models
        }
        print(f"\n[short] {args.n_short} prompts per model")
        all_short = [r for m in models for r in short[m]]
        short_res, wall = await run_batch(pool, all_short)
        for req, res in zip(all_short, short_res):
            print(f"  {req.model:6s} {res.status:6s} used={res.model_used:6s} att={res.attempts} "
                  f"finish={res.finish_reason} reasoning={len(res.reasoning):5d} ch/{res.reasoning_tokens:5d} tok "
                  f"content={len(res.content):4d} ch latency={res.latency_s:6.1f}s parsed={res.parsed}")
            if res.status != "ok" or res.parsed is None:
                failures.append(f"short {req.model} failed: {res.error}")
            if res.reasoning_empty:
                failures.append(f"short {req.model}: EMPTY reasoning")
            if res.model_used != req.model:
                failures.append(f"short {req.model}: needed fallback")
            if req.model == "gemma":
                for problem in gemma_leaks(res.content, res.reasoning):
                    failures.append(f"short gemma: {problem}")
        with (out_dir / "reasoning_samples.jsonl").open("w", encoding="utf-8") as fh:
            for req, res in zip(all_short, short_res):
                fh.write(json.dumps({
                    "model": req.model, "model_used": res.model_used, "status": res.status,
                    "schema": req.schema_name, "user": req.user, "reasoning_tokens": res.reasoning_tokens,
                    "reasoning": res.reasoning, "content": res.content, "error": res.error,
                }, ensure_ascii=False) + "\n")
        if args.show_example:
            for m in models:
                r = short_res[[q.model for q in all_short].index(m)]
                print(f"\n--- {m} example reasoning (first 600 chars) ---\n{r.reasoning[:600]}\n--- content ---\n{r.content}")

        # 4. throughput -----------------------------------------------------------------
        summaries = []
        tp_reqs: list = []
        tp_results: dict[str, list] = {m: [] for m in models}
        if args.n_throughput > 0:
            prompts = pairwise_prompts(args.n_throughput)
            tp_reqs = [
                LLMRequest(model=m, system=PAIRWISE_SYSTEM, user=p, schema_name=tp_schema[m],
                           seed_key=["smoke", "tp", str(i)], max_tokens=args.tp_max_tokens)
                for m in models for i, p in enumerate(prompts)
            ]
            print(f"\n[throughput] {args.n_throughput} concurrent 'pairwise' requests per model, all models at once")

            async def per_model(m):
                rs = [r for r in tp_reqs if r.model == m]
                res, w = await run_batch(pool, rs)
                return m, res, w

            t_all = time.monotonic()
            outs = await asyncio.gather(*[per_model(m) for m in models])
            wall_all = time.monotonic() - t_all
            for m, res, w in outs:
                tp_results[m] = res
                s = summarise(m, res, w)
                s["fallbacks"] = sum(r.model_used != m for r in res)
                summaries.append(s)
                print(f"  {m}: {json.dumps(s)}")
                prefs = Counter((r.parsed or {}).get("preferred") for r in res)
                print(f"  {m}: preferred distribution {dict(prefs)}")
                if s["ok"] < 0.95 * s["n"]:
                    failures.append(f"throughput {m}: only {s['ok']}/{s['n']} ok")
                if s["reasoning_empty"]:
                    failures.append(f"throughput {m}: {s['reasoning_empty']} empty reasoning")
                if m == "gemma":
                    leaky = sum(bool(gemma_leaks(r.content, r.reasoning)) for r in res if r.status == "ok")
                    if leaky:
                        failures.append(f"throughput gemma: {leaky} results with thought-channel leakage")
            print(f"  total wall {wall_all:.1f}s")
        print(f"\n[stats] {json.dumps(pool.stats(), indent=1)}")

        # 5. rerun from cache -------------------------------------------------------------
        gpu_before = pool.counters["gpu_attempts"]
        rerun, wall = await run_batch(pool, all_short + tp_reqs)
        hits = sum(r.from_cache for r in rerun)
        new_gpu = pool.counters["gpu_attempts"] - gpu_before
        same = all(a.parsed == b.parsed for a, b in zip(short_res, rerun[: len(short_res)]))
        print(f"\n[cache] rerun of {len(rerun)} requests: {hits} from cache, {new_gpu} GPU attempts, "
              f"{wall:.2f}s, short results identical={same}")
        failed_first = sum(1 for r in rerun if r.status != "ok")
        if new_gpu != 0 or hits != len(rerun):
            # Requests that failed the first time are retried by design (retry_failed=True).
            if hits + failed_first < len(rerun):
                failures.append(f"cache replay incomplete: {hits}/{len(rerun)} hits, {new_gpu} GPU attempts")
            else:
                print(f"  note: {failed_first} previously failed requests were retried (retry_failed=True)")
        if not same:
            failures.append("cached results differ from the originals")

        # 6. parent process must not hold a CUDA context / GPU memory --------------------
        torch = sys.modules.get("torch")
        cuda_init = bool(torch is not None and torch.cuda.is_initialized())
        parent_on_gpu = gpu_apps().get(os.getpid(), [])
        print(f"[cuda] parent: torch.cuda.is_initialized()={cuda_init} (driver init by vLLM's FLA import is "
              f"expected with Qwen3.5); parent GPU contexts: {parent_on_gpu or 'none'}")
        if parent_on_gpu:
            failures.append(f"parent process holds GPU memory: {parent_on_gpu}")

        per_model = {}
        for m in models:
            b = pool.backends[m]
            mine_short = [r for q, r in zip(all_short, short_res) if q.model == m]
            per_model[m] = {
                "model_id": b.model_id,
                "render_params": b.render_params(),
                "replicas": [r.gpus for r in b.replicas],
                "short": reasoning_stats(mine_short),
                "throughput": reasoning_stats(tp_results[m]) if tp_results[m] else None,
                "reasoning_excerpts": excerpts(mine_short + tp_results[m]),
            }
        summary = {
            "config": str(args.config),
            "run_name": args.run_name,
            "models": models,
            "gemma_thinking": gemma_mode if "gemma" in models else None,
            "result": "PASS" if not failures else "FAIL",
            "failures": failures,
            "warnings": warnings,
            "per_model": per_model,
            "throughput": summaries,
            "stats": pool.stats(),
            "gpu_memory_after_load_mib": gpu_memory(),
        }
        out = out_dir / "summary.json"
        out.write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"summary -> {out}")
    finally:
        await pool.close()

    print("\nRESULT:", "PASS" if not failures else "FAIL")
    for f in failures:
        print("  -", f)
    for w in warnings:
        print("  (warning)", w)
    return 0 if not failures else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(CONFIG_DIR / "refine" / "default.yaml"))
    ap.add_argument("--run-name", default="engine_smoke",
                    help="output directory name under $RUNS_DIR (or refine/runs)")
    ap.add_argument("--out-dir", help="output directory (overrides --run-name)")
    ap.add_argument("--cache-dir", help="LLM cache directory (default <out-dir>/cache)")
    ap.add_argument("--fresh", action="store_true", help="delete the cache first")
    ap.add_argument("--models", nargs="*", metavar="MODELS",
                    help="comma- or space-separated subset of gptoss,qwen,gemma (default: all configured)")
    ap.add_argument("--placement", choices=["check", "warn", "skip"], default="check",
                    help="GPU placement check via nvidia-smi: fail, only warn, or skip")
    ap.add_argument("--n-short", type=int, default=5)
    ap.add_argument("--n-throughput", type=int, default=200)
    ap.add_argument("--tp-max-tokens", type=int, default=4096)
    ap.add_argument("--show-example", action="store_true")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
