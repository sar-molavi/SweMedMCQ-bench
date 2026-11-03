"""Run one job: generate with vLLM, then score. Resumable at every step.

    python -m swemedmcq.evaluate.run sv_L4a/lfm-trace-ensv-reasoning
    python -m swemedmcq.evaluate.run sv_L4a/gpt-oss-20b --stage score
    python -m swemedmcq.evaluate.run sv_L4a/gpt-oss-20b --dry-run

GPUs: the runner uses every visible GPU (tensor parallel), so set
CUDA_VISIBLE_DEVICES to the job's `gpus` count (run_local.py does this).

Generation skips items that already have a result pickle. The generation settings
are written to ``raw/<set>/<name>/settings.json``; a run whose settings differ
from that file stops instead of mixing results. The exception is eager vs compiled
mode: an unfinished run continues in the current mode, and the file records which
results were made in which mode. Scoring runs only when
all items have a result (``--allow-incomplete`` to override) and writes
``scored/<set>/<name>/DONE`` at the end; a job with DONE is skipped entirely.

Scoring
  benchmark  swemedmcq.evaluate.decode_benchmark (answer extraction with the shared
             response parser and schema validation; both languages parsed the same way)
  students   swemedmcq.evaluate.{trace,answer_level}.extract_eval_predictions + eval
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

from swemedmcq import paths
from swemedmcq.evaluate.registry import Job, hf_overrides_arg, parse_job

PYTHON = os.environ.get("SWEMEDMCQ_PYTHON", sys.executable)
DECODER = ["-m", "swemedmcq.evaluate.decode_benchmark"]


def run(cmd: list[str], log: Path | None, dry: bool) -> None:
    print("+ " + shlex.join(cmd), flush=True)
    if dry:
        return
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    if log is None:
        subprocess.run(cmd, cwd=paths.ROOT, env=env, check=True)
        return
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as handle:
        proc = subprocess.run(cmd, cwd=paths.ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
    if proc.returncode:
        raise SystemExit(f"Command failed ({proc.returncode}); see {log}")


def count_results(job: Job) -> int:
    return len(list(job.raw_dir.glob("result_*.pkl"))) if job.raw_dir.exists() else 0


# ---------------------------------------------------------------- generation
def token_set(job: Job, recipe: str, dry: bool) -> Path:
    date = job.model.date
    name = f"{recipe}_{date}_{job.set}.pkl" if date else f"{recipe}_{job.set}.pkl"
    path = paths.path("results_dir") / "token_sets" / name
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        run([PYTHON, "-m", "swemedmcq.evaluate.no_reasoning_tokens", "--recipe", recipe,
             "--input", str(job.prompts()), "--output", str(path), "--show", "1",
             *(["--date", date] if date else [])], None, dry)
    return path


def generate_cmd(job: Job, dry: bool) -> list[str]:
    m = job.model
    inputs = job.prompts()
    flags: list[str] = []
    if m.mode == "no_think":
        flags += ["--no-reasoning"]
    elif m.mode.startswith("token_set:"):
        inputs = token_set(job, m.mode.split(":", 1)[1], dry)
    elif m.mode == "reasoning":
        flags += ["--structured-reasoning-parser", m.parser]
    elif m.mode == "forced_json":
        flags += ["--no-reasoning", "--force-structured-output"]
    elif m.mode != "default":
        raise ValueError(f"{m.name}: unknown mode {m.mode!r}")
    flags += ["--eager" if m.eager else "--no-eager"]
    if m.max_len:
        flags += ["--max-model-len", str(m.max_len)]
    if m.prefix:
        flags += ["--assistant-prefix", m.prefix]
    return [
        PYTHON, "-m", job.module(),
        "--input-path", str(inputs),
        "--output-dir", str(job.raw_dir),
        "--model-name", m.model_path,
        "--tokenizer-path", m.tokenizer_path,
        *(["--revision", m.revision] if m.revision else []),
        "--temperature", "0",
        "--max-output-tokens", str(m.max_out),
        "--gpu-memory-utilization", os.environ.get("GPU_MEMORY_UTILIZATION", "0.90"),
        *flags, *hf_overrides_arg(m),
    ]


def settings(job: Job) -> dict:
    """Settings that change the generations; results made with other settings are not mixed."""
    m = job.model
    return {"model": m.model, "mode": m.mode, "max_out": m.max_out, "max_len": m.max_len,
            "prefix": m.prefix, "hf_overrides": m.hf_overrides, "date": m.date, "revision": m.revision}


def check_settings(job: Job, have: int, dry: bool) -> None:
    """Record the settings in raw/<set>/<name>/settings.json and refuse to mix other ones.

    Eager vs compiled mode is the one setting a run may change: an unfinished run continues
    in the current mode, and ``segments`` records from which result on each mode was used,
    with the ids generated before the switch in ``items_before_segment_<k>.txt``. Results
    found without a settings.json are recorded as eager.
    """
    path = job.raw_dir / "settings.json"
    want = settings(job)
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        segments = old.get("segments") or [{"eager": old.get("eager", True), "from_results": 0}]
        diff = {k: (old.get(k), v) for k, v in want.items() if old.get(k) != v}
        if diff:
            raise SystemExit(f"{job.id}: existing results were generated with other settings "
                             f"(old, new): {diff}. Move {job.raw_dir} aside to start over.")
    elif have:
        segments = [{"eager": True, "from_results": 0, "note": "made before settings were recorded"}]
    else:
        segments = []
    if not segments or (segments[-1]["eager"] != job.model.eager and have < job.expected()):
        segments.append({"eager": job.model.eager, "from_results": have, "date": time.strftime("%Y-%m-%d")})
        if have and not dry:
            ids = sorted(f.name.removeprefix("result_").removesuffix(".pkl") for f in job.raw_dir.glob("result_*.pkl"))
            (job.raw_dir / f"items_before_segment_{len(segments) - 1}.txt").write_text("\n".join(ids) + "\n", encoding="utf-8")
    if not dry:
        job.raw_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({**want, "segments": segments}, indent=2) + "\n", encoding="utf-8")
    if len(segments) > 1:
        print(f"{job.id}: modes by segment: {[(s['from_results'], 'eager' if s['eager'] else 'compiled') for s in segments]}")


def generate(job: Job, dry: bool) -> None:
    expected, have = job.expected(), count_results(job)
    check_settings(job, have, dry)
    if have >= expected:
        print(f"{job.id}: {have}/{expected} results, generation done")
        return
    if job.model.model_path.startswith("/") and not Path(job.model.model_path).exists():
        raise SystemExit(f"{job.id}: checkpoint not found: {job.model.model_path}")
    print(f"{job.id}: {have}/{expected} results, generating")
    run(generate_cmd(job, dry), job.log, dry)


# ---------------------------------------------------------------- scoring
def score_benchmark(job: Job, dry: bool) -> None:
    # The decoder expects <root>/<lang>_response/<model_dir>/; build that view with a symlink.
    view = paths.path("results_dir") / "decode_view" / job.set / f"{job.lang}_response" / job.model.name
    if not dry:
        view.parent.mkdir(parents=True, exist_ok=True)
        if not view.is_symlink():
            view.symlink_to(job.raw_dir.resolve())
    run([PYTHON, *DECODER, "--input-root", str(view.parent.parent), "--tag", job.set,
         "--gold", str(job.answer_split()),
         "--cache-dir", str(job.scored_dir.parent.parent / "bench_items"),
         "--langs", job.lang, "--only", f"^{job.model.name}$", "--workers", "1", "--no-merge", "--force"],
        job.log, dry)


def score_student(job: Job, dry: bool) -> None:
    m = job.model
    if m.prompt == "trace":
        source = paths.path("results_dir") / "sources" / f"{job.set}_trace.jsonl"
        if not source.exists() and not dry:
            from swemedmcq.evaluate.student_sources import load_rows_by_prompt_id, write_source

            fallback = load_rows_by_prompt_id(job.answer_split())
            write_source(prompt_path=job.prompts(), output_path=source, language=job.lang, fallback_rows=fallback)
        extract, evaluate = "swemedmcq.evaluate.trace.extract_eval_predictions", "swemedmcq.evaluate.trace.eval"
    else:
        source = job.answer_split()
        extract = "swemedmcq.evaluate.answer_level.extract_eval_predictions"
        evaluate = "swemedmcq.evaluate.answer_level.eval"
    out = job.scored_dir
    run([PYTHON, "-m", extract, "--input-dir", str(job.raw_dir), "--source-jsonl", str(source),
         "--tokenizer-path", m.tokenizer_path, "--output-path", str(out / "extracted.jsonl")], job.log, dry)
    run([PYTHON, "-m", evaluate, str(out / "extracted.jsonl"), "--output-dir", str(out / "scored")], job.log, dry)


def score(job: Job, dry: bool, allow_incomplete: bool) -> None:
    expected, have = job.expected(), count_results(job)
    if have < expected and not allow_incomplete and not dry:
        raise SystemExit(f"{job.id}: only {have}/{expected} results; not scoring (--allow-incomplete to force)")
    if not dry:
        job.scored_dir.mkdir(parents=True, exist_ok=True)
    (score_benchmark if job.model.part == "benchmark" else score_student)(job, dry)
    if not dry:
        job.done_marker.write_text(f"{have}/{expected}\n", encoding="utf-8")
    print(f"{job.id}: scored")


def acquire_lock(job: Job) -> Path | None:
    """One process per job: a second queue that reaches a running job skips it."""
    lock = paths.path("results_dir") / "locks" / f"{job.set}__{job.model.name}.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    if lock.exists():
        try:
            os.kill(int(lock.read_text().strip()), 0)
            return None  # the owner is alive
        except (ValueError, ProcessLookupError, PermissionError):
            lock.unlink(missing_ok=True)  # stale lock
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return None
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    return lock


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("job", help="<set>/<model name>, e.g. sv_L4a/gpt-oss-20b")
    ap.add_argument("--stage", choices=["all", "generate", "score"], default="all")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--allow-incomplete", action="store_true")
    ap.add_argument("--force", action="store_true", help="Ignore the DONE marker.")
    args = ap.parse_args()
    job = parse_job(args.job)
    if job.done_marker.exists() and not args.force and not args.dry_run:
        print(f"{job.id}: already done ({job.done_marker})")
        return
    lock = None if args.dry_run else acquire_lock(job)
    if lock is None and not args.dry_run:
        print(f"{job.id}: already running in another process; skipped")
        return
    try:
        if args.stage in ("all", "generate"):
            generate(job, args.dry_run)
        if args.stage in ("all", "score"):
            score(job, args.dry_run, args.allow_incomplete)
    finally:
        if lock is not None:
            lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
