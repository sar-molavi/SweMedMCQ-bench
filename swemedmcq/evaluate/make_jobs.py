"""Write the job lists (one job id per line) and slurm array files.

    python -m swemedmcq.evaluate.make_jobs                 # all jobs
    python -m swemedmcq.evaluate.make_jobs --only 'euro-'  # regex on model name

Outputs (<results_dir>/jobs/):
  all.txt               every job; run with swemedmcq.evaluate.run_local
  <n>gpu.txt            the jobs that need n GPUs (models.yaml `gpus`)
  <n>gpu.sbatch         one slurm array per GPU count (task i runs line i of <n>gpu.txt,
                        generation only; edit the #SBATCH lines and submit yourself)
Nothing is submitted or started here. Scoring needs no GPU: after generation, run
``python -m swemedmcq.evaluate.run <job> --stage score`` for each line.
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict

from swemedmcq import paths
from swemedmcq.evaluate.registry import jobs

SBATCH = """#!/usr/bin/env bash
#SBATCH --job-name=swemedmcq_eval_{n}gpu
#SBATCH --ntasks=1
#SBATCH --time={time}
#SBATCH --account=<ACCOUNT>
#SBATCH --partition=<GPU_PARTITION>
#SBATCH --gpus={n}
#SBATCH --cpus-per-gpu=16
#SBATCH --mem-per-gpu=64G
#SBATCH --array=1-{count}
#SBATCH --output=logs/eval_{n}gpu_%A_%a.log

# Task i runs line i of {n}gpu.txt (generation only). Re-submitting is safe: items that
# already have a result are skipped. Run from the repository root (<CLUSTER_DIR>), with
# the inference environment activated (or ENV_SETUP=<file to source>).
set -euo pipefail
cd "${{PROJECT:-<CLUSTER_DIR>}}"
JOB="$(sed -n "${{SLURM_ARRAY_TASK_ID:?}}p" "{jobs_dir}/{n}gpu.txt")"
[[ -n "$JOB" ]] || {{ echo "No job on line $SLURM_ARRAY_TASK_ID" >&2; exit 1; }}
exec bash scripts/slurm/eval_job.sh "$JOB"
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", help="Regex on the model name.")
    ap.add_argument("--time", default="24:00:00", help="Slurm time limit per array task.")
    args = ap.parse_args()

    out = paths.path("results_dir") / "jobs"
    out.mkdir(parents=True, exist_ok=True)
    selected = [j for j in jobs() if not args.only or re.search(args.only, j.model.name)]
    # Small test sets first, so the delta runs finish early and give a quick check.
    selected.sort(key=lambda j: (j.set != "sv_L4_delta", j.model.part, j.model.name, j.set))

    (out / "all.txt").write_text("".join(f"{j.id}\n" for j in selected), encoding="utf-8")
    by_gpus: dict[int, list[str]] = defaultdict(list)
    for j in selected:
        by_gpus[j.model.gpus].append(j.id)
    for path in list(out.glob("*gpu.txt")) + list(out.glob("*gpu.sbatch")):
        path.unlink()
    try:
        jobs_dir = out.relative_to(paths.ROOT)
    except ValueError:
        jobs_dir = out
    for n, ids in sorted(by_gpus.items()):
        (out / f"{n}gpu.txt").write_text("".join(f"{j}\n" for j in ids), encoding="utf-8")
        (out / f"{n}gpu.sbatch").write_text(
            SBATCH.format(n=n, count=len(ids), time=args.time, jobs_dir=jobs_dir), encoding="utf-8")
    print(f"all: {len(selected)} jobs -> {out / 'all.txt'}")
    for n, ids in sorted(by_gpus.items()):
        print(f"{n} GPU: {len(ids)} jobs -> {out / f'{n}gpu.txt'} (+ .sbatch)")


if __name__ == "__main__":
    main()
