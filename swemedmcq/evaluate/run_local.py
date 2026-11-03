"""Run a job list on this machine, several jobs in parallel on the given GPUs.

    python -m swemedmcq.evaluate.run_local --gpus 0,1,2,3
    python -m swemedmcq.evaluate.run_local --gpus 0,1 --jobs runs/eval/jobs/1gpu.txt --only euro-

Each job gets as many GPUs as its `gpus` field asks for; finished jobs (DONE
marker) are skipped, so the command can be re-run after an interruption. Before
starting a job the GPUs are checked with nvidia-smi and skipped while busy
(another process). Per-job logs: <results_dir>/logs/<set>/<name>.log; this queue's
own log is printed to stdout.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from swemedmcq import paths
from swemedmcq.evaluate.registry import parse_job

BUSY_MIB = 2000


def busy_gpus() -> set[str]:
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=True).stdout
    return {line.split(",")[0].strip() for line in out.splitlines() if int(line.split(",")[1]) > BUSY_MIB}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpus", required=True, help="Comma-separated GPU ids this queue may use.")
    ap.add_argument("--jobs", type=Path, default=paths.path("results_dir") / "jobs/all.txt")
    ap.add_argument("--only", help="Regex on the job id.")
    args = ap.parse_args()

    pending = [parse_job(line.strip()) for line in args.jobs.read_text().splitlines() if line.strip()]
    pending = [j for j in pending if (not args.only or re.search(args.only, j.id)) and not j.done_marker.exists()]
    free = [g.strip() for g in args.gpus.split(",")]
    running: list[tuple[subprocess.Popen, list[str], str]] = []
    failed: list[str] = []
    print(f"{len(pending)} jobs to run on GPUs {free}", flush=True)
    while pending or running:
        for proc, gpus, job_id in list(running):
            if proc.poll() is not None:
                running.remove((proc, gpus, job_id))
                free.extend(gpus)
                status = "ok" if proc.returncode == 0 else f"FAILED ({proc.returncode})"
                if proc.returncode:
                    failed.append(job_id)
                print(f"[{time.strftime('%H:%M:%S')}] {job_id}: {status}", flush=True)
        busy = busy_gpus()
        for job in list(pending):
            usable = [g for g in free if g not in busy]
            if len(usable) < job.model.gpus:
                continue
            gpus = usable[: job.model.gpus]
            for g in gpus:
                free.remove(g)
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": ",".join(gpus)}
            job.log.parent.mkdir(parents=True, exist_ok=True)
            proc = subprocess.Popen([sys.executable, "-m", "swemedmcq.evaluate.run", job.id], cwd=paths.ROOT, env=env,
                                    stdout=job.log.open("a"), stderr=subprocess.STDOUT)
            running.append((proc, gpus, job.id))
            pending.remove(job)
            print(f"[{time.strftime('%H:%M:%S')}] {job.id}: started on GPU {','.join(gpus)}", flush=True)
            busy = busy_gpus() | set(gpus)
        time.sleep(20)
    print(f"finished; {len(failed)} failed: {failed}", flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
