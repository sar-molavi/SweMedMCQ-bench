"""Smoke test: run a few items of a job with its real command, then decode them.

Used to check that a model starts and answers (e.g. in vLLM's compiled mode) before
a long run. Results go to --out, never to the results folder.

    CUDA_VISIBLE_DEVICES=0,1,2,3 python -m swemedmcq.evaluate.smoke sv_L4a/Qwen3.8-27B \
        --out DIR --n 4 --max-len 40960

Tensor parallel = number of visible GPUs. --max-len only caps the context so the KV
cache fits on smaller GPUs; prompt + 30,720 output tokens fit in 40,960.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import pandas as pd

from swemedmcq.evaluate.registry import parse_job
from swemedmcq.evaluate.run import DECODER, PYTHON, generate_cmd


def spread_ids(job, n: int) -> list[str]:
    """n prompt ids spread evenly over the file (so several datasets are covered)."""
    with job.prompts().open(encoding="utf-8") as handle:
        ids = [json.loads(line)["prompt_id"] for line in handle if line.strip()]
    step = max(1, len(ids) // n)
    return ids[::step][:n]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("job")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=None)
    ap.add_argument("--ids", nargs="+", help="These prompt ids instead of --n spread ones.")
    ap.add_argument("--eager", action="store_true", help="Eager mode instead of the job's setting (comparison).")
    ap.add_argument("--tag", default="", help="Suffix of the output folder name (e.g. -eager).")
    args = ap.parse_args()
    job = parse_job(args.job)
    root = args.out / job.set.replace("/", "_")
    root = root.with_name(root.name + args.tag)
    raw = root / f"{job.lang}_response" / job.model.name
    raw.mkdir(parents=True, exist_ok=True)
    ids = root / f"{job.model.name}.ids"
    ids.write_text("\n".join(args.ids or spread_ids(job, args.n)) + "\n", encoding="utf-8")

    cmd = generate_cmd(job, dry=False)
    cmd[cmd.index("--output-dir") + 1] = str(raw)
    cmd += ["--prompt-ids-file", str(ids)]
    if args.eager:
        cmd = [("--eager" if c == "--no-eager" else c) for c in cmd]
    if args.max_len:
        if "--max-model-len" in cmd:
            cmd[cmd.index("--max-model-len") + 1] = str(args.max_len)
        else:
            cmd += ["--max-model-len", str(args.max_len)]
    log = root / f"{job.model.name}.log"
    print("+", " ".join(cmd), flush=True)
    start = time.time()
    with log.open("w", encoding="utf-8") as handle:
        rc = subprocess.run(cmd, stdout=handle, stderr=subprocess.STDOUT).returncode
    wall = time.time() - start
    if rc:
        raise SystemExit(f"{job.id}: generation failed ({rc}); see {log}")

    cache = root / "decoded"
    subprocess.run([PYTHON, *DECODER, "--input-root", str(root), "--tag", "smoke", "--cache-dir", str(cache),
                    "--gold", str(job.answer_split()),
                    "--langs", job.lang, "--only", f"^{job.model.name}$", "--workers", "1", "--no-merge", "--force"],
                   check=True, stdout=subprocess.DEVNULL)
    files = [f for f in (cache / "smoke").glob(f"{job.model.name}__{job.lang}*.parquet") if ".text." not in f.name]
    df = pd.concat(pd.read_parquet(f) for f in files)
    df = df[df.status != "missing"]
    summary = {
        "job": job.id, "items": len(df), "wall_s": round(wall), "status": df.status.value_counts().to_dict(),
        "finish": df.finish_reason.value_counts().to_dict(), "correct": int(df.is_correct.sum()),
        "gen_tokens": df.n_gen_tokens.tolist(), "compiled": "--no-eager" in cmd,
    }
    (root / f"{job.model.name}.summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
