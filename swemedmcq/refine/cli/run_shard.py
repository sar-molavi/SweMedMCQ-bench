"""Step 2 (GPU, one job per shard): run the whole refinement loop for one shard.

    python -m swemedmcq.refine.cli.run_shard --run-dir runs/refine/full --shard 3 --config configs/refine/with_gemma.yaml

Both models are loaded once (ModelPool, one AsyncLLMEngine per replica), and up
to `max_items_in_flight` items run their loops concurrently, so every GPU stays
busy without waiting for global steps.

Resumable: every finished item has `shards/<k>/results/<id>.json` (atomic);
those are skipped.  Unfinished items are re-run from the start; their finished
LLM calls are replayed from the request cache (`<run>/cache`, no GPU work), so a
restart continues where it stopped and gives identical results.  When all items
have a result, `shards/<k>/DONE` is written.  Programming errors are written to
`shards/<k>/errors/<id>.json` and the item is retried next time.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
import traceback
from pathlib import Path

import yaml

from ..engine import ModelPool
from ..io import atomic_write_json, read_jsonl, resources_dir
from ..loop.item_loop import Context, ItemLoop, LoopConfig
from ..schemas import SCHEMAS
from ..termbase import Lexicon, TermBase
from ..types import Item

def load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


async def run(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    shard_dir = args.run_dir / "shards" / f"{args.shard:03d}"
    results = shard_dir / "results"
    errors = shard_dir / "errors"
    results.mkdir(parents=True, exist_ok=True)
    errors.mkdir(parents=True, exist_ok=True)
    items = [Item.model_validate(r) for r in read_jsonl(args.run_dir / "prep" / f"shard_{args.shard:03d}.jsonl")]
    if args.limit:
        items = items[: args.limit]
    todo = []
    for it in items:
        path = results / f"{it.prompt_id}.json"
        if path.exists() and not (args.retry_failed and json.loads(path.read_text()).get("status") == "failed"):
            continue
        todo.append(it)
    print(f"shard {args.shard}: {len(items)} items, {len(items) - len(todo)} done, {len(todo)} to run", flush=True)
    if not todo:
        if len(items) == len(list(results.glob("*.json"))):
            (shard_dir / "DONE").write_text("done\n")
        return 0

    resources = resources_dir(cfg)
    termbase, lexicon = TermBase.load(resources), Lexicon.load(resources)
    loop_cfg = LoopConfig.from_dict(cfg.get("loop"))
    pool = await ModelPool.create(cfg, SCHEMAS, args.run_dir / "cache")
    ctx = Context(pool=pool, termbase=termbase, lexicon=lexicon, cfg=loop_cfg)
    sem = asyncio.Semaphore(cfg.get("max_items_in_flight", 512))
    done = failed = 0
    t0 = time.time()

    async def one(it: Item) -> None:
        nonlocal done, failed
        async with sem:
            try:
                rec = await ItemLoop(it, ctx).run()
            except Exception:  # programming error: keep the item pending, record the traceback
                failed += 1
                atomic_write_json(errors / f"{it.prompt_id}.json", {"traceback": traceback.format_exc()})
                return
            atomic_write_json(results / f"{it.prompt_id}.json", rec)
            (errors / f"{it.prompt_id}.json").unlink(missing_ok=True)
            done += 1
            if done % 50 == 0:
                rate = done / max(1e-9, time.time() - t0) * 3600
                print(f"[{time.strftime('%H:%M:%S')}] done {done}/{len(todo)} ({rate:.0f} items/h), "
                      f"errors {failed}; pool {json.dumps(pool.stats())}", flush=True)

    try:
        await asyncio.gather(*(one(it) for it in todo))
    finally:
        atomic_write_json(shard_dir / "pool_stats.json", pool.stats())
        await pool.close()
    n_results = len(list(results.glob("*.json")))
    print(f"shard {args.shard}: finished {done}, errors {failed}, results {n_results}/{len(items)}", flush=True)
    if n_results == len(items) and failed == 0:
        (shard_dir / "DONE").write_text("done\n")
        return 0
    return 1


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--no-retry-failed", dest="retry_failed", action="store_false",
                    help="Do not re-run items whose result has status 'failed'.")
    args = ap.parse_args(argv)
    raise SystemExit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
