"""CPU-only dry run of a student training config.

Loads the config with the training loader (bilingual configs with the bilingual
loader), checks every path the job reads, verifies the preprocessed data (row
counts against the preprocessing manifest, sequence lengths, labels, token ids;
only the two teachers' runs in the trace data) and prints the expected schedule. With
``--compare OTHER.json`` it also lists the fields that differ from another
config. Exits non-zero on any error.

It does not import unsloth or torch, so it runs on a login node:

    python -m swemedmcq.train.check_config -c configs/train/SweMedMCQ-EuroLLM-1.7B-Trace-EN.json
    python -m swemedmcq.train.check_config -c configs/train/SweMedMCQ-EuroLLM-1.7B-Trace-EN.json \\
        --compare configs/train/SweMedMCQ-LFM2.5-1.2B-Trace-EN.json --skip-data
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import sys
from pathlib import Path

from swemedmcq.paths import ROOT, CONFIG_DIR
from swemedmcq.train.bilingual.config import BilingualTrainingConfig
from swemedmcq.train.config import TrainingConfig

DEFAULT_ACCELERATE = CONFIG_DIR / "train" / "accelerate_deepspeed_zero2_4gpu.yaml"
TEACHER_RUNS = ("gpt_oss_120b", "qwen_3.6_35B_A3B")  # source runs of the two teachers

errors: list[str] = []
warnings: list[str] = []


def ok(msg: str) -> None:
    print(f"  ok    {msg}")


def warn(msg: str) -> None:
    warnings.append(msg)
    print(f"  WARN  {msg}")


def fail(msg: str) -> None:
    errors.append(msg)
    print(f"  FAIL  {msg}")


def check(cond: bool, msg: str) -> bool:
    (ok if cond else fail)(msg)
    return cond


def fmt(value) -> str:
    text = json.dumps(value) if not isinstance(value, str) else value
    return text if len(text) <= 70 else text[:67] + "..."


def load_raw(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_config(path: Path):
    raw = load_raw(path)
    if "train_dataset_paths" in raw:
        return BilingualTrainingConfig.from_json(path)
    return TrainingConfig.from_json(str(path))


def training_dict(cfg) -> dict:
    return dataclasses.asdict(cfg.training if isinstance(cfg, BilingualTrainingConfig) else cfg)


def load_yaml_flat(path: Path) -> dict:
    """Minimal parser for the two-level accelerate yaml files (avoids a yaml dependency)."""
    out: dict = {}
    section = None
    for line in path.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        key, _, value = line.strip().partition(":")
        value = value.strip().strip("'\"")
        if line.startswith((" ", "\t")) and section:
            out[f"{section}.{key}"] = value
        elif value == "":
            section = key
        else:
            section = None
            out[key] = value
    return out


def check_dataset(name: str, path: Path, expected_rows: int | None, vocab_size: int | None, max_len: int | None):
    import pyarrow.compute as pc
    from datasets import load_from_disk

    ds = load_from_disk(str(path))  # memory-mapped; the trainer uses keep_in_memory=True
    if expected_rows is not None:
        check(len(ds) == expected_rows, f"{name}: {len(ds):,} rows (manifest: {expected_rows:,})")
    else:
        ok(f"{name}: {len(ds):,} rows")
    cols = set(ds.column_names)
    check({"input_ids", "labels"} <= cols, f"{name}: columns {sorted(cols)}")
    table = ds.data.table
    ids = table.column("input_ids")
    lens = pc.list_value_length(ids)
    max_seen = pc.max(lens).as_py()
    if max_len:
        check(max_seen <= max_len, f"{name}: longest sequence {max_seen:,} tokens <= max_seq_len {max_len:,}")
    label_lens = pc.list_value_length(table.column("labels"))
    check(pc.all(pc.equal(lens, label_lens)).as_py(), f"{name}: labels have the same length as input_ids")
    total = pc.sum(lens).as_py()
    print(f"        {name}: {total:,} tokens, mean {total / len(ds):,.0f} per row")
    if "num_trainable_tokens" in cols:
        zero = pc.sum(pc.equal(table.column("num_trainable_tokens"), 0)).as_py()
        (ok if zero == 0 else warn)(f"{name}: {zero} rows without trainable tokens")
    if vocab_size is not None:
        max_id = pc.max(pc.list_flatten(ids)).as_py()
        check(max_id < vocab_size, f"{name}: max token id {max_id:,} < vocab size {vocab_size:,}")
    return len(ds)


def base_model_file(repo: str, name: str) -> Path | None:
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return None
    hit = try_to_load_from_cache(repo, name)
    return Path(hit) if isinstance(hit, str) else None


def manifest_rows(data_dir: Path, split: str) -> int | None:
    path = data_dir / "manifest.json"
    if not path.is_file():
        return None
    entry = json.loads(path.read_text()).get(split)
    return entry.get("num_examples") if isinstance(entry, dict) else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-c", "--config", type=Path, required=True)
    ap.add_argument("--accelerate-config", type=Path, default=DEFAULT_ACCELERATE)
    ap.add_argument("--compare", type=Path, default=None, help="Another config to list the differing fields.")
    ap.add_argument("--skip-data", action="store_true", help="skip loading the datasets (config checks only)")
    args = ap.parse_args()

    os.chdir(ROOT)
    print(f"repo root: {ROOT}")

    print("\n[1] config")
    cfg = load_config(args.config)
    bilingual = isinstance(cfg, BilingualTrainingConfig)
    ok(f"{args.config} ({'bilingual' if bilingual else 'single-language'} trainer)")
    new = training_dict(cfg)
    if args.compare:
        other = training_dict(load_config(args.compare))
        keys = list(new) + [k for k in other if k not in new]
        differing = [k for k in keys if new.get(k) != other.get(k)]
        print(f"        fields that differ from {args.compare}:")
        for key in differing:
            print(f"          {key}\n              this : {fmt(new.get(key))}\n              other: {fmt(other.get(key))}")

    print("\n[2] run conventions")
    acc = load_yaml_flat(args.accelerate_config) if args.accelerate_config.is_file() else {}
    n_proc = int(acc.get("num_processes", 0) or 0)
    global_batch = n_proc * cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps
    check(global_batch == 128, f"global batch {n_proc} GPUs x {cfg.per_device_train_batch_size} x "
                               f"{cfg.gradient_accumulation_steps} = {global_batch}")
    expectations = {
        "use_lora": False, "use_qlora": False, "freeze_embeddings": False, "learning_rate": 5e-5,
        "lr_scheduler_type": "cosine", "warmup_ratio": 0.05, "weight_decay": 0.01, "max_grad_norm": 1.0,
        "num_train_epochs": 2, "seed": 13, "bf16": True, "fp16": False, "padding_free": True,
        "gradient_checkpointing": True, "loss_type": "nll", "eval_strategy": "steps", "save_strategy": "steps",
        "eval_steps": 200, "save_steps": 200, "load_best_model_at_end": True,
        # early stopping: patience 4 for the trace students, 10 for the answer-level students
        "early_stopping_patience": 10 if getattr(cfg, "answer_level_sft", False) else 4,
    }
    bad = {k: getattr(cfg, k) for k, v in expectations.items() if getattr(cfg, k) != v}
    check(not bad, "full FT, trained embeddings, lr 5e-5 cosine, warm-up 0.05, wd 0.01, clip 1.0, 2 ep, seed 13, "
                   f"bf16, padding-free, grad ckpt, NLL, eval/save 200, patience {expectations['early_stopping_patience']}, best-at-end"
          + (f" (differs: {bad})" if bad else ""))

    print("\n[3] paths")
    if bilingual:
        data = {"train": Path(cfg.combined_train_dataset_path),
                **{f"eval_{k}": Path(v) for k, v in cfg.eval_dataset_paths.items()}}
        for lang, p in cfg.train_dataset_paths.items():
            (ok if Path(p, "state.json").is_file() else warn)(f"train source {lang}: {p} (input of distill.bilingual.build_dataset)")
    else:
        data = {"train": Path(cfg.dataset_path)}
        if cfg.validation_dataset_path:
            data["validation"] = Path(cfg.validation_dataset_path)
    for label, p in data.items():
        check(p.is_dir() and (p / "state.json").is_file(), f"{label}: {p}")
    check(Path(cfg.deepspeed_config_path).is_file(), f"deepspeed_config_path: {cfg.deepspeed_config_path}")
    check(args.accelerate_config.is_file(), f"accelerate config: {args.accelerate_config}")
    if acc:
        ds_file = acc.get("deepspeed_config.deepspeed_config_file")
        check(Path(ds_file or "").is_file(), f"accelerate -> deepspeed file: {ds_file}")
    out = Path(cfg.output_dir)
    if out.exists():
        ckpts = sorted(out.glob("checkpoint-*"), key=lambda p: p.stat().st_mtime)
        final = (out / "final_checkpoint" / "model.safetensors").is_file()
        print(f"        output_dir exists: {len(ckpts)} checkpoint(s), newest {ckpts[-1].name if ckpts else '-'}, "
              f"final_checkpoint {'present' if final else 'absent'} (the trainer resumes from the newest)")
    else:
        ok(f"output_dir does not exist yet (fresh run): {out}")

    print("\n[4] preprocessing manifest")
    train_dir = data["train"]
    manifest_path = train_dir.parent / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        if "model_name_or_path" in manifest:
            check(manifest["model_name_or_path"] == cfg.model_name_or_path, f"tokenised for {manifest['model_name_or_path']}")
            if cfg.max_seq_len:
                check(manifest["max_length"] == cfg.max_seq_len, f"preprocessing max_length {manifest['max_length']}")
            check(bool(manifest.get("answer_level_sft")) == bool(cfg.answer_level_sft),
                  f"answer_level_sft {manifest.get('answer_level_sft')} matches the config")
            flat_manifest = Path(manifest["train"]["input_path"]).parent / "manifest.json"
            if flat_manifest.is_file():
                flat = json.loads(flat_manifest.read_text())
                sources = flat.get("train", {}).get("source_counts", {})
                other = [s for s in sources if not any(t in s for t in TEACHER_RUNS)]
                check(not other, f"train sources are the two teachers' runs: {sorted(sources)}")
        else:
            ok(f"combined dataset manifest: {manifest.get('source_lengths')}")
    else:
        warn(f"no preprocessing manifest at {manifest_path}")

    print("\n[5] base model")
    cfg_file = base_model_file(cfg.model_name_or_path, "config.json")
    vocab_size = None
    if cfg_file:
        vocab_size = json.loads(cfg_file.read_text()).get("vocab_size")
        ok(f"{cfg.model_name_or_path} config in HF cache (vocab_size {vocab_size:,})")
    else:
        warn(f"{cfg.model_name_or_path} config not in the local HF cache (the job will try to download it)")

    print("\n[6] datasets")
    n_train = None
    if args.skip_data:
        warn("dataset loading skipped (--skip-data)")
        n_train = manifest_rows(train_dir.parent, "train")
    else:
        for label, p in data.items():
            if p.is_dir():
                split = "train" if label == "train" else "validation"
                rows = check_dataset(label, p, manifest_rows(p.parent, split) if not bilingual else None,
                                     vocab_size, cfg.max_seq_len)
                if label == "train":
                    n_train = rows

    print("\n[7] expected schedule (mirrors resolve_dataloader_drop_last + HF Trainer)")
    if global_batch and n_train:
        drop_last = n_train % global_batch != 0 if cfg.dataloader_drop_last is None else cfg.dataloader_drop_last
        steps_per_epoch = n_train // global_batch if drop_last else math.ceil(n_train / global_batch)
        total = steps_per_epoch * cfg.num_train_epochs
        print(f"        drop_last={drop_last}, {steps_per_epoch:,} steps/epoch, {total:,} total steps, "
              f"{math.ceil(cfg.warmup_ratio * total)} warm-up steps, ~{total // cfg.eval_steps} evals/checkpoints")

    print("\n" + "=" * 70)
    print(f"{len(errors)} error(s), {len(warnings)} warning(s)")
    for e in errors:
        print(f"  error: {e}")
    for w in warnings:
        print(f"  warning: {w}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
