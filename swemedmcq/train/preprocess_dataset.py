"""Pre-tokenize SFT JSONL files and save reusable Hugging Face datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from transformers import AutoConfig, AutoTokenizer

from .config import TrainingConfig
from .dataset import get_dataset
from .utils import get_max_length


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def prepare_split(
    input_path: str | Path,
    output_path: Path,
    tokenizer,
    model_name_or_path: str,
    max_num_samples: int | None,
    max_length: int,
    batch_size: int,
    num_proc: int | None,
    answer_level_sft: bool,
) -> int:
    dataset = get_dataset(
        path=input_path,
        tokenizer=tokenizer,
        model_name_or_path=model_name_or_path,
        max_num_samples=max_num_samples,
        max_length=max_length,
        batch_size=batch_size,
        num_proc=num_proc,
        answer_level_sft=answer_level_sft,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(output_path))
    return len(dataset)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Preprocess SFT JSONL files once and save tokenized datasets to disk."
    )
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        required=True,
        help="Training JSON config containing model and dataset paths.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory that will contain train/ and validation/ datasets.",
    )
    parser.add_argument(
        "--max-training-sample",
        type=int,
        default=None,
        help="Optional override for limiting the train split during preprocessing.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10_000,
        help="Batch size passed to datasets.map during chat-template preprocessing.",
    )
    parser.add_argument(
        "--num-proc",
        type=int,
        default=128,
        help="Number of worker processes passed to datasets.map/filter.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    cfg = TrainingConfig.from_json(str(args.config))

    base_config = AutoConfig.from_pretrained(cfg.model_name_or_path)
    max_length = cfg.max_seq_len or get_max_length(model=None, config=base_config)
    if max_length is None:
        raise ValueError("Set max_seq_len in the training config.")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_limit = (
        args.max_training_sample
        if args.max_training_sample is not None
        else cfg.max_training_sample
    )
    train_input_path = cfg.preprocessing_dataset_path or cfg.dataset_path
    validation_input_path = (
        cfg.preprocessing_validation_dataset_path or cfg.validation_dataset_path
    )
    train_output = args.output_dir / "train"
    train_count = prepare_split(
        input_path=train_input_path,
        output_path=train_output,
        tokenizer=tokenizer,
        model_name_or_path=cfg.model_name_or_path,
        max_num_samples=train_limit,
        max_length=max_length,
        batch_size=args.batch_size,
        num_proc=args.num_proc,
        answer_level_sft=cfg.answer_level_sft,
    )

    validation_output = None
    validation_count = None
    if validation_input_path:
        validation_output = args.output_dir / "validation"
        validation_count = prepare_split(
            input_path=validation_input_path,
            output_path=validation_output,
            tokenizer=tokenizer,
            model_name_or_path=cfg.model_name_or_path,
            max_num_samples=None,
            max_length=max_length,
            batch_size=args.batch_size,
            num_proc=args.num_proc,
            answer_level_sft=cfg.answer_level_sft,
        )

    manifest = {
        "config_path": str(args.config),
        "model_name_or_path": cfg.model_name_or_path,
        "max_length": max_length,
        "batch_size": args.batch_size,
        "num_proc": args.num_proc,
        "answer_level_sft": cfg.answer_level_sft,
        "train": {
            "input_path": str(train_input_path),
            "output_path": str(train_output),
            "num_examples": train_count,
            "max_num_samples": train_limit,
        },
        "validation": None
        if validation_output is None
        else {
            "input_path": str(validation_input_path),
            "output_path": str(validation_output),
            "num_examples": validation_count,
        },
    }
    write_json(args.output_dir / "manifest.json", manifest)

    print(f"Wrote train dataset with {train_count} examples to {train_output}")
    if validation_output is not None:
        print(
            f"Wrote validation dataset with {validation_count} examples to {validation_output}"
        )
    print(f"Wrote manifest to {args.output_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
