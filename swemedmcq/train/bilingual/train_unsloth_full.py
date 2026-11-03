"""Full-model Unsloth SFT training for the bilingual EN/SV MCQ dataset."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Mapping

import torch
import unsloth
from datasets import load_from_disk
from transformers import AutoConfig, EarlyStoppingCallback
from trl import SFTConfig
from unsloth import FastLanguageModel

from .config import BilingualTrainingConfig
from .trainer import BilingualEvalSFTTrainer
from ..dataset import DataCollatorForSupervisedFinetuning
from ..train_unsloth_full import (
    ensure_configured_tied_embeddings,
    freeze_input_embeddings,
)
from ..utils import cache_vram, get_max_length, resolve_dataloader_drop_last


def _load_arrow_dataset(path: str | Path, max_num_samples: int | None = None):
    dataset = load_from_disk(str(path), keep_in_memory=True)
    if max_num_samples is not None:
        dataset = dataset.select(range(min(max_num_samples, len(dataset))))
    return dataset


def load_eval_datasets(paths: Mapping[str, str | Path]) -> dict[str, Any]:
    """Load language-keyed validation datasets for Trainer dict evaluation."""

    if not paths:
        raise ValueError("At least one eval dataset path is required.")
    return {str(language): _load_arrow_dataset(path) for language, path in paths.items()}


def train(config_path: str | Path) -> None:
    print("Loading bilingual configuration...")
    cfg = BilingualTrainingConfig.from_json(config_path)

    if cfg.use_lora:
        raise ValueError("The bilingual full-finetune entrypoint expects use_lora=False.")

    base_model_config = AutoConfig.from_pretrained(cfg.model_name_or_path)
    max_seq_length = cfg.max_seq_len or get_max_length(
        model=None,
        config=base_model_config,
    )
    if max_seq_length is None:
        raise ValueError(
            "Unable to determine the model's maximum sequence length. "
            "Set `max_seq_len` in the training configuration."
        )

    if cfg.use_qlora and os.environ.get("LOCAL_RANK") is not None:
        local_rank = int(os.environ["LOCAL_RANK"])
        if torch.cuda.is_available() and local_rank < torch.cuda.device_count():
            torch.cuda.set_device(local_rank)
            print(f"Set CUDA device to: {local_rank}")

    model_kwargs: dict[str, Any] = {
        "load_in_4bit": False,
        "text_only": True,
        "gpu_memory_utilization": 0.9,
        "device_map": None,
        "full_finetuning": True,
    }
    if cfg.use_qlora:
        if cfg.load_in_8bit:
            print("Using QLoRA (8-bit quantization).")
            model_kwargs["load_in_8bit"] = True
        else:
            print("Using QLoRA (4-bit quantization).")
            model_kwargs["load_in_4bit"] = True
            model_kwargs["full_finetuning"] = False

    if cfg.bf16 or cfg.fp16:
        precision = "bf16" if cfg.bf16 else "fp16"
        print(f"Using standard precision: {precision}.")
        model_kwargs["dtype"] = torch.bfloat16 if cfg.bf16 else torch.float16

    print(f"Loading base model: {cfg.model_name_or_path}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        cfg.model_name_or_path,
        max_seq_length=max_seq_length,
        use_gradient_checkpointing=("unsloth" if cfg.gradient_checkpointing else False),
        **model_kwargs,
    )
    ensure_configured_tied_embeddings(model)

    if cfg.freeze_embeddings:
        freeze_input_embeddings(model)
        if cfg.gradient_checkpointing:
            model.enable_input_require_grads()
            print(
                "Enabled gradients on embedding outputs for gradient checkpointing; "
                "the embedding weights remain frozen."
            )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"Tokenizer EOS token set to: {tokenizer.eos_token}")

    print(f"Loading combined training dataset: {cfg.combined_train_dataset_path}")
    train_dataset = _load_arrow_dataset(
        cfg.combined_train_dataset_path,
        max_num_samples=cfg.max_training_sample,
    )
    print(f"Training dataset size: {len(train_dataset)} samples.")

    eval_dataset = load_eval_datasets(cfg.eval_dataset_paths)
    for language, dataset in eval_dataset.items():
        print(f"Validation dataset size ({language}): {len(dataset)} samples.")

    if cfg.padding_free:
        data_collator = None
        print("Using TRL padding-free collation.")
    else:
        data_collator = DataCollatorForSupervisedFinetuning(
            pad_token_id=tokenizer.pad_token_id,
            max_length=max_seq_length,
        )

    eval_strategy = cfg.eval_strategy
    if eval_dataset and cfg.eval_strategy == "no":
        eval_strategy = "steps"

    dataloader_drop_last = resolve_dataloader_drop_last(
        configured_drop_last=cfg.dataloader_drop_last,
        dataset_length=len(train_dataset),
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
    )

    training_args_kwargs = {
        "output_dir": cfg.output_dir,
        "per_device_train_batch_size": cfg.per_device_train_batch_size,
        "per_device_eval_batch_size": cfg.per_device_eval_batch_size,
        "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
        "num_train_epochs": cfg.num_train_epochs,
        "learning_rate": cfg.learning_rate,
        "lr_scheduler_type": cfg.lr_scheduler_type,
        "logging_steps": cfg.logging_steps,
        "save_strategy": cfg.save_strategy,
        "save_steps": cfg.save_steps,
        "fp16": cfg.fp16,
        "bf16": cfg.bf16,
        "report_to": cfg.report_to,
        "gradient_checkpointing": cfg.gradient_checkpointing,
        "ddp_find_unused_parameters": False,
        "eval_strategy": eval_strategy,
        "remove_unused_columns": False,
        "warmup_ratio": cfg.warmup_ratio,
        "dataloader_num_workers": 4,
        "dataloader_drop_last": dataloader_drop_last,
        "optim": "paged_adamw_32bit" if cfg.use_qlora else "adamw_torch_fused",
        "weight_decay": cfg.weight_decay,
        "max_grad_norm": cfg.max_grad_norm,
        "logging_strategy": "steps",
        "greater_is_better": False,
        "seed": cfg.seed if cfg.seed is not None else 42,
        "data_seed": cfg.seed if cfg.seed is not None else 42,
        "loss_type": cfg.loss_type,
        "padding_free": cfg.padding_free,
        "prediction_loss_only": True,
        "load_best_model_at_end": cfg.load_best_model_at_end,
        "metric_for_best_model": "eval_loss",
        "dataset_kwargs": {"skip_prepare_dataset": True},
        "max_length": None if cfg.padding_free else max_seq_length,
    }
    if cfg.run_name:
        training_args_kwargs["run_name"] = cfg.run_name
    if cfg.deepspeed_config_path:
        training_args_kwargs["deepspeed"] = cfg.deepspeed_config_path
    if cfg.fsdp_config_path:
        if cfg.deepspeed_config_path:
            raise ValueError("Choose either DeepSpeed or FSDP, not both.")
        training_args_kwargs["fsdp"] = "full_shard auto_wrap"
        training_args_kwargs["fsdp_config"] = cfg.fsdp_config_path
    if cfg.eval_steps is not None:
        training_args_kwargs["eval_steps"] = cfg.eval_steps
    if cfg.save_total_limit is not None:
        training_args_kwargs["save_total_limit"] = cfg.save_total_limit
    if cfg.max_steps is not None:
        training_args_kwargs["max_steps"] = cfg.max_steps

    training_args = SFTConfig(**training_args_kwargs)
    callbacks = []
    if cfg.early_stopping_patience and cfg.early_stopping_patience > 0:
        callbacks.append(
            EarlyStoppingCallback(early_stopping_patience=cfg.early_stopping_patience)
        )

    trainer = BilingualEvalSFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
        callbacks=callbacks,
        eval_loss_weighting=cfg.eval_loss_weighting,
    )

    print("\nVerifying training setup...")
    print(f"Model device: {next(model.parameters()).device}")
    print(f"Model dtype: {next(model.parameters()).dtype}")
    print(f"Training samples: {len(train_dataset)}")
    print(
        "Validation samples: "
        + ", ".join(f"{language}={len(dataset)}" for language, dataset in eval_dataset.items())
    )
    print(
        "Effective batch size: "
        f"{cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps * torch.cuda.device_count() if torch.cuda.is_available() else cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps}"
    )

    checkpoint = None
    if os.path.isdir(cfg.output_dir):
        checkpoints = [
            os.path.join(cfg.output_dir, entry)
            for entry in os.listdir(cfg.output_dir)
            if entry.startswith("checkpoint-")
        ]
        if checkpoints:
            checkpoint = max(checkpoints, key=os.path.getmtime)

    trainer.train(resume_from_checkpoint=checkpoint)

    if trainer.is_world_process_zero():
        final_output_dir = os.path.join(cfg.output_dir, "final_checkpoint")
        print(f"\nSaving final model to {final_output_dir}...")
        trainer.save_model(final_output_dir)
        tokenizer.save_pretrained(final_output_dir)
        print(f"Training complete. Final model saved to: {os.path.abspath(final_output_dir)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--config", type=Path, required=True)
    parser.add_argument("--local_rank", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--cache-vram", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.cache_vram and torch.cuda.is_available():
        local_rank = os.environ.get("LOCAL_RANK")
        cache_vram(device=int(local_rank) if local_rank is not None else 0)
    train(args.config)
