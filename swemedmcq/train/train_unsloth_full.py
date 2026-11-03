"""Fine-tune a student with Unsloth and TRL's ``SFTTrainer`` (full fine-tuning for all
released students; the LoRA/QLoRA options of the config are kept but unused).

The dataset is pre-tokenised with labels (``swemedmcq.train.preprocess_dataset``), so
TRL dataset preparation is skipped and the trainer uses the configured TRL SFT loss.

    accelerate launch --config_file configs/train/accelerate_deepspeed_zero2_4gpu.yaml \
        -m swemedmcq.train.train_unsloth_full --config configs/train/<student>.json
"""

import unsloth

import torch
import os
import argparse
from transformers import (
    AutoConfig,
    EarlyStoppingCallback,
)
from trl import SFTConfig,SFTTrainer
from unsloth import FastLanguageModel

from .config import TrainingConfig
from .dataset import (
    get_dataset,
    DataCollatorForSupervisedFinetuning,
)
from .utils import get_max_length, resolve_dataloader_drop_last, cache_vram


def unfreeze_lora_parameters(model):
    """Manually unfreeze LoRA parameters that may have been incorrectly frozen."""
    print("Manually unfreezing LoRA parameters...")
    unfrozen_count = 0
    for name, param in model.named_parameters():
        # Check if it's a LoRA parameter (contains 'lora' in the name)
        name = name.lower()
        if "lora" in name:
            param.requires_grad = True
            unfrozen_count += 1

    print(f"Unfroze {unfrozen_count} LoRA parameters")
    return unfrozen_count


def freeze_input_embeddings(model):
    """Freeze token embeddings and report whether the output head is tied."""
    input_embeddings = model.get_input_embeddings()
    if input_embeddings is None:
        raise ValueError("The model does not expose an input embedding module.")

    frozen_parameters = 0
    frozen_elements = 0
    for param in input_embeddings.parameters():
        param.requires_grad_(False)
        frozen_parameters += 1
        frozen_elements += param.numel()

    output_embeddings = model.get_output_embeddings()
    output_head_is_tied = (
        output_embeddings is not None
        and hasattr(input_embeddings, "weight")
        and hasattr(output_embeddings, "weight")
        and output_embeddings.weight is input_embeddings.weight
    )
    tied_message = " (including the tied output head)" if output_head_is_tied else ""
    print(
        f"Froze {frozen_parameters} input embedding parameter tensor(s) "
        f"containing {frozen_elements:,} elements{tied_message}."
    )


def ensure_configured_tied_embeddings(model):
    """Restore input/output weight tying when the model configuration requires it."""
    if not getattr(model.config, "tie_word_embeddings", False):
        return

    if not hasattr(model, "tie_weights"):
        raise ValueError(
            "The model configuration requires tied word embeddings, but the model "
            "does not expose tie_weights()."
        )

    model.tie_weights()
    input_embeddings = model.get_input_embeddings()
    output_embeddings = model.get_output_embeddings()
    if (
        input_embeddings is None
        or output_embeddings is None
        or not hasattr(input_embeddings, "weight")
        or not hasattr(output_embeddings, "weight")
        or output_embeddings.weight is not input_embeddings.weight
    ):
        raise RuntimeError(
            "The model configuration requires tied word embeddings, but tying "
            "the input embeddings and output head failed."
        )

    print("Confirmed that input embeddings and output head are tied.")


def train(config_path):
    """
    The main training function that orchestrates the entire fine-tuning process.

    Args:
        config_path (str): Path to the JSON configuration file.
    """
    # --- 1. Load Configuration ---
    print("Loading configuration...")
    cfg = TrainingConfig.from_json(config_path)

    if cfg.use_lora and cfg.lora_config is None:
        raise ValueError("`use_lora` is True but no `lora_config` was provided.")

    base_model_config = AutoConfig.from_pretrained(cfg.model_name_or_path)
    max_seq_length = cfg.max_seq_len or get_max_length(
        model=None, config=base_model_config
    )
    if max_seq_length is None:
        raise ValueError(
            "Unable to determine the model's maximum sequence length. "
            "Set `max_seq_len` in the training configuration."
        )

    # --- 0. Set the device if QLoRA is active ---
    if cfg.use_qlora and (os.environ.get("LOCAL_RANK", None) is not None):
        local_rank = int(os.environ["LOCAL_RANK"])
        if torch.cuda.is_available() and local_rank < torch.cuda.device_count():
            torch.cuda.set_device(local_rank)
            print(f"Set CUDA device to: {local_rank}")

    # --- 3. Configure Model Loading (Quantization vs. Full Precision) ---
    print("Configuring model loading...")
    # model_kwargs = {"attn_implementation": "eager"}
    model_kwargs = {
        "load_in_4bit": False,
        "text_only": True,
        "gpu_memory_utilization": 0.9,
        "device_map": None,
        "full_finetuning": False if (cfg.use_lora or cfg.use_qlora) else True,
    }  # "device_map": "auto"
    if cfg.use_qlora:
        # Configure 4-bit quantization for QLoRA
        if not cfg.load_in_8bit:
            print("Using QLoRA (4-bit quantization).")
            model_kwargs["load_in_4bit"] = True
        else:
            print("Using QLoRA (8-bit quantization).")
            model_kwargs["load_in_8bit"] = True

    if cfg.bf16 or cfg.fp16:
        # Load in standard bfloat16 or float16 precision
        precision = "bf16" if cfg.bf16 else ("fp16" if cfg.fp16 else "Auto-selected")
        print(f"Using standard precision: {precision}.")
        model_kwargs["dtype"] = (
            torch.bfloat16 if cfg.bf16 else (torch.float16 if cfg.fp16 else None)
        )

    # --- 4. Load Base Model ---
    print(f"Loading base model: {cfg.model_name_or_path}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        cfg.model_name_or_path,
        max_seq_length=max_seq_length,
        use_gradient_checkpointing=("unsloth" if cfg.gradient_checkpointing else False),
        **model_kwargs,
    )
    ensure_configured_tied_embeddings(model)

    if cfg.freeze_embeddings and not (cfg.use_lora or cfg.use_qlora):
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

    # --- 5. Configure and Apply LoRA / QLoRA ---
    lora_cfg = None
    if cfg.use_lora:
        for param in model.parameters():
            param.requires_grad = False

        print("Applying LoRA/QLoRA modifications...")
        model = FastLanguageModel.get_peft_model(
            model,
            r=cfg.lora_config.r,
            target_modules=cfg.lora_config.target_modules,
            lora_alpha=cfg.lora_config.lora_alpha,
            lora_dropout=cfg.lora_config.lora_dropout,  # Supports any, but = 0 is optimized
            bias=cfg.lora_config.bias,  # Supports any, but = "none" is optimized
            # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
            use_gradient_checkpointing=(
                "unsloth" if cfg.gradient_checkpointing else False
            ),  # True or "unsloth" for very long context
            random_state=cfg.seed,
            use_rslora=False,  # We support rank stabilized LoRA
            loftq_config=None,  # And LoftQ
            max_seq_length=max_seq_length,
        )

        print("Trainable parameters after applying LoRA:")
        model.print_trainable_parameters()

    # --- 6. Load and Prepare Dataset ---
    print(f"Loading training dataset '{cfg.dataset_name}'...")
    train_dataset = get_dataset(
        path=cfg.dataset_path,
        tokenizer=tokenizer,
        model_name_or_path=cfg.model_name_or_path,
        max_num_samples=cfg.max_training_sample,
        max_length=max_seq_length,
        answer_level_sft=cfg.answer_level_sft,
    )
    print(f"Training dataset size: {len(train_dataset)} samples.")

    eval_dataset = None
    if cfg.validation_dataset_path:
        eval_name = cfg.dataset_name
        print(f"Loading validation dataset '{eval_name}'...")
        eval_dataset = get_dataset(
            path=cfg.validation_dataset_path,
            tokenizer=tokenizer,
            model_name_or_path=cfg.model_name_or_path,
            max_num_samples=None,
            max_length=max_seq_length,
            answer_level_sft=cfg.answer_level_sft,
        )
        print(f"Validation dataset size: {len(eval_dataset)} samples.")

    # --- 7. Data Collator ---
    if cfg.padding_free:
        data_collator = None
        print("Using TRL padding-free collation.")
    else:
        data_collator = DataCollatorForSupervisedFinetuning(
            pad_token_id=tokenizer.pad_token_id, max_length=max_seq_length
        )

    # --- 8. Set Up SFT Arguments ---
    print("Setting up SFT arguments...")

    eval_strategy = cfg.eval_strategy
    if cfg.validation_dataset_path and cfg.eval_strategy == "no":
        eval_strategy = "steps"
    dataloader_drop_last = resolve_dataloader_drop_last(
        configured_drop_last=cfg.dataloader_drop_last,
        dataset_length=len(train_dataset),
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
    )

    # Prepare SFTConfig kwargs. TRL keeps the pre-tokenized input_ids/labels
    # untouched and uses the configured SFT loss.
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
        "eval_strategy": eval_strategy if cfg.validation_dataset_path else "no",
        "remove_unused_columns": False,
        "warmup_ratio": cfg.warmup_ratio,
        "dataloader_num_workers": 4,  # Parallel data loading
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
        "prediction_loss_only":True
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
    if eval_strategy == "steps":
        training_args_kwargs["load_best_model_at_end"] = cfg.load_best_model_at_end

    if cfg.validation_dataset_path:
        training_args_kwargs["metric_for_best_model"] = "eval_loss"

    if cfg.validation_dataset_path and cfg.eval_steps is not None:
        training_args_kwargs["eval_steps"] = cfg.eval_steps
    if cfg.save_total_limit is not None:
        training_args_kwargs["save_total_limit"] = cfg.save_total_limit
    training_args_kwargs["dataset_kwargs"] = {"skip_prepare_dataset": True}
    # Inputs are already truncated during get_dataset(..., max_length=max_seq_length).
    # TRL requires max_length=None when padding_free=True without packing.
    training_args_kwargs["max_length"] = None if cfg.padding_free else max_seq_length
    training_args = SFTConfig(**training_args_kwargs)

    # --- 9. Initialize Trainer ---
    print("Initializing SFTTrainer...")
    callbacks = []
    if (
        eval_dataset is not None
        and cfg.early_stopping_patience
        and cfg.early_stopping_patience > 0
    ):
        callbacks.append(
            EarlyStoppingCallback(early_stopping_patience=cfg.early_stopping_patience)
        )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
        callbacks=callbacks,
    )

    # --- 9.5 Ensure that the model is trainable ---
    # if cfg.use_lora:
    #    unfreeze_lora_parameters(model)

    # Print trainable parameters for verification
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()

    # --- 10. Verify training setup ---
    print("\nVerifying training setup...")
    print(f"Model device: {next(model.parameters()).device}")
    print(f"Model dtype: {next(model.parameters()).dtype}")
    print(f"Training samples: {len(train_dataset)}")
    if eval_dataset:
        print(f"Validation samples: {len(eval_dataset)}")
    print(
        f"Effective batch size: {cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps * torch.cuda.device_count() if torch.cuda.is_available() else cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps}"
    )

    # --- 11. Start Training ---
    print("\nStarting model training with SFTTrainer...")
    print("=" * 60)

    # Check if we should resume from checkpoint
    checkpoint = None
    if os.path.isdir(cfg.output_dir):
        checkpoints = [
            os.path.join(cfg.output_dir, d)
            for d in os.listdir(cfg.output_dir)
            if d.startswith("checkpoint-")
        ]
        if checkpoints:
            checkpoint = max(checkpoints, key=os.path.getmtime)  # latest checkpoint

    trainer.train(resume_from_checkpoint=checkpoint)

    # --- 12. Save Final Model ---
    if trainer.is_world_process_zero():
        final_output_dir = os.path.join(cfg.output_dir, "final_checkpoint")
        artifact_name = "LoRA adapter" if cfg.use_lora else "model"
        print(f"\nSaving final {artifact_name} to {final_output_dir}...")
        trainer.save_model(final_output_dir)

        # Save tokenizer as well
        tokenizer.save_pretrained(final_output_dir)

        print("=" * 60)
        print(
            f"Training complete! Final {artifact_name} saved to: "
            f"{os.path.abspath(final_output_dir)}"
        )
        print("You can now use it for inference or further training.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tune a student with Unsloth and TRL (full fine-tuning for the released configs)."
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="Path to the JSON configuration file.",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--cache-vram",
        action="store_true"
    )

    args = parser.parse_args()
    
    if args.cache_vram and torch.cuda.is_available():
        local_rank = os.environ.get("LOCAL_RANK", None)
        if local_rank is not None:
            # Multi-GPU / distributed processing:
            # cache VRAM only on the GPU assigned to this process.
            cache_vram(device=int(local_rank))
        else:
            # Single-GPU or non-distributed execution.
            cache_vram(device=0)

    try:
        train(args.config)
    except Exception as e:
        print(f"Training failed with error: {e}")
        raise
