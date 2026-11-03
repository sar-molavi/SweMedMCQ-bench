"""
Defines the configuration for the fine-tuning process using Pydantic dataclasses.

This module provides a structured and type-safe way to manage hyperparameters
for the training script. The configuration is loaded from a JSON file and parsed
into a nested dataclass structure.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, fields


@dataclass
class LoraConfigData:
    """Dataclass for LoRA (Low-Rank Adaptation) specific parameters."""

    r: int = 64
    lora_alpha: int = None
    lora_dropout: float = 0.1
    target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "up_proj",
            "down_proj",
        ]
    )
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    modules_to_save: list[str] | None = None

    def __post_init__(self):
        if self.lora_alpha is None:
            self.lora_alpha = self.r * 2


@dataclass
class TrainingConfig:
    """
    Main configuration class for the fine-tuning script.

    This dataclass holds all the hyperparameters and settings needed for the
    training process, from model and dataset paths to training arguments and
    LoRA settings.
    """

    # Model and tokenizer parameters
    model_name_or_path: str

    # LoRA and QLoRA settings
    use_lora: bool
    use_qlora: bool  # Flag to enable 4-bit quantization. Default False
    load_in_8bit: bool  # By default is False
    lora_config: LoraConfigData

    # Dataset and output paths
    dataset_name: str | list[str]
    dataset_path: str | list[str]
    output_dir: str
    validation_dataset_path: str | list[str] | None
    preprocessing_dataset_path: str | list[str] | None
    preprocessing_validation_dataset_path: str | list[str] | None
    load_best_model_at_end: bool
    seed: int | None
    max_training_sample: int | None
    max_seq_len: int | None  # Default None

    # Training hyperparameters
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    num_train_epochs: int
    learning_rate: float
    lr_scheduler_type: str
    warmup_ratio: float
    max_grad_norm: float
    weight_decay: float
    early_stopping_patience: int

    # Logging and saving settings
    save_strategy: str  # Default epoch
    eval_strategy: str  # Default no
    logging_steps: int
    save_steps: int
    eval_steps: int
    save_total_limit: int

    # Precision and performance settings
    fp16: bool
    bf16: bool
    gradient_checkpointing: bool
    freeze_embeddings: bool = True
    padding_free: bool = True
    answer_level_sft: bool = False
    loss_type: str = "dft"
    dataloader_drop_last: bool | None = None

    # Reporting and logging
    report_to: str = "tensorboard"
    run_name: str | None = None

    # DeepSpeed configuration (path to JSON config). None disables DeepSpeed.
    deepspeed_config_path: str | None = None

    # FSDP configuration (path to JSON config). None disables FSDP.
    fsdp_config_path: str | None = None

    @classmethod
    def from_json(cls, json_path: str) -> "TrainingConfig":
        """
        Loads configuration from a JSON file and creates an instance of TrainingConfig.

        The method reads the specified JSON file, and its structure should match the
        fields defined in this dataclass, including the nested LoraConfigData.

        Args:
            json_path (str): The path to the JSON configuration file.

        Returns:
            TrainingConfig: An instance of the TrainingConfig class populated with
                            data from the JSON file.
        """
        with open(json_path, "r") as f:
            config_dict = json.load(f)

        # Set default values for fields that might be missing
        config_dict["per_device_eval_batch_size"] = config_dict.get(
            "per_device_eval_batch_size", 1
        )
        config_dict["use_qlora"] = config_dict.get("use_qlora", False)
        config_dict["load_in_8bit"] = config_dict.get("load_in_8bit", False)
        config_dict["learning_rate"] = config_dict.get("learning_rate", 1e-6)
        config_dict["lr_scheduler_type"] = config_dict.get(
            "lr_scheduler_type", "cosine"
        )
        config_dict["max_grad_norm"] = config_dict.get("max_grad_norm", 0.5)
        config_dict["weight_decay"] = config_dict.get("weight_decay", 0.01)
        config_dict["early_stopping_patience"] = config_dict.get(
            "early_stopping_patience", 10
        )
        config_dict["warmup_ratio"] = config_dict.get("warmup_ratio", 0.05)
        config_dict["save_strategy"] = config_dict.get("save_strategy", "epoch")
        config_dict["eval_strategy"] = config_dict.get("eval_strategy", "no")
        config_dict["eval_steps"] = config_dict.get("eval_steps", None)
        config_dict["save_steps"] = config_dict.get("save_steps", 100)
        config_dict["save_total_limit"] = config_dict.get("save_total_limit", None)
        config_dict["deepspeed_config_path"] = config_dict.get(
            "deepspeed_config_path", None
        )
        config_dict["fsdp_config_path"] = config_dict.get("fsdp_config_path", None)

        # Dataset and output paths
        config_dict["validation_dataset_path"] = config_dict.get(
            "validation_dataset_path", None
        )
        config_dict["preprocessing_dataset_path"] = config_dict.get(
            "preprocessing_dataset_path", None
        )
        config_dict["preprocessing_validation_dataset_path"] = config_dict.get(
            "preprocessing_validation_dataset_path", None
        )
        config_dict["load_best_model_at_end"] = config_dict.get(
            "load_best_model_at_end", False
        )
        config_dict["seed"] = config_dict.get("seed", None)
        config_dict["max_training_sample"] = config_dict.get(
            "max_training_sample", None
        )
        config_dict["max_seq_len"] = config_dict.get("max_seq_len", None)
        config_dict["freeze_embeddings"] = config_dict.get(
            "freeze_embeddings", True
        )
        config_dict["padding_free"] = config_dict.get("padding_free", True)
        config_dict["answer_level_sft"] = config_dict.get("answer_level_sft", False)
        config_dict["dataloader_drop_last"] = config_dict.get(
            "dataloader_drop_last", None
        )
        config_dict["loss_type"] = config_dict.get("loss_type", "dft")
        config_dict["run_name"] = config_dict.get("run_name", None)
        if config_dict["loss_type"] not in {"nll", "dft"}:
            raise ValueError(
                "loss_type must be one of {'nll', 'dft'}, got "
                f"{config_dict['loss_type']!r}."
            )

        # Create LoraConfigData instance from the nested dictionary
        if "lora_config" in config_dict:
            lora_config_data = LoraConfigData(**config_dict.pop("lora_config"))
        else:
            lora_config_data = LoraConfigData()

        valid_fields = {field.name for field in fields(cls)}
        unknown_fields = sorted(set(config_dict) - valid_fields)
        if unknown_fields:
            raise ValueError(
                "Unknown training config field(s): "
                f"{', '.join(unknown_fields)}. "
                "DeepSpeed-only options such as gradient_clipping belong in the "
                "DeepSpeed JSON referenced by deepspeed_config_path."
            )

        # Create TrainingConfig instance
        return cls(lora_config=lora_config_data, **config_dict)
