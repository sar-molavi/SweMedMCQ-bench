"""Configuration wrapper for bilingual SFT training."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping

from ..config import LoraConfigData, TrainingConfig


EvalLossWeighting = Literal["num_examples", "equal"]


@dataclass
class BilingualTrainingConfig:
    """Existing SFT config plus bilingual dataset metadata."""

    training: TrainingConfig
    train_dataset_paths: dict[str, str]
    eval_dataset_paths: dict[str, str]
    combined_train_dataset_path: str
    combined_train_shuffle_seed: int = 2000
    eval_loss_weighting: EvalLossWeighting = "num_examples"
    max_steps: int | None = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self.training, name)

    @classmethod
    def from_json(cls, json_path: str | Path) -> "BilingualTrainingConfig":
        with Path(json_path).open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        if not isinstance(raw, dict):
            raise ValueError(f"{json_path}: expected a JSON object.")

        train_paths = _string_mapping(raw.get("train_dataset_paths"), "train_dataset_paths")
        eval_paths = _string_mapping(raw.get("eval_dataset_paths"), "eval_dataset_paths")
        combined_path = raw.get("combined_train_dataset_path")
        if not isinstance(combined_path, str) or not combined_path.strip():
            raise ValueError("combined_train_dataset_path must be a non-empty string.")

        shuffle_seed = raw.get("combined_train_shuffle_seed", 2000)
        if not isinstance(shuffle_seed, int):
            raise ValueError("combined_train_shuffle_seed must be an integer.")

        weighting = raw.get("eval_loss_weighting", "num_examples")
        if weighting not in {"num_examples", "equal"}:
            raise ValueError("eval_loss_weighting must be one of {'num_examples', 'equal'}.")

        max_steps = raw.get("max_steps")
        if max_steps is not None and (not isinstance(max_steps, int) or max_steps <= 0):
            raise ValueError("max_steps must be a positive integer when set.")

        training_raw = {
            key: value
            for key, value in raw.items()
            if key
            not in {
                "train_dataset_paths",
                "eval_dataset_paths",
                "combined_train_dataset_path",
                "combined_train_shuffle_seed",
                "eval_loss_weighting",
                "max_steps",
            }
        }
        training_raw["dataset_path"] = combined_path
        training_raw["validation_dataset_path"] = None
        training_raw.setdefault("preprocessing_dataset_path", None)
        training_raw.setdefault("preprocessing_validation_dataset_path", None)

        training = _training_config_from_dict(training_raw)
        return cls(
            training=training,
            train_dataset_paths=train_paths,
            eval_dataset_paths=eval_paths,
            combined_train_dataset_path=combined_path,
            combined_train_shuffle_seed=shuffle_seed,
            eval_loss_weighting=weighting,
            max_steps=max_steps,
        )


def _string_mapping(value: Any, field_name: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{field_name} must be a non-empty object.")
    result = {}
    for key, path in value.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError(f"{field_name} contains an empty language key.")
        if not isinstance(path, str) or not path.strip():
            raise ValueError(f"{field_name}.{key} must be a non-empty string.")
        result[key] = path
    return result


def _training_config_from_dict(config_dict: dict[str, Any]) -> TrainingConfig:
    config_dict = dict(config_dict)
    config_dict["per_device_eval_batch_size"] = config_dict.get(
        "per_device_eval_batch_size", 1
    )
    config_dict["use_qlora"] = config_dict.get("use_qlora", False)
    config_dict["load_in_8bit"] = config_dict.get("load_in_8bit", False)
    config_dict["learning_rate"] = config_dict.get("learning_rate", 1e-6)
    config_dict["lr_scheduler_type"] = config_dict.get("lr_scheduler_type", "cosine")
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
    config_dict["deepspeed_config_path"] = config_dict.get("deepspeed_config_path", None)
    config_dict["fsdp_config_path"] = config_dict.get("fsdp_config_path", None)
    config_dict["validation_dataset_path"] = config_dict.get("validation_dataset_path", None)
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
    config_dict["max_training_sample"] = config_dict.get("max_training_sample", None)
    config_dict["max_seq_len"] = config_dict.get("max_seq_len", None)
    config_dict["freeze_embeddings"] = config_dict.get("freeze_embeddings", True)
    config_dict["padding_free"] = config_dict.get("padding_free", True)
    config_dict["answer_level_sft"] = config_dict.get("answer_level_sft", False)
    config_dict["dataloader_drop_last"] = config_dict.get("dataloader_drop_last", None)
    config_dict["loss_type"] = config_dict.get("loss_type", "dft")
    config_dict["run_name"] = config_dict.get("run_name", None)
    if config_dict["loss_type"] not in {"nll", "dft"}:
        raise ValueError(
            "loss_type must be one of {'nll', 'dft'}, got "
            f"{config_dict['loss_type']!r}."
        )

    lora_config = config_dict.pop("lora_config", {})
    lora_config_data = LoraConfigData(**lora_config)

    valid_fields = set(TrainingConfig.__dataclass_fields__)
    unknown_fields = sorted(set(config_dict) - valid_fields)
    if unknown_fields:
        raise ValueError(
            "Unknown training config field(s): " + ", ".join(unknown_fields)
        )

    return TrainingConfig(lora_config=lora_config_data, **config_dict)
