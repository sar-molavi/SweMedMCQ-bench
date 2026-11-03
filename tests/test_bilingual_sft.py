from __future__ import annotations

import json

import pytest

from swemedmcq.distill.bilingual import build_dataset
from swemedmcq.distill.bilingual.check_prompt_id_overlap import check_disjoint_prompt_ids
from swemedmcq.train.bilingual.config import BilingualTrainingConfig
from swemedmcq.train.bilingual.trainer import weighted_eval_loss


class FakeDataset:
    column_names = ["input_ids", "labels", "num_trainable_tokens"]

    def __init__(self, values):
        self.values = list(values)
        self.saved_to = None

    def __len__(self):
        return len(self.values)

    def shuffle(self, seed):
        return FakeDataset([f"{seed}:{value}" for value in reversed(self.values)])

    def save_to_disk(self, path):
        self.saved_to = path


def test_weighted_eval_loss_uses_dataset_lengths() -> None:
    metrics = {
        "en": {"eval_en_loss": 2.0},
        "sv": {"eval_sv_loss": 4.0},
    }
    datasets = {"en": FakeDataset(range(2048)), "sv": FakeDataset(range(3072))}

    assert weighted_eval_loss(metrics, datasets) == (2.0 * 2048 + 4.0 * 3072) / 5120


def test_combined_dataset_builder_preserves_lengths_and_seed(monkeypatch, tmp_path) -> None:
    loaded = {
        "en_path": FakeDataset(["en1", "en2"]),
        "sv_path": FakeDataset(["sv1", "sv2", "sv3"]),
    }
    combined_holder = {}

    def fake_load(path):
        return loaded[str(path)]

    def fake_concat(datasets):
        values = []
        for dataset in datasets:
            values.extend(dataset.values)
        combined = FakeDataset(values)
        combined_holder["combined"] = combined
        return combined

    monkeypatch.setattr(build_dataset, "_load_from_disk", fake_load)
    monkeypatch.setattr(build_dataset, "_concatenate_datasets", fake_concat)

    manifest = build_dataset.build_combined_train_dataset(
        {"en": "en_path", "sv": "sv_path"},
        tmp_path / "train",
        shuffle_seed=2000,
    )

    assert manifest["source_lengths"] == {"en": 2, "sv": 3}
    assert manifest["combined_length"] == 5
    assert manifest["shuffle_seed"] == 2000
    assert combined_holder["combined"].values == ["en1", "en2", "sv1", "sv2", "sv3"]


def test_bilingual_config_loads_existing_training_fields(tmp_path) -> None:
    config_path = tmp_path / "config.json"
    config = {
        "model_name_or_path": "unsloth/LFM2.5-1.2B-Instruct",
        "use_lora": False,
        "use_qlora": False,
        "load_in_8bit": False,
        "lora_config": {},
        "dataset_name": "bilingual",
        "dataset_path": "ignored",
        "train_dataset_paths": {"en": "en_train", "sv": "sv_train"},
        "combined_train_dataset_path": "combined_train",
        "validation_dataset_path": "ignored",
        "eval_dataset_paths": {"en": "en_eval", "sv": "sv_eval"},
        "output_dir": "out",
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "num_train_epochs": 1,
        "learning_rate": 5e-5,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.05,
        "max_grad_norm": 1.0,
        "weight_decay": 0.01,
        "early_stopping_patience": 4,
        "max_steps": 2,
        "save_strategy": "steps",
        "eval_strategy": "steps",
        "logging_steps": 10,
        "save_steps": 200,
        "eval_steps": 200,
        "save_total_limit": 10,
        "fp16": False,
        "bf16": True,
        "gradient_checkpointing": True,
        "freeze_embeddings": True,
        "padding_free": True,
        "loss_type": "nll",
        "report_to": "tensorboard",
    }
    config_path.write_text(json.dumps(config), encoding="utf-8")

    loaded = BilingualTrainingConfig.from_json(config_path)

    assert loaded.dataset_path == "combined_train"
    assert loaded.validation_dataset_path is None
    assert loaded.train_dataset_paths == {"en": "en_train", "sv": "sv_train"}
    assert loaded.eval_dataset_paths == {"en": "en_eval", "sv": "sv_eval"}
    assert loaded.combined_train_shuffle_seed == 2000
    assert loaded.max_steps == 2


def _write_jsonl(path, rows) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_prompt_overlap_check_allows_duplicates_within_split(tmp_path) -> None:
    train_path = tmp_path / "train.jsonl"
    validation_path = tmp_path / "validation.jsonl"
    _write_jsonl(
        train_path,
        [
            {"prompt_id": "p1"},
            {"prompt_id": "p1"},
            {"prompt_id": "p2"},
        ],
    )
    _write_jsonl(validation_path, [{"prompt_id": "p3"}, {"prompt_id": "p3"}])

    summary = check_disjoint_prompt_ids([train_path], [validation_path])

    assert summary["train_unique_prompt_ids"] == 2
    assert summary["train_duplicate_rows"] == 1
    assert summary["validation_unique_prompt_ids"] == 1
    assert summary["validation_duplicate_rows"] == 1
    assert summary["overlap_unique_prompt_ids"] == 0


def test_prompt_overlap_check_rejects_cross_split_overlap(tmp_path) -> None:
    train_path = tmp_path / "train.jsonl"
    validation_path = tmp_path / "validation.jsonl"
    _write_jsonl(train_path, [{"prompt_id": "p1"}, {"prompt_id": "p1"}])
    _write_jsonl(validation_path, [{"prompt_id": "p1"}])

    with pytest.raises(ValueError, match="overlap detected"):
        check_disjoint_prompt_ids([train_path], [validation_path])
