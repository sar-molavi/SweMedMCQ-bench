"""Training data: JSONL loading, chat templating per student, and the data collators (padded and padding-free)."""

from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
import json

from datasets import Dataset, load_from_disk
from transformers import PreTrainedTokenizerBase
import torch
from torch.nn.utils.rnn import pad_sequence

from .prepare_answer_level_sft import main as answer_level_main
from .prepare_lfm_sft import main as lfm_main
from .prepare_eurollm_sft import main as eurollm_main


def _load_json_dataset(path: Path, max_num_samples: int | None) -> Dataset:
    """Load a JSONL dataset from a single file."""
    if not path.is_file():
        raise FileNotFoundError(f"Dataset path does not exist: {path}")

    max_num_samples = max_num_samples if max_num_samples is not None else 0
    records = []
    j = 1
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
            if j == max_num_samples:
                break
            j += 1
    return Dataset.from_list(records)


def _get_sft_preparer(model_name_or_path: str):
    """Return the chat-template preparer compatible with ``model_name_or_path``."""

    normalized_name = model_name_or_path.casefold()
    if "lfm2.5" in normalized_name:
        return lfm_main
    if "eurollm" in normalized_name:
        return eurollm_main

    raise ValueError(
        "No SFT dataset preparer is registered for model "
        f"{model_name_or_path!r}. Supported model families: LFM2.5 and EuroLLM."
    )


def get_dataset(
    path: str | Path,
    tokenizer: PreTrainedTokenizerBase,
    model_name_or_path: str,
    max_num_samples: int | None,
    max_length: int | None,
    batch_size:int = 10_000,
    num_proc:int=128,
    answer_level_sft: bool = False,
) -> Dataset:
    """Load JSONL data and apply the chat template for the training model."""

    path = Path(path)
    if path.is_dir():
        dataset = load_from_disk(str(path),keep_in_memory=True)
        if max_num_samples is not None:
            dataset = dataset.select(range(min(max_num_samples, len(dataset))))
        return dataset

    dataset = _load_json_dataset(path, max_num_samples=max_num_samples)
    preparer = (
        answer_level_main
        if answer_level_sft
        else _get_sft_preparer(model_name_or_path)
    )
    dataset = preparer(
        dataset=dataset,
        tokenizer=tokenizer,
        max_length=max_length,
        batch_size=batch_size,
        num_proc=num_proc,
    )
    return dataset


@dataclass
class DataCollatorForSupervisedFinetuning:
    pad_token_id: int
    max_length: int | None = None
    label_pad_token_id: int = -100

    def __call__(self, examples: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        if not examples:
            return {}

        # Extract sequences
        input_ids = [torch.tensor(e["input_ids"], dtype=torch.long) for e in examples]
        labels = [torch.tensor(e["labels"], dtype=torch.long) for e in examples]

        # Pad input_ids
        input_ids_padded = pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.pad_token_id,
        )

        # Pad labels (ignore loss on padding)
        labels_padded = pad_sequence(
            labels,
            batch_first=True,
            padding_value=self.label_pad_token_id,
        )

        # Attention mask: 1 for real tokens, 0 for padding
        attention_mask = input_ids_padded.ne(self.pad_token_id).long()

        if self.max_length is not None:
            input_ids_padded = input_ids_padded[:, : self.max_length]
            labels_padded = labels_padded[:, : self.max_length]
            attention_mask = attention_mask[:, : self.max_length]

        batch = {
            "input_ids": input_ids_padded,
            "labels": labels_padded,
            "attention_mask": attention_mask,
        }

        return batch


@dataclass
class PaddingFreeDataCollatorForSupervisedFinetuning:
    """Flatten a batch into one sequence for padding-free FlashAttention paths.

    Not used by the trainers: the TRL SFTTrainer uses TRL's own padding-free
    collator when ``SFTConfig.padding_free=True``.
    """

    pad_token_id: int
    max_length: int | None = None
    label_pad_token_id: int = -100

    def __call__(self, examples: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        if not examples:
            return {}

        flat_input_ids = []
        flat_labels = []
        flat_position_ids = []

        for example in examples:
            input_ids = list(example["input_ids"])
            labels = list(example["labels"])
            if len(input_ids) != len(labels):
                raise ValueError(
                    "Padding-free collation requires input_ids and labels to "
                    "have the same length."
                )

            if self.max_length is not None:
                input_ids = input_ids[: self.max_length]
                labels = labels[: self.max_length]

            position_ids = list(range(len(input_ids)))
            if labels:
                labels[0] = self.label_pad_token_id

            flat_input_ids.extend(input_ids)
            flat_labels.extend(labels)
            flat_position_ids.extend(position_ids)

        return {
            "input_ids": torch.tensor([flat_input_ids], dtype=torch.long),
            "labels": torch.tensor([flat_labels], dtype=torch.long),
            "position_ids": torch.tensor([flat_position_ids], dtype=torch.long),
        }


__all__ = [
    "get_dataset",
    "DataCollatorForSupervisedFinetuning",
    "PaddingFreeDataCollatorForSupervisedFinetuning",
]
