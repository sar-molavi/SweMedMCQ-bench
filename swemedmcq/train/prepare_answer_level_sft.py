"""Chat formatting and label masking of the answer-level training data (the answer JSON is the only target)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
import json

from datasets import Dataset

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase
else:
    PreTrainedTokenizerBase = Any

PROMPT_COLUMN = "prompt"
ANSWER_COLUMN = "gold_answer"
CHAT_TEMPLATE_KWARGS = {
    "tokenize": True,
    "return_dict": False,
    "enable_thinking": False,
}


def make_assistant_content(answer: str) -> str:
    return json.dumps(
        {"answer_index": str(answer).strip().upper()},
        ensure_ascii=False,
        separators=(",", ":"),
    )


def build_messages(
    prompt: list[dict[str, Any]],
    answer: str,
) -> list[dict[str, Any]]:
    """Build the user/assistant conversation."""

    messages = [dict(message) for message in prompt]
    if not messages:
        raise ValueError("prompt must contain at least one message")
    messages.append({"role": "assistant", "content": make_assistant_content(answer)})
    return messages


def tokenize_batch(
    batch: dict[str, list[Any]],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int | None = None,
) -> dict[str, list[Any]]:
    """Tokenize and mask one batched ``datasets.Dataset.map`` input."""

    prompts = batch[PROMPT_COLUMN]
    answers = batch[ANSWER_COLUMN]

    prompt_messages: list[list[dict[str, Any]]] = []
    input_messages: list[list[dict[str, Any]]] = []

    for prompt, answer in zip(prompts, answers):
        prompt_messages.append(prompt)
        input_messages.append(build_messages(prompt, answer))

    prompt_id_batch = tokenizer.apply_chat_template(
        prompt_messages,
        add_generation_prompt=True,
        **CHAT_TEMPLATE_KWARGS,
    )
    input_id_batch = tokenizer.apply_chat_template(
        input_messages,
        add_generation_prompt=False,
        **CHAT_TEMPLATE_KWARGS,
    )
    if len(prompts) == 1:
        if prompt_id_batch and isinstance(prompt_id_batch[0], int):
            prompt_id_batch = [prompt_id_batch]
        if input_id_batch and isinstance(input_id_batch[0], int):
            input_id_batch = [input_id_batch]

    output: dict[str, list[Any]] = {
        "input_ids": [],
        "labels": [],
        "num_trainable_tokens": [],
    }

    for prompt_ids, input_ids in zip(
        prompt_id_batch,
        input_id_batch,
    ):
        masked_prefix_length = len(prompt_ids)

        labels = input_ids.copy()
        labels[:masked_prefix_length] = [-100] * masked_prefix_length

        if max_length is not None:
            input_ids = input_ids[:max_length]
            labels = labels[:max_length]
            num_trainable_tokens = sum(label != -100 for label in labels)
            if num_trainable_tokens == 0:
                continue
        else:
            num_trainable_tokens = len(labels) - masked_prefix_length

        output["input_ids"].append(input_ids)
        output["labels"].append(labels)
        output["num_trainable_tokens"].append(num_trainable_tokens)

    return output


def prepare_dataset(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    batch_size: int = 10_000,
    max_length: int | None = None,
    num_proc: int | None = None,
) -> Dataset:
    """Apply formatting, tokenization, masking, and empty-sample filtering."""

    required_columns = {PROMPT_COLUMN, ANSWER_COLUMN}
    missing_columns = required_columns - set(dataset.column_names)
    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: {sorted(missing_columns)}"
        )
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    map_num_proc = num_proc if num_proc is not None and num_proc > 1 else None
    processed_dataset = dataset.map(
        lambda batch: tokenize_batch(
            batch,
            tokenizer,
            max_length,
        ),
        batched=True,
        batch_size=batch_size,
        remove_columns=list(dataset.column_names),
        num_proc=map_num_proc,
        desc="Tokenizing answer-level SFT examples",
    )
    return processed_dataset


def inspect_sample(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    index: int = 0,
) -> None:
    """Print a processed sample and the tokens which contribute to loss."""

    if not 0 <= index < len(dataset):
        raise IndexError(f"index={index} is outside dataset of length {len(dataset)}")
    sample = dataset[index]
    trainable_ids = [
        token_id
        for token_id, label in zip(sample["input_ids"], sample["labels"])
        if label != -100
    ]
    print("FULL INPUT")
    print(tokenizer.decode(sample["input_ids"], skip_special_tokens=False))
    print("\nTOKENS THAT CONTRIBUTE TO LOSS")
    print(tokenizer.decode(trainable_ids, skip_special_tokens=False))
    print(f"\nTrainable tokens: {sample['num_trainable_tokens']}")


def main(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase | None = None,
    max_length: int | None = None,
    batch_size: int = 10_000,
    num_proc: int | None = None,
    inspect: bool = False,
) -> Dataset:
    """Main LFM preprocessing entry point."""

    processed_dataset = prepare_dataset(
        dataset=dataset,
        tokenizer=tokenizer,
        max_length=max_length,
        batch_size=batch_size,
        num_proc=num_proc,
    )
    if inspect and len(processed_dataset) > 0:
        inspect_sample(processed_dataset, tokenizer)
    return processed_dataset


if __name__ == "__main__":
    from transformers import AutoTokenizer

    MODEL_NAME = "unsloth/LFM2.5-1.2B-Instruct"
    dataset = Dataset.from_list(
        [
            {
                "prompt": [{"role": "user", "content": "What is 17 * 6?"}],
                "gold_answer": "A",
            },
            {
                "prompt": [{"role": "user", "content": "What is the chemical symbol for water?"}],
                "gold_answer": "B",
            },
        ]
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    processed_dataset = main(
        dataset,
        tokenizer=tokenizer,
        max_length=4096,
        inspect=True,
    )
    inspect_sample(processed_dataset, tokenizer, index=1)
