"""Prepare a Hugging Face dataset for LiquidAI/LFM2.5-1.2B-Instruct SFT.

The input dataset must contain ``question``, ``reasoning``, and ``answer``.
``reasoning`` may be null, empty, or whitespace-only. In that case an empty,
masked thinking block disables reasoning and only the answer is trainable.

For samples with reasoning, loss is calculated on::

    reasoning</think>answer<|im_end|>

For samples without reasoning, loss is calculated on::

    answer<|im_end|>

The user message, ``<|im_start|>assistant\n`` generation header, and opening
``<think>`` marker are masked with ``-100``. Registering ``<think>`` as a
special token makes that final masked marker a single token ID.
For no-reasoning samples, ``<think>\n\n</think>\n\n`` is also masked.
"""

from __future__ import annotations

from typing import Any

from datasets import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase


MODEL_NAME = "LiquidAI/LFM2.5-1.2B-Instruct"
QUESTION_COLUMN = "question"
REASONING_COLUMN = "reasoning"
ANSWER_COLUMN = "answer"
EMPTY_THINKING_TEXT = "<think>\n\n</think>\n\n"
CHAT_TEMPLATE_KWARGS = {
    "tokenize": True,
    "return_dict": False,
    "preserve_thinking": False,
    "enable_thinking": True,
}


def _has_reasoning(reasoning: str | None) -> bool:
    return isinstance(reasoning, str) and bool(reasoning.strip())


def make_assistant_content(reasoning: str | None, answer: str) -> str:
    if _has_reasoning(reasoning):
        return f"<think>\n{reasoning.strip()}\n</think>\n\n{answer}"
    return f"{EMPTY_THINKING_TEXT}{answer}"


def build_messages(
    question: str,
    reasoning: str | None,
    answer: str,
) -> list[dict[str, Any]]:
    """Build the user/assistant conversation expected by the LFM template."""

    return [
        {"role": "user", "content": question},
        {
            "role": "assistant",
            "content": make_assistant_content(reasoning, answer),
        },
    ]


def tokenize_example(
    question: str,
    reasoning: str | None,
    answer: str,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int | None = None,
) -> dict[str, Any]:
    """Tokenize one example and mask the prompt plus the opening think marker."""

    has_reasoning = _has_reasoning(reasoning)
    prompt_messages = [{"role": "user", "content": question}]
    template_kwargs = {
        "tokenize": True,
        "return_dict": False,
        # The current assistant is always the last assistant turn, but making
        # this explicit ensures prior turns never retain hidden thinking.
        "preserve_thinking": False,
        # The official LFM template currently ignores this argument. Passing
        # it documents the intended behavior and supports a future customized
        # template that consumes the switch.
        "enable_thinking": has_reasoning,
    }

    prompt_ids = list(
        tokenizer.apply_chat_template(
            prompt_messages,
            add_generation_prompt=True,
            **template_kwargs,
        )
    )
    input_ids = list(
        tokenizer.apply_chat_template(
            build_messages(question, reasoning, answer),
            add_generation_prompt=False,
            **template_kwargs,
        )
    )

    if len(input_ids) < len(prompt_ids):
        raise ValueError(
            "Full sequence is shorter than the prompt sequence. "
            "The LFM chat-template assumptions are invalid."
        )
    if input_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError(
            "Full sequence does not begin with the generation prompt. "
            "The LFM chat-template assumptions are invalid."
        )

    masked_thinking_prefix_length = 0
    if has_reasoning:
        think_start_ids = list(tokenizer.encode("<think>\n", add_special_tokens=False))
        if not think_start_ids:
            raise ValueError("Tokenizer produced no token IDs for '<think>\\n'.")

        # A future template may add <think> to its generation prompt. The
        # current official LFM template places it immediately after the prompt
        # instead. Both layouts mask the same marker.
        if prompt_ids[-len(think_start_ids) :] == think_start_ids:
            pass
        elif input_ids[
            len(prompt_ids) : len(prompt_ids) + len(think_start_ids)
        ] == think_start_ids:
            masked_thinking_prefix_length = len(think_start_ids)
        else:
            raise ValueError(
                "The LFM assistant response does not begin with the '<think>' "
                "token sequence expected for a sample with reasoning."
            )
    else:
        empty_thinking_ids = list(
            tokenizer.encode(EMPTY_THINKING_TEXT, add_special_tokens=False)
        )
        if not empty_thinking_ids:
            raise ValueError("Tokenizer produced no token IDs for empty thinking.")

        if prompt_ids[-len(empty_thinking_ids) :] == empty_thinking_ids:
            # A future custom template can render the disabled-thinking block
            # as part of the generation prompt.
            pass
        elif input_ids[
            len(prompt_ids) : len(prompt_ids) + len(empty_thinking_ids)
        ] == empty_thinking_ids:
            # A future custom template can render it in the full assistant
            # message but not in the generation prompt.
            prompt_ids.extend(empty_thinking_ids)
        else:
            # The official LFM template currently omits empty thinking. Insert
            # it into both sequences so it is model context but never a label.
            prompt_ids.extend(empty_thinking_ids)
            input_ids = (
                input_ids[: len(prompt_ids) - len(empty_thinking_ids)]
                + empty_thinking_ids
                + input_ids[len(prompt_ids) - len(empty_thinking_ids) :]
            )

        if input_ids[: len(prompt_ids)] != prompt_ids:
            raise ValueError(
                "The empty thinking block is not aligned with the LFM prompt."
            )

    labels = input_ids.copy()
    labels[: len(prompt_ids)] = [-100] * len(prompt_ids)
    if masked_thinking_prefix_length:
        start = len(prompt_ids)
        end = start + masked_thinking_prefix_length
        labels[start:end] = [-100] * masked_thinking_prefix_length

    if max_length is not None:
        input_ids = input_ids[:max_length]
        labels = labels[:max_length]

    num_trainable_tokens = sum(label != -100 for label in labels)

    return {
        "input_ids": input_ids,
        "labels": labels,
        "num_trainable_tokens": num_trainable_tokens,
    }


def tokenize_batch(
    batch: dict[str, list[Any]],
    tokenizer: PreTrainedTokenizerBase,
    think_start_length: int,
    empty_thinking_ids: list[int],
    empty_thinking_length: int,
    max_length: int | None = None,
) -> dict[str, list[Any]]:
    """Tokenize and mask one batched ``datasets.Dataset.map`` input."""

    questions = batch[QUESTION_COLUMN]
    reasonings = batch[REASONING_COLUMN]
    answers = batch[ANSWER_COLUMN]

    prompt_messages: list[list[dict[str, Any]]] = []
    input_messages: list[list[dict[str, Any]]] = []
    has_reasonings: list[bool] = []
    for question, reasoning, answer in zip(questions, reasonings, answers):
        has_reasoning = _has_reasoning(reasoning)
        has_reasonings.append(has_reasoning)
        user_message = {"role": "user", "content": question}
        prompt_messages.append([user_message])
        input_messages.append(
            [
                user_message,
                {
                    "role": "assistant",
                    "content": make_assistant_content(reasoning, answer),
                },
            ]
        )

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
    if len(questions) == 1:
        if prompt_id_batch and isinstance(prompt_id_batch[0], int):
            prompt_id_batch = [prompt_id_batch]
        if input_id_batch and isinstance(input_id_batch[0], int):
            input_id_batch = [input_id_batch]

    output: dict[str, list[Any]] = {
        "input_ids": [],
        "labels": [],
        "num_trainable_tokens": [],
    }

    for prompt_ids, input_ids, has_reasoning in zip(
        prompt_id_batch,
        input_id_batch,
        has_reasonings,
    ):
        prompt_length = len(prompt_ids)
        masked_prefix_length = prompt_length
        if has_reasoning:
            masked_prefix_length += think_start_length
        else:
            masked_prefix_length += empty_thinking_length

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
    """Apply LFM formatting, tokenization, masking, and empty-sample filtering."""

    required_columns = {QUESTION_COLUMN, REASONING_COLUMN, ANSWER_COLUMN}
    missing_columns = required_columns - set(dataset.column_names)
    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: {sorted(missing_columns)}"
        )
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    think_start_ids = list(tokenizer.encode("<think>\n", add_special_tokens=False))
    empty_thinking_ids = list(
        tokenizer.encode(EMPTY_THINKING_TEXT, add_special_tokens=False)
    )
    if not think_start_ids:
        raise ValueError("Tokenizer produced no token IDs for '<think>\\n'.")
    if not empty_thinking_ids:
        raise ValueError("Tokenizer produced no token IDs for empty thinking.")

    map_num_proc = num_proc if num_proc is not None and num_proc > 1 else None
    processed_dataset = dataset.map(
        lambda batch: tokenize_batch(
            batch,
            tokenizer,
            len(think_start_ids),
            empty_thinking_ids,
            len(empty_thinking_ids),
            max_length,
        ),
        batched=True,
        batch_size=batch_size,
        remove_columns=list(dataset.column_names),
        num_proc=map_num_proc,
        desc="Tokenizing LFM SFT examples",
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

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

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
    dataset = Dataset.from_list(
        [
            {
                "question": "What is 17 * 6?",
                "reasoning": "17 times 6 is 102.",
                "answer": "102",
            },
            {
                "question": "What is the chemical symbol for water?",
                "reasoning": None,
                "answer": "H₂O.",
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
