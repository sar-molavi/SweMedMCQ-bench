"""Prepare EuroLLM-1.7B-Instruct SFT data using its native chat template.

Inputs contain question, reasoning, and answer columns. Like the LFM preparer,
mask the prompt and opening thinking block; train on reasoning, closing think
marker, answer, and the template's end-of-turn suffix. When reasoning is absent,
mask an empty thinking block and train only on the answer and turn suffix.
Thinking markers are ordinary text: no tokenizer vocabulary changes are needed.
"""

from __future__ import annotations

from typing import Any

from datasets import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase

MODEL_NAME = "utter-project/EuroLLM-1.7B-Instruct"
EMPTY_THINKING_TEXT = "<think>\n\n</think>\n\n"


def make_assistant_content(reasoning: str | None, answer: str) -> str:
    if isinstance(reasoning, str) and reasoning.strip():
        return f"<think>\n{reasoning.strip()}\n</think>\n\n{answer}"
    return EMPTY_THINKING_TEXT + answer


def build_messages(question: str, reasoning: str | None, answer: str) -> list[dict[str, str]]:
    return [
        {"role": "user", "content": question},
        {"role": "assistant", "content": make_assistant_content(reasoning, answer)},
    ]


def tokenize_batch(
    batch: dict[str, list[Any]],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int | None = None,
) -> dict[str, list[Any]]:
    """Batch tokenize rendered chats and mask by character offsets.

    Offsets avoid assuming that a separately tokenized thinking marker has the
    same token boundaries when followed by response text.
    """
    if not tokenizer.is_fast:
        raise ValueError("EuroLLM preprocessing requires a fast tokenizer for offsets.")
    if max_length is not None and max_length <= 0:
        raise ValueError("max_length must be positive.")
    output = {"input_ids": [], "labels": [], "num_trainable_tokens": []}
    texts, boundaries = [], []
    for question, reasoning, answer in zip(batch["question"], batch["reasoning"], batch["answer"]):
        messages = build_messages(question, reasoning, answer)
        prompt = tokenizer.apply_chat_template(
            messages[:1], tokenize=False, add_generation_prompt=True,
        )
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
        )
        thinking_prefix = (
            "<think>\n" if isinstance(reasoning, str) and reasoning.strip()
            else EMPTY_THINKING_TEXT
        )
        prefix = prompt + thinking_prefix
        if not text.startswith(prefix):
            raise ValueError("EuroLLM chat template does not preserve the expected assistant prefix.")
        texts.append(text)
        boundaries.append(len(prefix))
    if not texts:
        return output
    encoded = tokenizer(
        texts, add_special_tokens=False, return_offsets_mapping=True,
        truncation=max_length is not None, max_length=max_length,
        padding=False,
    )
    for ids, offsets, boundary in zip(encoded["input_ids"], encoded["offset_mapping"], boundaries):
        # A token crossing the boundary is masked to avoid supervising prompt text.
        labels = [token if start >= boundary and end > start else -100
                  for token, (start, end) in zip(ids, offsets)]
        count = sum(label != -100 for label in labels)
        output["input_ids"].append(ids)
        output["labels"].append(labels)
        output["num_trainable_tokens"].append(count)
    return output


def tokenize_example(
    question: str, reasoning: str | None, answer: str,
    tokenizer: PreTrainedTokenizerBase, max_length: int | None = None,
) -> dict[str, Any]:
    result = tokenize_batch(
        {"question": [question], "reasoning": [reasoning], "answer": [answer]},
        tokenizer, max_length,
    )
    return {key: values[0] for key, values in result.items()}


def prepare_dataset(
    dataset: Dataset, tokenizer: PreTrainedTokenizerBase,
    batch_size: int = 10_000, max_length: int | None = None,
    num_proc: int | None = None,
) -> Dataset:
    missing = {"question", "reasoning", "answer"} - set(dataset.column_names)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    workers = num_proc if num_proc is not None and num_proc > 1 else None
    processed = dataset.map(
        tokenize_batch, fn_kwargs={"tokenizer": tokenizer, "max_length": max_length},
        batched=True, batch_size=batch_size, num_proc=workers,
        remove_columns=list(dataset.column_names), desc="Tokenizing EuroLLM SFT examples",
    )
    return processed.filter(lambda row: row["num_trainable_tokens"] > 0, num_proc=workers)


def inspect_sample(dataset: Dataset, tokenizer: PreTrainedTokenizerBase, index: int = 0) -> None:
    if not 0 <= index < len(dataset):
        raise IndexError(f"index={index} is outside dataset of length {len(dataset)}")
    sample = dataset[index]
    print("FULL INPUT")
    print(tokenizer.decode(sample["input_ids"], skip_special_tokens=False))
    print("\nMASKED TOKENS (NO LOSS; repr shows whitespace)")
    print(repr(tokenizer.decode(
        [token for token, label in zip(sample["input_ids"], sample["labels"]) if label == -100],
        skip_special_tokens=False,
    )))
    print("\nTOKENS THAT CONTRIBUTE TO LOSS")
    print(tokenizer.decode([label for label in sample["labels"] if label != -100], skip_special_tokens=False))
    print(f"\nTrainable tokens: {sample['num_trainable_tokens']}")


def main(
    dataset: Dataset, tokenizer: PreTrainedTokenizerBase | None = None,
    max_length: int | None = None, batch_size: int = 10_000,
    num_proc: int | None = None, inspect: bool = False,
) -> Dataset:
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    processed = prepare_dataset(dataset, tokenizer, batch_size, max_length, num_proc)
    if inspect and len(processed):
        inspect_sample(processed, tokenizer)
    return processed


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    rows = [
        {"question": "What is 17 * 6?", "reasoning": "17 times 6 is 102.", "answer": "102"},
        *[{"question": "What is the chemical symbol for water?", "reasoning": value, "answer": "H₂O."}
          for value in (None, "", "   ")],
        {"question": "Vad är 8 + 5?", "reasoning": "Åtta plus fem är tretton.", "answer": "Svaret är 13."},
        {"question": "Quelle est la capitale de la France ?", "reasoning": None, "answer": "Paris."},
        {"question": "Explain 12 / 3 in two steps.", "reasoning": "  Split 12 into 3 equal groups.\nEach group contains 4.  ", "answer": "4\n\nCheck: 3 × 4 = 12."},
        {"question": "Return two bullet points.", "reasoning": "\n\t ", "answer": "- First\n- Second"},
        {"question": "Repeat the word with its leading whitespace.", "reasoning": None, "answer": "  hello"},
        {"question": "Return a JSON object with a true value.", "reasoning": "Use a JSON boolean.", "answer": '{"value": true}'},
    ]
    dataset = Dataset.from_list(rows)
    processed = main(dataset, tokenizer=tokenizer, max_length=4096, batch_size=2)
    assert len(processed) == len(rows)
    for index, row in enumerate(rows):
        sample = processed[index]
        print(f"\n{'=' * 60}\nEXAMPLE {index + 1}: reasoning={row['reasoning']!r}")
        inspect_sample(processed, tokenizer, index=index)
        assert sample == tokenize_example(**row, tokenizer=tokenizer, max_length=4096)
        assert len(sample["input_ids"]) == len(sample["labels"])
        trainable = [label for label in sample["labels"] if label != -100]
        assert len(trainable) == sample["num_trainable_tokens"]
        expected = (f"{row['reasoning'].strip()}\n</think>\n\n" if row["reasoning"] and row["reasoning"].strip() else "")
        expected += row["answer"] + "<|im_end|>\n"
        messages = build_messages(**row)
        assert sample["input_ids"] == tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False,
        )
        # Check the original rendered text, since decode() can insert spaces
        # around special tokens and need not round-trip the template exactly.
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        encoded = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
        boundary = len(rendered) - len(expected)
        assert rendered[boundary:] == expected
        first_trainable = next(i for i, label in enumerate(sample["labels"]) if label != -100)
        assert all(label == -100 for label in sample["labels"][:first_trainable])
        assert sample["labels"][first_trainable:] == sample["input_ids"][first_trainable:]
        start = encoded["offset_mapping"][first_trainable][0]
        assert start >= boundary
        # Allow uncovered whitespace, but never silently lose response content.
        assert not rendered[boundary:start].strip(), "Masking dropped response content"
        assert all(end <= boundary or not rendered[boundary:end].strip()
                   for _, end in encoded["offset_mapping"][:first_trainable])
        end_token = tokenizer.convert_tokens_to_ids("<|im_end|>")
        assert end_token in trainable, "Assistant end-of-turn token must be trainable"
    assert len(main(dataset, tokenizer=tokenizer, max_length=1)) == 0
    truncated = tokenize_example(**rows[0], tokenizer=tokenizer, max_length=len(processed[0]["input_ids"]) - 3)
    assert truncated["input_ids"] == processed[0]["input_ids"][:-3]
    assert truncated["labels"] == processed[0]["labels"][:-3]
    print(f"\n{'=' * 60}\nTRUNCATED EXAMPLE")
    inspect_sample(Dataset.from_list([truncated]), tokenizer)
    print("EuroLLM preprocessing checks passed.")
