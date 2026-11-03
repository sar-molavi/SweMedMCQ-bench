"""English explanation prompts for the teacher traces, one per training (or test) question of the source benchmarks."""

from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import dataclass, field
from collections.abc import Mapping
from pathlib import Path
from typing import Sequence

from datasets import Dataset, concatenate_datasets
from transformers import AutoTokenizer

from ..common.prompts import RESPONSE_PROMPT


# Benchmark datasets, in prompt-file order.
DEFAULT_DATASET_NAMES: Sequence[str] = (
    "medexqa",
    "mmlu-pro",
    "afrimedqa",
    "mmlu",
    "medqa",
    "medxpertqa-u",
    "medmcqa",
    "medxpertqa-r",
    "pubmedqa",
    "medbullets",
)


@dataclass
class ResponsePromptConfig:
    """Configuration bundle for response prompt generation."""

    input_dir: Path
    file_name: str
    output_name: str
    output_dir: Path = Path("prompts")
    datasets: Sequence[str] = field(default_factory=lambda: list(DEFAULT_DATASET_NAMES))
    response_prompt: str = RESPONSE_PROMPT
    prefix: str = "en"
    model_name: str = None


def parse_args() -> ResponsePromptConfig:
    parser = argparse.ArgumentParser(
        description="Create response prompts and token sets for MedAgents benchmark data."
    )
    parser.add_argument(
        "-i",
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing benchmark subsets (each with a test_hard.jsonl file).",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="Model identifier used to instantiate the tokenizer.",
    )
    parser.add_argument(
        "--file-name", type=str, required=True, help="The name of the JSONL file."
    )
    parser.add_argument(
        "--output-name",
        type=str,
        required=True,
        help="The name identifier of the output file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("prompts"),
        help="Output directory for prompt and token files (default: prompts).",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=list(DEFAULT_DATASET_NAMES),
        help="Optional list of dataset subdirectories to process.",
    )
    namespace = parser.parse_args()
    return ResponsePromptConfig(
        input_dir=namespace.input_dir,
        model_name=namespace.model_name,
        file_name=namespace.file_name,
        output_name=namespace.output_name,
        output_dir=namespace.output_dir,
        datasets=namespace.datasets,
    )


def choices_to_str(choices: Mapping[str, str | None]) -> str:
    """Render multiple-choice options as a labeled block of text."""
    return "\n\n".join(f"{key}) {value}" for key, value in choices.items() if value)


def make_prompt(
    example: Mapping[str, object], *, prompt_template: str
) -> dict[str, object]:
    """Format the example into the prompt structure required downstream."""
    options = choices_to_str(example["options"])
    prompt = prompt_template.format(options=options, question=example["question"])
    return {"prompt": [{"content": prompt, "role": "user"}]}


def load_dataset(example_path: Path, config: ResponsePromptConfig) -> Dataset:
    """Load and normalize a single benchmark subset."""
    if not example_path.exists():
        raise FileNotFoundError(f"Expected dataset file not found: {example_path}")

    dataset = Dataset.from_json(str(example_path), encoding="utf-8")

    dataset = dataset.remove_columns(
        [
            column
            for column in dataset.column_names
            if column not in ["options", "question", "realidx"]
        ]
    )

    dataset = dataset.rename_column("realidx", "prompt_id")
    dataset = dataset.map(
        lambda record: make_prompt(record, prompt_template=config.response_prompt)
    )

    keep_columns = [
        "prompt_id",
        "prompt",
    ]
    dataset = dataset.remove_columns(
        [column for column in dataset.column_names if column not in keep_columns]
    )
    dataset = dataset.map(
        lambda row: {"prompt_id": f"{example_path.parent.name}_{row['prompt_id']}"}
    )
    return dataset


def build_dataset(
    config: ResponsePromptConfig,
) -> Dataset:
    datasets = []
    for name in config.datasets:
        dataset_path = config.input_dir / name / f"{config.file_name}.jsonl"
        datasets.append(load_dataset(dataset_path, config))
    return concatenate_datasets(datasets)


def write_prompt_file(dataset: Dataset, output_path: Path) -> list[dict[str, object]]:
    """Write the HF dataset to JSONL and return the materialized records."""
    records = list(dataset)
    with output_path.open("wt", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False))
            file.write("\n")
    return records


def write_token_file(
    prompt_records: list[dict[str, object]],
    config: ResponsePromptConfig,
) -> None:
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    prompt_map = {record["prompt_id"]: record["prompt"] for record in prompt_records}
    tokenized = tokenizer.apply_chat_template(
        list(prompt_map.values()), add_generation_prompt=True
    )
    token_set = dict(zip(prompt_map.keys(), tokenized))

    output_path = (
        config.output_dir / f"{config.prefix}_{config.output_name}_token_set.pkl"
    )
    with output_path.open("wb") as file:
        pickle.dump(token_set, file)


def main(config: ResponsePromptConfig) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)

    dataset = build_dataset(config)

    prompt_records = write_prompt_file(
        dataset,
        config.output_dir / f"{config.prefix}_{config.output_name}_prompt_set.jsonl",
    )

    if config.model_name is not None:
        write_token_file(
            prompt_records,
            config,
        )


if __name__ == "__main__":
    config = parse_args()
    main(config)
