"""Create the training-split translation prompts from benchmark JSONL data."""

from __future__ import annotations

import argparse
import json
import pickle
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from datasets import Dataset, concatenate_datasets
from transformers import AutoTokenizer

from .prompt import FEW_SHOT_EXAMPLES, TRANSLATION_PROMPT


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
class TranslationPromptConfig:
    """Configuration bundle for training-split translation prompt generation."""

    input_dir: Path
    file_name: str
    output_name: str
    output_dir: Path = Path("prompts")
    datasets: Sequence[str] = field(default_factory=lambda: list(DEFAULT_DATASET_NAMES))
    prompt_template: str = TRANSLATION_PROMPT
    prefix: str = "translate_train"
    model_name: str | None = None


def parse_args() -> TranslationPromptConfig:
    """Parse command-line arguments for training-split prompt generation."""

    parser = argparse.ArgumentParser(
        description="Create training-split translation prompts and optional token sets."
    )
    parser.add_argument("-i", "--input-dir", type=Path, required=True)
    parser.add_argument("--file-name", required=True)
    parser.add_argument("--output-name", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("prompts"))
    parser.add_argument("--datasets", nargs="*", default=list(DEFAULT_DATASET_NAMES))
    parser.add_argument(
        "--model-name",
        default=None,
        help="Optional tokenizer model used to write a vLLM-compatible token set.",
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        default=None,
        help="Optional replacement template containing {FEW_SHOT_EXAMPLES} and {INPUT}.",
    )
    args = parser.parse_args()
    prompt_template = (
        args.prompt_file.read_text(encoding="utf-8").strip()
        if args.prompt_file is not None
        else TRANSLATION_PROMPT
    )
    return TranslationPromptConfig(
        input_dir=args.input_dir,
        file_name=args.file_name,
        output_name=args.output_name,
        output_dir=args.output_dir,
        datasets=args.datasets,
        prompt_template=prompt_template,
        model_name=args.model_name,
    )


def _ordered_options(options: Mapping[str, object]) -> list[tuple[str, object]]:
    """Sort numeric labels numerically while preserving lexical labels predictably."""

    def sort_key(item: tuple[str, object]) -> tuple[int, str]:
        label = item[0]
        try:
            return (0, f"{int(label):08d}")
        except (TypeError, ValueError):
            return (1, str(label))

    return sorted(options.items(), key=sort_key)


def _coerce_choices(options: object) -> list[dict[str, str]]:
    """Normalize mapping- and list-style benchmark options into ordered choices."""

    if isinstance(options, Mapping):
        return [
            {"label": str(label), "text": str(value)}
            for label, value in _ordered_options(options)
            if value is not None and str(value).strip()
        ]
    if isinstance(options, Sequence) and not isinstance(options, (str, bytes)):
        choices: list[dict[str, str]] = []
        for item in options:
            if isinstance(item, Mapping):
                label, text = item.get("label"), item.get("text")
                if label is None or text is None or not str(text).strip():
                    continue
                choices.append({"label": str(label), "text": str(text)})
            else:
                choices.append({"label": "", "text": str(item)})
        return choices
    raise TypeError(f"Unsupported options structure: {type(options)!r}")


def make_prompt(
    example: Mapping[str, object], *, prompt_template: str = TRANSLATION_PROMPT
) -> dict[str, object]:
    """Format one benchmark record into the training-split structured prompt."""

    payload = {
        "target_language": "Swedish",
        "question": example["question"],
        "choices": _coerce_choices(example["options"]),
    }
    prompt = prompt_template.format(
        FEW_SHOT_EXAMPLES=FEW_SHOT_EXAMPLES,
        INPUT=json.dumps(payload, ensure_ascii=False, indent=2),
    )
    return {"prompt": [{"content": prompt, "role": "user"}]}


def load_dataset(example_path: Path, config: TranslationPromptConfig) -> Dataset:
    """Load, normalize, and prompt one benchmark dataset file."""

    if not example_path.exists():
        raise FileNotFoundError(f"Expected dataset file not found: {example_path}")
    dataset = Dataset.from_json(str(example_path), encoding="utf-8")
    dataset = dataset.remove_columns(
        [
            column
            for column in dataset.column_names
            if column not in ["options", "question", "realidx"]
        ]
    ).rename_column("realidx", "prompt_id")
    dataset = dataset.map(
        lambda record: make_prompt(record, prompt_template=config.prompt_template)
    )
    dataset = dataset.remove_columns(
        [column for column in dataset.column_names if column not in ["prompt_id", "prompt"]]
    )
    return dataset.map(
        lambda row: {"prompt_id": f"{example_path.parent.name}_{row['prompt_id']}"}
    )


def build_dataset(config: TranslationPromptConfig) -> Dataset:
    """Build the combined training-split translation-prompt dataset."""

    return concatenate_datasets(
        [
            load_dataset(config.input_dir / name / f"{config.file_name}.jsonl", config)
            for name in config.datasets
        ]
    )


def write_prompt_file(dataset: Dataset, output_path: Path) -> list[dict[str, object]]:
    """Write materialized prompt records to JSONL and return them."""

    records = list(dataset)
    with output_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False))
            file.write("\n")
    return records


def write_token_file(
    prompt_records: list[dict[str, object]], config: TranslationPromptConfig
) -> None:
    """Write an optional vLLM-compatible pickled chat-template token map."""

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    prompt_map = {record["prompt_id"]: record["prompt"] for record in prompt_records}
    tokenized = tokenizer.apply_chat_template(
        list(prompt_map.values()), add_generation_prompt=True
    )
    with (
        config.output_dir / f"{config.prefix}_{config.output_name}_token_set.pkl"
    ).open("wb") as file:
        pickle.dump(dict(zip(prompt_map, tokenized)), file)


def main(config: TranslationPromptConfig) -> None:
    """Run the training-split translation prompt-generation pipeline."""

    config.output_dir.mkdir(parents=True, exist_ok=True)
    records = write_prompt_file(
        build_dataset(config),
        config.output_dir / f"{config.prefix}_{config.output_name}_prompt_set.jsonl",
    )
    if config.model_name is not None:
        write_token_file(records, config)


if __name__ == "__main__":
    main(parse_args())


__all__ = [
    "DEFAULT_DATASET_NAMES",
    "TranslationPromptConfig",
    "build_dataset",
    "load_dataset",
    "main",
    "make_prompt",
    "parse_args",
    "write_prompt_file",
    "write_token_file",
]
