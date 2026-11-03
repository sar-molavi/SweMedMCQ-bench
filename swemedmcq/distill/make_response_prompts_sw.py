"""Create Swedish response prompts from extracted translation outputs.

Unlike the English prompt builder, this script consumes the normalized output
produced by the translation extractor (``swemedmcq.translate.train_gptoss.extract``,
after the warning filter) rather than the source benchmark folders. The result is
written as a prompt JSONL file and, optionally, a pickled token set compatible with
``swemedmcq.common.vllm_runner``.
"""

from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from string import ascii_uppercase
from typing import Any, Sequence

from datasets import Dataset, load_from_disk
from transformers import AutoTokenizer

from ..common.prompts import SWE_RESPONSE_PROMPT


@dataclass
class SwedishResponsePromptConfig:
    """Configuration bundle for Swedish response prompt generation."""

    input_path: Path
    output_name: str
    output_dir: Path = Path("prompts")
    model_name: str | None = None
    prompt_template: str = SWE_RESPONSE_PROMPT
    prefix: str = "sw"


def parse_args() -> SwedishResponsePromptConfig:
    """Parse command-line arguments for Swedish prompt generation."""

    parser = argparse.ArgumentParser(
        description="Create Swedish response prompts from extracted translations."
    )
    parser.add_argument(
        "-i",
        "--input-path",
        type=Path,
        required=True,
        help="Path to the extracted translation JSON/JSONL file or HF dataset directory.",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="Model identifier used to instantiate the tokenizer.",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        required=True,
        help="Name identifier used in the prompt and token output filenames.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("prompts"),
        help="Output directory for prompt and token files.",
    )
    args = parser.parse_args()
    return SwedishResponsePromptConfig(
        input_path=args.input_path,
        output_name=args.output_name,
        output_dir=args.output_dir,
        model_name=args.model_name,
    )


def load_translations(path: Path) -> Dataset:
    """Load extracted translations from JSON/JSONL or a saved HF dataset."""

    if path.is_dir():
        return load_from_disk(str(path))

    suffix = path.suffix.lower()
    if suffix in {".json", ".jsonl"}:
        return Dataset.from_json(str(path), encoding="utf-8")

    raise ValueError(f"Unsupported input format: {path}")


def _choice_text(choice: Any) -> str:
    """A translated choice: plain text, or ``{"label", "text"}`` as written by the translation extractor."""
    if isinstance(choice, dict):
        return str(choice["text"])
    return str(choice)


def _normalize_choices(record: dict[str, Any]) -> list[str]:
    """Return the translated answer options as a plain ordered list."""

    choices = record.get("choices")
    if isinstance(choices, list):
        return [_choice_text(choice) for choice in choices]

    translation_output = record.get("translation_output")
    if isinstance(translation_output, dict):
        nested_choices = translation_output.get("choices")
        if isinstance(nested_choices, list):
            return [_choice_text(choice) for choice in nested_choices]

    raise ValueError(
        f"Missing translated choices for prompt_id={record.get('prompt_id', '<unknown>')}"
    )


def _normalize_question(record: dict[str, Any]) -> str:
    """Return the translated Swedish question text."""

    question = record.get("question")
    if isinstance(question, str):
        return question

    translation_output = record.get("translation_output")
    if isinstance(translation_output, dict) and isinstance(
        translation_output.get("question"), str
    ):
        return str(translation_output["question"])

    raise ValueError(
        f"Missing translated question for prompt_id={record.get('prompt_id', '<unknown>')}"
    )


def _prompt_id(record: dict[str, Any]) -> str:
    """Return the prompt id expected downstream by the batch runners."""

    prompt_id = record.get("prompt_id")
    if prompt_id:
        return str(prompt_id)

    dataset_name = record.get("dataset_name")
    realidx = record.get("realidx")
    if dataset_name is None or realidx is None:
        raise ValueError(f"Missing prompt metadata: {record}")
    return f"{dataset_name}_{realidx}"


def choices_to_str(choices: Sequence[str]) -> str:
    """Render ordered answer options with alphabetical labels."""

    if len(choices) > len(ascii_uppercase):
        raise ValueError("choices_to_str supports at most 26 answer options.")
    return "\n\n".join(
        f"{ascii_uppercase[index]}) {choice}" for index, choice in enumerate(choices)
    )


def make_prompt(
    record: dict[str, Any], *, prompt_template: str
) -> dict[str, object]:
    """Convert one translated example into the chat prompt payload."""

    question = _normalize_question(record)
    choices = _normalize_choices(record)
    options = choices_to_str(choices)
    prompt = prompt_template.format(options=options, question=question)
    return {
        "prompt_id": _prompt_id(record),
        "prompt": [{"content": prompt, "role": "user"}],
    }


def build_prompt_records(dataset: Dataset, prompt_template: str) -> list[dict[str, object]]:
    """Materialize the translated dataset as prompt records."""

    records = []
    for record in dataset:
        records.append(make_prompt(record, prompt_template=prompt_template))
    return records


def write_prompt_file(
    prompt_records: list[dict[str, object]], output_path: Path
) -> None:
    """Write prompt records to JSONL for use with batch runners."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for record in prompt_records:
            file.write(json.dumps(record, ensure_ascii=False))
            file.write("\n")


def write_token_file(
    prompt_records: list[dict[str, object]],
    config: SwedishResponsePromptConfig,
) -> None:
    """Write a pickled token map compatible with ``swemedmcq.common.vllm_runner``."""

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


def main(config: SwedishResponsePromptConfig) -> None:
    """Build prompt files from extracted Swedish translations."""

    config.output_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_translations(config.input_path)
    prompt_records = build_prompt_records(dataset, config.prompt_template)
    write_prompt_file(
        prompt_records,
        config.output_dir / f"{config.prefix}_{config.output_name}_prompt_set.jsonl",
    )

    if config.model_name is not None:
        write_token_file(prompt_records, config)


if __name__ == "__main__":
    main(parse_args())
