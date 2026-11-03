"""Flatten prompt-clustered MCQ SFT JSONL into the training schema of swemedmcq.train.

Input rows are prompt-level records with a ``responses`` list. Output rows are
example-level records with ``question``, ``reasoning``, and ``answer`` fields,
which are the columns consumed by the student SFT preparers.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

from ..common.schemas import ResponseOutput


DEFAULT_SOURCE_PREFERENCE = (
    "en_response_train_gpt_oss_120b_t=0.7",
    "en_response_train_gpt_oss_120b",
    "en_response_train_qwen_3.6_35B_A3B_t=0.7",
    "en_response_train_qwen_3.6_35B_A3B",
    "en_response_train_gpt_oss_120b_t=0.7_v2",
)


@dataclass
class ExportStats:
    input_prompts: int = 0
    skipped_prompts: int = 0
    output_examples: int = 0
    examples_with_reasoning: int = 0
    source_counts: Counter[str] | None = None
    dataset_counts: Counter[str] | None = None

    def __post_init__(self) -> None:
        self.source_counts = Counter()
        self.dataset_counts = Counter()

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_prompts": self.input_prompts,
            "skipped_prompts": self.skipped_prompts,
            "output_examples": self.output_examples,
            "examples_with_reasoning": self.examples_with_reasoning,
            "source_counts": dict(sorted((self.source_counts or Counter()).items())),
            "dataset_counts": dict(sorted((self.dataset_counts or Counter()).items())),
        }


def load_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=True, indent=2, sort_keys=True)
        handle.write("\n")


def prompt_text(record: dict[str, Any]) -> str:
    prompt = record.get("prompt")
    if not isinstance(prompt, list):
        raise ValueError(f"{record.get('prompt_id')}: prompt must be a list of messages")
    user_messages = [msg for msg in prompt if msg.get("role") == "user"]
    if len(user_messages) != 1:
        raise ValueError(
            f"{record.get('prompt_id')}: expected exactly one user message, got {len(user_messages)}"
        )
    content = user_messages[0].get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError(f"{record.get('prompt_id')}: user message content is empty")
    return content


def response_is_usable(response: dict[str, Any]) -> bool:
    return bool(str(response.get("explanation") or "").strip()) and bool(
        str(response.get("final_answer") or "").strip()
    )


def response_sort_key(
    response: dict[str, Any],
    source_rank: dict[str, int],
) -> tuple[int, int, int, str]:
    source_run = str(response.get("source_run") or "")
    rank = source_rank.get(source_run, len(source_rank))
    explanation_length = int(response.get("explanation_word_count") or 0)
    reasoning_length = int(response.get("reasoning_word_count") or 0)
    return (rank, explanation_length + reasoning_length, int(response.get("sample_index") or 0), source_run)


def selected_responses(
    record: dict[str, Any],
    selection: str,
    source_preference: Sequence[str],
) -> list[dict[str, Any]]:
    responses = [response for response in record.get("responses", []) if response_is_usable(response)]
    if selection == "all":
        return responses
    if selection != "preferred-one":
        raise ValueError(f"Unsupported selection mode: {selection}")
    if not responses:
        return []
    source_rank = {source_run: index for index, source_run in enumerate(source_preference)}
    return [min(responses, key=lambda response: response_sort_key(response, source_rank))]


def make_answer(response: dict[str, Any]) -> str:
    explanation = str(response["explanation"]).strip()
    final_answer = str(response["final_answer"]).strip().upper()
    answer = ResponseOutput.model_validate(
        {"explanation": explanation, "answer_index": final_answer}
    )
    return json.dumps(
        answer.model_dump(),
        ensure_ascii=True,
        indent=2,
    )


def make_example(record: dict[str, Any], response: dict[str, Any]) -> dict[str, Any]:
    reasoning = response.get("reasoning")
    if not isinstance(reasoning, str) or not reasoning.strip():
        reasoning = None
    else:
        reasoning = reasoning.strip()

    return {
        "question": prompt_text(record),
        "reasoning": reasoning,
        "answer": make_answer(response),
        "prompt_id": record.get("prompt_id"),
        "dataset_name": record.get("dataset_name"),
        "gold_answer": record.get("gold_answer"),
        "source_model": response.get("source_model"),
        "source_run": response.get("source_run"),
        "sample_index": response.get("sample_index"),
    }


def flatten_file(
    input_path: Path,
    output_path: Path,
    selection: str,
    source_preference: Sequence[str],
) -> ExportStats:
    stats = ExportStats()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output:
        for record in load_jsonl(input_path):
            stats.input_prompts += 1
            responses = selected_responses(record, selection, source_preference)
            if not responses:
                stats.skipped_prompts += 1
                continue
            for response in responses:
                example = make_example(record, response)
                output.write(json.dumps(example, ensure_ascii=True) + "\n")
                stats.output_examples += 1
                if example["reasoning"] is not None:
                    stats.examples_with_reasoning += 1
                stats.source_counts[str(example["source_run"])] += 1
                stats.dataset_counts[str(example["dataset_name"])] += 1
    return stats


def parse_source_preference(values: Iterable[str] | None) -> tuple[str, ...]:
    if not values:
        return DEFAULT_SOURCE_PREFERENCE
    result = tuple(value for value in values if value)
    if not result:
        raise ValueError("source preference cannot be empty")
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Flatten prompt-clustered SFT split JSONL files for swemedmcq.train."
    )
    parser.add_argument("--train-input", type=Path, default=Path("work/sft/en/splits/train.jsonl"))
    parser.add_argument(
        "--validation-input",
        type=Path,
        default=Path("work/sft/en/splits/validation.jsonl"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("work/sft/en/splits/flat"))
    parser.add_argument(
        "--selection",
        choices=("all", "preferred-one"),
        default="all",
        help="Flatten every retained response by default; preferred-one is an optional downsampling mode.",
    )
    parser.add_argument(
        "--source-preference",
        action="append",
        help="Source run priority for preferred-one mode. Repeat in priority order.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    source_preference = parse_source_preference(args.source_preference)

    train_output = args.output_dir / "train.jsonl"
    validation_output = args.output_dir / "validation.jsonl"
    train_stats = flatten_file(args.train_input, train_output, args.selection, source_preference)
    validation_stats = flatten_file(
        args.validation_input,
        validation_output,
        args.selection,
        source_preference,
    )

    manifest = {
        "selection": args.selection,
        "source_preference": list(source_preference),
        "train": {
            "input_path": str(args.train_input),
            "output_path": str(train_output),
            **train_stats.to_dict(),
        },
        "validation": {
            "input_path": str(args.validation_input),
            "output_path": str(validation_output),
            **validation_stats.to_dict(),
        },
    }
    write_json(args.output_dir / "manifest.json", manifest)

    print(f"Wrote {train_stats.output_examples} train examples to {train_output}")
    print(f"Wrote {validation_stats.output_examples} validation examples to {validation_output}")
    print(f"Wrote manifest to {args.output_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
