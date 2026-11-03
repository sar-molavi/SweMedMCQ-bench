"""Extraction of the translations from vLLM pickle outputs."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..test_gpt54.schema import MedicalMCQTranslationOutput
from ...common.result_io import (
    ExtractionConfig,
    ParsedPayloadStrategy,
    ResultExtractor,
)
from ...common.utils import (
    extract_final_answer_text,
    extract_final_block_gpt_oss,
    strip_generation_suffix_tokens,
)


@dataclass
class TranslationExtractionConfig(ExtractionConfig):
    """Runtime configuration for translation extraction."""

    task: str = "translation"


class TranslationMetadataMixin:
    """Parse translation result filenames: ``result_<dataset_name>_<realidx>``."""

    def metadata_from_result_stem(self, stem: str) -> dict[str, str]:
        prompt_id = stem.removeprefix("result_")
        if "_" not in prompt_id:
            raise ValueError(
                "translation result filename must be formatted as "
                f"result_<dataset_name>_<realidx>. Got: {stem}"
            )
        dataset_name, realidx = prompt_id.split("_", 1)
        return {
            "dataset_name": dataset_name,
            "realidx": realidx,
            "prompt_id": prompt_id,
        }


class TranslationStrategy(TranslationMetadataMixin, ParsedPayloadStrategy):
    """Strategy for ``MedicalMCQTranslationOutput`` translation records."""

    output_key = "translation_output"
    model = MedicalMCQTranslationOutput

    def candidates(
        self, decoded_text: str, response_format: str
    ) -> tuple[list[str], str]:
        extracted = extract_final_block_gpt_oss(decoded_text)
        text = extracted or decoded_text
        return [text], text

    def payload_fields(self, payload: dict[str, Any] | None) -> dict[str, Any]:
        payload = payload or {}
        return {
            "question": payload.get("question"),
            "choices": payload.get("choices"),
            "metadata": payload.get("metadata"),
        }


class RawResponseStrategy(TranslationMetadataMixin, ParsedPayloadStrategy):
    """Strategy for extracting plain decoded translation text with optional reasoning."""

    output_key = "final_answer"

    @property
    def model(self) -> type[Any]:
        raise NotImplementedError("Raw response extraction does not validate JSON.")

    def candidates(
        self, decoded_text: str, response_format: str
    ) -> tuple[list[str], str]:
        final_answer = extract_final_answer_text(decoded_text, response_format)
        text = final_answer or strip_generation_suffix_tokens(decoded_text)
        candidates = [text] if text else []
        return candidates, text

    def parse(self, decoded_text: str, response_format: str) -> tuple[Any, str]:
        final_answer = extract_final_answer_text(decoded_text, response_format)
        text = final_answer or strip_generation_suffix_tokens(decoded_text)
        return final_answer, text

    def payload_fields(self, payload: Any) -> dict[str, Any]:
        return {}


def build_strategy(task: str) -> ParsedPayloadStrategy:
    """Create the translation-specific strategy implementation."""
    if task == "translation":
        return TranslationStrategy()
    if task == "raw_response":
        return RawResponseStrategy()
    raise ValueError(f"Unsupported translation extraction task={task!r}.")


def parse_args(
    default_task: str | None = None, allow_task_override: bool = True
) -> TranslationExtractionConfig:
    """Parse the command-line interface for translation extraction."""
    parser = argparse.ArgumentParser(
        description="Extract translation outputs from a directory of vLLM pickle results."
    )
    parser.add_argument(
        "-i",
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing the saved vLLM pickle result files.",
    )
    parser.add_argument(
        "-o",
        "--output-path",
        type=Path,
        required=True,
        help="Output path for the extracted dataset.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Tokenizer model used to decode the saved token ids.",
    )
    if default_task is None or allow_task_override:
        parser.add_argument(
            "--task",
            type=str,
            required=default_task is None,
            default=default_task,
            choices=["translation", "raw_response"],
            help="Which translation extraction mode to use.",
        )
    parser.add_argument(
        "--response-format",
        type=str,
        default="gpt_oss",
        choices=["auto", "gpt_oss", "qwen", "medgemma"],
        help="Structured-output wrapper format used by response tasks.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="result_*.pkl",
        help="Glob pattern used to discover result pickle files.",
    )
    args = parser.parse_args()
    return TranslationExtractionConfig(
        input_dir=args.input_dir,
        output_path=args.output_path,
        model_name=args.model_name,
        task=getattr(args, "task", default_task),
        response_format=args.response_format,
        pattern=args.pattern,
    )


def main(config: TranslationExtractionConfig) -> None:
    """Run the configured translation extraction task."""
    extractor = ResultExtractor(config, build_strategy(config.task))
    extractor.run()


if __name__ == "__main__":
    main(parse_args())
