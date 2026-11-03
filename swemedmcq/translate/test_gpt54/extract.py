"""Extraction of the translations from OpenAI Batch JSONL outputs."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datasets import Dataset
from tqdm import tqdm

from .schema import MedicalMCQTranslationOutput

from ...common.schemas import ResponseOutput, SweResponseOutput

from ...common.result_io import parse_json_candidates
from ...common.utils import build_structured_output_candidates


@dataclass
class OpenAITranslationExtractionConfig:
    """Runtime configuration for OpenAI batch extraction."""

    input_path: Path
    output_path: Path
    task: str = "translation"
    response_format: str = "auto"


class OpenAITranslationMetadataMixin:
    """Parse prompt ids: ``<dataset_name>_<realidx>``."""

    def metadata_from_prompt_id(self, prompt_id: str) -> dict[str, str]:
        if "_" not in prompt_id:
            raise ValueError(
                "OpenAI custom_id must be formatted as "
                f"<dataset_name>_<realidx>. Got: {prompt_id}"
            )
        dataset_name, realidx = prompt_id.split("_", 1)
        return {
            "dataset_name": dataset_name,
            "realidx": realidx,
            "prompt_id": prompt_id,
        }


class TranslationStrategy(OpenAITranslationMetadataMixin):
    """Strategy for ``MedicalMCQTranslationOutput`` OpenAI records."""

    output_key = "translation_output"
    model = MedicalMCQTranslationOutput

    def parse(self, text: str, response_format: str) -> dict[str, Any] | None:
        candidates = build_structured_output_candidates(text, response_format)
        return parse_json_candidates(candidates, self.model)

    def payload_fields(self, payload: dict[str, Any] | None) -> dict[str, Any]:
        payload = payload or {}
        return {
            "question": payload.get("question"),
            "choices": payload.get("choices"),
            "metadata": payload.get("metadata"),
        }


class ResponseStrategy(OpenAITranslationMetadataMixin):
    """Strategy for English and Swedish MCQ response outputs."""

    output_key = "response_output"

    def __init__(
        self, model: type[ResponseOutput] | type[SweResponseOutput]
    ) -> None:
        self._model = model

    @property
    def model(self) -> type[Any]:
        return self._model

    def parse(self, text: str, response_format: str) -> dict[str, Any] | None:
        candidates = build_structured_output_candidates(text, response_format)
        return parse_json_candidates(candidates, self.model)

    def payload_fields(self, payload: dict[str, Any] | None) -> dict[str, Any]:
        payload = payload or {}
        return {
            "answer_index": payload.get("answer_index"),
            "explanation": payload.get("explanation"),
        }


class RawResponseStrategy(OpenAITranslationMetadataMixin):
    """Strategy for extracting plain OpenAI response text."""

    output_key = "final_answer"

    def parse(self, text: str, response_format: str) -> str | None:
        return text or None

    def payload_fields(self, payload: Any) -> dict[str, Any]:
        return {}


def build_strategy(
    task: str,
) -> TranslationStrategy | ResponseStrategy | RawResponseStrategy:
    """Create the OpenAI strategy implementation."""
    if task == "translation":
        return TranslationStrategy()
    if task == "en_response":
        return ResponseStrategy(ResponseOutput)
    if task == "sw_response":
        return ResponseStrategy(SweResponseOutput)
    if task == "raw_response":
        return RawResponseStrategy()
    raise ValueError(f"Unsupported OpenAI extraction task={task!r}.")


class OpenAIBatchResultExtractor:
    """End-to-end extraction pipeline for OpenAI Batch result JSONL."""

    def __init__(
        self,
        config: OpenAITranslationExtractionConfig,
        strategy: TranslationStrategy | ResponseStrategy | RawResponseStrategy,
    ) -> None:
        self.config = config
        self.strategy = strategy

    def read_rows(self) -> list[dict[str, Any]]:
        """Load one JSON object per non-empty JSONL line."""
        if not self.config.input_path.exists():
            raise FileNotFoundError(f"Input file not found: {self.config.input_path}")

        rows: list[dict[str, Any]] = []
        with self.config.input_path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                stripped = raw_line.strip()
                if not stripped:
                    continue
                try:
                    row = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON on line {line_number} of "
                        f"{self.config.input_path}: {exc}"
                    ) from exc
                if not isinstance(row, dict):
                    raise TypeError(
                        f"Expected JSON object on line {line_number} of "
                        f"{self.config.input_path}, got {type(row).__name__}."
                    )
                rows.append(row)
        return rows

    def read_data(self, row: dict[str, Any]) -> dict[str, Any]:
        """Normalize one OpenAI batch result row into an extracted dataset row."""
        prompt_id = self.extract_prompt_id(row)
        metadata = self.strategy.metadata_from_prompt_id(prompt_id)
        text = self.extract_output_text(row)
        parsed = (
            self.strategy.parse(text, self.config.response_format) if text else None
        )
        response = row.get("response") if isinstance(row.get("response"), dict) else {}
        body = response.get("body") if isinstance(response.get("body"), dict) else {}

        return {
            **metadata,
            **self.strategy.payload_fields(parsed),
            self.strategy.output_key: parsed,
            "reasoning_text": self.extract_reasoning_text(body),
            "text": text,
            "decoded_text": text,
            "response_status": response.get("status_code"),
            "response_id": body.get("id"),
            "request_id": response.get("request_id"),
            "error": self.extract_error(row),
        }

    @staticmethod
    def extract_prompt_id(row: dict[str, Any]) -> str:
        """Find the prompt id from the batch row or nested response metadata."""
        custom_id = row.get("custom_id")
        if isinstance(custom_id, str) and custom_id:
            return custom_id

        response = row.get("response") if isinstance(row.get("response"), dict) else {}
        body = response.get("body") if isinstance(response.get("body"), dict) else {}
        metadata = body.get("metadata") if isinstance(body.get("metadata"), dict) else {}
        prompt_id = metadata.get("prompt_id")
        if isinstance(prompt_id, str) and prompt_id:
            return prompt_id

        raise KeyError(f"OpenAI batch row is missing custom_id/prompt_id: {row}")

    @staticmethod
    def extract_output_text(row: dict[str, Any]) -> str | None:
        """Extract assistant output text from a Responses API batch result."""
        response = row.get("response") if isinstance(row.get("response"), dict) else {}
        body = response.get("body") if isinstance(response.get("body"), dict) else {}

        output_text = body.get("output_text")
        if isinstance(output_text, str) and output_text:
            return output_text

        parts: list[str] = []
        output_items = body.get("output")
        if isinstance(output_items, list):
            for output_item in output_items:
                if not isinstance(output_item, dict):
                    continue
                content_items = output_item.get("content")
                if not isinstance(content_items, list):
                    continue
                for content_item in content_items:
                    if not isinstance(content_item, dict):
                        continue
                    text = content_item.get("text")
                    if isinstance(text, str) and text:
                        parts.append(text)

        return "\n".join(parts) if parts else None

    @staticmethod
    def extract_reasoning_text(body: dict[str, Any]) -> str | None:
        """Extract requested OpenAI reasoning summaries, when present."""
        parts: list[str] = []
        output_items = body.get("output")
        if not isinstance(output_items, list):
            return None

        for output_item in output_items:
            if (
                not isinstance(output_item, dict)
                or output_item.get("type") != "reasoning"
            ):
                continue
            summaries = output_item.get("summary")
            if not isinstance(summaries, list):
                continue
            for summary in summaries:
                if not isinstance(summary, dict):
                    continue
                text = summary.get("text")
                if isinstance(text, str) and text:
                    parts.append(text)

        return "\n\n".join(parts) if parts else None

    @staticmethod
    def extract_error(row: dict[str, Any]) -> Any:
        """Return top-level or nested OpenAI error details, when present."""
        if row.get("error") is not None:
            return row.get("error")

        response = row.get("response") if isinstance(row.get("response"), dict) else {}
        body = response.get("body") if isinstance(response.get("body"), dict) else {}
        return body.get("error")

    def build_dataset(self) -> Dataset:
        """Build a Hugging Face dataset from all OpenAI batch result rows."""
        records = [
            self.read_data(row)
            for row in tqdm(self.read_rows(), desc="Extracting results", unit="row")
        ]
        return Dataset.from_list(records)

    def build_valid_dataset(self, dataset: Dataset) -> Dataset:
        """Return only rows whose task payload was extracted successfully."""
        return dataset.filter(
            lambda record: record[self.strategy.output_key] is not None
        )

    @staticmethod
    def write_dataset(dataset: Dataset, output_path: Path) -> None:
        """Persist the dataset as JSON/JSONL or a Hugging Face dataset directory."""
        if output_path.suffix.lower() in {".json", ".jsonl"}:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            dataset.to_json(str(output_path), force_ascii=False)
            return

        output_path.mkdir(parents=True, exist_ok=True)
        dataset.save_to_disk(str(output_path))

    @staticmethod
    def make_valid_output_path(output_path: Path) -> Path:
        """Return the sibling path used for the valid-only dataset."""
        if output_path.suffix:
            return output_path.with_name(
                f"{output_path.stem}_valid{output_path.suffix}"
            )
        return output_path.with_name(f"{output_path.name}_valid")

    @staticmethod
    def make_failure_report_path(output_path: Path) -> Path:
        """Return the sibling path used for the failed-row text report."""
        if output_path.suffix:
            return output_path.with_name(f"{output_path.stem}_failed.txt")
        return output_path.with_name(f"{output_path.name}_failed.txt")

    def write_failure_report(self, dataset: Dataset, output_path: Path) -> None:
        """Write ids for rows whose task payload could not be extracted."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            for record in dataset:
                if record[self.strategy.output_key] is not None:
                    continue
                dataset_name = record.get("dataset_name", "")
                realidx = record.get("realidx", "")
                handle.write(f"{dataset_name}\t{realidx}\t{record['prompt_id']}\n")

    def run(self) -> None:
        """Execute the extraction pipeline."""
        dataset = self.build_dataset()
        valid_dataset = self.build_valid_dataset(dataset)
        self.write_dataset(dataset, self.config.output_path)
        self.write_dataset(
            valid_dataset, self.make_valid_output_path(self.config.output_path)
        )
        self.write_failure_report(
            dataset, self.make_failure_report_path(self.config.output_path)
        )


def parse_args(
    default_task: str | None = "translation", allow_task_override: bool = True
) -> OpenAITranslationExtractionConfig:
    """Parse the command-line interface for OpenAI extraction."""
    parser = argparse.ArgumentParser(
        description="Extract translation outputs from an OpenAI Batch result JSONL file."
    )
    parser.add_argument(
        "-i",
        "--input-path",
        type=Path,
        required=True,
        help="OpenAI Batch result JSONL file.",
    )
    parser.add_argument(
        "-o",
        "--output-path",
        type=Path,
        required=True,
        help="Output path for the extracted dataset.",
    )
    if default_task is None or allow_task_override:
        parser.add_argument(
            "--task",
            type=str,
            required=default_task is None,
            default=default_task,
            choices=["translation", "en_response", "sw_response", "raw_response"],
            help="Which OpenAI extraction mode to use.",
        )
    parser.add_argument(
        "--response-format",
        type=str,
        default="auto",
        choices=["auto", "gpt_oss", "qwen", "medgemma"],
        help="Structured-output wrapper format used by response tasks.",
    )
    args = parser.parse_args()
    return OpenAITranslationExtractionConfig(
        input_path=args.input_path,
        output_path=args.output_path,
        task=getattr(args, "task", default_task),
        response_format=args.response_format,
    )


def main(config: OpenAITranslationExtractionConfig) -> None:
    """Run the configured OpenAI extraction task."""
    extractor = OpenAIBatchResultExtractor(config, build_strategy(config.task))
    extractor.run()


if __name__ == "__main__":
    main(parse_args())
