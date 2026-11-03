"""Shared mechanics for extracting decoded vLLM pickle outputs.

This module intentionally does not know about task schemas,
filename conventions, or output columns. Dataset packages provide those via a
strategy object.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datasets import Dataset
from transformers import AutoTokenizer
from tqdm import tqdm

from .utils import extract_reasoning_trace, parse_json_payload


@dataclass
class ExtractionConfig:
    """Common runtime configuration for extracting result pickle directories."""

    input_dir: Path
    output_path: Path
    model_name: str
    max_input_tokens: int | None = None
    response_format: str = "gpt_oss"
    pattern: str = "result_*.pkl"
    sample_index: int = 0


class ParsedPayloadStrategy:
    """Dataset-specific extraction behavior."""

    output_key: str

    @property
    def model(self) -> type[Any]:
        """Schema class used to validate the extracted JSON payload."""
        raise NotImplementedError

    def metadata_from_result_stem(self, stem: str) -> dict[str, str]:
        """Recover dataset metadata from a result filename stem."""
        raise NotImplementedError

    def candidates(
        self, decoded_text: str, response_format: str
    ) -> tuple[list[str], str]:
        """Return JSON candidates and the text shown in the output dataset."""
        raise NotImplementedError

    def parse(self, decoded_text: str, response_format: str) -> tuple[Any, str]:
        """Validate decoded text and return the parsed payload plus display text."""
        candidates, text = self.candidates(decoded_text, response_format)
        return parse_json_candidates(candidates, self.model), text

    def payload_fields(self, payload: dict[str, Any] | None) -> dict[str, Any]:
        """Map the validated payload into dataset-specific dataset columns."""
        raise NotImplementedError


class ResultExtractor:
    """Generic end-to-end extraction pipeline for one dataset strategy."""

    def __init__(self, config: ExtractionConfig, strategy: ParsedPayloadStrategy):
        self.config = config
        self.strategy = strategy
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    def read_data(self, path: str | Path) -> dict[str, object]:
        """Load one pickle result file and normalize it into a row."""
        path = Path(path)
        metadata = self.strategy.metadata_from_result_stem(path.stem)

        with path.open("rb") as handle:
            obj = pickle.load(handle)

        outputs = list(obj.outputs)
        if self.config.sample_index < 0 or self.config.sample_index >= len(outputs):
            raise ValueError(
                f"Requested sample_index={self.config.sample_index} from {path.name}, "
                f"but the result contains only {len(outputs)} completion(s)."
            )

        token_ids = list(outputs[self.config.sample_index].token_ids)
        if (
            self.config.max_input_tokens is not None
            and len(token_ids) > self.config.max_input_tokens
        ):
            print(
                f"Trimming token_ids for {path.name}: "
                f"{len(token_ids)} -> {self.config.max_input_tokens}",
                flush=True,
            )
            token_ids = token_ids[: self.config.max_input_tokens]

        decoded = self.tokenizer.decode(token_ids)
        parsed, text = self.strategy.parse(decoded, self.config.response_format)
        reasoning_text = extract_reasoning_trace(decoded, self.config.response_format)
        return {
            **metadata,
            "sample_index": self.config.sample_index,
            **self.strategy.payload_fields(parsed),
            f"{self.strategy.output_key}": parsed,
            "reasoning_text": reasoning_text,
            "text": text,
            "decoded_text": decoded,
        }

    def build_dataset(self) -> Dataset:
        """Build a Hugging Face dataset from all matching result files."""
        paths = [
            path
            for path in sorted(self.config.input_dir.glob(self.config.pattern))
            if path.is_file()
        ]
        records = [
            self.read_data(path)
            for path in tqdm(paths, desc="Extracting results", unit="file")
        ]
        return Dataset.from_list(records)

    def build_valid_dataset(self, dataset: Dataset) -> Dataset:
        """Return only rows whose structured payload was parsed successfully."""
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
        """Write ids for rows whose parsing failed."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            for record in dataset:
                if record[self.strategy.output_key] is not None:
                    continue
                dataset_name = record.get("dataset_name", "")
                realidx = record.get("realidx", "")
                handle.write(
                    f"{dataset_name}\t{realidx}\t{record['prompt_id']}\n"
                )

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


def parse_json_candidates(candidates: list[str], model: type[Any]) -> dict | None:
    """Validate candidate JSON strings against a schema model."""
    if not candidates:
        return None

    for candidate in candidates:
        if not candidate:
            continue
        parsed = parse_json_payload(candidate, model)
        if parsed is not None:
            return parsed
    return None
