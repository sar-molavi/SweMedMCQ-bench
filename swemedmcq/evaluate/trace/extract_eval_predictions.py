"""Convert vLLM result pickles into JSONL rows consumed by ``sft.eval``.

The source JSONL supplies stable evaluation metadata, while each trusted pickle
supplies one generated completion. Result filenames must follow
``result_<dataset_name>_<prompt_id>.pkl``.
"""

from __future__ import annotations

import argparse
import json
import pickle
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from ...common.schemas import ResponseOutput
from ...common.utils import (
    build_structured_output_candidates,
    parse_json_payload,
    split_qwen_reasoning_content,
    strip_generation_suffix_tokens,
)


class TokenDecoder(Protocol):
    """Describe the tokenizer method required by the extraction pipeline."""

    def decode(self, token_ids: list[int], **kwargs: Any) -> str:
        """Convert generated token IDs into text."""
        ...


@dataclass(frozen=True)
class SourceRecord:
    """Store validated metadata for one expected evaluation prediction."""

    prompt_id: str
    dataset_name: str
    question: str
    gold_answer: str
    inference_prompt_id: str

    @property
    def result_key(self) -> str:
        """Return the filename identifier used by ``swemedmcq.common.vllm_runner`` results."""
        return self.inference_prompt_id


def read_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    """Yield JSON objects and line numbers from a UTF-8 JSONL file."""
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            yield line_number, record


def parse_source_record(record: dict[str, Any], *, location: str) -> SourceRecord:
    """Validate the source metadata needed to join and score one prediction."""
    inference_prompt_id = record.get("prompt_id")
    prompt_id = record.get("source_prompt_id")
    dataset_name = record.get("dataset_name")
    question = record.get("question")
    gold_answer = record.get("gold_answer")

    if not isinstance(inference_prompt_id, str) or not inference_prompt_id.strip():
        raise ValueError(f"{location}: prompt_id must be a non-empty string")
    if not isinstance(dataset_name, str) or not dataset_name.strip():
        raise ValueError(f"{location}: dataset_name must be a non-empty string")
    inference_prompt_id = inference_prompt_id.strip()
    dataset_name = dataset_name.strip()
    if prompt_id is None:
        prefix = f"{dataset_name}_"
        if inference_prompt_id.startswith(prefix):
            prompt_id = inference_prompt_id.removeprefix(prefix)
        else:
            prompt_id = inference_prompt_id
            inference_prompt_id = f"{dataset_name}_{inference_prompt_id}"
    if not isinstance(prompt_id, str) or not prompt_id.strip():
        raise ValueError(f"{location}: source_prompt_id must be a non-empty string")
    if not isinstance(question, str) or not question.strip():
        raise ValueError(f"{location}: question must be a non-empty string")
    if not isinstance(gold_answer, str) or len(gold_answer.strip()) != 1:
        raise ValueError(f"{location}: gold_answer must be a single option letter")
    gold_answer = gold_answer.strip().upper()
    if not gold_answer.isascii() or not gold_answer.isalpha():
        raise ValueError(f"{location}: gold_answer must be an ASCII option letter")

    return SourceRecord(
        prompt_id=prompt_id.strip(),
        dataset_name=dataset_name,
        question=question,
        gold_answer=gold_answer,
        inference_prompt_id=inference_prompt_id,
    )


def load_source_records(path: Path) -> list[SourceRecord]:
    """Load source metadata and reject ambiguous dataset/prompt identifiers."""
    if not path.is_file():
        raise FileNotFoundError(f"Source JSONL does not exist: {path}")
    records = [
        parse_source_record(record, location=f"{path}:{line_number}")
        for line_number, record in read_jsonl(path)
    ]
    if not records:
        raise ValueError(f"Source JSONL contains no records: {path}")

    seen: set[str] = set()
    duplicates: list[str] = []
    for record in records:
        if record.result_key in seen:
            duplicates.append(record.result_key)
        seen.add(record.result_key)
    if duplicates:
        preview = ", ".join(duplicates[:10])
        raise ValueError(f"Duplicate source result identifier(s): {preview}")
    return records


def discover_results(input_dir: Path, pattern: str) -> dict[str, Path]:
    """Index result pickle paths by the identifier after the ``result_`` prefix."""
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Result directory does not exist: {input_dir}")
    results: dict[str, Path] = {}
    for path in sorted(input_dir.glob(pattern)):
        if not path.is_file():
            continue
        if not path.stem.startswith("result_"):
            raise ValueError(f"Result filename lacks the 'result_' prefix: {path.name}")
        result_key = path.stem.removeprefix("result_")
        if result_key in results:
            raise ValueError(f"Duplicate result identifier: {result_key}")
        results[result_key] = path
    return results


def split_lfm_completion(completion: str) -> tuple[str | None, str]:
    """Separate optional LFM thinking text from its final response text."""
    reasoning, final_text = split_qwen_reasoning_content(completion)
    if reasoning is None and "</think>" not in completion:
        return None, strip_generation_suffix_tokens(completion)
    return reasoning, strip_generation_suffix_tokens(final_text or "")


def parse_response(final_text: str) -> dict[str, str] | None:
    """Extract and validate the final JSON object against ``ResponseOutput``."""
    candidates = build_structured_output_candidates(final_text, response_format="qwen")
    for candidate in candidates:
        payload = parse_json_payload(candidate, ResponseOutput)
        if payload is not None:
            return payload
    return None


def _base_row(source: SourceRecord, result_path: Path) -> dict[str, Any]:
    """Create the metadata shared by successful and failed extraction rows."""
    return {
        "unique_id": result_path.name,
        "prompt_id": source.prompt_id,
        "dataset_name": source.dataset_name,
        "question": source.question,
        "completion": None,
        "gold_answer": source.gold_answer,
        "reasoning": None,
        "explanation": None,
        "answer_index": None,
        "schema_valid": False,
        "extraction_error": None,
        "finish_reason": None,
        "stop_reason": None,
        "request_finished": None,
        "num_generated_tokens": None,
        "final_token_id": None,
        "last_token_is_eos": None,
    }


def extract_result(
    source: SourceRecord,
    result_path: Path,
    tokenizer: TokenDecoder,
) -> dict[str, Any]:
    """Detokenize and parse one trusted vLLM ``RequestOutput`` pickle."""
    row = _base_row(source, result_path)
    try:
        with result_path.open("rb") as handle:
            result = pickle.load(handle)  # noqa: S301 - vLLM outputs are trusted.
        outputs = getattr(result, "outputs", None)
        if not isinstance(outputs, list) or len(outputs) != 1:
            count = len(outputs) if isinstance(outputs, list) else 0
            raise ValueError(f"expected exactly one completion, found {count}")
        output = outputs[0]
        token_ids = list(getattr(output, "token_ids"))
        eos_token_ids = set()
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        if eos_token_id is not None:
            eos_token_ids.add(int(eos_token_id))
        for token_id in getattr(tokenizer, "additional_special_tokens_ids", []) or []:
            token_text = tokenizer.decode(
                [token_id],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            if token_text in {"<|im_end|>", "<|return|>", "<eos>", "</s>"}:
                eos_token_ids.add(int(token_id))

        row["request_finished"] = bool(getattr(result, "finished", False))
        row["num_generated_tokens"] = len(token_ids)
        row["final_token_id"] = token_ids[-1] if token_ids else None
        row["last_token_is_eos"] = bool(
            token_ids and int(token_ids[-1]) in eos_token_ids
        )
        row["finish_reason"] = getattr(output, "finish_reason", None)
        stop_reason = getattr(output, "stop_reason", None)
        row["stop_reason"] = str(stop_reason) if stop_reason is not None else None
        completion = tokenizer.decode(
            token_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        reasoning, final_text = split_lfm_completion(completion)
        payload = parse_response(final_text)

        row["completion"] = completion
        row["reasoning"] = reasoning
        if payload is None:
            row["extraction_error"] = "completion does not match ResponseOutput"
            return row

        row["explanation"] = payload["explanation"]
        row["answer_index"] = payload["answer_index"]
        row["schema_valid"] = True
        return row
    except Exception as exc:
        row["extraction_error"] = f"{type(exc).__name__}: {exc}"
        return row


def missing_result_row(source: SourceRecord, expected_path: Path) -> dict[str, Any]:
    """Create an invalid prediction row when an expected result is absent."""
    row = _base_row(source, expected_path)
    row["extraction_error"] = "result pickle not found"
    return row


def write_jsonl(path: Path, records: Sequence[dict[str, Any]]) -> None:
    """Write records as UTF-8 JSONL while creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            # Model text can contain unpaired surrogate code points.  Escaping
            # non-ASCII keeps the audit payload lossless and UTF-8 writable.
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def sibling_output(path: Path, suffix: str) -> Path:
    """Create a sibling output path by appending a label to the filename stem."""
    return path.with_name(f"{path.stem}_{suffix}{path.suffix or '.jsonl'}")


def extract_directory(
    *,
    input_dir: Path,
    source_jsonl: Path,
    output_path: Path,
    tokenizer: TokenDecoder,
    pattern: str = "result_*.pkl",
) -> dict[str, int]:
    """Join source metadata to result pickles and write evaluation JSONL files."""
    sources = load_source_records(source_jsonl)
    results = discover_results(input_dir, pattern)
    records: list[dict[str, Any]] = []

    for source in sources:
        result_path = results.pop(source.result_key, None)
        if result_path is None:
            expected = input_dir / f"result_{source.result_key}.pkl"
            records.append(missing_result_row(source, expected))
        else:
            records.append(extract_result(source, result_path, tokenizer))

    orphan_results = len(results)
    valid_records = [record for record in records if record["schema_valid"]]
    failed_records = [record for record in records if not record["schema_valid"]]
    write_jsonl(output_path, records)
    write_jsonl(sibling_output(output_path, "valid"), valid_records)
    write_jsonl(sibling_output(output_path, "failed"), failed_records)

    summary = {
        "num_source_records": len(sources),
        "num_valid": len(valid_records),
        "num_failed": len(failed_records),
        "num_orphan_results": orphan_results,
    }
    summary_path = sibling_output(output_path, "summary").with_suffix(".json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    """Construct command-line arguments for result extraction."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--source-jsonl", type=Path, required=True)
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--pattern", default="result_*.pkl")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Load the tokenizer and run extraction from command-line arguments."""
    from transformers import AutoTokenizer

    args = build_arg_parser().parse_args(argv)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    summary = extract_directory(
        input_dir=args.input_dir,
        source_jsonl=args.source_jsonl,
        output_path=args.output_path,
        tokenizer=tokenizer,
        pattern=args.pattern,
    )
    print(
        f"Extracted {summary['num_valid']}/{summary['num_source_records']} valid "
        f"predictions; failed={summary['num_failed']} "
        f"orphan_results={summary['num_orphan_results']}"
    )
    print(f"Predictions: {args.output_path}")


if __name__ == "__main__":
    main()
