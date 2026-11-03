"""Extract answer-only validation predictions from vLLM result pickles."""

from __future__ import annotations

import argparse
import json
import pickle
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from ...common.utils import (
    build_structured_output_candidates,
    parse_json_payload,
    split_qwen_reasoning_content,
    strip_generation_suffix_tokens,
)
from ...common.schemas import AnswerLevelResponseOutput


class TokenDecoder(Protocol):
    eos_token_id: int | None
    additional_special_tokens_ids: list[int]

    def decode(self, token_ids: list[int], **kwargs: Any) -> str:
        """Convert generated token IDs into text."""


QUESTION_PATTERNS = {
    "en": re.compile(r"\*\*Question:\*\*\s*(.*?)\s*\*\*Options:\*\*", re.DOTALL),
    "sv": re.compile(r"\*\*Fråga:\*\*\s*(.*?)\s*\*\*Alternativ:\*\*", re.DOTALL),
}


@dataclass(frozen=True)
class SourceRecord:
    prompt_id: str
    dataset_name: str
    language: str | None
    question: str | None
    gold_answer: str
    allowed_choices: tuple[str, ...]

    @property
    def result_key(self) -> str:
        return self.prompt_id


def read_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
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


def normalize_letter(value: Any, *, location: str, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{location}: {field_name} must be a string")
    letter = value.strip().upper()
    if len(letter) != 1 or not letter.isascii() or not letter.isalpha():
        raise ValueError(f"{location}: {field_name} must be one ASCII option letter")
    return letter


def question_from_prompt(prompt: Any, language: str | None) -> str | None:
    """Extract displayed question text from an answer-level chat prompt."""
    if not isinstance(prompt, list) or not prompt:
        return None
    first = prompt[0]
    content = first.get("content") if isinstance(first, dict) else None
    if not isinstance(content, str) or not content.strip():
        return None
    patterns = []
    if language in QUESTION_PATTERNS:
        patterns.append(QUESTION_PATTERNS[language])
    patterns.extend(
        pattern
        for key, pattern in QUESTION_PATTERNS.items()
        if key != language
    )
    for pattern in patterns:
        match = pattern.search(content)
        if match:
            return match.group(1).strip()
    return content.strip()


def parse_source_record(record: dict[str, Any], *, location: str) -> SourceRecord:
    prompt_id = record.get("prompt_id")
    dataset_name = record.get("dataset_name")
    if not isinstance(prompt_id, str) or not prompt_id.strip():
        raise ValueError(f"{location}: prompt_id must be a non-empty string")
    if not isinstance(dataset_name, str) or not dataset_name.strip():
        raise ValueError(f"{location}: dataset_name must be a non-empty string")
    allowed_choices_value = record.get("allowed_choices") or []
    if not isinstance(allowed_choices_value, list):
        raise ValueError(f"{location}: allowed_choices must be a list when present")
    allowed_choices = tuple(
        str(choice).strip().upper()
        for choice in allowed_choices_value
        if str(choice).strip()
    )
    language = record.get("language") if isinstance(record.get("language"), str) else None
    question = record.get("question")
    if not isinstance(question, str) or not question.strip():
        question = question_from_prompt(record.get("prompt"), language)
    return SourceRecord(
        prompt_id=prompt_id.strip(),
        dataset_name=dataset_name.strip(),
        language=language,
        question=question.strip() if isinstance(question, str) and question.strip() else None,
        gold_answer=normalize_letter(
            record.get("gold_answer"),
            location=location,
            field_name="gold_answer",
        ),
        allowed_choices=allowed_choices,
    )


def load_source_records(path: Path) -> list[SourceRecord]:
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
        raise ValueError(f"Duplicate source prompt_id(s): {preview}")
    return records


def discover_results(input_dir: Path, pattern: str) -> dict[str, Path]:
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
    reasoning, final_text = split_qwen_reasoning_content(completion)
    if reasoning is None and "</think>" not in completion:
        return None, strip_generation_suffix_tokens(completion)
    return reasoning, strip_generation_suffix_tokens(final_text or "")


def parse_response(final_text: str) -> dict[str, str] | None:
    candidates = build_structured_output_candidates(final_text, response_format="qwen")
    for candidate in candidates:
        payload = parse_json_payload(candidate, AnswerLevelResponseOutput)
        if payload is not None:
            return payload
    return None


def _base_row(source: SourceRecord, result_path: Path) -> dict[str, Any]:
    return {
        "unique_id": result_path.name,
        "prompt_id": source.prompt_id,
        "dataset_name": source.dataset_name,
        "language": source.language,
        "question": source.question,
        "completion": None,
        "gold_answer": source.gold_answer,
        "allowed_choices": list(source.allowed_choices),
        "reasoning": None,
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


def eos_token_ids(tokenizer: TokenDecoder) -> set[int]:
    token_ids: set[int] = set()
    if tokenizer.eos_token_id is not None:
        token_ids.add(int(tokenizer.eos_token_id))
    for token_id in getattr(tokenizer, "additional_special_tokens_ids", []) or []:
        token_text = tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        if token_text in {"<|im_end|>", "<|return|>", "<eos>", "</s>"}:
            token_ids.add(int(token_id))
    return token_ids


def extract_result(
    source: SourceRecord,
    result_path: Path,
    tokenizer: TokenDecoder,
) -> dict[str, Any]:
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
        completion = tokenizer.decode(
            token_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        reasoning, final_text = split_lfm_completion(completion)
        payload = parse_response(final_text)

        row["request_finished"] = bool(getattr(result, "finished", False))
        row["num_generated_tokens"] = len(token_ids)
        row["final_token_id"] = token_ids[-1] if token_ids else None
        row["last_token_is_eos"] = bool(
            token_ids and int(token_ids[-1]) in eos_token_ids(tokenizer)
        )
        row["finish_reason"] = getattr(output, "finish_reason", None)
        stop_reason = getattr(output, "stop_reason", None)
        row["stop_reason"] = str(stop_reason) if stop_reason is not None else None
        row["completion"] = completion
        row["reasoning"] = reasoning
        if payload is None:
            row["extraction_error"] = "completion does not match AnswerLevelResponseOutput"
            return row

        row["answer_index"] = payload["answer_index"]
        row["schema_valid"] = True
        return row
    except Exception as exc:
        row["extraction_error"] = f"{type(exc).__name__}: {exc}"
        return row


def missing_result_row(source: SourceRecord, expected_path: Path) -> dict[str, Any]:
    row = _base_row(source, expected_path)
    row["extraction_error"] = "result pickle not found"
    return row


def write_jsonl(path: Path, records: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def sibling_output(path: Path, suffix: str) -> Path:
    return path.with_name(f"{path.stem}_{suffix}{path.suffix or '.jsonl'}")


def extract_directory(
    *,
    input_dir: Path,
    source_jsonl: Path,
    output_path: Path,
    tokenizer: TokenDecoder,
    pattern: str = "result_*.pkl",
) -> dict[str, int]:
    sources = load_source_records(source_jsonl)
    results = discover_results(input_dir, pattern)
    records: list[dict[str, Any]] = []

    for source in sources:
        result_path = results.pop(source.result_key, None)
        if result_path is None:
            records.append(missing_result_row(source, input_dir / f"result_{source.result_key}.pkl"))
        else:
            records.append(extract_result(source, result_path, tokenizer))

    valid_records = [record for record in records if record["schema_valid"]]
    failed_records = [record for record in records if not record["schema_valid"]]
    write_jsonl(output_path, records)
    write_jsonl(sibling_output(output_path, "valid"), valid_records)
    write_jsonl(sibling_output(output_path, "failed"), failed_records)
    summary = {
        "num_source_records": len(sources),
        "num_valid": len(valid_records),
        "num_failed": len(failed_records),
        "num_orphan_results": len(results),
    }
    summary_path = sibling_output(output_path, "summary").with_suffix(".json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--source-jsonl", type=Path, required=True)
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--pattern", default="result_*.pkl")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
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
