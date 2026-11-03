"""Decode vLLM English MCQ pickles into their three semantic components.

This is deliberately a first-stage extractor.  It does not join gold labels,
score answers, select SFT examples, or merge generation runs.
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

from pydantic import ValidationError
from transformers import AutoTokenizer

from ..common.schemas import ResponseOutput
from ..common.utils import (
    extract_final_answer_text,
    extract_last_braced_json,
    extract_reasoning_trace,
    strip_generation_suffix_tokens,
)


def repair_json_once(candidate: str) -> tuple[str, str | None]:
    """Apply one unambiguous punctuation repair, without changing content."""
    value = candidate.strip()
    repaired = re.sub(r",\s*}$", "}", value)
    if repaired != value:
        return repaired, "remove_trailing_comma"
    if not value.startswith("{") and value.endswith("}") and value.count("}") == 1:
        return "{" + value, "insert_open_brace"
    if value.startswith("{") and not value.endswith("}") and value.count("{") == 1:
        return value + "}", "insert_close_brace"
    return value, None


def parse_final_response(text: str) -> tuple[dict[str, Any] | None, str | None, str | None]:
    """Parse the final JSON, optionally using one conservative repair."""
    candidates: list[str] = []
    braced = extract_last_braced_json(text)
    if braced:
        candidates.append(braced)
    if text.strip() not in candidates:
        candidates.append(text.strip())

    last_error: str | None = "no_final_response"
    for candidate in candidates:
        for value, repair in ((candidate, None), repair_json_once(candidate)):
            if repair is None and value != candidate:
                continue
            try:
                obj = json.loads(value)
                payload = ResponseOutput.model_validate(obj).model_dump()
                return payload, repair, None
            except json.JSONDecodeError as exc:
                last_error = f"json_decode:{exc.msg}"
            except ValidationError as exc:
                kinds = sorted({str(error["type"]) for error in exc.errors()})
                last_error = "schema:" + ",".join(kinds)
    return None, None, last_error


def unravel_text(decoded: str, response_format: str) -> dict[str, Any]:
    """Split one decoded generation into reasoning and structured final fields."""
    cleaned = strip_generation_suffix_tokens(decoded)
    reasoning = extract_reasoning_trace(cleaned, response_format)
    final_text = extract_final_answer_text(cleaned, response_format) or ""

    # If a wrapper marker is malformed but a complete final JSON object exists,
    # recover that object from the full generation.  Reasoning is never inferred.
    if not extract_last_braced_json(final_text):
        fallback = extract_last_braced_json(cleaned)
        if fallback:
            final_text = fallback

    payload, repair, error = parse_final_response(final_text)
    return {
        "reasoning": reasoning,
        "final_response": final_text or None,
        "explanation": payload.get("explanation") if payload else None,
        "final_answer": payload.get("answer_index") if payload else None,
        "parse_status": "repaired" if repair else "ok" if payload else "failed",
        "repair": repair,
        "parse_error": error,
    }


def prompt_id(path: Path) -> str:
    return path.stem.removeprefix("result_")


def rows(input_dir: Path, tokenizer: Any, response_format: str) -> Iterator[dict[str, Any]]:
    for path in sorted(input_dir.glob("result_*.pkl")):
        try:
            with path.open("rb") as handle:
                request = pickle.load(handle)
            outputs = list(request.outputs)
            if not outputs:
                raise ValueError("request has no outputs")
            for sample_index, output in enumerate(outputs):
                decoded = tokenizer.decode(output.token_ids, skip_special_tokens=False)
                yield {
                    "prompt_id": prompt_id(path),
                    "sample_index": sample_index,
                    **unravel_text(decoded, response_format),
                    "source_file": str(path),
                }
        except Exception as exc:
            yield {
                "prompt_id": prompt_id(path),
                "sample_index": None,
                "reasoning": None,
                "final_response": None,
                "explanation": None,
                "final_answer": None,
                "parse_status": "read_failed",
                "repair": None,
                "parse_error": f"{type(exc).__name__}: {exc}",
                "source_file": str(path),
            }


def run(args: argparse.Namespace) -> None:
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output_path.with_suffix(args.output_path.suffix + ".tmp")
    counts: Counter[str] = Counter()
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows(args.input_dir, tokenizer, args.response_format):
            counts[row["parse_status"]] += 1
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(args.output_path)
    print(json.dumps({"output": str(args.output_path), "counts": counts}, default=dict))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument(
        "--response-format",
        required=True,
        choices=("gpt_oss", "qwen"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
