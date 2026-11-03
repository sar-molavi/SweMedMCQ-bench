"""Build vLLM-ready prompts for Swedish trace rewrite."""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any

from .prompts import PROMPT_VERSION, PROMPTS, TRACE_REWRITE_VARIANTS, TraceRewriteVariant


DEFAULT_INPUT_PATH = Path("work/sft/sv_rw/train_with_reasoning.jsonl")
DEFAULT_OUTPUT_PATH = Path("work/sft/sv_rw/rewrite_prompts.jsonl")
DEFAULT_MANIFEST_PATH = Path("work/sft/sv_rw/rewrite_prompts_manifest.json")


def iter_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            yield line_number, row


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def safe_prompt_id(value: Any) -> str:
    return re.sub(r"[^A-Za-z0-9_.=-]+", "_", str(value or "").strip())


def prompt_text(row: dict[str, Any], *, location: str) -> str:
    prompt = row.get("prompt")
    if not isinstance(prompt, list):
        raise ValueError(f"{location}: prompt must be a list")
    user_messages = [
        message
        for message in prompt
        if isinstance(message, dict) and message.get("role") == "user"
    ]
    if len(user_messages) != 1:
        raise ValueError(f"{location}: expected exactly one user message")
    content = user_messages[0].get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError(f"{location}: user prompt content must be non-empty")
    content = content.strip()
    marker = "## Indata"
    if marker in content:
        return content.split(marker, 1)[1].strip()
    return content


def response_text(response: dict[str, Any], field: str, *, location: str) -> str:
    value = response.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{location}: response {field} must be a non-empty string")
    return value.strip()


def make_rewrite_prompt(
    row: dict[str, Any],
    response: dict[str, Any],
    *,
    response_index: int,
    line_number: int,
    variant: TraceRewriteVariant,
) -> dict[str, Any]:
    location = f"line {line_number}, response {response_index}"
    source_prompt_id = str(row.get("prompt_id") or "").strip()
    if not source_prompt_id:
        raise ValueError(f"{location}: prompt_id must be non-empty")

    content = PROMPTS[variant].format(
        question=prompt_text(row, location=location),
        reasoning=response_text(response, "reasoning", location=location),
        explanation=response_text(response, "explanation", location=location),
        answer_index=response_text(response, "final_answer", location=location).upper(),
    )

    teacher_prompt_id = f"{safe_prompt_id(source_prompt_id)}__response_{response_index:03d}"
    return {
        "prompt_id": teacher_prompt_id,
        "source_prompt_id": source_prompt_id,
        "source_response_index": response_index,
        "dataset_name": row.get("dataset_name"),
        "gold_answer": row.get("gold_answer"),
        "source_model": response.get("source_model"),
        "source_run": response.get("source_run"),
        "sample_index": response.get("sample_index"),
        "source_final_answer": response.get("final_answer"),
        "trace_rewrite_variant": variant,
        "trace_rewrite_prompt_version": PROMPT_VERSION,
        "prompt": [{"role": "user", "content": content}],
    }


def build_rewrite_prompts(
    input_path: Path,
    output_path: Path,
    manifest_path: Path,
    *,
    variant: TraceRewriteVariant = "conservative_sv_rewrite",
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    input_rows = 0
    input_responses = 0

    for line_number, row in iter_jsonl(input_path):
        input_rows += 1
        responses = row.get("responses")
        if not isinstance(responses, list):
            raise ValueError(f"{input_path}:{line_number}: responses must be a list")
        for response_index, response in enumerate(responses):
            if not isinstance(response, dict):
                raise ValueError(f"{input_path}:{line_number}: response must be an object")
            input_responses += 1
            rows.append(
                make_rewrite_prompt(
                    row,
                    response,
                    response_index=response_index,
                    line_number=line_number,
                    variant=variant,
                )
            )

    output_count = write_jsonl(output_path, rows)
    manifest = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "variant": variant,
        "prompt_version": PROMPT_VERSION,
        "input_rows": input_rows,
        "input_responses": input_responses,
        "output_prompts": output_count,
    }
    write_json(manifest_path, manifest)
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--manifest-path", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--variant", choices=TRACE_REWRITE_VARIANTS, default="conservative_sv_rewrite")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    manifest = build_rewrite_prompts(
        input_path=args.input_path,
        output_path=args.output_path,
        manifest_path=args.manifest_path,
        variant=args.variant,
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
