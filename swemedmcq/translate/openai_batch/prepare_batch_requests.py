"""
Utility for creating OpenAI Batch API request payloads from prompt JSONL files.

The input JSONL is expected to contain one JSON object per line with at least
``prompt_id`` and one of:

* ``prompt``
* ``input`` (supported for compatibility with some batch export formats)

The value may be either a string, an object representing a single message, or a
list of message-like objects as produced by LangChain workflows (each containing
``role`` and ``content`` keys).

This script transforms those prompts into the JSONL format required by the
OpenAI Batch API when targeting the ``/v1/responses`` endpoint with a structured
output schema.
"""

import argparse
import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from pydantic import BaseModel


@dataclass
class BatchPrepConfig:
    """Configuration for converting prompt JSONL into batch request JSONL."""

    input: Path
    output: Path
    model: str
    prompt_ids_file: Path | None = None
    temperature: float | None = None
    reasoning_effort: str | None = None
    reasoning_summary: bool = True
    max_output_tokens: int | None = 4096
    response_output: type[BaseModel] | None = None


def parse_args() -> BatchPrepConfig:
    """Parse CLI arguments into a batch preparation config."""
    parser = argparse.ArgumentParser(
        description="Convert prompt JSONL into OpenAI Batch API request payloads."
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        type=Path,
        help="Path to the source JSONL file with prompts.",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        type=Path,
        help="Destination JSONL path for the batch requests.",
    )
    parser.add_argument(
        "--model",
        required=True,
        type=str,
        help="Model name to use in the batch requests.",
    )
    parser.add_argument(
        "--prompt-ids-file",
        type=Path,
        default=None,
        help=(
            "Optional text file limiting requests to matching prompt ids. "
            "Each non-empty line may be either a prompt id or a tab-separated "
            "failure-report row whose last field is the prompt id."
        ),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature for the model. Omitted when not provided.",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=4096,
        help="Upper bound on tokens generated per response (default: 4096).",
    )
    parser.add_argument(
        "--reasoning-effort",
        default="medium",
        choices=["none", "minimal", "low", "medium", "high", "xhigh"],
        help="Optional reasoning effort level for reasoning-capable models.",
    )
    parser.add_argument(
        "--reasoning-summary",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Request a reasoning summary from reasoning-capable models "
            "(default: enabled). Use --no-reasoning-summary to disable."
        ),
    )
    args = parser.parse_args()
    return BatchPrepConfig(
        input=args.input,
        output=args.output,
        model=args.model,
        prompt_ids_file=args.prompt_ids_file,
        temperature=args.temperature,
        reasoning_effort=args.reasoning_effort,
        reasoning_summary=args.reasoning_summary,
        max_output_tokens=args.max_output_tokens,
    )


def load_prompts(path: Path) -> list[dict[str, Any]]:
    """Load one JSON object per line from a JSONL file."""

    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    prompts: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue
            try:
                prompts.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {path}: {exc}"
                ) from exc
    return prompts


def load_prompt_ids(path: Path) -> set[str]:
    """Load prompt ids from a plain text or tab-separated failure report file."""

    if not path.exists():
        raise FileNotFoundError(f"Prompt ids file not found: {path}")

    prompt_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue
            prompt_id = stripped.split("\t")[-1].strip()
            if not prompt_id:
                raise ValueError(
                    f"Missing prompt id on line {line_number} of {path}."
                )
            prompt_ids.add(prompt_id)

    if not prompt_ids:
        raise ValueError(f"Prompt ids file did not contain any ids: {path}")

    return prompt_ids


def filter_prompts_by_id(
    prompts: Iterable[dict[str, Any]], prompt_ids: set[str]
) -> list[dict[str, Any]]:
    """Return source prompts whose ``prompt_id`` is present in ``prompt_ids``."""

    filtered: list[dict[str, Any]] = []
    seen: set[str] = set()

    for entry in prompts:
        prompt_id = entry.get("prompt_id")
        if not isinstance(prompt_id, str):
            continue
        if prompt_id not in prompt_ids:
            continue
        filtered.append(entry)
        seen.add(prompt_id)

    missing = sorted(prompt_ids - seen)
    if missing:
        preview = ", ".join(missing[:10])
        suffix = "..." if len(missing) > 10 else ""
        raise ValueError(
            f"{len(missing)} prompt ids were not found in the input prompt JSONL: "
            f"{preview}{suffix}"
        )

    return filtered


def _normalize_message_like(item: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a message-like mapping and return a plain dict copy."""
    if "role" not in item:
        raise KeyError(f"Prompt message is missing 'role': {item}")
    if "content" not in item:
        raise KeyError(f"Prompt message is missing 'content': {item}")
    return dict(item)


def coerce_prompt(raw_prompt: Any) -> Any:
    """Normalize prompt payloads without flattening structured message lists."""

    if isinstance(raw_prompt, str):
        return raw_prompt

    if isinstance(raw_prompt, Mapping):
        return _normalize_message_like(raw_prompt)

    if isinstance(raw_prompt, Sequence) and not isinstance(raw_prompt, (str, bytes)):
        items = []
        for item in raw_prompt:
            if isinstance(item, str):
                items.append(item)
            elif isinstance(item, Mapping):
                items.append(_normalize_message_like(item))
            else:
                raise TypeError(f"Unsupported prompt sequence item: {item}")

        if not items:
            raise ValueError("Prompt sequence did not yield any items.")

        return items

    raise TypeError(f"Unsupported prompt type: {type(raw_prompt)}")


def extract_prompt_payload(entry: Mapping[str, Any]) -> Any:
    """Accept both `input` and `prompt` payload fields."""
    if "prompt" in entry:
        return entry["prompt"]
    if "input" in entry:
        return entry["input"]
    raise KeyError(
        f"Prompt entry missing required key 'prompt' or 'input': {json.dumps(entry, ensure_ascii=False)}"
    )


def enforce_no_additional_properties(schema: dict[str, Any]) -> dict[str, Any]:
    """Force `additionalProperties=False` throughout a JSON schema tree."""

    def _walk(node: Any) -> None:
        if not isinstance(node, dict):
            return

        if node.get("type") == "object":
            node["additionalProperties"] = False
            properties = node.get("properties")
            if isinstance(properties, dict):
                for child in properties.values():
                    _walk(child)

        items = node.get("items")
        if isinstance(items, dict):
            _walk(items)
        elif isinstance(items, list):
            for item in items:
                _walk(item)

        for value in node.values():
            if isinstance(value, dict):
                _walk(value)
            elif isinstance(value, list):
                for entry in value:
                    _walk(entry)

    new_schema = deepcopy(schema)
    _walk(new_schema)
    return new_schema


def build_text_format(config: BatchPrepConfig) -> dict[str, Any]:
    """Build the Responses API text-format schema wrapper."""

    schema = enforce_no_additional_properties(
        config.response_output.model_json_schema()
    )
    format_name = schema.get("title") or config.response_output.__name__
    return {
        "name": format_name,
        "type": "json_schema",
        "schema": schema,
        "strict": True,
    }


def generate_requests(
    prompts: Iterable[dict[str, Any]],
    config: BatchPrepConfig,
) -> list[dict[str, Any]]:
    """Convert prompt entries into OpenAI batch request objects."""

    text_format = build_text_format(config) if config.response_output else None
    requests: list[dict[str, Any]] = []

    for entry in prompts:
        if "prompt_id" not in entry:
            raise KeyError(
                f"Prompt entry missing required keys: {json.dumps(entry, ensure_ascii=False)}"
            )

        prompt_text = coerce_prompt(extract_prompt_payload(entry))
        body: dict[str, Any] = {
            "model": config.model,
            "input": prompt_text,
            "metadata": {"prompt_id": entry["prompt_id"]},
        }

        if config.temperature is not None:
            body["temperature"] = config.temperature

        if config.reasoning_effort is not None:
            body["reasoning"] = {"effort": config.reasoning_effort}
            if config.reasoning_summary:
                body["reasoning"]["summary"] = "auto"

        if text_format is not None:
            body["text"] = {"format": text_format}

        if config.max_output_tokens is not None:
            body["max_output_tokens"] = config.max_output_tokens

        request = {
            "custom_id": str(entry["prompt_id"]),
            "method": "POST",
            "url": "/v1/responses",
            "body": body,
        }
        requests.append(request)

    return requests


def write_jsonl(rows: Iterable[dict[str, Any]], output_path: Path) -> None:
    """Write request rows to a UTF-8 JSONL file."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def main(config: BatchPrepConfig) -> None:
    """Run the end-to-end batch preparation workflow."""
    prompts = load_prompts(config.input)
    if config.prompt_ids_file is not None:
        prompt_ids = load_prompt_ids(config.prompt_ids_file)
        prompts = filter_prompts_by_id(prompts, prompt_ids)
    requests = generate_requests(prompts=prompts, config=config)
    write_jsonl(requests, config.output)


if __name__ == "__main__":
    config = parse_args()
    main(config)
