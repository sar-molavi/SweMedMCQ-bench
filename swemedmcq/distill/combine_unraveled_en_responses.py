"""Merge unraveled English MCQ response JSONL files by shared prompt ID."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Iterator


MODEL_BY_RUN_PREFIX = {
    "en_response_train_gpt_oss_120b": "openai/gpt-oss-120b",
    "en_response_train_qwen_3.6_35B_A3B": "Qwen/Qwen3.6-35B-A3B",
    "sw_response_train_gpt_oss_120b": "openai/gpt-oss-120b",
    "sw_response_train_qwen_3.6_35B_A3B": "Qwen/Qwen3.6-35B-A3B",
}


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Yield JSON objects from a JSONL file."""
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                yield json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc


def load_prompt_records(prompt_path: Path) -> dict[str, Any]:
    """Load prompt messages keyed by prompt ID."""
    prompts: dict[str, Any] = {}
    for row in iter_jsonl(prompt_path):
        prompt_id = str(row["prompt_id"])
        if prompt_id in prompts:
            raise ValueError(f"Duplicate prompt_id in {prompt_path}: {prompt_id}")
        prompts[prompt_id] = row.get("prompt")
    return prompts


def load_prompt_metadata(
    data_dir: Path, prompt_path: Path
) -> dict[str, dict[str, Any]]:
    """Load prompt text and gold answers keyed by prompt ID."""
    prompts = load_prompt_records(prompt_path)
    metadata: dict[str, dict[str, Any]] = {}

    for train_path in sorted(data_dir.glob("*/train.jsonl")):
        dataset_name = train_path.parent.name
        for row in iter_jsonl(train_path):
            prompt_id = f"{dataset_name}_{row['realidx']}"
            answer = row.get("answer_idx")
            if prompt_id in prompts:
                metadata[prompt_id] = {
                    "prompt_id": prompt_id,
                    "dataset_name": dataset_name,
                    "prompt": prompts[prompt_id],
                    "gold_answer": (
                        str(answer).strip().upper() if answer is not None else None
                    ),
                }

    missing = sorted(set(prompts) - set(metadata))
    if missing:
        preview = ", ".join(missing[:5])
        suffix = "..." if len(missing) > 5 else ""
        raise ValueError(
            f"{len(missing)} prompt IDs from {prompt_path} were not found in "
            f"{data_dir}/*/train.jsonl: {preview}{suffix}"
        )
    return {prompt_id: metadata[prompt_id] for prompt_id in prompts}


def infer_source_model(path: Path) -> str:
    """Infer the model identifier from an unraveled output filename."""
    run_name = path.stem
    matches = [
        (prefix, model)
        for prefix, model in MODEL_BY_RUN_PREFIX.items()
        if run_name.startswith(prefix)
    ]
    if not matches:
        return run_name
    return max(matches, key=lambda item: len(item[0]))[1]


def response_rows_for_file(path: Path) -> list[dict[str, Any]]:
    """Read one unraveled JSONL file and return compact response candidates."""
    source_run = path.stem
    source_model = infer_source_model(path)
    rows: list[dict[str, Any]] = []
    for row in iter_jsonl(path):
        rows.append(
            {
                "prompt_id": str(row["prompt_id"]),
                "sample_index": row.get("sample_index"),
                "source_model": source_model,
                "source_run": source_run,
                "reasoning": row.get("reasoning"),
                "explanation": row.get("explanation"),
                "final_answer": (
                    str(row["final_answer"]).strip().upper()
                    if row.get("final_answer") is not None
                    else None
                ),
                "parse_status": row.get("parse_status"),
                "parse_error": row.get("parse_error"),
                "source_file": row.get("source_file"),
            }
        )
    return rows


def discover_input_files(
    input_dir: Path,
    output_path: Path | None,
    source_runs: list[str] | None = None,
) -> list[Path]:
    """Find unraveled JSONL files, excluding the destination when it is nearby."""
    files = sorted(path for path in input_dir.glob("*.jsonl") if path.is_file())
    if output_path is None:
        result = files
    else:
        resolved_output = output_path.resolve()
        result = [path for path in files if path.resolve() != resolved_output]
    if source_runs:
        allowed = set(source_runs)
        result = [path for path in result if path.stem in allowed]
    return result


def combine_files(
    input_files: list[Path],
    output_path: Path,
    prompt_metadata: dict[str, dict[str, Any]],
    workers: int,
    include_empty_prompts: bool,
) -> None:
    """Merge response files into one row per shared prompt."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")

    if workers > 1 and len(input_files) > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            rows_by_file = list(pool.map(response_rows_for_file, input_files))
    else:
        rows_by_file = [response_rows_for_file(path) for path in input_files]

    responses_by_prompt: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rows in rows_by_file:
        for row in rows:
            prompt_id = row.pop("prompt_id")
            responses_by_prompt[prompt_id].append(row)

    for responses in responses_by_prompt.values():
        responses.sort(
            key=lambda row: (
                str(row.get("source_run") or ""),
                int(row.get("sample_index") or 0),
            )
        )

    prompt_ids = list(prompt_metadata) if include_empty_prompts else [
        prompt_id for prompt_id in prompt_metadata if prompt_id in responses_by_prompt
    ]

    with temporary.open("w", encoding="utf-8") as handle:
        for prompt_id in prompt_ids:
            meta = prompt_metadata[prompt_id]
            merged = {
                "prompt_id": prompt_id,
                "dataset_name": meta["dataset_name"],
                "prompt": meta["prompt"],
                "gold_answer": meta["gold_answer"],
                "responses": responses_by_prompt.get(prompt_id, []),
            }
            handle.write(json.dumps(merged, ensure_ascii=True, separators=(",", ":")))
            handle.write("\n")
    temporary.replace(output_path)

    print(
        json.dumps(
            {
                "output_path": str(output_path),
                "prompt_rows": len(prompt_ids),
                "response_rows": sum(len(rows) for rows in responses_by_prompt.values()),
            },
            ensure_ascii=True,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("work/sft/en/unraveled_en_responses"),
        help="Directory containing per-run unraveled JSONL files.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("work/sft/en/merged_unraveled_en_responses.jsonl"),
        help="Path for the merged JSONL file.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/source"),
        help="Root data directory containing <dataset>/train.jsonl files.",
    )
    parser.add_argument(
        "--prompt-path",
        type=Path,
        default=Path("work/prompts/en_train_prompt_set.jsonl"),
        help="JSONL prompt file created by make_response_prompts_en.py.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes for reading input files.",
    )
    parser.add_argument(
        "--include-empty-prompts",
        action="store_true",
        help="Also write prompts that have no response rows in the input files.",
    )
    parser.add_argument(
        "--source-runs",
        nargs="*",
        default=None,
        help="Optional allowlist of unraveled source-run file stems to merge.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_files = discover_input_files(args.input_dir, args.output_path, args.source_runs)
    if not input_files:
        raise FileNotFoundError(f"No JSONL files found in {args.input_dir}")
    prompt_metadata = load_prompt_metadata(args.data_dir, args.prompt_path)
    combine_files(
        input_files=input_files,
        output_path=args.output_path,
        prompt_metadata=prompt_metadata,
        workers=max(args.workers, 1),
        include_empty_prompts=args.include_empty_prompts,
    )


if __name__ == "__main__":
    main()
