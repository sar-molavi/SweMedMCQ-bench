"""Extract GPT-OSS-120B trace-rewrite results from vLLM pickles."""

from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path
from typing import Any

from joblib import Parallel, delayed
from tqdm import tqdm

os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
os.environ.setdefault("RAYON_NUM_THREADS", str(os.cpu_count() or 1))

from transformers import AutoTokenizer

from ...common.utils import (
    build_structured_output_candidates,
    extract_reasoning_trace,
    parse_json_payload,
)
from .schema import TeacherTraceRewriteOutput


MODEL_NAME = "openai/gpt-oss-120b"


def parse_data(decoded_text: str) -> dict[str, Any] | None:
    """Parse the structured final-channel payload."""

    for candidate in build_structured_output_candidates(decoded_text, "gpt_oss"):
        data = parse_json_payload(candidate, TeacherTraceRewriteOutput)
        if data is not None:
            return data
    return None


def load_pickle(path: Path) -> tuple[str, list[int] | None, str | None, str | None]:
    """Load one trusted local vLLM pickle."""
    prompt_id = path.stem.removeprefix("result_")
    try:
        with path.open("rb") as handle:
            result = pickle.load(handle)  # noqa: S301 - trusted local artifact
        if len(result.outputs) != 1:
            raise ValueError(f"expected one completion, found {len(result.outputs)}")
        output = result.outputs[0]
        return prompt_id, list(output.token_ids), output.finish_reason, None
    except Exception as exc:
        return prompt_id, None, None, f"{type(exc).__name__}: {exc}"


def parse_record(
    prompt_id: str,
    decoded_text: str | None,
    finish_reason: str | None,
    error: str | None,
) -> dict[str, Any]:
    """Extract reasoning and structured data from decoded text."""
    if decoded_text is None:
        return {
            "prompt_id": prompt_id,
            "data": None,
            "reasoning": None,
            "decoded_text": None,
            "finish_reason": finish_reason,
            "error": error,
        }
    data = parse_data(decoded_text)
    if finish_reason == "length":
        data = None
        error = "finish_reason=length"
    return {
        "prompt_id": prompt_id,
        "data": data,
        "reasoning": extract_reasoning_trace(decoded_text, "gpt_oss"),
        "decoded_text": decoded_text,
        "finish_reason": finish_reason,
        "error": error if error is not None else (
            None if data is not None else "invalid structured output"
        ),
    }


def sibling_path(output_path: Path, suffix: str, extension: str | None = None) -> Path:
    extension = output_path.suffix if extension is None else extension
    return output_path.with_name(f"{output_path.stem}_{suffix}{extension}")


def extract(input_dir: Path, output_path: Path, n_jobs: int) -> dict[str, Any]:
    """Extract every result pickle and write all/valid/failed artifacts."""

    paths = sorted(input_dir.glob("result_*.pkl"))
    if not paths:
        raise FileNotFoundError(f"No result_*.pkl files found in {input_dir}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    valid_path = sibling_path(output_path, "valid")
    valid_ids_path = sibling_path(output_path, "valid_prompt_ids", ".txt")
    failed_ids_path = sibling_path(output_path, "failed_prompt_ids", ".txt")
    summary_path = sibling_path(output_path, "summary", ".json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    loaded = list(
        tqdm(
            Parallel(n_jobs=n_jobs, prefer="threads", return_as="generator")(
                delayed(load_pickle)(path) for path in paths
            ),
            total=len(paths),
            desc="Loading pickles",
            unit="file",
        )
    )

    token_batches = [token_ids for _, token_ids, _, _ in loaded if token_ids is not None]
    decoded = iter(tokenizer.batch_decode(token_batches, skip_special_tokens=False))
    parse_inputs = [
        (prompt_id, next(decoded) if token_ids is not None else None, finish_reason, error)
        for prompt_id, token_ids, finish_reason, error in loaded
    ]
    del loaded, token_batches, decoded
    records = Parallel(n_jobs=n_jobs, prefer="threads", return_as="generator")(
        delayed(parse_record)(*values) for values in parse_inputs
    )
    valid = 0
    failed = 0
    length_failures = 0
    with (
        output_path.open("w", encoding="utf-8") as all_output,
        valid_path.open("w", encoding="utf-8") as valid_output,
        valid_ids_path.open("w", encoding="utf-8") as valid_ids,
        failed_ids_path.open("w", encoding="utf-8") as failed_ids,
    ):
        for record in tqdm(records, total=len(paths), desc="Parsing results", unit="file"):
            line = json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
            all_output.write(line)
            if record["data"] is not None:
                valid += 1
                valid_output.write(line)
                valid_ids.write(record["prompt_id"] + "\n")
            else:
                failed += 1
                if record["finish_reason"] == "length":
                    length_failures += 1
                failed_ids.write(record["prompt_id"] + "\n")

    summary = {
        "model_name": MODEL_NAME,
        "total": len(paths),
        "valid": valid,
        "failed": failed,
        "length_failures": length_failures,
        "output_path": str(output_path),
        "valid_output_path": str(valid_path),
        "valid_prompt_ids_path": str(valid_ids_path),
        "failed_prompt_ids_path": str(failed_ids_path),
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-i", "--input-dir", type=Path, required=True)
    parser.add_argument("-o", "--output-path", type=Path, required=True)
    parser.add_argument("-j", "--n-jobs", type=int, default=-1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = extract(args.input_dir, args.output_path, args.n_jobs)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
