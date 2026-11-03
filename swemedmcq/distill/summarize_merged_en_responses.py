"""Write per-source statistics for merged English MCQ responses."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterator


WORD_FIELDS = ("reasoning", "explanation", "final_answer")


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


def has_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def word_count(value: Any) -> int:
    """Count words using split(' '), ignoring empty chunks."""
    if not has_text(value):
        return 0
    return len([word for word in value.split(" ") if word])


def normalize_answer(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().upper()
    return normalized or None


def percentile(sorted_values: list[int], q: float) -> float | None:
    """Return a linearly interpolated percentile from sorted integer values."""
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return float(sorted_values[0])

    position = (len(sorted_values) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * fraction


def word_stats(values: list[int]) -> dict[str, float | int | None]:
    """Compute descriptive stats plus Tukey whisker thresholds."""
    ordered = sorted(values)
    if not ordered:
        return {
            "count": 0,
            "min": None,
            "p05": None,
            "p25": None,
            "median": None,
            "mean": None,
            "p75": None,
            "p95": None,
            "max": None,
            "iqr": None,
            "lower_fence": None,
            "upper_fence": None,
            "lower_whisker": None,
            "upper_whisker": None,
        }

    p25 = percentile(ordered, 0.25)
    p75 = percentile(ordered, 0.75)
    assert p25 is not None and p75 is not None
    iqr = p75 - p25
    lower_fence = p25 - 1.5 * iqr
    upper_fence = p75 + 1.5 * iqr
    inlier_values = [value for value in ordered if lower_fence <= value <= upper_fence]

    return {
        "count": len(ordered),
        "min": ordered[0],
        "p05": percentile(ordered, 0.05),
        "p25": p25,
        "median": percentile(ordered, 0.50),
        "mean": sum(ordered) / len(ordered),
        "p75": p75,
        "p95": percentile(ordered, 0.95),
        "max": ordered[-1],
        "iqr": iqr,
        "lower_fence": lower_fence,
        "upper_fence": upper_fence,
        "lower_whisker": inlier_values[0] if inlier_values else ordered[0],
        "upper_whisker": inlier_values[-1] if inlier_values else ordered[-1],
    }


def empty_source_bucket(source_run: str, source_model: str | None) -> dict[str, Any]:
    return {
        "source_run": source_run,
        "source_model": source_model,
        "counts": Counter(),
        "word_counts": {field: [] for field in WORD_FIELDS},
    }


def collect_stats(input_path: Path) -> dict[str, Any]:
    """Collect per-source response statistics from merged prompt rows."""
    sources: dict[str, dict[str, Any]] = {}
    prompts_total = 0
    responses_total = 0
    prompts_with_responses: set[str] = set()

    dataset_counts: dict[str, Counter[str]] = defaultdict(Counter)

    for prompt in iter_jsonl(input_path):
        prompts_total += 1
        prompt_id = str(prompt.get("prompt_id"))
        gold_answer = normalize_answer(prompt.get("gold_answer"))
        responses = prompt.get("responses") or []
        if responses:
            prompts_with_responses.add(prompt_id)

        for response in responses:
            responses_total += 1
            source_run = str(response.get("source_run") or "unknown")
            source_model = response.get("source_model")
            bucket = sources.setdefault(
                source_run, empty_source_bucket(source_run, source_model)
            )
            if bucket.get("source_model") is None and source_model is not None:
                bucket["source_model"] = source_model

            final_answer = normalize_answer(response.get("final_answer"))
            has_reasoning = has_text(response.get("reasoning"))
            has_explanation = has_text(response.get("explanation"))
            has_final_answer = final_answer is not None
            answer_correct = has_final_answer and final_answer == gold_answer
            eligible = has_explanation and has_final_answer and answer_correct

            bucket["counts"]["responses"] += 1
            bucket["counts"]["parse_ok"] += int(
                response.get("parse_status") in {"ok", "repaired"}
            )
            bucket["counts"]["has_reasoning"] += int(has_reasoning)
            bucket["counts"]["has_explanation"] += int(has_explanation)
            bucket["counts"]["has_final_answer"] += int(has_final_answer)
            bucket["counts"]["answer_correct"] += int(answer_correct)
            bucket["counts"]["eligible"] += int(eligible)

            dataset_name = str(prompt.get("dataset_name") or "unknown")
            dataset_key = f"{source_run}|{dataset_name}"
            dataset_counts[dataset_key]["responses"] += 1
            dataset_counts[dataset_key]["answer_correct"] += int(answer_correct)
            dataset_counts[dataset_key]["eligible"] += int(eligible)

            for field in WORD_FIELDS:
                bucket["word_counts"][field].append(word_count(response.get(field)))

    source_rows = []
    for source_run in sorted(sources):
        bucket = sources[source_run]
        counts = dict(bucket["counts"])
        total = counts.get("responses", 0)
        rates = {
            f"{name}_rate": (count / total if total else None)
            for name, count in counts.items()
            if name != "responses"
        }
        source_rows.append(
            {
                "source_run": source_run,
                "source_model": bucket.get("source_model"),
                "counts": counts,
                "rates": rates,
                "word_counts": {
                    field: word_stats(values)
                    for field, values in bucket["word_counts"].items()
                },
            }
        )

    return {
        "input_path": str(input_path),
        "prompt_count": prompts_total,
        "prompt_count_with_responses": len(prompts_with_responses),
        "response_count": responses_total,
        "sources": source_rows,
        "dataset_counts": {
            key: dict(counts) for key, counts in sorted(dataset_counts.items())
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-path",
        type=Path,
        default=Path("work/sft/en/merged_unraveled_en_responses.jsonl"),
        help="Merged prompt-level JSONL from combine_unraveled_en_responses.py.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("work/sft/en/merged_en_response_source_stats.json"),
        help="Output JSON path for per-source statistics.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = collect_stats(args.input_path)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_path.open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, ensure_ascii=True, indent=2)
        handle.write("\n")
    print(
        json.dumps(
            {
                "output_path": str(args.output_path),
                "sources": len(stats["sources"]),
                "responses": stats["response_count"],
            },
            ensure_ascii=True,
        )
    )


if __name__ == "__main__":
    main()
