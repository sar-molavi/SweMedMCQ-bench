"""Filter merged English MCQ responses for SFT using source-level whiskers."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterator


GPT_OSS_MODEL = "openai/gpt-oss-120b"


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


def is_gpt_oss(response: dict[str, Any], gpt_oss_model: str) -> bool:
    """Return whether this response should keep reasoning for SFT."""
    source_model = response.get("source_model")
    source_run = str(response.get("source_run") or "").lower()
    return source_model == gpt_oss_model or "gpt_oss" in source_run


def load_source_thresholds(stats_path: Path) -> dict[str, dict[str, int | float | None]]:
    """Load upper-whisker thresholds keyed by source_run."""
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    thresholds = {}
    for source in stats.get("sources", []):
        source_run = source["source_run"]
        word_counts = source.get("word_counts") or {}
        thresholds[source_run] = {
            "reasoning_upper_whisker": (
                (word_counts.get("reasoning") or {}).get("upper_whisker")
            ),
            "explanation_upper_whisker": (
                (word_counts.get("explanation") or {}).get("upper_whisker")
            ),
        }
    return thresholds


def is_eligible(response: dict[str, Any], gold_answer: Any) -> bool:
    """Eligible means explanation exists and final_answer is correct."""
    final_answer = normalize_answer(response.get("final_answer"))
    return (
        has_text(response.get("explanation"))
        and final_answer is not None
        and final_answer == normalize_answer(gold_answer)
    )


def above_threshold(value: int, threshold: int | float | None) -> bool:
    if threshold is None:
        return False
    return value > threshold


def source_allowed(response: dict[str, Any], args: argparse.Namespace) -> bool:
    """Apply optional source allowlists."""
    if args.source_runs and response.get("source_run") not in args.source_runs:
        return False
    if args.source_models and response.get("source_model") not in args.source_models:
        return False
    return True


def filter_response(
    response: dict[str, Any],
    gold_answer: Any,
    thresholds: dict[str, dict[str, int | float | None]],
    args: argparse.Namespace,
) -> tuple[dict[str, Any] | None, str]:
    """Return a filtered response or the reason it was removed."""
    if not source_allowed(response, args):
        return None, "source_excluded"
    if not is_eligible(response, gold_answer):
        return None, "not_eligible"

    source_run = response.get("source_run")
    if source_run not in thresholds:
        return None, "missing_source_stats"

    source_thresholds = thresholds[source_run]
    explanation_words = word_count(response.get("explanation"))
    if above_threshold(
        explanation_words, source_thresholds.get("explanation_upper_whisker")
    ):
        return None, "explanation_outlier"

    keep_reasoning = is_gpt_oss(response, args.gpt_oss_model)
    reasoning_words = word_count(response.get("reasoning"))
    if keep_reasoning and above_threshold(
        reasoning_words, source_thresholds.get("reasoning_upper_whisker")
    ):
        return None, "reasoning_outlier"

    filtered = {
        "sample_index": response.get("sample_index"),
        "source_model": response.get("source_model"),
        "source_run": source_run,
        "reasoning": response.get("reasoning") if keep_reasoning else None,
        "explanation": response.get("explanation"),
        "final_answer": normalize_answer(response.get("final_answer")),
        "reasoning_word_count": reasoning_words if keep_reasoning else 0,
        "explanation_word_count": explanation_words,
        "reasoning_kept": keep_reasoning,
    }
    return filtered, "kept"


def filter_file(args: argparse.Namespace) -> None:
    """Filter merged prompt rows and write output JSONL plus report JSON."""
    thresholds = load_source_thresholds(args.stats_path)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output_path.with_suffix(args.output_path.suffix + ".tmp")

    counts: Counter[str] = Counter()
    by_source: dict[str, Counter[str]] = defaultdict(Counter)
    prompts_with_kept = 0

    with temporary.open("w", encoding="utf-8") as handle:
        for prompt in iter_jsonl(args.input_path):
            counts["input_prompts"] += 1
            kept_responses = []
            for response in prompt.get("responses") or []:
                source_run = str(response.get("source_run") or "unknown")
                counts["input_responses"] += 1
                by_source[source_run]["input_responses"] += 1
                filtered, reason = filter_response(
                    response, prompt.get("gold_answer"), thresholds, args
                )
                counts[reason] += 1
                by_source[source_run][reason] += 1
                if filtered is None:
                    continue
                kept_responses.append(filtered)

            if not kept_responses and not args.include_empty_prompts:
                continue

            if kept_responses:
                prompts_with_kept += 1
            output_row = {
                "prompt_id": prompt.get("prompt_id"),
                "dataset_name": prompt.get("dataset_name"),
                "prompt": prompt.get("prompt"),
                "gold_answer": normalize_answer(prompt.get("gold_answer")),
                "responses": kept_responses,
            }
            handle.write(json.dumps(output_row, ensure_ascii=True, separators=(",", ":")))
            handle.write("\n")
            counts["output_prompts"] += 1

    temporary.replace(args.output_path)
    report = {
        "input_path": str(args.input_path),
        "stats_path": str(args.stats_path),
        "output_path": str(args.output_path),
        "policy": {
            "eligible": "explanation exists and final_answer equals gold_answer",
            "reasoning_sources": [args.gpt_oss_model, "source_run contains gpt_oss"],
            "outlier_filter": (
                "explanation <= source upper_whisker for all sources; "
                "reasoning <= source upper_whisker for GPT-OSS sources"
            ),
        },
        "counts": dict(counts),
        "prompts_with_kept_responses": prompts_with_kept,
        "by_source": {source: dict(counter) for source, counter in sorted(by_source.items())},
        "thresholds": thresholds,
    }
    with args.report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=True, indent=2, sort_keys=True)
        handle.write("\n")

    print(
        json.dumps(
            {
                "output_path": str(args.output_path),
                "report_path": str(args.report_path),
                "kept": counts.get("kept", 0),
                "input_responses": counts.get("input_responses", 0),
            },
            ensure_ascii=True,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-path",
        type=Path,
        default=Path("work/sft/en/merged_unraveled_en_responses.jsonl"),
        help="Merged prompt-level JSONL from combine_unraveled_en_responses.py.",
    )
    parser.add_argument(
        "--stats-path",
        type=Path,
        default=Path("work/sft/en/merged_en_response_source_stats.json"),
        help="Per-source stats JSON from summarize_merged_en_responses.py.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("work/sft/en/sft_filtered_en_responses.jsonl"),
        help="Filtered prompt-level JSONL output path.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("work/sft/en/sft_filtered_en_responses_report.json"),
        help="JSON report path for filter counts and thresholds.",
    )
    parser.add_argument(
        "--source-runs",
        nargs="*",
        default=None,
        help="Optional allowlist of source_run values.",
    )
    parser.add_argument(
        "--source-models",
        nargs="*",
        default=None,
        help="Optional allowlist of source_model values.",
    )
    parser.add_argument(
        "--gpt-oss-model",
        default=GPT_OSS_MODEL,
        help="Model ID whose responses keep reasoning for SFT.",
    )
    parser.add_argument(
        "--include-empty-prompts",
        action="store_true",
        help="Also write prompt rows whose responses are all filtered out.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    filter_file(parse_args())
