"""Filter extracted MCQ translations by structured translation warnings."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator


SEVERITY_RANK = {
    "low": 0,
    "medium": 1,
    "high": 2,
}


@dataclass
class FilterTranslationWarningsConfig:
    input_path: Path
    output_path: Path
    summary_path: Path | None = None
    warning_type: str = "possible_answer_change_risk"
    severity: str = "low"


def parse_args() -> FilterTranslationWarningsConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Filter extracted MCQ translation JSONL rows that contain a selected "
            "TranslationWarning type at or above a minimum severity."
        )
    )
    parser.add_argument(
        "-i",
        "--input-path",
        type=Path,
        default=Path("train_trans/results/train_trans_valid.jsonl"),
        help="Input extracted translations JSONL file.",
    )
    parser.add_argument(
        "-o",
        "--output-path",
        type=Path,
        default=Path("train_trans/results/train_trans_valid_filtered.jsonl"),
        help="Output JSONL file containing rows that pass the warning filter.",
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=None,
        help="Optional JSON summary path. Defaults beside the output file.",
    )
    parser.add_argument(
        "--warning-type",
        default="possible_answer_change_risk",
        help="TranslationWarning.type value to filter.",
    )
    parser.add_argument(
        "--severity",
        choices=tuple(SEVERITY_RANK),
        default="low",
        help="Minimum warning severity to filter.",
    )
    args = parser.parse_args()
    return FilterTranslationWarningsConfig(
        input_path=args.input_path,
        output_path=args.output_path,
        summary_path=args.summary_path,
        warning_type=args.warning_type,
        severity=args.severity,
    )


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {path}: {exc}"
                ) from exc


def translation_warnings(row: dict[str, Any]) -> list[dict[str, Any]]:
    translation_output = row.get("translation_output")
    if isinstance(translation_output, dict):
        warnings = translation_output.get("warnings")
        if isinstance(warnings, list):
            return [warning for warning in warnings if isinstance(warning, dict)]

    warnings = row.get("warnings")
    if isinstance(warnings, list):
        return [warning for warning in warnings if isinstance(warning, dict)]
    return []


def matching_filter_warnings(
    row: dict[str, Any],
    *,
    warning_type: str,
    min_severity: str,
) -> list[dict[str, Any]]:
    min_rank = SEVERITY_RANK[min_severity]
    matches = []
    for warning in translation_warnings(row):
        severity = str(warning.get("severity", "")).lower()
        if warning.get("type") != warning_type:
            continue
        if SEVERITY_RANK.get(severity, -1) >= min_rank:
            matches.append(warning)
    return matches


def should_filter(
    row: dict[str, Any],
    *,
    warning_type: str,
    min_severity: str,
) -> bool:
    return bool(
        matching_filter_warnings(
            row,
            warning_type=warning_type,
            min_severity=min_severity,
        )
    )


def default_summary_path(output_path: Path) -> Path:
    if output_path.suffix:
        return output_path.with_name(f"{output_path.stem}_summary.json")
    return output_path.with_name(f"{output_path.name}_summary.json")


def write_summary(summary: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")


def filter_translations(config: FilterTranslationWarningsConfig) -> dict[str, Any]:
    config.output_path.parent.mkdir(parents=True, exist_ok=True)

    total_count = 0
    kept_count = 0
    filtered_count = 0
    filtered_by_severity: Counter[str] = Counter()
    filtered_prompt_ids: list[str] = []

    with config.output_path.open("w", encoding="utf-8") as output:
        for row in iter_jsonl(config.input_path):
            total_count += 1
            matching_warnings = matching_filter_warnings(
                row,
                warning_type=config.warning_type,
                min_severity=config.severity,
            )
            if matching_warnings:
                filtered_count += 1
                filtered_prompt_ids.append(str(row.get("prompt_id", "")))
                for warning in matching_warnings:
                    severity = str(warning.get("severity", "")).lower()
                    filtered_by_severity[severity] += 1
                continue

            kept_count += 1
            output.write(json.dumps(row, ensure_ascii=False))
            output.write("\n")

    return {
        "input_path": str(config.input_path),
        "output_path": str(config.output_path),
        "warning_type": config.warning_type,
        "minimum_severity": config.severity,
        "total_count": total_count,
        "kept_count": kept_count,
        "filtered_count": filtered_count,
        "filtered_warning_counts_by_severity": dict(
            sorted(filtered_by_severity.items())
        ),
        "filtered_prompt_ids": filtered_prompt_ids,
    }


def main(config: FilterTranslationWarningsConfig) -> None:
    summary = filter_translations(config)
    summary_path = config.summary_path or default_summary_path(config.output_path)
    write_summary(summary, summary_path)


if __name__ == "__main__":
    main(parse_args())
