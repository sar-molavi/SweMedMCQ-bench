"""Score answer-only MCQ validation predictions."""

from __future__ import annotations

import argparse
import json
import string
from collections import Counter
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


INVALID_PREDICTION = "__INVALID__"
VALID_ANSWER_LABELS = frozenset(string.ascii_uppercase)


@dataclass(frozen=True)
class EvaluatedRow:
    source: dict[str, Any]
    prompt_id: str
    dataset_name: str
    language: str | None
    gold_answer: str
    predicted_answer: str
    is_correct: bool
    schema_valid: bool
    is_valid_prediction: bool
    prediction_status: str
    num_generated_tokens: int | None
    validation_errors: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.source,
            "normalized_gold_answer": self.gold_answer,
            "normalized_answer_index": (
                None
                if self.predicted_answer == INVALID_PREDICTION
                else self.predicted_answer
            ),
            "is_correct": self.is_correct,
            "schema_valid": self.schema_valid,
            "is_valid_prediction": self.is_valid_prediction,
            "prediction_status": self.prediction_status,
            "validation_errors": list(self.validation_errors),
        }


def normalize_answer_label(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    label = value.strip().upper()
    return label if label in VALID_ANSWER_LABELS else None


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


def evaluate_record(record: dict[str, Any], *, location: str) -> EvaluatedRow:
    errors: list[str] = []
    prompt_id_value = record.get("prompt_id")
    if isinstance(prompt_id_value, (str, int)) and str(prompt_id_value).strip():
        prompt_id = str(prompt_id_value).strip()
    else:
        raise ValueError(f"{location}: prompt_id must be a non-empty string or integer")

    dataset_value = record.get("dataset_name")
    if not isinstance(dataset_value, str) or not dataset_value.strip():
        raise ValueError(f"{location}: dataset_name must be a non-empty string")
    dataset_name = dataset_value.strip()
    language = record.get("language") if isinstance(record.get("language"), str) else None

    gold_answer = normalize_answer_label(record.get("gold_answer"))
    if gold_answer is None:
        raise ValueError(f"{location}: gold_answer must be one letter from A to Z")

    extraction_schema_valid = record.get("schema_valid") is True
    if not extraction_schema_valid:
        errors.append("completion does not match the final answer-only schema")

    predicted_answer = normalize_answer_label(record.get("answer_index"))
    if predicted_answer is None:
        errors.append("answer_index must be one letter from A to Z")
        predicted_answer = INVALID_PREDICTION

    allowed_choices = record.get("allowed_choices")
    if isinstance(allowed_choices, list) and allowed_choices:
        normalized_allowed = {str(choice).strip().upper() for choice in allowed_choices}
        if predicted_answer != INVALID_PREDICTION and predicted_answer not in normalized_allowed:
            errors.append("answer_index is not one of the prompt choices")
            predicted_answer = INVALID_PREDICTION

    token_count_value = record.get("num_generated_tokens")
    num_generated_tokens = (
        token_count_value
        if isinstance(token_count_value, int) and token_count_value >= 0
        else None
    )

    extraction_error = record.get("extraction_error")
    request_finished = record.get("request_finished")
    finish_reason = record.get("finish_reason")
    last_token_is_eos = record.get("last_token_is_eos")
    if extraction_error == "result pickle not found":
        prediction_status = "missing_result"
    elif record.get("completion") is None:
        prediction_status = "result_read_error"
    elif request_finished is not True or finish_reason != "stop":
        prediction_status = "generation_not_finished"
    elif last_token_is_eos is not True:
        prediction_status = "eos_not_observed"
    elif not extraction_schema_valid:
        prediction_status = "final_schema_invalid"
    elif predicted_answer == INVALID_PREDICTION:
        prediction_status = "answer_label_invalid"
    elif predicted_answer == gold_answer:
        prediction_status = "correct"
    else:
        prediction_status = "wrong_answer"

    is_valid_prediction = prediction_status in {"correct", "wrong_answer"}
    if not is_valid_prediction:
        predicted_answer = INVALID_PREDICTION

    return EvaluatedRow(
        source=record,
        prompt_id=prompt_id,
        dataset_name=dataset_name,
        language=language,
        gold_answer=gold_answer,
        predicted_answer=predicted_answer,
        is_correct=prediction_status == "correct",
        schema_valid=not errors,
        is_valid_prediction=is_valid_prediction,
        prediction_status=prediction_status,
        num_generated_tokens=num_generated_tokens,
        validation_errors=tuple(errors),
    )


def _safe_divide(numerator: int | float, denominator: int | float) -> float:
    return numerator / denominator if denominator else 0.0


def classification_metrics(rows: Sequence[EvaluatedRow]) -> dict[str, Any]:
    labels = sorted(
        {row.gold_answer for row in rows}
        | {
            row.predicted_answer
            for row in rows
            if row.predicted_answer != INVALID_PREDICTION
        }
    )
    per_class: dict[str, dict[str, float | int]] = {}
    total_tp = total_fp = total_fn = 0

    for label in labels:
        tp = sum(row.gold_answer == label and row.predicted_answer == label for row in rows)
        fp = sum(row.gold_answer != label and row.predicted_answer == label for row in rows)
        fn = sum(row.gold_answer == label and row.predicted_answer != label for row in rows)
        support = sum(row.gold_answer == label for row in rows)
        precision = _safe_divide(tp, tp + fp)
        recall = _safe_divide(tp, tp + fn)
        f1 = _safe_divide(2 * precision * recall, precision + recall)
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
        total_tp += tp
        total_fp += fp
        total_fn += fn

    total = len(rows)
    correct = sum(row.is_correct for row in rows)
    invalid = sum(not row.is_valid_prediction for row in rows)
    wrong = sum(row.prediction_status == "wrong_answer" for row in rows)
    valid = sum(row.is_valid_prediction for row in rows)
    schema_invalid = sum(not row.schema_valid for row in rows)
    token_counts = [
        row.num_generated_tokens
        for row in rows
        if row.num_generated_tokens is not None
    ]
    macro_precision = _safe_divide(
        sum(float(v["precision"]) for v in per_class.values()),
        len(labels),
    )
    macro_recall = _safe_divide(
        sum(float(v["recall"]) for v in per_class.values()),
        len(labels),
    )
    macro_f1 = _safe_divide(sum(float(v["f1"]) for v in per_class.values()), len(labels))
    micro_precision = _safe_divide(total_tp, total_tp + total_fp)
    micro_recall = _safe_divide(total_tp, total_tp + total_fn)
    micro_f1 = _safe_divide(
        2 * micro_precision * micro_recall,
        micro_precision + micro_recall,
    )

    return {
        "num_samples": total,
        "num_correct": correct,
        "num_incorrect": total - correct,
        "num_wrong_answers": wrong,
        "num_valid_predictions": valid,
        "accuracy": _safe_divide(correct, total),
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "num_invalid_predictions": invalid,
        "invalid_prediction_rate": _safe_divide(invalid, total),
        "num_schema_invalid": schema_invalid,
        "schema_invalid_rate": _safe_divide(schema_invalid, total),
        "prediction_status_counts": dict(
            sorted(Counter(row.prediction_status for row in rows).items())
        ),
        "invalid_diagnostics": {
            "missing_result": sum(
                row.source.get("extraction_error") == "result pickle not found"
                for row in rows
            ),
            "result_read_error": sum(
                row.source.get("completion") is None
                and row.source.get("extraction_error") != "result pickle not found"
                for row in rows
            ),
            "generation_not_finished": sum(
                row.source.get("request_finished") is not True
                or row.source.get("finish_reason") != "stop"
                for row in rows
                if row.source.get("completion") is not None
            ),
            "eos_not_observed": sum(
                row.source.get("last_token_is_eos") is not True
                for row in rows
                if row.source.get("completion") is not None
            ),
            "final_schema_invalid": sum(
                row.source.get("schema_valid") is not True for row in rows
            ),
            "answer_label_invalid": sum(
                normalize_answer_label(row.source.get("answer_index")) is None
                for row in rows
            ),
        },
        "num_generations_with_token_count": len(token_counts),
        "total_generated_tokens": sum(token_counts),
        "mean_generated_tokens": _safe_divide(sum(token_counts), len(token_counts)),
        "min_generated_tokens": min(token_counts) if token_counts else None,
        "max_generated_tokens": max(token_counts) if token_counts else None,
        "per_class": per_class,
    }


def confusion_matrix(rows: Sequence[EvaluatedRow]) -> dict[str, dict[str, int]]:
    gold_labels = sorted({row.gold_answer for row in rows})
    predicted_labels = sorted(
        {row.predicted_answer for row in rows},
        key=lambda label: (label == INVALID_PREDICTION, label),
    )
    return {
        gold: {
            predicted: sum(
                row.gold_answer == gold and row.predicted_answer == predicted
                for row in rows
            )
            for predicted in predicted_labels
        }
        for gold in gold_labels
    }


def build_summary(rows: Sequence[EvaluatedRow], input_path: Path) -> dict[str, Any]:
    by_dataset = {}
    for dataset_name in sorted({row.dataset_name for row in rows}):
        dataset_rows = [row for row in rows if row.dataset_name == dataset_name]
        by_dataset[dataset_name] = {
            **classification_metrics(dataset_rows),
            "confusion_matrix": confusion_matrix(dataset_rows),
        }
    by_language = {}
    for language in sorted({row.language for row in rows if row.language}):
        language_rows = [row for row in rows if row.language == language]
        by_language[language] = {
            **classification_metrics(language_rows),
            "confusion_matrix": confusion_matrix(language_rows),
        }
    return {
        "input_path": str(input_path),
        "overall": {
            **classification_metrics(rows),
            "confusion_matrix": confusion_matrix(rows),
        },
        "by_dataset": by_dataset,
        "by_language": by_language,
        "dataset_sample_counts": dict(
            sorted(Counter(row.dataset_name for row in rows).items())
        ),
    }


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=True, indent=2, sort_keys=True)
        handle.write("\n")


def evaluate_file(input_path: Path, output_dir: Path) -> dict[str, Any]:
    if not input_path.is_file():
        raise FileNotFoundError(f"Evaluation input does not exist: {input_path}")
    rows = [
        evaluate_record(record, location=f"{input_path}:{line_number}")
        for line_number, record in read_jsonl(input_path)
    ]
    if not rows:
        raise ValueError(f"Evaluation input contains no records: {input_path}")

    seen: set[tuple[str, str, str | None]] = set()
    duplicates: list[str] = []
    for row in rows:
        key = (row.dataset_name, row.prompt_id, row.language)
        if key in seen:
            duplicates.append(f"{row.language or 'unknown'}:{row.dataset_name}:{row.prompt_id}")
        seen.add(key)
    if duplicates:
        preview = ", ".join(duplicates[:10])
        suffix = "..." if len(duplicates) > 10 else ""
        raise ValueError(f"Found {len(duplicates)} duplicate prediction row(s): {preview}{suffix}")

    summary = build_summary(rows, input_path)
    write_jsonl(output_dir / "predictions.jsonl", (row.to_dict() for row in rows))
    write_json(output_dir / "summary.json", summary)
    return summary


def print_summary(summary: dict[str, Any]) -> None:
    overall = summary["overall"]
    print(
        f"Overall: n={overall['num_samples']} accuracy={overall['accuracy']:.4f} "
        f"macro_f1={overall['macro_f1']:.4f} "
        f"wrong={overall['num_wrong_answers']} "
        f"invalid={overall['num_invalid_predictions']} "
        f"generated_tokens={overall['total_generated_tokens']} "
        f"mean_tokens={overall['mean_generated_tokens']:.1f}"
    )
    for dataset_name, metrics in summary["by_dataset"].items():
        print(
            f"{dataset_name}: n={metrics['num_samples']} accuracy={metrics['accuracy']:.4f} "
            f"macro_f1={metrics['macro_f1']:.4f} "
            f"wrong={metrics['num_wrong_answers']} "
            f"invalid={metrics['num_invalid_predictions']}"
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_path", type=Path, help="Extracted completion JSONL")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("work/sft/answer_level/sft_evaluation"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    summary = evaluate_file(args.input_path, args.output_dir)
    print_summary(summary)
    print(f"Detailed predictions: {args.output_dir / 'predictions.jsonl'}")
    print(f"Summary: {args.output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
