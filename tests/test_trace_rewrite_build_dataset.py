from __future__ import annotations

import json

import pytest

from swemedmcq.distill.trace_rewrite.build_dataset import build_dataset


def write_jsonl(path, rows) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_build_dataset_rewrites_valid_drops_failed_and_keeps_no_reasoning(tmp_path) -> None:
    source = tmp_path / "source.jsonl"
    rewrites = tmp_path / "rewrites.jsonl"
    output = tmp_path / "output.jsonl"
    write_jsonl(
        source,
        [{
            "prompt_id": "p1",
            "responses": [
                {"reasoning": "English one"},
                {"reasoning": None},
                {"reasoning": "English two"},
            ],
        }],
    )
    write_jsonl(
        rewrites,
        [{"prompt_id": "p1__response_000", "data": {"reasoning": "Svenska"}}],
    )

    summary = build_dataset(source, rewrites, output, tmp_path / "summary.json")
    row = json.loads(output.read_text(encoding="utf-8"))

    assert row["responses"] == [{"reasoning": "Svenska"}, {"reasoning": None}]
    assert summary["rewritten_responses"] == 1
    assert summary["dropped_responses"] == 1


def test_build_dataset_rejects_empty_rewrite_file(tmp_path) -> None:
    source = tmp_path / "source.jsonl"
    rewrites = tmp_path / "rewrites.jsonl"
    write_jsonl(source, [{"prompt_id": "p1", "responses": [{"reasoning": None}]}])
    write_jsonl(rewrites, [])

    with pytest.raises(ValueError, match="No valid rewrites"):
        build_dataset(source, rewrites, tmp_path / "output.jsonl", tmp_path / "summary.json")
