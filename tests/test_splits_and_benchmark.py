from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import pytest

from swemedmcq.distill import split_english_sft
from swemedmcq.distill.prompt_ids import read_prompt_ids
from swemedmcq.testset.make_benchmark import read_hard

ROOT = Path(__file__).resolve().parents[1]


def write_ids(path: Path, ids: list[str]) -> Path:
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write("\n".join(ids) + "\n")
    return path


def test_read_prompt_ids_text_gzip_and_jsonl(tmp_path: Path) -> None:
    assert read_prompt_ids(write_ids(tmp_path / "ids.txt.gz", ["a_1", "b_2", "a_1"])) == {"a_1", "b_2"}
    jsonl = tmp_path / "split.jsonl"
    jsonl.write_text('{"prompt_id": "a_1"}\n{"prompt_id": "a_1"}\n{"prompt_id": "c_3"}\n', encoding="utf-8")
    assert read_prompt_ids(jsonl) == {"a_1", "c_3"}


def test_split_english_sft_follows_released_ids(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pool = tmp_path / "pool.jsonl"
    rows = [{"prompt_id": f"medqa_{i}", "dataset_name": "medqa", "responses": [{"source_run": "r"}]} for i in range(6)]
    rows.append({"prompt_id": "medqa_9", "dataset_name": "medqa", "responses": []})  # no kept response
    pool.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    train = write_ids(tmp_path / "train.txt.gz", ["medqa_0", "medqa_1", "medqa_2", "medqa_9"])
    val = write_ids(tmp_path / "val.txt.gz", ["medqa_3"])
    out = tmp_path / "splits"
    monkeypatch.setattr(sys, "argv", ["split", "--input-path", str(pool), "--output-dir", str(out),
                                      "--train-ids", str(train), "--validation-ids", str(val)])
    split_english_sft.main()
    ids = lambda name: [json.loads(l)["prompt_id"] for l in (out / name).read_text().splitlines()]  # noqa: E731
    assert ids("train.jsonl") == ["medqa_0", "medqa_1", "medqa_2"]
    assert ids("validation.jsonl") == ["medqa_3"]  # medqa_4/5 are in neither file


def test_released_split_ids_are_disjoint() -> None:
    split = ROOT / "data/train/split"
    train = read_prompt_ids(split / "train_prompt_ids.txt.gz")
    val = read_prompt_ids(split / "validation_prompt_ids.txt.gz")
    assert (len(train), len(val)) == (126141, 6642)
    assert not train & val


def test_read_hard(tmp_path: Path) -> None:
    (tmp_path / "medqa").mkdir()
    (tmp_path / "medqa" / "test_hard.jsonl").write_text('{"realidx": 7}\n{"realidx": "x"}\n', encoding="utf-8")
    assert read_hard(tmp_path, {"medqa"}) == {"medqa_7", "medqa_x"}
    with pytest.raises(SystemExit):
        read_hard(tmp_path, {"pubmedqa"})


def test_benchmark_flags() -> None:
    rows = [json.loads(l) for l in (ROOT / "data/test/benchmark.jsonl").open(encoding="utf-8")]
    assert len(rows) == 10363
    assert sum(r["core"] for r in rows) == 9733
    assert sum(r["hard"] for r in rows) == 894
    assert sum(r["hard"] and r["core"] for r in rows) == 840
