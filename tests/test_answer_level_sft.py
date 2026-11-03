from __future__ import annotations

import json
from pathlib import Path

import pytest
from datasets import Dataset

from swemedmcq.distill.answer_level import build_answer_level_dataset as builder
from swemedmcq.distill.answer_level import build_bilingual_answer_level_dataset as bilingual
from swemedmcq.distill.answer_level import extract_swedish_answer_level_rows as sv_extract
from swemedmcq.train.prepare_answer_level_sft import (
    build_messages,
    make_assistant_content,
    prepare_dataset,
    tokenize_batch,
)
from swemedmcq.common.prompts import ANSWER_RESPONSE_PROMPT as RESPONSE_PROMPT
from swemedmcq.common.prompts import ANSWER_SWE_RESPONSE_PROMPT as SWE_RESPONSE_PROMPT
from swemedmcq.common.schemas import AnswerLevelResponseOutput


class FakeTokenizer:
    def apply_chat_template(self, messages, add_generation_prompt, **kwargs):
        results = []
        for conversation in messages:
            token_ids = []
            for message in conversation:
                role_offset = 10 if message["role"] == "user" else 100
                if message["role"] == "assistant":
                    token_ids.append(999)
                token_ids.extend(
                    range(role_offset, role_offset + len(message["content"].split()))
                )
            if add_generation_prompt:
                token_ids.append(999)
            results.append(token_ids)
        return results


def test_answer_prompts_do_not_request_explanations() -> None:
    english = RESPONSE_PROMPT.format(question="Question?", options="A) One\n\nB) Two")
    swedish = SWE_RESPONSE_PROMPT.format(question="Fraga?", options="A) Ett\n\nB) Tva")

    assert "explanation" not in english.casefold()
    assert "förklaring" not in swedish.casefold()
    assert json.loads('{"answer_index":"A"}') == {"answer_index": "A"}
    assert AnswerLevelResponseOutput.model_validate({"answer_index": "b"}).answer_index == "b"
    with pytest.raises(Exception):
        AnswerLevelResponseOutput.model_validate({"answer_index": "A", "explanation": "x"})


def test_english_record_builds_prompt_from_raw_mcq_row() -> None:
    record = builder.make_record(
        {
            "realidx": 7,
            "question": "Best next step?",
            "options": {"A": "Observe", "B": "Treat"},
            "answer_idx": "b",
        },
        language="en",
        dataset_name="demo",
    )

    assert record["prompt_id"] == "demo_7"
    assert record["gold_answer"] == "B"
    assert record["allowed_choices"] == ["A", "B"]
    content = record["prompt"][0]["content"]
    assert "Best next step?" in content
    assert "A) Observe" in content
    assert "explanation" not in content.casefold()


def test_swedish_record_uses_existing_evaluation_row() -> None:
    record = builder.make_record(
        {
            "prompt_id": "demo_8",
            "dataset_name": "demo",
            "question": "Vilket svar ar bast?",
            "choices": ["Ett", "Tva", "Tre"],
            "gold_answer": "c",
        },
        language="sv",
    )

    assert record["gold_answer"] == "C"
    assert record["language"] == "sv"
    assert record["allowed_choices"] == ["A", "B", "C"]
    assert "C) Tre" in record["prompt"][0]["content"]
    assert "{'label'" not in record["prompt"][0]["content"]


def test_extract_swedish_rows_joins_gold_answer_from_english_train(tmp_path) -> None:
    data_dir = tmp_path / "data" / "demo"
    data_dir.mkdir(parents=True)
    (data_dir / "train.jsonl").write_text(
        json.dumps({"realidx": 1, "answer_idx": "b"}) + "\n",
        encoding="utf-8",
    )
    translated = tmp_path / "translated.jsonl"
    translated.write_text(
        json.dumps(
            {
                "prompt_id": "demo_1",
                "dataset_name": "demo",
                "realidx": "1",
                "translation_output": {
                    "question": "Vilket svar?",
                    "choices": [
                        {"label": "A", "text": "Ett"},
                        {"label": "B", "text": "Tva"},
                    ],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report = sv_extract.extract_rows(
        input_path=translated,
        answer_key_data_dir=tmp_path / "data",
        answer_key_file_name="train",
        output_path=tmp_path / "out" / "sv.jsonl",
        report_path=tmp_path / "out" / "manifest.json",
        datasets=["demo"],
    )

    row = json.loads((tmp_path / "out" / "sv.jsonl").read_text().splitlines()[0])
    assert report["output_row_count"] == 1
    assert row["prompt_id"] == "demo_1"
    assert row["gold_answer"] == "B"
    assert row["question"] == "Vilket svar?"


def test_builds_train_validation_test_from_prompt_ids(tmp_path) -> None:
    data_dir = tmp_path / "data"
    demo_dir = data_dir / "demo"
    demo_dir.mkdir(parents=True)
    (demo_dir / "train.jsonl").write_text(
        "".join(
            json.dumps(row) + "\n"
            for row in [
                {
                    "realidx": 1,
                    "question": "Train?",
                    "options": {"A": "One", "B": "Two"},
                    "answer_idx": "a",
                },
                {
                    "realidx": 2,
                    "question": "Validation?",
                    "options": {"A": "One", "B": "Two"},
                    "answer_idx": "b",
                },
            ]
        ),
        encoding="utf-8",
    )
    (demo_dir / "test.jsonl").write_text(
        json.dumps(
            {
                "realidx": 3,
                "question": "Test?",
                "options": {"A": "One", "B": "Two"},
                "answer_idx": "a",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    validation_ids = tmp_path / "validation_prompt_ids.txt"
    validation_ids.write_text("demo_2\n", encoding="utf-8")

    manifest = builder.build_language_splits(
        language="en",
        input_path=data_dir,
        train_file_name="train",
        test_file_name="test",
        validation_prompt_ids_path=validation_ids,
        output_dir=tmp_path / "out",
        datasets=["demo"],
    )

    assert manifest["train_prompt_count"] == 1
    assert manifest["validation_prompt_count"] == 1
    assert manifest["test_prompt_count"] == 1
    assert manifest["validation_prompt_set_count"] == 1
    assert manifest["test_prompt_set_count"] == 1
    assert manifest["train_validation_overlap_count"] == 0
    train_row = json.loads(Path(manifest["train_path"]).read_text().splitlines()[0])
    validation_row = json.loads(
        Path(manifest["validation_path"]).read_text().splitlines()[0]
    )
    assert train_row["prompt_id"] == "demo_1"
    assert validation_row["prompt_id"] == "demo_2"
    prompt_only = json.loads(
        Path(manifest["validation_prompt_path"]).read_text().splitlines()[0]
    )
    assert sorted(prompt_only) == ["prompt", "prompt_id"]


def test_builds_train_validation_without_test(tmp_path) -> None:
    input_path = tmp_path / "sv.jsonl"
    validation_ids = tmp_path / "validation_prompt_ids.txt"
    input_path.write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                {
                    "prompt_id": "demo_1",
                    "dataset_name": "demo",
                    "question": "Trana?",
                    "choices": [{"label": "A", "text": "Ja"}, {"label": "B", "text": "Nej"}],
                    "gold_answer": "a",
                },
                {
                    "prompt_id": "demo_2",
                    "dataset_name": "demo",
                    "question": "Validera?",
                    "choices": [{"label": "A", "text": "Ja"}, {"label": "B", "text": "Nej"}],
                    "gold_answer": "b",
                },
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    validation_ids.write_text("demo_2\n", encoding="utf-8")

    manifest = builder.build_language_splits(
        language="sv",
        input_path=input_path,
        train_file_name="ignored",
        test_file_name=None,
        validation_prompt_ids_path=validation_ids,
        output_dir=tmp_path / "out",
        datasets=None,
    )

    assert manifest["train_prompt_count"] == 1
    assert manifest["validation_prompt_count"] == 1
    assert manifest["test_prompt_count"] == 0
    assert manifest["test_path"] is None
    assert Path(manifest["validation_prompt_path"]).is_file()


def test_builds_en_sv_evaluation_files_and_combines_them(tmp_path) -> None:
    en_path = tmp_path / "en.jsonl"
    sv_path = tmp_path / "sv.jsonl"
    en_path.write_text(
        json.dumps(
            {
                "prompt_id": "demo_1",
                "dataset_name": "demo",
                "question": "Best option?",
                "options": {"A": "One", "B": "Two"},
                "answer_idx": "a",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    sv_path.write_text(
        json.dumps(
            {
                "prompt_id": "demo_2",
                "dataset_name": "demo",
                "question": "Basta alternativet?",
                "choices": ["Ett", "Tva"],
                "gold_answer": "b",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    output_dir = tmp_path / "out"
    en_manifest = builder.build_language_eval(
        language="en",
        input_path=en_path,
        file_name="ignored",
        output_name="eval",
        output_dir=output_dir,
        datasets=None,
    )
    sv_manifest = builder.build_language_eval(
        language="sv",
        input_path=sv_path,
        file_name="ignored",
        output_name="eval",
        output_dir=output_dir,
        datasets=None,
    )
    combined = bilingual.combine_eval(
        english_path=Path(en_manifest["eval_path"]),
        swedish_path=Path(sv_manifest["eval_path"]),
        output_path=output_dir / "bilingual" / "eval.jsonl",
    )

    rows = [
        json.loads(line)
        for line in Path(combined["output_path"]).read_text(encoding="utf-8").splitlines()
    ]
    assert combined["english_count"] == 1
    assert combined["swedish_count"] == 1
    assert [row["language"] for row in rows] == ["en", "sv"]
    assert [row["gold_answer"] for row in rows] == ["A", "B"]


def test_build_language_eval_can_filter_prompt_ids(tmp_path) -> None:
    input_path = tmp_path / "en.jsonl"
    prompt_ids_path = tmp_path / "prompt_ids.txt"
    input_path.write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                {
                    "prompt_id": "demo_1",
                    "dataset_name": "demo",
                    "question": "One?",
                    "options": {"A": "Yes", "B": "No"},
                    "answer_idx": "a",
                },
                {
                    "prompt_id": "demo_2",
                    "dataset_name": "demo",
                    "question": "Two?",
                    "options": {"A": "Yes", "B": "No"},
                    "answer_idx": "b",
                },
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    prompt_ids_path.write_text("demo_2\n", encoding="utf-8")

    manifest = builder.build_language_eval(
        language="en",
        input_path=input_path,
        file_name="ignored",
        output_name="eval_validation",
        output_dir=tmp_path / "out",
        datasets=None,
        prompt_ids_file=prompt_ids_path,
    )

    rows = [
        json.loads(line)
        for line in Path(manifest["eval_path"]).read_text(encoding="utf-8").splitlines()
    ]
    assert manifest["requested_prompt_count"] == 1
    assert [row["prompt_id"] for row in rows] == ["demo_2"]


def test_cli_can_build_english_only(tmp_path) -> None:
    data_root = tmp_path / "data"
    dataset_dir = data_root / "demo"
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "test_hard.jsonl").write_text(
        json.dumps(
            {
                "realidx": 1,
                "question": "Best option?",
                "options": {"A": "One", "B": "Two"},
                "answer_idx": "a",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    output_dir = tmp_path / "out"
    builder.main(
        [
            "--language",
            "en",
            "--english-input",
            str(data_root),
            "--datasets",
            "demo",
            "--output-dir",
            str(output_dir),
        ]
    )

    assert (output_dir / "en" / "splits" / "eval.jsonl").is_file()
    manifest = json.loads((output_dir / "manifest_eval.json").read_text(encoding="utf-8"))
    assert set(manifest["languages"]) == {"en"}


def test_prepare_dataset_masks_prompt_and_targets_answer_json() -> None:
    tokenizer = FakeTokenizer()
    prompt = [{"role": "user", "content": "choose one"}]
    batch = {"prompt": [prompt], "gold_answer": ["b"]}

    tokenized = tokenize_batch(batch, tokenizer)
    labels = tokenized["labels"][0]
    input_ids = tokenized["input_ids"][0]
    trainable = [token for token, label in zip(input_ids, labels) if label != -100]

    assert labels[:3] == [-100, -100, -100]
    assert trainable == [100]
    assert make_assistant_content(" b ") == '{"answer_index":"B"}'
    assert build_messages(prompt, "A")[-1] == {
        "role": "assistant",
        "content": '{"answer_index":"A"}',
    }

    dataset = Dataset.from_list([{"prompt": prompt, "gold_answer": "A", "extra": "x"}])
    prepared = prepare_dataset(dataset, tokenizer, batch_size=1)
    assert prepared.column_names == ["input_ids", "labels", "num_trainable_tokens"]
