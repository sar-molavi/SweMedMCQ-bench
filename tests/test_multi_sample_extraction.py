from __future__ import annotations

import pickle
from pathlib import Path
from types import SimpleNamespace

import pytest

from swemedmcq.common.result_io import ExtractionConfig, ParsedPayloadStrategy, ResultExtractor


class FakeTokenizer:
    def decode(self, token_ids: list[int]) -> str:
        return ",".join(str(token_id) for token_id in token_ids)


class FakeStrategy(ParsedPayloadStrategy):
    output_key = "response_output"

    @property
    def model(self) -> type[object]:
        return object

    def metadata_from_result_stem(self, stem: str) -> dict[str, str]:
        return {"prompt_id": stem.removeprefix("result_")}

    def candidates(self, decoded_text: str, response_format: str) -> tuple[list[str], str]:
        return [decoded_text], decoded_text

    def parse(self, decoded_text: str, response_format: str) -> tuple[dict[str, str], str]:
        return {"decoded": decoded_text}, decoded_text

    def payload_fields(self, payload: dict[str, str] | None) -> dict[str, str | None]:
        return {"decoded": payload["decoded"] if payload else None}


def make_extractor(tmp_path: Path, sample_index: int = 0) -> ResultExtractor:
    extractor = ResultExtractor.__new__(ResultExtractor)
    extractor.config = ExtractionConfig(
        input_dir=tmp_path,
        output_path=tmp_path / "response.jsonl",
        model_name="unused",
        sample_index=sample_index,
    )
    extractor.strategy = FakeStrategy()
    extractor.tokenizer = FakeTokenizer()
    return extractor


def write_result(path: Path) -> None:
    request = SimpleNamespace(
        outputs=[
            SimpleNamespace(token_ids=[10, 11]),
            SimpleNamespace(token_ids=[20, 21]),
        ]
    )
    with path.open("wb") as handle:
        pickle.dump(request, handle)


def test_default_sample_is_first_completion(tmp_path: Path) -> None:
    path = tmp_path / "result_example.pkl"
    write_result(path)

    row = make_extractor(tmp_path).read_data(path)

    assert row["sample_index"] == 0
    assert row["decoded_text"] == "10,11"


def test_can_extract_second_completion(tmp_path: Path) -> None:
    path = tmp_path / "result_example.pkl"
    write_result(path)

    row = make_extractor(tmp_path, sample_index=1).read_data(path)

    assert row["sample_index"] == 1
    assert row["decoded_text"] == "20,21"


def test_missing_completion_has_clear_error(tmp_path: Path) -> None:
    path = tmp_path / "result_example.pkl"
    write_result(path)

    with pytest.raises(ValueError, match=r"sample_index=2.*only 2 completion"):
        make_extractor(tmp_path, sample_index=2).read_data(path)
