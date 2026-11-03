"""Build a preprocessed bilingual SFT train dataset from EN and SV Arrow inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


REQUIRED_COLUMNS = frozenset({"input_ids", "labels", "num_trainable_tokens"})
DEFAULT_TRAIN_DATASET_PATHS = {
    "en": "work/preprocessed/lfm/trace_en/train",
    "sv": "work/preprocessed/lfm/trace_sv/train",
}
DEFAULT_OUTPUT_PATH = "work/preprocessed/lfm/trace_en_sv/train"
DEFAULT_MANIFEST_PATH = (
    "work/preprocessed/lfm/trace_en_sv/manifest.json"
)
DEFAULT_SHUFFLE_SEED = 2000


def _load_from_disk(path: str | Path):
    from datasets import load_from_disk

    return load_from_disk(str(path), keep_in_memory=True)


def _concatenate_datasets(datasets: Sequence[Any]):
    from datasets import concatenate_datasets

    return concatenate_datasets(list(datasets))


def validate_columns(dataset: Any, *, path: str | Path) -> None:
    """Require the columns produced by SFT preprocessing."""

    columns = set(getattr(dataset, "column_names", []) or [])
    missing = REQUIRED_COLUMNS - columns
    if missing:
        raise ValueError(f"{path}: missing required columns: {sorted(missing)}")


def load_train_datasets(paths: Mapping[str, str | Path]) -> dict[str, Any]:
    """Load and validate language-keyed preprocessed datasets."""

    if not paths:
        raise ValueError("At least one train dataset path is required.")

    loaded = {}
    for language, path in paths.items():
        if not language:
            raise ValueError("Language keys must be non-empty.")
        dataset = _load_from_disk(path)
        validate_columns(dataset, path=path)
        loaded[str(language)] = dataset
    return loaded


def build_combined_train_dataset(
    train_dataset_paths: Mapping[str, str | Path],
    output_path: str | Path,
    *,
    shuffle_seed: int = DEFAULT_SHUFFLE_SEED,
) -> dict[str, Any]:
    """Concatenate all language train datasets, shuffle globally, and save."""

    output_path = Path(output_path)
    datasets_by_language = load_train_datasets(train_dataset_paths)
    source_lengths = {
        language: len(dataset) for language, dataset in datasets_by_language.items()
    }
    combined = _concatenate_datasets(list(datasets_by_language.values()))
    shuffled = combined.shuffle(seed=shuffle_seed)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    shuffled.save_to_disk(str(output_path))

    return {
        "train_dataset_paths": {
            language: str(path) for language, path in train_dataset_paths.items()
        },
        "source_lengths": source_lengths,
        "combined_length": len(shuffled),
        "shuffle_seed": shuffle_seed,
        "output_path": str(output_path),
        "required_columns": sorted(REQUIRED_COLUMNS),
    }


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=True, indent=2, sort_keys=True)
        handle.write("\n")


def parse_language_path(values: Sequence[str]) -> dict[str, str]:
    """Parse CLI values shaped as ``lang=path``."""

    result = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected LANG=PATH, got {value!r}.")
        language, path = value.split("=", 1)
        language = language.strip()
        path = path.strip()
        if not language or not path:
            raise ValueError(f"Expected non-empty LANG=PATH, got {value!r}.")
        result[language] = path
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-dataset",
        action="append",
        default=[],
        metavar="LANG=PATH",
        help="Preprocessed Arrow train dataset. Repeat per language.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path(DEFAULT_OUTPUT_PATH),
        help="Directory where the combined shuffled Arrow dataset is saved.",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=Path(DEFAULT_MANIFEST_PATH),
        help="JSON manifest describing the combined artifact.",
    )
    parser.add_argument(
        "--shuffle-seed",
        type=int,
        default=DEFAULT_SHUFFLE_SEED,
        help="Deterministic global shuffle seed.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    train_paths = (
        parse_language_path(args.train_dataset)
        if args.train_dataset
        else dict(DEFAULT_TRAIN_DATASET_PATHS)
    )
    manifest = build_combined_train_dataset(
        train_paths,
        args.output_path,
        shuffle_seed=args.shuffle_seed,
    )
    write_json(args.manifest_path, manifest)
    print(
        json.dumps(
            {
                "output_path": manifest["output_path"],
                "combined_length": manifest["combined_length"],
                "shuffle_seed": manifest["shuffle_seed"],
                "manifest_path": str(args.manifest_path),
            },
            ensure_ascii=True,
        )
    )


if __name__ == "__main__":
    main()

