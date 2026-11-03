"""Build the evaluation prompt files from the benchmark file.

The benchmark file (paths.yaml ``benchmark``) has one row per item: ``prompt_id``,
``dataset``, ``question``, ``options``, ``sv_question``, ``sv_options``, ``answer``
(gold letter), ``status``, ``tier`` and ``core``. Its Swedish text is the reported
layer (L4a). This writes, to ``<testset_dir>/data/``:

  en_trace.jsonl              English explanation ("trace") prompt (prompt_id, prompt)
  en_answer_split.jsonl       English answer-only prompt + gold (scoring source)
  en_answer_prompts.jsonl     English answer-only prompt (prompt_id, prompt)
  sv_L4a_trace.jsonl          Swedish explanation prompt + gold
  sv_L4a_answer_split.jsonl   Swedish answer-only prompt + gold
  sv_L4a_answer_prompts.jsonl Swedish answer-only prompt (prompt_id, prompt)
  items.csv                   one row per item: dataset, status, tier, core (+ differs_L4a)

With ``--refined`` (paths.yaml ``refined``, the L4 layer) the L4 files are written for
the items whose Swedish text differs from L4a (``sv_L4_delta_*``); ``--refined`` with
``--full-layer`` writes all items of that layer instead (``sv_L4_*``). Any other layer
file in the same schema (prompt_id, question, options, sv_question, sv_options) can be
given with ``--layer-file NAME=PATH``.

``manifest.json`` lists the counts and the sha256 of every file. ``--compare DIR``
checks the built files byte for byte against files of the same name in DIR.

    python -m swemedmcq.testset.build
    python -m swemedmcq.testset.build --refined data/test/layers/layer4_refined.jsonl
    python -m swemedmcq.testset.build --benchmark my_benchmark.jsonl --out-dir /tmp/testset
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from swemedmcq import paths
from swemedmcq.common import jsonl
from swemedmcq.common.prompts import RESPONSE_PROMPT, SWE_RESPONSE_PROMPT
from swemedmcq.distill.answer_level.build_answer_level_dataset import make_record

MAIN_LAYER = "L4a"
ITEM_COLUMNS = ["prompt_id", "dataset", "status", "tier", "core", "differs_L4a"]


def as_options(value: dict[str, str] | str) -> dict[str, str]:
    """Options as a dict; a Python-literal string (``"{'A': ...}"``) is parsed."""
    return ast.literal_eval(value) if isinstance(value, str) else value


def load_items(path: Path) -> list[dict[str, Any]]:
    rows = list(jsonl.read(path))
    for row in rows:
        for key in ("options", "sv_options"):
            if key in row:
                row[key] = as_options(row[key])
    return rows


# ---------------------------------------------------------------- prompt rows
def options_text(options: dict[str, str]) -> str:
    """Options block of the Swedish explanation prompt (sorted labels)."""
    return "\n\n".join(f"{key}) {options[key]}" for key in sorted(options))


def en_options_text(options: dict[str, str]) -> str:
    """Options block of the English explanation prompt (option order, empty options left out)."""
    return "\n\n".join(f"{key}) {value}" for key, value in options.items() if value)


def gold_of(item: dict[str, Any]) -> str:
    return str(item.get("answer", item.get("gold_answer"))).strip().upper()


def allowed_choices(options: dict[str, str]) -> list[str]:
    return [str(key).strip().upper() for key, value in options.items() if value is not None]


def en_trace_row(item: dict[str, Any]) -> dict[str, Any]:
    prompt = RESPONSE_PROMPT.format(options=en_options_text(item["options"]), question=item["question"])
    return {"prompt_id": item["prompt_id"], "prompt": [{"content": prompt, "role": "user"}]}


def sv_trace_row(item: dict[str, Any], ref: dict[str, Any]) -> dict[str, Any]:
    prompt = SWE_RESPONSE_PROMPT.format(
        question=item["sv_question"], options=options_text(item["sv_options"])
    )
    return {
        "prompt_id": item["prompt_id"],
        "dataset_name": ref["dataset"],
        "language": "sv",
        "prompt": [{"content": prompt, "role": "user"}],
        "gold_answer": gold_of(ref),
        "allowed_choices": allowed_choices(ref["options"]),
    }


def answer_row(item: dict[str, Any], ref: dict[str, Any], language: str) -> dict[str, Any]:
    prefix = "sv_" if language == "sv" else ""
    record = make_record(
        {
            "prompt_id": item["prompt_id"],
            "dataset_name": ref["dataset"],
            "question": item[f"{prefix}question"],
            "options": item[f"{prefix}options"],
            "gold_answer": gold_of(ref),
        },
        language=language,
    )
    if record["allowed_choices"] != allowed_choices(ref["options"]):
        raise ValueError(f"{item['prompt_id']}: allowed choices differ from the English item")
    return record


# ---------------------------------------------------------------- writers
def write_english(items: list[dict[str, Any]], out_dir: Path) -> dict[str, Path]:
    paths_ = {
        "en_trace": out_dir / "en_trace.jsonl",
        "en_answer_split": out_dir / "en_answer_split.jsonl",
        "en_answer_prompts": out_dir / "en_answer_prompts.jsonl",
    }
    answers = [answer_row(item, item, "en") for item in items]
    # Serialisation of the original files: the English trace prompts use json.dumps defaults
    # without ASCII escapes; the answer-level files are compact and ASCII-escaped.
    out_dir.mkdir(parents=True, exist_ok=True)
    with paths_["en_trace"].open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(en_trace_row(item), ensure_ascii=False) + "\n")
    jsonl.write(paths_["en_answer_split"], answers, ascii=True)
    jsonl.write(
        paths_["en_answer_prompts"],
        ({"prompt_id": row["prompt_id"], "prompt": row["prompt"]} for row in answers),
        ascii=True,
    )
    return paths_


def write_swedish(items: list[dict[str, Any]], refs: dict[str, dict[str, Any]], out_dir: Path,
                  name: str) -> dict[str, Path]:
    missing = [item["prompt_id"] for item in items if item["prompt_id"] not in refs]
    if missing:
        raise KeyError(f"{len(missing)} items are not in the benchmark file, e.g. {missing[:3]}")
    paths_ = {
        f"sv_{name}_trace": out_dir / f"sv_{name}_trace.jsonl",
        f"sv_{name}_answer_split": out_dir / f"sv_{name}_answer_split.jsonl",
        f"sv_{name}_answer_prompts": out_dir / f"sv_{name}_answer_prompts.jsonl",
    }
    answers = [answer_row(item, refs[item["prompt_id"]], "sv") for item in items]
    # Trace prompts keep UTF-8; answer-level files are ASCII-escaped.
    jsonl.write(paths_[f"sv_{name}_trace"], (sv_trace_row(item, refs[item["prompt_id"]]) for item in items))
    jsonl.write(paths_[f"sv_{name}_answer_split"], answers, ascii=True)
    jsonl.write(
        paths_[f"sv_{name}_answer_prompts"],
        ({"prompt_id": row["prompt_id"], "prompt": row["prompt"]} for row in answers),
        ascii=True,
    )
    return paths_


def text_of(item: dict[str, Any]) -> tuple[str, tuple]:
    return item["sv_question"], tuple(sorted(item["sv_options"].items()))


def write_items(items: list[dict[str, Any]], differs: set[str] | None, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ITEM_COLUMNS)
        writer.writeheader()
        for item in items:
            writer.writerow({
                "prompt_id": item["prompt_id"],
                "dataset": item["dataset"],
                "status": item.get("status", item.get("category")),
                "tier": item.get("tier"),
                "core": item.get("core"),
                "differs_L4a": None if differs is None else item["prompt_id"] in differs,
            })


def compare(files: dict[str, Path], reference_dir: Path) -> bool:
    ok = True
    for path in files.values():
        ref = reference_dir / path.name
        if not ref.exists():
            print(f"skip {path.name} (not in {reference_dir})")
            continue
        same = path.read_bytes() == ref.read_bytes()
        ok &= same
        print(f"{'OK  ' if same else 'DIFF'} {path.name}  {jsonl.sha256(path)}")
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--benchmark", type=Path, default=None, help="Default: paths.yaml `benchmark`.")
    parser.add_argument("--out-dir", type=Path, default=None, help="Default: <testset_dir>/data (paths.yaml).")
    parser.add_argument("--refined", type=Path, nargs="?", const=paths.path("refined"), default=None,
                        help="L4 layer file; without a value: paths.yaml `refined`.")
    parser.add_argument("--layer-file", action="append", default=[], metavar="NAME=PATH",
                        help="Another layer in the draft schema, written like --refined.")
    parser.add_argument("--full-layer", action="store_true",
                        help="Write every item of the extra layers, not only those that differ from L4a.")
    parser.add_argument("--no-english", action="store_true", help="Skip the English files.")
    parser.add_argument("--compare", type=Path, default=None, help="Compare the built files with this folder.")
    args = parser.parse_args()

    benchmark = args.benchmark or paths.path("benchmark")
    out_dir = args.out_dir or paths.path("testset_dir") / "data"
    items = load_items(benchmark)
    refs = {item["prompt_id"]: item for item in items}
    base = {pid: text_of(item) for pid, item in refs.items()}

    files: dict[str, Path] = {}
    counts: dict[str, int] = {}
    if not args.no_english:
        files.update(write_english(items, out_dir))
        counts["en"] = len(items)
    files.update(write_swedish(items, refs, out_dir, MAIN_LAYER))
    counts[MAIN_LAYER] = len(items)

    layers = dict(spec.split("=", 1) for spec in args.layer_file)
    if args.refined is not None:
        layers = {"L4": str(args.refined), **layers}
    differs: set[str] | None = None
    for layer, layer_path in layers.items():
        layer_items = load_items(paths.resolve(layer_path))
        changed = [item for item in layer_items if text_of(item) != base[item["prompt_id"]]]
        if layer == "L4":
            differs = {item["prompt_id"] for item in changed}
        name = layer if args.full_layer else f"{layer}_delta"
        selected = layer_items if args.full_layer else changed
        counts[name] = len(selected)
        files.update(write_swedish(selected, refs, out_dir, name))

    write_items(items, differs, out_dir / "items.csv")
    files["items"] = out_dir / "items.csv"
    manifest = {
        "benchmark": str(benchmark),
        "layers": [MAIN_LAYER, *layers],
        "items_per_file": counts,
        "status": dict(Counter(str(item.get("status", item.get("category"))) for item in items)),
        "core_items": sum(bool(item.get("core")) for item in items),
        "core_per_dataset": dict(sorted(Counter(i["dataset"] for i in items if i.get("core")).items())),
        "sha256": {key: jsonl.sha256(path) for key, path in sorted(files.items())},
    }
    (out_dir.parent / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("items_per_file", "status", "core_items")}, indent=2))
    if args.compare is not None and not compare(files, args.compare):
        raise SystemExit("Some files differ from the reference folder.")


if __name__ == "__main__":
    main()
