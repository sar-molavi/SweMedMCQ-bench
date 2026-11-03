#!/usr/bin/env python
"""GPU check: does each reasoning parser put the JSON where it should?

Runs the real runner (the same ``python -m`` modules as the evaluation) on a few
items per configuration, decodes the result pickles with the model tokenizer and
checks

  (a) reasoning is present in "on" mode and absent in "off"/"single" mode;
  (b) the JSON starts right after the reasoning boundary (on) or at the first
      token (off), so the grammar was not applied inside the reasoning and was
      applied after it;
  (c) the JSON validates against the schema and answer_index is an allowed letter;
  (d) finish reasons (``length`` = truncated at max output tokens).

    CUDA_VISIBLE_DEVICES=0 python -m swemedmcq.evaluate.check_parsers --out DIR [--only REGEX]
    python -m swemedmcq.evaluate.check_parsers --out DIR --skip-generate   # re-check only

Configurations follow configs/models.yaml (small models, plus Qwen3.8-27B and
Gemma-4-31B-it on two 40 GB GPUs: CUDA_VISIBLE_DEVICES=0,1). Needs the test-set files
(swemedmcq.testset.build) and, for the student checks, the released students (local or Hub).
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ValidationError
from transformers import AutoTokenizer

from swemedmcq.common.schemas import AnswerLevelResponseOutput, ResponseOutput, SweResponseOutput
from swemedmcq.evaluate.registry import data_dir
from swemedmcq.paths import ROOT, path

PROMPTS = {
    ("trace", "sv"): data_dir() / "sv_L4a_trace.jsonl",
    ("answer", "sv"): data_dir() / "sv_L4a_answer_prompts.jsonl",
    ("trace", "en"): data_dir() / "en_trace.jsonl",
}
MODULES = {
    ("trace", "sv"): ("swemedmcq.common.sw_response", SweResponseOutput),
    ("trace", "en"): ("swemedmcq.common.en_response", ResponseOutput),
    ("answer", "sv"): ("swemedmcq.common.answer_response", AnswerLevelResponseOutput),
}
# Released students (the LFM trace students carry the reasoning chat template in their folder).
def student(name: str) -> str:
    """Local copy in paths.yaml `models_dir` if present, else the Hugging Face model."""
    local = path("models_dir") / name
    return str(local) if local.is_dir() else f"Anonymiserad/{name}"


LFM = student("SweMedMCQ-LFM2.5-1.2B-Trace-EN-SV")
EURO = student("SweMedMCQ-EuroLLM-1.7B-Trace-EN-SV")
EURO_FIX = ["--no-eager", "--hf-overrides", '{"max_position_embeddings": 12288}', "--max-model-len", "12288"]
STUDENT = ["--max-model-len", "12288", "--max-output-tokens", "10240"]

GPT_OSS_DATE = "2026-04-07"  # system-prompt date of the GPT-OSS reasoning-off runs (models.yaml)

# Reasoning boundary per family: (start marker or None, end marker).
THINK = (None, "</think>")
HARMONY = ("<|channel|>analysis<|message|>", "<|channel|>final")
GEMMA4 = (None, "<channel|>")  # the thought block is opened in the prompt (--assistant-prefix)
# Markers that must not appear in "off" / "single" outputs.
LEAK_MARKERS = ("</think>", "<|channel|>analysis", "<|channel>", "<channel|>")

# The two refinement-loop models (models.yaml; one 141 GB GPU each in the full runs). Here on
# two 40 GB GPUs (tensor parallel 2) with a shorter context and output.
QWEN38 = "Qwen/Qwen3.8-27B"
GEMMA4_31B = "google/gemma-4-31B-it"
QWEN38_FIX = ("--max-model-len", "16384", "--max-output-tokens", "8192")
GEMMA4_FIX = ("--max-model-len", "8192", "--max-output-tokens", "4096", "--gpu-memory-utilization", "0.92")


@dataclass(frozen=True)
class Check:
    name: str
    model: str
    prompt: str            # trace | answer
    mode: str              # on | off | single
    boundary: tuple | None  # reasoning markers in "on" mode
    flags: tuple[str, ...] = ()
    tokenizer: str | None = None
    token_set: str | None = None  # no_reasoning_tokens.py recipe
    langs: tuple[str, ...] = ("sv", "en")
    expect_fail: bool = False  # negative control


CHECKS = [
    Check("qwen3-4b-thinking-on", "Qwen/Qwen3-4B-Thinking-2507", "trace", "on", THINK,
          ("--max-output-tokens", "8192", "--max-model-len", "16384")),
    Check("qwen3-4b-thinking-off", "Qwen/Qwen3-4B-Thinking-2507", "trace", "off", None,
          ("--max-output-tokens", "8192", "--max-model-len", "16384"), token_set="qwen3_4b_thinking"),
    Check("gpt-oss-20b-on", "openai/gpt-oss-20b", "trace", "on", HARMONY,
          ("--max-output-tokens", "4096", "--max-model-len", "16384")),
    # Negative control: vLLM 0.28's own openai_gptoss parser (what "none" falls back to)
    # starts the JSON grammar at the first token, so there is no analysis channel.
    Check("gpt-oss-20b-on-builtin", "openai/gpt-oss-20b", "trace", "on", HARMONY,
          ("--structured-reasoning-parser", "none", "--max-output-tokens", "4096", "--max-model-len", "16384"),
          langs=("sv",), expect_fail=True),
    Check("gpt-oss-20b-off", "openai/gpt-oss-20b", "trace", "off", None,
          ("--max-output-tokens", "4096", "--max-model-len", "16384"), token_set="gpt_oss"),
    Check("qwen2.5-3b", "Qwen/Qwen2.5-3B-Instruct", "trace", "single", None, ("--max-output-tokens", "4096")),
    Check("lfm-student-on", LFM, "trace", "on", THINK,
          ("--structured-reasoning-parser", "lfm_thinking", "--no-eager", *STUDENT)),
    Check("lfm-student-off", LFM, "trace", "off", None,
          ("--no-reasoning", "--force-structured-output", "--no-eager", *STUDENT)),
    # EuroLLM students were trained with "<think>\n" in the prompt; their template does not add it.
    Check("euro-student-on", EURO, "trace", "on", THINK,
          ("--structured-reasoning-parser", "eurollm_thinking", "--assistant-prefix", "<think>\\n",
           *EURO_FIX, "--max-output-tokens", "10240")),
    # Negative control: without the prefix the model never writes "</think>", so the grammar never starts.
    Check("euro-student-on-noprefix", EURO, "trace", "on", THINK,
          ("--structured-reasoning-parser", "eurollm_thinking", *EURO_FIX, "--max-output-tokens", "10240"),
          langs=("sv",), expect_fail=True),
    Check("euro-student-off", EURO, "trace", "off", None,
          ("--no-reasoning", "--force-structured-output", *EURO_FIX, "--max-output-tokens", "10240")),
    Check("lfm-base-trace", "unsloth/LFM2.5-1.2B-Instruct", "trace", "single", None, ("--no-eager", *STUDENT)),
    # Template default: thinking on at reasoning effort "xhigh"; auto -> qwen3.
    Check("qwen3.8-27b-on", QWEN38, "trace", "on", THINK, QWEN38_FIX),
    Check("qwen3.8-27b-off", QWEN38, "trace", "off", None, ("--no-reasoning", *QWEN38_FIX)),
    # Thinking on only adds <|think|> to the system turn; the prefix opens the thought block.
    Check("gemma-4-31b-on", GEMMA4_31B, "trace", "on", GEMMA4,
          ("--structured-reasoning-parser", "gemma4", "--assistant-prefix", "<|channel>thought\\n", *GEMMA4_FIX)),
    Check("gemma-4-31b-off", GEMMA4_31B, "trace", "off", None, ("--no-reasoning", *GEMMA4_FIX)),
    Check("lfm-base-answer", "unsloth/LFM2.5-1.2B-Instruct", "answer", "single", None,
          ("--no-reasoning", "--force-structured-output", "--no-eager", *STUDENT), langs=("sv",)),
    Check("euro-base-trace", "utter-project/EuroLLM-1.7B-Instruct", "trace", "single", None,
          (*EURO_FIX, "--max-output-tokens", "10240")),
    Check("euro-base-answer", "utter-project/EuroLLM-1.7B-Instruct", "answer", "single", None,
          ("--no-reasoning", "--force-structured-output", *EURO_FIX, "--max-output-tokens", "10240"), langs=("sv",)),
]


# ---------------------------------------------------------------- inputs
def read_rows(path: Path) -> dict[str, dict]:
    with path.open(encoding="utf-8") as handle:
        return {row["prompt_id"]: row for row in map(json.loads, filter(str.strip, handle))}


def user_text(row: dict) -> str:
    return row["prompt"][-1]["content"]


def allowed_letters(row: dict) -> set[str]:
    if row.get("allowed_choices"):
        return set(row["allowed_choices"])
    return set(re.findall(r"^([A-Z])\) ", user_text(row), flags=re.M))


def pick_ids(n: int) -> list[str]:
    """The n shortest Swedish trace prompts that exist in every prompt file."""
    files = [read_rows(path) for path in PROMPTS.values()]
    common = set.intersection(*(set(rows) for rows in files))
    sv = files[0]
    return sorted(common, key=lambda pid: (len(user_text(sv[pid])), pid))[:n]


def subset_file(out: Path, key: tuple[str, str], ids: list[str]) -> Path:
    rows = read_rows(PROMPTS[key])
    path = out / "inputs" / f"{key[0]}_{key[1]}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(rows[pid], ensure_ascii=False) + "\n" for pid in ids), encoding="utf-8")
    return path


# ---------------------------------------------------------------- generation
def generate(check: Check, lang: str, out: Path, ids_file: Path, ids: list[str]) -> Path:
    raw = out / "raw" / check.name / lang
    if all((raw / f"result_{pid}.pkl").exists() for pid in ids):
        return raw
    inputs = subset_file(out, (check.prompt, lang), ids)
    if check.token_set:
        token_path = out / "inputs" / f"{check.token_set}_{lang}.pkl"
        date = ["--date", GPT_OSS_DATE] if check.token_set.startswith("gpt_oss") else []
        subprocess.run([sys.executable, "-m", "swemedmcq.evaluate.no_reasoning_tokens", "--recipe", check.token_set,
                        "--input", str(inputs), "--output", str(token_path), *date], cwd=ROOT, check=True)
        inputs = token_path
    module = MODULES[(check.prompt, lang)][0]
    cmd = [sys.executable, "-m", module, "--input-path", str(inputs), "--output-dir", str(raw),
           "--model-name", check.model, "--tokenizer-path", check.tokenizer or check.model,
           "--temperature", "0", "--gpu-memory-utilization", "0.85", "--prompt-ids-file", str(ids_file), *check.flags]
    log = out / "logs" / f"{check.name}_{lang}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    print("+", " ".join(cmd), flush=True)
    with log.open("w") as handle:
        subprocess.run(cmd, cwd=ROOT, check=True, stdout=handle, stderr=subprocess.STDOUT,
                       env={**os.environ, "PYTHONUNBUFFERED": "1"})
    return raw


# ---------------------------------------------------------------- checks
def split_output(text: str, check: Check) -> tuple[str | None, str]:
    """(reasoning or None, text after the reasoning boundary)."""
    if check.mode != "on":
        return None, text
    start, end = check.boundary
    reasoning, sep, rest = text.partition(end)
    if not sep:  # no boundary: no (finished) reasoning and no JSON after it
        return None, ""
    if start is not None:
        reasoning = reasoning.partition(start)[2] if start in reasoning else reasoning
        rest = rest.partition("<|message|>")[2]  # skip the rest of the final header
    return reasoning, rest


def json_part(content: str) -> str:
    end = content.rfind("}")
    return content[: end + 1] if end >= 0 else content


def check_item(text: str, check: Check, schema: type[BaseModel], letters: set[str]) -> dict:
    reasoning, content = split_output(text, check)
    has_reasoning = bool(reasoning and reasoning.strip())
    leaked = any(marker in text for marker in LEAK_MARKERS)
    a = has_reasoning if check.mode == "on" else not leaked
    # Grammar on from the boundary: JSON is the first thing after it (or at token 1).
    b = content.lstrip().startswith("{") and not (reasoning or "").lstrip().startswith("{")
    try:
        payload = schema.model_validate_json(json_part(content.strip()))
        c = payload.answer_index in letters
        answer = payload.answer_index
    except ValidationError:
        c, answer = False, None
    return {"a": a, "b": b, "c": c, "answer": answer, "reasoning_chars": len(reasoning or "")}


def evaluate(check: Check, lang: str, raw: Path, ids: list[str]) -> dict:
    name = check.tokenizer or check.model
    tokenizer = AutoTokenizer.from_pretrained(str(ROOT / name) if (ROOT / name).exists() else name)
    rows = read_rows(PROMPTS[(check.prompt, lang)])
    schema = MODULES[(check.prompt, lang)][1]
    items, finish = [], {}
    for pid in ids:
        with (raw / f"result_{pid}.pkl").open("rb") as handle:
            result = pickle.load(handle)
        output = result.outputs[0]
        text = tokenizer.decode(output.token_ids, skip_special_tokens=False)
        finish[output.finish_reason] = finish.get(output.finish_reason, 0) + 1
        item = check_item(text, check, schema, allowed_letters(rows[pid]))
        item.update(prompt_id=pid, finish=output.finish_reason, tokens=len(output.token_ids),
                    prompt_tail=tokenizer.decode(result.prompt_token_ids[-12:], skip_special_tokens=False),
                    head=text[:160], tail=text[-160:])
        items.append(item)
    n = len(items)
    return {
        "check": check.name, "lang": lang, "mode": check.mode, "n": n,
        "a": sum(i["a"] for i in items), "b": sum(i["b"] for i in items), "c": sum(i["c"] for i in items),
        "finish": finish, "expect_fail": check.expect_fail,
        "ok": all(i["a"] and i["b"] and i["c"] for i in items if i["finish"] != "length") != check.expect_fail,
        "items": items,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--only", default=".", help="Regex on check names.")
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--skip-generate", action="store_true")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    ids_file = args.out / "prompt_ids.txt"
    if not ids_file.exists():
        ids_file.write_text("\n".join(pick_ids(args.n)) + "\n")
    ids = ids_file.read_text().split()

    reports = []
    for check in CHECKS:
        if not re.search(args.only, check.name):
            continue
        for lang in (check.langs if check.prompt == "trace" else ("sv",)):
            raw = args.out / "raw" / check.name / lang
            if not args.skip_generate:
                raw = generate(check, lang, args.out, ids_file, ids)
            report = evaluate(check, lang, raw, ids)
            reports.append(report)
            print(f"{check.name:24s} {lang}  a={report['a']}/{report['n']} b={report['b']}/{report['n']} "
                  f"c={report['c']}/{report['n']} finish={report['finish']}  {'OK' if report['ok'] else 'FAIL'}{' (expected to fail)' if check.expect_fail else ''}",
                  flush=True)
            (args.out / "reports").mkdir(exist_ok=True)
            (args.out / "reports" / f"{check.name}_{lang}.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    if not all(r["ok"] for r in reports):
        raise SystemExit("Some checks failed.")


if __name__ == "__main__":
    main()
