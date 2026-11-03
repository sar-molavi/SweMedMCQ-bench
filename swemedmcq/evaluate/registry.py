"""Model registry (configs/models.yaml) and test sets.

A *job* is one model configuration on one test set. Everything about a job
(prompt file, scoring source, output folders, runner flags) is derived here, so
generation, scoring, job lists and collection agree on every path.

Test sets (files written by ``swemedmcq.testset.build`` to ``<testset_dir>/data``):
  en           English items (``en_*``)
  sv_L4a       Swedish items, reported layer L4a (``sv_L4a_*``)
  sv_L4_delta  Swedish items whose L4 text differs from L4a (``sv_L4_delta_*``)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from swemedmcq import paths

SETS = ("en", "sv_L4a", "sv_L4_delta")
# Model values starting with these prefixes are folders in the repository; others are HF ids.
LOCAL_PREFIXES = ("hf/", "runs/", "work/", "./", "/")


@dataclass(frozen=True)
class Model:
    name: str
    part: str            # benchmark | student
    prompt: str          # trace | answer
    model: str
    mode: str
    max_out: int
    sets: tuple[str, ...]
    gpus: int = 1
    tokenizer: str | None = None
    revision: str | None = None
    parser: str | None = None
    date: str | None = None
    prefix: str | None = None
    max_len: int | None = None
    eager: bool = False  # True: vLLM eager mode (no torch.compile / CUDA graphs)
    hf_overrides: dict[str, Any] | None = None
    labels: dict[str, str] = field(default_factory=dict)

    @property
    def decoding(self) -> str:
        """reasoning | no_reasoning for the trace students, default otherwise."""
        if self.part == "student" and self.prompt == "trace" and self.labels.get("regime") != "base":
            return "reasoning" if self.mode == "reasoning" else "no_reasoning"
        return "default"

    def path(self, value: str | None) -> str | None:
        """Local checkpoints are relative to the repository root; HF ids are kept as they are."""
        if value is None or not value.startswith(LOCAL_PREFIXES):
            return value
        return str(paths.resolve(value))

    @property
    def model_path(self) -> str:
        return self.path(self.model)

    @property
    def tokenizer_path(self) -> str:
        return self.path(self.tokenizer or self.model)


def models() -> list[Model]:
    raw = paths.load_config("models")["models"]
    out = []
    for entry in raw:
        entry = {k: v for k, v in entry.items() if k in Model.__dataclass_fields__}
        entry["sets"] = tuple(entry["sets"])
        out.append(Model(**entry))
    names = [m.name for m in out]
    dupes = {n for n in names if names.count(n) > 1}
    if dupes:
        raise ValueError(f"Duplicate model names in models.yaml: {sorted(dupes)}")
    return out


def model(name: str) -> Model:
    for m in models():
        if m.name == name:
            return m
    raise KeyError(f"No model named {name!r} in configs/models.yaml")


def data_dir() -> Path:
    return paths.path("testset_dir") / "data"


@dataclass(frozen=True)
class Job:
    model: Model
    set: str

    @property
    def id(self) -> str:
        return f"{self.set}/{self.model.name}"

    @property
    def lang(self) -> str:
        return "en" if self.set == "en" else "sv"

    @property
    def layer(self) -> str:
        """en | L4a | L4_delta: the part of the prompt file names after the language."""
        return self.set.removeprefix("sv_")

    # ------------------------------------------------------------ inputs
    def prompts(self) -> Path:
        if self.model.prompt == "trace":
            return data_dir() / ("en_trace.jsonl" if self.lang == "en" else f"sv_{self.layer}_trace.jsonl")
        return data_dir() / ("en_answer_prompts.jsonl" if self.lang == "en" else f"sv_{self.layer}_answer_prompts.jsonl")

    def answer_split(self) -> Path:
        """Gold labels with the answer-only prompt (also the scoring source)."""
        return data_dir() / ("en_answer_split.jsonl" if self.lang == "en" else f"sv_{self.layer}_answer_split.jsonl")

    def module(self) -> str:
        if self.model.prompt == "answer":
            return "swemedmcq.common.answer_response"
        return "swemedmcq.common.en_response" if self.lang == "en" else "swemedmcq.common.sw_response"

    # ------------------------------------------------------------ outputs
    @property
    def raw_dir(self) -> Path:
        return paths.path("results_dir") / "raw" / self.set / self.model.name

    @property
    def scored_dir(self) -> Path:
        return paths.path("results_dir") / "scored" / self.set / self.model.name

    @property
    def log(self) -> Path:
        return paths.path("results_dir") / "logs" / self.set / f"{self.model.name}.log"

    @property
    def done_marker(self) -> Path:
        return self.scored_dir / "DONE"

    def expected(self) -> int:
        with self.prompts().open(encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())


def jobs(names: list[str] | None = None) -> list[Job]:
    out = []
    for m in models():
        if names and m.name not in names:
            continue
        out.extend(Job(m, s) for s in m.sets)
    return out


def parse_job(job_id: str) -> Job:
    set_, name = job_id.split("/", 1)
    if set_ not in SETS:
        raise ValueError(f"Unknown test set {set_!r}; expected one of {SETS}")
    return Job(model(name), set_)


def hf_overrides_arg(m: Model) -> list[str]:
    return ["--hf-overrides", json.dumps(m.hf_overrides)] if m.hf_overrides else []
