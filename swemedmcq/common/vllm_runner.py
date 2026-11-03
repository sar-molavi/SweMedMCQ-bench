"""Batch generation with vLLM: one pickled ``RequestOutput`` per prompt.

Callers set ``config.structured_output`` to a pydantic schema and call ``main``::

    python -m swemedmcq.common.sw_response -i prompts.jsonl -o out/ --model-name M [flags]

Input is a ``.jsonl`` of ``{"prompt_id", "prompt": [chat messages]}`` (rendered
with the tokenizer's chat template) or a ``.pkl`` of ``{prompt_id: [token ids]}``.
Results are written to ``<output-dir>/result_<prompt_id>.pkl``; existing results
are skipped, so a run can be resumed.

Reasoning and JSON: vLLM applies the JSON grammar only after the reasoning
parser reports the end of the reasoning (``is_reasoning_end`` on the prompt,
then on the generated tokens). ``REASONING_PARSERS`` lists the parsers;
``--structured-reasoning-parser auto`` picks one from the model name.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from pydantic import BaseModel
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import SamplingParams, TokensPrompt
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.outputs import RequestOutput
from vllm.sampling_params import StructuredOutputsParams

logger = logging.getLogger(__name__)

_PARSERS = Path(__file__).parent / "parsers"

# Reasoning parser name -> plugin file (None: built into vLLM).
REASONING_PARSERS: dict[str, Path | None] = {
    "qwen3": None,  # JSON after </think>; a </think> prefilled in the prompt counts
    # JSON after <channel|>. With thinking on, the template only adds <|think|> to the system
    # turn, so pass --assistant-prefix '<|channel>thought\n' to open the thought block (else
    # the model may answer directly and the grammar never starts). Off: the template closes
    # an empty block, which counts as ended.
    "gemma4": None,
    "gptoss_final": _PARSERS / "gptoss_final_parser.py",  # JSON after <|channel|>final..<|message|>
    "lfm_thinking": _PARSERS / "lfm_thinking_parser.py",  # JSON after the </think> token
    "eurollm_thinking": _PARSERS / "eurollm_thinking_parser.py",  # JSON after "</think>" text
    "force_structured": _PARSERS / "force_structured_parser.py",  # JSON from the first token
}
# "none": no parser, JSON from the first token. (For GPT-OSS vLLM then sets its
# own `openai_gptoss` parser, which also starts JSON at the first token.)
PARSER_CHOICES = ("auto", "none", *REASONING_PARSERS)

# "auto": first model-name fragment that matches picks the parser; no match -> none.
# Matched against the lower-cased name; "gemma-4" does not match medgemma-27b or Gemma 2/3.
AUTO_PARSERS = (("gpt-oss", "gptoss_final"), ("qwen3", "qwen3"), ("gemma-4", "gemma4"))


@dataclass
class VLLMRunnerConfig:
    """Settings for one generation run."""

    input_path: Path
    output_dir: Path
    model_name: str
    structured_output: type[BaseModel] | None = None
    tokenizer_path: str | None = None
    revision: str | None = None
    temperature: float = 0.0
    max_output_tokens: int = 30 * 1024
    max_model_len: int | None = None
    num_samples: int = 1
    gpu_memory_utilization: float = 0.90
    cuda_visible_devices: str | None = None
    data_parallel_size: int = 1
    enforce_eager: bool = False
    hf_overrides: dict[str, Any] = field(default_factory=dict)
    enable_thinking: bool = True
    assistant_prefix: str = ""
    structured_reasoning_parser: str = "auto"
    force_structured_output: bool = False
    prompt_ids_file: Path | None = None

    def __post_init__(self) -> None:
        self.tokenizer_path = self.tokenizer_path or self.model_name
        if self.data_parallel_size < 1:
            raise ValueError("data_parallel_size must be positive")
        if self.structured_reasoning_parser not in PARSER_CHOICES:
            raise ValueError(
                f"structured_reasoning_parser must be one of {PARSER_CHOICES}, "
                f"got {self.structured_reasoning_parser!r}."
            )
        if self.force_structured_output and self.structured_reasoning_parser not in (
            "auto",
            "force_structured",
        ):
            raise ValueError(
                "--force-structured-output conflicts with "
                f"--structured-reasoning-parser {self.structured_reasoning_parser}."
            )

    @property
    def reasoning_parser(self) -> str | None:
        """Parser name gating the JSON grammar, or None (grammar from the first token)."""
        if self.structured_output is None:
            return None
        if self.force_structured_output:
            return "force_structured"
        if self.structured_reasoning_parser == "none":
            return None
        if self.structured_reasoning_parser != "auto":
            return self.structured_reasoning_parser
        name = self.model_name.lower()
        return next((parser for key, parser in AUTO_PARSERS if key in name), None)


# ---------------------------------------------------------------- engine
def structured_outputs_config(config: VLLMRunnerConfig) -> dict[str, Any] | None:
    """Engine-level reasoning parser settings for the structured-output grammar."""
    parser = config.reasoning_parser
    if parser is None:
        return None
    out: dict[str, Any] = {"reasoning_parser": parser, "enable_in_reasoning": False}
    plugin = REASONING_PARSERS[parser]
    if plugin is not None:
        out["reasoning_parser_plugin"] = str(plugin)
    return out


def engine_kwargs(config: VLLMRunnerConfig) -> dict[str, Any]:
    """Keyword arguments for ``AsyncEngineArgs``."""
    visible_gpus = torch.cuda.device_count()
    if visible_gpus < 1 or visible_gpus % config.data_parallel_size:
        raise ValueError(
            f"{visible_gpus} visible GPUs cannot be divided into "
            f"{config.data_parallel_size} data-parallel replicas"
        )
    kwargs: dict[str, Any] = dict(
        model=config.model_name,
        tokenizer=config.tokenizer_path,
        tensor_parallel_size=visible_gpus // config.data_parallel_size,
        data_parallel_size=config.data_parallel_size,
        gpu_memory_utilization=config.gpu_memory_utilization,
        enforce_eager=config.enforce_eager,
        enable_prefix_caching=True,
        language_model_only=True,
    )
    if config.max_model_len is not None:
        kwargs["max_model_len"] = config.max_model_len
    if config.hf_overrides:
        kwargs["hf_overrides"] = config.hf_overrides
    if config.revision is not None:
        kwargs["revision"] = config.revision
        if config.tokenizer_path == config.model_name:
            kwargs["tokenizer_revision"] = config.revision
    so_config = structured_outputs_config(config)
    if so_config is not None:
        kwargs["structured_outputs_config"] = so_config
    return kwargs


def build_engine(config: VLLMRunnerConfig) -> AsyncLLMEngine:
    return AsyncLLMEngine.from_engine_args(AsyncEngineArgs(**engine_kwargs(config)))


def sampling_params(config: VLLMRunnerConfig) -> SamplingParams:
    params = SamplingParams(
        temperature=config.temperature,
        max_tokens=config.max_output_tokens,
        n=config.num_samples,
        detokenize=False,
    )
    if config.structured_output is not None:
        schema = config.structured_output.model_json_schema()
        params.structured_outputs = StructuredOutputsParams(json=schema)
    return params


# ---------------------------------------------------------------- prompts
def read_prompt_ids_file(path: Path) -> set[str]:
    """Plain ids, one per line, or the tab-separated ``*_failed.txt`` format (id last)."""
    ids = {line.split("\t")[-1].strip() for line in path.read_text(encoding="utf-8").splitlines()}
    ids.discard("")
    if not ids:
        raise ValueError(f"No prompt ids in {path}")
    return ids


def load_prompts(config: VLLMRunnerConfig) -> dict[str, list[int]]:
    """All prompts of the input file as token ids, keyed by prompt id."""
    suffix = config.input_path.suffix.lower()
    if suffix == ".pkl":
        with config.input_path.open("rb") as handle:
            return pickle.load(handle)
    if suffix != ".jsonl":
        raise ValueError(f"Unsupported input format: {config.input_path}")
    with config.input_path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    revision = config.revision if config.tokenizer_path == config.model_name else None
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_path, revision=revision)
    texts = tokenizer.apply_chat_template(
        [row["prompt"] for row in rows],
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=config.enable_thinking,
    )
    # Same tokenisation as apply_chat_template(tokenize=True), plus the prefix.
    texts = [text + config.assistant_prefix for text in texts]
    token_ids = tokenizer(texts, add_special_tokens=False)["input_ids"]
    return {row["prompt_id"]: ids for row, ids in zip(rows, token_ids, strict=True)}


def completed_ids(output_dir: Path) -> set[str]:
    return {path.stem.removeprefix("result_") for path in output_dir.glob("result_*.pkl")}


def select_pending(config: VLLMRunnerConfig, prompts: dict[str, list[int]]) -> dict[str, list[int]]:
    """Prompts without a result; with ``--prompt-ids-file`` exactly those ids (rerun)."""
    if config.prompt_ids_file is None:
        done = completed_ids(config.output_dir)
        logger.info("Found %d completed prompts in %s.", len(done), config.output_dir)
        return {pid: ids for pid, ids in prompts.items() if pid not in done}
    wanted = read_prompt_ids_file(config.prompt_ids_file)
    missing = sorted(wanted - prompts.keys())
    if missing:
        raise ValueError(
            f"{len(missing)} ids from {config.prompt_ids_file} are not in "
            f"{config.input_path}: {', '.join(missing[:10])}"
        )
    logger.info("Regenerating %d prompts listed in %s.", len(wanted), config.prompt_ids_file)
    return {pid: prompts[pid] for pid in sorted(wanted)}


# ---------------------------------------------------------------- generation
def save_result(output_dir: Path, prompt_id: str, result: RequestOutput) -> None:
    """Atomic write: a crash never leaves a partial ``result_*.pkl``."""
    path = output_dir / f"result_{prompt_id}.pkl"
    tmp = output_dir / f".result_{prompt_id}.pkl.tmp"
    try:
        with tmp.open("wb") as handle:
            pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


async def generate_one(
    engine: AsyncLLMEngine,
    config: VLLMRunnerConfig,
    params: SamplingParams,
    prompt_id: str,
    token_ids: list[int],
) -> None:
    final = None
    async for output in engine.generate(
        TokensPrompt(prompt_token_ids=token_ids),
        sampling_params=params,
        request_id=f"prompt-{prompt_id}",
        # Same parser kwargs as vLLM's chat server, so parsers see enable_thinking.
        reasoning_parser_kwargs={"chat_template_kwargs": {"enable_thinking": config.enable_thinking}},
    ):
        final = output
    if final is None:
        raise RuntimeError(f"No output for prompt {prompt_id}")
    # Pickling and fsync block; keep them off the loop that drives vLLM.
    await asyncio.to_thread(save_result, config.output_dir, prompt_id, final)


async def generate_all(engine: AsyncLLMEngine, config: VLLMRunnerConfig, prompts: dict[str, list[int]]) -> None:
    params = sampling_params(config)
    tasks = [
        asyncio.create_task(generate_one(engine, config, params, pid, ids))
        for pid, ids in prompts.items()
    ]
    failed = 0
    for future in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Prompts"):
        try:
            await future
        except Exception as exc:  # keep the other prompts running; resume retries it
            failed += 1
            logger.error("A prompt failed: %r", exc)
    logger.info("Finished: %d ok, %d failed.", len(tasks) - failed, failed)


def main(config: VLLMRunnerConfig) -> None:
    """Generate every pending prompt of ``config.input_path``."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    if config.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = config.cuda_visible_devices
    config.output_dir.mkdir(parents=True, exist_ok=True)

    prompts = select_pending(config, load_prompts(config))
    if not prompts:
        logger.info("All prompts have already been processed.")
        return
    logger.info("Reasoning parser: %s", config.reasoning_parser)
    engine = build_engine(config)
    try:
        asyncio.run(generate_all(engine, config, prompts))
    finally:
        engine.shutdown()


# ---------------------------------------------------------------- CLI
def _non_negative_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError:
        parsed = -1.0
    if not 0 <= parsed < float("inf"):
        raise argparse.ArgumentTypeError(f"Expected a non-negative number, got {value!r}.")
    return parsed


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError:
        parsed = 0
    if parsed < 1:
        raise argparse.ArgumentTypeError(f"Expected a positive integer, got {value!r}.")
    return parsed


def _unescape(value: str) -> str:
    """Backslash escapes as in Python strings, so '<think>\\n' ends with a newline."""
    return value.encode("latin-1", "backslashreplace").decode("unicode_escape")


def _json_object(value: str) -> dict[str, Any]:
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError(f"Expected a JSON object, got {value!r}.")
    return parsed


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-i", "--input-path", type=Path, required=True, help=".jsonl chat prompts or .pkl token ids.")
    p.add_argument("-o", "--output-dir", type=Path, required=True)
    p.add_argument("--model-name", required=True)
    p.add_argument("--tokenizer-path", default=None, help="Defaults to --model-name.")
    p.add_argument("--revision", default=None, help="Hugging Face snapshot (commit) of --model-name; default: the cached refs/main.")
    p.add_argument("--temperature", type=_non_negative_float, default=0.0)
    p.add_argument("--max-output-tokens", type=int, default=30 * 1024)
    p.add_argument("--max-model-len", type=int, default=None, help="Input + output tokens.")
    p.add_argument("--num-samples", type=int, default=1, help="Completions per prompt.")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    p.add_argument("--cuda-visible-devices", default=None, help="Restrict GPUs visible to vLLM.")
    p.add_argument("--data-parallel-size", type=_positive_int, default=1,
                   help="Replicate the model across this many GPUs; remaining GPUs per replica use tensor parallelism.")
    eager = p.add_mutually_exclusive_group()
    eager.add_argument("--eager", dest="enforce_eager", action="store_true", help="No CUDA graphs or torch.compile.")
    eager.add_argument("--no-eager", dest="enforce_eager", action="store_false", help="torch.compile + CUDA graphs (default).")
    p.set_defaults(enforce_eager=False)
    p.add_argument(
        "--hf-overrides",
        type=_json_object,
        default={},
        help='JSON merged into the HF config, e.g. \'{"max_position_embeddings": 12288}\'.',
    )
    p.add_argument("--no-reasoning", action="store_true", help="Render .jsonl prompts with enable_thinking=False.")
    p.add_argument(
        "--assistant-prefix",
        type=_unescape,
        default="",
        help="Text appended to rendered .jsonl prompts after the generation prompt (backslash "
        "escapes allowed), e.g. '<think>\\n' for EuroLLM students (their template does not open the trained think block).",
    )
    p.add_argument(
        "--structured-reasoning-parser",
        choices=PARSER_CHOICES,
        default="auto",
        help="Parser that decides when the JSON grammar starts (auto: from the model name).",
    )
    p.add_argument(
        "--force-structured-output",
        action="store_true",
        help="JSON grammar from the first generated token (same as the force_structured parser).",
    )
    p.add_argument(
        "--prompt-ids-file",
        type=Path,
        default=None,
        help="Only these prompt ids (plain or *_failed.txt format); existing results are overwritten.",
    )
    return p


def parse_args(argv: list[str] | None = None) -> VLLMRunnerConfig:
    args = build_arg_parser().parse_args(argv)
    return VLLMRunnerConfig(
        input_path=args.input_path,
        output_dir=args.output_dir,
        model_name=args.model_name,
        tokenizer_path=args.tokenizer_path,
        revision=args.revision,
        temperature=args.temperature,
        max_output_tokens=args.max_output_tokens,
        max_model_len=args.max_model_len,
        num_samples=args.num_samples,
        gpu_memory_utilization=args.gpu_memory_utilization,
        cuda_visible_devices=args.cuda_visible_devices,
        data_parallel_size=args.data_parallel_size,
        enforce_eager=args.enforce_eager,
        hf_overrides=args.hf_overrides,
        enable_thinking=not args.no_reasoning,
        assistant_prefix=args.assistant_prefix,
        structured_reasoning_parser=args.structured_reasoning_parser,
        force_structured_output=args.force_structured_output,
        prompt_ids_file=args.prompt_ids_file,
    )


if __name__ == "__main__":
    main(parse_args())
