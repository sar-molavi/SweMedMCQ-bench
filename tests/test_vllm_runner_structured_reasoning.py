"""Engine settings of the vLLM batch runner per model family and flag (no GPU)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import BaseModel

import swemedmcq.common.vllm_runner as vllm_oop
from swemedmcq.common.parsers.eurollm_thinking_parser import EuroLLMThinkingParser
from swemedmcq.common.parsers.gptoss_final_parser import GptOssFinalChannelParser
from swemedmcq.common.parsers.lfm_thinking_parser import LFMThinkingParser


class DummyOutput(BaseModel):
    answer: str


LFM = "outputs_bilingual/sft_runs/lfm/final_checkpoint"
EURO = "outputs_bilingual/sft_runs/eurollm/final_checkpoint"


def parse(*flags: str, model: str = LFM, schema: type[BaseModel] | None = DummyOutput):
    config = vllm_oop.parse_args(["-i", "prompts.jsonl", "-o", "out", "--model-name", model, *flags])
    config.structured_output = schema
    return config


@pytest.fixture
def engine_kwargs(monkeypatch):
    """Run build_engine with fake vLLM classes; return the AsyncEngineArgs kwargs."""
    captured: dict = {}

    class FakeEngineArgs:
        def __init__(self, **kwargs):
            captured.clear()
            captured.update(kwargs)

    class FakeEngine:
        @staticmethod
        def from_engine_args(engine_args):
            return engine_args

    monkeypatch.setattr(vllm_oop, "AsyncEngineArgs", FakeEngineArgs)
    monkeypatch.setattr(vllm_oop, "AsyncLLMEngine", FakeEngine)
    monkeypatch.setattr(vllm_oop.torch.cuda, "device_count", lambda: 2)

    def build(config):
        vllm_oop.build_engine(config)
        return dict(captured)

    return build


# ------------------------------------------------------------ parser table
@pytest.mark.parametrize(
    ("model", "flags", "parser"),
    [
        # "auto" (default): chosen from the model name.
        ("Qwen/Qwen3-4B-Thinking-2507", (), "qwen3"),
        ("Qwen/Qwen3.5-35B-A3B", ("--no-reasoning",), "qwen3"),
        ("Qwen/Qwen3.8-27B", (), "qwen3"),
        ("Qwen/Qwen3.8-27B", ("--no-reasoning",), "qwen3"),
        ("google/gemma-4-31B-it", (), "gemma4"),
        ("google/gemma-4-31B-it", ("--no-reasoning",), "gemma4"),
        ("google/gemma-4-31B-it", ("--structured-reasoning-parser", "gemma4"), "gemma4"),
        ("/cache/models--google--gemma-4-31B-it/snapshots/842da37", (), "gemma4"),
        ("openai/gpt-oss-20b", (), "gptoss_final"),
        ("openai/gpt-oss-120b", (), "gptoss_final"),
        ("Qwen/Qwen2.5-3B-Instruct", (), None),
        ("google/medgemma-27b-text-it", (), None),
        ("unsloth/LFM2.5-1.2B-Instruct", (), None),
        ("utter-project/EuroLLM-1.7B-Instruct", (), None),
        (LFM, ("--no-reasoning",), None),
        # Explicit values win over the name match.
        (LFM, ("--structured-reasoning-parser", "lfm_thinking"), "lfm_thinking"),
        (EURO, ("--structured-reasoning-parser", "eurollm_thinking"), "eurollm_thinking"),
        ("Qwen/Qwen3-4B-Thinking-2507", ("--structured-reasoning-parser", "none"), None),
        ("Qwen/Qwen3-4B-Thinking-2507", ("--force-structured-output",), "force_structured"),
        (LFM, ("--no-reasoning", "--force-structured-output"), "force_structured"),
        (EURO, ("--structured-reasoning-parser", "force_structured"), "force_structured"),
    ],
)
def test_parser_selection(model, flags, parser, engine_kwargs) -> None:
    kwargs = engine_kwargs(parse(*flags, model=model))
    so_config = kwargs.get("structured_outputs_config")
    if parser is None:
        assert so_config is None
        return
    assert so_config["reasoning_parser"] == parser
    assert so_config["enable_in_reasoning"] is False
    plugin = vllm_oop.REASONING_PARSERS[parser]
    if plugin is None:
        assert "reasoning_parser_plugin" not in so_config
    else:
        assert so_config["reasoning_parser_plugin"] == str(plugin)
        assert Path(so_config["reasoning_parser_plugin"]).is_file()


def test_no_schema_means_no_parser(engine_kwargs) -> None:
    kwargs = engine_kwargs(parse(model="Qwen/Qwen3-4B-Thinking-2507", schema=None))
    assert "structured_outputs_config" not in kwargs
    assert vllm_oop.sampling_params(parse(schema=None)).structured_outputs is None


def test_force_conflicts_with_other_explicit_parser() -> None:
    with pytest.raises(ValueError):
        parse("--force-structured-output", "--structured-reasoning-parser", "lfm_thinking")


# ------------------------------------------------------------ engine and sampling
def test_engine_defaults(engine_kwargs) -> None:
    kwargs = engine_kwargs(parse())
    assert kwargs["model"] == kwargs["tokenizer"] == LFM
    assert kwargs["enforce_eager"] is False  # compiled unless --eager
    assert kwargs["tensor_parallel_size"] == 2
    assert kwargs["enable_prefix_caching"] is True
    assert kwargs["gpu_memory_utilization"] == 0.90
    assert "max_model_len" not in kwargs and "hf_overrides" not in kwargs


def test_engine_flags(engine_kwargs) -> None:
    kwargs = engine_kwargs(
        parse(
            "--no-eager",
            "--tokenizer-path", "outputs/lfm-2.5-1.2b-dynamic-tokenizer",
            "--max-model-len", "12288",
            "--gpu-memory-utilization", "0.8",
            "--hf-overrides", '{"max_position_embeddings": 12288}',
            model=EURO,
        )
    )
    assert kwargs["enforce_eager"] is False
    assert kwargs["tokenizer"] == "outputs/lfm-2.5-1.2b-dynamic-tokenizer"
    assert kwargs["max_model_len"] == 12288
    assert kwargs["gpu_memory_utilization"] == 0.8
    assert kwargs["hf_overrides"] == {"max_position_embeddings": 12288}


def test_eager_flags() -> None:
    assert parse("--eager").enforce_eager is True
    assert parse("--no-eager").enforce_eager is False
    with pytest.raises(SystemExit):
        parse("--eager", "--no-eager")


def test_sampling_params() -> None:
    config = parse("--temperature", "0.7", "--max-output-tokens", "4096", "--num-samples", "2")
    params = vllm_oop.sampling_params(config)
    assert params.temperature == 0.7 and params.max_tokens == 4096 and params.n == 2
    assert params.structured_outputs.json == DummyOutput.model_json_schema()
    assert parse("--no-reasoning").enable_thinking is False
    with pytest.raises(SystemExit):
        parse("--temperature", "-1")


def test_jsonl_prompts_use_chat_template_and_prefix(tmp_path, monkeypatch) -> None:
    class Tok:
        def apply_chat_template(self, conversations, add_generation_prompt, tokenize, enable_thinking):
            assert add_generation_prompt and not tokenize and enable_thinking is False
            return [f"<user>{c[0]['content']}<assistant>" for c in conversations]

        def __call__(self, texts, add_special_tokens):
            assert add_special_tokens is False
            return {"input_ids": [[ord(ch) for ch in text] for text in texts]}

    monkeypatch.setattr(vllm_oop.AutoTokenizer, "from_pretrained", lambda path, **_: Tok())
    path = tmp_path / "p.jsonl"
    path.write_text('{"prompt_id": "q1", "prompt": [{"role": "user", "content": "hi"}]}\n')
    config = parse("--no-reasoning", "--assistant-prefix", "<think>\\n")
    config.input_path = path
    assert config.assistant_prefix == "<think>\n"
    assert vllm_oop.load_prompts(config) == {"q1": [ord(ch) for ch in "<user>hi<assistant><think>\n"]}


def test_prompt_selection(tmp_path) -> None:
    out = tmp_path / "out"
    out.mkdir()
    (out / "result_a.pkl").write_bytes(b"")
    config = parse()
    config.output_dir = out
    prompts = {"a": [1], "b": [2], "c": [3]}
    assert vllm_oop.select_pending(config, prompts) == {"b": [2], "c": [3]}

    ids = tmp_path / "ids.txt"
    ids.write_text("x\ta\nb\n\n")
    config.prompt_ids_file = ids
    assert vllm_oop.select_pending(config, prompts) == {"a": [1], "b": [2]}  # rerun listed ids
    ids.write_text("zzz\n")
    with pytest.raises(ValueError):
        vllm_oop.select_pending(config, prompts)


# ------------------------------------------------------------ parser plugins
class FakeTokenizer:
    """One token per character; special tokens are single ids."""

    SPECIAL = ["<think>", "</think>", "<|channel|>", "<|message|>", "<|end|>", "<|start|>", "<|constrain|>"]

    def get_vocab(self) -> dict[str, int]:
        return {token: 1000 + i for i, token in enumerate(self.SPECIAL)}

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        ids, i = [], 0
        while i < len(text):
            special = next((s for s in self.SPECIAL if text.startswith(s, i)), None)
            if special:
                ids.append(self.get_vocab()[special])
                i += len(special)
            else:
                ids.append(ord(text[i]))
                i += 1
        return ids

    def decode(self, ids, skip_special_tokens: bool = False) -> str:
        inverse = {v: k for k, v in self.get_vocab().items()}
        return "".join(inverse.get(i, chr(i)) for i in ids)


TOK = FakeTokenizer()


def streaming_end_index(parser, prompt: str, output: str) -> int | None:
    """Index into the output at which the decode-step check first fires (one token per step)."""
    ids = TOK.encode(prompt)
    for index, token in enumerate(TOK.encode(output)):
        ids.append(token)
        if parser.is_reasoning_end_streaming(ids, [token]):
            return index
    return None


def test_lfm_parser() -> None:
    parser = LFMThinkingParser(TOK)
    assert parser.extract_reasoning('work it out</think>{"answer":"B"}', request=None) == (
        "work it out",
        '{"answer":"B"}',
    )
    assert not parser.is_reasoning_end(TOK.encode("user: q\nassistant: "))
    output = 'abc</think>{"a"}'
    assert streaming_end_index(parser, "q", output) == output.index("</think>")


class CharTokenizer(FakeTokenizer):
    """EuroLLM: <think> and </think> are plain text (several tokens), not vocabulary items."""

    SPECIAL: list[str] = []


def test_eurollm_parser() -> None:
    tok = CharTokenizer()
    parser = EuroLLMThinkingParser(tok)
    assert not parser.is_reasoning_end(tok.encode("<|im_start|>assistant\n"))
    output = "x" * 50 + "\n</think>\n\n{}"
    ids = tok.encode("q")
    fired = None
    for index, token in enumerate(tok.encode(output)):
        ids.append(token)
        if parser.is_reasoning_end_streaming(ids, [token]):
            fired = index
            break
    assert fired == output.index("{") - 1  # after the blank line that follows </think>
    assert parser.extract_content_ids(tok.encode(output)) == tok.encode("{}")
    assert not parser.is_reasoning_end(tok.encode("x</think>"))
    off = EuroLLMThinkingParser(tok, chat_template_kwargs={"enable_thinking": False})
    assert off.is_reasoning_end(tok.encode("q")) and off.is_reasoning_end_streaming([1], [1])


def test_gptoss_final_parser() -> None:
    parser = GptOssFinalChannelParser(TOK)
    prompt = "<|start|>user<|message|>q<|end|><|start|>assistant"
    assert not parser.is_reasoning_end(TOK.encode(prompt))
    # "Reasoning off" token set: the prompt already ends with the final header.
    assert parser.is_reasoning_end(TOK.encode(prompt + "<|channel|>analysis<|message|><|end|><|start|>assistant<|channel|>final<|message|>"))
    for header in ("<|channel|>final<|message|>", "<|channel|>final <|constrain|>json<|message|>"):
        output = "<|channel|>analysis<|message|>think<|end|><|start|>assistant" + header + '{"a":1}'
        assert streaming_end_index(parser, prompt, output) == len(TOK.encode(output)) - len('{"a":1}') - 1
        assert parser.extract_content_ids(TOK.encode(output)) == TOK.encode('{"a":1}')
    # The analysis header does not end the reasoning.
    assert streaming_end_index(parser, prompt, "<|channel|>analysis<|message|>final answer") is None


# ------------------------------------------------------------ real templates + vLLM parsers
# Tokenizers from the local HF cache (skipped where they are not downloaded). These check
# the rendered prompt (template + --assistant-prefix) against vLLM's built-in parsers.
GEMMA4 = "google/gemma-4-31B-it"
QWEN38 = "Qwen/Qwen3.8-27B"
MESSAGES = [{"role": "user", "content": "Fråga: A) ja B) nej"}]


_TOKENIZERS: dict = {}


def real_tokenizer(name: str):
    # Loaded once, before load_prompts' AutoTokenizer is monkeypatched to return it.
    if name not in _TOKENIZERS:
        try:
            _TOKENIZERS[name] = vllm_oop.AutoTokenizer.from_pretrained(name, local_files_only=True)
        except OSError:
            pytest.skip(f"{name} tokenizer not in the local HF cache")
    return _TOKENIZERS[name]


def real_prompt(tmp_path, monkeypatch, model: str, *flags: str) -> list[int]:
    tokenizer = real_tokenizer(model)
    monkeypatch.setattr(vllm_oop.AutoTokenizer, "from_pretrained", lambda path, **_: tokenizer)
    path = tmp_path / "p.jsonl"
    path.write_text(json.dumps({"prompt_id": "q", "prompt": MESSAGES}, ensure_ascii=False) + "\n")
    config = parse(*flags, model=model)
    config.input_path = path
    return vllm_oop.load_prompts(config)["q"]


def vllm_parser(name: str, tokenizer, enable_thinking: bool):
    from vllm.reasoning import ReasoningParserManager

    cls = ReasoningParserManager.get_reasoning_parser(name)
    # The runner passes exactly these kwargs (reasoning_parser_kwargs of each request).
    return cls(tokenizer, chat_template_kwargs={"enable_thinking": enable_thinking})


def end_index(parser, prompt_ids: list[int], output_ids: list[int]) -> int | None:
    ids = list(prompt_ids)
    for index, token in enumerate(output_ids):
        ids.append(token)
        if parser.is_reasoning_end_streaming(ids, [token]):
            return index
    return None


def test_gemma4_prefix_and_parser(tmp_path, monkeypatch) -> None:
    tok = real_tokenizer(GEMMA4)
    start, end = tok.convert_tokens_to_ids(["<|channel>", "<channel|>"])
    on = real_prompt(tmp_path, monkeypatch, GEMMA4, "--assistant-prefix", "<|channel>thought\\n")
    # The prefix adds exactly <|channel> thought \n after the template's "<|turn>model\n".
    assert on[-3:] == [start, *tok.encode("thought\n", add_special_tokens=False)] == [100, 45518, 107]
    assert tok.decode(on[:-3]).endswith("<|turn>model\n") and "<|think|>" in tok.decode(on)
    parser = vllm_parser("gemma4", tok, enable_thinking=True)
    assert not parser.is_reasoning_end(on)
    output = tok.encode("Tänker.", add_special_tokens=False) + [end] + tok.encode('{"a": 1}', add_special_tokens=False)
    assert end_index(parser, on, output) == output.index(end)

    # Without the prefix the thought block is not open: an answer written directly never
    # ends the reasoning (the backward scan stops at <|turn>), so the grammar never starts.
    bare = real_prompt(tmp_path, monkeypatch, GEMMA4)
    assert not parser.is_reasoning_end(bare)
    assert end_index(parser, bare, tok.encode('{"a": 1}', add_special_tokens=False)) is None

    # Reasoning off: the template closes an empty thought block, so JSON starts at once.
    off = real_prompt(tmp_path, monkeypatch, GEMMA4, "--no-reasoning")
    assert tok.decode(off).endswith("<|turn>model\n<|channel>thought\n<channel|>")
    assert vllm_parser("gemma4", tok, enable_thinking=False).is_reasoning_end(off)


def test_qwen38_template_and_parser(tmp_path, monkeypatch) -> None:
    tok = real_tokenizer(QWEN38)
    end = tok.convert_tokens_to_ids("</think>")
    on = real_prompt(tmp_path, monkeypatch, QWEN38)
    assert tok.decode(on).endswith("<|im_start|>assistant\n<think>\n")
    parser = vllm_parser("qwen3", tok, enable_thinking=True)
    assert not parser.is_reasoning_end(on)
    output = tok.encode("Tänker.\n", add_special_tokens=False) + [end] + tok.encode('\n\n{"a": 1}', add_special_tokens=False)
    assert end_index(parser, on, output) == output.index(end)
    off = real_prompt(tmp_path, monkeypatch, QWEN38, "--no-reasoning")
    assert tok.decode(off).endswith("<think>\n\n</think>\n\n")
    assert vllm_parser("qwen3", tok, enable_thinking=False).is_reasoning_end(off)
