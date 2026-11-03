"""Engine: Gemma backend (split / prefill / modes), per-model effort, 3-model routing, configs.

Offline: no vLLM import, no GPU; Gemma's tokenizer is replaced by a tiny fake
with the real special ids (<|channel>=100, <channel|>=101, 'thought'=45518,
'\\n'=107, <eos>=1, <turn|>=106, <|tool_response>=50).
"""

from __future__ import annotations

import asyncio
from typing import Literal

import pytest
import yaml
from pydantic import BaseModel

from swemedmcq.paths import CONFIG_DIR
from swemedmcq.refine.engine import MODEL_NAMES, FakeOutput, FakePool, LLMRequest, LLMResult
from swemedmcq.refine.engine.gemma import (
    GemmaBackend,
    analysis_from_content,
    split_gemma,
    with_prefill,
)
from swemedmcq.refine.engine.gptoss import GptOssBackend
from swemedmcq.refine.engine.qwen import QwenBackend

CONFIG = CONFIG_DIR / "refine"

# --------------------------------------------------------------------------- #
# Fake Gemma tokenizer
# --------------------------------------------------------------------------- #

SPECIAL = {"<eos>": 1, "<bos>": 2, "<|tool_response>": 50, "<|channel>": 100, "<channel|>": 101,
           "<|turn>": 105, "<turn|>": 106, "<|think|>": 98}
WORDS = {"thought": 45518, "\n": 107, "Let": 1000, " me": 1001, " think": 1002, ".": 1003,
         '{"answer": "ja", "confidence": "high"}': 2000, "model": 4368, "user": 2364, "system": 9000}
VOCAB = {**SPECIAL, **WORDS}
ID2TOK = {v: k for k, v in VOCAB.items()}


class FakeGemmaTokenizer:
    def get_vocab(self):
        return dict(VOCAB)

    def encode(self, text, add_special_tokens=False):
        ids, rest = [], text
        keys = sorted(VOCAB, key=len, reverse=True)
        while rest:
            for k in keys:
                if rest.startswith(k):
                    ids.append(VOCAB[k])
                    rest = rest[len(k):]
                    break
            else:
                ids.append(3000 + ord(rest[0]))  # any other character
                rest = rest[1:]
        return ids

    def decode(self, ids, skip_special_tokens=False):
        out = []
        for i in ids:
            tok = ID2TOK.get(i, chr(i - 3000) if i >= 3000 and i < 3000 + 0x110000 else f"<{i}>")
            if skip_special_tokens and i in SPECIAL.values():
                continue
            out.append(tok)
        return "".join(out)

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=False, return_dict=False,
                            enable_thinking=False, **kw):
        sys_msg = messages[0]["content"]
        user = messages[1]["content"]
        text = "<bos><|turn>system\n" + ("<|think|>\n" if enable_thinking else "") + sys_msg + "<turn|>\n"
        text += "<|turn>user\n" + user + "<turn|>\n<|turn>model\n"
        if not enable_thinking:
            text += "<|channel>thought\n<channel|>"
        return self.encode(text) if tokenize else text


TOK = FakeGemmaTokenizer()
ANSWER = '{"answer": "ja", "confidence": "high"}'


def _dec(ids, skip):
    return TOK.decode(ids, skip_special_tokens=skip)


def _split(ids, thinking=True):
    return split_gemma(ids, _dec, start_id=100, end_id=101, label_ids=[45518, 107], stop_ids=[1, 106, 50],
                       thinking=thinking)


def _gemma(**cfg) -> GemmaBackend:
    b = GemmaBackend("gemma", {"path": "google/gemma-4-31B-it", **cfg})
    b.tokenizer = TOK
    b.init_special_ids()
    return b


# --------------------------------------------------------------------------- #
# split / prefill
# --------------------------------------------------------------------------- #


def test_fake_tokenizer_matches_real_prefill_ids():
    assert TOK.encode("<|channel>thought\n") == [100, 45518, 107]


def test_split_after_prefill():
    # prompt ended with <|channel>thought\n, so generation starts inside the thought block
    ids = TOK.encode("Let me think.<channel|>" + ANSWER) + [106]
    reasoning, content, n = _split(ids)
    assert reasoning == "Let me think." and content == ANSWER and n == 4


def test_split_strips_repeated_channel_and_label():
    ids = TOK.encode("<|channel>thought\nLet me think.<channel|>" + ANSWER + "<turn|><eos>")
    reasoning, content, n = _split(ids)
    assert reasoning == "Let me think." and content == ANSWER
    assert "thought" not in reasoning.split()[0]


def test_split_without_channel_end_is_reasoning_only():
    reasoning, content, n = _split(TOK.encode("Let me think. Let me think."))
    assert reasoning.startswith("Let me think") and content == "" and n == 9


def test_split_keeps_leaked_markers_visible_in_content():
    ids = TOK.encode("x<channel|><|channel>thought\n" + ANSWER)
    _, content, _ = _split(ids)
    assert content.startswith("<|channel>")  # not silently dropped -> JSON validation fails


def test_split_schema_mode_is_all_content():
    reasoning, content, n = _split(TOK.encode(ANSWER + "<turn|>"), thinking=False)
    assert (reasoning, content, n) == ("", ANSWER, 0)


def test_with_prefill_is_idempotent():
    assert with_prefill([5, 6], [100, 45518, 107]) == [5, 6, 100, 45518, 107]
    assert with_prefill([5, 100, 45518, 107], [100, 45518, 107]) == [5, 100, 45518, 107]


def test_analysis_from_content():
    assert analysis_from_content('{"analysis": " because ", "verdict": "A"}') == "because"
    assert analysis_from_content("not json") == "" and analysis_from_content("[1]") == ""


# --------------------------------------------------------------------------- #
# Backend rendering / parsing / engine args (no vLLM)
# --------------------------------------------------------------------------- #


def test_gemma_native_render_prefills_thought_channel():
    b = _gemma()
    ids = b.render("SYS", "USER")
    assert ids[-3:] == [100, 45518, 107]
    text = b.render_text("SYS", "USER")
    assert "<|think|>" in text and text.endswith("<|turn>model\n<|channel>thought\n")
    assert b.chat_template_kwargs() == {"enable_thinking": True}
    assert b.render_params() == {"enable_thinking": True, "thinking": "native"}


def test_gemma_native_parse_and_failure_without_end():
    b = _gemma()
    assert b.parse(TOK.encode("Let me think.<channel|>" + ANSWER + "<turn|>")) == ("Let me think.", ANSWER, 4)
    assert b.parse(TOK.encode("Let me think."))[1] == ""


def test_gemma_schema_mode_no_prefill_and_analysis_as_reasoning():
    b = _gemma(thinking="schema")
    ids = b.render("SYS", "USER")
    assert ids[-4:] == [100, 45518, 107, 101]  # template's closed empty thought block
    assert "<|think|>" not in b.render_text("SYS", "USER")
    assert b.chat_template_kwargs() == {"enable_thinking": False}
    content = '{"analysis": "weighing both", "verdict": "A"}'
    reasoning, out, n = b.parse(TOK.encode(content + "<turn|>"))
    assert out == content and reasoning == "weighing both" and n == 0


def test_gemma_bad_thinking_mode_raises():
    with pytest.raises(ValueError):
        GemmaBackend("gemma", {"path": "google/gemma-4-31B-it", "thinking": "maybe"})


def test_gemma_sampling_defaults_and_seed_schedule():
    b = _gemma()
    a1, a2 = b.attempt(1, 41, retry={}), b.attempt(2, 41, retry={})
    assert (a1.temperature, a1.top_p, a1.top_k, a1.seed) == (1.0, 0.95, 64, 41)
    assert a2.seed == 42


def test_engine_kwargs_builtin_parser_has_no_plugin():
    g = _gemma().engine_kwargs()
    assert g["structured_outputs_config"] == {"reasoning_parser": "gemma4", "enable_in_reasoning": False}
    assert g["language_model_only"] is True
    q = QwenBackend("qwen", {"path": "Qwen/Qwen3.8-27B"}).engine_kwargs()["structured_outputs_config"]
    assert q["reasoning_parser"] == "qwen3_8_thinking" and q["reasoning_parser_plugin"].endswith(".py")


def test_reasoning_effort_is_per_model():
    assert QwenBackend("qwen", {"path": "Qwen/Qwen3.8-27B"}).reasoning_effort == "xhigh"
    assert QwenBackend("qwen", {"path": "Qwen/Qwen3.8-27B", "reasoning_effort": "low"}).reasoning_effort == "low"
    assert GptOssBackend("gptoss", {"path": "openai/gpt-oss-120b"}).reasoning_effort == "medium"
    q = QwenBackend("qwen", {"path": "Qwen/Qwen3.8-27B"})
    assert q.chat_template_kwargs()["reasoning_effort"] == "xhigh"


# --------------------------------------------------------------------------- #
# Records and 3-model routing with the fake pool
# --------------------------------------------------------------------------- #


class Tiny(BaseModel):
    answer: str
    confidence: Literal["low", "medium", "high"]


SCHEMAS = {"tiny": Tiny}
OK = {"answer": "ja", "confidence": "high"}


def _req(model, key=("p1", "r1", "judge", "k0"), user="u"):
    return LLMRequest(model=model, system="s", user=user, schema_name="tiny", seed_key=list(key))


def test_records_accept_gemma():
    assert MODEL_NAMES == ("gptoss", "qwen", "gemma")
    assert _req("gemma").model == "gemma"
    with pytest.raises(Exception):
        _req("llama")
    r = LLMResult(status="ok", parsed=OK, reasoning="r", content="{}", finish_reason="stop", prompt_tokens=1,
                  completion_tokens=1, reasoning_empty=False, error=None, request_hash="h", from_cache=False,
                  attempts=1, model_used="gemma")
    assert r.model_used == "gemma"


def test_fake_pool_routes_three_models(tmp_path):
    seen = []

    def responder(req, attempt):
        seen.append((req.model, attempt.top_k))
        return OK

    async def go():
        pool = FakePool(responder, SCHEMAS, tmp_path, models=("gptoss", "qwen", "gemma"), replicas=2)
        res = await asyncio.gather(*[pool.generate(_req(m, ("p", str(i)), user=f"u{i}")) for i in range(4)
                                     for m in ("gptoss", "qwen", "gemma")])
        return pool, res

    pool, res = asyncio.run(go())
    assert all(r.status == "ok" for r in res)
    assert [r.model_used for r in res] == ["gptoss", "qwen", "gemma"] * 4
    stats = pool.stats()["models"]
    assert {m: stats[m]["done"] for m in stats} == {"gptoss": 4, "qwen": 4, "gemma": 4}
    assert ("gemma", 64) in seen and ("qwen", 20) in seen
    for m in ("gptoss", "qwen", "gemma"):
        assert sorted(r["assigned"] for r in pool.stats()["replicas"].values() if r["model"] == m) == [2, 2]


def test_gemma_reasoning_only_output_retries_next_seed_without_fallback(tmp_path):
    seeds = []

    def responder(req, attempt):
        seeds.append((req.model, attempt.seed))
        if attempt.index == 1:
            return FakeOutput(content="", reasoning="thinking forever", finish_reason="length")
        return OK

    r = asyncio.run(FakePool(responder, SCHEMAS, tmp_path, models=("gptoss", "qwen", "gemma")).generate(_req("gemma")))
    assert r.status == "ok" and r.attempts == 2 and r.model_used == "gemma"
    assert [m for m, _ in seeds] == ["gemma", "gemma"] and seeds[1][1] == seeds[0][1] + 1


def test_k_samples_differ_by_seed_key(tmp_path):
    seeds = []

    def responder(req, attempt):
        seeds.append(attempt.seed)
        return OK

    async def go():
        pool = FakePool(responder, SCHEMAS, tmp_path)
        for k in range(3):
            await pool.generate(_req("qwen", ("p1", "r1", "judge", "c1", "candidate_first", f"k{k}")))

    asyncio.run(go())
    assert len(set(seeds)) == 3


# --------------------------------------------------------------------------- #
# Config profiles
# --------------------------------------------------------------------------- #

LOOP_SETTINGS = {
    "qwen_judge_samples": 3,
    "accept_high": 0.9,
    "accept_medium": 0.75,
    "max_veto_medium": 1,
    "max_rounds": 3,
    "passes": ["meaning", "terminology", "language"],
    "dual_meaning_detect": True,
}


def _load(name):
    return yaml.safe_load((CONFIG / name).read_text(encoding="utf-8"))


def _gpus(cfg):
    return [g for m in cfg["models"].values() for rep in m["replicas"] for g in rep]


@pytest.mark.parametrize(
    "name, replicas, judges",
    [
        ("default.yaml", {"gptoss": [[0]], "qwen": [[1], [2], [3]]}, ["qwen", "gptoss"]),
        ("with_gemma.yaml", {"gptoss": [[0]], "qwen": [[1], [2]], "gemma": [[3]]}, ["qwen", "gptoss", "gemma"]),
    ],
)
def test_config_profiles(name, replicas, judges, tmp_path):
    cfg = _load(name)
    assert {m: c["replicas"] for m, c in cfg["models"].items()} == replicas
    gpus = _gpus(cfg)
    assert len(gpus) == len(set(gpus)), "a GPU is shared by two engines"
    assert sorted(gpus) == [0, 1, 2, 3]
    for key, value in LOOP_SETTINGS.items():
        assert cfg["loop"][key] == value, (name, key)
    assert cfg["loop"]["judge_models"] == judges
    assert set(cfg["loop"]["judge_models"]) <= set(cfg["models"])
    if "qwen" in cfg["models"]:
        assert cfg["models"]["qwen"]["reasoning_effort"] == "xhigh"
    if "gptoss" in cfg["models"]:
        assert cfg["models"]["gptoss"]["reasoning_effort"] == "medium"
    if "gemma" in cfg["models"]:
        g = cfg["models"]["gemma"]
        assert g["path"] == "google/gemma-4-31B-it" and g["thinking"] == "native"
        assert (g["temperature"], g["top_p"], g["top_k"]) == (1.0, 0.95, 64)
    pool = FakePool(lambda req: OK, SCHEMAS, tmp_path, cfg=cfg)
    assert set(pool.backends) == set(cfg["models"])


# --------------------------------------------------------------------------- #
# engine_smoke helpers
# --------------------------------------------------------------------------- #


def test_engine_smoke_helpers():
    from swemedmcq.refine.cli.engine_smoke import gemma_leaks, parse_models

    assert parse_models(["gptoss,qwen", "gemma"], ["gptoss", "qwen", "gemma"]) == ["gptoss", "qwen", "gemma"]
    assert parse_models(None, ["gptoss", "qwen"]) == ["gptoss", "qwen"]
    with pytest.raises(SystemExit):
        parse_models(["gemma"], ["gptoss", "qwen"])
    assert gemma_leaks(ANSWER, "Let me think.") == []
    assert gemma_leaks("<|channel>" + ANSWER)
    assert gemma_leaks("thought\n" + ANSWER)
    assert gemma_leaks(ANSWER, "thought\nLet me")
