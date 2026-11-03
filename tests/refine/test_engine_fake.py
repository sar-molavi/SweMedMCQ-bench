"""FakePool: retry policy, fallback, routing, seeds, error handling."""

from __future__ import annotations

import asyncio
from typing import Literal

import pytest
from pydantic import BaseModel

from swemedmcq.refine.engine import FakeOutput, FakePool, LLMRequest, LLMResult, seed_from_key
from swemedmcq.refine.engine.gptoss import patch_chat_template
from swemedmcq.refine.engine.qwen import split_think


class Tiny(BaseModel):
    answer: str
    confidence: Literal["low", "medium", "high"]


SCHEMAS = {"tiny": Tiny}
OK = {"answer": "ja", "confidence": "high"}


def _req(model="gptoss", seed_key=("p1", "r1", "judge"), user="u"):
    return LLMRequest(model=model, system="s", user=user, schema_name="tiny", seed_key=list(seed_key))


def run(coro):
    return asyncio.run(coro)


def test_seed_is_stable_and_32bit():
    s = seed_from_key(["p1", "r1"])
    assert s == seed_from_key(["p1", "r1"]) and 0 <= s < 2**32 and s != seed_from_key(["p1", "r2"])


def test_gptoss_first_try_ok_is_greedy(tmp_path):
    seen = []

    def responder(req, attempt):
        seen.append((req.model, attempt.temperature, attempt.seed))
        return OK

    r = run(FakePool(responder, SCHEMAS, tmp_path).generate(_req()))
    assert r.status == "ok" and r.attempts == 1 and r.model_used == "gptoss"
    assert seen == [("gptoss", 0.0, None)]
    assert not r.reasoning_empty and r.finish_reason == "stop"


@pytest.mark.parametrize(
    "bad",
    [
        "not json",
        '{"answer": "x", "confidence": "maybe"}',  # schema error
        FakeOutput(content='{"answer": "x", "confidence": "low"}', finish_reason="length"),
        "",
        RuntimeError("engine died"),
    ],
)
def test_gptoss_retry_then_qwen_fallback(tmp_path, bad):
    seen = []

    def responder(req, attempt):
        seen.append((req.model, attempt.temperature, attempt.seed))
        return OK if req.model == "qwen" else bad

    r = run(FakePool(responder, SCHEMAS, tmp_path).generate(_req()))
    base = seed_from_key(["p1", "r1", "judge"])
    assert seen == [("gptoss", 0.0, None), ("gptoss", 0.6, base), ("qwen", 1.0, base)]
    assert r.status == "ok" and r.attempts == 3 and r.model_used == "qwen"
    assert len(r.attempt_hashes) == 3 and r.request_hash == r.attempt_hashes[0]


def test_gptoss_retry_succeeds_without_fallback(tmp_path):
    def responder(req, attempt):
        return OK if attempt.temperature > 0 else "nope"

    r = run(FakePool(responder, SCHEMAS, tmp_path).generate(_req()))
    assert r.status == "ok" and r.attempts == 2 and r.model_used == "gptoss"


def test_everything_fails_returns_failed_never_raises(tmp_path):
    def responder(req):
        raise ValueError("boom")

    r = run(FakePool(responder, SCHEMAS, tmp_path).generate(_req()))
    assert isinstance(r, LLMResult) and r.status == "failed" and r.attempts == 3
    assert "boom" in r.error and r.parsed is None


def test_qwen_retries_with_next_seed_then_fails(tmp_path):
    seeds = []

    def responder(req, attempt):
        seeds.append(attempt.seed)
        return "bad"

    r = run(FakePool(responder, SCHEMAS, tmp_path).generate(_req(model="qwen")))
    base = seed_from_key(["p1", "r1", "judge"])
    assert seeds == [base, (base + 1) % 2**32]
    assert r.status == "failed" and r.attempts == 2 and r.model_used == "qwen"


def test_no_fallback_when_qwen_not_loaded(tmp_path):
    r = run(FakePool(lambda req: "bad", SCHEMAS, tmp_path, models=("gptoss",)).generate(_req()))
    assert r.status == "failed" and r.attempts == 2


def test_reasoning_empty_is_recorded(tmp_path):
    r = run(FakePool(lambda req: FakeOutput(content='{"answer":"a","confidence":"low"}', reasoning=""),
                     SCHEMAS, tmp_path).generate(_req()))
    assert r.status == "ok" and r.reasoning_empty


def test_programming_errors_raise(tmp_path):
    pool = FakePool(lambda req: OK, SCHEMAS, tmp_path, models=("gptoss",))
    with pytest.raises(KeyError):
        run(pool.generate(LLMRequest(model="gptoss", system="", user="", schema_name="nope", seed_key=[])))
    with pytest.raises(KeyError):
        run(pool.generate(_req(model="qwen")))


def test_least_in_flight_routing_spreads_load(tmp_path):
    async def responder(req):
        await asyncio.sleep(0.01)
        return OK

    async def go():
        pool = FakePool(responder, SCHEMAS, tmp_path, replicas=3)
        await asyncio.gather(*[pool.generate(_req(user=str(i))) for i in range(30)])
        return pool.stats()

    stats = run(go())
    assigned = [r["assigned"] for r in stats["replicas"].values() if r["model"] == "gptoss"]
    assert assigned == [10, 10, 10]
    assert stats["counters"]["calls"] == 30 and stats["counters"]["ok"] == 30
    assert all(r["in_flight"] == 0 for r in stats["replicas"].values())


def test_patch_chat_template_fixes_date():
    tmpl = 'x {{- "Current date: " + strftime_now("%Y-%m-%d") + "\\n\\n" }} y'
    out = patch_chat_template(tmpl, "2000-01-02")
    assert "strftime_now" not in out and '"2000-01-02"' in out
    assert patch_chat_template(tmpl, None) == tmpl
    with pytest.raises(ValueError):
        patch_chat_template(tmpl, "today")


def test_split_think():
    class Tok:
        def decode(self, ids, skip_special_tokens=True):
            return " ".join(str(i) for i in ids)

    assert split_think([5, 1, 2, 9, 3, 4], Tok(), 5, 9) == ("1 2", "3 4", 2)
    assert split_think([1, 2], Tok(), 5, 9) == ("1 2", "", 2)
