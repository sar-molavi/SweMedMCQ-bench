"""Cache: content addressing, atomic writes, replay without the GPU, failed retries."""

from __future__ import annotations

import asyncio
import json
from typing import Literal

from pydantic import BaseModel

from swemedmcq.refine.engine import FakePool, LLMRequest
from swemedmcq.refine.engine.cache import ResultCache, _stable_hash, request_key


class Tiny(BaseModel):
    answer: str
    confidence: Literal["low", "medium", "high"]


SCHEMAS = {"tiny": Tiny}


def _req(model="gptoss", user="hej", seed_key=("p1", "r1")):
    return LLMRequest(model=model, system="sys", user=user, schema_name="tiny", seed_key=list(seed_key))


def run(coro):
    return asyncio.run(coro)


def test_stable_hash_is_canonical():
    assert _stable_hash({"b": 1, "a": 2}, "x") == _stable_hash({"a": 2, "b": 1}, "x")
    assert _stable_hash("a", "b") != _stable_hash("b", "a")
    assert len(_stable_hash("x")) == 64


def test_request_key_depends_on_every_part():
    base = dict(model_id="m@1", vllm_version="0.28.0", prompt_token_ids=[1, 2], sampling={"t": 0},
                schema_json={"type": "object"}, max_tokens=10)
    k0 = request_key(**base)
    for field, value in [("model_id", "m@2"), ("vllm_version", "0.29"), ("prompt_token_ids", [1, 3]),
                         ("sampling", {"t": 1}), ("schema_json", {}), ("max_tokens", 11)]:
        assert request_key(**{**base, field: value}) != k0, field


def test_cache_put_get_layout(tmp_path):
    cache = ResultCache(tmp_path)
    key = "ab" + "0" * 62
    path = cache.put("gptoss", key, {"status": "ok"})
    assert path == tmp_path / "gptoss" / "ab" / f"{key}.json"
    assert cache.get("gptoss", key) == {"status": "ok"}
    assert cache.get("qwen", key) is None
    assert not list(path.parent.glob("*.tmp"))  # no temp leftovers
    path.write_text("{torn", encoding="utf-8")
    assert cache.get("gptoss", key) is None  # unreadable entry = miss, never a crash


def test_hit_replays_without_calling_the_model(tmp_path):
    calls = []

    def responder(req):
        calls.append(req.user)
        return {"answer": "ja", "confidence": "high"}

    async def go():
        pool = FakePool(responder, SCHEMAS, tmp_path)
        r1 = await pool.generate(_req())
        pool2 = FakePool(lambda req: (_ for _ in ()).throw(AssertionError("GPU touched")), SCHEMAS, tmp_path)
        r2 = await pool2.generate(_req())
        return r1, r2

    r1, r2 = run(go())
    assert calls == ["hej"]
    assert r1.status == r2.status == "ok" and not r1.from_cache and r2.from_cache
    assert r1.parsed == r2.parsed == {"answer": "ja", "confidence": "high"}
    assert r1.request_hash == r2.request_hash
    f = tmp_path / "gptoss" / r1.request_hash[:2] / f"{r1.request_hash}.json"
    stored = json.loads(f.read_text())
    assert stored["status"] == "ok" and "from_cache" not in stored and stored["_meta"]["seed_key"] == ["p1", "r1"]


def test_different_prompt_or_schema_is_a_miss(tmp_path):
    n = []

    def responder(req):
        n.append(1)
        return {"answer": req.user, "confidence": "low"}

    async def go():
        pool = FakePool(responder, SCHEMAS, tmp_path)
        a = await pool.generate(_req(user="a"))
        b = await pool.generate(_req(user="b"))
        return a, b

    a, b = run(go())
    assert len(n) == 2 and a.request_hash != b.request_hash


def test_failed_results_are_retried_unless_frozen(tmp_path):
    state = {"ok": False, "n": 0}

    def responder(req):
        state["n"] += 1
        return {"answer": "x", "confidence": "low"} if state["ok"] else "not json"

    async def go(retry_failed):
        pool = FakePool(responder, SCHEMAS, tmp_path, models=("qwen",), retry_failed=retry_failed)
        return await pool.generate(_req(model="qwen"))

    r = run(go(True))
    assert r.status == "failed" and state["n"] == 2  # qwen: 2 attempts
    n_before = state["n"]
    r = run(go(False))  # frozen: replay the cached failures
    assert r.status == "failed" and r.from_cache and state["n"] == n_before
    state["ok"] = True
    r = run(go(True))  # retried on the next call
    assert r.status == "ok" and not r.from_cache and state["n"] == n_before + 1


def test_concurrent_identical_requests_share_one_call(tmp_path):
    n = []

    async def responder(req):
        n.append(1)
        await asyncio.sleep(0.01)
        return {"answer": "x", "confidence": "low"}

    async def go():
        pool = FakePool(responder, SCHEMAS, tmp_path)
        return await asyncio.gather(*[pool.generate(_req()) for _ in range(20)])

    results = run(go())
    assert len(n) == 1
    assert len({r.request_hash for r in results}) == 1
    assert all(r.status == "ok" for r in results)
