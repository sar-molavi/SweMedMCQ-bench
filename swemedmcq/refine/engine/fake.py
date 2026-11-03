"""GPU-free test double with the same cache, routing and retry semantics.

``FakePool(responder, schemas, cache_dir)`` behaves like ``ModelPool`` but each
attempt calls ``responder`` instead of vLLM.  The responder receives a copy of
the request whose ``model`` is the model of the *attempt* (``"qwen"`` for a
GPT-OSS fallback) and, if it accepts a second positional parameter, the
:class:`~swemedmcq.refine.engine.pool.Attempt` (index, temperature, seed).  It may be a
plain function or a coroutine function and returns:

* ``dict``         -> serialised as the JSON content (validated against the schema),
* ``str``          -> raw content (e.g. invalid JSON to exercise retries),
* ``FakeOutput``   -> full control (reasoning, content, finish_reason),
* ``Exception``    -> (returned or raised) an engine error for this attempt.

Models: ``"gptoss"``, ``"qwen"`` and ``"gemma"`` (``models=`` selects them; by
default the keys of ``cfg["models"]``, else GPT-OSS + Qwen).  Gemma and Qwen
share the seeded-sampling schedule (attempt k uses seed + k - 1).
"""

from __future__ import annotations

import asyncio
import contextvars
import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel

from .pool import MODEL_NAMES, Attempt, Backend, LLMRequest, ModelPool, RawOutput, Replica

_CURRENT_REQUEST: contextvars.ContextVar[LLMRequest] = contextvars.ContextVar("fake_current_request")

FAKE_DEFAULTS: dict[str, dict[str, Any]] = {
    "gptoss": {"path": "fake/gptoss", "temperature": 0.0, "reasoning_effort": "medium"},
    "qwen": {"path": "fake/qwen", "temperature": 1.0, "top_p": 0.95, "top_k": 20, "reasoning_effort": "xhigh"},
    "gemma": {"path": "fake/gemma", "temperature": 1.0, "top_p": 0.95, "top_k": 64, "thinking": "native"},
}
DEFAULT_FAKE_MODELS: tuple[str, ...] = ("gptoss", "qwen")


@dataclass
class FakeOutput:
    content: str
    reasoning: str = "fake reasoning"
    finish_reason: str | None = "stop"


class FakeBackend(Backend):
    def __init__(self, name: str, cfg: dict[str, Any], responder: Callable[..., Any], n_replicas: int = 1) -> None:
        super().__init__(name, {**FAKE_DEFAULTS.get(name, {}), **cfg})
        self.model_id = f"{self.cfg.get('path', name)}@fake"
        self.responder = responder
        self._pass_attempt = _accepts_two_args(responder)
        self.calls: list[tuple[LLMRequest, Attempt]] = []
        self.replicas = [Replica(model=name, index=i, gpus=[i]) for i in range(n_replicas)]

    def render_params(self) -> dict[str, Any]:
        if self.name == "gemma":
            return {"thinking": self.cfg.get("thinking", "native")}
        return {"reasoning_effort": self.cfg.get("reasoning_effort")}

    def _render(self, system: str, user: str) -> list[int]:
        return list(f"<sys>{system}<user>{user}".encode("utf-8"))

    def attempt(self, index: int, base_seed: int, *, retry: dict[str, Any]) -> Attempt:
        # Same seed/temperature schedule as the real backends.
        if self.name == "gptoss":
            temperature = float(self.cfg["temperature"]) if index == 1 else float(retry.get("gptoss_retry_temperature", 0.6))
            seed = None if temperature == 0.0 else (base_seed + max(0, index - 2)) % (2**32)
            return Attempt(model=self.name, index=index, temperature=temperature, seed=seed)
        return Attempt(
            model=self.name,
            index=index,
            temperature=float(self.cfg["temperature"]),
            top_p=float(self.cfg["top_p"]),
            top_k=int(self.cfg["top_k"]),
            seed=(base_seed + index - 1) % (2**32),
        )

    async def run(self, replica, prompt_ids, attempt, schema_json, max_tokens, request_id) -> RawOutput:
        req = _CURRENT_REQUEST.get().model_copy(update={"model": self.name})
        self.calls.append((req, attempt))
        out = self.responder(req, attempt) if self._pass_attempt else self.responder(req)
        if inspect.isawaitable(out):
            out = await out
        await asyncio.sleep(0)
        if isinstance(out, BaseException):
            raise out
        if isinstance(out, dict):
            out = FakeOutput(content=json.dumps(out, ensure_ascii=False))
        elif isinstance(out, str):
            out = FakeOutput(content=out)
        if not isinstance(out, FakeOutput):
            raise TypeError(f"responder returned unsupported {type(out).__name__}")
        return RawOutput(
            reasoning=out.reasoning,
            content=out.content,
            finish_reason=out.finish_reason,
            prompt_tokens=len(prompt_ids),
            completion_tokens=len(out.reasoning.split()) + len(out.content.split()),
            reasoning_tokens=len(out.reasoning.split()),
        )


def _accepts_two_args(fn: Callable[..., Any]) -> bool:
    try:
        params = list(inspect.signature(fn).parameters.values())
    except (TypeError, ValueError):
        return False
    positional = [p for p in params if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
    return len(positional) >= 2 or any(p.kind == p.VAR_POSITIONAL for p in params)


class FakePool(ModelPool):
    """``ModelPool`` with ``FakeBackend``s (no GPU, no vLLM import)."""

    def __init__(
        self,
        responder: Callable[..., Any],
        schemas: dict[str, type[BaseModel]],
        cache_dir: Path | str,
        *,
        cfg: dict[str, Any] | None = None,
        retry_failed: bool = True,
        models: tuple[str, ...] | None = None,
        replicas: int = 1,
    ) -> None:
        cfg = cfg or {}
        model_cfgs = cfg.get("models", {}) or {}
        if models is None:
            # Default: the models configured under ``models:`` (e.g. with_gemma.yaml
            # adds "gemma"), else GPT-OSS + Qwen.
            models = tuple(model_cfgs) or DEFAULT_FAKE_MODELS
        unknown = set(models) - set(MODEL_NAMES)
        if unknown:
            raise ValueError(f"unknown fake model(s) {sorted(unknown)}; expected {MODEL_NAMES}")
        backends: dict[str, Backend] = {
            name: FakeBackend(name, model_cfgs.get(name, {}), responder, replicas) for name in models
        }
        super().__init__(
            backends, schemas, cache_dir, retry=cfg.get("retry"), retry_failed=retry_failed, vllm_version="fake"
        )

    @classmethod
    async def create(  # type: ignore[override]
        cls,
        cfg: dict[str, Any],
        schemas: dict[str, type[BaseModel]],
        cache_dir: Path,
        *,
        responder: Callable[..., Any],
        retry_failed: bool = True,
        **kwargs: Any,
    ) -> "FakePool":
        return cls(responder, schemas, cache_dir, cfg=cfg, retry_failed=retry_failed, **kwargs)

    async def _compute_attempt(self, req, attempt, *args, **kwargs):  # type: ignore[override]
        _CURRENT_REQUEST.set(req)
        return await super()._compute_attempt(req, attempt, *args, **kwargs)

    @property
    def calls(self) -> list[tuple[LLMRequest, Attempt]]:
        """Every responder call (request as seen by the attempt's model, attempt)."""
        out: list[tuple[LLMRequest, Attempt]] = []
        for backend in self.backends.values():
            out.extend(getattr(backend, "calls", []))
        return out
