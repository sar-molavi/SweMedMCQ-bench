"""Model pool: cached, routed, retried structured-output LLM calls.

``ModelPool`` owns one or more *backends* (``"gptoss"``, ``"qwen"``, ``"gemma"``), each with
one or more *replicas* (one vLLM ``AsyncLLMEngine`` per replica, TP = number of
GPUs in the replica).  ``generate()`` is the only entry point for the loop:

    cache lookup -> least-in-flight replica -> parse + validate -> atomic cache
    write -> retry policy (GPT-OSS: sampled retry, then Qwen fallback; Qwen:
    one retry with the next seed) -> ``LLMResult``.

Model failures (engine errors, invalid JSON, schema errors, truncation, empty
content, prompt too long) never raise; they come back as ``status="failed"``.
Programming errors (unknown schema or model, bad config) raise.

Multi-engine strategy (verified by ``cli/engine_smoke.py``): all engines live
in this one process.  Right before each ``AsyncLLMEngine.from_engine_args`` we
set ``CUDA_VISIBLE_DEVICES`` to the replica's GPUs; vLLM V1 runs the engine
core and its TP workers in subprocesses that inherit that environment, so each
engine only sees its own GPUs.  The parent never allocates GPU memory or
creates a CUDA context.  Caveat (vLLM 0.28): resolving the Qwen3.5 config
imports vLLM's flash-linear-attention ops, whose import calls
``torch.cuda.current_device()`` and so initialises the CUDA *driver* in the
parent.  vLLM then switches to the ``spawn`` start method; spawned children
read ``CUDA_VISIBLE_DEVICES`` at start-up, so placement stays correct.  We set
``VLLM_WORKER_MULTIPROC_METHOD=spawn`` up front so every engine is started the
same way (entry points therefore need an ``if __name__ == "__main__"`` guard).

vLLM is imported lazily, so ``FakePool`` (``engine/fake.py``) and the tests run
without GPUs.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .cache import ResultCache, _stable_hash, request_key

log = logging.getLogger(__name__)

DEFAULT_MAX_TOKENS = 16384

ModelName = Literal["gptoss", "qwen", "gemma"]
MODEL_NAMES: tuple[str, ...] = ("gptoss", "qwen", "gemma")


# --------------------------------------------------------------------------- #
# Public request / result records
# --------------------------------------------------------------------------- #


class LLMRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: ModelName
    system: str
    user: str
    schema_name: str  # key into the ``schemas`` registry given to the pool
    seed_key: list[str]  # e.g. [prompt_id, "r2", "judge", cand_id, "candidate_first"]
    max_tokens: int | None = None


class LLMResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["ok", "failed"]
    parsed: dict | None  # validated JSON (model_dump of the schema)
    reasoning: str
    content: str
    finish_reason: str | None
    prompt_tokens: int
    completion_tokens: int
    reasoning_empty: bool
    error: str | None
    request_hash: str  # key of the FIRST attempt = stable id of the logical request
    from_cache: bool  # True when no attempt of this call touched the GPU
    attempts: int  # number of attempts made (1 = first try succeeded / failed alone)
    # --- additions to the contract (all defaulted) ---
    model_used: ModelName = "gptoss"  # model that produced this result ("qwen" after a fallback)
    attempt_hashes: list[str] = Field(default_factory=list)  # cache key of every attempt made
    reasoning_tokens: int = 0
    latency_s: float = 0.0  # wall time of the producing attempt when it ran on the GPU


# --------------------------------------------------------------------------- #
# Backend plumbing
# --------------------------------------------------------------------------- #


@dataclass
class Attempt:
    """One concrete request in the retry chain."""

    model: str
    index: int  # 1-based within the chain
    temperature: float
    top_p: float | None = None
    top_k: int | None = None
    seed: int | None = None

    def sampling_dict(self) -> dict[str, Any]:
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "seed": self.seed,
        }


@dataclass
class RawOutput:
    """What a backend returns for one attempt (before schema validation)."""

    reasoning: str
    content: str
    finish_reason: str | None
    prompt_tokens: int
    completion_tokens: int
    reasoning_tokens: int = 0


@dataclass
class Replica:
    """One engine (or fake) plus routing statistics."""

    model: str
    index: int
    gpus: list[int]  # logical ids from the config
    physical_gpus: list[str] = field(default_factory=list)  # ids as given to CUDA_VISIBLE_DEVICES
    engine: Any = None
    pids: list[int] = field(default_factory=list)  # engine subprocesses (for GPU checks)
    in_flight: int = 0
    assigned: int = 0
    done: int = 0
    failed: int = 0
    busy_seconds: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    max_in_flight: int = 0
    _busy_since: float | None = None

    @property
    def name(self) -> str:
        return f"{self.model}#{self.index}[{','.join(map(str, self.gpus))}]"

    def begin(self) -> None:
        if self.in_flight == 0:
            self._busy_since = time.monotonic()
        self.in_flight += 1
        self.assigned += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)

    def end(self, ok: bool, raw: RawOutput | None) -> None:
        self.in_flight -= 1
        if ok:
            self.done += 1
        else:
            self.failed += 1
        if raw is not None:
            self.prompt_tokens += raw.prompt_tokens
            self.completion_tokens += raw.completion_tokens
        if self.in_flight == 0 and self._busy_since is not None:
            self.busy_seconds += time.monotonic() - self._busy_since
            self._busy_since = None

    def stats(self) -> dict[str, Any]:
        busy = self.busy_seconds
        if self._busy_since is not None:
            busy += time.monotonic() - self._busy_since
        return {
            "model": self.model,
            "gpus": self.gpus,
            "physical_gpus": self.physical_gpus,
            "in_flight": self.in_flight,
            "max_in_flight": self.max_in_flight,
            "assigned": self.assigned,
            "done": self.done,
            "failed": self.failed,
            "busy_seconds": round(busy, 3),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
        }


class Backend:
    """Model-specific rendering, sampling and generation.  Subclassed per model."""

    name: str = ""

    def __init__(self, name: str, cfg: dict[str, Any]) -> None:
        self.name = name
        self.cfg = dict(cfg)
        self.replicas: list[Replica] = []
        self.max_model_len: int = int(self.cfg.get("max_model_len", 32768))
        self.default_max_tokens: int = int(self.cfg.get("max_tokens", DEFAULT_MAX_TOKENS))
        self.model_id: str = str(self.cfg.get("path", name))  # path + revision once resolved
        self._render_cache: dict[str, list[int]] = {}

    # -- prompt -------------------------------------------------------------
    def render_params(self) -> dict[str, Any]:
        """Template parameters that change the prompt (logged in the cache key)."""
        return {}

    def _render(self, system: str, user: str) -> list[int]:
        raise NotImplementedError

    def render(self, system: str, user: str) -> list[int]:
        key = _stable_hash(system, user)
        ids = self._render_cache.get(key)
        if ids is None:
            ids = self._render(system, user)
            if len(self._render_cache) > 4096:
                self._render_cache.clear()
            self._render_cache[key] = ids
        return ids

    # -- sampling -----------------------------------------------------------
    def attempt(self, index: int, base_seed: int, *, retry: dict[str, Any]) -> Attempt:
        """Sampling for the ``index``-th (1-based) attempt on this model."""
        raise NotImplementedError

    # -- generation ---------------------------------------------------------
    async def run(
        self,
        replica: Replica,
        prompt_ids: list[int],
        attempt: Attempt,
        schema_json: dict[str, Any],
        max_tokens: int,
        request_id: str,
    ) -> RawOutput:
        raise NotImplementedError

    async def close(self) -> None:
        return None


def seed_from_key(seed_key: list[str]) -> int:
    """Deterministic 32-bit seed from the request's ``seed_key``."""
    return int(_stable_hash(*seed_key)[:8], 16)


# --------------------------------------------------------------------------- #
# The pool
# --------------------------------------------------------------------------- #


class ModelPool:
    """Cache + routing + retry policy over per-model backends."""

    def __init__(
        self,
        backends: dict[str, Backend],
        schemas: dict[str, type[BaseModel]],
        cache_dir: Path | str,
        *,
        retry: dict[str, Any] | None = None,
        retry_failed: bool = True,
        vllm_version: str = "unknown",
    ) -> None:
        self.backends = backends
        self.schemas = dict(schemas)
        self.cache = ResultCache(cache_dir)
        self.retry = {
            "gptoss_retry_temperature": 0.6,
            "max_attempts": 2,
            "fallback_model": "qwen",
            "fallback_attempts": 1,
            **(retry or {}),
        }
        self.retry_failed = retry_failed
        self.vllm_version = vllm_version
        self._ids = itertools.count()
        self._schema_json: dict[str, dict[str, Any]] = {}
        self.counters: dict[str, int] = {
            "calls": 0,
            "ok": 0,
            "failed": 0,
            "cache_hits": 0,
            "gpu_attempts": 0,
            "retries": 0,
            "fallbacks": 0,
        }
        self._closed = False
        # Identical concurrent attempts share one GPU call.
        self._inflight: dict[str, asyncio.Future] = {}

    # -- construction -------------------------------------------------------
    @classmethod
    async def create(
        cls,
        cfg: dict[str, Any],
        schemas: dict[str, type[BaseModel]],
        cache_dir: Path,
        *,
        retry_failed: bool = True,
        models: list[str] | None = None,
    ) -> "ModelPool":
        """Load one AsyncLLMEngine per replica of every configured model.

        ``models`` restricts loading to a subset (e.g. ``["gptoss"]``); the
        fallback is then skipped when its model is not loaded.
        """
        from .gemma import GemmaBackend
        from .gptoss import GptOssBackend
        from .qwen import QwenBackend

        backend_classes: dict[str, type[VLLMBackend]] = {
            "gptoss": GptOssBackend,
            "qwen": QwenBackend,
            "gemma": GemmaBackend,
        }
        model_cfgs: dict[str, Any] = cfg["models"]
        wanted = list(model_cfgs) if models is None else list(models)
        engine_cfg = cfg.get("engine", {}) or {}
        backends: dict[str, Backend] = {}
        try:
            for name in wanted:
                if name not in backend_classes:
                    raise ValueError(f"unknown model {name!r}; expected one of {sorted(backend_classes)}")
                if name not in model_cfgs:
                    raise ValueError(f"model {name!r} requested but not configured under models: in the config")
                backend = backend_classes[name](name, model_cfgs[name], engine_cfg)
                backends[name] = backend
                await backend.start()
        except BaseException:
            for backend in backends.values():
                await backend.close()
            raise
        import vllm

        return cls(
            backends,
            schemas,
            cache_dir,
            retry=cfg.get("retry"),
            retry_failed=retry_failed,
            vllm_version=vllm.__version__,
        )

    # -- helpers --------------------------------------------------------------
    def _schema(self, name: str) -> tuple[type[BaseModel], dict[str, Any]]:
        if name not in self.schemas:
            raise KeyError(f"unknown schema_name {name!r}; registry has {sorted(self.schemas)}")
        if name not in self._schema_json:
            self._schema_json[name] = self.schemas[name].model_json_schema()
        return self.schemas[name], self._schema_json[name]

    def _plan(self, req: LLMRequest) -> list[Attempt]:
        """The retry chain for a request (attempts are made until one is ok)."""
        if req.model not in self.backends:
            raise KeyError(f"model {req.model!r} is not loaded in this pool ({sorted(self.backends)})")
        base_seed = seed_from_key(req.seed_key)
        n = max(1, int(self.retry["max_attempts"]))
        primary = self.backends[req.model]
        plan = [primary.attempt(i, base_seed, retry=self.retry) for i in range(1, n + 1)]
        fallback = self.retry.get("fallback_model")
        if req.model == "gptoss" and fallback and fallback != req.model and fallback in self.backends:
            fb = self.backends[fallback]
            for i in range(1, int(self.retry.get("fallback_attempts", 1)) + 1):
                plan.append(fb.attempt(i, base_seed, retry=self.retry))
        for i, attempt in enumerate(plan, start=1):
            attempt.index = i
        return plan

    def _pick_replica(self, backend: Backend) -> Replica:
        if not backend.replicas:
            raise RuntimeError(f"backend {backend.name!r} has no replicas")
        return min(backend.replicas, key=lambda r: (r.in_flight, r.assigned, r.index))

    @staticmethod
    def _validate(
        raw: RawOutput, schema_cls: type[BaseModel]
    ) -> tuple[dict | None, str | None]:
        if raw.finish_reason == "length":
            return None, "finish_reason=length (output truncated)"
        content = raw.content.strip()
        if not content:
            return None, f"empty content (finish_reason={raw.finish_reason})"
        try:
            obj = json.loads(content)
        except json.JSONDecodeError as exc:
            return None, f"invalid JSON: {exc}"
        try:
            parsed = schema_cls.model_validate(obj)
        except ValidationError as exc:
            return None, f"schema error: {exc.errors(include_url=False)}"
        return parsed.model_dump(mode="json"), None

    # -- one attempt ------------------------------------------------------------
    async def _run_attempt(
        self,
        req: LLMRequest,
        attempt: Attempt,
        schema_cls: type[BaseModel],
        schema_json: dict[str, Any],
    ) -> tuple[LLMResult, str, bool]:
        """Return (result, cache key, from_cache) for one attempt."""
        backend = self.backends[attempt.model]
        prompt_ids = backend.render(req.system, req.user)
        requested = int(req.max_tokens or backend.default_max_tokens)
        max_tokens = min(requested, backend.max_model_len - len(prompt_ids))
        sampling = {**attempt.sampling_dict(), **backend.render_params()}
        key = request_key(
            model_id=backend.model_id,
            vllm_version=self.vllm_version,
            prompt_token_ids=prompt_ids,
            sampling=sampling,
            schema_json=schema_json,
            max_tokens=max_tokens,
        )
        cached = self.cache.get(attempt.model, key)
        if cached is not None and (cached.get("status") == "ok" or not self.retry_failed):
            try:
                fields = {k: v for k, v in cached.items() if not k.startswith("_")}
                result = LLMResult.model_validate({**fields, "from_cache": True})
                self.counters["cache_hits"] += 1
                return result, key, True
            except ValidationError:
                log.warning("ignoring malformed cache entry %s", self.cache.path(attempt.model, key))

        shared = self._inflight.get(key)
        if shared is not None:
            await asyncio.wait([shared])  # our own cancellation still propagates
            if not shared.cancelled():
                return shared.result().model_copy(update={"from_cache": True}), key, True
            # the owner was cancelled: compute it ourselves
        future: asyncio.Future = asyncio.get_running_loop().create_future()
        self._inflight[key] = future
        try:
            result = await self._compute_attempt(
                req, attempt, schema_cls, schema_json, backend, prompt_ids, max_tokens, sampling, key
            )
        except BaseException:
            future.cancel()
            raise
        else:
            future.set_result(result)
        finally:
            if self._inflight.get(key) is future:
                del self._inflight[key]
        return result, key, False

    async def _compute_attempt(
        self,
        req: LLMRequest,
        attempt: Attempt,
        schema_cls: type[BaseModel],
        schema_json: dict[str, Any],
        backend: Backend,
        prompt_ids: list[int],
        max_tokens: int,
        sampling: dict[str, Any],
        key: str,
    ) -> LLMResult:
        raw: RawOutput | None = None
        parsed: dict | None = None
        error: str | None = None
        latency = 0.0
        if max_tokens <= 0:
            error = f"over_context: prompt has {len(prompt_ids)} tokens, max_model_len={backend.max_model_len}"
        else:
            replica = self._pick_replica(backend)
            request_id = f"{key[:16]}-{next(self._ids)}"
            self.counters["gpu_attempts"] += 1
            replica.begin()
            t0 = time.monotonic()
            try:
                raw = await backend.run(replica, prompt_ids, attempt, schema_json, max_tokens, request_id)
            except asyncio.CancelledError:
                replica.end(False, None)
                raise
            except Exception as exc:  # engine / model failure: recorded, never raised
                error = f"engine_error: {type(exc).__name__}: {exc}"
                log.warning("%s attempt %d failed on %s: %s", attempt.model, attempt.index, replica.name, error)
            latency = time.monotonic() - t0
            if raw is not None:
                parsed, error = self._validate(raw, schema_cls)
            replica.end(error is None, raw)

        result = LLMResult(
            status="ok" if error is None else "failed",
            parsed=parsed,
            reasoning=raw.reasoning if raw else "",
            content=raw.content if raw else "",
            finish_reason=raw.finish_reason if raw else None,
            prompt_tokens=raw.prompt_tokens if raw else len(prompt_ids),
            completion_tokens=raw.completion_tokens if raw else 0,
            reasoning_empty=not (raw.reasoning.strip() if raw else ""),
            error=error,
            request_hash=key,
            from_cache=False,
            attempts=1,
            model_used=attempt.model,
            attempt_hashes=[key],
            reasoning_tokens=raw.reasoning_tokens if raw else 0,
            latency_s=round(latency, 4),
        )
        record = result.model_dump(exclude={"from_cache"})
        record["_meta"] = {
            "model_id": backend.model_id,
            "vllm_version": self.vllm_version,
            "sampling": sampling,
            "max_tokens": max_tokens,
            "schema_name": req.schema_name,
            "seed_key": req.seed_key,
            "attempt_index": attempt.index,
            "written_at": time.time(),
        }
        self.cache.put(attempt.model, key, record)
        return result

    # -- public API -------------------------------------------------------------
    async def generate(self, req: LLMRequest) -> LLMResult:
        """Cached, routed, retried structured generation (never raises for model errors)."""
        if self._closed:
            raise RuntimeError("ModelPool is closed")
        schema_cls, schema_json = self._schema(req.schema_name)
        plan = self._plan(req)
        self.counters["calls"] += 1
        hashes: list[str] = []
        all_cached = True
        result: LLMResult | None = None
        for attempt in plan:
            if attempt.index > 1:
                if attempt.model != req.model:
                    self.counters["fallbacks"] += 1
                else:
                    self.counters["retries"] += 1
            result, key, cached = await self._run_attempt(req, attempt, schema_cls, schema_json)
            hashes.append(key)
            all_cached = all_cached and cached
            if result.status == "ok":
                break
        assert result is not None
        final = result.model_copy(
            update={
                "request_hash": hashes[0],
                "from_cache": all_cached,
                "attempts": len(hashes),
                "attempt_hashes": hashes,
            }
        )
        self.counters["ok" if final.status == "ok" else "failed"] += 1
        return final

    def stats(self) -> dict[str, Any]:
        replicas = {r.name: r.stats() for b in self.backends.values() for r in b.replicas}
        per_model: dict[str, dict[str, Any]] = {}
        for name, backend in self.backends.items():
            rs = [r.stats() for r in backend.replicas]
            per_model[name] = {
                "replicas": len(rs),
                "in_flight": sum(r["in_flight"] for r in rs),
                "done": sum(r["done"] for r in rs),
                "failed": sum(r["failed"] for r in rs),
                "busy_seconds": round(sum(r["busy_seconds"] for r in rs), 3),
                "prompt_tokens": sum(r["prompt_tokens"] for r in rs),
                "completion_tokens": sum(r["completion_tokens"] for r in rs),
            }
        return {"replicas": replicas, "models": per_model, "counters": dict(self.counters)}

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for backend in self.backends.values():
            try:
                await backend.close()
            except Exception as exc:  # shutting down: log, keep closing the others
                log.warning("error while closing %s: %s", backend.name, exc)

    async def __aenter__(self) -> "ModelPool":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()


# --------------------------------------------------------------------------- #
# vLLM backends (shared part; model specifics in gptoss.py / qwen.py / gemma.py)
# --------------------------------------------------------------------------- #


def resolve_revision(path: str, revision: str | None = None) -> str:
    """Snapshot revision of ``path`` (HF repo id or local dir) for the cache key."""
    if revision:
        return revision
    local = Path(path)
    if local.is_dir():
        cfg = local / "config.json"
        return "local-" + (_stable_hash(cfg.read_text(encoding="utf-8"))[:16] if cfg.exists() else "unknown")
    hub = os.environ.get("HF_HUB_CACHE") or os.path.join(
        os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub"
    )
    ref = Path(hub) / ("models--" + path.replace("/", "--")) / "refs" / "main"
    try:
        return ref.read_text(encoding="utf-8").strip()
    except OSError:
        return "unknown"


def _descendant_pids(pid: int | None = None) -> set[int]:
    try:
        import psutil
    except ImportError:  # pragma: no cover - psutil ships with vLLM
        return set()
    try:
        return {p.pid for p in psutil.Process(pid).children(recursive=True)}
    except psutil.Error:
        return set()


def physical_gpu_ids(gpus: list[int]) -> list[str]:
    """Map config GPU indices to ids for CUDA_VISIBLE_DEVICES.

    Config indices are relative to the devices visible to this job (e.g. under
    slurm ``CUDA_VISIBLE_DEVICES=4,5,6,7`` makes replica ``[0]`` physical GPU 4).
    """
    base = os.environ.get("CUDA_VISIBLE_DEVICES")
    if base is None or base.strip() == "":
        return [str(g) for g in gpus]
    visible = [d.strip() for d in base.split(",") if d.strip()]
    try:
        return [visible[g] for g in gpus]
    except IndexError as exc:
        raise ValueError(f"replica GPUs {gpus} exceed CUDA_VISIBLE_DEVICES={base!r}") from exc


class VLLMBackend(Backend):
    """Shared vLLM engine management; subclasses add rendering and parsing."""

    reasoning_parser: str = ""

    def __init__(self, name: str, cfg: dict[str, Any], engine_cfg: dict[str, Any] | None = None) -> None:
        super().__init__(name, cfg)
        self.engine_cfg = dict(engine_cfg or {})
        self.path = str(self.cfg["path"])
        self.revision = resolve_revision(self.path, self.cfg.get("revision"))
        self.model_id = f"{self.path}@{self.revision}"
        self.tokenizer: Any = None

    # subclasses -------------------------------------------------------------
    def extra_engine_kwargs(self) -> dict[str, Any]:
        return {}

    def chat_template_kwargs(self) -> dict[str, Any]:
        return {}

    def parse(self, token_ids: list[int]) -> tuple[str, str, int]:
        """(reasoning, content, reasoning_tokens) from generated token ids."""
        raise NotImplementedError

    # engine -----------------------------------------------------------------
    def load_tokenizer(self) -> None:
        from transformers import AutoTokenizer

        rev = None if self.revision in ("unknown",) or self.revision.startswith("local-") else self.revision
        self.tokenizer = AutoTokenizer.from_pretrained(self.path, revision=rev)

    def structured_outputs_config(self) -> dict[str, Any]:
        """Reasoning parser for the JSON grammar gate.

        Our own parsers are loaded from a plugin file (``parsers.PLUGIN_PATHS``);
        vLLM built-in parsers (e.g. ``gemma4``) have no entry and need none.
        """
        from .parsers import PLUGIN_PATHS

        soc: dict[str, Any] = {"reasoning_parser": self.reasoning_parser, "enable_in_reasoning": False}
        plugin = PLUGIN_PATHS.get(self.reasoning_parser)
        if plugin is not None:
            soc["reasoning_parser_plugin"] = str(plugin)
        return soc

    def engine_kwargs(self) -> dict[str, Any]:
        kw: dict[str, Any] = {
            "model": self.path,
            "max_model_len": self.max_model_len,
            "gpu_memory_utilization": float(self.cfg.get("gpu_memory_utilization", 0.9)),
            "enable_prefix_caching": bool(self.cfg.get("enable_prefix_caching", True)),
            "disable_log_stats": bool(self.engine_cfg.get("disable_log_stats", True)),
            "distributed_executor_backend": "mp",
            "structured_outputs_config": self.structured_outputs_config(),
        }
        if not (self.revision == "unknown" or self.revision.startswith("local-")):
            kw["revision"] = self.revision
        for opt in ("max_num_seqs", "max_num_batched_tokens", "enforce_eager", "kv_cache_dtype", "seed"):
            if self.cfg.get(opt) is not None:
                kw[opt] = self.cfg[opt]
        kw.update(self.extra_engine_kwargs())
        kw.update(self.cfg.get("engine_kwargs", {}) or {})
        return kw

    def _create_engine(self, replica: Replica) -> None:
        """Create one AsyncLLMEngine pinned to ``replica``'s GPUs (in this process)."""
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.engine.async_llm_engine import AsyncLLMEngine

        # Same start method for every engine (see module docstring).
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        kwargs = self.engine_kwargs()
        kwargs["tensor_parallel_size"] = len(replica.gpus)
        previous = os.environ.get("CUDA_VISIBLE_DEVICES")
        before = _descendant_pids()
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(replica.physical_gpus)
        t0 = time.monotonic()
        try:
            log.info("loading %s on GPUs %s (TP=%d)", replica.name, replica.physical_gpus, len(replica.gpus))
            replica.engine = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(**kwargs))
        finally:
            if previous is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = previous
        replica.pids = sorted(_descendant_pids() - before)
        log.info("loaded %s in %.0fs (engine pids %s)", replica.name, time.monotonic() - t0, replica.pids)

    async def start(self) -> None:
        self.load_tokenizer()
        replicas_cfg = self.cfg.get("replicas") or [[0]]
        for i, gpus in enumerate(replicas_cfg):
            gpus = [int(g) for g in gpus]
            replica = Replica(model=self.name, index=i, gpus=gpus, physical_gpus=physical_gpu_ids(gpus))
            self.replicas.append(replica)
            self._create_engine(replica)

    async def close(self) -> None:
        for replica in self.replicas:
            if replica.engine is not None:
                try:
                    replica.engine.shutdown()
                finally:
                    replica.engine = None

    # generation ---------------------------------------------------------------
    async def run(
        self,
        replica: Replica,
        prompt_ids: list[int],
        attempt: Attempt,
        schema_json: dict[str, Any],
        max_tokens: int,
        request_id: str,
    ) -> RawOutput:
        from vllm import SamplingParams, TokensPrompt
        from vllm.sampling_params import RequestOutputKind, StructuredOutputsParams

        sp_kwargs: dict[str, Any] = {
            "temperature": attempt.temperature,
            "max_tokens": max_tokens,
            "detokenize": False,
            "output_kind": RequestOutputKind.FINAL_ONLY,
            "structured_outputs": StructuredOutputsParams(json=schema_json),
        }
        for name in ("top_p", "top_k", "seed"):
            value = getattr(attempt, name)
            if value is not None:
                sp_kwargs[name] = value
        final = None
        async for out in replica.engine.generate(
            TokensPrompt(prompt_token_ids=prompt_ids),
            SamplingParams(**sp_kwargs),
            request_id=request_id,
            reasoning_parser_kwargs={"chat_template_kwargs": self.chat_template_kwargs()},
        ):
            final = out
        if final is None or not final.outputs:
            raise RuntimeError("vLLM returned no output")
        output = final.outputs[0]
        token_ids = list(output.token_ids)
        reasoning, content, reasoning_tokens = self.parse(token_ids)
        return RawOutput(
            reasoning=reasoning,
            content=content,
            finish_reason=output.finish_reason,
            prompt_tokens=len(prompt_ids),
            completion_tokens=len(token_ids),
            reasoning_tokens=reasoning_tokens,
        )
