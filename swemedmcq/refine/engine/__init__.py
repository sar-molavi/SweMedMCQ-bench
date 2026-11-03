"""LLM engine: cached, routed, retried structured generation on vLLM.

Public API::

    from swemedmcq.refine.engine import LLMRequest, LLMResult, ModelPool, FakePool

``ModelPool.create(cfg, schemas, cache_dir)`` loads vLLM engines (imported
lazily); ``FakePool(responder, schemas, cache_dir)`` is the GPU-free double.
"""

from .fake import FakeOutput, FakePool
from .pool import MODEL_NAMES, Attempt, LLMRequest, LLMResult, ModelName, ModelPool, seed_from_key

__all__ = [
    "Attempt",
    "FakeOutput",
    "FakePool",
    "LLMRequest",
    "LLMResult",
    "MODEL_NAMES",
    "ModelName",
    "ModelPool",
    "seed_from_key",
]
