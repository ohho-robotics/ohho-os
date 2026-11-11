"""Registry of LLM serving backends."""

from __future__ import annotations

from .base import ServingBackend
from .errors import ServingBackendError
from .llama_cpp import LlamaCppBackend
from .sglang import SglangBackend
from .tensorrt_llm import TensorRtLlmBackend
from .vllm import VllmBackend

_ADAPTERS: tuple[type[ServingBackend], ...] = (
    VllmBackend,
    SglangBackend,
    TensorRtLlmBackend,
    LlamaCppBackend,
)
_ALIASES = {
    "tensorrt": "tensorrt-llm",
    "tensorrt_llm": "tensorrt-llm",
    "trtllm": "tensorrt-llm",
    "llama.cpp": "llama-cpp",
    "llama_cpp": "llama-cpp",
    "llamacpp": "llama-cpp",
}


def backend_names() -> tuple[str, ...]:
    """Return the canonical backend ids."""
    return tuple(cls.name for cls in _ADAPTERS)


def open_backend(name: str) -> ServingBackend:
    """Return a new adapter for ``name``.

    The adapter is stateless until :meth:`ServingBackend.launch`.
    """
    key = _ALIASES.get(name.strip().lower(), name.strip().lower())
    for cls in _ADAPTERS:
        if cls.name == key:
            return cls()
    known = ", ".join(backend_names())
    raise ServingBackendError(f"unknown serving backend {name!r}; known: {known}")
