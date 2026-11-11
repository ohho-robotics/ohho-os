"""OpenAI-compatible LLM serving backends.

The engines (vLLM, SGLang, TensorRT-LLM, llama.cpp) are external programs.
Importing this package does not import them.
"""

from __future__ import annotations

from .base import LaunchPlan, ServingBackend
from .errors import ServingBackendError
from .registry import backend_names, open_backend

__all__ = [
    "LaunchPlan",
    "ServingBackend",
    "ServingBackendError",
    "backend_names",
    "open_backend",
]
