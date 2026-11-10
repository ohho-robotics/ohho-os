"""llama.cpp server adapter for CPU and Jetson.

``llama-server`` is an external binary. It is not a dependency of ohho-os.
Weight precision is the GGUF file. The precision argument selects the KV
cache type (``--cache-type-k`` and ``--cache-type-v``). ``n_gpu_layers``
defaults to 0 so a CPU machine does not offload. Set it to ``auto`` or a
layer count for Jetson.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .base import LaunchPlan, ServingBackend
from .cmdline import (
    command_prefix,
    extra_args,
    openai_base,
    optional_positive_int,
    origin,
    resolve_host,
    resolve_port,
    semver_from_text,
)
from .errors import ServingBackendError

_CACHE = {
    "fp16": "f16",
    "float16": "f16",
    "f16": "f16",
    "half": "f16",
    "bf16": "bf16",
    "bfloat16": "bf16",
    "fp32": "f32",
    "float32": "f32",
    "f32": "f32",
    "q8_0": "q8_0",
    "q4_0": "q4_0",
    "q4_1": "q4_1",
    "q5_0": "q5_0",
    "q5_1": "q5_1",
    "iq4_nl": "iq4_nl",
}
_VERSION = re.compile(r"version:\s*(\d+)(?:\s*\(([0-9a-fA-F]+)\))?", re.IGNORECASE)


class LlamaCppBackend(ServingBackend):
    """Launch ``llama-server`` with an OpenAI-compatible ``/v1`` URL."""

    name = "llama-cpp"
    default_executable = "llama-server"

    def build_plan(
        self,
        model: str,
        precision: str,
        config: Mapping[str, Any],
    ) -> LaunchPlan:
        if (
            not config.get("dry_run")
            and not config.get("argv_prefix")
            and not Path(model).is_file()
        ):
            raise ServingBackendError(f"llama.cpp model file does not exist: {model}")
        host = resolve_host(config)
        port = resolve_port(config, 8080)
        layers = config.get("n_gpu_layers", 0)
        command = [
            *command_prefix(config, [self.default_executable]),
            "-m",
            model,
            "--host",
            host,
            "--port",
            str(port),
            "--n-gpu-layers",
            str(layers),
        ]
        ctx = optional_positive_int(config, "ctx_size")
        if ctx is not None:
            command.extend(["--ctx-size", str(ctx)])
        threads = optional_positive_int(config, "threads")
        if threads is not None:
            command.extend(["--threads", str(threads)])
        command.extend(_cache_args(precision))
        command.extend(extra_args(config))
        return LaunchPlan(
            commands=(tuple(command),),
            endpoint_url=openai_base(host, port),
            health_url=origin(host, port) + "/health",
            version_argv=tuple(self.version_command(config)),
            precision=precision.strip().lower() or "auto",
        )

    def parse_version_text(self, text: str) -> str | None:
        match = _VERSION.search(text)
        if match:
            commit = match.group(2)
            if commit:
                return f"{match.group(1)} ({commit})"
            return match.group(1)
        return semver_from_text(text)


def _cache_args(precision: str) -> list[str]:
    key = precision.strip().lower() or "auto"
    if key == "auto":
        return []
    cache = _CACHE.get(key)
    if cache is None:
        raise ServingBackendError(
            f"unsupported precision {precision!r} for llama-cpp; "
            "use auto, fp16, bf16, fp32, or a llama.cpp cache type "
            "such as q8_0"
        )
    return ["--cache-type-k", cache, "--cache-type-v", cache]
