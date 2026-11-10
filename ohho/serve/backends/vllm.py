"""vLLM OpenAI server adapter.

The ``vllm`` program is optional and is not installed with ohho-os.
"""

from __future__ import annotations

from collections.abc import Mapping
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
)
from .errors import ServingBackendError

_DTYPE = {
    "auto": "auto",
    "fp16": "float16",
    "float16": "float16",
    "half": "half",
    "bf16": "bfloat16",
    "bfloat16": "bfloat16",
    "fp32": "float32",
    "float32": "float32",
    "float": "float",
}


class VllmBackend(ServingBackend):
    """Launch ``vllm serve`` and expose ``http://host:port/v1``."""

    name = "vllm"
    default_executable = "vllm"

    def build_plan(
        self,
        model: str,
        precision: str,
        config: Mapping[str, Any],
    ) -> LaunchPlan:
        host = resolve_host(config)
        port = resolve_port(config, 8000)
        command = [
            *command_prefix(config, [self.default_executable]),
            "serve",
            model,
            "--host",
            host,
            "--port",
            str(port),
            *_precision_args(precision),
        ]
        parallel = optional_positive_int(config, "tensor_parallel_size")
        if parallel is not None:
            command.extend(["--tensor-parallel-size", str(parallel)])
        length = optional_positive_int(config, "max_model_len")
        if length is not None:
            command.extend(["--max-model-len", str(length)])
        command.extend(extra_args(config))
        return LaunchPlan(
            commands=(tuple(command),),
            endpoint_url=openai_base(host, port),
            health_url=origin(host, port) + "/health",
            version_argv=tuple(self.version_command(config)),
            precision=precision.strip().lower() or "auto",
        )


def _precision_args(precision: str) -> list[str]:
    key = precision.strip().lower() or "auto"
    if key in {"fp8", "float8"}:
        return ["--quantization", "fp8"]
    dtype = _DTYPE.get(key)
    if dtype is None:
        raise ServingBackendError(
            f"unsupported precision {precision!r} for vllm; "
            "use auto, fp16, bf16, fp32, half, or fp8"
        )
    return ["--dtype", dtype]
