"""SGLang OpenAI server adapter.

SGLang is imported or launched only as an external program. It is not a
dependency of ohho-os.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping
from typing import Any

from .base import LaunchPlan, ServingBackend
from .cmdline import (
    extra_args,
    openai_base,
    optional_positive_int,
    origin,
    resolve_host,
    resolve_port,
)
from .errors import ServingBackendError
from .sweep_support import launch_args

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


class SglangBackend(ServingBackend):
    """Launch ``python -m sglang.launch_server``."""

    name = "sglang"
    default_executable = sys.executable

    def build_plan(
        self,
        model: str,
        precision: str,
        config: Mapping[str, Any],
    ) -> LaunchPlan:
        host = resolve_host(config)
        port = resolve_port(config, 30000)
        command = [
            *self._prefix(config),
            "--model-path",
            model,
            "--host",
            host,
            "--port",
            str(port),
            *_precision_args(precision),
        ]
        parallel = optional_positive_int(config, "tp_size")
        if parallel is None:
            parallel = optional_positive_int(config, "tensor_parallel_size")
        if parallel is not None:
            command.extend(["--tp-size", str(parallel)])
        command.extend(launch_args(self.name, config))
        command.extend(extra_args(config))
        return LaunchPlan(
            commands=(tuple(command),),
            endpoint_url=openai_base(host, port),
            health_url=origin(host, port) + "/health",
            version_argv=tuple(self.version_command(config)),
            precision=precision.strip().lower() or "auto",
        )

    def version_command(self, config: Mapping[str, Any]) -> list[str]:
        if (
            config.get("version_argv") is not None
            or config.get("argv_prefix") is not None
        ):
            return super().version_command(config)
        python = str(config.get("executable") or sys.executable)
        return [python, "-c", "import sglang; print(sglang.__version__)"]

    def _prefix(self, config: Mapping[str, Any]) -> list[str]:
        prefix = config.get("argv_prefix")
        if prefix is not None:
            if isinstance(prefix, str):
                return [prefix]
            if not isinstance(prefix, (list, tuple)) or not prefix:
                raise ServingBackendError(
                    "argv_prefix must be a non-empty list of strings"
                )
            return [str(part) for part in prefix]
        python = str(config.get("executable") or sys.executable)
        return [python, "-m", "sglang.launch_server"]


def _precision_args(precision: str) -> list[str]:
    key = precision.strip().lower() or "auto"
    if key in {"fp8", "float8"}:
        return ["--quantization", "fp8"]
    dtype = _DTYPE.get(key)
    if dtype is None:
        raise ServingBackendError(
            f"unsupported precision {precision!r} for sglang; "
            "use auto, fp16, bf16, fp32, half, or fp8"
        )
    return ["--dtype", dtype]
