"""TensorRT-LLM adapter.

``trtllm-serve`` is the OpenAI-compatible server. ``tritonserver`` is the
Triton path: its health probe is ``/v2/health/ready``, and the OpenAI base
URL is ``http://host:openai_port/v1``. Neither binary is a dependency of
ohho-os. Weight dtype is fixed by the checkpoint. ``fp8`` and ``nvfp4``
select ``--kv_cache_dtype`` on ``trtllm-serve``.
"""

from __future__ import annotations

import sys
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
)
from .errors import ServingBackendError
from .sweep_support import launch_args

_LAUNCHERS = {"trtllm-serve", "triton"}
_KV_CACHE = {"fp8": "fp8", "float8": "fp8", "fp4": "nvfp4", "nvfp4": "nvfp4"}
_WEIGHT = {"auto", "fp16", "float16", "bf16", "bfloat16", "fp32", "float32", "float"}


class TensorRtLlmBackend(ServingBackend):
    """Launch TensorRT-LLM through ``trtllm-serve`` or Triton."""

    name = "tensorrt-llm"
    default_executable = "trtllm-serve"

    def build_plan(
        self,
        model: str,
        precision: str,
        config: Mapping[str, Any],
    ) -> LaunchPlan:
        launcher = str(config.get("launcher") or "trtllm-serve").strip().lower()
        if launcher not in _LAUNCHERS:
            raise ServingBackendError(
                f"unsupported TensorRT-LLM launcher {launcher!r}; "
                "use trtllm-serve or triton"
            )
        if launcher == "triton":
            return self._triton(model, precision, config)
        return self._trtllm(model, precision, config)

    def version_command(self, config: Mapping[str, Any]) -> list[str]:
        if (
            config.get("version_argv") is not None
            or config.get("argv_prefix") is not None
        ):
            return super().version_command(config)
        if config.get("executable"):
            return super().version_command(config)
        launcher = str(config.get("launcher") or "trtllm-serve").strip().lower()
        if launcher == "triton":
            return ["tritonserver", "--version"]
        return [
            sys.executable,
            "-c",
            "import tensorrt_llm; print(tensorrt_llm.__version__)",
        ]

    def _trtllm(
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
            "--host",
            host,
            "--port",
            str(port),
        ]
        parallel = optional_positive_int(config, "tp_size")
        if parallel is None:
            parallel = optional_positive_int(config, "tensor_parallel_size")
        if parallel is not None:
            command.extend(["--tp_size", str(parallel)])
        command.extend(_precision_args(precision))
        config_path = config.get("config_file") or config.get("config_path")
        if config_path:
            command.extend(["--config", str(config_path)])
        command.extend(launch_args(self.name, config))
        command.append(model)
        command.extend(extra_args(config))
        return LaunchPlan(
            commands=(tuple(command),),
            endpoint_url=openai_base(host, port),
            health_url=origin(host, port) + "/health",
            version_argv=tuple(self.version_command(config)),
            precision=precision.strip().lower() or "auto",
        )

    def _triton(
        self,
        model: str,
        precision: str,
        config: Mapping[str, Any],
    ) -> LaunchPlan:
        key = precision.strip().lower() or "auto"
        if key not in _WEIGHT and key not in _KV_CACHE:
            raise ServingBackendError(
                f"unsupported precision {precision!r} for tensorrt-llm; "
                "use auto, fp16, bf16, fp32, fp8, or nvfp4"
            )
        host = resolve_host(config)
        port = resolve_port(config, 8000)
        repository = str(config.get("model_repository") or model)
        if not config.get("dry_run") and not config.get("argv_prefix"):
            if not Path(repository).is_dir():
                raise ServingBackendError(
                    f"Triton model repository does not exist: {repository}"
                )
        grpc = _configured_port(config, "grpc_port", port + 1)
        metrics = _configured_port(config, "metrics_port", port + 2)
        executable = str(config.get("triton_executable") or "tritonserver")
        prefix = config.get("argv_prefix")
        if isinstance(prefix, str):
            command = [prefix]
        elif isinstance(prefix, (list, tuple)) and prefix:
            command = [str(part) for part in prefix]
        elif prefix is not None:
            raise ServingBackendError("argv_prefix must be a non-empty list of strings")
        elif config.get("executable"):
            command = [str(config["executable"])]
        else:
            command = [executable]
        command.extend(
            [
                "--model-repository",
                repository,
                "--http-port",
                str(port),
                "--grpc-port",
                str(grpc),
                "--metrics-port",
                str(metrics),
            ]
        )
        command.extend(launch_args(self.name, config))
        command.extend(extra_args(config))
        commands = [tuple(command)]
        openai = config.get("openai_argv")
        if openai:
            if not isinstance(openai, (list, tuple)) or not all(
                isinstance(part, str) for part in openai
            ):
                raise ServingBackendError("openai_argv must be a list of strings")
            commands.append(tuple(str(part) for part in openai))
        openai_port = _configured_port(config, "openai_port", port)
        return LaunchPlan(
            commands=tuple(commands),
            endpoint_url=openai_base(host, openai_port),
            health_url=origin(host, port) + "/v2/health/ready",
            version_argv=tuple(self.version_command(config)),
            precision=key,
        )


def _precision_args(precision: str) -> list[str]:
    key = precision.strip().lower() or "auto"
    kv = _KV_CACHE.get(key)
    if kv is not None:
        return ["--kv_cache_dtype", kv]
    if key in _WEIGHT:
        # trtllm-serve has no weight-dtype flag. fp16/bf16/fp32 come from
        # the checkpoint. auto leaves the server default in place.
        return []
    raise ServingBackendError(
        f"unsupported precision {precision!r} for tensorrt-llm; "
        "use auto, fp16, bf16, fp32, fp8, or nvfp4"
    )


def _configured_port(config: Mapping[str, Any], key: str, default: int) -> int:
    if config.get(key) is None:
        return _check_port(default, key)
    try:
        port = int(config[key])
    except (TypeError, ValueError) as exc:
        raise ServingBackendError(f"invalid {key} {config[key]!r}") from exc
    return _check_port(port, key)


def _check_port(port: int, name: str) -> int:
    if port < 1 or port > 65535:
        raise ServingBackendError(f"{name} out of range: {port}")
    return port
