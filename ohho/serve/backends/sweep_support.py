"""Which serving backends can sweep batching and cache options.

The engines stay external programs. This module only records the flags a
later launch would pass. It does not start a process.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .errors import ServingBackendError

SWEEP_AXES: tuple[str, ...] = (
    "concurrency",
    "max_batch_tokens",
    "prefix_cache",
    "chunked_prefill",
    "speculative_decoding",
)

DRAFT_REQUIRED_REASON = "speculative decoding needs a draft model (speculative_model)"
STANDIN_SPECULATIVE_REASON = (
    "the CPU stand-in has no draft model, so speculative decoding is not measured"
)
LLAMA_CHUNKED_REASON = (
    "llama.cpp has no chunked-prefill scheduler; prompt chunks follow the "
    "fixed ubatch size"
)
TRT_PREFIX_REASON = (
    "tensorrt-llm applies prefix-cache reuse in the engine config file, "
    "not as a sweep flag"
)
TRT_CHUNKED_REASON = (
    "tensorrt-llm applies chunked context in the engine config file, "
    "not as a sweep flag"
)
TRT_SPECULATIVE_REASON = (
    "tensorrt-llm applies speculative decoding in the engine config file, "
    "not as a sweep flag"
)
TRITON_BATCH_REASON = (
    "the triton launcher reads batch limits from the model repository, "
    "not from a sweep flag"
)

_ALIASES = {
    "standin": "stand-in",
    "cpu-stand-in": "stand-in",
    "tensorrt": "tensorrt-llm",
    "tensorrt_llm": "tensorrt-llm",
    "trtllm": "tensorrt-llm",
    "llama.cpp": "llama-cpp",
    "llama_cpp": "llama-cpp",
    "llamacpp": "llama-cpp",
}


@dataclass(frozen=True)
class AxisSupport:
    """Whether one sweep axis can be changed on this backend."""

    name: str
    supported: bool
    reason: str = ""


@dataclass(frozen=True)
class BackendSweepSupport:
    """Support for every sweep axis on one backend."""

    backend: str
    axes: tuple[AxisSupport, ...]

    def axis(self, name: str) -> AxisSupport:
        """Return the support record for ``name``."""
        for item in self.axes:
            if item.name == name:
                return item
        raise KeyError(name)


def canonical_backend(name: str) -> str:
    """Return the sweep id for ``name``.

    ``stand-in`` is the CPU model. The other ids are serving adapters.
    """
    if not isinstance(name, str) or not name.strip():
        raise ServingBackendError("backend is required")
    key = _ALIASES.get(name.strip().lower(), name.strip().lower())
    if key == "stand-in":
        return key
    from .registry import backend_names

    if key in backend_names():
        return key
    known = ", ".join(("stand-in", *backend_names()))
    raise ServingBackendError(f"unknown serving backend {name!r}; known: {known}")


def support_for(
    backend: str,
    *,
    speculative_model: str = "",
    launcher: str = "",
) -> BackendSweepSupport:
    """Return which axes ``backend`` can sweep.

    Speculative decoding is supported only when ``speculative_model`` is
    set and the engine has a draft-model flag. An empty model is reported
    as unsupported with a reason, so a sweep skips it instead of failing.
    """
    name = canonical_backend(backend)
    launcher_name = (launcher or "trtllm-serve").strip().lower() or "trtllm-serve"
    draft = speculative_model.strip()
    axes = (
        AxisSupport("concurrency", True, ""),
        _max_batch(name, launcher_name),
        _prefix(name),
        _chunked(name),
        _speculative(name, draft),
    )
    return BackendSweepSupport(name, axes)


def launch_args(backend: str, config: Mapping[str, Any]) -> list[str]:
    """Argv for sweep keys that are present in ``config``.

    Unsupported requests that would turn a feature on, or set a batch
    limit the engine cannot express, raise :class:`ServingBackendError`.
    Turning an unsupported feature off is a no-op: that is already the
    only behavior the engine has.
    """
    name = canonical_backend(backend)
    draft = str(config.get("speculative_model") or "").strip()
    launcher = str(config.get("launcher") or "")
    support = support_for(name, speculative_model=draft, launcher=launcher)
    refused = _refused(support, config, draft)
    if refused:
        raise ServingBackendError(refused)
    if name == "stand-in":
        return []
    args: list[str] = []
    batch = _optional_positive(config, "max_batch_tokens")
    prefix = _optional_bool(config, "prefix_cache")
    chunked = _optional_bool(config, "chunked_prefill")
    tokens = _optional_positive(config, "speculative_tokens") or 4
    parallel = _optional_positive(config, "concurrency")
    if name == "vllm":
        args.extend(_vllm_args(batch, prefix, chunked, draft, tokens))
    elif name == "sglang":
        args.extend(_sglang_args(batch, prefix, chunked, draft, tokens))
    elif name == "tensorrt-llm":
        args.extend(_tensorrt_args(batch))
    elif name == "llama-cpp":
        args.extend(_llama_args(batch, prefix, draft, tokens, parallel))
    return args


def _max_batch(backend: str, launcher: str) -> AxisSupport:
    if backend == "tensorrt-llm" and launcher == "triton":
        return AxisSupport("max_batch_tokens", False, TRITON_BATCH_REASON)
    return AxisSupport("max_batch_tokens", True, "")


def _prefix(backend: str) -> AxisSupport:
    if backend == "tensorrt-llm":
        return AxisSupport("prefix_cache", False, TRT_PREFIX_REASON)
    return AxisSupport("prefix_cache", True, "")


def _chunked(backend: str) -> AxisSupport:
    if backend == "llama-cpp":
        return AxisSupport("chunked_prefill", False, LLAMA_CHUNKED_REASON)
    if backend == "tensorrt-llm":
        return AxisSupport("chunked_prefill", False, TRT_CHUNKED_REASON)
    return AxisSupport("chunked_prefill", True, "")


def _speculative(backend: str, draft: str) -> AxisSupport:
    if backend == "stand-in":
        return AxisSupport("speculative_decoding", False, STANDIN_SPECULATIVE_REASON)
    if backend == "tensorrt-llm":
        return AxisSupport("speculative_decoding", False, TRT_SPECULATIVE_REASON)
    if draft:
        return AxisSupport("speculative_decoding", True, "")
    return AxisSupport("speculative_decoding", False, DRAFT_REQUIRED_REASON)


def _refused(
    support: BackendSweepSupport,
    config: Mapping[str, Any],
    draft: str,
) -> str:
    if (
        "max_batch_tokens" in config
        and config.get("max_batch_tokens") is not None
        and not support.axis("max_batch_tokens").supported
    ):
        return support.axis("max_batch_tokens").reason
    if (
        config.get("prefix_cache") is True
        and not support.axis("prefix_cache").supported
    ):
        return support.axis("prefix_cache").reason
    if (
        config.get("chunked_prefill") is True
        and not support.axis("chunked_prefill").supported
    ):
        return support.axis("chunked_prefill").reason
    wants_speculative = config.get("speculative_decoding") is True or bool(draft)
    if wants_speculative and not support.axis("speculative_decoding").supported:
        return support.axis("speculative_decoding").reason
    return ""


def _optional_positive(config: Mapping[str, Any], key: str) -> int | None:
    if key not in config or config.get(key) is None:
        return None
    raw = config[key]
    if isinstance(raw, bool) or not isinstance(raw, int):
        try:
            if isinstance(raw, str) and raw.strip().isdigit():
                value = int(raw.strip())
            else:
                raise TypeError
        except (TypeError, ValueError) as exc:
            raise ServingBackendError(f"{key} must be a positive integer") from exc
    else:
        value = raw
    if value < 1:
        raise ServingBackendError(f"{key} must be a positive integer")
    return value


def _optional_bool(config: Mapping[str, Any], key: str) -> bool | None:
    if key not in config or config.get(key) is None:
        return None
    value = config[key]
    if not isinstance(value, bool):
        raise ServingBackendError(f"{key} must be a boolean")
    return value


def _vllm_args(
    batch: int | None,
    prefix: bool | None,
    chunked: bool | None,
    draft: str,
    tokens: int,
) -> list[str]:
    args: list[str] = []
    if batch is not None:
        args.extend(["--max-num-batched-tokens", str(batch)])
    if prefix is True:
        args.append("--enable-prefix-caching")
    elif prefix is False:
        args.append("--no-enable-prefix-caching")
    if chunked is True:
        args.append("--enable-chunked-prefill")
    elif chunked is False:
        args.append("--no-enable-chunked-prefill")
    if draft:
        payload = {"model": draft, "num_speculative_tokens": tokens}
        args.extend(
            ["--speculative-config", json.dumps(payload, separators=(",", ":"))]
        )
    return args


def _sglang_args(
    batch: int | None,
    prefix: bool | None,
    chunked: bool | None,
    draft: str,
    tokens: int,
) -> list[str]:
    args: list[str] = []
    if batch is not None:
        args.extend(["--max-prefill-tokens", str(batch)])
    if prefix is False:
        args.append("--disable-radix-cache")
    if chunked is True:
        # The chunk size is the prefill budget. The on state needs a size.
        size = batch if batch is not None else 8192
        args.extend(["--chunked-prefill-size", str(size)])
    elif chunked is False:
        args.extend(["--chunked-prefill-size", "-1"])
    if draft:
        args.extend(
            [
                "--speculative-algorithm",
                "EAGLE",
                "--speculative-draft-model-path",
                draft,
                "--speculative-num-steps",
                str(tokens),
                "--speculative-eagle-topk",
                "1",
                "--speculative-num-draft-tokens",
                str(tokens),
            ]
        )
    return args


def _tensorrt_args(batch: int | None) -> list[str]:
    if batch is None:
        return []
    return ["--max_num_tokens", str(batch)]


def _llama_args(
    batch: int | None,
    prefix: bool | None,
    draft: str,
    tokens: int,
    parallel: int | None,
) -> list[str]:
    args: list[str] = []
    if batch is not None:
        args.extend(["--batch-size", str(batch)])
    if prefix is True:
        args.extend(["--cache-reuse", "256"])
    elif prefix is False:
        args.extend(["--cache-reuse", "0"])
    if parallel is not None:
        args.extend(["--parallel", str(parallel)])
    if draft:
        args.extend(["--model-draft", draft, "--draft-max", str(tokens)])
    return args
