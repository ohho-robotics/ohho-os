"""Deterministic training switches for a recipe.

``deterministic: true`` on an ``ohho.recipe/v1`` document means three things:

* batch order comes from a ``torch.Generator`` seeded with ``hyperparameters.seed``
* ``DataLoader`` workers are pinned to :data:`FIXED_DATALOADER_WORKERS`
* torch deterministic algorithms and cuDNN deterministic mode are turned on
  when those knobs exist, and restored when the trainer returns

The helpers here do not import torch. Callers pass the module they already
imported. A missing knob is skipped. CUDA exactness can also need
``CUBLAS_WORKSPACE_CONFIG=:4096:8`` in the environment before the process
starts. This module does not set that variable.
"""

from __future__ import annotations

import random
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any

FIXED_DATALOADER_WORKERS = 0


def deterministic_requested(recipe: Mapping[str, Any]) -> bool:
    """True only when the recipe sets ``deterministic`` to true."""
    return recipe.get("deterministic") is True


def ensure_deterministic_seed(recipe: dict[str, Any]) -> int | None:
    """Return the seed the trainer should use.

    When ``deterministic`` is true and ``hyperparameters.seed`` is omitted,
    set that seed to ``0`` on ``recipe`` so the manifest records the value
    the run actually used. An existing integer seed is left unchanged.
    """
    hyperparameters = recipe.get("hyperparameters")
    if not isinstance(hyperparameters, dict):
        hyperparameters = {}
    seed = hyperparameters.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        seed = None
    if not deterministic_requested(recipe):
        return seed
    if seed is None:
        if not isinstance(recipe.get("hyperparameters"), dict):
            recipe["hyperparameters"] = {}
        recipe["hyperparameters"]["seed"] = 0
        return 0
    return seed


def dataloader_kwargs(
    torch_module: Any,
    *,
    deterministic: bool,
    seed: int | None,
    shuffle: bool,
) -> dict[str, Any]:
    """Keyword arguments for one ``DataLoader``.

    Outside deterministic mode the loader keeps today's shuffle flag and the
    library's default worker count. Inside it, workers are
    :data:`FIXED_DATALOADER_WORKERS` and a shuffled loader draws its
    permutation from a generator seeded with ``seed``.
    """
    if not deterministic:
        return {"shuffle": shuffle}
    kwargs: dict[str, Any] = {
        "shuffle": shuffle,
        "num_workers": FIXED_DATALOADER_WORKERS,
    }
    generator_type = getattr(torch_module, "Generator", None)
    if shuffle and seed is not None and callable(generator_type):
        generator = generator_type()
        generator.manual_seed(int(seed))
        kwargs["generator"] = generator
    return kwargs


def reseed_runtime(torch_module: Any, seed: int) -> None:
    """Seed Python, NumPy when installed, and torch from ``seed``.

    Called after the seeded loader generator is built, so model initialisation
    and later ``randn`` calls do not depend on how many values that generator
    consumed.
    """
    random.seed(seed)
    try:
        import numpy as np
    except ImportError:
        np = None
    if np is not None:
        np.random.seed(seed)
    manual_seed = getattr(torch_module, "manual_seed", None)
    if callable(manual_seed):
        manual_seed(int(seed))
    cuda = getattr(torch_module, "cuda", None)
    available = getattr(cuda, "is_available", None)
    manual_all = getattr(cuda, "manual_seed_all", None)
    if callable(available) and callable(manual_all):
        try:
            if available():
                manual_all(int(seed))
        except Exception:
            return


@contextmanager
def deterministic_algorithms(
    torch_module: Any, *, enabled: bool
) -> Iterator[dict[str, Any]]:
    """Turn on deterministic torch and cuDNN behavior when ``enabled``.

    Restores the previous flags on the way out, including when training
    raises. When ``enabled`` is false this is a no-op. Knobs the installed
    torch does not have are left alone. A knob that raises is skipped.
    """
    if not enabled:
        yield {"enabled": False, "workers": None}
        return
    snapshot = _snapshot(torch_module)
    applied = _enable(torch_module)
    try:
        yield applied
    finally:
        _restore(torch_module, snapshot)


def _enable(torch_module: Any) -> dict[str, Any]:
    applied: dict[str, Any] = {
        "enabled": True,
        "algorithms": False,
        "cudnn_deterministic": False,
        "workers": FIXED_DATALOADER_WORKERS,
    }
    use = getattr(torch_module, "use_deterministic_algorithms", None)
    if callable(use):
        if _call(use, True):
            applied["algorithms"] = True
        elif _call(use, True, warn_only=True):
            applied["algorithms"] = "warn_only"
    cudnn = _cudnn(torch_module)
    if cudnn is not None:
        if _set_attr(cudnn, "deterministic", True):
            applied["cudnn_deterministic"] = True
        _set_attr(cudnn, "benchmark", False)
        _set_attr(cudnn, "allow_tf32", False)
    matmul = _cuda_matmul(torch_module)
    if matmul is not None:
        _set_attr(matmul, "allow_tf32", False)
    cuda_backend = _cuda_backend(torch_module)
    if cuda_backend is not None:
        _call_method(cuda_backend, "enable_flash_sdp", False)
        _call_method(cuda_backend, "enable_mem_efficient_sdp", False)
        _call_method(cuda_backend, "enable_math_sdp", True)
    mkldnn = _mkldnn(torch_module)
    if mkldnn is not None:
        _set_attr(mkldnn, "enabled", False)
    return applied


def _snapshot(torch_module: Any) -> dict[str, Any]:
    state: dict[str, Any] = {}
    enabled = getattr(torch_module, "are_deterministic_algorithms_enabled", None)
    if callable(enabled):
        try:
            state["algorithms"] = bool(enabled())
        except Exception:
            state["algorithms"] = False
    warn = getattr(torch_module, "is_deterministic_algorithms_warn_only_enabled", None)
    if callable(warn):
        try:
            state["warn_only"] = bool(warn())
        except Exception:
            state["warn_only"] = False
    cudnn = _cudnn(torch_module)
    if cudnn is not None:
        _remember(state, cudnn, "cudnn_deterministic", "deterministic")
        _remember(state, cudnn, "cudnn_benchmark", "benchmark")
        _remember(state, cudnn, "cudnn_allow_tf32", "allow_tf32")
    matmul = _cuda_matmul(torch_module)
    if matmul is not None:
        _remember(state, matmul, "matmul_allow_tf32", "allow_tf32")
    cuda_backend = _cuda_backend(torch_module)
    if cuda_backend is not None:
        for key, getter in (
            ("flash_sdp", "flash_sdp_enabled"),
            ("mem_efficient_sdp", "mem_efficient_sdp_enabled"),
            ("math_sdp", "math_sdp_enabled"),
        ):
            fn = getattr(cuda_backend, getter, None)
            if callable(fn):
                try:
                    state[key] = bool(fn())
                except Exception:
                    continue
    mkldnn = _mkldnn(torch_module)
    if mkldnn is not None:
        _remember(state, mkldnn, "mkldnn_enabled", "enabled")
    return state


def _restore(torch_module: Any, snapshot: Mapping[str, Any]) -> None:
    use = getattr(torch_module, "use_deterministic_algorithms", None)
    if callable(use) and "algorithms" in snapshot:
        warn_only = bool(snapshot.get("warn_only", False))
        if not _call(use, bool(snapshot["algorithms"]), warn_only=warn_only):
            _call(use, bool(snapshot["algorithms"]))
    cudnn = _cudnn(torch_module)
    if cudnn is not None:
        _restore_attr(cudnn, "deterministic", snapshot.get("cudnn_deterministic"))
        _restore_attr(cudnn, "benchmark", snapshot.get("cudnn_benchmark"))
        _restore_attr(cudnn, "allow_tf32", snapshot.get("cudnn_allow_tf32"))
    matmul = _cuda_matmul(torch_module)
    if matmul is not None:
        _restore_attr(matmul, "allow_tf32", snapshot.get("matmul_allow_tf32"))
    cuda_backend = _cuda_backend(torch_module)
    if cuda_backend is not None:
        if "flash_sdp" in snapshot:
            _call_method(cuda_backend, "enable_flash_sdp", snapshot["flash_sdp"])
        if "mem_efficient_sdp" in snapshot:
            _call_method(
                cuda_backend,
                "enable_mem_efficient_sdp",
                snapshot["mem_efficient_sdp"],
            )
        if "math_sdp" in snapshot:
            _call_method(cuda_backend, "enable_math_sdp", snapshot["math_sdp"])
    mkldnn = _mkldnn(torch_module)
    if mkldnn is not None:
        _restore_attr(mkldnn, "enabled", snapshot.get("mkldnn_enabled"))


def _cudnn(torch_module: Any) -> Any:
    backends = getattr(torch_module, "backends", None)
    if backends is None:
        return None
    return getattr(backends, "cudnn", None)


def _cuda_backend(torch_module: Any) -> Any:
    backends = getattr(torch_module, "backends", None)
    if backends is None:
        return None
    return getattr(backends, "cuda", None)


def _cuda_matmul(torch_module: Any) -> Any:
    cuda = _cuda_backend(torch_module)
    if cuda is None:
        return None
    return getattr(cuda, "matmul", None)


def _mkldnn(torch_module: Any) -> Any:
    backends = getattr(torch_module, "backends", None)
    if backends is None:
        return None
    return getattr(backends, "mkldnn", None)


def _remember(state: dict[str, Any], obj: Any, key: str, attr: str) -> None:
    if hasattr(obj, attr):
        try:
            state[key] = getattr(obj, attr)
        except Exception:
            return


def _set_attr(obj: Any, attr: str, value: Any) -> bool:
    if not hasattr(obj, attr):
        return False
    try:
        setattr(obj, attr, value)
    except Exception:
        return False
    return True


def _restore_attr(obj: Any, attr: str, value: Any) -> None:
    if value is None:
        return
    _set_attr(obj, attr, value)


def _call(fn: Any, *args: Any, **kwargs: Any) -> bool:
    try:
        fn(*args, **kwargs)
    except Exception:
        return False
    return True


def _call_method(obj: Any, name: str, *args: Any) -> bool:
    fn = getattr(obj, name, None)
    if not callable(fn):
        return False
    return _call(fn, *args)
