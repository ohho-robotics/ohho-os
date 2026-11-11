"""One W&B sweep trial, run as an OhhO train job.

W&B launches this module with the sweep command from
:mod:`ohho.train.sweep_wandb`:

    python -m ohho.train.sweep_agent ${args}

``${args}`` becomes ``--path=value`` flags. This module does not import
``wandb`` and does not call the W&B API. Recognized overrides are passed to
:func:`ohho.train.finetune`.

The agent runs on CPU. A trial that asks for CUDA, TensorRT, or another
non-CPU device fails before training.
"""

from __future__ import annotations

import json
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .sweep import SweepError

_KEY_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_.-]*")
_FINETUNE_FIELDS = {
    "dataset": "dataset",
    "train.dataset": "dataset",
    "policy": "policy",
    "checkpoint": "checkpoint",
    "output_dir": "output_dir",
    "num_epochs": "num_epochs",
    "train.num_epochs": "num_epochs",
    "epochs": "num_epochs",
    "train.epochs": "num_epochs",
    "batch_size": "batch_size",
    "train.batch_size": "batch_size",
    "lr": "lr",
    "optimizer.lr": "lr",
    "train.lr": "lr",
    "mock": "mock",
}
_CPU_DEVICES = frozenset({"", "cpu", "auto"})


@dataclass(frozen=True)
class TrainJobResult:
    """One finished OhhO train job."""

    job: str
    overrides: dict[str, Any]
    checkpoint: str


def parse_agent_args(argv: Sequence[str]) -> dict[str, Any]:
    """Parse W&B ``${args}`` flags and an optional JSON object.

    Later flags replace earlier ones. A JSON object is flattened so nested
    keys become dotted paths.
    """

    overrides: dict[str, Any] = {}
    index = 0
    args = list(argv)
    while index < len(args):
        token = args[index]
        if token == "--overrides-json":
            if index + 1 >= len(args):
                raise SweepError("sweep agent --overrides-json needs a value")
            overrides.update(_json_object(args[index + 1]))
            index += 2
            continue
        if token.startswith("--") and "=" in token:
            key, raw = token[2:].split("=", 1)
            overrides[_key(key, token)] = _parse_scalar(raw)
            index += 1
            continue
        if token.startswith("--"):
            key = _key(token[2:], token)
            if index + 1 < len(args) and not args[index + 1].startswith("--"):
                overrides[key] = _parse_scalar(args[index + 1])
                index += 2
                continue
            overrides[key] = True
            index += 1
            continue
        if token[:1] in "{[":
            overrides.update(_json_object(token))
            index += 1
            continue
        if "=" in token and not token.startswith("="):
            key, raw = token.split("=", 1)
            overrides[_key(key, token)] = _parse_scalar(raw)
            index += 1
            continue
        raise SweepError(f"unrecognized sweep agent argument {token!r}")
    return overrides


def resolve_train_device(
    device: str,
    *,
    available: Callable[[str], bool] | None = None,
) -> str:
    """Return the device passed to training.

    Omitted, ``cpu``, and ``auto`` run on CPU. Any other device is used only
    when ``available`` reports it. The default probe does not use a GPU or
    TensorRT and raises :class:`ohho.train.TrainUnavailable` before training.
    """

    if not isinstance(device, str):
        raise SweepError("device must be a string")
    folded = device.strip().lower()
    if folded in _CPU_DEVICES:
        return "cpu"
    probe = _gpu_available if available is None else available
    if not probe(folded):
        from . import TrainUnavailable

        raise TrainUnavailable(
            f"OhhO train job asked for device {device!r}. "
            "This sweep agent runs on CPU. "
            "GPU and TensorRT trials are not available. "
            "No training was started."
        )
    return device.strip()


def execute_train_job(
    overrides: Mapping[str, Any],
    *,
    trainer: Callable[..., str] | None = None,
    device_available: Callable[[str], bool] | None = None,
) -> TrainJobResult:
    """Run one OhhO train job for a sweep trial.

    ``trainer`` defaults to :func:`ohho.train.finetune`. Unknown overrides
    stay on the result and are not forwarded.
    """

    flat = _flatten(overrides)
    raw_device = flat.get("device", flat.get("train.device", "cpu"))
    if not isinstance(raw_device, str):
        raise SweepError("device must be a string")
    device = resolve_train_device(raw_device, available=device_available)
    kwargs = _finetune_kwargs(flat)
    kwargs["device"] = device
    if "dataset" not in kwargs:
        raise SweepError(
            "OhhO train job needs a dataset override or a dataset constant. "
            "No training was started."
        )
    run = _default_trainer if trainer is None else trainer
    checkpoint = run(**kwargs)
    if not isinstance(checkpoint, str) or not checkpoint:
        raise SweepError("OhhO train job did not return a checkpoint path")
    return TrainJobResult(
        job="ohho.train",
        overrides=_copy_overrides(flat),
        checkpoint=checkpoint,
    )


def main(
    argv: Sequence[str] | None = None,
    *,
    trainer: Callable[..., str] | None = None,
) -> int:
    """CLI entry for ``python -m ohho.train.sweep_agent``."""

    from . import TrainUnavailable

    args = list(sys.argv[1:] if argv is None else argv)
    try:
        result = execute_train_job(parse_agent_args(args), trainer=trainer)
    except (SweepError, TrainUnavailable, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"job: {result.job}")
    print(f"checkpoint: {result.checkpoint}")
    return 0


def _default_trainer(**kwargs: Any) -> str:
    from ohho.train import finetune

    return finetune(**kwargs)


def _gpu_available(device: str) -> bool:
    """Report no GPU and no TensorRT.

    This agent does not probe CUDA or TensorRT. A machine with a GPU still
    takes the CPU path unless a caller passes its own ``available`` probe.
    """

    del device
    return False


def _finetune_kwargs(flat: Mapping[str, Any]) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    for source, dest in _FINETUNE_FIELDS.items():
        if source in flat:
            kwargs[dest] = flat[source]
    return kwargs


def _flatten(data: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in data.items():
        if not isinstance(key, str) or not key:
            raise SweepError("override names must be strings")
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            out.update(_flatten(value, path))
        else:
            out[path] = value
    return out


def _copy_overrides(flat: Mapping[str, Any]) -> dict[str, Any]:
    copied: dict[str, Any] = {}
    for key, value in flat.items():
        copied[key] = value
    return copied


def _json_object(text: str) -> dict[str, Any]:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SweepError(f"invalid sweep agent JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise SweepError("sweep agent JSON overrides must be an object")
    return _flatten(parsed)


def _parse_scalar(raw: str) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def _key(key: str, token: str) -> str:
    if _KEY_RE.fullmatch(key) is None:
        raise SweepError(f"invalid override {token!r}")
    return key


if __name__ == "__main__":
    raise SystemExit(main())
