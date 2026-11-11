"""Defaults and ranges for ACT and Diffusion Policy recipe hyperparameters.

Schema validation accepts extra keys. These checks run in the family, before
the dataset is opened and before ``train_start``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

from ..recipe_schema import RecipeConfigError

_UNSET = object()
_MANUAL = frozenset(
    {
        "train_steps",
        "num_epochs",
        "seed",
        "chunk_size",
        "horizon",
        "num_inference_steps",
    }
)
_TRAIN_STEPS_MAX = 1_000_000
_ACT_CHUNK = 10
_DIFF_CHUNK = 16


@dataclass(frozen=True)
class ParamSpec:
    """One tunable as it is documented and checked."""

    name: str
    family: str
    kind: str
    doc_default: str
    doc_range: str
    default: Any = _UNSET
    lo: float | None = None
    hi: float | None = None
    lo_exclusive: bool = False
    choices: tuple[str, ...] = ()
    fill: bool = True

    def parse(self, value: Any) -> Any:
        """Return ``value`` checked against this spec, or raise."""
        if not _matches_kind(self.kind, value):
            raise RecipeConfigError(
                f"hyperparameters.{self.name} must be {self.doc_range}, got {value!r}"
            )
        if self.kind == "enum":
            if value not in self.choices:
                raise RecipeConfigError(
                    f"hyperparameters.{self.name} must be {self.doc_range}, "
                    f"got {value!r}"
                )
            return value
        if self.kind == "int":
            number = int(value)
            if (self.lo is not None and number < self.lo) or (
                self.hi is not None and number > self.hi
            ):
                raise RecipeConfigError(
                    f"hyperparameters.{self.name} must be {self.doc_range}, "
                    f"got {value!r}"
                )
            return number
        number_f = float(value)
        if not math.isfinite(number_f):
            raise RecipeConfigError(
                f"hyperparameters.{self.name} must be {self.doc_range}, got {value!r}"
            )
        if self.lo is not None:
            too_low = number_f <= self.lo if self.lo_exclusive else number_f < self.lo
            if too_low or (self.hi is not None and number_f > self.hi):
                raise RecipeConfigError(
                    f"hyperparameters.{self.name} must be {self.doc_range}, "
                    f"got {value!r}"
                )
        elif self.hi is not None and number_f > self.hi:
            raise RecipeConfigError(
                f"hyperparameters.{self.name} must be {self.doc_range}, got {value!r}"
            )
        return number_f


def _matches_kind(kind: str, value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if kind == "int":
        return isinstance(value, int)
    if kind == "float":
        return isinstance(value, (int, float))
    if kind == "enum":
        return isinstance(value, str)
    return False


def _spec(
    name: str,
    family: str,
    kind: str,
    doc_default: str,
    doc_range: str,
    *,
    default: Any = _UNSET,
    lo: float | None = None,
    hi: float | None = None,
    lo_exclusive: bool = False,
    choices: tuple[str, ...] = (),
    fill: bool = True,
) -> ParamSpec:
    return ParamSpec(
        name=name,
        family=family,
        kind=kind,
        doc_default=doc_default,
        doc_range=doc_range,
        default=default,
        lo=lo,
        hi=hi,
        lo_exclusive=lo_exclusive,
        choices=choices,
        fill=fill,
    )


def _shared(family: str) -> list[ParamSpec]:
    return [
        _spec(
            "train_steps",
            family,
            "int",
            "200",
            "integer in [1, 1000000]",
            lo=1,
            hi=_TRAIN_STEPS_MAX,
            fill=False,
        ),
        _spec(
            "num_epochs",
            family,
            "int",
            "unset",
            "integer in [1, 1000000]; if train_steps is omitted, steps = num_epochs * 2",
            lo=1,
            hi=_TRAIN_STEPS_MAX,
            fill=False,
        ),
        _spec(
            "batch_size",
            family,
            "int",
            "8",
            "integer in [1, 1024]",
            default=8,
            lo=1,
            hi=1024,
        ),
        _spec(
            "lr",
            family,
            "float",
            "0.0001",
            "float in (0, 1]",
            default=0.0001,
            lo=0.0,
            hi=1.0,
            lo_exclusive=True,
        ),
        _spec(
            "vision_backbone",
            family,
            "enum",
            "none",
            "one of none, resnet18, resnet34",
            default="none",
            choices=("none", "resnet18", "resnet34"),
        ),
        _spec(
            "val_every",
            family,
            "int",
            "50",
            "integer in [1, 1000000]",
            default=50,
            lo=1,
            hi=_TRAIN_STEPS_MAX,
        ),
        _spec(
            "val_fraction",
            family,
            "float",
            "0.2",
            "float in (0, 0.5]",
            default=0.2,
            lo=0.0,
            hi=0.5,
            lo_exclusive=True,
        ),
        _spec(
            "early_stop_patience",
            family,
            "int",
            "0",
            "integer in [0, 100000]; 0 disables early stopping",
            default=0,
            lo=0,
            hi=100_000,
        ),
        _spec(
            "early_stop_min_delta",
            family,
            "float",
            "0.0",
            "float in [0, 1e12]",
            default=0.0,
            lo=0.0,
            hi=1e12,
        ),
        _spec(
            "seed",
            family,
            "int",
            "unset",
            "integer in [-2147483648, 2147483647]",
            lo=-2_147_483_648,
            hi=2_147_483_647,
            fill=False,
        ),
        _spec(
            "log_every",
            family,
            "int",
            "25",
            "integer in [1, 1000000]",
            default=25,
            lo=1,
            hi=_TRAIN_STEPS_MAX,
        ),
    ]


def _act_specs() -> list[ParamSpec]:
    return [
        *_shared("act"),
        _spec(
            "chunk_size",
            "act",
            "int",
            "10",
            "integer in [1, 1024]",
            lo=1,
            hi=1024,
            fill=False,
        ),
        _spec(
            "d_model",
            "act",
            "int",
            "64",
            "integer in [1, 1024], and divisible by nhead",
            default=64,
            lo=1,
            hi=1024,
        ),
        _spec(
            "nhead",
            "act",
            "int",
            "2",
            "integer in [1, 16]",
            default=2,
            lo=1,
            hi=16,
        ),
        _spec(
            "num_encoder_layers",
            "act",
            "int",
            "1",
            "integer in [1, 12]",
            default=1,
            lo=1,
            hi=12,
        ),
        _spec(
            "num_decoder_layers",
            "act",
            "int",
            "1",
            "integer in [1, 12]",
            default=1,
            lo=1,
            hi=12,
        ),
        _spec(
            "dim_feedforward",
            "act",
            "int",
            "128",
            "integer in [1, 4096]",
            default=128,
            lo=1,
            hi=4096,
        ),
        _spec(
            "latent_dim",
            "act",
            "int",
            "16",
            "integer in [1, 256]",
            default=16,
            lo=1,
            hi=256,
        ),
        _spec(
            "kl_weight",
            "act",
            "float",
            "10.0",
            "float in [0, 1000000]",
            default=10.0,
            lo=0.0,
            hi=1_000_000.0,
        ),
    ]


def _diffusion_specs() -> list[ParamSpec]:
    return [
        *_shared("diffusion"),
        _spec(
            "chunk_size",
            "diffusion",
            "int",
            "16",
            "integer in [1, 1024]",
            lo=1,
            hi=1024,
            fill=False,
        ),
        _spec(
            "horizon",
            "diffusion",
            "int",
            "16",
            "integer in [1, 1024]; alias of chunk_size; both must match when set",
            lo=1,
            hi=1024,
            fill=False,
        ),
        _spec(
            "diffusion_steps",
            "diffusion",
            "int",
            "100",
            "integer in [1, 1000]",
            default=100,
            lo=1,
            hi=1000,
        ),
        _spec(
            "scheduler",
            "diffusion",
            "enum",
            "ddpm",
            "one of ddpm, ddim",
            default="ddpm",
            choices=("ddpm", "ddim"),
        ),
        _spec(
            "num_inference_steps",
            "diffusion",
            "int",
            "10",
            "integer in [1, diffusion_steps]; omitted values use min(10, diffusion_steps)",
            lo=1,
            hi=1000,
            fill=False,
        ),
        _spec(
            "d_model",
            "diffusion",
            "int",
            "64",
            "integer in [8, 1024]",
            default=64,
            lo=8,
            hi=1024,
        ),
    ]


SPECS: tuple[ParamSpec, ...] = tuple(_act_specs() + _diffusion_specs())


def documented_parameter_lines() -> list[str]:
    """Markdown table rows for every tunable, including default and range."""
    return [
        f"| `{spec.name}` | {spec.doc_default} | {spec.doc_range} |" for spec in SPECS
    ]


def resolve_hyperparameters(family: str, raw: Mapping[str, Any]) -> dict[str, Any]:
    """Fill defaults and reject values outside the family range.

    ``train_steps`` wins when both ``train_steps`` and ``num_epochs`` are set.
    When only ``num_epochs`` is set, the step count is ``num_epochs * 2``.
    When neither is set, the step count is 200.
    """
    if family not in {"act", "diffusion"}:
        raise RecipeConfigError(
            f"no imitation-learning parameters for family {family!r}"
        )
    if not isinstance(raw, Mapping):
        raise RecipeConfigError("hyperparameters must be a mapping")
    if "mock" in raw:
        raise RecipeConfigError(
            "hyperparameters.mock is not accepted; ohho train does not run a fake trainer"
        )
    specs = [spec for spec in SPECS if spec.family == family]
    by_name = {spec.name: spec for spec in specs}
    unknown = sorted(set(raw) - set(by_name))
    if unknown:
        raise RecipeConfigError(
            f"unknown hyperparameters for family {family}: {', '.join(unknown)}"
        )
    resolved: dict[str, Any] = {}
    for spec in specs:
        if spec.name in _MANUAL:
            continue
        if spec.name in raw:
            resolved[spec.name] = spec.parse(raw[spec.name])
        elif spec.fill:
            resolved[spec.name] = spec.default
    if "seed" in raw:
        resolved["seed"] = by_name["seed"].parse(raw["seed"])
    if "num_epochs" in raw:
        resolved["num_epochs"] = by_name["num_epochs"].parse(raw["num_epochs"])
    if "train_steps" in raw:
        resolved["train_steps"] = by_name["train_steps"].parse(raw["train_steps"])
    elif "num_epochs" in resolved:
        steps = int(resolved["num_epochs"]) * 2
        if steps > _TRAIN_STEPS_MAX:
            raise RecipeConfigError(
                "hyperparameters.num_epochs implies train_steps "
                f"{steps}, which is outside [1, 1000000]"
            )
        resolved["train_steps"] = steps
    else:
        resolved["train_steps"] = 200
    _resolve_chunk(family, raw, by_name, resolved)
    if family == "act":
        d_model = int(resolved["d_model"])
        nhead = int(resolved["nhead"])
        if d_model % nhead != 0:
            raise RecipeConfigError(
                f"hyperparameters.d_model ({d_model}) must be divisible by "
                f"nhead ({nhead})"
            )
    if family == "diffusion":
        _resolve_inference_steps(raw, by_name, resolved)
    return resolved


def _resolve_chunk(
    family: str,
    raw: Mapping[str, Any],
    by_name: dict[str, ParamSpec],
    resolved: dict[str, Any],
) -> None:
    if family == "act":
        if "chunk_size" in raw:
            resolved["chunk_size"] = by_name["chunk_size"].parse(raw["chunk_size"])
        else:
            resolved["chunk_size"] = _ACT_CHUNK
        return
    has_chunk = "chunk_size" in raw
    has_horizon = "horizon" in raw
    if has_chunk and has_horizon:
        chunk = by_name["chunk_size"].parse(raw["chunk_size"])
        horizon = by_name["horizon"].parse(raw["horizon"])
        if chunk != horizon:
            raise RecipeConfigError(
                f"hyperparameters.horizon ({horizon}) and chunk_size ({chunk}) "
                "must match"
            )
        resolved["chunk_size"] = chunk
        resolved["horizon"] = horizon
    elif has_horizon:
        horizon = by_name["horizon"].parse(raw["horizon"])
        resolved["horizon"] = horizon
        resolved["chunk_size"] = horizon
    elif has_chunk:
        chunk = by_name["chunk_size"].parse(raw["chunk_size"])
        resolved["chunk_size"] = chunk
        resolved["horizon"] = chunk
    else:
        resolved["chunk_size"] = _DIFF_CHUNK
        resolved["horizon"] = _DIFF_CHUNK


def _resolve_inference_steps(
    raw: Mapping[str, Any],
    by_name: dict[str, ParamSpec],
    resolved: dict[str, Any],
) -> None:
    diffusion_steps = int(resolved["diffusion_steps"])
    if "num_inference_steps" in raw:
        value = int(by_name["num_inference_steps"].parse(raw["num_inference_steps"]))
        if value > diffusion_steps:
            raise RecipeConfigError(
                "hyperparameters.num_inference_steps must be "
                f"integer in [1, diffusion_steps], got {value} "
                f"(diffusion_steps is {diffusion_steps})"
            )
        resolved["num_inference_steps"] = value
        return
    resolved["num_inference_steps"] = min(10, diffusion_steps)
