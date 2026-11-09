"""Defaults and ranges for SmolVLA and OpenVLA recipe hyperparameters.

Schema validation accepts extra keys. These checks run in the family, before
any base weights are loaded. BF16 is a family parameter (``precision``).
The shared recipe ``precision`` field is unchanged; OHH-524 will unify it.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Mapping

from ..recipe_schema import RecipeConfigError

_UNSET = object()
_MANUAL = frozenset({"train_steps", "num_epochs", "seed"})
_TRAIN_STEPS_MAX = 1_000_000
_DEFAULT_STEPS = 200
_MODULE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")
_MODULE_RANGE = "all-linear, or 1 to 32 names (letters, digits, underscore; each at most 64 characters)"
_FAMILIES = frozenset({"smolvla", "openvla"})


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
        if self.kind == "modules":
            return _parse_modules(self, value)
        if self.kind == "bool":
            if not isinstance(value, bool):
                raise RecipeConfigError(
                    f"hyperparameters.{self.name} must be {self.doc_range}, got {value!r}"
                )
            return value
        if not _matches_kind(self.kind, value):
            raise RecipeConfigError(
                f"hyperparameters.{self.name} must be {self.doc_range}, got {value!r}"
            )
        if self.kind == "enum":
            if value not in self.choices:
                raise RecipeConfigError(
                    f"hyperparameters.{self.name} must be {self.doc_range}, got {value!r}"
                )
            return value
        if self.kind == "int":
            number = int(value)
            if (self.lo is not None and number < self.lo) or (
                self.hi is not None and number > self.hi
            ):
                raise RecipeConfigError(
                    f"hyperparameters.{self.name} must be {self.doc_range}, got {value!r}"
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


def _parse_modules(spec: ParamSpec, value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        text = value.strip()
        parts = (
            tuple(part.strip() for part in text.split(","))
            if "," in text
            else ((text,) if text else ())
        )
    elif isinstance(value, (list, tuple)):
        if any(not isinstance(item, str) for item in value):
            raise RecipeConfigError(
                f"hyperparameters.{spec.name} must be {spec.doc_range}, got {value!r}"
            )
        parts = tuple(item.strip() for item in value)
    else:
        raise RecipeConfigError(
            f"hyperparameters.{spec.name} must be {spec.doc_range}, got {value!r}"
        )
    if not parts or any(part == "" for part in parts):
        raise RecipeConfigError(
            f"hyperparameters.{spec.name} must be {spec.doc_range}, got {value!r}"
        )
    if "all-linear" in parts and parts != ("all-linear",):
        raise RecipeConfigError(
            "hyperparameters.lora_target_modules cannot combine all-linear with "
            f"other names, got {value!r}"
        )
    if parts == ("all-linear",):
        return parts
    if len(parts) > 32:
        raise RecipeConfigError(
            f"hyperparameters.{spec.name} must be {spec.doc_range}, got {value!r}"
        )
    if len(set(parts)) != len(parts):
        raise RecipeConfigError(
            f"hyperparameters.{spec.name} contains a duplicate name, got {value!r}"
        )
    for name in parts:
        if _MODULE.fullmatch(name) is None:
            raise RecipeConfigError(
                f"hyperparameters.{spec.name} must be {spec.doc_range}, got {value!r}"
            )
    return parts


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


def _family_specs(family: str) -> list[ParamSpec]:
    if family == "smolvla":
        batch, batch_doc = 8, "8"
        lr, lr_doc = 1e-4, "0.0001"
        tune, tune_doc = "full", "full"
    else:
        batch, batch_doc = 16, "16"
        lr, lr_doc = 5e-4, "0.0005"
        tune, tune_doc = "lora", "lora"
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
            batch_doc,
            "integer in [1, 256]",
            default=batch,
            lo=1,
            hi=256,
        ),
        _spec(
            "grad_accumulation_steps",
            family,
            "int",
            "1",
            "integer in [1, 1024]",
            default=1,
            lo=1,
            hi=1024,
        ),
        _spec(
            "lr",
            family,
            "float",
            lr_doc,
            "float in (0, 1]",
            default=lr,
            lo=0.0,
            hi=1.0,
            lo_exclusive=True,
        ),
        _spec(
            "precision",
            family,
            "enum",
            "bf16",
            "one of bf16, fp16, fp32",
            default="bf16",
            choices=("bf16", "fp16", "fp32"),
        ),
        _spec(
            "gradient_checkpointing",
            family,
            "bool",
            "true",
            "boolean",
            default=True,
        ),
        _spec(
            "tune_mode",
            family,
            "enum",
            tune_doc,
            "one of lora, full",
            default=tune,
            choices=("lora", "full"),
        ),
        _spec(
            "lora_rank",
            family,
            "int",
            "32",
            "integer in [1, 256]",
            default=32,
            lo=1,
            hi=256,
        ),
        _spec(
            "lora_alpha",
            family,
            "int",
            "16",
            "integer in [1, 1024]",
            default=16,
            lo=1,
            hi=1024,
        ),
        _spec(
            "lora_dropout",
            family,
            "float",
            "0.0",
            "float in [0, 0.5]",
            default=0.0,
            lo=0.0,
            hi=0.5,
        ),
        _spec(
            "lora_target_modules",
            family,
            "modules",
            "all-linear",
            _MODULE_RANGE,
            default=("all-linear",),
        ),
        _spec(
            "warmup_steps",
            family,
            "int",
            "0",
            "integer in [0, 1000000]",
            default=0,
            lo=0,
            hi=_TRAIN_STEPS_MAX,
        ),
        _spec(
            "max_grad_norm",
            family,
            "float",
            "1.0",
            "float in (0, 100]",
            default=1.0,
            lo=0.0,
            hi=100.0,
            lo_exclusive=True,
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


SPECS: tuple[ParamSpec, ...] = tuple(
    _family_specs("smolvla") + _family_specs("openvla")
)


def documented_parameter_lines() -> list[str]:
    """Markdown table rows for every tunable, including default and range."""
    return [
        f"| `{spec.name}` | {spec.doc_default} | {spec.doc_range} |" for spec in SPECS
    ]


def recipe_hyperparameters(recipe: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the hyperparameters mapping. A missing key is an empty mapping."""
    if not isinstance(recipe, Mapping):
        raise RecipeConfigError("recipe must be a mapping")
    value = recipe.get("hyperparameters")
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise RecipeConfigError("hyperparameters must be a mapping")
    return value


def resolve_hyperparameters(family: str, raw: Mapping[str, Any]) -> dict[str, Any]:
    """Fill defaults and reject values outside the family range.

    ``train_steps`` wins when both ``train_steps`` and ``num_epochs`` are set.
    When only ``num_epochs`` is set, the step count is ``num_epochs * 2``.
    When neither is set, the step count is 200.

    ``precision`` here is the family compute dtype. It defaults to ``bf16``.
    This function does not read the shared recipe ``precision`` field.
    """
    if family not in _FAMILIES:
        raise RecipeConfigError(f"no VLA parameters for family {family!r}")
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
        resolved["train_steps"] = _DEFAULT_STEPS
    return resolved
