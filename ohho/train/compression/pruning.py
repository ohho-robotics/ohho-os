"""Magnitude, structured, and 2:4 pruning for ``ohho train``.

The recipe block is optional. Omit it and the run stays dense. When it is
set, :class:`PruningSession` applies one method on one schedule:

* ``unstructured`` zeroes the smallest-magnitude weights. The module shapes
  stay the same.
* ``structured_channel`` drops output channels of a ``Linear`` or ``Conv2d``
  and the matching input channels of the next layer in the same parent, then
  rebuilds those modules so the checkpoint has fewer parameters.
* ``structured_head`` drops attention heads and rebuilds each
  ``MultiheadAttention`` so the projection matrices are smaller. The residual
  width stays ``embed_dim``.
* ``nm_2_4`` keeps the two largest-magnitude weights in every group of four
  along the last dimension. ``target_sparsity`` is 0.5. The export under
  ``pruning_export/nm24`` is what a TensorRT sparse engine build consumes.
  Building that engine needs a CUDA GPU and TensorRT, which this build does
  not do.

Schedules, from Zhu and Gupta for the cubic one:

* ``one_shot`` trains dense, then prunes once to the target.
* ``gradual`` raises sparsity with
  ``s + (s0 - s) * (1 - (step - start) / (end - start)) ** 3``.
* ``iterative`` prunes to the next fraction of the target, fine-tunes, and
  repeats.

``target_sparsity``, ``schedule``, ``include``, and ``exclude`` are ordinary
recipe fields, so ``--set`` can sweep them. Patterns are module names
(``encoder.*``, ``action_head``). An empty ``include`` selects every module
the method can prune. ``exclude`` wins.
"""

from __future__ import annotations

import importlib.util
import json
import math
import struct
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from ..recipe_schema import RecipeConfigError

METHODS: tuple[str, ...] = (
    "unstructured",
    "structured_channel",
    "structured_head",
    "nm_2_4",
)
SCHEDULES: tuple[str, ...] = ("one_shot", "gradual", "iterative")
NM_SCHEMA = "ohho.pruning.2to4/v1"
EXPORT_DIR_NAME = "pruning_export"

_FIELDS = frozenset(
    {
        "method",
        "schedule",
        "target_sparsity",
        "include",
        "exclude",
        "start_step",
        "end_step",
        "iterations",
        "finetune_steps",
    }
)


class PruningCapabilityError(RecipeConfigError):
    """A TensorRT sparse engine cannot be built in this process.

    The 2:4 export is still written. Building the engine needs a CUDA GPU
    and TensorRT, and this build does not run that build.
    """


def cubic_sparsity(
    step: int, start: int, end: int, initial: float, final: float
) -> float:
    """Sparsity at ``step`` on the cubic gradual schedule.

    At ``start`` the value is ``initial``. At ``end`` it is ``final``.
    Between them the remaining gap shrinks with the cube of the time left,
    which is the Zhu and Gupta schedule. ``end <= start`` jumps to
    ``final`` once ``step`` reaches ``end``.
    """
    if end <= start:
        return final if step >= end else initial
    if step <= start:
        return initial
    if step >= end:
        return final
    progress = (step - start) / (end - start)
    return final + (initial - final) * (1.0 - progress) ** 3


def normalize_pruning(
    block: Mapping[str, Any], *, train_steps: int | None = None
) -> dict[str, Any]:
    """Return a recipe pruning block with defaults filled in.

    Raises :class:`RecipeConfigError` when the block cannot be run.
    ``train_steps`` checks that an iterative schedule has a fine-tune step
    for every iteration. Omit it to check only the block itself.
    """
    if not isinstance(block, Mapping):
        raise RecipeConfigError("pruning must be a mapping")
    unknown = sorted(set(block) - _FIELDS)
    if unknown:
        listed = ", ".join(unknown)
        raise RecipeConfigError(f"unknown pruning fields: {listed}")
    method = block.get("method")
    schedule = block.get("schedule")
    if method not in METHODS:
        listed = ", ".join(METHODS)
        raise RecipeConfigError(f"pruning.method {method!r} is not one of {listed}")
    if schedule not in SCHEDULES:
        listed = ", ".join(SCHEDULES)
        raise RecipeConfigError(f"pruning.schedule {schedule!r} is not one of {listed}")
    target = _finite_number(block.get("target_sparsity"), "target_sparsity")
    if target < 0 or target >= 1:
        raise RecipeConfigError("pruning.target_sparsity must be >= 0 and < 1")
    if method == "nm_2_4" and abs(target - 0.5) > 1e-9:
        raise RecipeConfigError(
            "pruning.method nm_2_4 keeps 2 of every 4 weights, so "
            "target_sparsity must be 0.5"
        )
    include = _string_list(block.get("include"), "include")
    exclude = _string_list(block.get("exclude"), "exclude")
    start = _optional_step(block.get("start_step"), "start_step")
    end = _optional_step(block.get("end_step"), "end_step")
    iterations = _optional_step(block.get("iterations"), "iterations")
    finetune = _optional_step(block.get("finetune_steps"), "finetune_steps")
    if schedule == "gradual":
        if iterations is not None or finetune is not None:
            raise RecipeConfigError(
                "pruning.iterations and pruning.finetune_steps apply only "
                "to schedule iterative"
            )
        if start is None:
            start = 0
        if end is not None and end < start:
            raise RecipeConfigError("pruning.end_step must be >= pruning.start_step")
    elif schedule == "iterative":
        if start is not None or end is not None:
            raise RecipeConfigError(
                "pruning.start_step and pruning.end_step apply only to schedule gradual"
            )
        if iterations is None:
            iterations = 3
        if iterations < 1:
            raise RecipeConfigError("pruning.iterations must be >= 1")
        if finetune is not None and finetune < 1:
            raise RecipeConfigError("pruning.finetune_steps must be >= 1")
        if train_steps is not None:
            _check_budget(iterations, finetune, train_steps)
    elif any(value is not None for value in (start, end, iterations, finetune)):
        raise RecipeConfigError(
            "one_shot pruning uses method, target_sparsity, include, and "
            "exclude. start_step, end_step, iterations, and finetune_steps "
            "belong to gradual or iterative"
        )
    config: dict[str, Any] = {
        "method": method,
        "schedule": schedule,
        "target_sparsity": target,
        "include": include,
        "exclude": exclude,
    }
    if start is not None:
        config["start_step"] = start
    if end is not None:
        config["end_step"] = end
    if iterations is not None:
        config["iterations"] = iterations
    if finetune is not None:
        config["finetune_steps"] = finetune
    return config


def optimizer_steps(config: Mapping[str, Any], train_steps: int) -> int:
    """Optimizer steps a normalized pruning block will run.

    Iterative pruning with ``finetune_steps`` uses
    ``iterations * finetune_steps``. Every other schedule uses
    ``train_steps``.
    """
    if train_steps < 1:
        raise RecipeConfigError("pruning needs train_steps >= 1")
    if (
        config.get("schedule") == "iterative"
        and config.get("finetune_steps") is not None
    ):
        return int(config["iterations"]) * int(config["finetune_steps"])
    return int(train_steps)


def pending_report(block: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Manifest pruning object written at ``train_start``, before metrics exist."""
    if block is None:
        return None
    if not isinstance(block, Mapping):
        return None
    config = normalize_pruning(block)
    return _blank_report(config)


def build_tensorrt_sparse_engine(
    export_dir: str | Path,
    *,
    builder: Callable[[Path], str] | None = None,
    cuda_available: bool | None = None,
    tensorrt_available: bool | None = None,
) -> str:
    """Build a TensorRT engine from a 2:4 export, or raise.

    ``builder`` is the GPU follow-up hook. It receives the export directory
    and returns the engine path. Without it, this function checks for CUDA
    and TensorRT and raises :class:`PruningCapabilityError`. No engine file
    is written.
    """
    directory = Path(export_dir)
    if builder is not None:
        result = builder(directory)
        if not isinstance(result, str) or not result.strip():
            raise PruningCapabilityError(
                "TensorRT sparse engine builder returned an empty path. "
                "No engine was written."
            )
        return result
    reason = _capability_reason(cuda_available, tensorrt_available)
    if reason:
        raise PruningCapabilityError(
            "TensorRT sparse engine build needs a CUDA GPU and TensorRT. "
            f"{reason} The 2:4 export is the input. No engine was written."
        )
    manifest = _read_nm_manifest(directory)
    if not manifest.get("complete"):
        raise PruningCapabilityError(
            "TensorRT sparse engine build needs a complete 2:4 pattern on "
            "every exported tensor. This export is partial. No engine was "
            "written."
        )
    raise PruningCapabilityError(
        "TensorRT sparse engine build is a GPU follow-up and does not run "
        "in this build, even when CUDA and TensorRT are installed. The 2:4 "
        "export is the input. No engine was written."
    )


def load_2to4_tensor(export_dir: str | Path, name: str) -> Any:
    """Rebuild the dense tensor ``name`` from an ``ohho.pruning.2to4/v1`` export.

    ``export_dir`` may be the checkpoint directory, ``pruning_export``, or
    the ``nm24`` directory itself. Values are float32. Indexes in each
    bitmask byte are ascending positions in the group of four.
    """
    torch = _require_torch()
    manifest_path = _nm_manifest_path(Path(export_dir))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tensors = manifest.get("tensors")
    if not isinstance(tensors, list):
        raise RecipeConfigError("2:4 export is missing tensors")
    match = None
    for item in tensors:
        if isinstance(item, dict) and item.get("name") == name:
            match = item
            break
    if match is None:
        raise RecipeConfigError(f"2:4 export has no tensor {name!r}")
    if not match.get("packed"):
        raise RecipeConfigError(f"2:4 tensor {name!r} is not a complete group packing")
    shape = tuple(int(dim) for dim in match["shape"])
    root = manifest_path.parent
    values = (root / str(match["values"])).read_bytes()
    bitmask = (root / str(match["bitmask"])).read_bytes()
    return _unpack_2to4(torch, shape, values, bitmask)


class PruningSession:
    """Apply one pruning block around a step loop, then write the export.

    Call :meth:`before_step` before the forward and :meth:`after_step` after
    the optimizer. :meth:`finish` applies a one-shot prune, rebuilds modules
    for structured methods, and returns the manifest report.
    """

    def __init__(
        self,
        model: Any,
        pruning: Mapping[str, Any],
        *,
        train_steps: int,
        export_dir: str | Path,
    ) -> None:
        self._torch = _require_torch()
        self.model = model
        self.config = normalize_pruning(pruning, train_steps=train_steps)
        self.method = str(self.config["method"])
        self.optimizer_steps = optimizer_steps(self.config, train_steps)
        self.export_root = Path(export_dir) / EXPORT_DIR_NAME
        self._masks: dict[str, Any] = {}
        self._chains: list[_Chain] = []
        self._heads: list[_Head] = []
        self._weights: list[_Weight] = []
        self._skips: list[dict[str, str]] = []
        self._patterned: dict[str, set[int]] = {}
        self._group_keeps: dict[str, dict[int, tuple[int, int]]] = {}
        self._spans: list[tuple[int, int, float]] = []
        self._discover()
        self._params_before = _count_params(model)
        self._build_spans()
        self.export_root.mkdir(parents=True, exist_ok=True)
        self._before_path = self.export_root / "before.pt"
        self._torch.save(model.state_dict(), self._before_path)
        self._bytes_before = self._before_path.stat().st_size

    def before_step(self, step: int) -> None:
        """Prune at the start of an iterative fine-tune cycle."""
        if self.config["schedule"] != "iterative":
            return
        for start, _end, level in self._spans:
            if step == start:
                self._apply_level(level, fresh=False)
                return

    def after_step(self, step: int) -> None:
        """Update a gradual mask, and keep already-pruned weights at zero."""
        if self.config["schedule"] == "gradual":
            self._apply_level(self._schedule_value(step), fresh=False)
        elif self.config["schedule"] == "iterative":
            self._reapply()

    def finish(self) -> dict[str, Any]:
        """Prune one-shot runs, shrink structured modules, and return the report."""
        if self.config["schedule"] == "one_shot":
            self._apply_level(self._final_level(), fresh=True)
        else:
            last = self.optimizer_steps - 1
            self._apply_level(self._schedule_value(last), fresh=False)
        if self.method in {"structured_channel", "structured_head"}:
            self._shrink()
        report = self._report()
        after_path = self.export_root / "after.pt"
        self._torch.save(self.model.state_dict(), after_path)
        report["export_bytes_before"] = self._bytes_before
        report["export_bytes_after"] = after_path.stat().st_size
        report["export_dir"] = EXPORT_DIR_NAME
        if self.method == "nm_2_4":
            self._write_nm_export(report)
            report["tensorrt_error"] = _caught_engine_error(self.export_root)
            report["tensorrt_engine"] = None
        else:
            report["export_format"] = "dense"
            report["tensorrt_engine"] = None
            report["tensorrt_error"] = None
        _write_json(self.export_root / "report.json", report)
        return report

    def _discover(self) -> None:
        if self.method == "structured_channel":
            self._discover_channels()
        elif self.method == "structured_head":
            self._discover_heads()
        else:
            self._discover_weights()
        if self._chains or self._heads or self._weights:
            return
        detail = "; ".join(
            f"{item['name']}: {item['reason']}" for item in self._skips[:6]
        )
        message = f"pruning matched no layers for method {self.method!r}"
        if detail:
            message = f"{message}. {detail}"
        raise RecipeConfigError(message)

    def _discover_channels(self) -> None:
        torch = self._torch
        linear = torch.nn.Linear
        conv = torch.nn.Conv2d
        for parent_name, parent in self.model.named_modules():
            for kind, cls in (("linear", linear), ("conv2d", conv)):
                kids: list[tuple[str, Any]] = []
                for child_name, child in parent.named_children():
                    if not isinstance(child, cls):
                        continue
                    qname = _join(parent_name, child_name)
                    if isinstance(child, conv) and int(child.groups) != 1:
                        if _selected(qname, self.config):
                            self._skips.append(
                                {
                                    "name": qname,
                                    "reason": ("grouped convolution is left dense"),
                                }
                            )
                        continue
                    if _under_attention(self.model, qname, torch):
                        continue
                    kids.append((qname, child))
                self._consume_chains(kids, kind)

    def _consume_chains(self, kids: list[tuple[str, Any]], kind: str) -> None:
        chains = _split_chains(kids)
        used: set[str] = set()
        for chain in chains:
            names = [name for name, _mod in chain]
            if not all(_selected(name, self.config) for name in names):
                for name in names:
                    if _selected(name, self.config):
                        self._skips.append(
                            {
                                "name": name,
                                "reason": (
                                    "left dense because the adjacent layer "
                                    "in the same parent is not included"
                                ),
                            }
                        )
                continue
            self._chains.append(_Chain(chain, kind))
            used.update(names)
        for name, _mod in kids:
            if name in used or not _selected(name, self.config):
                continue
            self._skips.append(
                {
                    "name": name,
                    "reason": (
                        "no adjacent Linear or Conv2d in the same parent "
                        "shares this channel width"
                    ),
                }
            )

    def _discover_heads(self) -> None:
        torch = self._torch
        for name, module in self.model.named_modules():
            if not isinstance(module, torch.nn.MultiheadAttention):
                continue
            if not name or not _selected(name, self.config):
                continue
            if module.in_proj_weight is None:
                self._skips.append(
                    {
                        "name": name,
                        "reason": (
                            "separate QKV projections are left dense in this build"
                        ),
                    }
                )
                continue
            heads = int(module.num_heads)
            embed = int(module.embed_dim)
            if heads < 2 or embed % heads != 0:
                self._skips.append(
                    {
                        "name": name,
                        "reason": (
                            "attention needs at least two heads whose width "
                            "divides embed_dim"
                        ),
                    }
                )
                continue
            self._heads.append(_Head(name, module))

    def _discover_weights(self) -> None:
        torch = self._torch
        linear = torch.nn.Linear
        conv = torch.nn.Conv2d
        attention = torch.nn.MultiheadAttention
        for name, module in self.model.named_modules():
            label = name or "model"
            if not _selected(label, self.config):
                continue
            if isinstance(module, attention):
                if module.in_proj_weight is None:
                    self._skips.append(
                        {
                            "name": label,
                            "reason": "separate QKV projections are left dense",
                        }
                    )
                    continue
                prefix = f"{name}." if name else ""
                self._weights.append(
                    _Weight(
                        label,
                        "attention",
                        f"{prefix}in_proj_weight",
                        module_name=name,
                    )
                )
                self._weights.append(
                    _Weight(
                        label,
                        "attention",
                        f"{prefix}out_proj.weight",
                        module_name=name,
                    )
                )
                continue
            if _under_attention(self.model, name, torch):
                continue
            if isinstance(module, (linear, conv)):
                weight = module.weight
                if int(weight.numel()) < 2:
                    self._skips.append(
                        {
                            "name": label,
                            "reason": "weight has fewer than 2 elements",
                        }
                    )
                    continue
                if self.method == "nm_2_4" and int(weight.shape[-1]) < 4:
                    self._skips.append(
                        {
                            "name": label,
                            "reason": "last dimension is shorter than 4",
                        }
                    )
                    continue
                kind = "conv2d" if isinstance(module, conv) else "linear"
                param = f"{name}.weight" if name else "weight"
                self._weights.append(_Weight(label, kind, param, module_name=name))

    def _build_spans(self) -> None:
        if self.config["schedule"] != "iterative":
            self._spans = []
            return
        iterations = int(self.config["iterations"])
        total = self.optimizer_steps
        base, rem = divmod(total, iterations)
        lengths = [base + (1 if index < rem else 0) for index in range(iterations)]
        final = self._final_level()
        spans: list[tuple[int, int, float]] = []
        cursor = 0
        for index, length in enumerate(lengths):
            level = final * ((index + 1) / iterations)
            spans.append((cursor, cursor + length, level))
            cursor += length
        self._spans = spans

    def _final_level(self) -> float:
        if self.method == "nm_2_4":
            return 1.0
        return float(self.config["target_sparsity"])

    def _schedule_value(self, step: int) -> float:
        schedule = self.config["schedule"]
        final = self._final_level()
        if schedule == "one_shot":
            return final
        if schedule == "iterative":
            for start, end, level in self._spans:
                if start <= step < end:
                    return level
            return self._spans[-1][2]
        start = int(self.config.get("start_step") or 0)
        end = self.config.get("end_step")
        if end is None:
            end = self.optimizer_steps - 1
        return cubic_sparsity(step, start, int(end), 0.0, final)

    def _apply_level(self, level: float, *, fresh: bool) -> None:
        if fresh:
            self._masks = {}
            self._patterned = {}
            self._group_keeps = {}
            for chain in self._chains:
                for bound in chain.boundaries:
                    bound.order = None
                    bound.keep = None
            for head in self._heads:
                head.order = None
                head.keep = None
            for weight in self._weights:
                weight.order = None
        if level <= 0:
            return
        if self.method == "structured_channel":
            self._mask_channels(level, fresh=fresh)
        elif self.method == "structured_head":
            self._mask_heads(level, fresh=fresh)
        elif self.method == "nm_2_4":
            self._mask_nm(level, fresh=fresh)
        else:
            self._mask_magnitude(level, fresh=fresh)
        self._reapply()

    def _mask_magnitude(self, sparsity: float, *, fresh: bool) -> None:
        del fresh
        torch = self._torch
        for weight in self._weights:
            param = _parameter(self.model, weight.param)
            if weight.order is None:
                flat = param.detach().abs().flatten().cpu()
                weight.order = torch.argsort(flat, descending=True)
            count = _keep_count(int(param.numel()), sparsity)
            mask = torch.zeros(int(param.numel()), dtype=torch.bool)
            mask[weight.order[:count]] = True
            self._store_mask(weight.param, mask.view_as(param).to(param.device))

    def _mask_channels(self, sparsity: float, *, fresh: bool) -> None:
        del fresh
        torch = self._torch
        for chain in self._chains:
            for index, bound in enumerate(chain.boundaries):
                producer = self.model.get_submodule(chain.names[index])
                consumer = self.model.get_submodule(chain.names[index + 1])
                if bound.order is None:
                    scores = _channel_scores(producer, consumer).cpu()
                    bound.order = torch.argsort(scores, descending=True)
                count = _keep_count(bound.n, sparsity)
                chosen = bound.order[:count]
                bound.keep = torch.sort(chosen).values
                self._zero_channels(producer, consumer, bound.keep, torch)

    def _zero_channels(
        self, producer: Any, consumer: Any, keep: Any, torch: Any
    ) -> None:
        del torch
        n = int(producer.weight.shape[0])
        drop_idx = _drop_index(keep, n, self._torch).to(producer.weight.device)
        if int(drop_idx.numel()) == 0:
            return
        producer.weight.data.index_fill_(0, drop_idx, 0.0)
        if producer.bias is not None:
            producer.bias.data.index_fill_(0, drop_idx, 0.0)
        consumer.weight.data.index_fill_(1, drop_idx, 0.0)

    def _mask_heads(self, sparsity: float, *, fresh: bool) -> None:
        del fresh
        torch = self._torch
        for head in self._heads:
            module = self.model.get_submodule(head.name)
            if head.order is None:
                scores = _head_scores(module, torch)
                head.order = torch.argsort(scores, descending=True)
            count = _keep_count(head.n, sparsity)
            chosen = head.order[:count]
            head.keep = [int(index) for index in torch.sort(chosen).values.tolist()]
            _zero_heads(module, head.keep, head.head_dim, torch)

    def _mask_nm(self, fraction: float, *, fresh: bool) -> None:
        torch = self._torch
        for weight in self._weights:
            param = _parameter(self.model, weight.param)
            grouped, groups = _group_view(param)
            if groups == 0:
                continue
            if fresh or weight.order is None:
                magnitude = grouped.detach().abs()
                smallest = magnitude.topk(2, dim=-1, largest=False).values.sum(-1)
                weight.order = torch.argsort(smallest.cpu())
                if fresh:
                    self._group_keeps[weight.param] = {}
                    self._patterned[weight.param] = set()
            total = int(weight.order.numel())
            want = int(math.floor(total * fraction + 0.5))
            want = min(total, max(0, want))
            chosen = {int(index) for index in weight.order[:want].tolist()}
            chosen |= self._patterned.get(weight.param, set())
            self._patterned[weight.param] = chosen
            saved = self._group_keeps.setdefault(weight.param, {})
            mask = torch.ones_like(param, dtype=torch.bool)
            usable = groups * 4
            flat_mask = mask[..., :usable].reshape(-1, 4)
            flat_value = param.detach()[..., :usable].reshape(-1, 4)
            for group in chosen:
                if group not in saved:
                    top = torch.topk(flat_value[group].abs(), 2).indices
                    pair = tuple(sorted(int(index) for index in top.tolist()))
                    saved[group] = (pair[0], pair[1])
                first, second = saved[group]
                flat_mask[group] = False
                flat_mask[group, first] = True
                flat_mask[group, second] = True
            self._store_mask(weight.param, mask)

    def _store_mask(self, name: str, mask: Any) -> None:
        previous = self._masks.get(name)
        if previous is None:
            self._masks[name] = mask
            return
        self._masks[name] = previous & mask.to(device=previous.device)

    def _reapply(self) -> None:
        if self.method == "structured_channel":
            torch = self._torch
            for chain in self._chains:
                for index, bound in enumerate(chain.boundaries):
                    if bound.keep is None:
                        continue
                    producer = self.model.get_submodule(chain.names[index])
                    consumer = self.model.get_submodule(chain.names[index + 1])
                    self._zero_channels(producer, consumer, bound.keep, torch)
            return
        if self.method == "structured_head":
            torch = self._torch
            for head in self._heads:
                if not head.keep:
                    continue
                module = self.model.get_submodule(head.name)
                if not isinstance(module, torch.nn.MultiheadAttention):
                    continue
                _zero_heads(module, head.keep, head.head_dim, torch)
            return
        for name, mask in self._masks.items():
            param = _parameter(self.model, name)
            param.data.mul_(mask.to(device=param.device, dtype=param.dtype))

    def _shrink(self) -> None:
        if self.method == "structured_channel":
            self._shrink_channels()
        else:
            self._shrink_heads()

    def _shrink_channels(self) -> None:
        torch = self._torch
        for chain in self._chains:
            keeps: list[Any | None] = []
            for bound in chain.boundaries:
                if bound.keep is None or int(bound.keep.numel()) >= bound.n:
                    keeps.append(None)
                else:
                    keeps.append(bound.keep)
            if all(item is None for item in keeps):
                continue
            built: list[tuple[str, Any]] = []
            in_idx = None
            for index, name in enumerate(chain.names):
                module = self.model.get_submodule(name)
                out_idx = keeps[index] if index < len(keeps) else None
                built.append((name, _shrink_module(module, in_idx, out_idx, torch)))
                in_idx = out_idx
            for name, module in built:
                _replace(self.model, name, module)

    def _shrink_heads(self) -> None:
        torch = self._torch
        for head in self._heads:
            if not head.keep or len(head.keep) >= head.n:
                continue
            module = self.model.get_submodule(head.name)
            if not isinstance(module, torch.nn.MultiheadAttention):
                continue
            _replace(
                self.model,
                head.name,
                _shrink_attention(module, head.keep, torch),
            )

    def _report(self) -> dict[str, Any]:
        if self.method == "structured_channel":
            layers = self._channel_rows()
        elif self.method == "structured_head":
            layers = self._head_rows()
        else:
            layers = self._weight_rows()
        layers.sort(key=lambda row: row["name"])
        removed = 0
        total = 0
        for row in layers:
            before = int(row.pop("_before"))
            after = int(row.pop("_after"))
            total += before
            removed += before - after
        achieved = _sparsity_ratio(removed, total) if total else 0.0
        report = _blank_report(self.config)
        report["layers"] = layers
        report["skipped"] = [dict(item) for item in self._skips]
        report["achieved_sparsity"] = achieved
        report["params_before"] = self._params_before
        report["params_after"] = _count_params(self.model)
        report["export_bytes_before"] = self._bytes_before
        report["export_format"] = NM_SCHEMA if self.method == "nm_2_4" else "dense"
        report["export_dir"] = EXPORT_DIR_NAME
        return report

    def _channel_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for chain in self._chains:
            in_keep = None
            for index, name in enumerate(chain.names):
                module = self.model.get_submodule(name)
                out_keep = None
                if index < len(chain.boundaries):
                    out_keep = chain.boundaries[index].keep
                before_w = chain.weight_before[name]
                after_w = int(module.weight.numel())
                rows.append(
                    _layer_row(
                        name,
                        chain.kind,
                        before_w,
                        after_w,
                        chain.params_before[name],
                        _count_params(module),
                        after_w,
                    )
                )
                in_keep = out_keep
            del in_keep
        return rows

    def _head_rows(self) -> list[dict[str, Any]]:
        torch = self._torch
        rows = []
        for head in self._heads:
            module = self.model.get_submodule(head.name)
            if isinstance(module, torch.nn.MultiheadAttention):
                weight_after = head.weight_before
            else:
                weight_after = int(
                    module.q_proj.weight.numel()
                    + module.k_proj.weight.numel()
                    + module.v_proj.weight.numel()
                    + module.out_proj.weight.numel()
                )
            rows.append(
                _layer_row(
                    head.name,
                    "attention",
                    head.weight_before,
                    weight_after,
                    head.params_before,
                    _count_params(module),
                    weight_after,
                )
            )
        return rows

    def _weight_rows(self) -> list[dict[str, Any]]:
        grouped: dict[str, list[_Weight]] = {}
        for weight in self._weights:
            grouped.setdefault(weight.layer, []).append(weight)
        rows = []
        for name, weights in grouped.items():
            before = 0
            nonzero = 0
            for weight in weights:
                param = _parameter(self.model, weight.param)
                before += int(param.numel())
                nonzero += int(self._torch.count_nonzero(param.data).item())
            module = self.model.get_submodule(weights[0].module_name)
            params = _count_params(module)
            rows.append(
                _layer_row(
                    name,
                    weights[0].kind,
                    before,
                    nonzero,
                    params,
                    params,
                    nonzero,
                )
            )
        return rows

    def _write_nm_export(self, report: dict[str, Any]) -> None:
        directory = self.export_root / "nm24"
        directory.mkdir(parents=True, exist_ok=True)
        tensors = []
        complete = bool(self._weights)
        for weight in self._weights:
            param = _parameter(self.model, weight.param)
            mask = self._masks.get(weight.param)
            packed = _pack_tensor(param, mask)
            entry: dict[str, Any] = {
                "name": weight.param,
                "shape": [int(dim) for dim in param.shape],
                "dtype": "float32",
                "groups": packed["groups"],
                "patterned_groups": packed["patterned"],
                "tail": packed["tail"],
                "packed": packed["complete"],
            }
            if packed["complete"]:
                stem = weight.param
                values_name = f"{stem}.values.f32"
                mask_name = f"{stem}.bitmask.u8"
                (directory / values_name).write_bytes(packed["values"])
                (directory / mask_name).write_bytes(packed["bitmask"])
                entry["values"] = values_name
                entry["bitmask"] = mask_name
            else:
                complete = False
            tensors.append(entry)
        if not tensors:
            complete = False
        manifest = {
            "schema": NM_SCHEMA,
            "pattern": "2:4",
            "layout": "groups_of_4_along_the_last_dimension",
            "bitmask": (
                "one byte per group; bits 0-1 and bits 2-3 are the kept "
                "indexes in ascending order"
            ),
            "values": (
                "float32 little-endian, two values per group, in the same "
                "index order as the bitmask"
            ),
            "complete": complete,
            "tensors": tensors,
            "tensorrt": {
                "engine": None,
                "note": (
                    "Building a TensorRT sparse engine needs a CUDA GPU and "
                    "TensorRT. This directory is the input. The build is a "
                    "GPU follow-up."
                ),
            },
        }
        _write_json(directory / "manifest.json", manifest)
        report["export_format"] = NM_SCHEMA


class _Chain:
    def __init__(self, modules: list[tuple[str, Any]], kind: str) -> None:
        self.names = [name for name, _mod in modules]
        self.kind = kind
        self.params_before = {name: _count_params(module) for name, module in modules}
        self.weight_before = {
            name: int(module.weight.numel()) for name, module in modules
        }
        self.boundaries: list[_Bound] = []
        for (_name, producer), (_next, _consumer) in zip(modules, modules[1:]):
            self.boundaries.append(_Bound(int(producer.weight.shape[0])))


class _Bound:
    def __init__(self, n: int) -> None:
        self.n = n
        self.order: Any = None
        self.keep: Any = None


class _Head:
    def __init__(self, name: str, module: Any) -> None:
        self.name = name
        self.n = int(module.num_heads)
        self.head_dim = int(module.embed_dim) // self.n
        self.params_before = _count_params(module)
        self.weight_before = int(module.in_proj_weight.numel()) + int(
            module.out_proj.weight.numel()
        )
        self.order: Any = None
        self.keep: list[int] | None = None


class _Weight:
    def __init__(
        self,
        layer: str,
        kind: str,
        param: str,
        module_name: str | None = None,
    ) -> None:
        self.layer = layer
        self.kind = kind
        self.param = param
        self.module_name = layer if module_name is None else module_name
        self.order: Any = None


def _attention_type(torch: Any) -> type:
    cached = getattr(_attention_type, "cls", None)
    if cached is not None:
        return cached

    class PrunedMultiheadAttention(torch.nn.Module):
        """Fewer heads than ``embed_dim / head_dim``, still emitting ``embed_dim``.

        PyTorch's ``MultiheadAttention`` cannot store a head count whose
        width is narrower than ``embed_dim``. This module is the shrunk
        replacement. ``in_proj_bias`` stays unset so ``TransformerEncoderLayer``
        takes its Python attention path.
        """

        def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            head_dim: int,
            *,
            batch_first: bool,
            dropout: float,
            bias: bool,
        ) -> None:
            super().__init__()
            self.embed_dim = embed_dim
            self.num_heads = num_heads
            self.head_dim = head_dim
            self.batch_first = batch_first
            self.dropout_p = dropout
            self.in_proj_bias = None
            self._qkv_same_embed_dim = True
            inner = num_heads * head_dim
            self.q_proj = torch.nn.Linear(embed_dim, inner, bias=bias)
            self.k_proj = torch.nn.Linear(embed_dim, inner, bias=bias)
            self.v_proj = torch.nn.Linear(embed_dim, inner, bias=bias)
            self.out_proj = torch.nn.Linear(inner, embed_dim, bias=bias)

        def forward(
            self,
            query: Any,
            key: Any,
            value: Any,
            key_padding_mask: Any = None,
            need_weights: bool = True,
            attn_mask: Any = None,
            average_attn_weights: bool = True,
            is_causal: bool = False,
        ) -> tuple[Any, None]:
            del need_weights, average_attn_weights
            if not self.batch_first:
                query = query.transpose(0, 1)
                key = key.transpose(0, 1)
                value = value.transpose(0, 1)
            batch, queries, _embed = query.shape
            keys = key.shape[1]
            heads = self.num_heads
            width = self.head_dim
            q = self.q_proj(query).view(batch, queries, heads, width).transpose(1, 2)
            k = self.k_proj(key).view(batch, keys, heads, width).transpose(1, 2)
            v = self.v_proj(value).view(batch, keys, heads, width).transpose(1, 2)
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(width)
            if is_causal:
                causal = torch.ones(
                    queries, keys, device=scores.device, dtype=torch.bool
                ).triu(diagonal=1)
                scores = scores.masked_fill(causal, float("-inf"))
            if attn_mask is not None:
                if attn_mask.dtype == torch.bool:
                    scores = scores.masked_fill(attn_mask, float("-inf"))
                else:
                    scores = scores + attn_mask
            if key_padding_mask is not None:
                pad = key_padding_mask
                if pad.dtype != torch.bool:
                    pad = pad != 0
                scores = scores.masked_fill(pad.view(batch, 1, 1, keys), float("-inf"))
            weights = torch.softmax(scores, dim=-1)
            if self.training and self.dropout_p:
                weights = torch.nn.functional.dropout(weights, self.dropout_p, True)
            out = torch.matmul(weights, v)
            out = out.transpose(1, 2).contiguous().view(batch, queries, heads * width)
            out = self.out_proj(out)
            if not self.batch_first:
                out = out.transpose(0, 1)
            return out, None

    _attention_type.cls = PrunedMultiheadAttention
    return PrunedMultiheadAttention


def _shrink_attention(module: Any, keep: Sequence[int], torch: Any) -> Any:
    embed = int(module.embed_dim)
    head_dim = embed // int(module.num_heads)
    q, k, v = module.in_proj_weight.detach().chunk(3, dim=0)
    qb = kb = vb = None
    if module.in_proj_bias is not None:
        qb, kb, vb = module.in_proj_bias.detach().chunk(3, dim=0)
    cls = _attention_type(torch)
    pruned = cls(
        embed,
        len(keep),
        head_dim,
        batch_first=bool(module.batch_first),
        dropout=float(module.dropout),
        bias=module.in_proj_bias is not None,
    )
    pruned = pruned.to(device=q.device, dtype=q.dtype)
    with torch.no_grad():
        pruned.q_proj.weight.copy_(_take_rows(q, keep, head_dim))
        pruned.k_proj.weight.copy_(_take_rows(k, keep, head_dim))
        pruned.v_proj.weight.copy_(_take_rows(v, keep, head_dim))
        pruned.out_proj.weight.copy_(
            _take_cols(module.out_proj.weight.detach(), keep, head_dim)
        )
        if qb is not None:
            pruned.q_proj.bias.copy_(_take_rows(qb, keep, head_dim))
            pruned.k_proj.bias.copy_(_take_rows(kb, keep, head_dim))
            pruned.v_proj.bias.copy_(_take_rows(vb, keep, head_dim))
        if module.out_proj.bias is not None:
            pruned.out_proj.bias.copy_(module.out_proj.bias.detach())
    pruned.train(module.training)
    return pruned


def _shrink_module(module: Any, in_idx: Any, out_idx: Any, torch: Any) -> Any:
    weight = module.weight.detach()
    if out_idx is not None:
        weight = weight[out_idx.to(weight.device)]
    if in_idx is not None:
        weight = weight[:, in_idx.to(weight.device)]
    bias = None if module.bias is None else module.bias.detach()
    if bias is not None and out_idx is not None:
        bias = bias[out_idx.to(bias.device)]
    if isinstance(module, torch.nn.Conv2d):
        new = torch.nn.Conv2d(
            in_channels=int(weight.shape[1]),
            out_channels=int(weight.shape[0]),
            kernel_size=module.kernel_size,
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            groups=module.groups,
            bias=bias is not None,
            padding_mode=module.padding_mode,
        )
    else:
        new = torch.nn.Linear(
            int(weight.shape[1]), int(weight.shape[0]), bias=bias is not None
        )
    new = new.to(device=weight.device, dtype=weight.dtype)
    with torch.no_grad():
        new.weight.copy_(weight)
        if bias is not None and new.bias is not None:
            new.bias.copy_(bias)
    new.train(module.training)
    return new


def _take_rows(matrix: Any, heads: Sequence[int], head_dim: int) -> Any:
    return _cat([matrix[head * head_dim : (head + 1) * head_dim] for head in heads], 0)


def _take_cols(matrix: Any, heads: Sequence[int], head_dim: int) -> Any:
    return _cat(
        [matrix[:, head * head_dim : (head + 1) * head_dim] for head in heads], 1
    )


def _cat(parts: list[Any], dim: int) -> Any:
    torch = _require_torch()
    return torch.cat(parts, dim=dim)


def _zero_heads(module: Any, keep: Sequence[int], head_dim: int, torch: Any) -> None:
    dropped = [head for head in range(int(module.num_heads)) if head not in set(keep)]
    if not dropped:
        return
        q_end = int(module.embed_dim)
        for head in dropped:
            sl = slice(head * head_dim, (head + 1) * head_dim)
            for part in range(3):
                start = part * q_end + head * head_dim
                module.in_proj_weight.data[start : start + head_dim] = 0
                if module.in_proj_bias is not None:
                    module.in_proj_bias.data[start : start + head_dim] = 0
            module.out_proj.weight.data[:, sl] = 0


def _head_scores(module: Any, torch: Any) -> Any:
    embed = int(module.embed_dim)
    heads = int(module.num_heads)
    width = embed // heads
    q, k, v = module.in_proj_weight.detach().abs().chunk(3, dim=0)
    out = module.out_proj.weight.detach().abs()
    scores = []
    for head in range(heads):
        sl = slice(head * width, (head + 1) * width)
        score = q[sl].sum() + k[sl].sum() + v[sl].sum() + out[:, sl].sum()
        scores.append(score)
    return torch.stack(scores)


def _channel_scores(producer: Any, consumer: Any) -> Any:
    prod = producer.weight.detach().abs()
    cons = consumer.weight.detach().abs()
    prod_score = prod.sum(dim=tuple(range(1, prod.ndim)))
    cons_dims = (0,) + tuple(range(2, cons.ndim))
    cons_score = cons.sum(dim=cons_dims)
    return prod_score + cons_score


def _split_chains(kids: list[tuple[str, Any]]) -> list[list[tuple[str, Any]]]:
    """Link consecutive siblings only when that width has one producer and one consumer.

    A feed-forward pair (``linear1`` then ``linear2``) matches. Two heads that
    merely share an output width, such as ``mu_head`` and ``logvar_head``, do
    not, because more than one sibling has that width.
    """

    def linked(index: int) -> bool:
        left = kids[index][1]
        right = kids[index + 1][1]
        width = int(left.weight.shape[0])
        if int(right.weight.shape[1]) != width:
            return False
        producers = sum(int(mod.weight.shape[0]) == width for _name, mod in kids)
        consumers = sum(int(mod.weight.shape[1]) == width for _name, mod in kids)
        return producers == 1 and consumers == 1

    chains: list[list[tuple[str, Any]]] = []
    current: list[tuple[str, Any]] = []
    for index, kid in enumerate(kids):
        if not current:
            current = [kid]
            continue
        if linked(index - 1):
            current.append(kid)
            continue
        if len(current) >= 2:
            chains.append(current)
        current = [kid]
    if len(current) >= 2:
        chains.append(current)
    return chains


def _group_view(param: Any) -> tuple[Any, int]:
    length = int(param.shape[-1])
    groups = length // 4
    if groups <= 0:
        return param, 0
    usable = groups * 4
    return param[..., :usable].reshape(-1, 4), groups


def _pack_tensor(param: Any, mask: Any) -> dict[str, Any]:
    length = int(param.shape[-1])
    groups = length // 4
    tail = length - groups * 4
    leading = int(param.numel() // length) if length else 0
    total_groups = leading * groups
    if mask is None or groups == 0:
        return {
            "groups": total_groups,
            "patterned": 0,
            "tail": tail,
            "complete": False,
            "values": b"",
            "bitmask": b"",
        }
    flat_mask = mask.detach().cpu()[..., : groups * 4].reshape(-1, 4)
    flat_value = param.detach().cpu()[..., : groups * 4].reshape(-1, 4)
    values: list[float] = []
    bitmask = bytearray()
    patterned = 0
    complete = tail == 0 and int(flat_mask.shape[0]) == total_groups
    for group in range(int(flat_mask.shape[0])):
        kept = flat_mask[group].nonzero().view(-1).tolist()
        if len(kept) != 2:
            complete = False
            continue
        patterned += 1
        first, second = int(kept[0]), int(kept[1])
        if first > second:
            first, second = second, first
        bitmask.append(first | (second << 2))
        values.append(float(flat_value[group, first].item()))
        values.append(float(flat_value[group, second].item()))
    if patterned != total_groups:
        complete = False
    return {
        "groups": total_groups,
        "patterned": patterned,
        "tail": tail,
        "complete": complete,
        "values": struct.pack("<" + "f" * len(values), *values) if values else b"",
        "bitmask": bytes(bitmask),
    }


def _unpack_2to4(
    torch: Any, shape: tuple[int, ...], values: bytes, bitmask: bytes
) -> Any:
    length = shape[-1]
    groups = length // 4
    out = torch.zeros(shape, dtype=torch.float32)
    if groups == 0:
        return out
    flat = out.view(-1, length)
    expected = int(flat.shape[0]) * groups
    if len(bitmask) != expected or len(values) != expected * 8:
        raise RecipeConfigError("2:4 export byte length does not match the tensor")
    offset = 0
    cursor = 0
    for row in range(int(flat.shape[0])):
        for group in range(groups):
            byte = bitmask[cursor]
            first = byte & 3
            second = (byte >> 2) & 3
            left, right = struct.unpack_from("<ff", values, offset)
            flat[row, group * 4 + first] = left
            flat[row, group * 4 + second] = right
            cursor += 1
            offset += 8
    return out


def _layer_row(
    name: str,
    kind: str,
    weight_before: int,
    weight_after: int,
    params_before: int,
    params_after: int,
    nonzero: int,
) -> dict[str, Any]:
    return {
        "name": name,
        "kind": kind,
        "sparsity": _sparsity_ratio(weight_before - weight_after, weight_before),
        "params_before": int(params_before),
        "params_after": int(params_after),
        "nonzero": int(nonzero),
        "_before": int(weight_before),
        "_after": int(weight_after),
    }


def _blank_report(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "method": config["method"],
        "schedule": config["schedule"],
        "target_sparsity": config["target_sparsity"],
        "include": list(config.get("include") or []),
        "exclude": list(config.get("exclude") or []),
        "start_step": config.get("start_step"),
        "end_step": config.get("end_step"),
        "iterations": config.get("iterations"),
        "finetune_steps": config.get("finetune_steps"),
        "layers": [],
        "skipped": [],
        "achieved_sparsity": None,
        "params_before": None,
        "params_after": None,
        "export_bytes_before": None,
        "export_bytes_after": None,
        "export_format": None,
        "export_dir": None,
        "tensorrt_engine": None,
        "tensorrt_error": None,
    }


def _selected(name: str, config: Mapping[str, Any]) -> bool:
    include = list(config.get("include") or [])
    exclude = list(config.get("exclude") or [])
    if include and not any(_match(name, pattern) for pattern in include):
        return False
    if exclude and any(_match(name, pattern) for pattern in exclude):
        return False
    return True


def _match(name: str, pattern: str) -> bool:
    from fnmatch import fnmatchcase

    if fnmatchcase(name, pattern):
        return True
    return fnmatchcase(name, pattern.rstrip(".") + ".*") if pattern else False


def _under_attention(model: Any, name: str, torch: Any) -> bool:
    parts = name.split(".")
    for index in range(1, len(parts)):
        prefix = ".".join(parts[:index])
        module = model.get_submodule(prefix)
        if isinstance(module, torch.nn.MultiheadAttention):
            return True
    return False


def _join(parent: str, child: str) -> str:
    if not parent:
        return child
    return f"{parent}.{child}"


def _parameter(model: Any, name: str) -> Any:
    found = dict(model.named_parameters())
    if name not in found:
        raise RecipeConfigError(f"pruning lost parameter {name!r}")
    return found[name]


def _replace(model: Any, name: str, module: Any) -> None:
    parent_name, sep, attr = name.rpartition(".")
    parent = model if not sep else model.get_submodule(parent_name)
    setattr(parent, attr, module)


def _drop_index(keep: Any, n: int, torch: Any) -> Any:
    keep_cpu = keep.detach().cpu().long()
    mask = torch.ones(n, dtype=torch.bool)
    mask[keep_cpu] = False
    return mask.nonzero(as_tuple=False).view(-1)


def _keep_count(n: int, sparsity: float) -> int:
    if n <= 1 or sparsity <= 0:
        return n
    kept = int(math.floor(n * (1.0 - sparsity) + 0.5))
    return min(n, max(1, kept))


def _count_params(module: Any) -> int:
    return sum(int(param.numel()) for param in module.parameters())


def _sparsity_ratio(removed: int, total: int) -> float:
    if total <= 0 or removed <= 0:
        return 0.0
    if removed >= total:
        return 1.0
    return round(removed / total, 6)


def _check_budget(iterations: int, finetune: int | None, train_steps: int) -> None:
    if train_steps < 1:
        raise RecipeConfigError("pruning needs train_steps >= 1")
    if finetune is not None:
        return
    if iterations > train_steps:
        raise RecipeConfigError(
            f"iterative pruning has {iterations} iterations but only "
            f"{train_steps} train steps. Each iteration needs a fine-tune "
            "step. Lower pruning.iterations or set pruning.finetune_steps."
        )


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RecipeConfigError(f"pruning.{field} must be a number")
    number = round(float(value), 6)
    if number != number or number in (float("inf"), float("-inf")):
        raise RecipeConfigError(f"pruning.{field} must be finite")
    return number


def _optional_step(value: Any, field: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise RecipeConfigError(f"pruning.{field} must be an integer")
    if value < 0:
        raise RecipeConfigError(f"pruning.{field} must be >= 0")
    return value


def _string_list(value: Any, field: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise RecipeConfigError(f"pruning.{field} must be a list of strings")
    return list(value)


def _capability_reason(
    cuda_available: bool | None, tensorrt_available: bool | None
) -> str:
    cuda = _cuda_ok(cuda_available)
    tensorrt = _tensorrt_ok(tensorrt_available)
    if cuda and tensorrt:
        return ""
    if not cuda and not tensorrt:
        return "TensorRT is not installed and CUDA is not available."
    if not tensorrt:
        return "TensorRT is not installed."
    return "CUDA is not available."


def _cuda_ok(flag: bool | None) -> bool:
    if flag is not None:
        return bool(flag)
    torch = _try_torch()
    if torch is None:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _tensorrt_ok(flag: bool | None) -> bool:
    if flag is not None:
        return bool(flag)
    try:
        return importlib.util.find_spec("tensorrt") is not None
    except Exception:
        return False


def _caught_engine_error(export_dir: Path) -> str:
    try:
        build_tensorrt_sparse_engine(export_dir)
    except PruningCapabilityError as exc:
        return str(exc)
    return (
        "TensorRT sparse engine build returned without an engine path. "
        "No engine was written."
    )


def _read_nm_manifest(export_dir: Path) -> dict[str, Any]:
    path = _nm_manifest_path(export_dir)
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PruningCapabilityError(
            f"could not read the 2:4 export manifest: {exc}. No engine was written."
        ) from exc
    if not isinstance(document, dict):
        raise PruningCapabilityError(
            "2:4 export manifest is not an object. No engine was written."
        )
    return document


def _nm_manifest_path(export_dir: Path) -> Path:
    candidates = (
        export_dir / "manifest.json",
        export_dir / "nm24" / "manifest.json",
        export_dir / "pruning_export" / "nm24" / "manifest.json",
    )
    for path in candidates:
        if path.is_file():
            return path
    raise PruningCapabilityError(
        "no ohho.pruning.2to4/v1 manifest was found. No engine was written."
    )


def _write_json(path: Path, document: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")


def _try_torch() -> Any | None:
    try:
        import torch
    except ImportError:
        return None
    return torch


def _require_torch() -> Any:
    torch = _try_torch()
    if torch is None:
        raise RecipeConfigError(
            "pruning needs torch. Install it with: pip install 'ohho-os[train]'"
        )
    return torch
