"""Quantisation-aware training and Q/DQ ONNX export.

Recipes carry an optional ``qat`` block. When it is enabled, linear layers
are wrapped with :class:`torch.ao.quantization.FakeQuantize` (weights, and
activations when the scope says so). A calibration warm-up updates the
observers. A fine-tune phase then trains through the fake-quant
straight-through estimator. Bit width, granularity, symmetry, and the two
phase lengths are plain fields on that block.

Export writes an ONNX graph of ``QuantizeLinear`` and ``DequantizeLinear``
nodes around each linear. That is the Q/DQ form TensorRT consumes. Building
a TensorRT engine needs the ``tensorrt`` package and a CUDA GPU; this
module raises :class:`QatCapabilityError` instead of pretending to build one.

Torch, ONNX, and ONNX Runtime are imported only when a call needs them.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ohho.train.recipe_schema import RecipeConfigError

BIT_WIDTHS: tuple[int, ...] = (8, 4)
SCOPES: tuple[str, ...] = ("weight_only", "weight_and_activation")
GRANULARITIES: tuple[str, ...] = ("per_tensor", "per_channel")
QDQ_ATOL = 1e-4
_QAT_KEYS = frozenset(
    {
        "enabled",
        "bit_width",
        "scope",
        "granularity",
        "activation_granularity",
        "symmetric",
        "activation_symmetric",
        "calibration_steps",
        "finetune_steps",
        "export_onnx",
    }
)
_FQ_TYPE: type | None = None
_ACT_LIKE: type | None = None


class QatError(RecipeConfigError):
    """The QAT block or an export cannot be used as given."""


class QatCapabilityError(QatError):
    """ONNX or TensorRT cannot run on this machine.

    The run does not switch to another export format.
    """


@dataclass(frozen=True)
class QatSpec:
    """Resolved ``qat`` block.

    ``granularity`` and ``symmetric`` apply to weights. Activation fields
    default to those same values when the recipe omits them, so one sweep
    of granularity or symmetry changes both. ``calibration_steps`` is the
    warm-up (observers on, no weight update). ``finetune_steps`` is the
    fake-quant fine-tune.
    """

    enabled: bool
    bit_width: int
    scope: str
    granularity: str
    activation_granularity: str
    symmetric: bool
    activation_symmetric: bool
    calibration_steps: int
    finetune_steps: int
    export_onnx: bool

    def to_dict(self) -> dict[str, Any]:
        """JSON object stored on the run manifest."""
        return {
            "enabled": self.enabled,
            "bit_width": self.bit_width,
            "scope": self.scope,
            "granularity": self.granularity,
            "activation_granularity": self.activation_granularity,
            "symmetric": self.symmetric,
            "activation_symmetric": self.activation_symmetric,
            "calibration_steps": self.calibration_steps,
            "finetune_steps": self.finetune_steps,
            "export_onnx": self.export_onnx,
        }


def spec_from_mapping(block: Mapping[str, Any]) -> QatSpec:
    """Fill defaults and reject a block the schema would not accept."""
    if not isinstance(block, Mapping):
        raise QatError("qat must be a mapping")
    unknown = sorted(str(key) for key in block if key not in _QAT_KEYS)
    if unknown:
        listed = ", ".join(unknown)
        raise QatError(f"unknown qat fields: {listed}")
    granularity = _choice(block, "granularity", "per_channel", GRANULARITIES)
    symmetric = _flag(block, "symmetric", True)
    if "activation_granularity" in block:
        activation_granularity = _choice(
            block, "activation_granularity", granularity, GRANULARITIES
        )
    else:
        activation_granularity = granularity
    if "activation_symmetric" in block:
        activation_symmetric = _flag(block, "activation_symmetric", symmetric)
    else:
        activation_symmetric = symmetric
    return QatSpec(
        enabled=_flag(block, "enabled", True),
        bit_width=_bit_width(block),
        scope=_choice(block, "scope", "weight_only", SCOPES),
        granularity=granularity,
        activation_granularity=activation_granularity,
        symmetric=symmetric,
        activation_symmetric=activation_symmetric,
        calibration_steps=_count(block, "calibration_steps"),
        finetune_steps=_count(block, "finetune_steps"),
        export_onnx=_flag(block, "export_onnx", False),
    )


def settings_from_recipe(recipe: Mapping[str, Any]) -> dict[str, Any] | None:
    """Resolved QAT settings, or ``None`` when the recipe has no ``qat`` block."""
    if not isinstance(recipe, Mapping) or "qat" not in recipe:
        return None
    block = recipe.get("qat")
    if block is None:
        return None
    return spec_from_mapping(block).to_dict()


def prepare_qat_recipe(
    block: Mapping[str, Any], *, family_id: str | None = None
) -> QatSpec:
    """Check a schema-valid block before training starts.

    Weight-and-activation QAT needs a calibration warm-up so activation
    scales exist. Family ``act`` cannot yet lower its transformer to Q/DQ
    ONNX, so ``export_onnx`` is refused before ``train_start``.
    """
    spec = spec_from_mapping(block)
    if not spec.enabled:
        return spec
    if spec.export_onnx and family_id == "act":
        raise QatError(
            "qat.export_onnx cannot lower the ACT transformer to Q/DQ ONNX. "
            "Set export_onnx to false for family act, or export "
            "ohho.compression.qat.TinyActLike with export_qdq_onnx. "
            "Full-transformer Q/DQ lowering is a follow-up. "
            "No fallback graph is written."
        )
    if spec.scope == "weight_and_activation" and spec.calibration_steps < 1:
        raise QatError(
            "qat.calibration_steps must be >= 1 when qat.scope is "
            "weight_and_activation. Activation scales are estimated during "
            "the calibration warm-up. No fallback scale is applied."
        )
    return spec


def insert_fake_quant(model: Any, spec: QatSpec) -> Any:
    """Wrap every ``nn.Linear`` in ``model`` with fake quantisation.

    Weights always get a fake-quant module. Activations get one only when
    ``spec.scope`` is ``weight_and_activation``. Existing wrappers are left
    in place.
    """
    if not spec.enabled:
        return model
    torch = _require_torch()
    fq_type = _fake_quant_linear_type()
    for name, child in list(model.named_children()):
        if isinstance(child, fq_type):
            continue
        if isinstance(child, torch.nn.Linear):
            setattr(model, name, fq_type(child, spec))
        else:
            insert_fake_quant(child, spec)
    return model


def unwrap_fake_quant(model: Any) -> Any:
    """Restore the original ``nn.Linear`` modules. Parameters are the same objects."""
    fq_type = _built_fake_quant_type()
    if fq_type is None:
        return model
    for name, child in list(model.named_children()):
        if isinstance(child, fq_type):
            setattr(model, name, child.linear)
        else:
            unwrap_fake_quant(child)
    return model


def run_calibration(model: Any, steps: int, forward_once: Callable[[], None]) -> None:
    """Collect observer ranges for ``steps`` forwards. No optimizer step.

    Fake quant stays off so the ranges are those of the float activations
    and weights. With ``steps`` of 0, each weight tensor is observed once
    and activations are not.
    """
    _set_observers(model, observe=True, quantize=False)
    was_training = bool(model.training)
    model.eval()
    torch = _require_torch()
    with torch.no_grad():
        for _ in range(steps):
            forward_once()
        if steps == 0:
            _observe_weights(model)
    _set_observers(model, observe=False, quantize=True)
    model.train(was_training)


def run_finetune(
    model: Any,
    steps: int,
    optimizer: Any,
    loss_once: Callable[[], Any],
) -> float | None:
    """Fine-tune for ``steps`` with fake quant on and observers frozen.

    ``loss_once`` returns a scalar loss. Gradients use the fake-quant
    straight-through estimator. Returns the last loss, or ``None`` when
    ``steps`` is 0.
    """
    _set_observers(model, observe=False, quantize=True)
    model.train()
    last: float | None = None
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_once()
        loss.backward()
        optimizer.step()
        last = float(loss.detach().float().item())
    return last


def apply_qat(
    model: Any,
    spec: QatSpec,
    *,
    calibration_forwards: Callable[[], None],
    loss_fn: Callable[[], Any],
    lr: float,
    export_path: str | Path | None = None,
    export_example: Any = None,
) -> dict[str, Any]:
    """Insert fake quant, calibrate, fine-tune, optionally export, then unwrap.

    The returned mapping is the resolved settings plus ``finetune_loss``,
    ``onnx_path``, and ``ran``. The module's ``nn.Linear`` parameters stay
    the fp32 master weights. The Q/DQ file is the quantized graph.
    """
    report = spec.to_dict()
    report["finetune_loss"] = None
    report["onnx_path"] = None
    report["ran"] = False
    if not spec.enabled:
        return report
    prepare_qat_recipe(spec.to_dict(), family_id=None)
    torch = _require_torch()
    try:
        device = next(model.parameters()).device
    except StopIteration as exc:
        raise QatError("QAT needs a module with parameters") from exc
    training = bool(model.training)
    try:
        insert_fake_quant(model, spec)
        model.to(device)
        run_calibration(model, spec.calibration_steps, calibration_forwards)
        if spec.finetune_steps:
            optimizer = torch.optim.AdamW(
                [param for param in model.parameters() if param.requires_grad],
                lr=lr,
            )
            report["finetune_loss"] = run_finetune(
                model, spec.finetune_steps, optimizer, loss_fn
            )
        if spec.export_onnx:
            if export_path is None or export_example is None:
                raise QatError(
                    "qat.export_onnx needs an export path and an example input. "
                    "No ONNX file was written."
                )
            written = export_qdq_onnx(model, export_example, export_path)
            report["onnx_path"] = str(written)
        report["ran"] = True
        return report
    finally:
        unwrap_fake_quant(model)
        model.train(training)


def export_qdq_onnx(model: Any, example: Any, path: str | Path) -> Path:
    """Write Q/DQ ONNX for a chain of fake-quant linears.

    ``example`` is a rank-2 float tensor ``[batch, features]``. The graph
    uses ``QuantizeLinear`` and ``DequantizeLinear`` (opset 13 for INT8,
    opset 21 for INT4) and matches the fake-quant module on that example.
    """
    torch = _require_torch()
    if not torch.is_tensor(example) or example.dim() != 2:
        raise QatError(
            "Q/DQ export expects a rank-2 example tensor [batch, features]. "
            "No ONNX file was written."
        )
    linears = _chain_linears(model)
    if not linears:
        raise QatError(
            "Q/DQ export needs fake-quant linear layers. Call insert_fake_quant "
            "first. No ONNX file was written."
        )
    was_training = bool(model.training)
    model.eval()
    try:
        if not _chain_matches(model, linears, example):
            raise QatError(
                "Q/DQ ONNX export needs a module whose forward is a chain of "
                "linear layers (ohho.compression.qat.TinyActLike). This module "
                "is not that chain. The ACT transformer is not lowered to "
                "QuantizeLinear/DequantizeLinear in this build. "
                "No ONNX file was written."
            )
        _assert_scales(linears)
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        _write_qdq(linears, example, destination)
    finally:
        model.train(was_training)
    return destination


def compare_ptq_qat(
    spec: QatSpec,
    *,
    seed: int = 0,
    state_dim: int = 4,
    d_model: int = 8,
    action_dim: int = 4,
    batch: int = 8,
    eval_batch: int = 4,
    lr: float = 0.05,
) -> dict[str, Any]:
    """Compare post-training quantisation with QAT on one tiny ACT-like stack.

    Both copies start from the same weights and see the same calibration
    forwards. PTQ stops after that warm-up. QAT then runs
    ``spec.finetune_steps`` of fake-quant SGD. The report is the eval MSE of
    each copy on a held-out batch. It does not measure GPU time or a
    TensorRT engine. See ``docs/train/qat.md``.
    """
    prepare_qat_recipe(spec.to_dict(), family_id=None)
    torch = _require_torch()
    if not spec.enabled:
        raise QatError("compare_ptq_qat needs qat.enabled true")
    torch.manual_seed(seed)
    cls = _tiny_act_like_type()
    init = cls(state_dim=state_dim, d_model=d_model, action_dim=action_dim)
    state = {key: value.detach().clone() for key, value in init.state_dict().items()}
    generator = torch.Generator().manual_seed(seed + 1)
    inputs = torch.randn(batch, state_dim, generator=generator)
    targets = torch.randn(batch, action_dim, generator=generator)
    eval_x = torch.randn(eval_batch, state_dim, generator=generator)
    eval_y = torch.randn(eval_batch, action_dim, generator=generator)

    def score(finetune_steps: int) -> float:
        copy = cls(state_dim=state_dim, d_model=d_model, action_dim=action_dim)
        copy.load_state_dict(state)
        insert_fake_quant(copy, spec)

        def forward() -> None:
            copy(inputs)

        run_calibration(copy, spec.calibration_steps, forward)
        if finetune_steps:
            optimizer = torch.optim.SGD(copy.parameters(), lr=lr)

            def loss_once() -> Any:
                predicted = copy(inputs)
                return torch.nn.functional.mse_loss(predicted, targets)

            run_finetune(copy, finetune_steps, optimizer, loss_once)
        copy.eval()
        with torch.no_grad():
            predicted = copy(eval_x)
            value = float(torch.nn.functional.mse_loss(predicted, eval_y).item())
        unwrap_fake_quant(copy)
        return value

    ptq_loss = score(0)
    qat_loss = score(spec.finetune_steps)
    report = spec.to_dict()
    report.update(
        {
            "method": "shared_calibration_then_optional_fake_quant_finetune",
            "ptq_finetune_steps": 0,
            "qat_finetune_steps": spec.finetune_steps,
            "ptq_eval_loss": ptq_loss,
            "qat_eval_loss": qat_loss,
            "gpu_measured": False,
            "tensorrt_measured": False,
        }
    )
    return report


def build_tensorrt_engine(
    onnx_path: str | Path, *, engine_path: str | Path | None = None
) -> str:
    """Build a TensorRT engine from a Q/DQ ONNX file.

    Raises :class:`QatCapabilityError` when TensorRT or a CUDA GPU is
    missing. This build does not call the TensorRT builder even when the
    package is installed; the ONNX file is the artifact a GPU host consumes.
    """
    del engine_path
    path = Path(onnx_path)
    if importlib.util.find_spec("tensorrt") is None:
        raise QatCapabilityError(
            "TensorRT engine build needs the tensorrt Python package and a "
            f"CUDA GPU. {path.name} is the Q/DQ ONNX TensorRT consumes. "
            "Building the engine is a follow-up on a GPU host. "
            "No fallback engine is produced."
        )
    cuda = _cuda_visible()
    if not cuda:
        raise QatCapabilityError(
            "TensorRT is installed but no CUDA GPU is visible. "
            "Engine build stays on the GPU host. No fallback engine is produced."
        )
    raise QatCapabilityError(
        "TensorRT and a CUDA GPU are visible, and ohho-os does not call the "
        "TensorRT builder in this version. Pass the Q/DQ ONNX to trtexec. "
        "GPU engine timing is a follow-up. No fallback engine is produced."
    )


def _choice(
    block: Mapping[str, Any], key: str, default: str, allowed: tuple[str, ...]
) -> str:
    if key not in block:
        return default
    value = block[key]
    if not isinstance(value, str) or value not in allowed:
        listed = ", ".join(allowed)
        raise QatError(f"qat.{key} must be one of {listed}")
    return value


def _flag(block: Mapping[str, Any], key: str, default: bool) -> bool:
    if key not in block:
        return default
    value = block[key]
    if not isinstance(value, bool):
        raise QatError(f"qat.{key} must be a boolean")
    return value


def _bit_width(block: Mapping[str, Any]) -> int:
    if "bit_width" not in block:
        return 8
    value = block["bit_width"]
    if isinstance(value, bool) or value not in BIT_WIDTHS:
        raise QatError("qat.bit_width must be 8 or 4")
    return int(value)


def _count(block: Mapping[str, Any], key: str) -> int:
    if key not in block:
        return 0
    value = block[key]
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise QatError(f"qat.{key} must be an integer >= 0")
    return value


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise QatError(
            "quantisation-aware training needs torch. "
            "Install it with: pip install 'ohho-os[train]'"
        ) from exc
    return torch


def _import_onnx() -> tuple[Any, Any, Any, Any]:
    try:
        import onnx
        from onnx import TensorProto, helper, numpy_helper
    except ImportError as exc:
        raise QatCapabilityError(
            "Q/DQ ONNX export needs the onnx package "
            "(pip install 'onnx>=1.15'). No ONNX file was written. "
            "No fallback format is produced."
        ) from exc
    return onnx, TensorProto, helper, numpy_helper


def _fake_quant_linear_type() -> type:
    global _FQ_TYPE
    if _FQ_TYPE is None:
        _FQ_TYPE = _build_fake_quant_linear()
    return _FQ_TYPE


def _built_fake_quant_type() -> type | None:
    return _FQ_TYPE


def _tiny_act_like_type() -> type:
    global _ACT_LIKE
    if _ACT_LIKE is None:
        _ACT_LIKE = _build_tiny_act_like()
    return _ACT_LIKE


def __getattr__(name: str) -> Any:
    if name == "TinyActLike":
        cls = _tiny_act_like_type()
        globals()["TinyActLike"] = cls
        return cls
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _build_tiny_act_like() -> type:
    torch = _require_torch()

    class TinyActLike(torch.nn.Module):
        """Linear stack shaped like ACT's state, latent, and action projections.

        The full ACT transformer is trained with the same fake-quant wrappers.
        This module is the chain :func:`export_qdq_onnx` lowers to
        QuantizeLinear and DequantizeLinear nodes.
        """

        def __init__(
            self, state_dim: int = 4, d_model: int = 8, action_dim: int = 4
        ) -> None:
            super().__init__()
            self.state_proj = torch.nn.Linear(state_dim, d_model)
            self.latent_proj = torch.nn.Linear(d_model, d_model)
            self.action_head = torch.nn.Linear(d_model, action_dim)

        def forward(self, state: Any) -> Any:
            hidden = self.state_proj(state)
            hidden = self.latent_proj(hidden)
            return self.action_head(hidden)

    return TinyActLike


def _build_fake_quant_linear() -> type:
    torch = _require_torch()

    class FakeQuantLinear(torch.nn.Module):
        """``nn.Linear`` with torch fake quant on the weight and, optionally, the output."""

        def __init__(self, linear: Any, spec: QatSpec) -> None:
            super().__init__()
            self.linear = linear
            self.spec = spec
            self.weight_fq = _make_fake_quant(
                spec.bit_width, spec.granularity, spec.symmetric, 0
            )
            self.act_fq = None
            if spec.scope == "weight_and_activation":
                self.act_fq = _make_fake_quant(
                    spec.bit_width,
                    spec.activation_granularity,
                    spec.activation_symmetric,
                    1,
                )
            self.weight_observed = False
            self.activation_observed = False
            # The encoder fast path reads ``linear.weight`` and skips
            # ``forward`` when the module is in eval and grad is off.
            # A hook forces that path to call this module instead.
            self.register_forward_hook(lambda _module, _inputs, output: output)

        @property
        def weight(self) -> Any:
            return self.linear.weight

        @property
        def bias(self) -> Any:
            return self.linear.bias

        def forward(self, inputs: Any) -> Any:
            if int(self.weight_fq.observer_enabled[0]) == 1:
                self.weight_observed = True
            weight = self.weight_fq(self.linear.weight)
            output = torch.nn.functional.linear(inputs, weight, self.linear.bias)
            if self.act_fq is None:
                return output
            if int(self.act_fq.observer_enabled[0]) == 1:
                self.activation_observed = True
            return self.act_fq(output)

    return FakeQuantLinear


def _make_fake_quant(
    bit_width: int, granularity: str, symmetric: bool, ch_axis: int
) -> Any:
    torch = _require_torch()
    from torch.ao.quantization.fake_quantize import FakeQuantize
    from torch.ao.quantization.observer import MinMaxObserver, PerChannelMinMaxObserver

    qmin, qmax, dtype, qscheme = _torch_qparams(
        torch, bit_width, granularity, symmetric
    )
    if granularity == "per_channel":
        return FakeQuantize(
            observer=PerChannelMinMaxObserver,
            quant_min=qmin,
            quant_max=qmax,
            dtype=dtype,
            qscheme=qscheme,
            ch_axis=ch_axis,
        )
    return FakeQuantize(
        observer=MinMaxObserver,
        quant_min=qmin,
        quant_max=qmax,
        dtype=dtype,
        qscheme=qscheme,
    )


def _torch_qparams(
    torch: Any, bit_width: int, granularity: str, symmetric: bool
) -> tuple[int, int, Any, Any]:
    per = "per_channel" if granularity == "per_channel" else "per_tensor"
    kind = "symmetric" if symmetric else "affine"
    qscheme = getattr(torch, f"{per}_{kind}")
    if symmetric and bit_width == 8:
        return -128, 127, torch.qint8, qscheme
    if symmetric:
        return -8, 7, torch.qint8, qscheme
    if bit_width == 8:
        return 0, 255, torch.quint8, qscheme
    return 0, 15, torch.quint8, qscheme


def _onnx_elem_type(bit_width: int, symmetric: bool) -> int:
    from onnx import TensorProto

    if bit_width == 8 and symmetric:
        return int(TensorProto.INT8)
    if bit_width == 8:
        return int(TensorProto.UINT8)
    if symmetric:
        return int(TensorProto.INT4)
    return int(TensorProto.UINT4)


def _set_observers(model: Any, *, observe: bool, quantize: bool) -> None:
    for module in _fq_modules(model):
        module.weight_fq.enable_observer(observe)
        module.weight_fq.enable_fake_quant(quantize)
        if module.act_fq is not None:
            module.act_fq.enable_observer(observe)
            module.act_fq.enable_fake_quant(quantize)


def _observe_weights(model: Any) -> None:
    for module in _fq_modules(model):
        module.weight_fq(module.linear.weight.detach())


def _fq_modules(model: Any) -> list[Any]:
    fq_type = _fake_quant_linear_type()
    return [module for module in model.modules() if isinstance(module, fq_type)]


def _chain_linears(model: Any) -> list[Any]:
    return _fq_modules(model)


def _chain_matches(model: Any, linears: list[Any], example: Any) -> bool:
    torch = _require_torch()
    try:
        with torch.no_grad():
            reference = model(example)
            value = example
            for layer in linears:
                value = layer(value)
    except Exception:
        return False
    if not torch.is_tensor(reference) or not torch.is_tensor(value):
        return False
    if reference.shape != value.shape:
        return False
    return bool(torch.allclose(reference, value, atol=1e-5, rtol=1e-5))


def _assert_scales(linears: list[Any]) -> None:
    for index, layer in enumerate(linears):
        if not layer.weight_observed:
            raise QatError(
                f"linear {index} has no weight scale. Run calibration before "
                "export. No fallback scale is applied."
            )
        if layer.act_fq is not None and not layer.activation_observed:
            raise QatError(
                "weight-and-activation export needs activation scales from the "
                "calibration warm-up. Set qat.calibration_steps to at least 1. "
                "No fallback scale is applied."
            )


def _write_qdq(linears: list[Any], example: Any, path: Path) -> None:
    import numpy as np

    onnx, tensor_proto, helper, numpy_helper = _import_onnx()
    nodes: list[Any] = []
    initializers: list[Any] = []
    current = "state"
    features = int(example.shape[1])
    bit_width = int(linears[0].spec.bit_width)
    for index, layer in enumerate(linears):
        weight = layer.linear.weight.detach().cpu().float()
        if index == 0 and int(weight.shape[1]) != features:
            raise QatError(
                "the example feature count does not match the first linear. "
                "No ONNX file was written."
            )
        bias = layer.linear.bias
        if bias is None:
            bias_np = np.zeros((int(weight.shape[0]),), dtype=np.float32)
        else:
            bias_np = bias.detach().cpu().float().numpy().astype(np.float32)
        initializers.append(
            numpy_helper.from_array(weight.numpy().astype(np.float32), f"W{index}")
        )
        initializers.append(numpy_helper.from_array(bias_np, f"B{index}"))
        produced = _append_qdq(
            nodes,
            initializers,
            helper,
            numpy_helper,
            value_name=f"W{index}",
            fake_quant=layer.weight_fq,
            bit_width=layer.spec.bit_width,
            symmetric=layer.spec.symmetric,
            per_channel=layer.spec.granularity == "per_channel",
            axis=0,
            prefix=f"w{index}",
        )
        gemm_out = f"gemm{index}"
        nodes.append(
            helper.make_node(
                "Gemm",
                [current, produced, f"B{index}"],
                [gemm_out],
                transB=1,
            )
        )
        current = gemm_out
        if layer.act_fq is not None:
            current = _append_qdq(
                nodes,
                initializers,
                helper,
                numpy_helper,
                value_name=current,
                fake_quant=layer.act_fq,
                bit_width=layer.spec.bit_width,
                symmetric=layer.spec.activation_symmetric,
                per_channel=layer.spec.activation_granularity == "per_channel",
                axis=1,
                prefix=f"a{index}",
            )
    batch = int(example.shape[0])
    out_features = int(linears[-1].linear.weight.shape[0])
    graph = helper.make_graph(
        nodes,
        "ohho_qdq",
        [helper.make_tensor_value_info("state", tensor_proto.FLOAT, [batch, features])],
        [
            helper.make_tensor_value_info(
                current, tensor_proto.FLOAT, [batch, out_features]
            )
        ],
        initializers,
    )
    opset = 21 if bit_width == 4 else 13
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    # onnxruntime 1.31 accepts IR <= 13. Pin a version it loads.
    model.ir_version = 10
    try:
        onnx.checker.check_model(model)
    except onnx.checker.ValidationError as exc:
        raise QatError(f"Q/DQ ONNX graph is not valid: {exc}") from exc
    onnx.save(model, str(path))


def _append_qdq(
    nodes: list[Any],
    initializers: list[Any],
    helper: Any,
    numpy_helper: Any,
    *,
    value_name: str,
    fake_quant: Any,
    bit_width: int,
    symmetric: bool,
    per_channel: bool,
    axis: int,
    prefix: str,
) -> str:
    import numpy as np

    scale_np = (
        fake_quant.scale.detach().cpu().float().numpy().astype(np.float32).reshape(-1)
    )
    zero_points = [
        int(value) for value in fake_quant.zero_point.detach().cpu().flatten().tolist()
    ]
    if scale_np.size == 0 or float(np.min(scale_np)) <= 0:
        raise QatError(
            f"{prefix} scale is not a positive finite value. "
            "No fallback scale is applied."
        )
    if not per_channel:
        scale_tensor = np.array(float(scale_np[0]), dtype=np.float32)
        zero_points = [zero_points[0]]
        q_kwargs: dict[str, Any] = {}
    else:
        scale_tensor = scale_np
        q_kwargs = {"axis": axis}
    _check_zero_points(zero_points, bit_width, symmetric, prefix)
    initializers.append(numpy_helper.from_array(scale_tensor, f"{prefix}_scale"))
    initializers.append(
        helper.make_tensor(
            f"{prefix}_zp",
            _onnx_elem_type(bit_width, symmetric),
            [] if len(zero_points) == 1 else [len(zero_points)],
            zero_points,
        )
    )
    quantized = f"{prefix}_q"
    dequantized = f"{prefix}_dq"
    nodes.append(
        helper.make_node(
            "QuantizeLinear",
            [value_name, f"{prefix}_scale", f"{prefix}_zp"],
            [quantized],
            **q_kwargs,
        )
    )
    nodes.append(
        helper.make_node(
            "DequantizeLinear",
            [quantized, f"{prefix}_scale", f"{prefix}_zp"],
            [dequantized],
            **q_kwargs,
        )
    )
    return dequantized


def _check_zero_points(
    values: list[int], bit_width: int, symmetric: bool, prefix: str
) -> None:
    if symmetric:
        lo, hi = (-128, 127) if bit_width == 8 else (-8, 7)
    else:
        lo, hi = (0, 255) if bit_width == 8 else (0, 15)
    for value in values:
        if value < lo or value > hi:
            raise QatError(
                f"{prefix} zero_point {value} is outside {lo}..{hi} for "
                f"INT{bit_width}. No ONNX file was written."
            )


def _cuda_visible() -> bool:
    try:
        import torch
    except ImportError:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False
