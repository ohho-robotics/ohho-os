"""Locomotion policy runner for ONNX and TorchScript checkpoints.

The runner does not know a robot. It loads a file the plugin named, on CPU,
and returns the network output. Observation layout stays with the caller.
``onnxruntime`` and ``torch`` are imported only when that checkpoint is used.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from .humanoid_sim import DEFAULT_POLICY_RATE_HZ


class PolicyUnavailable(RuntimeError):
    """The checkpoint's runtime (ONNX Runtime or torch) is not installed."""


class LocomotionPolicy:
    """One ONNX or TorchScript policy, evaluated on CPU.

    ``.onnx`` uses ONNX Runtime. ``.pt`` is a TorchScript module from
    ``torch.jit.save``. The module's forward takes a float tensor of shape
    ``(1, obs)`` and returns ``(1, action)``. ``calls`` counts inferences.
    """

    def __init__(
        self, checkpoint: str, rate_hz: float = DEFAULT_POLICY_RATE_HZ
    ) -> None:
        path = Path(checkpoint)
        if not path.is_file():
            raise FileNotFoundError(f"locomotion checkpoint not found: {checkpoint}")
        suffix = path.suffix.lower()
        if suffix == ".onnx":
            kind = "onnx"
        elif suffix == ".pt":
            kind = "torchscript"
        else:
            raise ValueError(
                "locomotion checkpoint must be a .onnx or TorchScript .pt file"
            )
        if rate_hz <= 0:
            raise ValueError("rate_hz must be > 0")
        self.path = path
        self.rate_hz = float(rate_hz)
        self.calls = 0
        self._kind = kind
        self._session: object | None = None
        self._module: object | None = None
        self._input_name = "obs"

    @property
    def kind(self) -> str:
        """``onnx`` or ``torchscript``."""
        return self._kind

    def load(self) -> None:
        """Import the runtime and load the file. Repeated calls do nothing."""
        if self._session is not None or self._module is not None:
            return
        if self._kind == "onnx":
            try:
                import onnxruntime as ort
            except ImportError as exc:
                raise PolicyUnavailable(
                    "onnxruntime is not installed; pip install 'ohho-os[humanoid-sim]'"
                ) from exc
            session = ort.InferenceSession(
                str(self.path), providers=["CPUExecutionProvider"]
            )
            inputs = session.get_inputs()
            if not inputs:
                raise ValueError("ONNX policy has no inputs")
            self._input_name = str(inputs[0].name)
            self._session = session
            return
        try:
            import torch
        except ImportError as exc:
            raise PolicyUnavailable(
                "torch is not installed; pip install 'ohho-os[humanoid-sim]'"
            ) from exc
        module = torch.jit.load(str(self.path), map_location="cpu")
        module.eval()
        self._module = module

    def act(self, observation: Sequence[float]) -> tuple[float, ...]:
        """Run one inference. ``observation`` is the plugin's flat vector."""
        self.load()
        self.calls += 1
        if self._kind == "onnx":
            return self._act_onnx(observation)
        return self._act_torch(observation)

    def _act_onnx(self, observation: Sequence[float]) -> tuple[float, ...]:
        import numpy as np

        session = self._session
        obs = np.asarray(observation, dtype=np.float32).reshape(1, -1)
        outputs = session.run(None, {self._input_name: obs})  # type: ignore[union-attr]
        flat = np.asarray(outputs[0], dtype=np.float64).reshape(-1)
        return tuple(float(value) for value in flat)

    def _act_torch(self, observation: Sequence[float]) -> tuple[float, ...]:
        import torch

        module = self._module
        if not callable(module):
            raise RuntimeError("TorchScript policy is not loaded")
        flat_in = torch.tensor(list(observation), dtype=torch.float32)
        batched = flat_in.unsqueeze(0)
        with torch.no_grad():
            try:
                out = module(batched)
            except RuntimeError:
                # Some checkpoints are traced on a flat vector, not a batch.
                out = module(flat_in)
        flat = out.detach().reshape(-1).tolist()
        return tuple(float(value) for value in flat)


__all__ = ["LocomotionPolicy", "PolicyUnavailable"]
