"""Base-velocity policy adapter for the quadruped inference server."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Names ``/load_model`` already accepted, including the former module path.
MODEL_NAMES = frozenset(
    {
        "go2",
        "go2_base",
        "Go2PolicyModel",
        "ohho.serve.go2.Go2PolicyModel",
        "ohho.robots.unitree.policy_model.Go2PolicyModel",
    }
)


class Go2PolicyModel:
    """Inference wrapper for quadruped policies that emit base velocity commands."""

    def __init__(self, device: str = "cpu") -> None:
        self.device = device
        self.config: dict[str, Any] = {}
        self._path = ""
        self._mock = False

    def load_model(self, model_path: str, **kwargs: Any) -> None:
        if "device" in kwargs and kwargs["device"]:
            self.device = str(kwargs["device"])
        self._path = model_path
        p = Path(model_path).expanduser()
        config_file = p / "config.json" if p.is_dir() else p.parent / "config.json"
        if config_file.exists():
            with open(config_file, encoding="utf-8") as f:
                self.config = json.load(f)
        else:
            self.config = {
                "policy": "go2_base_velocity",
                "state_dim": 3,
                "action_dim": 3,
            }
        self._mock = bool(self.config.get("mock")) or str(
            self.config.get("policy", "")
        ).lower() in ("mock", "go2_mock")

    def predict_action(
        self, image: Any = None, instruction: str = "", **kwargs: Any
    ) -> dict[str, Any]:
        del image
        state: list[float] = list(kwargs.get("state") or [0.0, 0.0, 0.0])
        while len(state) < 3:
            state.append(0.0)
        odom = kwargs.get("odom") or {}
        odom_x = float(odom.get("x", 0.0)) if isinstance(odom, dict) else 0.0

        vx, vy, wz = self._select_base_velocity(state, instruction, odom_x)
        out: dict[str, Any] = {
            "base": {"vx": float(vx), "vy": float(vy), "wz": float(wz)},
        }
        mode = self.config.get("mode")
        if mode:
            out["mode"] = str(mode)
        elif self._mock:
            out["mode"] = "mock"
        return out

    def _select_base_velocity(
        self, state: list[float], instruction: str, odom_x: float
    ) -> tuple[float, float, float]:
        """Return (vx, vy, wz). Mock / missing weights use a simple go-forward heuristic."""
        target_vx = float(self.config.get("default_vx", 0.2))
        target_vy = float(self.config.get("default_vy", 0.0))
        target_wz = float(self.config.get("default_wz", 0.0))

        text = instruction.lower()
        if "stop" in text or "halt" in text:
            return 0.0, 0.0, 0.0
        if "left" in text:
            target_wz = float(self.config.get("turn_wz", 0.3))
        elif "right" in text:
            target_wz = -float(self.config.get("turn_wz", 0.3))
        elif "back" in text or "reverse" in text:
            target_vx = -abs(target_vx)
        elif "forward" in text or "go" in text or self._mock:
            target_vx = abs(target_vx)

        goal_x = self.config.get("goal_x")
        if goal_x is not None and odom_x >= float(goal_x):
            return 0.0, 0.0, 0.0
        if goal_x is not None:
            remaining = float(goal_x) - odom_x
            target_vx = min(target_vx, max(0.05, remaining))

        return target_vx, target_vy, target_wz
