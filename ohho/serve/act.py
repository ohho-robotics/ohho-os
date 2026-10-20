"""ACT policy model adapter for ohho.serve."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


class ACTModel:
    """Inference wrapper for TinyACTPolicy compatible with ohho.serve."""

    def __init__(self, device: str = "cpu") -> None:
        self.device = device
        self.policy: Any = None
        self.config: Dict[str, Any] = {}

    def load_model(self, model_path: str, **kwargs: Any) -> None:
        if "device" in kwargs and kwargs["device"]:
            self.device = kwargs["device"]
        p = Path(model_path).expanduser()
        config_file = p / "config.json"
        if not config_file.exists() and (p.parent / "config.json").exists():
            config_file = p.parent / "config.json"

        if config_file.exists():
            with open(config_file, encoding="utf-8") as f:
                self.config = json.load(f)
        else:
            self.config = {"state_dim": 9, "action_dim": 9, "chunk_size": 10}

        pt_file = p / "policy.pt" if (p / "policy.pt").exists() else p
        if pt_file.is_file():
            from ..train.act import load_act_checkpoint

            self.policy, loaded_config = load_act_checkpoint(str(p), device=self.device)
            if loaded_config:
                self.config.update(loaded_config)

    def predict_action(
        self, image: Any = None, instruction: str = "", **kwargs: Any
    ) -> Dict[str, Any]:
        state = kwargs.get("state")
        state_dim = self.config.get("state_dim", 9)
        action_dim = self.config.get("action_dim", 9)

        if state is None:
            state = [0.0] * state_dim

        if self.policy is not None:
            action = self.policy.select_action(state)
        else:
            action = [0.0] * action_dim

        vector = [float(x) for x in action]
        result: Dict[str, Any] = {
            "vector": vector,
            "instruction": instruction,
            "mock": self.policy is None,
        }
        names = self.config.get("action_names")
        if (
            isinstance(names, list)
            and len(names) == len(vector)
            and all(isinstance(name, str) for name in names)
        ):
            result["names"] = list(names)
            result["values"] = {name: value for name, value in zip(names, vector)}
        state_names = self.config.get("state_names")
        if isinstance(state_names, list) and state_names:
            result["state_names"] = list(state_names)
        if self.config.get("arm_action"):
            result["arm_action"] = self.config["arm_action"]
        return result
