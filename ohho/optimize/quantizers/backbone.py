"""Select a SmolVLA language-model backbone before AWQ or GPTQ.

The role ``lm`` walks the attribute path used by the LeRobot SmolVLA
policy. A loaded module is written to a small directory with
``config.json`` so the existing AWQ and GPTQ loaders can read it. A
checkpoint that is already a directory is left as-is.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from ohho.optimize.models import LoadedModel

_LM = "lm"


def prepare_backbone(model: LoadedModel, scratch: Path) -> LoadedModel:
    """Return a model whose path is the LM directory when ``submodule`` is ``lm``."""
    if model.submodule not in {None, _LM, "language", "text"}:
        return model
    if model.submodule not in {_LM, "language", "text"}:
        return model
    if model.module is None:
        return model
    from ohho.optimize.exporters import select_submodule

    backbone = select_submodule(model.module, _LM)
    dest = scratch / "smolvla-lm"
    dest.mkdir(parents=True, exist_ok=True)
    config = {
        "model_type": "smollm",
        "architectures": ["SmolLMForCausalLM"],
        "ohho_backbone": _LM,
    }
    (dest / "config.json").write_text(json.dumps(config), encoding="utf-8")
    import torch

    torch.save(backbone.state_dict(), dest / "model.pt")
    return replace(
        model,
        path=dest,
        module=backbone,
        kind="smolvla" if model.kind == "smolvla" else model.kind,
        submodule=_LM,
    )
