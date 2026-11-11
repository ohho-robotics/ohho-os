"""Placeholder for a small VLM bench on vLLM.

The LLM bench command is not in this tree. The script names that gap and
prints no token-rate or accuracy numbers. It does not contact vLLM.
"""

from __future__ import annotations


def _support() -> object:
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "support.py"
    spec = importlib.util.spec_from_file_location("ohho_serve_example_support", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    """Print the placeholder status for the vLLM bench."""
    support = _support()
    return support.emit_placeholder(
        example_id="vlm-vllm",
        hardware="GPU host for vLLM",
        pending="ohho bench llm on vLLM (OHH-436)",
        checks=("vllm",),
    )


if __name__ == "__main__":
    raise SystemExit(main())
