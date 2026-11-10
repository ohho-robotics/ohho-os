"""Placeholder for W&B and Prometheus export.

The bench report exporters are not in this tree. The script names that gap.
It does not call W&B, does not push to Prometheus, and prints no metric values.
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
    """Print the placeholder status for metric export."""
    support = _support()
    return support.emit_placeholder(
        example_id="export-wandb-prometheus",
        hardware="cpu",
        pending="W&B and Prometheus export from the bench report",
        checks=("export:metrics",),
    )


if __name__ == "__main__":
    raise SystemExit(main())
