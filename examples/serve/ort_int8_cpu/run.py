"""Placeholder for CPU ORT INT8 on a Raspberry Pi 5 class board.

``ohho optimize`` is not in this tree. The script names that gap and prints
no latency or accuracy numbers.
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
    """Print the placeholder status for the CPU INT8 path."""
    support = _support()
    from ohho.profiles import get_profile

    get_profile("edge_cpu")
    return support.emit_placeholder(
        example_id="ort-int8-cpu",
        hardware="Raspberry Pi 5 CPU (profile edge_cpu)",
        pending="ohho optimize (OHH-429) CPU ORT INT8",
        checks=("ohho.optimize.ort_int8",),
    )


if __name__ == "__main__":
    raise SystemExit(main())
