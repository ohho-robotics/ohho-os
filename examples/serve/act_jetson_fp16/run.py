"""Placeholder for FP16 TensorRT ACT compile, parity, and bench.

``ohho optimize`` is not in this tree. The script names that gap and prints
no latency, parity, or throughput numbers.
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
    """Print the placeholder status for the Jetson FP16 path."""
    support = _support()
    from ohho.profiles import get_profile

    get_profile("jetson_single")
    return support.emit_placeholder(
        example_id="act-jetson-fp16",
        hardware="Jetson-class GPU (profile jetson_single)",
        pending=(
            "ohho optimize (OHH-429) FP16 TensorRT, parity, and the bench harness"
        ),
        checks=("tensorrt", "bench:launch"),
    )


if __name__ == "__main__":
    raise SystemExit(main())
