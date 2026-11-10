"""Run the Serve examples whose target hardware is a GPU.

Usage::

    python examples/serve/run_gpu.py

On this tree the scripts are placeholders. They exit 0 without a GPU and
without benchmark numbers. When the pending command is importable they exit 1
until this example calls it.
"""

from __future__ import annotations


def main() -> int:
    import support

    return support.run_manifest("gpu")


if __name__ == "__main__":
    raise SystemExit(main())
