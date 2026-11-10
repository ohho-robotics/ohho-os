"""Run every Serve example that is safe on CPU.

Usage::

    python examples/serve/run_cpu.py

Exit 0 when every script and notebook exits 0. A failing example exits 1.
Placeholders for commands that are not in this tree exit 0 and print no
benchmark numbers. GPU-target examples are included because their scripts do
not need a GPU until ``ohho optimize`` or the vLLM bench command lands.
"""

from __future__ import annotations


def main() -> int:
    import support

    return support.run_manifest(None)


if __name__ == "__main__":
    raise SystemExit(main())
