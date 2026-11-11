"""Shared helpers for the Serve examples.

CPU scripts call into this module. It uses the standard library and the
``ohho`` commands that exist in this tree. A command that is not on main stays
a placeholder: the script names it and prints no benchmark numbers.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any, Iterable

SERVE_DIR = Path(__file__).resolve().parent
REPO_ROOT = SERVE_DIR.parents[1]
MANIFEST_PATH = SERVE_DIR / "manifest.json"

FIXTURES = (
    "ohho/bench/examples/vision.json",
    "ohho/bench/examples/vla.json",
    "ohho/bench/examples/llm.json",
)

_REQUIRED = ("id", "kind", "script", "notebook", "readme")


def prefer_checkout() -> None:
    """Import ``ohho`` from this checkout ahead of any other install."""
    root = str(REPO_ROOT)
    try:
        sys.path.remove(root)
    except ValueError:
        pass
    sys.path.insert(0, root)


def child_env() -> dict[str, str]:
    """Environment that puts this checkout on ``PYTHONPATH``."""
    env = os.environ.copy()
    root = str(REPO_ROOT)
    previous = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = root if not previous else os.pathsep.join((root, previous))
    return env


def module_exists(name: str) -> bool:
    """True when ``name`` can be imported. A missing parent is absence."""
    try:
        return importlib.util.find_spec(name) is not None
    except ModuleNotFoundError:
        return False


prefer_checkout()


def load_manifest() -> list[dict[str, str]]:
    """Return the example rows from ``manifest.json``."""
    document = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    examples = document.get("examples")
    if not isinstance(examples, list) or not examples:
        raise RuntimeError(f"manifest has no examples: {MANIFEST_PATH}")
    rows: list[dict[str, str]] = []
    for item in examples:
        if not isinstance(item, dict):
            raise RuntimeError("manifest example is not an object")
        missing = [key for key in _REQUIRED if not isinstance(item.get(key), str)]
        if missing:
            raise RuntimeError(f"manifest example missing {missing}")
        rows.append({key: str(item[key]) for key in _REQUIRED})
    return rows


def top_commands() -> set[str]:
    """Names of the top-level ``ohho`` subcommands in this tree."""
    from ohho.cli import build_parser

    parser = build_parser()
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return set(action.choices)
    return set()


def bench_commands() -> set[str]:
    """Names of the ``ohho bench`` subcommands in this tree."""
    from ohho.cli import build_parser

    parser = build_parser()
    for action in parser._actions:
        if not isinstance(action, argparse._SubParsersAction):
            continue
        bench = action.choices.get("bench")
        if bench is None:
            continue
        for sub in bench._actions:
            if isinstance(sub, argparse._SubParsersAction):
                return set(sub.choices)
    return set()


def exporters_present() -> bool:
    """True when this tree can render Prometheus text or log to W&B."""
    if not module_exists("ohho.bench.report"):
        return False
    try:
        module = importlib.import_module("ohho.bench.report")
    except ImportError:
        return False
    return hasattr(module, "render_prometheus") or hasattr(module, "log_wandb")


def feature_ready(check: str) -> bool:
    """True when ``check`` names a command or module that is importable here.

    ``cli:<name>`` is a top-level subcommand. ``bench:<name>`` is an
    ``ohho bench`` subcommand. ``export:metrics`` is the Prometheus / W&B
    report API. Any other string is a module name.
    """
    if check == "export:metrics":
        return exporters_present()
    if check.startswith("cli:"):
        return check.split(":", 1)[1] in top_commands()
    if check.startswith("bench:"):
        return check.split(":", 1)[1] in bench_commands()
    return module_exists(check)


def emit_placeholder(
    *,
    example_id: str,
    hardware: str,
    pending: str,
    checks: Iterable[str],
) -> int:
    """Print the placeholder contract.

    Returns 0 while every probe is absent. Returns 1 when a probe is present,
    so the example fails until someone points it at the real command. It still
    prints no benchmark numbers.
    """
    ready = [check for check in checks if feature_ready(check)]
    print(f"example: {example_id}")
    if ready:
        joined = ", ".join(ready)
        print("status: blocked")
        print(f"hardware: {hardware}")
        print(
            "pending: "
            + pending
            + " is now present ("
            + joined
            + "); this example has no pinned invocation and will not "
            + "invent benchmark numbers"
        )
        print("runtime: not-run")
        print("result: none")
        return 1
    print("status: placeholder")
    print(f"hardware: {hardware}")
    print(f"pending: {pending}")
    print("runtime: not-run")
    print("result: none")
    return 0


def release_text_problems(text: str) -> list[str]:
    """Problems that would let a broken CPU example skip the release."""
    problems: list[str] = []
    if "  serve-examples:" not in text:
        problems.append("release workflow has no serve-examples job")
    if "python examples/serve/run_cpu.py" not in text:
        problems.append("release workflow does not run examples/serve/run_cpu.py")
    if "needs: [serve-examples]" not in text:
        problems.append("publish job does not need serve-examples")
    return problems


def gpu_workflow_problems(text: str) -> list[str]:
    """Problems in the nightly self-hosted GPU workflow."""
    problems: list[str] = []
    if "self-hosted" not in text:
        problems.append("GPU workflow does not target a self-hosted runner")
    if "python examples/serve/run_gpu.py" not in text:
        problems.append("GPU workflow does not run examples/serve/run_gpu.py")
    if "gpu_runner_probe.py" not in text:
        problems.append("GPU workflow does not probe for an online runner")
    needle = "needs.probe.outputs.available == 'true'"
    if needle not in text:
        problems.append("GPU job does not run only when a runner is online")
    return problems


def execute_notebook(path: Path) -> int:
    """Execute code cells with the standard library. No Jupyter install."""
    document = json.loads(path.read_text(encoding="utf-8"))
    cells = document.get("cells")
    if not isinstance(cells, list):
        print(f"notebook has no cells: {path}")
        return 1
    namespace: dict[str, Any] = {"__name__": "__main__", "__file__": str(path)}
    for cell in cells:
        if not isinstance(cell, dict) or cell.get("cell_type") != "code":
            continue
        source = cell.get("source", "")
        if isinstance(source, list):
            source = "".join(str(part) for part in source)
        if not isinstance(source, str) or not source.strip():
            continue
        try:
            exec(compile(source, str(path), "exec"), namespace, namespace)
        except SystemExit as exc:
            code = exc.code
            if code in (None, 0):
                continue
            if isinstance(code, int):
                return code
            print(code)
            return 1
        except Exception:
            traceback.print_exc()
            return 1
    return 0


def run_scripts(scripts: Iterable[Path]) -> int:
    """Run each script. Return 1 if any exits non-zero or the list is empty."""
    failed = False
    count = 0
    for script in scripts:
        count += 1
        print(f"--- {script}")
        completed = subprocess.run(
            [sys.executable, str(script)], check=False, env=child_env()
        )
        if completed.returncode != 0:
            print(f"failed: {script} exit {completed.returncode}")
            failed = True
    if count == 0 or failed:
        return 1
    return 0


def run_manifest(kind: str | None = None) -> int:
    """Run manifest examples.

    ``kind=None`` runs every row (the CPU release gate). ``kind="gpu"`` runs
    the GPU rows. Their scripts are CPU-safe placeholders until the pending
    command exists.
    """
    failed = False
    selected = 0
    label = "serve-examples-gpu" if kind == "gpu" else "serve-examples"
    for example in load_manifest():
        if kind is not None and example["kind"] != kind:
            continue
        selected += 1
        if _run_one(example) != 0:
            failed = True
    if selected == 0 or failed:
        print(f"{label}: failed")
        if selected == 0:
            print("problem: no examples selected")
        return 1
    print(f"{label}: ok")
    return 0


def _run_one(example: dict[str, str]) -> int:
    script = SERVE_DIR / example["script"]
    notebook = SERVE_DIR / example["notebook"]
    print(f"--- {example['id']}")
    if not script.is_file():
        print(f"failed: missing script {script}")
        return 1
    completed = subprocess.run(
        [sys.executable, str(script)], check=False, env=child_env()
    )
    if completed.returncode != 0:
        print(f"failed: {example['id']} script exit {completed.returncode}")
        return completed.returncode
    print(f"--- {example['id']} notebook")
    if not notebook.is_file():
        print(f"failed: missing notebook {notebook}")
        return 1
    code = execute_notebook(notebook)
    if code != 0:
        print(f"failed: {example['id']} notebook exit {code}")
        return code
    return 0
