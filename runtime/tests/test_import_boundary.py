"""``ohho`` must not import ``ohho_runtime``. Runtime stays on public ohho APIs."""

from __future__ import annotations

import os
import re
import subprocess
import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_OHHO = _ROOT / "ohho"
_RUNTIME_SRC = _ROOT / "runtime" / "src"

_OHHO_RUNTIME_IMPORTS = (
    re.compile(r"(?m)^\s*import\s+ohho_runtime\b"),
    re.compile(r"(?m)^\s*from\s+ohho_runtime\b"),
    re.compile(r"""importlib\.import_module\(\s*['"]ohho_runtime"""),
    re.compile(r"""__import__\(\s*['"]ohho_runtime"""),
)

_OHHO_IMPORT = re.compile(r"(?m)^(?:from|import)\s+(ohho(?:\.[\w]+)*)\b")

_PRIVATE_SAFETY = (
    "_last_state_t",
    "_state_stale",
    "OHHO_ARM_HARDWARE",
    "ohho.adapters",
    ".emergency_stop(",
    ".release_stop(",
    ".send_velocity(",
    ".send_joint_command(",
    ".disarm(",
    ".tick(",
    ".arm(",
)

# GPL, LGPL, or proprietary modules the base import must not load.
_COPYLEFT_OR_PROPRIETARY = ("pymavlink", "MAVProxy", "unitree_sdk2py")


class ImportBoundaryTest(unittest.TestCase):
    def test_ohho_sources_do_not_import_ohho_runtime(self) -> None:
        offenders: list[str] = []
        for path in sorted(_OHHO.rglob("*.py")):
            text = path.read_text(encoding="utf-8")
            for pattern in _OHHO_RUNTIME_IMPORTS:
                if pattern.search(text):
                    offenders.append(
                        f"{path.relative_to(_ROOT)} matches {pattern.pattern}"
                    )
        self.assertEqual(offenders, [])

    def test_root_project_does_not_depend_on_ohho_runtime(self) -> None:
        text = (_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        self.assertNotIn("ohho-runtime", text)
        self.assertNotIn("ohho_runtime", text)

    def test_importing_ohho_does_not_load_ohho_runtime(self) -> None:
        env = dict(os.environ)
        prior = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = str(_ROOT) if not prior else str(_ROOT) + os.pathsep + prior
        probe = (
            "import ohho, sys\n"
            "raise SystemExit(0 if 'ohho_runtime' not in sys.modules else 1)\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", probe],
            cwd=_ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(
            result.returncode,
            0,
            result.stderr or result.stdout,
        )

    def test_runtime_imports_only_public_ohho_modules(self) -> None:
        for path in sorted(_RUNTIME_SRC.rglob("*.py")):
            text = path.read_text(encoding="utf-8")
            for match in _OHHO_IMPORT.finditer(text):
                module = match.group(1)
                parts = module.split(".")
                self.assertFalse(
                    any(part.startswith("_") for part in parts),
                    f"{path} imports private module {module}",
                )
                self.assertNotIn(
                    "adapters",
                    parts,
                    f"{path} imports an adapter module",
                )

    def test_runtime_source_has_no_vendor_or_private_safety_use(self) -> None:
        from vendor_grep import scan_runtime

        self.assertEqual(scan_runtime(_ROOT), [])
        files = list(_RUNTIME_SRC.rglob("*.py"))
        files.append(_ROOT / "runtime" / "pyproject.toml")
        for path in files:
            text = path.read_text(encoding="utf-8")
            for token in _PRIVATE_SAFETY:
                self.assertNotIn(token, text, f"{path} contains {token}")
            for name in _COPYLEFT_OR_PROPRIETARY:
                self.assertNotIn(name, text, f"{path} names {name}")

    def test_manifest_import_does_not_load_copyleft_or_proprietary(self) -> None:
        env = dict(os.environ)
        prior = env.get("PYTHONPATH", "")
        extra = os.pathsep.join((str(_ROOT), str(_RUNTIME_SRC)))
        env["PYTHONPATH"] = extra if not prior else extra + os.pathsep + prior
        banned = ", ".join(repr(name) for name in _COPYLEFT_OR_PROPRIETARY)
        probe = (
            "import sys\n"
            "import ohho_runtime.manifest\n"
            f"banned = {{{banned}}}\n"
            "roots = {name.split('.')[0] for name in sys.modules}\n"
            "bad = sorted(roots & banned)\n"
            "raise SystemExit(0 if not bad else ','.join(bad))\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", probe],
            cwd=_ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(
            result.returncode,
            0,
            result.stderr or result.stdout,
        )


if __name__ == "__main__":
    unittest.main()
