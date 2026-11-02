"""AST scan: runtime code must not swallow exceptions."""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"

_BROAD = frozenset({"Exception", "BaseException"})


def _func_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def _caught_names(node: ast.expr | None) -> set[str]:
    if isinstance(node, ast.Name):
        return {node.id}
    if isinstance(node, ast.Attribute):
        return {node.attr}
    if isinstance(node, ast.Tuple):
        names: set[str] = set()
        for elt in node.elts:
            if isinstance(elt, ast.Name):
                names.add(elt.id)
        return names
    return set()


def _accounted(body: list[ast.stmt]) -> bool:
    """True when the handler re-raises or calls ``note_exception``."""
    for stmt in body:
        for node in ast.walk(stmt):
            if isinstance(node, ast.Raise):
                return True
            if isinstance(node, ast.Call) and _func_name(node.func) == "note_exception":
                return True
    return False


def _silent(body: list[ast.stmt]) -> bool:
    """True for an empty body, ``pass``, or ``...``."""
    if not body:
        return True
    for stmt in body:
        if isinstance(stmt, ast.Pass):
            continue
        if (
            isinstance(stmt, ast.Expr)
            and isinstance(stmt.value, ast.Constant)
            and stmt.value.value is Ellipsis
        ):
            continue
        return False
    return True


def handler_swallows(handler: ast.ExceptHandler) -> bool:
    """Bare ``except`` and ``except Exception: pass`` style handlers."""
    if handler.type is None:
        return True
    if _silent(handler.body):
        return True
    names = _caught_names(handler.type)
    if names & _BROAD and not _accounted(handler.body):
        return True
    return False


def find_swallows(source: str, filename: str = "<snippet>") -> list[str]:
    tree = ast.parse(source, filename=filename)
    findings: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ExceptHandler) and handler_swallows(node):
            findings.append(f"{filename}:{node.lineno}: swallowed exception")
    return findings


def scan_tree(root: Path) -> list[str]:
    findings: list[str] = []
    for path in sorted(root.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        findings.extend(find_swallows(text, str(path.relative_to(root.parent.parent))))
    return findings


class NoSwallowTest(unittest.TestCase):
    def test_runtime_src_does_not_swallow(self) -> None:
        self.assertEqual(scan_tree(_SRC), [])

    def test_scanner_flags_bare_except_and_pass(self) -> None:
        bare = "def f():\n    try:\n        g()\n    except:\n        raise\n"
        self.assertTrue(find_swallows(bare))
        passed = (
            "def f():\n    try:\n        g()\n    except Exception:\n        pass\n"
        )
        self.assertTrue(find_swallows(passed))
        ellipsis = (
            "def f():\n    try:\n        g()\n    except Exception:\n        ...\n"
        )
        self.assertTrue(find_swallows(ellipsis))
        cleanup = "def f():\n    try:\n        g()\n    except OSError:\n        pass\n"
        self.assertTrue(find_swallows(cleanup))
        broad = (
            "def f():\n"
            "    while True:\n"
            "        try:\n"
            "            g()\n"
            "        except Exception:\n"
            "            continue\n"
        )
        self.assertTrue(find_swallows(broad))

    def test_scanner_allows_raise_and_note(self) -> None:
        raised = (
            "def f():\n"
            "    try:\n"
            "        g()\n"
            "    except KeyError:\n"
            "        raise RuntimeError('missing') from None\n"
        )
        self.assertEqual(find_swallows(raised), [])
        noted = (
            "def f():\n"
            "    try:\n"
            "        g()\n"
            "    except Exception as exc:\n"
            "        note_exception(exc, service='svc', event='failed')\n"
        )
        self.assertEqual(find_swallows(noted), [])
        reraised = (
            "def f():\n"
            "    try:\n"
            "        g()\n"
            "    except Exception as exc:\n"
            "        note_exception(exc, service='svc', event='failed')\n"
            "        raise\n"
        )
        self.assertEqual(find_swallows(reraised), [])
        waiting = (
            "def f():\n"
            "    try:\n"
            "        g()\n"
            "    except socket.timeout:\n"
            "        continue\n"
        )
        self.assertEqual(find_swallows(waiting), [])


if __name__ == "__main__":
    unittest.main()
