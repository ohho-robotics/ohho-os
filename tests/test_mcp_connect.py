"""Golden-file tests for ``ohho mcp connect``.

Every case passes a temporary HOME / USERPROFILE / APPDATA. These tests must
not read or write the developer machine's real client configs, and they must
not spawn the real ``claude`` or ``codex`` CLIs.
"""

from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from typing import List, Optional
from unittest.mock import patch

from ohho.cli import build_parser, main
from ohho.mcp_connect import (
    CLIENTS,
    HOSTED_MESSAGE,
    McpConnectError,
    client_config_path,
    connect_clients,
    resolve_ohho_executable,
)
from ohho.mcp_toml import TomlError, edit_codex_server, parse_toml

EXE = "/opt/ohho/bin/ohho"
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "mcp_connect"

STDIO_OTHER = {
    "theme": "dark",
    "mcpServers": {"other": {"command": "npx", "args": ["-y", "other-server"]}},
}
STDIO_STALE = {
    "mcpServers": {
        "ohho": {"command": "old-binary", "args": ["mcp"]},
        "other": {"command": "npx", "args": ["-y", "other-server"]},
    }
}
VS_OTHER = {
    "theme": "dark",
    "servers": {
        "other": {"type": "stdio", "command": "npx", "args": ["-y", "other-server"]}
    },
}
VS_STALE = {
    "servers": {
        "ohho": {"type": "stdio", "command": "old-binary", "args": ["mcp"]},
        "other": {"type": "stdio", "command": "npx", "args": ["-y", "other-server"]},
    },
}
OC_OTHER = {
    "theme": "dark",
    "mcp": {"other": {"type": "local", "command": ["npx", "-y", "other-server"]}},
}
OC_STALE = {
    "mcp": {
        "ohho": {"type": "local", "command": ["old"]},
        "other": {"type": "local", "command": ["npx", "-y", "other-server"]},
    },
}
OC_V2 = {
    "mcp": {
        "servers": {
            "other": {"type": "local", "command": ["npx", "-y", "other-server"]}
        }
    }
}
CODEX_OTHER = """# keep me
model = "gpt-5.1"

[mcp_servers.other]
command = "npx"
args = ["-y", "other-server"]
"""
CODEX_STALE = """model = "gpt-5.1"

[mcp_servers.ohho]
command = "old"
args = ["mcp"]

[mcp_servers.other]
command = "npx"
args = ["-y", "other-server"]
"""

JSON_CASES = (
    ("cursor", "stdio", STDIO_OTHER, STDIO_STALE),
    ("gemini", "stdio", STDIO_OTHER, STDIO_STALE),
    ("windsurf", "stdio", STDIO_OTHER, STDIO_STALE),
    ("claude-code", "stdio", STDIO_OTHER, STDIO_STALE),
    ("vscode", "vscode", VS_OTHER, VS_STALE),
    ("opencode", "opencode", OC_OTHER, OC_STALE),
)


def golden(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def _refuse_runner(_argv: List[str]) -> int:
    raise AssertionError("external CLI runner must not be called")


class TempConnect(unittest.TestCase):
    """Connect against a throwaway home directory."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        self.home = root / "home"
        self.appdata = root / "AppData" / "Roaming"
        self.home.mkdir()
        self.appdata.mkdir(parents=True)
        self.environ = {
            "HOME": str(self.home),
            "USERPROFILE": str(self.home),
            "APPDATA": str(self.appdata),
        }

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def connect(self, clients: Optional[List[str]], **kwargs):
        stdout = io.StringIO()
        stderr = io.StringIO()
        code = connect_clients(
            clients,
            environ=self.environ,
            platform_name=kwargs.pop("platform_name", "linux"),
            which=kwargs.pop("which", lambda _name: None),
            runner=kwargs.pop("runner", _refuse_runner),
            executable=kwargs.pop("executable", EXE),
            stdout=stdout,
            stderr=stderr,
            **kwargs,
        )
        self.assert_no_files_outside_temp()
        return code, stdout.getvalue(), stderr.getvalue()

    def config(self, client: str, platform_name: str = "linux") -> Path:
        return client_config_path(
            client,
            home=self.home,
            environ=self.environ,
            platform_name=platform_name,
        )

    def put(self, client: str, payload: object) -> Path:
        path = self.config(client)
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(payload, str):
            text = payload
        else:
            text = json.dumps(payload)
        path.write_text(text, encoding="utf-8", newline="\n")
        return path

    def assert_no_files_outside_temp(self) -> None:
        root = Path(self._tmp.name).resolve()
        for dirpath, _dirs, files in os.walk(root):
            for name in files:
                Path(dirpath, name).resolve().relative_to(root)


class TestClientPaths(TempConnect):
    def test_linux_macos_windows_targets(self) -> None:
        home = Path("/home/ada")
        env = {"HOME": "/home/ada", "USERPROFILE": "/home/ada"}
        expect = {
            "claude-code": home / ".claude.json",
            "codex": home / ".codex" / "config.toml",
            "cursor": home / ".cursor" / "mcp.json",
            "gemini": home / ".gemini" / "settings.json",
            "windsurf": home / ".codeium" / "windsurf" / "mcp_config.json",
            "opencode": home / ".config" / "opencode" / "opencode.json",
            "vscode": home / ".config" / "Code" / "User" / "mcp.json",
        }
        for client, path in expect.items():
            self.assertEqual(
                client_config_path(
                    client, home=home, environ=env, platform_name="linux"
                ),
                path,
            )
        xdg = {**env, "XDG_CONFIG_HOME": "/xdg"}
        self.assertEqual(
            client_config_path("vscode", home=home, environ=xdg, platform_name="linux"),
            Path("/xdg/Code/User/mcp.json"),
        )
        for value in ("$HOME/.config", "${HOME}/.config", "~/.config"):
            self.assertEqual(
                client_config_path(
                    "vscode",
                    home=home,
                    environ={**env, "XDG_CONFIG_HOME": value},
                    platform_name="linux",
                ),
                home / ".config" / "Code" / "User" / "mcp.json",
                value,
            )
        mac = Path("/Users/ada")
        self.assertEqual(
            client_config_path(
                "vscode",
                home=mac,
                environ={"HOME": "/Users/ada"},
                platform_name="darwin",
            ),
            mac / "Library" / "Application Support" / "Code" / "User" / "mcp.json",
        )
        win_home = Path(r"C:\Users\ada")
        win_env = {
            "USERPROFILE": r"C:\Users\ada",
            "APPDATA": r"C:\Users\ada\AppData\Roaming",
        }
        self.assertEqual(
            client_config_path(
                "vscode", home=win_home, environ=win_env, platform_name="win32"
            ),
            Path(r"C:\Users\ada\AppData\Roaming\Code\User\mcp.json"),
        )
        # Windows separators. On a POSIX host, `win_home / ".cursor"` inserts
        # forward slashes because `Path(r"C:\Users\ada")` is a single component.
        self.assertEqual(
            client_config_path(
                "cursor", home=win_home, environ=win_env, platform_name="win32"
            ),
            Path(r"C:\Users\ada\.cursor\mcp.json"),
        )

    def test_windows_vscode_requires_appdata(self) -> None:
        with self.assertRaises(McpConnectError) as ctx:
            client_config_path(
                "vscode",
                home=self.home,
                environ={"HOME": str(self.home)},
                platform_name="win32",
            )
        self.assertIn("APPDATA", str(ctx.exception))

    def test_list_names_every_client(self) -> None:
        code, out, err = self.connect(None, list_only=True)
        self.assertEqual(code, 0)
        self.assertEqual(err, "")
        for spec in CLIENTS:
            self.assertIn(spec.id, out)

    def test_hosted_is_refused(self) -> None:
        code, out, err = self.connect(["cursor", "claude-code"], hosted=True)
        self.assertEqual(code, 2)
        self.assertIn(HOSTED_MESSAGE, err)
        self.assertIn("OHH-273", err)
        self.assertIn("OHH-274", err)
        self.assertEqual(out, "")
        self.assertFalse(self.config("cursor").exists())

    def test_unknown_and_missing_client(self) -> None:
        code, _out, err = self.connect(["not-a-client"])
        self.assertEqual(code, 2)
        self.assertIn("unknown client", err)
        code, _out, err = self.connect([])
        self.assertEqual(code, 2)
        self.assertIn("--client", err)


class TestGoldenFiles(TempConnect):
    def test_json_clients_match_goldens(self) -> None:
        for client, shape, other, stale in JSON_CASES:
            with self.subTest(client=client, case="empty"):
                code, _out, err = self.connect([client])
                self.assertEqual(code, 0, err)
                path = self.config(client)
                self.assertEqual(
                    path.read_text(encoding="utf-8"), golden(f"{shape}_empty.json")
                )
                self.assertFalse(path.with_name(path.name + ".bak").exists())
                self.assertEqual(list(path.parent.glob(f".{path.name}.*.tmp")), [])

            with self.subTest(client=client, case="merge"):
                path = self.put(client, other)
                original = path.read_bytes()
                code, _out, err = self.connect([client])
                self.assertEqual(code, 0, err)
                self.assertEqual(
                    path.read_text(encoding="utf-8"), golden(f"{shape}_merged.json")
                )
                bak = path.with_name(path.name + ".bak")
                self.assertEqual(bak.read_bytes(), original)
                written = json.loads(path.read_text(encoding="utf-8"))
                container = (
                    written["mcp"]
                    if client == "opencode"
                    else written["servers" if client == "vscode" else "mcpServers"]
                )
                other_cmd = container["other"]["command"]
                if isinstance(other_cmd, list):
                    self.assertEqual(other_cmd[0], "npx")
                else:
                    self.assertEqual(other_cmd, "npx")
                self.assertIn("ohho", container)
                if "theme" in other:
                    self.assertEqual(written["theme"], "dark")

            with self.subTest(client=client, case="update"):
                path = self.put(client, stale)
                code, _out, err = self.connect([client])
                self.assertEqual(code, 0, err)
                self.assertEqual(
                    path.read_text(encoding="utf-8"), golden(f"{shape}_updated.json")
                )
                written = json.loads(path.read_text(encoding="utf-8"))
                container = (
                    written["mcp"]
                    if client == "opencode"
                    else written["servers" if client == "vscode" else "mcpServers"]
                )
                other_cmd = container["other"]["command"]
                if isinstance(other_cmd, list):
                    self.assertEqual(other_cmd[0], "npx")
                else:
                    self.assertEqual(other_cmd, "npx")
                ohho = container["ohho"]
                if client == "opencode":
                    self.assertEqual(ohho["type"], "local")
                    self.assertEqual(ohho["command"], [EXE, "mcp", "serve"])
                else:
                    self.assertEqual(ohho["command"], EXE)
                    self.assertEqual(ohho["args"], ["mcp", "serve"])
                if client == "vscode":
                    self.assertEqual(ohho["type"], "stdio")

            with self.subTest(client=client, case="remove"):
                path = self.put(client, stale)
                code, _out, err = self.connect([client], remove=True)
                self.assertEqual(code, 0, err)
                self.assertEqual(
                    path.read_text(encoding="utf-8"), golden(f"{shape}_removed.json")
                )
                written = json.loads(path.read_text(encoding="utf-8"))
                blob = json.dumps(written)
                self.assertNotIn('"ohho"', blob)
                self.assertIn("other-server", blob)

    def test_codex_goldens(self) -> None:
        path = self.config("codex")
        code, _out, err = self.connect(["codex"])
        self.assertEqual(code, 0, err)
        self.assertEqual(path.read_text(encoding="utf-8"), golden("codex_empty.toml"))
        self.assertFalse(path.with_name(path.name + ".bak").exists())

        path = self.put("codex", CODEX_OTHER)
        original = path.read_bytes()
        code, _out, err = self.connect(["codex"])
        self.assertEqual(code, 0, err)
        self.assertEqual(path.read_text(encoding="utf-8"), golden("codex_merged.toml"))
        self.assertEqual(path.with_name(path.name + ".bak").read_bytes(), original)
        doc = parse_toml(path.read_text(encoding="utf-8"))
        self.assertEqual(doc["model"], "gpt-5.1")
        self.assertEqual(doc["mcp_servers"]["other"]["command"], "npx")
        self.assertEqual(doc["mcp_servers"]["ohho"]["command"], EXE)
        self.assertIn("# keep me", path.read_text(encoding="utf-8"))

        path = self.put("codex", CODEX_STALE)
        code, _out, err = self.connect(["codex"])
        self.assertEqual(code, 0, err)
        self.assertEqual(path.read_text(encoding="utf-8"), golden("codex_updated.toml"))

        path = self.put("codex", CODEX_STALE)
        code, _out, err = self.connect(["codex"], remove=True)
        self.assertEqual(code, 0, err)
        text = path.read_text(encoding="utf-8")
        self.assertEqual(text, golden("codex_removed.toml"))
        self.assertNotIn("ohho", text)
        self.assertIn("other-server", text)

    def test_opencode_v2_servers_object_is_preserved(self) -> None:
        path = self.put("opencode", OC_V2)
        code, _out, err = self.connect(["opencode"])
        self.assertEqual(code, 0, err)
        self.assertEqual(
            path.read_text(encoding="utf-8"), golden("opencode_v2_merged.json")
        )
        written = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(written["mcp"]["servers"]["other"]["command"][0], "npx")
        self.assertEqual(
            written["mcp"]["servers"]["ohho"]["command"], [EXE, "mcp", "serve"]
        )
        self.assertNotIn("ohho", written["mcp"])

    def test_invalid_json_and_toml_are_refused(self) -> None:
        cursor = self.put("cursor", "{ not json")
        before = cursor.read_bytes()
        code, _out, err = self.connect(["cursor"])
        self.assertEqual(code, 1)
        self.assertIn("not valid JSON", err)
        self.assertIn("Refusing", err)
        self.assertEqual(cursor.read_bytes(), before)
        self.assertFalse(cursor.with_name(cursor.name + ".bak").exists())

        broken = self.put("codex", '[mcp_servers.ohho\ncommand = "x"\n')
        before = broken.read_bytes()
        code, _out, err = self.connect(["codex"])
        self.assertEqual(code, 1)
        self.assertIn("not valid TOML", err)
        self.assertEqual(broken.read_bytes(), before)
        self.assertFalse(broken.with_name(broken.name + ".bak").exists())

        root = self.put("gemini", "[]")
        code, _out, err = self.connect(["gemini"])
        self.assertEqual(code, 1)
        self.assertIn("not an object", err)
        self.assertEqual(root.read_text(encoding="utf-8"), "[]")

        self.put("windsurf", {"mcpServers": ["nope"]})
        code, _out, err = self.connect(["windsurf"])
        self.assertEqual(code, 1)
        self.assertIn("not a JSON object", err)

    def test_directory_is_refused(self) -> None:
        path = self.config("cursor")
        path.mkdir(parents=True)
        code, _out, err = self.connect(["cursor"])
        self.assertEqual(code, 1)
        self.assertIn("not a file", err)

    def test_dry_run_prints_diff_and_writes_nothing(self) -> None:
        path = self.put("cursor", STDIO_OTHER)
        before = path.read_bytes()
        code, out, err = self.connect(["cursor"], dry_run=True)
        self.assertEqual(code, 0, err)
        self.assertIn("dry-run:", out)
        self.assertIn("+", out)
        self.assertIn(EXE, out)
        self.assertEqual(path.read_bytes(), before)
        self.assertFalse(path.with_name(path.name + ".bak").exists())

        missing = self.config("gemini")
        code, out, _err = self.connect(["gemini"], dry_run=True)
        self.assertEqual(code, 0)
        self.assertIn("dry-run:", out)
        self.assertFalse(missing.exists())
        self.assertFalse(missing.parent.exists())

    def test_print_and_remove_missing(self) -> None:
        code, out, err = self.connect(["cursor"], print_config=True)
        self.assertEqual(code, 0, err)
        self.assertIn('"mcpServers"', out)
        self.assertTrue(self.config("cursor").is_file())

        code, out, err = self.connect(["gemini"], remove=True)
        self.assertEqual(code, 0, err)
        self.assertIn("does not exist", out)
        self.assertFalse(self.config("gemini").exists())

    def test_second_write_is_unchanged(self) -> None:
        code, _out, err = self.connect(["cursor"])
        self.assertEqual(code, 0, err)
        code, out, err = self.connect(["cursor"])
        self.assertEqual(code, 0, err)
        self.assertIn("unchanged", out)
        self.assertFalse(self.config("cursor").with_name("mcp.json.bak").exists())

    def test_aliases_and_multiple_clients(self) -> None:
        code, _out, err = self.connect(["claude", "gemini-cli", "vs-code"])
        self.assertEqual(code, 0, err)
        self.assertTrue(self.config("claude-code").is_file())
        self.assertTrue(self.config("gemini").is_file())
        self.assertTrue(self.config("vscode").is_file())

    def test_absolute_command_round_trip(self) -> None:
        windows = r"C:\Program Files\ohho\ohho.exe"
        code, _out, err = self.connect(["cursor"], executable=windows)
        self.assertEqual(code, 0, err)
        data = json.loads(self.config("cursor").read_text(encoding="utf-8"))
        self.assertEqual(data["mcpServers"]["ohho"]["command"], windows)
        spaced = "/opt/oh ho/bin/ohho"
        code, _out, err = self.connect(["codex"], executable=spaced)
        self.assertEqual(code, 0, err)
        doc = parse_toml(self.config("codex").read_text(encoding="utf-8"))
        self.assertEqual(doc["mcp_servers"]["ohho"]["command"], spaced)
        self.assertEqual(doc["mcp_servers"]["ohho"]["args"], ["mcp", "serve"])

    def test_codex_keeps_crlf(self) -> None:
        path = self.config("codex")
        path.parent.mkdir(parents=True)
        path.write_bytes(b'model = "x"\r\n')
        code, _out, err = self.connect(["codex"])
        self.assertEqual(code, 0, err)
        raw = path.read_bytes()
        self.assertIn(b'model = "x"\r\n', raw)
        self.assertIn(b"[mcp_servers.ohho]\r\n", raw)

    def test_does_not_use_path_home(self) -> None:
        with patch.object(Path, "home", side_effect=AssertionError("real home")):
            code, _out, err = self.connect(["cursor"])
        self.assertEqual(code, 0, err)
        self.assertTrue(str(self.config("cursor")).startswith(str(self.home)))


class TestCliInjection(TempConnect):
    def test_claude_and_codex_use_the_fake_runner(self) -> None:
        calls: List[List[str]] = []

        def runner(argv: List[str]) -> int:
            calls.append(list(argv))
            return 0

        def which(name: str) -> Optional[str]:
            if name == "claude":
                return "/usr/bin/claude"
            if name == "codex":
                return "/usr/bin/codex"
            return None

        code, out, err = self.connect(
            ["claude-code", "codex"], which=which, runner=runner
        )
        self.assertEqual(code, 0, err)
        self.assertEqual(
            calls,
            [
                [
                    "/usr/bin/claude",
                    "mcp",
                    "add",
                    "--scope",
                    "user",
                    "ohho",
                    "--",
                    EXE,
                    "mcp",
                    "serve",
                ],
                ["/usr/bin/codex", "mcp", "add", "ohho", "--", EXE, "mcp", "serve"],
            ],
        )
        self.assertIn("claude-code:", out)
        self.assertFalse(self.config("claude-code").exists())
        self.assertFalse(self.config("codex").exists())

        calls.clear()
        code, out, err = self.connect(
            ["claude-code", "codex"], which=which, runner=runner, remove=True
        )
        self.assertEqual(code, 0, err)
        self.assertEqual(
            calls[0],
            ["/usr/bin/claude", "mcp", "remove", "--scope", "user", "ohho"],
        )
        self.assertEqual(calls[1], ["/usr/bin/codex", "mcp", "remove", "ohho"])

    def test_dry_run_does_not_call_runner(self) -> None:
        def runner(_argv: List[str]) -> int:
            raise AssertionError("dry-run called the runner")

        code, out, err = self.connect(
            ["claude-code"],
            which=lambda name: "/usr/bin/claude" if name == "claude" else None,
            runner=runner,
            dry_run=True,
        )
        self.assertEqual(code, 0, err)
        self.assertIn("dry-run:", out)
        self.assertIn("--scope", out)
        self.assertIn(EXE, out)
        self.assertFalse(self.config("claude-code").exists())

    def test_runner_failure_writes_no_fallback_file(self) -> None:
        code, _out, err = self.connect(
            ["codex"],
            which=lambda name: "/usr/bin/codex" if name == "codex" else None,
            runner=lambda _argv: 3,
        )
        self.assertEqual(code, 3)
        self.assertIn("exited 3", err)
        self.assertFalse(self.config("codex").exists())

    def test_missing_cli_falls_back_to_the_file(self) -> None:
        code, _out, err = self.connect(["claude-code", "codex"])
        self.assertEqual(code, 0, err)
        self.assertEqual(
            self.config("claude-code").read_text(encoding="utf-8"),
            golden("stdio_empty.json"),
        )
        self.assertEqual(
            self.config("codex").read_text(encoding="utf-8"),
            golden("codex_empty.toml"),
        )


class TestExecutableResolution(unittest.TestCase):
    def test_argv0_which_and_venv_siblings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            script = root / "ohho"
            script.write_text("#!/bin/sh\n", encoding="utf-8")
            found = resolve_ohho_executable(
                argv0=str(script),
                which=lambda _name: None,
                python_executable=str(root / "missing"),
                platform_name="linux",
            )
            self.assertEqual(found, str(script.resolve()))

            shim = root / "shim" / "ohho"
            shim.parent.mkdir()
            shim.write_text("", encoding="utf-8")
            found = resolve_ohho_executable(
                argv0="python",
                which=lambda name: str(shim) if name == "ohho" else None,
                python_executable=str(root / "missing"),
                platform_name="linux",
            )
            self.assertEqual(found, str(shim.resolve()))

            bindir = root / "venv" / "bin"
            bindir.mkdir(parents=True)
            (bindir / "python").write_text("", encoding="utf-8")
            sibling = bindir / "ohho"
            sibling.write_text("", encoding="utf-8")
            found = resolve_ohho_executable(
                argv0="unittest",
                which=lambda _name: None,
                python_executable=str(bindir / "python"),
                platform_name="linux",
            )
            self.assertEqual(found, str(sibling.resolve()))

            scripts = root / "Scripts"
            scripts.mkdir()
            (scripts / "python.exe").write_text("", encoding="utf-8")
            win = scripts / "ohho.exe"
            win.write_text("", encoding="utf-8")
            found = resolve_ohho_executable(
                argv0="unittest",
                which=lambda _name: None,
                python_executable=str(scripts / "python.exe"),
                platform_name="win32",
            )
            self.assertEqual(found, str(win.resolve()))

            empty = root / "empty-venv"
            empty.mkdir()
            found = resolve_ohho_executable(
                argv0="unittest",
                which=lambda _name: None,
                python_executable=str(empty / "no-python-here"),
                platform_name="linux",
            )
            self.assertEqual(found, "ohho")


class TestTomlEditor(unittest.TestCase):
    def test_rejects_invalid_and_keeps_unrelated_tables(self) -> None:
        for text in ("{", "[mcp_servers.ohho", "command =", "n = 01", '"unterminated'):
            with self.assertRaises(TomlError):
                parse_toml(text)
        doc = parse_toml(
            """
enabled = true
count = 20_000
ratio = 1.5

[mcp_servers.other]
command = "npx"
args = ["-y", "pkg"]
env = { TOKEN = "x" }

[projects."/tmp/demo"]
trust_level = "trusted"

[[profiles]]
name = "a"
"""
        )
        self.assertIs(doc["enabled"], True)
        self.assertEqual(doc["count"], 20000)
        self.assertEqual(doc["ratio"], 1.5)
        self.assertEqual(doc["mcp_servers"]["other"]["env"]["TOKEN"], "x")
        self.assertEqual(doc["projects"]["/tmp/demo"]["trust_level"], "trusted")
        self.assertEqual(doc["profiles"][0]["name"], "a")

    def test_inline_and_nested_ohho_entries(self) -> None:
        inline = """[mcp_servers]
other = { command = "npx", args = ["-y", "other-server"] }
ohho = { command = "old", args = ["mcp"] }
"""
        edited = edit_codex_server(
            inline, command=EXE, args=["mcp", "serve"], remove=False
        )
        doc = parse_toml(edited)
        self.assertEqual(doc["mcp_servers"]["other"]["args"], ["-y", "other-server"])
        self.assertEqual(doc["mcp_servers"]["ohho"]["command"], EXE)
        self.assertNotIn("old", edited)

        nested = """[mcp_servers.ohho]
command = "old"

[mcp_servers.ohho.env]
FOO = "bar"

[mcp_servers.other]
command = "npx"
"""
        edited = edit_codex_server(
            nested, command=EXE, args=["mcp", "serve"], remove=False
        )
        doc = parse_toml(edited)
        self.assertEqual(doc["mcp_servers"]["other"]["command"], "npx")
        self.assertEqual(doc["mcp_servers"]["ohho"]["args"], ["mcp", "serve"])
        self.assertNotIn("FOO", edited)

        split = """[mcp_servers.ohho]
command = "old"

[mcp_servers.other]
command = "npx"

[mcp_servers.ohho.env]
FOO = "bar"
"""
        edited = edit_codex_server(
            split, command=EXE, args=["mcp", "serve"], remove=False
        )
        doc = parse_toml(edited)
        self.assertEqual(doc["mcp_servers"]["other"]["command"], "npx")
        self.assertNotIn("FOO", edited)
        self.assertEqual(doc["mcp_servers"]["ohho"]["command"], EXE)


class TestMcpCli(unittest.TestCase):
    def test_help_and_serve_delegation(self) -> None:
        with redirect_stdout(io.StringIO()) as out:
            with self.assertRaises(SystemExit) as exited:
                main(["mcp", "serve", "--help"])
        self.assertEqual(exited.exception.code, 0)
        self.assertIn("--sim", out.getvalue())
        self.assertIn("nav_goto", out.getvalue())

        with redirect_stdout(io.StringIO()) as out:
            with self.assertRaises(SystemExit) as exited:
                main(["mcp", "connect", "--help"])
        text = out.getvalue()
        self.assertEqual(exited.exception.code, 0)
        self.assertIn("--client", text)
        self.assertIn("--dry-run", text)
        self.assertIn("--print", text)
        self.assertIn("--remove", text)
        self.assertIn("--hosted", text)
        self.assertIn("OHH-273", text)
        self.assertIn("OHH-274", text)

        args = build_parser().parse_args(["mcp", "--sim"])
        self.assertIsNone(args.mcp_command)
        self.assertTrue(args.sim)

        with patch("ohho.mcp_server.run_stdio", return_value=0) as run:
            code = main(["mcp", "serve", "--sim", "--robot", "sim"])
        self.assertEqual(code, 0)
        session = run.call_args.kwargs["session"]
        self.assertEqual(session.robot_id, "sim")
        self.assertEqual(session.transport, "sim://")

        with patch("ohho.mcp_server.run_stdio", return_value=0) as run:
            code = main(["mcp"])
        self.assertEqual(code, 0)
        self.assertEqual(run.call_args.kwargs["session"].transport, "sim://")

    def test_cli_connect_stays_inside_temp_home(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            appdata = Path(tmp) / "AppData" / "Roaming"
            home.mkdir()
            appdata.mkdir(parents=True)
            env = {
                "HOME": str(home),
                "USERPROFILE": str(home),
                "APPDATA": str(appdata),
            }
            with patch.dict(os.environ, env, clear=False):
                # GitHub-hosted Ubuntu sets XDG_CONFIG_HOME=/home/runner/.config.
                # Leaving that absolute path in place writes VS Code's mcp.json
                # outside this temporary HOME. $HOME/.config / unset both mean
                # the default under the relocated home.
                os.environ.pop("XDG_CONFIG_HOME", None)
                with patch("shutil.which", return_value=None):
                    with patch.object(
                        Path, "home", side_effect=AssertionError("real home")
                    ):
                        code = main(
                            [
                                "mcp",
                                "connect",
                                "--client",
                                "cursor",
                                "--client",
                                "vscode",
                            ]
                        )
            self.assertEqual(code, 0)
            cursor = home / ".cursor" / "mcp.json"
            self.assertTrue(cursor.is_file())
            cursor.resolve().relative_to(Path(tmp).resolve())
            data = json.loads(cursor.read_text(encoding="utf-8"))
            self.assertEqual(data["mcpServers"]["ohho"]["args"], ["mcp", "serve"])
            if os.name == "nt":
                vscode = appdata / "Code" / "User" / "mcp.json"
            elif os.uname().sysname == "Darwin":
                vscode = (
                    home
                    / "Library"
                    / "Application Support"
                    / "Code"
                    / "User"
                    / "mcp.json"
                )
            else:
                vscode = home / ".config" / "Code" / "User" / "mcp.json"
            self.assertTrue(vscode.is_file(), vscode)
            vscode.resolve().relative_to(Path(tmp).resolve())
            vs = json.loads(vscode.read_text(encoding="utf-8"))
            self.assertEqual(vs["servers"]["ohho"]["type"], "stdio")

    def test_doctor_mentions_connect(self) -> None:
        with redirect_stdout(io.StringIO()) as out:
            code = main(["doctor"])
        self.assertEqual(code, 0)
        self.assertIn("mcp connect --list", out.getvalue())


if __name__ == "__main__":
    unittest.main()
