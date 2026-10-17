"""Write AI-client MCP config for the local ``ohho mcp serve`` stdio server.

``ohho mcp connect`` merges an ``ohho`` entry into each client's config. It
never replaces the rest of the file. Existing files are copied to ``*.bak``
and the new text is written atomically. Invalid JSON or TOML is refused.

The default entry is the local simulator server (``sim://`` only). ``--hosted``
is reserved until ``ohho login`` (OHH-273) and hosted MCP auth (OHH-274).

Tests must pass a temporary ``environ`` (HOME / USERPROFILE / APPDATA /
XDG_CONFIG_HOME) and a ``which`` callback. The real process environment is
used only by the CLI.
"""

from __future__ import annotations

import difflib
import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Callable, List, Mapping, Optional, Sequence, TextIO, Tuple

from .mcp_toml import TomlError, edit_codex_server

SERVER_NAME = "ohho"
SERVER_ARGS: Tuple[str, ...] = ("mcp", "serve")
HOSTED_MESSAGE = "needs `ohho login` (OHH-273) and hosted MCP auth (OHH-274)"

Which = Callable[[str], Optional[str]]
Runner = Callable[[Sequence[str]], int]


class McpConnectError(Exception):
    """A connect request was refused. ``code`` is the process exit status."""

    def __init__(self, message: str, code: int = 1) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class ClientSpec:
    """One AI client ``ohho mcp connect`` knows how to configure."""

    id: str
    title: str
    aliases: Tuple[str, ...]
    summary: str


CLIENTS: Tuple[ClientSpec, ...] = (
    ClientSpec(
        "claude-code",
        "Claude Code",
        ("claude", "claude_code"),
        "claude mcp add --scope user, else ~/.claude.json",
    ),
    ClientSpec(
        "codex",
        "Codex",
        (),
        "codex mcp add, else ~/.codex/config.toml",
    ),
    ClientSpec(
        "cursor",
        "Cursor",
        (),
        "~/.cursor/mcp.json (mcpServers.ohho)",
    ),
    ClientSpec(
        "vscode",
        "VS Code",
        ("vs-code", "code", "visual-studio-code"),
        "user mcp.json (servers.ohho, type stdio)",
    ),
    ClientSpec(
        "gemini",
        "Gemini CLI",
        ("gemini-cli",),
        "~/.gemini/settings.json (mcpServers.ohho)",
    ),
    ClientSpec(
        "windsurf",
        "Windsurf",
        (),
        "~/.codeium/windsurf/mcp_config.json (mcpServers.ohho)",
    ),
    ClientSpec(
        "opencode",
        "OpenCode",
        ("open-code",),
        "~/.config/opencode/opencode.json (mcp.ohho, type local)",
    ),
)

_BY_ALIAS = {spec.id: spec.id for spec in CLIENTS}
for _spec in CLIENTS:
    for _alias in _spec.aliases:
        _BY_ALIAS[_alias] = _spec.id

_JSON_KEY = {
    "claude-code": "mcpServers",
    "cursor": "mcpServers",
    "gemini": "mcpServers",
    "windsurf": "mcpServers",
    "vscode": "servers",
}


def normalize_client(name: str) -> str:
    """Return the canonical client id or raise :class:`McpConnectError`."""
    key = name.strip().lower()
    found = _BY_ALIAS.get(key)
    if found is None:
        known = ", ".join(spec.id for spec in CLIENTS)
        raise McpConnectError(f"unknown client {name!r}. known: {known}", code=2)
    return found


def format_client_list() -> str:
    """Text for ``ohho mcp connect --list``."""
    lines = [f"{spec.id:<14} {spec.title:<16} {spec.summary}" for spec in CLIENTS]
    return "\n".join(lines) + "\n"


def resolve_home(environ: Mapping[str, str], platform_name: str) -> Path:
    """Home directory from ``environ`` (never :meth:`Path.home`)."""
    if platform_name == "win32":
        raw = environ.get("USERPROFILE") or environ.get("HOME")
    else:
        raw = environ.get("HOME") or environ.get("USERPROFILE")
    if not raw:
        raise McpConnectError(
            "cannot resolve the home directory (set HOME or USERPROFILE)",
            code=2,
        )
    return Path(raw)


def _target_path(platform_name: str, first: object, *rest: str) -> Path:
    """Join ``first`` and ``rest`` using the target platform's separators.

    Host :class:`Path` follows the machine that is running the tests, so
    ``Path(appdata) / "Code"`` on Linux turns a Windows ``APPDATA`` value
    into ``C:\\Users\\.../Code/User/mcp.json``. Parse and join with
    :class:`PureWindowsPath` or :class:`PurePosixPath`, then wrap the
    result in a concrete :class:`Path` so the caller can create the file.
    """
    flavour = PureWindowsPath if platform_name == "win32" else PurePosixPath
    pure = flavour(os.fspath(first))
    for part in rest:
        pure /= part
    return Path(os.fspath(pure))


def _xdg_anchor(home: Path, raw: Optional[str]) -> Tuple[object, ...]:
    """Directory parts for ``$XDG_CONFIG_HOME`` or ``HOME/.config``.

    ``$HOME/.config``, ``${HOME}/.config`` and ``~/.config`` follow the
    resolved home, so relocating ``HOME`` does not keep another user's
    config directory. Any other value (including an absolute path outside
    that home) is used as given.
    """
    if raw is None or not raw.strip():
        return (home, ".config")
    text = raw.strip()
    relative: Optional[str] = None
    for prefix in ("${HOME}", "$HOME", "~"):
        if text == prefix:
            return (home,)
        for sep in ("/", "\\"):
            token = prefix + sep
            if text.startswith(token):
                relative = text[len(token) :]
                break
        if relative is not None:
            break
    if relative is None:
        return (text,)
    parts = tuple(part for part in relative.replace("\\", "/").split("/") if part)
    return (home, *parts)


def client_config_path(
    client_id: str,
    *,
    home: Path,
    environ: Mapping[str, str],
    platform_name: str,
) -> Path:
    """Absolute config path for ``client_id`` on ``platform_name``."""
    if client_id == "claude-code":
        return _target_path(platform_name, home, ".claude.json")
    if client_id == "codex":
        return _target_path(platform_name, home, ".codex", "config.toml")
    if client_id == "cursor":
        return _target_path(platform_name, home, ".cursor", "mcp.json")
    if client_id == "vscode":
        return _vscode_user_mcp(home, environ, platform_name)
    if client_id == "gemini":
        return _target_path(platform_name, home, ".gemini", "settings.json")
    if client_id == "windsurf":
        return _target_path(
            platform_name, home, ".codeium", "windsurf", "mcp_config.json"
        )
    if client_id == "opencode":
        return _target_path(platform_name, home, ".config", "opencode", "opencode.json")
    raise McpConnectError(f"unknown client {client_id!r}", code=2)


def _vscode_user_mcp(
    home: Path, environ: Mapping[str, str], platform_name: str
) -> Path:
    # Same folder as VS Code user settings.json, file name mcp.json.
    # https://code.visualstudio.com/docs/configure/settings
    # https://code.visualstudio.com/docs/agent-customization/mcp-servers
    if platform_name == "win32":
        appdata = environ.get("APPDATA")
        if not appdata:
            raise McpConnectError(
                "APPDATA is not set; cannot locate VS Code user mcp.json",
                code=2,
            )
        return _target_path(platform_name, appdata, "Code", "User", "mcp.json")
    if platform_name == "darwin":
        return _target_path(
            platform_name,
            home,
            "Library",
            "Application Support",
            "Code",
            "User",
            "mcp.json",
        )
    anchor = _xdg_anchor(home, environ.get("XDG_CONFIG_HOME"))
    return _target_path(platform_name, *anchor, "Code", "User", "mcp.json")


def resolve_ohho_executable(
    *,
    argv0: Optional[str] = None,
    which: Optional[Which] = None,
    python_executable: Optional[str] = None,
    platform_name: Optional[str] = None,
) -> str:
    """Absolute path of the ``ohho`` console script.

    GUI clients do not inherit a shell ``PATH``, so config files store this
    path. Resolution order: ``argv0`` when it is the console script,
    ``which("ohho")`` (pipx shims and venvs on ``PATH``), then ``ohho`` next
    to the current Python. Returns the bare name ``ohho`` when nothing on
    disk matches.
    """
    plat = platform_name if platform_name is not None else sys.platform
    which_fn = which if which is not None else shutil.which
    script = sys.argv[0] if argv0 is None else argv0
    python = sys.executable if python_executable is None else python_executable

    candidate = Path(script)
    if candidate.name.lower() in {"ohho", "ohho.exe", "ohho.cmd", "ohho.bat"}:
        found = _existing_file(candidate)
        if found:
            return found

    located = which_fn("ohho")
    if located:
        found = _existing_file(Path(located))
        if found:
            return found

    py = Path(python)
    names = (
        ("ohho.exe", "ohho.cmd", "ohho.bat", "ohho") if plat == "win32" else ("ohho",)
    )
    folders = [py.parent, py.parent / "bin", py.parent / "Scripts"]
    parent = py.parent.parent
    folders.extend([parent / "bin", parent / "Scripts"])
    for folder in folders:
        for name in names:
            found = _existing_file(folder / name)
            if found:
                return found
    return "ohho"


def _existing_file(path: Path) -> Optional[str]:
    try:
        if path.is_file():
            return str(path.resolve())
    except OSError:
        return None
    return None


def _stdio_entry(executable: str) -> dict:
    return {"command": executable, "args": list(SERVER_ARGS)}


def _vscode_entry(executable: str) -> dict:
    return {"type": "stdio", "command": executable, "args": list(SERVER_ARGS)}


def _opencode_entry(executable: str) -> dict:
    return {"type": "local", "command": [executable, *SERVER_ARGS]}


def _dump_json(data: dict) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False) + "\n"


def _load_json(path: Path, text: str) -> dict:
    body = text.lstrip("\ufeff").strip()
    if not body:
        return {}
    try:
        data = json.loads(body)
    except json.JSONDecodeError as exc:
        raise McpConnectError(
            f"{path} is not valid JSON ({exc}). Refusing to modify it."
        ) from exc
    if not isinstance(data, dict):
        raise McpConnectError(
            f"{path} JSON root is not an object. Refusing to modify it."
        )
    return data


def _touch_json_key(
    data: dict, key: str, entry: dict, remove: bool, path: Path
) -> bool:
    if key not in data or data[key] is None:
        if remove:
            return False
        data[key] = {}
    container = data[key]
    if not isinstance(container, dict):
        raise McpConnectError(
            f"{path} key {key!r} is not a JSON object. Refusing to modify it."
        )
    if remove:
        if SERVER_NAME not in container:
            return False
        del container[SERVER_NAME]
        return True
    if container.get(SERVER_NAME) == entry:
        return False
    container[SERVER_NAME] = entry
    return True


def _is_server_object(value: object) -> bool:
    return isinstance(value, dict) and ("command" in value or "type" in value)


def _touch_opencode(data: dict, entry: dict, remove: bool, path: Path) -> bool:
    """Update ``mcp.ohho`` (v1) or ``mcp.servers.ohho`` when that object exists.

    New files and v1 configs get ``mcp.ohho`` (``type: local``, command
    array), which is the shape OHH-275 names. OpenCode v2 stores servers
    under ``mcp.servers``; when that object is already present, the ``ohho``
    entry is written there so unrelated v2 servers stay put. A v1 ``mcp.ohho``
    server object is updated in either case.
    """
    mcp_value = data.get("mcp")
    if mcp_value is not None and not isinstance(mcp_value, dict):
        raise McpConnectError(
            f"{path} key 'mcp' is not a JSON object. Refusing to modify it."
        )
    if isinstance(mcp_value, dict) and "servers" in mcp_value:
        if not isinstance(mcp_value["servers"], dict):
            raise McpConnectError(
                f"{path} key 'mcp.servers' is not a JSON object. Refusing to modify it."
            )
    servers = mcp_value.get("servers") if isinstance(mcp_value, dict) else None
    if isinstance(servers, dict):
        changed = _touch_json_key(data["mcp"], "servers", entry, remove, path)
        mcp = data["mcp"]
        if _is_server_object(mcp.get(SERVER_NAME)):
            if remove:
                del mcp[SERVER_NAME]
                changed = True
            elif mcp.get(SERVER_NAME) != entry:
                mcp[SERVER_NAME] = entry
                changed = True
        return changed
    return _touch_json_key(data, "mcp", entry, remove, path)


def _read_text(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    if not path.is_file():
        raise McpConnectError(f"{path} is not a file. Refusing to modify it.")
    # Decode bytes so Windows CRLF survives. Path.read_text() would
    # translate newlines before the TOML editor can preserve them.
    try:
        return path.read_bytes().decode("utf-8")
    except UnicodeDecodeError as exc:
        raise McpConnectError(
            f"{path} is not valid UTF-8 ({exc}). Refusing to modify it."
        ) from exc


def _backup_and_write(path: Path, text: str) -> Optional[Path]:
    """Copy an existing file to ``name.bak``, then replace it atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    bak: Optional[Path] = None
    if path.exists():
        bak = path.with_name(path.name + ".bak")
        shutil.copy2(path, bak)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(text, encoding="utf-8", newline="\n")
        os.replace(tmp, path)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise
    if tmp.exists():
        tmp.unlink()
    return bak


def _unified_diff(path: Path, old: str, new: str) -> str:
    fromfile = str(path)
    diff = difflib.unified_diff(
        old.splitlines(keepends=True),
        new.splitlines(keepends=True),
        fromfile=fromfile,
        tofile=fromfile,
    )
    return "".join(diff)


def _claude_argv(cli: str, executable: str, remove: bool) -> List[str]:
    # Current docs: `claude mcp add [options] <name> -- <command> [args...]`
    # https://code.claude.com/docs/en/mcp
    if remove:
        return [cli, "mcp", "remove", "--scope", "user", SERVER_NAME]
    return [
        cli,
        "mcp",
        "add",
        "--scope",
        "user",
        SERVER_NAME,
        "--",
        executable,
        *SERVER_ARGS,
    ]


def _codex_argv(cli: str, executable: str, remove: bool) -> List[str]:
    # https://learn.chatgpt.com/docs/extend/mcp
    if remove:
        return [cli, "mcp", "remove", SERVER_NAME]
    return [cli, "mcp", "add", SERVER_NAME, "--", executable, *SERVER_ARGS]


def _format_argv(argv: Sequence[str]) -> str:
    import shlex

    return " ".join(shlex.quote(part) for part in argv)


@dataclass
class _Plan:
    client_id: str
    action: str  # write, unchanged, cli
    path: Optional[Path] = None
    old: str = ""
    new: str = ""
    argv: Optional[List[str]] = None
    missing: bool = False


def _entry_for(client_id: str, executable: str) -> dict:
    if client_id == "vscode":
        return _vscode_entry(executable)
    if client_id == "opencode":
        return _opencode_entry(executable)
    return _stdio_entry(executable)


def _plan_toml(
    client_id: str,
    path: Path,
    old: Optional[str],
    executable: str,
    remove: bool,
) -> _Plan:
    if old is None:
        if remove:
            return _Plan(client_id, "unchanged", path=path, missing=True)
        new = edit_codex_server(
            "",
            command=executable,
            args=list(SERVER_ARGS),
            remove=False,
        )
        return _Plan(client_id, "write", path=path, old="", new=new)
    try:
        new = edit_codex_server(
            old,
            command=executable,
            args=list(SERVER_ARGS),
            remove=remove,
        )
    except TomlError as exc:
        raise McpConnectError(
            f"{path} is not valid TOML ({exc}). Refusing to modify it."
        ) from exc
    if new == old:
        return _Plan(client_id, "unchanged", path=path, old=old, new=new)
    return _Plan(client_id, "write", path=path, old=old, new=new)


def _plan_file(
    client_id: str,
    path: Path,
    executable: str,
    remove: bool,
) -> _Plan:
    old = _read_text(path)
    if client_id == "codex":
        return _plan_toml(client_id, path, old, executable, remove)
    if old is None:
        if remove:
            return _Plan(client_id, "unchanged", path=path, missing=True)
        data: dict = {}
        old_text = ""
    else:
        data = _load_json(path, old)
        old_text = old
    entry = _entry_for(client_id, executable)
    if client_id == "opencode":
        changed = _touch_opencode(data, entry, remove, path)
    else:
        changed = _touch_json_key(data, _JSON_KEY[client_id], entry, remove, path)
    if not changed:
        return _Plan(client_id, "unchanged", path=path, old=old_text, new=old_text)
    new_text = _dump_json(data)
    if old is not None and new_text == old_text:
        return _Plan(client_id, "unchanged", path=path, old=old_text, new=new_text)
    return _Plan(client_id, "write", path=path, old=old_text, new=new_text)


def _default_runner(argv: Sequence[str]) -> int:
    import subprocess

    completed = subprocess.run(list(argv), check=False)
    return int(completed.returncode)


def connect_clients(
    clients: Optional[Sequence[str]],
    *,
    list_only: bool = False,
    dry_run: bool = False,
    print_config: bool = False,
    remove: bool = False,
    hosted: bool = False,
    executable: Optional[str] = None,
    environ: Optional[Mapping[str, str]] = None,
    platform_name: Optional[str] = None,
    which: Optional[Which] = None,
    runner: Optional[Runner] = None,
    stdout: Optional[TextIO] = None,
    stderr: Optional[TextIO] = None,
) -> int:
    """Configure ``clients``. Returns a process exit code.

    ``which`` and ``runner`` inject the ``claude`` / ``codex`` lookup and
    process spawn so tests never execute those CLIs. ``environ`` supplies
    HOME, USERPROFILE, APPDATA, and XDG_CONFIG_HOME; pass a temporary
    directory from tests. An ambient absolute ``XDG_CONFIG_HOME`` wins over
    ``HOME/.config``.
    """
    out = stdout if stdout is not None else sys.stdout
    err = stderr if stderr is not None else sys.stderr
    try:
        return _connect(
            clients,
            list_only=list_only,
            dry_run=dry_run,
            print_config=print_config,
            remove=remove,
            hosted=hosted,
            executable=executable,
            environ=os.environ if environ is None else environ,
            platform_name=sys.platform if platform_name is None else platform_name,
            which=shutil.which if which is None else which,
            runner=_default_runner if runner is None else runner,
            stdout=out,
            stderr=err,
        )
    except McpConnectError as exc:
        print(f"error: {exc}", file=err)
        return exc.code


def _connect(
    clients: Optional[Sequence[str]],
    *,
    list_only: bool,
    dry_run: bool,
    print_config: bool,
    remove: bool,
    hosted: bool,
    executable: Optional[str],
    environ: Mapping[str, str],
    platform_name: str,
    which: Which,
    runner: Runner,
    stdout: TextIO,
    stderr: TextIO,
) -> int:
    if hosted:
        raise McpConnectError(f"--hosted {HOSTED_MESSAGE}", code=2)
    if list_only:
        stdout.write(format_client_list())
        return 0
    if not clients:
        raise McpConnectError(
            "pass --client <name> (repeatable), or --list",
            code=2,
        )

    ordered: List[str] = []
    for name in clients:
        client_id = normalize_client(name)
        if client_id not in ordered:
            ordered.append(client_id)

    exe = executable if executable is not None else resolve_ohho_executable(which=which)
    if exe == "ohho":
        print(
            "warning: could not resolve the ohho executable; "
            "GUI clients may not find 'ohho' on PATH",
            file=stderr,
        )

    home = resolve_home(environ, platform_name)
    plans: List[_Plan] = []
    for client_id in ordered:
        cli = _cli_path(client_id, which)
        if cli is not None:
            plans.append(
                _Plan(
                    client_id,
                    "cli",
                    argv=_argv_for(client_id, cli, exe, remove),
                )
            )
            continue
        path = client_config_path(
            client_id, home=home, environ=environ, platform_name=platform_name
        )
        plans.append(_plan_file(client_id, path, exe, remove))

    for plan in plans:
        code = _apply_plan(
            plan,
            dry_run=dry_run,
            print_config=print_config,
            remove=remove,
            runner=runner,
            stdout=stdout,
            stderr=stderr,
        )
        if code != 0:
            return code
    return 0


def _cli_path(client_id: str, which: Which) -> Optional[str]:
    program = {"claude-code": "claude", "codex": "codex"}.get(client_id)
    if program is None:
        return None
    return which(program) or None


def _argv_for(client_id: str, cli: str, executable: str, remove: bool) -> List[str]:
    if client_id == "claude-code":
        return _claude_argv(cli, executable, remove)
    if client_id == "codex":
        return _codex_argv(cli, executable, remove)
    raise McpConnectError(f"{client_id} has no CLI installer", code=2)


def _apply_plan(
    plan: _Plan,
    *,
    dry_run: bool,
    print_config: bool,
    remove: bool,
    runner: Runner,
    stdout: TextIO,
    stderr: TextIO,
) -> int:
    if plan.action == "cli":
        assert plan.argv is not None
        rendered = _format_argv(plan.argv)
        if dry_run:
            print(f"dry-run: {rendered}", file=stdout)
            return 0
        if print_config:
            print(rendered, file=stdout)
        code = runner(plan.argv)
        if code != 0:
            print(f"error: {rendered} exited {code}", file=stderr)
            return code
        verb = "remove" if remove else "add"
        print(
            f"{plan.client_id}: ran {verb} via {_format_argv(plan.argv[:3])}",
            file=stdout,
        )
        return 0

    assert plan.path is not None
    label = f"{plan.client_id}: {plan.path}"
    if plan.action == "unchanged":
        if plan.missing:
            why = "file does not exist" if remove else "nothing to write"
            print(f"unchanged: {label} ({why})", file=stdout)
        else:
            print(f"unchanged: {label}", file=stdout)
        if print_config and plan.new:
            stdout.write(plan.new)
        return 0

    if dry_run:
        print(f"dry-run: {label}", file=stdout)
        stdout.write(_unified_diff(plan.path, plan.old, plan.new))
        if print_config:
            stdout.write(plan.new)
        return 0

    if print_config:
        stdout.write(plan.new)
    try:
        bak = _backup_and_write(plan.path, plan.new)
    except OSError as exc:
        print(f"error: could not write {plan.path}: {exc}", file=stderr)
        return 1
    print(f"wrote {label}", file=stdout)
    if bak is not None:
        print(f"backup {plan.client_id}: {bak}", file=stdout)
    return 0
