"""Robot-id resolver and per-robot profiles (OHH-314).

The robot id is chosen in one place, in this order:

1. an explicit CLI flag or function argument (``--robot``, positional robot)
2. the ``OHHO_ROBOT`` environment variable
3. ``robot = "..."`` in ``~/.ohho/config.toml``
4. ``sim``

``source`` on the result is ``flag``, ``env``, ``config`` or ``default``.

Per-robot settings live in ``~/.ohho/robots/<id>.toml`` or a repo-local
``ohho.toml`` (working directory). They may set the default transport URI,
sim backend, asset path and network interface. Safety keys may only tighten
the robot's current caps. A profile that raises a cap is refused at load.

``OHHO_HOME``, when set, is the directory that contains ``.ohho`` (tests pass
a temporary directory here). Nothing in this module reads a caller's real
home unless ``home`` and ``OHHO_HOME`` are both left unset.

This parser accepts the TOML subset those files use (strings, numbers, bools,
arrays, and dotted tables). The base install stays dependency-free, including
on Python 3.10 where ``tomllib`` is absent.
"""

from __future__ import annotations

import dataclasses
import math
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any

from .registry import RobotSpec, UnknownRobot, get_spec
from .robot_spec import RobotAssets
from .safety import SafetyConfig, SafetyProfile, normalize_action, profile_for

ROBOT_ENV = "OHHO_ROBOT"
HOME_ENV = "OHHO_HOME"
DEFAULT_ROBOT_ID = "sim"
SOURCE_FLAG = "flag"
SOURCE_ENV = "env"
SOURCE_CONFIG = "config"
SOURCE_DEFAULT = "default"

_PROFILE_KEYS = frozenset(
    {
        "robot",
        "transport",
        "sim_backend",
        "asset_path",
        "network_interface",
        "safety",
    }
)
_STRING_FIELDS = ("transport", "sim_backend", "asset_path", "network_interface")
# Higher numbers are looser caps. A profile may only set a smaller value.
_MAX_CAPS = frozenset(
    {
        "max_vx",
        "max_vy",
        "max_wz",
        "max_accel_x",
        "max_accel_y",
        "max_accel_wz",
        "max_joint_step",
        "cmd_timeout_s",
        "state_timeout_s",
        "tilt_limit_rad",
        "euler_limit_rad",
    }
)
_POSITIVE_CAPS = frozenset(
    {
        "cmd_timeout_s",
        "state_timeout_s",
        "tilt_limit_rad",
        "euler_limit_rad",
    }
)
_BARE_KEY = re.compile(r"[A-Za-z0-9_-]+")
_NUMBER = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?\Z")


class ConfigError(ValueError):
    """A robot config or profile cannot be read or is not allowed."""


class SafetyCapError(ConfigError):
    """A robot profile tried to raise a safety cap. Overrides may only tighten."""


@dataclass(frozen=True)
class ResolvedRobot:
    """A robot id and which switch won.

    ``source`` is ``flag``, ``env``, ``config`` or ``default``.
    """

    robot_id: str
    source: str

    def __post_init__(self) -> None:
        if self.source not in (SOURCE_FLAG, SOURCE_ENV, SOURCE_CONFIG, SOURCE_DEFAULT):
            raise ValueError(f"unknown robot source {self.source!r}")
        if not isinstance(self.robot_id, str) or not self.robot_id.strip():
            raise ValueError("robot id must be a non-empty string")


@dataclass(frozen=True)
class RobotProfile:
    """Per-robot defaults loaded from TOML. Not a compute :class:`HardwareProfile`.

    ``safety_overrides`` is empty when the file sets no safety keys. Constructing
    a profile with overrides validates them and refuses any cap that is looser
    than the robot's current safety profile.
    """

    robot_id: str
    transport: str | None = None
    sim_backend: str | None = None
    asset_path: str | None = None
    network_interface: str | None = None
    safety_overrides: Mapping[str, Any] = field(default_factory=dict)
    source: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.robot_id, str) or not self.robot_id.strip():
            raise ValueError("robot id must be a non-empty string")
        for name in _STRING_FIELDS:
            value = getattr(self, name)
            if value is not None and not isinstance(value, str):
                raise TypeError(f"{name} must be a string")
        raw = dict(self.safety_overrides)
        if raw:
            cleaned = _safety_overrides(
                self.robot_id, raw, self.source or self.robot_id
            )
        else:
            cleaned = {}
        object.__setattr__(self, "safety_overrides", MappingProxyType(cleaned))


def default_home() -> Path:
    """Directory that contains ``.ohho``.

    ``OHHO_HOME`` overrides the real user home so tests never need that tree.
    """
    override = os.environ.get(HOME_ENV)
    if override is not None and override.strip():
        return Path(override).expanduser()
    return Path.home()


def resolve_robot_id(
    flag: str | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    config_path: Path | str | None = None,
    home: Path | str | None = None,
) -> ResolvedRobot:
    """Resolve the robot id. Flag, then ``OHHO_ROBOT``, then config, then ``sim``.

    An empty flag or environment value is skipped. A missing config file is
    skipped. A present config file that is not valid TOML, or that sets
    ``robot`` to something other than a string, raises :class:`ConfigError`.
    ``home`` and ``config_path`` are injectable; when both are omitted the
    config file is ``<default_home>/.ohho/config.toml``.
    """
    chosen = _blank_to_none(flag)
    if chosen is not None:
        return ResolvedRobot(chosen, SOURCE_FLAG)

    env = os.environ if environ is None else environ
    chosen = _blank_to_none(env.get(ROBOT_ENV))
    if chosen is not None:
        return ResolvedRobot(chosen, SOURCE_ENV)

    path = _config_file(home, config_path)
    if path.is_file():
        data = load_toml(path)
        if not isinstance(data, dict):
            raise ConfigError(f"{path}: top level must be a table")
        unknown = set(data) - {"robot"}
        if unknown:
            raise ConfigError(
                f"{path}: unknown keys {sorted(unknown)} "
                "(this file only sets robot; per-robot settings belong in "
                "~/.ohho/robots/<id>.toml or ohho.toml)"
            )
        if "robot" in data:
            robot = data["robot"]
            if not isinstance(robot, str) or not robot.strip():
                raise ConfigError(f"{path}: robot must be a non-empty string")
            return ResolvedRobot(robot.strip(), SOURCE_CONFIG)
    return ResolvedRobot(DEFAULT_ROBOT_ID, SOURCE_DEFAULT)


def load_robot_profile(
    robot_id: str,
    *,
    home: Path | str | None = None,
    cwd: Path | str | None = None,
    path: Path | str | None = None,
) -> RobotProfile:
    """Load the profile for ``robot_id``.

    Search order when ``path`` is omitted: repo-local ``ohho.toml`` in ``cwd``
    (default: the process working directory), then
    ``<home>/.ohho/robots/<id>.toml``. A repo file overrides the home file
    field by field. When neither exists the profile is empty and ``source``
    is ``None``. ``path`` loads that file only.

    Safety overrides are checked against the robot's current safety profile.
    Raising any cap raises :class:`SafetyCapError`.
    """
    robot_id = _safe_robot_id(robot_id)
    if path is not None:
        file = Path(path)
        if not file.is_file():
            raise ConfigError(f"robot profile not found: {file}")
        section = _section_for(load_toml(file), robot_id, source=file, required=True)
        if section is None:
            raise ConfigError(f"{file}: no profile fields for {robot_id!r}")
        return _build_profile(robot_id, section, str(file))

    found: list[tuple[Path, dict[str, Any]]] = []
    home_file = _home_dir(home) / ".ohho" / "robots" / f"{robot_id}.toml"
    if home_file.is_file():
        section = _section_for(
            load_toml(home_file), robot_id, source=home_file, required=True
        )
        if section is None:
            raise ConfigError(f"{home_file}: no profile fields for {robot_id!r}")
        found.append((home_file, section))

    repo = _repo_file(cwd)
    if repo is not None:
        section = _section_for(load_toml(repo), robot_id, source=repo, required=False)
        if section is not None:
            found.append((repo, section))

    if not found:
        return RobotProfile(robot_id)

    # Home first so the repo file wins on keys it sets.
    found.sort(key=lambda item: item[0] == repo)
    merged: dict[str, Any] = {}
    sources: list[str] = []
    for file, section in found:
        merged = _merge_tables(merged, section)
        sources.append(str(file))
    return _build_profile(robot_id, merged, " + ".join(sources))


def format_resolved(
    resolved: ResolvedRobot, profile: RobotProfile | None = None
) -> str:
    """Text for ``ohho config show``: the robot id and its source, then profile fields."""
    lines = [f"robot: {resolved.robot_id}", f"source: {resolved.source}"]
    if profile is None:
        return "\n".join(lines)
    if profile.source:
        lines.append(f"profile: {profile.source}")
    if profile.transport:
        lines.append(f"transport: {profile.transport}")
    if profile.sim_backend:
        lines.append(f"sim_backend: {profile.sim_backend}")
    if profile.asset_path:
        lines.append(f"asset_path: {profile.asset_path}")
    if profile.network_interface:
        lines.append(f"network_interface: {profile.network_interface}")
    if profile.safety_overrides:
        lines.append("safety: " + ", ".join(sorted(profile.safety_overrides)))
    return "\n".join(lines)


def apply_profile_to_spec(spec: RobotSpec, profile: RobotProfile) -> RobotSpec:
    """Copy sim backend and asset path onto ``spec``. Other fields stay put.

    An ``.xml`` or ``.mjcf`` asset path is stored as MJCF. Any other path is
    stored as URDF. The path is not opened.
    """
    return spec_with_overrides(
        spec, sim_backend=profile.sim_backend, asset_path=profile.asset_path
    )


def spec_with_overrides(
    spec: RobotSpec,
    *,
    sim_backend: str | None = None,
    asset_path: str | None = None,
) -> RobotSpec:
    """Return ``spec`` with a sim backend and/or asset path filled in."""
    assets = spec.assets
    if asset_path:
        if not isinstance(assets, RobotAssets):
            assets = RobotAssets()
        lower = asset_path.lower()
        if lower.endswith((".xml", ".mjcf")):
            assets = replace(assets, mjcf=asset_path)
        else:
            assets = replace(assets, urdf=asset_path)
    backend = sim_backend if sim_backend else spec.sim_backend
    if assets is spec.assets and backend == spec.sim_backend:
        return spec
    return replace(spec, assets=assets, sim_backend=backend)


def merge_safety_overrides(
    config: SafetyConfig, robot_id: str, overrides: Mapping[str, Any]
) -> SafetyConfig:
    """Add profile caps onto a safety config.

    Numeric caps take the smaller of the config value and the profile value,
    so a profile cannot undo a tighter safety-config file. Non-numeric keys
    already set in the safety config are left as that file wrote them.
    ``SafetyGate`` is unchanged; this only builds the config object it already
    accepts.
    """
    if not overrides:
        return config
    profiles = {key: dict(value) for key, value in dict(config.profiles).items()}
    current = dict(profiles.get(robot_id, {}))
    for key, value in overrides.items():
        if key not in current:
            current[key] = value
            continue
        if _is_number(value) and _is_number(current[key]):
            current[key] = min(float(current[key]), float(value))
    profiles[robot_id] = current
    return replace(config, profiles=profiles)


def load_toml(path: Path | str) -> dict[str, Any]:
    """Read a TOML subset file. IO and syntax errors raise :class:`ConfigError`."""
    file = Path(path)
    try:
        text = file.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"cannot read {file}: {exc}") from exc
    return parse_toml(text, source=str(file))


def parse_toml(text: str, *, source: str = "<toml>") -> dict[str, Any]:
    """Parse the TOML subset used by OhhO config files."""
    if text.startswith("\ufeff"):
        text = text[1:]
    root: dict[str, Any] = {}
    current: list[str] = []
    logical = ""
    start_line = 0
    for line_no, raw in enumerate(text.splitlines(), start=1):
        piece = _strip_comment(raw).strip()
        if not piece and not logical:
            continue
        if not logical:
            start_line = line_no
        logical = f"{logical} {piece}".strip()
        if _brackets_open(logical):
            continue
        try:
            _parse_logical(logical, root, current, source, start_line)
        except ConfigError as exc:
            if str(exc).startswith(f"{source}:"):
                raise
            raise ConfigError(f"{source}:{start_line}: {exc}") from exc
        logical = ""
    if logical:
        raise ConfigError(f"{source}:{start_line}: unclosed '['")
    return root


def _parse_logical(
    logical: str,
    root: dict[str, Any],
    current: list[str],
    source: str,
    line_no: int,
) -> None:
    if logical.startswith("["):
        if not logical.endswith("]"):
            raise ConfigError(f"{source}:{line_no}: invalid table header")
        body = logical[1:-1].strip()
        if body.startswith("[") or body.endswith("]"):
            raise ConfigError(f"{source}:{line_no}: array-of-tables is not supported")
        parts = _split_key(body, source, line_no)
        _enter_table(root, parts, source, line_no)
        current[:] = parts
        return
    key, sep, raw_value = logical.partition("=")
    if not sep:
        raise ConfigError(f"{source}:{line_no}: expected key = value")
    parts = current + _split_key(key.strip(), source, line_no)
    value = _parse_value(raw_value.strip(), source, line_no)
    _assign(root, parts, value, source, line_no)


def _enter_table(
    root: dict[str, Any], parts: list[str], source: str, line_no: int
) -> None:
    node = root
    for part in parts:
        nxt = node.get(part)
        if nxt is None:
            nxt = {}
            node[part] = nxt
        elif not isinstance(nxt, dict):
            raise ConfigError(
                f"{source}:{line_no}: {'.'.join(parts)} conflicts with a value"
            )
        node = nxt


def _assign(
    root: dict[str, Any],
    parts: list[str],
    value: Any,
    source: str,
    line_no: int,
) -> None:
    node = root
    for part in parts[:-1]:
        nxt = node.get(part)
        if nxt is None:
            nxt = {}
            node[part] = nxt
        elif not isinstance(nxt, dict):
            raise ConfigError(
                f"{source}:{line_no}: {'.'.join(parts)} conflicts with a value"
            )
        node = nxt
    leaf = parts[-1]
    if leaf in node:
        raise ConfigError(f"{source}:{line_no}: duplicate key {'.'.join(parts)}")
    node[leaf] = value


def _split_key(text: str, source: str, line_no: int) -> list[str]:
    if not text:
        raise ConfigError(f"{source}:{line_no}: empty key")
    parts = text.split(".")
    for part in parts:
        if not _BARE_KEY.fullmatch(part):
            raise ConfigError(f"{source}:{line_no}: unsupported key {text!r}")
    return parts


def _brackets_open(text: str) -> bool:
    if text.startswith("["):
        return False
    return text.count("[") > text.count("]")


def _strip_comment(line: str) -> str:
    in_string = False
    quote = ""
    index = 0
    while index < len(line):
        char = line[index]
        if in_string:
            if char == "\\" and quote == '"':
                index += 2
                continue
            if char == quote:
                in_string = False
            index += 1
            continue
        if char in {'"', "'"}:
            in_string = True
            quote = char
            index += 1
            continue
        if char == "#":
            return line[:index]
        index += 1
    return line


def _parse_value(text: str, source: str, line_no: int) -> Any:
    if not text:
        raise ConfigError(f"{source}:{line_no}: missing value")
    value: Any
    if text[0] in {'"', "'"}:
        value, index = _parse_string(text, 0, source, line_no)
        _expect_end(text, index, source, line_no)
        return value
    if text[0] == "[":
        value, index = _parse_array(text, 0, source, line_no)
        _expect_end(text, index, source, line_no)
        return value
    token = text.split(None, 1)[0]
    if token in {"true", "false"}:
        _expect_end(text, len(token), source, line_no)
        return token == "true"
    if _NUMBER.fullmatch(token):
        _expect_end(text, len(token), source, line_no)
        if any(mark in token for mark in (".", "e", "E")):
            return float(token)
        return int(token)
    raise ConfigError(f"{source}:{line_no}: unsupported value {text!r}")


def _expect_end(text: str, index: int, source: str, line_no: int) -> None:
    rest = text[index:].strip()
    if rest:
        raise ConfigError(f"{source}:{line_no}: unexpected {rest!r}")


def _parse_string(text: str, index: int, source: str, line_no: int) -> tuple[str, int]:
    quote = text[index]
    index += 1
    chars: list[str] = []
    while index < len(text):
        char = text[index]
        if char == quote:
            return "".join(chars), index + 1
        if char == "\\" and quote == '"':
            index += 1
            if index >= len(text):
                raise ConfigError(f"{source}:{line_no}: unterminated string")
            escaped = text[index]
            mapped = {"n": "\n", "t": "\t", "r": "\r", "\\": "\\", '"': '"'}.get(
                escaped
            )
            if mapped is None:
                raise ConfigError(f"{source}:{line_no}: unsupported escape \\{escaped}")
            chars.append(mapped)
        elif char == "\n":
            raise ConfigError(f"{source}:{line_no}: unterminated string")
        else:
            chars.append(char)
        index += 1
    raise ConfigError(f"{source}:{line_no}: unterminated string")


def _parse_array(text: str, index: int, source: str, line_no: int) -> tuple[list, int]:
    index += 1
    items: list[Any] = []
    while index < len(text):
        while index < len(text) and text[index] in " \t\r\n":
            index += 1
        if index < len(text) and text[index] == "]":
            return items, index + 1
        if index >= len(text):
            break
        value: Any
        if text[index] in {'"', "'"}:
            value, index = _parse_string(text, index, source, line_no)
        elif text[index] == "[":
            value, index = _parse_array(text, index, source, line_no)
        else:
            end = index
            while end < len(text) and text[end] not in ",]":
                end += 1
            value = _parse_value(text[index:end].strip(), source, line_no)
            index = end
        items.append(value)
        while index < len(text) and text[index] in " \t\r\n":
            index += 1
        if index < len(text) and text[index] == ",":
            index += 1
            continue
        if index < len(text) and text[index] == "]":
            return items, index + 1
        raise ConfigError(f"{source}:{line_no}: expected ',' or ']' in array")
    raise ConfigError(f"{source}:{line_no}: unclosed array")


def _config_file(home: Path | str | None, config_path: Path | str | None) -> Path:
    if config_path is not None:
        return Path(config_path)
    return _home_dir(home) / ".ohho" / "config.toml"


def _home_dir(home: Path | str | None) -> Path:
    if home is not None:
        return Path(home)
    return default_home()


def _repo_file(cwd: Path | str | None) -> Path | None:
    directory = Path.cwd() if cwd is None else Path(cwd)
    candidate = directory / "ohho.toml"
    if candidate.is_file():
        return candidate
    return None


def _blank_to_none(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ConfigError(f"robot id must be a string, got {type(value).__name__}")
    text = value.strip()
    return text or None


def _safe_robot_id(robot_id: str) -> str:
    text = _blank_to_none(robot_id)
    if text is None or text in {".", ".."} or any(char in text for char in "/\\\x00"):
        raise ConfigError(f"invalid robot id {robot_id!r}")
    return text


def _section_for(
    data: dict[str, Any], robot_id: str, *, source: Path, required: bool
) -> dict[str, Any] | None:
    if not isinstance(data, dict):
        raise ConfigError(f"{source}: top level must be a table")
    robots = data.get("robots")
    if robots is not None and not isinstance(robots, dict):
        raise ConfigError(f"{source}: robots must be a table")
    if isinstance(robots, dict) and robot_id in robots:
        section = robots[robot_id]
        if not isinstance(section, dict):
            raise ConfigError(f"{source}: robots.{robot_id} must be a table")
        _reject_unknown(section, source)
        _reject_unknown(
            {key: value for key, value in data.items() if key != "robots"}, source
        )
        return section
    flat = {key: value for key, value in data.items() if key != "robots"}
    _reject_unknown(flat, source)
    named = data.get("robot")
    if named is not None and not isinstance(named, str):
        raise ConfigError(f"{source}: robot must be a string")
    if named is not None and named != robot_id:
        if required:
            raise ConfigError(f"{source} is for robot {named!r}, not {robot_id!r}")
        return None
    if not _has_profile_fields(flat):
        if required:
            raise ConfigError(f"{source}: no profile fields for {robot_id!r}")
        return None
    return flat


def _has_profile_fields(data: Mapping[str, Any]) -> bool:
    return any(key in data for key in _PROFILE_KEYS - {"robot"})


def _reject_unknown(data: Mapping[str, Any], source: Path) -> None:
    unknown = set(data) - _PROFILE_KEYS
    if unknown:
        raise ConfigError(f"{source}: unknown keys {sorted(unknown)}")


def _merge_tables(base: dict[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overlay.items():
        if (
            key == "safety"
            and isinstance(value, dict)
            and isinstance(merged.get("safety"), dict)
        ):
            safety = dict(merged["safety"])
            safety.update(value)
            merged["safety"] = safety
        else:
            merged[key] = value
    return merged


def _build_profile(
    robot_id: str, table: Mapping[str, Any], source: str
) -> RobotProfile:
    strings = {
        name: _optional_str(table.get(name), name, source) for name in _STRING_FIELDS
    }
    safety = table.get("safety", {})
    if safety is None:
        safety = {}
    if not isinstance(safety, dict):
        raise ConfigError(f"{source}: safety must be a table")
    return RobotProfile(
        robot_id=robot_id,
        transport=strings["transport"],
        sim_backend=strings["sim_backend"],
        asset_path=strings["asset_path"],
        network_interface=strings["network_interface"],
        safety_overrides=safety,
        source=source,
    )


def _optional_str(value: Any, name: str, source: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ConfigError(f"{source}: {name} must be a string")
    text = value.strip()
    return text or None


def _safety_overrides(
    robot_id: str, raw: Mapping[str, Any], source: str
) -> dict[str, Any]:
    known = {item.name for item in dataclasses.fields(SafetyProfile)}
    unknown = set(raw) - known
    if unknown:
        raise ConfigError(f"{source}: unknown safety keys {sorted(unknown)}")
    try:
        spec: RobotSpec | None = get_spec(robot_id)
    except UnknownRobot:
        spec = None
    baseline = profile_for(robot_id, spec)
    cleaned: dict[str, Any] = {}
    for key, value in raw.items():
        if key in _MAX_CAPS:
            number = _finite_number(f"safety.{key}", value, source)
            if key in _POSITIVE_CAPS and number <= 0:
                raise ConfigError(f"{source}: safety.{key} must be > 0")
            base = float(getattr(baseline, key))
            if number > base + 1e-9:
                raise SafetyCapError(
                    f"profile for {robot_id!r} raises safety cap {key} "
                    f"from {base} to {number} (overrides may only tighten)"
                )
            cleaned[key] = number
        elif key == "effort_limits":
            efforts = _tighter_efforts(robot_id, baseline, value, source)
            if efforts:
                cleaned[key] = efforts
        elif key == "joint_limits":
            joints = _tighter_joints(robot_id, baseline, value, source)
            if joints:
                cleaned[key] = joints
        elif key == "allowlist":
            cleaned[key] = _tighter_allowlist(robot_id, baseline, value, source)
        elif key == "require_orientation":
            flag = _as_bool(f"safety.{key}", value, source)
            if baseline.require_orientation and not flag:
                raise SafetyCapError(
                    f"profile for {robot_id!r} loosens require_orientation "
                    "(overrides may only tighten)"
                )
            if flag != baseline.require_orientation:
                cleaned[key] = flag
        elif key in {"estop_damp", "lying_body_height"}:
            if _changes_gate_policy(key, value, baseline, source):
                raise SafetyCapError(
                    f"profile for {robot_id!r} cannot change {key} "
                    "(overrides may only tighten caps)"
                )
        else:
            raise ConfigError(
                f"{source}: safety.{key} cannot be set from a robot profile"
            )
    if cleaned:
        try:
            dataclasses.replace(baseline, **cleaned)
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"profile for {robot_id!r}: {exc}") from exc
    return cleaned


def _changes_gate_policy(
    key: str, value: Any, baseline: SafetyProfile, source: str
) -> bool:
    if key == "estop_damp":
        return _as_bool(f"safety.{key}", value, source) != baseline.estop_damp
    number = _finite_number(f"safety.{key}", value, source)
    return abs(number - float(getattr(baseline, key))) > 1e-9


def _tighter_efforts(
    robot_id: str, baseline: SafetyProfile, value: Any, source: str
) -> dict[str, float]:
    if not isinstance(value, dict):
        raise ConfigError(f"{source}: safety.effort_limits must be a table")
    cleaned: dict[str, float] = {}
    for name, limit in value.items():
        number = _finite_number(f"safety.effort_limits.{name}", limit, source)
        if name not in baseline.effort_limits:
            raise SafetyCapError(
                f"profile for {robot_id!r} raises safety cap "
                f"effort_limits[{name!r}] "
                "(no effort cap is set; adding one would allow torque; "
                "overrides may only tighten)"
            )
        base = float(baseline.effort_limits[name])
        if number > base + 1e-9:
            raise SafetyCapError(
                f"profile for {robot_id!r} raises safety cap "
                f"effort_limits[{name!r}] from {base} to {number} "
                "(overrides may only tighten)"
            )
        cleaned[str(name)] = number
    return cleaned


def _tighter_joints(
    robot_id: str, baseline: SafetyProfile, value: Any, source: str
) -> dict[str, tuple[float, float]]:
    if not isinstance(value, dict):
        raise ConfigError(f"{source}: safety.joint_limits must be a table")
    cleaned: dict[str, tuple[float, float]] = {}
    for name, limit in value.items():
        if (
            isinstance(limit, bool)
            or not isinstance(limit, (list, tuple))
            or len(limit) != 2
        ):
            raise ConfigError(f"{source}: safety.joint_limits.{name} must be [lo, hi]")
        lo = _signed_number(f"safety.joint_limits.{name} lo", limit[0], source)
        hi = _signed_number(f"safety.joint_limits.{name} hi", limit[1], source)
        if lo > hi:
            raise ConfigError(f"{source}: safety.joint_limits.{name} has lo > hi")
        if name not in baseline.joint_limits:
            raise SafetyCapError(
                f"profile for {robot_id!r} raises safety cap "
                f"joint_limits[{name!r}] (joint is not allowed; adding a range would allow motion)"
            )
        base_lo, base_hi = (float(item) for item in baseline.joint_limits[name])
        if lo < base_lo - 1e-9 or hi > base_hi + 1e-9:
            raise SafetyCapError(
                f"profile for {robot_id!r} raises safety cap "
                f"joint_limits[{name!r}] from ({base_lo}, {base_hi}) to ({lo}, {hi}) "
                "(overrides may only tighten)"
            )
        cleaned[str(name)] = (lo, hi)
    return cleaned


def _tighter_allowlist(
    robot_id: str, baseline: SafetyProfile, value: Any, source: str
) -> frozenset[str]:
    if isinstance(value, (str, bytes)) or not isinstance(
        value, (list, tuple, set, frozenset)
    ):
        raise ConfigError(f"{source}: safety.allowlist must be an array of strings")
    proposed: list[str] = []
    for action in value:
        if not isinstance(action, str):
            raise ConfigError(f"{source}: safety.allowlist must be an array of strings")
        name = normalize_action(action)
        if name not in baseline.allowlist:
            raise SafetyCapError(
                f"profile for {robot_id!r} raises safety cap allowlist "
                f"with {name!r} (overrides may only tighten)"
            )
        proposed.append(name)
    return frozenset(proposed)


def _signed_number(where: str, value: Any, source: str) -> float:
    if not _is_number(value):
        raise ConfigError(f"{source}: {where} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise ConfigError(f"{source}: {where} must be a finite number")
    return number


def _finite_number(where: str, value: Any, source: str) -> float:
    number = _signed_number(where, value, source)
    if number < 0:
        raise ConfigError(f"{source}: {where} must be a finite number >= 0")
    return number


def _as_bool(where: str, value: Any, source: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{source}: {where} must be true or false")
    return value


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)
