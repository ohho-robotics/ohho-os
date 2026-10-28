"""HIL harness for the four bench tests (OHH-35).

Importing this module does not open a serial port, a DDS interface, a camera,
or a lidar. Hardware runs only from :func:`execute` after every opt-in gate
passes, and then only by spawning ``unittest discover`` on ``tests/hil``.

The bench tests construct adapters directly. They are not wrapped in
``SafetyGate``. The motion they command is the pulse below — do not raise it.
"""

from __future__ import annotations

import argparse
import os
import platform
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence

ENV_HIL = "OHHO_HIL"
# Device variables for a plugin's bench cases are filled from that plugin.
ENV_BASE: str = ""
ENV_ARM: str = ""
ENV_GO2 = "OHHO_GO2_IFACE"
ENV_BAG = "OHHO_HIL_BAG"
ENV_OPERATOR = "OHHO_HIL_OPERATOR"

# Bench pulse shared with tests/hil. The HIL path is not clamped by SafetyGate.
HIL_BASE_VX = 0.05
HIL_BASE_VY = 0.0
HIL_BASE_W = 0.0
HIL_GRIPPER_OPEN = 0.2
HIL_GRIPPER_SHUT = 0.0
HIL_GO2_VX = 0.0
HIL_GO2_VY = 0.0
HIL_GO2_W = 0.0

PREFLIGHT_CHECKS: tuple[str, ...] = (
    "Batteries",
    "Wiring",
    "E-stop",
    "Lidar",
    "Cameras",
    "Pi 5 Jazzy image",
)

_RESULT_LINE = re.compile(
    r"^(?P<head>.+?) \.\.\. (?P<status>ok|FAIL|ERROR|skipped)\b(?P<rest>.*)$"
)

Runner = Callable[[list[str], dict[str, str], str], "subprocess.CompletedProcess[str]"]


@dataclass(frozen=True)
class HilCase:
    """One of the four bench scenarios."""

    case_id: str
    title: str
    adapter: str
    unittest_class: str
    required_env: tuple[str, ...]


_UNITREE_CASE = HilCase(
    "unitree_go2",
    "Unitree Go2 drive and read",
    "unitree DDS",
    "TestHilUnitreeGo2",
    (ENV_GO2,),
)

# Plugin benches are appended in :func:`_apply_benches` before the Go2 case.
CASES: tuple[HilCase, ...] = ()
_REPORT_FIELDS: tuple[tuple[str, str], ...] = ()
_MOCKS: dict[str, Callable[[], str]] = {}


@dataclass
class HilCaseResult:
    """Outcome of one scenario, mock or hardware."""

    case_id: str
    title: str
    adapter: str
    status: str
    detail: str


@dataclass
class HilReport:
    """Fields rendered into ``docs/hil-results.md``."""

    when_utc: str
    operator: str
    host: str
    revision: str
    mode: str
    hil_env: str
    base_port: str
    arm_port: str
    go2_iface: str
    bag: str
    preflight: list[tuple[str, str, str]]
    cases: list[HilCaseResult]
    notes: str


def repo_root() -> Path:
    """Repository root (parent of the ``ohho`` package)."""
    return Path(__file__).resolve().parents[1]


def template_path(root: Optional[Path] = None) -> Path:
    """Committed empty results file."""
    return (root or repo_root()) / "docs" / "hil-results.md"


def hardware_skip_reason(
    required: Sequence[str],
    env: Optional[Mapping[str, str]] = None,
) -> Optional[str]:
    """Why a bench test must skip, or ``None`` when it may run.

    ``OHHO_HIL=1`` alone is not enough. Each device variable has to be set
    to a non-empty string. Unset variables are skipped rather than filled
    with ``/dev/ttyUSB0``, ``/dev/ttyACM0``, or ``eth0``.
    """
    source = os.environ if env is None else env
    needed = ", ".join(required)
    if source.get(ENV_HIL, "") != "1":
        return (
            f"set {ENV_HIL}=1 and explicit {needed} to run this hardware test; "
            "there is no default port"
        )
    missing = [name for name in required if not source.get(name, "").strip()]
    if missing:
        return (
            f"{ENV_HIL}=1 but missing explicit device env: "
            + ", ".join(missing)
            + "; there is no default port"
        )
    return None


def hardware_refused_reason(
    *,
    preflight_ok: bool,
    tty: bool,
    env: Mapping[str, str],
) -> Optional[str]:
    """Why ``--hardware`` must exit before spawning unittest."""
    if not preflight_ok:
        return (
            "refusing hardware: pass --preflight-ok after docs/hil-preflight.md "
            "(batteries, wiring, e-stop, lidar, cameras, Pi 5 Jazzy image)"
        )
    if env.get(ENV_HIL, "") != "1":
        return f"refusing hardware: set {ENV_HIL}=1 on this command"
    ready = [
        case for case in CASES if hardware_skip_reason(case.required_env, env) is None
    ]
    if not ready:
        return (
            "refusing hardware: set at least one explicit device variable "
            f"({ENV_BASE}, {ENV_ARM}, {ENV_GO2}); there is no default port"
        )
    if not tty:
        return (
            "refusing hardware: need an interactive terminal so the operator "
            "is present (ssh -t). The harness did not open a device"
        )
    return None


def dry_run_text() -> str:
    """Operator-facing text for the default command. No device access."""
    return (
        "OhhO HIL harness (OHH-35)\n"
        "\n"
        "No device was opened. This command did not touch a serial port, a DDS\n"
        "interface, a camera, or a lidar, and it did not command motion.\n"
        "\n"
        f"The four bench tests stay skipped until {ENV_HIL}=1 and the matching\n"
        "device variable are both set. Unset variables are skipped.\n"
        "\n"
        "Before hardware, complete docs/hil-preflight.md:\n"
        "  batteries, wiring, e-stop, lidar, cameras, Pi 5 Jazzy image\n"
        "Record the 2-minute MCAP with docs/hil-bag.md.\n"
        "\n"
        "These tests construct adapters directly. They are not wrapped in\n"
        "SafetyGate. Keep a physical e-stop in hand. The pulse is "
        f"{HIL_BASE_VX} m/s on the base, gripper {HIL_GRIPPER_OPEN} rad, "
        "and zero velocity on the Go2.\n"
        "\n"
        "Mock (safe anywhere, including CI):\n"
        "  ohho hil --mock\n"
        "\n"
        "Hardware, from an interactive terminal, after the checklist:\n"
        f"  {ENV_HIL}=1 {ENV_BASE}=<base> {ENV_ARM}=<arm> {ENV_GO2}=<iface> \\\n"
        "    ohho hil --hardware --preflight-ok\n"
        "\n"
        "That writes docs/hil-results.md. Omit a variable to skip that device.\n"
        "Example device names are in docs/hil-preflight.md. Use the nodes you\n"
        "confirmed on this machine.\n"
    )


def blank_report() -> HilReport:
    """Empty template. Every result cell is blank."""
    return HilReport(
        when_utc="",
        operator="",
        host="",
        revision="",
        mode="",
        hil_env="",
        base_port="",
        arm_port="",
        go2_iface="",
        bag="",
        preflight=[(name, "", "") for name in PREFLIGHT_CHECKS],
        cases=[
            HilCaseResult(case.case_id, case.title, case.adapter, "", "")
            for case in CASES
        ],
        notes="",
    )


def template_text() -> str:
    """Markdown for the committed empty results file."""
    return render_report(blank_report())


def render_report(report: HilReport) -> str:
    """Render a report in the ``docs/hil-results.md`` shape."""
    lines = [
        "# HIL results (OHH-35)",
        "",
        "Bench record for the four hardware-in-the-loop tests in `tests/hil/test_hil.py`.",
        "The committed copy of this file is the empty template. On the bench,",
        "`ohho hil --hardware --preflight-ok` rewrites it after `docs/hil-preflight.md`.",
        "CI runs `ohho hil --mock` and leaves this file empty.",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Date (UTC) | {_cell(report.when_utc)} |",
        f"| Operator | {_cell(report.operator)} |",
        f"| Host | {_cell(report.host)} |",
        f"| Revision | {_cell(report.revision)} |",
        f"| Mode | {_cell(report.mode)} |",
        f"| OHHO_HIL | {_cell(report.hil_env)} |",
    ]
    for label, attr in _REPORT_FIELDS:
        lines.append(f"| {label} | {_cell(getattr(report, attr))} |")
    lines.extend(
        [
            f"| Go2 interface | {_cell(report.go2_iface)} |",
            f"| Bag (2-min MCAP) | {_cell(report.bag)} |",
            "",
            "## Pre-flight",
            "",
            "Checklist: `docs/hil-preflight.md`. The harness does not measure these.",
            "",
            "| Check | Result | Notes |",
            "|---|---|---|",
        ]
    )
    for name, result, note in report.preflight:
        lines.append(f"| {name} | {_cell(result)} | {_cell(note)} |")
    lines.extend(
        [
            "",
            "## Tests",
            "",
            "| Test | Adapter | Status | Detail |",
            "|---|---|---|---|",
        ]
    )
    for case in report.cases:
        lines.append(
            f"| {case.title} | {case.adapter} | {_cell(case.status)} | {_cell(case.detail)} |"
        )
    lines.extend(["", "## Notes", ""])
    if report.notes:
        lines.append(report.notes)
        lines.append("")
    return "\n".join(lines) + "\n"


def parse_unittest_verbose(text: str) -> dict[str, tuple[str, str]]:
    """Map case id → (PASS|FAIL|SKIP, detail) from ``unittest -v`` output."""
    found: dict[str, tuple[str, str]] = {}
    for line in text.splitlines():
        match = _RESULT_LINE.match(line.strip())
        if match is None:
            continue
        head = match.group("head")
        status = match.group("status")
        rest = match.group("rest").strip()
        if len(rest) >= 2 and rest[0] == rest[-1] and rest[0] in {"'", '"'}:
            rest = rest[1:-1]
        for case in CASES:
            if case.unittest_class not in head:
                continue
            if status == "ok":
                found[case.case_id] = ("PASS", rest or "ok")
            elif status == "skipped":
                found[case.case_id] = ("SKIP", rest or "skipped")
            else:
                detail = _failure_detail(text, case.unittest_class) or status
                found[case.case_id] = ("FAIL", detail)
    return found


def results_from_unittest(
    text: str,
    env: Mapping[str, str],
) -> list[HilCaseResult]:
    """Turn a unittest log into one result per bench case."""
    parsed = parse_unittest_verbose(text)
    results: list[HilCaseResult] = []
    for case in CASES:
        if case.case_id in parsed:
            status, detail = parsed[case.case_id]
        else:
            reason = hardware_skip_reason(case.required_env, env)
            if reason:
                status, detail = "SKIP", reason
            else:
                status, detail = "FAIL", "no unittest result"
        results.append(
            HilCaseResult(case.case_id, case.title, case.adapter, status, detail)
        )
    return results


def run_mock_suite() -> list[HilCaseResult]:
    """Run all four scenarios on fake transports. Does not read device env."""
    results: list[HilCaseResult] = []
    for case in CASES:
        fn = _MOCKS[case.case_id]
        try:
            detail = fn()
        except Exception as exc:
            results.append(
                HilCaseResult(
                    case.case_id,
                    case.title,
                    case.adapter,
                    "FAIL",
                    f"{type(exc).__name__}: {exc}",
                )
            )
            continue
        results.append(
            HilCaseResult(case.case_id, case.title, case.adapter, "PASS", detail)
        )
    return results


def mock_unitree_go2() -> str:
    """Fake DDS version of ``TestHilUnitreeGo2``. Sends zero velocity only."""
    from .adapters.unitree import UnitreeDdsTransport
    from .registry import get_spec
    from .schema import Velocity

    _assert_go2_pulse()
    state = {
        "position": [1.25, 0.0, 0.0],
        "velocity": [0.0, 0.0, 0.0],
        "yaw_speed": 0.0,
        "imu_rpy": [0.0, 0.0, 0.3],
        "battery": 0.8,
        "error_code": 0,
    }
    client = _RecordingSport()
    transport = UnitreeDdsTransport(
        get_spec("unitree-go2"),
        "mock0",
        state_factory=lambda: lambda: state,
    )
    if transport._state_factory is None:
        raise AssertionError("Go2 mock must inject state_factory")
    transport.connect()
    try:
        transport._on_state(state)
        reading = transport.read()
        if reading.odom is None or abs(reading.odom.x - 1.25) > 1e-6:
            raise AssertionError("Go2 odom did not come from the fake state")
        transport._client = client
        transport.send_velocity(Velocity(HIL_GO2_VX, HIL_GO2_VY, HIL_GO2_W))
        if client.calls != [("Move", 0.0, 0.0, 0.0)]:
            raise AssertionError(f"Go2 command was not zero velocity: {client.calls}")
    finally:
        transport.disconnect()
    if ("StopMove",) not in client.calls:
        raise AssertionError("Go2 disconnect did not call StopMove")
    return "odom.x=1.250; Move(0, 0, 0); StopMove on disconnect"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Flags shared by ``ohho hil`` and ``scripts/hil_run.py``."""
    parser.add_argument(
        "--mock",
        action="store_true",
        help="run the four scenarios on fake transports (no hardware)",
    )
    parser.add_argument(
        "--hardware",
        action="store_true",
        help="run tests/hil on real devices (requires every opt-in gate)",
    )
    parser.add_argument(
        "--preflight-ok",
        action="store_true",
        help="confirm docs/hil-preflight.md is done",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the checklist and the bench command, then exit",
    )
    parser.add_argument(
        "--out",
        default="",
        help="results markdown path (hardware default: docs/hil-results.md)",
    )
    parser.add_argument(
        "--notes",
        default="",
        help="extra text for the Notes section",
    )


def build_parser() -> argparse.ArgumentParser:
    """Parser for ``python -m`` / ``scripts/hil_run.py``."""
    parser = argparse.ArgumentParser(
        prog="ohho hil",
        description="HIL harness. Default is a dry run. Hardware is opt-in.",
    )
    add_arguments(parser)
    return parser


def execute(
    args: argparse.Namespace,
    *,
    env: Optional[Mapping[str, str]] = None,
    tty: Optional[bool] = None,
    runner: Optional[Runner] = None,
    root: Optional[Path] = None,
) -> int:
    """Run dry-run, mock, or hardware. Hardware is refused until every gate passes."""
    source = os.environ if env is None else env
    if tty is None:
        tty = sys.stdin.isatty()
    base = root or repo_root()
    if args.dry_run or not (args.mock or args.hardware):
        print(dry_run_text(), end="")
        return 0
    if args.mock and args.hardware:
        print("pass only one of --mock and --hardware", file=sys.stderr)
        return 2
    if args.mock:
        return _run_mock(args, source, base)
    return _run_hardware(args, source, tty=tty, runner=runner, root=base)


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry for ``scripts/hil_run.py``."""
    return execute(build_parser().parse_args(argv))


def _run_mock(args: argparse.Namespace, env: Mapping[str, str], root: Path) -> int:
    out = _resolve_out(args.out)
    if out is not None and _same_path(out, template_path(root)):
        print(
            "refusing to overwrite docs/hil-results.md from a mock run; "
            "that file stays empty until the bench",
            file=sys.stderr,
        )
        return 2
    results = run_mock_suite()
    report = _filled_report(args, env, root, mode="mock", results=results)
    text = render_report(report)
    _print_summary(results)
    if out is None:
        print(
            "Results not written. Pass --out PATH to save them. "
            "Leave docs/hil-results.md empty until the bench run."
        )
    else:
        _write(out, text)
        print(f"Results written to: {out}")
    if any(item.status == "FAIL" for item in results):
        return 1
    return 0


def _run_hardware(
    args: argparse.Namespace,
    env: Mapping[str, str],
    *,
    tty: bool,
    runner: Optional[Runner],
    root: Path,
) -> int:
    reason = hardware_refused_reason(preflight_ok=args.preflight_ok, tty=tty, env=env)
    if reason:
        print(reason, file=sys.stderr)
        print("see: ohho hil", file=sys.stderr)
        return 2
    out = _resolve_out(args.out) or template_path(root)
    cmd = hardware_unittest_cmd(root)
    child_env = {key: value for key, value in env.items()}
    launch = runner or _default_runner
    try:
        completed = launch(cmd, child_env, str(root))
    except subprocess.TimeoutExpired as exc:
        text = _process_text(exc.stdout, exc.stderr) + "\nharness timeout"
        results = results_from_unittest(text, env)
        for index, item in enumerate(results):
            if item.status != "SKIP":
                results[index] = HilCaseResult(
                    item.case_id,
                    item.title,
                    item.adapter,
                    "FAIL",
                    "harness timeout; use the physical e-stop",
                )
        _write_hardware(args, env, root, out, results)
        print("hardware run timed out; use the physical e-stop", file=sys.stderr)
        return 1
    text = _process_text(completed.stdout, completed.stderr)
    results = results_from_unittest(text, env)
    _write_hardware(args, env, root, out, results)
    _print_summary(results)
    print(f"Results written to: {out}")
    if any(item.status == "FAIL" for item in results):
        return 1
    if results and all(item.status == "SKIP" for item in results):
        print("no hardware test ran", file=sys.stderr)
        return 3
    if completed.returncode not in (0, None):
        print(text, file=sys.stderr)
        return 1
    return 0


def _write_hardware(
    args: argparse.Namespace,
    env: Mapping[str, str],
    root: Path,
    out: Path,
    results: list[HilCaseResult],
) -> None:
    report = _filled_report(args, env, root, mode="hardware", results=results)
    _write(out, render_report(report))


def _filled_report(
    args: argparse.Namespace,
    env: Mapping[str, str],
    root: Path,
    *,
    mode: str,
    results: list[HilCaseResult],
) -> HilReport:
    if mode == "mock":
        preflight = [
            (name, "not run", "mock transports; checklist not performed")
            for name in PREFLIGHT_CHECKS
        ]
        base_port = arm_port = go2_iface = "not used"
        hil_env = "ignored"
        bag = "not recorded"
    else:
        preflight = [
            (
                name,
                "confirmed",
                "operator passed --preflight-ok; not measured by the harness",
            )
            for name in PREFLIGHT_CHECKS
        ]
        base_port = env.get(ENV_BASE, "").strip()
        arm_port = env.get(ENV_ARM, "").strip()
        go2_iface = env.get(ENV_GO2, "").strip()
        hil_env = "1" if env.get(ENV_HIL, "") == "1" else "unset"
        bag = env.get(ENV_BAG, "").strip()
    notes = (getattr(args, "notes", "") or "").strip()
    if not notes:
        if mode == "mock":
            notes = (
                "Mock run on fake transports. No serial port, DDS interface, "
                "camera, or lidar was opened."
            )
        else:
            notes = (
                "Filled by ohho hil --hardware. This command does not record "
                "the MCAP bag; see docs/hil-bag.md."
            )
    return HilReport(
        when_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        operator=env.get(ENV_OPERATOR, "").strip(),
        host=platform.node(),
        revision=_git_revision(root),
        mode=mode,
        hil_env=hil_env,
        base_port=base_port,
        arm_port=arm_port,
        go2_iface=go2_iface,
        bag=bag,
        preflight=preflight,
        cases=results,
        notes=notes,
    )


def _print_summary(results: Sequence[HilCaseResult]) -> None:
    passed = sum(item.status == "PASS" for item in results)
    failed = sum(item.status == "FAIL" for item in results)
    skipped = sum(item.status == "SKIP" for item in results)
    print(f"HIL: {passed} passed, {failed} failed, {skipped} skipped")
    for item in results:
        print(f"  {item.status:<4}  {item.title} — {item.detail}")


def hardware_unittest_cmd(root: Path) -> list[str]:
    """Load only ``tests/hil/test_hil.py``.

    ``tests`` is not an importable package from the repo root, so the bench
    command discovers that directory instead of ``python -m unittest
    tests.hil.test_hil``.
    """
    return [
        sys.executable,
        "-m",
        "unittest",
        "discover",
        "-s",
        str(root / "tests" / "hil"),
        "-p",
        "test_hil.py",
        "-v",
    ]


def _default_runner(
    cmd: list[str],
    env: dict[str, str],
    cwd: str,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        env=env,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )


def _process_text(stdout: object, stderr: object) -> str:
    out = stdout.decode() if isinstance(stdout, bytes) else (stdout or "")
    err = stderr.decode() if isinstance(stderr, bytes) else (stderr or "")
    return f"{out}\n{err}"


def _resolve_out(raw: str) -> Optional[Path]:
    if not raw:
        return None
    path = Path(raw)
    if not path.is_absolute():
        path = Path.cwd() / path
    return path


def _same_path(left: Path, right: Path) -> bool:
    try:
        return left.resolve() == right.resolve()
    except OSError:
        return False


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def _git_revision(root: Path) -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=root,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return out.strip()


def _cell(value: str) -> str:
    return _one_line(value) if value else ""


def _one_line(text: str, limit: int = 240) -> str:
    flat = " ".join(text.split()).replace("|", "/")
    if len(flat) <= limit:
        return flat
    return flat[: limit - 3] + "..."


def _failure_detail(text: str, class_name: str) -> str:
    chunks = re.split(r"\n=+\n", text)
    for chunk in chunks:
        head = chunk.lstrip()
        if class_name not in chunk:
            continue
        if not head.startswith(("FAIL:", "ERROR:")):
            continue
        lines = [line.strip() for line in chunk.splitlines() if line.strip()]
        if lines:
            return _one_line(lines[-1])
    return ""


def _assert_base_pulse() -> None:
    if (HIL_BASE_VX, HIL_BASE_VY, HIL_BASE_W) != (0.05, 0.0, 0.0):
        raise AssertionError("base pulse must stay 0.05 m/s straight")


def _assert_gripper_pulse() -> None:
    if (HIL_GRIPPER_OPEN, HIL_GRIPPER_SHUT) != (0.2, 0.0):
        raise AssertionError("gripper pulse must stay 0.2 rad then 0")


def _assert_go2_pulse() -> None:
    if (HIL_GO2_VX, HIL_GO2_VY, HIL_GO2_W) != (0.0, 0.0, 0.0):
        raise AssertionError("Go2 HIL command must stay zero velocity")


class _FakeSerial:
    """In-memory serial port. ``read`` returns nothing so the RX thread idles."""

    def __init__(self) -> None:
        self.written = bytearray()

    def write(self, data: bytes) -> int:
        self.written.extend(data)
        return len(data)

    def read(self, _n: int = 1) -> bytes:
        return b""

    def close(self) -> None:
        return None


class _FakeFeetechBus:
    """In-memory Feetech bus. Goal writes update present position."""

    def __init__(self, names: tuple[str, ...]) -> None:
        from .adapters.feetech import HOME_TICKS

        self.present = {name: HOME_TICKS for name in names}
        self.goals: dict[str, int] = {}
        self.torque: dict[str, int] = {}
        self.connected = False

    def connect(self) -> None:
        self.connected = True

    def disconnect(self) -> None:
        self.connected = False

    def read(self, key: str) -> dict[str, int]:
        if key == "Present_Position":
            return dict(self.present)
        return {}

    def write(self, key: str, values: dict[str, int]) -> None:
        if key == "Goal_Position":
            self.goals.update(values)
            self.present.update(values)
        elif key == "Torque_Enable":
            self.torque.update(values)


class _RecordingSport:
    """Records SportClient calls. Implements only Move and StopMove."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def Move(self, vx: float, vy: float, vyaw: float) -> None:
        self.calls.append(("Move", vx, vy, vyaw))

    def StopMove(self) -> None:
        self.calls.append(("StopMove",))


def _apply_benches() -> None:
    """Load bench cases from robot plugins, then append the Go2 case.

    Device-variable names and fake-transport checks stay in the plugin that
    owns them. Discovery does not call this.
    """
    global ENV_BASE, ENV_ARM, CASES, _MOCKS, _REPORT_FIELDS
    from ohho.plugins import loaded_plugins

    cases: list[HilCase] = []
    mocks: dict[str, Callable[[], str]] = {}
    fields: list[tuple[str, str]] = []
    exports: dict[str, object] = {}
    for info in loaded_plugins():
        if info.error is not None or info.plugin is None:
            continue
        hook = getattr(info.plugin, "hil_bench", None)
        if not callable(hook):
            continue
        data = hook()
        cases.extend(data["cases"])
        mocks.update(data["mocks"])
        fields.extend(data["report_fields"])
        exports.update(data["exports"])
    base = exports.get("ENV_BASE")
    arm = exports.get("ENV_ARM")
    if not isinstance(base, str) or not base.strip():
        raise RuntimeError("bench base device variable is not registered")
    if not isinstance(arm, str) or not arm.strip():
        raise RuntimeError("bench arm device variable is not registered")
    ENV_BASE = base
    ENV_ARM = arm
    for name, value in exports.items():
        if name.startswith("mock_") and callable(value):
            globals()[name] = value
    CASES = tuple(cases) + (_UNITREE_CASE,)
    _MOCKS = dict(mocks)
    _MOCKS[_UNITREE_CASE.case_id] = mock_unitree_go2
    _REPORT_FIELDS = tuple(fields)


_apply_benches()


if __name__ == "__main__":
    sys.exit(main())
