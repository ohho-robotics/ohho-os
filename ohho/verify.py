"""``ohho verify --sim`` — drive a robot in simulation and report the distance.

The only backend this module constructs is ``SimTransport`` (``sim://``).
A non-sim URI from a flag, environment variable, or robot config is refused
by :func:`non_sim_refusal` before any transport is created. ``--sim`` is
required; nothing here defaults to hardware.

The sim link is wrapped in :class:`~ohho.safety.SafetyGate` so the same caps,
300 ms command watchdog, and action allowlist apply as on a real link.
``Robot.connect`` does not wrap ``sim://`` (the gate is opted out there).
Wrapping a simulator explicitly is the supported path: ``arm()`` enables
motion without a TTY prompt because ``requires_hardware_arming`` is false for
``protocol == "simulated"``. This module calls that ``arm()``. It does not
read ``OHHO_ARM_HARDWARE``, does not prompt, does not call ``release_stop``,
and does not load a safety config that would change the built-in limits.

Velocity is refreshed faster than the 300 ms deadman, then an explicit stop
is sent. The stop also runs from ``finally``, including when the drive raises
or the process is interrupted.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Optional, TextIO
from urllib.request import urlopen

from .adapters.sim import SimTransport
from .auth import AuthError, console_origin
from .registry import UnknownRobot, get_spec
from .safety import CMD_TIMEOUT_S, SafetyConfig, SafetyGate
from .schema import Velocity

# Faster than the SafetyGate command watchdog (CMD_TIMEOUT_S, 300 ms).
COMMAND_PERIOD_S = 0.05
DRIVE_MPS = 0.1
DRIVE_S = 1.0
TOLERANCE_M = 0.05
_ARM_WAIT_S = 2.0

# Names an agent or a shell might use to point a command at hardware.
# A value of sim:// is ignored. Any other non-empty value is refused.
TRANSPORT_ENV_KEYS = (
    "OHHO_TRANSPORT",
    "OHHO_ROBOT_TRANSPORT",
    "OHHO_VERIFY_TRANSPORT",
    "OHHO_URI",
)
_ROBOT_CONFIG_ENV = "OHHO_ROBOT_CONFIG"
_CONFIG_URI_KEYS = ("transport", "uri", "url")


class VerifyRefused(Exception):
    """verify was pointed at something other than ``sim://``, or ``--sim`` was omitted."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.code = 2


def is_sim_uri(value: str) -> bool:
    """True for ``sim://`` and the ``simulated://`` alias. Nothing else."""
    text = value.strip().lower()
    return text in {"sim://", "sim:", "simulated://"} or text.startswith("sim://")


def non_sim_refusal(
    environ: Mapping[str, str],
    *,
    transport: Optional[str] = None,
    robot: str | None = None,
) -> Optional[str]:
    """A refusal message when something tries to select a non-sim URI.

    The message names the flag, variable, or config key. It does not repeat
    the URI. ``None`` means nothing is requesting a non-sim backend.
    """
    if transport is not None and transport.strip() and not is_sim_uri(transport):
        return "ohho verify accepts only sim:// (refusing --transport)"
    robot_text = (robot or "").strip()
    if "://" in robot_text and not is_sim_uri(robot_text):
        return "ohho verify accepts only sim:// (refusing --robot)"
    for key in TRANSPORT_ENV_KEYS:
        raw = environ.get(key)
        if isinstance(raw, str) and raw.strip() and not is_sim_uri(raw):
            return f"ohho verify accepts only sim:// (refusing {key})"
    config = environ.get(_ROBOT_CONFIG_ENV)
    if isinstance(config, str) and config.strip():
        reason = _robot_config_refusal(config.strip())
        if reason is not None:
            return reason
    return None


def _robot_config_refusal(path_text: str) -> Optional[str]:
    """Refuse a robot-config file that names a non-sim transport URI.

    A manifest that only names a hardware adapter (and no URI) is not a
    request to connect to that adapter. verify still builds ``sim://``.
    """
    path = Path(path_text)
    try:
        if not path.is_file():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return "ohho verify accepts only sim:// (robot config is unreadable)"
    if not isinstance(data, dict):
        return "ohho verify accepts only sim:// (robot config is unreadable)"
    for key in _CONFIG_URI_KEYS:
        raw = data.get(key)
        if isinstance(raw, str) and raw.strip() and not is_sim_uri(raw):
            return "ohho verify accepts only sim:// (refusing robot config transport)"
    return None


def _no_prompt(_text: str) -> str:
    """Never accept an arming phrase. Sim arming does not ask; hardware must not."""
    return ""


def _publish(sim: SimTransport) -> None:
    """Hand the gate a fresh snapshot so the state watchdog has a timestamp.

    ``SimTransport.read`` does not notify subscribers. The gate only hears
    telemetry that is emitted.
    """
    sim._emit_telemetry(sim.read())


@dataclass
class ConsoleProbe:
    """Result of the optional console ``/sim`` GET. Never a crash."""

    status: str
    detail: str

    def as_dict(self) -> dict[str, str]:
        return {"status": self.status, "detail": self.detail}


@dataclass
class VerifyResult:
    """Distance moved on ``sim://``. ``gate`` is not part of the JSON object."""

    ok: bool
    robot: str
    transport: str
    distance_m: float
    tolerance_m: float
    commanded_mps: float
    duration_s: float
    detail: str
    gated: bool
    console: Optional[ConsoleProbe] = None
    gate: Optional[SafetyGate] = None

    def as_dict(self) -> dict[str, object]:
        body: dict[str, object] = {
            "ok": self.ok,
            "robot": self.robot,
            "transport": self.transport,
            "distance_m": self.distance_m,
            "tolerance_m": self.tolerance_m,
            "commanded_mps": self.commanded_mps,
            "duration_s": self.duration_s,
            "detail": self.detail,
            "console": None if self.console is None else self.console.as_dict(),
        }
        return body


def probe_console(
    origin: str,
    opener: Callable[..., object] = urlopen,
    timeout_s: float = 5.0,
) -> ConsoleProbe:
    """GET ``{origin}/sim``. 404 and network errors are ``skip``, not a crash.

    Production ``/sim`` is 404 today. A 200 is ``pass``. The response body
    is not read into the report.
    """
    url = origin.rstrip("/") + "/sim"
    request = urllib.request.Request(url, method="GET")
    try:
        response = opener(request, timeout=timeout_s)
    except urllib.error.HTTPError as exc:
        return ConsoleProbe("skip", f"unavailable (HTTP {exc.code})")
    except Exception:
        return ConsoleProbe("skip", "unavailable")
    status = 0
    try:
        if hasattr(response, "__enter__"):
            with response:  # type: ignore[union-attr]
                status = int(getattr(response, "status", 0) or response.getcode())
        else:
            status = int(getattr(response, "status", 0) or response.getcode())
    except Exception:
        return ConsoleProbe("skip", "unavailable")
    if status == 200:
        return ConsoleProbe("pass", "reachable")
    if status == 0:
        return ConsoleProbe("skip", "unavailable")
    return ConsoleProbe("skip", f"unavailable (HTTP {status})")


def _stop(gate: SafetyGate) -> None:
    """Zero velocity. Safe to call more than once. Does not release an e-stop."""
    try:
        gate.send_velocity(Velocity())
    except Exception:
        return


def _distance(sim: SimTransport) -> float:
    telemetry = sim.read()
    odom = telemetry.odom if telemetry is not None else None
    if odom is None:
        return 0.0
    return float(math.hypot(odom.x, odom.y))


def _passed(distance: float, vx: float, duration_s: float, tolerance_m: float) -> bool:
    if not math.isfinite(distance) or distance <= 0.0:
        return False
    upper = vx * duration_s + tolerance_m
    return distance <= upper + 1e-9


def run_verify(
    robot: str | None = None,
    *,
    duration_s: float = DRIVE_S,
    vx: float = DRIVE_MPS,
    period_s: float = COMMAND_PERIOD_S,
    tolerance_m: float = TOLERANCE_M,
    clock: Optional[Callable[[], float]] = None,
    sleep: Optional[Callable[[float], None]] = None,
    environ: Optional[Mapping[str, str]] = None,
    console: bool = False,
    opener: Optional[Callable[..., object]] = None,
    on_step: Optional[Callable[[int], None]] = None,
    command_log: Optional[list] = None,
) -> VerifyResult:
    """Connect ``robot`` on ``sim://``, drive, stop, and measure distance.

    ``clock`` and ``sleep`` default to real time, and the simulator thread
    integrates physics. Tests pass a manual clock whose ``sleep`` advances
    it; physics is then stepped by ``period_s`` on each command. There is no
    transport argument: the link is always ``SimTransport``.

    ``on_step`` runs at the start of each command. It may raise (including
    ``KeyboardInterrupt``); the ``finally`` clause still sends a stop.
    """
    if period_s <= 0 or period_s >= CMD_TIMEOUT_S:
        raise ValueError(
            f"command period must be > 0 and < {CMD_TIMEOUT_S} s so the deadman stays fed"
        )
    if (clock is None) != (sleep is None):
        raise ValueError("clock and sleep must be passed together")
    realtime = clock is None
    monotonic = time.monotonic if clock is None else clock
    pause = time.sleep if sleep is None else sleep
    env: Mapping[str, str] = os.environ if environ is None else environ
    if not robot:
        from .plugins import preferred_robot_id

        robot = preferred_robot_id()
    spec = get_spec(robot)
    sim = SimTransport(spec)
    if command_log is not None:
        original = sim.send_velocity

        def _record(
            vel: Velocity, _orig: Callable[[Velocity], None] = original
        ) -> None:
            command_log.append(vel)
            _orig(vel)

        sim.send_velocity = _record  # type: ignore[method-assign]
    # Built-in profile only. A safety config file must not raise these caps.
    gate = SafetyGate(
        sim,
        spec,
        config=SafetyConfig(),
        clock=monotonic,
        environ=env,
        auto_tick=realtime,
        isatty=lambda: False,
        input_fn=_no_prompt,
    )
    console_probe: Optional[ConsoleProbe] = None
    try:
        gate.connect()
        if realtime:
            sim.start_sim()
        if not _arm(gate, sim, realtime=realtime, clock=monotonic, sleep=pause):
            dist = _distance(sim)
            return _result(
                robot,
                dist,
                vx,
                duration_s,
                tolerance_m,
                ok=False,
                detail="simulator safety gate did not arm",
                gate=gate,
                console=None,
            )
        _drive(
            gate,
            sim,
            vx=vx,
            duration_s=duration_s,
            period_s=period_s,
            realtime=realtime,
            clock=monotonic,
            sleep=pause,
            on_step=on_step,
        )
        _stop(gate)
        if not realtime:
            sim.step(period_s)
        distance = _distance(sim)
        ok = _passed(distance, vx, duration_s, tolerance_m)
        detail = (
            f"moved {distance:.3f} m"
            if ok
            else f"moved {distance:.3f} m, outside tolerance"
        )
        if console:
            console_probe = _optional_console(env, opener)
        return _result(
            robot,
            distance,
            vx,
            duration_s,
            tolerance_m,
            ok=ok,
            detail=detail,
            gate=gate,
            console=console_probe,
        )
    finally:
        _stop(gate)
        try:
            sim.stop_sim()
        except Exception:
            pass
        try:
            gate.disconnect()
        except Exception:
            pass


def _optional_console(
    env: Mapping[str, str],
    opener: Optional[Callable[..., object]],
) -> ConsoleProbe:
    """Console ``/sim`` probe. A bad ``OHHO_CONSOLE_URL`` is unavailable, not a crash."""
    try:
        origin = console_origin(env)
    except (AuthError, OSError, ValueError):
        return ConsoleProbe("skip", "unavailable")
    return probe_console(origin, opener=opener or urlopen)


def _arm(
    gate: SafetyGate,
    sim: SimTransport,
    *,
    realtime: bool,
    clock: Callable[[], float],
    sleep: Callable[[float], None],
) -> bool:
    """Arm the simulator gate. Hardware confirmation is not possible here.

    ``isatty`` is false and the prompt returns an empty string, so a
    mis-detected hardware link fails closed instead of asking. Simulator
    arming does not read the hardware-arming environment variable.
    """
    if not realtime:
        _publish(sim)
        return bool(gate.arm())
    deadline = clock() + _ARM_WAIT_S
    while True:
        _publish(sim)
        if gate.arm():
            return True
        if clock() >= deadline:
            return False
        sleep(min(COMMAND_PERIOD_S, _ARM_WAIT_S))


def _drive(
    gate: SafetyGate,
    sim: SimTransport,
    *,
    vx: float,
    duration_s: float,
    period_s: float,
    realtime: bool,
    clock: Callable[[], float],
    sleep: Callable[[float], None],
    on_step: Optional[Callable[[int], None]],
) -> None:
    start = clock()
    step = 0
    while clock() - start < duration_s:
        if on_step is not None:
            on_step(step)
        _publish(sim)
        gate.send_velocity(Velocity(vx, 0.0, 0.0))
        if not realtime:
            sim.step(period_s)
        sleep(period_s)
        if not realtime:
            gate.tick()
        step += 1


def _result(
    robot: str,
    distance: float,
    vx: float,
    duration_s: float,
    tolerance_m: float,
    *,
    ok: bool,
    detail: str,
    gate: SafetyGate,
    console: Optional[ConsoleProbe],
) -> VerifyResult:
    return VerifyResult(
        ok=ok,
        robot=robot,
        transport="sim://",
        distance_m=distance,
        tolerance_m=tolerance_m,
        commanded_mps=vx,
        duration_s=duration_s,
        detail=detail,
        gated=isinstance(gate, SafetyGate),
        console=console,
        gate=gate,
    )


def format_verify(result: VerifyResult, *, as_json: bool) -> str:
    """One JSON object, or a short text report. No secrets."""
    if as_json:
        return json.dumps(result.as_dict(), ensure_ascii=False, indent=2) + "\n"
    lines = [
        f"verify sim:// {result.robot}: {'pass' if result.ok else 'fail'}",
        f"  distance: {result.distance_m:.3f} m",
        f"  tolerance: {result.tolerance_m:.3f} m",
        (f"  commanded: {result.commanded_mps:g} m/s for {result.duration_s:g} s"),
    ]
    if result.console is not None:
        lines.append(f"  console: {result.console.status}  {result.console.detail}")
    return "\n".join(lines) + "\n"


def format_refusal(message: str, *, robot: str, as_json: bool) -> str:
    if not as_json:
        return ""
    body = {
        "ok": False,
        "robot": robot,
        "transport": "sim://",
        "error": message,
    }
    return json.dumps(body, ensure_ascii=False, indent=2) + "\n"


def run_verify_cli(
    *,
    sim: bool,
    as_json: bool,
    robot: str,
    console: bool,
    transport: Optional[str],
    environ: Optional[Mapping[str, str]] = None,
    stdout: Optional[TextIO] = None,
    stderr: Optional[TextIO] = None,
    opener: Optional[Callable[..., object]] = None,
) -> int:
    """CLI entry. Exit 2 on refusal, 1 when the sim drive fails, 0 on pass."""
    out = stdout if stdout is not None else sys.stdout
    err = stderr if stderr is not None else sys.stderr
    env = os.environ if environ is None else environ
    reason = non_sim_refusal(env, transport=transport, robot=robot)
    if reason is None and not sim:
        reason = (
            "ohho verify requires --sim. It only drives sim://. "
            "Hardware is never the default."
        )
    if reason is not None:
        print(f"error: {reason}", file=err)
        text = format_refusal(reason, robot=robot, as_json=as_json)
        if text:
            out.write(text)
        return 2
    try:
        result = run_verify(
            robot,
            environ=env,
            console=console,
            opener=opener,
        )
    except UnknownRobot as exc:
        message = f"unknown robot {robot}"
        print(f"error: {message}", file=err)
        text = format_refusal(str(exc) or message, robot=robot, as_json=as_json)
        if text:
            out.write(text)
        return 2
    except KeyboardInterrupt:
        print("stopped.", file=err)
        return 1
    except Exception:
        print("error: verify failed", file=err)
        return 1
    out.write(format_verify(result, as_json=as_json))
    return 0 if result.ok else 1
