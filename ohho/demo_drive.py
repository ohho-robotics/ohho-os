"""Shared drive demo for sim:// and ros2:// — one transport argument switch.

Used by ``examples/demo_drive.py`` and CI (fake ROS 2 robot node).
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass
from typing import Optional

from .config import default_home, load_robot_profile, resolve_robot_id
from .registry import UnknownRobot
from .robot import Robot


@dataclass(frozen=True)
class DriveResult:
    """Odometry before/after a timed velocity command."""

    x0: float
    y0: float
    theta0: float
    x1: float
    y1: float
    theta1: float
    seconds: float

    @property
    def dx(self) -> float:
        return self.x1 - self.x0

    @property
    def dy(self) -> float:
        return self.y1 - self.y0

    @property
    def displacement(self) -> float:
        return math.hypot(self.dx, self.dy)

    def summary_line(self) -> str:
        return (
            f"odom delta: dx={self.dx:.4f} dy={self.dy:.4f} "
            f"disp={self.displacement:.4f} dtheta={self.theta1 - self.theta0:.4f} "
            f"({self.seconds:.1f}s)"
        )


def default_runtime_for_transport(transport: str) -> str:
    """Pick runtime so callers can change only ``--transport`` between sim and ROS 2."""
    scheme = transport.partition("://")[0].lower()
    if scheme == "ros2":
        return "ros2"
    return "native"


def _has_live_odom(bot: Robot) -> bool:
    """True when odometry is from the backend, not a zero placeholder."""
    tp = bot.transport
    flag = getattr(tp, "odom_received", None)
    if flag is not None:
        return bool(flag)
    t = bot.telemetry()
    return t.odom is not None


def wait_for_odom(bot: Robot, timeout_s: float = 8.0) -> bool:
    """Block until telemetry reports odometry or timeout."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if _has_live_odom(bot):
            return True
        time.sleep(0.05)
    return False


def run_drive(
    bot: Robot,
    *,
    seconds: float,
    vx: float = 0.15,
    vy: float = 0.0,
    w: float = 0.0,
    rate_hz: float = 10.0,
    settle_s: float = 0.3,
    odom_timeout_s: float = 8.0,
) -> DriveResult:
    """Drive at constant velocity and return odometry delta."""
    if not wait_for_odom(bot, timeout_s=odom_timeout_s):
        gated = bot.gated
        proto = getattr(bot.transport, "protocol", "unknown")
        raise RuntimeError(
            "no odometry received before drive "
            f"(gated={gated}, protocol={proto}, waited {odom_timeout_s}s for live odom)"
        )
    if bot.gated and not bot.arm():
        raise RuntimeError("failed to arm safety gate before drive")
    t0 = bot.telemetry()
    o0 = t0.odom
    if o0 is None:
        raise RuntimeError("no odometry snapshot before drive")
    period = 1.0 / max(rate_hz, 1.0)
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        bot.drive(vx=vx, vy=vy, w=w)
        time.sleep(period)
    bot.stop()
    time.sleep(settle_s)
    t1 = bot.telemetry()
    o1 = t1.odom or o0
    return DriveResult(
        x0=o0.x,
        y0=o0.y,
        theta0=o0.theta,
        x1=o1.x,
        y1=o1.y,
        theta1=o1.theta,
        seconds=seconds,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Drive a robot with the same script on sim:// or ros2:// "
            "(change --transport only; runtime defaults per transport)."
        )
    )
    p.add_argument(
        "--robot",
        default=None,
        help="robot spec id (else OHHO_ROBOT, ~/.ohho/config.toml, or sim)",
    )
    p.add_argument(
        "--transport",
        default="sim://",
        help="sim:// for in-process sim, ros2:// for /cmd_vel + /odom",
    )
    p.add_argument(
        "--runtime",
        default=None,
        choices=["auto", "native", "ros2"],
        help="defaults to native for sim:// and ros2 for ros2://",
    )
    p.add_argument("--seconds", type=float, default=2.0, help="drive duration")
    p.add_argument("--vx", type=float, default=0.15)
    p.add_argument("--vy", type=float, default=0.0)
    p.add_argument("--w", type=float, default=0.0)
    p.add_argument(
        "--min-dx",
        type=float,
        default=None,
        help="exit 1 if forward odom delta is below this (CI assertion)",
    )
    p.add_argument(
        "--min-displacement",
        type=float,
        default=None,
        help="exit 1 if planar displacement is below this",
    )
    p.add_argument(
        "--odom-timeout",
        type=float,
        default=8.0,
        help="seconds to wait for live odometry before driving",
    )
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    resolved = resolve_robot_id(args.robot, home=default_home())
    profile = load_robot_profile(resolved.robot_id, home=default_home())
    runtime = args.runtime or default_runtime_for_transport(args.transport)
    print(
        f"demo_drive: robot={resolved.robot_id} transport={args.transport} runtime={runtime} "
        f"vx={args.vx} for {args.seconds}s"
    )
    try:
        bot = Robot.connect(
            resolved.robot_id,
            transport=args.transport,
            runtime=runtime,
            sim_backend=profile.sim_backend,
            asset_path=profile.asset_path,
            safety_overrides=dict(profile.safety_overrides) or None,
        )
    except UnknownRobot as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    try:
        result = run_drive(
            bot,
            seconds=args.seconds,
            vx=args.vx,
            vy=args.vy,
            w=args.w,
            odom_timeout_s=args.odom_timeout,
        )
        print(result.summary_line())
        if args.min_dx is not None and result.dx < args.min_dx:
            print(
                f"FAIL: dx {result.dx:.4f} < --min-dx {args.min_dx}",
                file=sys.stderr,
            )
            return 1
        if (
            args.min_displacement is not None
            and result.displacement < args.min_displacement
        ):
            print(
                f"FAIL: displacement {result.displacement:.4f} < "
                f"--min-displacement {args.min_displacement}",
                file=sys.stderr,
            )
            return 1
        return 0
    finally:
        bot.disconnect()


__all__ = [
    "DriveResult",
    "default_runtime_for_transport",
    "wait_for_odom",
    "run_drive",
    "build_arg_parser",
    "main",
]
