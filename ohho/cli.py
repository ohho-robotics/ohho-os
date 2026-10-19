"""The ``ohho`` command-line interface."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

from . import __version__
from .adapters import available_adapters
from .agent import Agent
from .config import (
    ConfigError,
    RobotProfile,
    default_home,
    format_resolved,
    load_robot_profile,
    resolve_robot_id,
)
from .hardware import resolve_device
from .registry import UnknownRobot, list_specs
from .robot import Robot
from .runtime import available_runtimes

_ROBOT_HELP = "robot id (omit to use OHHO_ROBOT, ~/.ohho/config.toml, or sim)"


def _cmd_version(args) -> int:
    print(f"OhhO OS {__version__}")
    return 0


def _explicit_robot(args) -> Optional[str]:
    if not hasattr(args, "robot"):
        return None
    value = args.robot
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text or None


def _normalize_robot_args(args) -> None:
    """Keep ``ohho agent ROBOT GOAL`` working when the robot positional is optional.

    One positional on ``agent`` is the goal. One positional on ``market run`` is
    the skill. One positional on ``memory where`` is the label.
    """
    command = getattr(args, "command", None)
    robot = _explicit_robot(args)
    if command == "agent" and getattr(args, "goal", None) is None and robot:
        args.goal = robot
        args.robot = None
    elif (
        command == "market"
        and getattr(args, "action", None) == "run"
        and getattr(args, "skill_name", None) is None
        and robot
    ):
        args.skill_name = robot
        args.robot = None
    elif (
        command == "memory"
        and getattr(args, "action", None) == "where"
        and getattr(args, "label", None) is None
        and robot
    ):
        args.label = robot
        args.robot = None


def _wants_profile(args) -> bool:
    command = getattr(args, "command", None)
    action = getattr(args, "action", None)
    if command in {
        "doctor",
        "config",
        "connect",
        "sim",
        "drive",
        "agent",
        "sim-serve",
        "sim-loop",
        "selftest",
        "look",
    }:
        return True
    if command == "market" and action == "run":
        return True
    if command == "nav" and action in {"goto", "explore", "map"}:
        return True
    if command == "memory" and action in {"show", "where", "near", "clear"}:
        return True
    return False


def _bind_robot(args) -> None:
    """Resolve the robot once for this invocation and attach it to ``args``."""
    _normalize_robot_args(args)
    flag = _explicit_robot(args)
    home = default_home()
    resolved = resolve_robot_id(flag, home=home)
    args.resolved_robot = resolved
    if hasattr(args, "robot") and flag is None:
        args.robot = resolved.robot_id
    if _wants_profile(args):
        args.robot_profile = load_robot_profile(
            resolved.robot_id, home=home, cwd=Path.cwd()
        )
    else:
        args.robot_profile = RobotProfile(resolved.robot_id)


def _connect_kwargs(args, transport: Optional[str], *, use_profile_transport: bool):
    """Transport plus profile fields. Empty profiles add no keyword arguments."""
    profile = getattr(args, "robot_profile", None)
    extra: dict = {}
    if isinstance(profile, RobotProfile):
        if use_profile_transport and transport is None and profile.transport:
            transport = profile.transport
        if profile.sim_backend:
            extra["sim_backend"] = profile.sim_backend
        if profile.asset_path:
            extra["asset_path"] = profile.asset_path
        if profile.safety_overrides:
            extra["safety_overrides"] = dict(profile.safety_overrides)
    return transport, extra


def _cmd_doctor(args) -> int:
    import importlib.util

    from .brains import harness_available

    resolved = getattr(args, "resolved_robot", None)
    if resolved is None:
        resolved = resolve_robot_id(_explicit_robot(args), home=default_home())

    def have(mod: str) -> bool:
        return importlib.util.find_spec(mod) is not None

    runtimes = available_runtimes()
    print(f"OhhO OS {__version__}")
    print(f"  python      : {sys.version.split()[0]}")
    print(f"  runtimes    : {', '.join(runtimes)}")
    print(f"  adapters    : {', '.join(available_adapters())}  (others via extras)")
    print(f"  device      : {resolve_device('auto')}")
    print(f"  robots      : {', '.join(s.id for s in list_specs())}")
    print(f"  robot       : {resolved.robot_id} ({resolved.source})")
    profile = getattr(args, "robot_profile", None)
    if isinstance(profile, RobotProfile) and profile.source:
        print(f"  profile     : {profile.source}")
    brain = (
        "harness — full perceive→reason→act→reflect"
        if harness_available()
        else "scripted fallback  (install 'ohho-os[agent]' + repo agent_engine)"
    )
    print(f"  agent brain : {brain}")
    print(
        f"  serve       : {'ready (fastapi)' if have('fastapi') else 'needs [serve] extra'}"
    )
    print(
        f"  data writer : {'parquet (pyarrow)' if have('pyarrow') else 'json fallback  (add [data] for parquet)'}"
    )
    print(
        f"  train       : {'torch available' if have('torch') else 'mock only  (add [train] for real training)'}"
    )
    print(f"  recommended : runtime={'ros2' if 'ros2' in runtimes else 'native'}")
    return 0


def _cmd_list(args) -> int:
    if getattr(args, "plugins", False):
        from .plugins import format_plugin_report

        # Register clean plugins first so ``ohho list`` and ``--plugins`` agree.
        list_specs()
        print(format_plugin_report())
        return 0
    for s in list_specs():
        caps = ", ".join(s.capabilities) or "—"
        print(f"{s.id:<14} {s.name:<18} {s.category:<20} [{caps}]")
    return 0


def _connect(args) -> Robot:
    transport, extra = _connect_kwargs(args, args.transport, use_profile_transport=True)
    bot = Robot.connect(
        args.robot,
        transport=transport,
        runtime=args.runtime,
        safety_config=args.safety_config,
        estop_damp=args.estop_damp,
        **extra,
    )
    if bot.gated:
        if args.arm:
            if not bot.arm():
                print(
                    "safety: arming refused — motion stays disabled "
                    "(reason in ~/.ohho/safety/<date>.jsonl)",
                    file=sys.stderr,
                )
        else:
            print(
                "safety: hardware link is disarmed — pass --arm to enable motion",
                file=sys.stderr,
            )
    return bot


def _cmd_connect(args) -> int:
    try:
        bot = _connect(args)
    except UnknownRobot as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    try:
        print(f"connected: {bot!r}")
        s = bot.status()
        print(f"  status : {s.state.value} — {s.label}")
        time.sleep(0.3)
        t = bot.telemetry()
        if t and t.odom:
            print(
                f"  odom   : x={t.odom.x:.3f} y={t.odom.y:.3f} theta={t.odom.theta:.3f}"
            )
        if t and t.battery is not None:
            print(f"  battery: {t.battery * 100:.0f}%")
    finally:
        bot.disconnect()
    return 0


def _cmd_sim(args) -> int:
    transport, extra = _connect_kwargs(args, "sim://", use_profile_transport=False)
    try:
        bot = Robot.connect(args.robot, transport=transport, runtime="native", **extra)
    except UnknownRobot as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(
        f"simulating {bot.spec.name} — driving a pattern for {args.seconds:.0f}s (Ctrl-C to stop)"
    )
    try:
        i = 0
        end = time.monotonic() + args.seconds
        while time.monotonic() < end:
            phase = i % 4
            if phase == 0:
                bot.drive(vx=0.15)
            elif phase == 1:
                bot.drive(w=0.6)
            elif phase == 2:
                bot.drive(vx=0.15, vy=0.1)
            else:
                bot.drive(w=-0.6)
            time.sleep(0.5)
            t = bot.telemetry()
            if t and t.odom:
                batt = (t.battery or 0.0) * 100
                print(
                    f"  t+{i * 0.5:>4.1f}s  x={t.odom.x:6.3f} y={t.odom.y:6.3f} "
                    f"th={t.odom.theta:6.3f}  batt={batt:3.0f}%"
                )
            i += 1
    except KeyboardInterrupt:
        print("\nstopped.")
    finally:
        bot.stop()
        bot.disconnect()
    return 0


def _cmd_drive(args) -> int:
    try:
        bot = _connect(args)
    except UnknownRobot as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    try:
        print(
            f"driving {bot.spec.id}: vx={args.vx} vy={args.vy} w={args.w} for {args.seconds:.0f}s"
        )
        end = time.monotonic() + args.seconds
        while time.monotonic() < end:
            bot.drive(vx=args.vx, vy=args.vy, w=args.w)
            time.sleep(0.1)
        bot.stop()
        time.sleep(0.1)
        t = bot.telemetry()
        if t and t.odom:
            print(
                f"final odom: x={t.odom.x:.3f} y={t.odom.y:.3f} theta={t.odom.theta:.3f}"
            )
    finally:
        bot.disconnect()
    return 0


def _cmd_agent(args) -> int:
    if not getattr(args, "goal", None):
        print('usage: ohho agent [ROBOT] "goal"', file=sys.stderr)
        return 2
    try:
        bot = _connect(args)
    except UnknownRobot as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    try:
        for line in Agent(bot).run(args.goal):
            print(f"  {line}")
    finally:
        bot.disconnect()
    return 0


def _cmd_serve(args) -> int:
    from .serve import serve, ServeUnavailable

    try:
        serve(
            checkpoint=args.checkpoint,
            port=args.port,
            host=args.host,
            device=args.device,
            mock=args.mock,
        )
    except ServeUnavailable as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    return 0


def _cmd_sim_serve(args) -> int:
    if args.backend in ("isaac", "mujoco"):
        print("not built", file=sys.stderr)
        return 1

    from .sim_serve import SimServeUnavailable, serve_sim

    try:
        serve_sim(
            backend=args.backend,
            robot=args.robot,
            host=args.host,
            port=args.port,
            rosbridge=args.rosbridge,
            rate_hz=args.rate,
            home=default_home(),
        )
    except SimServeUnavailable as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nstopped.")
    return 0


def _cmd_sim_loop(args) -> int:
    from .sim_loop import run_sim_loop

    try:
        run_sim_loop(
            robot_id=args.robot,
            episodes=args.episodes,
            steps_per_episode=args.steps_per_episode,
            train_steps=args.train_steps,
            eval_steps=args.eval_steps,
            device=args.device,
            output_dir=args.output_dir,
            mock=args.mock,
            home=str(default_home()),
        )
        return 0
    except Exception as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


def _cmd_market(args) -> int:
    from .market import list_skills, run_skill, SkillRequirementsNotMet

    if args.action == "list":
        skills = list_skills()
        if not skills:
            print("no skills registered")
            return 0
        for s in skills:
            reqs = ", ".join(s.requires) or "—"
            tags = ", ".join(s.tags) or "—"
            print(
                f"{s.name:<16} v{s.version:<6} [{reqs}]  "
                f"tags: {tags}\n  {s.description}"
            )
        return 0
    if args.action == "run":
        if not getattr(args, "skill_name", None):
            print("usage: ohho market run [ROBOT] SKILL", file=sys.stderr)
            return 1
        try:
            bot = _connect(args)
        except Exception as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
        try:
            result = run_skill(args.skill_name, bot)
            print(f"skill '{args.skill_name}': {result}")
        except SkillRequirementsNotMet as e:
            print(f"error: {e}", file=sys.stderr)
            return 3
        except KeyError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
        finally:
            bot.disconnect()
        return 0
    print("usage: ohho market [list|run]")
    return 1


def _cmd_profile(args) -> int:
    from . import profiles

    if args.action == "list":
        for name in profiles.list_profiles():
            p = profiles.get_profile(name)
            print(f"{name:<22} {p.description}")
        return 0
    if args.action == "show":
        p = profiles.get_profile(args.profile_name)
        print(f"profile: {p.name}")
        print(f"  {p.description}")
        for node in p.nodes:
            print(f"  node: {node.name}")
            print(f"    roles: {', '.join(node.roles)}")
            print(f"    device: {node.device}  accelerator: {node.accelerator or '—'}")
            if node.notes:
                print(f"    notes: {node.notes}")
        return 0
    if args.action == "detect":
        p = profiles.detect_profile()
        desc = profiles.describe()
        print(f"detected: {p.name}")
        print(f"  {p.description}")
        print(f"  platform: {desc['platform']}")
        print(f"  device: {desc['device']}")
        return 0
    print("usage: ohho profile [list|show|detect]")
    return 1


def _maybe_world(bot: Robot, world: str) -> None:
    """Populate the sim world when ``--world demo`` is given (no-op on hardware)."""
    if world != "demo":
        return
    add = getattr(bot.transport, "add_object", None)
    if callable(add):
        from .adapters.sim import demo_world

        for ob in demo_world():
            add(ob.x, ob.y, ob.radius, ob.label)


def _cmd_nav(args) -> int:
    from .nav import Navigator

    if not getattr(args, "action", None):
        print("usage: ohho nav [goto|explore|map]")
        return 1
    try:
        bot = _connect(args)
    except UnknownRobot as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    _maybe_world(bot, args.world)
    try:
        nav = Navigator(bot)
        if args.action == "goto":
            print(f"navigating {bot.spec.id} to ({args.x:.2f}, {args.y:.2f}) …")
            result = nav.navigate_to(args.x, args.y, timeout=args.timeout)
            print(result)
            return 0 if result.reached else 3
        if args.action == "explore":
            for line in nav.explore(max_frontiers=args.frontiers):
                print(line)
            print(nav.map_ascii())
            return 0
        if args.action == "map":
            print("spin-scanning to build the map …")
            nav.spin_scan()
            print(nav.map_ascii())
            c = nav.grid.counts()
            print(
                f"cells: {c['free']} free · {c['occupied']} occupied · {c['unknown']} unknown"
            )
            return 0
        return 1
    finally:
        bot.disconnect()


def _cmd_look(args) -> int:
    from .memory import SpatialMemory, default_memory_path
    from .perception import SimPerceptor, remember_detections

    try:
        bot = _connect(args)
    except UnknownRobot as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    _maybe_world(bot, args.world)
    try:
        detections = SimPerceptor(bot).look()
        if not detections:
            print("nothing detected")
            return 0
        for d in detections:
            print(f"  {d}")
        path = default_memory_path(bot.spec.id)
        memory = SpatialMemory(path=path)
        n = remember_detections(memory, detections)
        memory.save()
        print(f"remembered {n} object(s) → {path}")
        return 0
    finally:
        bot.disconnect()


def _cmd_memory(args) -> int:
    from .memory import SpatialMemory, default_memory_path

    if not getattr(args, "action", None):
        print("usage: ohho memory [show|where|near|clear]")
        return 1
    if args.action == "where" and not getattr(args, "label", None):
        print("usage: ohho memory where [ROBOT] LABEL", file=sys.stderr)
        return 1
    memory = SpatialMemory(path=default_memory_path(args.robot))
    if args.action == "show":
        print(memory.describe())
        events = memory.timeline()[-5:]
        if events:
            print("recent events:")
            for ev in events:
                where = f" @({ev.x:.2f},{ev.y:.2f})" if ev.x is not None else ""
                print(f"  [{ev.kind}] {ev.label}{where} {ev.note}".rstrip())
        return 0
    if args.action == "where":
        e = memory.where_is(args.label)
        if e is None:
            print(f"no memory of '{args.label}'")
            return 3
        print(
            f"{e.label} last seen at ({e.x:.2f}, {e.y:.2f}), "
            f"{e.age():.0f}s ago (seen {e.count}×)"
        )
        return 0
    if args.action == "near":
        found = memory.near(args.x, args.y, args.radius)
        if not found:
            print(f"nothing within {args.radius:.1f} m of ({args.x:.2f}, {args.y:.2f})")
            return 0
        for e in found:
            print(f"  {e.label} at ({e.x:.2f}, {e.y:.2f})")
        return 0
    if args.action == "clear":
        memory.clear()
        memory.save()
        print(f"memory cleared for '{args.robot}'")
        return 0
    print("usage: ohho memory [show|where|near|clear]")
    return 1


def _cmd_selftest(args) -> int:
    from .selftest import default_report_path, format_report_text, run_selftest

    allow_spin = getattr(args, "allow_spin", False)
    if not args.sim and not allow_spin:
        if sys.stdin.isatty():
            try:
                ans = (
                    input(
                        "Allow wheel spin test on bench? (wheels will spin briefly at low speed) [y/N]: "
                    )
                    .strip()
                    .lower()
                )
                if ans in ("y", "yes"):
                    allow_spin = True
            except (EOFError, KeyboardInterrupt):
                allow_spin = False

    out_path = args.out or default_report_path()
    try:
        report = run_selftest(
            robot=args.robot,
            sim=args.sim,
            allow_spin=allow_spin,
            out_path=out_path,
            base_port=getattr(args, "base_port", ""),
            arm_port=getattr(args, "arm_port", ""),
            camera_idx=getattr(args, "camera_idx", 0),
        )
    except Exception as e:
        print(f"error running selftest: {e}", file=sys.stderr)
        return 2

    print(format_report_text(report))
    print(f"Report saved to: {out_path}")
    if report.overall_status == "FAIL":
        return 1
    if report.mode == "hardware" and report.summary.get("passed", 0) == 0:
        if not getattr(args, "allow_all_skip", False):
            print(
                "Error: all hardware checks were skipped (no hardware verified). Exiting with code 3.",
                file=sys.stderr,
            )
            return 3
    return 0


def _cmd_config(args) -> int:
    if getattr(args, "action", None) != "show":
        print("usage: ohho config show")
        return 1
    resolved = getattr(args, "resolved_robot", None)
    if resolved is None:
        resolved = resolve_robot_id(_explicit_robot(args), home=default_home())
    profile = getattr(args, "robot_profile", None)
    if profile is None:
        profile = load_robot_profile(resolved.robot_id, home=default_home())
    print(format_resolved(resolved, profile))
    return 0


def _add_robot_args(
    sp: argparse.ArgumentParser, transport_default: Optional[str] = None
) -> None:
    sp.add_argument("robot", nargs="?", default=None, help=_ROBOT_HELP)
    sp.add_argument(
        "--transport", default=transport_default, help="transport URI, e.g. sim://"
    )
    sp.add_argument("--runtime", default="auto", choices=["auto", "native", "ros2"])
    sp.add_argument(
        "--arm",
        action="store_true",
        help="arm a hardware link (needs OHHO_ARM_HARDWARE=1 and typed confirmation)",
    )
    sp.add_argument(
        "--estop-damp",
        action="store_true",
        help="damp motors after an e-stop even if the robot is standing",
    )
    sp.add_argument(
        "--safety-config", default=None, help="safety config JSON (caps, ceilings)"
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ohho", description="OhhO OS — the open robot engine."
    )
    p.add_argument(
        "-V", "--version", action="store_true", help="print version and exit"
    )
    sub = p.add_subparsers(dest="command")

    doctor = sub.add_parser(
        "doctor", help="show environment, runtimes, adapters and robots"
    )
    doctor.add_argument("--robot", default=None, help=_ROBOT_HELP)
    doctor.set_defaults(func=_cmd_doctor)
    listed = sub.add_parser("list", help="list registered robots")
    listed.add_argument(
        "--plugins",
        action="store_true",
        help="show each robot plugin, its specs, and any load error",
    )
    listed.set_defaults(func=_cmd_list)
    sub.add_parser("version", help="print version").set_defaults(func=_cmd_version)

    cfg = sub.add_parser("config", help="show the resolved robot id")
    cfg_sub = cfg.add_subparsers(dest="action")
    cfg_show = cfg_sub.add_parser(
        "show", help="print the resolved robot and where it came from"
    )
    cfg_show.add_argument("--robot", default=None, help=_ROBOT_HELP)
    cfg_show.set_defaults(action="show")
    cfg.set_defaults(func=_cmd_config)

    c = sub.add_parser("connect", help="connect and print status + telemetry")
    _add_robot_args(c)
    c.set_defaults(func=_cmd_connect)

    s = sub.add_parser("sim", help="run a robot in simulation")
    s.add_argument("--robot", default=None, help=_ROBOT_HELP)
    s.add_argument("--seconds", type=float, default=8.0)
    s.set_defaults(func=_cmd_sim)

    d = sub.add_parser("drive", help="drive a robot for a duration")
    _add_robot_args(d)
    d.add_argument("--vx", type=float, default=0.1)
    d.add_argument("--vy", type=float, default=0.0)
    d.add_argument("--w", type=float, default=0.0)
    d.add_argument("--seconds", type=float, default=3.0)
    d.set_defaults(func=_cmd_drive)

    a = sub.add_parser("agent", help="hand a natural-language goal to the agent")
    _add_robot_args(a)
    a.add_argument(
        "goal",
        nargs="?",
        default=None,
        help='the goal, e.g. "explore the room"',
    )
    a.set_defaults(func=_cmd_agent)

    sv = sub.add_parser("serve", help="serve a trained policy over REST")
    sv.add_argument("--checkpoint", default="", help="path to trained checkpoint")
    sv.add_argument("--port", type=int, default=8000)
    sv.add_argument("--host", default="0.0.0.0")
    sv.add_argument("--device", default="auto")
    sv.add_argument(
        "--mock", action="store_true", help="use a no-op model (sim loop, no GPU)"
    )
    sv.set_defaults(func=_cmd_serve)

    ss = sub.add_parser("sim-serve", help="serve robot simulation over WebSocket")
    ss.add_argument(
        "--backend",
        default="sim",
        choices=["sim", "ros2", "isaac", "mujoco"],
        help="simulation backend (sim, ros2, isaac, mujoco)",
    )
    ss.add_argument("--robot", default=None, help=_ROBOT_HELP)
    ss.add_argument(
        "--host", default="0.0.0.0", help="host to bind the WebSocket server"
    )
    ss.add_argument(
        "--port", type=int, default=8765, help="port to bind the WebSocket server"
    )
    ss.add_argument(
        "--rosbridge",
        default="ws://localhost:9090",
        help="rosbridge WebSocket URL for ros2 backend",
    )
    ss.add_argument(
        "--rate", type=float, default=20.0, help="telemetry broadcast rate in Hz"
    )
    ss.set_defaults(func=_cmd_sim_serve)

    sl = sub.add_parser(
        "sim-loop",
        help="run nightly CPU sim loop (record -> validate -> train ACT -> serve -> drive sim)",
    )
    sl.add_argument("--robot", default=None, help=_ROBOT_HELP)
    sl.add_argument(
        "--episodes", type=int, default=5, help="number of episodes to record"
    )
    sl.add_argument(
        "--steps-per-episode",
        type=int,
        default=20,
        help="number of frames recorded per episode",
    )
    sl.add_argument(
        "--train-steps",
        type=int,
        default=200,
        help="number of CPU training steps for ACT",
    )
    sl.add_argument(
        "--eval-steps",
        type=int,
        default=50,
        help="number of closed-loop policy evaluation steps",
    )
    sl.add_argument(
        "--device", default="cpu", help="device to run training and inference on"
    )
    sl.add_argument(
        "--output-dir",
        default="./sim_loop_output",
        help="output directory for dataset and checkpoints",
    )
    sl.add_argument(
        "--mock",
        action="store_true",
        help="run in mock mode (no torch or GPU dependencies required)",
    )
    sl.set_defaults(func=_cmd_sim_loop)

    mk = sub.add_parser("market", help="list and run robot skills")
    mk_sub = mk.add_subparsers(dest="action")
    mk_sub.add_parser("list", help="list registered skills").set_defaults(action="list")
    mk_run = mk_sub.add_parser("run", help="run a skill on a robot")
    _add_robot_args(mk_run)
    mk_run.add_argument(
        "skill_name",
        nargs="?",
        default=None,
        help="skill name (see 'ohho market list')",
    )
    mk_run.set_defaults(action="run")
    mk.set_defaults(func=_cmd_market)

    nv = sub.add_parser("nav", help="native navigation: goto / explore / map")
    nv_sub = nv.add_subparsers(dest="action")
    nv_goto = nv_sub.add_parser("goto", help="navigate to a world-frame point")
    _add_robot_args(nv_goto)
    nv_goto.add_argument("--x", type=float, required=True)
    nv_goto.add_argument("--y", type=float, required=True)
    nv_goto.add_argument("--timeout", type=float, default=30.0)
    nv_goto.add_argument("--world", default="", help="'demo' adds demo obstacles (sim)")
    nv_goto.set_defaults(action="goto")
    nv_exp = nv_sub.add_parser("explore", help="frontier exploration (builds the map)")
    _add_robot_args(nv_exp)
    nv_exp.add_argument("--frontiers", type=int, default=4)
    nv_exp.add_argument("--world", default="", help="'demo' adds demo obstacles (sim)")
    nv_exp.set_defaults(action="explore")
    nv_map = nv_sub.add_parser("map", help="spin-scan and print the occupancy map")
    _add_robot_args(nv_map)
    nv_map.add_argument("--world", default="", help="'demo' adds demo obstacles (sim)")
    nv_map.set_defaults(action="map")
    nv.set_defaults(func=_cmd_nav)

    lk = sub.add_parser("look", help="perceive surroundings and remember them")
    _add_robot_args(lk)
    lk.add_argument("--world", default="", help="'demo' adds demo obstacles (sim)")
    lk.set_defaults(func=_cmd_look)

    me = sub.add_parser("memory", help="query the robot's spatio-temporal memory")
    me_sub = me.add_subparsers(dest="action")
    me_show = me_sub.add_parser("show", help="summarize everything remembered")
    me_show.add_argument("robot", nargs="?", default=None, help=_ROBOT_HELP)
    me_show.set_defaults(action="show")
    me_where = me_sub.add_parser("where", help="recall an object's last position")
    me_where.add_argument("robot", nargs="?", default=None, help=_ROBOT_HELP)
    me_where.add_argument("label", nargs="?", default=None)
    me_where.set_defaults(action="where")
    me_near = me_sub.add_parser("near", help="remembered objects near a point")
    me_near.add_argument("robot", nargs="?", default=None, help=_ROBOT_HELP)
    me_near.add_argument("--x", type=float, required=True)
    me_near.add_argument("--y", type=float, required=True)
    me_near.add_argument("--radius", type=float, default=1.5)
    me_near.set_defaults(action="near")
    me_clear = me_sub.add_parser("clear", help="forget everything")
    me_clear.add_argument("robot", nargs="?", default=None, help=_ROBOT_HELP)
    me_clear.set_defaults(action="clear")
    me.set_defaults(func=_cmd_memory)

    pf = sub.add_parser("profile", help="hardware deployment profiles")
    pf_sub = pf.add_subparsers(dest="action")
    pf_sub.add_parser("list", help="list built-in profiles").set_defaults(action="list")
    pf_show = pf_sub.add_parser("show", help="show a specific profile")
    pf_show.add_argument("profile_name", help="profile name")
    pf_show.set_defaults(action="show")
    pf_sub.add_parser(
        "detect", help="auto-detect the current machine's profile"
    ).set_defaults(action="detect")
    pf.set_defaults(func=_cmd_profile)

    st = sub.add_parser("selftest", help="hardware bring-up self-test with JSON report")
    st.add_argument(
        "--robot",
        default=None,
        help=_ROBOT_HELP,
    )
    st.add_argument(
        "--sim",
        action="store_true",
        help="run against simulated hardware (safe for CI)",
    )
    st.add_argument(
        "--allow-spin",
        "--spin-wheels",
        action="store_true",
        dest="allow_spin",
        help="allow bench wheel spin check (low-speed, short pulse)",
    )
    st.add_argument(
        "--out",
        default="",
        help="custom path for JSON report (default: selftest-YYYYMMDD-HHMMSS.json)",
    )
    st.add_argument(
        "--base-port", default="", help="serial port for base (e.g. /dev/ttyUSB0)"
    )
    st.add_argument(
        "--arm-port", default="", help="serial port for arm bus (e.g. /dev/ttyACM0)"
    )
    st.add_argument(
        "--camera-idx", type=int, default=0, help="camera device index (default: 0)"
    )
    st.add_argument(
        "--allow-all-skip",
        action="store_true",
        help="allow exit code 0 when all checks are skipped in hardware mode",
    )
    st.set_defaults(func=_cmd_selftest)

    return p


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if getattr(args, "version", False):
            return _cmd_version(args)
        if not getattr(args, "command", None):
            parser.print_help()
            return 0
        _bind_robot(args)
        return args.func(args)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except BrokenPipeError:
        # stdout was closed early (e.g. piped into `head`) — exit quietly.
        import os

        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
