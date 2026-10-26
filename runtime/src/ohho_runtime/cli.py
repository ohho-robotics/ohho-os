"""Operator CLI for the onboard runtime.

Console script ``ohho-runtime``: ``up``, ``down``, ``status``, ``health``,
``logs``, and ``restart``. ``up`` runs in the foreground. The other
commands talk to it over the loopback control channel.

This does not arm a robot and does not open a non-loopback socket.
``ohho`` does not load external subcommands, so this package does not
register an ``ohho runtime`` alias.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import TextIO

from ohho_runtime.control import (
    LOOPBACK_IPV4,
    ControlError,
    ControlSession,
    LogBuffer,
    default_state_path,
    ping_state,
    read_state,
    require_loopback_host,
    transact,
)
from ohho_runtime.log import (
    RuntimeLog,
    resolve_log_format,
    resolve_log_level,
    resolve_run_id,
)
from ohho_runtime.manifest import ManifestError
from ohho_runtime.manifest_file import class_manifest, load_manifest_path
from ohho_runtime.metrics import Registry
from ohho_runtime.supervisor import Supervisor, SupervisorError

_FOLLOW_INTERVAL_S = 0.25


def build_parser() -> argparse.ArgumentParser:
    """Argument parser for ``ohho-runtime``."""
    parser = argparse.ArgumentParser(
        prog="ohho-runtime",
        description="Operate the onboard runtime supervisor.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    up = sub.add_parser("up", help="start the supervisor in the foreground")
    source = up.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--class",
        dest="robot_class",
        metavar="ROBOT_CLASS",
        help="robot class whose default manifest to run",
    )
    source.add_argument("--manifest", type=Path, help="JSON runtime manifest")
    _add_state(up)
    up.add_argument(
        "--host",
        default=LOOPBACK_IPV4,
        help="loopback address (127.0.0.1 or ::1)",
    )
    up.add_argument(
        "--port",
        type=int,
        default=0,
        help="control port, or 0 to let the OS choose",
    )
    up.add_argument(
        "--poll-interval",
        type=float,
        default=0.05,
        dest="poll_interval",
        help="seconds between supervisor polls",
    )
    up.add_argument(
        "--metrics",
        action="store_true",
        help="serve /metrics on 127.0.0.1",
    )

    down = sub.add_parser("down", help="stop the supervisor")
    _add_state(down)

    status = sub.add_parser("status", help="show per-service state")
    _add_state(status)
    status.add_argument("--json", action="store_true", help="print one JSON object")

    health = sub.add_parser(
        "health", help="exit non-zero when a critical service is down"
    )
    _add_state(health)
    health.add_argument("--json", action="store_true", help="print one JSON object")

    logs = sub.add_parser("logs", help="show structured logs")
    _add_state(logs)
    logs.add_argument("--service", default="", help="only this service")
    logs.add_argument("--level", default="", help="minimum level (DEBUG..CRITICAL)")
    logs.add_argument("--tail", type=int, default=200, help="maximum lines to print")
    logs.add_argument(
        "--follow",
        action="store_true",
        help="keep printing new lines until interrupted",
    )

    restart = sub.add_parser("restart", help="restart one service")
    restart.add_argument("service", help="service role to restart")
    _add_state(restart)
    return parser


def main(
    argv: list[str] | None = None,
    *,
    out: TextIO | None = None,
    err: TextIO | None = None,
) -> int:
    """Run one operator command. Returns a process exit code."""
    stdout = sys.stdout if out is None else out
    stderr = sys.stderr if err is None else err
    args = build_parser().parse_args(argv)
    try:
        return _dispatch(args, stdout, stderr)
    except (ControlError, ManifestError, SupervisorError, OSError, ValueError) as exc:
        print(str(exc), file=stderr)
        return 1


def _dispatch(args: argparse.Namespace, stdout: TextIO, stderr: TextIO) -> int:
    command = args.command
    if command == "up":
        return _cmd_up(args, stdout, stderr)
    if command == "down":
        return _cmd_down(args, stdout, stderr)
    if command == "status":
        return _cmd_status(args, stdout, stderr)
    if command == "health":
        return _cmd_health(args, stdout, stderr)
    if command == "logs":
        return _cmd_logs(args, stdout, stderr)
    if command == "restart":
        return _cmd_restart(args, stdout, stderr)
    raise ControlError(f"unknown command {command!r}")


def _cmd_up(args: argparse.Namespace, stdout: TextIO, stderr: TextIO) -> int:
    del stderr
    host = require_loopback_host(args.host)
    state_path = _state_path(args)
    if state_path.is_file() and ping_state(state_path):
        raise ControlError("supervisor is already running")
    if args.manifest is not None:
        manifest, robot_class = load_manifest_path(args.manifest)
    else:
        manifest, robot_class = class_manifest(args.robot_class)
    run_id = resolve_run_id(None)
    logs = LogBuffer()
    logger = RuntimeLog(
        service="supervisor",
        robot_class=robot_class.value,
        run_id=run_id,
        level=resolve_log_level(manifest.log_level),
        log_format=resolve_log_format(manifest.log_format),
        stream=logs,
    )
    metrics = Registry()
    supervisor = Supervisor(
        manifest,
        robot_class=robot_class,
        logger=logger,
        run_id=run_id,
        metrics=metrics,
        metrics_enabled=True if args.metrics else False,
        metrics_host=LOOPBACK_IPV4 if args.metrics else None,
    )
    session = ControlSession(
        supervisor,
        manifest,
        logs,
        state_path=state_path,
        host=host,
        port=args.port,
        poll_interval=args.poll_interval,
        metrics=metrics,
    )

    def _ready(bound_host: str, port: int) -> None:
        print(f"ohho-runtime listening on {bound_host}:{port}", file=stdout, flush=True)

    try:
        session.serve(ready=_ready)
    except KeyboardInterrupt:
        print("ohho-runtime stopped", file=stdout, flush=True)
        return 0
    print("ohho-runtime stopped", file=stdout, flush=True)
    return 0


def _cmd_down(args: argparse.Namespace, stdout: TextIO, stderr: TextIO) -> int:
    del stderr
    _call(args, {"op": "down"})
    print("stopped", file=stdout)
    return 0


def _cmd_status(args: argparse.Namespace, stdout: TextIO, stderr: TextIO) -> int:
    del stderr
    report = _call(args, {"op": "status"})
    _emit_report(report, json_output=bool(args.json), health=False, stdout=stdout)
    return 0


def _cmd_health(args: argparse.Namespace, stdout: TextIO, stderr: TextIO) -> int:
    del stderr
    report = _call(args, {"op": "health"})
    _emit_report(report, json_output=bool(args.json), health=True, stdout=stdout)
    return 0 if report.get("healthy") is True else 1


def _cmd_logs(args: argparse.Namespace, stdout: TextIO, stderr: TextIO) -> int:
    if args.tail < 0:
        print("tail must be >= 0", file=stderr)
        return 1
    after = 0
    printed = False
    try:
        while True:
            result = _call(
                args,
                {
                    "op": "logs",
                    "service": args.service,
                    "level": args.level,
                    "tail": args.tail,
                    "after": after,
                },
            )
            lines = result.get("lines")
            if not isinstance(lines, list):
                raise ControlError("control response has no lines")
            for line in lines:
                if not isinstance(line, str):
                    raise ControlError("log line is not a string")
                print(line, file=stdout)
                printed = True
            if not args.follow:
                return 0
            cursor = result.get("cursor")
            if isinstance(cursor, int) and not isinstance(cursor, bool):
                after = cursor
            time.sleep(_FOLLOW_INTERVAL_S)
    except KeyboardInterrupt:
        return 0
    except ControlError as exc:
        if args.follow and printed:
            return 0
        print(str(exc), file=stderr)
        return 1


def _cmd_restart(args: argparse.Namespace, stdout: TextIO, stderr: TextIO) -> int:
    del stderr
    result = _call(args, {"op": "restart", "service": args.service})
    service = result.get("service", args.service)
    restarts = result.get("restarts", 0)
    state = result.get("state", "")
    print(f"restarted {service} restarts={restarts} state={state}", file=stdout)
    return 0


def _call(args: argparse.Namespace, payload: Mapping[str, object]) -> dict[str, object]:
    path = _state_path(args)
    if not path.is_file():
        raise ControlError("supervisor is not running")
    state = read_state(path)
    response = transact(state.host, state.port, payload)
    if response.get("ok") is not True:
        error = response.get("error")
        raise ControlError(str(error or "control request failed"))
    result = response.get("result")
    if not isinstance(result, dict):
        raise ControlError("control response has no result")
    return result


def _emit_report(
    report: Mapping[str, object],
    *,
    json_output: bool,
    health: bool,
    stdout: TextIO,
) -> None:
    if json_output:
        print(
            json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False),
            file=stdout,
        )
        return
    if health:
        print(_format_health(report), file=stdout)
        return
    print(_format_status(report), file=stdout)


def _format_status(report: Mapping[str, object]) -> str:
    lines = [
        f"state: {report.get('state', '')}",
        f"run_id: {report.get('run_id', '')}",
        f"robot_class: {report.get('robot_class', '')}",
        f"do_not_operate: {str(report.get('do_not_operate', False)).lower()}",
        f"{'service':<24} {'state':<12} {'restarts':>8} "
        f"{'heartbeat_age_s':>16} {'health':<10}",
    ]
    services = report.get("services")
    if isinstance(services, list):
        for row in services:
            if not isinstance(row, dict):
                continue
            age = row.get("heartbeat_age_s")
            if isinstance(age, bool) or not isinstance(age, (int, float)):
                age_text = "-"
            else:
                age_text = f"{float(age):.3f}"
            restarts = row.get("restarts", 0)
            if isinstance(restarts, bool) or not isinstance(restarts, int):
                restarts = 0
            lines.append(
                f"{str(row.get('service', '')):<24} "
                f"{str(row.get('state', '')):<12} "
                f"{restarts:>8} "
                f"{age_text:>16} "
                f"{str(row.get('health', '')):<10}"
            )
    return "\n".join(lines)


def _format_health(report: Mapping[str, object]) -> str:
    if report.get("healthy") is True:
        return "healthy"
    unhealthy = report.get("unhealthy")
    if isinstance(unhealthy, list) and unhealthy:
        names = ", ".join(str(name) for name in unhealthy)
        return f"unhealthy: {names}"
    if report.get("do_not_operate") is True:
        return "unhealthy: do_not_operate"
    return "unhealthy"


def _add_state(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--state",
        type=Path,
        help="control state file (default: ~/.ohho/runtime/control.json)",
    )


def _state_path(args: argparse.Namespace) -> Path:
    path = args.state
    if path is None:
        return default_state_path()
    return Path(path)


if __name__ == "__main__":
    raise SystemExit(main())
