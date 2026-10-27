"""Simulation-target check and an in-memory command log (OHH-285).

Non-loopback targets are refused. With ``sitl=True``, docker-local names are
also allowed. This module does not open a socket, resolve DNS, or run a
simulator. The command log is the stand-in the tests use for "what would
have been sent". Force-disarm and flight termination are dropped before
they enter the log.
"""

from __future__ import annotations

import ipaddress
import threading
from dataclasses import dataclass

from .estop import EstopCommand, command_is_forbidden, estop_commands

_LOOPBACK_NAMES = frozenset({"localhost"})
_DOCKER_LOCAL_NAMES = frozenset(
    {
        "host.docker.internal",
        "gateway.docker.internal",
        "host.containers.internal",
    }
)


def _wire_host(host: str) -> str:
    """Host string with surrounding brackets and whitespace removed."""
    text = str(host).strip()
    if text.startswith("[") and text.endswith("]"):
        return text[1:-1]
    return text


def _normalize_host(host: str) -> str:
    return _wire_host(host).lower()


def _parse_ip(host: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    try:
        return ipaddress.ip_address(_normalize_host(host))
    except ValueError:
        return None


def _is_loopback_host(host: str) -> bool:
    """True for 127.0.0.0/8, ::1, IPv4-mapped loopback, and the name localhost."""
    text = _normalize_host(host)
    if text in _LOOPBACK_NAMES:
        return True
    parsed = _parse_ip(host)
    if parsed is None:
        return False
    mapped = getattr(parsed, "ipv4_mapped", None)
    if mapped is not None:
        return bool(mapped.is_loopback)
    return bool(parsed.is_loopback)


def _is_docker_local_host(host: str) -> bool:
    """True for well-known local container DNS names, not routable LAN addresses."""
    text = _normalize_host(host)
    if text in _DOCKER_LOCAL_NAMES:
        return True
    return text.endswith(".docker.internal") or text.endswith(".containers.internal")


def simulation_target_allowed(host: str, *, sitl: bool = False) -> bool:
    """Return whether a simulation link may target ``host``.

    Loopback is always allowed. Docker-local names are allowed only when
    ``sitl`` is set. Every other host is refused. No socket is opened.
    """
    if _is_loopback_host(host):
        return True
    return bool(sitl and _is_docker_local_host(host))


def simulation_refusal_reason(host: str, *, sitl: bool = False) -> str | None:
    """Error string when ``host`` is outside the simulation allow-list."""
    if simulation_target_allowed(host, sitl=sitl):
        return None
    if sitl:
        return (
            f"Refusing connection to {host!r}: SITL mode is simulation-only and "
            "still requires a loopback host (127.0.0.0/8, ::1, localhost) or a "
            "docker-local host such as host.docker.internal. Other hosts are refused."
        )
    return (
        f"Refusing connection to {host!r}: this transport is simulation-only and "
        "only accepts loopback hosts (127.0.0.0/8, ::1, localhost). "
        "Pass sitl=True to also allow docker-local hosts. Other hosts are refused."
    )


@dataclass(frozen=True, slots=True)
class SimulationTarget:
    """Result of the allow-list check. ``allowed`` false means no commands are logged."""

    host: str
    sitl: bool
    allowed: bool
    reason: str | None


def check_simulation_target(host: str, *, sitl: bool = False) -> SimulationTarget:
    """Check ``host`` without opening a socket."""
    reason = simulation_refusal_reason(host, sitl=sitl)
    return SimulationTarget(host=host, sitl=sitl, allowed=reason is None, reason=reason)


class MockCommandLog:
    """Command log gated by the simulation allow-list.

    ``connect`` records the allow-list decision and does not open a socket.
    ``record`` drops force-disarm and flight termination. ``apply_estop``
    logs :func:`ohho.drone.estop.estop_commands` only after an allowed connect.
    """

    def __init__(self, host: str = "127.0.0.1", *, sitl: bool = False) -> None:
        self.host = host
        self.sitl = sitl
        self._commands: list[EstopCommand] = []
        self._allowed = False
        self._decision: SimulationTarget | None = None
        self._dropped_forbidden = 0
        self._lock = threading.Lock()

    @property
    def connected(self) -> bool:
        """True after ``connect`` allowed this host."""
        with self._lock:
            return self._allowed

    @property
    def refusal_reason(self) -> str | None:
        """Refusal from the last ``connect``, or None if none was refused."""
        with self._lock:
            if self._decision is None:
                return None
            return self._decision.reason

    @property
    def dropped_forbidden(self) -> int:
        """How many force-disarm or flight-termination commands were dropped."""
        with self._lock:
            return self._dropped_forbidden

    def connect(self) -> SimulationTarget:
        """Apply the allow-list. Does not open a socket or resolve a name."""
        decision = check_simulation_target(self.host, sitl=self.sitl)
        with self._lock:
            self._decision = decision
            self._allowed = decision.allowed
        return decision

    def record(self, command: EstopCommand) -> bool:
        """Append ``command`` if the link is allowed and the command is not forbidden.

        Returns False when the command is dropped. A disconnected or refused
        link records nothing.
        """
        if not isinstance(command, EstopCommand):
            raise TypeError("command must be an EstopCommand")
        if command_is_forbidden(command):
            with self._lock:
                self._dropped_forbidden += 1
            return False
        with self._lock:
            if not self._allowed:
                return False
            self._commands.append(command)
        return True

    def apply_estop(
        self,
        landed_state: object,
        altitude_m: object = None,
    ) -> tuple[EstopCommand, ...]:
        """Log the e-stop decision. A refused link logs nothing and returns ``()``."""
        if not self.connected:
            return ()
        commands = estop_commands(landed_state, altitude_m)
        recorded: list[EstopCommand] = []
        for command in commands:
            if self.record(command):
                recorded.append(command)
        return tuple(recorded)

    def received_commands(self) -> tuple[EstopCommand, ...]:
        """Copy of the log, in arrival order."""
        with self._lock:
            return tuple(self._commands)

    def received_tuples(self) -> tuple[tuple[str, int, float, float], ...]:
        """``(name, mav_command, param1, param2)`` in arrival order."""
        return tuple(
            (command.name, command.mav_command, command.param1, command.param2)
            for command in self.received_commands()
        )


__all__ = [
    "MockCommandLog",
    "SimulationTarget",
    "check_simulation_target",
    "simulation_refusal_reason",
    "simulation_target_allowed",
]
