"""Read-only doctor checks for this plugin.

The hook reports whether the SDK is installed, the robot answers, the model
is one this plugin supports, the firmware is new enough, and a network
interface is named. It never publishes and never calls a state-changing RPC.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

from ohho.plugins import DoctorReport

from .messages import UnknownRobotModel, is_supported_model

MIN_FIRMWARE: tuple[int, ...] = (1, 7, 2)
MIN_FIRMWARE_TEXT = "1.7.2"
_VERSION = re.compile(r"(\d+(?:\.\d+)*)")


class Unreachable(RuntimeError):
    """The read-only probe could not reach the robot."""


@dataclass(frozen=True)
class ProbeResult:
    """What a read-only probe learned. ``firmware`` may be empty."""

    model: object
    firmware: object
    interface: str


def parse_firmware(value: object) -> tuple[int, ...] | None:
    """Numeric firmware tuple, or ``None`` when ``value`` has no version."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        text = str(value)
    elif isinstance(value, str):
        text = value.strip()
        if text.lower().startswith("v"):
            text = text[1:].strip()
    else:
        return None
    match = _VERSION.search(text)
    if match is None:
        return None
    return tuple(int(part) for part in match.group(1).split("."))


def firmware_ok(value: object, minimum: tuple[int, ...] = MIN_FIRMWARE) -> bool:
    """True when ``value`` is ``minimum`` or newer."""
    parsed = parse_firmware(value)
    if parsed is None:
        return False
    width = max(len(parsed), len(minimum))
    left = parsed + (0,) * (width - len(parsed))
    right = minimum + (0,) * (width - len(minimum))
    return left >= right


def _address_body(uri: str | None) -> str:
    if not isinstance(uri, str):
        return ""
    text = uri.strip()
    if "://" in text:
        text = text.split("://", 1)[1]
    return text.strip()


def _interface_name(address: str, network_interface: str | None) -> str:
    if isinstance(network_interface, str) and network_interface.strip():
        return network_interface.strip()
    if not address:
        return ""
    from .sdk import parse_address

    try:
        host, _domain = parse_address(address)
    except ValueError:
        return ""
    return host.strip()


def _default_loader() -> object:
    from .sdk import _load_sdk

    return _load_sdk()


def _default_probe(address: str) -> ProbeResult:
    """Connect read-only and return model plus firmware. Does not command."""
    from .sdk import CommandFailed, open_session

    session = open_session(address, read_only=True)
    try:
        try:
            session.connect()
        except UnknownRobotModel as exc:
            return ProbeResult(model=exc.model, firmware=None, interface=session.host)
        model, firmware = session.robot_info()
        return ProbeResult(model=model, firmware=firmware, interface=session.host)
    except CommandFailed as exc:
        raise Unreachable(str(exc)) from exc
    finally:
        session.close()


def diagnose(
    uri: str | None,
    *,
    network_interface: str | None = None,
    loader: Callable[[], object] | None = None,
    probe: Callable[[str], ProbeResult] | None = None,
) -> DoctorReport:
    """Run the five checks. Failures name SDK, reachability, model or firmware.

    A missing network interface fails before the SDK is imported and before
    any probe runs. The probe, when it runs, is the read-only one.
    """
    address = _address_body(uri)
    interface = _interface_name(address, network_interface)
    if not interface:
        return DoctorReport(
            ok=False,
            summary="network interface missing",
            details=("network interface: missing",),
        )
    if not address:
        return DoctorReport(
            ok=False,
            summary="unreachable: no transport URI",
            details=("reachable: no", f"network interface: {interface}"),
        )
    from .sdk import SdkUnavailable

    load = loader or _default_loader
    try:
        load()
    except SdkUnavailable as exc:
        return DoctorReport(
            ok=False,
            summary=f"SDK missing: {exc}",
            details=("sdk: missing",),
        )
    ask = probe or _default_probe
    try:
        facts = ask(address)
    except UnknownRobotModel as exc:
        return DoctorReport(
            ok=False,
            summary=f"unknown model: {exc.model!r}",
            details=(
                "reachable: yes",
                "model: unknown",
                f"network interface: {interface}",
            ),
        )
    except (Unreachable, OSError, TimeoutError, ConnectionError) as exc:
        return DoctorReport(
            ok=False,
            summary=f"unreachable: {exc}",
            details=("reachable: no", f"network interface: {interface}"),
        )
    model = facts.model
    if not is_supported_model(model):
        return DoctorReport(
            ok=False,
            summary=f"unknown model: {model!r}",
            details=(
                "sdk: installed",
                "reachable: yes",
                "model: unknown",
                f"network interface: {interface}",
            ),
        )
    firmware = facts.firmware
    if not firmware_ok(firmware):
        shown = "missing" if firmware in (None, "") else firmware
        return DoctorReport(
            ok=False,
            summary=f"firmware too old: {shown} (need {MIN_FIRMWARE_TEXT} or later)",
            details=(
                "sdk: installed",
                "reachable: yes",
                f"model: {model}",
                "firmware: too old",
                f"network interface: {interface}",
            ),
        )
    reported = interface or facts.interface
    return DoctorReport(
        ok=True,
        summary="ok",
        details=(
            "sdk: installed",
            "reachable: yes",
            f"model: {model}",
            f"firmware: {firmware}",
            f"network interface: {reported}",
        ),
    )
