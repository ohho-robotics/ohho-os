"""The Robot Abstraction Layer — the public API.

``Robot`` is what your code, the agent, and the training pipelines talk to. It is
backend- and protocol-agnostic: the same calls work on any robot and either
runtime, with capabilities gating what actually happens (a robot that can't do
something turns the call into a safe no-op rather than crashing).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Iterable, Optional, Union

from . import capabilities as caps
from . import humanoid
from .adapters import resolve_transport
from .config import RobotProfile, merge_safety_overrides, spec_with_overrides
from .registry import RobotSpec, get_spec
from .runtime import Runtime, get_runtime
from .safety import SafetyGate, resolve_config
from .schema import Telemetry, TransportStatus, Velocity, clamp
from .transport import Transport


class Robot:
    def __init__(self, spec: RobotSpec, transport: Transport, runtime: Runtime) -> None:
        self.spec = spec
        self._t = transport
        self._rt = runtime
        self._latest: Optional[Telemetry] = None
        self._unsub = transport.on_telemetry(self._on_tele)

    @classmethod
    def connect(
        cls,
        robot_id: str,
        transport: Optional[str] = None,
        runtime: str = "auto",
        *,
        safety_config: Optional[str] = None,
        estop_damp: bool = False,
        sim_backend: Optional[str] = None,
        asset_path: Optional[str] = None,
        safety_overrides: Optional[Mapping[str, Any]] = None,
    ) -> "Robot":
        """Connect to a robot and start its runtime.

        ``transport`` is a URI like ``"sim://"`` or ``"dds://192.168.1.10"``; when
        omitted, the robot's declared adapter is used (falling back to simulation
        if it isn't bundled). ``runtime`` is ``"auto"`` | ``"native"`` | ``"ros2"``.

        Every non-simulated transport is wrapped in a :class:`~ohho.safety.SafetyGate`
        and starts **disarmed** — call :meth:`arm` before driving.
        ``safety_config`` is a safety config JSON path (default:
        ``$OHHO_SAFETY_CONFIG``); ``estop_damp`` damps the motors after an
        e-stop even when standing.

        ``sim_backend``, ``asset_path`` and ``safety_overrides`` come from a
        :class:`~ohho.config.RobotProfile`. They are omitted by existing callers.
        Overrides may only tighten caps; a looser profile is refused at load,
        before this method runs. The gate's own rules are unchanged.
        """
        spec = get_spec(robot_id)
        if sim_backend or asset_path:
            spec = spec_with_overrides(
                spec, sim_backend=sim_backend, asset_path=asset_path
            )
        rt = get_runtime(runtime)
        tp = resolve_transport(transport, spec, rt)
        if tp.protocol != "simulated":
            cfg = resolve_config(safety_config)
            if safety_overrides:
                checked = RobotProfile(spec.id, safety_overrides=dict(safety_overrides))
                cfg = merge_safety_overrides(cfg, spec.id, checked.safety_overrides)
            tp = SafetyGate(tp, spec, config=cfg, estop_damp=estop_damp)
        tp.connect()
        bot = cls(spec, tp, rt)
        # live backends that own a loop start it here
        start_sim = getattr(tp, "start_sim", None)
        if callable(start_sim):
            start_sim()
        rt.start()
        return bot

    # ── introspection ─────────────────────────────────────────────────────────
    @property
    def simulated(self) -> bool:
        return self._t.protocol == "simulated"

    @property
    def runtime(self) -> Runtime:
        return self._rt

    @property
    def transport(self) -> Transport:
        return self._t

    @property
    def gated(self) -> bool:
        """True when commands pass through a :class:`~ohho.safety.SafetyGate`."""
        return isinstance(self._t, SafetyGate)

    def has(self, capability: str) -> bool:
        return self.spec.has(capability)

    # ── safety ────────────────────────────────────────────────────────────────
    def arm(self) -> bool:
        """Enable motion on a gated robot (see :meth:`SafetyGate.arm`).
        Ungated (simulated) robots are always armed."""
        if isinstance(self._t, SafetyGate):
            return self._t.arm()
        return True

    def disarm(self) -> "Robot":
        if isinstance(self._t, SafetyGate):
            self._t.disarm()
        return self

    # ── control ───────────────────────────────────────────────────────────────
    def drive(self, vx: float = 0.0, vy: float = 0.0, w: float = 0.0) -> "Robot":
        """Drive the base. Clamped to the robot's limits; ``vy`` ignored on
        non-holonomic bases; a no-op if the robot can't drive.

        A humanoid walks with this same call (``vx``, ``vy``, yaw rate). It
        does not change posture; :meth:`walk_mode` does that.
        """
        if not self.has(caps.BASE_DRIVE):
            return self
        if not self.has(caps.BASE_HOLONOMIC):
            vy = 0.0
        vx = clamp(vx, -self.spec.max_lin, self.spec.max_lin)
        vy = clamp(vy, -self.spec.max_lin, self.spec.max_lin)
        w = clamp(w, -self.spec.max_ang, self.spec.max_ang)
        self._t.send_velocity(Velocity(vx, vy, w))
        return self

    def stop(self) -> "Robot":
        self._t.send_velocity(Velocity())
        return self

    def move_joints(
        self,
        positions: Union[str, Iterable[float]],
        q: Optional[Iterable[float]] = None,
        kp: Optional[float] = None,
        kd: Optional[float] = None,
    ) -> "Robot":
        """Command joints.

        A sequence of positions commands the arm in spec order and is a no-op
        without ``manipulation``. A group name commands that whole-body group
        (``move_joints(group, q, kp=None, kd=None)``) and is a no-op without
        ``whole_body.joint``. Gains on the arm form are a ``TypeError``.
        """
        if isinstance(positions, str):
            return humanoid.move_joints(self, positions, q, kp=kp, kd=kd)
        if q is not None or kp is not None or kd is not None:
            raise TypeError(
                "move_joints(positions) commands the arm; "
                "move_joints(group, q) commands a whole-body group"
            )
        if not self.has(caps.MANIPULATION):
            return self
        for name, p in zip(self.spec.joint_names, positions):
            self._t.send_joint_command(name, float(p))
        return self

    def emergency_stop(self) -> "Robot":
        self._t.emergency_stop()
        return self

    def release_stop(self) -> "Robot":
        self._t.release_stop()
        return self

    # ── humanoid (no-op without the capability; gated calls enter the gate) ──
    def prepare(self) -> "Robot":
        """Stand and hold balance. No-op without ``humanoid.posture``."""
        return humanoid.prepare(self)

    def walk_mode(self) -> "Robot":
        """Enable locomotion. No-op without ``humanoid.posture``.

        Walking speed is :meth:`drive`.
        """
        return humanoid.walk_mode(self)

    def damp(self) -> "Robot":
        """Damping on every joint. The robot may collapse.

        No-op without ``humanoid.posture``. Not an e-stop.
        """
        return humanoid.damp(self)

    def get_up(self) -> "Robot":
        """Get-up recovery. No-op without ``humanoid.posture``."""
        return humanoid.get_up(self)

    def lie_down(self) -> "Robot":
        """Lie-down recovery. No-op without ``humanoid.posture``."""
        return humanoid.lie_down(self)

    def look(self, pitch: float, yaw: float) -> "Robot":
        """Point the head, radians. No-op without ``head.gaze``."""
        return humanoid.look(self, pitch, yaw)

    def move_ee(self, side: str, pose: Iterable[float], t: float) -> "Robot":
        """Move one hand. No-op without ``manipulation.dual_arm``."""
        return humanoid.move_ee(self, side, pose, t)

    def gripper(self, side: str, cmd: object) -> "Robot":
        """Command one gripper. No-op without ``end_effector.gripper``."""
        return humanoid.gripper(self, side, cmd)

    def hand(self, side: str, fingers: Iterable[float]) -> "Robot":
        """Command one dexterous hand. No-op without ``end_effector.dex_hand``."""
        return humanoid.hand(self, side, fingers)

    # ── state ─────────────────────────────────────────────────────────────────
    def telemetry(self) -> Telemetry:
        if self._latest is not None:
            return self._latest
        return self._t.read()

    def status(self) -> TransportStatus:
        return self._t.status()

    def _on_tele(self, t: Telemetry) -> None:
        self._latest = t

    # ── teardown ──────────────────────────────────────────────────────────────
    def disconnect(self) -> None:
        try:
            self._unsub()
        except Exception:
            pass
        stop_sim = getattr(self._t, "stop_sim", None)
        if callable(stop_sim):
            stop_sim()
        try:
            self._t.disconnect()
        finally:
            self._rt.stop()

    def __enter__(self) -> "Robot":
        return self

    def __exit__(self, *exc) -> None:
        self.disconnect()

    def __repr__(self) -> str:
        kind = "sim" if self.simulated else self._t.protocol
        return f"<Robot {self.spec.id} via {kind} on {self._rt.name}>"


def connect(
    robot_id: str,
    transport: Optional[str] = None,
    runtime: str = "auto",
    *,
    safety_config: Optional[str] = None,
    estop_damp: bool = False,
    sim_backend: Optional[str] = None,
    asset_path: Optional[str] = None,
    safety_overrides: Optional[Mapping[str, Any]] = None,
) -> Robot:
    """Module-level shortcut for :meth:`Robot.connect`."""
    return Robot.connect(
        robot_id,
        transport=transport,
        runtime=runtime,
        safety_config=safety_config,
        estop_damp=estop_damp,
        sim_backend=sim_backend,
        asset_path=asset_path,
        safety_overrides=safety_overrides,
    )
