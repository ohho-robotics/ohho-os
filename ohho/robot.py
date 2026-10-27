"""The Robot Abstraction Layer — the public API.

``Robot`` is what your code, the agent, and the training pipelines talk to. It is
backend- and protocol-agnostic: the same calls work on any robot and either
runtime, with capabilities gating what actually happens (a robot that can't do
something turns the call into a safe no-op rather than crashing).

Flight methods (``arm_flight``, ``disarm_flight``, ``takeoff``, ``land``,
``rtl``, ``hold``, ``offboard_position``, ``offboard_velocity``, ``set_yaw``)
no-op unless the spec includes ``aerial``. They are experimental: nothing has
flown. Legged pose methods (``stand``, ``lie_down``, ``sit``, ``recover``,
``set_body_pose``) no-op unless the spec includes ``legged.pose``. ``drive``
is a planar base command and does not move a quadruped that is lying or sitting.
"""

from __future__ import annotations

from typing import Iterable, Optional

from . import capabilities as caps
from .adapters import resolve_transport
from .aerial import Frame, finite_number, require_frame
from .registry import RobotSpec, get_spec
from .runtime import Runtime, get_runtime
from .planner import agent_max_linear_speed
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
        """
        spec = get_spec(robot_id)
        rt = get_runtime(runtime)
        tp = resolve_transport(transport, spec, rt)
        if tp.protocol != "simulated":
            tp = SafetyGate(
                tp, spec, config=resolve_config(safety_config), estop_damp=estop_damp
            )
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
        """Drive the base in the plane.

        Clamped to the robot's limits. ``vy`` is ignored on non-holonomic
        bases. No-op when the robot cannot drive. The command is vx, vy, and
        yaw rate; vertical speed is an offboard velocity setpoint.
        """
        if not self.has(caps.BASE_DRIVE):
            return self
        if not self.has(caps.BASE_HOLONOMIC):
            vy = 0.0
        if self.has(caps.LEGGED_POSE) and not self._legged_ready_to_drive():
            return self
        lin_cap = agent_max_linear_speed(self)
        vx = clamp(vx, -lin_cap, lin_cap)
        vy = clamp(vy, -lin_cap, lin_cap)
        w = clamp(w, -self.spec.max_ang, self.spec.max_ang)
        self._t.send_velocity(Velocity(vx, vy, w))
        return self

    def stop(self) -> "Robot":
        self._t.send_velocity(Velocity())
        return self

    def move_joints(self, positions: Iterable[float]) -> "Robot":
        """Command arm joints (in spec order). No-op if the robot has no arm."""
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

    # ── legged pose (one API: capability no-op, main's limits and sim behaviour) ─
    def stand(self) -> "Robot":
        """Stand up (quadruped). No-op unless ``legged.pose`` is supported."""
        self._legged_action("stand_up")
        return self

    def lie_down(self) -> "Robot":
        """Lie down / stand down. No-op unless ``legged.pose`` is supported."""
        self._legged_action("stand_down")
        return self

    def sit(self) -> "Robot":
        """Sit. No-op unless ``legged.pose`` is supported."""
        self._legged_action("sit")
        return self

    def recover(self) -> "Robot":
        """Recovery stand from prone. No-op unless ``legged.pose`` is supported."""
        self._legged_action("recovery_stand")
        return self

    def set_body_pose(
        self, roll: float = 0.0, pitch: float = 0.0, yaw: float = 0.0
    ) -> "Robot":
        """Set body orientation (radians), clamped to the safety euler limit.

        No-op unless ``legged.pose`` is supported. The clamp is the module
        ceiling; it is not widened.
        """
        if not self.has(caps.LEGGED_POSE):
            return self
        from .safety import EULER_LIMIT_RAD

        lim = EULER_LIMIT_RAD
        r = clamp(float(roll), -lim, lim)
        p = clamp(float(pitch), -lim, lim)
        y = clamp(float(yaw), -lim, lim)
        self._legged_action("euler", r, p, y)
        return self

    def legged_posture(self) -> str:
        """Simulator / telemetry hint: ``standing``, ``lying``, ``sitting``, or ``unknown``."""
        if not self.has(caps.LEGGED_POSE):
            return "n/a"
        inner = getattr(self._t, "_inner", self._t)
        getter = getattr(inner, "legged_posture", None)
        if callable(getter):
            return str(getter())
        custom = self._t.read().custom or {}
        return str(custom.get("legged_posture", custom.get("posture", "unknown")))

    def _legged_ready_to_drive(self) -> bool:
        if not self.has(caps.LEGGED_POSE):
            return True
        return self.legged_posture() in ("standing", "unknown")

    def _legged_action(self, action: str, *args: float) -> bool:
        if not self.has(caps.LEGGED_POSE):
            return False
        gate = self._t
        if isinstance(gate, SafetyGate):
            return bool(gate.command(action, *args))
        cmd = getattr(gate, "command", None)
        if callable(cmd):
            return bool(cmd(action, *args))
        sport = getattr(gate, "sport_command", None)
        if callable(sport):
            sport(action, *args)
            return True
        mode = getattr(gate, "send_mode", None)
        if callable(mode):
            if action == "euler" and len(args) == 3:
                mode("euler", roll=args[0], pitch=args[1], yaw=args[2])
            else:
                mode(action)
            return True
        return False

    # ── flight (experimental: nothing has flown) ─────────────────────────────
    def _dispatch_flight(
        self, action: str, args: tuple[object, ...], /, **fields: object
    ) -> "Robot":
        """Send one flight request.

        No-op without ``aerial``. A SafetyGate receives ``command``; any other
        transport must implement ``send_flight``.
        """
        if not self.has(caps.AERIAL):
            return self
        if isinstance(self._t, SafetyGate):
            self._t.command(action, *args)
            return self
        send = getattr(self._t, "send_flight", None)
        if not callable(send):
            raise AttributeError(
                f"{type(self._t).__name__} does not implement send_flight"
            )
        send(action, **fields)
        return self

    def arm_flight(self) -> "Robot":
        """Ask the autopilot to arm the motors.

        :meth:`arm` unlocks the SafetyGate. This method sends an autopilot arm
        request and does not call :meth:`arm`, so the gate's armed state stays
        as it was. On a gated transport the request is
        ``SafetyGate.command("arm")``. No-op without ``aerial``.

        Experimental: nothing has flown.
        """
        if not self.has(caps.AERIAL):
            return self
        return self._dispatch_flight("arm", ())

    def disarm_flight(self) -> "Robot":
        """Ask the autopilot for a normal disarm.

        The request is the action ``disarm`` with no force flag. No-op without
        ``aerial``. Whether an airborne disarm is refused is up to the backend.

        Experimental: nothing has flown.
        """
        if not self.has(caps.AERIAL):
            return self
        return self._dispatch_flight("disarm", ())

    def takeoff(self, alt_m: float) -> "Robot":
        """Climb to ``alt_m`` metres above the local origin, then hold.

        ``alt_m`` is positive up in every frame. No-op without ``aerial``.
        """
        if not self.has(caps.AERIAL):
            return self
        altitude = _positive_up("alt_m", alt_m)
        return self._dispatch_flight("takeoff", (altitude,), alt_m=altitude)

    def land(self) -> "Robot":
        """Land at the current horizontal position. No-op without ``aerial``."""
        if not self.has(caps.AERIAL):
            return self
        return self._dispatch_flight("land", ())

    def rtl(self) -> "Robot":
        """Ask the autopilot to return to launch. No-op without ``aerial``.

        Ohho does not fly the return itself.
        """
        if not self.has(caps.AERIAL):
            return self
        return self._dispatch_flight("rtl", ())

    def hold(self) -> "Robot":
        """Ask the autopilot to loiter. No-op without ``aerial``."""
        if not self.has(caps.AERIAL):
            return self
        return self._dispatch_flight("hold", ())

    def offboard_position(
        self, x: float, y: float, z: float, yaw: float, frame: Frame
    ) -> "Robot":
        """Position setpoint. ``frame`` is required. No-op without ``aerial``.

        ``z`` is forwarded in ``frame`` (down is positive in ``LOCAL_NED`` and
        ``BODY_FRD``). This method does not rewrite one local frame as the other.
        """
        frame = require_frame(frame)
        if not self.has(caps.AERIAL):
            return self
        x = finite_number("x", x)
        y = finite_number("y", y)
        z = finite_number("z", z)
        yaw = finite_number("yaw", yaw)
        return self._dispatch_flight(
            "offboard_position",
            (x, y, z, yaw, frame),
            x=x,
            y=y,
            z=z,
            yaw=yaw,
            frame=frame,
        )

    def offboard_velocity(
        self, vx: float, vy: float, vz: float, yaw_rate: float, frame: Frame
    ) -> "Robot":
        """Velocity setpoint. ``frame`` is required. No-op without ``aerial``.

        ``vz`` is forwarded in ``frame``. Positive ``vz`` descends in
        ``LOCAL_NED`` and ``BODY_FRD``.
        """
        frame = require_frame(frame)
        if not self.has(caps.AERIAL):
            return self
        vx = finite_number("vx", vx)
        vy = finite_number("vy", vy)
        vz = finite_number("vz", vz)
        yaw_rate = finite_number("yaw_rate", yaw_rate)
        return self._dispatch_flight(
            "offboard_velocity",
            (vx, vy, vz, yaw_rate, frame),
            vx=vx,
            vy=vy,
            vz=vz,
            yaw_rate=yaw_rate,
            frame=frame,
        )

    def set_yaw(self, yaw: float, frame: Frame) -> "Robot":
        """Absolute yaw in ``frame``. ``frame`` is required.

        No-op without ``aerial``.
        """
        frame = require_frame(frame)
        if not self.has(caps.AERIAL):
            return self
        yaw = finite_number("yaw", yaw)
        return self._dispatch_flight("set_yaw", (yaw, frame), yaw=yaw, frame=frame)

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


def _positive_up(name: str, value: object) -> float:
    """Metres, positive up. Used by :meth:`Robot.takeoff`."""
    number = finite_number(name, value)
    if number <= 0.0:
        raise ValueError(f"{name} must be positive up, in metres")
    return number


def connect(
    robot_id: str,
    transport: Optional[str] = None,
    runtime: str = "auto",
    *,
    safety_config: Optional[str] = None,
    estop_damp: bool = False,
) -> Robot:
    """Module-level shortcut for :meth:`Robot.connect`."""
    return Robot.connect(
        robot_id,
        transport=transport,
        runtime=runtime,
        safety_config=safety_config,
        estop_damp=estop_damp,
    )
