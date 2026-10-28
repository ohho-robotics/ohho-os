"""Drone SafetyGate profile (OHH-288).

Fake clock, fake transports, injected TTY. No sleeps and no hardware.
Go2 and OmniBot rules are covered by tests.test_safety; cases here that
touch those profiles only check the drone work did not adopt them.
"""

from __future__ import annotations

import json
import logging
import math
import shutil
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

from ohho.aerial import Frame
from ohho.drone.estop import EstopCommand
from ohho.registry import RobotSpec, get_spec
from ohho.safety import (
    ARM_ENV,
    ARM_PHRASE,
    DRONE_PROFILE,
    GO2_PROFILE,
    RELEASE_PHRASE,
    AuditLog,
    SafetyConfig,
    SafetyGate,
    SafetyProfile,
    clip_horizontal,
    drone_mcp_allowed,
    fence_violation,
    load_safety_config,
    profile_for,
)
from ohho.schema import ConnectionState, Telemetry, TransportStatus, Velocity
from ohho.transport import BaseTransport

WALL = 1_760_000_000.0


def setUpModule() -> None:
    logging.getLogger("ohho.safety").addHandler(logging.NullHandler())


class FakeClock:
    def __init__(self, t: float = 100.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


class DroneLink(BaseTransport):
    """Records flight requests. Does not open a socket."""

    def __init__(
        self, backend_id: str = "px4-dds", protocol: str = "simulated"
    ) -> None:
        super().__init__()
        self.backend_id = backend_id
        self.protocol = protocol
        self.state = ConnectionState.CONNECTED
        self.flight: list[tuple[str, dict]] = []
        self.velocities: list[tuple[float, float, float]] = []
        self.events: list[str] = []
        self.fail: set[str] = set()
        self.latest = Telemetry()

    def connect(self) -> TransportStatus:
        self.state = ConnectionState.CONNECTED
        return self.status()

    def disconnect(self) -> None:
        self.state = ConnectionState.DISCONNECTED

    def status(self) -> TransportStatus:
        return TransportStatus(protocol=self.protocol, state=self.state, label="drone")

    def read(self) -> Telemetry:
        return self.latest

    def send_velocity(self, vel: Velocity) -> None:
        if "vel" in self.fail:
            raise RuntimeError("vel")
        self.velocities.append((vel.linear_x, vel.linear_y, vel.angular_z))

    def send_joint_command(self, name: str, position: float) -> None:
        return None

    def emergency_stop(self) -> None:
        if "estop" in self.fail:
            raise RuntimeError("estop")
        self.events.append("estop")

    def release_stop(self) -> None:
        self.events.append("release")

    def motor_cut(self) -> None:
        if "motor_cut" in self.fail:
            raise RuntimeError("motor_cut")
        self.events.append("motor_cut")

    def simulated_descent(self) -> None:
        if "descent" in self.fail:
            raise RuntimeError("descent")
        self.events.append("descent")

    def send_flight(self, action: str, **fields: object) -> None:
        if "send_flight" in self.fail or action in self.fail:
            raise RuntimeError(action)
        self.flight.append((action, dict(fields)))

    def emit(self, tele: Telemetry) -> Telemetry:
        self.latest = tele
        self._emit_telemetry(tele)
        return tele

    def actions(self) -> list[str]:
        return [name for name, _fields in self.flight]


def fast(profile: SafetyProfile = DRONE_PROFILE) -> SafetyProfile:
    return replace(profile, max_accel_x=1e6, max_accel_y=1e6, max_accel_wz=1e6)


def drone_telemetry(
    *,
    landed: str = "on_ground",
    x: float = 0.0,
    y: float = 0.0,
    z: float = -0.1,
    frame: str = "LOCAL_NED",
    battery: float | None = 0.9,
    link_ok: bool | None = True,
    include_pose: bool = True,
    pose_frame: str | None = None,
    roll: float | None = None,
    error_code: object = None,
    drone_battery: float | None = None,
) -> Telemetry:
    drone: dict[str, object] = {
        "landed_state": landed,
        "experimental": True,
        "flight_ready": False,
    }
    if link_ok is not None:
        drone["link_ok"] = link_ok
    if include_pose:
        drone["pose"] = {
            "x": x,
            "y": y,
            "z": z,
            "roll": 0.0,
            "pitch": 0.0,
            "yaw": 0.0,
        }
        drone["pose_frame"] = pose_frame or frame
    if drone_battery is not None:
        drone["battery"] = drone_battery
    elif battery is not None:
        drone["battery"] = battery
    custom: dict[str, object] = {"drone": drone}
    if roll is not None:
        custom["roll"] = roll
        custom["pitch"] = 0.0
    if error_code is not None:
        custom["error_code"] = error_code
    return Telemetry(battery=battery, custom=custom)


def _aerial_spec(robot_id: str = "px4-sitl") -> RobotSpec:
    return RobotSpec(
        id=robot_id,
        name="Aerial test",
        category="aerial",
        capabilities=("aerial", "base.drive", "base.holonomic_drive"),
        max_lin=2.0,
        max_ang=0.8,
    )


class DroneCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="ohho-drone-safety-")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.clock = FakeClock()
        self.env: dict[str, str] = {}
        self.tty = True
        self.answer: object = ARM_PHRASE
        self.prompts: list[str] = []

    def _input(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if isinstance(self.answer, BaseException):
            raise self.answer
        return str(self.answer)

    def make(
        self,
        link: DroneLink | None = None,
        *,
        profile: SafetyProfile | None = None,
        spec: RobotSpec | None = None,
        arm: bool = True,
        tele: Telemetry | None = None,
        **kw: object,
    ) -> tuple[SafetyGate, DroneLink]:
        link = link if link is not None else DroneLink()
        kw.setdefault("audit", AuditLog(self.tmp, wall_clock=lambda: WALL))
        kw.setdefault("auto_tick", False)
        gate = SafetyGate(
            link,
            spec if spec is not None else _aerial_spec(),
            profile=profile if profile is not None else DRONE_PROFILE,
            clock=self.clock,
            environ=self.env,
            isatty=lambda: self.tty,
            input_fn=self._input,
            **kw,  # type: ignore[arg-type]
        )
        gate.connect()
        link.emit(tele if tele is not None else drone_telemetry())
        if arm:
            self.assertTrue(gate.arm())
        link.flight.clear()
        link.velocities.clear()
        link.events.clear()
        return gate, link

    def events(self, name: str | None = None) -> list[dict]:
        out: list[dict] = []
        for path in sorted(Path(self.tmp).glob("*.jsonl")):
            for line in path.read_text(encoding="utf-8").splitlines():
                rec = json.loads(line)
                if name is None or rec["event"] == name:
                    out.append(rec)
        return out

    def audit_text(self) -> str:
        return "".join(
            path.read_text(encoding="utf-8") for path in Path(self.tmp).glob("*.jsonl")
        )


class TestDroneProfile(unittest.TestCase):
    def test_defaults_and_known_ids(self):
        self.assertIs(profile_for("px4-sitl"), DRONE_PROFILE)
        self.assertIs(profile_for("crazyflie-cf21"), DRONE_PROFILE)
        self.assertIs(profile_for("px4-x500"), DRONE_PROFILE)
        self.assertIs(profile_for("unitree-go2"), GO2_PROFILE)
        self.assertEqual(profile_for("omnibot").max_vx, 0.2)
        self.assertEqual(
            DRONE_PROFILE.allowlist,
            frozenset(
                {"hold", "land", "rtl", "offboard_velocity", "offboard_position"}
            ),
        )
        self.assertNotIn("land", GO2_PROFILE.allowlist)
        self.assertNotIn("flip", DRONE_PROFILE.allowlist)
        self.assertNotIn("arm", DRONE_PROFILE.allowlist)
        self.assertNotIn("takeoff", DRONE_PROFILE.allowlist)
        self.assertFalse(DRONE_PROFILE.require_orientation)
        self.assertFalse(DRONE_PROFILE.tilt_guard)
        self.assertTrue(GO2_PROFILE.require_orientation)
        self.assertTrue(GO2_PROFILE.tilt_guard)
        self.assertEqual(
            (
                DRONE_PROFILE.max_horizontal,
                DRONE_PROFILE.max_vz,
                DRONE_PROFILE.max_wz,
                DRONE_PROFILE.takeoff_alt_m,
                DRONE_PROFILE.fence_radius_m,
                DRONE_PROFILE.fence_max_alt_m,
            ),
            (1.5, 1.0, 0.8, 3.0, 10.0, 3.0),
        )
        spec = RobotSpec(
            id="aerial-test",
            name="Aerial",
            category="aerial",
            capabilities=("aerial",),
        )
        self.assertIs(profile_for(spec.id, spec), DRONE_PROFILE)
        ground = RobotSpec(
            id="diff", name="Diff", category="wheeled", capabilities=("base.drive",)
        )
        self.assertFalse(profile_for(ground.id, ground).aerial)
        self.assertEqual(profile_for("unitree-go2").max_vx, 0.5)

    def test_ground_profile_cannot_grow_drone_actions_or_drop_tilt(self):
        with self.assertRaises(ValueError):
            SafetyProfile(allowlist=frozenset({"land"}))
        with self.assertRaises(ValueError):
            SafetyProfile(tilt_guard=False)
        with self.assertRaises(ValueError):
            SafetyProfile(aerial=1)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            SafetyProfile(
                aerial=True,
                tilt_guard=False,
                allowlist=frozenset({"takeoff"}),
                fence_max_alt_m=3.0,
                takeoff_alt_m=3.0,
            )
        with self.assertRaises(ValueError):
            SafetyProfile(
                aerial=True,
                tilt_guard=False,
                allowlist=DRONE_PROFILE.allowlist,
                fence_radius_m=10.0,
                fence_max_alt_m=2.0,
                takeoff_alt_m=3.0,
            )

    def test_config_cannot_change_family(self):
        path = Path(tempfile.mkdtemp()) / "safety.json"
        self.addCleanup(shutil.rmtree, path.parent, True)
        cases = [
            {"profiles": {"unitree-go2": {"aerial": True, "tilt_guard": False}}},
            {"profiles": {"px4-sitl": {"aerial": False}}},
            {"profiles": {"omnibot": {"tilt_guard": False}}},
            {"profiles": {"px4-sitl": {"allowlist": ["flip", "land"]}}},
            {"profiles": {"px4-sitl": {"fence_max_alt_m": 1.0}}},
        ]
        for body in cases:
            with self.subTest(body=body):
                path.write_text(json.dumps(body), encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_safety_config(path)
        path.write_text(
            json.dumps(
                {
                    "profiles": {
                        "px4-sitl": {
                            "fence_radius_m": 5.0,
                            "takeoff_alt_m": 2.0,
                            "fence_max_alt_m": 2.0,
                        }
                    }
                }
            ),
            encoding="utf-8",
        )
        cfg = load_safety_config(path)
        tuned = profile_for("px4-sitl", config=cfg)
        self.assertEqual(tuned.fence_radius_m, 5.0)
        self.assertEqual(tuned.takeoff_alt_m, 2.0)
        self.assertTrue(tuned.aerial)
        self.assertIn("land", tuned.allowlist)
        path.write_text(
            json.dumps(
                {
                    "profiles": {
                        "not-a-known-robot": {
                            "aerial": True,
                            "tilt_guard": False,
                            "allowlist": [
                                "hold",
                                "land",
                                "rtl",
                                "offboard_velocity",
                                "offboard_position",
                            ],
                            "fence_radius_m": 10,
                            "fence_max_alt_m": 3,
                            "takeoff_alt_m": 3,
                            "max_vz": 1,
                            "max_horizontal": 1.5,
                        }
                    }
                }
            ),
            encoding="utf-8",
        )
        with self.assertRaises(ValueError) as raised:
            load_safety_config(path)
        self.assertIn("cannot change aerial", str(raised.exception))

    def test_mcp_denylist(self):
        for name in ("hold", "Land", "offboard_velocity", "offboard_position"):
            self.assertTrue(drone_mcp_allowed(name), name)
        for name in (
            "takeoff",
            "arm_flight",
            "drone_arm",
            "rtl",
            "drone_rtl",
            "release_stop",
            "disarm",
            "flip",
            "orbit",
            "",
        ):
            self.assertFalse(drone_mcp_allowed(name), name)
        self.assertFalse(drone_mcp_allowed(None))

    def test_clip_and_fence_helpers(self):
        self.assertEqual(clip_horizontal(0.0, 0.0, 1.5), (0.0, 0.0))
        self.assertEqual(clip_horizontal(1.0, 0.0, 1.5), (1.0, 0.0))
        vx, vy = clip_horizontal(3.0, 4.0, 1.5)
        self.assertAlmostEqual(math.hypot(vx, vy), 1.5)
        self.assertIsNone(
            fence_violation(6.0, 8.0, 3.0, Frame.LOCAL_ENU, DRONE_PROFILE)
        )
        self.assertEqual(
            fence_violation(11.0, 0.0, 0.0, Frame.LOCAL_ENU, DRONE_PROFILE),
            "radius",
        )
        self.assertEqual(
            fence_violation(0.0, 0.0, -3.5, Frame.LOCAL_NED, DRONE_PROFILE),
            "altitude",
        )
        self.assertEqual(
            fence_violation(0.0, 0.0, 0.4, Frame.LOCAL_NED, DRONE_PROFILE),
            "below_origin",
        )
        self.assertEqual(
            fence_violation(0.0, 0.0, 0.0, Frame.BODY_FRD, DRONE_PROFILE),
            "body_frame",
        )

        class _Odd(str):
            pass

        self.assertEqual(
            fence_violation(0.0, 0.0, 0.0, _Odd("NOPE"), DRONE_PROFILE),  # type: ignore[arg-type]
            "frame",
        )


class TestDroneCommands(DroneCase):
    def test_disarmed_flight_is_not_forwarded(self):
        gate, link = self.make(
            arm=False, tele=drone_telemetry(landed="on_ground", z=-0.1)
        )
        self.assertFalse(gate.command("land"))
        self.assertFalse(gate.command("takeoff", 2.0))
        self.assertFalse(gate.command("disarm"))
        self.assertFalse(
            gate.command("offboard_position", 1.0, 0.0, -1.0, 0.0, Frame.LOCAL_NED)
        )
        self.assertFalse(gate.command("arm", 1))
        self.assertEqual(link.flight, [])
        self.assertEqual(gate.last_error, "bad_arguments")

    def test_transport_error_on_land_latches(self):
        gate, link = self.make(tele=drone_telemetry(landed="in_air", z=-2.0))
        link.fail.add("land")
        self.assertFalse(gate.command("land"))
        self.assertTrue(gate.latched)
        self.assertEqual(link.flight, [])
        self.assertNotIn("disarm", link.actions())

    def test_flight_audit_failure_drops_the_command(self):
        class FlightAudit(AuditLog):
            def write(self, event: str, **fields: object) -> bool:
                if event == "flight":
                    return False
                return super().write(event, **fields)

        gate, link = self.make(audit=FlightAudit(self.tmp))
        self.assertFalse(
            gate.command("offboard_velocity", 0.2, 0.0, 0.0, 0.0, Frame.BODY_FRD)
        )
        self.assertEqual(link.flight, [])
        self.assertFalse(gate.armed)

    def test_non_finite_pose_is_not_treated_as_a_breach(self):
        gate, link = self.make(
            tele=drone_telemetry(z=float("nan"), frame="LOCAL_NED")  # type: ignore[arg-type]
        )
        self.assertTrue(
            gate.command("offboard_velocity", 0.2, 0.0, 0.0, 0.0, Frame.BODY_FRD)
        )
        self.assertEqual(link.actions(), ["offboard_velocity"])
        gate, link = self.make(
            tele=drone_telemetry(landed="on_ground", z="high", frame="LOCAL_NED")  # type: ignore[arg-type]
        )
        self.assertFalse(gate.command("disarm"))
        self.assertEqual(gate.last_error, "not_on_ground")
        self.assertEqual(link.flight, [])

    def test_accepts_land_and_rejects_flip_and_unknown(self):
        gate, link = self.make()
        self.assertTrue(gate.command("Land"))
        self.assertTrue(gate.command("hold"))
        self.assertTrue(gate.command("rtl"))
        self.assertEqual(link.actions(), ["land", "hold", "rtl"])
        for name, fields in link.flight:
            self.assertEqual(fields, {})
            self.assertNotIn(21196, fields.values())
        self.assertFalse(gate.command("flip"))
        self.assertFalse(gate.command("front_flip"))
        self.assertFalse(gate.command("orbit"))
        self.assertFalse(gate.command("land", 1))
        self.assertEqual(link.actions(), ["land", "hold", "rtl"])
        reasons = {rec["reason"] for rec in self.events("reject")}
        self.assertIn("denied", reasons)
        self.assertIn("not_allowlisted", reasons)
        self.assertIn("bad_arguments", reasons)
        flight = self.events("flight")
        self.assertEqual(flight[0]["flight_request"], "land")
        self.assertIsNone(flight[0]["frame"])
        self.assertEqual(flight[0]["backend"], "px4-dds")
        self.assertNotIn("21196", self.audit_text())
        self.assertNotIn("force_disarm", self.audit_text())

    def test_go2_still_refuses_land_and_stops_via_estop(self):
        link = DroneLink(backend_id="go2", protocol="simulated")
        gate, link = self.make(
            link,
            profile=GO2_PROFILE,
            spec=get_spec("unitree-go2"),
            tele=Telemetry(
                custom={"roll": 0.0, "pitch": 0.0, "error_code": 0, "body_height": 0.3}
            ),
        )
        self.assertFalse(gate.command("land"))
        self.assertEqual(link.flight, [])
        self.assertEqual(self.events("reject")[-1]["reason"], "not_allowlisted")
        gate.emergency_stop()
        self.assertEqual(link.events, ["estop"])
        self.assertEqual(link.flight, [])
        self.assertTrue(gate.latched)
        self.assertNotIn("flight_request", self.events("estop")[-1])

    def test_three_mps_is_clamped_and_reported(self):
        gate, link = self.make()
        self.assertTrue(
            gate.command("offboard_velocity", 3.0, 0.0, 0.0, 0.0, Frame.BODY_FRD)
        )
        action, fields = link.flight[-1]
        self.assertEqual(action, "offboard_velocity")
        self.assertAlmostEqual(fields["vx"], 1.5)
        self.assertAlmostEqual(fields["vy"], 0.0)
        self.assertLessEqual(math.hypot(fields["vx"], fields["vy"]), 1.5)
        self.assertIs(fields["frame"], Frame.BODY_FRD)
        clamp = self.events("clamp")[-1]
        self.assertEqual(clamp["requested"], [3.0, 0.0, 0.0, 0.0])
        self.assertEqual(clamp["applied"], [1.5, 0.0, 0.0, 0.0])
        self.assertIn("velocity_cap", clamp["reasons"])
        self.assertEqual(clamp["flight_request"], "offboard_velocity")
        self.assertEqual(clamp["frame"], "BODY_FRD")
        self.assertEqual(clamp["backend"], "px4-dds")

    def test_vertical_yaw_and_diagonal_caps(self):
        gate, link = self.make()
        self.assertTrue(
            gate.command("offboard_velocity", 3.0, 4.0, -5.0, 2.0, Frame.LOCAL_NED)
        )
        _action, fields = link.flight[-1]
        self.assertAlmostEqual(math.hypot(fields["vx"], fields["vy"]), 1.5)
        self.assertAlmostEqual(fields["vz"], -1.0)
        self.assertAlmostEqual(fields["yaw_rate"], 0.8)
        self.assertLessEqual(abs(fields["vz"]), 1.0)
        self.assertLessEqual(abs(fields["yaw_rate"]), 0.8)

    def test_velocity_inside_caps_is_not_clamped(self):
        gate, link = self.make()
        self.assertTrue(
            gate.command("offboard_velocity", 0.4, -0.2, 0.3, 0.1, Frame.LOCAL_ENU)
        )
        self.assertEqual(self.events("clamp"), [])
        _action, fields = link.flight[-1]
        self.assertEqual(
            (fields["vx"], fields["vy"], fields["vz"], fields["yaw_rate"]),
            (0.4, -0.2, 0.3, 0.1),
        )

    def test_planar_velocity_magnitude_is_clipped(self):
        gate, link = self.make(profile=fast())
        self.clock.advance(0.1)
        gate.send_velocity(Velocity(1.5, 1.5, 0.0))
        vx, vy, _wz = link.velocities[-1]
        self.assertAlmostEqual(math.hypot(vx, vy), 1.5)
        self.assertLess(abs(vx), 1.5)

    def test_fence_breach_returns_error_and_hold(self):
        gate, link = self.make()
        with self.assertLogs("ohho.safety", level="ERROR") as logs:
            self.assertFalse(
                gate.command("offboard_position", 11.0, 0.0, 1.0, 0.2, Frame.LOCAL_ENU)
            )
        self.assertEqual(gate.last_error, "geofence")
        self.assertTrue(any("geofence" in line for line in logs.output))
        self.assertEqual(link.flight, [("hold", {})])
        rec = self.events("reject")[-1]
        self.assertEqual(rec["reason"], "geofence")
        self.assertEqual(rec["violation"], "radius")
        self.assertEqual(rec["flight_request"], "offboard_position")
        self.assertEqual(rec["frame"], "LOCAL_ENU")
        self.assertEqual(rec["backend"], "px4-dds")
        self.assertNotIn("offboard_position", link.actions())

    def test_fence_altitude_floor_body_and_boundary(self):
        gate, link = self.make()
        self.assertFalse(
            gate.command("offboard_position", 0.0, 0.0, -3.5, 0.0, Frame.LOCAL_NED)
        )
        self.assertEqual(self.events("reject")[-1]["violation"], "altitude")
        self.assertFalse(
            gate.command("offboard_position", 0.0, 0.0, 0.5, 0.0, Frame.LOCAL_NED)
        )
        self.assertEqual(self.events("reject")[-1]["violation"], "below_origin")
        self.assertFalse(
            gate.command("offboard_position", 1.0, 0.0, 0.0, 0.0, Frame.BODY_FRD)
        )
        self.assertEqual(self.events("reject")[-1]["violation"], "body_frame")
        self.assertEqual(link.actions(), ["hold", "hold", "hold"])
        self.assertTrue(
            gate.command("offboard_position", 6.0, 8.0, 3.0, 0.4, Frame.LOCAL_ENU)
        )
        self.assertTrue(
            gate.command("offboard_position", 0.0, 0.0, -3.0, 0.0, Frame.LOCAL_NED)
        )
        kinds = link.actions()
        self.assertEqual(kinds[-2:], ["offboard_position", "offboard_position"])
        self.assertIs(link.flight[-1][1]["frame"], Frame.LOCAL_NED)
        self.assertEqual(link.flight[-1][1]["z"], -3.0)

    def test_velocity_while_pose_is_outside_holds(self):
        gate, link = self.make(
            tele=drone_telemetry(frame="LOCAL_ENU", x=20.0, y=0.0, z=1.0)
        )
        self.assertFalse(
            gate.command("offboard_velocity", 0.2, 0.0, 0.0, 0.0, Frame.BODY_FRD)
        )
        self.assertEqual(gate.last_error, "geofence")
        self.assertEqual(link.actions(), ["hold"])

    def test_velocity_without_a_pose_is_clipped_not_fenced(self):
        gate, link = self.make(tele=drone_telemetry(include_pose=False))
        self.assertTrue(
            gate.command("offboard_velocity", 3.0, 0.0, 0.0, 0.0, Frame.BODY_FRD)
        )
        self.assertAlmostEqual(link.flight[-1][1]["vx"], 1.5)
        self.assertNotIn("hold", link.actions())

    def test_unreadable_pose_does_not_invent_a_fence_breach(self):
        gate, link = self.make(
            tele=drone_telemetry(include_pose=True, pose_frame="NOPE", z="nope")  # type: ignore[arg-type]
        )
        self.assertTrue(
            gate.command("offboard_velocity", 0.2, 0.0, 0.0, 0.0, Frame.BODY_FRD)
        )
        self.assertEqual(link.actions(), ["offboard_velocity"])

    def test_bad_velocity_arguments(self):
        gate, link = self.make()
        self.assertFalse(gate.command("offboard_velocity", 1.0, 0.0, 0.0, 0.0))
        self.assertFalse(
            gate.command(
                "offboard_velocity", float("nan"), 0.0, 0.0, 0.0, Frame.BODY_FRD
            )
        )
        self.assertFalse(
            gate.command("offboard_velocity", 0.0, 0.0, 0.0, 0.0, "BODY_FRD")
        )
        self.assertEqual(link.flight, [])
        self.assertEqual(gate.last_error, "bad_arguments")

    def test_takeoff_ceiling_is_a_fence_error(self):
        gate, link = self.make()
        self.assertTrue(gate.command("takeoff", 3.0))
        self.assertEqual(link.flight, [("takeoff", {"alt_m": 3.0})])
        link.flight.clear()
        self.assertFalse(gate.command("takeoff", 3.01))
        self.assertEqual(gate.last_error, "geofence")
        self.assertEqual(link.actions(), ["hold"])
        self.assertFalse(gate.command("takeoff", 0))
        self.assertFalse(gate.command("takeoff", -1.0))
        self.assertFalse(gate.command("takeoff", float("nan")))
        self.assertEqual(gate.last_error, "bad_arguments")

    def test_takeoff_refuses_low_battery_and_bad_link(self):
        gate, link = self.make(tele=drone_telemetry(battery=0.29))
        self.assertFalse(gate.command("takeoff", 2.0))
        self.assertEqual(gate.last_error, "battery")
        gate, link = self.make(tele=drone_telemetry(battery=None, include_pose=True))
        self.assertFalse(gate.command("takeoff", 2.0))
        self.assertEqual(gate.last_error, "battery")
        gate, link = self.make(tele=drone_telemetry(battery=0.9, drone_battery=0.1))
        self.assertFalse(gate.command("takeoff", 2.0))
        self.assertEqual(gate.last_error, "battery")
        gate, link = self.make(tele=drone_telemetry(link_ok=False, battery=0.9))
        self.assertFalse(gate.command("takeoff", 2.0))
        self.assertEqual(gate.last_error, "link")
        gate, link = self.make(tele=drone_telemetry(link_ok=None, battery=0.9))
        self.assertFalse(gate.command("takeoff", 2.0))
        self.assertEqual(gate.last_error, "link")
        self.assertEqual(link.flight, [])

    def test_takeoff_at_battery_boundary_with_link(self):
        gate, link = self.make(tele=drone_telemetry(battery=0.3))
        self.assertTrue(gate.command("takeoff", 2.5))
        self.assertEqual(link.flight, [("takeoff", {"alt_m": 2.5})])
        rec = self.events("flight")[-1]
        self.assertEqual(rec["flight_request"], "takeoff")
        self.assertEqual(rec["backend"], "px4-dds")
        self.assertIsNone(rec["frame"])

    def test_takeoff_and_arm_need_a_human_tty(self):
        self.tty = False
        gate, link = self.make()
        self.assertFalse(gate.command("takeoff", 2.0))
        self.assertEqual(gate.last_error, "no_tty")
        self.assertFalse(gate.command("arm"))
        self.assertEqual(gate.last_error, "no_tty")
        self.assertEqual(link.flight, [])

        self.tty = True
        self.env = {"CI": "true"}
        gate, link = self.make()
        self.assertFalse(gate.command("takeoff", 2.0))
        self.assertEqual(gate.last_error, "ci")
        self.assertEqual(link.flight, [])

    def test_arm_flight_needs_a_live_setpoint_stream(self):
        gate, link = self.make()
        self.assertFalse(gate.command("arm"))
        self.assertEqual(gate.last_error, "no_setpoint_stream")
        self.assertTrue(
            gate.command("offboard_velocity", 0.2, 0.0, 0.0, 0.0, Frame.BODY_FRD)
        )
        self.assertTrue(gate.command("arm_flight"))
        self.assertEqual(link.actions()[-1], "arm")
        self.assertEqual(link.flight[-1][1], {})
        self.clock.advance(0.31)
        link.emit(drone_telemetry())
        self.assertFalse(gate.command("arm"))
        self.assertEqual(gate.last_error, "no_setpoint_stream")
        self.assertNotIn("21196", self.audit_text())

    def test_hardware_takeoff_needs_the_phrase(self):
        self.env = {ARM_ENV: "1"}
        link = DroneLink(backend_id="px4-dds", protocol="serial")
        gate, link = self.make(link)
        self.assertTrue(self.prompts)
        before = len(link.flight)
        self.answer = "nope"
        self.assertFalse(gate.command("takeoff", 2.0))
        self.assertEqual(gate.last_error, "not_confirmed")
        self.assertEqual(len(link.flight), before)
        self.answer = ARM_PHRASE
        self.assertTrue(gate.command("takeoff", 2.0))
        self.assertEqual(link.actions()[-1], "takeoff")
        self.env.pop(ARM_ENV)
        self.assertFalse(gate.command("takeoff", 2.0))
        self.assertEqual(gate.last_error, "env")

    def test_disarm_only_on_the_ground(self):
        gate, link = self.make(tele=drone_telemetry(landed="in_air", z=-2.5))
        self.assertFalse(gate.command("disarm"))
        self.assertEqual(gate.last_error, "not_on_ground")
        self.assertFalse(gate.command("disarm", 21196))
        self.assertEqual(gate.last_error, "bad_arguments")
        self.assertEqual(link.flight, [])

        gate, link = self.make(tele=drone_telemetry(landed="on_ground", z=-1.0))
        self.assertFalse(gate.command("disarm"))
        self.assertEqual(link.flight, [])

        gate, link = self.make(tele=drone_telemetry(landed="on_ground", z=-0.25))
        self.assertTrue(gate.command("disarm"))
        self.assertEqual(link.flight, [("disarm", {})])
        self.assertNotIn(21196, link.flight[0][1].values())

    def test_tilt_does_not_latch_and_fault_still_does(self):
        gate, link = self.make(tele=drone_telemetry(roll=1.2))
        self.assertFalse(gate.latched)
        self.assertTrue(gate.command("land"))
        self.assertEqual(link.actions(), ["land"])

        gate, link = self.make()
        link.emit(drone_telemetry(error_code=2, landed="in_air", z=-2.0))
        self.assertTrue(gate.latched)
        self.assertEqual(gate.latch_reason, "fault")
        self.assertNotIn("disarm", link.actions())
        self.assertIn("land", link.actions())

    def test_hard_ceiling_lowers_drone_speed(self):
        gate, _link = self.make(
            profile=DRONE_PROFILE,
            arm=False,
            config=SafetyConfig(max_lin_ceiling=0.4, max_ang_ceiling=0.5),
        )
        self.assertEqual(gate.profile.max_horizontal, 0.4)
        self.assertEqual(gate.profile.max_vz, 0.4)
        self.assertEqual(gate.profile.max_vx, 0.4)
        self.assertEqual(gate.profile.max_wz, 0.5)

    def test_failed_audit_does_not_forward(self):
        class Selective(AuditLog):
            def write(self, event: str, **fields: object) -> bool:
                if event in {"clamp", "flight"}:
                    return False
                return super().write(event, **fields)

        gate, link = self.make(audit=Selective(self.tmp))
        self.assertTrue(gate.armed)
        self.assertFalse(
            gate.command("offboard_velocity", 3.0, 0.0, 0.0, 0.0, Frame.BODY_FRD)
        )
        self.assertNotIn("offboard_velocity", link.actions())
        self.assertFalse(gate.armed)


class TestDroneEstop(DroneCase):
    def test_in_air_estop_lands_and_never_disarms(self):
        gate, link = self.make(tele=drone_telemetry(landed="in_air", z=-2.5))
        gate.emergency_stop()
        self.assertTrue(gate.latched)
        self.assertFalse(gate.armed)
        self.assertEqual(link.actions(), ["land"])
        self.assertNotIn("disarm", link.actions())
        self.assertNotIn("estop", link.events)
        self.assertNotIn("motor_cut", link.events)
        rec = self.events("estop")[-1]
        self.assertEqual(rec["flight_request"], ["land"])
        self.assertEqual(rec["backend"], "px4-dds")
        self.assertEqual(rec["estop_layer"], "ohho_land")
        self.assertIsNone(rec["frame"])
        self.assertFalse(rec["inner_error"])
        self.assertNotIn("21196", self.audit_text())
        self.assertNotIn("force_disarm", self.audit_text())
        self.assertNotIn("flight_termination", self.audit_text())

        gate, link = self.make(tele=drone_telemetry(landed="in_air", z=0.0))
        gate.emergency_stop()
        self.assertEqual(link.actions(), ["land"])
        self.assertNotIn("disarm", link.actions())

    def test_unknown_holds_then_lands(self):
        gate, link = self.make(tele=drone_telemetry(landed="unknown", z=-5.0))
        gate.emergency_stop()
        self.assertEqual(link.actions(), ["hold", "land"])
        self.assertEqual(self.events("estop")[-1]["flight_request"], ["hold", "land"])
        self.assertNotIn("disarm", link.actions())

        gate, link = self.make(tele=drone_telemetry(landed="made-up", z=-5.0))
        gate.emergency_stop()
        self.assertEqual(link.actions(), ["hold", "land"])

    def test_ground_disarms_only_under_altitude_gate(self):
        gate, link = self.make(tele=drone_telemetry(landed="on_ground", z=-0.1))
        gate.emergency_stop()
        self.assertEqual(link.actions(), ["disarm"])
        self.assertEqual(link.flight[0][1], {})
        self.assertNotIn("21196", self.audit_text())

        gate, link = self.make(tele=drone_telemetry(landed="on_ground", z=-1.0))
        gate.emergency_stop()
        self.assertEqual(link.actions(), ["land"])
        self.assertNotIn("disarm", link.actions())

    def test_crazyflie_cuts_motors_and_records_the_backend(self):
        link = DroneLink(backend_id="Crazyflie")
        gate, link = self.make(link, tele=drone_telemetry(landed="in_air", z=-1.0))
        gate.emergency_stop()
        self.assertEqual(link.events, ["motor_cut"])
        self.assertEqual(link.flight, [])
        rec = self.events("estop")[-1]
        self.assertEqual(rec["backend"], "crazyflie")
        self.assertEqual(rec["flight_request"], "motor_cut")
        self.assertEqual(rec["estop_layer"], "crazyflie_motor_cut")
        self.assertNotIn("disarm", link.actions())
        self.assertNotIn("21196", self.audit_text())

        bare = DroneLink(backend_id="crazyflie-cf21")
        bare.motor_cut = None  # type: ignore[method-assign]
        gate, bare = self.make(bare, tele=drone_telemetry(landed="in_air", z=-1.0))
        gate.emergency_stop()
        self.assertEqual(bare.events, ["estop"])
        self.assertEqual(bare.flight, [])
        self.assertEqual(self.events("estop")[-1]["backend"], "crazyflie-cf21")

    def test_sim_descends_and_does_not_disarm_in_air(self):
        link = DroneLink(backend_id="drone-sim")
        gate, link = self.make(link, tele=drone_telemetry(landed="in_air", z=-2.0))
        gate.emergency_stop()
        self.assertEqual(link.actions(), ["land"])
        self.assertIn("descent", link.events)
        self.assertNotIn("disarm", link.actions())
        self.assertNotIn("motor_cut", link.events)
        self.assertEqual(self.events("estop")[-1]["estop_layer"], "simulated_descent")

        plain = DroneLink(protocol="simulated")
        del plain.backend_id
        gate, plain = self.make(plain, tele=drone_telemetry(landed="in_air", z=-2.0))
        plain.simulated_descent = None  # type: ignore[method-assign]
        gate.emergency_stop()
        self.assertEqual(plain.actions(), ["land"])
        self.assertNotIn("descent", plain.events)
        self.assertEqual(self.events("estop")[-1]["backend"], "drone-sim")
        self.assertNotIn("disarm", plain.actions())

    def test_forbidden_estop_command_is_not_sent(self):
        gate, link = self.make(tele=drone_telemetry(landed="in_air", z=-2.0))
        bad = EstopCommand(
            name="force_disarm", mav_command=400, param1=0.0, param2=21196.0
        )
        with mock.patch("ohho.safety.estop_commands", return_value=(bad,)):
            gate.emergency_stop()
        self.assertEqual(link.flight, [])
        self.assertTrue(self.events("estop")[-1]["inner_error"])
        self.assertNotIn("21196", self.audit_text())
        self.assertTrue(gate.latched)

    def test_missing_send_flight_latches_without_a_disarm(self):
        link = DroneLink(backend_id="mavlink", protocol="serial")
        link.send_flight = None  # type: ignore[method-assign]
        self.env = {ARM_ENV: "1"}
        gate, link = self.make(link, tele=drone_telemetry(landed="in_air", z=-2.0))
        gate.emergency_stop()
        self.assertTrue(gate.latched)
        self.assertTrue(self.events("estop")[-1]["inner_error"])
        self.assertNotIn("disarm", link.actions())
        self.assertEqual(self.events("estop")[-1]["backend"], "mavlink")

    def test_release_clears_the_latch_only(self):
        gate, link = self.make(tele=drone_telemetry(landed="in_air", z=-2.0))
        gate.emergency_stop()
        self.assertTrue(gate.release_stop())
        self.assertFalse(gate.latched)
        self.assertFalse(gate.armed)
        self.assertNotIn("release", link.events)
        self.assertNotIn("arm", link.actions())
        self.assertNotIn("offboard_velocity", link.actions())
        self.assertNotIn("offboard_position", link.actions())
        rec = self.events("estop_release")[-1]
        self.assertFalse(rec["resumed_offboard"])
        self.assertFalse(rec["autopilot_armed"])
        self.assertEqual(rec["backend"], "px4-dds")
        self.assertIsNone(rec["flight_request"])
        self.assertFalse(
            gate.command("offboard_velocity", 0.2, 0.0, 0.0, 0.0, Frame.BODY_FRD)
        )

    def test_hardware_release_still_needs_the_phrase(self):
        self.env = {ARM_ENV: "1"}
        link = DroneLink(backend_id="px4-dds", protocol="serial")
        gate, link = self.make(link, tele=drone_telemetry(landed="in_air", z=-1.0))
        gate.emergency_stop()
        self.tty = False
        self.assertFalse(gate.release_stop())
        self.assertTrue(gate.latched)
        self.assertEqual(self.events("release_refused")[-1]["reason"], "no_tty")
        self.tty = True
        self.answer = "no"
        self.assertFalse(gate.release_stop())
        self.assertTrue(gate.latched)
        self.answer = RELEASE_PHRASE
        self.assertTrue(gate.release_stop())
        self.assertFalse(gate.latched)
        self.assertFalse(gate.armed)
        self.assertNotIn("release", link.events)
        self.assertNotIn("arm", link.actions())

    def test_release_refuses_a_fault_and_ignores_roll(self):
        gate, _link = self.make(tele=drone_telemetry(roll=1.1, landed="in_air", z=-1.0))
        gate.emergency_stop()
        self.assertTrue(gate.release_stop())
        self.assertFalse(gate.latched)

        gate, _link = self.make(
            tele=drone_telemetry(error_code=1, landed="in_air", z=-1.0), arm=False
        )
        self.assertTrue(gate.latched)
        self.tty = True
        self.assertFalse(gate.release_stop())
        self.assertEqual(self.events("release_refused")[-1]["reason"], "unsafe_state")

    def test_hold_failure_on_a_fence_breach_latches(self):
        gate, link = self.make(tele=drone_telemetry(landed="in_air", z=-1.0))
        link.fail.add("hold")
        self.assertFalse(
            gate.command("offboard_position", 12.0, 0.0, 0.0, 0.0, Frame.LOCAL_ENU)
        )
        self.assertTrue(gate.latched)
        self.assertNotIn("offboard_position", link.actions())
        self.assertIn("land", link.actions())

    def test_state_watchdog_still_latches_in_the_air(self):
        gate, link = self.make(tele=drone_telemetry(landed="in_air", z=-2.0))
        self.clock.advance(0.5)
        gate.tick()
        self.assertTrue(gate.latched)
        self.assertEqual(gate.latch_reason, "watchdog_state")
        self.assertEqual(link.actions(), ["land"])
        self.assertNotIn("disarm", link.actions())

    def test_command_watchdog_still_zeroes_velocity(self):
        gate, link = self.make(profile=fast())
        self.clock.advance(0.05)
        gate.send_velocity(Velocity(0.2, 0.0, 0.0))
        self.assertNotEqual(link.velocities[-1], (0.0, 0.0, 0.0))
        self.clock.advance(0.31)
        link.emit(drone_telemetry())
        gate.tick()
        self.assertFalse(gate.latched)
        self.assertEqual(link.velocities[-1], (0.0, 0.0, 0.0))

    def test_backend_without_an_id_uses_the_protocol(self):
        link = DroneLink(backend_id="ignored", protocol="ros2")
        del link.backend_id
        self.env = {ARM_ENV: "1"}
        gate, link = self.make(link, tele=drone_telemetry(landed="takeoff", z=-1.0))
        gate.emergency_stop()
        self.assertEqual(link.actions(), ["land"])
        self.assertEqual(self.events("estop")[-1]["backend"], "ros2")
        self.assertEqual(self.events("estop")[-1]["estop_layer"], "ohho_land")


if __name__ == "__main__":
    unittest.main()
