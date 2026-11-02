"""Kinematic drone-sim: square flight, caps, frames, e-stop, routing."""

from __future__ import annotations

import math
import unittest

from ohho import capabilities as caps
from ohho.adapters import AdapterUnavailable, resolve_transport
from ohho.adapters.drone_sim import (
    OHHO_MAX_CLIMB_MPS,
    OHHO_MAX_HORIZONTAL_MPS,
    OHHO_MAX_YAW_RATE_RAD_S,
    DroneSimTransport,
)
from ohho.adapters.sim import SimTransport
from ohho.aerial import Frame
from ohho.registry import get_spec, list_specs, spec_from_dict
from ohho.robot import Robot
from ohho.runtime import NativeRuntime
from ohho.schema import Telemetry, Velocity

DT = 0.05

_DRONE_KEYS = (
    "armed",
    "flight_mode",
    "landed_state",
    "pose",
    "pose_frame",
    "velocity",
    "battery",
    "gps_fix",
    "failsafe",
    "link_ok",
    "experimental",
    "flight_ready",
)


def _drone(snap: Telemetry) -> dict:
    raw = snap.custom["drone"]
    if not isinstance(raw, dict):
        raise AssertionError("Telemetry.custom['drone'] is missing")
    return raw


def _alt(snap: Telemetry) -> float:
    return -float(_drone(snap)["pose"]["z"])


def _pair(robot_id: str = "px4-sitl") -> tuple[Robot, DroneSimTransport]:
    spec = get_spec(robot_id)
    transport = resolve_transport("sim://", spec)
    if not isinstance(transport, DroneSimTransport):
        raise AssertionError(f"{robot_id} did not select drone-sim")
    transport.connect()
    return Robot(spec, transport, NativeRuntime()), transport


def _step(transport: DroneSimTransport, seconds: float, dt: float = DT) -> Telemetry:
    last = transport.read()
    steps = int(round(seconds / dt))
    for _ in range(steps):
        last = transport.step(dt)
    return last


def _until(
    transport: DroneSimTransport,
    pred,
    limit: float,
    dt: float = DT,
) -> Telemetry:
    elapsed = 0.0
    last = transport.read()
    while elapsed < limit:
        last = transport.step(dt)
        elapsed += dt
        if pred(last):
            return last
    raise AssertionError(f"timeout after {limit}s: {_drone(last)}")


def _takeoff(bot: Robot, transport: DroneSimTransport, alt: float = 2.5) -> None:
    bot.arm_flight()
    bot.takeoff(alt)

    def reached(snap: Telemetry) -> bool:
        drone = _drone(snap)
        return (
            drone["landed_state"] == "in_air"
            and drone["flight_mode"] == "hold"
            and bool(drone["armed"])
            and abs(_alt(snap) - alt) <= 0.05
        )

    _until(transport, reached, 20.0)


class TestDroneSimRegistry(unittest.TestCase):
    def test_aerial_specs_and_ohho_caps(self):
        ids = {spec.id for spec in list_specs()}
        self.assertTrue({"px4-sitl", "crazyflie-sim", "px4-x500"}.issubset(ids))
        for robot_id in ("px4-sitl", "crazyflie-sim", "px4-x500"):
            spec = get_spec(robot_id)
            self.assertEqual(spec.category, "aerial")
            self.assertIn(caps.AERIAL, spec.capabilities)
            self.assertNotIn(caps.MANIPULATION, spec.capabilities)
            self.assertEqual(spec.max_lin, OHHO_MAX_HORIZONTAL_MPS)
            self.assertEqual(spec.max_climb, OHHO_MAX_CLIMB_MPS)
            self.assertEqual(spec.max_ang, OHHO_MAX_YAW_RATE_RAD_S)
            self.assertTrue(spec.experimental)
            self.assertFalse(spec.flight_ready)
        self.assertEqual(get_spec("px4-sitl").adapter, "sim")
        self.assertEqual(get_spec("px4-sitl").gps_fix, "3d")
        self.assertFalse(get_spec("px4-sitl").requires_hardware_arming)
        self.assertEqual(get_spec("crazyflie-sim").adapter, "sim")
        self.assertEqual(get_spec("crazyflie-sim").gps_fix, "none")
        self.assertTrue(get_spec("px4-x500").requires_hardware_arming)
        self.assertEqual(get_spec("px4-x500").adapter, "px4-dds")
        ground = get_spec("omnibot")
        self.assertFalse(ground.experimental)
        self.assertFalse(ground.requires_hardware_arming)
        self.assertEqual(ground.gps_fix, "")
        self.assertEqual(ground.max_climb, 0.0)
        self.assertFalse(ground.flight_ready)

    def test_manifest_reads_aerial_fields_and_never_flight_ready(self):
        spec = spec_from_dict(
            {
                "id": "manifest-drone",
                "name": "Manifest",
                "category": "aerial",
                "capabilities": ["aerial"],
                "limits": {"max_lin": 2.0, "max_ang": 0.8, "max_climb": 1.0},
                "gps_fix": "none",
                "requires_hardware_arming": True,
                "experimental": True,
                "flight_ready": True,
            }
        )
        self.assertEqual(spec.gps_fix, "none")
        self.assertTrue(spec.requires_hardware_arming)
        self.assertTrue(spec.experimental)
        self.assertFalse(spec.flight_ready)
        self.assertAlmostEqual(spec.max_climb, 1.0)

    def test_sim_scheme_selects_drone_sim_only_for_sitl_ids(self):
        for robot_id in ("px4-sitl", "crazyflie-sim"):
            for uri in ("sim://", "simulated://", None):
                transport = resolve_transport(uri, get_spec(robot_id))
                self.assertIs(type(transport), DroneSimTransport)
                self.assertEqual(transport.backend, "drone-sim")
                self.assertEqual(transport.protocol, "simulated")
        for robot_id in ("omnibot", "unitree-go2", "sim"):
            for uri in ("sim://", None):
                transport = resolve_transport(uri, get_spec(robot_id))
                self.assertIs(type(transport), SimTransport)

    def test_ground_sim_still_drives(self):
        transport = resolve_transport("sim://", get_spec("sim"))
        self.assertIs(type(transport), SimTransport)
        transport.connect()
        transport.send_velocity(Velocity(1.0, 0.0, 0.0))
        transport.step(1.0)
        odom = transport.read().odom
        self.assertIsNotNone(odom)
        assert odom is not None
        self.assertGreater(odom.x, 0.5)

    def test_px4_x500_spec_does_not_connect(self):
        spec = get_spec("px4-x500")
        for uri in (
            None,
            "sim://",
            "px4+dds://10.1.1.1:8888",
            "mavlink+serial:///dev/ttyACM0",
        ):
            with self.assertRaises(AdapterUnavailable):
                resolve_transport(uri, spec)


class TestDroneSimFlight(unittest.TestCase):
    def test_telemetry_keys_are_experimental(self):
        _bot, transport = _pair("px4-sitl")
        snap = transport.read()
        drone = _drone(snap)
        for key in _DRONE_KEYS:
            self.assertIn(key, drone)
        self.assertIs(drone["experimental"], True)
        self.assertIs(drone["flight_ready"], False)
        self.assertEqual(drone["pose_frame"], "LOCAL_NED")
        self.assertEqual(drone["velocity_frame"], "LOCAL_NED")
        self.assertEqual(drone["gps_fix"], "3d")
        self.assertEqual(drone["landed_state"], "on_ground")
        self.assertEqual(drone["flight_mode"], "disarmed")
        self.assertIs(drone["armed"], False)
        self.assertIs(drone["link_ok"], True)
        self.assertEqual(drone["link_quality"], 1.0)
        self.assertEqual(drone["failsafe"], [])
        self.assertEqual(
            set(drone["pose"]),
            {"x", "y", "z", "roll", "pitch", "yaw"},
        )
        self.assertEqual(set(drone["velocity"]), {"vx", "vy", "vz", "yaw_rate"})
        battery = drone["battery"]
        self.assertEqual(set(battery), {"fraction", "voltage_v", "current_a"})
        self.assertAlmostEqual(snap.battery, battery["fraction"])
        self.assertEqual(drone["voltage_v"], battery["voltage_v"])
        self.assertEqual(drone["current_a"], battery["current_a"])
        self.assertGreater(battery["voltage_v"], 12.0)
        self.assertLess(battery["voltage_v"], 17.0)
        self.assertIsNotNone(snap.odom)

    def test_step_is_deterministic_and_rejects_a_negative_dt(self):
        def sample() -> tuple[dict, float]:
            bot, transport = _pair()
            _takeoff(bot, transport, 1.0)
            bot.offboard_velocity(0.5, 0.0, 0.0, 0.0, Frame.LOCAL_NED)
            snap = _step(transport, 1.0)
            return _drone(snap)["pose"], snap.timestamp

        first = sample()
        second = sample()
        self.assertEqual(first, second)
        _bot, transport = _pair()
        with self.assertRaises(ValueError):
            transport.step(-0.01)
        transport.step(0.0)
        self.assertEqual(transport.read().timestamp, 0.0)

    def test_takeoff_requires_arm_and_disarm_is_refused_in_air(self):
        bot, transport = _pair()
        bot.takeoff(2.5)
        _step(transport, 2.0)
        self.assertFalse(_drone(transport.read())["armed"])
        self.assertEqual(_drone(transport.read())["landed_state"], "on_ground")
        self.assertAlmostEqual(_alt(transport.read()), 0.0, places=6)

        bot.arm_flight()
        self.assertTrue(_drone(transport.read())["armed"])
        bot.disarm_flight()
        self.assertFalse(_drone(transport.read())["armed"])

        _takeoff(bot, transport, 2.5)
        bot.disarm_flight()
        transport.send_flight("disarm", force=True)
        transport.send_flight("disarm", param2=21196)
        drone = _drone(transport.read())
        self.assertTrue(drone["armed"])
        self.assertEqual(drone["landed_state"], "in_air")
        self.assertGreater(_alt(transport.read()), 2.0)

    def test_square_at_half_meter_per_second_then_land(self):
        bot, transport = _pair("px4-sitl")
        _takeoff(bot, transport, 2.5)
        _step(transport, 1.0)
        hover = _drone(transport.read())["pose"]
        self.assertAlmostEqual(hover["x"], 0.0, delta=0.02)
        self.assertAlmostEqual(hover["y"], 0.0, delta=0.02)
        self.assertAlmostEqual(hover["z"], -2.5, delta=0.02)
        self.assertLessEqual(math.hypot(hover["x"], hover["y"]), 0.8)

        # 4 m sides, LOCAL_NED: north, east, south, west. Speed command 0.5 m/s.
        legs = (
            (0.5, 0.0, 4.0, 0.0),
            (0.0, 0.5, 4.0, 4.0),
            (-0.5, 0.0, 0.0, 4.0),
            (0.0, -0.5, 0.0, 0.0),
        )
        for index, (vx, vy, target_x, target_y) in enumerate(legs):
            bot.offboard_velocity(vx, vy, 0.0, 0.0, Frame.LOCAL_NED)
            elapsed = 0.0
            best = 1e9
            cruise: tuple[float, float, float] | None = None
            while elapsed < 16.0:
                snap = transport.step(DT)
                elapsed += DT
                drone = _drone(snap)
                self.assertTrue(drone["armed"])
                self.assertEqual(drone["landed_state"], "in_air")
                self.assertIs(drone["experimental"], True)
                pose = drone["pose"]
                self.assertAlmostEqual(pose["z"], -2.5, delta=0.05)
                self.assertAlmostEqual(pose["yaw"], 0.0, delta=0.02)
                vel = drone["velocity"]
                speed = math.hypot(float(vel["vx"]), float(vel["vy"]))
                self.assertLessEqual(speed, 0.6)
                self.assertAlmostEqual(float(vel["vz"]), 0.0, delta=0.05)
                dist = math.hypot(pose["x"] - target_x, pose["y"] - target_y)
                best = min(best, dist)
                odom = snap.odom
                self.assertIsNotNone(odom)
                assert odom is not None
                self.assertAlmostEqual(odom.x, pose["x"], places=6)
                self.assertAlmostEqual(odom.y, pose["y"], places=6)
                self.assertAlmostEqual(odom.theta, pose["yaw"], places=6)
                if index == 0 and cruise is None and elapsed >= 1.5:
                    cruise = (float(vel["vx"]), float(vel["vy"]), speed)
                if dist <= 0.25:
                    break
            else:
                self.fail(f"leg {index} missed {(target_x, target_y)}; best={best}")
            self.assertGreaterEqual(elapsed, 6.0)
            self.assertLessEqual(best, 0.8)
            if index == 0:
                self.assertIsNotNone(cruise)
                assert cruise is not None
                self.assertAlmostEqual(cruise[0], 0.5, delta=0.05)
                self.assertAlmostEqual(cruise[1], 0.0, delta=0.05)
                self.assertAlmostEqual(cruise[2], 0.5, delta=0.05)

        bot.land()
        before = transport.read()
        self.assertEqual(_drone(before)["landed_state"], "landing")
        self.assertTrue(_drone(before)["armed"])
        self.assertGreater(_alt(before), 2.0)
        armed_while_landing = False
        prev_alt = _alt(before)
        for _ in range(int(20.0 / DT)):
            snap = transport.step(DT)
            drone = _drone(snap)
            alt = _alt(snap)
            self.assertLessEqual(prev_alt - alt, OHHO_MAX_CLIMB_MPS * DT + 1e-4)
            self.assertGreaterEqual(alt, -1e-6)
            if drone["landed_state"] != "on_ground":
                self.assertTrue(drone["armed"])
                self.assertEqual(drone["landed_state"], "landing")
                self.assertGreater(alt, 0.0)
                armed_while_landing = True
            else:
                self.assertFalse(drone["armed"])
                self.assertAlmostEqual(alt, 0.0, places=5)
                self.assertEqual(drone["flight_mode"], "disarmed")
                break
            prev_alt = alt
        else:
            self.fail("land did not reach the ground")
        self.assertTrue(armed_while_landing)
        done = _drone(transport.read())
        self.assertEqual(done["landed_state"], "on_ground")
        self.assertFalse(done["armed"])
        battery = transport.read().battery
        self.assertIsNotNone(battery)
        assert battery is not None
        self.assertLess(battery, 0.999)

    def test_estop_descends_then_disarms_without_an_altitude_drop(self):
        bot, transport = _pair()
        _takeoff(bot, transport, 2.5)
        bot.offboard_velocity(0.5, -0.4, 0.0, 0.0, Frame.LOCAL_NED)
        _step(transport, 1.5)
        before = transport.read()
        pose = _drone(before)["pose"]
        x0 = float(pose["x"])
        y0 = float(pose["y"])
        alt0 = _alt(before)
        self.assertGreater(alt0, 2.0)
        moving = _drone(before)["velocity"]
        self.assertGreater(math.hypot(float(moving["vx"]), float(moving["vy"])), 0.2)

        bot.emergency_stop()
        held = transport.read()
        self.assertAlmostEqual(_alt(held), alt0, places=6)
        self.assertTrue(_drone(held)["armed"])
        self.assertEqual(_drone(held)["landed_state"], "landing")
        self.assertAlmostEqual(float(_drone(held)["pose"]["x"]), x0, places=6)
        self.assertAlmostEqual(float(_drone(held)["pose"]["y"]), y0, places=6)

        prev = alt0
        saw_airborne = False
        for _ in range(int(20.0 / DT)):
            snap = transport.step(DT)
            drone = _drone(snap)
            alt = _alt(snap)
            self.assertLessEqual(prev - alt, OHHO_MAX_CLIMB_MPS * DT + 1e-4)
            self.assertGreaterEqual(prev - alt, -1e-6)
            self.assertAlmostEqual(float(drone["pose"]["x"]), x0, delta=1e-6)
            self.assertAlmostEqual(float(drone["pose"]["y"]), y0, delta=1e-6)
            if drone["landed_state"] != "on_ground":
                self.assertTrue(drone["armed"])
                self.assertGreater(alt, 0.0)
                if prev > 2.0:
                    self.assertGreater(alt, 1.0)
                saw_airborne = True
            else:
                self.assertFalse(drone["armed"])
                self.assertAlmostEqual(alt, 0.0, places=5)
                self.assertEqual(drone["flight_mode"], "disarmed")
                break
            prev = alt
        else:
            self.fail("e-stop did not land")
        self.assertTrue(saw_airborne)
        bot.arm_flight()
        self.assertFalse(_drone(transport.read())["armed"])
        bot.release_stop()
        bot.arm_flight()
        released = _drone(transport.read())
        self.assertTrue(released["armed"])
        self.assertEqual(released["landed_state"], "on_ground")
        self.assertAlmostEqual(_alt(transport.read()), 0.0, places=5)

    def test_caps_clamp_with_first_order_response(self):
        bot, transport = _pair()
        _takeoff(bot, transport, 2.5)
        bot.offboard_velocity(10.0, 10.0, -10.0, 5.0, Frame.LOCAL_NED)
        first = transport.step(DT)
        vel = _drone(first)["velocity"]
        speed = math.hypot(float(vel["vx"]), float(vel["vy"]))
        self.assertGreater(speed, 0.0)
        self.assertLess(speed, 1.0)
        self.assertGreater(float(vel["vz"]), -0.5)
        self.assertLess(abs(float(vel["yaw_rate"])), 0.4)

        snap = _step(transport, 3.0)
        vel = _drone(snap)["velocity"]
        speed = math.hypot(float(vel["vx"]), float(vel["vy"]))
        self.assertLessEqual(speed, OHHO_MAX_HORIZONTAL_MPS + 1e-6)
        self.assertGreater(speed, OHHO_MAX_HORIZONTAL_MPS - 0.05)
        self.assertAlmostEqual(float(vel["vx"]), float(vel["vy"]), delta=0.05)
        self.assertGreaterEqual(float(vel["vz"]), -OHHO_MAX_CLIMB_MPS - 1e-6)
        self.assertAlmostEqual(float(vel["vz"]), -OHHO_MAX_CLIMB_MPS, delta=0.05)
        self.assertLessEqual(
            abs(float(vel["yaw_rate"])), OHHO_MAX_YAW_RATE_RAD_S + 1e-6
        )
        self.assertAlmostEqual(
            float(vel["yaw_rate"]), OHHO_MAX_YAW_RATE_RAD_S, delta=0.05
        )

    def test_enu_and_ned_setpoints_reach_the_same_ned_pose(self):
        # NED target: 2 m north, 1 m west, 2.5 m up, yaw 0.4 clockwise from north.
        # ENU of that point is east=-1, north=2, up=2.5, yaw = pi/2 - 0.4.
        ned_target = (2.0, -1.0, -2.5, 0.4)
        enu_command = (-1.0, 2.0, 2.5, math.pi / 2.0 - 0.4)
        poses = []
        grounded = []
        for frame, command in (
            (Frame.LOCAL_NED, ned_target),
            (Frame.LOCAL_ENU, enu_command),
        ):
            bot, transport = _pair()
            _takeoff(bot, transport, 2.5)
            bot.offboard_position(*command, frame)
            first = transport.step(DT)
            moved = math.hypot(
                float(_drone(first)["pose"]["x"]),
                float(_drone(first)["pose"]["y"]),
            )
            self.assertLess(moved, 0.2)
            _step(transport, 20.0)
            pose = _drone(transport.read())["pose"]
            self.assertAlmostEqual(pose["roll"], 0.0)
            self.assertAlmostEqual(pose["pitch"], 0.0)
            for value, expected in zip(
                (pose["x"], pose["y"], pose["z"], pose["yaw"]),
                ned_target,
                strict=True,
            ):
                self.assertAlmostEqual(float(value), expected, delta=0.05)
            poses.append(pose)
            bot.land()
            _until(
                transport,
                lambda snap: (
                    _drone(snap)["landed_state"] == "on_ground"
                    and not _drone(snap)["armed"]
                ),
                20.0,
            )
            landed = _drone(transport.read())
            self.assertEqual(landed["landed_state"], "on_ground")
            self.assertFalse(landed["armed"])
            grounded.append(landed["pose"])
        for key in ("x", "y", "z", "yaw"):
            self.assertAlmostEqual(
                float(poses[0][key]), float(poses[1][key]), delta=0.05
            )
        for key in ("x", "y"):
            self.assertAlmostEqual(
                float(grounded[0][key]), float(grounded[1][key]), delta=0.05
            )
            self.assertAlmostEqual(
                float(grounded[0][key]), ned_target[0 if key == "x" else 1], delta=0.08
            )
        self.assertAlmostEqual(float(grounded[0]["z"]), 0.0, places=5)
        self.assertAlmostEqual(float(grounded[1]["z"]), 0.0, places=5)

    def test_enu_yaw_rate_is_nose_left(self):
        bot, transport = _pair()
        _takeoff(bot, transport, 2.5)
        bot.offboard_velocity(0.0, 0.0, 0.0, 0.3, Frame.LOCAL_ENU)
        _step(transport, 2.0)
        drone = _drone(transport.read())
        self.assertLess(float(drone["pose"]["yaw"]), -0.3)
        self.assertAlmostEqual(float(drone["velocity"]["yaw_rate"]), -0.3, delta=0.05)
        yaw_before = float(drone["pose"]["yaw"])
        bot.offboard_velocity(0.0, 0.0, 0.0, 0.3, Frame.LOCAL_NED)
        _step(transport, 2.0)
        later = _drone(transport.read())
        self.assertGreater(float(later["pose"]["yaw"]), yaw_before)
        self.assertAlmostEqual(float(later["velocity"]["yaw_rate"]), 0.3, delta=0.05)

    def test_planar_drive_does_not_climb(self):
        bot, transport = _pair()
        _takeoff(bot, transport, 2.5)
        bot.drive(vx=0.4, vy=-0.1, w=0.0)
        _step(transport, 2.0)
        pose = _drone(transport.read())["pose"]
        self.assertGreater(float(pose["x"]), 0.4)
        self.assertLess(float(pose["y"]), -0.05)
        self.assertAlmostEqual(_alt(transport.read()), 2.5, delta=0.02)
        transport.send_velocity(Velocity(5.0, 0.0, 2.0))
        snap = _step(transport, 2.0)
        vel = _drone(snap)["velocity"]
        speed = math.hypot(float(vel["vx"]), float(vel["vy"]))
        self.assertLessEqual(speed, OHHO_MAX_HORIZONTAL_MPS + 1e-6)
        self.assertGreater(speed, 1.5)
        self.assertAlmostEqual(float(vel["vz"]), 0.0, delta=0.05)
        self.assertAlmostEqual(_alt(snap), 2.5, delta=0.05)

    def test_body_frame_velocity_follows_yaw(self):
        bot, transport = _pair()
        _takeoff(bot, transport, 2.5)
        bot.set_yaw(math.pi / 2.0, Frame.LOCAL_NED)

        def faced(snap: Telemetry) -> bool:
            return abs(float(_drone(snap)["pose"]["yaw"]) - math.pi / 2.0) < 0.05

        _until(transport, faced, 10.0)
        origin = _drone(transport.read())["pose"]
        transport.send_velocity(Velocity(0.4, 0.0, 0.0))
        _step(transport, 2.0)
        pose = _drone(transport.read())["pose"]
        self.assertGreater(float(pose["y"]) - float(origin["y"]), 0.3)
        self.assertLess(abs(float(pose["x"]) - float(origin["x"])), 0.15)
        self.assertAlmostEqual(_alt(transport.read()), 2.5, delta=0.05)
        with self.assertRaises(TypeError):
            bot.offboard_velocity(0.0, 0.0, 0.0, 0.0, "LOCAL_NED")  # type: ignore[arg-type]

    def test_rtl_returns_home_then_disarms_on_the_ground(self):
        bot, transport = _pair()
        _takeoff(bot, transport, 2.5)
        bot.offboard_velocity(0.5, 0.0, 0.0, 0.0, Frame.LOCAL_NED)
        _step(transport, 4.0)
        self.assertGreater(float(_drone(transport.read())["pose"]["x"]), 1.0)
        bot.rtl()
        self.assertEqual(_drone(transport.read())["flight_mode"], "rtl")
        self.assertTrue(_drone(transport.read())["armed"])
        _until(
            transport,
            lambda snap: (
                _drone(snap)["landed_state"] == "on_ground"
                and not _drone(snap)["armed"]
            ),
            40.0,
        )
        pose = _drone(transport.read())["pose"]
        self.assertLessEqual(math.hypot(float(pose["x"]), float(pose["y"])), 0.8)
        self.assertAlmostEqual(float(pose["z"]), 0.0, places=5)
        self.assertEqual(_drone(transport.read())["flight_mode"], "disarmed")

    def test_crazyflie_sim_gps_fix_stays_none(self):
        bot, transport = _pair("crazyflie-sim")
        self.assertEqual(_drone(transport.read())["gps_fix"], "none")
        self.assertIs(_drone(transport.read())["experimental"], True)
        self.assertIs(_drone(transport.read())["flight_ready"], False)
        _takeoff(bot, transport, 1.0)
        self.assertEqual(_drone(transport.read())["gps_fix"], "none")
        self.assertAlmostEqual(_alt(transport.read()), 1.0, delta=0.05)

    def test_battery_drains_faster_in_the_air(self):
        bot, transport = _pair()
        idle = transport.read().battery
        assert idle is not None
        _step(transport, 20.0)
        rested = transport.read().battery
        assert rested is not None
        rested_current = float(_drone(transport.read())["current_a"])
        _takeoff(bot, transport, 1.0)
        _step(transport, 20.0)
        flown = transport.read().battery
        assert flown is not None
        flown_current = float(_drone(transport.read())["current_a"])
        self.assertLess(rested, idle)
        self.assertLess(flown, rested)
        self.assertGreater(flown_current, rested_current)

    def test_connect_path_is_simulated_and_ungated(self):
        bot = Robot.connect("px4-sitl", "sim://", runtime="native")
        try:
            self.assertIsInstance(bot.transport, DroneSimTransport)
            self.assertFalse(bot.gated)
            self.assertEqual(bot.transport.protocol, "simulated")
            bot.arm_flight()
            drone = _drone(bot.telemetry())
            self.assertTrue(drone["armed"])
            self.assertIs(drone["flight_ready"], False)
        finally:
            bot.disconnect()


if __name__ == "__main__":
    unittest.main()
