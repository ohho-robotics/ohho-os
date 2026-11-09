"""OHH-285 drone telemetry keys, frames, and the planar odom shadow."""

from __future__ import annotations

import pathlib
import unittest
from dataclasses import FrozenInstanceError

import ohho
from ohho.drone.telemetry import (
    BODY_FRD,
    DRONE_TELEMETRY_KEYS,
    FAILSAFE_FLAGS,
    LOCAL_ENU,
    LOCAL_NED,
    DronePose,
    DroneTelemetry,
    DroneVelocity,
    FlightMode,
    GpsFix,
    LandedState,
    gps_fix_from_mav,
    landed_state_from_mav,
    positive_up_m,
)
from ohho.schema import Odometry

_PLAN_KEYS = [
    "armed",
    "flight_mode",
    "landed_state",
    "pose",
    "pose_frame",
    "velocity",
    "velocity_frame",
    "battery",
    "voltage_v",
    "current_a",
    "gps_fix",
    "failsafe",
    "link_ok",
    "link_quality",
    "experimental",
    "flight_ready",
]


def _sample() -> DroneTelemetry:
    return DroneTelemetry(
        pose_frame=LOCAL_NED,
        velocity_frame=LOCAL_NED,
        armed=True,
        flight_mode=FlightMode.HOLD,
        landed_state=LandedState.IN_AIR,
        pose=DronePose(x=10.0, y=5.0, z=-2.5, roll=0.1, pitch=-0.2, yaw=1.57),
        velocity=DroneVelocity(vx=0.4, vy=-0.1, vz=0.2, yaw_rate=0.05),
        battery=0.85,
        voltage_v=12.1,
        current_a=4.2,
        gps_fix=GpsFix.FIX_3D,
        failsafe=["geofence"],
        link_ok=True,
        link_quality=0.9,
        timestamp=123.0,
    )


class TestDroneTelemetry(unittest.TestCase):
    def test_plan_keys_and_fixed_flags(self) -> None:
        drone = _sample().to_ohho_telemetry().custom["drone"]
        self.assertIsInstance(drone, dict)
        assert isinstance(drone, dict)
        self.assertEqual(list(drone), _PLAN_KEYS)
        self.assertEqual(DRONE_TELEMETRY_KEYS, tuple(_PLAN_KEYS))
        self.assertIs(drone["experimental"], True)
        self.assertIs(drone["flight_ready"], False)
        self.assertEqual(
            list(drone["pose"]),
            ["x", "y", "z", "roll", "pitch", "yaw"],
        )
        self.assertEqual(list(drone["velocity"]), ["vx", "vy", "vz", "yaw_rate"])

    def test_full_snapshot_values(self) -> None:
        out = _sample().to_ohho_telemetry()
        drone = out.custom["drone"]
        assert isinstance(drone, dict)
        self.assertEqual(out.battery, 0.85)
        self.assertEqual(out.timestamp, 123.0)
        self.assertEqual(out.joints, [])
        self.assertIsNone(out.scan)
        self.assertIsNone(out.odom)
        self.assertIs(drone["armed"], True)
        self.assertEqual(drone["flight_mode"], "hold")
        self.assertIsInstance(drone["flight_mode"], str)
        self.assertNotIsInstance(drone["flight_mode"], FlightMode)
        self.assertEqual(drone["landed_state"], "in_air")
        self.assertNotIsInstance(drone["landed_state"], LandedState)
        self.assertEqual(
            drone["pose"],
            {"x": 10.0, "y": 5.0, "z": -2.5, "roll": 0.1, "pitch": -0.2, "yaw": 1.57},
        )
        self.assertEqual(
            drone["velocity"],
            {"vx": 0.4, "vy": -0.1, "vz": 0.2, "yaw_rate": 0.05},
        )
        self.assertEqual(drone["pose_frame"], "LOCAL_NED")
        self.assertEqual(drone["velocity_frame"], "LOCAL_NED")
        self.assertEqual(drone["battery"], 0.85)
        self.assertEqual(drone["voltage_v"], 12.1)
        self.assertEqual(drone["current_a"], 4.2)
        self.assertEqual(drone["gps_fix"], "3d")
        self.assertEqual(drone["failsafe"], ["geofence"])
        self.assertIsInstance(drone["failsafe"], list)
        self.assertIs(drone["link_ok"], True)
        self.assertEqual(drone["link_quality"], 0.9)
        self.assertIsInstance(drone["pose"]["z"], float)

    def test_experimental_and_flight_ready_ignore_caller_overrides(self) -> None:
        drone_override: dict[str, object] = {
            "experimental": False,
            "flight_ready": True,
            "armed": False,
        }
        extra: dict[str, object] = {"source": "sitl", "drone": drone_override}
        telemetry = DroneTelemetry(
            pose_frame=LOCAL_NED,
            velocity_frame=LOCAL_NED,
            armed=True,
            custom=extra,
        )
        extra["source"] = "mutated"
        drone_override["flight_ready"] = True
        out = telemetry.to_ohho_telemetry()
        drone = out.custom["drone"]
        assert isinstance(drone, dict)
        self.assertEqual(out.custom["source"], "sitl")
        self.assertIs(drone["experimental"], True)
        self.assertIs(drone["flight_ready"], False)
        self.assertIs(drone["armed"], True)
        self.assertNotIn("flight_ready", {key for key in out.custom if key != "drone"})

    def test_flight_ready_is_not_a_field(self) -> None:
        with self.assertRaises(TypeError):
            DroneTelemetry(
                pose_frame=LOCAL_NED,
                velocity_frame=LOCAL_NED,
                flight_ready=True,  # type: ignore[call-arg]
            )
        telemetry = DroneTelemetry(pose_frame=LOCAL_ENU, velocity_frame=LOCAL_ENU)
        with self.assertRaises((AttributeError, FrozenInstanceError)):
            telemetry.flight_ready = True  # type: ignore[attr-defined]
        with self.assertRaises(FrozenInstanceError):
            telemetry.armed = True  # type: ignore[misc]
        drone = telemetry.to_ohho_telemetry().custom["drone"]
        assert isinstance(drone, dict)
        self.assertIs(drone["flight_ready"], False)
        self.assertIs(drone["experimental"], True)

    def test_unknown_landed_state_is_not_on_ground(self) -> None:
        for raw in (None, 0, LandedState.UNKNOWN, "unknown", "UNKNOWN"):
            telemetry = DroneTelemetry(
                pose_frame=LOCAL_NED,
                velocity_frame=LOCAL_NED,
                landed_state=raw,
            )
            self.assertEqual(telemetry.landed_state, LandedState.UNKNOWN)
            landed = telemetry.to_ohho_telemetry().custom["drone"]["landed_state"]
            self.assertEqual(landed, "unknown")
            self.assertNotEqual(landed, "on_ground")

    def test_landed_state_enum_and_mav_mapping(self) -> None:
        self.assertEqual(
            {state.value for state in LandedState},
            {"on_ground", "in_air", "takeoff", "landing", "unknown"},
        )
        expected = {
            None: LandedState.UNKNOWN,
            0: LandedState.UNKNOWN,
            1: LandedState.ON_GROUND,
            2: LandedState.IN_AIR,
            3: LandedState.TAKEOFF,
            4: LandedState.LANDING,
            9: LandedState.UNKNOWN,
            -1: LandedState.UNKNOWN,
        }
        for raw, landed in expected.items():
            self.assertEqual(landed_state_from_mav(raw), landed)
        self.assertEqual(
            DroneTelemetry(
                pose_frame=LOCAL_NED,
                velocity_frame=LOCAL_NED,
                landed_state=2,
            ).landed_state,
            LandedState.IN_AIR,
        )
        with self.assertRaises(ValueError):
            DroneTelemetry(
                pose_frame=LOCAL_NED,
                velocity_frame=LOCAL_NED,
                landed_state=9,
            )
        with self.assertRaises(ValueError):
            DroneTelemetry(
                pose_frame=LOCAL_NED,
                velocity_frame=LOCAL_NED,
                landed_state="flying",
            )

    def test_gps_fix_enum_and_mav_mapping(self) -> None:
        self.assertEqual(
            {fix.value for fix in GpsFix},
            {"none", "2d", "3d", "dgps", "rtk_float", "rtk_fixed"},
        )
        for raw, name in (
            (0, "none"),
            (1, "none"),
            (2, "2d"),
            (3, "3d"),
            (4, "dgps"),
            (5, "rtk_float"),
            (6, "rtk_fixed"),
            (7, "none"),
            (8, "none"),
        ):
            self.assertEqual(gps_fix_from_mav(raw).value, name)
        with self.assertRaises(TypeError):
            gps_fix_from_mav(True)  # type: ignore[arg-type]
        telemetry = DroneTelemetry(
            pose_frame=LOCAL_NED,
            velocity_frame=LOCAL_NED,
            gps_fix="rtk_fixed",
        )
        self.assertEqual(telemetry.gps_fix, GpsFix.RTK_FIXED)
        self.assertEqual(
            telemetry.to_ohho_telemetry().custom["drone"]["gps_fix"], "rtk_fixed"
        )

    def test_frames_are_explicit_strings(self) -> None:
        with self.assertRaises(TypeError):
            DroneTelemetry()  # type: ignore[call-arg]
        enu = DroneTelemetry(pose_frame="LOCAL_ENU", velocity_frame="LOCAL_ENU")
        drone = enu.to_ohho_telemetry().custom["drone"]
        assert isinstance(drone, dict)
        self.assertEqual(drone["pose_frame"], "LOCAL_ENU")
        self.assertEqual(drone["velocity_frame"], "LOCAL_ENU")
        bad_frames: tuple[tuple[object, object], ...] = (
            ("local_ned", LOCAL_NED),
            ("BODY_FRD", "BODY_FRD"),
            (None, LOCAL_NED),
            (LOCAL_NED, "body_frd"),
            ("", LOCAL_NED),
        )
        for pose_frame, velocity_frame in bad_frames:
            with self.assertRaises(ValueError):
                DroneTelemetry(
                    pose_frame=pose_frame,  # type: ignore[arg-type]
                    velocity_frame=velocity_frame,  # type: ignore[arg-type]
                )

    def test_body_frd_is_velocity_only(self) -> None:
        telemetry = DroneTelemetry(pose_frame=LOCAL_NED, velocity_frame=BODY_FRD)
        drone = telemetry.to_ohho_telemetry().custom["drone"]
        assert isinstance(drone, dict)
        self.assertEqual(drone["velocity_frame"], "BODY_FRD")
        self.assertIsNone(telemetry.to_ohho_telemetry().odom)
        with self.assertRaises(ValueError):
            telemetry.planar_odom()
        with self.assertRaises(ValueError):
            telemetry.to_ohho_telemetry(include_planar_odom=True)
        with self.assertRaises(ValueError):
            positive_up_m(1.0, BODY_FRD)

    def test_positive_up_follows_the_pose_frame(self) -> None:
        self.assertEqual(positive_up_m(-2.5, LOCAL_NED), 2.5)
        self.assertEqual(positive_up_m(0.2, LOCAL_NED), -0.2)
        self.assertEqual(positive_up_m(2.5, LOCAL_ENU), 2.5)
        ned = DroneTelemetry(
            pose_frame=LOCAL_NED,
            velocity_frame=LOCAL_NED,
            pose=DronePose(z=-3.0),
        )
        enu = DroneTelemetry(
            pose_frame=LOCAL_ENU,
            velocity_frame=LOCAL_ENU,
            pose=DronePose(z=3.0),
        )
        self.assertEqual(ned.local_altitude_up_m(), 3.0)
        self.assertEqual(enu.local_altitude_up_m(), 3.0)

    def test_planar_odom_shadow_is_optional_and_not_altitude(self) -> None:
        telemetry = DroneTelemetry(
            pose_frame=LOCAL_NED,
            velocity_frame=LOCAL_NED,
            pose=DronePose(x=3.0, y=4.0, z=-4.0, yaw=0.5),
            velocity=DroneVelocity(vx=0.1, vy=0.2, vz=9.0, yaw_rate=0.3),
        )
        self.assertIsNone(telemetry.to_ohho_telemetry().odom)
        shadow = telemetry.to_ohho_telemetry(include_planar_odom=True).odom
        self.assertEqual(
            shadow, Odometry(x=3.0, y=4.0, theta=0.5, vx=0.1, vy=0.2, omega=0.3)
        )
        drone = telemetry.as_custom()
        assert isinstance(drone["pose"], dict)
        assert isinstance(drone["velocity"], dict)
        self.assertEqual(drone["pose"]["z"], -4.0)
        self.assertEqual(drone["velocity"]["vz"], 9.0)
        explicit = Odometry(x=1.0, y=2.0, theta=0.3)
        mixed = DroneTelemetry(
            pose_frame=LOCAL_NED,
            velocity_frame=BODY_FRD,
            odom=explicit,
        )
        kept = mixed.to_ohho_telemetry(include_planar_odom=True).odom
        self.assertIsNotNone(kept)
        assert kept is not None
        self.assertEqual(kept.x, 1.0)
        self.assertEqual(kept.y, 2.0)
        self.assertEqual(kept.theta, 0.3)

    def test_battery_fraction_and_electrical(self) -> None:
        empty = DroneTelemetry(pose_frame=LOCAL_NED, velocity_frame=LOCAL_NED)
        self.assertIsNone(empty.to_ohho_telemetry().battery)
        self.assertIsNone(empty.as_custom()["battery"])
        zero = DroneTelemetry(
            pose_frame=LOCAL_NED, velocity_frame=LOCAL_NED, battery=0.0
        )
        self.assertEqual(zero.to_ohho_telemetry().battery, 0.0)
        full = DroneTelemetry(pose_frame=LOCAL_NED, velocity_frame=LOCAL_NED, battery=1)
        self.assertEqual(full.as_custom()["battery"], 1.0)
        for battery in (1.1, -0.01, float("nan")):
            with self.assertRaises(ValueError):
                DroneTelemetry(
                    pose_frame=LOCAL_NED, velocity_frame=LOCAL_NED, battery=battery
                )

    def test_link_quality_bounds_and_failsafe_copy(self) -> None:
        flags = ["geofence"]
        telemetry = DroneTelemetry(
            pose_frame=LOCAL_NED,
            velocity_frame=LOCAL_NED,
            failsafe=flags,
            link_quality=0.5,
        )
        flags.append("rc_loss")
        self.assertEqual(telemetry.failsafe, ("geofence",))
        first = telemetry.as_custom()["failsafe"]
        assert isinstance(first, list)
        first.append("battery")
        self.assertEqual(telemetry.as_custom()["failsafe"], ["geofence"])
        carried = DroneTelemetry(
            pose_frame=LOCAL_NED,
            velocity_frame=LOCAL_NED,
            failsafe=list(FAILSAFE_FLAGS),
            link_ok=True,
            link_quality=0.0,
        )
        drone = carried.as_custom()
        self.assertEqual(drone["failsafe"], list(FAILSAFE_FLAGS))
        self.assertEqual(drone["link_quality"], 0.0)
        self.assertIs(drone["link_ok"], True)
        edge = DroneTelemetry(
            pose_frame=LOCAL_NED, velocity_frame=LOCAL_NED, link_quality=1
        )
        self.assertEqual(edge.link_quality, 1.0)
        self.assertIsNone(
            DroneTelemetry(pose_frame=LOCAL_NED, velocity_frame=LOCAL_NED).link_quality
        )
        for quality in (1.1, -0.1, float("inf")):
            with self.assertRaises(ValueError):
                DroneTelemetry(
                    pose_frame=LOCAL_NED,
                    velocity_frame=LOCAL_NED,
                    link_quality=quality,
                )

    def test_backend_mode_string_is_preserved(self) -> None:
        telemetry = DroneTelemetry(
            pose_frame=LOCAL_NED,
            velocity_frame=LOCAL_NED,
            flight_mode="  AUTO.LOITER  ",
        )
        self.assertEqual(telemetry.flight_mode, "AUTO.LOITER")
        with self.assertRaises(ValueError):
            DroneTelemetry(
                pose_frame=LOCAL_NED, velocity_frame=LOCAL_NED, flight_mode="  "
            )

    def test_rejects_bad_types(self) -> None:
        with self.assertRaises(TypeError):
            DroneTelemetry(pose_frame=LOCAL_NED, velocity_frame=LOCAL_NED, armed=1)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            DroneTelemetry(pose_frame=LOCAL_NED, velocity_frame=LOCAL_NED, link_ok=1)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            DroneTelemetry(
                pose_frame=LOCAL_NED,
                velocity_frame=LOCAL_NED,
                failsafe="geofence",  # type: ignore[arg-type]
            )
        with self.assertRaises(TypeError):
            DroneTelemetry(
                pose_frame=LOCAL_NED,
                velocity_frame=LOCAL_NED,
                voltage_v=True,  # type: ignore[arg-type]
            )
        with self.assertRaises(ValueError):
            DroneTelemetry(
                pose_frame=LOCAL_NED,
                velocity_frame=LOCAL_NED,
                pose=DronePose(z=float("nan")),
            )

    def test_base_dependencies_stay_empty(self) -> None:
        root = pathlib.Path(ohho.__file__).resolve().parent.parent
        text = (root / "pyproject.toml").read_text(encoding="utf-8")
        self.assertIn("\ndependencies = []\n", text)


if __name__ == "__main__":
    unittest.main()
