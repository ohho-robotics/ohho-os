"""Robot-agnostic sim-serve stream: posture decisions, pose, joint order."""

from __future__ import annotations

import math
import unittest
from types import SimpleNamespace

from ohho.capabilities import HUMANOID_POSTURE
from ohho.humanoid_sim import rpy_to_quat_wxyz
from ohho.schema import Odometry, Telemetry
from ohho.sim_serve import SimServe
from ohho.sim_stream import (
    STREAM_HZ,
    build_stream_frame,
    decide_mode_request,
    ordered_joint_state,
    pose_from_sources,
    stream_joint_names,
)


def _spec(modes: tuple[str, ...] = ("prepare", "walk")) -> SimpleNamespace:
    return SimpleNamespace(
        id="desc-humanoid",
        joint_names=("hip", "knee"),
        capabilities=(HUMANOID_POSTURE,),
        capability_params={"humanoid.posture": {"modes": list(modes)}},
        has=lambda cap: cap == HUMANOID_POSTURE,
    )


class StreamContractTest(unittest.TestCase):
    def test_default_rate_is_thirty_hertz(self) -> None:
        self.assertEqual(STREAM_HZ, 30.0)
        self.assertEqual(SimServe(backend="sim").rate_hz, 30.0)

    def test_prepare_and_walk_are_accepted_on_sim(self) -> None:
        spec = _spec()
        for request, action in (("prepare", "prepare"), ("walk", "walk_mode")):
            decision = decide_mode_request(spec, request, backend="sim", uri="sim://")
            self.assertTrue(decision.accepted, decision.reason)
            self.assertEqual(decision.action, action)

    def test_undeclared_and_unknown_requests_are_refused(self) -> None:
        spec = _spec(("prepare",))
        undeclared = decide_mode_request(spec, "walk", backend="sim", uri="sim://")
        self.assertFalse(undeclared.accepted)
        self.assertEqual(undeclared.reason, "undeclared")
        unknown = decide_mode_request(spec, "flip", backend="sim", uri="sim://")
        self.assertFalse(unknown.accepted)
        self.assertEqual(unknown.reason, "undeclared")
        blank = decide_mode_request(spec, "  ", backend="sim", uri="sim://")
        self.assertEqual(blank.reason, "request")

    def test_robot_backend_needs_the_arming_phrase_and_env(self) -> None:
        spec = _spec()
        missing = decide_mode_request(
            spec,
            "prepare",
            backend="robot",
            uri="serial://bus",
            env={},
        )
        self.assertFalse(missing.accepted)
        self.assertEqual(missing.reason, "confirm")
        armed = decide_mode_request(
            spec,
            "prepare",
            backend="robot",
            uri="serial://bus",
            confirm="I am physically present, the area is clear, the remote is in my hand",
            env={"OHHO_ARM_HARDWARE": "1"},
        )
        self.assertTrue(armed.accepted)

    def test_missing_capability_is_refused(self) -> None:
        spec = SimpleNamespace(
            capabilities=(),
            capability_params={},
            has=lambda _cap: False,
        )
        decision = decide_mode_request(spec, "prepare", backend="sim", uri="sim://")
        self.assertFalse(decision.accepted)
        self.assertEqual(decision.reason, "capability")

    def test_ordered_positions_follow_names_and_fill_gaps(self) -> None:
        state = ordered_joint_state(("hip", "knee", "ankle"), {"knee": 0.4})
        self.assertEqual(state["names"], ["hip", "knee", "ankle"])
        self.assertEqual(state["positions"], [0.0, 0.4, 0.0])

    def test_pose_quaternion_matches_the_humanoid_helper(self) -> None:
        telemetry = Telemetry(
            custom={"humanoid": {"base_rpy": [0.1, -0.2, 0.3]}},
        )
        pose = pose_from_sources(
            {"x": 1.0, "y": 2.0, "yaw": 0.3},
            telemetry=telemetry,
            base_position=(1.0, 2.0, 0.7),
        )
        w, x, y, z = rpy_to_quat_wxyz(0.1, -0.2, 0.3)
        self.assertEqual(pose["position"], {"x": 1.0, "y": 2.0, "z": 0.7})
        self.assertEqual(
            pose["orientation"],
            {"x": x, "y": y, "z": z, "w": w},
        )
        self.assertAlmostEqual(math.sqrt(w * w + x * x + y * y + z * z), 1.0, places=6)

    def test_frame_keeps_legacy_fields_and_adds_the_stream(self) -> None:
        frame = build_stream_frame(
            backend="sim",
            odom={"x": 0.2, "y": 0.0, "yaw": 0.1},
            joints={"hip": 0.5},
            names=("hip", "knee"),
            pose=pose_from_sources({"x": 0.2, "y": 0.0, "yaw": 0.1}),
            estop=False,
            deadman=True,
            connected=True,
            posture="WALK",
        )
        self.assertEqual(frame["odom"], {"x": 0.2, "y": 0.0, "yaw": 0.1})
        self.assertEqual(frame["joints"], {"hip": 0.5})
        self.assertEqual(frame["backend"], "sim")
        self.assertTrue(frame["no_camera"])
        self.assertEqual(frame["joint_state"]["names"], ["hip", "knee"])
        self.assertEqual(frame["joint_state"]["positions"], [0.5, 0.0])
        self.assertEqual(frame["status"]["posture"], "WALK")
        self.assertNotIn("modes", frame)

    def test_unregistered_spec_falls_back_to_joint_names(self) -> None:
        spec = SimpleNamespace(id="not-a-registered-robot", joint_names=("a", "b"))
        self.assertEqual(stream_joint_names(spec), ("a", "b"))

    def test_humanoid_block_supplies_roll_pitch_and_height(self) -> None:
        telemetry = Telemetry(
            odom=Odometry(x=0.0, y=0.0, theta=0.0),
            custom={
                "humanoid": {"base_rpy": [0.0, 0.0, 0.0], "base_position": [0, 0, 0.8]}
            },
        )
        pose = pose_from_sources({"x": 0.0, "y": 0.0, "yaw": 0.0}, telemetry=telemetry)
        self.assertEqual(pose["position"]["z"], 0.8)


if __name__ == "__main__":
    unittest.main()
