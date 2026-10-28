"""Humanoid telemetry block on Telemetry.custom['humanoid'] (OHH-327).

No hardware. SafetyGate is not involved.
"""

from __future__ import annotations

import json
import re
import unittest
from pathlib import Path

from ohho import capabilities as caps
from ohho.descriptor import robot_descriptor, telemetry_schema_docs
from ohho.humanoid import (
    FallState,
    HumanoidMode,
    fall_state_of,
    is_standing,
    is_upright,
    mode_of,
)
from ohho.humanoid_telemetry import (
    HUMANOID_BLOCK_KEYS,
    HUMANOID_BLOCK_SCHEMA,
    apply_humanoid_telemetry,
    build_humanoid_telemetry,
    dumps_humanoid_telemetry,
    dumps_telemetry,
    loads_humanoid_telemetry,
    loads_telemetry,
    validate_humanoid_telemetry,
)
from ohho.registry import RobotSpec, _REGISTRY, register
from ohho.robot_spec import RobotClass
from ohho.schema import (
    TELEMETRY_SCHEMA_ID,
    JointReading,
    Odometry,
    Scan,
    Telemetry,
)

# Names that must not appear in the new core modules. This file may list them.
_VENDOR = re.compile(
    r"booster|unitree|\bgo2\b|\bt2\b|px4|crazyflie|yahboom|omnibot|\bdji\b|mavlink",
    re.IGNORECASE,
)

_GROUPS = {
    "left_leg": ("hip", "knee", "ankle"),
    "right_arm": ("shoulder", "elbow"),
}


def _raw(**overrides: object) -> dict:
    body: dict = {
        "mode": "WALK",
        "vendor_mode": "walk-high",
        "fall_state": "upright",
        "base_rpy": [0.01, -0.02, 0.3],
        "base_ang_vel": [0.1, 0.0, -0.2],
        "joints": {
            "left_leg": {
                "q": [0.0, 0.1, -0.1],
                "dq": [0.0, 0.0, 0.0],
                "tau": [1.0, 2.0, 3.0],
                "temp": [30.0, 31.0, 32.0],
            },
            "right_arm": {
                "q": [0.2, -0.4],
                "dq": [0.0, 0.1],
                "tau": [0.5, 0.5],
                "temp": [28.0, 29.0],
            },
        },
        "odom": {"x": 1.5, "y": -0.25, "yaw": 0.4},
        "battery": {"fraction": 0.8, "voltage": 48.0},
        "experimental": True,
    }
    body.update(overrides)
    return body


def _spec() -> RobotSpec:
    return RobotSpec(
        id="humanoid-telemetry-test",
        name="Humanoid telemetry test",
        category="humanoid",
        robot_class=RobotClass.HUMANOID,
        joint_groups=dict(_GROUPS),
        max_lin=0.5,
        max_ang=0.6,
    )


class TestHumanoidBlock(unittest.TestCase):
    def test_every_section_key_is_present_and_typed(self):
        block = build_humanoid_telemetry(**_raw(), spec=_spec())
        body = block.to_dict()
        self.assertEqual(list(body), list(HUMANOID_BLOCK_KEYS))
        self.assertEqual(set(HUMANOID_BLOCK_SCHEMA["keys"]), set(HUMANOID_BLOCK_KEYS))
        self.assertEqual(body["mode"], "WALK")
        self.assertEqual(block.mode, HumanoidMode.WALK)
        self.assertEqual(body["vendor_mode"], "walk-high")
        self.assertEqual(body["fall_state"], "upright")
        self.assertEqual(block.fall_state, FallState.UPRIGHT)
        self.assertEqual(body["base_rpy"], [0.01, -0.02, 0.3])
        self.assertEqual(body["base_ang_vel"], [0.1, 0.0, -0.2])
        self.assertEqual(
            body["joints"]["left_leg"],
            {
                "q": [0.0, 0.1, -0.1],
                "dq": [0.0, 0.0, 0.0],
                "tau": [1.0, 2.0, 3.0],
                "temp": [30.0, 31.0, 32.0],
            },
        )
        self.assertEqual(body["joints"]["right_arm"]["q"], [0.2, -0.4])
        self.assertEqual(body["odom"], {"x": 1.5, "y": -0.25, "yaw": 0.4})
        self.assertEqual(body["battery"], {"fraction": 0.8, "voltage": 48.0})
        self.assertIs(body["experimental"], True)
        self.assertEqual(block.base_rpy, (0.01, -0.02, 0.3))
        self.assertEqual(block.base_ang_vel, (0.1, 0.0, -0.2))

    def test_experimental_defaults_to_true(self):
        raw = _raw()
        del raw["experimental"]
        block = build_humanoid_telemetry(**raw, spec=_spec())
        self.assertIs(block.experimental, True)
        recorded = build_humanoid_telemetry(**_raw(experimental=False), spec=_spec())
        self.assertIs(recorded.experimental, False)

    def test_vendor_mode_is_stored_and_not_a_mode(self):
        block = build_humanoid_telemetry(
            **_raw(mode="UNKNOWN", vendor_mode="kWalking", fall_state="unknown"),
            spec=_spec(),
        )
        self.assertEqual(block.vendor_mode, "kWalking")
        self.assertIs(block.mode, HumanoidMode.UNKNOWN)
        self.assertFalse(is_standing(block.mode))
        self.assertFalse(is_upright(block))
        with self.assertRaises(ValueError):
            build_humanoid_telemetry(**_raw(mode="kWalking"), spec=_spec())

    def test_unreported_joint_channels_are_omitted(self):
        joints = _raw()["joints"]
        del joints["left_leg"]["temp"]
        joints["left_leg"]["tau"] = None
        joints["right_arm"] = {"q": [0.2, -0.4]}
        block = validate_humanoid_telemetry(_raw(joints=joints), spec=_spec())
        left = block.to_dict()["joints"]["left_leg"]
        self.assertNotIn("temp", left)
        self.assertNotIn("tau", left)
        self.assertIsNone(block.joints["left_leg"].temp)
        self.assertEqual(block.joints["right_arm"].q, (0.2, -0.4))
        self.assertIsNone(block.joints["right_arm"].dq)

    def test_json_round_trip(self):
        block = build_humanoid_telemetry(**_raw(), joint_groups=_GROUPS)
        again = loads_humanoid_telemetry(
            dumps_humanoid_telemetry(block), joint_groups=_GROUPS
        )
        self.assertEqual(again, block)
        parsed = json.loads(dumps_humanoid_telemetry(block))
        self.assertEqual(set(parsed), set(HUMANOID_BLOCK_KEYS))

        snap = telemetry_from(block)
        text = dumps_telemetry(snap)
        back = loads_telemetry(text, spec=_spec())
        self.assertEqual(back.custom["humanoid"], block.to_dict())
        self.assertEqual(back.to_dict()["schema_id"], TELEMETRY_SCHEMA_ID)
        self.assertEqual(json.loads(text)["schema_id"], "ohho.telemetry.v1")
        self.assertEqual(back.timestamp, snap.timestamp)

    def test_plain_snapshot_round_trip(self):
        snap = Telemetry(
            odom=Odometry(1.0, 2.0, 3.0, 0.1, 0.2, 0.3),
            joints=[JointReading("a", 0.5, 0.1)],
            battery=0.4,
            scan=Scan(0.0, 0.1, [1.0, 2.0], 4.0),
            custom={"flag": True},
            timestamp=3.0,
        )
        back = Telemetry.from_dict(json.loads(json.dumps(snap.to_dict())))
        self.assertEqual(back.odom, snap.odom)
        self.assertEqual(back.joints, snap.joints)
        self.assertEqual(back.battery, 0.4)
        self.assertEqual(back.scan, snap.scan)
        self.assertEqual(back.custom, {"flag": True})
        self.assertEqual(back.timestamp, 3.0)
        self.assertIsNone(
            loads_telemetry(dumps_telemetry(Telemetry(timestamp=1.0))).odom
        )
        bad = snap.to_dict()
        bad["schema_id"] = "nope"
        with self.assertRaises(ValueError):
            Telemetry.from_dict(bad)
        bad = snap.to_dict()
        bad["battery"] = True
        with self.assertRaises(ValueError):
            Telemetry.from_dict(bad)
        raw = Telemetry(timestamp=1.0).to_dict()
        raw["custom"] = {"humanoid": "upright"}
        with self.assertRaises(TypeError):
            loads_telemetry(json.dumps(raw))


class TestJointGroups(unittest.TestCase):
    def test_groups_must_match_the_spec(self):
        block = validate_humanoid_telemetry(_raw(), spec=_spec())
        self.assertEqual(tuple(block.joints), ("left_leg", "right_arm"))

        reversed_joints = {
            "right_arm": _raw()["joints"]["right_arm"],
            "left_leg": _raw()["joints"]["left_leg"],
        }
        ordered = validate_humanoid_telemetry(
            _raw(joints=reversed_joints), spec=_spec()
        )
        self.assertEqual(tuple(ordered.joints), ("left_leg", "right_arm"))

        extra = _raw()["joints"]
        extra["head"] = {"q": [0.0]}
        with self.assertRaises(ValueError) as unknown:
            validate_humanoid_telemetry(_raw(joints=extra), spec=_spec())
        self.assertIn("unknown", str(unknown.exception))

        missing = _raw()["joints"]
        del missing["right_arm"]
        with self.assertRaises(ValueError) as gone:
            validate_humanoid_telemetry(_raw(joints=missing), spec=_spec())
        self.assertIn("missing", str(gone.exception))

    def test_channel_length_matches_the_group(self):
        joints = _raw()["joints"]
        joints["left_leg"]["q"] = [0.0, 0.1]
        with self.assertRaises(ValueError):
            validate_humanoid_telemetry(_raw(joints=joints), joint_groups=_GROUPS)

        mismatched = _raw()["joints"]
        mismatched["right_arm"]["dq"] = [0.0]
        with self.assertRaises(ValueError):
            validate_humanoid_telemetry(_raw(joints=mismatched), spec=_spec())

        empty = {"left_leg": {}, "right_arm": {}}
        quiet = validate_humanoid_telemetry(_raw(joints=empty), spec=_spec())
        self.assertIsNone(quiet.joints["left_leg"].q)
        self.assertIsNone(quiet.joints["right_arm"].temp)

    def test_without_a_spec_channels_in_a_group_share_a_length(self):
        joints = {
            "head": {"q": [0.1, 0.2], "dq": [0.0, 0.0]},
        }
        block = validate_humanoid_telemetry(_raw(joints=joints))
        self.assertEqual(block.joints["head"].q, (0.1, 0.2))
        joints["head"]["tau"] = [1.0]
        with self.assertRaises(ValueError):
            validate_humanoid_telemetry(_raw(joints=joints))

    def test_spec_and_joint_groups_together_are_refused(self):
        with self.assertRaises(TypeError):
            validate_humanoid_telemetry(_raw(), spec=_spec(), joint_groups=_GROUPS)


class TestFallState(unittest.TestCase):
    def test_unknown_and_missing_are_not_upright(self):
        for state in FallState:
            block = build_humanoid_telemetry(**_raw(fall_state=state), spec=_spec())
            snap = telemetry_from(block)
            upright = state is FallState.UPRIGHT
            self.assertIs(is_upright(state), upright)
            self.assertIs(is_upright(state.value), upright)
            self.assertIs(is_upright(block), upright)
            self.assertIs(is_upright(snap), upright)
            self.assertIs(is_upright(snap.custom), upright)
            self.assertIs(is_upright(snap.custom["humanoid"]), upright)

        self.assertFalse(is_upright(None))
        self.assertFalse(is_upright(Telemetry()))
        self.assertFalse(is_upright(""))
        self.assertFalse(is_upright("unknown"))
        self.assertFalse(is_upright("UNKNOWN"))
        self.assertFalse(is_upright("  unknown  "))
        self.assertFalse(is_upright(FallState.UNKNOWN))
        self.assertFalse(is_upright(HumanoidMode.PREPARE))
        self.assertFalse(is_upright(HumanoidMode.WALK))
        self.assertTrue(is_upright(" upright "))
        self.assertTrue(is_upright(FallState.UPRIGHT))

        missing = Telemetry(custom={"humanoid": {"mode": "WALK"}})
        self.assertIs(mode_of(missing), HumanoidMode.WALK)
        self.assertTrue(is_standing(mode_of(missing)))
        self.assertFalse(is_upright(missing))
        self.assertIs(fall_state_of(missing), FallState.UNKNOWN)

        self.assertFalse(is_upright(Telemetry(custom={"fall_state": "upright"})))
        self.assertFalse(is_upright(Telemetry(custom={"humanoid": "upright"})))
        self.assertFalse(
            is_upright({"humanoid": {"fall_state": "unknown"}, "fall_state": "upright"})
        )
        self.assertTrue(
            is_upright({"humanoid": {"fall_state": "upright"}, "fall_state": "fallen"})
        )

        for name in ("falling", "fallen", "recovering", "unknown"):
            self.assertFalse(is_upright(name))
            self.assertFalse(is_upright(parse_block(name)))

    def test_walking_with_unknown_fall_state_is_not_upright(self):
        block = build_humanoid_telemetry(
            **_raw(mode="WALK", fall_state="unknown"), spec=_spec()
        )
        snap = telemetry_from(block)
        self.assertTrue(is_standing(mode_of(snap)))
        self.assertFalse(is_upright(snap))


class TestOdomShadow(unittest.TestCase):
    def test_planar_odom_is_copied_onto_telemetry_odom(self):
        block = build_humanoid_telemetry(**_raw(), spec=_spec())
        snap = Telemetry(
            odom=Odometry(x=9.0, y=9.0, theta=9.0, vx=0.2, vy=-0.1, omega=0.05),
            custom={"humanoid": block.to_dict(), "note": "keep"},
            timestamp=4.0,
        )
        self.assertEqual(snap.odom.x, 1.5)
        self.assertEqual(snap.odom.y, -0.25)
        self.assertEqual(snap.odom.theta, 0.4)
        self.assertEqual(snap.odom.vx, 0.2)
        self.assertEqual(snap.odom.vy, -0.1)
        self.assertEqual(snap.odom.omega, 0.05)
        self.assertEqual(snap.custom["note"], "keep")

        fresh = Telemetry(custom={"note": "keep"})
        apply_humanoid_telemetry(fresh, block)
        self.assertEqual(
            (fresh.odom.x, fresh.odom.y, fresh.odom.theta),
            (block.odom.x, block.odom.y, block.odom.yaw),
        )
        self.assertEqual(fresh.odom.vx, 0.0)
        self.assertEqual(fresh.custom["note"], "keep")
        self.assertIsNotNone(fresh.odom)

    def test_shadow_survives_json(self):
        block = build_humanoid_telemetry(**_raw(), spec=_spec())
        snap = telemetry_from(block)
        back = Telemetry.from_dict(json.loads(json.dumps(snap.to_dict())))
        self.assertEqual(back.odom.x, block.odom.x)
        self.assertEqual(back.odom.y, block.odom.y)
        self.assertEqual(back.odom.theta, block.odom.yaw)
        self.assertEqual(loads_telemetry(dumps_telemetry(snap)).odom.theta, 0.4)


class TestTelemetryDocs(unittest.TestCase):
    def test_humanoid_spec_lists_the_block(self):
        register(
            RobotSpec(
                id="desc-humanoid",
                name="Desc Humanoid",
                category="humanoid",
                capabilities=(caps.BASE_DRIVE,),
                max_lin=0.2,
                max_ang=0.6,
                robot_class=RobotClass.HUMANOID,
                joint_names=("hip", "knee"),
                joint_groups={"legs": ("hip", "knee")},
            )
        )
        self.addCleanup(lambda: _REGISTRY.pop("desc-humanoid", None))
        doc = robot_descriptor("desc-humanoid")
        self.assertEqual(doc["telemetry"]["schema_id"], "ohho.telemetry.v1")
        self.assertEqual(doc["telemetry"]["fields"], ["base", "joints", "humanoid"])
        self.assertEqual(doc["drive"]["kind"], "bipedal")
        docs = telemetry_schema_docs()
        self.assertEqual(docs["schema_id"], TELEMETRY_SCHEMA_ID)
        self.assertEqual(tuple(docs["humanoid"]["keys"]), HUMANOID_BLOCK_KEYS)
        self.assertIn("humanoid", docs["blocks"])
        self.assertIs(docs["humanoid"]["unknown_fall_state_is_upright"], False)
        self.assertNotIn("humanoid", robot_descriptor("sim")["telemetry"]["fields"])

    def test_core_modules_do_not_name_a_vendor(self):
        root = Path(__file__).resolve().parents[1]
        for rel in (
            "ohho/humanoid_telemetry.py",
            "ohho/humanoid.py",
            "ohho/schema.py",
        ):
            text = (root / rel).read_text(encoding="utf-8")
            self.assertIsNone(_VENDOR.search(text), rel)


def telemetry_from(block):
    from ohho.humanoid_telemetry import telemetry_with_humanoid

    return telemetry_with_humanoid(block, timestamp=12.0)


def parse_block(fall_state: str):
    return build_humanoid_telemetry(**_raw(fall_state=fall_state), spec=_spec())


if __name__ == "__main__":
    unittest.main()
