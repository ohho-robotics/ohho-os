"""Conformance kit and vendor-name grep (OHH-315)."""

from __future__ import annotations

import contextlib
import importlib
import io
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from ohho import capabilities as caps
from ohho.adapters.composite import CompositeTransport
from ohho.adapters.feetech import FeetechTransport
from ohho.adapters.ros2 import Ros2Transport
from ohho.adapters.unitree import UnitreeDdsTransport
from ohho.adapters.yahboom import YahboomTransport
from ohho.plugins import loaded_plugins
from ohho.registry import get_spec
from ohho.robot_spec import ArmLimits, RobotClass
from ohho.testing.conformance import (
    CLASS_SCENARIOS,
    evaluate_mcp,
    format_result,
    main,
    open_sim_backend,
    run_all,
    run_robot,
    run_spec,
)
from ohho.testing.conformance.vendors import (
    UNALLOWABLE,
    format_violation,
    imported_vendor_sdks,
    load_allowlist,
    scan_paths,
    scan_repository,
    violations,
)


class TestClassRegistry(unittest.TestCase):
    def test_unregistered_classes_are_not_a_pass(self):
        for robot_class in (RobotClass.QUADRUPED, RobotClass.AERIAL):
            scenario = CLASS_SCENARIOS[robot_class]
            self.assertIsNone(scenario.sim_backend)
            self.assertIn("not registered", scenario.reason)
            self.assertTrue(scenario.name)

    def test_registered_classes_name_a_sim_backend(self):
        expected = {
            RobotClass.MOBILE_BASE: ("kinematic", "drive_square"),
            RobotClass.MOBILE_MANIPULATOR: ("kinematic", "drive_square"),
            RobotClass.ARM: ("kinematic", "arm_home"),
            RobotClass.HUMANOID: ("humanoid-mujoco", "humanoid_walk"),
        }
        for robot_class, (backend, name) in expected.items():
            scenario = CLASS_SCENARIOS[robot_class]
            self.assertEqual(scenario.sim_backend, backend)
            self.assertEqual(scenario.name, name)
            self.assertEqual(scenario.reason, "")


class TestScenarios(unittest.TestCase):
    def test_sim_and_mobile_manipulator_pass(self):
        for robot_id in ("sim", "omnibot"):
            result = run_robot(robot_id)
            self.assertEqual(result.status, "pass", format_result(result))
            names = {check.name: check.status for check in result.checks}
            self.assertEqual(names["drive_square"], "pass")
            self.assertEqual(names["watchdog_zeroes"], "pass")
            self.assertEqual(names["estop_latches"], "pass")
            self.assertEqual(names["telemetry_schema"], "pass")
            self.assertEqual(names["sim_backend"], "pass")

    def test_quadruped_is_skipped(self):
        result = run_robot("unitree-go2")
        self.assertEqual(result.status, "skipped", format_result(result))
        self.assertIn("not registered", result.reason)
        self.assertNotIn("unitree-go2 (quadruped): pass", format_result(result))
        scenario = next(check for check in result.checks if check.name == "scenario")
        self.assertEqual(scenario.status, "skipped")

    def test_declared_backend_does_not_pass_an_unregistered_class(self):
        spec = replace(get_spec("unitree-go2"), sim_backend="kinematic")
        result = run_spec(spec)
        self.assertEqual(result.status, "skipped", format_result(result))
        self.assertIn("not registered", result.reason)

    def test_arm_scenario_passes_on_the_kinematic_backend(self):
        spec = replace(
            get_spec("omnibot"),
            robot_class=RobotClass.ARM,
            category="arm",
            capabilities=(caps.MANIPULATION,),
            dof_base=0,
            limits=ArmLimits(),
            sim_backend="kinematic",
        )
        result = run_spec(spec)
        self.assertEqual(result.status, "pass", format_result(result))
        names = {check.name: check.status for check in result.checks}
        self.assertEqual(names["move_home"], "pass")
        self.assertEqual(names["joint_limit_clamp"], "pass")
        self.assertEqual(names["estop_torque"], "pass")
        self.assertEqual(get_spec("omnibot").robot_class, RobotClass.MOBILE_MANIPULATOR)

    def test_classes_without_a_robot_are_skipped(self):
        from ohho.registry import list_specs

        # The humanoid class has its own scenario test. Hide those specs so
        # this check still covers a registered class with no robot.
        specs = [
            spec for spec in list_specs() if spec.robot_class != RobotClass.HUMANOID
        ]
        with patch("ohho.testing.conformance.runner.list_specs", return_value=specs):
            results = {result.robot_class: result for result in run_all()}
        for robot_class in ("arm", "aerial", "humanoid"):
            self.assertEqual(results[robot_class].status, "skipped", robot_class)
            self.assertNotEqual(results[robot_class].status, "pass")
        self.assertEqual(results["quadruped"].status, "skipped")
        self.assertEqual(results["mobile_base"].status, "pass")


class TestHardwareRefusal(unittest.TestCase):
    def test_hardware_constructors_are_not_called(self):
        opened: list[str] = []

        def boom(label):
            def _init(self, *args, **kwargs):
                del self, args, kwargs
                opened.append(label)
                raise AssertionError(label)

            return _init

        with (
            patch.object(YahboomTransport, "__init__", boom("yahboom")),
            patch.object(FeetechTransport, "__init__", boom("feetech")),
            patch.object(UnitreeDdsTransport, "__init__", boom("unitree")),
            patch.object(Ros2Transport, "__init__", boom("ros2")),
            patch.object(CompositeTransport, "__init__", boom("composite")),
            patch("ohho.plugins.plugin_transport", boom("plugin_transport")),
        ):
            sim = run_robot("sim")
            arm = run_robot("omnibot")
        self.assertEqual(opened, [], opened)
        self.assertEqual(sim.status, "pass", format_result(sim))
        self.assertEqual(arm.status, "pass", format_result(arm))

    def test_non_sim_protocol_is_refused_before_connect(self):
        connected = {"called": False}

        class _Serial:
            protocol = "serial"

            def connect(self):
                connected["called"] = True
                raise AssertionError("connected hardware")

            def disconnect(self):
                connected["called"] = True

        serial = _Serial()

        class _Info:
            error = None
            name = "fake"
            plugin = None
            sim_factories = {"kinematic": lambda spec, address, runtime: serial}

        real = loaded_plugins

        def mixed():
            return (_Info(),) + tuple(real())

        with patch("ohho.plugins.loaded_plugins", mixed):
            with self.assertRaises(RuntimeError) as caught:
                open_sim_backend(get_spec("sim"), "kinematic")
        self.assertIn("refusing", str(caught.exception))
        self.assertIn("serial", str(caught.exception))
        self.assertFalse(connected["called"])


class TestVendorGrep(unittest.TestCase):
    def test_existing_core_names_are_allowlisted(self):
        bad = violations(scan_repository(), load_allowlist())
        self.assertEqual([format_violation(hit) for hit in bad], [])

    def test_allowlist_rejects_booster_and_t2(self):
        self.assertEqual(UNALLOWABLE, frozenset({"booster", "t2"}))
        for (_path, token), issue in load_allowlist().items():
            self.assertNotIn(token, UNALLOWABLE)
            self.assertIn(issue, {"R-01", "R-02", "R-03"})

    def test_plugin_package_is_not_scanned(self):
        hits = scan_repository()
        self.assertFalse(any(hit.path.startswith("ohho/robots/") for hit in hits))
        self.assertFalse(any(hit.path.startswith("plugins/") for hit in hits))

    def test_planted_booster_outside_a_plugin_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            planted = Path(tmp) / "core_snippet.py"
            planted.write_text('robot = "booster"\n', encoding="utf-8")
            bad = violations(scan_paths([planted]), load_allowlist())
        self.assertEqual(len(bad), 1)
        message = format_violation(bad[0])
        self.assertIn("core_snippet.py", message)
        self.assertRegex(message, r":1: booster")


class TestPluginChecks(unittest.TestCase):
    def test_builtin_plugin_imports_no_vendor_sdk(self):
        import ohho.robots.builtin as builtin

        self.assertEqual(imported_vendor_sdks(lambda: importlib.reload(builtin)), [])

    def test_sentinel_catches_a_vendor_import(self):
        def load() -> None:
            import serial  # noqa: F401

        self.assertIn("serial", imported_vendor_sdks(load))

    def test_mcp_skips_when_absent_and_fails_when_it_can_arm(self):
        skipped = {check.name: check for check in evaluate_mcp(None, None)}
        self.assertEqual(skipped["mcp_cannot_arm"].status, "skipped")
        self.assertEqual(skipped["mcp_backend"].status, "skipped")
        self.assertNotEqual(skipped["mcp_cannot_arm"].status, "pass")

        armed = {check.name: check for check in evaluate_mcp(["arm", "drive"], [])}
        self.assertEqual(armed["mcp_cannot_arm"].status, "fail")
        self.assertIn("arm", armed["mcp_cannot_arm"].reason)

        clean = {
            check.name: check
            for check in evaluate_mcp(
                ["emergency_stop", "drive"],
                [{"backend": "kinematic", "cmd": "stop"}],
            )
        }
        self.assertEqual(clean["mcp_cannot_arm"].status, "pass")
        self.assertEqual(clean["mcp_backend"].status, "pass")

        missing = {
            check.name: check for check in evaluate_mcp(["drive"], [{"cmd": "stop"}])
        }
        self.assertEqual(missing["mcp_backend"].status, "fail")
        self.assertIn("backend", missing["mcp_backend"].reason)


class TestCli(unittest.TestCase):
    def test_unknown_robot_exits_2(self):
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main(["--robot", "missing-robot"]), 2)

    def test_skipped_robot_exits_0(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(["--robot", "unitree-go2"]), 0)

    def test_module_entry_point(self):
        root = Path(__file__).resolve().parents[1]
        proc = subprocess.run(
            [sys.executable, "-m", "ohho.testing.conformance", "--robot", "sim"],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn("sim (mobile_base): pass", proc.stdout)


if __name__ == "__main__":
    unittest.main()
