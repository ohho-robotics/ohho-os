"""CI coverage for the four HIL scenarios and the harness gates.

These tests use fake transports. They must not open a serial port, a DDS
interface, a camera, or a lidar, even if OHHO_HIL is set in the environment.
"""

from __future__ import annotations

import importlib
import io
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ohho import hil
from ohho.adapters.feetech import DEFAULT_JOINT_MAX, DEFAULT_JOINT_MIN

ROOT = Path(__file__).resolve().parents[1]
_DEVICE_ENV = (
    hil.ENV_HIL,
    hil.ENV_BASE,
    hil.ENV_ARM,
    hil.ENV_GO2,
    hil.ENV_BAG,
    hil.ENV_OPERATOR,
)


class _EnvGuard:
    def __enter__(self):
        self._saved = {key: os.environ.get(key) for key in _DEVICE_ENV}
        for key in _DEVICE_ENV:
            os.environ.pop(key, None)
        return self

    def __exit__(self, *_args):
        for key, value in self._saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        return False


class TestHilMotionLock(unittest.TestCase):
    def test_pulse_stays_at_the_bench_values(self):
        self.assertEqual(
            (hil.HIL_BASE_VX, hil.HIL_BASE_VY, hil.HIL_BASE_W), (0.05, 0.0, 0.0)
        )
        self.assertEqual((hil.HIL_GRIPPER_OPEN, hil.HIL_GRIPPER_SHUT), (0.2, 0.0))
        self.assertEqual(
            (hil.HIL_GO2_VX, hil.HIL_GO2_VY, hil.HIL_GO2_W), (0.0, 0.0, 0.0)
        )
        self.assertGreaterEqual(hil.HIL_GRIPPER_OPEN, DEFAULT_JOINT_MIN[5])
        self.assertLessEqual(hil.HIL_GRIPPER_OPEN, DEFAULT_JOINT_MAX[5])


class TestHilTemplate(unittest.TestCase):
    def test_committed_template_is_empty(self):
        text = (ROOT / "docs" / "hil-results.md").read_text(encoding="utf-8")
        text = text.replace("\r\n", "\n")
        self.assertEqual(text, hil.template_text())
        self.assertNotIn("PASS", text)
        self.assertNotIn("FAIL", text)
        for case in hil.CASES:
            self.assertIn(f"| {case.title} | {case.adapter} |  |  |", text)
        for name in hil.PREFLIGHT_CHECKS:
            self.assertIn(f"| {name} |  |  |", text)

    def test_preflight_and_bag_docs(self):
        pre = (ROOT / "docs" / "hil-preflight.md").read_text(encoding="utf-8")
        bag = (ROOT / "docs" / "hil-bag.md").read_text(encoding="utf-8")
        for word in ("Batteries", "Wiring", "E-stop", "Lidar", "Cameras", "Jazzy"):
            self.assertIn(word, pre)
        self.assertIn("--preflight-ok", pre)
        self.assertIn("OHHO_HIL=1", pre)
        self.assertIn("mcap", bag.lower())
        self.assertIn("120", bag)
        self.assertIn("/cmd_vel", bag)
        self.assertIn("/odom", bag)
        self.assertIn("/scan", bag)


class TestHilOptIn(unittest.TestCase):
    def test_skip_reasons(self):
        self.assertIsNotNone(hil.hardware_skip_reason((hil.ENV_BASE,), {}))
        self.assertIn(
            "no default port",
            hil.hardware_skip_reason((hil.ENV_BASE,), {hil.ENV_HIL: "1"}),
        )
        self.assertIsNone(
            hil.hardware_skip_reason(
                (hil.ENV_BASE,),
                {hil.ENV_HIL: "1", hil.ENV_BASE: "/dev/ttyUSB0"},
            )
        )
        both = {hil.ENV_HIL: "1", hil.ENV_BASE: "COM5", hil.ENV_ARM: "COM6"}
        self.assertIsNone(hil.hardware_skip_reason((hil.ENV_BASE, hil.ENV_ARM), both))
        self.assertIsNotNone(hil.hardware_skip_reason((hil.ENV_GO2,), both))
        self.assertIsNone(
            hil.hardware_skip_reason(
                (hil.ENV_GO2,), {hil.ENV_HIL: "1", hil.ENV_GO2: "eth0"}
            )
        )
        self.assertIsNotNone(
            hil.hardware_skip_reason(
                (hil.ENV_BASE,), {hil.ENV_HIL: "1", hil.ENV_BASE: "  "}
            )
        )

    def test_refused_until_every_gate(self):
        empty: dict[str, str] = {}
        self.assertIn(
            "preflight",
            hil.hardware_refused_reason(preflight_ok=False, tty=True, env=empty) or "",
        )
        flagged = {hil.ENV_HIL: "1", hil.ENV_BASE: "COM5"}
        self.assertIn(
            "OHHO_HIL",
            hil.hardware_refused_reason(preflight_ok=True, tty=True, env=empty) or "",
        )
        self.assertIn(
            "explicit",
            hil.hardware_refused_reason(
                preflight_ok=True, tty=True, env={hil.ENV_HIL: "1"}
            )
            or "",
        )
        self.assertIn(
            "interactive",
            hil.hardware_refused_reason(preflight_ok=True, tty=False, env=flagged)
            or "",
        )
        self.assertIsNone(
            hil.hardware_refused_reason(preflight_ok=True, tty=True, env=flagged)
        )

    def test_suite_skips_without_opening_adapters(self):
        for flag in (None, "1"):
            with self.subTest(flag=flag):
                self._assert_skipped(flag)

    def _assert_skipped(self, flag: str | None) -> None:
        with _EnvGuard():
            if flag is not None:
                os.environ[hil.ENV_HIL] = flag
            try:
                import hil.test_hil as hw
            except ImportError:
                import tests.hil.test_hil as hw

            importlib.reload(hw)
            for case in hil.CASES:
                self.assertTrue(hasattr(hw, case.unittest_class), case.unittest_class)

            def boom(self, *args, **kwargs):
                raise AssertionError(f"adapter constructed: {args!r} {kwargs!r}")

            stream = io.StringIO()
            with (
                patch("ohho.adapters.yahboom.YahboomTransport.__init__", boom),
                patch("ohho.adapters.feetech.FeetechTransport.__init__", boom),
                patch("ohho.adapters.unitree.UnitreeDdsTransport.__init__", boom),
            ):
                suite = unittest.defaultTestLoader.loadTestsFromModule(hw)
                result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
            self.assertEqual(len(result.skipped), 4, stream.getvalue())
            self.assertFalse(result.failures, stream.getvalue())
            self.assertFalse(result.errors, stream.getvalue())
            if flag == "1":
                parsed = hil.parse_unittest_verbose(stream.getvalue())
                self.assertEqual(set(parsed), {case.case_id for case in hil.CASES})
                self.assertTrue(
                    all(status == "SKIP" for status, _detail in parsed.values())
                )


class TestHilMockScenarios(unittest.TestCase):
    def test_omnibot_base(self):
        detail = hil.mock_omnibot_base()
        self.assertIn("0.05", detail)
        self.assertIn("odom.vx", detail)

    def test_omnibot_arm(self):
        detail = hil.mock_omnibot_arm()
        self.assertIn("joints=6", detail)

    def test_omnibot_composite(self):
        detail = hil.mock_omnibot_composite()
        self.assertIn("joints=6", detail)
        self.assertIn("odom.vx", detail)

    def test_unitree_go2_zero_velocity(self):
        detail = hil.mock_unitree_go2()
        self.assertIn("Move(0, 0, 0)", detail)
        self.assertIn("StopMove", detail)

    def test_suite_ignores_device_env(self):
        from ohho.adapters.feetech import FeetechTransport
        from ohho.adapters.unitree import UnitreeDdsTransport
        from ohho.adapters.yahboom import YahboomTransport

        opened: list[tuple[str, str]] = []
        orig_y = YahboomTransport.__init__
        orig_f = FeetechTransport.__init__
        orig_u = UnitreeDdsTransport.__init__

        def y_init(self, spec, address="", **kwargs):
            opened.append(("yahboom", str(address)))
            if kwargs.get("serial_factory") is None or not str(address).startswith(
                "mock:"
            ):
                raise AssertionError(f"yahboom address {address}")
            return orig_y(self, spec, address, **kwargs)

        def f_init(self, spec, address="", **kwargs):
            opened.append(("feetech", str(address)))
            if kwargs.get("bus_factory") is None or not str(address).startswith(
                "mock:"
            ):
                raise AssertionError(f"feetech address {address}")
            return orig_f(self, spec, address, **kwargs)

        def u_init(self, spec, address="", **kwargs):
            opened.append(("unitree", str(address)))
            if kwargs.get("state_factory") is None or not str(address).startswith(
                "mock"
            ):
                raise AssertionError(f"unitree address {address}")
            return orig_u(self, spec, address, **kwargs)

        sentinel = "DO-NOT-OPEN-SERIAL"
        with _EnvGuard():
            os.environ[hil.ENV_HIL] = "1"
            os.environ[hil.ENV_BASE] = sentinel
            os.environ[hil.ENV_ARM] = sentinel
            os.environ[hil.ENV_GO2] = sentinel
            with (
                patch.object(YahboomTransport, "__init__", y_init),
                patch.object(FeetechTransport, "__init__", f_init),
                patch.object(UnitreeDdsTransport, "__init__", u_init),
            ):
                results = hil.run_mock_suite()
        self.assertTrue(all(item.status == "PASS" for item in results), results)
        self.assertEqual(len(results), 4)
        self.assertTrue(opened)
        for _kind, address in opened:
            self.assertNotIn(sentinel, address)
            self.assertNotIn("/dev/", address)
            self.assertNotIn("COM", address)


class TestHilRunner(unittest.TestCase):
    def test_default_is_dry_run_and_leaves_the_template(self):
        path = ROOT / "docs" / "hil-results.md"
        before = path.read_bytes()
        code = hil.main([])
        self.assertEqual(code, 0)
        self.assertEqual(path.read_bytes(), before)
        text = hil.dry_run_text()
        self.assertIn("No device was opened", text)
        self.assertIn("serial port", text)
        self.assertIn("--preflight-ok", text)
        self.assertIn("SafetyGate", text)
        self.assertIn("docs/hil-bag.md", text)

    def test_dry_run_wins_over_hardware(self):
        calls: list[object] = []

        def runner(*args):
            calls.append(args)
            raise AssertionError("runner invoked")

        args = hil.build_parser().parse_args(
            ["--dry-run", "--hardware", "--preflight-ok"]
        )
        code = hil.execute(
            args, env={hil.ENV_HIL: "1", hil.ENV_BASE: "COM5"}, tty=True, runner=runner
        )
        self.assertEqual(code, 0)
        self.assertEqual(calls, [])

    def test_mock_and_hardware_together_refused(self):
        code = hil.main(["--mock", "--hardware"])
        self.assertEqual(code, 2)

    def test_mock_writes_results_and_not_the_template(self):
        template = (ROOT / "docs" / "hil-results.md").read_bytes()
        sentinel = "DO-NOT-OPEN-SERIAL"
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "mock-results.md"
            env = os.environ.copy()
            env[hil.ENV_HIL] = "1"
            env[hil.ENV_BASE] = sentinel
            env[hil.ENV_ARM] = sentinel
            env[hil.ENV_GO2] = sentinel
            args = hil.build_parser().parse_args(["--mock", "--out", str(out)])
            code = hil.execute(args, env=env, tty=False, root=ROOT)
            written = out.read_text(encoding="utf-8")
        self.assertEqual(code, 0)
        self.assertEqual((ROOT / "docs" / "hil-results.md").read_bytes(), template)
        self.assertIn("| Mode | mock |", written)
        self.assertNotIn(sentinel, written)
        for case in hil.CASES:
            self.assertIn(f"| {case.title} | {case.adapter} | PASS |", written)
        for name in hil.PREFLIGHT_CHECKS:
            self.assertIn(f"| {name} | not run |", written)

    def test_mock_refuses_to_overwrite_template(self):
        template = (ROOT / "docs" / "hil-results.md").read_bytes()
        args = hil.build_parser().parse_args(["--mock", "--out", "docs/hil-results.md"])
        code = hil.execute(args, env={}, tty=False, root=ROOT)
        self.assertEqual(code, 2)
        self.assertEqual((ROOT / "docs" / "hil-results.md").read_bytes(), template)

    def test_hardware_refused_does_not_spawn(self):
        calls: list[object] = []

        def runner(*args):
            calls.append(args)
            raise AssertionError("runner invoked")

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "should-not-exist.md"
            args = hil.build_parser().parse_args(
                ["--hardware", "--preflight-ok", "--out", str(out)]
            )
            code = hil.execute(
                args,
                env={hil.ENV_HIL: "1", hil.ENV_BASE: "DO-NOT-OPEN-SERIAL"},
                tty=False,
                runner=runner,
                root=ROOT,
            )
            self.assertFalse(out.exists())
        self.assertEqual(code, 2)
        self.assertEqual(calls, [])

    def test_hardware_fake_runner_records_pass(self):
        template = (ROOT / "docs" / "hil-results.md").read_bytes()
        log = "\n".join(
            [
                "test_drive_and_read (tests.hil.test_hil.TestHilOmniBotBase.test_drive_and_read) ... ok",
                "test_move_joints_and_read (tests.hil.test_hil.TestHilOmniBotArm.test_move_joints_and_read) ... ok",
                "test_composite_drive_and_arm (tests.hil.test_hil.TestHilOmniBotComposite.test_composite_drive_and_arm) ... ok",
                "test_drive_and_read (tests.hil.test_hil.TestHilUnitreeGo2.test_drive_and_read) ... ok",
            ]
        )
        seen: dict[str, object] = {}

        def runner(cmd, env, cwd):
            seen["cmd"] = cmd
            seen["env"] = env
            seen["cwd"] = cwd
            return subprocess.CompletedProcess(cmd, 0, stdout=log, stderr="")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = hil.build_parser().parse_args(["--hardware", "--preflight-ok"])
            env = {
                hil.ENV_HIL: "1",
                hil.ENV_BASE: "COM5",
                hil.ENV_ARM: "COM6",
                hil.ENV_GO2: "eth0",
                hil.ENV_BAG: str(root / "bag"),
                "PATH": os.environ.get("PATH", ""),
            }
            code = hil.execute(args, env=env, tty=True, runner=runner, root=root)
            written = (root / "docs" / "hil-results.md").read_text(encoding="utf-8")
        self.assertEqual(code, 0)
        self.assertEqual((ROOT / "docs" / "hil-results.md").read_bytes(), template)
        self.assertEqual(seen["cmd"], hil.hardware_unittest_cmd(root))
        self.assertIn("tests", seen["cmd"][seen["cmd"].index("-s") + 1])
        self.assertTrue(str(seen["cmd"][seen["cmd"].index("-s") + 1]).endswith("hil"))
        self.assertEqual(seen["env"][hil.ENV_BASE], "COM5")
        self.assertIn("| Mode | hardware |", written)
        self.assertIn("| Bag (2-min MCAP) |", written)
        self.assertIn("COM5", written)
        for case in hil.CASES:
            self.assertIn(f"| {case.title} | {case.adapter} | PASS |", written)

    def test_hardware_failure_and_all_skip_exit_codes(self):
        fail_log = (
            "test_drive_and_read (tests.hil.test_hil.TestHilOmniBotBase.test_drive_and_read) ... FAIL\n"
            "\n======================================================================\n"
            "FAIL: test_drive_and_read (tests.hil.test_hil.TestHilOmniBotBase.test_drive_and_read)\n"
            "----------------------------------------------------------------------\n"
            "AssertionError: odom missing\n"
        )
        skip_log = (
            "test_drive_and_read (tests.hil.test_hil.TestHilOmniBotBase.test_drive_and_read) "
            "... skipped 'missing bench'\n"
        )

        def fail_runner(cmd, env, cwd):
            return subprocess.CompletedProcess(cmd, 1, stdout=fail_log, stderr="")

        def skip_runner(cmd, env, cwd):
            return subprocess.CompletedProcess(cmd, 0, stdout=skip_log, stderr="")

        env = {
            hil.ENV_HIL: "1",
            hil.ENV_BASE: "COM5",
            "PATH": os.environ.get("PATH", ""),
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = hil.build_parser().parse_args(["--hardware", "--preflight-ok"])
            code = hil.execute(args, env=env, tty=True, runner=fail_runner, root=root)
            text = (root / "docs" / "hil-results.md").read_text(encoding="utf-8")
        self.assertEqual(code, 1)
        self.assertIn("AssertionError: odom missing", text)
        self.assertIn("| OmniBot base drive and read | yahboom | FAIL |", text)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            code = hil.execute(args, env=env, tty=True, runner=skip_runner, root=root)
        self.assertEqual(code, 3)

    def test_script_mock_does_not_touch_the_template(self):
        script = ROOT / "scripts" / "hil_run.py"
        template = (ROOT / "docs" / "hil-results.md").read_bytes()
        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        for key in _DEVICE_ENV:
            env.pop(key, None)
        env[hil.ENV_HIL] = "1"
        env[hil.ENV_BASE] = "DO-NOT-OPEN-SERIAL"
        env[hil.ENV_ARM] = "DO-NOT-OPEN-SERIAL"
        env[hil.ENV_GO2] = "DO-NOT-OPEN-SERIAL"
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "from-script.md"
            mock = subprocess.run(
                [sys.executable, str(script), "--mock", "--out", str(out)],
                cwd=ROOT,
                env=env,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(mock.returncode, 0, mock.stderr)
            written = out.read_text(encoding="utf-8")
        self.assertNotIn("DO-NOT-OPEN-SERIAL", written)
        self.assertIn("| Mode | mock |", written)
        self.assertEqual((ROOT / "docs" / "hil-results.md").read_bytes(), template)

    def test_runner_command_skips_without_opt_in(self):
        env = os.environ.copy()
        for key in _DEVICE_ENV:
            env.pop(key, None)
        proc = subprocess.run(
            hil.hardware_unittest_cmd(ROOT),
            cwd=ROOT,
            env=env,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            check=False,
        )
        combined = proc.stdout + "\n" + proc.stderr
        self.assertEqual(proc.returncode, 0, combined)
        self.assertIn("skipped=4", combined)
        parsed = hil.parse_unittest_verbose(combined)
        self.assertEqual(set(parsed), {case.case_id for case in hil.CASES})
        self.assertTrue(all(status == "SKIP" for status, _detail in parsed.values()))


if __name__ == "__main__":
    unittest.main()
