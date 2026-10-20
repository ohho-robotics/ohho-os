"""Read-only doctor hook and ``ohho doctor --robot`` dispatch. No robot, no SDK wheel."""

from __future__ import annotations

import io
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from ohho.cli import main

try:
    from test_booster_transport import FakeSdk
    from test_ohho_booster import (
        _Entry,
        _builtin_entry,
        _discovered,
        _ensure_plugin_path,
    )
except ImportError:  # python -m unittest tests.test_booster_doctor
    from tests.test_booster_transport import FakeSdk  # type: ignore[no-redef]
    from tests.test_ohho_booster import (  # type: ignore[no-redef]
        _Entry,
        _builtin_entry,
        _discovered,
        _ensure_plugin_path,
    )

_ensure_plugin_path()

from ohho_booster.doctor import (  # noqa: E402
    ProbeResult,
    Unreachable,
    diagnose,
    firmware_ok,
)
from ohho_booster.messages import (  # noqa: E402
    TOPIC_JOINT_CTRL,
    TOPIC_LOCO_API,
    CommandRefused,
)
from ohho_booster.plugin import BoosterPlugin, plugin as plugin_factory  # noqa: E402
from ohho_booster.sdk import (  # noqa: E402
    STATE_CHANGING_RPCS,
    SdkSession,
    SdkUnavailable,
)
from ohho_booster.spec import booster_t2_spec  # noqa: E402

_URI = "booster://192.0.2.10"


def _rpc_names(calls: list[object]) -> set[object]:
    names: set[object] = set()
    for item in calls:
        names.add(item[0] if isinstance(item, tuple) else item)
    return names


def _write_profile(home: Path) -> None:
    path = home / ".ohho" / "robots" / "booster-t2.toml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                'transport = "booster://192.0.2.10"',
                'network_interface = "eth0"',
                "",
            ]
        ),
        encoding="utf-8",
    )
    (home / ".ohho" / "config.toml").write_text('robot = "sim"\n', encoding="utf-8")


class FirmwareCase(unittest.TestCase):
    def test_minimum_is_inclusive_and_newer_builds_pass(self) -> None:
        self.assertTrue(firmware_ok("1.7.2"))
        self.assertTrue(firmware_ok("v1.7.2"))
        self.assertTrue(firmware_ok("1.7.2-rc1"))
        self.assertTrue(firmware_ok("1.8.3"))
        self.assertTrue(firmware_ok("1.10.0"))
        self.assertFalse(firmware_ok("1.7.1"))
        self.assertFalse(firmware_ok("1.7"))
        self.assertFalse(firmware_ok("1.6.9"))
        self.assertFalse(firmware_ok(None))
        self.assertFalse(firmware_ok(""))


class DiagnoseCase(unittest.TestCase):
    def test_sdk_missing_does_not_probe(self) -> None:
        def loader() -> object:
            raise SdkUnavailable("wheel missing")

        def probe(_address: str) -> ProbeResult:
            raise AssertionError("probe ran without an SDK")

        report = diagnose(
            _URI,
            network_interface="eth0",
            loader=loader,
            probe=probe,
        )
        self.assertFalse(report.ok)
        self.assertIn("SDK missing", report.summary)
        self.assertIn("sdk: missing", report.details)

    def test_unreachable_unknown_model_and_old_firmware(self) -> None:
        def loader() -> object:
            return object()

        cases = (
            (
                "unreachable",
                lambda _address: (_ for _ in ()).throw(Unreachable("timed out")),
                "unreachable",
            ),
            (
                "model",
                lambda _address: ProbeResult("K1", "1.8.3", "eth0"),
                "unknown model",
            ),
            (
                "firmware",
                lambda _address: ProbeResult("T2", "1.7.1", "eth0"),
                "firmware too old",
            ),
        )
        for name, probe, phrase in cases:
            with self.subTest(name=name):
                report = diagnose(
                    _URI,
                    network_interface="eth0",
                    loader=loader,
                    probe=probe,  # type: ignore[arg-type]
                )
                self.assertFalse(report.ok)
                self.assertIn(phrase, report.summary)

    def test_current_and_newer_firmware_pass(self) -> None:
        def loader() -> object:
            return object()

        for version in ("1.7.2", "1.8.3", "1.10.0"):
            with self.subTest(version=version):
                report = diagnose(
                    _URI,
                    network_interface="eth0",
                    loader=loader,
                    probe=lambda _address, version=version: ProbeResult(
                        "T2", version, "eth0"
                    ),
                )
                self.assertTrue(report.ok, report.summary)
                self.assertIn(f"firmware: {version}", report.details)
                self.assertIn("network interface: eth0", report.details)
                self.assertIn("model: T2", report.details)

    def test_default_probe_is_read_only(self) -> None:
        sdk = FakeSdk(model="T2", mode=1)
        sdk.firmware = "1.8.3"

        def opener(
            address: str, sdk_module: object = None, *, read_only: bool = False
        ) -> SdkSession:
            del sdk_module
            self.assertTrue(read_only)
            self.assertNotIn("://", address)
            return SdkSession(address, 0, sdk, read_only=read_only)

        with (
            patch("ohho_booster.doctor._default_loader", lambda: sdk),
            patch("ohho_booster.sdk.open_session", opener),
        ):
            report = diagnose(_URI, network_interface="eth0")
        self.assertTrue(report.ok, report.summary)
        self.assertEqual(sdk.publishers, [])
        self.assertNotIn(TOPIC_JOINT_CTRL, repr(sdk.events))
        self.assertNotIn(TOPIC_LOCO_API, repr(sdk.events))
        assert sdk.client is not None
        self.assertTrue(STATE_CHANGING_RPCS.isdisjoint(_rpc_names(sdk.client.calls)))

        old = FakeSdk(model="T2", mode=1)
        old.firmware = "1.7.1"
        unknown = FakeSdk(model="K1", mode=1)

        def opener_for(
            target: FakeSdk,
        ):
            def _open(
                address: str, sdk_module: object = None, *, read_only: bool = False
            ) -> SdkSession:
                del sdk_module
                self.assertTrue(read_only)
                return SdkSession(address, 0, target, read_only=read_only)

            return _open

        with (
            patch("ohho_booster.doctor._default_loader", lambda: old),
            patch("ohho_booster.sdk.open_session", opener_for(old)),
        ):
            report = diagnose(_URI, network_interface="eth0")
        self.assertFalse(report.ok)
        self.assertIn("firmware too old", report.summary)
        self.assertEqual(old.publishers, [])

        with (
            patch("ohho_booster.doctor._default_loader", lambda: unknown),
            patch("ohho_booster.sdk.open_session", opener_for(unknown)),
        ):
            report = diagnose(_URI, network_interface="eth0")
        self.assertFalse(report.ok)
        self.assertIn("unknown model", report.summary)
        self.assertEqual(unknown.publishers, [])
        assert unknown.client is not None
        self.assertTrue(
            STATE_CHANGING_RPCS.isdisjoint(_rpc_names(unknown.client.calls))
        )

        down = FakeSdk(model="T2", mode=1)
        down.firmware = "1.8.3"

        def refuse(address: str, sdk_module: object = None, *, read_only: bool = False):
            del address, sdk_module, read_only
            raise TimeoutError("no reply")

        with (
            patch("ohho_booster.doctor._default_loader", lambda: down),
            patch("ohho_booster.sdk.open_session", refuse),
        ):
            report = diagnose(_URI, network_interface="eth0")
        self.assertFalse(report.ok)
        self.assertIn("unreachable", report.summary)
        self.assertEqual(down.publishers, [])


class DoctorCliCase(unittest.TestCase):
    def _run(self, loader, probe) -> tuple[int, str, dict[str, object]]:
        seen: dict[str, object] = {}
        real = BoosterPlugin.doctor

        def wrapped(self, spec, uri=None, read_only=True):  # type: ignore[no-untyped-def]
            seen["id"] = spec.id
            seen["uri"] = uri
            seen["read_only"] = read_only
            return real(self, spec, uri, read_only)

        entry = _Entry("booster", "ohho_booster:plugin", plugin_factory)
        out = io.StringIO()
        err = io.StringIO()
        with (
            patch.object(BoosterPlugin, "doctor", wrapped),
            patch("ohho_booster.doctor._default_loader", loader),
            patch("ohho_booster.doctor._default_probe", probe),
            redirect_stdout(out),
            redirect_stderr(err),
        ):
            with _discovered(lambda: [_builtin_entry(), entry]):
                code = main(["doctor", "--robot", "booster-t2"])
        return code, out.getvalue(), seen

    def test_doctor_routes_through_the_resolver(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            _write_profile(home)
            env = {"OHHO_HOME": tmp, "OHHO_ROBOT": "omnibot"}
            with patch.dict(os.environ, env):
                code, text, seen = self._run(
                    lambda: object(),
                    lambda _address: ProbeResult("T2", "1.8.3", "eth0"),
                )
        self.assertEqual(code, 0, text)
        self.assertEqual(seen["id"], "booster-t2")
        self.assertEqual(seen["uri"], _URI)
        self.assertIs(seen["read_only"], True)
        self.assertIn("booster-t2 (flag)", text)
        self.assertIn("plugin doctor: ok", text)
        self.assertIn("network interface: eth0", text)
        self.assertNotIn("omnibot (flag)", text)

    def test_doctor_reports_each_failure(self) -> None:
        failures = (
            (
                "sdk",
                lambda: (_ for _ in ()).throw(SdkUnavailable("wheel missing")),
                lambda _address: (_ for _ in ()).throw(AssertionError("probed")),
                "SDK missing",
            ),
            (
                "reach",
                lambda: object(),
                lambda _address: (_ for _ in ()).throw(Unreachable("timed out")),
                "unreachable",
            ),
            (
                "model",
                lambda: object(),
                lambda _address: ProbeResult("K1", "1.8.3", "eth0"),
                "unknown model",
            ),
            (
                "firmware",
                lambda: object(),
                lambda _address: ProbeResult("T2", "1.7.1", "eth0"),
                "firmware too old",
            ),
        )
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            _write_profile(home)
            env = {"OHHO_HOME": tmp, "OHHO_ROBOT": "omnibot"}
            with patch.dict(os.environ, env):
                for name, loader, probe, phrase in failures:
                    with self.subTest(name=name):
                        code, text, seen = self._run(loader, probe)
                        self.assertEqual(code, 1, text)
                        self.assertIn(phrase, text)
                        self.assertEqual(seen["id"], "booster-t2")
                        self.assertIs(seen["read_only"], True)
                        self.assertIn("booster-t2 (flag)", text)


class ConnectCliCase(unittest.TestCase):
    def test_read_only_connect_uses_the_flag_and_prints_the_block(self) -> None:
        block = {
            "mode": "PREPARE",
            "fall_state": "upright",
            "joints": {"head": {"q": [0.1, -0.2]}},
            "battery": {"fraction": 0.5, "voltage": 48.2},
        }
        telemetry = SimpleNamespace(
            odom=None,
            battery=0.5,
            custom={"humanoid": block},
        )
        fake = Mock()
        fake.gated = False
        fake.telemetry.return_value = telemetry
        fake.transport.on_telemetry.return_value = lambda: None
        state = Mock()
        state.value = "connected"
        fake.status.return_value = Mock(state=state, label="link")
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"OHHO_HOME": tmp, "OHHO_ROBOT": "sim"}):
                with (
                    patch("ohho.cli.Robot.connect", return_value=fake) as conn,
                    patch("ohho.cli.time.sleep"),
                    patch("ohho.cli._stdio_is_tty", return_value=False),
                    redirect_stdout(io.StringIO()) as out,
                    redirect_stderr(io.StringIO()),
                ):
                    code = main(
                        [
                            "connect",
                            "--robot",
                            "booster-t2",
                            "--read-only",
                            "booster://192.0.2.10",
                        ]
                    )
        self.assertEqual(code, 0)
        conn.assert_called_once_with(
            "booster-t2",
            transport="booster://192.0.2.10",
            runtime="auto",
            safety_config=None,
            estop_damp=False,
            read_only=True,
        )
        text = out.getvalue()
        self.assertIn("publishing nothing", text)
        self.assertIn("mode   : PREPARE", text)
        self.assertIn("fall   : upright", text)
        self.assertIn("joints.head", text)
        self.assertIn("fraction=0.5", text)
        self.assertIn("voltage=48.2", text)

    def test_plugin_doctor_stays_read_only_when_asked_to_write(self) -> None:
        sdk = FakeSdk(model="T2", mode=1)
        sdk.firmware = "1.7.2"
        spec = booster_t2_spec()

        def opener(
            address: str, sdk_module: object = None, *, read_only: bool = False
        ) -> SdkSession:
            del sdk_module
            self.assertTrue(read_only)
            return SdkSession(address, 0, sdk, read_only=read_only)

        with tempfile.TemporaryDirectory() as tmp:
            _write_profile(Path(tmp))
            with patch.dict(os.environ, {"OHHO_HOME": tmp}):
                with (
                    patch("ohho_booster.doctor._default_loader", lambda: sdk),
                    patch("ohho_booster.sdk.open_session", opener),
                ):
                    report = BoosterPlugin().doctor(spec, None, False)
        self.assertTrue(report.ok, report.summary)
        with self.assertRaises(CommandRefused):
            SdkSession("192.0.2.10", 0, sdk, read_only=True).move(1.0, 0.0, 0.0)
        self.assertEqual(sdk.publishers, [])
        assert sdk.client is not None
        self.assertTrue(STATE_CHANGING_RPCS.isdisjoint(_rpc_names(sdk.client.calls)))


if __name__ == "__main__":
    unittest.main()
