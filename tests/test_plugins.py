"""RobotPlugin discovery via the ``ohho.robots`` entry-point group."""

from __future__ import annotations

import importlib.metadata
import io
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from contextlib import contextmanager, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from ohho.adapters.sim import SimTransport
from ohho.cli import main
from ohho.plugins import (
    BUILTIN_ENTRY_VALUE,
    ENTRY_GROUP,
    format_plugin_report,
    loaded_plugins,
)
from ohho.registry import RobotSpec, UnknownRobot, get_spec, list_specs
from ohho.safety import profile_for

_ROOT = Path(__file__).resolve().parents[1]


class _MemEP:
    """Entry point whose ``load`` returns an in-memory object."""

    def __init__(self, name: str, target: object, value: str = "mem:plugin"):
        self.name = name
        self.value = value
        self.group = ENTRY_GROUP
        self.dist = None
        self._target = target

    def load(self) -> object:
        return self._target


def _plugin_class(name, specs, schemes=None, backends=None, calls=None):
    class Plugin:
        def specs(self):
            return list(specs)

        def transports(self):
            return dict(schemes or {})

        def sim_backends(self):
            return dict(backends or {})

        def safety_policy(self, spec):
            if calls is not None:
                calls.append("safety_policy")
            return None

        def descriptor(self, spec):
            if calls is not None:
                calls.append("descriptor")
            return None

        def doctor(self, spec, uri=None, read_only=True):
            if calls is not None:
                calls.append("doctor")
            return None

    Plugin.name = name
    return Plugin


def _builtin_entry():
    return importlib.metadata.EntryPoint(
        name="builtin",
        value=BUILTIN_ENTRY_VALUE,
        group=ENTRY_GROUP,
    )


@contextmanager
def _discovered(reader):
    """Reload plugins from ``reader`` and restore the previous registry."""
    import ohho.plugins as plugins
    import ohho.registry as registry

    saved = (
        dict(registry._REGISTRY),
        registry._plugins_loaded,
        registry._plugins_loading,
        plugins._CACHE,
        plugins._in_load,
    )
    try:
        with patch.object(plugins, "_read_entry_points", reader):
            plugins.reload_plugins()
            yield
    finally:
        reg, loaded, loading, cache, in_load = saved
        plugins._CACHE = cache
        plugins._in_load = in_load
        registry._REGISTRY.clear()
        registry._REGISTRY.update(reg)
        registry._plugins_loaded = loaded
        registry._plugins_loading = loading


@contextmanager
def _on_path(module: str, source: str):
    """Expose ``source`` as importable ``module`` without leaving it imported."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        Path(tmp, f"{module}.py").write_text(source, encoding="utf-8")
        sys.path.insert(0, tmp)
        try:
            yield
        finally:
            try:
                sys.path.remove(tmp)
            except ValueError:
                pass
            for key in list(sys.modules):
                if key == module or key.startswith(module + "."):
                    sys.modules.pop(key, None)


def _stdout(argv: list[str]) -> tuple[int, str]:
    buf = io.StringIO()
    with redirect_stdout(buf):
        code = main(argv)
    return code, buf.getvalue()


class TestPluginDiscovery(unittest.TestCase):
    def test_pyproject_registers_builtin_entry_point(self):
        text = (_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        self.assertIn('[project.entry-points."ohho.robots"]', text)
        self.assertIn(f'builtin = "{BUILTIN_ENTRY_VALUE}"', text)
        self.assertIn('omnibot = "ohho.robots.omnibot:plugin"', text)

    def test_entry_point_object_loads_builtin_specs(self):
        plugin = _builtin_entry().load()()
        self.assertEqual(plugin.name, "builtin")
        self.assertEqual(
            {spec.id for spec in plugin.specs()},
            {
                "sim",
                "unitree-go2",
                "px4-sitl",
                "crazyflie-sim",
                "px4-x500",
            },
        )
        for spec in plugin.specs():
            self.assertIsNone(plugin.safety_policy(spec))
            self.assertIsNone(plugin.descriptor(spec))
            self.assertIsNone(plugin.doctor(spec, "sim://", read_only=True))

    def test_source_checkout_falls_back_when_group_is_empty(self):
        def reader():
            return []

        with _discovered(reader):
            infos = loaded_plugins()
            builtin = [info for info in infos if info.name == "builtin"]
            self.assertEqual(len(builtin), 1)
            self.assertTrue(builtin[0].fallback)
            self.assertIsNone(builtin[0].error)
            self.assertEqual(
                set(builtin[0].spec_ids),
                {
                    "sim",
                    "unitree-go2",
                    "px4-sitl",
                    "crazyflie-sim",
                    "px4-x500",
                },
            )
            mobile = next(info for info in infos if info.name == "omnibot")
            self.assertTrue(mobile.fallback)
            self.assertEqual(mobile.spec_ids, ("omnibot",))
            self.assertEqual(get_spec("omnibot").name, "OmniBot")
            self.assertEqual(get_spec("omnibot").source, "omnibot")
            self.assertEqual(get_spec("unitree-go2").max_lin, 1.5)
            code, text = _stdout(["list", "--plugins"])
            self.assertEqual(code, 0)
            self.assertIn("builtin (source fallback):", text)
            self.assertIn("omnibot (source fallback):", text)
            self.assertNotIn("error:", text)
            code, listed = _stdout(["list"])
            self.assertEqual(code, 0)
            self.assertIn("OmniBot", listed)
            self.assertIn("Unitree Go2", listed)
            self.assertNotIn("source fallback", listed)

    def test_installed_entry_point_skips_fallback(self):
        def reader():
            return [
                _builtin_entry(),
                importlib.metadata.EntryPoint(
                    name="omnibot",
                    value="ohho.robots.omnibot:plugin",
                    group=ENTRY_GROUP,
                ),
            ]

        with _discovered(reader):
            infos = [info for info in loaded_plugins() if info.name == "builtin"]
            self.assertEqual(len(infos), 1)
            self.assertFalse(infos[0].fallback)
            self.assertIsNone(infos[0].error)
            self.assertEqual(
                set(infos[0].spec_ids),
                {
                    "sim",
                    "unitree-go2",
                    "px4-sitl",
                    "crazyflie-sim",
                    "px4-x500",
                },
            )
            mobile = next(info for info in loaded_plugins() if info.name == "omnibot")
            self.assertFalse(mobile.fallback)
            self.assertEqual(mobile.spec_ids, ("omnibot",))
            self.assertNotIn("source fallback", format_plugin_report())
            self.assertIn("sim", {spec.id for spec in list_specs()})
            self.assertIn("omnibot", {spec.id for spec in list_specs()})

    def test_builtin_declares_current_schemes(self):
        def reader():
            return [_builtin_entry()]

        with _discovered(reader):
            info = next(item for item in loaded_plugins() if item.name == "builtin")
            self.assertEqual(
                set(info.schemes),
                {
                    "sim",
                    "simulated",
                    "feetech",
                    "dds",
                    "unitree-dds",
                    "ros2",
                },
            )
            self.assertEqual(info.sim_backends, ("kinematic",))

    def test_external_plugin_is_listed_and_core_does_not_import_it(self):
        mod = "ohho_c02_external_plugin"
        self.assertNotIn(mod, sys.modules)
        import ohho
        import ohho.adapters
        import ohho.cli
        import ohho.registry
        import ohho.robot

        self.assertNotIn(mod, sys.modules)
        self.assertIsNotNone(ohho.robot)
        self.assertIsNotNone(ohho.adapters)
        self.assertIsNotNone(ohho.cli)
        self.assertIsNotNone(ohho.registry)

        source = textwrap.dedent(
            """\
            from ohho.registry import RobotSpec

            class External:
                name = "external"

                def specs(self):
                    return [
                        RobotSpec(
                            id="external-bot",
                            name="External Bot",
                            category="wheeled",
                            capabilities=("base.drive",),
                            adapter="sim",
                            source="external",
                        )
                    ]

                def transports(self):
                    def factory(spec, address, runtime):
                        raise AssertionError("factory ran during discovery")

                    return {"external": factory}

                def sim_backends(self):
                    return {}

                def safety_policy(self, spec):
                    return None

                def descriptor(self, spec):
                    return None

                def doctor(self, spec, uri=None, read_only=True):
                    return None

            def plugin():
                return External()
            """
        )
        with _on_path(mod, source):
            self.assertNotIn(mod, sys.modules)

            def reader():
                return [
                    _builtin_entry(),
                    importlib.metadata.EntryPoint(
                        name="external",
                        value=f"{mod}:plugin",
                        group=ENTRY_GROUP,
                    ),
                ]

            with _discovered(reader):
                self.assertIn(mod, sys.modules)
                self.assertEqual(get_spec("external-bot").source, "external")
                self.assertEqual(get_spec("omnibot").dof_arm, 6)
                code, plugins_text = _stdout(["list", "--plugins"])
                self.assertEqual(code, 0)
                self.assertIn("external: external-bot", plugins_text)
                self.assertNotIn("error:", plugins_text)
                code, listed = _stdout(["list"])
                self.assertEqual(code, 0)
                self.assertIn("external-bot", listed)
                self.assertIn("omnibot", listed)

    def test_import_error_is_listed_and_other_robots_work(self):
        mod = "ohho_c02_bad_plugin"
        with _on_path(mod, 'raise ImportError("vendor sdk missing")\n'):

            def reader():
                return [
                    importlib.metadata.EntryPoint(
                        name="badvendor",
                        value=f"{mod}:plugin",
                        group=ENTRY_GROUP,
                    )
                ]

            with _discovered(reader):
                infos = {info.name: info for info in loaded_plugins()}
                self.assertIn("badvendor", infos)
                error = infos["badvendor"].error or ""
                self.assertIn("vendor sdk missing", error)
                self.assertIn("ImportError", error)
                self.assertIsNone(infos["builtin"].error)
                self.assertEqual(get_spec("sim").id, "sim")
                self.assertEqual(get_spec("unitree-go2").id, "unitree-go2")
                self.assertEqual(get_spec("omnibot").name, "OmniBot")
                code, text = _stdout(["list", "--plugins"])
                self.assertEqual(code, 0)
                self.assertIn("badvendor", text)
                self.assertIn("vendor sdk missing", text)
                code, listed = _stdout(["list"])
                self.assertEqual(code, 0)
                self.assertIn("omnibot", listed)
                self.assertIn("unitree-go2", listed)
                with self.assertRaises(UnknownRobot):
                    get_spec("badvendor-bot")

    def test_specs_exception_does_not_drop_builtins(self):
        boom = _plugin_class("boom", specs=[])

        def specs(self):
            raise RuntimeError("spec build failed")

        boom.specs = specs

        def reader():
            return [_MemEP("boom", boom)]

        with _discovered(reader):
            info = next(item for item in loaded_plugins() if item.name == "boom")
            self.assertIn("spec build failed", info.error or "")
            self.assertIn("RuntimeError", info.error or "")
            self.assertEqual(get_spec("sim").category, "wheeled")

    def test_duplicate_robot_id_names_both_plugins(self):
        acme = _plugin_class(
            "acme",
            specs=[
                RobotSpec(
                    id="omnibot",
                    name="Impostor",
                    category="wheeled",
                    source="acme",
                ),
                RobotSpec(
                    id="acme-bot",
                    name="Acme",
                    category="wheeled",
                    source="acme",
                ),
            ],
        )

        def reader():
            return [_MemEP("acme", acme)]

        with _discovered(reader):
            info = next(item for item in loaded_plugins() if item.name == "acme")
            self.assertIn("duplicate robot id 'omnibot'", info.error or "")
            self.assertIn("plugins 'omnibot' and 'acme'", info.error or "")
            self.assertEqual(get_spec("omnibot").name, "OmniBot")
            with self.assertRaises(UnknownRobot):
                get_spec("acme-bot")
            text = format_plugin_report()
            self.assertIn("acme: omnibot, acme-bot", text)
            self.assertIn("error:", text)

    def test_duplicate_scheme_names_both_plugins(self):
        def factory(spec, address, runtime):
            raise AssertionError("scheme factory ran during discovery")

        other = _plugin_class(
            "schemeplug",
            specs=[
                RobotSpec(
                    id="scheme-bot",
                    name="Scheme",
                    category="wheeled",
                    source="schemeplug",
                )
            ],
            schemes={"DDS": factory},
        )

        def reader():
            return [_MemEP("schemeplug", other)]

        with _discovered(reader):
            info = next(item for item in loaded_plugins() if item.name == "schemeplug")
            self.assertIn("duplicate transport scheme 'dds'", info.error or "")
            self.assertIn("'builtin'", info.error or "")
            self.assertIn("'schemeplug'", info.error or "")
            with self.assertRaises(UnknownRobot):
                get_spec("scheme-bot")
            self.assertEqual(get_spec("unitree-go2").adapter, "unitree-dds")

    def test_duplicate_between_third_parties_keeps_the_first(self):
        def factory(spec, address, runtime):
            return None

        alpha = _plugin_class(
            "alpha",
            specs=[
                RobotSpec(
                    id="alpha-bot",
                    name="Alpha",
                    category="wheeled",
                    source="alpha",
                )
            ],
            schemes={"widget": factory},
        )
        beta = _plugin_class(
            "beta",
            specs=[
                RobotSpec(
                    id="beta-bot",
                    name="Beta",
                    category="wheeled",
                    source="beta",
                )
            ],
            schemes={"widget": factory},
        )

        def reader():
            return [_builtin_entry(), _MemEP("beta", beta), _MemEP("alpha", alpha)]

        with _discovered(reader):
            names = [info.name for info in loaded_plugins()]
            self.assertEqual(names[0], "builtin")
            self.assertLess(names.index("alpha"), names.index("beta"))
            self.assertEqual(get_spec("alpha-bot").name, "Alpha")
            with self.assertRaises(UnknownRobot):
                get_spec("beta-bot")
            beta_info = next(item for item in loaded_plugins() if item.name == "beta")
            self.assertIn("'alpha'", beta_info.error or "")
            self.assertIn("'beta'", beta_info.error or "")
            self.assertIn("widget", beta_info.error or "")

    def test_plugin_that_repeats_an_id_is_rejected(self):
        twice = _plugin_class(
            "twice",
            specs=[
                RobotSpec(id="twice-bot", name="A", category="wheeled", source="twice"),
                RobotSpec(id="twice-bot", name="B", category="wheeled", source="twice"),
            ],
        )

        def reader():
            return [_MemEP("twice", twice)]

        with _discovered(reader):
            info = next(item for item in loaded_plugins() if item.name == "twice")
            self.assertIn("declared twice", info.error or "")
            self.assertIn("'twice'", info.error or "")
            with self.assertRaises(UnknownRobot):
                get_spec("twice-bot")

    def test_empty_source_is_the_plugin_name(self):
        plain = _plugin_class(
            "plain",
            specs=[RobotSpec(id="plain-bot", name="Plain", category="wheeled")],
        )

        def reader():
            return [_MemEP("plain", plain)]

        with _discovered(reader):
            self.assertEqual(get_spec("plain-bot").source, "plain")

    def test_discovery_does_not_call_safety_descriptor_or_doctor(self):
        calls: list[str] = []
        quiet = _plugin_class(
            "quiet",
            specs=[
                RobotSpec(
                    id="quiet-bot", name="Quiet", category="wheeled", source="quiet"
                )
            ],
            calls=calls,
        )

        def reader():
            return [_MemEP("quiet", quiet)]

        with _discovered(reader):
            _stdout(["list", "--plugins"])
            self.assertEqual(calls, [])
            self.assertEqual(get_spec("quiet-bot").id, "quiet-bot")

    def test_incomplete_plugin_is_an_error(self):
        class Incomplete:
            name = "incomplete"

            def specs(self):
                return [
                    RobotSpec(
                        id="incomplete-bot",
                        name="Incomplete",
                        category="wheeled",
                        source="incomplete",
                    )
                ]

            def transports(self):
                return {}

            def sim_backends(self):
                return {}

        def reader():
            return [_MemEP("incomplete", Incomplete)]

        with _discovered(reader):
            info = next(item for item in loaded_plugins() if item.name == "incomplete")
            self.assertIn("doctor", info.error or "")
            with self.assertRaises(UnknownRobot):
                get_spec("incomplete-bot")
            self.assertEqual(get_spec("sim").id, "sim")

    def test_broken_builtin_entry_point_is_not_masked_by_fallback(self):
        class BrokenBuiltin:
            name = "builtin"
            value = BUILTIN_ENTRY_VALUE
            group = ENTRY_GROUP
            dist = None

            def load(self):
                raise ImportError("broken builtin entry point")

        def reader():
            return [BrokenBuiltin()]

        with _discovered(reader):
            infos = loaded_plugins()
            self.assertEqual(infos[0].name, "builtin")
            self.assertFalse(infos[0].fallback)
            self.assertIn("broken builtin entry point", infos[0].error or "")
            mobile = next(info for info in infos if info.name == "omnibot")
            self.assertTrue(mobile.fallback)
            self.assertIsNone(mobile.error)
            with self.assertRaises(UnknownRobot):
                get_spec("sim")
            self.assertEqual(get_spec("omnibot").name, "OmniBot")

    def test_resolve_transport_and_safety_profiles_unchanged(self):
        from ohho.adapters import resolve_transport

        spec = get_spec("sim")
        self.assertIsInstance(resolve_transport("sim://", spec), SimTransport)
        self.assertIsInstance(resolve_transport(None, spec), SimTransport)
        go2 = profile_for("unitree-go2")
        self.assertEqual((go2.max_vx, go2.max_vy, go2.max_wz), (0.5, 0.3, 1.0))
        omni = profile_for("omnibot")
        self.assertEqual((omni.max_vx, omni.max_vy, omni.max_wz), (0.2, 0.2, 1.0))


class TestBuiltinPluginDoesNotImportVendorSdks(unittest.TestCase):
    def test_import_and_factories_stay_off_vendor_sdks(self):
        script = textwrap.dedent(
            """\
            import sys

            import ohho
            import ohho.plugins

            preloaded = sorted(
                name for name in sys.modules if name.startswith("ohho.robots")
            )
            assert "ohho.robots.builtin" not in sys.modules, preloaded
            assert "ohho.robots.omnibot" not in sys.modules, preloaded
            from ohho.registry import get_spec

            spec = get_spec("omnibot")
            assert spec.source == "omnibot"
            assert "ohho.robots.omnibot" in sys.modules
            assert "ohho.robots.builtin" in sys.modules

            from ohho.robots.builtin import plugin as builtin_plugin
            from ohho.robots.omnibot import plugin as omnibot_plugin

            bot = builtin_plugin()
            arm = omnibot_plugin()
            specs = {item.id: item for item in bot.specs()}
            assert "omnibot" not in specs
            factories = bot.transports()
            assert "serial" not in factories
            assert "yahboom-serial" not in factories
            backends = bot.sim_backends()
            assert type(factories["sim"](specs["sim"], "", None)).__name__ == "SimTransport"
            assert (
                type(factories["dds"](specs["unitree-go2"], "lo", None)).__name__
                == "UnitreeDdsTransport"
            )
            assert (
                type(factories["feetech"](spec, "", None)).__name__ == "FeetechTransport"
            )
            assert type(factories["ros2"](spec, "", None)).__name__ == "Ros2Transport"
            assert (
                type(backends["kinematic"](specs["sim"], "", None)).__name__
                == "SimTransport"
            )
            serial = arm.transports()
            made = serial["serial"](spec, "/dev/ttyUSB0,/dev/ttyACM0", None)
            assert type(made).__name__ == "CompositeTransport"
            single = serial["yahboom-serial"](spec, "/dev/ttyUSB0", None)
            assert type(single).__name__ == "CompositeTransport"
            base = serial["serial"](specs["sim"], "/dev/ttyUSB0", None)
            assert type(base).__name__ == "YahboomTransport"
            policy = arm.safety_policy(spec)
            assert policy is not None and policy.name == "ground+arm"
            assert arm.descriptor(spec)["id"] == "omnibot"
            report = arm.doctor(spec, "sim://", True)
            assert report is not None and report.ok
            refused = arm.doctor(spec, "serial:///dev/ttyUSB0", False)
            assert refused is not None and refused.ok is False
            forbidden = {
                "serial",
                "lerobot",
                "cyclonedds",
                "unitree_sdk2py",
                "rclpy",
                "pymavlink",
            }
            leaked = sorted(forbidden & set(sys.modules))
            if leaked:
                raise SystemExit("imported " + ",".join(leaked))
            """
        )
        env = os.environ.copy()
        env["PYTHONPATH"] = str(_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        env.pop("OHHO_ARM_HARDWARE", None)
        proc = subprocess.run(
            [sys.executable, "-c", script],
            cwd=_ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)


if __name__ == "__main__":
    unittest.main()
