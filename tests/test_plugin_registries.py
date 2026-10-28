"""Plugin transport and sim-backend registries (OHH-311).

``resolve_transport`` asks plugins first. The scheme dict in
``ohho.adapters`` stays the fallback, with the same entries as before.
"""

from __future__ import annotations

import importlib.metadata
import unittest
from contextlib import contextmanager
from dataclasses import replace
from unittest.mock import patch

from ohho.adapters import _HARDWARE, AdapterUnavailable, resolve_transport
from ohho.adapters.composite import CompositeTransport
from ohho.adapters.feetech import FeetechTransport
from ohho.adapters.ros2 import Ros2Transport
from ohho.adapters.sim import SimTransport
from ohho.adapters.unitree import UnitreeDdsTransport
from ohho.robots.omnibot.yahboom import YahboomTransport
from ohho.plugins import ENTRY_GROUP, describe_plugin_schemes, loaded_plugins
from ohho.registry import RobotSpec, UnknownRobot, get_spec, spec_from_dict


class _MemEP:
    def __init__(self, name: str, target: object):
        self.name = name
        self.value = "mem:plugin"
        self.group = ENTRY_GROUP
        self.dist = None
        self._target = target

    def load(self) -> object:
        return self._target


def _plugin(name, specs=(), schemes=None, backends=None):
    class Plugin:
        def specs(self):
            return list(specs)

        def transports(self):
            return dict(schemes or {})

        def sim_backends(self):
            return dict(backends or {})

        def safety_policy(self, spec):
            return None

        def descriptor(self, spec):
            return None

        def doctor(self, spec, uri=None, read_only=True):
            return None

    Plugin.name = name
    return Plugin


@contextmanager
def _discovered(reader):
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


def _builtin_reader(*extra):
    def reader():
        return list(extra)

    return reader


class _Marker:
    def __init__(self, spec, address, runtime):
        self.spec = spec
        self.address = address
        self.runtime = runtime
        self.protocol = "marker"


class _DictOnly:
    protocol = "dict-only"

    def __init__(self, spec, address):
        self.spec = spec
        self.address = address


class TestSchemeDictUnchanged(unittest.TestCase):
    def test_hardware_dict_entries(self):
        self.assertEqual(
            set(_HARDWARE),
            {"feetech", "dds", "unitree-dds", "ros2"},
        )
        self.assertNotIn("serial", _HARDWARE)
        self.assertNotIn("yahboom-serial", _HARDWARE)
        self.assertIs(_HARDWARE["feetech"], FeetechTransport)
        self.assertIs(_HARDWARE["dds"], UnitreeDdsTransport)
        self.assertIs(_HARDWARE["unitree-dds"], UnitreeDdsTransport)
        self.assertIs(_HARDWARE["ros2"], Ros2Transport)


class TestPluginSchemesResolve(unittest.TestCase):
    def test_plugin_scheme_resolves_and_is_not_in_the_dict(self):
        ran: list[str] = []

        def factory(spec, address, runtime):
            ran.append("widget")
            made = _Marker(spec, address, runtime)
            made.protocol = "widget"
            return made

        plugin = _plugin("lab", schemes={"Widget": factory})
        spec = RobotSpec(id="lab-bot", name="Lab", category="wheeled")
        runtime = object()
        with _discovered(_builtin_reader(_MemEP("lab", plugin))):
            self.assertEqual(ran, [])
            self.assertNotIn("widget", _HARDWARE)
            made = resolve_transport("widget://radio-1", spec, runtime=runtime)
            self.assertEqual(ran, ["widget"])
            self.assertEqual(made.protocol, "widget")
            self.assertIs(made.spec, spec)
            self.assertEqual(made.address, "radio-1")
            self.assertIs(made.runtime, runtime)

    def test_plugin_wins_over_the_same_dict_entry(self):
        def factory(spec, address, runtime):
            made = _Marker(spec, address, runtime)
            made.protocol = "from-plugin"
            return made

        plugin = _plugin("lab", schemes={"overlap": factory})
        spec = RobotSpec(id="lab-bot", name="Lab", category="wheeled")
        _HARDWARE["overlap"] = _DictOnly
        try:
            with _discovered(_builtin_reader(_MemEP("lab", plugin))):
                made = resolve_transport("overlap://addr", spec)
                self.assertEqual(made.protocol, "from-plugin")
                self.assertEqual(made.address, "addr")
        finally:
            del _HARDWARE["overlap"]

    def test_dict_resolves_a_scheme_no_plugin_registered(self):
        spec = RobotSpec(id="lab-bot", name="Lab", category="wheeled")
        _HARDWARE["dict-only"] = _DictOnly
        try:
            made = resolve_transport("dict-only://port", spec)
            self.assertIsInstance(made, _DictOnly)
            self.assertEqual(made.protocol, "dict-only")
            self.assertEqual(made.address, "port")
            self.assertIs(made.spec, spec)
        finally:
            del _HARDWARE["dict-only"]

    def test_builtin_schemes_come_from_the_registry(self):
        import ohho.plugins as plugins

        calls: list[str] = []
        real = plugins.plugin_transport

        def wrapped(scheme, spec, address, runtime):
            calls.append(scheme)
            return real(scheme, spec, address, runtime)

        go2 = get_spec("unitree-go2")
        omni = get_spec("omnibot")
        sim = get_spec("sim")
        with patch.object(plugins, "plugin_transport", wrapped):
            dds = resolve_transport("dds://eth0", go2)
            self.assertEqual(calls, ["dds"])
            self.assertIsInstance(dds, UnitreeDdsTransport)
            self.assertEqual(dds.protocol, "dds")
            self.assertEqual(dds.address, "eth0")

            alias = resolve_transport("unitree-dds://lo", go2)
            self.assertIsInstance(alias, UnitreeDdsTransport)
            self.assertEqual(alias.address, "lo")

            arm = resolve_transport("feetech:///dev/ttyACM0", omni)
            self.assertIsInstance(arm, FeetechTransport)
            self.assertEqual(arm.port, "/dev/ttyACM0")

            composite = resolve_transport("serial:///dev/ttyUSB0,/dev/ttyACM0", omni)
            self.assertIsInstance(composite, CompositeTransport)

            single = resolve_transport("yahboom-serial:///dev/ttyUSB0", omni)
            self.assertIsInstance(single, CompositeTransport)

            base = resolve_transport("serial:///dev/ttyUSB0", sim)
            self.assertIsInstance(base, YahboomTransport)
            self.assertEqual(base.protocol, "serial")

            bare = resolve_transport("serial:///dev/ttyUSB0", None)
            self.assertIsInstance(bare, YahboomTransport)

            ros = resolve_transport("ros2:///robot1", omni, runtime=object())
            self.assertIsInstance(ros, Ros2Transport)
            self.assertEqual(ros.address, "/robot1")

        self.assertNotIn("sim", calls)
        self.assertNotIn("simulated", calls)

    def test_fallback_dict_resolves_every_current_scheme(self):
        import ohho.plugins as plugins

        go2 = get_spec("unitree-go2")
        omni = get_spec("omnibot")
        sim = get_spec("sim")
        with patch.object(plugins, "plugin_transport", return_value=None):
            self.assertIsInstance(resolve_transport("sim://", sim), SimTransport)
            self.assertIsInstance(resolve_transport("simulated://", sim), SimTransport)
            self.assertIsInstance(resolve_transport(None, go2), SimTransport)

            for uri in (
                "serial:///dev/ttyUSB0,/dev/ttyACM0",
                "yahboom-serial:///dev/ttyUSB0",
            ):
                with self.assertRaises(AdapterUnavailable):
                    resolve_transport(uri, omni)
            with self.assertRaises(AdapterUnavailable):
                resolve_transport("serial:///dev/ttyUSB0", sim)
            with self.assertRaises(AdapterUnavailable):
                resolve_transport("serial:///dev/ttyUSB0", None)

            arm = resolve_transport("feetech:///dev/ttyACM0", omni)
            self.assertIsInstance(arm, FeetechTransport)
            self.assertEqual(arm.port, "/dev/ttyACM0")

            dds = resolve_transport("dds://eth0", go2)
            self.assertIsInstance(dds, UnitreeDdsTransport)
            self.assertEqual(dds.address, "eth0")
            alias = resolve_transport("unitree-dds://lo", go2)
            self.assertIsInstance(alias, UnitreeDdsTransport)
            self.assertEqual(alias.address, "lo")

            ros = resolve_transport("ros2://", omni)
            self.assertIsInstance(ros, Ros2Transport)
            self.assertEqual(ros.protocol, "ros2")

    def test_explicit_hardware_uri_ignores_sim_backend(self):
        def factory(spec, address, runtime):
            made = _Marker(spec, address, runtime)
            made.protocol = "lab-sim"
            return made

        plugin = _plugin("lab", backends={"lab-sim": factory})
        with _discovered(_builtin_reader(_MemEP("lab", plugin))):
            spec = replace(get_spec("unitree-go2"), sim_backend="lab-sim")
            dds = resolve_transport("dds://eth0", spec)
            self.assertIsInstance(dds, UnitreeDdsTransport)
            self.assertEqual(dds.address, "eth0")


class TestUnknownSchemeListsPluginSchemes(unittest.TestCase):
    def test_error_names_each_plugins_schemes(self):
        plugin = _plugin(
            "lab",
            schemes={
                "widget": lambda spec, address, runtime: _Marker(spec, address, runtime)
            },
        )
        spec = get_spec("sim")
        with _discovered(_builtin_reader(_MemEP("lab", plugin))):
            with self.assertRaises(AdapterUnavailable) as raised:
                resolve_transport("wat://nope", spec)
            message = str(raised.exception)
            self.assertIn("Unknown transport scheme 'wat'", message)
            self.assertIn("Known: sim, serial, feetech, dds, ros2", message)
            self.assertIn(
                "builtin: sim, simulated, feetech, dds, unitree-dds, ros2", message
            )
            self.assertIn("omnibot: serial, yahboom-serial", message)
            self.assertIn("lab: widget", message)
            self.assertIn(describe_plugin_schemes(), message)
            names = [info.name for info in loaded_plugins() if info.error is None]
            self.assertEqual(names[0], "builtin")
            self.assertIn("lab", names)


class TestSimBackendRegistry(unittest.TestCase):
    def _lab(self, ran: list[tuple]):
        def factory(spec, address, runtime):
            ran.append((spec, address, runtime))
            made = _Marker(spec, address, runtime)
            made.protocol = "lab-sim"
            return made

        return _plugin("lab", backends={"lab-sim": factory})

    def test_sim_uri_uses_declared_backend(self):
        ran: list[tuple] = []
        plugin = self._lab(ran)
        spec = RobotSpec(
            id="lab-bot",
            name="Lab",
            category="wheeled",
            sim_backend="  Lab-Sim  ",
        )
        runtime = object()
        with _discovered(_builtin_reader(_MemEP("lab", plugin))):
            self.assertEqual(spec.sim_backend, "Lab-Sim")
            self.assertEqual(ran, [])
            made = resolve_transport("sim://arena", spec, runtime=runtime)
            self.assertEqual(made.protocol, "lab-sim")
            self.assertEqual(len(ran), 1)
            self.assertIs(ran[0][0], spec)
            self.assertEqual(ran[0][1], "arena")
            self.assertIs(ran[0][2], runtime)

            again = resolve_transport("simulated://pit", spec, runtime=runtime)
            self.assertEqual(again.protocol, "lab-sim")
            self.assertEqual(ran[1][1], "pit")

            auto = resolve_transport(None, spec, runtime=runtime)
            self.assertEqual(auto.protocol, "lab-sim")
            self.assertEqual(ran[2][1], "")

    def test_undeclared_sim_stays_on_todays_simulator(self):
        import ohho.plugins as plugins

        calls: list[str] = []
        real = plugins.plugin_sim_backend

        def wrapped(spec, address, runtime):
            calls.append("sim")
            return real(spec, address, runtime)

        spec = get_spec("sim")
        self.assertEqual(spec.sim_backend, "")
        with patch.object(plugins, "plugin_sim_backend", wrapped):
            transport = resolve_transport("sim://", spec)
            self.assertIsInstance(transport, SimTransport)
            self.assertIs(transport.spec, spec)
            self.assertEqual(calls, ["sim"])
            self.assertIsInstance(resolve_transport(None, spec), SimTransport)

    def test_builtin_kinematic_backend(self):
        spec = replace(get_spec("sim"), sim_backend="kinematic")
        transport = resolve_transport("sim://", spec)
        self.assertIsInstance(transport, SimTransport)
        self.assertIs(transport.spec, spec)
        folded = replace(spec, sim_backend="Kinematic")
        self.assertIsInstance(resolve_transport("sim://", folded), SimTransport)

    def test_unknown_backend_lists_each_plugin(self):
        plugin = _plugin(
            "lab",
            backends={
                "lab-sim": lambda spec, address, runtime: _Marker(
                    spec, address, runtime
                )
            },
        )
        spec = RobotSpec(
            id="lab-bot", name="Lab", category="wheeled", sim_backend="missing"
        )
        with _discovered(_builtin_reader(_MemEP("lab", plugin))):
            with self.assertRaises(AdapterUnavailable) as raised:
                resolve_transport("sim://", spec)
            message = str(raised.exception)
            self.assertIn("Unknown sim backend 'missing'", message)
            self.assertIn("builtin: kinematic", message)
            self.assertIn("lab: lab-sim", message)

    def test_duplicate_sim_backend_names_both_plugins(self):
        def alpha_shared(spec, address, runtime):
            made = _Marker(spec, address, runtime)
            made.protocol = "alpha-shared"
            return made

        def alpha_only(spec, address, runtime):
            made = _Marker(spec, address, runtime)
            made.protocol = "alpha-only"
            return made

        def beta_only(spec, address, runtime):
            made = _Marker(spec, address, runtime)
            made.protocol = "beta-only"
            return made

        alpha = _plugin(
            "alpha",
            specs=[
                RobotSpec(
                    id="alpha-bot", name="Alpha", category="wheeled", source="alpha"
                )
            ],
            backends={"shared-sim": alpha_shared, "alpha-only": alpha_only},
        )
        beta = _plugin(
            "beta",
            specs=[
                RobotSpec(id="beta-bot", name="Beta", category="wheeled", source="beta")
            ],
            backends={"shared-sim": alpha_shared, "beta-only": beta_only},
        )

        def reader():
            return [_MemEP("beta", beta), _MemEP("alpha", alpha)]

        with _discovered(reader):
            info = next(item for item in loaded_plugins() if item.name == "beta")
            self.assertIn("duplicate sim backend 'shared-sim'", info.error or "")
            self.assertIn("'alpha'", info.error or "")
            self.assertIn("'beta'", info.error or "")
            self.assertEqual(get_spec("alpha-bot").name, "Alpha")
            with self.assertRaises(UnknownRobot):
                get_spec("beta-bot")

            shared = RobotSpec(
                id="user", name="User", category="wheeled", sim_backend="shared-sim"
            )
            self.assertEqual(
                resolve_transport("sim://", shared).protocol, "alpha-shared"
            )
            only = replace(shared, sim_backend="alpha-only")
            self.assertEqual(resolve_transport("sim://", only).protocol, "alpha-only")
            rejected = replace(shared, sim_backend="beta-only")
            with self.assertRaises(AdapterUnavailable):
                resolve_transport("sim://", rejected)

    def test_sim_backend_manifest_round_trip(self):
        spec = spec_from_dict(
            {
                "id": "declared",
                "name": "Declared",
                "category": "wheeled",
                "sim_backend": "kinematic",
            }
        )
        self.assertEqual(spec.sim_backend, "kinematic")
        again = spec_from_dict(spec.to_dict())
        self.assertEqual(again, spec)
        omitted = spec_from_dict({"id": "plain", "category": "wheeled"})
        self.assertEqual(omitted.sim_backend, "")
        with self.assertRaises(ValueError):
            spec_from_dict({"id": "bad", "sim_backend": 3})
        with self.assertRaises(TypeError):
            RobotSpec(id="bad", name="Bad", category="wheeled", sim_backend=3)  # type: ignore[arg-type]


class TestBuiltinEntryStillLoads(unittest.TestCase):
    def test_installed_style_entry_point_object(self):
        entry = importlib.metadata.EntryPoint(
            name="builtin",
            value="ohho.robots.builtin:plugin",
            group=ENTRY_GROUP,
        )

        def reader():
            return [entry]

        with _discovered(reader):
            info = next(item for item in loaded_plugins() if item.name == "builtin")
            self.assertIsNone(info.error)
            self.assertIn("dds", info.transport_factories)
            self.assertIn("kinematic", info.sim_factories)
            go2 = get_spec("unitree-go2")
            transport = info.transport_factories["dds"](go2, "eth0", None)
            self.assertIsInstance(transport, UnitreeDdsTransport)
            self.assertEqual(transport.address, "eth0")


if __name__ == "__main__":
    unittest.main()
