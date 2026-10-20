"""Booster T2 plugin: discovery, pin, and no vendor SDK at import."""

from __future__ import annotations

import ast
import hashlib
import importlib.metadata
import io
import json
import os
import re
import subprocess
import sys
import tarfile
import tempfile
import unittest
import urllib.error
import urllib.request
from contextlib import contextmanager, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from ohho.cli import main
from ohho.humanoid import HumanoidMode
from ohho.plugins import BUILTIN_ENTRY_VALUE, ENTRY_GROUP
from ohho.registry import list_specs
from ohho.robot_spec import HumanoidLimits, RobotClass, default_limits
from ohho.safety import PROFILES, profile_for

_REPO = Path(__file__).resolve().parents[1]
_PLUGIN = _REPO / "plugins" / "ohho-booster"
_OHHO = _REPO / "ohho"
_SDK_MODULES = {"booster_robotics_sdk_python", "booster_robotics_sdk"}
_VENDOR = re.compile(r"booster|\bt2\b", re.IGNORECASE)


def _ensure_plugin_path() -> None:
    root = str(_PLUGIN)
    if root not in sys.path:
        sys.path.insert(0, root)


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


def _stdout(argv: list[str]) -> tuple[int, str]:
    buf = io.StringIO()
    with redirect_stdout(buf):
        code = main(argv)
    return code, buf.getvalue()


class _Entry:
    def __init__(self, name: str, value: str, target: object):
        self.name = name
        self.value = value
        self.group = ENTRY_GROUP
        self.dist = None
        self._target = target

    def load(self) -> object:
        return self._target


def _builtin_entry():
    return importlib.metadata.EntryPoint(
        name="builtin",
        value=BUILTIN_ENTRY_VALUE,
        group=ENTRY_GROUP,
    )


def _archive(path: Path, commit: str, files: dict[str, bytes]) -> None:
    root = f"booster_assets-{commit}"
    with tarfile.open(path, "w:gz") as tar:
        for rel, data in files.items():
            info = tarfile.TarInfo(f"{root}/{rel}")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))


class TestBoosterSpec(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _ensure_plugin_path()
        from ohho_booster.joints import JOINT_GROUPS, T2_JOINT_NAMES
        from ohho_booster.modes import VENDOR_MODE_MAP, generic_mode
        from ohho_booster.spec import booster_t2_spec

        cls.groups = JOINT_GROUPS
        cls.names = T2_JOINT_NAMES
        cls.mode_map = VENDOR_MODE_MAP
        cls.generic_mode = staticmethod(generic_mode)
        cls.spec = booster_t2_spec()

    def test_identity_and_joint_groups(self):
        spec = self.spec
        self.assertEqual(spec.id, "booster-t2")
        self.assertEqual(spec.robot_class, RobotClass.HUMANOID)
        self.assertEqual(spec.category, "humanoid")
        self.assertEqual(spec.source, "booster")
        self.assertEqual(
            tuple(self.groups),
            ("head", "left_arm", "right_arm", "waist", "left_leg", "right_leg"),
        )
        self.assertEqual(
            [len(self.groups[name]) for name in self.groups],
            [2, 7, 7, 3, 6, 6],
        )
        flat = tuple(joint for name in self.groups for joint in self.groups[name])
        self.assertEqual(flat, self.names)
        self.assertEqual(len(self.names), 31)
        self.assertEqual(len(set(self.names)), 31)
        self.assertEqual(tuple(spec.joint_names), self.names)
        self.assertEqual(spec.assets.licence, "BSD-3-Clause")
        self.assertTrue(spec.assets.urdf.endswith("T2_31dof.urdf"))
        self.assertTrue(spec.assets.mjcf.endswith("T2_31dof.xml"))
        self.assertEqual(spec.sim_backend, "humanoid-mujoco")
        self.assertFalse(spec.ci_sim.enabled)

    def test_limits_are_not_looser_than_the_humanoid_defaults(self):
        generic = default_limits(RobotClass.HUMANOID, 0.5, 1.0)
        self.assertIsInstance(generic, HumanoidLimits)
        limits = self.spec.limits
        self.assertIsInstance(limits, HumanoidLimits)
        self.assertLessEqual(self.spec.max_lin, 0.5)
        self.assertLessEqual(self.spec.max_ang, 1.0)
        self.assertLessEqual(limits.max_vx, generic.max_vx)
        self.assertLessEqual(limits.max_vy, generic.max_vy)
        self.assertLessEqual(limits.max_wz, generic.max_wz)
        self.assertEqual(limits.max_joint_step, generic.max_joint_step)
        self.assertEqual(limits.max_kp, generic.max_kp)
        self.assertEqual(limits.max_tau, generic.max_tau)
        speed = self.spec.capability_params["humanoid.locomotion"]["speed_max"]
        self.assertLessEqual(speed, generic.max_vx)
        self.assertNotIn("booster-t2", PROFILES)
        profile = profile_for("booster-t2", self.spec)
        self.assertLessEqual(profile.max_vx, 0.3)
        self.assertLessEqual(profile.max_vy, 0.3)
        self.assertLessEqual(profile.max_wz, 1.0)

    def test_mode_map_is_plain_data_and_omits_soccer(self):
        self.assertEqual(
            dict(self.mode_map),
            {
                "kDamping": "DAMPING",
                "kPrepare": "PREPARE",
                "kWalking": "WALK",
                "kCustom": "CUSTOM",
                "kUnknown": "UNKNOWN",
            },
        )
        for vendor, generic in self.mode_map.items():
            self.assertEqual(HumanoidMode[generic].value, generic)
            self.assertEqual(self.generic_mode(vendor), generic)
        self.assertNotIn("kSoccer", self.mode_map)
        self.assertEqual(self.generic_mode("kSoccer"), "UNKNOWN")
        self.assertEqual(self.generic_mode(None), "UNKNOWN")


class TestBoosterDiscovery(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _ensure_plugin_path()
        from ohho_booster import plugin as plugin_factory

        cls.factory = staticmethod(plugin_factory)

    def test_install_lists_spec_and_removal_drops_it(self):
        booster = _Entry("booster", "ohho_booster:plugin", self.factory)

        def with_plugin():
            return [_builtin_entry(), booster]

        def without_plugin():
            return [_builtin_entry()]

        with _discovered(with_plugin):
            self.assertIn("booster-t2", {spec.id for spec in list_specs()})
            code, listed = _stdout(["list"])
            self.assertEqual(code, 0)
            self.assertIn("booster-t2", listed)
            code, report = _stdout(["list", "--plugins"])
            self.assertEqual(code, 0)
            self.assertIn("booster: booster-t2", report)
            self.assertNotIn("error:", report)
            info_schemes = report
            self.assertNotIn("booster://", info_schemes)

        with _discovered(without_plugin):
            self.assertNotIn("booster-t2", {spec.id for spec in list_specs()})
            code, listed = _stdout(["list"])
            self.assertEqual(code, 0)
            self.assertNotIn("booster-t2", listed)

    def test_plugin_declares_no_transport(self):
        bot = self.factory()
        spec = bot.specs()[0]
        self.assertEqual(bot.transports(), {})
        self.assertEqual(bot.sim_backends(), {})
        self.assertIsNone(bot.safety_policy(spec))
        self.assertIsNone(bot.descriptor(spec))
        self.assertIsNone(bot.doctor(spec, None, True))


class TestNoSdkImport(unittest.TestCase):
    def test_source_has_no_sdk_import(self):
        root = _PLUGIN / "ohho_booster"
        for path in root.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                modules: list[str] = []
                if isinstance(node, ast.Import):
                    modules = [alias.name.split(".")[0] for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    modules = [node.module.split(".")[0]]
                overlap = _SDK_MODULES.intersection(modules)
                self.assertFalse(overlap, f"{path} imports {sorted(overlap)}")

    def test_import_does_not_load_the_sdk(self):
        script = (
            "import sys\n"
            f"sys.path.insert(0, {str(_PLUGIN)!r})\n"
            "import ohho_booster\n"
            "from ohho_booster import plugin\n"
            "bot = plugin()\n"
            "spec = bot.specs()[0]\n"
            "assert spec.id == 'booster-t2'\n"
            "assert bot.transports() == {}\n"
            "assert 'booster_robotics_sdk_python' not in sys.modules\n"
            "assert 'booster_robotics_sdk' not in sys.modules\n"
        )
        env = os.environ.copy()
        env.pop("OHHO_ARM_HARDWARE", None)
        proc = subprocess.run(
            [sys.executable, "-c", script],
            cwd=_REPO,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)


class TestAssetPin(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _ensure_plugin_path()
        from ohho_booster.assets import archive_url, load_manifest

        cls.manifest = load_manifest()
        cls.archive_url = staticmethod(archive_url)

    def test_manifest_and_notice_record_the_pin(self):
        pin = self.manifest
        self.assertEqual(
            pin["upstream"], "https://github.com/BoosterRobotics/booster_assets"
        )
        self.assertEqual(pin["commit"], "12a97516f57469078707afb587a5f84c1cb110d1")
        self.assertEqual(pin["licence"], "BSD-3-Clause")
        self.assertIn("BoosterRobotics", pin["copyright"])
        notice = (_PLUGIN / "NOTICE").read_text(encoding="utf-8")
        self.assertIn(pin["commit"], notice)
        self.assertIn(pin["sha256"], notice)
        self.assertEqual(len(pin["sha256"]), 64)
        self.assertIn(pin["upstream"], notice)
        self.assertIn(self.archive_url(pin), notice)
        self.assertIn("BSD-3-Clause", notice)
        self.assertIn("Copyright (c) 2025, BoosterRobotics", notice)
        walk = pin["policies"]["walk"]
        self.assertEqual(walk["commit"], "7bb1462e2ac742eba5c01659b815e2e99fbe617f")
        self.assertEqual(walk["licence"], "Apache-2.0")
        self.assertEqual(len(walk["sha256"]), 64)
        self.assertIn(walk["commit"], notice)
        self.assertIn(walk["sha256"], notice)
        self.assertIn("Apache-2.0", notice)
        pyproject = (_PLUGIN / "pyproject.toml").read_text(encoding="utf-8")
        self.assertIn('[project.entry-points."ohho.robots"]', pyproject)
        self.assertIn('booster = "ohho_booster:plugin"', pyproject)

    def test_fetch_accepts_a_local_archive_and_refuses_a_different_commit(self):
        from ohho_booster import assets as assets_mod

        pin = dict(self.manifest)
        commit = pin["commit"]
        files = {
            pin["paths"]["urdf"]: b"<robot/>",
            pin["paths"]["mjcf"]: b"<mujoco/>",
            pin["paths"]["meshes"] + "/marker.txt": b"fixture",
            pin["paths"]["joint_names_module"]: b"T2_JOINT_NAMES = []\n",
        }
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            archive = folder / "ok.tar.gz"
            _archive(archive, commit, files)
            pin["sha256"] = hashlib.sha256(archive.read_bytes()).hexdigest()
            with patch.object(assets_mod, "load_manifest", return_value=pin):
                root = assets_mod.fetch_assets(folder / "cache", url=archive.as_uri())
                self.assertTrue((root / pin["paths"]["urdf"]).is_file())
                again = assets_mod.fetch_assets(folder / "cache")
                self.assertEqual(again, root)
                receipt = json.loads(
                    (folder / "cache" / commit / "receipt.json").read_text(
                        encoding="utf-8"
                    )
                )
                self.assertEqual(receipt["commit"], commit)
                self.assertEqual(receipt["sha256"], pin["sha256"])
                self.assertEqual(receipt["licence"], "BSD-3-Clause")
                self.assertEqual(receipt["upstream"], pin["upstream"])

                wrong = folder / "wrong.tar.gz"
                _archive(wrong, "0" * 40, files)
                with self.assertRaises(ValueError) as caught:
                    assets_mod.fetch_assets(folder / "cache-wrong", url=wrong.as_uri())
                self.assertIn("sha256", str(caught.exception))

            wrong_pin = dict(pin)
            wrong_pin["sha256"] = hashlib.sha256(wrong.read_bytes()).hexdigest()
            with patch.object(assets_mod, "load_manifest", return_value=wrong_pin):
                with self.assertRaises(ValueError) as caught:
                    assets_mod.fetch_assets(folder / "cache-root", url=wrong.as_uri())
                self.assertIn("pinned commit", str(caught.exception))

    def test_policy_fetch_checks_sha256(self):
        from ohho_booster import assets as assets_mod

        payload = b"checkpoint-bytes"
        digest = hashlib.sha256(payload).hexdigest()
        pin = json.loads(json.dumps(self.manifest))
        pin["policies"]["walk"]["sha256"] = digest
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            blob = folder / "policy.pt"
            blob.write_bytes(payload)
            with patch.object(assets_mod, "load_manifest", return_value=pin):
                path = assets_mod.fetch_policy(folder / "cache", url=blob.as_uri())
                self.assertEqual(path.read_bytes(), payload)
                self.assertEqual(path, assets_mod.fetch_policy(folder / "cache"))
            pin["policies"]["walk"]["sha256"] = "0" * 64
            with patch.object(assets_mod, "load_manifest", return_value=pin):
                with self.assertRaises(ValueError) as caught:
                    assets_mod.fetch_policy(folder / "other", url=blob.as_uri())
                self.assertIn("sha256", str(caught.exception))

    def test_network_fetch_matches_pinned_joint_names(self):
        """Compare upstream T2_JOINT_NAMES with the copy in the plugin.

        Skips only when that small file cannot be downloaded. The mesh archive
        is not fetched here.
        """
        from ohho_booster.joints import T2_JOINT_NAMES

        pin = self.manifest
        url = (
            "https://raw.githubusercontent.com/BoosterRobotics/booster_assets/"
            f"{pin['commit']}/src/booster_assets/motions.py"
        )
        try:
            with urllib.request.urlopen(url, timeout=15) as response:
                text = response.read().decode("utf-8")
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            self.skipTest(f"network fetch unavailable: {exc}")
        tree = ast.parse(text)
        found = None
        for node in tree.body:
            if not isinstance(node, ast.Assign):
                continue
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "T2_JOINT_NAMES":
                    found = ast.literal_eval(node.value)
        self.assertIsNotNone(found)
        self.assertEqual(tuple(found), T2_JOINT_NAMES)


class TestVendorNamesStayOutOfCore(unittest.TestCase):
    def test_ohho_package_does_not_name_the_vendor(self):
        hits: list[str] = []
        for path in _OHHO.rglob("*"):
            if path.suffix not in {".py", ".json"}:
                continue
            text = path.read_text(encoding="utf-8")
            for lineno, line in enumerate(text.splitlines(), 1):
                if _VENDOR.search(line):
                    hits.append(f"{path.relative_to(_REPO)}:{lineno}:{line.strip()}")
        self.assertEqual(hits, [])


class TestPipInstallAndUninstall(unittest.TestCase):
    def test_editable_install_lists_spec_and_uninstall_removes_it(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            venv = Path(tmp) / "venv"
            proc = subprocess.run(
                [sys.executable, "-m", "venv", str(venv)],
                cwd=_REPO,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            python = (
                venv / "Scripts" / "python.exe"
                if os.name == "nt"
                else venv / "bin" / "python"
            )
            env = os.environ.copy()
            env.pop("PYTHONPATH", None)
            env.pop("OHHO_ARM_HARDWARE", None)
            env["PYTHONNOUSERSITE"] = "1"

            def run(*args: str) -> subprocess.CompletedProcess[str]:
                return subprocess.run(
                    [str(python), *args],
                    cwd=_REPO,
                    env=env,
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=180,
                )

            installed = run("-m", "pip", "install", "-e", str(_REPO))
            self.assertEqual(
                installed.returncode, 0, installed.stdout + installed.stderr
            )
            plugin = run("-m", "pip", "install", "-e", str(_PLUGIN))
            self.assertEqual(plugin.returncode, 0, plugin.stdout + plugin.stderr)
            listed = run(
                "-c",
                "from ohho.cli import main; raise SystemExit(main(['list']))",
            )
            self.assertEqual(listed.returncode, 0, listed.stdout + listed.stderr)
            self.assertIn("booster-t2", listed.stdout)
            removed = run("-m", "pip", "uninstall", "-y", "ohho-booster")
            self.assertEqual(removed.returncode, 0, removed.stdout + removed.stderr)
            after = run(
                "-c",
                "from ohho.cli import main; raise SystemExit(main(['list']))",
            )
            self.assertEqual(after.returncode, 0, after.stdout + after.stderr)
            self.assertNotIn("booster-t2", after.stdout)


if __name__ == "__main__":
    unittest.main()
