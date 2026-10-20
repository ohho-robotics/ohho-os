"""Sim CI matrix rows come from registered specs (OHH-316)."""

from __future__ import annotations

import importlib.metadata
import io
import json
import unittest
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from ohho.cli import main
from ohho.plugins import (
    BUILTIN_ENTRY_VALUE,
    ENTRY_GROUP,
    AssetSource,
    ci_sim_matrix_rows,
    format_matrix_runtime,
    loaded_plugins,
)
from ohho.registry import RobotSpec, register
from ohho.robot_spec import CiSim

_ROOT = Path(__file__).resolve().parents[1]
_WORKFLOW = _ROOT / ".github" / "workflows" / "sim-matrix.yml"
_CI = _ROOT / ".github" / "workflows" / "ci.yml"
_PIN = "0123456789abcdef0123456789abcdef01234567"
_ROBOT = "matrix-rover"
_PARKED = "matrix-parked"
_REPO = "example-org/mesh-pack"


class _MemEP:
    def __init__(self, name: str, target: object):
        self.name = name
        self.value = "mem:plugin"
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


def _stdout(argv: list[str]) -> tuple[int, str]:
    buf = io.StringIO()
    with redirect_stdout(buf):
        code = main(argv)
    return code, buf.getvalue()


def _stderr(argv: list[str]) -> tuple[int, str]:
    buf = io.StringIO()
    with redirect_stdout(io.StringIO()), redirect_stderr(buf):
        code = main(argv)
    return code, buf.getvalue()


def _fixture_plugin():
    pin = AssetSource(repository=_REPO, commit=_PIN.upper())

    class Plugin:
        name = "matrix-fixture"

        def specs(self):
            return [
                RobotSpec(
                    id=_ROBOT,
                    name="Matrix Rover",
                    category="wheeled",
                    capabilities=("base.drive",),
                    adapter="sim",
                    source="matrix-fixture",
                    ci_sim=CiSim(enabled=True, scenario="drive_square"),
                ),
                RobotSpec(
                    id=_PARKED,
                    name="Parked",
                    category="wheeled",
                    capabilities=("base.drive",),
                    adapter="sim",
                    source="matrix-fixture",
                    ci_sim=CiSim(enabled=False, scenario="drive_square"),
                ),
            ]

        def transports(self):
            return {}

        def sim_backends(self):
            return {}

        def safety_policy(self, spec):
            del spec
            return None

        def descriptor(self, spec):
            del spec
            return None

        def doctor(self, spec, uri=None, read_only=True):
            del spec, uri, read_only
            return None

        def asset_source(self):
            return pin

    return Plugin


def _broken_assets():
    class Plugin:
        name = "matrix-broken"

        def specs(self):
            return [
                RobotSpec(
                    id="matrix-broken-bot",
                    name="Broken",
                    category="wheeled",
                    capabilities=("base.drive",),
                    adapter="sim",
                    source="matrix-broken",
                    ci_sim=CiSim(enabled=True, scenario="drive_square"),
                )
            ]

        def transports(self):
            return {}

        def sim_backends(self):
            return {}

        def safety_policy(self, spec):
            del spec
            return None

        def descriptor(self, spec):
            del spec
            return None

        def doctor(self, spec, uri=None, read_only=True):
            del spec, uri, read_only
            return None

        def asset_source(self):
            return "meshes/link.stl"

    return Plugin


class TestCiSimList(unittest.TestCase):
    def test_builtin_rows_are_the_opted_in_specs(self):
        code, text = _stdout(["list", "--json", "--ci-sim"])
        self.assertEqual(code, 0)
        self.assertFalse(text.endswith("\n\n"))
        rows = json.loads(text)
        self.assertEqual([row["robot"] for row in rows], ["omnibot", "sim"])
        by_id = {row["robot"]: row for row in rows}
        self.assertEqual(
            by_id["sim"],
            {
                "robot": "sim",
                "robot_class": "mobile_base",
                "plugin": "builtin",
                "scenario": "drive_square",
                "asset_repository": "",
                "asset_commit": "",
            },
        )
        self.assertEqual(by_id["omnibot"]["robot_class"], "mobile_manipulator")
        self.assertEqual(by_id["omnibot"]["plugin"], "builtin")
        self.assertEqual(by_id["omnibot"]["scenario"], "drive_square")
        self.assertEqual(by_id["omnibot"]["asset_commit"], "")
        self.assertNotIn("unitree-go2", by_id)

    def test_plain_list_stays_a_table(self):
        code, text = _stdout(["list"])
        self.assertEqual(code, 0)
        self.assertIn("omnibot", text)
        self.assertIn("Generic Sim Bot", text)
        self.assertNotIn("{", text)
        self.assertNotIn("asset_repository", text)
        self.assertNotIn("drive_square", text)

    def test_flags_must_be_paired(self):
        code, err = _stderr(["list", "--json"])
        self.assertEqual(code, 2)
        self.assertIn("--json", err)
        code, err = _stderr(["list", "--ci-sim"])
        self.assertEqual(code, 2)
        self.assertIn("--ci-sim", err)
        code, err = _stderr(["list", "--plugins", "--json", "--ci-sim"])
        self.assertEqual(code, 2)
        self.assertIn("--plugins", err)

    def test_added_spec_is_a_row_without_a_workflow_edit(self):
        workflow = _WORKFLOW.read_text(encoding="utf-8")
        self.assertNotIn(_ROBOT, workflow)
        self.assertNotIn(_PARKED, workflow)
        self.assertNotIn("matrix-fixture", workflow)
        self.assertNotIn(_REPO, workflow)

        def reader():
            return [_builtin_entry(), _MemEP("matrix-fixture", _fixture_plugin())]

        with _discovered(reader):
            register(
                RobotSpec(
                    id="matrix-blank-scenario",
                    name="Blank Scenario",
                    category="wheeled",
                    capabilities=("base.drive",),
                    adapter="sim",
                    source="hand-registered",
                    ci_sim=CiSim(enabled=True, scenario=""),
                )
            )
            code, text = _stdout(["list", "--json", "--ci-sim"])
            self.assertEqual(code, 0)
            rows = json.loads(text)
            by_id = {row["robot"]: row for row in rows}
            self.assertEqual(
                by_id[_ROBOT],
                {
                    "robot": _ROBOT,
                    "robot_class": "mobile_base",
                    "plugin": "matrix-fixture",
                    "scenario": "drive_square",
                    "asset_repository": _REPO,
                    "asset_commit": _PIN,
                },
            )
            self.assertNotIn(_PARKED, by_id)
            self.assertEqual(by_id["matrix-blank-scenario"]["scenario"], "drive_square")
            self.assertEqual(
                by_id["matrix-blank-scenario"]["plugin"], "hand-registered"
            )
            self.assertEqual(by_id["matrix-blank-scenario"]["asset_commit"], "")
            self.assertIn("omnibot", by_id)
            self.assertIn("sim", by_id)

            code, plain = _stdout(["list"])
            self.assertEqual(code, 0)
            self.assertIn(_ROBOT, plain)
            self.assertNotIn("{", plain)
            self.assertNotIn("asset_commit", plain)

        self.assertNotIn(_ROBOT, _WORKFLOW.read_text(encoding="utf-8"))
        self.assertEqual(
            [
                row["robot"]
                for row in json.loads(_stdout(["list", "--json", "--ci-sim"])[1])
            ],
            ["omnibot", "sim"],
        )

    def test_bad_asset_source_is_a_load_error_and_not_a_row(self):
        def reader():
            return [
                _builtin_entry(),
                _MemEP("matrix-fixture", _fixture_plugin()),
                _MemEP("matrix-broken", _broken_assets()),
            ]

        with _discovered(reader):
            broken = next(
                info for info in loaded_plugins() if info.name == "matrix-broken"
            )
            self.assertIsNotNone(broken.error)
            self.assertIn("AssetSource", broken.error or "")
            robots = {row["robot"] for row in ci_sim_matrix_rows()}
            self.assertNotIn("matrix-broken-bot", robots)
            self.assertIn(_ROBOT, robots)
            self.assertIn("sim", robots)

    def test_asset_source_rejects_a_path_a_url_and_a_branch(self):
        with self.assertRaises(ValueError):
            AssetSource(repository="assets/meshes/link.stl", commit=_PIN)
        with self.assertRaises(ValueError):
            AssetSource(repository="https://example.com/meshes.git", commit=_PIN)
        with self.assertRaises(ValueError):
            AssetSource(repository=_REPO, commit="main")
        pin = AssetSource(repository=_REPO, commit=_PIN.upper())
        self.assertEqual(pin.commit, _PIN)
        self.assertEqual(pin.repository, _REPO)


class TestWorkflow(unittest.TestCase):
    def test_workflow_records_runtime_and_does_not_arm_hardware(self):
        text = _WORKFLOW.read_text(encoding="utf-8")
        self.assertIn("ohho list --json --ci-sim", text)
        self.assertIn("fromJson(needs.discover.outputs.matrix)", text)
        self.assertIn("python -m ohho.testing.conformance --robot", text)
        self.assertIn("GITHUB_STEP_SUMMARY", text)
        self.assertIn("format_matrix_runtime", text)
        self.assertIn("actions/cache/restore@v4", text)
        self.assertIn("actions/cache/save@v4", text)
        self.assertIn(
            "key: plugin-assets-${{ matrix.plugin }}-${{ matrix.asset_commit }}",
            text,
        )
        self.assertIn("repository: ${{ matrix.asset_repository }}", text)
        self.assertIn("ref: ${{ matrix.asset_commit }}", text)
        self.assertIn("path: .sim-assets/${{ matrix.plugin }}", text)
        self.assertIn("R-05", text)
        self.assertIn("reusable workflows", text)
        self.assertIn("robot:", text)
        self.assertIn("backend:", text)
        self.assertIn("PX4 Gazebo", text)
        self.assertIn("ArduPilot SITL", text)
        self.assertIn("Isaac", text)
        self.assertNotIn("OHHO_ARM_HARDWARE", text)
        self.assertIn(
            ".sim-assets/", (_ROOT / ".gitignore").read_text(encoding="utf-8")
        )

    def test_existing_ci_jobs_remain(self):
        ci = _CI.read_text(encoding="utf-8")
        self.assertIn("python -m unittest discover -s tests", ci)
        self.assertIn("--fail-under=95", ci)
        self.assertIn("ohho doctor", ci)
        self.assertIn("ohho selftest --sim", ci)
        sim_loop = (_ROOT / ".github" / "workflows" / "sim-loop.yml").read_text(
            encoding="utf-8"
        )
        self.assertIn("sim-loop", sim_loop)
        demo = (_ROOT / ".github" / "workflows" / "demo-drive-ros2.yml").read_text(
            encoding="utf-8"
        )
        self.assertIn("demo-drive-fake-robot", demo)

    def test_summary_line_includes_the_runtime(self):
        text = format_matrix_runtime(
            {
                "robot": "sim",
                "robot_class": "mobile_base",
                "plugin": "builtin",
                "scenario": "drive_square",
            },
            3,
        )
        self.assertIn("| sim | mobile_base | builtin | drive_square | 3 |", text)
        self.assertIn("runtime_s", text)
        with self.assertRaises(ValueError):
            format_matrix_runtime({"robot": "sim"}, -1)
        with self.assertRaises(TypeError):
            format_matrix_runtime({"robot": "sim"}, True)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
