"""OHH-314: one robot-id switch, and profiles that may only tighten caps."""

from __future__ import annotations

import io
import json
import os
import re
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

from ohho.adapters.sim import SimTransport
from ohho.cli import _cmd_sim, _connect, _normalize_robot_args, build_parser, main
from ohho.config import (
    ConfigError,
    ResolvedRobot,
    RobotProfile,
    SafetyCapError,
    apply_profile_to_spec,
    format_resolved,
    load_robot_profile,
    merge_safety_overrides,
    resolve_robot_id,
    spec_with_overrides,
)
from ohho.data.recorder import Recorder
from ohho.registry import get_spec
from ohho.robot import Robot
from ohho.runtime import NativeRuntime
from ohho.safety import SafetyConfig, profile_for
from ohho.session import RobotSession
from ohho.sim_loop import record_sim_episodes
from ohho.sim_serve import SimServe

ROOT = Path(__file__).resolve().parents[1]
_OMNIBOT_DEFAULT = re.compile(r"""default\s*=\s*['"]omnibot['"]""")


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


class TestPrecedence(unittest.TestCase):
    def test_flag_then_env_then_config_then_sim(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            config = home / ".ohho" / "config.toml"
            _write(config, 'robot = "unitree-go2"\n')
            env = {"OHHO_ROBOT": "omnibot"}

            flag = resolve_robot_id(
                "  sim  ", environ=env, home=home, config_path=config
            )
            self.assertEqual(flag, ResolvedRobot("sim", "flag"))

            blank_flag = resolve_robot_id(
                "   ", environ=env, home=home, config_path=config
            )
            self.assertEqual(blank_flag, ResolvedRobot("omnibot", "env"))

            from_env = resolve_robot_id(None, environ=env, home=home)
            self.assertEqual(from_env.source, "env")
            self.assertEqual(from_env.robot_id, "omnibot")

            from_config = resolve_robot_id(None, environ={}, home=home)
            self.assertEqual(from_config, ResolvedRobot("unitree-go2", "config"))

            stripped = resolve_robot_id(
                None, environ={}, config_path=self._file(home, 'robot = "  sim  "\n')
            )
            self.assertEqual(stripped.robot_id, "sim")
            self.assertEqual(stripped.source, "config")

            missing = resolve_robot_id(
                None, environ={}, home=home / "absent", config_path=home / "no.toml"
            )
            self.assertEqual(missing, ResolvedRobot("sim", "default"))

    def test_bad_config_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            broken = home / "broken.toml"
            _write(broken, "robot = [\n")
            with self.assertRaises(ConfigError):
                resolve_robot_id(None, environ={}, config_path=broken)

            numbered = home / "number.toml"
            _write(numbered, "robot = 1\n")
            with self.assertRaises(ConfigError) as raised:
                resolve_robot_id(None, environ={}, config_path=numbered)
            self.assertIn("non-empty string", str(raised.exception))

            extra = home / "extra.toml"
            _write(extra, 'robot = "sim"\ntransport = "serial://"\n')
            with self.assertRaises(ConfigError) as raised:
                resolve_robot_id(None, environ={}, config_path=extra)
            message = str(raised.exception)
            self.assertIn("unknown keys", message)
            self.assertIn("robots/", message)

    def _file(self, home: Path, text: str) -> Path:
        path = home / "named.toml"
        _write(path, text)
        return path


class TestProfiles(unittest.TestCase):
    def test_missing_files_are_an_empty_profile(self):
        with tempfile.TemporaryDirectory() as tmp:
            profile = load_robot_profile("omnibot", home=tmp, cwd=tmp)
        self.assertIsNone(profile.source)
        self.assertIsNone(profile.transport)
        self.assertEqual(dict(profile.safety_overrides), {})
        self.assertIsNone(RobotProfile("not-a-registered-robot").source)

    def test_repo_file_overrides_home_field_by_field_and_does_not_walk_parents(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(
                root / ".ohho" / "robots" / "omnibot.toml",
                "\n".join(
                    [
                        'transport = "sim://home"',
                        'network_interface = "eth0"',
                        'asset_path = "home.urdf"',
                        "",
                        "[safety]",
                        "max_vx = 0.1",
                        "max_wz = 0.4",
                        "",
                    ]
                ),
            )
            _write(
                root / "ohho.toml",
                "\n".join(
                    [
                        "[robots.omnibot]",
                        'transport = "sim://repo"',
                        'sim_backend = "kinematic"',
                        "",
                        "[robots.omnibot.safety]",
                        "max_vx = 0.05",
                        "",
                    ]
                ),
            )
            nested = root / "nested"
            nested.mkdir()
            # cwd is not walked upward, so the parent ohho.toml does not apply.
            profile = load_robot_profile("omnibot", home=nested, cwd=nested)
            self.assertIsNone(profile.source)
            self.assertIsNone(profile.transport)

            profile = load_robot_profile("omnibot", home=root, cwd=root)
            self.assertEqual(profile.transport, "sim://repo")
            self.assertEqual(profile.network_interface, "eth0")
            self.assertEqual(profile.sim_backend, "kinematic")
            self.assertEqual(profile.asset_path, "home.urdf")
            self.assertAlmostEqual(profile.safety_overrides["max_vx"], 0.05)
            self.assertAlmostEqual(profile.safety_overrides["max_wz"], 0.4)
            self.assertIn(" + ", profile.source or "")

    def test_flat_repo_file_applies_only_to_its_robot(self):
        with tempfile.TemporaryDirectory() as tmp:
            cwd = Path(tmp)
            _write(cwd / "ohho.toml", 'robot = "sim"\ntransport = "sim://flat"\n')
            other = load_robot_profile("omnibot", home=cwd / "empty", cwd=cwd)
            self.assertIsNone(other.transport)
            matched = load_robot_profile("sim", home=cwd / "empty", cwd=cwd)
            self.assertEqual(matched.transport, "sim://flat")

    def test_explicit_path_and_asset_backend_copy(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "one.toml"
            _write(path, 'transport = "sim://"\nasset_path = "bot.xml"\n')
            profile = load_robot_profile("sim", path=path)
            spec = apply_profile_to_spec(get_spec("sim"), profile)
            self.assertEqual(spec.assets.mjcf, "bot.xml")
            urdf = spec_with_overrides(get_spec("sim"), asset_path="bot.urdf")
            self.assertEqual(urdf.assets.urdf, "bot.urdf")
            self.assertEqual(urdf.sim_backend, get_spec("sim").sim_backend)

    def test_invalid_robot_id_and_unknown_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ConfigError):
                load_robot_profile("../omnibot", home=tmp, cwd=tmp)
            path = Path(tmp) / "bad.toml"
            _write(path, 'colour = "red"\n')
            with self.assertRaises(ConfigError) as raised:
                load_robot_profile("omnibot", path=path)
            self.assertIn("unknown keys", str(raised.exception))


class TestSafetyCaps(unittest.TestCase):
    def _profile(self, body: str, robot: str = "omnibot") -> RobotProfile:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / f"{robot}.toml"
            _write(path, body)
            return load_robot_profile(robot, path=path)

    def test_tighter_profile_loads_and_equal_cap_is_kept(self):
        profile = self._profile(
            "\n".join(
                [
                    "[safety]",
                    "max_vx = 0.1",
                    "cmd_timeout_s = 0.1",
                    "require_orientation = true",
                    "allowlist = [",
                    '  "stop_move",',
                    "]",
                    "estop_damp = false",
                    "",
                    "[safety.joint_limits]",
                    "arm_gripper = [-0.05, 0.4]",
                    "",
                ]
            )
        )
        overrides = profile.safety_overrides
        self.assertAlmostEqual(overrides["max_vx"], 0.1)
        self.assertAlmostEqual(overrides["cmd_timeout_s"], 0.1)
        self.assertEqual(overrides["allowlist"], frozenset({"stop_move"}))
        self.assertEqual(overrides["joint_limits"]["arm_gripper"], (-0.05, 0.4))
        self.assertTrue(overrides["require_orientation"])
        self.assertNotIn("estop_damp", overrides)
        applied = profile_for(
            "omnibot",
            get_spec("omnibot"),
            SafetyConfig(profiles={"omnibot": dict(overrides)}),
        )
        self.assertAlmostEqual(applied.max_vx, 0.1)

        equal = self._profile("[safety]\nmax_vx = 0.2\n")
        self.assertAlmostEqual(equal.safety_overrides["max_vx"], 0.2)

    def test_raised_caps_are_refused(self):
        cases = [
            ("[safety]\nmax_vx = 0.5\n", "max_vx"),
            ("[safety]\ncmd_timeout_s = 1.0\n", "cmd_timeout_s"),
            (
                "[safety.joint_limits]\narm_gripper = [-1.0, 0.8]\n",
                "joint_limits",
            ),
            (
                "[safety.effort_limits]\narm_gripper = 1.0\n",
                "effort_limits",
            ),
            ('[safety]\nallowlist = ["move", "stand_up"]\n', "allowlist"),
        ]
        for body, needle in cases:
            with self.subTest(needle=needle):
                with self.assertRaises(SafetyCapError) as raised:
                    self._profile(body)
                self.assertIn(needle, str(raised.exception))
                self.assertIn("tighten", str(raised.exception))

        with self.assertRaises(SafetyCapError):
            self._profile(
                "[safety]\nrequire_orientation = false\n", robot="unitree-go2"
            )
        with self.assertRaises(SafetyCapError):
            self._profile("[safety]\nestop_damp = true\n")
        with self.assertRaises(ConfigError):
            self._profile("[safety]\ndance = 1\n")

    def test_merge_keeps_the_tighter_number_and_an_existing_list(self):
        tighter = merge_safety_overrides(
            SafetyConfig(profiles={"omnibot": {"max_vx": 0.05}}),
            "omnibot",
            {"max_vx": 0.1},
        )
        self.assertAlmostEqual(tighter.profiles["omnibot"]["max_vx"], 0.05)
        kept = merge_safety_overrides(
            SafetyConfig(profiles={"omnibot": {"allowlist": ["move"]}}),
            "omnibot",
            {"allowlist": frozenset({"stop_move"}), "max_vx": 0.1},
        )
        self.assertEqual(kept.profiles["omnibot"]["allowlist"], ["move"])
        self.assertAlmostEqual(kept.profiles["omnibot"]["max_vx"], 0.1)

    def test_connect_passes_a_tighter_cap_and_refuses_a_looser_one(self):
        inner = mock.Mock()
        inner.protocol = "serial"
        gate = mock.Mock()
        gate.on_telemetry.return_value = lambda: None
        runtime = mock.Mock()
        with (
            mock.patch("ohho.robot.resolve_transport", return_value=inner),
            mock.patch("ohho.robot.get_runtime", return_value=runtime),
            mock.patch("ohho.robot.resolve_config", return_value=SafetyConfig()),
            mock.patch("ohho.robot.SafetyGate", return_value=gate) as gate_cls,
        ):
            Robot.connect(
                "omnibot",
                transport="serial://loopback",
                safety_overrides={"max_vx": 0.1},
            )
            cfg = gate_cls.call_args.kwargs["config"]
            applied = profile_for("omnibot", get_spec("omnibot"), cfg)
            self.assertAlmostEqual(applied.max_vx, 0.1)
            with self.assertRaises(SafetyCapError):
                Robot.connect(
                    "omnibot",
                    transport="serial://loopback",
                    safety_overrides={"max_vx": 5},
                )
        self.assertEqual(gate_cls.call_count, 1)


class TestEntryPoints(unittest.TestCase):
    def test_sources_name_the_resolver_and_cli_has_no_omnibot_default(self):
        required = {
            "ohho/cli.py": "resolve_robot_id",
            "ohho/sim_serve.py": "resolve_robot_id",
            "ohho/sim_loop.py": "resolve_robot_id",
            "ohho/session.py": "class RobotSession",
            "ohho/data/recorder.py": "resolve_robot_id",
        }
        for rel, needle in required.items():
            text = (ROOT / rel).read_text(encoding="utf-8")
            with self.subTest(rel=rel):
                self.assertIn(needle, text)
        for rel in ("ohho/cli.py", "ohho/sim_loop.py", "ohho/demo_drive.py"):
            text = (ROOT / rel).read_text(encoding="utf-8")
            with self.subTest(default=rel):
                self.assertIsNone(_OMNIBOT_DEFAULT.search(text))
        cli = (ROOT / "ohho" / "cli.py").read_text(encoding="utf-8")
        self.assertLess(
            cli.index("_bind_robot(args)"), cli.index("return args.func(args)")
        )
        commands = {
            name
            for action in build_parser()._actions
            if action.dest == "command"
            for name in action.choices
        }
        self.assertTrue(
            {
                "doctor",
                "list",
                "version",
                "config",
                "connect",
                "sim",
                "drive",
                "agent",
                "serve",
                "sim-serve",
                "sim-loop",
                "market",
                "profile",
                "nav",
                "look",
                "memory",
                "selftest",
            }.issubset(commands)
        )

    def test_sim_serve_session_and_recorder_follow_the_switch(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            _write(home / ".ohho" / "config.toml", 'robot = "unitree-go2"\n')
            server = SimServe(robot=None, home=home, cwd=home, environ={})
            self.assertEqual(server.robot_id, "unitree-go2")
            self.assertEqual(server.resolved.source, "config")

            pinned = SimServe(robot="omnibot", environ={"OHHO_ROBOT": "unitree-go2"})
            self.assertEqual(pinned.robot_id, "omnibot")
            self.assertEqual(pinned.resolved.source, "flag")
            self.assertIsNone(pinned.profile.source)

            session = RobotSession(home=home, cwd=home, environ={})
            self.assertEqual(session.robot_id, "unitree-go2")
            self.assertEqual(session.resolved.source, "config")
            explicit = RobotSession("sim", environ={"OHHO_ROBOT": "omnibot"})
            self.assertEqual(explicit.resolved.source, "flag")
            self.assertIsNone(explicit.profile.transport)

            spec = get_spec("sim")
            transport = SimTransport(spec)
            transport.connect()
            bot = Robot(spec, transport, NativeRuntime())
            try:
                recorder = Recorder(bot)
                self.assertEqual(recorder.resolved, ResolvedRobot("sim", "flag"))
                recorder.start_episode(task="look")
                transport.step(0.1)
                recorder.capture_frame()
                recorder.stop_episode()
                saved = Path(recorder.save(str(home / "dataset")))
            finally:
                bot.disconnect()
            info = json.loads(
                (saved / "meta" / "info.json").read_text(encoding="utf-8")
            )
            self.assertEqual(info["robot_type"], "sim")

            recorded = record_sim_episodes(
                robot_id=None,
                num_episodes=1,
                steps_per_episode=1,
                output_dir=str(home / "loop"),
                home=str(home),
                cwd=str(home),
                environ={},
            )
            loop_info = json.loads(
                (Path(recorded) / "meta" / "info.json").read_text(encoding="utf-8")
            )
            self.assertEqual(loop_info["robot_type"], "unitree-go2")

    def test_optional_positionals_keep_the_old_two_argument_form(self):
        agent = build_parser().parse_args(["agent", "omnibot", "explore the room"])
        _normalize_robot_args(agent)
        self.assertEqual(agent.robot, "omnibot")
        self.assertEqual(agent.goal, "explore the room")
        goal_only = build_parser().parse_args(["agent", "explore the room"])
        _normalize_robot_args(goal_only)
        self.assertIsNone(goal_only.robot)
        self.assertEqual(goal_only.goal, "explore the room")

        skill = build_parser().parse_args(["market", "run", "stop"])
        _normalize_robot_args(skill)
        self.assertIsNone(skill.robot)
        self.assertEqual(skill.skill_name, "stop")
        where = build_parser().parse_args(["memory", "where", "cup"])
        _normalize_robot_args(where)
        self.assertIsNone(where.robot)
        self.assertEqual(where.label, "cup")

        sim = build_parser().parse_args(["sim"])
        self.assertIsNone(sim.robot)
        connect = build_parser().parse_args(["connect", "omnibot"])
        self.assertEqual(connect.robot, "omnibot")
        self.assertIsNone(connect.transport)


class TestCli(unittest.TestCase):
    def _home(self):
        return tempfile.TemporaryDirectory()

    def test_config_show_and_doctor_print_the_source(self):
        with self._home() as tmp:
            with mock.patch.dict(os.environ, {"OHHO_HOME": tmp}):
                os.environ.pop("OHHO_ROBOT", None)
                out = io.StringIO()
                with redirect_stdout(out):
                    code = main(["config", "show"])
                self.assertEqual(code, 0)
                self.assertEqual(
                    out.getvalue().strip(),
                    format_resolved(
                        ResolvedRobot("sim", "default"), RobotProfile("sim")
                    ),
                )
                out = io.StringIO()
                with redirect_stdout(out):
                    code = main(["config", "show", "--robot", "omnibot"])
                self.assertEqual(code, 0)
                self.assertIn("robot: omnibot", out.getvalue())
                self.assertIn("source: flag", out.getvalue())

                _write(
                    Path(tmp) / ".ohho" / "config.toml",
                    'robot = "unitree-go2"\n',
                )
                out = io.StringIO()
                with redirect_stdout(out):
                    self.assertEqual(main(["config", "show"]), 0)
                self.assertIn("robot: unitree-go2", out.getvalue())
                self.assertIn("source: config", out.getvalue())

                os.environ["OHHO_ROBOT"] = "omnibot"
                out = io.StringIO()
                with redirect_stdout(out):
                    self.assertEqual(main(["config", "show"]), 0)
                self.assertIn("robot: omnibot", out.getvalue())
                self.assertIn("source: env", out.getvalue())

                out = io.StringIO()
                with redirect_stdout(out):
                    self.assertEqual(main(["doctor"]), 0)
                self.assertIn("robot       : omnibot (env)", out.getvalue())

    def test_flag_beats_env_on_connect_and_sim_ignores_profile_transport(self):
        with self._home() as tmp:
            home = Path(tmp)
            _write(
                home / ".ohho" / "robots" / "omnibot.toml",
                "\n".join(
                    [
                        'transport = "serial:///dev/ttyUSB0"',
                        'network_interface = "eth0"',
                        'sim_backend = "kinematic"',
                        "",
                        "[safety]",
                        "max_vx = 0.1",
                        "",
                    ]
                ),
            )
            with mock.patch.dict(os.environ, {"OHHO_HOME": tmp, "OHHO_ROBOT": "sim"}):
                fake = mock.Mock()
                fake.gated = False
                fake.status.return_value = mock.Mock(
                    state=mock.Mock(value="connected"), label="sim"
                )
                fake.telemetry.return_value = None
                with (
                    mock.patch("ohho.cli.time.sleep"),
                    mock.patch("ohho.cli.Robot.connect", return_value=fake) as conn,
                    redirect_stdout(io.StringIO()),
                ):
                    code = main(["connect", "omnibot", "--transport", "sim://"])
                self.assertEqual(code, 0)
                conn.assert_called_once_with(
                    "omnibot",
                    transport="sim://",
                    runtime="auto",
                    safety_config=None,
                    estop_damp=False,
                    read_only=False,
                    sim_backend="kinematic",
                    safety_overrides={"max_vx": 0.1},
                )

                args = build_parser().parse_args(
                    ["sim", "--robot", "omnibot", "--seconds", "0"]
                )
                args.resolved_robot = ResolvedRobot("omnibot", "flag")
                args.robot_profile = load_robot_profile("omnibot", home=home, cwd=home)
                sim_bot = mock.Mock()
                sim_bot.spec.name = "OmniBot"
                with (
                    mock.patch("ohho.cli.Robot.connect", return_value=sim_bot) as conn,
                    redirect_stdout(io.StringIO()),
                ):
                    self.assertEqual(_cmd_sim(args), 0)
                self.assertEqual(conn.call_args.args[0], "omnibot")
                self.assertEqual(conn.call_args.kwargs["transport"], "sim://")
                self.assertNotIn("serial://", str(conn.call_args.kwargs))

    def test_direct_connect_without_a_profile_keeps_the_old_call(self):
        args = build_parser().parse_args(["connect", "omnibot"])
        fake = mock.Mock()
        fake.gated = False
        with mock.patch("ohho.cli.Robot.connect", return_value=fake) as conn:
            self.assertIs(_connect(args), fake)
        conn.assert_called_once_with(
            "omnibot",
            transport=None,
            runtime="auto",
            safety_config=None,
            estop_damp=False,
            read_only=False,
        )

    def test_every_quiet_command_resolves_and_a_bad_config_exits_2(self):
        with self._home() as tmp:
            with mock.patch.dict(os.environ, {"OHHO_HOME": tmp}):
                os.environ.pop("OHHO_ROBOT", None)
                calls = {"n": 0}
                real = resolve_robot_id

                def _spy(*args, **kwargs):
                    calls["n"] += 1
                    return real(*args, **kwargs)

                with mock.patch("ohho.cli.resolve_robot_id", side_effect=_spy):
                    out = io.StringIO()
                    with redirect_stdout(out), redirect_stderr(io.StringIO()):
                        self.assertEqual(main(["version"]), 0)
                        self.assertEqual(main(["list"]), 0)
                        self.assertEqual(main(["doctor"]), 0)
                        self.assertEqual(main(["profile", "list"]), 0)
                        self.assertEqual(main(["market", "list"]), 0)
                        self.assertEqual(main(["config", "show"]), 0)
                        self.assertEqual(main(["sim-serve", "--backend", "isaac"]), 1)
                self.assertGreaterEqual(calls["n"], 7)

                _write(Path(tmp) / ".ohho" / "config.toml", "robot = [\n")
                err = io.StringIO()
                with redirect_stderr(err):
                    self.assertEqual(main(["version"]), 2)
                self.assertIn("error:", err.getvalue())

                _write(
                    Path(tmp) / ".ohho" / "robots" / "omnibot.toml",
                    "[safety]\nmax_vx = 1.0\n",
                )
                _write(Path(tmp) / ".ohho" / "config.toml", 'robot = "omnibot"\n')
                err = io.StringIO()
                with redirect_stderr(err):
                    self.assertEqual(main(["config", "show"]), 2)
                self.assertIn("max_vx", err.getvalue())

    def test_sim_without_a_robot_flag_uses_the_resolver(self):
        with self._home() as tmp:
            with mock.patch.dict(os.environ, {"OHHO_HOME": tmp}):
                os.environ.pop("OHHO_ROBOT", None)
                out = io.StringIO()
                with redirect_stdout(out):
                    code = main(["sim", "--seconds", "0"])
                self.assertEqual(code, 0)
                self.assertIn("Generic Sim Bot", out.getvalue())


if __name__ == "__main__":
    unittest.main()
