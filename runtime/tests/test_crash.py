"""Crash bundles: one directory per crash, redaction, and retention."""

from __future__ import annotations

import io
import json
import os
import unittest
from pathlib import Path

from ohho.registry import RobotSpec
from ohho.robot_spec import AerialLimits, RobotClass

from ohho_runtime.crash import (
    BUNDLE_FILES,
    CRASH_DIR_NAME,
    STATE_DIR_ENV,
    CrashBundleError,
    classify_crash,
    default_state_dir,
    resolve_state_dir,
    write_crash_bundle,
)
from ohho_runtime.log import RuntimeLog
from ohho_runtime.manifest import (
    ROLE_TRANSPORT,
    Criticality,
    ManifestError,
    PythonCallable,
    RestartKind,
    RestartPolicy,
    class_default_manifest,
    load_manifest,
    make_service,
    resolve_manifest,
    with_services,
)
from ohho_runtime.metrics import Registry
from ohho_runtime.service import LifecycleState, ManualClock, ServiceBase
from ohho_runtime.supervisor import Supervisor
from ohho_runtime.timing import LoopTimer

_README = Path(__file__).resolve().parents[1] / "README.md"
_ARMING = "I am physically present, the area is clear"


def _noop() -> None:
    return None


def _service(
    role: str = "recorder",
    *,
    readiness: float = 1000.0,
    liveness: float = 1000.0,
    max_restarts: int = 3,
) -> object:
    return make_service(
        role,
        PythonCallable(_noop),
        criticality=Criticality.DEGRADABLE,
        readiness_timeout_s=readiness,
        liveness_timeout_s=liveness,
        restart=RestartPolicy(
            RestartKind.ON_FAILURE,
            max_restarts=max_restarts,
            window_s=60.0,
            backoff_s=0.5,
        ),
        loop_hz=1.0,
    )


def _manifest(
    *,
    readiness: float = 1000.0,
    liveness: float = 1000.0,
    max_restarts: int = 3,
    state_dir: str = "",
):
    return load_manifest(
        (_service(readiness=readiness, liveness=liveness, max_restarts=max_restarts),),
        state_dir=state_dir,
    )


class _Boom(ServiceBase):
    def on_start(self) -> None:
        raise RuntimeError("boom")


class _BeatOnce(ServiceBase):
    def on_start(self) -> None:
        self.mark_ready()
        self.heartbeat()


class _Exit(ServiceBase):
    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self.exit_code = 7

    def on_start(self) -> None:
        self.mark_ready()
        self.heartbeat()

    def crash(self) -> None:
        self._fail("child exited")


class _Hold(ServiceBase):
    def on_start(self) -> None:
        self.mark_ready()
        self.heartbeat()

    def crash(self) -> None:
        self._fail("crashed")


class _Idle(ServiceBase):
    def on_start(self) -> None:
        return None


def _factory(kind: type[ServiceBase]):
    class Factory:
        def __call__(self, spec, *, clock, grace_s):  # type: ignore[no-untyped-def]
            return kind(
                clock=clock,
                readiness_timeout_s=spec.readiness_timeout_s,
                liveness_timeout_s=spec.liveness_timeout_s,
                grace_s=grace_s,
            )

    return Factory()


def _bundles(root: Path) -> list[Path]:
    crashes = root / CRASH_DIR_NAME
    if not crashes.is_dir():
        return []
    return sorted(path for path in crashes.iterdir() if path.is_dir())


def _load(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise AssertionError(path.name)
    return payload


def _text(path: Path) -> str:
    return "".join(
        item.read_text(encoding="utf-8") for item in path.rglob("*") if item.is_file()
    )


def _size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


class ResolveStateDirTest(unittest.TestCase):
    def test_explicit_wins_over_env_and_manifest(self) -> None:
        path = resolve_state_dir(
            explicit="explicit-dir",
            manifest_dir="manifest-dir",
            environ={STATE_DIR_ENV: "env-dir"},
        )
        self.assertEqual(path, Path("explicit-dir"))

    def test_env_overrides_manifest(self) -> None:
        path = resolve_state_dir(
            manifest_dir="manifest-dir",
            environ={STATE_DIR_ENV: "env-dir"},
        )
        self.assertEqual(path, Path("env-dir"))

    def test_manifest_is_used_when_env_is_empty(self) -> None:
        path = resolve_state_dir(manifest_dir="manifest-dir", environ={})
        self.assertEqual(path, Path("manifest-dir"))

    def test_default_is_the_runtime_state_dir(self) -> None:
        path = resolve_state_dir(manifest_dir="", environ={})
        self.assertEqual(path, default_state_dir())
        self.assertEqual(path, Path.home() / ".ohho" / "runtime")


class ClassifyCrashTest(unittest.TestCase):
    def test_reasons(self) -> None:
        self.assertEqual(
            classify_crash(
                detail="crashed", exc=None, exit_code=3, outcome="restart_scheduled"
            ),
            "nonzero_exit",
        )
        self.assertEqual(
            classify_crash(
                detail="start failed: boom",
                exc=RuntimeError("boom"),
                exit_code=None,
                outcome="restart_scheduled",
            ),
            "unhandled_exception",
        )
        self.assertEqual(
            classify_crash(
                detail="liveness timeout",
                exc=None,
                exit_code=None,
                outcome="restart_scheduled",
            ),
            "heartbeat_timeout",
        )
        self.assertEqual(
            classify_crash(
                detail="liveness timeout",
                exc=None,
                exit_code=None,
                outcome="restart_limit",
            ),
            "restart_limit",
        )
        self.assertIsNone(
            classify_crash(
                detail="liveness timeout",
                exc=None,
                exit_code=None,
                outcome="restart_refused",
            )
        )
        self.assertIsNone(
            classify_crash(
                detail="readiness timeout",
                exc=None,
                exit_code=None,
                outcome="restart_scheduled",
            )
        )
        self.assertIsNone(
            classify_crash(
                detail="crashed",
                exc=None,
                exit_code=None,
                outcome="restart_scheduled",
            )
        )


class RedactionRetentionTest(unittest.TestCase):
    def _logger(self) -> tuple[io.StringIO, RuntimeLog]:
        buf = io.StringIO()
        logger = RuntimeLog(
            service="supervisor",
            robot_class="mobile_base",
            run_id="run-1",
            stream=buf,
        )
        return buf, logger

    def test_on_demand_bundle_redacts_and_counts(self) -> None:
        import tempfile

        buf, logger = self._logger()
        registry = Registry()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            secret = "plain-value-should-hide"
            token = "super-secret-token"
            bundle = write_crash_bundle(
                service="recorder",
                run_id="run-1",
                reason="requested",
                state_dir=str(root),
                timestamp="20261021T010203000000Z",
                manifest={
                    "services": [
                        {
                            "role": "recorder",
                            "loop_hz": 1.0,
                            "api_token": secret,
                            "note": _ARMING,
                        }
                    ],
                    "password": "hunter2",
                },
                service_logs=[
                    {
                        "timestamp": "2026-10-21T01:02:03.000000Z",
                        "level": "ERROR",
                        "service": "recorder",
                        "event": "note",
                        "robot_class": "mobile_base",
                        "run_id": "run-1",
                        "message": f"Authorization: Bearer {token}",
                    }
                ],
                supervisor_logs=[
                    {
                        "timestamp": "2026-10-21T01:02:03.000000Z",
                        "level": "INFO",
                        "service": "supervisor",
                        "event": "supervisor_start",
                        "robot_class": "mobile_base",
                        "run_id": "run-1",
                        "message": "sk-abcdefghijklmnop",
                    }
                ],
                history=["created", "failed"],
                state="failed",
                restarts=1,
                failure_reason="boom",
                metrics_text="# ok\n",
                robot_class="mobile_base",
                logger=logger,
                metrics=registry,
                environ={},
            )
            names = sorted(path.name for path in bundle.path.iterdir())
            self.assertEqual(names, sorted(BUNDLE_FILES))
            body = _text(bundle.path)
            self.assertNotIn(secret, body)
            self.assertNotIn("hunter2", body)
            self.assertNotIn("physically present", body)
            self.assertNotIn(token, body)
            self.assertNotIn("sk-abcdefghijklmnop", body)
            manifest = _load(bundle.path / "manifest.json")
            services = manifest["services"]
            self.assertIsInstance(services, list)
            service = services[0]
            self.assertIsInstance(service, dict)
            self.assertEqual(service["role"], "recorder")
            self.assertEqual(service["loop_hz"], 1.0)
            self.assertEqual(service["api_token"], "***")
            self.assertIn('"event":"crash_bundle"', buf.getvalue())
            text = registry.render()
            self.assertIn('reason="requested"', text)
            self.assertIn('service="recorder"', text)
            self.assertEqual(
                [
                    sample
                    for name, _labels, sample in _samples(text)
                    if name == "ohho_runtime_crash_bundles_total"
                ],
                [1.0],
            )

    def test_count_cap_drops_the_oldest(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            written: list[Path] = []
            for index in range(3):
                bundle = write_crash_bundle(
                    service="recorder",
                    run_id=f"run-{index}",
                    state_dir=str(root),
                    timestamp=f"20261021T00000000000{index}Z",
                    metrics=Registry(),
                    logger=self._logger()[1],
                    environ={},
                    max_bundles=2,
                    max_bytes=8 * 1024 * 1024,
                )
                written.append(bundle.path)
            remaining = {path.name for path in _bundles(root)}
            self.assertEqual(len(remaining), 2)
            self.assertNotIn(written[0].name, remaining)
            self.assertIn(written[2].name, remaining)

    def test_size_cap_drops_the_oldest(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            logger = self._logger()[1]
            first = write_crash_bundle(
                service="recorder",
                run_id="run-0",
                state_dir=str(root),
                timestamp="20261021T000000000000Z",
                service_logs=[_fat("recorder", "a")],
                metrics=Registry(),
                logger=logger,
                environ={},
                max_bundles=10,
                max_bytes=8 * 1024 * 1024,
            )
            cap = max(1, _size(first.path) // 2)
            second = write_crash_bundle(
                service="recorder",
                run_id="run-1",
                state_dir=str(root),
                timestamp="20261021T000000000001Z",
                service_logs=[_fat("recorder", "b")],
                metrics=Registry(),
                logger=logger,
                environ={},
                max_bundles=10,
                max_bytes=cap,
            )
            third = write_crash_bundle(
                service="recorder",
                run_id="run-2",
                state_dir=str(root),
                timestamp="20261021T000000000002Z",
                service_logs=[_fat("recorder", "c")],
                metrics=Registry(),
                logger=logger,
                environ={},
                max_bundles=10,
                max_bytes=cap,
            )
            remaining = _bundles(root)
            self.assertEqual([path.name for path in remaining], [third.path.name])
            self.assertFalse(first.path.exists())
            self.assertFalse(second.path.exists())
            self.assertTrue(third.path.is_dir())


def _fat(service: str, marker: str) -> dict[str, object]:
    return {
        "timestamp": "2026-10-21T00:00:00.000000Z",
        "level": "INFO",
        "service": service,
        "event": "note",
        "robot_class": "mobile_base",
        "run_id": "run",
        "message": marker + ("x" * 4000),
    }


def _samples(text: str) -> list[tuple[str, dict[str, str], float]]:
    found: list[tuple[str, dict[str, str], float]] = []
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        if "{" in line:
            name, rest = line.split("{", 1)
            label_text, value_text = rest.split("} ", 1)
            labels = {}
            for part in label_text.split(","):
                key, raw = part.split("=", 1)
                labels[key] = raw[1:-1]
            found.append((name, labels, float(value_text)))
        else:
            name, value_text = line.rsplit(" ", 1)
            found.append((name, {}, float(value_text)))
    return found


class SupervisorCrashTest(unittest.TestCase):
    def _supervisor(
        self,
        root: Path,
        kind: type[ServiceBase],
        *,
        readiness: float = 1000.0,
        liveness: float = 1000.0,
        max_restarts: int = 3,
        state_dir: str = "",
        crash_dir: str | None = "",
    ) -> tuple[Supervisor, io.StringIO, Registry, ManualClock]:
        buf = io.StringIO()
        logger = RuntimeLog(
            service="supervisor",
            robot_class="mobile_base",
            run_id="run-1",
            stream=buf,
        )
        registry = Registry()
        clock = ManualClock()
        explicit = str(root) if crash_dir == "" else crash_dir
        sup = Supervisor(
            _manifest(
                readiness=readiness,
                liveness=liveness,
                max_restarts=max_restarts,
                state_dir=state_dir,
            ),
            robot_class=RobotClass.MOBILE_BASE,
            clock=clock,
            service_factory=_factory(kind),
            logger=logger,
            metrics=registry,
            run_id="run-1",
            metrics_enabled=False,
            crash_dir=explicit,
        )
        return sup, buf, registry, clock

    def test_unhandled_exception_writes_exactly_one_bundle(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sup, buf, registry, _clock = self._supervisor(root, _Boom)
            sup.start()
            bundles = _bundles(root)
            self.assertEqual(len(bundles), 1)
            bundle = bundles[0]
            self.assertEqual(
                sorted(path.name for path in bundle.iterdir()), sorted(BUNDLE_FILES)
            )
            self.assertIn("run-1", bundle.name)
            self.assertTrue(bundle.name.startswith("recorder-"))
            failure = _load(bundle / "failure.json")
            self.assertEqual(failure["reason"], "unhandled_exception")
            self.assertIn("RuntimeError", str(failure["traceback"]))
            self.assertIn("boom", str(failure["traceback"]))
            state = _load(bundle / "state.json")
            self.assertEqual(state["service"], "recorder")
            self.assertEqual(state["restarts"], 0)
            self.assertIn("failed", state["history"])
            self.assertEqual(state["state"], LifecycleState.FAILED.value)
            loop = _load(bundle / "loop.json")
            self.assertIs(loop["attached"], False)
            manifest = _load(bundle / "manifest.json")
            services = manifest["services"]
            self.assertIsInstance(services, list)
            self.assertEqual(services[0]["role"], "recorder")  # type: ignore[index]
            platform_info = _load(bundle / "platform.json")
            self.assertIn("python_version", platform_info)
            self.assertIn("ohho_version", platform_info)
            self.assertIn("ohho_runtime_version", platform_info)
            self.assertIn(
                "ohho_runtime_service_state",
                (bundle / "metrics.txt").read_text("utf-8"),
            )
            service_logs = (bundle / "logs-service.jsonl").read_text(encoding="utf-8")
            supervisor_logs = (bundle / "logs-supervisor.jsonl").read_text(
                encoding="utf-8"
            )
            self.assertIn("recorder", service_logs)
            self.assertIn("supervisor", supervisor_logs)
            self.assertIn('"event":"crash_bundle"', buf.getvalue())
            self.assertIn('reason="unhandled_exception"', registry.render())
            self.assertIs(sup.service("recorder").state, LifecycleState.FAILED)

    def test_heartbeat_timeout_that_restarts_writes_loop_stats(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sup, _buf, _registry, clock = self._supervisor(
                root, _BeatOnce, liveness=1.0
            )
            sup.start()
            self.assertIs(sup.service("recorder").state, LifecycleState.READY)
            loop_clock = ManualClock()
            timer = LoopTimer(0.01, clock=loop_clock)
            timer.run(lambda: loop_clock.advance(0.02))
            sup.observe_loop("recorder", timer)
            clock.advance(1.0)
            sup.poll()
            bundles = _bundles(root)
            self.assertEqual(len(bundles), 1)
            failure = _load(bundles[0] / "failure.json")
            self.assertEqual(failure["reason"], "heartbeat_timeout")
            loop = _load(bundles[0] / "loop.json")
            self.assertIs(loop["attached"], True)
            stats = loop["stats"]
            self.assertIsInstance(stats, dict)
            self.assertGreater(stats["samples"], 0)
            self.assertIn("p50_s", stats)
            self.assertTrue(sup.events.of("restart_scheduled"))

    def test_nonzero_exit_writes_the_exit_code(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sup, _buf, _registry, _clock = self._supervisor(root, _Exit)
            sup.start()
            service = sup.service("recorder")
            self.assertIsInstance(service, _Exit)
            service.crash()
            sup.poll()
            bundles = _bundles(root)
            self.assertEqual(len(bundles), 1)
            failure = _load(bundles[0] / "failure.json")
            self.assertEqual(failure["reason"], "nonzero_exit")
            self.assertEqual(failure["exit_code"], 7)

    def test_restart_limit_writes_one_bundle(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sup, _buf, _registry, clock = self._supervisor(root, _Hold, max_restarts=3)
            sup.start()
            for index in range(3):
                service = sup.service("recorder")
                self.assertIsInstance(service, _Hold)
                service.crash()
                sup.poll()
                if index < 2:
                    self.assertEqual(_bundles(root), [])
                    delay = float(sup.events.of("restart_scheduled")[-1].detail)
                    clock.advance(delay)
                    sup.poll()
                    self.assertIs(sup.service("recorder").state, LifecycleState.READY)
            bundles = _bundles(root)
            self.assertEqual(len(bundles), 1)
            failure = _load(bundles[0] / "failure.json")
            self.assertEqual(failure["reason"], "restart_limit")
            state = _load(bundles[0] / "state.json")
            self.assertEqual(state["restarts"], 2)
            self.assertIs(state["restart_exhausted"], True)

    def test_readiness_timeout_does_not_write_a_bundle(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sup, _buf, _registry, clock = self._supervisor(root, _Idle, readiness=1.0)
            sup.start()
            clock.advance(1.0)
            sup.poll()
            self.assertIs(sup.service("recorder").state, LifecycleState.FAILED)
            self.assertTrue(sup.events.of("restart_scheduled"))
            self.assertEqual(_bundles(root), [])

    def test_manifest_state_dir_is_used_when_env_is_unset(self) -> None:
        import tempfile

        previous = os.environ.get(STATE_DIR_ENV)
        os.environ.pop(STATE_DIR_ENV, None)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                sup, _buf, _registry, _clock = self._supervisor(
                    root,
                    _Boom,
                    state_dir=str(root),
                    crash_dir=None,
                )
                sup.start()
                self.assertEqual(len(_bundles(root)), 1)
        finally:
            if previous is None:
                os.environ.pop(STATE_DIR_ENV, None)
            else:
                os.environ[STATE_DIR_ENV] = previous

    def test_env_overrides_manifest_state_dir(self) -> None:
        import tempfile

        previous = os.environ.get(STATE_DIR_ENV)
        try:
            with tempfile.TemporaryDirectory() as manifest_tmp:
                with tempfile.TemporaryDirectory() as env_tmp:
                    os.environ[STATE_DIR_ENV] = env_tmp
                    sup, _buf, _registry, _clock = self._supervisor(
                        Path(manifest_tmp),
                        _Boom,
                        state_dir=manifest_tmp,
                        crash_dir=None,
                    )
                    sup.start()
                    self.assertEqual(_bundles(Path(manifest_tmp)), [])
                    self.assertEqual(len(_bundles(Path(env_tmp))), 1)
        finally:
            if previous is None:
                os.environ.pop(STATE_DIR_ENV, None)
            else:
                os.environ[STATE_DIR_ENV] = previous


class ManifestStateDirTest(unittest.TestCase):
    def test_state_dir_is_stored_and_copied(self) -> None:
        manifest = load_manifest((_service(),), state_dir="runtime-state")
        self.assertEqual(manifest.state_dir, "runtime-state")
        copied = with_services(manifest, _service("perception"))
        self.assertEqual(copied.state_dir, "runtime-state")
        with self.assertRaises(ManifestError):
            load_manifest((_service(),), state_dir=1)  # type: ignore[arg-type]

    def test_aerial_gate_keeps_state_dir(self) -> None:
        spec = RobotSpec(
            id="flyer-experimental",
            name="Flyer",
            category="aerial",
            limits=AerialLimits(flight_ready=True, experimental=True),
        )
        base = class_default_manifest(RobotClass.AERIAL)
        manifest = load_manifest(base.services, state_dir="runtime-state")

        class Plugin:
            def services(self, robot: RobotSpec):
                del robot
                return manifest

        resolved = resolve_manifest(spec, Plugin())
        self.assertEqual(resolved.state_dir, "runtime-state")
        self.assertTrue(resolved.service(ROLE_TRANSPORT).do_not_operate)


class PublicApiTest(unittest.TestCase):
    def test_package_exports_the_writer(self) -> None:
        import ohho_runtime

        self.assertIs(ohho_runtime.write_crash_bundle, write_crash_bundle)

    def test_readme_documents_the_bundle_layout(self) -> None:
        text = _README.read_text(encoding="utf-8")
        self.assertIn("OHHO_RUNTIME_STATE_DIR", text)
        self.assertIn("`ohho_runtime_crash_bundles_total`", text)
        self.assertIn("crash_bundle", text)
        for name in BUNDLE_FILES:
            self.assertIn(f"`{name}`", text)
        self.assertIn("~/.ohho/runtime", text)

    def test_rejects_a_blank_service(self) -> None:
        with self.assertRaises(CrashBundleError):
            write_crash_bundle(service=" ", run_id="run-1", metrics=Registry())


if __name__ == "__main__":
    unittest.main()
