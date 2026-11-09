"""Prometheus text, loop timing, and the /metrics endpoint."""

from __future__ import annotations

import time
import unittest
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

from ohho.robot_spec import RobotClass

from ohho_runtime.manifest import (
    Criticality,
    PythonCallable,
    RestartKind,
    RestartPolicy,
    load_manifest,
    make_service,
)
from ohho_runtime.metrics import (
    DEFAULT_METRICS_HOST,
    DEFAULT_METRICS_PORT,
    METRIC_NAMES,
    METRICS_ENABLED_ENV,
    METRICS_PORT_ENV,
    PROMETHEUS_CONTENT_TYPE,
    MetricsEndpoint,
    Registry,
    resolve_metrics_bind,
)
from ohho_runtime.service import ManualClock, ServiceBase
from ohho_runtime.service.lifecycle import LifecycleState
from ohho_runtime.supervisor import Supervisor
from ohho_runtime.timing import LoopTimer

_README = Path(__file__).resolve().parents[1] / "README.md"


def _noop() -> None:
    return None


def _spec(
    role: str, *, restart: RestartPolicy | None = None, criticality=Criticality.CRITICAL
):
    return make_service(
        role,
        PythonCallable(_noop),
        criticality=criticality,
        readiness_timeout_s=1000.0,
        liveness_timeout_s=1000.0,
        restart=RestartPolicy() if restart is None else restart,
        loop_hz=1.0,
    )


def _samples(text: str) -> list[tuple[str, dict[str, str], float]]:
    found: list[tuple[str, dict[str, str], float]] = []
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        if "{" in line:
            name, rest = line.split("{", 1)
            label_text, value_text = rest.split("} ", 1)
            labels: dict[str, str] = {}
            for part in label_text.split(","):
                key, raw = part.split("=", 1)
                labels[key] = raw[1:-1]
            found.append((name, labels, float(value_text)))
        else:
            name, value_text = line.rsplit(" ", 1)
            found.append((name, {}, float(value_text)))
    return found


def _values(text: str, name: str, **labels: str) -> list[float]:
    matches: list[float] = []
    for metric, got, value in _samples(text):
        if metric != name:
            continue
        if all(got.get(key) == expected for key, expected in labels.items()):
            matches.append(value)
    return matches


class MetricsTextTest(unittest.TestCase):
    def test_catalog_names_and_types_are_in_the_exposition(self) -> None:
        text = Registry().render()
        self.assertTrue(text.endswith("\n"))
        for name in METRIC_NAMES:
            self.assertIn(f"# HELP {name} ", text)
            self.assertIn(f"# TYPE {name} ", text)
        self.assertIn("# TYPE ohho_runtime_service_restarts_total counter", text)
        self.assertIn("# TYPE ohho_runtime_loop_cycle_seconds gauge", text)
        self.assertIn("# TYPE ohho_runtime_exceptions_total counter", text)

    def test_readme_documents_every_metric_and_the_scrape(self) -> None:
        text = _README.read_text(encoding="utf-8")
        for name in METRIC_NAMES:
            self.assertIn(f"`{name}`", text)
        self.assertIn("127.0.0.1:9470", text)
        self.assertIn("OHHO_RUNTIME_METRICS", text)
        self.assertIn("quantile", text)
        self.assertIn(PROMETHEUS_CONTENT_TYPE, text)


class MetricsBindTest(unittest.TestCase):
    def test_default_is_disabled_loopback(self) -> None:
        bind = resolve_metrics_bind(environ={})
        self.assertFalse(bind.enabled)
        self.assertEqual(bind.host, DEFAULT_METRICS_HOST)
        self.assertEqual(bind.host, "127.0.0.1")
        self.assertEqual(bind.port, DEFAULT_METRICS_PORT)
        self.assertEqual(bind.port, 9470)

    def test_env_enables_and_sets_port(self) -> None:
        bind = resolve_metrics_bind(
            environ={
                METRICS_ENABLED_ENV: "1",
                METRICS_PORT_ENV: "18080",
                "OHHO_RUNTIME_METRICS_HOST": "127.0.0.1",
            }
        )
        self.assertTrue(bind.enabled)
        self.assertEqual(bind.host, "127.0.0.1")
        self.assertEqual(bind.port, 18080)

    def test_explicit_disable_wins_over_env(self) -> None:
        bind = resolve_metrics_bind(
            enabled=False,
            environ={METRICS_ENABLED_ENV: "on"},
        )
        self.assertFalse(bind.enabled)

    def test_bad_port_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            resolve_metrics_bind(port=70000, environ={})
        with self.assertRaises(ValueError):
            resolve_metrics_bind(environ={METRICS_PORT_ENV: "nope"})


class MetricsEndpointTest(unittest.TestCase):
    def test_disabled_does_not_bind(self) -> None:
        endpoint = MetricsEndpoint(
            Registry(),
            enabled=False,
            host="127.0.0.1",
            port=1,
            environ={},
        )
        endpoint.start()
        self.addCleanup(endpoint.stop)
        self.assertFalse(endpoint.enabled)
        self.assertEqual(endpoint.host, "127.0.0.1")
        self.assertFalse(endpoint.listening)
        self.assertIsNone(endpoint.address)

    def test_enabled_binds_loopback_and_serves_metrics(self) -> None:
        registry = Registry()
        registry.set(
            "ohho_runtime_allows_motion",
            1,
            {"robot_class": "arm", "run_id": "run-1"},
        )
        endpoint = MetricsEndpoint(
            registry,
            enabled=True,
            host="127.0.0.1",
            port=0,
            environ={},
        )
        self.addCleanup(endpoint.stop)
        status, content_type, body = endpoint.handle_path("/metrics")
        self.assertEqual(status, 200)
        self.assertEqual(content_type, PROMETHEUS_CONTENT_TYPE)
        self.assertIn(b"ohho_runtime_allows_motion", body)
        missing, _, _ = endpoint.handle_path("/health")
        self.assertEqual(missing, 404)

        endpoint.start()
        address = endpoint.address
        self.assertIsNotNone(address)
        assert address is not None
        self.assertEqual(address[0], "127.0.0.1")
        self.assertGreater(address[1], 0)
        self.assertNotEqual(address[1], 9470)

        deadline = time.monotonic() + 2.0
        payload = ""
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            try:
                with urlopen(
                    f"http://127.0.0.1:{address[1]}/metrics",
                    timeout=0.5,
                ) as response:
                    self.assertEqual(response.status, 200)
                    header = response.headers.get("Content-Type", "")
                    self.assertIn("version=0.0.4", header)
                    payload = response.read().decode("utf-8")
                break
            except (URLError, TimeoutError, ConnectionError, OSError) as exc:
                last_error = exc
                time.sleep(0.02)
        else:
            self.fail(f"metrics endpoint did not answer: {last_error}")
        self.assertIn('robot_class="arm"', payload)
        self.assertIn("ohho_runtime_allows_motion", payload)

    def test_scrape_failure_is_logged_and_counted(self) -> None:
        registry = Registry()

        def boom() -> None:
            raise RuntimeError("scrape broke")

        endpoint = MetricsEndpoint(registry, enabled=False, collect=boom, environ={})
        status, _, body = endpoint.handle_path("/metrics")
        self.assertEqual(status, 500)
        self.assertEqual(body, b"scrape failed\n")
        text = registry.render()
        self.assertIn('event="scrape_failed"', text)
        self.assertIn('exc_type="RuntimeError"', text)
        self.assertEqual(_values(text, "ohho_runtime_exceptions_total")[0], 1.0)


class _Once(ServiceBase):
    """Fails its first start, then becomes ready."""

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self._left = 1

    def on_start(self) -> None:
        if self._left:
            self._left -= 1
            raise RuntimeError("not yet")
        self.mark_ready()
        self.heartbeat()


class _OnceFactory:
    def __call__(self, spec, *, clock, grace_s):
        return _Once(
            clock=clock,
            readiness_timeout_s=spec.readiness_timeout_s,
            liveness_timeout_s=spec.liveness_timeout_s,
            grace_s=grace_s,
        )


class SupervisorMetricsTest(unittest.TestCase):
    def test_service_heartbeat_restart_and_loop_metrics(self) -> None:
        clock = ManualClock()
        registry = Registry()
        manifest = load_manifest(
            (
                _spec(
                    "recorder",
                    criticality=Criticality.DEGRADABLE,
                    restart=RestartPolicy(
                        RestartKind.ON_FAILURE,
                        max_restarts=3,
                        window_s=60.0,
                        backoff_s=0.5,
                    ),
                ),
            )
        )
        sup = Supervisor(
            manifest,
            robot_class=RobotClass.MOBILE_BASE,
            clock=clock,
            metrics=registry,
            run_id="run-1",
            metrics_enabled=False,
            service_factory=_OnceFactory(),
        )
        sup.start()
        self.assertEqual(sup.metrics_endpoint.address, None)
        clock.advance(0.5)
        sup.poll()
        self.assertEqual(sup.restart_count("recorder"), 1)
        self.assertIs(sup.service("recorder").state, LifecycleState.READY)
        clock.advance(1.25)

        loop_clock = ManualClock()
        timer = LoopTimer(0.01, clock=loop_clock)

        def slow() -> None:
            loop_clock.advance(0.05)

        timer.run(slow)
        sup.observe_loop("recorder", timer)
        text = sup.metrics_text()

        labels = {
            "service": "recorder",
            "robot_class": "mobile_base",
            "run_id": "run-1",
        }
        self.assertEqual(
            _values(text, "ohho_runtime_service_restarts_total", **labels),
            [1.0],
        )
        self.assertEqual(
            _values(text, "ohho_runtime_service_state", **labels, state="ready"),
            [1.0],
        )
        age = _values(text, "ohho_runtime_heartbeat_age_seconds", **labels)
        self.assertEqual(len(age), 1)
        self.assertAlmostEqual(age[0], 1.25)
        self.assertEqual(
            _values(text, "ohho_runtime_heartbeat_misses_total", **labels),
            [0.0],
        )
        self.assertGreater(
            _values(text, "ohho_runtime_start_duration_seconds", **labels)[0],
            -1.0,
        )
        self.assertEqual(
            _values(text, "ohho_runtime_do_not_operate", run_id="run-1"), [0.0]
        )
        self.assertEqual(
            _values(
                text,
                "ohho_runtime_supervisor_state",
                state="running",
                run_id="run-1",
            ),
            [1.0],
        )
        for quantile in ("0.5", "0.95", "0.99"):
            cycle = _values(
                text,
                "ohho_runtime_loop_cycle_seconds",
                service="recorder",
                quantile=quantile,
            )
            self.assertEqual(len(cycle), 1)
            self.assertGreater(cycle[0], 0.0)
        self.assertGreater(
            _values(text, "ohho_runtime_loop_deadline_misses_total", **labels)[0],
            0.0,
        )
        self.assertEqual(
            len(_values(text, "ohho_runtime_loop_jitter_seconds", **labels)),
            1,
        )
        self.assertEqual(
            _values(text, "ohho_runtime_loop_samples", **labels),
            [1.0],
        )

        sup.shutdown()
        text = sup.metrics_text()
        self.assertEqual(
            _values(text, "ohho_runtime_service_state", **labels, state="stopped"),
            [1.0],
        )
        self.assertGreaterEqual(
            _values(text, "ohho_runtime_stop_duration_seconds", **labels)[0],
            0.0,
        )
        self.assertIsNone(sup.metrics_endpoint.address)


if __name__ == "__main__":
    unittest.main()
