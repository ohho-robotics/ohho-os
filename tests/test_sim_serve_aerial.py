"""Aerial teleop on ``ohho sim-serve`` (OHH-289).

Ground-robot cases stay in ``test_sim_serve.py``. These tests use a fake
aircraft or the in-process drone sim. They do not set ``OHHO_ARM_HARDWARE``
and they do not arm a SafetyGate.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import time
import unittest
from contextlib import contextmanager
from typing import Any, Iterator, Optional

from ohho.aerial import Frame
from ohho.safety import ARM_ENV, ARM_PHRASE, DRONE_TAKEOFF_ALT_M
from ohho.schema import Odometry, Telemetry
from ohho.sim_serve import (
    SETPOINT_SILENCE_S,
    SimServe,
    is_sim_transport_uri,
    normalize_phrase,
    parse_frame,
    parse_takeoff_altitude,
    parse_twist,
    parse_velocity,
    privileged_flight_block,
    requires_flight_confirmation,
)


def _has_websockets() -> bool:
    return importlib.util.find_spec("websockets") is not None


@contextmanager
def _arm_env_absent() -> Iterator[None]:
    """Hide a pre-existing arm variable for one test, then put it back.

    The test does not assign ``OHHO_ARM_HARDWARE``. Restoration only happens
    when the process already had the variable.
    """
    saved = os.environ.pop(ARM_ENV, None)
    try:
        yield
    finally:
        if saved is not None:
            os.environ[ARM_ENV] = saved


class _FakeAerial:
    """Records flight calls. ``arm`` is the gate; ``arm_flight`` is the autopilot."""

    def __init__(self) -> None:
        self.velocity: list[tuple[Any, ...]] = []
        self.flight: list[tuple[Any, ...]] = []
        self.gate_arm: list[str] = []
        self.estops = 0
        self.gated = False
        self.last_error: Optional[str] = None
        self.transport = self

    def has(self, capability: str) -> bool:
        return capability == "aerial"

    def drive(self, vx: float = 0.0, vy: float = 0.0, w: float = 0.0) -> None:
        self.velocity.append(("drive", vx, vy, w))

    def offboard_velocity(
        self, vx: float, vy: float, vz: float, yaw_rate: float, frame: Frame
    ) -> None:
        self.velocity.append((vx, vy, vz, yaw_rate, frame))

    def takeoff(self, alt_m: float) -> None:
        self.flight.append(("takeoff", alt_m))

    def land(self) -> None:
        self.flight.append(("land",))

    def rtl(self) -> None:
        self.flight.append(("rtl",))

    def hold(self) -> None:
        self.flight.append(("hold",))

    def arm(self) -> bool:
        self.gate_arm.append("arm")
        return True

    def arm_flight(self) -> None:
        self.flight.append(("arm_flight",))

    def disarm_flight(self) -> None:
        self.flight.append(("disarm_flight",))

    def emergency_stop(self) -> None:
        self.estops += 1

    def release_stop(self) -> None:
        return None

    def disconnect(self) -> None:
        return None

    def telemetry(self) -> Telemetry:
        return Telemetry(odom=Odometry())

    def read(self) -> Telemetry:
        return self.telemetry()


class TestAerialParsing(unittest.TestCase):
    def test_linear_z_is_climb_axis_and_bare_z_stays_yaw(self):
        vx, vy, vz, w = parse_twist(
            {"linear": {"x": 0.2, "y": -0.1, "z": 0.15}, "angular": {"z": 0.25}}
        )
        self.assertEqual((vx, vy, vz, w), (0.2, -0.1, 0.15, 0.25))
        # Ground helper still returns yaw in the third slot and drops climb.
        self.assertEqual(
            parse_velocity({"linear": {"x": 0.2, "z": 0.15}}), (0.2, 0.0, 0.0)
        )
        vx, vy, vz, w = parse_twist({"x": 0.4, "y": -0.2, "z": 0.1})
        self.assertEqual((vx, vy, vz, w), (0.4, -0.2, 0.0, 0.1))

    def test_dotted_and_flat_vertical(self):
        self.assertEqual(parse_twist({"linear.z": -0.4, "linear.x": 0.1})[2], -0.4)
        self.assertEqual(
            parse_twist({"linear_z": -0.2, "angular_z": 0.3})[2:], (-0.2, 0.3)
        )
        self.assertEqual(parse_twist({"linear": {"z": None}})[2], 0.0)
        self.assertEqual(parse_twist({"linear": {"z": "up"}})[2], 0.0)

    def test_frame_names_have_no_default(self):
        self.assertIs(parse_frame("BODY_FRD"), Frame.BODY_FRD)
        self.assertIs(parse_frame(" body_frd "), Frame.BODY_FRD)
        self.assertIs(parse_frame("local_ned"), Frame.LOCAL_NED)
        self.assertIs(parse_frame("LOCAL_ENU"), Frame.LOCAL_ENU)
        self.assertIsNone(parse_frame("ENU"))
        self.assertIsNone(parse_frame(""))
        self.assertIsNone(parse_frame(None))

    def test_confirmation_rules(self):
        self.assertFalse(requires_flight_confirmation("sim", None))
        self.assertFalse(requires_flight_confirmation("robot", "sim://"))
        self.assertFalse(requires_flight_confirmation("robot", "SIM://drone"))
        self.assertFalse(requires_flight_confirmation("ros2", "dds://eth0"))
        self.assertTrue(requires_flight_confirmation("robot", "dds://eth0"))
        self.assertTrue(
            requires_flight_confirmation("robot", "px4+dds://127.0.0.1:8888")
        )
        self.assertTrue(is_sim_transport_uri("simulated://local"))
        self.assertFalse(is_sim_transport_uri("dds://lo"))

        phrase = "  " + ARM_PHRASE.upper() + ". "
        self.assertEqual(normalize_phrase(phrase), normalize_phrase(ARM_PHRASE))
        # A mapping, not the process environment.
        allowed = privileged_flight_block(phrase, {ARM_ENV: "1"})
        self.assertEqual(allowed, "")
        self.assertEqual(privileged_flight_block(phrase, {}), "env")
        self.assertEqual(privileged_flight_block(phrase, {ARM_ENV: "0"}), "env")
        self.assertEqual(
            privileged_flight_block("release", {ARM_ENV: "1"}), "not_confirmed"
        )
        self.assertEqual(privileged_flight_block(None, {ARM_ENV: "1"}), "not_confirmed")

    def test_takeoff_altitude_ceiling(self):
        self.assertEqual(parse_takeoff_altitude(None), (DRONE_TAKEOFF_ALT_M, ""))
        self.assertEqual(parse_takeoff_altitude(2.5), (2.5, ""))
        self.assertEqual(parse_takeoff_altitude(10)[1], "altitude")
        self.assertEqual(parse_takeoff_altitude(0)[1], "bad_altitude")
        self.assertEqual(parse_takeoff_altitude(True)[1], "bad_altitude")
        self.assertEqual(parse_takeoff_altitude("high")[1], "bad_altitude")


class TestFlightPolicy(unittest.TestCase):
    """Flight requests without opening a socket."""

    def _server(
        self, backend: str, transport: Optional[str] = None
    ) -> tuple[SimServe, _FakeAerial]:
        robot = "px4-sitl"
        if backend == "robot" and transport is None:
            transport = "dds://eth0"
        server = SimServe(backend=backend, robot=robot, transport=transport)
        fake = _FakeAerial()
        server._aerial = True
        server._bot = fake
        return server, fake

    def test_sim_takeoff_without_phrase_does_not_arm_the_gate(self):
        server, fake = self._server("sim")
        server._on_flight({"request": "arm"})
        server._on_flight({"request": "takeoff"})
        self.assertEqual(
            fake.flight, [("arm_flight",), ("takeoff", DRONE_TAKEOFF_ALT_M)]
        )
        self.assertEqual(fake.gate_arm, [])
        self.assertTrue(server._flight_status["accepted"])

    def test_robot_sim_uri_does_not_require_the_phrase(self):
        server, fake = self._server("robot", "sim://")
        server._on_flight({"request": "takeoff", "alt_m": 1.5})
        self.assertEqual(fake.flight, [("takeoff", 1.5)])
        self.assertEqual(fake.gate_arm, [])

    def test_robot_backend_without_phrase_or_env_is_refused(self):
        with _arm_env_absent():
            server, fake = self._server("robot", "px4+dds://127.0.0.1:8888")
            server._on_flight(
                {
                    "request": "takeoff",
                    "alt_m": 2.5,
                    "confirm": ARM_PHRASE,
                    ARM_ENV: "1",
                    "env": {ARM_ENV: "1"},
                }
            )
            self.assertEqual(fake.flight, [])
            self.assertEqual(fake.gate_arm, [])
            self.assertFalse(server._flight_status["accepted"])
            self.assertEqual(server._flight_status["reason"], "env")
            self.assertIsNone(os.environ.get(ARM_ENV))

            server._on_flight({"request": "Takeoff", "alt_m": 2.5})
            self.assertEqual(server._flight_status["reason"], "env")
            server._on_flight({"request": "arm", "confirm": "no"})
            self.assertEqual(fake.flight, [])
            self.assertEqual(fake.gate_arm, [])

    def test_land_does_not_require_the_phrase(self):
        with _arm_env_absent():
            server, fake = self._server("robot", "dds://eth0")
            server._on_flight({"request": "land"})
            server._on_flight({"request": "rtl"})
            server._on_flight({"request": "hold"})
            server._on_flight({"request": "disarm"})
            self.assertEqual(
                [name for name, *_rest in fake.flight],
                ["land", "rtl", "hold", "disarm_flight"],
            )
            self.assertEqual(fake.gate_arm, [])

    def test_unknown_request_and_high_altitude_are_refused(self):
        server, fake = self._server("sim")
        server._on_flight({"request": "flip"})
        server._on_flight({"request": "force_disarm"})
        server._on_flight({"request": "takeoff", "alt_m": 12})
        self.assertEqual(fake.flight, [])
        self.assertEqual(server._flight_status["reason"], "altitude")
        self.assertEqual(fake.gate_arm, [])

    def test_gate_refusal_is_reported_and_does_not_count_as_accepted(self):
        server, fake = self._server("sim")
        fake.gated = True

        def _takeoff(alt_m: float) -> None:
            fake.flight.append(("takeoff", alt_m))
            fake.last_error = "not_armed"

        fake.takeoff = _takeoff  # type: ignore[method-assign]
        server._on_flight({"request": "takeoff", "alt_m": 2.5})
        self.assertFalse(server._flight_status["accepted"])
        self.assertEqual(server._flight_status["reason"], "not_armed")

    def test_ground_robot_ignores_flight(self):
        server = SimServe(backend="sim", robot="omnibot")
        fake = _FakeAerial()
        server._bot = fake
        server._on_flight({"request": "takeoff", "alt_m": 1})
        self.assertEqual(fake.flight, [])
        self.assertEqual(server._flight_status["reason"], "not_aerial")


@unittest.skipUnless(_has_websockets(), "websockets not installed")
class TestAerialWebSocket(unittest.TestCase):
    async def _serve(self, backend: str, transport: Optional[str], fake: _FakeAerial):
        import websockets
        from unittest.mock import patch

        server = SimServe(
            backend=backend,
            robot="px4-sitl",
            transport=transport,
            host="127.0.0.1",
            port=0,
            rate_hz=40.0,
        )
        with patch("ohho.robot.Robot.connect", return_value=fake):
            await server.start()
        return server, websockets

    def test_body_frame_sign_deadman_and_watchdog(self):
        """Positive BODY_FRD yaw is nose-right and is not negated.

        A mode-2 radio's left stick, rightward, and the VR aerial scheme both
        use that sign (AerialDriveScheme does not negate left-X). Positive
        linear.z is down; this server does not turn stick-up into climb.
        """
        fake = _FakeAerial()

        async def run() -> None:
            server, websockets = await self._serve("sim", None, fake)
            try:
                async with websockets.connect(
                    f"ws://127.0.0.1:{server.port}"
                ) as client:
                    await asyncio.wait_for(client.recv(), timeout=1.0)
                    # No frame: do not invent BODY_FRD and do not stream.
                    await client.send(
                        json.dumps(
                            {
                                "deadman": True,
                                "velocity": {
                                    "linear": {"x": 0.2, "y": 0.0, "z": -0.1},
                                    "angular": {"z": 0.2},
                                },
                            }
                        )
                    )
                    await asyncio.sleep(0.08)
                    self.assertEqual(fake.velocity, [])
                    self.assertEqual(fake.flight, [])

                    await client.send(
                        json.dumps(
                            {
                                "deadman": True,
                                "frame": "BODY_FRD",
                                "velocity": {
                                    "linear": {"x": 0.2, "y": -0.1, "z": 0.15},
                                    "angular": {"z": 0.25},
                                },
                            }
                        )
                    )
                    await asyncio.sleep(0.08)
                    self.assertTrue(fake.velocity)
                    self.assertNotIn("drive", {item[0] for item in fake.velocity})
                    sample = fake.velocity[-1]
                    self.assertEqual(sample[:4], (0.2, -0.1, 0.15, 0.25))
                    self.assertIs(sample[4], Frame.BODY_FRD)

                    # Deadman false does not land and does not keep this command.
                    await client.send(
                        json.dumps(
                            {
                                "deadman": False,
                                "frame": "BODY_FRD",
                                "velocity": {
                                    "linear": {"x": 9.0, "y": 0.0, "z": -9.0},
                                    "angular": {"z": 9.0},
                                },
                            }
                        )
                    )
                    await asyncio.sleep(0.08)
                    self.assertNotIn(9.0, {item[0] for item in fake.velocity})
                    self.assertEqual(fake.velocity[-1][:4], (0.0, 0.0, 0.0, 0.0))
                    self.assertNotIn("land", [name for name, *_rest in fake.flight])
                    self.assertEqual(fake.estops, 0)
                    quiet = len(fake.velocity)
                    await asyncio.sleep(0.25)
                    self.assertEqual(len(fake.velocity), quiet)

                    # Fresh non-zero, then 300 ms of silence: one zero, then stop.
                    await client.send(
                        json.dumps(
                            {
                                "deadman": True,
                                "frame": "BODY_FRD",
                                "velocity": {
                                    "linear": {"x": 0.4, "y": 0.0, "z": 0.0},
                                    "angular": {"z": 0.0},
                                },
                            }
                        )
                    )
                    await asyncio.sleep(0.08)
                    self.assertEqual(fake.velocity[-1][0], 0.4)
                    await asyncio.sleep(SETPOINT_SILENCE_S + 0.2)
                    nonzero = [
                        i for i, item in enumerate(fake.velocity) if item[0] == 0.4
                    ]
                    self.assertTrue(nonzero)
                    tail = fake.velocity[nonzero[-1] + 1 :]
                    self.assertEqual(tail, [(0.0, 0.0, 0.0, 0.0, Frame.BODY_FRD)])
                    stopped = len(fake.velocity)
                    await asyncio.sleep(0.25)
                    self.assertEqual(len(fake.velocity), stopped)
                    self.assertEqual(fake.estops, 0)
                    self.assertNotIn("land", [name for name, *_rest in fake.flight])

                    # A deadman heartbeat must not revive the cleared 0.4 m/s.
                    await client.send(
                        json.dumps({"deadman": True, "frame": "BODY_FRD"})
                    )
                    await asyncio.sleep(0.12)
                    revived = fake.velocity[stopped:]
                    self.assertTrue(revived)
                    self.assertTrue(all(item[0] == 0.0 for item in revived))
            finally:
                await server.stop()

        asyncio.run(run())

    def test_estop_calls_emergency_stop_and_stops_the_stream(self):
        fake = _FakeAerial()

        async def run() -> None:
            server, websockets = await self._serve("sim", None, fake)
            try:
                async with websockets.connect(
                    f"ws://127.0.0.1:{server.port}"
                ) as client:
                    await asyncio.wait_for(client.recv(), timeout=1.0)
                    await client.send(
                        json.dumps(
                            {
                                "deadman": True,
                                "frame": "BODY_FRD",
                                "velocity": {
                                    "linear": {"x": 0.3, "z": 0.0},
                                    "angular": {"z": 0.0},
                                },
                            }
                        )
                    )
                    await asyncio.sleep(0.05)
                    await client.send(
                        json.dumps(
                            {
                                "estop": True,
                                "deadman": True,
                                "frame": "BODY_FRD",
                                "velocity": {
                                    "linear": {"x": 1.0},
                                    "angular": {"z": 0.0},
                                },
                            }
                        )
                    )
                    await asyncio.sleep(0.08)
                    msg = None
                    while True:
                        try:
                            raw = await asyncio.wait_for(client.recv(), timeout=0.05)
                            msg = json.loads(raw)
                        except asyncio.TimeoutError:
                            break
                    self.assertIsNotNone(msg)
                    assert msg is not None
                    self.assertEqual(fake.estops, 1)
                    self.assertNotIn("land", [name for name, *_rest in fake.flight])
                    self.assertFalse(msg["streaming"])
                    self.assertNotIn(1.0, {item[0] for item in fake.velocity})
            finally:
                await server.stop()

        asyncio.run(run())

    def test_websocket_payload_cannot_set_env_or_arm_hardware(self):
        fake = _FakeAerial()

        async def run() -> None:
            server, websockets = await self._serve("robot", "dds://eth0", fake)
            before = dict(os.environ)
            try:
                async with websockets.connect(
                    f"ws://127.0.0.1:{server.port}"
                ) as client:
                    await asyncio.wait_for(client.recv(), timeout=1.0)
                    await client.send(
                        json.dumps(
                            {
                                ARM_ENV: "1",
                                "env": {ARM_ENV: "1"},
                                "environ": {ARM_ENV: "1"},
                                "arm": True,
                                "flight": {
                                    "request": "takeoff",
                                    "alt_m": 2.5,
                                    "confirm": ARM_PHRASE,
                                    ARM_ENV: "1",
                                },
                            }
                        )
                    )
                    await asyncio.sleep(0.08)
                    await client.send(
                        json.dumps(
                            {"flight": {"request": "arm", "confirm": ARM_PHRASE}}
                        )
                    )
                    await asyncio.sleep(0.08)
                    msg = None
                    while True:
                        try:
                            raw = await asyncio.wait_for(client.recv(), timeout=0.05)
                            msg = json.loads(raw)
                        except asyncio.TimeoutError:
                            break
                    self.assertIsNotNone(msg)
            finally:
                await server.stop()
            self.assertEqual(dict(os.environ), before)
            self.assertIsNone(os.environ.get(ARM_ENV))
            self.assertEqual(fake.gate_arm, [])
            self.assertEqual(fake.flight, [])
            assert msg is not None
            self.assertFalse(msg["flight_status"]["accepted"])
            self.assertEqual(msg["flight_status"]["reason"], "env")

        with _arm_env_absent():
            asyncio.run(run())


@unittest.skipUnless(_has_websockets(), "websockets not installed")
class TestDroneSimSession(unittest.TestCase):
    """In-process px4-sitl: deadman does not land, estop does."""

    def test_takeoff_deadman_and_estop(self):
        import websockets

        async def drain(client: Any, timeout: float = 0.05) -> Optional[dict[str, Any]]:
            latest = None
            while True:
                try:
                    raw = await asyncio.wait_for(client.recv(), timeout=timeout)
                    latest = json.loads(raw)
                    timeout = 0.01
                except asyncio.TimeoutError:
                    break
            return latest

        async def wait_until(
            client: Any, pred: Any, timeout: float = 6.0
        ) -> dict[str, Any]:
            deadline = time.monotonic() + timeout
            last: Optional[dict[str, Any]] = None
            while time.monotonic() < deadline:
                last = await drain(client)
                if last is not None and pred(last):
                    return last
                await asyncio.sleep(0.02)
            raise AssertionError(last)

        def drone(msg: dict[str, Any]) -> dict[str, Any]:
            body = msg.get("drone")
            if not isinstance(body, dict):
                raise AssertionError(body)
            return body

        async def run() -> None:
            server = SimServe(
                backend="sim",
                robot="px4-sitl",
                host="127.0.0.1",
                port=0,
                rate_hz=40.0,
            )
            await server.start()
            try:
                async with websockets.connect(
                    f"ws://127.0.0.1:{server.port}"
                ) as client:
                    first = await wait_until(
                        client,
                        lambda msg: drone(msg).get("landed_state") == "on_ground",
                    )
                    self.assertEqual(first["backend"], "sim")
                    self.assertTrue(first["drone"]["experimental"])
                    self.assertFalse(first["drone"]["flight_ready"])
                    self.assertEqual(first["drone"]["pose_frame"], "LOCAL_NED")

                    await client.send(json.dumps({"flight": {"request": "arm"}}))
                    armed = await wait_until(
                        client, lambda msg: drone(msg).get("armed") is True
                    )
                    self.assertTrue(armed["flight_status"]["accepted"])

                    await client.send(
                        json.dumps({"flight": {"request": "takeoff", "alt_m": 0.35}})
                    )
                    holding = await wait_until(
                        client,
                        lambda msg: (
                            drone(msg).get("flight_mode") == "hold"
                            and drone(msg).get("landed_state") == "in_air"
                        ),
                    )
                    self.assertLess(holding["drone"]["pose"]["z"], -0.2)
                    self.assertTrue(holding["flight_status"]["accepted"])
                    self.assertEqual(holding["flight_status"]["request"], "takeoff")

                    for _ in range(6):
                        await client.send(
                            json.dumps(
                                {
                                    "deadman": True,
                                    "frame": "BODY_FRD",
                                    "velocity": {
                                        "linear": {"x": 0.6, "y": 0.0, "z": 0.0},
                                        "angular": {"z": 0.0},
                                    },
                                }
                            )
                        )
                        await asyncio.sleep(0.05)
                    moving = await wait_until(
                        client, lambda msg: msg["odom"]["x"] > 0.02
                    )
                    self.assertEqual(drone(moving).get("landed_state"), "in_air")

                    await client.send(
                        json.dumps(
                            {
                                "deadman": False,
                                "frame": "BODY_FRD",
                                "velocity": {
                                    "linear": {"x": 2.0, "y": 0.0, "z": -1.0},
                                    "angular": {"z": 0.0},
                                },
                            }
                        )
                    )
                    quiet = await wait_until(
                        client, lambda msg: msg.get("streaming") is False
                    )
                    self.assertEqual(drone(quiet).get("landed_state"), "in_air")
                    self.assertNotEqual(drone(quiet).get("flight_mode"), "land")
                    await asyncio.sleep(0.8)
                    settled = await drain(client)
                    assert settled is not None
                    x1 = settled["odom"]["x"]
                    self.assertEqual(drone(settled).get("landed_state"), "in_air")
                    await asyncio.sleep(0.35)
                    later = await drain(client)
                    assert later is not None
                    self.assertAlmostEqual(later["odom"]["x"], x1, delta=0.05)
                    self.assertEqual(drone(later).get("landed_state"), "in_air")
                    self.assertNotIn(
                        drone(later).get("landed_state"), ("landing", "on_ground")
                    )

                    await client.send(json.dumps({"estop": True, "deadman": True}))
                    landed = await wait_until(
                        client,
                        lambda msg: (
                            drone(msg).get("landed_state") == "on_ground"
                            and drone(msg).get("armed") is False
                        ),
                    )
                    self.assertGreaterEqual(landed["drone"]["pose"]["z"], -0.05)
                    self.assertFalse(landed["streaming"])
            finally:
                await server.stop()

        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
