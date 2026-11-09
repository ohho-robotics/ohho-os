import struct
import threading
import time
import unittest

from ohho.robots.omnibot import proto
from ohho.robots.omnibot.yahboom import YahboomTransport
from ohho.registry import get_spec
from ohho.schema import ConnectionState, Velocity

try:
    from test_yahboom_proto import make_rx_velocity
except ImportError:
    from tests.test_yahboom_proto import make_rx_velocity


class FakeClock:
    def __init__(self, start: float = 1000.0) -> None:
        self.t = start
        self._lock = threading.Lock()

    def now(self) -> float:
        with self._lock:
            return self.t

    def advance(self, dt: float) -> None:
        with self._lock:
            self.t += dt


class FakeSerial:
    """In-memory serial stand-in: records writes, replays fed RX bytes."""

    def __init__(self) -> None:
        self.written = bytearray()
        self._rx = bytearray()
        self.fail_read = False

    def write(self, data: bytes) -> int:
        self.written.extend(data)
        return len(data)

    def read(self, n: int = 1) -> bytes:
        if self.fail_read:
            raise OSError("serial read failed")
        chunk = bytes(self._rx[:n])
        del self._rx[:n]
        return chunk

    def feed(self, data: bytes) -> None:
        self._rx.extend(data)

    def close(self) -> None:
        pass


def _tp(fs: FakeSerial) -> YahboomTransport:
    return YahboomTransport(get_spec("omnibot"), "/dev/fake", serial_factory=lambda: fs)


class TestYahboomAdapter(unittest.TestCase):
    def test_connect_sets_car_type(self):
        fs = FakeSerial()
        tp = _tp(fs)
        tp.connect()
        try:
            self.assertIn(bytes(proto.packet_set_car_type()), bytes(fs.written))
        finally:
            tp.disconnect()

    def test_send_velocity_writes_motion_packet(self):
        fs = FakeSerial()
        tp = _tp(fs)
        tp.connect()
        try:
            fs.written.clear()
            tp.send_velocity(Velocity(0.1, 0.0, 0.2))
            self.assertEqual(bytes(fs.written), proto.packet_motion(0.1, 0.0, 0.2))
        finally:
            tp.disconnect()

    def test_emergency_stop_blocks_motion(self):
        fs = FakeSerial()
        tp = _tp(fs)
        tp.connect()
        try:
            fs.written.clear()
            tp.emergency_stop()
            self.assertEqual(bytes(fs.written), proto.packet_motion(0.0, 0.0, 0.0))
            fs.written.clear()
            tp.send_velocity(Velocity(0.1, 0.0, 0.0))  # ignored while stopped
            self.assertEqual(len(fs.written), 0)
            tp.release_stop()
            tp.send_velocity(Velocity(0.1, 0.0, 0.0))
            self.assertGreater(len(fs.written), 0)
        finally:
            tp.disconnect()

    def test_ingest_decodes_telemetry(self):
        # No connect() -> no reader thread; feed bytes deterministically.
        fs = FakeSerial()
        tp = _tp(fs)
        tp._ingest(make_rx_velocity(0.3, 0.0, 0.5))
        t = tp.read()
        self.assertAlmostEqual(t.odom.vx, 0.3, places=3)
        self.assertAlmostEqual(t.odom.omega, 0.5, places=3)

    def test_telemetry_callback_fires_on_ingest(self):
        fs = FakeSerial()
        tp = _tp(fs)
        seen = []
        tp.on_telemetry(seen.append)
        tp._ingest(make_rx_velocity(0.1, 0.0, 0.0))
        self.assertEqual(len(seen), 1)

    def test_send_motor_writes_packet(self):
        fs = FakeSerial()
        tp = _tp(fs)
        tp.connect()
        try:
            fs.written.clear()
            tp.send_motor(50, -50, 50, -50)
            self.assertEqual(bytes(fs.written), proto.packet_motor(50, -50, 50, -50))
        finally:
            tp.disconnect()

    def test_wheel_encoders_not_derived_from_velocity(self):
        fs = FakeSerial()
        tp = _tp(fs)
        tp._ingest(make_rx_velocity(0.2, 0.0, 0.0))
        tp._ingest(make_rx_velocity(0.2, 0.0, 0.0))
        # Finding 1: Yahboom protocol does not report raw encoder counts;
        # adapter must never derive synthetic encoder counts from velocity packets.
        encs = tp.get_wheel_encoders()
        self.assertEqual(encs, {})

    def test_imu_single_stream_timestamps(self):
        # Finding 2: Only attitude packets update IMU timestamps to prevent 3x count from burst packets
        fs = FakeSerial()
        tp = _tp(fs)

        # Helper to construct an RX packet
        def make_pkt(pkt_type: int, payload: bytes) -> bytes:
            length = 3 + len(payload)
            body = bytes([length, pkt_type]) + payload
            cs = sum(body) & 0xFF
            return bytes([proto.HEAD_TX, proto.HEAD_RX]) + body + bytes([cs])

        # Accel and Gyro should not append to _imu_timestamps
        tp._ingest(make_pkt(proto.TYPE_ACCEL, struct.pack("<hhh", 100, 200, 300)))
        tp._ingest(make_pkt(proto.TYPE_GYRO, struct.pack("<hhh", 10, 20, 30)))
        self.assertEqual(len(tp.get_imu_samples()), 0)

        # Attitude packet should append
        tp._ingest(make_pkt(proto.TYPE_ATTITUDE, struct.pack("<hhh", 0, 0, 1000)))
        self.assertEqual(len(tp.get_imu_samples()), 1)

    def test_link_loss_watchdog_sends_stop(self):
        fs = FakeSerial()
        tp = _tp(fs)
        tp.connect()
        try:
            fs.written.clear()
            fs.fail_read = True
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline:
                if tp.status().state == ConnectionState.RECONNECTING:
                    break
                time.sleep(0.01)
            self.assertEqual(tp.status().state, ConnectionState.RECONNECTING)
            self.assertIn(proto.packet_motion(0.0, 0.0, 0.0), bytes(fs.written))
        finally:
            tp.disconnect()

    def test_reconnect_opens_fresh_serial(self):
        clock = FakeClock()
        opened: list[FakeSerial] = []

        def factory() -> FakeSerial:
            ser = FakeSerial()
            opened.append(ser)
            return ser

        tp = YahboomTransport(
            get_spec("omnibot"),
            "/dev/fake",
            serial_factory=factory,
            telemetry_timeout_s=0.05,
            reconnect_interval_s=0.02,
            monotonic=clock.now,
        )
        tp.connect()
        try:
            opened[0].feed(make_rx_velocity(0.0, 0.0, 0.0))
            clock.advance(0.1)
            tp._on_link_lost()
            self.assertEqual(len(opened), 1)
            clock.advance(0.03)
            tp._try_reconnect()
            self.assertEqual(len(opened), 2)
            self.assertEqual(tp.status().state, ConnectionState.CONNECTED)
        finally:
            tp.disconnect()


if __name__ == "__main__":
    unittest.main()
