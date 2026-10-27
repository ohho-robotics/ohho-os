import unittest

from ohho.adapters._link_resilience import LinkResilience


class FakeClock:
    def __init__(self, start: float = 100.0) -> None:
        self.t = start

    def now(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


class TestLinkResilience(unittest.TestCase):
    def test_telemetry_stale_after_timeout(self):
        clock = FakeClock()
        link = LinkResilience(
            telemetry_timeout_s=1.0, reconnect_interval_s=0.5, monotonic=clock.now
        )
        link.on_connected()
        clock.advance(0.5)
        self.assertFalse(link.telemetry_stale())
        clock.advance(0.6)
        self.assertTrue(link.telemetry_stale())

    def test_reconnect_backoff(self):
        clock = FakeClock()
        link = LinkResilience(
            telemetry_timeout_s=2.0, reconnect_interval_s=0.5, monotonic=clock.now
        )
        link.on_connected()
        link.on_link_lost()
        self.assertFalse(link.should_reconnect())
        clock.advance(0.5)
        self.assertTrue(link.should_reconnect())
        link.on_reconnect_failed()
        clock.advance(0.5)
        self.assertFalse(link.should_reconnect())
        clock.advance(0.5)
        self.assertTrue(link.should_reconnect())


if __name__ == "__main__":
    unittest.main()
