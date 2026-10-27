"""Frame enum, ENU↔NED conversion, and BODY_FRD stick signs."""

from __future__ import annotations

import math
import unittest

from ohho.aerial import (
    Frame,
    body_z_from_climb,
    convert_vector,
    convert_yaw,
    convert_yaw_rate,
    require_frame,
    stick_to_body_frd,
)


class TestFrame(unittest.TestCase):
    def test_members(self):
        self.assertEqual(
            [member.value for member in Frame],
            ["LOCAL_NED", "LOCAL_ENU", "BODY_FRD"],
        )

    def test_require_frame_rejects_strings_and_none(self):
        self.assertIs(require_frame(Frame.LOCAL_NED), Frame.LOCAL_NED)
        for bad in ("LOCAL_NED", "LOCAL_ENU", "BODY_FRD", None, 0):
            with self.assertRaises(TypeError):
                require_frame(bad)

    def test_require_frame_has_no_default_value(self):
        with self.assertRaises(TypeError):
            require_frame()  # type: ignore[call-arg]


class TestLocalConversion(unittest.TestCase):
    def test_enu_ned_swaps_horizontal_axes_and_negates_z(self):
        samples = [(0, 0, 0), (1, 2, 3), (-4, 0.5, -1), (8, -7, 0.25)]
        for x, y, z in samples:
            ned = convert_vector(
                x, y, z, source=Frame.LOCAL_ENU, target=Frame.LOCAL_NED
            )
            self.assertEqual(ned, (float(y), float(x), -float(z)))
            back = convert_vector(*ned, source=Frame.LOCAL_NED, target=Frame.LOCAL_ENU)
            self.assertEqual(back, (float(x), float(y), float(z)))

    def test_positive_up_is_negative_ned_z(self):
        # ENU velocity: 0 east, 0 north, +1 up → NED 0 north, 0 east, -1 down.
        north, east, down = convert_vector(
            0.0, 0.0, 1.0, source=Frame.LOCAL_ENU, target=Frame.LOCAL_NED
        )
        self.assertEqual((north, east, down), (0.0, 0.0, -1.0))

    def test_yaw_zero_ned_is_north_and_pi_over_2_enu(self):
        for yaw in (0.0, math.pi / 2, math.pi, -math.pi / 2, 0.3, -2.0):
            enu = convert_yaw(yaw, source=Frame.LOCAL_NED, target=Frame.LOCAL_ENU)
            self.assertAlmostEqual(enu, math.pi / 2.0 - yaw)
            back = convert_yaw(enu, source=Frame.LOCAL_ENU, target=Frame.LOCAL_NED)
            self.assertAlmostEqual(back, yaw)
        # Facing east: NED +pi/2 (clockwise from north) is ENU yaw 0.
        self.assertAlmostEqual(
            convert_yaw(math.pi / 2, source=Frame.LOCAL_NED, target=Frame.LOCAL_ENU),
            0.0,
        )

    def test_yaw_rate_sign_flips_between_ned_and_enu(self):
        # Positive NED yaw rate is nose-right; ENU positive is nose-left.
        self.assertEqual(
            convert_yaw_rate(0.25, source=Frame.LOCAL_NED, target=Frame.LOCAL_ENU),
            -0.25,
        )
        self.assertEqual(
            convert_yaw_rate(-0.25, source=Frame.LOCAL_ENU, target=Frame.LOCAL_NED),
            0.25,
        )
        self.assertAlmostEqual(
            convert_yaw_rate(
                convert_yaw_rate(0.4, source=Frame.LOCAL_NED, target=Frame.LOCAL_ENU),
                source=Frame.LOCAL_ENU,
                target=Frame.LOCAL_NED,
            ),
            0.4,
        )

    def test_same_frame_is_identity(self):
        for frame in Frame:
            self.assertEqual(
                convert_vector(1.0, -2.0, 3.5, source=frame, target=frame),
                (1.0, -2.0, 3.5),
            )
            self.assertEqual(convert_yaw(-0.4, source=frame, target=frame), -0.4)
            self.assertEqual(
                convert_yaw_rate(0.2, source=frame, target=frame),
                0.2,
            )

    def test_body_frame_is_not_silently_treated_as_local(self):
        for fn, value in (
            (convert_vector, (1.0, 0.0, 0.0)),
            (convert_yaw, (0.0,)),
            (convert_yaw_rate, (0.1,)),
        ):
            with self.assertRaises(ValueError):
                fn(*value, source=Frame.BODY_FRD, target=Frame.LOCAL_NED)
            with self.assertRaises(ValueError):
                fn(*value, source=Frame.LOCAL_ENU, target=Frame.BODY_FRD)

    def test_source_and_target_are_required(self):
        with self.assertRaises(TypeError) as missing_both:
            convert_vector(1.0, 2.0, 3.0)  # type: ignore[call-arg]
        self.assertIn("source", str(missing_both.exception))
        with self.assertRaises(TypeError) as missing_target:
            convert_vector(1.0, 2.0, 3.0, source=Frame.LOCAL_ENU)  # type: ignore[call-arg]
        self.assertIn("target", str(missing_target.exception))
        with self.assertRaises(TypeError):
            convert_yaw(0.0)  # type: ignore[call-arg]
        with self.assertRaises(TypeError):
            convert_yaw_rate(0.1, source=Frame.LOCAL_NED)  # type: ignore[call-arg]
        with self.assertRaises(TypeError):
            convert_vector(
                0.0,
                0.0,
                0.0,
                Frame.LOCAL_NED,  # type: ignore[misc]
                Frame.LOCAL_ENU,
            )

    def test_strings_and_non_finite_numbers_are_rejected(self):
        with self.assertRaises(TypeError):
            convert_vector(0.0, 0.0, 0.0, source="LOCAL_ENU", target=Frame.LOCAL_NED)
        with self.assertRaises(TypeError):
            convert_vector(
                True, 0.0, 0.0, source=Frame.LOCAL_NED, target=Frame.LOCAL_ENU
            )
        with self.assertRaises(ValueError):
            convert_yaw(float("nan"), source=Frame.LOCAL_NED, target=Frame.LOCAL_ENU)
        with self.assertRaises(ValueError):
            convert_yaw_rate(
                float("inf"), source=Frame.LOCAL_ENU, target=Frame.LOCAL_NED
            )


class TestBodyFrd(unittest.TestCase):
    def test_stick_up_is_negative_body_z(self):
        self.assertEqual(body_z_from_climb(1.0), -1.0)
        self.assertEqual(body_z_from_climb(0.0), 0.0)
        self.assertEqual(body_z_from_climb(-0.5), 0.5)

    def test_stick_yaw_right_is_positive_body_yaw_rate(self):
        """Rightward left-stick is nose-right in BODY_FRD.

        Positive ``yaw_right`` stays a positive yaw rate. Body z is down, so
        nose-right is the right-hand sense about +z. A mode-2 radio and the VR
        aerial scheme both use this unnegated left-stick sign.
        """
        vx, vy, vz, yaw_rate = stick_to_body_frd(
            forward=0.4, right=-0.2, climb=1.0, yaw_right=0.3
        )
        self.assertEqual((vx, vy, vz, yaw_rate), (0.4, -0.2, -1.0, 0.3))
        descend = stick_to_body_frd(0.0, 0.0, climb=-0.4, yaw_right=0.0)
        self.assertEqual(descend, (0.0, 0.0, 0.4, 0.0))

    def test_stick_helper_rejects_non_finite_climb(self):
        with self.assertRaises(ValueError):
            stick_to_body_frd(0.0, 0.0, float("nan"), 0.0)
        with self.assertRaises(TypeError):
            body_z_from_climb(True)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
