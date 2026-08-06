from pathlib import Path
import sys
import unittest

import numpy as np


_MINIMAL = Path(__file__).resolve().parents[1]
_PACKAGE = _MINIMAL.parent / "src"
sys.path.insert(0, str(_MINIMAL))
sys.path.insert(0, str(_PACKAGE))

import deterministic_multiple_shooting_estimator as estimator  # noqa: E402
import estimate_recorded_control as entrypoint  # noqa: E402
from grape_param_estim.geometry import (  # noqa: E402
    so3_exp,
    so3_left_jacobian,
)
from grape_param_estim.system import ActuatorState, RigidBodyState  # noqa: E402


class SegmentScheduleTests(unittest.TestCase):
    def test_schedule_covers_full_grid_without_continuation(self):
        boundaries = estimator.segment_boundaries(101, 0.05, 0.5)
        np.testing.assert_array_equal(boundaries, np.arange(0, 101, 10))

    def test_last_short_segment_is_retained(self):
        boundaries = estimator.segment_boundaries(12, 0.05, 0.2)
        np.testing.assert_array_equal(boundaries, (0, 4, 8, 11))


class Se3ResidualTests(unittest.TestCase):
    def test_identical_poses_have_zero_error(self):
        position = np.asarray((1.0, -2.0, 0.5))
        rotation = so3_exp((0.2, -0.1, 0.3))
        np.testing.assert_allclose(
            estimator.se3_log_error(position, rotation, position, rotation),
            np.zeros(6),
            atol=1.0e-12,
        )

    def test_translation_is_pushed_to_se3_coordinate(self):
        observed_position = np.asarray((0.2, -0.3, 0.4))
        observed_rotation = so3_exp((0.1, 0.2, -0.15))
        rho = np.asarray((0.4, -0.2, 0.1))
        phi = np.asarray((0.3, -0.1, 0.2))
        relative_translation = so3_left_jacobian(phi) @ rho
        simulated_rotation = observed_rotation @ so3_exp(phi)
        simulated_position = (
            observed_position + observed_rotation @ relative_translation
        )
        residual = estimator.se3_log_error(
            observed_position,
            observed_rotation,
            simulated_position,
            simulated_rotation,
        )
        np.testing.assert_allclose(residual[:3], rho, atol=1.0e-10)
        np.testing.assert_allclose(residual[3:], phi, atol=1.0e-10)


class NodeChartTests(unittest.TestCase):
    def test_node_encode_decode_round_trip(self):
        reference = estimator.NodeReference(
            position=np.asarray((1.0, 2.0, 3.0)),
            rotation=so3_exp((0.1, -0.2, 0.05)),
            linear_velocity=np.asarray((0.2, 0.3, -0.4)),
            angular_velocity=np.asarray((0.4, -0.5, 0.6)),
            thrust=np.asarray((5.0, 6.0, 7.0, 8.0)),
            gimbal=np.asarray((0.1, -0.1, 0.2, -0.2)),
        )
        correction = np.linspace(-0.05, 0.05, estimator.NODE_DIMENSION)
        rigid, actuator, _, _ = estimator._decode_node(reference, correction)
        encoded = estimator._encode_node(reference, rigid, actuator)
        np.testing.assert_allclose(encoded, correction, atol=1.0e-10)


class EntryPointTests(unittest.TestCase):
    def test_multiple_shooting_is_default(self):
        self.assertEqual(
            entrypoint.DEFAULT_METHOD,
            "deterministic_multiple_shooting",
        )

    def test_parser_uses_only_requested_physical_family(self):
        arguments = estimator.create_argument_parser().parse_args([])
        self.assertFalse(hasattr(arguments, "thrust_time_constant_scale_bounds"))
        self.assertFalse(hasattr(arguments, "gimbal_time_constant_scale_bounds"))
        self.assertEqual(estimator.PHYSICAL_DIMENSION, 13)
        self.assertEqual(estimator.NODE_DIMENSION, 20)


if __name__ == "__main__":
    unittest.main()
