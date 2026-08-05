from argparse import Namespace
from pathlib import Path
import sys
import unittest

import numpy as np


_MINIMAL = Path(__file__).resolve().parents[1]
_PACKAGE = _MINIMAL.parent / "src"
sys.path.insert(0, str(_MINIMAL))
sys.path.insert(0, str(_PACKAGE))

import deterministic_tempered_estimator as estimator  # noqa: E402


class ProposalGeometryTests(unittest.TestCase):
    def test_gauss_newton_covariance_is_scaled_and_ridge_aligned(self):
        diagonal = np.linspace(1.0, 15.0, estimator.sobol.SMOOTH_DIMENSION)
        jacobian = np.diag(diagonal)
        scales = np.linspace(0.1, 0.3, estimator.sobol.SMOOTH_DIMENSION)
        root, diagnostic = estimator.proposal_geometry_from_jacobian(
            jacobian,
            scales,
            damping=0.5,
            minimum_variance=1.0e-8,
            maximum_variance=10.0,
        )
        expected = np.diag(
            scales**2 / ((diagonal * scales) ** 2 + 0.5)
        )
        np.testing.assert_allclose(root @ root.T, expected, atol=1.0e-14)
        self.assertEqual(root.shape, (15, 15))
        self.assertEqual(len(diagnostic["singular_values"]), 15)

    def test_underdetermined_jacobian_keeps_clipped_null_directions(self):
        jacobian = np.zeros((1, estimator.sobol.SMOOTH_DIMENSION))
        jacobian[0, 0] = 2.0
        root, _diagnostic = estimator.proposal_geometry_from_jacobian(
            jacobian,
            np.ones(estimator.sobol.SMOOTH_DIMENSION),
            damping=0.1,
            minimum_variance=0.01,
            maximum_variance=0.4,
        )
        covariance = root @ root.T
        self.assertEqual(root.shape, (15, 15))
        self.assertAlmostEqual(covariance[0, 0], 1.0 / 4.1)
        np.testing.assert_allclose(np.diag(covariance)[1:], 0.4)


class TemperatureTests(unittest.TestCase):
    def test_pilot_loss_scale_sets_geometric_temperature_ladder(self):
        temperatures, diagnostic = estimator.calibrate_temperature_ladder(
            100.0,
            (101.0, 104.0, 109.0),
            replica_count=4,
            low_typical_uphill_acceptance=0.02,
            high_typical_uphill_acceptance=0.30,
        )
        self.assertAlmostEqual(diagnostic["typical_loss_increment"], 4.0)
        self.assertAlmostEqual(temperatures[0], 4.0 / -np.log(0.02))
        self.assertAlmostEqual(temperatures[-1], 4.0 / -np.log(0.30))
        np.testing.assert_allclose(
            temperatures[1:] / temperatures[:-1],
            np.full(3, temperatures[1] / temperatures[0]),
        )

    def test_replica_exchange_direction_is_correct(self):
        unfavorable = estimator.replica_swap_log_acceptance(1.0, 5.0, 1.0, 2.0)
        favorable = estimator.replica_swap_log_acceptance(5.0, 1.0, 1.0, 2.0)
        self.assertAlmostEqual(unfavorable, -2.0)
        self.assertAlmostEqual(favorable, 2.0)


class TemperedDriverTests(unittest.TestCase):
    class QuadraticEvaluator:
        def evaluate(self, coordinate, *, problem=None):
            del problem
            value = np.asarray(coordinate, dtype=float)
            if np.any(np.abs(value[:15]) > 3.0):
                return {"valid": False, "reason": "outside_search_bounds"}
            return {
                "valid": True,
                "reason": "accepted",
                "trajectory_loss": 0.5 * float(value[:15] @ value[:15]),
                "physical": {"normalized_inertia_triangle_margin": 1.0},
                "pid": {"group_scales": np.ones(4)},
            }

    def test_parallel_tempering_records_proposals_and_swaps(self):
        center = {
            "coordinate": np.zeros(16),
            "trajectory_loss": 0.0,
            "valid": True,
        }
        pilot_coordinate = np.zeros(16)
        pilot_coordinate[0] = 0.4
        pilot = {
            "coordinate": pilot_coordinate,
            "trajectory_loss": 0.08,
            "valid": True,
        }
        arguments = Namespace(
            proposal_scale=0.2,
            maximum_temperature_proposal_scale=2.0,
            local_prior_strength=1.0,
            sweeps=4,
            exchange_interval=1,
            command_delay=0.01,
            saved_top_count=2,
            saved_top_minimum_distance=0.1,
        )
        result = estimator._run_parallel_tempering(
            self.QuadraticEvaluator(),
            center,
            (pilot,),
            np.eye(15),
            np.asarray((1.0, 2.0)),
            np.ones(15),
            arguments,
            np.random.default_rng(4),
            (object(), object()),
            None,
        )
        np.testing.assert_array_equal(result["proposal_attempts"], (4, 4))
        self.assertEqual(result["loss_trace"].shape, (5, 2))
        self.assertEqual(int(np.sum(result["swap_attempts"])), 2)
        self.assertLessEqual(result["best_record"]["trajectory_loss"], 0.0)

    def test_pid_gate_is_disabled_by_default_and_requires_both_limits(self):
        arguments = estimator.create_argument_parser().parse_args([])
        gate, scales = estimator._pid_gate(arguments, np.ones((4, 3)))
        self.assertFalse(gate.enabled)
        self.assertEqual(scales, (None, None))
        arguments.pid_gain_min_scale = 0.8
        with self.assertRaises(ValueError):
            estimator._pid_gate(arguments, np.ones((4, 3)))


if __name__ == "__main__":
    unittest.main()
