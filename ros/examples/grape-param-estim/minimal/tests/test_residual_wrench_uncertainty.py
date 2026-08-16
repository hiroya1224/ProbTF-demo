from __future__ import annotations

import math
import unittest

import numpy as np

from _support import synthetic_problem_parts
from single_bag_savgol_core import (
    COMMON_SCALE_DIRECTION,
    SingleBagDynamicsProblem,
)
from single_bag_savgol_covariance import (
    parameter_covariances,
    residual_wrench_uncertainty,
    wrench_acceleration_closure_maps,
)


def _samples_with_covariance(covariance: np.ndarray, count: int = 17) -> np.ndarray:
    """Construct zero-mean samples with the requested ddof=1 covariance."""

    value = np.asarray(covariance, dtype=float)
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (value + value.T))
    if np.any(eigenvalues < -1.0e-13):
        raise ValueError("sample covariance must be PSD")
    root = eigenvectors @ np.diag(np.sqrt(np.maximum(eigenvalues, 0.0)))
    centered_seed = np.eye(count, 6) - np.ones((count, 6)) / count
    basis, _ = np.linalg.qr(centered_seed)
    samples = math.sqrt(count - 1) * basis[:, :6] @ root.T
    if not np.allclose(np.mean(samples, axis=0), 0.0, atol=2.0e-15):
        raise RuntimeError("constructed covariance samples are not centered")
    return samples


def _residual_case(excess_raw: np.ndarray):
    mass = 2.3
    inertia = np.asarray(
        ((1.7, 0.12, -0.08), (0.12, 1.3, 0.09), (-0.08, 0.09, 1.1))
    )
    lever = np.asarray((0.21, -0.17, 0.08))
    acceleration_to_wrench, _ = wrench_acceleration_closure_maps(
        mass, inertia, lever
    )
    sigma_z = np.diag((0.8, 0.7, 0.9, 0.6, 0.75, 0.65))
    sg_wrench = acceleration_to_wrench @ sigma_z @ acceleration_to_wrench.T
    empirical = 0.5 * (sg_wrench + excess_raw + (sg_wrench + excess_raw).T)
    samples = _samples_with_covariance(empirical)
    return residual_wrench_uncertainty(
        raw_residual_wrench=samples,
        modeled_wrench=np.zeros_like(samples),
        required_wrench=samples,
        estimated_mass_kg=mass,
        estimated_inertia_kg_m2=inertia,
        fixed_mass_kg=mass,
        lever_arm_m=lever,
        reference_sigma_z=np.repeat(sigma_z[None, :, :], samples.shape[0], axis=0),
    )


class ResidualWrenchUncertaintyTests(unittest.TestCase):
    def test_closure_map_inverse_and_covariance_round_trip(self):
        mass = 3.7
        inertia = np.asarray(
            ((1.4, 0.13, -0.07), (0.13, 1.1, 0.04), (-0.07, 0.04, 0.9))
        )
        lever = np.asarray((0.23, -0.11, 0.17))
        forward, inverse = wrench_acceleration_closure_maps(
            mass, inertia, lever
        )
        self.assertTrue(np.allclose(inverse @ forward, np.eye(6), atol=2e-15))
        generator = np.asarray(
            [
                [1.0, 0.2, 0.0, 0.1, 0.0, -0.1],
                [0.2, 1.2, 0.1, 0.0, 0.2, 0.0],
                [0.0, 0.1, 0.9, -0.1, 0.0, 0.1],
                [0.1, 0.0, -0.1, 0.8, 0.1, 0.0],
                [0.0, 0.2, 0.0, 0.1, 1.1, 0.2],
                [-0.1, 0.0, 0.1, 0.0, 0.2, 1.0],
            ]
        )
        sigma_z = generator @ generator.T
        sigma_w = forward @ sigma_z @ forward.T
        self.assertTrue(
            np.allclose(inverse @ sigma_w @ inverse.T, sigma_z, atol=2e-14)
        )

    def test_common_scale_invariance(self):
        mass = 2.6
        inertia = np.asarray(
            ((1.2, 0.1, 0.03), (0.1, 1.0, -0.04), (0.03, -0.04, 0.8))
        )
        lever = np.asarray((0.2, -0.3, 0.1))
        wrench_covariance = np.asarray(
            [
                [2.0, 0.1, 0.0, 0.2, 0.0, 0.0],
                [0.1, 1.8, 0.1, 0.0, -0.2, 0.0],
                [0.0, 0.1, 1.6, 0.0, 0.0, 0.15],
                [0.2, 0.0, 0.0, 0.7, 0.1, 0.0],
                [0.0, -0.2, 0.0, 0.1, 0.8, 0.1],
                [0.0, 0.0, 0.15, 0.0, 0.1, 0.6],
            ]
        )
        _, base_inverse = wrench_acceleration_closure_maps(
            mass, inertia, lever
        )
        expected = base_inverse @ wrench_covariance @ base_inverse.T
        for scale in (0.37, 1.83, 4.71):
            _, scaled_inverse = wrench_acceleration_closure_maps(
                scale * mass, scale * inertia, lever
            )
            actual = scaled_inverse @ (scale**2 * wrench_covariance) @ scaled_inverse.T
            self.assertTrue(np.allclose(actual, expected, rtol=2e-14, atol=2e-14))

    def test_known_full_excess_and_force_torque_coupling(self):
        generator = np.asarray(
            [
                [0.6, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.1, 0.5, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.1, 0.45, 0.0, 0.0, 0.0],
                [0.14, 0.0, 0.0, 0.3, 0.0, 0.0],
                [0.0, -0.12, 0.0, 0.04, 0.28, 0.0],
                [0.0, 0.0, 0.11, 0.0, 0.03, 0.25],
            ]
        )
        known = generator @ generator.T
        result = _residual_case(known)
        self.assertTrue(
            np.allclose(result.excess_covariance_raw, known, atol=3e-14)
        )
        self.assertTrue(
            np.allclose(result.model_discrepancy_covariance, known, atol=3e-14)
        )
        self.assertGreater(
            np.linalg.norm(result.model_discrepancy_covariance[:3, 3:]), 0.0
        )

    def test_psd_projection_preserves_negative_raw_eigenvalue(self):
        raw = np.diag((-0.2, 0.3, 0.4, 0.15, 0.2, 0.25))
        raw[0, 3] = raw[3, 0] = 0.03
        result = _residual_case(raw)
        self.assertLess(result.excess_covariance_raw_eigenvalues[0], -0.1)
        self.assertGreater(result.appreciably_negative_raw_eigenvalues.size, 0)
        self.assertGreaterEqual(
            np.min(np.linalg.eigvalsh(result.model_discrepancy_covariance)),
            -2e-15,
        )
        self.assertEqual(result.model_discrepancy_eigenvalues[0], 0.0)

    def test_no_double_counting_and_corrected_parameter_covariance_psd(self):
        result = _residual_case(np.zeros((6, 6)))
        self.assertTrue(
            np.allclose(result.model_discrepancy_covariance, 0.0, atol=4e-14)
        )
        dataset, _model, _actuator = synthetic_problem_parts()
        rng = np.random.default_rng(8)
        jacobian = rng.standard_normal((dataset.time.size, 6, 14))
        uncertainty = parameter_covariances(
            jacobian,
            dataset.covariance,
            COMMON_SCALE_DIRECTION,
            result.acceleration_model_discrepancy_covariance,
        )
        self.assertTrue(
            np.allclose(
                uncertainty.wrench_corrected,
                uncertainty.overlap_corrected,
                rtol=2e-12,
                atol=2e-12,
            )
        )
        self.assertTrue(
            np.allclose(
                uncertainty.wrench_corrected,
                uncertainty.wrench_corrected.T,
                atol=2e-14,
            )
        )
        self.assertGreaterEqual(
            np.min(np.linalg.eigvalsh(uncertainty.wrench_corrected)), -2e-12
        )

    def test_reference_full_covariance_used_when_optimization_is_identity(self):
        dataset, model, actuator = synthetic_problem_parts()
        self.assertEqual(dataset.covariance.mode, "identity")
        self.assertEqual(dataset.reference_covariance.mode, "full")
        problem = SingleBagDynamicsProblem(dataset, model, actuator)
        evaluation = problem.evaluate_physical(np.zeros(14), 0.0)
        residual = residual_wrench_uncertainty(
            raw_residual_wrench=evaluation.raw_residual_wrench,
            modeled_wrench=evaluation.modeled_wrench,
            required_wrench=evaluation.required_wrench,
            estimated_mass_kg=evaluation.parameters.mass,
            estimated_inertia_kg_m2=evaluation.parameters.inertia,
            fixed_mass_kg=model.parameters.mass,
            lever_arm_m=(
                dataset.pose_sensor_position_in_body
                - evaluation.parameters.cog_offset
            ),
            reference_sigma_z=dataset.reference_covariance.local_sigma_z,
        )
        expected = np.asarray(
            [
                residual.acceleration_to_wrench
                @ item
                @ residual.acceleration_to_wrench.T
                for item in dataset.reference_covariance.local_sigma_z
            ]
        )
        identity_wrong = np.asarray(
            [
                residual.acceleration_to_wrench
                @ item
                @ residual.acceleration_to_wrench.T
                for item in dataset.covariance.local_sigma_z
            ]
        )
        self.assertTrue(np.allclose(residual.sg_covariance_per_time, expected))
        self.assertFalse(np.allclose(residual.sg_covariance_per_time, identity_wrong))

    def test_direct_mass_scaling_matches_exact_chart_regauge(self):
        dataset, model, actuator = synthetic_problem_parts()
        problem = SingleBagDynamicsProblem(dataset, model, actuator)
        coordinate = 0.23 * COMMON_SCALE_DIRECTION
        original = problem.evaluate_physical(coordinate, 0.0)
        scale = model.parameters.mass / original.parameters.mass
        regauged_coordinate = coordinate + math.log(scale) * COMMON_SCALE_DIRECTION
        regauged = problem.evaluate_physical(regauged_coordinate, 0.0)
        self.assertTrue(
            np.allclose(
                scale * original.raw_residual_wrench,
                regauged.raw_residual_wrench,
                rtol=2e-14,
                atol=2e-14,
            )
        )
        self.assertTrue(
            np.allclose(scale * original.modeled_wrench, regauged.modeled_wrench)
        )
        self.assertTrue(
            np.allclose(scale * original.required_wrench, regauged.required_wrench)
        )

    def test_postfit_path_does_not_mutate_point_evaluation(self):
        dataset, model, actuator = synthetic_problem_parts()
        problem = SingleBagDynamicsProblem(dataset, model, actuator)
        evaluation = problem.evaluate_physical(np.zeros(14), 0.0)
        coordinate = evaluation.physical_coordinate.copy()
        raw = evaluation.raw_residual_wrench.copy()
        objective = evaluation.cost
        rotor_lag = evaluation.rotor_lag_seconds
        residual_wrench_uncertainty(
            raw_residual_wrench=evaluation.raw_residual_wrench,
            modeled_wrench=evaluation.modeled_wrench,
            required_wrench=evaluation.required_wrench,
            estimated_mass_kg=evaluation.parameters.mass,
            estimated_inertia_kg_m2=evaluation.parameters.inertia,
            fixed_mass_kg=model.parameters.mass,
            lever_arm_m=(
                dataset.pose_sensor_position_in_body
                - evaluation.parameters.cog_offset
            ),
            reference_sigma_z=dataset.reference_covariance.local_sigma_z,
        )
        self.assertTrue(np.array_equal(evaluation.physical_coordinate, coordinate))
        self.assertEqual(evaluation.rotor_lag_seconds, rotor_lag)
        self.assertEqual(evaluation.cost, objective)
        self.assertTrue(np.array_equal(evaluation.raw_residual_wrench, raw))


if __name__ == "__main__":
    unittest.main()
