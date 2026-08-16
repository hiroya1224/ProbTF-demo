from __future__ import annotations

import unittest

import numpy as np
from scipy.linalg import null_space

from _support import synthetic_problem_parts
from single_bag_cross_bag_consensus import (
    _pairwise_distance,
    fuse_quotient_gaussians,
)
from single_bag_savgol_core import (
    COMMON_SCALE_DIRECTION,
    SingleBagDynamicsProblem,
    nominal_mass_gauge_uncertainty,
)
from single_bag_savgol_covariance import (
    SgCovarianceEvaluation,
    parameter_covariances,
    residual_score_sandwich_middles,
    residual_wrench_uncertainty,
)


def _independent_identity_covariance(count: int) -> SgCovarianceEvaluation:
    operator = np.zeros((1, 12, 6))
    operator[0, :6, :] = np.eye(6)
    propagation = np.zeros((6, 12))
    propagation[:, :6] = np.eye(6)
    sigma_xi = np.zeros((12, 12))
    sigma_xi[:6, :6] = np.eye(6)
    return SgCovarianceEvaluation(
        mode="identity",
        z=np.zeros((count, 6)),
        local_omega=np.repeat(np.eye(6)[None, :, :], count, axis=0),
        local_sigma_xi=np.repeat(sigma_xi[None, :, :], count, axis=0),
        local_sigma_z=np.repeat(np.eye(6)[None, :, :], count, axis=0),
        propagation_jacobian=np.repeat(
            propagation[None, :, :], count, axis=0
        ),
        whitening=np.repeat(np.eye(6)[None, :, :], count, axis=0),
        xi_operators=tuple(operator.copy() for _ in range(count)),
        raw_indices=tuple(np.asarray((index,)) for index in range(count)),
        geometric_correction=True,
        include_position_rotation_cross=True,
        include_rotation_uncertainty_in_specific_force=True,
    )


def _gauge_projected_jacobian(count: int, seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    gauge = COMMON_SCALE_DIRECTION
    projector = np.eye(14) - np.outer(gauge, gauge) / (gauge @ gauge)
    return np.einsum("npi,ij->npj", rng.standard_normal((count, 6, 14)), projector)


class ConservativeFusionCovarianceTests(unittest.TestCase):
    def test_per_sample_score_outer_product_identity_and_psd(self):
        count = 8
        rng = np.random.default_rng(4)
        jacobian = _gauge_projected_jacobian(count)
        factors = rng.standard_normal((count, 6, 6))
        weights = np.einsum("nij,nkj->nik", factors, factors)
        residual = rng.standard_normal((count, 6))
        uncentered, centered, remainder = residual_score_sandwich_middles(
            jacobian, weights, residual
        )
        expected = np.zeros((14, 14))
        equivalent = np.zeros((14, 14))
        for index in range(count):
            score = jacobian[index].T @ weights[index] @ residual[index]
            expected += np.outer(score, score)
            equivalent += (
                jacobian[index].T
                @ weights[index]
                @ np.outer(residual[index], residual[index])
                @ weights[index]
                @ jacobian[index]
            )
        self.assertTrue(np.allclose(uncentered, expected, atol=2e-12))
        self.assertTrue(np.allclose(uncentered, equivalent, atol=2e-12))
        self.assertTrue(np.allclose(uncentered, centered + remainder))
        self.assertGreaterEqual(np.linalg.eigvalsh(uncentered)[0], -2e-12)

    def test_zero_residual_null_case_and_existing_wrench_path_unchanged(self):
        count = 7
        covariance = _independent_identity_covariance(count)
        jacobian = _gauge_projected_jacobian(count)
        additional = np.diag((0.2, 0.1, 0.3, 0.05, 0.08, 0.04))
        baseline = parameter_covariances(
            jacobian,
            covariance,
            COMMON_SCALE_DIRECTION,
            additional,
        )
        result = parameter_covariances(
            jacobian,
            covariance,
            COMMON_SCALE_DIRECTION,
            additional,
            np.zeros((count, 6)),
        )
        self.assertTrue(
            np.array_equal(result.wrench_corrected, baseline.wrench_corrected)
        )
        self.assertTrue(
            np.array_equal(
                result.sandwich_middle_wrench,
                baseline.sandwich_middle_wrench,
            )
        )
        self.assertTrue(
            np.allclose(
                result.conservative_fusion,
                result.overlap_corrected,
                rtol=2e-13,
                atol=2e-13,
            )
        )

    def test_constant_nonzero_residual_retains_mean_without_sg_subtraction(self):
        count = 9
        covariance = _independent_identity_covariance(count)
        jacobian = _gauge_projected_jacobian(count, seed=9)
        constant = np.asarray((0.4, -0.2, 0.3, 0.1, -0.05, 0.2))
        residual = np.repeat(constant[None, :], count, axis=0)
        result = parameter_covariances(
            jacobian,
            covariance,
            COMMON_SCALE_DIRECTION,
            np.eye(6),
            residual,
        )
        expected, centered, _remainder = residual_score_sandwich_middles(
            jacobian, covariance.weight, residual
        )
        self.assertTrue(
            np.allclose(result.sandwich_middle_residual_uncentered, expected)
        )
        self.assertTrue(np.allclose(centered, 0.0, atol=2e-28))
        self.assertGreater(np.linalg.norm(expected), 0.0)
        difference = (
            result.gauge_basis.T
            @ (result.conservative_fusion - result.overlap_corrected)
            @ result.gauge_basis
        )
        self.assertGreater(np.linalg.eigvalsh(difference)[-1], 0.0)
        # The additional centered-wrench covariance cannot reduce M_res.
        no_wrench = parameter_covariances(
            jacobian,
            covariance,
            COMMON_SCALE_DIRECTION,
            np.zeros((6, 6)),
            residual,
        )
        self.assertTrue(
            np.array_equal(
                result.sandwich_middle_residual_uncentered,
                no_wrench.sandwich_middle_residual_uncentered,
            )
        )

    def test_conservative_psd_order_and_weak_direction_behavior(self):
        count = 13
        covariance = _independent_identity_covariance(count)
        basis = null_space(COMMON_SCALE_DIRECTION.reshape(1, -1))
        jacobian = np.zeros((count, 6, 14))
        residual = np.zeros((count, 6))
        strength = np.ones(13)
        strength[0] = 10.0
        strength[1] = 0.1
        for index in range(13):
            jacobian[index, 0] = strength[index] * basis[:, index]
            residual[index, 0] = 1.0
        result = parameter_covariances(
            jacobian,
            covariance,
            COMMON_SCALE_DIRECTION,
            uncentered_residual=residual,
        )
        quotient_difference = (
            result.gauge_basis.T
            @ (result.conservative_fusion - result.overlap_corrected)
            @ result.gauge_basis
        )
        self.assertGreaterEqual(
            np.linalg.eigvalsh(quotient_difference)[0], -2e-12
        )
        quotient_covariance = (
            basis.T @ result.conservative_fusion @ basis
        )
        self.assertGreater(quotient_covariance[1, 1], quotient_covariance[0, 0])

    def test_common_scale_quotient_and_nominal_mass_physical_transform(self):
        dataset, model, actuator = synthetic_problem_parts()
        problem = SingleBagDynamicsProblem(dataset, model, actuator)
        coordinate = np.asarray(
            (0.37, 0.1, -0.05, 0.03, 0.02, -0.01, 0.01, 0.01, -0.02, 0.005, 0.1, -0.1, 0.05, 0.02)
        )
        evaluation = problem.evaluate_physical(coordinate, 0.0)
        residual_wrench = residual_wrench_uncertainty(
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
        uncertainty = parameter_covariances(
            evaluation.acceleration_jacobian,
            dataset.covariance,
            COMMON_SCALE_DIRECTION,
            residual_wrench.acceleration_model_discrepancy_covariance,
            evaluation.acceleration_residual,
        )
        quotient = uncertainty.gauge_basis
        base_coordinate = quotient.T @ coordinate
        base_covariance = quotient.T @ uncertainty.conservative_fusion @ quotient
        for scale_coordinate in (-1.17, 0.43, 1.91):
            shifted = coordinate + scale_coordinate * COMMON_SCALE_DIRECTION
            self.assertTrue(np.allclose(quotient.T @ shifted, base_coordinate))
            self.assertTrue(
                np.array_equal(
                    quotient.T @ uncertainty.conservative_fusion @ quotient,
                    base_covariance,
                )
            )

        nominal = nominal_mass_gauge_uncertainty(
            problem, evaluation, uncertainty
        )
        fixed_covariance = nominal["covariance_conservative_fusion"]
        self.assertLess(abs(float(fixed_covariance[0, 0])), 2e-14)
        fixed_coordinate = coordinate - coordinate[0] * COMMON_SCALE_DIRECTION
        _parameters, jacobian = problem.chart.decode_with_jacobian(
            fixed_coordinate
        )
        direct_force_covariance = (
            jacobian.force_effectiveness
            @ fixed_covariance
            @ jacobian.force_effectiveness.T
        )
        direct_std = np.sqrt(np.maximum(np.diag(direct_force_covariance), 0.0))
        self.assertTrue(
            np.allclose(
                direct_std,
                nominal["force_effectiveness_std_conservative_fusion"],
            )
        )

    def test_wrench_uncentered_second_moment_keeps_mean(self):
        raw = np.asarray(
            [
                (1.0, 2.0, 0.0, 0.1, 0.0, -0.1),
                (1.0, 2.0, 0.0, 0.1, 0.0, -0.1),
                (1.0, 2.0, 0.0, 0.1, 0.0, -0.1),
            ]
        )
        result = residual_wrench_uncertainty(
            raw_residual_wrench=raw,
            modeled_wrench=np.zeros_like(raw),
            required_wrench=raw,
            estimated_mass_kg=2.0,
            estimated_inertia_kg_m2=np.diag((0.8, 0.9, 1.0)),
            fixed_mass_kg=2.0,
            lever_arm_m=(0.1, -0.2, 0.05),
            reference_sigma_z=np.zeros((raw.shape[0], 6, 6)),
        )
        self.assertTrue(np.allclose(result.empirical_covariance, 0.0))
        self.assertTrue(
            np.allclose(
                result.uncentered_second_moment,
                np.outer(result.mean, result.mean),
            )
        )

    def test_postfit_conservative_path_does_not_change_point_result(self):
        dataset, model, actuator = synthetic_problem_parts()
        problem = SingleBagDynamicsProblem(dataset, model, actuator)
        evaluation = problem.evaluate_physical(np.zeros(14), 0.0)
        coordinate = evaluation.physical_coordinate.copy()
        residual = evaluation.acceleration_residual.copy()
        raw_wrench = evaluation.raw_residual_wrench.copy()
        objective = evaluation.cost
        lag = evaluation.rotor_lag_seconds
        wrench_uncertainty = residual_wrench_uncertainty(
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
        uncertainty = parameter_covariances(
            evaluation.acceleration_jacobian,
            dataset.covariance,
            COMMON_SCALE_DIRECTION,
            wrench_uncertainty.acceleration_model_discrepancy_covariance,
            evaluation.acceleration_residual,
        )
        nominal_mass_gauge_uncertainty(problem, evaluation, uncertainty)
        self.assertTrue(np.array_equal(evaluation.physical_coordinate, coordinate))
        self.assertEqual(evaluation.rotor_lag_seconds, lag)
        self.assertEqual(evaluation.cost, objective)
        self.assertTrue(np.array_equal(evaluation.acceleration_residual, residual))
        self.assertTrue(np.array_equal(evaluation.raw_residual_wrench, raw_wrench))

    def test_pairwise_distance_monotonicity_and_indefinite_failure(self):
        rng = np.random.default_rng(18)
        coordinate = rng.standard_normal((3, 13))
        overlap = []
        conservative = []
        for _index in range(3):
            left = rng.standard_normal((13, 13))
            base = left @ left.T + 0.5 * np.eye(13)
            right = rng.standard_normal((13, 13))
            overlap.append(base)
            conservative.append(base + right @ right.T)
        _squared_overlap, distance_overlap = _pairwise_distance(
            coordinate, np.asarray(overlap)
        )
        _squared_conservative, distance_conservative = _pairwise_distance(
            coordinate, np.asarray(conservative)
        )
        self.assertTrue(
            np.all(distance_conservative <= distance_overlap + 2e-13)
        )
        invalid = np.repeat(np.eye(2)[None, :, :], 2, axis=0)
        invalid[1, 0, 0] = -2.0
        with self.assertRaises(ValueError):
            _pairwise_distance(np.zeros((2, 2)), invalid)

    def test_three_gaussian_product_matches_full_rank_formula(self):
        rng = np.random.default_rng(23)
        coordinate = rng.standard_normal((3, 13))
        covariance = []
        for _index in range(3):
            factor = rng.standard_normal((13, 13))
            covariance.append(factor @ factor.T + np.eye(13))
        covariance = np.asarray(covariance)
        result = fuse_quotient_gaussians(coordinate, covariance)
        precision = np.sum(np.linalg.inv(covariance), axis=0)
        expected_covariance = np.linalg.inv(precision)
        expected_coordinate = expected_covariance @ np.sum(
            np.einsum("nij,nj->ni", np.linalg.inv(covariance), coordinate),
            axis=0,
        )
        self.assertEqual(result["rank"], 13)
        self.assertTrue(
            np.allclose(result["covariance"], expected_covariance, atol=2e-13)
        )
        self.assertTrue(
            np.allclose(result["coordinate"], expected_coordinate, atol=2e-13)
        )


if __name__ == "__main__":
    unittest.main()
