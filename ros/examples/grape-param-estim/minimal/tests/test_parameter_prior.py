from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from _support import MINIMAL, synthetic_problem_parts
from single_bag_parameter_prior import (
    PARAMETER_PRIOR_SCHEMA,
    QUOTIENT_COMPONENT_LABELS,
    load_parameter_prior,
    quotient_value_and_jacobian,
)
from single_bag_savgol_core import (
    COMMON_SCALE_DIRECTION,
    EstimatorConfig,
    LmSettings,
    ObjectiveEvaluation,
    PHYSICAL_DIMENSION,
    SiParameterChart,
    SingleBagDynamicsProblem,
    SolverResult,
    adaptive_kkt_lm,
    estimate_single_bag,
    load_vehicle_model,
)
from single_bag_savgol_reports import arrays_payload, result_payload


def _prior_object(factors):
    return {
        "schema": PARAMETER_PRIOR_SCHEMA,
        "name": "test_prior",
        "description": "test external information",
        "role": "test",
        "factors": factors,
    }


def _factor(
    *,
    name="factor",
    quantity="cog_position_body_m",
    components=("x",),
    target=None,
    std=None,
    covariance=None,
):
    result = {
        "name": name,
        "quantity": quantity,
        "components": list(components),
        "target": {"source": "vehicle_model_nominal"} if target is None else target,
    }
    if covariance is None:
        result["std"] = [1.0e-3] * len(components) if std is None else list(std)
    else:
        result["covariance"] = covariance
    return result


class ParameterPriorTests(unittest.TestCase):
    def setUp(self):
        self.model = load_vehicle_model(MINIMAL / "grape_vehicle_model.json")
        self.chart = SiParameterChart(self.model.parameters)

    def _load(self, value):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        path = Path(temporary.name) / "prior.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        return load_parameter_prior(path, self.model)

    def test_valid_nominal_explicit_std_and_full_covariance(self):
        scalar = self._load(_prior_object([_factor()]))
        self.assertEqual(scalar.factors[0].target_source, "vehicle_model_nominal")
        explicit = self._load(
            _prior_object(
                [
                    _factor(
                        quantity="inertia_over_mass_m2",
                        components=("xx", "xy"),
                        target={"value": [0.02, -0.001]},
                        covariance=[[4.0e-8, 1.0e-8], [1.0e-8, 9.0e-8]],
                    )
                ]
            )
        )
        self.assertTrue(np.allclose(explicit.factors[0].target, (0.02, -0.001)))
        self.assertTrue(np.all(np.linalg.eigvalsh(explicit.factors[0].covariance) > 0.0))

    def test_schema_validation_rejects_all_ambiguous_or_invalid_forms(self):
        invalid = []
        wrong_schema = _prior_object([_factor()])
        wrong_schema["schema"] = "wrong"
        invalid.append(wrong_schema)
        invalid.append(_prior_object([_factor(quantity="mass_kg")]))
        invalid.append(_prior_object([_factor(components=("bad",))]))
        invalid.append(_prior_object([_factor(components=("x", "x"))]))
        invalid.append(_prior_object([_factor(target={"source": "other"})]))
        invalid.append(_prior_object([_factor(target={"source": "vehicle_model_nominal", "value": [0.0]})]))
        invalid.append(_prior_object([_factor(components=("x", "y"), target={"value": [0.0]})]))
        invalid.append(_prior_object([_factor(std=(0.0,))]))
        both = _factor()
        both["covariance"] = [[1.0]]
        invalid.append(_prior_object([both]))
        neither = _factor()
        del neither["std"]
        invalid.append(_prior_object([neither]))
        invalid.append(_prior_object([_factor(components=("x", "y"), covariance=[[1.0, 2.0], [0.0, 1.0]])]))
        invalid.append(_prior_object([_factor(components=("x", "y"), covariance=[[1.0, 2.0], [2.0, 1.0]])]))
        for index, value in enumerate(invalid):
            with self.subTest(index=index), self.assertRaises(ValueError):
                self._load(value)

    def test_quotient_analytic_jacobian_covers_all_thirteen_components(self):
        rng = np.random.default_rng(91)
        coordinate = 0.08 * rng.standard_normal(PHYSICAL_DIMENSION)
        value, analytic = quotient_value_and_jacobian(self.chart, coordinate)
        numeric = np.empty_like(analytic)
        step = 2.0e-6
        for column in range(PHYSICAL_DIMENSION):
            delta = np.zeros(PHYSICAL_DIMENSION)
            delta[column] = step
            positive = quotient_value_and_jacobian(self.chart, coordinate + delta)[0]
            negative = quotient_value_and_jacobian(self.chart, coordinate - delta)[0]
            numeric[:, column] = (positive - negative) / (2.0 * step)
        self.assertEqual(value.shape, (len(QUOTIENT_COMPONENT_LABELS),))
        self.assertTrue(np.allclose(analytic, numeric, rtol=3e-7, atol=3e-9))

    def test_prior_whitening_jacobian_and_exact_scale_gauge(self):
        prior = self._load(
            _prior_object(
                [
                    _factor(
                        name="cog",
                        quantity="cog_position_body_m",
                        components=("x", "y", "z"),
                        std=(0.01, 0.02, 0.03),
                    ),
                    _factor(
                        name="inertia",
                        quantity="inertia_over_mass_m2",
                        components=("xx", "yy", "zz", "xy", "xz", "yz"),
                        std=(0.01,) * 6,
                    ),
                    _factor(
                        name="force",
                        quantity="force_effectiveness_over_mass",
                        components=("rotor_1", "rotor_2", "rotor_3", "rotor_4"),
                        covariance=[
                            [4.0e-4, 1.0e-4, 0.0, 0.0],
                            [1.0e-4, 3.0e-4, 0.0, 0.0],
                            [0.0, 0.0, 5.0e-4, 1.0e-4],
                            [0.0, 0.0, 1.0e-4, 4.0e-4],
                        ],
                    )
                ]
            )
        )
        coordinate = np.linspace(-0.04, 0.05, PHYSICAL_DIMENSION)
        evaluation = prior.evaluate(self.chart, coordinate)
        step = 1.0e-6
        numeric = np.empty_like(evaluation.jacobian)
        for column in range(PHYSICAL_DIMENSION):
            delta = np.zeros(PHYSICAL_DIMENSION)
            delta[column] = step
            numeric[:, column] = (
                prior.evaluate(self.chart, coordinate + delta).residual
                - prior.evaluate(self.chart, coordinate - delta).residual
            ) / (2.0 * step)
        shifted = prior.evaluate(self.chart, coordinate + 0.7 * COMMON_SCALE_DIRECTION)
        self.assertTrue(np.allclose(evaluation.jacobian, numeric, rtol=2e-7, atol=2e-8))
        self.assertTrue(np.allclose(evaluation.residual, shifted.residual, rtol=0.0, atol=2e-12))
        self.assertLess(np.linalg.norm(evaluation.jacobian @ COMMON_SCALE_DIRECTION), 2e-11)

    def test_prior_rows_are_active_in_smooth_and_strict_objectives_with_zero_lag_column(self):
        dataset, model, actuator = synthetic_problem_parts()
        prior = self._load(_prior_object([_factor(components=("x", "z"), std=(0.01, 0.02))]))
        plain = SingleBagDynamicsProblem(dataset, model, actuator)
        active = SingleBagDynamicsProblem(dataset, model, actuator, parameter_prior=prior)
        coordinate = np.linspace(-0.01, 0.01, 15)
        plain_residual, plain_jacobian, _ = plain.global_residual_jacobian(
            coordinate, command_mode="smooth", epsilon=0.5
        )
        residual, jacobian, _ = active.global_residual_jacobian(
            coordinate, command_mode="smooth", epsilon=0.5
        )
        prior_evaluation = prior.evaluate(active.chart, coordinate[:14])
        self.assertTrue(np.array_equal(residual[: plain_residual.size], plain_residual))
        self.assertTrue(np.array_equal(jacobian[: plain_residual.size], plain_jacobian))
        self.assertTrue(np.array_equal(residual[-2:], prior_evaluation.residual))
        self.assertTrue(np.array_equal(jacobian[-2:, :14], prior_evaluation.jacobian))
        self.assertTrue(np.array_equal(jacobian[-2:, 14], np.zeros(2)))
        self.assertLess(
            np.linalg.norm(jacobian[:, :14] @ COMMON_SCALE_DIRECTION),
            2e-10,
        )
        strict_residual, strict_jacobian, _ = active.physical_residual_jacobian(coordinate[:14], coordinate[14])
        self.assertTrue(np.array_equal(strict_residual[-2:], prior_evaluation.residual))
        self.assertTrue(np.array_equal(strict_jacobian[-2:], prior_evaluation.jacobian))

    def test_no_prior_path_returns_the_original_arrays_exactly(self):
        dataset, model, actuator = synthetic_problem_parts()
        problem = SingleBagDynamicsProblem(dataset, model, actuator, parameter_prior=None)
        coordinate = np.linspace(-0.01, 0.01, 15)
        direct = problem.evaluate_physical(coordinate[:14], coordinate[14], command_mode="smooth", epsilon=0.25)
        residual, jacobian, returned = problem.global_residual_jacobian(coordinate, command_mode="smooth", epsilon=0.25)
        expected_jacobian = np.zeros((direct.residual_vector.size, 15))
        expected_jacobian[:, :14] = direct.jacobian_matrix
        expected_jacobian[:, 14] = direct.whitened_lag_jacobian.reshape(-1)
        self.assertTrue(
            np.array_equal(returned.residual_vector, direct.residual_vector)
        )
        self.assertTrue(np.array_equal(residual, direct.residual_vector))
        self.assertTrue(np.array_equal(jacobian, expected_jacobian))

    def test_explicit_none_reproduces_initialization_gauge_and_lag_path(self):
        dataset, model, actuator = synthetic_problem_parts()
        config = EstimatorConfig(
            lag_mode="zero",
            lag_continuation_enabled=False,
            strict_max_nfev=1,
        )
        default = estimate_single_bag(
            SingleBagDynamicsProblem(dataset, model, actuator), config
        )
        explicit_none = estimate_single_bag(
            SingleBagDynamicsProblem(
                dataset, model, actuator, parameter_prior=None
            ),
            config,
        )
        self.assertTrue(
            np.array_equal(
                default.physical_coordinate,
                explicit_none.physical_coordinate,
            )
        )
        self.assertEqual(default.rotor_lag_seconds, explicit_none.rotor_lag_seconds)
        self.assertTrue(
            np.array_equal(
                default.evaluation.residual_vector,
                explicit_none.evaluation.residual_vector,
            )
        )
        self.assertEqual(
            default.diagnostics["lag"], explicit_none.diagnostics["lag"]
        )
        self.assertEqual(
            default.ridge["j_v_scale_norm"],
            explicit_none.ridge["j_v_scale_norm"],
        )
        self.assertEqual(default.prior_diagnostics, {"active": False})
        self.assertEqual(explicit_none.prior_diagnostics, {"active": False})

    def test_tighter_finite_factor_pulls_closer_and_charges_more_data_loss(self):
        target = quotient_value_and_jacobian(
            self.chart, np.zeros(PHYSICAL_DIMENSION)
        )[0][6]

        def solve(std):
            prior = (
                None
                if std is None
                else self._load(_prior_object([_factor(std=(std,))]))
            )

            def evaluator(coordinate):
                quotient, jacobian = quotient_value_and_jacobian(
                    self.chart, coordinate
                )
                # Synthetic data factor prefers a 0.1 m CoG-x displacement.
                data_residual = np.asarray(
                    (
                        10.0 * (quotient[6] - target - 0.1),
                        10.0 * (quotient[7] - quotient_at_origin[7] - 0.05),
                    )
                )
                data_jacobian = 10.0 * jacobian[6:8]
                if prior is None:
                    return ObjectiveEvaluation(data_residual, data_jacobian)
                prior_evaluation = prior.evaluate(self.chart, coordinate)
                return ObjectiveEvaluation(
                    np.concatenate((data_residual, prior_evaluation.residual)),
                    np.vstack((data_jacobian, prior_evaluation.jacobian)),
                )

            result = adaptive_kkt_lm(
                evaluator,
                np.zeros(PHYSICAL_DIMENSION),
                settings=LmSettings(),
                max_nfev=100,
                gauge_direction=COMMON_SCALE_DIRECTION,
            )
            quotient = quotient_value_and_jacobian(
                self.chart, result.coordinate
            )[0]
            error = abs(quotient[6] - target)
            data_loss = 0.5 * float(
                (10.0 * (quotient[6] - target - 0.1)) ** 2
                + (10.0 * (quotient[7] - quotient_at_origin[7] - 0.05)) ** 2
            )
            return result, error, data_loss, quotient[7]

        quotient_at_origin = quotient_value_and_jacobian(
            self.chart, np.zeros(PHYSICAL_DIMENSION)
        )[0]
        plain, plain_error, plain_data_loss, plain_y = solve(None)
        moderate, moderate_error, moderate_data_loss, moderate_y = solve(0.2)
        tight, tight_error, tight_data_loss, tight_y = solve(0.01)
        self.assertTrue(plain.success)
        self.assertTrue(moderate.success)
        self.assertTrue(tight.success)
        self.assertLess(moderate_error, plain_error)
        self.assertLess(tight_error, moderate_error)
        self.assertGreater(moderate_data_loss, plain_data_loss)
        self.assertGreater(tight_data_loss, moderate_data_loss)
        self.assertGreater(tight_error, 0.0)
        self.assertAlmostEqual(
            plain_y - quotient_at_origin[7], 0.05, places=10
        )
        self.assertAlmostEqual(moderate_y, plain_y, places=7)
        self.assertAlmostEqual(tight_y, plain_y, places=7)
        self.assertTrue(np.all(np.isfinite(tight.coordinate)))

    def test_completed_point_survives_postfit_uncertainty_failure(self):
        dataset, model, actuator = synthetic_problem_parts()
        prior = self._load(_prior_object([_factor()]))
        problem = SingleBagDynamicsProblem(
            dataset, model, actuator, parameter_prior=prior
        )

        def immediate_success(_problem, initial, evaluator, **_kwargs):
            objective = evaluator(np.asarray(initial, dtype=float))
            return SolverResult(
                coordinate=np.asarray(initial, dtype=float),
                evaluation=ObjectiveEvaluation(objective.residual, objective.jacobian, objective.payload),
                success=True,
                status="gtol",
                message="synthetic success",
                nfev=1,
                elapsed_seconds=0.0,
            )

        with patch("single_bag_savgol_core._solve_stage", side_effect=immediate_success), patch(
            "single_bag_savgol_core.residual_wrench_uncertainty",
            side_effect=RuntimeError("deliberate postfit failure"),
        ):
            result = estimate_single_bag(
                problem,
                EstimatorConfig(lag_mode="zero", lag_continuation_enabled=False),
            )
        self.assertTrue(result.success)
        self.assertEqual(result.optimization_status, "completed")
        self.assertEqual(result.postfit_uncertainty_status, "failed")
        self.assertEqual(result.overall_case_status, "point_estimate_completed")
        self.assertIsNone(result.uncertainty)
        self.assertTrue(np.all(np.isfinite(result.physical_coordinate)))
        self.assertTrue(np.isfinite(result.evaluation.cost))
        self.assertIn("prior_objective_sum", result.prior_diagnostics)
        self.assertAlmostEqual(
            result.prior_diagnostics["total_objective_sum"],
            result.evaluation.cost
            + result.prior_diagnostics["prior_objective_sum"],
        )
        self.assertEqual(result.postfit_uncertainty_failure["failure_stage"], "residual_wrench_uncertainty")
        augmented = result.prior_diagnostics[
            "parameter_covariance_prior_augmented_local_curvature"
        ]
        self.assertEqual(augmented.shape, (14, 14))
        self.assertTrue(np.allclose(augmented, augmented.T))
        self.assertLess(
            np.linalg.norm(augmented @ COMMON_SCALE_DIRECTION), 2e-8
        )
        payload = result_payload(
            case_name="postfit_failure",
            source_revision="abc",
            model=model,
            result=result,
            replay=None,
        )
        self.assertEqual(payload["optimization_status"], "completed")
        self.assertEqual(payload["postfit_uncertainty_status"], "failed")
        self.assertEqual(payload["status"], "point_estimate_completed")
        self.assertIn("prior_objective_sum", payload["optimization_objective"])
        arrays = arrays_payload(dataset, result, replay=None)
        self.assertIn("physical_coordinate", arrays)
        self.assertIn("prior_residual", arrays)
        self.assertNotIn("parameter_covariance_wrench_corrected", arrays)


if __name__ == "__main__":
    unittest.main()
