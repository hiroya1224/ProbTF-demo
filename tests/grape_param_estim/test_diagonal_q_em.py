from dataclasses import replace
import unittest

import numpy as np

from grape_param_estim.diagonal_q import (
    diagonal_q_em_sufficient_statistics,
    shared_diagonal_q_m_step,
)
from grape_param_estim.diagonal_q_em import (
    LOG_Q_TOLERANCE_TERMINATION,
    MAXIMUM_ITERATIONS_TERMINATION,
    DiagonalQBagExpectation,
    DiagonalQEmConfig,
    DiagonalQInitialPilot,
    initial_diagonal_q_from_pilots,
    run_diagonal_q_em,
)
from grape_param_estim.progress import (
    CancellationToken,
    ProgressCancelled,
)


def _one_boundary_expectation(bag_id, variance, likelihood):
    standard_deviation = np.sqrt(np.asarray(variance, dtype=float))
    wrench = np.stack(
        (standard_deviation, -standard_deviation), axis=0
    )[:, None, :]
    return DiagonalQBagExpectation(
        bag_id=bag_id,
        times=np.asarray((0.0,)),
        correlation_time=0.4,
        smoothed_wrench=wrench,
        approx_log_likelihood=likelihood,
    )


class DiagonalQEmOrchestrationTest(unittest.TestCase):
    def test_initial_q_is_boundary_weighted_pilot_std_squared(self):
        first_scale = np.asarray((1.0, 2.0, 3.0, 0.1, 0.2, 0.3))
        second_scale = np.asarray((4.0, 3.0, 2.0, 0.8, 0.6, 0.4))
        first = DiagonalQInitialPilot("bag-z", 2, first_scale)
        second = DiagonalQInitialPilot("bag-a", 5, second_scale)
        floor = np.asarray((0.01, 0.02, 0.03, 0.004, 0.005, 0.006))
        expected = (
            2.0 * first_scale**2 + 5.0 * second_scale**2
        ) / 7.0

        covariance = initial_diagonal_q_from_pilots(
            (first, second), floor
        )
        reversed_covariance = initial_diagonal_q_from_pilots(
            (second, first), floor.copy()
        )

        np.testing.assert_array_equal(
            covariance.stationary_variance, expected
        )
        np.testing.assert_array_equal(
            covariance.stationary_variance,
            reversed_covariance.stationary_variance,
        )
        first_scale[:] = 100.0
        second_scale[:] = 200.0
        np.testing.assert_array_equal(
            covariance.stationary_variance, expected
        )

    def test_trace_progress_and_log_q_convergence_are_one_based(self):
        initial = np.asarray((16.0, 9.0, 4.0, 1.0, 0.25, 0.0625))
        target = np.ones(6)
        pilot = DiagonalQInitialPilot("bag-a", 1, np.sqrt(initial))
        config = DiagonalQEmConfig(
            maximum_iterations=12,
            log_q_tolerance=0.1,
            component_floor=1.0e-8,
        )
        calls = []
        progress = []

        def expectation_step(covariance, iteration):
            calls.append(
                (iteration, covariance.stationary_variance.copy())
            )
            desired = np.exp(
                0.5
                * (
                    np.log(covariance.stationary_variance)
                    + np.log(target)
                )
            )
            return (_one_boundary_expectation("bag-a", desired, -iteration),)

        result = run_diagonal_q_em(
            (pilot,),
            expectation_step,
            config,
            progress_callback=progress.append,
            run_id="trace-test",
        )

        self.assertTrue(result.converged)
        self.assertEqual(
            result.termination_reason, LOG_Q_TOLERANCE_TERMINATION
        )
        self.assertEqual(len(result.iterations), 5)
        self.assertEqual(
            [value[0] for value in calls], [1, 2, 3, 4, 5, 6]
        )
        self.assertEqual(
            [value.iteration for value in result.iterations],
            [1, 2, 3, 4, 5],
        )
        for call, trace in zip(calls[:-1], result.iterations):
            np.testing.assert_array_equal(
                call[1], trace.input_covariance.stationary_variance
            )
            self.assertEqual(
                trace.approx_log_likelihood, -float(trace.iteration)
            )
            self.assertEqual(
                trace.converged,
                trace.maximum_absolute_log_q_change
                <= config.log_q_tolerance,
            )
        self.assertTrue(result.iterations[-1].converged)
        self.assertTrue(
            all(not value.converged for value in result.iterations[:-1])
        )
        self.assertEqual(len(progress), 2 * len(result.iterations) + 1)
        self.assertEqual(
            [value.stage_id for value in progress[:-1:2]],
            ["diagonal_q_expectation"] * len(result.iterations),
        )
        self.assertEqual(
            [value.stage_id for value in progress[1:-1:2]],
            ["diagonal_q_m_step"] * len(result.iterations),
        )
        self.assertEqual(
            [value.completed_units for value in progress[:-1:2]],
            [0, 1, 2, 3, 4],
        )
        self.assertEqual(
            [value.completed_units for value in progress[1:-1:2]],
            [1, 2, 3, 4, 5],
        )
        self.assertEqual(
            progress[-1].stage_id, "diagonal_q_final_expectation"
        )
        np.testing.assert_array_equal(
            calls[-1][1], result.covariance.stationary_variance
        )
        np.testing.assert_array_equal(
            result.final_expectation_input_covariance.stationary_variance,
            result.covariance.stationary_variance,
        )
        self.assertEqual(result.final_approx_log_likelihood, -6.0)

        final = result.final_expectations[0]
        with self.assertRaisesRegex(ValueError, "member count changed"):
            replace(
                result,
                final_expectations=(
                    replace(
                        final,
                        smoothed_wrench=np.concatenate(
                            (final.smoothed_wrench, final.smoothed_wrench[:1]),
                            axis=0,
                        ),
                    ),
                ),
            )

        with self.assertRaisesRegex(ValueError, "log_q_tolerance"):
            replace(
                result,
                converged=True,
                termination_reason=MAXIMUM_ITERATIONS_TERMINATION,
            )

    def test_multiple_bags_use_existing_shared_m_step_and_sorted_trace(self):
        first_times = np.asarray((0.0, 0.15, 0.5))
        second_times = np.asarray((0.0, 0.4))
        first_wrench = np.asarray(
            (
                np.arange(18, dtype=float).reshape(3, 6) / 10.0,
                -np.arange(18, dtype=float).reshape(3, 6) / 20.0,
            )
        )
        second_wrench = np.asarray(
            (
                ((0.2,) * 6, (0.5,) * 6),
                ((-0.3,) * 6, (0.1,) * 6),
            )
        )
        first = DiagonalQBagExpectation(
            "bag-z", first_times, 0.3, first_wrench, -12.5
        )
        second = DiagonalQBagExpectation(
            "bag-a", second_times, 0.8, second_wrench, -3.25
        )
        floor = np.full(6, 1.0e-12)
        expected_update = shared_diagonal_q_m_step(
            (
                diagonal_q_em_sufficient_statistics(
                    first.bag_id,
                    first.times,
                    first.correlation_time,
                    first.smoothed_wrench,
                ),
                diagonal_q_em_sufficient_statistics(
                    second.bag_id,
                    second.times,
                    second.correlation_time,
                    second.smoothed_wrench,
                ),
            ),
            floor,
        )
        pilots = (
            DiagonalQInitialPilot("bag-z", 3, np.ones(6)),
            DiagonalQInitialPilot("bag-a", 2, np.ones(6) * 0.5),
        )

        result = run_diagonal_q_em(
            pilots,
            lambda _covariance, _iteration: (first, second),
            DiagonalQEmConfig(1, 1.0e-14, floor),
        )

        self.assertFalse(result.converged)
        self.assertEqual(
            result.termination_reason, MAXIMUM_ITERATIONS_TERMINATION
        )
        self.assertEqual(result.bag_ids, ("bag-a", "bag-z"))
        self.assertEqual(
            tuple(value.bag_id for value in result.last_expectations),
            ("bag-a", "bag-z"),
        )
        iteration = result.iterations[0]
        self.assertEqual(
            iteration.bag_approx_log_likelihoods,
            (("bag-a", -3.25), ("bag-z", -12.5)),
        )
        self.assertEqual(iteration.approx_log_likelihood, -15.75)
        self.assertEqual(iteration.update.total_boundary_count, 5)
        self.assertEqual(iteration.update.total_transition_count, 3)
        np.testing.assert_array_equal(
            iteration.update.raw_stationary_variance,
            expected_update.raw_stationary_variance,
        )
        np.testing.assert_array_equal(
            result.covariance.stationary_variance,
            expected_update.covariance.stationary_variance,
        )

    def test_component_floor_applies_to_initial_and_each_m_step(self):
        floor = np.asarray((0.1, 0.2, 0.3, 0.04, 0.05, 0.06))
        pilot = DiagonalQInitialPilot("bag-a", 2, np.zeros(6))
        expectation = DiagonalQBagExpectation(
            "bag-a",
            np.asarray((0.0, 0.2)),
            0.5,
            np.zeros((3, 2, 6)),
            -1.0,
        )

        result = run_diagonal_q_em(
            (pilot,),
            lambda _covariance, _iteration: (expectation,),
            DiagonalQEmConfig(3, 1.0e-12, floor),
        )

        self.assertTrue(result.converged)
        self.assertEqual(len(result.iterations), 1)
        np.testing.assert_array_equal(
            result.initial_covariance.stationary_variance, floor
        )
        np.testing.assert_array_equal(
            result.covariance.stationary_variance, floor
        )
        np.testing.assert_array_equal(
            result.iterations[0].update.raw_stationary_variance,
            np.zeros(6),
        )
        np.testing.assert_array_equal(
            result.iterations[0].update.floor_applied,
            np.ones(6, dtype=bool),
        )

    def test_cancellation_is_checked_before_and_after_injected_work(self):
        pilot = DiagonalQInitialPilot("bag-a", 1, np.ones(6))
        config = DiagonalQEmConfig(3, 1.0e-6, 1.0e-8)
        pre_cancelled = CancellationToken()
        pre_cancelled.cancel("pre_cancelled")
        calls = []

        with self.assertRaises(ProgressCancelled) as context:
            run_diagonal_q_em(
                (pilot,),
                lambda covariance, iteration: calls.append(
                    (covariance, iteration)
                ),
                config,
                cancellation_token=pre_cancelled,
            )
        self.assertEqual(context.exception.reason, "pre_cancelled")
        self.assertEqual(calls, [])

        token = CancellationToken()
        stages = []
        expectation_calls = []

        def expectation_step(covariance, iteration):
            expectation_calls.append(iteration)
            return (
                _one_boundary_expectation(
                    "bag-a", covariance.stationary_variance, -1.0
                ),
            )

        def cancel_after_m_step(event):
            stages.append(event.stage_id)
            if event.stage_id == "diagonal_q_m_step":
                token.cancel("after_first_m_step")

        with self.assertRaises(ProgressCancelled) as context:
            run_diagonal_q_em(
                (pilot,),
                expectation_step,
                config,
                progress_callback=cancel_after_m_step,
                cancellation_token=token,
            )
        self.assertEqual(context.exception.reason, "after_first_m_step")
        self.assertEqual(expectation_calls, [1])
        self.assertEqual(
            stages, ["diagonal_q_expectation", "diagonal_q_m_step"]
        )

    def test_bag_set_boundary_grid_and_tau_are_fixed(self):
        pilots = (
            DiagonalQInitialPilot("bag-a", 2, np.ones(6)),
            DiagonalQInitialPilot("bag-b", 2, np.ones(6)),
        )
        config = DiagonalQEmConfig(3, 1.0e-12, 1.0e-8)
        valid_a = DiagonalQBagExpectation(
            "bag-a", (0.0, 0.2), 0.5, np.ones((2, 2, 6)), -1.0
        )
        valid_b = DiagonalQBagExpectation(
            "bag-b", (0.0, 0.3), 0.7, np.ones((2, 2, 6)) * 2.0, -2.0
        )

        with self.assertRaisesRegex(ValueError, "bag set changed"):
            run_diagonal_q_em(
                pilots,
                lambda _covariance, _iteration: (valid_a,),
                config,
            )

        wrong_count = DiagonalQBagExpectation(
            "bag-a",
            (0.0, 0.1, 0.2),
            0.5,
            np.ones((2, 3, 6)),
            -1.0,
        )
        with self.assertRaisesRegex(ValueError, "boundary count changed"):
            run_diagonal_q_em(
                pilots,
                lambda _covariance, _iteration: (wrong_count, valid_b),
                config,
            )

        shifted_a = DiagonalQBagExpectation(
            "bag-a", (0.0, 0.25), 0.5, np.ones((2, 2, 6)), -1.0
        )

        def moving_grid(_covariance, iteration):
            return (valid_a if iteration == 1 else shifted_a, valid_b)

        with self.assertRaisesRegex(ValueError, "time grid changed"):
            run_diagonal_q_em(pilots, moving_grid, config)

        changed_correlation_time = DiagonalQBagExpectation(
            "bag-a", (0.0, 0.2), 0.6, np.ones((2, 2, 6)), -1.0
        )

        def moving_correlation_time(_covariance, iteration):
            return (
                valid_a
                if iteration == 1
                else changed_correlation_time,
                valid_b,
            )

        with self.assertRaisesRegex(ValueError, "correlation time changed"):
            run_diagonal_q_em(pilots, moving_correlation_time, config)

    def test_maximum_reason_and_reproducibility_are_strict(self):
        first = DiagonalQInitialPilot(
            "bag-z", 1, np.asarray((1.0, 1.1, 1.2, 0.3, 0.4, 0.5))
        )
        second = DiagonalQInitialPilot(
            "bag-a", 1, np.asarray((2.0, 1.8, 1.6, 0.7, 0.6, 0.5))
        )
        config = DiagonalQEmConfig(3, 1.0e-8, 1.0e-10)

        def ordered_step(covariance, iteration):
            variance = covariance.stationary_variance * 2.0
            values = (
                _one_boundary_expectation("bag-a", variance * 0.5, -2.0),
                _one_boundary_expectation("bag-z", variance * 1.5, -3.0),
            )
            return values if iteration % 2 else tuple(reversed(values))

        first_result = run_diagonal_q_em(
            (first, second), ordered_step, config
        )
        second_result = run_diagonal_q_em(
            (second, first), ordered_step, config
        )

        self.assertFalse(first_result.converged)
        self.assertEqual(
            first_result.termination_reason,
            MAXIMUM_ITERATIONS_TERMINATION,
        )
        self.assertEqual(len(first_result.iterations), 3)
        np.testing.assert_array_equal(
            first_result.initial_covariance.stationary_variance,
            second_result.initial_covariance.stationary_variance,
        )
        np.testing.assert_array_equal(
            first_result.covariance.stationary_variance,
            second_result.covariance.stationary_variance,
        )
        for left, right in zip(
            first_result.iterations, second_result.iterations
        ):
            np.testing.assert_array_equal(
                left.input_covariance.stationary_variance,
                right.input_covariance.stationary_variance,
            )
            np.testing.assert_array_equal(
                left.update.raw_stationary_variance,
                right.update.raw_stationary_variance,
            )
            self.assertEqual(
                left.bag_approx_log_likelihoods,
                right.bag_approx_log_likelihoods,
            )
            self.assertEqual(
                left.maximum_absolute_log_q_change,
                right.maximum_absolute_log_q_change,
            )

        with self.assertRaisesRegex(ValueError, "maximum_iterations"):
            replace(
                first_result,
                converged=False,
                termination_reason=LOG_Q_TOLERANCE_TERMINATION,
            )

    def test_configuration_and_duplicate_pilot_validation_is_strict(self):
        for maximum in (0, -1, 1.5, True):
            with self.subTest(maximum=maximum), self.assertRaises(ValueError):
                DiagonalQEmConfig(maximum, 1.0e-3, 1.0e-8)
        for tolerance in (0.0, -1.0, np.nan, np.inf):
            with self.subTest(tolerance=tolerance), self.assertRaises(
                ValueError
            ):
                DiagonalQEmConfig(2, tolerance, 1.0e-8)
        for floor in (0.0, -1.0, np.nan, np.ones(5), np.zeros(6)):
            with self.subTest(floor=floor), self.assertRaises(ValueError):
                DiagonalQEmConfig(2, 1.0e-3, floor)

        pilot = DiagonalQInitialPilot("bag-a", 1, np.ones(6))
        with self.assertRaisesRegex(ValueError, "unique"):
            initial_diagonal_q_from_pilots((pilot, pilot), 1.0e-8)
        with self.assertRaisesRegex(ValueError, "at least one"):
            initial_diagonal_q_from_pilots((), 1.0e-8)


if __name__ == "__main__":
    unittest.main()
