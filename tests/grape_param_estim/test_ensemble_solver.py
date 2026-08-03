import unittest
from dataclasses import dataclass

import numpy as np

from grape_param_estim.ensemble_solver import (
    EstimationCancelled,
    InitialPriorForecastError,
    run_ensemble_space_ienks,
)


@dataclass(frozen=True)
class Configuration:
    maximum_iterations: int = 4
    convergence_tolerance: float = 1.0e-8
    minimum_line_search_step: float = 1.0 / 64.0
    maximum_initial_prior_backoff_trials: int = 8


class EnsembleSolverTest(unittest.TestCase):
    def setUp(self):
        generator = np.random.RandomState(4)
        self.prior = generator.normal(size=(8, 2))

    def test_linear_pose_residual_moves_ensemble_toward_solution(self):
        target = np.asarray((0.8, -0.4))

        def residual(controls):
            return 4.0 * (controls - target)

        result = run_ensemble_space_ienks(
            self.prior, residual, Configuration()
        )
        self.assertLess(
            np.linalg.norm(result.center_control - target),
            np.linalg.norm(np.mean(self.prior, axis=0) - target),
        )
        self.assertEqual(result.posterior_ensemble.shape, self.prior.shape)
        self.assertEqual(result.posterior_residuals.shape, self.prior.shape)
        self.assertEqual(result.ensemble_rank, 2)

    def test_progress_is_monotonic_and_cancel_is_cooperative(self):
        events = []
        cancelled = [False]

        def progress(stage, completed, total, message):
            events.append((stage, completed, total, message))
            if len(events) == 2:
                cancelled[0] = True

        with self.assertRaises(EstimationCancelled):
            run_ensemble_space_ienks(
                self.prior,
                lambda controls: controls,
                Configuration(),
                progress_callback=progress,
                cancel_requested=lambda: cancelled[0],
            )
        completed = [event[1] for event in events]
        self.assertEqual(completed, sorted(completed))

    def test_rank_deficient_prior_is_rejected(self):
        prior = np.ones((4, 2))
        with self.assertRaises(ValueError):
            run_ensemble_space_ienks(
                prior, lambda controls: controls, Configuration()
            )

    def test_high_dimensional_control_uses_member_ensemble_subspace(self):
        generator = np.random.RandomState(8)
        prior = generator.normal(size=(7, 12))
        result = run_ensemble_space_ienks(
            prior,
            lambda controls: controls[:, :3],
            Configuration(maximum_iterations=2),
        )
        self.assertEqual(result.ensemble_rank, 6)
        self.assertEqual(result.posterior_ensemble.shape, (7, 12))

    def test_initial_prior_uses_one_global_dyadic_backoff_without_dropping_rows(self):
        requested = self.prior.copy()
        center = np.mean(requested, axis=0)
        maximum_radius = np.max(np.abs(requested - center))
        threshold = 0.75 * maximum_radius

        def residual(controls):
            if (
                controls.shape[0] == requested.shape[0]
                and np.max(np.abs(controls - center)) > threshold
            ):
                raise ValueError("synthetic member divergence")
            return controls - np.asarray((0.2, -0.1))

        result = run_ensemble_space_ienks(
            requested,
            residual,
            Configuration(maximum_iterations=1),
        )
        audit = result.initial_prior_forecast
        self.assertEqual(audit.backoff_trials, 1)
        self.assertEqual(audit.radial_scale, 0.5)
        self.assertEqual(len(audit.failures), 1)
        self.assertEqual(audit.failures[0].radial_scale, 1.0)
        self.assertEqual(audit.failures[0].exception_type, "ValueError")
        np.testing.assert_array_equal(
            result.requested_prior_ensemble, requested
        )
        np.testing.assert_allclose(
            result.prior_ensemble,
            center + 0.5 * (requested - center),
            rtol=0.0,
            atol=1.0e-15,
        )
        np.testing.assert_allclose(
            np.mean(result.prior_ensemble, axis=0), center
        )
        self.assertEqual(result.prior_ensemble.shape, requested.shape)
        self.assertEqual(audit.requested_rank, audit.effective_rank)

    def test_initial_prior_exhaustion_reports_every_scale_and_reason(self):
        events = []

        def residual(controls):
            if controls.shape[0] > 1:
                raise FloatingPointError("always divergent")
            return controls

        with self.assertRaisesRegex(
            InitialPriorForecastError,
            r"scale 1 FloatingPointError: always divergent; scale 0.5",
        ):
            run_ensemble_space_ienks(
                self.prior,
                residual,
                Configuration(
                    maximum_iterations=1,
                    maximum_initial_prior_backoff_trials=1,
                ),
                progress_callback=lambda *event: events.append(event),
            )
        failures = [
            event
            for event in events
            if event[0] == "initial_prior_forecast_failed"
        ]
        self.assertEqual(len(failures), 2)
        self.assertIn("radial scale 1", failures[0][3])
        self.assertIn("radial scale 0.5", failures[1][3])

    def test_later_ensemble_failure_is_not_reinterpreted_as_prior_backoff(self):
        ensemble_batches = [0]

        def residual(controls):
            if controls.shape[0] == self.prior.shape[0]:
                ensemble_batches[0] += 1
                if ensemble_batches[0] == 2:
                    raise ValueError("posterior linearization failed")
            return controls

        with self.assertRaisesRegex(
            ValueError, "posterior linearization failed"
        ):
            run_ensemble_space_ienks(
                self.prior,
                residual,
                Configuration(maximum_iterations=1),
            )
        self.assertEqual(ensemble_batches[0], 2)


if __name__ == "__main__":
    unittest.main()
