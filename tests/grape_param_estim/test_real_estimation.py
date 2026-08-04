from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from grape_param_estim.batch.evidence import (
    compute_delay_static_laplace_geometry,
)
from grape_param_estim.batch.lag_profile import (
    LagProfilePoint,
    LagProfileResult,
)
from grape_param_estim.batch.variables import VariableKind
from grape_param_estim.batch_artifact import file_sha256
from grape_param_estim.batch_request import validate_batch_estimation_request
from grape_param_estim.posterior.delayed_acceptance import TargetEvaluation
from grape_param_estim.posterior.laplace_target import (
    ConditionalTrajectoryWarmStart,
)
from grape_param_estim.posterior.mcmc import McmcCancelled
from grape_param_estim.real_estimation import (
    RealEstimationInputs,
    prepare_real_estimation_inputs,
    production_state_scaling,
    sample_laplace_solution,
)
import tests.grape_param_estim.test_batch_artifact_export as artifact_support
from tests.grape_param_estim.test_batch_preparation import (
    _flight_data,
    _request_payload,
)


def _profile(points):
    coordinate = np.linspace(-0.2, 0.3, 18)
    values = tuple(
        LagProfilePoint(
            lag=float(lag),
            phase="coarse",
            objective=float(objective),
            converged=True,
            inner_iterations=1,
            termination_reason="synthetic",
            warm_start_lag=None,
            approximate_marginal_objective=float(objective) + 0.5,
            static_coordinate=coordinate,
        )
        for lag, objective in points
    )
    best = min(values, key=lambda value: (value.objective, value.lag))
    return LagProfileResult(
        best_lag=best.lag,
        best_objective=best.objective,
        best_state=None,
        initial_refinement_bracket=(0.0, 0.08),
        final_refinement_bracket=(0.0, 0.08),
        points=values,
    )


class RealEstimationTests(unittest.TestCase):
    def test_delay_uncertainty_uses_positive_local_profile_curvature(self):
        center = 0.035
        standard_deviation = 0.004
        profile = _profile(
            (
                (lag, 9.0 + 0.5 * ((lag - center) / standard_deviation) ** 2)
                for lag in (0.025, 0.03, 0.035, 0.04, 0.045)
            )
        )
        center_coordinate = next(
            point.static_coordinate
            for point in profile.points
            if point.lag == profile.best_lag
        )
        result = compute_delay_static_laplace_geometry(
            (profile,),
            (0.0, 0.08),
            np.eye(18),
            profile.best_lag,
            center_coordinate,
            1.0e-5,
        )
        self.assertTrue(result.valid)
        self.assertEqual(result.source, "three_point_final_q_map_profile_curvature")
        self.assertAlmostEqual(
            result.standard_deviation_seconds, standard_deviation, places=10
        )

    def test_delay_uncertainty_reports_uniform_prior_fallback(self):
        result = compute_delay_static_laplace_geometry(
            (),
            (0.0, 0.12),
            np.eye(18),
            0.04,
            np.zeros(18),
            1.0e-5,
        )
        self.assertIsNone(result.curvature)
        self.assertFalse(result.valid)
        self.assertEqual(result.reason, "final_q_profile_unavailable")
        self.assertAlmostEqual(
            result.standard_deviation_seconds, 0.12 / np.sqrt(12.0)
        )

    def test_real_input_preparation_authenticates_bag_and_preserves_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bag_path = root / "flight-a.bag"
            bag_path.write_bytes(b"synthetic real-estimation flight")
            digest = file_sha256(bag_path)
            flight = _flight_data(bag_path, digest, "flight-a")
            request = validate_batch_estimation_request(
                _request_payload(
                    root, (("flight-a", bag_path, digest),)
                )
            )
            progress = []

            def loader(bag, include_accelerometer, checkpoint):
                self.assertEqual(bag["bag_id"], "flight-a")
                self.assertFalse(include_accelerometer)
                checkpoint()
                return flight

            result = prepare_real_estimation_inputs(
                request,
                flight_loader=loader,
                progress=lambda *value: progress.append(value),
            )
            self.assertEqual(result.request, request)
            self.assertEqual(result.flight_data, (flight,))
            self.assertEqual(result.initializations[0].bag_id, "flight-a")
            self.assertEqual(progress[0][0], "preparing_trajectory")
            self.assertEqual(result.actuator_parameters.delay, 0.0)
            self.assertEqual(
                result.actuator_parameters.thrust_time_constant, 0.04
            )
            self.assertEqual(
                result.actuator_parameters.gimbal_time_constant, 0.03
            )

    def test_production_scaling_covers_every_batch_variable_kind(self):
        scaling = production_state_scaling()
        self.assertEqual(
            set(scaling.kind_scales), set(VariableKind)
        )
        self.assertTrue(
            all(value > 0.0 for value in scaling.kind_scales.values())
        )

    def test_sampling_target_and_chain_are_resume_identical_with_fixed_map_start(self):
        helper = artifact_support.BatchArtifactExportTests()
        helper.setUp()
        try:
            payload = helper._mcmc_payload_request(helper.helper.payload)
            request = validate_batch_estimation_request(payload)
            inputs = RealEstimationInputs(
                request=request,
                flight_data=(helper.helper.flight,),
                initializations=(helper.helper.initialization,),
                parameter_chart=helper.helper.chart,
                geometry=helper.helper.geometry,
                actuator_parameters=helper.helper.actuators,
                scaling=helper.prepared.scaling,
                loading_seconds=0.0,
            )
            received_warm_starts = []

            class HistoryWarmStart:
                pass

            class WarmStartSensitiveTarget:
                def __init__(self, _factory, _delay_prior, _settings):
                    pass

                def __call__(self, point, warm_start=None):
                    received_warm_starts.append(warm_start)
                    fixed = isinstance(
                        warm_start, ConditionalTrajectoryWarmStart
                    )
                    history_offset = 0.0 if fixed else 0.125
                    objective = (
                        0.5
                        * float(
                            point.static_coordinate
                            @ point.static_coordinate
                        )
                        + history_offset
                    )
                    local_log_determinant = (
                        0.2
                        + 0.01 * float(point.static_coordinate[0] ** 2)
                    )
                    delay_log_prior = 0.0
                    return TargetEvaluation(
                        point=point,
                        log_density=(
                            delay_log_prior
                            - objective
                            - 0.5 * local_log_determinant
                        ),
                        successful=True,
                        failure_reason="",
                        inner_iterations=1,
                        warm_start=HistoryWarmStart(),
                        graph_objective=objective,
                        local_log_determinant=local_log_determinant,
                        delay_log_prior=delay_log_prior,
                    )

            common = dict(
                inputs=inputs,
                mode_id="recorded-mode",
                final=helper.solution,
                static_geometry=helper.solution.static_geometry(),
                delay_static_geometry=helper._delay_geometry(),
            )
            with patch(
                "grape_param_estim.real_estimation.LaplaceMarginalTarget",
                WarmStartSensitiveTarget,
            ):
                uninterrupted = sample_laplace_solution(**common)
                latest = {}
                cancelled = [False]

                def checkpoint(chain_id, value):
                    latest[chain_id] = value
                    if chain_id == "chain-000" and value.completed_transition == 2:
                        cancelled[0] = True

                with self.assertRaises(McmcCancelled):
                    sample_laplace_solution(
                        **common,
                        cancellation_requested=lambda: cancelled[0],
                        checkpoint_chain_proposal=checkpoint,
                    )
                resumed = sample_laplace_solution(
                    **common,
                    chain_checkpoints={"chain-000": latest["chain-000"]},
                )

            self.assertTrue(received_warm_starts)
            self.assertTrue(
                all(
                    isinstance(value, ConditionalTrajectoryWarmStart)
                    for value in received_warm_starts
                )
            )
            expected_cache_key = np.asarray(
                np.concatenate(
                    (
                        helper.solution.lm.state.value(
                            helper.solution.lm.state.layout.variable_keys[0]
                        ),
                        np.asarray((helper.solution.prepared.fixed_delay,)),
                    )
                ),
                dtype="<f8",
            ).tobytes(order="C")
            self.assertTrue(
                all(
                    value.state is helper.solution.lm.state
                    and value.source_point_cache_key == expected_cache_key
                    for value in received_warm_starts
                )
            )
            for expected, actual in zip(
                uninterrupted.chains, resumed.chains
            ):
                for name in (
                    "sample_id",
                    "draw_index",
                    "static_coordinate",
                    "delay",
                    "log_density",
                    "attempted_kernel",
                    "accepted_kernel",
                    "accepted",
                    "stage_one_accepted",
                    "stage_two_attempted",
                    "full_target_cache_hit",
                    "inner_solve_failed",
                    "inner_iterations",
                    "graph_objective",
                    "local_log_determinant",
                    "delay_log_prior",
                ):
                    np.testing.assert_array_equal(
                        getattr(actual, name),
                        getattr(expected, name),
                        err_msg=name,
                    )
        finally:
            helper.tearDown()


if __name__ == "__main__":
    unittest.main()
