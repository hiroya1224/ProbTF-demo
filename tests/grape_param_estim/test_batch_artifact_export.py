import copy
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np

from grape_param_estim.batch.em_loop import (
    LaplaceEmIteration,
    LaplaceEmResult,
    LaplaceEmTerminationReason,
)
from grape_param_estim.batch.evidence import (
    compute_delay_static_laplace_geometry,
)
from grape_param_estim.batch.lag_profile import (
    LagProfilePoint,
    LagProfileResult,
)
from grape_param_estim.batch.laplace_em import (
    QInnerEvaluation,
    QUpdateAttempt,
    QUpdateResult,
    compute_diagonal_q_target,
)
from grape_param_estim.batch.lm import LMSettings
from grape_param_estim.batch.variables import VariableKey, VariableKind
from grape_param_estim.batch_artifact import (
    replace_batch_estimation_run,
    write_batch_estimation_run,
)
from grape_param_estim.batch_artifact_export import (
    ArtifactRunIdentity,
    BagPerformanceMeasurements,
    RunPerformanceMeasurements,
    SelectedConditionalTrajectory,
    _unit_quaternion_series,
    append_posterior_sampling_artifact_payload,
    complete_pending_mcmc_artifact_payload,
    export_batch_estimation_artifact_payload,
)
from grape_param_estim.batch_request import validate_batch_estimation_request
from grape_param_estim.estimation import solve_fixed_graph_laplace
from grape_param_estim.posterior.diagnostics import McmcDiagnostics
from grape_param_estim.posterior.laplace_target import (
    factorize_bag_local_hessian,
)
from grape_param_estim.posterior.mcmc import (
    KernelAcceptanceSummary,
    McmcChainResult,
)
from tests.grape_param_estim.test_batch_preparation import (
    BatchPreparationTests,
)


class BatchArtifactExportTests(unittest.TestCase):
    def test_quaternion_export_normalizes_and_unwraps_double_cover(self):
        values = np.asarray(
            (
                (0.0, 0.0, 0.0, 1.0000002),
                (0.0, 0.0, 0.01, -0.99995),
                (0.0, 0.0, 0.02, 0.9998),
            )
        )
        result = _unit_quaternion_series(values, "test")
        np.testing.assert_allclose(np.linalg.norm(result, axis=1), 1.0)
        self.assertTrue(
            np.all(np.sum(result[:-1] * result[1:], axis=1) >= 0.0)
        )

    def setUp(self):
        self.helper = BatchPreparationTests()
        self.helper.setUp()
        self.prepared = self.helper._prepare()

        def factory(q, delay, coordinate):
            return replace(
                self.prepared,
                dynamics=replace(self.prepared.dynamics, q=q),
                fixed_delay=delay,
                initial_parameter_coordinates=coordinate,
            )

        self.solution = solve_fixed_graph_laplace(
            factory,
            self.prepared.dynamics.q,
            self.prepared.fixed_delay,
            self.prepared.initial_parameter_coordinates,
            LMSettings(
                maximum_iterations=50,
                gradient_tolerance=1.0e-5,
                scaled_step_tolerance=1.0e-6,
                relative_objective_tolerance=1.0e-7,
            ),
        )
        self.em_result = self._em_result()

    def tearDown(self):
        self.helper.tearDown()

    def _em_result(self):
        step = self.solution.as_e_step_result()
        target = compute_diagonal_q_target(
            self.prepared.dynamics.q_definition,
            self.solution.dynamics.moments,
            self.solution.dynamics.time_step,
            np.full(6, 1.0e-8),
        )
        input_evaluation = QInnerEvaluation(
            q=step.q,
            successful=True,
            map_objective=step.map_objective,
            approximate_marginal_objective=(
                step.approximate_marginal_objective
            ),
            lag=step.lag,
            failure_reason="",
            warm_start=step.state,
        )
        failed_evaluation = QInnerEvaluation(
            q=target.target,
            successful=False,
            map_objective=float("inf"),
            approximate_marginal_objective=float("inf"),
            lag=step.lag,
            failure_reason="synthetic_final_iteration_boundary",
        )
        attempt = QUpdateAttempt(
            alpha=1.0,
            candidate_q=target.target,
            evaluation=failed_evaluation,
            accepted=False,
            rejection_reason="inner_failure",
        )
        update = QUpdateResult(
            input_evaluation=input_evaluation,
            target=target,
            attempts=(attempt,),
            accepted=False,
            accepted_q=step.q,
            accepted_alpha=0.0,
            max_log_q_change=0.0,
            termination_reason="all_damped_candidates_rejected",
        )
        iteration = LaplaceEmIteration(
            iteration=0,
            input_step=step,
            q_target=target,
            q_update=update,
            output_step=step,
            lag_refinement_failed=False,
            lag_refinement_failure_reason="",
            lag_change=0.0,
            map_objective_change=0.0,
            marginal_objective_change=0.0,
        )
        return LaplaceEmResult(
            definition=self.prepared.dynamics.q_definition,
            iterations=(iteration,),
            final_step=step,
            reason=LaplaceEmTerminationReason.MAXIMUM_ITERATIONS,
        )

    def _final_q_lag_profile(self):
        delay = self.prepared.fixed_delay
        map_objective = self.solution.lm.objective
        marginal = self.solution.marginal_objective.value
        coordinate = self.solution.lm.state.value(
            VariableKey(VariableKind.STATIC_PARAMETERS)
        )
        slope = np.linspace(-0.2, 0.3, coordinate.size)
        curvature = 1.0e6
        lags = (0.0, delay, 0.08)
        points = tuple(
            LagProfilePoint(
                lag=lag,
                phase="coarse",
                objective=(
                    map_objective + 0.5 * curvature * (lag - delay) ** 2
                ),
                converged=True,
                inner_iterations=len(self.solution.lm.iterations),
                termination_reason=self.solution.lm.reason.value,
                warm_start_lag=None if index == 0 else lags[index - 1],
                approximate_marginal_objective=(
                    marginal + 0.5 * curvature * (lag - delay) ** 2
                ),
                static_coordinate=coordinate + slope * (lag - delay),
            )
            for index, lag in enumerate(lags)
        )
        return LagProfileResult(
            best_lag=delay,
            best_objective=map_objective,
            best_state=self.solution.lm.state,
            initial_refinement_bracket=(0.0, 0.08),
            final_refinement_bracket=(0.03, 0.04),
            points=points,
        )

    def _delay_geometry(self, profile=True):
        delay = self.prepared.fixed_delay
        coordinate = self.solution.lm.state.value(
            VariableKey(VariableKind.STATIC_PARAMETERS)
        )
        delay_request = self.helper.payload["delay"]
        return compute_delay_static_laplace_geometry(
            ((self._final_q_lag_profile(),) if profile else ()),
            tuple(delay_request["bounds_seconds"]),
            self.solution.static_geometry().information.posterior.hessian,
            delay,
            coordinate,
            float(delay_request["refinement_tolerance_seconds"]),
        )

    def _performance(self, mcmc):
        bag_id = self.prepared.bags[0].bag_id
        bag_factors = tuple(
            factor
            for factor in self.solution.final_linearization.factors
            if any(
                block.variable_key.bag_id == bag_id
                for block in factor.jacobian_blocks
            )
        )
        return RunPerformanceMeasurements(
            bags=(
                BagPerformanceMeasurements(
                    bag_id=bag_id,
                    knot_count=len(self.prepared.bags[0].knots),
                    factor_count=len(bag_factors),
                    residual_dimension=sum(
                        value.residual.size for value in bag_factors
                    ),
                    jacobian_nnz=(
                        self.solution.final_linearization.sparse.jacobian.nnz
                    ),
                    assembly_seconds=0.011,
                    factorization_seconds=0.022,
                    schur_solve_seconds=0.007,
                ),
            ),
            nonlinear_iteration_seconds=(0.10, 0.08),
            em_iteration_seconds=(0.35,),
            mcmc_target_seconds=((0.20, 0.21) if mcmc else ()),
            peak_memory_bytes=1234567,
        )

    @staticmethod
    def _mcmc_payload_request(payload):
        result = copy.deepcopy(payload)
        result["run_mode"] = "estimate_and_sample"
        result["mcmc_settings"] = {
            "enabled": True,
            "chain_count": 2,
            "warmup_steps": 0,
            "retained_draws": 4,
            "thinning": 1,
            "random_seed": 17,
            "local_scale": 0.1,
            "exact_ridge_scale": 0.2,
            "near_ridge_scale": 0.1,
            "identified_scale": 0.05,
            "delay_scale_seconds": 0.001,
            "near_relative_threshold": 1.0e-6,
            "rhat_threshold": 1.01,
            "minimum_effective_sample_size": 4.0,
        }
        return result

    def _chains_and_diagnostics(self):
        coordinate = self.solution.lm.state.value(
            VariableKey(VariableKind.STATIC_PARAMETERS)
        )
        chains = []
        graph_objective = self.solution.lm.objective
        log_determinant_value = factorize_bag_local_hessian(
            self.solution.final_linearization.sparse
        ).value
        for chain_index in range(2):
            coordinates = np.tile(coordinate, (4, 1))
            graph = np.full(4, graph_objective)
            log_determinant = np.full(4, log_determinant_value)
            delay_bounds = self.helper.payload["delay"]["bounds_seconds"]
            delay_prior = np.full(
                4, -np.log(delay_bounds[1] - delay_bounds[0])
            )
            log_density = delay_prior - graph - 0.5 * log_determinant
            chains.append(
                McmcChainResult(
                    chain_id="chain-{}".format(chain_index),
                    mode_id="recorded-mode",
                    sample_id=np.asarray(
                        tuple(
                            "sample-{}-{}".format(chain_index, draw)
                            for draw in range(4)
                        )
                    ),
                    draw_index=np.arange(1, 5, dtype=np.int64),
                    static_coordinate=coordinates,
                    delay=np.full(4, self.prepared.fixed_delay),
                    log_density=log_density,
                    attempted_kernel=np.asarray(("ridge",) * 4),
                    accepted_kernel=np.asarray(("ridge", "", "ridge", "")),
                    accepted=np.asarray((True, False, True, False)),
                    stage_one_accepted=np.asarray((True, True, True, False)),
                    stage_two_attempted=np.asarray((True, True, True, False)),
                    full_target_cache_hit=np.zeros(4, dtype=bool),
                    inner_solve_failed=np.zeros(4, dtype=bool),
                    inner_iterations=np.asarray((1, 1, 1, 0), dtype=np.int64),
                    warmup_steps=0,
                    thinning=1,
                    total_transitions=4,
                    kernel_summaries={
                        "ridge": KernelAcceptanceSummary(
                            attempts=4,
                            stage_one_accepted=3,
                            stage_two_attempted=3,
                            stage_two_accepted=2,
                            full_target_cache_hits=0,
                            inner_solve_failures=0,
                            inner_iterations=3,
                        )
                    },
                    graph_objective=graph,
                    local_log_determinant=log_determinant,
                    delay_log_prior=delay_prior,
                )
            )
        aggregate = KernelAcceptanceSummary(
            attempts=8,
            stage_one_accepted=6,
            stage_two_attempted=6,
            stage_two_accepted=4,
            full_target_cache_hits=0,
            inner_solve_failures=0,
            inner_iterations=6,
        )
        diagnostics = McmcDiagnostics(
            chain_ids=("chain-0", "chain-1"),
            mode_id="recorded-mode",
            draws_per_chain=4,
            split_rhat=np.ones(19),
            effective_sample_size=np.full(19, 8.0),
            integrated_autocorrelation_time=np.ones(19),
            ridge_coordinate_trace=np.zeros((2, 4)),
            delay_trace=np.full((2, 4), self.prepared.fixed_delay),
            log_density_trace=np.stack(
                tuple(value.log_density for value in chains), axis=0
            ),
            kernel_summaries={"ridge": aggregate},
            completed=True,
            converged=True,
            rhat_threshold=1.01,
            minimum_effective_sample_size=4.0,
        )
        return tuple(chains), diagnostics

    def _selected_trajectory_and_policy(self, chains):
        dynamics = np.zeros((len(self.prepared.bags[0].knots) - 1, 6))
        valid = np.zeros(dynamics.shape[0], dtype=bool)
        for interval in self.solution.dynamics.linearizations.intervals:
            dynamics[interval.left_knot_index] = interval.residual
            valid[interval.left_knot_index] = True
        sample_id = str(chains[0].sample_id[0])
        selected = SelectedConditionalTrajectory(
            sample_id=sample_id,
            bag_id=self.prepared.bags[0].bag_id,
            state=self.solution.lm.state,
            dynamics_residual=dynamics,
            dynamics_residual_valid=valid,
            conditional_objective=float(chains[0].graph_objective[0]),
        )
        policy = {
            "policy": "deterministic_flattened_chain_draw_quantiles_v1",
            "sample_order": "chain_order_then_draw_index",
            "available_sample_count": 8,
            "maximum_sample_count": 1,
            "selected_sample_ids": [sample_id],
            "selected_bag_ids": [self.prepared.bags[0].bag_id],
            "conditional_evaluation_method": "fresh_conditional_sparse_map",
            "warm_start_policy": "selected_mode_map_local_state",
        }
        return (selected,), policy

    def test_pending_mcmc_core_is_completed_without_reexporting_map(self):
        request = validate_batch_estimation_request(
            self._mcmc_payload_request(self.helper.payload)
        )
        core = export_batch_estimation_artifact_payload(
            request=request,
            flight_data=(self.helper.flight,),
            initializations=(self.helper.initialization,),
            final_solution=self.solution,
            em_result=self.em_result,
            static_geometry=self.solution.static_geometry(),
            final_q_lag_profile=self._final_q_lag_profile(),
            delay_geometry=self._delay_geometry(),
            identity=ArtifactRunIdentity(
                estimator_revision="test-estimator-revision",
                configuration_fingerprint="sha256:" + "1" * 64,
                controller_snapshot_fingerprint="sha256:" + "2" * 64,
            ),
            performance=self._performance(mcmc=False),
            pending_mcmc_checkpoint=True,
        )
        self.assertIsNone(core.mcmc_samples)
        self.assertNotIn("mcmc", core.manifest_metadata["substage_status"])
        chains, diagnostics = self._chains_and_diagnostics()
        selected, selection = self._selected_trajectory_and_policy(chains)
        completed = complete_pending_mcmc_artifact_payload(
            core,
            request,
            self.solution,
            chains,
            diagnostics,
            (0.2, 0.21),
            (self.helper.initialization,),
            selected,
            selection,
        )
        self.assertIs(completed.map_static, core.map_static)
        self.assertEqual(completed.mcmc_samples["sample_id"].size, 8)
        self.assertIn("mcmc", completed.manifest_metadata["substage_status"])
        with tempfile.TemporaryDirectory() as temporary:
            loaded = write_batch_estimation_run(
                Path(temporary) / "run", **completed.writer_arguments
            )
        self.assertEqual(loaded.mcmc_samples["sample_id"].size, 8)

    def test_independent_sampling_replaces_estimate_only_with_rollback_boundary(self):
        request = validate_batch_estimation_request(self.helper.payload)
        core = export_batch_estimation_artifact_payload(
            request=request,
            flight_data=(self.helper.flight,),
            initializations=(self.helper.initialization,),
            final_solution=self.solution,
            em_result=self.em_result,
            static_geometry=self.solution.static_geometry(),
            final_q_lag_profile=self._final_q_lag_profile(),
            delay_geometry=self._delay_geometry(),
            identity=ArtifactRunIdentity(
                estimator_revision="test-estimator-revision",
                configuration_fingerprint="sha256:" + "1" * 64,
                controller_snapshot_fingerprint="sha256:" + "2" * 64,
            ),
            performance=self._performance(mcmc=False),
        )
        chains, diagnostics = self._chains_and_diagnostics()
        selected, selection = self._selected_trajectory_and_policy(chains)
        sampled = append_posterior_sampling_artifact_payload(
            core,
            self.solution,
            chains,
            diagnostics,
            (0.2, 0.21),
            {
                "enabled": True,
                "sampling_request_fingerprint": "sha256:" + "3" * 64,
                "upstream_estimation_request_fingerprint": request.fingerprint,
                "sampler_revision": "test-sampler",
            },
            (self.helper.initialization,),
            selected,
            selection,
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "run"
            original = write_batch_estimation_run(
                root, **core.writer_arguments
            )
            original_map_digest = original.manifest["artifacts"]["map_static"][
                "sha256"
            ]
            upgraded = replace_batch_estimation_run(
                root,
                expected_request_fingerprint=request.fingerprint,
                **sampled.writer_arguments
            )
        self.assertEqual(upgraded.mcmc_samples["sample_id"].size, 8)
        self.assertEqual(
            upgraded.manifest["artifacts"]["map_static"]["sha256"],
            original_map_digest,
        )
        self.assertEqual(
            upgraded.manifest["mcmc_settings"][
                "sampling_request_fingerprint"
            ],
            "sha256:" + "3" * 64,
        )

    def test_exports_raw_or_labelled_sha_and_round_trips_complete_mcmc_run(self):
        request = validate_batch_estimation_request(
            self._mcmc_payload_request(self.helper.payload)
        )
        chains, diagnostics = self._chains_and_diagnostics()
        selected, selection = self._selected_trajectory_and_policy(chains)
        common = dict(
            request=request,
            initializations=(self.helper.initialization,),
            final_solution=self.solution,
            em_result=self.em_result,
            static_geometry=self.solution.static_geometry(),
            final_q_lag_profile=self._final_q_lag_profile(),
            delay_geometry=self._delay_geometry(),
            identity=ArtifactRunIdentity(
                estimator_revision="test-estimator-revision",
                configuration_fingerprint="sha256:" + "1" * 64,
                controller_snapshot_fingerprint="sha256:" + "2" * 64,
            ),
            performance=self._performance(mcmc=True),
            mcmc_chains=chains,
            mcmc_diagnostics=diagnostics,
            selected_trajectories=selected,
            trajectory_selection=selection,
        )
        labelled_payload = export_batch_estimation_artifact_payload(
            flight_data=(self.helper.flight,), **common
        )
        mismatched = dict(common)
        mismatched["final_q_lag_profile"] = replace(
            self._final_q_lag_profile(),
            best_lag=self.prepared.fixed_delay + 0.001,
        )
        with self.assertRaisesRegex(ValueError, "best lag"):
            export_batch_estimation_artifact_payload(
                flight_data=(self.helper.flight,), **mismatched
            )
        raw_flight = replace(
            self.helper.flight,
            provenance=replace(
                self.helper.flight.provenance,
                bag_sha256=self.helper.flight.provenance.bag_sha256[7:],
            ),
        )
        payload = export_batch_estimation_artifact_payload(
            flight_data=(raw_flight,), **common
        )
        self.assertEqual(
            payload.manifest_metadata["selected_bag_sha256"],
            labelled_payload.manifest_metadata["selected_bag_sha256"],
        )
        with tempfile.TemporaryDirectory() as temporary:
            loaded = write_batch_estimation_run(
                Path(temporary) / "run", **payload.writer_arguments
            )
        self.assertEqual(loaded.manifest["status"], "complete")
        self.assertEqual(loaded.mcmc_samples["sample_id"].size, 8)
        self.assertEqual(
            loaded.trajectories["flight-a"]["sample_id"].tolist(),
            [str(chains[0].sample_id[0])],
        )
        self.assertIn("mcmc_split_rhat", loaded.diagnostics)
        self.assertIn("accelerometer", loaded.bags["flight-a"])
        self.assertNotIn("residual_wrench", loaded.bags["flight-a"])
        self.assertTrue(loaded.laplace["delay_profile_available"][0])
        self.assertTrue(loaded.laplace["delay_profile_curvature_valid"][0])
        self.assertEqual(
            loaded.laplace["static_covariance_conditioning"][0],
            "fixed_delay_conditional",
        )
        self.assertEqual(
            loaded.laplace["joint_parameter_delay_information"].shape,
            (19, 19),
        )
        self.assertGreater(
            np.linalg.norm(
                loaded.laplace["joint_parameter_delay_information"][:-1, -1]
            ),
            0.0,
        )
        np.testing.assert_allclose(
            loaded.q_em["expected_residual_second_moment"],
            loaded.q_em["map_residual_second_moment"]
            + loaded.q_em["covariance_correction"],
        )

        unavailable = dict(common)
        unavailable["final_q_lag_profile"] = None
        unavailable["delay_geometry"] = self._delay_geometry(profile=False)
        no_profile = export_batch_estimation_artifact_payload(
            flight_data=(raw_flight,), **unavailable
        )
        self.assertFalse(no_profile.laplace["delay_profile_available"][0])
        self.assertEqual(no_profile.laplace["delay_profile_grid"].size, 0)
        self.assertFalse(
            no_profile.laplace["delay_profile_curvature_valid"][0]
        )
        self.assertEqual(
            no_profile.laplace["joint_parameter_delay_covariance"].shape,
            (0, 0),
        )
        self.assertEqual(
            no_profile.laplace["parameter_delay_cross_covariance"].shape,
            (0,),
        )


if __name__ == "__main__":
    unittest.main()
