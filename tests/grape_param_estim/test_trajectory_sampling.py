from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from grape_param_estim.artifact_io import (
    ArtifactValidationError,
    load_npz_strict,
    read_json,
    write_npz_atomic,
    write_json_atomic,
)
from grape_param_estim.batch.variables import VariableKey, VariableKind
from grape_param_estim.batch_artifact import (
    file_sha256,
    load_batch_estimation_run,
    write_batch_estimation_run,
)
from grape_param_estim.batch_artifact_export import (
    ArtifactRunIdentity,
    export_batch_estimation_artifact_payload,
)
from grape_param_estim.batch_request import validate_batch_estimation_request
from grape_param_estim.posterior.delayed_acceptance import TargetEvaluation
from grape_param_estim.posterior.laplace_target import (
    ConditionalTrajectoryWarmStart,
    LaplaceMarginalTarget,
    factorize_bag_local_hessian,
)
from grape_param_estim.real_estimation import RealEstimationInputs
from grape_param_estim.trajectory_sampling import (
    CONDITIONAL_TRAJECTORY_SELECTION_POLICY,
    sample_selected_conditional_trajectories,
    select_conditional_trajectory_draws,
)
import tests.grape_param_estim.test_batch_artifact_export as artifact_support


class ConditionalTrajectorySamplingTests(unittest.TestCase):
    def setUp(self):
        self.helper = artifact_support.BatchArtifactExportTests()
        self.helper.setUp()
        payload = self.helper._mcmc_payload_request(
            self.helper.helper.payload
        )
        self.request = validate_batch_estimation_request(payload)
        self.inputs = RealEstimationInputs(
            request=self.request,
            flight_data=(self.helper.helper.flight,),
            initializations=(self.helper.helper.initialization,),
            parameter_chart=self.helper.helper.chart,
            geometry=self.helper.helper.geometry,
            actuator_parameters=self.helper.helper.actuators,
            scaling=self.helper.prepared.scaling,
            loading_seconds=0.0,
        )
        self.chains, self.diagnostics = self.helper._chains_and_diagnostics()

    def tearDown(self):
        self.helper.tearDown()

    def _fresh_evaluation(self, calls):
        objective = self.helper.solution.lm.objective
        log_determinant = factorize_bag_local_hessian(
            self.helper.solution.final_linearization.sparse
        ).value
        bounds = self.request.payload["delay"]["bounds_seconds"]
        delay_log_prior = -float(np.log(bounds[1] - bounds[0]))
        state = self.helper.solution.lm.state

        def evaluate(_target, point, warm_start=None):
            calls.append((point, warm_start))
            return TargetEvaluation(
                point=point,
                log_density=(
                    delay_log_prior - objective - 0.5 * log_determinant
                ),
                successful=True,
                failure_reason="",
                inner_iterations=3,
                warm_start=ConditionalTrajectoryWarmStart(
                    state, point.exact_cache_key
                ),
                graph_objective=objective,
                local_log_determinant=log_determinant,
                delay_log_prior=delay_log_prior,
            )

        return evaluate

    def test_selects_bounded_draws_and_freshly_materializes_every_bag(self):
        selected = select_conditional_trajectory_draws(
            self.chains, maximum_sample_count=2
        )
        self.assertEqual(
            tuple(value.sample_id for value in selected),
            ("sample-0-0", "sample-1-3"),
        )
        calls = []
        progress = []
        with patch.object(
            LaplaceMarginalTarget,
            "evaluate",
            autospec=True,
            side_effect=self._fresh_evaluation(calls),
        ):
            result = sample_selected_conditional_trajectories(
                self.inputs,
                "recorded-mode",
                self.helper.solution,
                self.chains,
                maximum_sample_count=2,
                progress=lambda *value: progress.append(value),
            )
        self.assertEqual(len(calls), 2)
        self.assertTrue(
            all(
                isinstance(warm_start, ConditionalTrajectoryWarmStart)
                for _point, warm_start in calls
            )
        )
        self.assertEqual(len(result.trajectories), 2)
        self.assertEqual(
            result.selection.selected_sample_ids,
            ("sample-0-0", "sample-1-3"),
        )
        self.assertEqual(
            result.selection.manifest_payload["policy"],
            CONDITIONAL_TRAJECTORY_SELECTION_POLICY,
        )
        self.assertEqual(progress[0][:2], (0, 2))
        self.assertEqual(progress[-1][:2], (2, 2))
        static_key = VariableKey(VariableKind.STATIC_PARAMETERS)
        for trajectory in result.trajectories:
            np.testing.assert_array_equal(
                trajectory.state.value(static_key),
                self.chains[0].static_coordinate[0],
            )
            self.assertEqual(
                trajectory.dynamics_residual.shape[1], 6
            )
            self.assertAlmostEqual(
                trajectory.conditional_objective,
                self.helper.solution.lm.objective,
            )

    def test_rejects_fake_retained_objective_and_honors_cancel_boundary(self):
        from dataclasses import replace

        forged_graph = self.chains[0].graph_objective.copy()
        forged_graph[0] += 1.0
        forged_log_density = self.chains[0].log_density.copy()
        forged_log_density[0] -= 1.0
        forged = (
            replace(
                self.chains[0],
                graph_objective=forged_graph,
                log_density=forged_log_density,
            ),
            self.chains[1],
        )
        calls = []
        with patch.object(
            LaplaceMarginalTarget,
            "evaluate",
            autospec=True,
            side_effect=self._fresh_evaluation(calls),
        ):
            with self.assertRaisesRegex(
                RuntimeError, "objective disagrees"
            ):
                sample_selected_conditional_trajectories(
                    self.inputs,
                    "recorded-mode",
                    self.helper.solution,
                    forged,
                    maximum_sample_count=1,
                )
        with patch.object(
            LaplaceMarginalTarget, "evaluate", autospec=True
        ) as evaluate:
            with self.assertRaisesRegex(RuntimeError, "cancelled"):
                sample_selected_conditional_trajectories(
                    self.inputs,
                    "recorded-mode",
                    self.helper.solution,
                    self.chains,
                    maximum_sample_count=1,
                    cancellation_requested=lambda: True,
                )
        evaluate.assert_not_called()

    def test_loader_rejects_forged_conditional_objective(self):
        selected, selection = self.helper._selected_trajectory_and_policy(
            self.chains
        )
        payload = export_batch_estimation_artifact_payload(
            request=self.request,
            flight_data=(self.helper.helper.flight,),
            initializations=(self.helper.helper.initialization,),
            final_solution=self.helper.solution,
            em_result=self.helper.em_result,
            static_geometry=self.helper.solution.static_geometry(),
            final_q_lag_profile=self.helper._final_q_lag_profile(),
            delay_geometry=self.helper._delay_geometry(),
            identity=ArtifactRunIdentity(
                estimator_revision="test-estimator-revision",
                configuration_fingerprint="sha256:" + "1" * 64,
                controller_snapshot_fingerprint="sha256:" + "2" * 64,
            ),
            performance=self.helper._performance(mcmc=True),
            mcmc_chains=self.chains,
            mcmc_diagnostics=self.diagnostics,
            selected_trajectories=selected,
            trajectory_selection=selection,
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "run"
            written = write_batch_estimation_run(
                root, **payload.writer_arguments
            )
            trajectory_path = root / written.manifest["artifacts"][
                "trajectories"
            ]["flight-a"]["path"]
            arrays = dict(load_npz_strict(trajectory_path))
            arrays["conditional_objective"] = (
                arrays["conditional_objective"] + 1.0
            )
            write_npz_atomic(trajectory_path, arrays)
            manifest = read_json(root / "manifest.json")
            manifest["artifacts"]["trajectories"]["flight-a"][
                "sha256"
            ] = file_sha256(trajectory_path)
            write_json_atomic(root / "manifest.json", manifest)
            with self.assertRaisesRegex(
                ArtifactValidationError, "not the retained target"
            ):
                load_batch_estimation_run(root)


if __name__ == "__main__":
    unittest.main()
