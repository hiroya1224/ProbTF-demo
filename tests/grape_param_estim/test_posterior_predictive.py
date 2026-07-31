from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np

from grape_param_estim.controller import (
    ControllerConfig,
    initial_controller_state,
)
from grape_param_estim.geometry import (
    correction_transform_path,
    euler_xyz_to_matrix,
    matrix_to_quaternion,
)
from grape_param_estim.mode_validation import (
    NOMINAL_MODE_ID,
    plant_wiring_mode,
)
from grape_param_estim.posterior_predictive import (
    ControllerParameterCandidate,
    PosteriorPredictiveInput,
    PosteriorPredictiveWeights,
    TrackingLossDefinition,
    apply_controller_candidate,
    default_controller_parameter_candidates,
    empirical_upper_cvar,
    evaluate_posterior_predictive,
    input_from_mode_posterior,
    input_from_real_assimilation,
    save_posterior_predictive_decision,
)
from grape_param_estim.strong_constraint import (
    CONTROL_DIMENSION,
    PARAMETER_OFFSET,
)
from grape_param_estim.strong_constraint_experiments import (
    _problem_from_synthetic,
)
from grape_param_estim.synthetic import run_synthetic_experiment
from grape_param_estim.parameterization import VehicleParameterChart
from grape_param_estim.system import (
    ActuatorParameters,
    ActuatorState,
    ControllerState,
    GrapeGeometry,
    RigidBodyState,
    VehicleParameters,
)


class PosteriorPredictiveDecisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.synthetic = run_synthetic_experiment(
            duration=0.20,
            time_step=0.04,
            truth_residual_wrench=lambda _time, _state: np.zeros(6),
            translation_noise=0.001,
            rotation_noise=0.001,
            seed=17,
        )
        cls.problem = _problem_from_synthetic(cls.synthetic)
        nominal = VehicleParameters.nominal()
        cls.members = tuple(
            replace(nominal, mass=nominal.mass * scale)
            for scale in (0.82, 1.0, 1.18)
        )
        cls.member_ids = np.asarray((9, 2, 15), dtype=np.int64)
        cls.zero_residual_wrench = np.zeros(
            (len(cls.members), cls.synthetic.truth.times.size - 1, 6)
        )
        configuration = ControllerConfig.grape()
        controller_state = initial_controller_state(
            configuration, trim_hover=True
        )
        cls.predictive_input = PosteriorPredictiveInput(
            selected_mode_id=NOMINAL_MODE_ID,
            member_ids=cls.member_ids,
            physical_parameter_members=cls.members,
            times=cls.synthetic.truth.times,
            references=cls.synthetic.references,
            initial_states=tuple(
                cls.problem.initial_state_anchor for _value in cls.members
            ),
            initial_controller_states=tuple(
                controller_state for _value in cls.members
            ),
            initial_actuator_states=(None, None, None),
            interval_residual_wrench=cls.zero_residual_wrench,
            controller_configuration=configuration,
            controller_parameters=nominal,
            controller_geometry=GrapeGeometry.grape(),
            plant_geometry=GrapeGeometry.grape(),
            actuator_parameters=ActuatorParameters(),
            provenance=(
                ("source_artifact", "/tmp/selected_mode.npz"),
                ("source_schema", "phase5-test"),
            ),
        )
        cls.candidates = (
            ControllerParameterCandidate("baseline"),
            ControllerParameterCandidate(
                "mass_and_attitude",
                controller_mass_scale=1.08,
                roll_pid_scale=0.90,
                pitch_pid_scale=0.95,
                yaw_pid_scale=1.10,
            ),
        )
        cls.loss_definition = TrackingLossDefinition(
            translation_scale=0.08,
            rotation_scale=np.deg2rad(8.0),
        )
        cls.weights = PosteriorPredictiveWeights(
            mean_tracking_loss=1.0,
            cvar_tracking_loss=0.4,
            failure_probability=2.5,
            parameter_change=0.2,
        )
        cls.decision = evaluate_posterior_predictive(
            cls.predictive_input,
            candidates=cls.candidates,
            failure_threshold=0.50,
            cvar_level=0.75,
            loss_definition=cls.loss_definition,
            weights=cls.weights,
        )

    def test_candidate_application_changes_only_declared_coordinates(self):
        configuration = ControllerConfig.grape()
        parameters = VehicleParameters.nominal()
        candidate = ControllerParameterCandidate(
            "declared",
            controller_mass_scale=1.10,
            roll_pid_scale=0.80,
            pitch_pid_scale=1.20,
            yaw_pid_scale=0.70,
        )
        changed_configuration, changed_parameters = (
            apply_controller_candidate(
                configuration, parameters, candidate
            )
        )
        self.assertAlmostEqual(
            changed_parameters.mass, 1.10 * parameters.mass
        )
        np.testing.assert_array_equal(
            changed_parameters.inertia, parameters.inertia
        )
        self.assertEqual(
            changed_configuration.pid[:3], configuration.pid[:3]
        )
        for axis, scale in ((3, 0.80), (4, 1.20), (5, 0.70)):
            before = configuration.pid[axis]
            after = changed_configuration.pid[axis]
            self.assertAlmostEqual(after.p_gain, scale * before.p_gain)
            self.assertAlmostEqual(after.i_gain, scale * before.i_gain)
            self.assertAlmostEqual(after.d_gain, scale * before.d_gain)
            self.assertEqual(after.limit_sum, before.limit_sum)

    def test_default_proposals_are_an_explicit_unique_set(self):
        candidates = default_controller_parameter_candidates()
        self.assertEqual(candidates[0].candidate_id, "baseline")
        self.assertEqual(candidates[0].scales.tolist(), [1.0] * 4)
        self.assertEqual(
            len({value.candidate_id for value in candidates}),
            len(candidates),
        )
        self.assertTrue(
            any(value.controller_mass_scale != 1.0 for value in candidates)
        )
        self.assertTrue(
            any(value.roll_pid_scale != 1.0 for value in candidates)
        )
        self.assertTrue(
            any(value.pitch_pid_scale != 1.0 for value in candidates)
        )
        self.assertTrue(
            any(value.yaw_pid_scale != 1.0 for value in candidates)
        )

    def test_empirical_cvar_integrates_fractional_tail_mass(self):
        losses = np.asarray((0.0, 1.0, 2.0, 9.0))
        self.assertAlmostEqual(empirical_upper_cvar(losses, 0.0), 3.0)
        self.assertAlmostEqual(empirical_upper_cvar(losses, 0.50), 5.5)
        self.assertAlmostEqual(empirical_upper_cvar(losses, 0.60), 6.375)
        self.assertAlmostEqual(empirical_upper_cvar(losses, 0.75), 9.0)

    def test_every_candidate_runs_every_raw_physical_member(self):
        decision = self.decision
        sample_count = self.synthetic.truth.times.size
        self.assertEqual(len(decision.evaluations), 2)
        for evaluation in decision.evaluations:
            np.testing.assert_array_equal(
                evaluation.member_ids, self.member_ids
            )
            self.assertEqual(len(evaluation.trajectories), 3)
            self.assertEqual(
                evaluation.correction_translation.shape,
                (3, sample_count, 3),
            )
            self.assertEqual(evaluation.tracking_loss.shape, (3,))
            self.assertAlmostEqual(
                evaluation.mean_tracking_loss,
                float(np.mean(evaluation.tracking_loss)),
            )
            self.assertAlmostEqual(
                evaluation.cvar_tracking_loss,
                empirical_upper_cvar(evaluation.tracking_loss, 0.75),
            )
            self.assertAlmostEqual(
                evaluation.failure_probability,
                float(np.mean(evaluation.tracking_loss >= 0.50)),
            )
        # Distinct raw plant masses remain distinct simulations.  Replacing
        # them with one mean vehicle would make these paths identical.
        self.assertGreater(
            np.max(
                np.abs(
                    decision.evaluations[0].trajectories[0].position
                    - decision.evaluations[0].trajectories[2].position
                )
            ),
            1.0e-5,
        )

    def test_member_aligned_correction_paths_and_score_are_reproducible(self):
        desired_position = np.asarray(
            [value.position for value in self.synthetic.references]
        )
        desired_orientation = np.asarray(
            [
                matrix_to_quaternion(euler_xyz_to_matrix(value.rpy))
                for value in self.synthetic.references
            ]
        )
        evaluation = self.decision.evaluations[1]
        expected_translation, expected_rotation = correction_transform_path(
            desired_position,
            desired_orientation,
            evaluation.trajectories[1].position,
            evaluation.trajectories[1].orientation_xyzw,
        )
        np.testing.assert_array_equal(
            evaluation.correction_translation[1], expected_translation
        )
        np.testing.assert_array_equal(
            evaluation.correction_rotation_vector[1], expected_rotation
        )
        expected_score = (
            self.weights.mean_tracking_loss * evaluation.mean_tracking_loss
            + self.weights.cvar_tracking_loss
            * evaluation.cvar_tracking_loss
            + self.weights.failure_probability
            * evaluation.failure_probability
            + evaluation.change_penalty
        )
        expected_change_magnitude = float(
            np.mean(np.log(evaluation.candidate.scales) ** 2)
        )
        self.assertAlmostEqual(
            evaluation.parameter_change_magnitude,
            expected_change_magnitude,
        )
        self.assertAlmostEqual(
            evaluation.change_penalty,
            self.weights.parameter_change * expected_change_magnitude,
        )
        self.assertAlmostEqual(evaluation.decision_score, expected_score)
        scores = [value.decision_score for value in self.decision.evaluations]
        self.assertEqual(
            self.decision.selected_candidate_index, int(np.argmin(scores))
        )

    def test_raw_member_model_error_changes_predictive_trajectory(self):
        nominal = VehicleParameters.nominal()
        residual = self.zero_residual_wrench.copy()
        residual[1, :, 0] = 0.8
        residual[2, :, 3] = 0.04
        source = replace(
            self.predictive_input,
            physical_parameter_members=(nominal, nominal, nominal),
            interval_residual_wrench=residual,
        )
        decision = evaluate_posterior_predictive(
            source,
            candidates=(ControllerParameterCandidate("baseline"),),
            failure_threshold=1.0,
        )
        trajectories = decision.evaluations[0].trajectories
        self.assertGreater(
            np.max(np.abs(trajectories[0].position - trajectories[1].position)),
            1.0e-5,
        )
        self.assertGreater(
            np.max(
                np.abs(
                    trajectories[0].orientation_xyzw
                    - trajectories[2].orientation_xyzw
                )
            ),
            1.0e-6,
        )

    def test_articulated_controller_mass_candidate_changes_the_forecast(self):
        nominal = VehicleParameters.nominal()
        source = replace(
            self.predictive_input,
            physical_parameter_members=(nominal, nominal, nominal),
        )
        decision = evaluate_posterior_predictive(
            source,
            candidates=(
                ControllerParameterCandidate("baseline"),
                ControllerParameterCandidate(
                    "mass_1p10", controller_mass_scale=1.10
                ),
            ),
            failure_threshold=100.0,
        )
        baseline = decision.evaluations[0].trajectories[0]
        changed = decision.evaluations[1].trajectories[0]
        self.assertGreater(
            np.max(
                np.abs(
                    baseline.commanded_thrust - changed.commanded_thrust
                )
            ),
            1.0e-6,
        )
        self.assertGreater(
            np.max(np.abs(baseline.position - changed.position)),
            1.0e-8,
        )

    def test_matching_mass_candidate_is_selected_for_known_mismatch(self):
        nominal = VehicleParameters.nominal()
        heavy = replace(nominal, mass=1.20 * nominal.mass)
        source = replace(
            self.predictive_input,
            physical_parameter_members=(heavy, heavy, heavy),
        )
        decision = evaluate_posterior_predictive(
            source,
            candidates=(
                ControllerParameterCandidate("baseline"),
                ControllerParameterCandidate(
                    "mass_match", controller_mass_scale=1.20
                ),
            ),
            failure_threshold=1000.0,
            weights=PosteriorPredictiveWeights(
                mean_tracking_loss=1.0,
                cvar_tracking_loss=0.0,
                failure_probability=0.0,
                parameter_change=0.0,
            ),
        )
        self.assertEqual(decision.selected_candidate.candidate_id, "mass_match")
        self.assertLess(
            decision.evaluations[1].mean_tracking_loss,
            0.01 * decision.evaluations[0].mean_tracking_loss,
        )

    def test_all_member_coordinates_survive_a_joint_permutation(self):
        base = self.predictive_input
        initial_states = tuple(
            RigidBodyState(
                position=value.position
                + np.asarray((0.01 * member, -0.004 * member, 0.0)),
                orientation_xyzw=value.orientation_xyzw,
                linear_velocity=value.linear_velocity
                + np.asarray((0.0, 0.002 * member, 0.0)),
                angular_velocity=value.angular_velocity,
            )
            for member, value in enumerate(base.initial_states)
        )
        initial_controller_states = tuple(
            ControllerState(
                value.integral_error
                + np.asarray((0.0, 0.0, 0.01 * member, 0.0, 0.0, 0.0)),
                value.roll_pitch_integration_active,
            )
            for member, value in enumerate(base.initial_controller_states)
        )
        initial_actuator_states = tuple(
            ActuatorState(
                np.full(4, 1.5 + 0.01 * member),
                np.full(4, 0.001 * member),
            )
            for member in range(len(self.members))
        )
        residual = base.interval_residual_wrench.copy()
        residual[0, :, 0] = 0.1
        residual[1, :, 2] = -0.2
        residual[2, :, 5] = 0.03
        source = replace(
            base,
            initial_states=initial_states,
            initial_controller_states=initial_controller_states,
            initial_actuator_states=initial_actuator_states,
            interval_residual_wrench=residual,
        )
        candidate = (ControllerParameterCandidate("baseline"),)
        original = evaluate_posterior_predictive(
            source, candidates=candidate, failure_threshold=100.0
        ).evaluations[0]
        permutation = np.asarray((2, 0, 1))
        permuted_source = replace(
            source,
            member_ids=source.member_ids[permutation],
            physical_parameter_members=tuple(
                source.physical_parameter_members[index]
                for index in permutation
            ),
            initial_states=tuple(
                source.initial_states[index] for index in permutation
            ),
            initial_controller_states=tuple(
                source.initial_controller_states[index]
                for index in permutation
            ),
            initial_actuator_states=tuple(
                source.initial_actuator_states[index]
                for index in permutation
            ),
            interval_residual_wrench=(
                source.interval_residual_wrench[permutation]
            ),
        )
        permuted = evaluate_posterior_predictive(
            permuted_source,
            candidates=candidate,
            failure_threshold=100.0,
        ).evaluations[0]
        np.testing.assert_array_equal(
            permuted.tracking_loss, original.tracking_loss[permutation]
        )
        for row, original_row in enumerate(permutation):
            np.testing.assert_array_equal(
                permuted.trajectories[row].position,
                original.trajectories[original_row].position,
            )

    def test_numerical_member_failure_is_retained_and_scored(self):
        residual = self.zero_residual_wrench.copy()
        residual[1, :, :] = 1.0e300
        source = replace(
            self.predictive_input,
            interval_residual_wrench=residual,
        )
        decision = evaluate_posterior_predictive(
            source,
            candidates=(ControllerParameterCandidate("baseline"),),
            failure_threshold=2.0,
        )
        evaluation = decision.evaluations[0]
        np.testing.assert_array_equal(
            evaluation.forecast_success, (True, False, True)
        )
        self.assertIsNone(evaluation.trajectories[1])
        self.assertTrue(np.all(np.isnan(evaluation.correction_translation[1])))
        self.assertTrue(evaluation.forecast_failure_reason[1])
        self.assertEqual(evaluation.tracking_loss[1], 2.0)
        self.assertGreaterEqual(evaluation.failure_probability, 1.0 / 3.0)
        self.assertTrue(np.isinf(evaluation.decision_score))
        self.assertFalse(decision.recommendation_available)
        self.assertEqual(decision.selected_candidate_index, -1)
        with self.assertRaisesRegex(RuntimeError, "no candidate"):
            _selected = decision.selected_candidate
        with tempfile.TemporaryDirectory() as directory:
            destination = save_posterior_predictive_decision(
                str(Path(directory) / "failed_member.npz"), decision
            )
            with np.load(str(destination), allow_pickle=False) as artifact:
                self.assertFalse(artifact["forecast_success"][0, 1])
                self.assertTrue(
                    np.all(np.isnan(artifact["prediction_position"][0, 1]))
                )
                self.assertTrue(artifact["forecast_failure_reason"][0, 1])
                self.assertFalse(artifact["recommendation_available"][0])
                self.assertEqual(artifact["selected_candidate_index"][0], -1)
                self.assertEqual(str(artifact["selected_candidate_id"][0]), "")
                for key in artifact.files:
                    self.assertFalse(artifact[key].dtype.hasobject)

    def test_selected_mode_adapter_preserves_raw_member_order(self):
        controls = np.zeros((len(self.members), CONTROL_DIMENSION))
        for index, parameters in enumerate(self.members):
            controls[index, PARAMETER_OFFSET:] = (
                self.problem.parameter_chart.encode(parameters)
            )
        selected = SimpleNamespace(
            mode=plant_wiring_mode(NOMINAL_MODE_ID),
            problem=self.problem,
            posterior=SimpleNamespace(control_ensemble=controls),
            member_ids=self.member_ids,
        )
        adapted = input_from_mode_posterior(
            selected, provenance={"source_artifact": "mode.npz"}
        )
        np.testing.assert_array_equal(adapted.member_ids, self.member_ids)
        np.testing.assert_allclose(
            [value.mass for value in adapted.physical_parameter_members],
            [value.mass for value in self.members],
            atol=2.0e-15,
        )
        self.assertEqual(adapted.selected_mode_id, NOMINAL_MODE_ID)
        self.assertIn(
            ("source_kind", "selected_mode_posterior"), adapted.provenance
        )
        np.testing.assert_array_equal(
            adapted.interval_residual_wrench, self.zero_residual_wrench
        )

    def test_real_assimilation_adapter_keeps_x0_c0_a0_theta_and_q_rows(self):
        trajectories = self.decision.evaluations[0].trajectories
        chart = VehicleParameterChart(VehicleParameters.nominal())
        coordinates = np.asarray(
            [chart.encode(value) for value in self.members]
        )
        residual = self.zero_residual_wrench.copy()
        residual[0, :, 2] = 0.13
        residual[1, :, 4] = -0.02
        posterior = SimpleNamespace(
            trajectory_ensemble=trajectories,
            parameter_ensemble=SimpleNamespace(coordinates=coordinates),
            residual_wrench_ensemble=residual,
        )
        episode = SimpleNamespace(
            observations=self.synthetic.observations,
            references=self.synthetic.references,
            initial_controller_state=(
                self.predictive_input.initial_controller_states[0]
            ),
            controller_configuration=(
                self.predictive_input.controller_configuration
            ),
            provenance=SimpleNamespace(
                bag_path="flight.bag",
                bag_sha256="abc123",
                time_basis="header",
            ),
        )
        result = SimpleNamespace(
            posterior=posterior,
            episode=episode,
            nominal_parameters=VehicleParameters.nominal(),
            actuator_parameters=ActuatorParameters(),
            mode_diagnostic=SimpleNamespace(
                selected_mode_id=NOMINAL_MODE_ID
            ),
        )
        adapted = input_from_real_assimilation(
            result, provenance={"run_id": "real-test"}
        )
        np.testing.assert_array_equal(adapted.member_ids, np.arange(3))
        np.testing.assert_array_equal(
            adapted.interval_residual_wrench, residual
        )
        np.testing.assert_allclose(
            [value.mass for value in adapted.physical_parameter_members],
            [value.mass for value in self.members],
            atol=2.0e-15,
        )
        for member, trajectory in enumerate(trajectories):
            np.testing.assert_array_equal(
                adapted.initial_states[member].position,
                trajectory.position[0],
            )
            np.testing.assert_array_equal(
                adapted.initial_controller_states[member].integral_error,
                trajectory.controller_integral[0],
            )
            np.testing.assert_array_equal(
                adapted.initial_actuator_states[member].thrust,
                trajectory.actuator_thrust[0],
            )
        self.assertIn(("bag_path", "flight.bag"), adapted.provenance)
        self.assertIn(("run_id", "real-test"), adapted.provenance)

    def test_pickle_free_artifact_retains_threshold_candidates_and_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = save_posterior_predictive_decision(
                str(Path(directory) / "decision.npz"), self.decision
            )
            with np.load(str(destination), allow_pickle=False) as artifact:
                self.assertEqual(
                    str(artifact["schema"][0]),
                    "grape-weak-constraint/phase6-posterior-predictive",
                )
                self.assertEqual(
                    str(artifact["selected_mode_id"][0]), NOMINAL_MODE_ID
                )
                np.testing.assert_array_equal(
                    artifact["source_mode_id"], (NOMINAL_MODE_ID,)
                )
                np.testing.assert_array_equal(
                    artifact["source_mode_weight"], (1.0,)
                )
                self.assertEqual(
                    str(artifact["scenario_assumption"][0]),
                    self.predictive_input.scenario_assumption,
                )
                self.assertTrue(artifact["recommendation_available"][0])
                np.testing.assert_array_equal(
                    artifact["posterior_member_id"], self.member_ids
                )
                np.testing.assert_allclose(
                    artifact["posterior_member_mass"],
                    [value.mass for value in self.members],
                )
                np.testing.assert_array_equal(
                    artifact["posterior_residual_wrench_interval"],
                    self.zero_residual_wrench,
                )
                np.testing.assert_array_equal(
                    artifact["candidate_id"],
                    [value.candidate_id for value in self.candidates],
                )
                np.testing.assert_allclose(
                    artifact["candidate_scale"],
                    [value.scales for value in self.candidates],
                )
                np.testing.assert_allclose(
                    artifact["candidate_controller_mass"],
                    self.predictive_input.controller_parameters.mass
                    * np.asarray((1.0, 1.08)),
                )
                self.assertEqual(artifact["candidate_pid"].shape[:2], (2, 6))
                self.assertEqual(artifact["failure_threshold"][0], 0.50)
                self.assertEqual(
                    dict(
                        zip(
                            artifact["provenance_key"].tolist(),
                            artifact["provenance_value"].tolist(),
                        )
                    ),
                    {
                        "source_artifact": "/tmp/selected_mode.npz",
                        "source_schema": "phase5-test",
                    },
                )
                self.assertEqual(
                    artifact["correction_translation"].shape[:2], (2, 3)
                )
                self.assertEqual(
                    artifact["prediction_position"].shape[:2], (2, 3)
                )
                self.assertTrue(np.all(artifact["forecast_success"]))
                for key in artifact.files:
                    self.assertFalse(artifact[key].dtype.hasobject)

    def test_single_mean_and_invalid_decision_inputs_are_rejected(self):
        source = self.predictive_input
        with self.assertRaisesRegex(ValueError, "at least two"):
            PosteriorPredictiveInput(
                selected_mode_id=source.selected_mode_id,
                member_ids=np.asarray((0,)),
                physical_parameter_members=(self.members[0],),
                times=source.times,
                references=source.references,
                initial_states=(source.initial_states[0],),
                initial_controller_states=(
                    source.initial_controller_states[0],
                ),
                initial_actuator_states=(None,),
                interval_residual_wrench=np.zeros(
                    (1, source.times.size - 1, 6)
                ),
                controller_configuration=source.controller_configuration,
                controller_parameters=source.controller_parameters,
                controller_geometry=source.controller_geometry,
                plant_geometry=source.plant_geometry,
                actuator_parameters=source.actuator_parameters,
            )
        with self.assertRaisesRegex(ValueError, "interval_residual_wrench"):
            replace(source, interval_residual_wrench=None)
        duplicate = (
            ControllerParameterCandidate("same"),
            ControllerParameterCandidate("same", yaw_pid_scale=1.1),
        )
        with self.assertRaisesRegex(ValueError, "unique"):
            evaluate_posterior_predictive(source, candidates=duplicate)
        with self.assertRaisesRegex(ValueError, "failure_threshold"):
            evaluate_posterior_predictive(source, failure_threshold=0.0)
        with self.assertRaisesRegex(ValueError, "cvar_level"):
            evaluate_posterior_predictive(source, cvar_level=1.0)


if __name__ == "__main__":
    unittest.main()
