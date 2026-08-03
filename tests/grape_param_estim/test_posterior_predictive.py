from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from grape_param_estim.artifact_io import (
    begin_bundle,
    load_pid_proposal_evaluation,
    read_manifest,
)
from grape_param_estim.controller import ControllerConfig, initial_controller_state
from grape_param_estim.controller_config import (
    ControllerSnapshotProvenance,
    PID_GROUPS,
    PidGainConfiguration,
)
from grape_param_estim.geometry import (
    euler_xyz_to_matrix,
    matrix_to_quaternion,
)
from grape_param_estim.pid_proposal import (
    derive_pid_proposal_ensemble,
    member_pid_candidate,
    user_pid_candidate,
)
from grape_param_estim.posterior_predictive import (
    COMPONENTWISE_IMPROVEMENT_RULE,
    CounterfactualBagScenario,
    ErrorThresholds,
    PHYSICAL_METRICS,
    PosteriorPredictiveInput,
    bag_equal_aggregate,
    correction_zero_coverage,
    empirical_upper_cvar,
    evaluate_pid_proposals,
    log_gain_change,
    pid_proposal_evaluation_manifest,
    save_pid_proposal_evaluation,
    summarize_member_metrics,
    time_integrated_error_metrics,
)
from grape_param_estim.progress import CancellationToken, ProgressCancelled
from grape_param_estim.synthetic import run_synthetic_experiment
from grape_param_estim.system import (
    ActuatorParameters,
    ActuatorState,
    ClosedLoopTrajectory,
    GrapeGeometry,
    RigidBodyState,
    VehicleParameters,
)


def _pid_values(configuration):
    rows = []
    for axis in (0, 2, 3, 5):
        value = configuration.pid[axis]
        rows.append((value.p_gain, value.i_gain, value.d_gain))
    return np.asarray(rows)


class PosteriorPredictiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.configuration = ControllerConfig.grape()
        cls.current = PidGainConfiguration(
            _pid_values(cls.configuration),
            ControllerSnapshotProvenance(
                bag_id="bag-a",
                topics=tuple(
                    "/recorded/controller/{}/parameter_updates".format(group)
                    for group in PID_GROUPS
                ),
                record_times=np.asarray((10.0, 10.1, 10.2, 10.3)),
                source_kinds=("dynamic_reconfigure",) * 4,
            ),
        )
        cls.nominal = VehicleParameters.nominal()
        cls.geometry = GrapeGeometry.grape()
        cls.member_id = np.asarray((41, 7, 99), dtype=np.int64)
        cls.physical = tuple(
            replace(cls.nominal, mass=cls.nominal.mass * scale)
            for scale in (0.88, 1.0, 1.13)
        )
        cls.constant_delay = np.asarray((0.0, 0.007, 0.024))
        cls.proposals = derive_pid_proposal_ensemble(
            cls.member_id,
            cls.physical,
            cls.constant_delay,
            ("nominal", "nominal", "nominal"),
            cls.nominal,
            cls.geometry,
            cls.current,
        )
        first = run_synthetic_experiment(
            duration=0.24,
            time_step=0.04,
            truth_parameters=cls.nominal,
            truth_actuators=ActuatorParameters(),
            truth_residual_wrench=lambda _time, _state: np.zeros(6),
            translation_noise=0.0,
            rotation_noise=0.0,
            seed=91,
        )
        second = run_synthetic_experiment(
            duration=0.16,
            time_step=0.04,
            truth_parameters=cls.nominal,
            truth_actuators=ActuatorParameters(),
            truth_residual_wrench=lambda _time, _state: np.zeros(6),
            translation_noise=0.0,
            rotation_noise=0.0,
            seed=92,
        )
        cls.scenarios = (
            cls._scenario("bag-a", first, "posterior_replay", 0.03),
            cls._scenario("bag-b", second, "zero", 0.07),
        )
        cls.predictive_input = PosteriorPredictiveInput(
            selected_mode_id="nominal",
            physical_parameter_members=cls.physical,
            proposal_ensemble=cls.proposals,
            bags=cls.scenarios,
            provenance=(("source_run_id", "run-a"),),
        )
        cls.member_candidate = member_pid_candidate(cls.proposals, 41)
        cls.user_candidate = user_pid_candidate(
            "user-soft",
            PidGainConfiguration(cls.current.values * 0.90),
        )

    @classmethod
    def _scenario(cls, bag_id, experiment, residual_policy, residual_value):
        truth = experiment.truth
        state = RigidBodyState(
            truth.position[0],
            truth.orientation_xyzw[0],
            truth.linear_velocity[0],
            truth.angular_velocity[0],
        )
        controller_state = initial_controller_state(
            cls.configuration, trim_hover=True
        )
        actuator_state = ActuatorState(
            truth.actuator_thrust[0], truth.actuator_gimbal_angle[0]
        )
        members = cls.member_id.size
        residual = np.full(
            (members, truth.times.size - 1, 6), residual_value
        )
        return CounterfactualBagScenario(
            bag_id=bag_id,
            times=truth.times,
            references=experiment.references,
            initial_states=tuple(state for _index in range(members)),
            initial_controller_states=tuple(
                controller_state for _index in range(members)
            ),
            initial_actuator_states=tuple(
                actuator_state for _index in range(members)
            ),
            posterior_residual_wrench=residual,
            controller_configuration=cls.configuration,
            controller_nominal_parameters=cls.nominal,
            controller_geometry=cls.geometry,
            plant_geometry=cls.geometry,
            actuator_parameters=ActuatorParameters(
                thrust_time_constant=0.02,
                gimbal_time_constant=0.02,
            ),
            residual_policy=residual_policy,
            provenance=(("bag_id", bag_id),),
        )

    def _fake_trajectory(self, keyword_arguments, error_scale=None):
        references = keyword_arguments["references"]
        times = np.asarray(keyword_arguments["times"])
        controller = keyword_arguments["controller"]
        scale = (
            controller.configuration.pid[0].p_gain
            / self.configuration.pid[0].p_gain
            if error_scale is None
            else float(error_scale)
        )
        reference_position = np.asarray([value.position for value in references])
        position = reference_position.copy()
        position[:, 0] += 0.05 * scale
        orientation = []
        error_rotation = euler_xyz_to_matrix((0.0, 0.0, 0.03 * scale))
        for reference in references:
            orientation.append(
                matrix_to_quaternion(
                    euler_xyz_to_matrix(reference.rpy) @ error_rotation
                )
            )
        count = times.size
        return ClosedLoopTrajectory(
            times=times,
            position=position,
            orientation_xyzw=np.asarray(orientation),
            linear_velocity=np.zeros((count, 3)),
            angular_velocity=np.zeros((count, 3)),
            controller_integral=np.zeros((count, 6)),
            commanded_thrust=np.zeros((count, 4)),
            commanded_gimbal_angle=np.zeros((count, 4)),
            actuator_thrust=np.zeros((count, 4)),
            actuator_gimbal_angle=np.zeros((count, 4)),
            body_wrench=np.zeros((count, 6)),
        )

    def test_current_is_always_included_and_no_representative_is_automatic(self):
        with patch(
            "grape_param_estim.posterior_predictive.simulate_closed_loop",
            side_effect=lambda **kwargs: self._fake_trajectory(kwargs),
        ):
            decision = evaluate_pid_proposals(
                self.predictive_input,
                candidates=(self.member_candidate,),
            )
        self.assertEqual(
            [value.candidate.candidate_id for value in decision.evaluations],
            ["current", "member_41"],
        )
        self.assertFalse(decision.recommendation_available)
        self.assertEqual(decision.recommended_candidate_id, "")
        self.assertIn("no automatic representative", decision.rejection_reason)
        self.assertEqual(decision.improvement_rule, COMPONENTWISE_IMPROVEMENT_RULE)
        self.assertIn(
            "same recorded reference",
            decision.predictive_input.scenario_assumption,
        )
        self.assertIn(
            "bag-b=zero", decision.predictive_input.scenario_assumption
        )
        self.assertIn(
            "not a forecast of a new disturbance realization",
            decision.predictive_input.scenario_assumption,
        )

    def test_multi_bag_raw_member_paths_are_never_averaged(self):
        decision = evaluate_pid_proposals(
            self.predictive_input,
            candidates=(self.member_candidate,),
            cvar_level=0.75,
        )
        self.assertEqual(len(decision.evaluations), 2)
        for evaluation in decision.evaluations:
            self.assertEqual(len(evaluation.bags), 2)
            for scenario in self.scenarios:
                bag = evaluation.bag(scenario.bag_id)
                np.testing.assert_array_equal(bag.member_id, self.member_id)
                self.assertEqual(
                    bag.prediction_position.shape,
                    (3, scenario.times.size, 3),
                )
                self.assertEqual(
                    bag.correction_rotation_vector.shape,
                    (3, scenario.times.size, 3),
                )
                self.assertEqual(bag.position_rmse.shape, (3,))
                self.assertTrue(np.all(bag.forecast_completed))
        current_a = decision.current_evaluation.bag("bag-a")
        self.assertGreater(
            np.max(
                np.abs(
                    current_a.prediction_position[0]
                    - current_a.prediction_position[2]
                )
            ),
            1.0e-7,
        )

    def test_controller_model_limits_member_tau_and_residual_policy_are_fixed(self):
        captured = []

        def fake_simulation(**kwargs):
            captured.append(kwargs)
            return self._fake_trajectory(kwargs)

        with patch(
            "grape_param_estim.posterior_predictive.simulate_closed_loop",
            side_effect=fake_simulation,
        ):
            evaluate_pid_proposals(
                self.predictive_input,
                candidates=(self.user_candidate,),
            )
        # current + user, then bag-a + bag-b, each with all three members.
        self.assertEqual(len(captured), 12)
        for candidate_index in range(2):
            start = candidate_index * 6
            for bag_index, scenario in enumerate(self.scenarios):
                for member_index in range(3):
                    call = captured[start + bag_index * 3 + member_index]
                    self.assertIs(
                        call["controller"].nominal_parameters,
                        scenario.controller_nominal_parameters,
                    )
                    np.testing.assert_array_equal(
                        call["plant"].parameters.inertia,
                        self.physical[member_index].inertia,
                    )
                    self.assertEqual(
                        call["actuator_parameters"].delay,
                        self.constant_delay[member_index],
                    )
                    expected_residual = (
                        scenario.posterior_residual_wrench[member_index]
                        if scenario.residual_policy == "posterior_replay"
                        else np.zeros((scenario.times.size - 1, 6))
                    )
                    np.testing.assert_array_equal(
                        call["interval_residual_wrench"], expected_residual
                    )
                    for before, after in zip(
                        scenario.controller_configuration.pid,
                        call["controller"].configuration.pid,
                    ):
                        self.assertEqual(before.limit_sum, after.limit_sum)
                        self.assertEqual(before.limit_p, after.limit_p)
                        self.assertEqual(before.limit_i, after.limit_i)
                        self.assertEqual(before.limit_d, after.limit_d)
        # Candidate changes do not replace the nominal mass/inertia/geometry.
        self.assertTrue(
            all(
                call["controller"].nominal_parameters is self.nominal
                for call in captured
            )
        )

    def test_time_integrated_metrics_keep_position_and_orientation_separate(self):
        times = np.asarray((0.0, 1.0, 3.0))
        position = np.asarray(((0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (2.0, 0.0, 0.0)))
        orientation = np.asarray(((0.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 3.0, 0.0)))
        metric = time_integrated_error_metrics(times, position, orientation)
        self.assertAlmostEqual(metric.position_rmse, np.sqrt(10.0 / 3.0))
        self.assertAlmostEqual(metric.orientation_rmse, np.sqrt(3.5))
        self.assertEqual(metric.maximum_position_error, 2.0)
        self.assertEqual(metric.maximum_orientation_error, 3.0)
        self.assertFalse(hasattr(metric, "tracking_loss"))

    def test_cvar_is_per_bag_then_equal_weighted_across_bags(self):
        completed = np.ones(4, dtype=bool)

        def summary(offset):
            values = {
                name: np.asarray((offset, offset + 1, offset + 2, offset + 9), dtype=float)
                for name in PHYSICAL_METRICS
            }
            return summarize_member_metrics(values, completed, 0.75)

        first = summary(0.0)
        second = summary(10.0)
        aggregate = bag_equal_aggregate((first, second))
        self.assertAlmostEqual(first.metrics["position_rmse"].upper_cvar, 9.0)
        self.assertAlmostEqual(second.metrics["position_rmse"].upper_cvar, 19.0)
        self.assertAlmostEqual(
            aggregate.metrics["position_rmse"].upper_cvar, 14.0
        )
        self.assertAlmostEqual(
            aggregate.metrics["position_rmse"].mean,
            0.5
            * (
                np.mean((0.0, 1.0, 2.0, 9.0))
                + np.mean((10.0, 11.0, 12.0, 19.0))
            ),
        )
        self.assertAlmostEqual(
            empirical_upper_cvar((0.0, 1.0, 2.0, 9.0), 0.60), 6.375
        )

    def test_correction_coverage_is_a_separate_componentwise_diagnostic(self):
        translation = np.zeros((3, 2, 3))
        translation[0, :, 0] = -1.0
        translation[1, :, 0] = 1.0
        translation[:2, :, 1] = 1.0
        translation[2] = np.nan
        rotation = np.zeros_like(translation)
        rotation[2] = np.nan
        completed = np.asarray((True, True, False))
        translation_coverage, rotation_coverage, combined = (
            correction_zero_coverage(translation, rotation, completed)
        )
        self.assertAlmostEqual(translation_coverage, 2.0 / 3.0)
        self.assertEqual(rotation_coverage, 1.0)
        self.assertAlmostEqual(combined, 5.0 / 6.0)

    def test_absent_threshold_is_not_configured_and_failure_is_separate(self):
        def sometimes_fails(**kwargs):
            if np.isclose(kwargs["plant"].parameters.mass, self.nominal.mass):
                raise FloatingPointError("member diverged")
            return self._fake_trajectory(kwargs, error_scale=1.0)

        thresholds = ErrorThresholds(position=0.01, orientation=None)
        with patch(
            "grape_param_estim.posterior_predictive.simulate_closed_loop",
            side_effect=sometimes_fails,
        ):
            decision = evaluate_pid_proposals(
                self.predictive_input,
                candidates=(),
                thresholds=thresholds,
            )
        bag = decision.current_evaluation.bag("bag-a")
        self.assertEqual(bag.forecast_completed.tolist(), [True, False, True])
        self.assertIn("FloatingPointError", bag.failure_reason[1])
        self.assertTrue(np.isnan(bag.position_rmse[1]))
        self.assertTrue(np.isnan(bag.position_threshold_exceeded[1]))
        self.assertIsNone(bag.orientation_threshold_exceeded)
        self.assertEqual(decision.thresholds.orientation_display(), "Not configured")
        self.assertEqual(bag.summary.numerical_failure_count, 1)
        # The numerical failure is not counted as threshold exceedance.
        self.assertEqual(bag.summary.position_threshold_exceedance, 1.0)

    def test_position_and_orientation_thresholds_remain_independent(self):
        thresholds = ErrorThresholds(position=0.04, orientation=0.04)
        with patch(
            "grape_param_estim.posterior_predictive.simulate_closed_loop",
            side_effect=lambda **kwargs: self._fake_trajectory(kwargs, 1.0),
        ):
            decision = evaluate_pid_proposals(
                self.predictive_input, candidates=(), thresholds=thresholds
            )
        bag = decision.current_evaluation.bag("bag-a")
        self.assertEqual(bag.summary.position_threshold_exceedance, 1.0)
        self.assertEqual(bag.summary.orientation_threshold_exceedance, 0.0)

    def test_pareto_and_explicit_componentwise_recommendation_rule(self):
        better = user_pid_candidate(
            "better", PidGainConfiguration(self.current.values * 0.5)
        )
        worse = user_pid_candidate(
            "worse", PidGainConfiguration(self.current.values * 1.5)
        )

        def deterministic(**kwargs):
            return self._fake_trajectory(kwargs)

        with patch(
            "grape_param_estim.posterior_predictive.simulate_closed_loop",
            side_effect=deterministic,
        ):
            unselected = evaluate_pid_proposals(
                self.predictive_input, candidates=(better, worse)
            )
            selected_better = evaluate_pid_proposals(
                self.predictive_input,
                candidates=(better, worse),
                selected_candidate_id="better",
            )
            selected_worse = evaluate_pid_proposals(
                self.predictive_input,
                candidates=(better, worse),
                selected_candidate_id="worse",
            )
        self.assertTrue(unselected.evaluation("better").improves_current)
        self.assertFalse(unselected.evaluation("better").pareto_dominated)
        self.assertTrue(unselected.evaluation("worse").pareto_dominated)
        self.assertFalse(unselected.recommendation_available)
        self.assertTrue(selected_better.recommendation_available)
        self.assertEqual(selected_better.recommended_candidate_id, "better")
        self.assertFalse(selected_worse.recommendation_available)
        self.assertIn("Pareto dominated", selected_worse.rejection_reason)

    def test_member_candidate_must_match_source_member_and_mode(self):
        fake = replace(
            self.member_candidate,
            configuration=PidGainConfiguration(
                self.member_candidate.configuration.values * 1.01
            ),
        )
        with self.assertRaisesRegex(ValueError, "must match"):
            evaluate_pid_proposals(self.predictive_input, candidates=(fake,))
        mixed = replace(
            self.proposals,
            source_mode_id=("nominal", "other", "nominal"),
        )
        with self.assertRaisesRegex(ValueError, "cannot average across modes"):
            PosteriorPredictiveInput(
                selected_mode_id="nominal",
                physical_parameter_members=self.physical,
                proposal_ensemble=mixed,
                bags=self.scenarios,
            )

    def test_progress_is_emitted_at_every_candidate_bag_member_boundary(self):
        events = []
        with patch(
            "grape_param_estim.posterior_predictive.simulate_closed_loop",
            side_effect=lambda **kwargs: self._fake_trajectory(kwargs),
        ):
            evaluate_pid_proposals(
                self.predictive_input,
                candidates=(self.user_candidate,),
                progress_callback=events.append,
                progress_run_id="evaluation-a",
            )
        self.assertEqual(len(events), 2 * 2 * 3)
        self.assertEqual(events[-1].fraction, 1.0)
        self.assertEqual(events[-1].completed_units, 12)
        self.assertEqual(
            {value.bag_id for value in events}, {"bag-a", "bag-b"}
        )
        self.assertEqual(
            {value.member_id for value in events}, {41, 7, 99}
        )
        self.assertEqual(
            {value.stage_id for value in events},
            {"candidate_member_bag_forecast"},
        )

    def test_pre_cancelled_evaluation_stops_before_the_first_forecast(self):
        token = CancellationToken()
        token.cancel("user_requested")
        with patch(
            "grape_param_estim.posterior_predictive.simulate_closed_loop"
        ) as simulation:
            with self.assertRaises(ProgressCancelled):
                evaluate_pid_proposals(
                    self.predictive_input,
                    candidates=(),
                    cancellation_token=token,
                )
        simulation.assert_not_called()

    def test_artifact_is_pickle_free_aligned_and_contains_no_weighted_score(self):
        with patch(
            "grape_param_estim.posterior_predictive.simulate_closed_loop",
            side_effect=lambda **kwargs: self._fake_trajectory(kwargs),
        ):
            decision = evaluate_pid_proposals(
                self.predictive_input,
                candidates=(self.user_candidate,),
                thresholds=ErrorThresholds(position=0.10),
                selected_candidate_id="user-soft",
            )
        with tempfile.TemporaryDirectory() as directory:
            root = save_pid_proposal_evaluation(
                str(Path(directory) / "pid_proposal_evaluation"),
                decision,
                evaluation_id="evaluation-a",
                source_run_id="run-a",
                created_at="2026-08-04T12:00:00+09:00",
            )
            bundle = load_pid_proposal_evaluation(root)
            np.testing.assert_array_equal(
                bundle.proposal_ensemble["source_member_id"], self.member_id
            )
            self.assertEqual(
                bundle.bags["bag-a"]["prediction_position"].shape,
                (2, 3, self.scenarios[0].times.size, 3),
            )
            self.assertEqual(
                bundle.summary["member_bag_position_rmse"].shape,
                (2, 2, 3),
            )
            self.assertIn("per_bag_position_rmse_upper_cvar", bundle.summary)
            self.assertIn("pareto_non_dominated", bundle.summary)
            self.assertIn(
                "same posterior member initial state",
                str(bundle.summary["scenario_assumption"][0]),
            )
            self.assertEqual(
                bundle.summary["current_pid_baseline_bag_id"].tolist(),
                ["bag-a"],
            )
            np.testing.assert_array_equal(
                bundle.summary["current_pid_snapshot_record_time"],
                self.current.provenance.record_times,
            )
            np.testing.assert_array_equal(
                bundle.summary["current_pid_snapshot_topic"],
                np.asarray(self.current.provenance.topics),
            )
            self.assertEqual(
                bundle.summary[
                    "per_bag_correction_transform_zero_coverage"
                ].shape,
                (2, 2),
            )
            np.testing.assert_allclose(
                bundle.summary[
                    "per_bag_correction_transform_zero_coverage"
                ][:, 0],
                bundle.bags["bag-a"][
                    "correction_transform_zero_coverage"
                ],
            )
            self.assertNotIn("decision_score", bundle.summary)
            self.assertNotIn("tracking_loss", bundle.summary)
            self.assertFalse(
                any(
                    "controller_mass" in key
                    for arrays in (
                        bundle.proposal_ensemble,
                        bundle.summary,
                        bundle.bags["bag-a"],
                    )
                    for key in arrays
                )
            )
            for arrays in (
                bundle.proposal_ensemble,
                bundle.summary,
                bundle.bags["bag-a"],
                bundle.bags["bag-b"],
            ):
                for value in arrays.values():
                    self.assertFalse(value.dtype.hasobject)
            yaml_text = bundle.proposed_yaml_path.read_text(encoding="utf-8")
            self.assertIn("xy:", yaml_text)
            self.assertIn("roll_pitch:", yaml_text)

    def test_prestarted_bundle_and_artifact_cancellation_are_authoritative(self):
        with patch(
            "grape_param_estim.posterior_predictive.simulate_closed_loop",
            side_effect=lambda **kwargs: self._fake_trajectory(kwargs),
        ):
            decision = evaluate_pid_proposals(
                self.predictive_input, candidates=(self.user_candidate,)
            )
        with tempfile.TemporaryDirectory() as directory:
            prestarted = Path(directory) / "prestarted"
            manifest = pid_proposal_evaluation_manifest(
                self.predictive_input,
                "evaluation-prestarted",
                "run-a",
                "2026-08-04T12:00:00+09:00",
            )
            begin_bundle(prestarted, manifest)
            save_pid_proposal_evaluation(
                str(prestarted),
                decision,
                evaluation_id="evaluation-prestarted",
                source_run_id="run-a",
                bundle_started=True,
            )
            self.assertEqual(read_manifest(prestarted)["status"], "complete")

            cancelled = Path(directory) / "cancelled"
            token = CancellationToken()
            token.cancel("signal_SIGTERM")
            with self.assertRaises(ProgressCancelled):
                save_pid_proposal_evaluation(
                    str(cancelled),
                    decision,
                    evaluation_id="evaluation-cancelled",
                    source_run_id="run-a",
                    cancellation_token=token,
                )
            cancelled_manifest = read_manifest(cancelled)
            self.assertEqual(cancelled_manifest["status"], "cancelled")
            self.assertEqual(
                cancelled_manifest["cancellation_reason"], "signal_SIGTERM"
            )

    def test_log_gain_change_has_no_hidden_zero_gain_regularizer(self):
        self.assertEqual(log_gain_change(self.current, self.current), 0.0)
        values = self.current.values.copy()
        zero = np.argwhere(values == 0.0)
        if zero.size:
            row, column = zero[0]
            values[row, column] = 0.1
            self.assertEqual(
                log_gain_change(self.current, PidGainConfiguration(values)),
                float("inf"),
            )


if __name__ == "__main__":
    unittest.main()
