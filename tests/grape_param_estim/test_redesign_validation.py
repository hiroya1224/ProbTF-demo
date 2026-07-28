import unittest
import hashlib
import json
import tempfile
import types
from dataclasses import replace
from pathlib import Path
from unittest import mock

import numpy as np

from grape_param_estim.controller import (
    evaluate_exact_closed_loop_gate,
)
from grape_param_estim.controller.external_oracle import (
    controller_backend_identity,
)
from grape_param_estim.validation.controller_design import (
    BoundParticleEvaluator,
    ControllerEvaluatorIdentity,
    ControllerRecommendationBinding,
    VerifiedPlantArtifactIdentity,
    measure_particle_evaluator_sha256,
)
from grape_param_estim.counterfactual import (
    SUPPORTED,
    UNSUPPORTED,
    SupportDiagnostics,
    SupportEvidenceIdentity,
    SupportReference,
    classify_support,
)
from grape_param_estim.inference.posterior import PlantPosterior
from grape_param_estim.output import (
    CONTROLLER_EVALUATION_ARTIFACTS,
    ControllerEvaluationArtifactWriter,
    ControllerEvaluationProvenance,
    PlantAssimilationArtifactWriter,
    PlantRunProvenance,
    evaluate_and_write_controller_candidate,
)
from grape_param_estim.plant.parameters import (
    ACTUATOR_PARAMETER_NAMES,
    PlantHypothesis,
)
from grape_param_estim.plant.actuator import (
    ActuatorCalibrationIdentity,
)
from grape_param_estim.validation import (
    CompositeFailureDetector,
    ControllerCandidate,
    ControllerParticleOutcome,
    ControllerRecommendationEvidence,
    ControllerRecommendationGates,
    FailureEvent,
    FirstMaskFailureDetector,
    SuccessGateConfig,
    ThresholdFailureDetector,
    ValidationDatasetIdentity,
    assert_success_episodes_validation_only,
    censor_after_failure,
    evaluate_controller_candidate,
    evaluate_success_episode,
    evaluate_success_gate,
    trajectory_envelope,
    validate_held_out_failure,
    validate_posterior_predictive,
    validate_trajectory_envelope,
)
from test_closed_loop_assimilation import (
    _factual_evidence,
    _oracle_identity,
)
from test_counterfactual import probability_calibration_report


def _controller_tube_evidence(
    success,
    maximum_position_ratio=0.0,
):
    return {
        "success": bool(success),
        "violations": (),
        "diagnostic_exceedances": (),
        "outside_duration_s": 0.0,
        "maximum_continuous_saturation_s": 0.0,
        "maximum_position_ratio": float(maximum_position_ratio),
        "maximum_velocity_ratio": 0.0,
    }


class FailureEventValidationTests(unittest.TestCase):
    def test_earliest_failure_is_typed_and_post_failure_samples_are_censored(self):
        timestamps = np.arange(5.0)
        threshold = ThresholdFailureDetector(
            "excessive_tilt", "tilt", upper=1.0
        )
        mask = FirstMaskFailureDetector(
            "emergency_stop", mask_key="stop"
        )
        event = CompositeFailureDetector((threshold, mask)).detect(
            timestamps,
            tilt=np.array([0.0, 0.2, 0.4, 1.2, 1.5]),
            stop=np.array([False, False, True, False, False]),
        )
        self.assertEqual(event.failure_type, "emergency_stop")
        self.assertEqual(event.failure_time, 2.0)

        censored = censor_after_failure(
            timestamps, event, include_failure_sample=True
        )
        np.testing.assert_array_equal(
            censored.score_mask,
            np.array([True, True, True, False, False]),
        )
        self.assertEqual(censored.censored_count, 2)

    def test_no_failure_preserves_an_existing_score_mask(self):
        result = censor_after_failure(
            [0.0, 1.0, 2.0], None, [True, False, True]
        )
        np.testing.assert_array_equal(
            result.score_mask, [True, False, True]
        )


class PosteriorPredictiveValidationTests(unittest.TestCase):
    def test_weighted_trajectory_envelope_and_heldout_failure_validate(self):
        timestamps = np.arange(5.0)
        observed = np.zeros((5, 2))
        predictive = np.stack(
            (
                np.full((5, 2), -0.2),
                np.full((5, 2), -0.1),
                np.full((5, 2), 0.1),
                np.full((5, 2), 0.2),
            )
        )
        weights = np.array([0.2, 0.3, 0.3, 0.2])
        envelope = trajectory_envelope(
            timestamps, predictive, weights, credible_probability=0.95
        )
        coverage = validate_trajectory_envelope(
            observed, envelope, minimum_coverage_fraction=0.95
        )
        self.assertTrue(coverage.passed)
        self.assertEqual(coverage.element_coverage_fraction, 1.0)

        observed_failure = FailureEvent(
            "excessive_tilt", 3.0, "bag_event/v1"
        )
        predicted = (
            FailureEvent("excessive_tilt", 3.0, "rollout/v1"),
            FailureEvent("excessive_tilt", 3.0, "rollout/v1"),
            FailureEvent("ground_contact", 2.5, "rollout/v1"),
            None,
        )
        failure = validate_held_out_failure(
            observed_failure,
            predicted,
            weights,
            minimum_failure_probability=0.8,
            minimum_matching_type_probability=0.5,
        )
        self.assertTrue(failure.passed)
        self.assertAlmostEqual(failure.predicted_failure_probability, 0.8)
        self.assertAlmostEqual(failure.matching_type_probability, 0.5)
        self.assertTrue(failure.observed_time_covered)

    def test_failure_trajectory_is_censored_and_both_components_must_pass(self):
        timestamps = np.arange(5.0)
        observed = np.zeros((5, 1))
        predictive = np.asarray(
            [
                [[-0.1], [-0.1], [-0.1], [100.0], [100.0]],
                [[0.1], [0.1], [0.1], [100.0], [100.0]],
            ]
        )
        observed_failure = FailureEvent(
            "ground_contact", 2.0, "bag_event/v1"
        )
        matching_events = (
            FailureEvent("ground_contact", 2.0, "rollout/v1"),
            FailureEvent("ground_contact", 2.0, "rollout/v1"),
        )
        passed = validate_posterior_predictive(
            timestamps=timestamps,
            observed_trajectory=observed,
            predictive_trajectories=predictive,
            observed_failure=observed_failure,
            predicted_failures=matching_events,
            minimum_coverage_fraction=1.0,
        )
        self.assertTrue(passed.trajectory.passed)
        self.assertTrue(passed.failure.passed)
        self.assertTrue(passed.passed)
        self.assertEqual(passed.trajectory.evaluated_time_count, 3)

        trajectory_miss = np.array(predictive, copy=True)
        trajectory_miss[:, 1, 0] = 5.0
        failed_trajectory = validate_posterior_predictive(
            timestamps=timestamps,
            observed_trajectory=observed,
            predictive_trajectories=trajectory_miss,
            observed_failure=observed_failure,
            predicted_failures=matching_events,
            minimum_coverage_fraction=1.0,
        )
        self.assertFalse(failed_trajectory.trajectory.passed)
        self.assertTrue(failed_trajectory.failure.passed)
        self.assertFalse(failed_trajectory.passed)

        failed_event = validate_posterior_predictive(
            timestamps=timestamps,
            observed_trajectory=observed,
            predictive_trajectories=predictive,
            observed_failure=observed_failure,
            predicted_failures=(None, None),
            minimum_coverage_fraction=1.0,
        )
        self.assertTrue(failed_event.trajectory.passed)
        self.assertFalse(failed_event.failure.passed)
        self.assertFalse(failed_event.passed)


class SuccessGateTests(unittest.TestCase):
    def predictive_inputs(self):
        timestamps = np.arange(4.0)
        observed = np.zeros((4, 1))
        predictive = np.stack(
            (
                np.full((4, 1), -0.1),
                np.zeros((4, 1)),
                np.full((4, 1), 0.1),
            )
        )
        return timestamps, observed, predictive

    def test_success_data_is_validation_only_and_checks_false_failures(self):
        timestamps, observed, predictive = self.predictive_inputs()
        passed = evaluate_success_episode(
            episode_id="hover-07",
            role="validation_success",
            timestamps=timestamps,
            observed_trajectory=observed,
            predictive_trajectories=predictive,
            predicted_failures=(None, None, None),
        )
        self.assertTrue(passed.passed)
        self.assertEqual(passed.credible_probability, 0.95)
        self.assertTrue(evaluate_success_gate((passed,)).passed)

        failed = evaluate_success_episode(
            episode_id="hover-used-for-inference",
            role="inference_success",
            timestamps=timestamps,
            observed_trajectory=observed,
            predictive_trajectories=predictive,
            predicted_failures=(
                FailureEvent("false_tilt", 2.0, "rollout/v1"),
                None,
                None,
            ),
            config=SuccessGateConfig(
                maximum_false_failure_probability=0.05
            ),
        )
        self.assertFalse(failed.passed)
        self.assertIn(
            "success_episode_not_validation_only", failed.reasons
        )
        self.assertIn("false_failure_probability", failed.reasons)

        with self.assertRaisesRegex(ValueError, "validation_success"):
            assert_success_episodes_validation_only(
                (
                    {
                        "episode_id": "hover-07",
                        "outcome": {"value": "success"},
                        "role": "inference_failure",
                    },
                )
            )


class ControllerDesignTests(unittest.TestCase):
    _reports = None
    _plant_artifact_temporary_directory = None
    _plant_artifact_identity = None

    @staticmethod
    def posterior():
        values = (0.8, 1.2, 2.0)
        particles = tuple(
            PlantHypothesis(
                model_id="controller-design-test/v1",
                plant_parameters=np.asarray([value]),
                actuator_parameters=np.empty(0),
                disturbance_parameters=np.empty(0),
                plant_parameter_names=("test_plant_value",),
                actuator_parameter_names=(),
            )
            for value in values
        )
        return PlantPosterior.from_arrays(
            particles=particles,
            weights=np.asarray([0.2, 0.3, 0.5]),
            log_likelihood=np.zeros(3),
            model_id="controller-design-test/v1",
            prior_id="test-prior/v1",
            likelihood_id="test-likelihood/v1",
            controller_snapshot_id="c" * 64,
        )

    @classmethod
    def plant_artifact_identity(cls, posterior):
        cached = cls._plant_artifact_identity
        if (
            cached is not None
            and cached.posterior_content_sha256
            == posterior.content_sha256
        ):
            return cached
        if cls._plant_artifact_temporary_directory is not None:
            cls._plant_artifact_temporary_directory.cleanup()
        temporary = tempfile.TemporaryDirectory()
        provenance = PlantRunProvenance(
            source_commit="test-source-commit",
            source_bag_sha256=("4" * 64,),
            normalized_episode_sha256=("5" * 64,),
            controller_snapshot_sha256=(
                posterior.controller_snapshot_id
            ),
            controller_artifact_sha256="6" * 64,
            plant_backend_id="test-plant-backend/v1",
            plant_backend_sha256="7" * 64,
            plant_geometry_profile_id="test-geometry/v1",
            plant_geometry_sha256="8" * 64,
            prior_id=posterior.prior_id,
            likelihood_id=posterior.likelihood_id,
            seed=7,
            config_sha256="9" * 64,
            fixture_sha256="0" * 64,
        )
        destination = PlantAssimilationArtifactWriter(
            temporary.name
        ).write(
            run_id="phase8-source-plant",
            posterior=posterior,
            provenance=provenance,
            controller_snapshot={
                "snapshot_id": posterior.controller_snapshot_id
            },
            controller_replay_audit={"passed": True},
            factual_replay_report={"passed": True},
            identifiability_report={"passed": True},
            likelihood_components=(),
            posterior_predictive={},
            failure_validation={"passed": True},
            success_validation={"passed": True},
            interpretation="effective_plant_posterior",
        )
        cls._plant_artifact_temporary_directory = temporary
        cls._plant_artifact_identity = VerifiedPlantArtifactIdentity(
            destination
        )
        return cls._plant_artifact_identity

    @classmethod
    def tearDownClass(cls):
        if cls._plant_artifact_temporary_directory is not None:
            cls._plant_artifact_temporary_directory.cleanup()
        cls._plant_artifact_temporary_directory = None
        cls._plant_artifact_identity = None
        super().tearDownClass()

    @classmethod
    def typed_reports(cls):
        if cls._reports is not None:
            return cls._reports
        oracle_identity = _oracle_identity()
        conformance = _factual_evidence(oracle_identity)
        exactness = evaluate_exact_closed_loop_gate(
            controller_backend_identity(oracle_identity),
            conformance,
        )
        probability, _, _ = probability_calibration_report(
            oracle_identity, conformance
        )
        support_reference = SupportReference(
            observed_candidate_vectors=np.zeros((10, 1)),
            observed_state_action_points=np.zeros((10, 1)),
            candidate_scale=np.ones(1),
            state_action_scale=np.ones(1),
            minimum_importance_ess=2.0,
            maximum_predictive_std=1.0,
        )
        support = classify_support(
            candidate_vector=np.zeros(1),
            rollout_state_action_points=np.zeros((3, 1)),
            predictive_std=0.1,
            reference=support_reference,
        )
        actuator = ActuatorCalibrationIdentity(
            artifact_sha256="a" * 64,
            actuator_model_id="test-actuator/v1",
        )
        timestamps = np.asarray([0.0, 1.0, 2.0])
        observed = np.zeros((3, 1))
        predictive = np.zeros((2, 3, 1))
        observed_failure = FailureEvent(
            "ground_contact", 2.0, "held-out/v1"
        )
        failure = validate_posterior_predictive(
            timestamps=timestamps,
            observed_trajectory=observed,
            predictive_trajectories=predictive,
            observed_failure=observed_failure,
            predicted_failures=(
                FailureEvent(
                    "ground_contact", 2.0, "rollout/v1"
                ),
                FailureEvent(
                    "ground_contact", 2.0, "rollout/v1"
                ),
            ),
            minimum_coverage_fraction=1.0,
            dataset_provenance_sha256="2" * 64,
        )
        success_episode = evaluate_success_episode(
            episode_id="success-1",
            role="validation_success",
            timestamps=timestamps,
            observed_trajectory=observed,
            predictive_trajectories=predictive,
            predicted_failures=(None, None),
            dataset_provenance_sha256="3" * 64,
        )
        success = evaluate_success_gate((success_episode,))
        cls._reports = {
            "exactness_report": exactness,
            "actuator_calibration_report": actuator,
            "support_report": support,
            "probability_calibration_report": probability,
            "failure_validation_report": failure,
            "success_validation_report": success,
        }
        return cls._reports

    @classmethod
    def evidence(cls, **changes):
        values = {
            **cls.typed_reports(),
        }
        values.update(changes)
        return ControllerRecommendationEvidence(**values)

    @classmethod
    def gates(cls, **changes):
        return ControllerRecommendationGates.from_evidence(
            cls.evidence(**changes)
        )

    @classmethod
    def evaluator_identity(cls, evaluator, **changes):
        reports = cls.typed_reports()
        values = {
            "evaluator_id": "test-closed-loop-evaluator/v1",
            "controller_backend_identity": (
                reports["exactness_report"].identity
            ),
            "evaluator_artifact_sha256": (
                measure_particle_evaluator_sha256(evaluator)
            ),
            "evaluation_config_sha256": "f" * 64,
            "actuator_model_id": (
                reports[
                    "actuator_calibration_report"
                ].actuator_model_id
            ),
            "actuator_backend_sha256": "b" * 64,
            "actuator_calibration_sha256": (
                reports[
                    "actuator_calibration_report"
                ].artifact_sha256
            ),
        }
        values.update(changes)
        return ControllerEvaluatorIdentity(**values)

    @classmethod
    def bound_evaluation(
        cls,
        candidate,
        posterior,
        evaluator,
        *,
        evaluator_identity=None,
        report_changes=None,
    ):
        reports = dict(cls.typed_reports())
        reports.update({} if report_changes is None else report_changes)
        identity = (
            cls.evaluator_identity(evaluator)
            if evaluator_identity is None
            else evaluator_identity
        )
        binding = ControllerRecommendationBinding.create(
            candidate=candidate,
            plant_posterior=posterior,
            plant_artifact_identity=(
                cls.plant_artifact_identity(posterior)
            ),
            evaluator_identity=identity,
            **reports
        )
        evidence = ControllerRecommendationEvidence(
            binding=binding, **reports
        )
        return (
            evidence,
            ControllerRecommendationGates.from_evidence(evidence),
            BoundParticleEvaluator(identity, binding, evaluator),
        )

    def test_controller_candidate_is_separate_and_all_gates_are_required(self):
        candidate = ControllerCandidate(
            "pid-a",
            {
                "p_gain": np.array([10.0, 10.0]),
                "d_gain": np.array([5.0, 5.0]),
            },
        )
        posterior = self.posterior()

        def evaluator(_candidate, plant):
            value = float(plant.plant_parameters[0])
            return {
                "success": value < 1.5,
                "failure": value >= 1.5,
                "saturated": value > 1.0,
                "trajectory": np.full((2, 2), value),
                "tube": _controller_tube_evidence(
                    value < 1.5,
                    maximum_position_ratio=value,
                ),
            }

        evidence, gates, bound_evaluator = self.bound_evaluation(
            candidate, posterior, evaluator
        )
        result = evaluate_controller_candidate(
            candidate,
            posterior,
            bound_evaluator,
            gates,
            recommendation_threshold=0.5,
        )
        self.assertAlmostEqual(result.success_probability, 0.5)
        self.assertAlmostEqual(result.failure_probability, 0.5)
        self.assertAlmostEqual(result.saturation_probability, 0.8)
        self.assertTrue(result.recommendation_allowed)
        self.assertTrue(result.output_evidence_gate_passed)
        self.assertTrue(result.phase8_gates_passed)
        self.assertEqual(result.trajectory_particle_count, 3)
        self.assertEqual(result.trajectory_tube_particle_count, 3)
        self.assertEqual(result.saturation_measurement_count, 3)
        self.assertEqual(len(result.particle_outcomes), 3)
        self.assertAlmostEqual(result.particle_outcomes[1].weight, 0.3)
        self.assertEqual(
            result.particle_outcomes[1].outcome.tube["success"], True
        )
        np.testing.assert_allclose(
            result.particle_outcomes[2].outcome.trajectory,
            np.full((2, 2), 2.0),
        )
        for particle_index, weighted in enumerate(
            result.particle_outcomes
        ):
            output_evidence = weighted.output_evidence
            self.assertIsNotNone(output_evidence)
            self.assertTrue(output_evidence.content_is_valid())
            self.assertEqual(
                output_evidence.evaluation_context_sha256,
                result.gates.evaluation_context_sha256,
            )
            self.assertEqual(
                output_evidence.candidate_sha256,
                candidate.content_sha256,
            )
            self.assertEqual(
                output_evidence.plant_posterior_sha256,
                posterior.content_sha256,
            )
            self.assertEqual(
                output_evidence.particle_index,
                particle_index,
            )
            self.assertEqual(
                output_evidence.plant_particle_sha256,
                result.plant_particle_sha256s[particle_index],
            )

        tampered_output_evidence = replace(
            result.particle_outcomes[0].output_evidence,
            candidate_sha256="0" * 64,
        )
        tampered_particle = replace(
            result.particle_outcomes[0],
            output_evidence=tampered_output_evidence,
        )
        with self.assertRaisesRegex(
            ValueError, "recommendation gate is inconsistent"
        ):
            replace(
                result,
                particle_outcomes=(
                    tampered_particle,
                    *result.particle_outcomes[1:],
                ),
            )
        changed_plant_particle = replace(
            result.plant_particles[0],
            plant_parameters=np.asarray([9.0]),
        )
        with self.assertRaisesRegex(
            ValueError, "recommendation gate is inconsistent"
        ):
            replace(
                result,
                plant_particles=(
                    changed_plant_particle,
                    *result.plant_particles[1:],
                ),
            )
        tampered_particle_evidence = replace(
            result.particle_outcomes[0].output_evidence,
            plant_particle_sha256="0" * 64,
        )
        tampered_particle = replace(
            result.particle_outcomes[0],
            output_evidence=tampered_particle_evidence,
        )
        with self.assertRaisesRegex(
            ValueError, "recommendation gate is inconsistent"
        ):
            replace(
                result,
                particle_outcomes=(
                    tampered_particle,
                    *result.particle_outcomes[1:],
                ),
            )

        blocked_evidence = ControllerRecommendationEvidence(
            **dict(
                self.typed_reports(),
                failure_validation_report={
                    "schema": "grape_failure_validation/v1",
                    "passed": False,
                    "held_out": [
                        {"episode_id": "failure-1", "passed": False}
                    ],
                },
            ),
            binding=evidence.binding,
        )
        blocked = evaluate_controller_candidate(
            candidate,
            posterior,
            bound_evaluator,
            ControllerRecommendationGates.from_evidence(
                blocked_evidence
            ),
            recommendation_threshold=0.5,
        )
        self.assertFalse(blocked.recommendation_allowed)
        self.assertEqual(blocked.reasons, ("failure_validation",))

        with self.assertRaisesRegex(ValueError, "plant/actuator"):
            ControllerCandidate("mixed", {"mass": 2.0, "p_gain": [1.0]})

    def test_final_evaluation_rejects_legacy_or_incomplete_outputs(self):
        candidate = ControllerCandidate(
            "strict-output-pid", {"p_gain": [1.0]}
        )
        posterior = self.posterior()
        legacy = ControllerParticleOutcome(
            success=True,
            failure=False,
        )
        self.assertFalse(legacy.promotion_evidence_complete)
        self.assertEqual(
            legacy.missing_promotion_evidence,
            (
                "trajectory",
                "trajectory_tube",
                "saturation_measurement",
            ),
        )
        complete_tube = _controller_tube_evidence(True)
        for invalid_trajectory in (True, object()):
            with self.subTest(
                invalid_trajectory=type(invalid_trajectory).__name__
            ):
                outcome = ControllerParticleOutcome(
                    success=True,
                    failure=False,
                    saturated=False,
                    trajectory=invalid_trajectory,
                    tube=complete_tube,
                )
                self.assertFalse(outcome.trajectory_evidence_present)
        for invalid_tube in (True, object()):
            with self.subTest(invalid_tube=type(invalid_tube).__name__):
                outcome = ControllerParticleOutcome(
                    success=True,
                    failure=False,
                    saturated=False,
                    trajectory=((0.0,),),
                    tube=invalid_tube,
                )
                self.assertFalse(
                    outcome.trajectory_tube_evidence_present
                )
        with self.assertRaisesRegex(ValueError, "finite"):
            ControllerParticleOutcome(
                success=True,
                failure=False,
                saturated=False,
                trajectory=np.asarray([[np.nan]]),
                tube=complete_tube,
            )
        nonfinite_tube = dict(complete_tube)
        nonfinite_tube["maximum_position_ratio"] = float("inf")
        with self.assertRaisesRegex(ValueError, "finite"):
            ControllerParticleOutcome(
                success=True,
                failure=False,
                saturated=False,
                trajectory=((0.0,),),
                tube=nonfinite_tube,
            )

        cases = (
            (
                "bool-only",
                lambda _candidate, _plant: True,
                "trajectory, trajectory_tube, saturation_measurement",
            ),
            (
                "missing-saturation",
                lambda _candidate, _plant: {
                    "success": True,
                    "failure": False,
                    "trajectory": ((0.0,), (0.1,)),
                    "tube": _controller_tube_evidence(True),
                },
                "saturation_measurement",
            ),
            (
                "missing-trajectory",
                lambda _candidate, _plant: {
                    "success": True,
                    "failure": False,
                    "saturated": False,
                    "tube": _controller_tube_evidence(True),
                },
                "trajectory",
            ),
            (
                "empty-trajectory-tube",
                lambda _candidate, _plant: {
                    "success": True,
                    "failure": False,
                    "saturated": False,
                    "trajectory": ((0.0,),),
                    "tube": {},
                },
                "trajectory_tube",
            ),
        )
        for name, evaluator, missing in cases:
            with self.subTest(name=name):
                _, gates, bound = self.bound_evaluation(
                    candidate,
                    posterior,
                    evaluator,
                )
                with self.assertRaisesRegex(
                    ValueError,
                    "lacks required evidence: {}".format(missing),
                ):
                    evaluate_controller_candidate(
                        candidate,
                        posterior,
                        bound,
                        gates,
                        recommendation_threshold=0.5,
                    )

    def test_candidate_rejects_every_actuator_name_and_physical_alias(self):
        names = tuple(ACTUATOR_PARAMETER_NAMES) + (
            "mass",
            "mass_kg",
            "cog",
            "inertia_xx",
            "controller_mass",
            "controller_inertia_diagonal",
            "thrust_scale",
        )
        for name in names:
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, "plant/actuator"):
                    ControllerCandidate(
                        "mixed-{}".format(name),
                        {"p_gain": [1.0], name: 1.0},
                    )

    def test_each_evidence_gate_is_required_and_raw_booleans_fail_closed(self):
        failing_reports = {
            "exactness": (
                "exactness_report",
                {
                    "schema": "grape_exact_closed_loop_gate/v1",
                    "passed": False,
                    "status": "REJECTED",
                    "factual_replay_passed": False,
                    "closed_loop_exact_allowed": False,
                },
            ),
            "actuator_calibration": (
                "actuator_calibration_report",
                {},
            ),
            "support": (
                "support_report",
                {
                    "schema": "grape_controller_support/v1",
                    "passed": False,
                    "label": "UNSUPPORTED",
                },
            ),
            "probability_calibration": (
                "probability_calibration_report",
                {
                    "schema": "grape_probability_calibration/v1",
                    "passed": False,
                    "content_sha256": "b" * 64,
                },
            ),
            "failure_validation": (
                "failure_validation_report",
                {
                    "schema": "grape_failure_validation/v1",
                    "passed": False,
                    "held_out": [{"passed": False}],
                },
            ),
            "success": (
                "success_validation_report",
                {
                    "schema": "grape_success_validation/v1",
                    "passed": False,
                    "episodes": [{"passed": False}],
                },
            ),
        }
        self.assertFalse(self.gates().passed)
        self.assertIn("evaluation_binding", self.gates().failed_gates)
        for expected, (field, report) in failing_reports.items():
            with self.subTest(gate=expected):
                gates = self.gates(**{field: report})
                self.assertFalse(gates.passed)
                self.assertIn(expected, gates.failed_gates)

        forged_passing_reports = {
            "exactness": (
                "exactness_report",
                {
                    "schema": "grape_exact_closed_loop_gate/v2",
                    "passed": True,
                    "status": "PASS",
                    "factual_replay_passed": True,
                    "factual_evidence_sha256": "a" * 64,
                },
            ),
            "actuator_calibration": (
                "actuator_calibration_report",
                {
                    "schema": "grape_actuator_calibration/v1",
                    "artifact_sha256": "a" * 64,
                    "actuator_model_id": "forged/v1",
                },
            ),
            "support": (
                "support_report",
                {"label": "SUPPORTED", "passed": True},
            ),
            "probability_calibration": (
                "probability_calibration_report",
                {
                    "schema": "grape_probability_calibration/v1",
                    "passed": True,
                    "content_sha256": "b" * 64,
                },
            ),
            "failure_validation": (
                "failure_validation_report",
                {
                    "schema": "grape_failure_validation/v1",
                    "passed": True,
                    "held_out": [{"passed": True}],
                },
            ),
            "success": (
                "success_validation_report",
                {
                    "schema": "grape_success_validation/v1",
                    "passed": True,
                    "episodes": [{"passed": True}],
                },
            ),
        }
        for expected, (field, forged) in forged_passing_reports.items():
            with self.subTest(forged_gate=expected):
                evidence = self.evidence(**{field: forged})
                gates = ControllerRecommendationGates.from_evidence(
                    evidence
                )
                self.assertFalse(gates.passed)
                self.assertIn(expected, gates.failed_gates)
                report_name = (
                    "success"
                    if expected == "success"
                    else expected
                )
                self.assertFalse(
                    evidence.report_validation[report_name]
                )

        genuine = self.evidence()
        forged_mapping_gate = ControllerRecommendationGates(
            exactness_gate_passed=True,
            actuator_calibration_gate_passed=True,
            support_gate_passed=True,
            probability_calibration_gate_passed=True,
            failure_validation_gate_passed=True,
            success_gate_passed=True,
            evidence=genuine.to_mapping(),
        )
        self.assertFalse(forged_mapping_gate.evidence_bound)
        self.assertFalse(forged_mapping_gate.passed)

        compatibility = ControllerRecommendationGates(
            exactness_gate_passed=True,
            actuator_calibration_gate_passed=True,
            support_gate_passed=True,
            probability_calibration_gate_passed=True,
            failure_validation_gate_passed=True,
            success_gate_passed=True,
        )
        self.assertFalse(compatibility.passed)
        self.assertEqual(
            compatibility.failed_gates,
            ("evidence_binding", "evaluation_binding"),
        )
        legacy = ControllerRecommendationGates(
            exactness_gate_passed=True,
            actuator_calibration_gate_passed=True,
            support_gate_passed=True,
            probability_calibration_gate_passed=True,
            success_gate_passed=True,
        )
        self.assertFalse(legacy.passed)
        self.assertEqual(
            legacy.failed_gates,
            (
                "evidence_binding",
                "evaluation_binding",
                "failure_validation",
            ),
        )

    def test_recommendation_binding_rejects_cross_context_reuse(self):
        candidate = ControllerCandidate(
            "bound-pid", {"p_gain": [1.0]}
        )
        posterior = self.posterior()

        def evaluator(_candidate, _plant):
            return {
                "success": True,
                "failure": False,
                "saturated": False,
                "trajectory": ((0.0,), (0.0,)),
                "tube": _controller_tube_evidence(True),
            }

        evidence, gates, bound = self.bound_evaluation(
            candidate, posterior, evaluator
        )
        relabel_attempts = (
            (
                self.typed_reports()["support_report"],
                "support_reference_sha256",
            ),
            (
                self.typed_reports()["failure_validation_report"],
                "dataset_sha256",
            ),
            (
                self.typed_reports()[
                    "success_validation_report"
                ].episodes[0],
                "dataset_sha256",
            ),
        )
        for report, field_name in relabel_attempts:
            with self.subTest(relabel=field_name):
                with self.assertRaises(TypeError):
                    replace(report, **{field_name: "9" * 64})
        passing_support = self.typed_reports()["support_report"]
        failed_support = classify_support(
            candidate_vector=np.zeros(1),
            rollout_state_action_points=np.zeros((3, 1)),
            predictive_std=0.1,
            reference=SupportReference(
                observed_candidate_vectors=np.full((10, 1), 100.0),
                observed_state_action_points=np.full(
                    (10, 1), 100.0
                ),
                candidate_scale=np.ones(1),
                state_action_scale=np.ones(1),
                minimum_importance_ess=2.0,
                maximum_predictive_std=1.0,
            ),
        )
        self.assertEqual(failed_support.label, UNSUPPORTED)
        with self.assertRaisesRegex(
            ValueError, "support evidence/result identity mismatch"
        ):
            replace(
                passing_support,
                support_evidence=failed_support.support_evidence,
            )
        with self.assertRaises(TypeError):
            SupportEvidenceIdentity(
                support_reference=failed_support.support_reference,
                candidate_vector=np.zeros(1),
                rollout_state_action_points=np.zeros((3, 1)),
                predictive_std=0.1,
                support_result={
                    "label": SUPPORTED,
                    "candidate_distance": 0.0,
                    "state_action_distance_p95": 0.0,
                    "importance_weight_ess": 10.0,
                    "maximum_predictive_std": 0.1,
                    "reasons": (),
                },
            )

        timestamps = np.asarray([0.0, 1.0, 2.0])
        observed = np.zeros((3, 1))
        failed_failure = validate_posterior_predictive(
            timestamps=timestamps,
            observed_trajectory=observed,
            predictive_trajectories=np.full((2, 3, 1), 10.0),
            observed_failure=FailureEvent(
                "ground_contact", 2.0, "held-out/v1"
            ),
            predicted_failures=(None, None),
            minimum_coverage_fraction=1.0,
            dataset_provenance_sha256="2" * 64,
        )
        self.assertFalse(failed_failure.passed)
        with self.assertRaisesRegex(
            TypeError, "held-out evaluation functions"
        ):
            ValidationDatasetIdentity(
                "grape_held_out_failure_dataset/v1",
                {"unrelated": True},
                {
                    "trajectory": (
                        self.typed_reports()[
                            "failure_validation_report"
                        ].trajectory
                    ),
                    "failure": (
                        self.typed_reports()[
                            "failure_validation_report"
                        ].failure
                    ),
                    "passed": True,
                },
            )
        with self.assertRaisesRegex(
            ValueError, "failure dataset/result identity mismatch"
        ):
            replace(
                self.typed_reports()["failure_validation_report"],
                dataset_identity=failed_failure.dataset_identity,
            )

        failed_success = evaluate_success_episode(
            episode_id="success-1",
            role="validation_success",
            timestamps=timestamps,
            observed_trajectory=observed,
            predictive_trajectories=np.full((2, 3, 1), 10.0),
            predicted_failures=(
                FailureEvent("ground_contact", 1.0, "rollout/v1"),
                FailureEvent("ground_contact", 1.0, "rollout/v1"),
            ),
            dataset_provenance_sha256="3" * 64,
        )
        self.assertFalse(failed_success.passed)
        with self.assertRaisesRegex(
            ValueError, "success dataset/result identity mismatch"
        ):
            replace(
                self.typed_reports()[
                    "success_validation_report"
                ].episodes[0],
                dataset_identity=failed_success.dataset_identity,
            )
        with self.assertRaises(TypeError):
            ControllerRecommendationBinding.create(
                candidate=candidate,
                plant_posterior=posterior,
                plant_artifact_identity=(
                    self.plant_artifact_identity(posterior)
                ),
                evaluator_identity=bound.identity,
                held_out_failure_dataset_sha256="9" * 64,
                **self.typed_reports()
            )

        def evaluate(actual_candidate, actual_posterior, actual_evaluator):
            return evaluate_controller_candidate(
                actual_candidate,
                actual_posterior,
                actual_evaluator,
                gates,
                recommendation_threshold=0.5,
            )

        accepted = evaluate(candidate, posterior, bound)
        self.assertTrue(accepted.recommendation_allowed)
        self.assertTrue(accepted.gates.evaluation_bound)

        raw = evaluate(candidate, posterior, evaluator)
        self.assertFalse(raw.recommendation_allowed)
        self.assertFalse(raw.output_evidence_gate_passed)
        self.assertIn("evaluation_binding", raw.reasons)
        self.assertIn("evaluation_output_evidence", raw.reasons)

        other_candidate = ControllerCandidate(
            "other-bound-pid", {"p_gain": [1.0]}
        )
        cross_candidate = evaluate(other_candidate, posterior, bound)
        self.assertFalse(cross_candidate.recommendation_allowed)
        self.assertIn("evaluation_binding", cross_candidate.reasons)

        other_posterior = replace(
            posterior, provenance={"run": "other-source-run"}
        )
        cross_posterior = evaluate(
            candidate, other_posterior, bound
        )
        self.assertFalse(cross_posterior.recommendation_allowed)
        self.assertIn("evaluation_binding", cross_posterior.reasons)

        def other_evaluator(_candidate, _plant):
            return {
                "success": True,
                "failure": False,
                "saturated": False,
                "trajectory": ((1.0,), (1.0,)),
                "tube": _controller_tube_evidence(True),
            }

        other_evaluator_identity = self.evaluator_identity(
            other_evaluator
        )
        _, _, other_bound = self.bound_evaluation(
            candidate,
            posterior,
            other_evaluator,
            evaluator_identity=other_evaluator_identity,
        )
        cross_evaluator = evaluate(candidate, posterior, other_bound)
        self.assertFalse(cross_evaluator.recommendation_allowed)
        self.assertIn("evaluation_binding", cross_evaluator.reasons)

        with self.assertRaisesRegex(
            ValueError, "callable/artifact identity mismatch"
        ):
            BoundParticleEvaluator(
                bound.identity,
                evidence.binding,
                other_evaluator,
            )

        def global_template(_candidate, _plant):
            return GLOBAL_OUTCOME

        common_globals = {
            "__builtins__": __builtins__,
            "__name__": __name__,
        }
        trusted_global_evaluator = types.FunctionType(
            global_template.__code__,
            {
                **common_globals,
                "GLOBAL_OUTCOME": {
                    "success": False,
                    "failure": True,
                    "saturated": False,
                },
            },
            global_template.__name__,
        )
        changed_global_evaluator = types.FunctionType(
            global_template.__code__,
            {
                **common_globals,
                "GLOBAL_OUTCOME": {
                    "success": True,
                    "failure": False,
                    "saturated": False,
                },
            },
            global_template.__name__,
        )
        global_evidence, _, global_bound = self.bound_evaluation(
            candidate, posterior, trusted_global_evaluator
        )
        with self.assertRaisesRegex(
            ValueError, "callable/artifact identity mismatch"
        ):
            BoundParticleEvaluator(
                global_bound.identity,
                global_evidence.binding,
                changed_global_evaluator,
            )

        def attribute_template(_candidate, _plant):
            return CONFIG.GLOBAL_OUTCOME

        def attribute_evaluator(config):
            return types.FunctionType(
                attribute_template.__code__,
                {
                    **common_globals,
                    "CONFIG": config,
                },
                attribute_template.__name__,
            )

        trusted_module = types.ModuleType(
            "controller_evaluator_config"
        )
        changed_module = types.ModuleType(
            "controller_evaluator_config"
        )
        trusted_module.GLOBAL_OUTCOME = {
            "success": False,
            "failure": True,
            "saturated": False,
        }
        changed_module.GLOBAL_OUTCOME = {
            "success": True,
            "failure": False,
            "saturated": False,
        }
        trusted_module.UNRELATED = {"revision": 1}
        changed_module.UNRELATED = {"revision": 1}
        trusted_module_evaluator = attribute_evaluator(
            trusted_module
        )
        changed_module_evaluator = attribute_evaluator(
            changed_module
        )
        trusted_module_hash = measure_particle_evaluator_sha256(
            trusted_module_evaluator
        )
        trusted_module.UNRELATED["revision"] = 2
        self.assertEqual(
            trusted_module_hash,
            measure_particle_evaluator_sha256(
                trusted_module_evaluator
            ),
        )
        self.assertNotEqual(
            trusted_module_hash,
            measure_particle_evaluator_sha256(
                changed_module_evaluator
            ),
        )
        module_evidence, _, module_bound = self.bound_evaluation(
            candidate, posterior, trusted_module_evaluator
        )
        with self.assertRaisesRegex(
            ValueError, "callable/artifact identity mismatch"
        ):
            BoundParticleEvaluator(
                module_bound.identity,
                module_evidence.binding,
                changed_module_evaluator,
            )

        trusted_class = type(
            "ControllerEvaluatorConfig",
            (),
            {
                "GLOBAL_OUTCOME": {
                    "success": False,
                    "failure": True,
                    "saturated": False,
                },
                "UNRELATED": {"revision": 1},
            },
        )
        changed_class = type(
            "ControllerEvaluatorConfig",
            (),
            {
                "GLOBAL_OUTCOME": {
                    "success": True,
                    "failure": False,
                    "saturated": False,
                },
                "UNRELATED": {"revision": 1},
            },
        )
        trusted_class.__module__ = __name__
        changed_class.__module__ = __name__
        trusted_class_evaluator = attribute_evaluator(trusted_class)
        changed_class_evaluator = attribute_evaluator(changed_class)
        trusted_class_hash = measure_particle_evaluator_sha256(
            trusted_class_evaluator
        )
        trusted_class.UNRELATED["revision"] = 2
        self.assertEqual(
            trusted_class_hash,
            measure_particle_evaluator_sha256(
                trusted_class_evaluator
            ),
        )
        self.assertNotEqual(
            trusted_class_hash,
            measure_particle_evaluator_sha256(
                changed_class_evaluator
            ),
        )
        class_evidence, _, class_bound = self.bound_evaluation(
            candidate, posterior, trusted_class_evaluator
        )
        with self.assertRaisesRegex(
            ValueError, "callable/artifact identity mismatch"
        ):
            BoundParticleEvaluator(
                class_bound.identity,
                class_evidence.binding,
                changed_class_evaluator,
            )

        def dynamic_attribute_evaluator(config):
            return types.FunctionType(
                attribute_template.__code__,
                {
                    **common_globals,
                    "CONFIG": config,
                },
                attribute_template.__name__,
            )

        def dynamic_module(outcome):
            module = types.ModuleType(
                "dynamic_controller_evaluator_config"
            )

            def resolve(name):
                if name != "GLOBAL_OUTCOME":
                    raise AttributeError(name)
                return outcome

            module.__getattr__ = resolve
            return module

        trusted_dynamic_evaluator = dynamic_attribute_evaluator(
            dynamic_module(
                {
                    "success": False,
                    "failure": True,
                    "saturated": False,
                }
            )
        )
        changed_dynamic_evaluator = dynamic_attribute_evaluator(
            dynamic_module(
                {
                    "success": True,
                    "failure": False,
                    "saturated": False,
                }
            )
        )
        dynamic_evidence, _, dynamic_bound = self.bound_evaluation(
            candidate, posterior, trusted_dynamic_evaluator
        )
        self.assertNotEqual(
            measure_particle_evaluator_sha256(
                trusted_dynamic_evaluator
            ),
            measure_particle_evaluator_sha256(
                changed_dynamic_evaluator
            ),
        )
        with self.assertRaisesRegex(
            ValueError, "callable/artifact identity mismatch"
        ):
            BoundParticleEvaluator(
                dynamic_bound.identity,
                dynamic_evidence.binding,
                changed_dynamic_evaluator,
            )

        def classmethod_template(_candidate, _plant):
            return CONFIG.outcome()

        def classmethod_evaluator(config):
            return types.FunctionType(
                classmethod_template.__code__,
                {
                    **common_globals,
                    "CONFIG": config,
                },
                classmethod_template.__name__,
            )

        def class_outcome(cls):
            return cls.GLOBAL_OUTCOME

        trusted_method_class = type(
            "ControllerEvaluatorMethodConfig",
            (),
            {
                "GLOBAL_OUTCOME": {
                    "success": False,
                    "failure": True,
                    "saturated": False,
                },
                "outcome": classmethod(class_outcome),
            },
        )
        changed_method_class = type(
            "ControllerEvaluatorMethodConfig",
            (),
            {
                "GLOBAL_OUTCOME": {
                    "success": True,
                    "failure": False,
                    "saturated": False,
                },
                "outcome": classmethod(class_outcome),
            },
        )
        trusted_method_class.__module__ = __name__
        changed_method_class.__module__ = __name__
        trusted_method_evaluator = classmethod_evaluator(
            trusted_method_class
        )
        changed_method_evaluator = classmethod_evaluator(
            changed_method_class
        )
        method_evidence, _, method_bound = self.bound_evaluation(
            candidate, posterior, trusted_method_evaluator
        )
        self.assertNotEqual(
            measure_particle_evaluator_sha256(
                trusted_method_evaluator
            ),
            measure_particle_evaluator_sha256(
                changed_method_evaluator
            ),
        )
        with self.assertRaisesRegex(
            ValueError, "callable/artifact identity mismatch"
        ):
            BoundParticleEvaluator(
                method_bound.identity,
                method_evidence.binding,
                changed_method_evaluator,
            )

        mutable_state = {"success": True}

        def mutable_evaluator(_candidate, _plant):
            return {
                "success": mutable_state["success"],
                "failure": False,
                "saturated": False,
            }

        mutable_evidence, mutable_gates, mutable_bound = (
            self.bound_evaluation(
                candidate, posterior, mutable_evaluator
            )
        )
        self.assertTrue(mutable_evidence.content_is_valid())
        mutable_state["success"] = False
        with self.assertRaisesRegex(
            RuntimeError, "changed after identity measurement"
        ):
            evaluate_controller_candidate(
                candidate,
                posterior,
                mutable_bound,
                mutable_gates,
                recommendation_threshold=0.5,
            )

        changed_failure_binding = replace(
            evidence.binding,
            held_out_failure_dataset_sha256="4" * 64,
        )
        changed_failure_evaluator = BoundParticleEvaluator(
            bound.identity,
            changed_failure_binding,
            evaluator,
        )
        cross_failure_dataset = evaluate(
            candidate, posterior, changed_failure_evaluator
        )
        self.assertFalse(
            cross_failure_dataset.recommendation_allowed
        )
        self.assertIn(
            "evaluation_binding", cross_failure_dataset.reasons
        )

        changed_success_binding = replace(
            evidence.binding,
            held_out_success_dataset_sha256="5" * 64,
        )
        changed_success_evaluator = BoundParticleEvaluator(
            bound.identity,
            changed_success_binding,
            evaluator,
        )
        cross_success_dataset = evaluate(
            candidate, posterior, changed_success_evaluator
        )
        self.assertFalse(
            cross_success_dataset.recommendation_allowed
        )

        changed_support_binding = replace(
            evidence.binding,
            support_reference_sha256="6" * 64,
        )
        changed_support_evaluator = BoundParticleEvaluator(
            bound.identity,
            changed_support_binding,
            evaluator,
        )
        self.assertFalse(
            evaluate(
                candidate, posterior, changed_support_evaluator
            ).recommendation_allowed
        )

        changed_probability_binding = replace(
            evidence.binding,
            probability_dataset_sha256="7" * 64,
        )
        changed_probability_evidence = ControllerRecommendationEvidence(
            **self.typed_reports(),
            binding=changed_probability_binding,
        )
        changed_probability_result = evaluate_controller_candidate(
            candidate,
            posterior,
            BoundParticleEvaluator(
                bound.identity,
                changed_probability_binding,
                evaluator,
            ),
            ControllerRecommendationGates.from_evidence(
                changed_probability_evidence
            ),
            recommendation_threshold=0.5,
        )
        self.assertFalse(
            changed_probability_result.recommendation_allowed
        )
        self.assertIn(
            "probability_calibration",
            changed_probability_result.reasons,
        )

        bad_calibration_identity = self.evaluator_identity(
            evaluator,
            actuator_calibration_sha256="8" * 64
        )
        with self.assertRaisesRegex(
            ValueError, "calibration artifact binding mismatch"
        ):
            self.bound_evaluation(
                candidate,
                posterior,
                evaluator,
                evaluator_identity=bad_calibration_identity,
            )

        mismatched_controller = replace(
            self.typed_reports()["exactness_report"].identity,
            artifact_sha256="9" * 64,
        )
        mismatched_controller_evaluator = self.evaluator_identity(
            evaluator,
            controller_backend_identity=mismatched_controller
        )
        with self.assertRaisesRegex(
            ValueError, "controller identity"
        ):
            self.bound_evaluation(
                candidate,
                posterior,
                evaluator,
                evaluator_identity=mismatched_controller_evaluator,
            )

    def test_real_posterior_and_positive_explicit_threshold_are_required(self):
        candidate = ControllerCandidate("pid-a", {"p_gain": [1.0]})

        with self.assertRaisesRegex(TypeError, "PlantPosterior"):
            evaluate_controller_candidate(
                candidate,
                type(
                    "PosteriorLike",
                    (),
                    {
                        "particles": (1.0,),
                        "weights": np.asarray([1.0]),
                    },
                )(),
                lambda _candidate, _plant: True,
                self.gates(),
                recommendation_threshold=0.5,
            )
        with self.assertRaisesRegex(ValueError, r"\(0, 1\]"):
            evaluate_controller_candidate(
                candidate,
                self.posterior(),
                lambda _candidate, _plant: True,
                self.gates(),
                recommendation_threshold=0.0,
            )
        conflicting_particle = PlantHypothesis(
            model_id="custom-plant/v1",
            plant_parameters=np.asarray([1.0]),
            actuator_parameters=np.empty(0),
            disturbance_parameters=np.empty(0),
            plant_parameter_names=("p_gain",),
            actuator_parameter_names=(),
        )
        conflicting_posterior = PlantPosterior.from_arrays(
            particles=(conflicting_particle,),
            weights=np.asarray([1.0]),
            log_likelihood=np.zeros(1),
            model_id="custom-plant/v1",
            prior_id="custom-prior/v1",
            likelihood_id="custom-likelihood/v1",
            controller_snapshot_id="test-controller-snapshot",
        )
        with self.assertRaisesRegex(ValueError, "overlaps"):
            evaluate_controller_candidate(
                candidate,
                conflicting_posterior,
                lambda _candidate, _plant: True,
                self.gates(),
                recommendation_threshold=0.5,
            )

    def test_controller_evaluation_artifact_is_hash_bound_and_non_overwriting(self):
        candidate = ControllerCandidate(
            "pid-artifact",
            {"p_gain": [10.0], "mode": "hover"},
        )
        posterior = self.posterior()
        exact_identity = self.typed_reports()[
            "exactness_report"
        ].identity
        provenance = ControllerEvaluationProvenance(
            source_commit="test-source-commit",
            plant_posterior_sha256=posterior.content_sha256,
            controller_backend_id=exact_identity.backend_id,
            controller_backend_sha256=exact_identity.artifact_sha256,
            config_sha256="f" * 64,
            plant_artifact_identity=(
                self.plant_artifact_identity(posterior)
            ),
        )

        def evaluator(_candidate, plant):
            value = float(plant.plant_parameters[0])
            return {
                "success": value < 1.5,
                "failure": value >= 1.5,
                "saturated": value > 1.0,
                "trajectory": np.asarray([[value], [value + 0.1]]),
                "tube": _controller_tube_evidence(value < 1.5),
            }

        evidence, _, bound_evaluator = self.bound_evaluation(
            candidate, posterior, evaluator
        )
        with tempfile.TemporaryDirectory() as directory:
            run = evaluate_and_write_controller_candidate(
                output_root=directory,
                run_id="phase8-test",
                candidate=candidate,
                plant_posterior=posterior,
                particle_evaluator=bound_evaluator,
                evidence=evidence,
                recommendation_threshold=0.5,
                provenance=provenance,
            )
            self.assertTrue(run.evaluation.recommendation_allowed)
            self.assertEqual(
                {item.name for item in run.artifact_directory.iterdir()},
                set(CONTROLLER_EVALUATION_ARTIFACTS),
            )
            manifest = json.loads(
                (
                    run.artifact_directory / "artifact_manifest.json"
                ).read_text(encoding="utf-8")
            )
            for name, identity in manifest["files"].items():
                payload = (run.artifact_directory / name).read_bytes()
                self.assertEqual(
                    hashlib.sha256(payload).hexdigest(),
                    identity["sha256"],
                )
                self.assertEqual(len(payload), identity["bytes"])
            outcomes = json.loads(
                (
                    run.artifact_directory / "particle_outcomes.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(len(outcomes["particle_outcomes"]), 3)
            self.assertEqual(
                outcomes["evaluation_sha256"],
                run.evaluation.content_sha256,
            )
            first_outcome = outcomes["particle_outcomes"][0]
            self.assertEqual(
                first_outcome["outcome"]["schema"],
                "grape_controller_particle_outcome/v2",
            )
            self.assertTrue(
                first_outcome["outcome"]["saturation_measured"]
            )
            self.assertTrue(
                first_outcome["outcome"][
                    "output_evidence_status"
                ]["complete"]
            )
            self.assertEqual(
                first_outcome["output_evidence"]["schema"],
                "grape_controller_particle_output_evidence/v1",
            )
            self.assertEqual(
                first_outcome["output_evidence"][
                    "evaluation_context_sha256"
                ],
                run.evaluation.gates.evaluation_context_sha256,
            )
            summary = json.loads(
                (
                    run.artifact_directory / "controller_evaluation.json"
                ).read_text(encoding="utf-8")
            )
            self.assertTrue(
                summary["evaluation"]["output_evidence_gate_passed"]
            )
            self.assertTrue(
                summary["evaluation"]["phase8_gates_passed"]
            )

            writer = ControllerEvaluationArtifactWriter(directory)
            provenance_mismatches = (
                (
                    "backend-id-mismatch",
                    replace(
                        provenance,
                        controller_backend_id="other-exact-controller/v1",
                    ),
                    "controller provenance mismatch",
                ),
                (
                    "backend-hash-mismatch",
                    replace(
                        provenance,
                        controller_backend_sha256="9" * 64,
                    ),
                    "controller provenance mismatch",
                ),
                (
                    "config-hash-mismatch",
                    replace(provenance, config_sha256="8" * 64),
                    "evaluator config provenance mismatch",
                ),
                (
                    "unverified-plant-artifact",
                    replace(
                        provenance,
                        plant_artifact_identity=None,
                    ),
                    "plant artifact provenance mismatch",
                ),
            )
            for run_id, bad_provenance, expected in provenance_mismatches:
                with self.subTest(run_id=run_id):
                    with self.assertRaisesRegex(ValueError, expected):
                        writer.write(
                            run_id=run_id,
                            evaluation=run.evaluation,
                            provenance=bad_provenance,
                        )

            forged_gates = replace(
                run.evaluation.gates,
                evaluation_context_sha256="0" * 64,
            )
            with self.assertRaisesRegex(
                ValueError, "recommendation gate is inconsistent"
            ):
                replace(
                    run.evaluation,
                    gates=forged_gates,
                    recommendation_allowed=True,
                    reasons=(),
                )

            with self.assertRaises(FileExistsError):
                evaluate_and_write_controller_candidate(
                    output_root=directory,
                    run_id="phase8-test",
                    candidate=candidate,
                    plant_posterior=posterior,
                    particle_evaluator=bound_evaluator,
                    evidence=evidence,
                    recommendation_threshold=0.5,
                    provenance=provenance,
                )

            race_destination = Path(directory) / "phase8-race"
            original_write_payloads = (
                ControllerEvaluationArtifactWriter._write_payloads
            )

            def write_payloads_then_compete(*arguments):
                original_write_payloads(*arguments)
                race_destination.mkdir()

            with mock.patch.object(
                ControllerEvaluationArtifactWriter,
                "_write_payloads",
                side_effect=write_payloads_then_compete,
            ):
                with self.assertRaises(FileExistsError):
                    evaluate_and_write_controller_candidate(
                        output_root=directory,
                        run_id="phase8-race",
                        candidate=candidate,
                        plant_posterior=posterior,
                        particle_evaluator=bound_evaluator,
                        evidence=evidence,
                        recommendation_threshold=0.5,
                        provenance=provenance,
                    )
            self.assertEqual(tuple(race_destination.iterdir()), ())
            self.assertFalse(
                any(
                    item.name.startswith(".phase8-race.staging.")
                    for item in Path(directory).iterdir()
                )
            )

    def test_target_tube_and_support_compatibility_wrappers_are_exact(self):
        from grape_param_estim.counterfactual import (
            SupportReference as ExistingSupportReference,
            TargetTube as ExistingTargetTube,
        )
        from grape_param_estim.validation import (
            SupportReference as PackageSupportReference,
            TargetTube as PackageTargetTube,
        )
        from grape_param_estim.validation.support import SupportReference
        from grape_param_estim.validation.trajectory_tube import TargetTube

        self.assertIs(TargetTube, ExistingTargetTube)
        self.assertIs(SupportReference, ExistingSupportReference)
        self.assertIs(PackageTargetTube, ExistingTargetTube)
        self.assertIs(PackageSupportReference, ExistingSupportReference)


if __name__ == "__main__":
    unittest.main()
