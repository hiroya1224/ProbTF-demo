from dataclasses import asdict, replace
from pathlib import Path
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from grape_param_estim.alternative_backends import (
    EXACT_ORACLE_PROTOCOL,
    REQUIRED_CONFORMANCE_CHANNELS,
    REQUIRED_ORACLE_CAPABILITIES,
    ExactOracleConformanceReport,
    ExactOracleFixtureProvenance,
    ExactOracleIdentity,
)
from grape_param_estim.controller_replay import (
    ControllerParameters,
    ControllerReplay,
    PidLimits,
    ReplayMetrics,
)
from grape_param_estim.counterfactual import (
    DEPENDENCE_APPROXIMATED,
    DEPENDENCE_JOINT_SAMPLES,
    EXTRAPOLATIVE,
    SUPPORTED,
    UNSUPPORTED,
    ClosedLoopCounterfactualEvaluator,
    CounterfactualCandidate,
    CounterfactualConfig,
    InitialStateSample,
    JointPosteriorSample,
    ProbabilityCalibrationReport,
    PythonControllerReplayBackend,
    SupportReference,
    TargetTrajectory,
    TargetTube,
    classify_support,
    connected_candidate_regions,
    evaluate_target_tube,
)
from grape_param_estim.episode import stable_hash
from grape_param_estim.selection import (
    DEFAULT_CANDIDATE,
    SELECTION_SCHEMA,
    load_selection_protocol,
)
from grape_param_estim.effective_response import (
    EffectiveResponseParameters,
    EffectiveResponsePosterior,
    IdentifiabilityReport,
    ResponseState,
)


def controller(delay_compensation=0.0, p_gain=3.0):
    return ControllerParameters(
        p_gain=np.full(6, p_gain),
        i_gain=np.zeros(6),
        d_gain=np.full(6, 1.2),
        limits=PidLimits(
            output=50.0,
            p_term=50.0,
            i_term=10.0,
            d_term=50.0,
            p_error=10.0,
            i_state=10.0,
            d_error=10.0,
        ),
        controller_mass=1.0,
        controller_inertia_diagonal=np.ones(3),
        allocation_scale=np.ones(6),
        thrust_scale=1.0,
        delay_compensation_s=delay_compensation,
    )


def response_posterior():
    samples = []
    for delay in (0.12, 0.15, 0.18):
        samples.append(
            EffectiveResponseParameters(
                effectiveness=np.eye(6),
                time_constant_s=np.full(6, 0.05),
                delay_s=np.full(6, delay),
                damping=np.full(6, 0.05),
                bias=np.zeros(6),
            )
        )
    identifiability = IdentifiabilityReport(
        design_rank=48,
        parameter_count=48,
        condition_number=2.0,
        per_axis_design_rank=np.full(6, 8),
        per_axis_condition_number=np.full(6, 2.0),
        singular_values=np.ones(48),
        null_directions=np.empty((0, 48)),
        posterior_maximum_absolute_correlation=0.2,
        identifiable=True,
        scope="conditional_linear_coefficients_given_delay_and_lag",
    )
    return EffectiveResponsePosterior(
        samples=tuple(samples),
        weights=np.full(3, 1.0 / 3.0),
        grid_delay_s=np.array([0.12, 0.15, 0.18]),
        grid_time_constant_s=np.full(3, 0.05),
        grid_weights=np.full(3, 1.0 / 3.0),
        identifiability=identifiability,
        log_evidence=0.0,
        approximation="test_posterior",
        source_sample_ids=(0,),
    )


def target_trajectory():
    times = np.arange(0.0, 5.0 + 1.0e-9, 0.05)
    position = np.zeros((times.size, 6))
    velocity = np.zeros_like(position)
    acceleration = np.zeros_like(position)
    frequency = 1.2
    position[:, 0] = 0.4 * np.sin(frequency * times)
    velocity[:, 0] = 0.4 * frequency * np.cos(frequency * times)
    acceleration[:, 0] = -0.4 * frequency ** 2 * np.sin(frequency * times)
    return TargetTrajectory(times, position, velocity, acceleration)


def support_reference(candidates):
    vectors = np.asarray([item.vector() for item in candidates])
    # Multiple factual settings keep importance ESS meaningful.
    observed = np.vstack((vectors, vectors + 1.0e-3, vectors - 1.0e-3))
    state_action = np.zeros((8, 18))
    return SupportReference(
        observed_candidate_vectors=observed,
        observed_state_action_points=state_action,
        candidate_scale=np.full(vectors.shape[1], 0.5),
        state_action_scale=np.full(18, 5.0),
        supported_distance=3.0,
        unsupported_distance=8.0,
        minimum_importance_ess=2.0,
        maximum_predictive_std=2.0,
    )


def probability_calibration_report(identity, exact_conformance_report):
    repository = Path(__file__).resolve().parents[2]
    protocol = load_selection_protocol(
        repository
        / "ros/examples/grape-param-estim/config/selection_protocol.yaml"
    )
    candidate_id = "bayesian_closed_loop"
    groups = {
        group_id: {
            "selected_default": (
                candidate_id
                if group_id == "counterfactual_usefulness"
                else group["candidates"][0]["candidate_id"]
            )
        }
        for group_id, group in protocol["candidate_groups"].items()
    }
    backend_identity = asdict(identity)
    backend_identity.update(
        {
            "is_exact": True,
            "supports_closed_loop_plant_callback": True,
            "applies_candidate_parameters": True,
            "applies_delay_compensation": True,
        }
    )
    normalized_hashes = tuple(
        stable_hash(("normalized-dataset", index)) for index in range(12)
    )
    candidates = {
        item["candidate_id"]: {
            "observation_count": 12,
            "missing_folds": (),
            "missing_metric_folds": (),
        }
        for group in protocol["candidate_groups"].values()
        for item in group["candidates"]
    }
    candidates[candidate_id].update(
        {
            "status": DEFAULT_CANDIDATE,
            "primary_metric": "held_out_brier_score",
            "failed_hard_gates": (),
            "run_hashes": tuple(
                stable_hash(("calibration-run", index))
                for index in range(12)
            ),
            "trajectory_sample_bundle_hashes": normalized_hashes,
            "model_versions": (
                "low_dimensional_effective_response/v1",
            ),
            "controller_backend_identity_hashes": (
                stable_hash(backend_identity),
            ),
            "exact_conformance_report_hashes": (
                exact_conformance_report.evidence_sha256,
            ),
            "statistics": {
                "episode_count": 12,
                "mean": 0.08,
                "bootstrap_lower": 0.04,
                "bootstrap_upper": 0.12,
                "standard_error": 0.02,
            },
        }
    )
    selection_result = {
        "schema": SELECTION_SCHEMA,
        "protocol_hash": stable_hash(protocol),
        "manifest_hash": protocol["manifest_hash"],
        "source_commit": "test-commit",
        "outer_fold_count": 12,
        "selection_complete": True,
        "groups": groups,
        "candidates": candidates,
    }
    selection_result["result_hash"] = stable_hash(selection_result)
    report = ProbabilityCalibrationReport.from_selection_result(
        selection_protocol=protocol,
        selection_result=selection_result,
        model_version="low_dimensional_effective_response/v1",
        controller_backend_identity=backend_identity,
    )
    return report, protocol, normalized_hashes


class CounterfactualTests(unittest.TestCase):
    def test_so3_error_and_physical_tilt_handle_coupled_rotation(self):
        times = np.array([0.0, 0.1, 0.2])
        desired = np.zeros((3, 6))
        actual = np.zeros((3, 6))
        desired_rotation = Rotation.from_euler("z", np.pi - 0.02)
        actual_rotation = desired_rotation * Rotation.from_euler("x", 0.05)
        desired[:, 3:] = desired_rotation.as_rotvec()
        actual[:, 3:] = actual_rotation.as_rotvec()
        target = TargetTrajectory(
            times, desired, np.zeros_like(desired), np.zeros_like(desired)
        )
        result = evaluate_target_tube(
            target,
            TargetTube(
                position_tolerance=np.full(6, 0.1),
                velocity_tolerance=np.ones(6),
                maximum_tilt_rad=0.1,
            ),
            actual,
            np.zeros_like(actual),
            np.zeros_like(actual, dtype=bool),
        )
        self.assertTrue(result.success)
        self.assertLess(result.maximum_position_ratio, 1.0)

    def test_target_tube_checks_tracking_saturation_ground_and_tilt(self):
        times = np.arange(0.0, 0.51, 0.1)
        zeros = np.zeros((times.size, 6))
        target = TargetTrajectory(times, zeros, zeros, zeros)
        actual = zeros.copy()
        actual[:, 2] = 0.4
        actual[3, 0] = 0.5
        actual[4, 3] = 0.7
        saturation = np.zeros_like(actual, dtype=bool)
        saturation[1:4, 0] = True
        result = evaluate_target_tube(
            target,
            TargetTube(
                position_tolerance=np.full(6, 0.2),
                velocity_tolerance=np.full(6, 0.3),
                maximum_continuous_saturation_s=0.15,
                minimum_height_m=0.5,
                maximum_tilt_rad=0.5,
            ),
            actual,
            zeros,
            saturation,
        )
        self.assertFalse(result.success)
        self.assertIn("position_tube", result.violations)
        self.assertIn("saturation_duration", result.violations)
        self.assertIn("ground_or_contact", result.violations)
        self.assertIn("tilt_safety", result.violations)

    def test_short_pointwise_excursion_can_use_explicit_duration_allowance(self):
        times = np.arange(0.0, 0.51, 0.1)
        zeros = np.zeros((times.size, 6))
        target = TargetTrajectory(times, zeros, zeros, zeros)
        actual = zeros.copy()
        actual[2, 0] = 0.25
        result = evaluate_target_tube(
            target,
            TargetTube(
                position_tolerance=np.full(6, 0.2),
                velocity_tolerance=np.ones(6),
                allowed_outside_duration_s=0.11,
            ),
            actual,
            zeros,
            np.zeros_like(actual, dtype=bool),
        )
        self.assertTrue(result.success)
        self.assertIn("position_tube", result.diagnostic_exceedances)
        self.assertNotIn("position_tube", result.violations)

    def test_support_labels_use_parameters_state_action_ess_and_uncertainty(self):
        reference = SupportReference(
            observed_candidate_vectors=np.array(
                [[0.0, 0.0], [0.1, 0.0], [-0.1, 0.0]]
            ),
            observed_state_action_points=np.array(
                [[0.0, 0.0], [0.2, 0.0], [-0.2, 0.0]]
            ),
            candidate_scale=np.ones(2),
            state_action_scale=np.ones(2),
            supported_distance=1.0,
            unsupported_distance=4.0,
            minimum_importance_ess=1.0,
            maximum_predictive_std=1.0,
        )
        supported = classify_support(
            np.array([0.0, 0.0]),
            np.array([[0.1, 0.0], [0.0, 0.1]]),
            0.1,
            reference,
        )
        self.assertEqual(supported.label, SUPPORTED)
        extrapolative = classify_support(
            np.array([2.0, 0.0]),
            np.array([[0.0, 0.0]]),
            0.1,
            reference,
        )
        self.assertEqual(extrapolative.label, EXTRAPOLATIVE)
        unsupported = classify_support(
            np.array([10.0, 0.0]),
            np.array([[10.0, 0.0]]),
            2.0,
            reference,
        )
        self.assertEqual(unsupported.label, UNSUPPORTED)
        self.assertIn("candidate_parameter_distance", unsupported.reasons)
        self.assertIn("state_action_distance", unsupported.reasons)
        self.assertIn("posterior_predictive_uncertainty", unsupported.reasons)
        nonfinite = classify_support(
            np.array([0.0, 0.0]),
            np.array([[0.0, 0.0]]),
            float("nan"),
            reference,
        )
        self.assertEqual(nonfinite.label, UNSUPPORTED)

    def test_closed_loop_recomputes_commands_and_delay_compensation_changes_rollout(self):
        target = target_trajectory()
        baseline = CounterfactualCandidate("baseline", controller(0.0))
        compensated = CounterfactualCandidate("compensated", controller(0.15))
        reference = support_reference((baseline, compensated))
        evaluator = ClosedLoopCounterfactualEvaluator(reference)
        initial = InitialStateSample(
            sample_id=7,
            state=ResponseState(
                target.position[0],
                target.velocity[0],
                np.zeros(6),
            ),
            weight=1.0,
            controller_integral_state=np.zeros(6),
            integrator_state_source="explicit_test_assumption",
        )
        tube = TargetTube(
            position_tolerance=np.full(6, 0.25),
            velocity_tolerance=np.full(6, 0.5),
            evaluation_start_offset_s=0.5,
            maximum_continuous_saturation_s=0.5,
            allowed_outside_duration_s=1.5,
        )
        config = CounterfactualConfig(
            process_noise_sigma=np.full(6, 0.002),
            process_noise_replicates=3,
            seed=22,
            source_bag_hashes=("a" * 64,),
            normalized_dataset_hashes=("b" * 64,),
        )
        first = evaluator.evaluate(
            baseline,
            target,
            tube,
            response_posterior(),
            [initial],
            config,
        )
        second = evaluator.evaluate(
            compensated,
            target,
            tube,
            response_posterior(),
            [initial],
            config,
        )
        repeated = evaluator.evaluate(
            baseline,
            target,
            tube,
            response_posterior(),
            [initial],
            config,
        )
        self.assertNotEqual(first.run_id, second.run_id)
        self.assertEqual(first.run_id, repeated.run_id)
        self.assertEqual(first.support.label, SUPPORTED)
        self.assertEqual(second.support.label, SUPPORTED)
        self.assertFalse(first.recommendable)
        self.assertEqual(first.workflow_status, "EXPERIMENTAL")
        self.assertEqual(first.dependence_handling, DEPENDENCE_APPROXIMATED)
        self.assertGreater(first.effective_rollout_sample_size, 1.0)
        self.assertGreaterEqual(first.credible_upper, first.credible_lower)
        self.assertGreaterEqual(first.success_probability, 0.0)
        self.assertLessEqual(first.success_probability, 1.0)
        baseline_position = first.rollouts[0].position[:, 0]
        compensated_position = second.rollouts[0].position[:, 0]
        self.assertFalse(np.allclose(baseline_position, compensated_position))
        self.assertFalse(
            np.allclose(
                first.rollouts[0].command,
                second.rollouts[0].command,
            )
        )
        np.testing.assert_allclose(
            first.rollouts[0].position,
            repeated.rollouts[0].position,
        )
        regions = connected_candidate_regions([first, second], gamma=0.0)
        self.assertEqual(regions, ())

    def test_online_prefix_provenance_requires_cutoff(self):
        with self.assertRaisesRegex(ValueError, "prefix_cutoff"):
            CounterfactualConfig(
                process_noise_sigma=np.zeros(6),
                analysis_mode="online_prefix",
            )
        valid = CounterfactualConfig(
            process_noise_sigma=np.zeros(6),
            analysis_mode="online_prefix",
            prefix_cutoff=2.0,
            inference_data_end_time=2.0,
        )
        self.assertEqual(valid.prefix_cutoff, 2.0)
        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            CounterfactualConfig(
                process_noise_sigma=np.zeros(6),
                analysis_mode="online_prefix",
                prefix_cutoff=2.0,
                inference_data_end_time=2.1,
            )

    def test_online_prefix_evaluator_rejects_unstamped_or_future_initial_state(self):
        target = target_trajectory()
        # Rebase target so it begins at the causal cutoff.
        shifted = TargetTrajectory(
            target.timestamps + 2.0,
            target.position,
            target.velocity,
            target.acceleration,
        )
        candidate = CounterfactualCandidate("causal", controller())
        evaluator = ClosedLoopCounterfactualEvaluator(
            support_reference((candidate,))
        )
        unstamped = InitialStateSample(
            1,
            ResponseState(
                shifted.position[0], shifted.velocity[0], np.zeros(6)
            ),
            1.0,
            controller_integral_state=np.zeros(6),
            integrator_state_source="explicit_test_assumption",
        )
        with self.assertRaisesRegex(ValueError, "stamped at prefix_cutoff"):
            evaluator.evaluate(
                candidate,
                shifted,
                TargetTube(np.ones(6), np.ones(6)),
                response_posterior(),
                [unstamped],
                CounterfactualConfig(
                    process_noise_sigma=np.zeros(6),
                    analysis_mode="online_prefix",
                    prefix_cutoff=2.0,
                    inference_data_end_time=2.0,
                ),
            )

    def test_rollout_arrays_are_immutable(self):
        target = target_trajectory()
        candidate = CounterfactualCandidate("immutable", controller())
        evaluator = ClosedLoopCounterfactualEvaluator(
            support_reference((candidate,))
        )
        result = evaluator.evaluate(
            candidate,
            target,
            TargetTube(
                np.ones(6),
                np.ones(6),
                allowed_outside_duration_s=5.0,
            ),
            response_posterior(),
            [
                InitialStateSample(
                    0,
                    ResponseState(
                        target.position[0],
                        target.velocity[0],
                        np.zeros(6),
                    ),
                    1.0,
                    controller_integral_state=np.zeros(6),
                    integrator_state_source="explicit_test_assumption",
                )
            ],
            CounterfactualConfig(process_noise_sigma=np.zeros(6)),
        )
        with self.assertRaises(ValueError):
            result.rollouts[0].position[0, 0] = 99.0

    def test_missing_integrator_state_is_never_silently_zero_filled(self):
        target = target_trajectory()
        candidate = CounterfactualCandidate("missing-integrator", controller())
        initial = InitialStateSample(
            0,
            ResponseState(
                target.position[0], target.velocity[0], np.zeros(6)
            ),
            1.0,
        )
        with self.assertRaisesRegex(
            ValueError, "controller_integral_state"
        ):
            ClosedLoopCounterfactualEvaluator(
                support_reference((candidate,))
            ).evaluate(
                candidate,
                target,
                TargetTube(np.ones(6), np.ones(6)),
                response_posterior(),
                [initial],
                CounterfactualConfig(process_noise_sigma=np.zeros(6)),
            )

    def test_verified_exact_backend_joint_samples_and_gates_are_required(self):
        identity = ExactOracleIdentity(
            protocol=EXACT_ORACLE_PROTOCOL,
            backend_id="test_cpp_closed_loop_controller/v1",
            implementation_language="C++",
            source_commit="0123456789abcdef",
            artifact_sha256="a" * 64,
            capabilities=REQUIRED_ORACLE_CAPABILITIES,
        )

        class CapturingExactBackend:
            is_exact = True
            supports_closed_loop_plant_callback = True
            applies_candidate_parameters = True
            applies_delay_compensation = True

            def run(self, *args, **kwargs):
                self.integral = np.array(
                    kwargs["initial_integral_state"], copy=True
                )
                replay = ControllerReplay().run(*args, **kwargs)
                return replace(
                    replay,
                    backend_id=self.backend_id,
                    is_exact=True,
                )

        CapturingExactBackend.backend_id = identity.backend_id
        CapturingExactBackend.identity = identity
        replay_metric = ReplayMetrics(
            normalized_rmse=np.zeros(6),
            normalized_maximum_error=np.zeros(6),
            event_agreement=1.0,
            passed=True,
            rmse_threshold=0.01,
            maximum_error_threshold=0.03,
            event_agreement_threshold=1.0,
        )
        timestamp_metric = ReplayMetrics(
            normalized_rmse=np.zeros(1),
            normalized_maximum_error=np.zeros(1),
            event_agreement=1.0,
            passed=True,
            rmse_threshold=0.0,
            maximum_error_threshold=0.0,
            event_agreement_threshold=1.0,
        )
        repository = Path(__file__).resolve().parents[2]
        fixture_protocol = load_selection_protocol(
            repository
            / "ros/examples/grape-param-estim/config/selection_protocol.yaml"
        )
        fixture_source_hash = next(
            iter(fixture_protocol["episodes"].values())
        )["bag_sha256"]
        fixture_payload = {"factual_replay": "unit-test"}
        fixture_continuous = {
            channel: np.zeros((3, 1))
            for channel in REQUIRED_CONFORMANCE_CHANNELS
        }
        fixture_events = np.zeros(3, dtype=int)
        fixture_provenance = ExactOracleFixtureProvenance.create(
            source_bag_sha256=fixture_source_hash,
            source_topics=tuple(
                "/fixture/" + channel
                for channel in REQUIRED_CONFORMANCE_CHANNELS
            ),
            interval_start_time_ns=1_000_000_000,
            interval_end_time_ns=2_000_000_000,
            frame_conventions={"controller": "body_flu"},
            unit_conventions={"controller": "SI"},
            motor_order=("motor1", "motor2", "motor3", "motor4"),
            request_payload=fixture_payload,
            continuous=fixture_continuous,
            events=fixture_events,
            extraction_config_sha256="c" * 64,
            source_commit="fixture-source-commit",
        )
        report = ExactOracleConformanceReport(
            passed=True,
            status="PASS",
            reasons=(),
            channel_metrics={
                channel: (
                    timestamp_metric
                    if channel == "command_timestamp"
                    else replay_metric
                )
                for channel in REQUIRED_CONFORMANCE_CHANNELS
            },
            identity=identity,
            fixture_provenance=fixture_provenance,
            fixture_content_sha256=fixture_provenance.content_sha256,
            request_payload_sha256=(
                fixture_provenance.fixture_input_payload_sha256
            ),
        )
        calibration, protocol, normalized_hashes = (
            probability_calibration_report(identity, report)
        )
        backend = CapturingExactBackend()
        target = TargetTrajectory(
            np.array([0.0, 0.1, 0.2]),
            np.zeros((3, 6)),
            np.zeros((3, 6)),
            np.zeros((3, 6)),
        )
        candidate = CounterfactualCandidate("verified-exact", controller())
        integral = np.linspace(0.01, 0.06, 6)
        initial = InitialStateSample(
            5,
            ResponseState(np.zeros(6), np.zeros(6), np.zeros(6)),
            1.0,
            controller_integral_state=integral,
            integrator_state_source="latent_posterior_sample",
        )
        evaluator = ClosedLoopCounterfactualEvaluator(
            support_reference((candidate,)),
            controller_backend_factory=lambda: backend,
            exact_oracle_conformance_report=report,
            probability_calibration_report=calibration,
        )
        source_hash = next(iter(protocol["episodes"].values()))["bag_sha256"]
        config = CounterfactualConfig(
            process_noise_sigma=np.zeros(6),
            recommendation_threshold=0.1,
            source_bag_hashes=(source_hash,),
            normalized_dataset_hashes=(normalized_hashes[0],),
            source_commit="test-commit",
        )
        joint = (JointPosteriorSample(11, 5, 0, 1.0),)
        result = evaluator.evaluate(
            candidate,
            target,
            TargetTube(
                np.full(6, 100.0),
                np.full(6, 100.0),
                allowed_outside_duration_s=1.0,
            ),
            response_posterior(),
            [initial],
            config,
            joint,
        )
        np.testing.assert_array_equal(backend.integral, integral)
        self.assertEqual(result.dependence_handling, DEPENDENCE_JOINT_SAMPLES)
        self.assertTrue(result.exact_controller_gate_passed)
        self.assertTrue(result.probability_calibration_gate_passed)
        self.assertTrue(result.integrator_state_gate_passed)
        self.assertTrue(result.recommendable)
        self.assertEqual(result.workflow_status, "MANUAL_REVIEW_REQUIRED")
        self.assertEqual(
            connected_candidate_regions([result], gamma=0.0),
            (("verified-exact",),),
        )

        other_source_hash = tuple(
            item["bag_sha256"] for item in protocol["episodes"].values()
        )[1]
        cross_bag_result = evaluator.evaluate(
            candidate,
            target,
            TargetTube(
                np.full(6, 100.0),
                np.full(6, 100.0),
                allowed_outside_duration_s=1.0,
            ),
            response_posterior(),
            [initial],
            replace(config, source_bag_hashes=(other_source_hash,)),
            joint,
        )
        self.assertFalse(cross_bag_result.exact_controller_gate_passed)
        self.assertFalse(cross_bag_result.recommendable)

        lenient_metric = ReplayMetrics(
            normalized_rmse=np.full(6, 0.02),
            normalized_maximum_error=np.full(6, 0.04),
            event_agreement=0.95,
            passed=True,
            rmse_threshold=0.10,
            maximum_error_threshold=0.10,
            event_agreement_threshold=0.90,
        )
        with self.assertRaisesRegex(ValueError, "frozen"):
            replace(
                report,
                channel_metrics={
                    channel: lenient_metric
                    for channel in REQUIRED_CONFORMANCE_CHANNELS
                },
            )

        with self.assertRaisesRegex(
            ValueError, "thresholds/status"
        ):
            replace(replay_metric, passed=np.bool_(True))
        with self.assertRaisesRegex(ValueError, "passed flag disagrees"):
            ReplayMetrics(
                normalized_rmse=np.full(6, 1.0),
                normalized_maximum_error=np.full(6, 1.0),
                event_agreement=0.0,
                passed=True,
                rmse_threshold=0.01,
                maximum_error_threshold=0.03,
                event_agreement_threshold=1.0,
            )

        no_report = ClosedLoopCounterfactualEvaluator(
            support_reference((candidate,)),
            controller_backend_factory=CapturingExactBackend,
        ).evaluate(
            candidate,
            target,
            TargetTube(
                np.full(6, 100.0),
                np.full(6, 100.0),
                allowed_outside_duration_s=1.0,
            ),
            response_posterior(),
            [initial],
            config,
            joint,
        )
        self.assertFalse(no_report.exact_controller_gate_passed)
        self.assertFalse(no_report.recommendable)
        self.assertEqual(no_report.workflow_status, "EXPERIMENTAL")

        assumed_integrator = evaluator.evaluate(
            candidate,
            target,
            TargetTube(
                np.full(6, 100.0),
                np.full(6, 100.0),
                allowed_outside_duration_s=1.0,
            ),
            response_posterior(),
            [
                replace(
                    initial,
                    integrator_state_source="explicit_test_assumption",
                )
            ],
            config,
            joint,
        )
        self.assertTrue(assumed_integrator.exact_controller_gate_passed)
        self.assertTrue(
            assumed_integrator.probability_calibration_gate_passed
        )
        self.assertFalse(assumed_integrator.integrator_state_gate_passed)
        self.assertFalse(assumed_integrator.recommendable)
        self.assertEqual(assumed_integrator.workflow_status, "EXPERIMENTAL")

        class SurrogateWrappedAsExact(PythonControllerReplayBackend):
            is_exact = True

        SurrogateWrappedAsExact.backend_id = identity.backend_id
        SurrogateWrappedAsExact.identity = identity
        with self.assertRaisesRegex(
            ValueError, "replay result does not match"
        ):
            ClosedLoopCounterfactualEvaluator(
                support_reference((candidate,)),
                controller_backend_factory=SurrogateWrappedAsExact,
                exact_oracle_conformance_report=report,
                probability_calibration_report=calibration,
            ).evaluate(
                candidate,
                target,
                TargetTube(
                    np.full(6, 100.0),
                    np.full(6, 100.0),
                    allowed_outside_duration_s=1.0,
                ),
                response_posterior(),
                [initial],
                config,
                joint,
            )

        class NonBooleanExactBackend(CapturingExactBackend):
            is_exact = "false"

        NonBooleanExactBackend.backend_id = identity.backend_id
        NonBooleanExactBackend.identity = identity
        with self.assertRaisesRegex(TypeError, "is_exact"):
            ClosedLoopCounterfactualEvaluator(
                support_reference((candidate,)),
                controller_backend_factory=NonBooleanExactBackend,
                exact_oracle_conformance_report=report,
                probability_calibration_report=calibration,
            ).evaluate(
                candidate,
                target,
                TargetTube(
                    np.full(6, 100.0),
                    np.full(6, 100.0),
                    allowed_outside_duration_s=1.0,
                ),
                response_posterior(),
                [initial],
                config,
                joint,
            )

        for capability in (
            "supports_closed_loop_plant_callback",
            "applies_candidate_parameters",
            "applies_delay_compensation",
        ):
            for invalid in ("false", 0, np.bool_(False)):
                with self.subTest(
                    capability=capability,
                    invalid=repr(invalid),
                ):
                    class NonBooleanCapabilityBackend(
                        CapturingExactBackend
                    ):
                        pass

                    setattr(
                        NonBooleanCapabilityBackend,
                        capability,
                        invalid,
                    )
                    with self.assertRaisesRegex(TypeError, capability):
                        ClosedLoopCounterfactualEvaluator(
                            support_reference((candidate,)),
                            controller_backend_factory=(
                                NonBooleanCapabilityBackend
                            ),
                            exact_oracle_conformance_report=report,
                            probability_calibration_report=calibration,
                        ).evaluate(
                            candidate,
                            target,
                            TargetTube(
                                np.full(6, 100.0),
                                np.full(6, 100.0),
                                allowed_outside_duration_s=1.0,
                            ),
                            response_posterior(),
                            [initial],
                            config,
                            joint,
                        )

    def test_run_id_hashes_every_behavioral_input(self):
        times = np.array([0.0, 0.1, 0.2])
        zeros = np.zeros((3, 6))
        target = TargetTrajectory(times, zeros, zeros, zeros)
        candidate = CounterfactualCandidate("content-hash", controller())
        initial = InitialStateSample(
            8,
            ResponseState(np.zeros(6), np.zeros(6), np.zeros(6)),
            1.0,
            controller_integral_state=np.zeros(6),
            integrator_state_source="latent_posterior_sample",
        )
        posterior = response_posterior()
        tube = TargetTube(
            np.full(6, 100.0),
            np.full(6, 100.0),
            allowed_outside_duration_s=1.0,
        )
        reference = support_reference((candidate,))
        config = CounterfactualConfig(
            process_noise_sigma=np.zeros(6), seed=77
        )

        def evaluate(
            selected_target=target,
            selected_initial=initial,
            selected_posterior=posterior,
            selected_reference=reference,
            selected_config=config,
        ):
            return ClosedLoopCounterfactualEvaluator(
                selected_reference
            ).evaluate(
                candidate,
                selected_target,
                tube,
                selected_posterior,
                [selected_initial],
                selected_config,
            )

        baseline = evaluate()
        self.assertEqual(baseline.run_id, evaluate().run_id)
        changed_integrator = replace(
            initial,
            controller_integral_state=np.ones(6) * 0.01,
        )
        changed_weights = replace(
            posterior, weights=np.array([0.8, 0.1, 0.1])
        )
        changed_target_position = np.array(zeros, copy=True)
        changed_target_position[1, 0] = 0.01
        changed_target = TargetTrajectory(
            times, changed_target_position, zeros, zeros
        )
        changed_reference = replace(
            reference, supported_distance=2.5
        )
        changed_config = replace(
            config, process_noise_replicates=2
        )
        identifiers = {
            baseline.run_id,
            evaluate(selected_initial=changed_integrator).run_id,
            evaluate(selected_posterior=changed_weights).run_id,
            evaluate(selected_target=changed_target).run_id,
            evaluate(selected_reference=changed_reference).run_id,
            evaluate(selected_config=changed_config).run_id,
        }
        self.assertEqual(len(identifiers), 6)


if __name__ == "__main__":
    unittest.main()
