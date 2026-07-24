import hashlib
import os
import unittest

import numpy as np

from grape_param_estim.alternative_backends import (
    EXACT_ORACLE_PROTOCOL,
    BatchImuPreintegrationSmoother,
    ExactOracleConformanceFixture,
    ExactOracleFixtureProvenance,
    ExactOracleIdentity,
    ExactOracleProtocolError,
    ExactOracleReplayOutput,
    ExactOracleUnavailable,
    FactorGraphSmootherConfig,
    LinearGaussianRandomWalkModel,
    ParticleLikelihoodDegeneracy,
    ParticleMarginalMetropolisHastings,
    ParticleMarginalMhConfig,
    ParticleStateSpaceModel,
    REQUIRED_ORACLE_CAPABILITIES,
    StructuredMechanicsParameters,
    StructuredSixDofMechanicsResponse,
    SubprocessExactControllerOracle,
    bootstrap_particle_log_likelihood,
    compare_pmmh_with_modular_smc,
    evaluate_conditional_candidate,
    evaluate_exact_oracle_conformance,
)
from grape_param_estim.inference import (
    BoundedLogitTransform,
    BoxUniformPrior,
    TemperedResampleMoveSmc,
    TemperedSmcConfig,
)
from grape_param_estim.state_smoother import (
    SmootherConfig,
    TrajectoryObservations,
    TrajectoryPosterior,
)


def trajectory_observations(future_shift=False):
    times = np.arange(0.0, 2.01, 0.1)
    acceleration = 0.4
    position = np.zeros((times.size, 3))
    position[:, 0] = 0.5 * acceleration * times * times
    accelerometer = np.zeros((times.size, 3))
    accelerometer[:, 0] = acceleration
    accelerometer[:, 2] = 9.80665
    if future_shift:
        position[times > 1.0, 0] += 100.0
        accelerometer[times > 1.0, 0] += 50.0
    quaternion = np.zeros((times.size, 4))
    quaternion[:, 3] = 1.0
    mocap_valid = ~((times >= 0.6) & (times <= 1.3))
    return TrajectoryObservations(
        mocap_times=times,
        mocap_positions_world=position,
        mocap_quaternions_xyzw=quaternion,
        imu_times=times,
        accelerometer_body=accelerometer,
        gyro_body=np.zeros((times.size, 3)),
        mocap_valid_mask=mocap_valid,
    )


def inertial_parameters():
    return np.array(
        [
            2.0,
            0.1,
            -0.05,
            0.2,
            0.8,
            0.02,
            -0.01,
            1.0,
            0.03,
            1.2,
        ]
    )


def exact_identity(digest="0" * 64):
    return ExactOracleIdentity(
        protocol=EXACT_ORACLE_PROTOCOL,
        backend_id="grape_pc_mcu_controller_cpp/v1",
        implementation_language="C++",
        source_commit="0123456789abcdef",
        artifact_sha256=digest,
        capabilities=REQUIRED_ORACLE_CAPABILITIES,
    )


class AlternativeSmootherTests(unittest.TestCase):
    def config(self, samples=8, maximum=128):
        bootstrap = SmootherConfig(
            trajectory_sample_count=samples,
            seed=41,
            mocap_position_sigma=0.01,
            accelerometer_noise_sigma=0.05,
            mocap_nis_gate=1.0e6,
        )
        return FactorGraphSmootherConfig(
            bootstrap_config=bootstrap,
            mocap_position_sigma=0.01,
            max_dense_event_count=maximum,
        )

    def test_factor_graph_uses_common_contract_and_joint_samples(self):
        observations = trajectory_observations()
        posterior = BatchImuPreintegrationSmoother(self.config()).smooth(
            observations
        )
        self.assertIsInstance(posterior, TrajectoryPosterior)
        self.assertTrue(posterior.is_smoothed)
        self.assertEqual(posterior.sample_count, 8)
        self.assertIn("imu_preintegration_factor_graph", posterior.sampling_approximation)
        self.assertEqual(
            posterior.sample_position_world.shape,
            (8, posterior.timestamps.size, 3),
        )
        # The graph carries one sample_id through the complete dropout rather
        # than drawing independent time marginals.
        midpoint = np.searchsorted(posterior.timestamps, 1.0)
        cross_time_motion = (
            posterior.sample_position_world[:, midpoint, 0]
            - posterior.sample_position_world[:, 0, 0]
        )
        self.assertGreater(np.std(cross_time_motion), 0.0)
        truth = 0.2 * posterior.timestamps**2
        self.assertLess(
            float(np.sqrt(np.mean((posterior.position_world[:, 0] - truth) ** 2))),
            0.03,
        )

    def test_online_prefix_disables_batch_and_cannot_see_future(self):
        backend = BatchImuPreintegrationSmoother(self.config(samples=3))
        first = backend.smooth(
            trajectory_observations(False),
            online_prefix=True,
            cutoff=1.0,
        )
        changed_future = backend.smooth(
            trajectory_observations(True),
            online_prefix=True,
            cutoff=1.0,
        )
        self.assertFalse(first.is_smoothed)
        self.assertIn("batch_factor_graph_disabled", first.sampling_approximation)
        self.assertLessEqual(float(first.timestamps[-1]), 1.0)
        np.testing.assert_array_equal(first.timestamps, changed_future.timestamps)
        np.testing.assert_array_equal(
            first.position_world, changed_future.position_world
        )
        np.testing.assert_array_equal(
            first.sample_position_world, changed_future.sample_position_world
        )

    def test_dense_vertical_slice_refuses_unbounded_problem_size(self):
        backend = BatchImuPreintegrationSmoother(self.config(maximum=10))
        with self.assertRaisesRegex(ValueError, "at most 10 events"):
            backend.smooth(trajectory_observations())

    def test_factor_graph_does_not_resurrect_bootstrap_rejected_mocap(self):
        source = trajectory_observations()
        position = np.array(source.mocap_positions_world, copy=True)
        outlier_index = int(np.searchsorted(source.mocap_times, 1.5))
        position[outlier_index, 0] += 50.0
        observations = TrajectoryObservations(
            mocap_times=source.mocap_times,
            mocap_positions_world=position,
            mocap_quaternions_xyzw=source.mocap_quaternions_xyzw,
            imu_times=source.imu_times,
            accelerometer_body=source.accelerometer_body,
            gyro_body=source.gyro_body,
            mocap_valid_mask=source.mocap_valid_mask,
        )
        config = FactorGraphSmootherConfig(
            bootstrap_config=SmootherConfig(
                trajectory_sample_count=0,
                seed=41,
                mocap_position_sigma=0.01,
                accelerometer_noise_sigma=0.05,
                mocap_nis_gate=30.0,
            ),
            mocap_position_sigma=0.01,
        )
        posterior = BatchImuPreintegrationSmoother(config).smooth(observations)
        event_index = int(np.searchsorted(posterior.timestamps, 1.5))
        self.assertTrue(posterior.mocap_rejected[event_index])
        self.assertLess(posterior.position_world[event_index, 0], 2.0)


class AlternativeInferenceTests(unittest.TestCase):
    def test_pmmh_and_modular_smc_agree_on_shared_synthetic_model(self):
        model = LinearGaussianRandomWalkModel(
            initial_sigma=0.4,
            process_sigma=0.15,
            observation_sigma=0.25,
        )
        truth = 0.055
        _, observations = model.simulate(truth, count=28, seed=3)
        prior = BoxUniformPrior([-0.15], [0.22])
        transform = BoundedLogitTransform(prior.lower, prior.upper)
        pmmh = ParticleMarginalMetropolisHastings(
            model,
            prior,
            transform,
            ParticleMarginalMhConfig(
                iteration_count=700,
                burn_in=200,
                thin=2,
                particle_count=96,
                proposal_scale=0.5,
                seed=33,
            ),
        ).run(observations)
        smc = TemperedResampleMoveSmc(
            prior,
            transform,
            TemperedSmcConfig(
                particle_count=512,
                target_ess_fraction=0.75,
                resample_ess_fraction=0.45,
                mcmc_steps=2,
                proposal_scale=0.8,
                seed=33,
            ),
        ).run(lambda particles: model.exact_log_likelihood(particles, observations))
        comparison = compare_pmmh_with_modular_smc(pmmh, smc, tolerance=0.035)
        self.assertTrue(comparison.passed)
        self.assertLess(abs(float(pmmh.mean()[0]) - truth), 0.05)
        self.assertGreater(pmmh.acceptance_rate, 0.05)
        self.assertLess(pmmh.acceptance_rate, 0.9)

    def test_particle_likelihood_fails_when_every_particle_is_impossible(self):
        class ImpossibleModel(ParticleStateSpaceModel):
            parameter_dimension = 1

            def sample_initial(self, parameters, count, rng):
                return np.zeros((count, 1))

            def sample_transition(
                self, states, parameters, time_index, rng
            ):
                return states

            def observation_log_likelihood(
                self, observation, states, parameters, time_index
            ):
                return np.full(states.shape[0], -np.inf)

        with self.assertRaises(ParticleLikelihoodDegeneracy):
            bootstrap_particle_log_likelihood(
                ImpossibleModel(),
                np.zeros(2),
                np.zeros(1),
                32,
                np.random.default_rng(4),
            )

    def test_pmmh_configuration_rejects_invalid_burn_in(self):
        with self.assertRaisesRegex(ValueError, "burn_in"):
            ParticleMarginalMhConfig(iteration_count=10, burn_in=10)


class StructuredMechanicsTests(unittest.TestCase):
    def test_uncalibrated_scale_gauge_is_explicit_and_exactly_invariant(self):
        model = StructuredSixDofMechanicsResponse()
        parameters = StructuredMechanicsParameters(
            inertial_parameters=inertial_parameters(),
            actuator_wrench_scale=np.array([2.0, 2.2, 2.4, 0.7, 0.8, 0.9]),
            calibrated_wrench=False,
        )
        specific = np.array([0.3, -0.2, 9.5])
        omega = np.array([0.2, -0.1, 0.4])
        alpha = np.array([-0.3, 0.5, 0.1])
        baseline = model.predict_observation(
            parameters, specific, omega, alpha
        )
        gauge_equivalent = model.apply_global_gauge(parameters, 3.5)
        transformed = model.predict_observation(
            gauge_equivalent, specific, omega, alpha
        )
        np.testing.assert_allclose(transformed, baseline, atol=1.0e-13)
        report = model.gauge_report(parameters)
        self.assertEqual(report.gauge_dimension, 1)
        self.assertIn("absolute mass", report.forbidden_claims)
        self.assertEqual(report.local_null_direction.shape, (16,))

    def test_calibrated_wrench_removes_scale_gauge_but_keeps_rank_caveat(self):
        model = StructuredSixDofMechanicsResponse()
        parameters = StructuredMechanicsParameters(inertial_parameters())
        report = model.gauge_report(parameters)
        self.assertEqual(report.gauge_dimension, 0)
        self.assertEqual(report.calibration_status, "CALIBRATED_WRENCH")
        with self.assertRaisesRegex(ValueError, "no scale gauge"):
            model.apply_global_gauge(parameters, 2.0)
        predicted = model.predict_observation(
            parameters,
            [0.0, 0.0, 9.8],
            [0.1, 0.2, 0.3],
            [0.2, 0.0, -0.1],
        )
        likelihood = model.gaussian_log_likelihood(
            parameters,
            [0.0, 0.0, 9.8],
            [0.1, 0.2, 0.3],
            [0.2, 0.0, -0.1],
            predicted,
            np.ones(6),
        )
        self.assertTrue(np.isfinite(likelihood))

    def test_uncalibrated_mode_cannot_silently_assume_unit_scale(self):
        with self.assertRaisesRegex(ValueError, "requires actuator_wrench_scale"):
            StructuredMechanicsParameters(
                inertial_parameters(), calibrated_wrench=False
            )

    def test_local_rank_separates_structural_gauge_from_poor_excitation(self):
        model = StructuredSixDofMechanicsResponse()
        rng = np.random.default_rng(8)
        specific = rng.normal(size=(50, 3))
        omega = rng.normal(size=(50, 3))
        alpha = rng.normal(size=(50, 3))
        uncalibrated = StructuredMechanicsParameters(
            inertial_parameters(),
            actuator_wrench_scale=[2.0, 2.2, 2.4, 0.7, 0.8, 0.9],
            calibrated_wrench=False,
        )
        excited = model.local_identifiability(
            uncalibrated, specific, omega, alpha
        )
        self.assertEqual(excited.jacobian_rank, 15)
        self.assertEqual(excited.structural_gauge_dimension, 1)
        self.assertEqual(excited.excitation_nullity, 0)
        self.assertTrue(excited.identifiable_up_to_declared_gauge)

        calibrated = StructuredMechanicsParameters(inertial_parameters())
        unexcited = model.local_identifiability(
            calibrated,
            np.zeros((10, 3)),
            np.zeros((10, 3)),
            np.zeros((10, 3)),
        )
        self.assertEqual(unexcited.structural_gauge_dimension, 0)
        self.assertGreater(unexcited.excitation_nullity, 0)
        self.assertFalse(unexcited.identifiable_up_to_declared_gauge)


class ExactOracleAndCandidateGateTests(unittest.TestCase):
    def fixture(self, payload=None, source_bag_sha256="b" * 64):
        samples = 31
        base = np.linspace(-1.0, 1.0, samples)[:, None]
        continuous = {
            "pid_terms": np.hstack((base, base**2, -base)),
            "four_axis_command": np.hstack(
                (base, -base, base**2, base**3)
            ),
            "vectoring_force": np.hstack((base, 2.0 * base)),
            "pwm": np.hstack(
                (1000.0 + 100.0 * base, 1200.0 - 80.0 * base)
            ),
        }
        events = np.arange(samples) % 4
        request = {"run_id": "unit-test"} if payload is None else payload
        provenance = ExactOracleFixtureProvenance.create(
            source_bag_sha256=source_bag_sha256,
            source_topics=(
                "/debug/pose/pid",
                "/four_axes/command",
                "/target_vectoring_force",
                "/motor_pwms",
            ),
            interval_start_time_ns=1_000_000_000,
            interval_end_time_ns=2_000_000_000,
            frame_conventions={"controller": "body_flu", "world": "enu"},
            unit_conventions={"angles": "rad", "pwm": "timer_count"},
            motor_order=("motor1", "motor2", "motor3", "motor4"),
            request_payload=request,
            continuous=continuous,
            events=events,
            extraction_config_sha256="c" * 64,
            source_commit="fixture-source-commit",
        )
        return ExactOracleConformanceFixture(
            continuous=continuous,
            events=events,
            provenance=provenance,
        )

    def test_verified_external_oracle_passes_all_factual_channels(self):
        identity = exact_identity()
        fixture = self.fixture()

        class ExactOracle:
            is_exact = True

            def __init__(self):
                self.identity = identity

            def replay(self, payload):
                self.payload = payload
                return ExactOracleReplayOutput(
                    identity=self.identity,
                    continuous=fixture.continuous,
                    events=fixture.events,
                )

        report = evaluate_exact_oracle_conformance(
            ExactOracle(), {"run_id": "unit-test"}, fixture
        )
        self.assertTrue(report.passed)
        self.assertEqual(report.status, "PASS")
        self.assertEqual(
            set(report.channel_metrics),
            {
                "pid_terms",
                "four_axis_command",
                "vectoring_force",
                "pwm",
            },
        )
        self.assertEqual(
            report.fixture_provenance.source_bag_sha256, "b" * 64
        )
        self.assertEqual(
            report.request_payload_sha256,
            fixture.provenance.fixture_input_payload_sha256,
        )

    def test_fixture_provenance_rejects_tampering_and_wrong_payload(self):
        fixture = self.fixture()
        changed = dict(fixture.continuous)
        changed["pid_terms"] = np.array(changed["pid_terms"], copy=True)
        changed["pid_terms"][0, 0] += 1.0
        with self.assertRaisesRegex(ValueError, "do not match"):
            ExactOracleConformanceFixture(
                continuous=changed,
                events=fixture.events,
                provenance=fixture.provenance,
            )

        class ExactOracle:
            is_exact = True
            identity = exact_identity()

            def replay(self, payload):
                raise AssertionError("mismatched payload must not execute")

        report = evaluate_exact_oracle_conformance(
            ExactOracle(), {"run_id": "different"}, fixture
        )
        self.assertFalse(report.passed)
        self.assertEqual(report.status, "FIXTURE_BINDING_REJECTED")

        fixture.continuous["pid_terms"] = np.ones((31, 3))
        with self.assertRaisesRegex(ValueError, "fixture data was mutated"):
            evaluate_exact_oracle_conformance(
                None, {"run_id": "unit-test"}, fixture
            )

        provenance_fixture = self.fixture()
        provenance_fixture.provenance.frame_conventions[
            "controller"
        ] = "mutated_frame"
        with self.assertRaisesRegex(ValueError, "provenance was mutated"):
            evaluate_exact_oracle_conformance(
                None, {"run_id": "unit-test"}, provenance_fixture
            )

    def test_missing_or_surrogate_oracle_fails_closed(self):
        fixture = self.fixture()
        missing = evaluate_exact_oracle_conformance(None, {}, fixture)
        self.assertFalse(missing.passed)
        self.assertEqual(missing.status, "ORACLE_UNAVAILABLE")

        class Surrogate:
            is_exact = False

            def replay(self, payload):
                raise AssertionError("a surrogate must not be invoked")

        surrogate = evaluate_exact_oracle_conformance(
            Surrogate(), {}, fixture
        )
        self.assertFalse(surrogate.passed)
        self.assertEqual(surrogate.status, "IDENTITY_REJECTED")

    def test_identity_rejects_python_or_surrogate_claims(self):
        values = dict(
            protocol=EXACT_ORACLE_PROTOCOL,
            backend_id="python_surrogate",
            implementation_language="Python",
            source_commit="abc",
            artifact_sha256="0" * 64,
            capabilities=REQUIRED_ORACLE_CAPABILITIES,
        )
        with self.assertRaisesRegex(ValueError, "surrogate"):
            ExactOracleIdentity(**values)

    def test_subprocess_adapter_checks_artifact_before_handshake(self):
        with self.assertRaises(ExactOracleUnavailable):
            SubprocessExactControllerOracle(
                ["/definitely/not/a/controller-oracle"],
                exact_identity(),
            )
        executable = "/bin/true"
        if os.path.isfile(executable):
            with open(executable, "rb") as stream:
                digest = hashlib.sha256(stream.read()).hexdigest()
            with self.assertRaises(ExactOracleProtocolError):
                SubprocessExactControllerOracle(
                    [executable],
                    exact_identity(digest=digest),
                )

    def test_conditional_gp_and_bayessim_remain_pruned_without_evidence(self):
        gp = evaluate_conditional_candidate("sparse_gp_residual", {})
        self.assertEqual(gp.decision, "PRUNE")
        self.assertIn(
            "heldout_parametric_residual_structure",
            gp.missing_prerequisites,
        )
        bayessim = evaluate_conditional_candidate(
            "likelihood_free_bayessim",
            {
                "likelihood_model_invalidated": True,
                "black_box_simulator_heldout_validated": True,
                "exact_controller_oracle_passed": False,
            },
        )
        self.assertEqual(bayessim.decision, "PRUNE")
        self.assertEqual(
            bayessim.missing_prerequisites,
            ("exact_controller_oracle_passed",),
        )
        start = evaluate_conditional_candidate(
            "sparse_gp_residual",
            {
                "heldout_parametric_residual_structure": True,
                "support_variance_growth_validated": True,
                "no_free_extrapolation": True,
            },
        )
        self.assertEqual(start.decision, "START")


if __name__ == "__main__":
    unittest.main()
