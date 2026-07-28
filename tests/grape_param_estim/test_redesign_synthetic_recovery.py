from types import SimpleNamespace
import unittest

import numpy as np

from grape_param_estim.episode import stable_hash
from grape_param_estim.forward import (
    OpenLoopForwardModel,
    RecordedCommandSeries,
)
from grape_param_estim.inference import (
    BatchPlantInference,
    BoundedLogitTransform,
    IndependentBoundedPrior,
    PriorDimension,
    TemperedSmcConfig,
    local_identifiability,
)
from grape_param_estim.plant import (
    ActuatorCalibrationIdentity,
    EpisodeNuisance,
    PlantHypothesis,
    effective_identifiable_quantities,
)
from grape_param_estim.plant.parameters import (
    ACTUATOR_PARAMETER_NAMES,
    CALIBRATED_RIGID_BODY_MODEL_ID,
    CALIBRATED_RIGID_BODY_PARAMETER_NAMES,
    EFFECTIVE_CLOSED_LOOP_MODEL_ID,
    EFFECTIVE_CLOSED_LOOP_PARAMETER_NAMES,
)


def _initial_state():
    return np.asarray(
        [
            0.0,
            0.0,
            5.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
        ]
    )


def _nuisance():
    return EpisodeNuisance(
        initial_plant_state=_initial_state(),
        initial_actuator_state=np.zeros(0),
        disturbance_parameters=np.zeros(6),
        sensor_bias=np.zeros(6),
    )


def _grids(end=1.0, count=11):
    times = np.linspace(0.0, float(end), int(count))
    return SimpleNamespace(
        plant_integration_grid=times,
        likelihood_grid=times,
        controller_tick_grid=np.zeros(0),
    )


def _actuator_values(common_thrust_scale=1.0):
    return np.asarray(
        [
            common_thrust_scale,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            100.0,
        ]
    )


def _effective_hypothesis(force_bias_z):
    plant = np.asarray(
        [
            1.0,
            1.0,
            1.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            float(force_bias_z),
            0.0,
            0.0,
        ]
    )
    actuator = _actuator_values()
    return PlantHypothesis(
        model_id=EFFECTIVE_CLOSED_LOOP_MODEL_ID,
        plant_parameters=plant,
        actuator_parameters=actuator,
        disturbance_parameters=np.zeros(0),
        plant_parameter_names=EFFECTIVE_CLOSED_LOOP_PARAMETER_NAMES,
        actuator_parameter_names=ACTUATOR_PARAMETER_NAMES,
        derived_quantities=effective_identifiable_quantities(
            plant, actuator
        ),
    )


class CalibratedRecoveryFixture:
    truth = np.asarray([2.0, 0.8])
    calibration = ActuatorCalibrationIdentity(
        artifact_sha256="a" * 64,
        actuator_model_id="synthetic_calibrated_actuator/v1",
    )

    def __init__(self):
        self.commands = RecordedCommandSeries(
            timestamps=np.asarray([0.0, 1.0]),
            base_thrust=np.zeros((2, 4)),
            gimbal_angle=np.zeros((2, 4)),
            generalized_wrench=np.tile(
                np.asarray([4.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
                (2, 1),
            ),
        )
        self.grids = _grids()
        self.nuisance = _nuisance()
        self.forward = OpenLoopForwardModel(
            actuator_calibration_identity=self.calibration
        )
        truth = self.rollout(self.truth)
        self.observed_x = truth.positions[:, 0]
        self.observed_omega_x = np.asarray(
            [
                item.angular_velocity_body[0]
                for item in truth.plant_states
            ]
        )

    @staticmethod
    def hypothesis(values):
        mass, inertia_xx = np.asarray(values, dtype=float)
        plant = np.asarray(
            [
                mass,
                0.0,
                0.0,
                0.0,
                inertia_xx,
                0.0,
                0.0,
                1.0,
                0.0,
                1.1,
            ]
        )
        return PlantHypothesis(
            model_id=CALIBRATED_RIGID_BODY_MODEL_ID,
            plant_parameters=plant,
            actuator_parameters=_actuator_values(),
            disturbance_parameters=np.zeros(0),
            plant_parameter_names=(
                CALIBRATED_RIGID_BODY_PARAMETER_NAMES
            ),
            actuator_parameter_names=ACTUATOR_PARAMETER_NAMES,
            derived_quantities={
                "mass_kg": float(mass),
                "inertia_xx_kg_m2": float(inertia_xx),
            },
        )

    def rollout(self, values):
        return self.forward.run(
            self.commands,
            self.hypothesis(values),
            self.nuisance,
            self.grids,
        )

    def log_likelihood(self, particles):
        values = np.asarray(particles, dtype=float)
        output = np.empty(values.shape[0])
        for index, particle in enumerate(values):
            rollout = self.rollout(particle)
            predicted_omega = np.asarray(
                [
                    item.angular_velocity_body[0]
                    for item in rollout.plant_states
                ]
            )
            position_residual = (
                rollout.positions[:, 0] - self.observed_x
            ) / 0.02
            angular_residual = (
                predicted_omega - self.observed_omega_x
            ) / 0.03
            output[index] = -0.5 * float(
                np.dot(position_residual, position_residual)
                + np.dot(angular_residual, angular_residual)
            )
        return output

    def infer(self, seed=19):
        prior = IndependentBoundedPrior(
            (
                PriorDimension(
                    "mass", "bounded_uniform", 1.4, 2.6
                ),
                PriorDimension(
                    "inertia_xx", "bounded_uniform", 0.55, 1.05
                ),
            )
        )
        bounds_identity = stable_hash(
            {
                "names": prior.names,
                "lower": prior.lower,
                "upper": prior.upper,
            }
        )
        return BatchPlantInference(
            prior=prior,
            transform=BoundedLogitTransform(
                prior.lower, prior.upper
            ),
            hypothesis_builder=self.hypothesis,
            log_likelihood=self.log_likelihood,
            config=TemperedSmcConfig(
                particle_count=64,
                target_ess_fraction=0.70,
                resample_ess_fraction=0.50,
                mcmc_steps=2,
                proposal_scale=0.55,
                seed=seed,
            ),
        ).run(
            model_id=CALIBRATED_RIGID_BODY_MODEL_ID,
            prior_id="synthetic_calibrated_prior/{}".format(
                bounds_identity
            ),
            likelihood_id="synthetic_calibrated_rollout/v1",
            controller_snapshot_id="b" * 64,
            provenance={
                "seed": seed,
                "prior_bounds_sha256": bounds_identity,
                "actuator_calibration_sha256": (
                    self.calibration.artifact_sha256
                ),
            },
        )


def _analytic_bias_posterior(
    *,
    lower=-2.0,
    upper=2.0,
    seed=31,
    bimodal=False,
    particle_count=64,
):
    prior = IndependentBoundedPrior(
        (
            PriorDimension(
                "force_bias_z",
                "bounded_uniform",
                lower,
                upper,
            ),
        )
    )
    bounds_identity = stable_hash(
        {
            "names": prior.names,
            "lower": prior.lower,
            "upper": prior.upper,
        }
    )

    def likelihood(particles):
        value = np.asarray(particles, dtype=float)[:, 0]
        if bimodal:
            left = -0.5 * ((value + 1.0) / 0.16) ** 2
            right = -0.5 * ((value - 1.0) / 0.16) ** 2
            maximum = np.maximum(left, right)
            return maximum + np.log(
                np.exp(left - maximum) + np.exp(right - maximum)
            )
        return -0.5 * ((value - 0.45) / 0.20) ** 2

    return BatchPlantInference(
        prior=prior,
        transform=BoundedLogitTransform(prior.lower, prior.upper),
        hypothesis_builder=lambda value: _effective_hypothesis(
            np.asarray(value, dtype=float)[0]
        ),
        log_likelihood=likelihood,
        config=TemperedSmcConfig(
            particle_count=particle_count,
            target_ess_fraction=0.70,
            resample_ess_fraction=0.50,
            mcmc_steps=2,
            proposal_scale=0.65,
            seed=seed,
        ),
    ).run(
        model_id=EFFECTIVE_CLOSED_LOOP_MODEL_ID,
        prior_id="synthetic_bias_prior/{}".format(bounds_identity),
        likelihood_id=(
            "synthetic_bimodal/v1"
            if bimodal
            else "synthetic_unimodal/v1"
        ),
        controller_snapshot_id="c" * 64,
        provenance={
            "seed": seed,
            "prior_bounds_sha256": bounds_identity,
        },
    )


class SyntheticRecoveryAcceptanceTests(unittest.TestCase):
    def test_calibrated_forward_recovers_mass_and_inertia(self):
        fixture = CalibratedRecoveryFixture()
        posterior = fixture.infer()
        mean = posterior.mean
        self.assertEqual(
            posterior.model_id, CALIBRATED_RIGID_BODY_MODEL_ID
        )
        self.assertTrue(
            all(
                item.model_id == CALIBRATED_RIGID_BODY_MODEL_ID
                for item in posterior.particles
            )
        )
        self.assertEqual(
            posterior.provenance["actuator_calibration_sha256"],
            fixture.calibration.artifact_sha256,
        )
        self.assertAlmostEqual(mean[0], fixture.truth[0], delta=0.12)
        self.assertAlmostEqual(mean[4], fixture.truth[1], delta=0.07)

    def test_uncalibrated_scale_gauge_is_invariant_and_reported(self):
        first_plant = np.asarray(
            [2.0, 4.0, 5.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        )
        second_plant = np.array(first_plant, copy=True)
        second_plant[:4] *= 0.5
        first_actuator = _actuator_values(3.0)
        second_actuator = _actuator_values(6.0)
        first_quantities = effective_identifiable_quantities(
            first_plant, first_actuator
        )
        second_quantities = effective_identifiable_quantities(
            second_plant, second_actuator
        )
        self.assertEqual(
            dict(first_quantities), dict(second_quantities)
        )

        jacobian = np.asarray([[3.0, 2.0]])
        report = local_identifiability(
            jacobian,
            (
                "specific_thrust_authority",
                "common_thrust_scale",
            ),
            EFFECTIVE_CLOSED_LOOP_MODEL_ID,
            structural_gauge_dimension=1,
        )
        self.assertEqual(report.jacobian_rank, 1)
        self.assertEqual(report.structural_gauge_dimension, 1)
        self.assertEqual(report.excitation_nullity, 0)
        self.assertEqual(report.null_directions.shape, (1, 2))
        self.assertAlmostEqual(
            float(jacobian @ report.null_directions[0]), 0.0
        )

    def test_weighted_posterior_preserves_both_synthetic_modes(self):
        posterior = _analytic_bias_posterior(
            bimodal=True, particle_count=128, seed=23
        )
        bias = posterior.raw_parameters[:, 8]
        left_mass = float(np.sum(posterior.weights[bias < -0.5]))
        right_mass = float(np.sum(posterior.weights[bias > 0.5]))
        self.assertGreater(left_mass, 0.20)
        self.assertGreater(right_mass, 0.20)
        self.assertGreater(posterior.covariance[8, 8], 0.50)
        diagnostic = posterior.multimodality_diagnostic()
        self.assertTrue(diagnostic["any_multimodal"])
        self.assertGreaterEqual(
            diagnostic["per_parameter_mode_count"]["force_bias_z"],
            2,
        )

    def test_same_seed_has_same_hash_and_prior_bounds_change_identity(self):
        first = _analytic_bias_posterior(seed=47)
        repeated = _analytic_bias_posterior(seed=47)
        self.assertEqual(first.content_sha256, repeated.content_sha256)
        np.testing.assert_array_equal(
            first.raw_parameters, repeated.raw_parameters
        )
        np.testing.assert_array_equal(first.weights, repeated.weights)

        changed = _analytic_bias_posterior(
            lower=-1.0, upper=1.0, seed=47
        )
        self.assertNotEqual(first.prior_id, changed.prior_id)
        self.assertNotEqual(
            first.provenance["prior_bounds_sha256"],
            changed.provenance["prior_bounds_sha256"],
        )
        self.assertNotEqual(
            first.content_sha256, changed.content_sha256
        )
        self.assertFalse(
            np.array_equal(
                first.raw_parameters, changed.raw_parameters
            )
        )


if __name__ == "__main__":
    unittest.main()
