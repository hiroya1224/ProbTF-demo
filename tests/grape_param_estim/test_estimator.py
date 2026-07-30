import tempfile
import unittest
from pathlib import Path

import numpy as np

from grape_param_estim.data import AnalysisData
from grape_param_estim.estimator import (
    LikelihoodWeights,
    PARAMETER_NAMES,
    estimate_parameters,
    load_result,
    relative_transform_from_nominal,
    save_result,
    weighted_quantile,
)
from grape_param_estim.model import (
    GrapeRigidBodyModel,
    RigidBodyParameters,
    replay_segments,
)
from test_model import make_analysis, nominal_parameters


PRIOR = {name: (0.55, 1.65) for name in PARAMETER_NAMES}


def generated_analysis(scales) -> AnalysisData:
    source = make_analysis()
    model = GrapeRigidBodyModel(maximum_time_step=0.02)
    segment = next(source.segments())[1]
    generated = model.simulate_segment(
        source, nominal_parameters(**scales), segment
    )
    return AnalysisData(
        bag_path="/tmp/generated.bag",
        start_time=source.start_time,
        end_time=source.end_time,
        segment_duration=source.segment_duration,
        times=source.times,
        position=generated.position,
        orientation_xyzw=generated.orientation_xyzw,
        linear_velocity=generated.linear_velocity,
        angular_velocity=generated.angular_velocity,
        specific_force=source.specific_force,
        base_thrust=source.base_thrust,
        gimbal_target_angle=source.gimbal_target_angle,
        gimbal_measured_angle=source.gimbal_measured_angle,
        flight_state=source.flight_state,
        segment_id=source.segment_id,
    )


def weighted_correlation(x_values, y_values, weights):
    mean_x = np.sum(weights * x_values)
    mean_y = np.sum(weights * y_values)
    delta_x = x_values - mean_x
    delta_y = y_values - mean_y
    covariance = np.sum(weights * delta_x * delta_y)
    variance_x = np.sum(weights * delta_x**2)
    variance_y = np.sum(weights * delta_y**2)
    return covariance / np.sqrt(variance_x * variance_y)


class WeightedParticleEstimatorTest(unittest.TestCase):
    def test_nominal_replay_pushes_forward_to_identity_transform(self):
        analysis = make_analysis()
        replay = GrapeRigidBodyModel(maximum_time_step=0.02)
        replay_result = replay_segments(
            analysis, replay, nominal_parameters()
        )

        translation, rotation = relative_transform_from_nominal(
            replay_result, replay_result
        )

        np.testing.assert_allclose(translation, 0.0, atol=1.0e-12)
        np.testing.assert_allclose(rotation, 0.0, atol=1.0e-12)

    def test_weighted_quantile_preserves_trailing_dimensions(self):
        values = np.asarray(((0.0, 10.0), (1.0, 20.0), (2.0, 30.0)))
        weights = np.asarray((0.1, 0.2, 0.7))

        quantiles = weighted_quantile(values, weights, (0.5, 0.95))

        self.assertEqual(quantiles.shape, (2, 2))
        np.testing.assert_allclose(quantiles[0], (1.28571429, 22.85714286))
        np.testing.assert_allclose(quantiles[1], (1.92857143, 29.28571429))

    def test_synthetic_posterior_recovers_ratios_and_keeps_ridges(self):
        truth = {
            "mass_scale": 1.20,
            "force_scale": 0.90,
            "inertia_scale": 1.40,
            "torque_scale": 0.70,
        }
        analysis = generated_analysis(truth)

        result = estimate_parameters(
            analyses=(analysis,),
            nominal_parameters=nominal_parameters(),
            prior_bounds=PRIOR,
            particle_count=96,
            likelihood_weights=LikelihoodWeights(
                translation=180.0, rotation=300.0
            ),
            seed=7,
            resample_ess_fraction=0.10,
            jitter_fraction=0.15,
            model=GrapeRigidBodyModel(maximum_time_step=0.02),
        )

        particles = result.posterior.particles
        weights = result.posterior.weights
        force_mass = particles[:, 1] / particles[:, 0]
        torque_inertia = particles[:, 3] / particles[:, 2]
        ratio_median = weighted_quantile(
            np.column_stack((force_mass, torque_inertia)),
            weights,
            (0.5,),
        )[0]
        np.testing.assert_allclose(ratio_median, (0.75, 0.5), atol=0.12)
        self.assertGreater(
            weighted_correlation(
                particles[:, 0], particles[:, 1], weights
            ),
            0.45,
        )
        self.assertGreater(
            weighted_correlation(
                particles[:, 2], particles[:, 3], weights
            ),
            0.45,
        )
        self.assertEqual(
            result.bags[0].delta_translation.shape,
            (96, analysis.times.size, 3),
        )

    def test_multiple_bags_add_likelihood_and_result_round_trips(self):
        analysis = generated_analysis(
            {
                "mass_scale": 1.1,
                "force_scale": 0.9,
                "inertia_scale": 1.2,
                "torque_scale": 0.8,
            }
        )
        common = dict(
            nominal_parameters=nominal_parameters(),
            prior_bounds=PRIOR,
            particle_count=8,
            likelihood_weights=LikelihoodWeights(
                translation=10.0, rotation=2.0
            ),
            seed=11,
            resample_ess_fraction=0.0,
            model=GrapeRigidBodyModel(maximum_time_step=0.02),
        )

        single = estimate_parameters(analyses=(analysis,), **common)
        multiple = estimate_parameters(
            analyses=(analysis, analysis), **common
        )

        np.testing.assert_allclose(
            multiple.posterior.particles, single.posterior.particles
        )
        np.testing.assert_allclose(
            multiple.posterior.log_likelihood,
            2.0 * single.posterior.log_likelihood,
            atol=1.0e-12,
        )
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "result.npz"
            save_result(str(destination), multiple)
            loaded = load_result(str(destination))

        np.testing.assert_allclose(
            loaded["particles"], multiple.posterior.particles
        )
        self.assertEqual(loaded["bag_paths"].shape, (2,))
        self.assertIn("bag_1_delta_rotation_vector", loaded)


if __name__ == "__main__":
    unittest.main()
