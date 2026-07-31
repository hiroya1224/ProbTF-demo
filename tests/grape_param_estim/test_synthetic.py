from dataclasses import fields
from pathlib import Path
import tempfile
import unittest

import numpy as np

from grape_param_estim.synthetic import (
    generate_pose_observations,
    run_synthetic_experiment,
    save_experiment,
)
from grape_param_estim.system import PoseObservations
from grape_param_estim.system import ActuatorParameters


class SyntheticExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.experiment = run_synthetic_experiment(
            duration=2.0, time_step=0.02, seed=19
        )

    def test_observation_contract_is_pose_only(self):
        names = {item.name for item in fields(PoseObservations)}
        self.assertEqual(
            names,
            {
                "times",
                "position",
                "orientation_xyzw",
                "translation_covariance",
                "rotation_covariance",
            },
        )
        for forbidden in (
            "linear_velocity",
            "angular_velocity",
            "acceleration",
            "imu",
            "command",
        ):
            self.assertFalse(hasattr(self.experiment.observations, forbidden))

    def test_pose_noise_is_reproducible_and_does_not_reset_forecast(self):
        truth_before = self.experiment.truth.position.copy()
        first = generate_pose_observations(
            self.experiment.truth, 0.01, 0.005, seed=11
        )
        repeat = generate_pose_observations(
            self.experiment.truth, 0.01, 0.005, seed=11
        )
        other = generate_pose_observations(
            self.experiment.truth, 0.01, 0.005, seed=12
        )
        np.testing.assert_array_equal(first.position, repeat.position)
        np.testing.assert_array_equal(
            first.orientation_xyzw, repeat.orientation_xyzw
        )
        self.assertFalse(np.array_equal(first.position, other.position))
        # Observation generation happens after the continuous forecast and
        # cannot feed/reset its latent state.
        np.testing.assert_array_equal(
            self.experiment.truth.position,
            truth_before,
        )

    def test_reference_and_real_episode_exercise_full_six_dof(self):
        reference_position = np.asarray(
            [item.position for item in self.experiment.references]
        )
        reference_rpy = np.asarray(
            [item.rpy for item in self.experiment.references]
        )
        self.assertTrue(np.all(np.ptp(reference_position, axis=0) > 0.01))
        self.assertTrue(np.all(np.ptp(reference_rpy, axis=0) > 0.01))
        self.assertTrue(np.all(np.isfinite(self.experiment.truth.position)))
        self.assertTrue(
            np.all(np.isfinite(self.experiment.truth.angular_velocity))
        )
        self.assertGreater(
            np.max(
                np.linalg.norm(
                    self.experiment.correction_translation, axis=1
                )
            ),
            0.01,
        )
        self.assertGreater(
            np.max(
                np.linalg.norm(
                    self.experiment.correction_rotation_vector, axis=1
                )
            ),
            np.deg2rad(1.0),
        )

    def test_default_episode_does_not_depend_on_hardware_saturation(self):
        limits = ActuatorParameters()
        self.assertGreater(
            np.min(self.experiment.truth.commanded_thrust),
            limits.minimum_thrust,
        )
        self.assertLess(
            np.max(self.experiment.truth.commanded_thrust),
            limits.maximum_thrust,
        )
        self.assertLess(
            np.max(np.abs(self.experiment.truth.commanded_gimbal_angle)),
            limits.maximum_gimbal_angle,
        )

    def test_phase1_npz_has_no_pickle_or_estimator_result(self):
        with tempfile.TemporaryDirectory() as directory:
            path = save_experiment(
                str(Path(directory) / "phase1.npz"), self.experiment
            )
            with np.load(str(path), allow_pickle=False) as result:
                self.assertEqual(
                    str(result["schema"][0]),
                    "grape-weak-constraint/phase1",
                )
                self.assertIn("nominal_position", result.files)
                self.assertIn("truth_position", result.files)
                self.assertIn("observed_position", result.files)
                self.assertIn("correction_translation", result.files)
                self.assertIn("controller_inertia", result.files)
                self.assertIn("truth_force_effectiveness", result.files)
                self.assertNotIn("particles", result.files)
                self.assertNotIn("weights", result.files)


if __name__ == "__main__":
    unittest.main()
