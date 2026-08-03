from pathlib import Path
import tempfile
import unittest

import numpy as np

from grape_param_estim.mode_validation import (
    ActuatorWiringMeasurement,
    ModeValidationResult,
    NOMINAL_MODE_ID,
    SWAPPED_MODE_ID,
    condition_on_actuator_wiring,
    run_mode_validation_experiment,
    save_mode_validation,
)
from grape_param_estim.strong_constraint import CONTROL_DIMENSION


class ModeValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.forward = run_mode_validation_experiment(
            truth_mode_id=SWAPPED_MODE_ID,
            mode_order=(NOMINAL_MODE_ID, SWAPPED_MODE_ID),
            duration=0.3,
            time_step=0.04,
            ensemble_size=38,
            maximum_iterations=1,
            seed=19,
        )
        # Reorder the completed independent components without altering or
        # recomputing either raw posterior law.
        cls.reverse = ModeValidationResult(
            synthetic=cls.forward.synthetic,
            truth_mode_id=SWAPPED_MODE_ID,
            mode_posteriors=tuple(reversed(cls.forward.mode_posteriors)),
            prior_mode_probabilities=(
                cls.forward.prior_mode_probabilities[::-1]
            ),
            pose_mode_probabilities=(
                cls.forward.pose_mode_probabilities[::-1]
            ),
        )
        cls.nominal_truth = run_mode_validation_experiment(
            truth_mode_id=NOMINAL_MODE_ID,
            mode_order=(NOMINAL_MODE_ID, SWAPPED_MODE_ID),
            duration=0.3,
            time_step=0.04,
            ensemble_size=38,
            maximum_iterations=1,
            seed=19,
        )

    def test_pose_laplace_weight_argmax_is_the_truth_mode(self):
        result = self.forward
        selected = result.mode_ids[
            int(np.argmax(result.pose_mode_probabilities))
        ]
        self.assertEqual(selected, SWAPPED_MODE_ID)
        truth_index = result.mode_ids.index(SWAPPED_MODE_ID)
        self.assertGreater(result.pose_mode_probabilities[truth_index], 0.99)
        self.assertAlmostEqual(
            float(np.sum(result.pose_mode_probabilities)), 1.0
        )
        self.assertIsNot(
            result.for_mode(NOMINAL_MODE_ID).problem,
            result.for_mode(SWAPPED_MODE_ID).problem,
        )
        self.assertIsNot(
            result.for_mode(NOMINAL_MODE_ID).posterior,
            result.for_mode(SWAPPED_MODE_ID).posterior,
        )
        for value in result.mode_posteriors:
            self.assertEqual(value.evidence.regression_rank, CONTROL_DIMENSION)
            np.testing.assert_array_equal(
                value.member_ids, np.arange(38, dtype=np.int64)
            )

    def test_pose_weights_follow_either_synthetic_truth_mode(self):
        for result in (self.forward, self.nominal_truth):
            truth_index = result.mode_ids.index(result.truth_mode_id)
            self.assertEqual(
                result.mode_ids[
                    int(np.argmax(result.pose_mode_probabilities))
                ],
                result.truth_mode_id,
            )
            self.assertGreater(
                result.pose_mode_probabilities[truth_index], 0.99
            )

    def test_mode_experiment_intentionally_shares_one_actuator_model(self):
        synthetic = self.forward.synthetic
        self.assertIs(
            synthetic.nominal_actuator_parameters,
            synthetic.truth_actuator_parameters,
        )
        self.assertEqual(synthetic.nominal_actuator_parameters.delay, 0.0)
        for value in self.forward.mode_posteriors:
            self.assertIs(
                value.problem.actuator_parameters,
                synthetic.nominal_actuator_parameters,
            )

    def test_mode_order_does_not_change_weights_or_raw_members(self):
        forward_weights = dict(
            zip(self.forward.mode_ids, self.forward.pose_mode_probabilities)
        )
        reverse_weights = dict(
            zip(self.reverse.mode_ids, self.reverse.pose_mode_probabilities)
        )
        for mode_id in (NOMINAL_MODE_ID, SWAPPED_MODE_ID):
            self.assertEqual(
                forward_weights[mode_id], reverse_weights[mode_id]
            )
            first = self.forward.for_mode(mode_id).posterior
            second = self.reverse.for_mode(mode_id).posterior
            np.testing.assert_array_equal(
                first.prior_control_ensemble,
                second.prior_control_ensemble,
            )
            np.testing.assert_array_equal(
                first.control_ensemble, second.control_ensemble
            )
            np.testing.assert_array_equal(
                first.correction_translation, second.correction_translation
            )

    def test_wiring_conditioning_is_decisive_and_nonmutating(self):
        raw_before = {}
        for value in self.forward.mode_posteriors:
            posterior = value.posterior
            raw_before[value.mode.mode_id] = (
                posterior.control_ensemble.tobytes(),
                posterior.parameter_ensemble.coordinates.tobytes(),
                posterior.correction_translation.tobytes(),
                posterior.correction_rotation_vector.tobytes(),
            )
        conditioned = condition_on_actuator_wiring(
            self.forward,
            ActuatorWiringMeasurement(
                channel_to_rotor=np.asarray((1, 0, 2, 3)),
                correctness_probability=0.995,
            ),
        )
        truth_index = self.forward.mode_ids.index(SWAPPED_MODE_ID)
        self.assertEqual(conditioned.selected_mode_id, SWAPPED_MODE_ID)
        self.assertGreater(
            conditioned.conditioned_mode_probabilities[truth_index], 0.99
        )
        self.assertIs(
            conditioned.selected_posterior,
            self.forward.for_mode(SWAPPED_MODE_ID),
        )
        for value in self.forward.mode_posteriors:
            posterior = value.posterior
            raw_after = (
                posterior.control_ensemble.tobytes(),
                posterior.parameter_ensemble.coordinates.tobytes(),
                posterior.correction_translation.tobytes(),
                posterior.correction_rotation_vector.tobytes(),
            )
            self.assertEqual(raw_before[value.mode.mode_id], raw_after)

    def test_artifact_is_pickle_free_and_preserves_mode_member_pairs(self):
        measurement = ActuatorWiringMeasurement(
            np.asarray((1, 0, 2, 3)), 0.995
        )
        conditioned = condition_on_actuator_wiring(self.forward, measurement)
        with tempfile.TemporaryDirectory() as directory:
            destination = save_mode_validation(
                str(Path(directory) / "mode_validation.npz"),
                self.forward,
                conditioned,
            )
            with np.load(str(destination), allow_pickle=False) as artifact:
                self.assertEqual(
                    str(artifact["schema"][0]),
                    "grape-param-estim/mode-validation/v1",
                )
                self.assertEqual(
                    artifact["posterior_control_ensemble"].shape,
                    (2, 38, CONTROL_DIMENSION),
                )
                self.assertEqual(
                    artifact["posterior_linear_velocity"].shape[:2],
                    (2, 38),
                )
                self.assertEqual(
                    str(artifact["selected_mode_id"][0]), SWAPPED_MODE_ID
                )
                self.assertEqual(
                    artifact["actuator_wiring_measurement"].tolist(),
                    [1, 0, 2, 3],
                )
                for prefix in ("nominal", "truth"):
                    for field in (
                        "commanded_thrust",
                        "commanded_gimbal_angle",
                        "actuator_thrust",
                        "actuator_gimbal_angle",
                        "body_wrench",
                    ):
                        self.assertIn(
                            "{}_{}".format(prefix, field), artifact.files
                        )
                pairs = set(
                    zip(
                        artifact["member_mode_id"].tolist(),
                        artifact["member_id"].tolist(),
                    )
                )
                self.assertEqual(len(pairs), 2 * 38)
                for mode_id in self.forward.mode_ids:
                    self.assertEqual(
                        {member for mode, member in pairs if mode == mode_id},
                        set(range(38)),
                    )
                truth_index = self.forward.mode_ids.index(SWAPPED_MODE_ID)
                self.assertGreater(
                    artifact["conditioned_mode_probability"][truth_index],
                    0.99,
                )


if __name__ == "__main__":
    unittest.main()
