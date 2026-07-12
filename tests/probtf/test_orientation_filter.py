import unittest

import numpy as np

from probtf.bingham import bingham_mode, canonical_bingham_parameter
from probtf_estimators.orientation_imu import (
    OrientationBinghamFilter,
    OrientationEvidence,
    delta_quaternion_second_moment,
    gravity_bingham_evidence,
    magnetic_bingham_evidence,
    predict_orientation_bingham,
    vector_alignment_bingham_evidence,
)


def _same_rotation(first, second, tolerance=1e-2):
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    first /= np.linalg.norm(first)
    second /= np.linalg.norm(second)
    return abs(float(first @ second)) >= 1.0 - tolerance


class DeltaQuaternionMomentTest(unittest.TestCase):
    def test_zero_rate_and_zero_dt_are_deterministic_identity(self):
        expected = np.diag([1.0, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(
            delta_quaternion_second_moment([0.0, 0.0, 0.0], 0.2),
            expected,
        )
        np.testing.assert_allclose(
            delta_quaternion_second_moment(
                [2.0, -1.0, 0.5],
                0.0,
                angular_velocity_covariance=np.eye(3),
            ),
            expected,
        )

    def test_deterministic_nonzero_rate_matches_quaternion_exponential(self):
        moment = delta_quaternion_second_moment([0.0, 0.0, np.pi], 0.5)
        expected_quaternion = np.array(
            [np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)],
            dtype=float,
        )
        np.testing.assert_allclose(
            moment,
            np.outer(expected_quaternion, expected_quaternion),
            atol=1e-12,
        )

    def test_uncertain_increment_is_a_valid_second_moment(self):
        moment = delta_quaternion_second_moment(
            [0.1, -0.2, 0.3],
            0.05,
            angular_velocity_covariance=np.diag([0.2, 0.1, 0.3]),
        )
        np.testing.assert_allclose(moment, moment.T, atol=1e-12)
        self.assertAlmostEqual(float(np.trace(moment)), 1.0)
        self.assertGreaterEqual(float(np.linalg.eigvalsh(moment)[0]), -1e-12)
        self.assertGreater(float(np.trace(moment[1:, 1:])), 0.0)

    def test_rejects_invalid_time_rate_and_covariance(self):
        invalid_calls = [
            lambda: delta_quaternion_second_moment([0.0, 0.0], 0.1),
            lambda: delta_quaternion_second_moment([0.0, np.nan, 0.0], 0.1),
            lambda: delta_quaternion_second_moment([0.0, 0.0, 0.0], -0.1),
            lambda: delta_quaternion_second_moment(
                [0.0, 0.0, 0.0],
                0.1,
                angular_velocity_covariance=np.eye(2),
            ),
            lambda: delta_quaternion_second_moment(
                [0.0, 0.0, 0.0],
                0.1,
                angular_velocity_covariance=np.triu(np.ones((3, 3))),
            ),
            lambda: delta_quaternion_second_moment(
                [0.0, 0.0, 0.0],
                0.1,
                angular_velocity_covariance=np.diag([1.0, 1.0, -0.1]),
            ),
        ]
        for invalid_call in invalid_calls:
            with self.subTest(invalid_call=invalid_call):
                with self.assertRaises(ValueError):
                    invalid_call()


class VectorAlignmentEvidenceTest(unittest.TestCase):
    def test_alignment_quadratic_prefers_the_expected_rotation(self):
        parameter = vector_alignment_bingham_evidence(
            reference_vector_parent=[0.0, 1.0, 0.0],
            observed_vector_body=[1.0, 0.0, 0.0],
            concentration=8.0,
        )
        expected_quaternion = np.array(
            [np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)],
            dtype=float,
        )
        identity = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)

        expected_score = float(expected_quaternion @ parameter @ expected_quaternion)
        identity_score = float(identity @ parameter @ identity)
        self.assertGreater(expected_score, identity_score)
        self.assertAlmostEqual(float(np.linalg.eigvalsh(parameter)[-1]), 0.0)

    def test_gravity_and_magnetic_functions_remain_separate(self):
        gravity = gravity_bingham_evidence(
            [0.0, 0.0, -1.0],
            [0.0, 0.0, -9.81],
            concentration=5.0,
        )
        magnetic = magnetic_bingham_evidence(
            [1.0, 0.0, 0.0],
            [0.2, 0.0, 0.0],
            concentration=3.0,
        )

        self.assertFalse(np.allclose(gravity, magnetic))
        self.assertEqual(np.count_nonzero(np.isclose(np.linalg.eigvalsh(gravity), 0.0)), 2)
        self.assertEqual(np.count_nonzero(np.isclose(np.linalg.eigvalsh(magnetic), 0.0)), 2)

    def test_rejects_zero_vectors_and_invalid_concentration(self):
        with self.assertRaises(ValueError):
            gravity_bingham_evidence([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 1.0)
        with self.assertRaises(ValueError):
            magnetic_bingham_evidence([1.0, 0.0, 0.0], [1.0, 0.0, 0.0], 0.0)


class OrientationBinghamFilterTest(unittest.TestCase):
    def test_zero_increment_prediction_preserves_canonical_prior(self):
        prior = np.diag([2.0, -3.0, -4.0, -5.0])
        predicted = predict_orientation_bingham(prior, [0.0, 0.0, 0.0], 0.1)
        np.testing.assert_allclose(predicted, canonical_bingham_parameter(prior))

    def test_nonzero_gyro_prediction_rotates_the_mode(self):
        identity_concentrated = np.diag([0.0, -15.0, -15.0, -15.0])
        predicted = predict_orientation_bingham(
            identity_concentrated,
            [0.0, 0.0, np.pi / 2.0],
            1.0,
            integration_steps=60,
        )
        expected = np.array([np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)])
        self.assertTrue(_same_rotation(bingham_mode(predicted), expected, tolerance=2e-2))

    def test_filter_exposes_prediction_and_evidence_components(self):
        orientation_filter = OrientationBinghamFilter(np.zeros((4, 4)))
        gravity = gravity_bingham_evidence(
            [0.0, 0.0, -1.0],
            [0.0, 0.0, -1.0],
            6.0,
        )
        magnetic = magnetic_bingham_evidence(
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            4.0,
        )
        visual = OrientationEvidence(
            source_id="vision",
            parameter=np.diag([0.0, -1.0, -1.0, -1.0]),
            kind="absolute_orientation",
        )

        update = orientation_filter.update(
            angular_velocity=[0.0, 0.0, 0.0],
            dt=0.0,
            gravity_evidence=gravity,
            magnetic_evidence=magnetic,
            independent_evidence=[visual],
        )

        np.testing.assert_allclose(update.prediction_parameter, np.zeros((4, 4)))
        np.testing.assert_allclose(update.gravity_evidence, gravity)
        np.testing.assert_allclose(update.magnetic_evidence, magnetic)
        np.testing.assert_allclose(
            update.posterior_parameter,
            canonical_bingham_parameter(gravity + magnetic + visual.parameter),
        )
        self.assertEqual(update.evidence_source_ids, ("gravity", "magnetic", "vision"))
        self.assertIs(orientation_filter.last_update, update)
        self.assertTrue(_same_rotation(bingham_mode(update.posterior_parameter), [1, 0, 0, 0]))

    def test_duplicate_sources_are_rejected_and_refusion_does_not_accumulate(self):
        orientation_filter = OrientationBinghamFilter(np.zeros((4, 4)))
        evidence = OrientationEvidence("vision", np.diag([0.0, -2.0, -2.0, -2.0]))
        orientation_filter.predict([0.0, 0.0, 0.0], 0.0)
        first = orientation_filter.fuse_independent_evidence(
            independent_evidence=[evidence]
        )
        second = orientation_filter.fuse_independent_evidence(
            independent_evidence=[evidence]
        )
        np.testing.assert_allclose(first.posterior_parameter, second.posterior_parameter)

        with self.assertRaisesRegex(ValueError, "double count"):
            orientation_filter.fuse_independent_evidence(
                independent_evidence=[evidence, evidence]
            )
        with self.assertRaisesRegex(ValueError, "double count"):
            orientation_filter.fuse_independent_evidence(
                gravity_evidence=np.zeros((4, 4)),
                independent_evidence=[
                    OrientationEvidence("gravity", np.zeros((4, 4)))
                ],
            )


if __name__ == "__main__":
    unittest.main()
