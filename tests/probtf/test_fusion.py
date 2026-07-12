import unittest

import numpy as np

from probtf.provenance import ApproximationInfo, ApproximationKind, Provenance
from probtf_estimators.evidence_fusion import TransformEvidence, fuse_evidence


class TransformEvidenceFusionTest(unittest.TestCase):
    def test_fuses_bingham_and_gaussian_natural_parameters(self):
        first_bingham = np.diag([-4.0, -3.0, -2.0, 0.0])
        second_bingham = np.array(
            [
                [-1.0, 0.2, 0.0, 0.0],
                [0.2, -2.0, 0.0, 0.0],
                [0.0, 0.0, -3.0, 0.1],
                [0.0, 0.0, 0.1, 0.0],
            ]
        )
        first = TransformEvidence.from_gaussian_position(
            source_id="camera",
            parent_frame_id="world",
            child_frame_id="tool",
            position_mean=[1.0, 0.0, -1.0],
            position_covariance=np.diag([0.5, 1.0, 2.0]),
            orientation_bingham=first_bingham,
            timestamp=10.25,
            sequence=7,
        )
        second = TransformEvidence.from_gaussian_position(
            source_id="imu",
            parent_frame_id="world",
            child_frame_id="tool",
            position_mean=[0.0, 2.0, 1.0],
            position_covariance=np.diag([1.0, 0.25, 0.5]),
            orientation_bingham=second_bingham,
            timestamp=10.5,
        )

        fused = fuse_evidence([first, second])

        np.testing.assert_allclose(
            fused.orientation_bingham,
            first_bingham + second_bingham,
        )
        expected_information = first.position_information + second.position_information
        expected_vector = (
            first.position_information_vector + second.position_information_vector
        )
        np.testing.assert_allclose(fused.position_information, expected_information)
        np.testing.assert_allclose(fused.position_information_vector, expected_vector)
        expected_covariance = np.linalg.inv(expected_information)
        np.testing.assert_allclose(fused.position_covariance, expected_covariance)
        np.testing.assert_allclose(
            fused.position_mean,
            expected_covariance @ expected_vector,
        )
        self.assertEqual(fused.source_ids, ("camera", "imu"))
        self.assertEqual(fused.evidence_provenance[0].evidence_kind, "likelihood")
        self.assertEqual(fused.evidence_provenance[0].timestamp, 10.25)
        self.assertEqual(fused.evidence_provenance[0].sequence, 7)
        self.assertTrue(fused.evidence_provenance[0].contributes_orientation)
        self.assertTrue(fused.evidence_provenance[0].contributes_position)
        self.assertEqual(fused.provenance.source_ids, ("camera", "imu"))
        self.assertEqual(
            fused.provenance.method,
            "independent_natural_parameter_fusion",
        )

    def test_allows_orientation_only_and_position_only_evidence(self):
        orientation = TransformEvidence(
            source_id="gyro",
            parent_frame_id="base",
            child_frame_id="imu",
            evidence_kind="prediction",
            orientation_bingham=np.diag([-8.0, -4.0, -2.0, 0.0]),
        )
        position = TransformEvidence(
            source_id="fixture",
            parent_frame_id="base",
            child_frame_id="imu",
            position_information=np.diag([2.0, 3.0, 4.0]),
            position_information_vector=[2.0, 6.0, 12.0],
        )

        fused = fuse_evidence(item for item in (orientation, position))

        np.testing.assert_allclose(fused.orientation_bingham, orientation.orientation_bingham)
        np.testing.assert_allclose(fused.position_mean, [1.0, 2.0, 3.0])
        self.assertFalse(fused.evidence_provenance[0].contributes_position)
        self.assertEqual(fused.evidence_provenance[0].evidence_kind, "prediction")
        self.assertFalse(fused.evidence_provenance[1].contributes_orientation)

    def test_preserves_structured_provenance_and_approximation(self):
        approximation = ApproximationInfo(
            kind=ApproximationKind.BINGHAM_CLOSURE,
            lossy=True,
            detail="Moment-matched gyro convolution.",
            source="gyro_predictor",
        )
        gyro = TransformEvidence(
            source_id="orientation_filter",
            parent_frame_id="world",
            child_frame_id="imu",
            evidence_kind="prediction",
            orientation_bingham=np.diag([-3.0, -2.0, -1.0, 0.0]),
            provenance=Provenance(
                source_ids=("raw_imu",),
                derived_from_edge_ids=("previous_orientation",),
                method="gyro_prediction",
            ),
            approximation=approximation,
        )
        gravity = TransformEvidence(
            source_id="gravity",
            parent_frame_id="world",
            child_frame_id="imu",
            orientation_bingham=np.diag([-1.0, -1.0, 0.0, 0.0]),
        )

        fused = fuse_evidence([gyro, gravity])

        self.assertEqual(fused.approximation, approximation)
        self.assertEqual(fused.provenance.source_ids, ("raw_imu", "gravity"))
        self.assertEqual(
            fused.provenance.derived_from_edge_ids,
            ("previous_orientation",),
        )
        self.assertEqual(fused.evidence_provenance[0].provenance, gyro.provenance)
        self.assertEqual(
            fused.evidence_provenance[0].approximation,
            approximation,
        )

    def test_rejects_duplicate_source_by_default(self):
        first = self._orientation_evidence("imu")
        second = self._orientation_evidence("imu")

        with self.assertRaisesRegex(ValueError, "double count"):
            fuse_evidence([first, second])

        fused = fuse_evidence([first, second], allow_duplicate_sources=True)
        np.testing.assert_allclose(
            fused.orientation_bingham,
            2.0 * first.orientation_bingham,
        )
        self.assertEqual(fused.source_ids, ("imu", "imu"))

    def test_rejects_frame_mismatch_and_empty_input(self):
        first = self._orientation_evidence("imu")
        reversed_frames = TransformEvidence(
            source_id="camera",
            parent_frame_id="tool",
            child_frame_id="world",
            orientation_bingham=np.zeros((4, 4)),
        )

        with self.assertRaisesRegex(ValueError, "same directed frame pair"):
            fuse_evidence([first, reversed_frames])
        with self.assertRaisesRegex(ValueError, "at least one"):
            fuse_evidence([])

    def test_validates_payload_shapes_symmetry_psd_and_metadata(self):
        base = {
            "source_id": "imu",
            "parent_frame_id": "world",
            "child_frame_id": "tool",
        }
        invalid_payloads = [
            {"orientation_bingham": np.zeros(16)},
            {"orientation_bingham": np.triu(np.ones((4, 4)))},
            {
                "position_information": np.diag([1.0, 1.0, -0.1]),
                "position_information_vector": np.zeros(3),
            },
            {
                "position_information": np.eye(3),
                "position_information_vector": np.zeros(2),
            },
            {"position_information": np.eye(3)},
        ]
        for payload in invalid_payloads:
            with self.subTest(payload=payload):
                with self.assertRaises(ValueError):
                    TransformEvidence(**base, **payload)

        with self.assertRaises(ValueError):
            TransformEvidence(**base, orientation_bingham=np.eye(4), timestamp=np.inf)
        with self.assertRaises(ValueError):
            TransformEvidence(**base, orientation_bingham=np.eye(4), sequence=-1)
        with self.assertRaises(ValueError):
            TransformEvidence(
                **base,
                orientation_bingham=np.eye(4),
                sequence=1 << 64,
            )
        with self.assertRaises(ValueError):
            TransformEvidence(
                source_id=" ",
                parent_frame_id="world",
                child_frame_id="tool",
                orientation_bingham=np.eye(4),
            )

    def test_accepts_consistent_singular_position_likelihood(self):
        evidence = TransformEvidence(
            source_id="planar_sensor",
            parent_frame_id="world",
            child_frame_id="tool",
            position_information=np.diag([2.0, 3.0, 0.0]),
            position_information_vector=[4.0, 9.0, 0.0],
        )
        fused = fuse_evidence([evidence])

        with self.assertRaisesRegex(ValueError, "singular"):
            fused.gaussian_position()
        with self.assertRaisesRegex(ValueError, "range"):
            TransformEvidence(
                source_id="invalid_planar_sensor",
                parent_frame_id="world",
                child_frame_id="tool",
                position_information=np.diag([1.0, 1.0, 0.0]),
                position_information_vector=[0.0, 0.0, 1.0],
            )

    def test_copies_and_freezes_input_arrays(self):
        matrix = np.diag([-3.0, -2.0, -1.0, 0.0])
        evidence = TransformEvidence(
            source_id="imu",
            parent_frame_id="world",
            child_frame_id="tool",
            orientation_bingham=matrix,
        )
        matrix[0, 0] = 99.0

        self.assertEqual(evidence.orientation_bingham[0, 0], -3.0)
        with self.assertRaises(ValueError):
            evidence.orientation_bingham[0, 0] = 5.0

    @staticmethod
    def _orientation_evidence(source_id):
        return TransformEvidence(
            source_id=source_id,
            parent_frame_id="world",
            child_frame_id="tool",
            orientation_bingham=np.diag([-3.0, -2.0, -1.0, 0.0]),
        )


if __name__ == "__main__":
    unittest.main()
