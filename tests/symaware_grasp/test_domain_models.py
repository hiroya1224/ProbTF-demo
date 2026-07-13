import unittest

import numpy as np

from symaware_grasp.grasp_targets import compose_grasp_targets
from symaware_grasp.models import GraspCandidate
from probtf.distributions import (
    BinghamOrientation,
    ConditionalGaussianTranslation,
    TransformComponent,
    TransformDistribution,
    TransformDistributionStamped,
)
from probtf.geometry import quat_mul, quat_to_rotmat


class DomainModelsTest(unittest.TestCase):
    def test_compose_grasp_targets_is_ros_free(self):
        orientation = BinghamOrientation.from_parameter_matrix(
            np.diag([0.0, -1.0, -2.0, -3.0])
        )
        object_transform = TransformDistributionStamped(
            "world",
            "object",
            1.0,
            "world_object",
            "test",
            TransformDistribution(
                (
                    TransformComponent(
                        "object",
                        1.0,
                        orientation,
                        ConditionalGaussianTranslation(
                            np.array([0.2, -0.1, 0.3]),
                            np.zeros((3, 3)),
                            np.zeros((3, 9)),
                        ),
                    ),
                )
            ),
        )
        candidate = GraspCandidate(
            grasp_id="front",
            object_to_grasp_position=[0.1, 0.0, 0.0],
            object_to_grasp_orientation_wxyz=[1.0, 0.0, 0.0, 0.0],
            approach_axis=[1.0, 0.0, 0.0],
            finger_axis=[0.0, 0.0, 1.0],
        )

        targets = compose_grasp_targets(
            object_transform,
            [candidate],
        )

        self.assertEqual(len(targets), 1)
        self.assertEqual(targets[0].child_frame_id, "front")
        component = targets[0].distribution.components[0]
        reference = orientation.reference_quaternion_wxyz
        np.testing.assert_allclose(
            component.conditional_translation_mean(reference),
            [0.3, -0.1, 0.3],
        )
        test_quaternion = np.array([np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)])
        target_quaternion = quat_mul(
            test_quaternion,
            candidate.object_to_grasp_orientation_wxyz,
        )
        np.testing.assert_allclose(
            component.conditional_translation_mean(target_quaternion),
            np.array([0.2, -0.1, 0.3])
            + quat_to_rotmat(test_quaternion) @ candidate.object_to_grasp_position,
        )
        self.assertGreater(np.linalg.norm(component.translation.rotation_coupling), 0.0)
        np.testing.assert_allclose(component.translation.residual_covariance, np.zeros((3, 3)))


if __name__ == "__main__":
    unittest.main()
