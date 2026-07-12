import unittest

import numpy as np

from symaware_grasp.grasp_targets import compose_grasp_targets
from symaware_grasp.models import GraspCandidate, ProbabilisticTransform


class DomainModelsTest(unittest.TestCase):
    def test_compose_grasp_targets_is_ros_free(self):
        object_transform = ProbabilisticTransform(
            parent_frame_id="world",
            child_frame_id="object",
            position_mean=[0.2, -0.1, 0.3],
            position_covariance=np.zeros((3, 3)),
            orientation_bingham=np.diag([-3.0, -2.0, -1.0, 0.0]),
            orientation_mode_wxyz=[1.0, 0.0, 0.0, 0.0],
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
            rotation_covariance_samples=0,
            covariance_floor=1e-4,
        )

        self.assertEqual(len(targets), 1)
        self.assertEqual(targets[0].child_frame_id, "front")
        np.testing.assert_allclose(targets[0].position_mean, [0.3, -0.1, 0.3])
        np.testing.assert_allclose(targets[0].position_covariance, 1e-4 * np.eye(3))
        np.testing.assert_allclose(targets[0].orientation_mode_wxyz, [1.0, 0.0, 0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
