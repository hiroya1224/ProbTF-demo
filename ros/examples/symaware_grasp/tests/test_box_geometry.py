"""Independent checks of physical box symmetry, native payloads and clearance."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
import unittest
from unittest import mock

import numpy as np
from scipy.spatial.transform import Rotation

from probtf.distributions import DistributionStatus, OrientationKind
from probtf.geometry import quat_to_rotmat, rpy_to_quat
from probtf.probability import sample_transform_component
from symaware_grasp_ros import object_belief_to_msg, record_from_app_message

DRAFT = Path(__file__).resolve().parents[1]
SPEC = spec_from_file_location('symaware_grasp.box_geometry', DRAFT / 'src/symaware_grasp/box_geometry.py')
geometry = module_from_spec(SPEC)
sys.modules[SPEC.name] = geometry
SPEC.loader.exec_module(geometry)
NODE_SPEC = spec_from_file_location('draft_box_pose_node', DRAFT / 'scripts/box_pose_node.py')
node_module = module_from_spec(NODE_SPEC)
NODE_SPEC.loader.exec_module(node_module)


def ordered(points):
    points = np.round(points, 12)
    return points[np.lexsort(points.T[::-1])]


class BoxGeometryTests(unittest.TestCase):
    def test_d2_rotations_preserve_all_physical_corners(self):
        modes = geometry.box_mode_quaternions()
        reference = quat_to_rotmat(modes[0])
        expected = ordered(geometry.box_corners(geometry.BOX_POSITION, reference))
        np.testing.assert_allclose(modes @ modes.T, np.eye(4), atol=1e-12)
        for mode in modes:
            rotation = quat_to_rotmat(mode)
            self.assertAlmostEqual(np.linalg.det(rotation), 1.0)
            np.testing.assert_allclose(ordered(geometry.box_corners(geometry.BOX_POSITION, rotation)), expected, atol=1e-12)
        # A quarter yaw turn is not a symmetry of this unequal-edge box.
        quarter = reference @ Rotation.from_euler('z', 90, degrees=True).as_matrix()
        self.assertFalse(np.allclose(ordered(geometry.box_corners(geometry.BOX_POSITION, quarter)), expected))

    def test_mixture_is_four_finite_equal_weight_native_components(self):
        record = geometry.make_box_record(stamp=12.5)
        normalized = record.distribution.normalize_weights()
        self.assertIs(normalized.status, DistributionStatus.OK)
        self.assertEqual(len(normalized.components), 4)
        self.assertEqual(len({item.component.component_id for item in normalized.components}), 4)
        self.assertEqual([item.weight for item in normalized.components], [0.25] * 4)
        for item, mode in zip(normalized.components, geometry.box_mode_quaternions()):
            self.assertIs(item.component.orientation.kind, OrientationKind.FINITE_BINGHAM)
            parameter = item.component.orientation.backend_parameter_matrix()
            np.testing.assert_allclose(np.linalg.eigvalsh(parameter), [-5000, -5000, -5000, 0], atol=1e-8)
            self.assertAlmostEqual(float(mode @ parameter @ mode), 0.0, places=7)
            np.testing.assert_allclose(item.component.conditional_translation_mean(mode), geometry.BOX_POSITION)
        restored = record_from_app_message(object_belief_to_msg(record, 'demo_box'))
        self.assertEqual(restored.stamp, record.stamp)
        self.assertEqual(len(restored.distribution.components), 4)
        for left, right in zip(record.distribution.components, restored.distribution.components):
            self.assertEqual(left.component_id, right.component_id)
            self.assertEqual(left.raw_weight, right.raw_weight)
            np.testing.assert_allclose(left.orientation.backend_parameter_matrix(), right.orientation.backend_parameter_matrix())

    def test_single_pose_keeps_the_same_true_reference(self):
        mixture = geometry.make_box_record()
        single = geometry.make_box_record(distribution='single_pose')
        normalized = single.distribution.normalized_components()
        self.assertEqual(len(normalized), 1)
        self.assertEqual(normalized[0].weight, 1.0)
        np.testing.assert_allclose(single.representative.translation, mixture.representative.translation)
        np.testing.assert_allclose(single.representative.rotation_wxyz, mixture.representative.rotation_wxyz)

    def test_finite_samples_near_each_mode_keep_exact_common_center(self):
        # This is a broad qualitative sampler check, not probability calibration.
        for component in geometry.make_box_record().distribution.components:
            sample = sample_transform_component(component, 96)
            errors = 2 * np.arccos(np.clip(np.abs(sample.rotations_wxyz @ component.orientation.reference_quaternion_wxyz), 0, 1))
            self.assertLess(float(np.median(errors)), np.deg2rad(8))
            self.assertLess(float(np.quantile(errors, 0.95)), np.deg2rad(20))
            self.assertGreater(float(np.var(errors)), 0.0)
            np.testing.assert_allclose(sample.translations, np.tile(geometry.BOX_POSITION, (96, 1)))

    def test_sat_separated_contact_overlap_and_rotated_support(self):
        identity = np.eye(3)
        half = np.array([0.05, 0.04, 0.09])
        self.assertAlmostEqual(geometry.box_box_gap([0, 0, 0], identity, half, [0.12, 0, 0], identity, half), 0.02)
        self.assertEqual(geometry.box_box_gap([0, 0, 0], identity, half, [0.10, 0, 0], identity, half), 0.0)
        self.assertLess(geometry.box_box_gap([0, 0, 0], identity, half, [0.08, 0, 0], identity, half), 0.0)
        rotated = Rotation.from_euler('z', 45, degrees=True).as_matrix()
        projected = (half[0] + half[1]) / np.sqrt(2)
        self.assertAlmostEqual(geometry.box_box_gap([0, 0, 0], identity, [0.05] * 3, [0.18, 0, 0], rotated, half), 0.18 - 0.05 - projected)
        self.assertLess(geometry.box_box_gap([0, 0, 0], identity, [0.05] * 3, [0.10, 0, 0], rotated, half), 0.0)

    def test_skew_edges_need_the_cross_product_separating_axes(self):
        a = Rotation.from_euler('xyz', [35, 20, -15], degrees=True).as_matrix()
        b = Rotation.from_euler('xyz', [-25, 60, 50], degrees=True).as_matrix()
        half = np.array([0.3, 0.04, 0.04])
        displacement = np.array([-0.4, -0.1, 0.25])
        # These slender boxes overlap in all six face-normal projections.
        normals = np.vstack((a.T, b.T))
        face_gaps = np.abs(normals @ displacement) - np.abs(normals @ a) @ half - np.abs(normals @ b) @ half
        self.assertLess(float(np.max(face_gaps)), 0.0)
        # Their skew edges nevertheless separate; a face-only test would fail.
        gap = geometry.box_box_gap([0, 0, 0], a, half, displacement, b, half)
        self.assertAlmostEqual(gap, 0.02189237100706136)

    def test_sat_symmetry_and_rigid_frame_invariance(self):
        a = Rotation.from_euler('xyz', [10, 30, 20], degrees=True).as_matrix()
        b = Rotation.from_euler('xyz', [-20, 15, 70], degrees=True).as_matrix()
        center_a, center_b = np.array([0.2, -0.1, 0.3]), np.array([0.5, 0.4, 0.2])
        half_a, half_b = np.array([0.10, 0.06, 0.08]), np.array([0.07, 0.12, 0.04])
        gap = geometry.box_box_gap(center_a, a, half_a, center_b, b, half_b)
        self.assertAlmostEqual(gap, geometry.box_box_gap(center_b, b, half_b, center_a, a, half_a))
        world = Rotation.from_euler('xyz', [50, -20, 35], degrees=True).as_matrix()
        translation = np.array([2, -3, 1])
        self.assertAlmostEqual(gap, geometry.box_box_gap(world @ center_a + translation, world @ a, half_a, world @ center_b + translation, world @ b, half_b))

    def test_collision_uses_fixed_rotation_and_inherited_path_validation(self):
        urdf = '<robot name="test"><link name="base_link"><collision name="body"><geometry><box size="0.1 0.1 0.1"/></geometry></collision></link></robot>'
        true_rotation = np.eye(3)
        model = geometry.BoxCollisionModel(urdf, object_rotation=true_rotation)
        true_rotation[:] = Rotation.from_euler('z', 45, degrees=True).as_matrix()
        q = np.zeros(6)
        distance, _ = model.clearance(q, 0.106, [0.12, 0, 0], [0, 0, 1])
        self.assertAlmostEqual(distance, 0.02)
        self.assertEqual(model.clearance(q, 0.106, [0.12, 0, 0], [0, 0, -1])[0], distance)
        self.assertEqual(model.validate_path(np.zeros((2, 6)), 0.106, [0.12, 0, 0], minimum=0.01)['clearance_lower_bound_mm'], 1000 * distance)
        with self.assertRaises(RuntimeError):
            model.validate_path(np.zeros((2, 6)), 0.106, [0.10, 0, 0], minimum=0.0005)

    def test_invalid_dimensions_and_rotations_fail(self):
        with self.assertRaises(ValueError):
            geometry.make_box_record(dimensions=(0.1, 0, 0.2))
        with self.assertRaises(ValueError):
            geometry.box_box_gap([0, 0, 0], np.diag([1, 1, -1]), [0.1] * 3, [0.3, 0, 0], np.eye(3), [0.1] * 3)
        with self.assertRaises(ValueError):
            geometry.make_box_record(distribution='component_split')

    def test_grasp_diagnostics_use_discrete_modes_and_true_box_width(self):
        modes = geometry.box_mode_quaternions()
        class Robot:
            dof = 6
            quaternion = modes[1]
            def within_limits(self, _): return True
            def forward_kinematics(self, _): return np.asarray(geometry.BOX_POSITION), self.quaternion, None
        robot = Robot()
        result = geometry.evaluate_box_grasp(np.zeros(6), 0.106, geometry.BOX_POSITION, modes, robot=robot, object_rotation=quat_to_rotmat(modes[0]))
        self.assertEqual(result['tcp_object_distance_mm'], 0.0)
        self.assertAlmostEqual(result['nearest_mode_error_deg'], 0.0)
        self.assertAlmostEqual(result['reference_mode_error_deg'], 180.0)
        self.assertAlmostEqual(result['jaw_axis_box_width_mm'], 80.0)
        self.assertAlmostEqual(result['jaw_gap_mm'], 26.0)
        robot.quaternion = rpy_to_quat(0, 0, np.deg2rad(67.5 + 45))
        result = geometry.evaluate_box_grasp(np.zeros(6), 0.106, geometry.BOX_POSITION, modes, robot=robot, object_rotation=quat_to_rotmat(modes[0]))
        self.assertAlmostEqual(result['nearest_mode_error_deg'], 45.0)
        self.assertLess(result['jaw_gap_mm'], 0.0)

    def test_publisher_keeps_whole_record_and_identical_true_geometry(self):
        saved = []
        markers = []
        for distribution, count in [('mixture', 4), ('single_pose', 1)]:
            params = {}
            def get_param(name, default=None):
                return distribution if name == '~distribution' else default
            with mock.patch.object(node_module.rospy, 'get_param', side_effect=get_param), \
                 mock.patch.object(node_module.rospy, 'set_param', side_effect=lambda key, value: params.__setitem__(key, value)), \
                 mock.patch.object(node_module.rospy, 'get_name', return_value='/box_pose_test'), \
                 mock.patch.object(node_module.rospy, 'Publisher', side_effect=lambda *args, **kwargs: mock.Mock()), \
                 mock.patch.object(node_module.rospy, 'Timer'), \
                 mock.patch.object(node_module.rospy.Time, 'now', return_value=node_module.rospy.Time.from_sec(12.5)), \
                 mock.patch.object(node_module, 'ProbTfBroadcaster'):
                node = node_module.BoxPoseNode()
                node.publish_message(None)
                record = node.broadcaster.send_transform.call_args[0][0]
                belief = node.publisher.publish.call_args[0][0]
                restored = record_from_app_message(belief)
                self.assertEqual(len(record.distribution.components), count)
                self.assertEqual(len(restored.distribution.components), count)
                self.assertEqual(restored.stamp, record.stamp)
                for expected, item in zip(record.distribution.components, restored.distribution.components):
                    self.assertEqual(expected.raw_weight, item.raw_weight)
                    np.testing.assert_allclose(expected.orientation.backend_parameter_matrix(), item.orientation.backend_parameter_matrix())
                self.assertEqual(params['~component_count'], count)
                self.assertEqual(sum(item['weight'] for item in params['~components']), 1.0)
                markers.append(node.geometry_publisher.publish.call_args[0][0])
                saved.append(params)
        for marker in markers:
            self.assertEqual(marker.type, node_module.Marker.CUBE)
            self.assertEqual(marker.header.frame_id, 'base_link')
        self.assertEqual(markers[0].pose, markers[1].pose)
        self.assertEqual(markers[0].scale, markers[1].scale)
        self.assertEqual(markers[0].color, markers[1].color)
        self.assertEqual(saved[0]['~true_quaternion_wxyz'], saved[1]['~true_quaternion_wxyz'])


if __name__ == '__main__':
    unittest.main()
