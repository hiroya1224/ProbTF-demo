"""Check C4 axis visibility without changing the native pose distribution."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import unittest
from unittest import mock

import numpy as np
from sensor_msgs import point_cloud2
from std_msgs.msg import Header

from probtf.geometry import quat_to_rotmat
from symaware_grasp import box_geometry
from symaware_grasp_ros import object_belief_to_msg


SCRIPT = Path(box_geometry.__file__).resolve().parents[2] / 'scripts/ptf_visualizer.py'
SPEC = spec_from_file_location('box_axis_display_visualizer', SCRIPT)
visualizer_module = module_from_spec(SPEC)
SPEC.loader.exec_module(visualizer_module)


def ordered(points):
    points = np.round(np.asarray(points), 10)
    return points[np.lexsort(points.T[::-1])]


def make_visualizer(axis_lengths=None):
    # Only ROS I/O is replaced. Constructor defaults and rendering stay real.
    parameters = {'~axis_length': 0.18, '~sample_count': 64}
    if axis_lengths is not None:
        parameters['~axis_lengths'] = axis_lengths
    with mock.patch.object(visualizer_module.rospy, 'get_param',
                           side_effect=lambda name, default=None: parameters.get(name, default)), \
         mock.patch.object(visualizer_module, 'RosProbTfListener'), \
         mock.patch.object(visualizer_module.rospy, 'Publisher', side_effect=lambda *args, **kwargs: mock.Mock()), \
         mock.patch.object(visualizer_module.rospy, 'Subscriber'):
        return visualizer_module.ProbTfV2Visualizer()


def colored_endpoints(marker):
    endpoints = {'red': [], 'green': [], 'blue': []}
    colors = {(1.0, 0.0, 0.0): 'red', (0.0, 1.0, 0.0): 'green', (0.0, 0.0, 1.0): 'blue'}
    for index in range(0, len(marker.points), 2):
        color = marker.colors[index]
        endpoint = marker.points[index + 1]
        endpoints[colors[(color.r, color.g, color.b)]].append([endpoint.x, endpoint.y, endpoint.z])
    return {name: np.asarray(points) for name, points in endpoints.items()}


class BoxAxisDisplayTests(unittest.TestCase):
    def setUp(self):
        self.record = box_geometry.make_box_record(stamp=12.5)
        self.header = Header(frame_id=self.record.parent_frame_id)
        self.center = np.asarray(box_geometry.BOX_POSITION)
        self.rotation = quat_to_rotmat(box_geometry.box_mode_quaternions()[0])
        self.side_normals = np.array([self.rotation[:, 0], self.rotation[:, 1],
                                      -self.rotation[:, 0], -self.rotation[:, 1]])

    def test_unequal_lengths_separate_four_red_and_four_green_endpoints(self):
        visualizer = make_visualizer([0.18, 0.12, 0.18])
        marker = visualizer.build_component_marker(self.header, self.record)
        endpoints = colored_endpoints(marker)
        self.assertEqual(len(marker.points), 24)
        for index in range(0, len(marker.points), 2):
            origin = marker.points[index]
            np.testing.assert_allclose([origin.x, origin.y, origin.z], self.center, atol=1e-12)
        np.testing.assert_allclose(ordered(endpoints['red']),
                                   ordered(self.center + 0.18 * self.side_normals), atol=1e-10)
        np.testing.assert_allclose(ordered(endpoints['green']),
                                   ordered(self.center + 0.12 * self.side_normals), atol=1e-10)
        self.assertEqual(len(np.unique(np.round(endpoints['red'], 10), axis=0)), 4)
        self.assertEqual(len(np.unique(np.round(endpoints['green'], 10), axis=0)), 4)
        self.assertTrue(np.all(endpoints['red'][:, 2] > 0))
        self.assertFalse(np.any(np.linalg.norm(endpoints['red'][:, None, :] - endpoints['green'][None, :, :], axis=2) < 1e-8))
        np.testing.assert_allclose(endpoints['blue'], np.tile(self.center + [0, 0, 0.18], (4, 1)), atol=1e-12)
        self.assertEqual(len(np.unique(np.round(endpoints['blue'], 10), axis=0)), 1)

    def test_scalar_default_keeps_all_three_lengths_at_018(self):
        visualizer = make_visualizer()
        np.testing.assert_array_equal(visualizer.axis_lengths, [0.18, 0.18, 0.18])
        endpoints = colored_endpoints(visualizer.build_component_marker(self.header, self.record))
        for points in endpoints.values():
            np.testing.assert_allclose(np.linalg.norm(points - self.center, axis=1), 0.18, atol=1e-12)
        np.testing.assert_allclose(ordered(endpoints['red']), ordered(endpoints['green']), atol=1e-10)
        np.testing.assert_allclose(endpoints['blue'], np.tile(self.center + [0, 0, 0.18], (4, 1)), atol=1e-12)

    def check_native_cloud_lengths(self, axis_lengths, expected_lengths):
        visualizer = make_visualizer(axis_lengths)
        message = object_belief_to_msg(self.record, 'demo_box')
        # Native finite-mixture sampling and point transformation are exercised.
        with mock.patch.object(visualizer_module, 'lookup_message_record', return_value=self.record):
            visualizer.handle_belief(message)
        cloud = visualizer.cloud_publisher.publish.call_args[0][0]
        self.assertEqual(cloud.header.frame_id, self.record.parent_frame_id)
        self.assertEqual(cloud.header.stamp, message.transform.header.stamp)
        rows = list(point_cloud2.read_points(cloud, field_names=('x', 'y', 'z', 'rgb'), skip_nans=False))
        self.assertEqual(len(rows), 3 * visualizer.sample_count)
        points = np.asarray([row[:3] for row in rows])
        colors = np.asarray([row[3] for row in rows], dtype=np.uint32)
        for color, length in zip(visualizer.axis_colors, expected_lengths):
            endpoints = points[colors == color]
            self.assertEqual(len(endpoints), visualizer.sample_count)
            np.testing.assert_allclose(np.linalg.norm(endpoints - self.center, axis=1), length, atol=1e-6)
        rendered = colored_endpoints(visualizer.marker_publisher.publish.call_args[0][0])
        for points, length in zip(rendered.values(), expected_lengths):
            np.testing.assert_allclose(np.linalg.norm(points - self.center, axis=1), length, atol=1e-12)

    def test_native_cloud_uses_separate_red_green_and_blue_lengths(self):
        self.check_native_cloud_lengths([0.18, 0.12, 0.18], [0.18, 0.12, 0.18])

    def test_native_cloud_uses_the_unchanged_scalar_default(self):
        self.check_native_cloud_lengths(None, [0.18, 0.18, 0.18])

    def test_display_extents_preserve_native_components_and_weights(self):
        before = [(item.raw_weight, item.orientation.reference_quaternion_wxyz.copy(),
                   item.orientation.backend_parameter_matrix().copy())
                  for item in self.record.distribution.components]
        for lengths in (None, [0.18, 0.12, 0.18]):
            visualizer = make_visualizer(lengths)
            visualizer.build_component_marker(self.header, self.record)
            with mock.patch.object(visualizer_module, 'lookup_message_record', return_value=self.record):
                visualizer.handle_belief(object_belief_to_msg(self.record, 'demo_box'))
        self.assertEqual(len(self.record.distribution.components), 4)
        for component, (weight, mode, parameter) in zip(self.record.distribution.components, before):
            self.assertEqual(component.raw_weight, weight)
            np.testing.assert_array_equal(component.orientation.reference_quaternion_wxyz, mode)
            np.testing.assert_array_equal(component.orientation.backend_parameter_matrix(), parameter)


if __name__ == '__main__':
    unittest.main()
