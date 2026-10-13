import ast
import copy
import unittest
from pathlib import Path

import numpy as np

from probtf.geometry import quat_mul, quat_to_rotmat, rpy_to_quat
from symaware_grasp.box_geometry import BOX_DIMENSIONS, BOX_POSITION, box_mode_quaternions, make_box_record


RUNNER = Path(__file__).resolve().parents[2] / 'ros/examples/symaware_grasp/scripts/run_box_grasp_demo.py'
tree = ast.parse(RUNNER.read_text())
function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'verify_publisher_parameters')
namespace = dict(np=np, quat_mul=quat_mul, quat_to_rotmat=quat_to_rotmat)
exec(compile(ast.Module(body=[function], type_ignores=[]), str(RUNNER), 'exec'), namespace)
verify = namespace['verify_publisher_parameters']


def fixture(distribution):
    modes = box_mode_quaternions()
    if distribution == 'single_pose':
        modes = modes[:1]
    manifest = dict(distribution=distribution, target_candidate_count=1,
                    physical_box=dict(size=list(BOX_DIMENSIONS), position=list(BOX_POSITION),
                                      rotation=quat_to_rotmat(modes[0]).tolist()),
                    target_quaternions=modes.tolist(),
                    bingham_components=[dict(weight=1 / len(modes), quaternion_wxyz=mode.tolist(),
                                              parameter_matrix=(5000 * (np.outer(mode, mode) - np.eye(4))).tolist())
                                        for mode in modes])
    record = make_box_record(distribution=distribution)
    components = [dict(component_id=item.component.component_id, weight=float(item.weight),
                       mode_wxyz=item.component.orientation.reference_quaternion_wxyz.tolist(),
                       parameter_matrix=item.component.orientation.backend_parameter_matrix().tolist())
                  for item in record.distribution.normalized_components()]
    parameters = dict(object_pose_node=dict(distribution=distribution, component_count=len(components),
                      components=components, dimensions=list(BOX_DIMENSIONS), concentration=5000,
                      position_mean=list(BOX_POSITION), position_covariance=[0.] * 9,
                      true_position=list(BOX_POSITION), true_quaternion_wxyz=modes[0].tolist()))
    return manifest, parameters


class PublisherManifestChecks(unittest.TestCase):
    def test_finite_four_and_single_components_save_actual_values(self):
        for distribution in ('mixture', 'single_pose'):
            with self.subTest(distribution=distribution):
                manifest, parameters = fixture(distribution)
                verify(manifest, parameters, [1., 0., 0., 0.])
                self.assertTrue(manifest['publisher_parameters_verified'])
                for saved, actual in zip(manifest['bingham_components'], parameters['object_pose_node']['components']):
                    self.assertEqual(saved['parameter_matrix'], actual['parameter_matrix'])
                    self.assertEqual(saved['quaternion_wxyz'], actual['mode_wxyz'])

    def test_quaternion_sign_is_not_a_physical_mismatch(self):
        manifest, parameters = fixture('mixture')
        for source in parameters['object_pose_node']['components']:
            source['mode_wxyz'] = (-np.asarray(source['mode_wxyz'])).tolist()
        parameters['object_pose_node']['true_quaternion_wxyz'] = (-np.asarray(
            parameters['object_pose_node']['true_quaternion_wxyz'])).tolist()
        verify(manifest, parameters, [1., 0., 0., 0.])
        self.assertTrue(manifest['publisher_parameters_verified'])

    def test_mismatched_physical_box_or_distribution_is_rejected(self):
        changes = [dict(dimensions=[.1, .1, .18]), dict(true_position=[.8, -.12, .46]),
                   dict(true_quaternion_wxyz=[1., 0., 0., 0.]), dict(distribution='single_pose'),
                   dict(component_count=1), dict(concentration=100.), dict(position_covariance=[1.] * 9)]
        for change in changes:
            with self.subTest(change=change):
                manifest, parameters = fixture('mixture')
                parameters['object_pose_node'].update(change)
                with self.assertRaises(RuntimeError):
                    verify(manifest, parameters, [1., 0., 0., 0.])

    def test_wrong_bingham_matrix_and_weight_are_rejected(self):
        for field in ('weight', 'parameter_matrix', 'mode_wxyz'):
            with self.subTest(field=field):
                manifest, parameters = fixture('mixture')
                first = parameters['object_pose_node']['components'][0]
                if field == 'weight':
                    first[field] = .5
                elif field == 'parameter_matrix':
                    first[field][0][0] += 1
                else:
                    first[field] = [1., 0., 0., 0.]
                with self.assertRaises(RuntimeError):
                    verify(manifest, parameters, [1., 0., 0., 0.])

    def test_incomplete_target_modes_are_rejected(self):
        manifest, parameters = fixture('mixture')
        manifest['target_quaternions'].pop()
        with self.assertRaises(RuntimeError):
            verify(manifest, parameters, [1., 0., 0., 0.])

    def test_wrong_grasp_composition_is_rejected(self):
        manifest, parameters = fixture('mixture')
        with self.assertRaises(RuntimeError):
            verify(manifest, parameters, rpy_to_quat(0, 0, np.pi / 2))


if __name__ == '__main__':
    unittest.main()
