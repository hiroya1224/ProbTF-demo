"""The unobserved body starts at the rigid command family, before fusion."""
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pinocchio as pin
import pytest
import rospkg
import yaml

from probtf.geometry import quat_to_rotmat
from probtf.spherical_law.first_order import point_query, pose_query
from probtf_observation_fusion_demo.moving_inference import MovingInference
from probtf_switch_press_demo.scene import write_robot


PACKAGE = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def initial_geometry(tmp_path_factory):
    """Generate attachment geometry without creating a physical plant or sensor."""
    config = yaml.safe_load((PACKAGE/'config/switch_press_scene.yaml').read_text())
    source = Path(rospkg.RosPack().get_path('symaware_grasp'))/'urdf/simple_six_dof_arm.urdf'
    urdf = write_robot(source, tmp_path_factory.mktemp('initial_prior')/'robot.urdf', config)
    return urdf, np.asarray(config['initial_command'], dtype=float)


def _paths(inference):
    return dict(inference.paths['links'], rod_tip=inference.paths['hand'],
                camera_optical_frame=inference.paths['camera'])


def _rigid_poses(urdf, command, names):
    model = pin.buildModelFromUrdf(str(urdf))
    data = model.createData()
    pin.forwardKinematics(model, data, command)
    pin.updateFramePlacements(model, data)
    base = data.oMf[model.getFrameId('base_link')]
    return {name: base.inverse()*data.oMf[model.getFrameId(name)] for name in names}


@pytest.mark.parametrize('command_kind', ('initial', 'reference', 'other'))
def test_unobserved_prior_modal_paths_are_rigid_URDF_FK(initial_geometry, command_kind):
    urdf, initial = initial_geometry
    inference = MovingInference(urdf.read_text(), ())
    command = {'initial': initial, 'reference': np.zeros(6),
               'other': np.array([.17, -1.2, 1.8, -.25, -.7, .31])}[command_kind]
    stamp, index = 1234.5, 7
    state = inference.prior_at(command, stamp, index)
    poses = _rigid_poses(urdf, command, _paths(inference))
    for name, path in _paths(inference).items():
        point = point_query(state, path)
        pose = pose_query(state, path)
        np.testing.assert_allclose(point.mode, poses[name].translation, atol=1e-12, rtol=0.)
        np.testing.assert_allclose(quat_to_rotmat(pose.rotation_wxyz),
                                   poses[name].rotation, atol=1e-12, rtol=0.)
        # These are queries of the original initial law, not a later posterior.
        for query in (point, pose):
            assert query.state_id == state.state_id
            assert query.stamp == stamp
            assert query.command == tuple(command)
            assert query.version == 3*index+1
            assert query.component_id == 'common_body_prior'
    assert inference.last_result is None
    assert inference.components == inference.goal_components == ()


def _leading_order_point_mean(state, path):
    """Independent right-rotation moments: E[R] = R(I+(C-tr(C)I)/2)."""
    edges = {edge.edge_id: (i, edge) for i, edge in enumerate(state.transforms)}
    rotation, rotation_correction = np.eye(3), np.zeros((3, 3))
    modal, correction = np.zeros(3), np.zeros(3)
    for edge_id, inverse in path:
        assert inverse is False
        index, edge = edges[edge_id]
        modal += rotation @ edge.translation
        correction += rotation_correction @ edge.translation
        local_rotation = quat_to_rotmat(edge.rotation_wxyz)
        covariance = state.covariance[6*index+3:6*index+6, 6*index+3:6*index+6]
        local_correction = .5*(covariance-np.trace(covariance)*np.eye(3))
        rotation_correction = (rotation_correction @ local_rotation
                               + rotation @ local_rotation @ local_correction)
        rotation = rotation @ local_rotation
    return modal, correction


def test_unobserved_prior_keeps_core_O_P_mean_correction(initial_geometry):
    urdf, command = initial_geometry
    inference = MovingInference(urdf.read_text(), ())
    state = inference.prior_at(command, 1234.5, 0)
    independent = np.zeros_like(state.covariance)
    for index in range(len(state.transforms)):
        block = slice(6*index, 6*index+6)
        independent[block, block] = state.covariance[block, block]
    np.testing.assert_array_equal(state.covariance, independent)
    for path in _paths(inference).values():
        query = point_query(state, path)
        modal, correction = _leading_order_point_mean(state, path)
        np.testing.assert_allclose(query.mode, modal, atol=1e-14, rtol=0.)
        np.testing.assert_allclose(query.correction, correction, atol=1e-14, rtol=0.)
        np.testing.assert_allclose(query.mean, modal+correction, atol=1e-14, rtol=0.)
    hand = point_query(state, inference.paths['hand'])
    assert np.linalg.norm(hand.mean-hand.mode) > .001
    # Querying must not recenter the retained modal transforms to force E[X]=FK(q).
    repeated = inference.prior_at(command, 1234.5, 0)
    assert repeated.state_id == state.state_id
    np.testing.assert_array_equal(repeated.covariance, state.covariance)


def test_initial_prior_ignores_inertia_and_never_constructs_physics_or_K_model(
        initial_geometry, monkeypatch):
    from probtf_switch_press_demo.body import EquilibriumBody
    from probtf_observation_fusion_demo.dynamic_physics import NativeDynamics
    from probtf_observation_fusion_demo.stiffness_identification import StiffnessIdentifier
    from probtf_observation_fusion_demo.auxiliary_equilibrium_native import NativeAuxiliaryEquilibrium

    def forbidden(*args, **kwargs):
        raise AssertionError('An initial direct transform law must not query a physical or K model.')

    for constructor in (EquilibriumBody, NativeDynamics, StiffnessIdentifier,
                        NativeAuxiliaryEquilibrium):
        monkeypatch.setattr(constructor, '__init__', forbidden)
    urdf, command = initial_geometry
    description = urdf.read_text()
    changed = ET.fromstring(description)
    inertials = changed.findall('./link/inertial')
    assert inertials
    for inertial in inertials:
        inertial.find('mass').set('value', '37.0')
        origin = inertial.find('origin')
        if origin is not None:
            origin.set('xyz', '0.3 -0.2 0.4')
            origin.set('rpy', '0.2 -0.3 0.4')
        inertia = inertial.find('inertia')
        for name in ('ixx', 'iyy', 'izz'):
            inertia.set(name, '17.0')
    # Only known geometry and the actual sent command cross the inference boundary.
    original_inference = MovingInference(description, ())
    changed_inference = MovingInference(ET.tostring(changed, encoding='unicode'), ())
    original = original_inference.prior_at(command, 1234.5, 0)
    actual = changed_inference.prior_at(command, 1234.5, 0)
    assert actual.state_id == original.state_id
    assert actual.command == original.command
    assert actual.stamp == original.stamp
    assert actual.component_id == original.component_id
    np.testing.assert_array_equal(actual.mean, original.mean)
    np.testing.assert_array_equal(actual.covariance, original.covariance)
    for left, right in zip(actual.transforms, original.transforms):
        assert left.edge_id == right.edge_id
        np.testing.assert_array_equal(left.translation, right.translation)
        np.testing.assert_array_equal(left.rotation_wxyz, right.rotation_wxyz)
        if left.bingham is not None:
            np.testing.assert_array_equal(left.bingham, right.bingham)
