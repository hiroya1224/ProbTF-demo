"""Attitude feedback, applied wrench, source alignment and no past resets."""
from dataclasses import replace
import importlib.util
from pathlib import Path
import sys
from unittest import mock

import numpy as np
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
_spec = importlib.util.spec_from_file_location('coupled_replay_fixture', HERE/'tests/test_imu_replay.py')
_fixture = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_fixture)
fixture, refresh_fit = _fixture.fixture, _fixture.refresh_fit
from imu_replay import replay
from imu_coupled_replay import prepare_coupled_model, replay_coupled_model
from grape_param_estim.system import GRAVITY


def orientation_law(data, recursive_rotations=None, output_rotations=None):
    n = len(data['time'])
    if recursive_rotations is None:
        recursive_rotations = Rotation.from_quat(np.tile([0., 0., 0., 1.], (n, 1)))
    output_rotations = recursive_rotations if output_rotations is None else output_rotations
    result = dict(time=data['time'].copy())
    for suffix, rotations in [('recursive', recursive_rotations), ('output', output_rotations)]:
        matrices = rotations.as_matrix()
        vec = matrices.transpose(0, 2, 1).reshape(n, 9)
        result['mean_rotation_'+suffix] = matrices
        result['rotation_vector_second_moment_'+suffix] = vec[:, :, None]*vec[:, None, :]
        result['q_'+suffix+'_wxyz'] = rotations.as_quat()[:, [3, 0, 1, 2]]
        result['covariance_'+suffix] = np.zeros((n, 3, 3))
    return result


def prepared_fixture(n=13, deterministic_carrier=True):
    data, fit, old_orientation = fixture(n=n)
    model = replay(data, fit, old_orientation)['model']
    if deterministic_carrier:
        model['state_covariance_factor'] = np.zeros_like(model['state_covariance_factor'])
    return data, fit, model


def test_applied_residual_force_changes_coupled_world_position():
    data, fit, _ = fixture(n=13)
    acceleration = np.array([.7, -.3, .2])
    data['accel'] += acceleration
    refresh_fit(data, fit)
    common = replay(data, fit, fixture(n=13)[2])['model']
    common['state_covariance_factor'] = np.zeros_like(common['state_covariance_factor'])
    shared = prepare_coupled_model(data, fit, common)
    result = replay_coupled_model(data, fit, shared, orientation_law(data))
    expected = data['initial_position']+.5*data['time'][:, None]**2*acceleration
    np.testing.assert_allclose(result['position_fc_world'], expected, atol=3e-13)
    np.testing.assert_allclose(result['applied_residual_wrench_body'][:, :3],
        fit['parameters'].mass*np.broadcast_to(acceleration, (len(data['time']), 3)), atol=4e-12)
    np.testing.assert_array_equal(result['covariance_rotation'], np.zeros((len(data['time']), 3, 3)))


def test_model_residual_torque_changes_the_supplied_angular_prediction():
    data, fit, old_orientation = fixture(n=13)
    still = replay(data, fit, old_orientation)['model']
    old = prepare_coupled_model(data, fit, still)
    data['gyro'] = data['time'][:, None]*np.array([.0, .0, .8])
    refresh_fit(data, fit)
    spinning = replay(data, fit, old_orientation)['model']
    changed = prepare_coupled_model(data, fit, spinning)
    before = old['orientation_prediction_options']['prediction_gyro']
    after = changed['orientation_prediction_options']['prediction_gyro']
    assert np.max(np.abs(after-before)) > .05
    assert np.max(np.abs(changed['applied_residual_wrench_body'][:, 3:])) > .01
    q = Rotation.from_quat(spinning['quaternion_wxyz'][:, [1, 2, 3, 0]])
    expected = (q[:-1].inv()*q[1:]).as_rotvec()/np.diff(data['time'])[:, None]
    np.testing.assert_allclose(after[1:], expected, atol=2e-16)
    factor = changed['orientation_prediction_options']['prediction_increment_factor_source']
    cov = changed['orientation_prediction_options']['prediction_gyro_covariance']
    np.testing.assert_allclose(cov[1:], factor[1:]@factor[1:].transpose(0, 2, 1)/np.diff(data['time'])[:, None, None]**2)


def test_current_observation_does_not_rotate_past_force_interval():
    data, fit, common = prepared_fixture()
    shared = prepare_coupled_model(data, fit, common)
    old = orientation_law(data)
    n = len(data['time'])
    rpy = np.zeros((n, 3));rpy[1:, 1] = 20.
    new = orientation_law(data, Rotation.from_euler('xyz', rpy, degrees=True))
    before = replay_coupled_model(data, fit, shared, old)
    after = replay_coupled_model(data, fit, shared, new)
    # At first endpoint its observation can change only the current lever.
    np.testing.assert_allclose(after['position_cog_world'][1], before['position_cog_world'][1], atol=2e-16)
    np.testing.assert_allclose(after['velocity_cog_world'][1], before['velocity_cog_world'][1], atol=2e-16)
    # It changes the actual world force from the following interval onwards.
    assert after['velocity_cog_world'][-1, 0] > .2
    assert np.max(np.abs(after['position_cog_world'][-1]-before['position_cog_world'][-1])) > .01


def test_output_only_magnetics_changes_only_current_lever_and_attitude():
    data, fit, common = prepared_fixture()
    shared = prepare_coupled_model(data, fit, common)
    base = orientation_law(data)
    n = len(data['time']);rpy = np.zeros((n, 3));rpy[3:, 2] = 50.
    magnetic = orientation_law(data, output_rotations=Rotation.from_euler('xyz',rpy,degrees=True))
    first = replay_coupled_model(data, fit, shared, base)
    second = replay_coupled_model(data, fit, shared, magnetic)
    np.testing.assert_array_equal(first['position_cog_world'], second['position_cog_world'])
    np.testing.assert_array_equal(first['velocity_cog_world'], second['velocity_cog_world'])
    assert np.max(abs(first['position_fc_world']-second['position_fc_world'])) > .005
    np.testing.assert_array_equal(second['quaternion_wxyz'], magnetic['q_output_wxyz'])
    np.testing.assert_array_equal(shared['orientation_prediction_options']['prediction_gyro'], first['prediction_gyro'])


def test_full_rotation_mean_not_mode_drives_the_force():
    data, fit, common = prepared_fixture()
    shared = prepare_coupled_model(data, fit, common)
    law = orientation_law(data)
    angle = np.deg2rad(20.)
    rotations = Rotation.from_rotvec([[0.,angle,0.],[0.,-angle,0.]]).as_matrix()
    vec = rotations.transpose(0, 2, 1).reshape(2, 9)
    mean = rotations.mean(axis=0)
    second = np.einsum('ni,nj->ij',vec,vec)/2
    for suffix in ['recursive','output']:
        law['mean_rotation_'+suffix][:] = mean
        law['rotation_vector_second_moment_'+suffix][:] = second
        law['covariance_'+suffix][:] = np.diag([0., angle**2, 0.])
    result = replay_coupled_model(data, fit, shared, law)
    expected = data['initial_position'][2]-.5*GRAVITY*(1.-np.cos(angle))*data['time']**2
    np.testing.assert_allclose(result['position_fc_world'][:, 2], expected, atol=2e-13)
    assert np.max(result['position_pose_history_sigma_bound']) > 0.
    # The representative stayed at identity; a mode-only replay would hover.
    np.testing.assert_array_equal(result['quaternion_wxyz'], np.tile([1.,0.,0.,0.], (len(data['time']),1)))


def test_shared_source_cancellation_and_exact_initial_fc_position():
    data, fit, common = prepared_fixture()
    shared = prepare_coupled_model(data, fit, common)
    result = replay_coupled_model(data, fit, shared, orientation_law(data))
    np.testing.assert_array_equal(result['position_fc_world'][0], data['initial_position'])
    np.testing.assert_array_equal(result['position_known_source_factor_fc'][0], np.zeros((3,31)))
    np.testing.assert_array_equal(result['covariance_position_fc'][0], np.zeros((3,3)))
    # Every non-CoG physical direction cancels between required m*f, actuator
    # and residual before the common source factor is applied in static hover.
    np.testing.assert_allclose(shared['stage_specific_force_factor'][:, :, :16], 0., atol=2e-14)
    factor = result['position_covariance_factor_fc']
    np.testing.assert_array_equal(result['covariance_position_fc'], factor@factor.transpose(0,2,1))
    assert factor.shape == (len(data['time']),3,34)
    assert not result['metadata']['joint_state_gaussian_claimed']


def test_real_unsupported_drag_and_clock_mismatch_are_explicit():
    data, fit, common = prepared_fixture()
    changed = dict(fit)
    changed['parameters'] = replace(fit['parameters'], linear_drag=np.array([.1,0.,0.]))
    try:
        prepare_coupled_model(data, changed, common)
    except NotImplementedError:
        pass
    else:
        raise AssertionError('A future nonzero drag model must not be silently discarded')
    shared = prepare_coupled_model(data, fit, common)
    orientation = orientation_law(data);orientation['time'] += .001
    try:
        replay_coupled_model(data, fit, shared, orientation)
    except ValueError:
        pass
    else:
        raise AssertionError('Original native clocks must match')


def test_mocap_after_initialization_and_rng_cannot_affect_replay():
    data, fit, common = prepared_fixture()
    def forbidden(*args,**kwargs):raise AssertionError('No Monte Carlo')
    with mock.patch.object(np.random,'default_rng',side_effect=forbidden):
        shared = prepare_coupled_model(data, fit, common)
        before = replay_coupled_model(data, fit, shared, orientation_law(data))
        data['mocap_position'][:] = 1e30;data['mocap_quaternion_xyzw'][:] = -123.
        after = replay_coupled_model(data, fit, shared, orientation_law(data))
    for key in ['position_fc_world','position_cog_world','covariance_position_fc','quaternion_wxyz']:
        np.testing.assert_array_equal(before[key],after[key])
