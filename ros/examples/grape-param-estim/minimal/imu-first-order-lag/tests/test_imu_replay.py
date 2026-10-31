"""Physical hover, lever, uncertainty and truth-boundary replay regressions."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from imu_replay import replay, _residual_wrench_linearization
from imu_model import ImuDynamicsProblem, SiParameterChart, joint_chart_basis, load_vehicle_model
from grape_param_estim.system import GRAVITY


def fixture(n=25, omega_z=0.):
    original = load_vehicle_model(HERE.parent / 'grape_vehicle_model.json')
    pars = replace(original.parameters, cog_offset=np.zeros(3),
        inertia=np.diag([.065, .065, .129]))
    model = SimpleNamespace(parameters=pars, geometry=original.geometry)
    t = np.linspace(0., .12, n)
    lever = np.array([.02, -.01, .03])
    omega = np.tile([0., 0., omega_z], (n, 1))
    specific = np.tile([0., 0., GRAVITY], (n, 1)) + np.cross(omega, np.cross(omega, lever))
    quaternion = np.column_stack((np.cos(.5 * omega_z * t), np.zeros(n),
                                  np.zeros(n), np.sin(.5 * omega_z * t)))
    data = {'time': t, 'gyro': omega.copy(), 'accel': specific.copy(),
        'gyro_noise_covariance': np.eye(3)*1e-6, 'accel_noise_covariance': np.eye(3)*1e-4,
        'controller_dt': .005,
        'gyro_sensor_position_in_body': lever, 'initial_quaternion_wxyz': quaternion[0],
        'initial_position': lever.copy(), 'initial_velocity': np.cross(omega[0], lever),
        'initial_velocity_covariance': np.eye(3)*1e-6,
        'rotor_command_time': np.array([-1., 1.]),
        'rotor_command': np.full((2, 4), pars.mass * GRAVITY/4),
        'gimbal_time': np.array([-1., 1.]), 'gimbal_angle': np.zeros((2, 4)),
        'mocap_position': np.full((n, 3), np.nan),
        'mocap_quaternion_xyzw': np.full((n, 4), np.nan)}
    fit = {'model': model, 'parameters': pars, 'chart': SiParameterChart(pars),
        'physical_coordinate': np.zeros(14), 'joint_basis': joint_chart_basis(),
        'taus': np.array([.02, .01, .015]),
        'parameter_covariance_factor_reduced': np.eye(16)*1e-4,
        'arrays': {'time': t, 'corrected_gyro': omega, 'corrected_angular_acceleration': np.zeros((n, 3)),
            'corrected_specific_force': specific, 'gyro_derivative': np.zeros((n, 3)),
            'gyro_second_derivative': np.zeros((n, 3)), 'accel_derivative': np.zeros((n, 3)),
            'raw_residual_wrench': np.zeros((n, 6))},
        'estimate': {'method': {'options': {'sg_window_seconds': .1, 'sg_degree': 3}}}}
    orientation = {'time': t, 'q_output_wxyz': quaternion,
        'covariance_output': np.tile(np.eye(3)*1e-4, (n, 1, 1)),
        'corrected_gyro_covariance': np.tile(np.eye(3)*1e-6, (n, 1, 1))}
    refresh_fit(data, fit)
    return data, fit, orientation


def refresh_fit(data, fit):
    """Reinfer the same wrench whenever a synthetic parameter candidate changes."""
    problem = ImuDynamicsProblem(data, fit['model'], fit['estimate']['method']['options'])
    coordinate = np.r_[fit['joint_basis'][:14, :13].T @ fit['physical_coordinate'], np.log(fit['taus'])]
    evaluation = problem.evaluate(coordinate)
    fit['problem'] = problem
    fit['parameters'] = evaluation['parameters']
    fit['arrays'] = dict(evaluation, time=np.asarray(data['time']),
        gyro_derivative=problem.gyro['derivative'],
        gyro_second_derivative=problem.gyro['second_derivative'],
        accel_derivative=problem.accel['derivative'],
        jacobian_reduced_log_tau=evaluation['jacobian'])


def test_static_hover_both_means_and_covariance_factors():
    data, fit, orientation = fixture()
    result = replay(data, fit, orientation)
    for name in ('model', 'imu_dead_reckoning'):
        branch = result[name]
        np.testing.assert_allclose(branch['position_fc_world'],
            np.broadcast_to(data['initial_position'], (len(data['time']), 3)), atol=1e-12)
        np.testing.assert_allclose(branch['velocity_fc_world'], 0., atol=1e-12)
        np.testing.assert_allclose(np.linalg.norm(branch['quaternion_wxyz'], axis=1), 1., atol=1e-14)
        factors = branch['state_covariance_factor']
        np.testing.assert_allclose(branch['covariance_state'], factors @ factors.transpose(0, 2, 1))
        np.testing.assert_array_equal(branch['position_covariance_factor_fc'], factors[:, :3])
        assert np.isfinite(factors).all()
        assert np.all(np.diagonal(branch['covariance_position_fc'], axis1=1, axis2=2) >= 0)
    assert not result['metadata']['monte_carlo_used']


def test_rotating_lever_correction_recovers_stationary_cog():
    data, fit, orientation = fixture(omega_z=.8)
    result = replay(data, fit, orientation)['imu_dead_reckoning']
    np.testing.assert_allclose(result['position_cog_world'], 0., atol=1e-12)
    np.testing.assert_allclose(result['velocity_cog_world'], 0., atol=1e-12)
    np.testing.assert_allclose(result['world_acceleration_cog'], 0., atol=2e-11)
    theta = .8*data['time']; r = data['gyro_sensor_position_in_body']
    expected = np.column_stack((np.cos(theta)*r[0]-np.sin(theta)*r[1],
        np.sin(theta)*r[0]+np.cos(theta)*r[1], np.full(len(theta), r[2])))
    np.testing.assert_allclose(result['position_fc_world'], expected, atol=1e-12)


def test_future_mocap_and_rng_do_not_enter_either_replay(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Monte Carlo is forbidden')
    monkeypatch.setattr(np.random, 'default_rng', forbidden)
    monkeypatch.setattr(np.random, 'normal', forbidden)
    data, fit, orientation = fixture(n=9)
    first = replay(data, fit, orientation)
    data['mocap_position'][:] = 1e50
    data['mocap_quaternion_xyzw'][:] = -12345.
    second = replay(data, fit, orientation)
    for name in ('model', 'imu_dead_reckoning'):
        for key, value in first[name].items():
            np.testing.assert_array_equal(second[name][key], value)


def test_model_sensitivity_factor_matches_physical_parameter_difference():
    data, fit, orientation = fixture(n=15)
    # The first factor block is sqrt(3)*G*S*L, with all 16 modes retained.
    result = replay(data, fit, orientation)['model']
    coordinate = fit['joint_basis'][:14, :13][:, 0]
    h = 2e-5
    branches = []
    for sign in (-1., 1.):
        trial = dict(fit)
        trial['physical_coordinate'] = sign*h*coordinate
        trial['parameters'] = fit['chart'].decode_with_jacobian(trial['physical_coordinate'])[0]
        refresh_fit(data, trial)
        branches.append(replay(data, trial, orientation)['model']['position_fc_world'])
    finite_difference = (branches[1]-branches[0])/(2*h)
    analytic = result['position_covariance_factor_fc'][:, :, 0] / (np.sqrt(3.)*1e-4)
    np.testing.assert_allclose(analytic, finite_difference, rtol=3e-5, atol=1e-9)


def test_thrust_and_initial_gyro_log_lag_sensitivities():
    data, fit, orientation = fixture(n=15)
    # A genuine transient actuator history and nonzero initial gyro derivative
    # exercise both the exact response derivative and parameterised initial ω.
    data['rotor_command_time'] = np.array([-1., -.025, .03, 1.])
    nominal = fit['parameters'].mass*GRAVITY/4
    data['rotor_command'] = np.tile([nominal]*4, (4, 1))
    data['rotor_command'][1:3, 0] *= 1.05
    data['gyro'] = data['time'][:, None] * np.array([.3, -.2, .1])
    data['accel'][:, 0] = .8 * data['time']
    refresh_fit(data, fit)
    result = replay(data, fit, orientation)['model']
    h = 2e-5
    for column, tau_index in ((13, 0), (14, 1), (15, 2)):
        branches = []
        for sign in (-1., 1.):
            trial = dict(fit)
            trial['taus'] = fit['taus'].copy()
            trial['taus'][tau_index] *= np.exp(sign*h)
            refresh_fit(data, trial)
            branches.append(replay(data, trial, orientation)['model']['position_fc_world'])
        finite_difference = (branches[1]-branches[0])/(2*h)
        analytic = result['position_covariance_factor_fc'][:, :, column] / (np.sqrt(3.)*1e-4)
        np.testing.assert_allclose(analytic, finite_difference, rtol=4e-5, atol=2e-9)


def test_rk4_stage_quaternion_hemisphere_crossing_preserves_spin():
    data, fit, orientation = fixture(n=25, omega_z=.8)
    initial_angle = np.pi - .043
    angle = initial_angle + .8*data['time']
    q = np.column_stack((np.cos(.5*angle), np.zeros(len(angle)),
        np.zeros(len(angle)), np.sin(.5*angle)))
    data['initial_quaternion_wxyz'] = q[0]
    orientation['q_output_wxyz'] = q
    result = replay(data, fit, orientation)['model']
    measured = result['quaternion_wxyz']
    sign = np.where(np.sum(measured*q, axis=1) < 0, -1., 1.)
    np.testing.assert_allclose(measured*sign[:, None], q, atol=2e-13)


def test_nonzero_imu_residual_wrench_drives_the_mean_trajectory():
    data, fit, orientation = fixture(n=25)
    acceleration = np.array([.7, -.3, .2])
    data['accel'] += acceleration
    refresh_fit(data, fit)
    result = replay(data, fit, orientation)['model']
    t = data['time']
    expected = data['initial_position'] + .5 * t[:, None]**2 * acceleration
    np.testing.assert_allclose(result['position_fc_world'], expected, atol=2e-13)
    np.testing.assert_allclose(result['applied_residual_wrench_body'][:, :3],
        np.broadcast_to(fit['parameters'].mass * acceleration, (len(t), 3)), atol=3e-12)
    factor = result['applied_residual_wrench_covariance_factor']
    assert factor.shape == (len(t), 6, 22)
    assert np.isfinite(factor).all()


def test_residual_wrench_derivative_keeps_all_shared_physical_and_lag_modes():
    data, fit, orientation = fixture(n=25)
    t = data['time']
    data['gyro'] = np.column_stack((.1 + .3*t, -.2 + .7*t, .05 - .2*t))
    data['accel'][:, 0] = .8*t
    data['accel'][:, 1] = -.4*t
    nominal = fit['parameters'].mass * GRAVITY / 4
    data['rotor_command_time'] = np.array([-1., -.025, .03, 1.])
    data['rotor_command'] = np.tile([nominal]*4, (4, 1))
    data['rotor_command'][1:3, 0] *= 1.05
    refresh_fit(data, fit)
    mean, derivative = _residual_wrench_linearization(data, fit)
    problem = fit['problem']
    coordinate = np.r_[np.zeros(13), np.log(fit['taus'])]
    h = 2e-6
    numerical = np.empty_like(derivative)
    for k in range(16):
        plus, minus = coordinate.copy(), coordinate.copy()
        plus[k] += h; minus[k] -= h
        numerical[:, :, k] = (problem.evaluate(plus)['raw_residual_wrench']
            - problem.evaluate(minus)['raw_residual_wrench']) / (2*h)
    np.testing.assert_allclose(derivative, numerical, rtol=5e-5, atol=8e-9)
    np.testing.assert_array_equal(mean, fit['arrays']['raw_residual_wrench'])
