"""Attitude-feedback reconstruction with the same applied body residual wrench.

The free model supplies *increments*, not absolute observer attitudes. World
translation integrates the actual actuator-plus-residual specific force under
E[R] of the carried attitude law and the same model increment law. Output-only
magnetics may change the current FC lever query, never past CoG translation.

Known parameter/initial/empirical source factors retain their aligned columns.
Pose-history correlations with those sources and across time are bounded by
Minkowski and a four-source Cauchy sum. Only position/velocity marginal upper
bounds are constructed; no exact joint state Gaussian is claimed.
"""
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
for directory in (HERE, HERE.parent, HERE.parent.parent/'src',
                  ROOT/'ros/core/probtf_core/src',
                  ROOT/'ros/examples/probtf_orientation_demo/scripts'):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from imu_replay import _residual_wrench_linearization, _factor, _gram
from imu_model import first_order_zoh_thrust_history
from grape_param_estim.dynamics import FullSixDofPlant
from grape_param_estim.geometry import skew
from grape_param_estim.system import ActuatorState, GRAVITY
from translational_prediction import gyro_increment


_SOURCE_WIDTH = 31
_SOURCE_SLICES = ((0, 16), (16, 25), (25, 31))


def _symmetric(value):
    return .5*(value+value.T)


def _rotation_series(quaternions):
    q = np.asarray(quaternions, float)
    return Rotation.from_quat(q[:, [1, 2, 3, 0]])


def _full_rotation_tensor(moment):
    # M[i+3*j,l+3*m] = E[R_ij R_lm] for column-major vec_F.
    return np.asarray(moment).reshape(3, 3, 3, 3, order='F')


def _rotated_vector_sigma(mean_rotation, rotation_second, vector):
    """Spectral-norm marginal majorant from the supplied full rotation law."""
    mean = np.asarray(mean_rotation) @ vector
    second = np.einsum('iajb,a,b->ij', rotation_second, vector, vector)
    covariance = _symmetric(second-np.outer(mean, mean))
    # This is the declared covariance operator norm, not an eigenvalue floor
    # or a replacement attitude variance. Its square root is a scalar bound.
    return float(np.sqrt(np.linalg.norm(covariance, ord=2)))


def _increment_rotation_moments(rate, covariance, elapsed):
    if elapsed == 0.:
        identity = np.eye(3)
        return identity, np.einsum('ij,lm->ijlm', identity, identity)
    increment = gyro_increment(rate, elapsed, covariance)
    nominal = increment['nominal_rotation']
    second = np.einsum('ac,bd,cjdm->ajbm', nominal, nominal,
                       increment['noise_rotation_second'])
    return increment['mean_rotation'], second


def prepare_coupled_model(data, fit, common_model, options=None):
    """Prepare common angular increments and body profiles once for both laws.

    ``common_model`` is the existing model+applied-residual free reconstruction
    with its aligned 31-column state factor (including its sqrt(3) bound).
    Stage thrust, gimbals, residual means and all 16 shared parameter
    derivatives are constructed once, then read by every attitude branch.
    """
    options = dict(options or {})
    times = np.asarray(data['time'], float)
    if not np.array_equal(times, np.asarray(common_model['time'])) or not np.array_equal(times, fit['arrays']['time']):
        raise ValueError('Coupled replay requires the identical native model/fit/IMU clock')
    if len(times) < 2 or np.any(np.diff(times) <= 0.) or not np.isfinite(times).all():
        raise ValueError('A finite increasing native clock with two rows is required')
    pars = fit['parameters']
    if np.any(np.asarray(pars.linear_drag) != 0.) or np.any(np.asarray(pars.angular_drag) != 0.):
        raise NotImplementedError('Coupled body-profile replay currently supports the declared zero-drag vehicle only')
    state_factor = np.asarray(common_model['state_covariance_factor'], float)
    if state_factor.shape != (len(times), 12, _SOURCE_WIDTH):
        raise ValueError('Common model must retain aligned 16+9+6 source columns')
    rotation_factor = state_factor[:, 3:6]
    rotations = _rotation_series(common_model['quaternion_wxyz'])
    delta_rotations = rotations[:-1].inv()*rotations[1:]
    rate = np.empty((len(times), 3))
    covariance = np.empty((len(times), 3, 3))
    increment_factor = np.zeros((len(times), 3, _SOURCE_WIDTH))
    rate[0] = np.asarray(common_model['angular_velocity_body'])[0]
    covariance[0] = _gram(state_factor[0, 9:12])
    for k, (delta, dt) in enumerate(zip(delta_rotations, np.diff(times)), 1):
        matrix = delta.as_matrix()
        rate[k] = delta.as_rotvec()/dt
        increment_factor[k] = rotation_factor[k]-matrix.T@rotation_factor[k-1]
        covariance[k] = _symmetric(_gram(increment_factor[k])/dt**2)
    prediction = dict(prediction_gyro=rate,
        prediction_gyro_covariance=covariance,
        prediction_increment_factor_source=increment_factor,
        prediction_gyro_source='body model plus applied residual wrench; right increments of free angular carrier, with aligned full source-factor differences')

    _, pj = fit['chart'].decode_with_jacobian(fit['physical_coordinate'])
    basis = np.asarray(fit['joint_basis'])[:14, :13]
    parameter_factor = np.asarray(fit['parameter_covariance_factor_reduced'])
    if parameter_factor.shape != (16, 16):
        raise ValueError('The full 16-mode parameter factor is required')
    lever = np.asarray(data['gyro_sensor_position_in_body'])-pars.cog_offset
    lever_factor = np.zeros((3, _SOURCE_WIDTH))
    lever_factor[:, :16] = np.column_stack((-pj.cog_offset@basis,
                                           np.zeros((3, 3))))@parameter_factor
    dm = np.r_[pj.mass@basis, np.zeros(3)]
    residual, residual_jacobian = _residual_wrench_linearization(data, fit)
    persistent = np.linalg.qr(residual/np.sqrt(len(residual)), mode='reduced')[1].T
    if persistent.shape != (6, 6):
        raise ValueError('The six-column persistent residual source is required')
    maximum_step = float(options.get('integration_max_step_seconds', .005))
    if not np.isfinite(maximum_step) or maximum_step <= 0.:
        raise ValueError('integration_max_step_seconds must be positive and finite')
    nodes = [times[0]]; native_nodes = [0]
    interval_substeps = []
    for left, right in zip(times[:-1], times[1:]):
        count = max(1, int(np.ceil((right-left)/maximum_step)))
        local = np.linspace(left, right, count+1)
        interval_substeps.append(tuple((float(a), float(b)) for a, b in zip(local[:-1], local[1:])))
        nodes.extend(local[1:]); native_nodes.append(len(nodes)-1)
    nodes = np.asarray(nodes)
    stages = np.unique(np.r_[nodes, .5*(nodes[:-1]+nodes[1:])])
    stage_residual = np.column_stack([np.interp(stages, times, residual[:, i]) for i in range(6)])
    stage_residual_j = np.column_stack([np.interp(stages, times, residual_jacobian[:, i, j])
        for i in range(6) for j in range(16)]).reshape(len(stages), 6, 16)
    fit_options = getattr(fit.get('problem'), 'options',
        fit.get('estimate', {}).get('method', {}).get('options', {}))
    history = first_order_zoh_thrust_history(command_times=data['rotor_command_time'],
        command_values=data['rotor_command'], evaluation_times=stages,
        time_constant=float(fit['taus'][0]),
        minimum_thrust=float(fit_options.get('minimum_thrust', 1.5)),
        maximum_thrust=float(fit_options.get('maximum_thrust', 27.6145)))
    gt = np.asarray(data['gimbal_time'])
    if times[0] < gt[0] or times[-1] > gt[-1]:
        raise ValueError('Coupled replay exceeds actual gimbal support')
    gimbals = np.column_stack([np.interp(stages, gt, np.asarray(data['gimbal_angle'])[:, j])
                              for j in range(4)])
    plant = FullSixDofPlant(pars, fit['model'].geometry)
    specific = np.empty((len(stages), 3))
    specific_factor = np.zeros((len(stages), 3, _SOURCE_WIDTH))
    actuator_wrench = np.empty((len(stages), 6))
    for j, (thrust, gimbal) in enumerate(zip(history.thrust, gimbals)):
        actuator = plant.actuator_wrench_evaluation(ActuatorState(thrust, gimbal))
        actuator_wrench[j] = actuator.body_wrench
        jac = actuator.jacobian
        derivative = np.zeros((6, 16))
        derivative[:, :13] = (jac.cog_offset@pj.cog_offset
            + jac.force_effectiveness@pj.force_effectiveness)@basis
        derivative[:, 13] = jac.actual_thrust@history.log_tau_jacobian[j]
        specific[j] = (actuator.body_wrench[:3]+stage_residual[j, :3])/pars.mass
        # Cancel force/mass and actuator/residual dependencies BEFORE applying
        # the shared parameter factor. No independent-force approximation.
        specific_j = (derivative[:3]+stage_residual_j[j, :3]
                      - specific[j, :, None]*dm[None, :])/pars.mass
        specific_factor[j, :, :16] = specific_j@parameter_factor
        specific_factor[j, :, 25:31] = persistent[:3]/pars.mass

    interval_profiles = []
    for k, substeps in enumerate(interval_substeps):
        cached = {}
        interval = []
        for left, right in substeps:
            mid = .5*(left+right)
            query = []
            for stamp in (left, mid, right):
                if stamp not in cached:
                    index = int(np.searchsorted(stages, stamp))
                    mean_delta, second_delta = _increment_rotation_moments(
                        rate[k+1], covariance[k+1], float(stamp-times[k]))
                    cached[stamp] = (index, mean_delta, second_delta)
                query.append(cached[stamp])
            interval.append((right-left, tuple(query)))
        interval_profiles.append(tuple(interval))
    return dict(time=times.copy(), orientation_prediction_options=prediction,
        interval_profiles=tuple(interval_profiles), stage_time=stages,
        stage_specific_force=specific, stage_specific_force_factor=specific_factor,
        stage_actuator_wrench=actuator_wrench, stage_residual_wrench=stage_residual,
        applied_residual_wrench_body=residual.copy(), persistent_wrench_factor=persistent,
        lever=lever, lever_factor=lever_factor,
        angular_velocity_body=np.asarray(common_model['angular_velocity_body']).copy(),
        angular_velocity_factor=state_factor[:, 9:12]/np.sqrt(3.),
        source_width=_SOURCE_WIDTH,
        metadata=dict(schema='imu-first-order-lag/coupled-body-profiles/v1',
            zero_drag_required=True, body_profile_shared_between_branches=True,
            interpolation='same native piecewise-linear residual mean/Jacobian; exact ZOH thrust response and actual-gimbal interpolation at all RK4 stages',
            increment_factor='L_delta[k]=L_rotation[k]-D[k].T L_rotation[k-1]; original aligned31 source columns retained',
            increment_covariance='L_delta L_delta.T / dt^2; first row is original initial model angular-velocity marginal',
            independent_bingham_increment='explicit per-interval approximation of the shared-parameter angular carrier; source factor differences remain saved',
            integration_max_step_seconds=maximum_step))


def _law_moments(orientation, suffix):
    mean = orientation.get('mean_rotation_'+suffix,
                           orientation.get('rotation_mean_'+suffix))
    second = orientation.get('rotation_vector_second_moment_'+suffix,
                             orientation.get('rotation_second_moment_'+suffix))
    if mean is None or second is None:
        raise ValueError('Coupled replay requires full rotation first/second moments from the actual '+suffix+' law')
    return np.asarray(mean), np.asarray(second)


def replay_coupled_model(data, fit, shared, orientation, options=None):
    """Feed carried attitude laws into world translation without past resets."""
    times = shared['time']; count = len(times)
    if not np.array_equal(times, data['time']) or not np.array_equal(times, orientation['time']):
        raise ValueError('Coupled branch must retain the identical original native clock')
    rec_mean, rec_second = _law_moments(orientation, 'recursive')
    out_mean, out_second = _law_moments(orientation, 'output')
    if (rec_mean.shape != (count, 3, 3) or rec_second.shape != (count, 9, 9)
            or out_mean.shape != rec_mean.shape or out_second.shape != rec_second.shape):
        raise ValueError('Native full rotation moment shapes are invalid')
    q_output = np.asarray(orientation['q_output_wxyz'])
    covariance_rotation = np.asarray(orientation['covariance_output'])
    lever = shared['lever']; lever_factor = shared['lever_factor']
    omega = shared['angular_velocity_body']; omega_factor = shared['angular_velocity_factor']
    rotation0 = rec_mean[0]
    body_velocity_lever0 = np.cross(omega[0], lever)
    p = np.asarray(data['initial_position'])-rotation0@lever
    v = np.asarray(data['initial_velocity'])-rotation0@body_velocity_lever0
    p_factor = -rotation0@lever_factor
    v_factor = (rotation0@skew(lever)@omega_factor[0]
                - rotation0@skew(omega[0])@lever_factor)
    v_factor[:, 19:22] += _factor(data['initial_velocity_covariance'])
    initial_rot_second = _full_rotation_tensor(rec_second[0])
    initial_position_pose_sigma = _rotated_vector_sigma(rotation0, initial_rot_second, lever)
    initial_velocity_pose_sigma = _rotated_vector_sigma(rotation0, initial_rot_second, body_velocity_lever0)
    integrated_pose_p = integrated_pose_v = 0.
    p_fc = np.empty((count, 3)); v_fc = np.empty_like(p_fc)
    p_cog = np.empty_like(p_fc); v_cog = np.empty_like(p_fc)
    known_p = np.empty((count, 3, _SOURCE_WIDTH)); known_v = np.empty_like(known_p)
    factor_p = np.empty((count, 3, _SOURCE_WIDTH+3)); factor_v = np.empty_like(factor_p)
    pose_p_sigma = np.empty(count); pose_v_sigma = np.empty(count)
    acceleration_world = np.empty_like(p_fc)
    gravity = np.array([0., 0., -GRAVITY])

    for k in range(count):
        if k:
            # Integrate interval(k-1,k] with the carried law at k-1 only.
            # The measurement at k has not acted on any past stage.
            root_mean = rec_mean[k-1]
            root_second = _full_rotation_tensor(rec_second[k-1])
            for dt, stages in shared['interval_profiles'][k-1]:
                means, sources, sigmas = [], [], []
                for index, delta_mean, delta_second in stages:
                    rotation_mean = root_mean@delta_mean
                    rotation_second = np.einsum('ialb,ajbm->ijlm', root_second, delta_second)
                    force = shared['stage_specific_force'][index]
                    means.append(rotation_mean@force+gravity)
                    sources.append(rotation_mean@shared['stage_specific_force_factor'][index])
                    sigmas.append(_rotated_vector_sigma(rotation_mean, rotation_second, force))
                a0, am, a1 = means; l0, lm, l1 = sources
                p = p + dt*v + dt**2/6.*(a0+2.*am)
                v = v + dt/6.*(a0+4.*am+a1)
                p_factor = p_factor + dt*v_factor + dt**2/6.*(l0+2.*lm)
                v_factor = v_factor + dt/6.*(l0+4.*lm+l1)
                integrated_pose_p += dt*integrated_pose_v + dt**2/6.*(sigmas[0]+2.*sigmas[1])
                integrated_pose_v += dt/6.*(sigmas[0]+4.*sigmas[1]+sigmas[2])
            acceleration_world[k] = means[-1]
        else:
            first_index = int(np.searchsorted(shared['stage_time'], times[0]))
            acceleration_world[0] = rec_mean[0]@shared['stage_specific_force'][first_index]+gravity
        p_cog[k], v_cog[k] = p, v
        p_fc[k] = p+out_mean[k]@lever
        v_fc[k] = v+out_mean[k]@np.cross(omega[k], lever)
        known_p[k] = p_factor+out_mean[k]@lever_factor
        known_v[k] = (v_factor-out_mean[k]@skew(lever)@omega_factor[k]
                      + out_mean[k]@skew(omega[k])@lever_factor)
        out_tensor = _full_rotation_tensor(out_second[k])
        elapsed = times[k]-times[0]
        current_position_pose_sigma = _rotated_vector_sigma(out_mean[k], out_tensor, lever)
        current_velocity_pose_sigma = _rotated_vector_sigma(out_mean[k], out_tensor, np.cross(omega[k], lever))
        pose_p_sigma[k] = (integrated_pose_p+elapsed*initial_velocity_pose_sigma
                           +initial_position_pose_sigma+current_position_pose_sigma)
        pose_v_sigma[k] = integrated_pose_v+initial_velocity_pose_sigma+current_velocity_pose_sigma
        if k == 0:
            # The very same initial lever variable occurs with both signs.
            # Cancel it before any unknown-time/source bound is introduced.
            p_fc[0] = data['initial_position']; v_fc[0] = data['initial_velocity']
            known_p[0] = 0.; pose_p_sigma[0] = 0.; pose_v_sigma[0] = 0.
            known_v[0] = 0.; known_v[0, :, 19:22] = _factor(data['initial_velocity_covariance'])
        # Four source groups: parameter16, initial9, persistent6, pose-history.
        # Known within-group cross terms were combined before this 4-sum bound.
        factor_p[k, :, :_SOURCE_WIDTH] = 2.*known_p[k]
        factor_p[k, :, _SOURCE_WIDTH:] = 2.*pose_p_sigma[k]*np.eye(3)
        factor_v[k, :, :_SOURCE_WIDTH] = 2.*known_v[k]
        factor_v[k, :, _SOURCE_WIDTH:] = 2.*pose_v_sigma[k]*np.eye(3)
    if not all(np.isfinite(a).all() for a in (p_fc, v_fc, factor_p, factor_v)):
        raise FloatingPointError('Coupled reconstruction produced a non-finite mean or covariance factor')
    return dict(time=times.copy(), position_fc_world=p_fc, velocity_fc_world=v_fc,
        position_cog_world=p_cog, velocity_cog_world=v_cog,
        quaternion_wxyz=q_output.copy(), angular_velocity_body=omega.copy(),
        covariance_rotation=covariance_rotation.copy(),
        covariance_position_fc=factor_p@factor_p.transpose(0, 2, 1),
        position_covariance_factor_fc=factor_p,
        covariance_velocity_fc=factor_v@factor_v.transpose(0, 2, 1),
        velocity_covariance_factor_fc=factor_v,
        position_known_source_factor_fc=known_p, velocity_known_source_factor_fc=known_v,
        position_pose_history_sigma_bound=pose_p_sigma, velocity_pose_history_sigma_bound=pose_v_sigma,
        world_acceleration_cog=acceleration_world,
        applied_residual_wrench_body=shared['applied_residual_wrench_body'].copy(),
        prediction_gyro=shared['orientation_prediction_options']['prediction_gyro'].copy(),
        prediction_gyro_covariance=shared['orientation_prediction_options']['prediction_gyro_covariance'].copy(),
        prediction_increment_factor_source=shared['orientation_prediction_options']['prediction_increment_factor_source'].copy(),
        metadata=dict(schema='imu-first-order-lag/coupled-attitude-feedback-replay/v1',
            mean_dynamics='actual (F_act+F_res)/mass rotated by E[R_recursive] E[D_model(s)], plus world gravity; no posterior IMU acceleration substitution',
            attitude_prediction='right model increment acts on current carried attitude law; absolute free model attitude never resets the observer',
            stage_clock='RK4 stages on original IMU intervals; endpoint observation does not act backward on the interval',
            magnetic_policy='only current output quaternion and FC lever query use magnetic output; CoG p/v and next carried law exclude it',
            known_source_columns={'parameter':[0,16], 'initial':[16,25], 'persistent_wrench':[25,31]},
            source_factors='unscaled aligned known columns; force/mass, actuator/residual and lever cancellation performed before propagation',
            position_covariance='4*(parameter-block Gram + initial-block Gram + persistent-block Gram + pose-history spectral/Minkowski majorant)',
            orientation_covariance='actual output attitude-law proxy supplied by orientation adapter, not inflated by position Cauchy bound',
            joint_state_gaussian_claimed=False, monte_carlo=False, ongoing_mocap_used=False,
            full_rotation_moments_used=True, zero_drag_required=True,
            independent_increment_approximation=shared['metadata']['independent_bingham_increment']))
