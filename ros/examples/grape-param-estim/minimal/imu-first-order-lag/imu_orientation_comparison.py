"""Independent real-IMU attitude comparison with output-only magnetics.

The historical gravity-only implementation is reused unchanged.  The second
branch carries the joint attitude/body-acceleration law from the standalone
translation experiment, but only its gyro+acceleration posterior recurs here.
Every magnetic output is a separate query and cannot change either the next
attitude prior or the conditional acceleration parameters.
"""
from __future__ import annotations

from pathlib import Path
import sys
import time as wall_time

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
SCRIPTS = ROOT / 'ros/examples/probtf_orientation_demo/scripts'
for path in (HERE, ROOT / 'ros/core/probtf_core/src', SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from imu_orientation import estimate_orientation, _isotropic_sensor_covariance, _prediction_inputs
from imu_signal import local_polynomial_signal, corrected_lag_signal
from probtf.bingham import (bingham_second_moment, bingham_fourth_moment,
                           rotation_first_moment, rotation_vector_second_moment)
from probtf.geometry import quat_to_rotmat
from translational_attitude_law import ExponentialAttitudeLaw, projective_basis
from translational_joint_filter import JointTranslationAttitudeFilter
from translational_observation_factors import magnetic_factor


def _continuous_principal_quaternion(second, previous):
    quaternion = np.linalg.eigh(.5 * (second + second.T))[1][:, -1]
    if quaternion @ previous < 0.:
        quaternion = -quaternion
    return quaternion


def _full_law_queries(integral, previous):
    """Query the same full law; no Bingham fit or A-only covariance query.

    The right-body covariance uses delta_theta ~= 2 E(q_ref).T q.  Its
    antipodal-even second moment is 4 E.T Mqq E.  This is an explicitly
    high-concentration coordinate proxy, not an exact rotation-log covariance
    or a separately assigned Gaussian attitude law.
    """
    quaternion = _continuous_principal_quaternion(integral['Mqq'], previous)
    tangent = projective_basis(quaternion)
    covariance = 4. * tangent.T @ integral['Mqq'] @ tangent
    vector_rotation = integral['R'].transpose(0, 2, 1).reshape(-1, 9)
    second_rotation = (vector_rotation.T * integral['weights']) @ vector_rotation
    return dict(quaternion=quaternion, covariance=.5 * (covariance + covariance.T),
                mean_rotation=integral['mean_R'].copy(),
                rotation_second_moment=.5 * (second_rotation + second_rotation.T))


def _add_rotation_aliases(sequence):
    for suffix in ('recursive', 'output'):
        sequence['rotation_mean_' + suffix] = sequence['mean_rotation_' + suffix]
        sequence['rotation_second_moment_' + suffix] = sequence['rotation_vector_second_moment_' + suffix]
        sequence['right_rotation_covariance_' + suffix] = sequence['covariance_' + suffix]


def _attach_gravity_moments(sequence, integration_steps):
    """Full rotation moments from the historical branch's actual Bingham."""
    count = len(sequence['time'])
    for suffix in ('recursive', 'output'):
        sequence['mean_rotation_' + suffix] = np.empty((count, 3, 3))
        sequence['rotation_vector_second_moment_' + suffix] = np.empty((count, 9, 9))
        sequence['Wg_' + suffix] = np.zeros((count, 3, 3))
        sequence['Wb_' + suffix] = np.zeros((count, 3, 3))
    for k in range(count):
        for suffix in ('recursive', 'output'):
            if suffix == 'output' and np.array_equal(sequence['A_recursive'][k], sequence['A_output'][k]):
                sequence['mean_rotation_output'][k] = sequence['mean_rotation_recursive'][k]
                sequence['rotation_vector_second_moment_output'][k] = sequence['rotation_vector_second_moment_recursive'][k]
                continue
            parameter = sequence['A_' + suffix][k]
            second = bingham_second_moment(parameter, integration_steps=integration_steps)
            fourth = bingham_fourth_moment(parameter, integration_steps=integration_steps)
            sequence['mean_rotation_' + suffix][k] = rotation_first_moment(second)
            sequence['rotation_vector_second_moment_' + suffix][k] = rotation_vector_second_moment(fourth)
    _add_rotation_aliases(sequence)
    sequence['metadata'].update(
        comparison_role='historical gravity-only acceleration evidence; implementation unchanged',
        full_rotation_moments='same native A law; E[R] and E[vec_F(R) vec_F(R).T]',
        rotation_vectorization='column-major, vec_F(R)=[R00,R10,R20,R01,...,R22]',
        full_rotation_moment_integration_steps=int(integration_steps))


def _prepare_signals(data, gyro_tau, accel_tau, lag_covariance, options):
    stamps = np.asarray(data['time'], float)
    static = {name: _isotropic_sensor_covariance(data[name + '_noise_covariance'])
              for name in ('gyro', 'accel', 'mag')}
    signals = {name: local_polynomial_signal(stamps, data[name], static[name],
        window_seconds=float(options.get('window_seconds', .08)),
        degree=int(options.get('degree', 3))) for name in ('gyro', 'accel')}
    gyro = corrected_lag_signal(signals['gyro'], gyro_tau, lag_covariance[0, 0])
    accel = corrected_lag_signal(signals['accel'], accel_tau, lag_covariance[1, 1])
    cross = lag_covariance[0, 1] * np.einsum('ni,nj->nij',
        signals['gyro']['derivative'], signals['accel']['derivative'])
    common = dict(time=stamps.copy(), corrected_gyro=gyro['mean'], corrected_accel=accel['mean'],
        corrected_gyro_covariance=gyro['covariance'], corrected_accel_covariance=accel['covariance'],
        corrected_gyro_accel_cross_covariance=cross,
        gyro_tau_sensitivity=signals['gyro']['derivative'],
        accel_tau_sensitivity=signals['accel']['derivative'],
        lag_covariance_2x2=lag_covariance.copy(),
        signal_operators={name: value['operators'] for name, value in signals.items()},
        corrected_gyro_operator=gyro['operator'], corrected_accel_operator=accel['operator'])
    prediction, prediction_source, model_prediction = _prediction_inputs(options, gyro)
    common.update(prediction)
    for key in ('imu_stamp_abs', 'imu_stamp_ns', 'imu_bag_time_abs', 'imu_bag_time_ns',
                'imu_source_index', 'imu_selected_record_index', 'mag_observed',
                'magnetic_source_index', 'magnetic_measurement_stamps'):
        if key in data:
            common[key] = np.asarray(data[key]).copy()
    return common, static, prediction_source, model_prediction


def _translation_sequence(data, common, static, initial_parameter, options,
                          prediction_source, model_prediction):
    stamps = common['time']; count = len(stamps)
    initial = np.asarray(data['initial_quaternion_wxyz'], float)
    initial = initial / np.linalg.norm(initial)
    magnitude = float(options.get('gravity_magnitude', 9.80665))
    if not np.isfinite(magnitude) or magnitude <= 0.:
        raise ValueError('Physical gravity magnitude must be positive and finite')
    # New y=a-R.T*g model uses physical downward gravity.  The historical
    # gravity evidence instead uses the upward apparent-gravity reference.
    physical_gravity = np.array([0., 0., -magnitude])
    magnetic = np.asarray(data.get('mag', np.zeros((count, 3))), float)
    observed = np.asarray(data.get('mag_observed', np.zeros(count, bool)), bool)
    if magnetic.shape != (count, 3) or observed.shape != (count,):
        raise ValueError('Magnetic measurements and acquisition mask must share the original IMU rows')
    valid = observed & np.isfinite(magnetic).all(axis=1) & (np.linalg.norm(magnetic, axis=1) > 0.)
    reference = np.asarray(data.get('initial_magnetic_reference_world', np.zeros(3)), float)
    available = reference.shape == (3,) and np.isfinite(reference).all() and np.linalg.norm(reference) > 0.
    if not available and valid[0]:
        reference = quat_to_rotmat(initial) @ magnetic[0]
        available = True
    # With no magnetic reference, this fixed vector defines an unused quartic
    # family basis only. It never creates a magnetic observation or factor.
    law_reference = reference if available else np.array([1., 0., 0.])
    concentration = float(options.get('initial_concentration', 100.))
    estimator = JointTranslationAttitudeFilter(initial, physical_gravity, law_reference,
        common['prediction_gyro_covariance'][0], common['corrected_accel_covariance'][0],
        2. * static['mag'], use_magnetic=False, estimate_translation=True,
        quadrature_order=int(options.get('quadrature_order', 7)),
        projection_iterations=int(options.get('projection_iterations', 3)),
        initial_attitude_std_deg=float(np.rad2deg(np.sqrt(2. / concentration))),
        initial_acceleration_std=float(options.get('initial_acceleration_std', .3)),
        acceleration_time_constant=float(options.get('acceleration_time_constant', .7)),
        acceleration_stationary_std=options.get('acceleration_stationary_std', (1.5, 1.5, 1.5)),
        quadrature_options=options.get('quadrature_options'))
    # Copy the actual historical initial A, including its harmless scalar
    # gauge, rather than introducing a slightly different finite prior.
    estimator.law = ExponentialAttitudeLaw(initial_parameter, np.zeros((3, 3)),
        np.zeros((3, 3)), physical_gravity, law_reference)
    estimator.reference = initial.copy()
    estimator.proposal = estimator.law.proposal_covariance(initial)
    estimator.moments = estimator.law.integrate(initial, estimator.proposal, estimator.order,
                                              **estimator.quadrature_options)
    sequence = {}
    for suffix in ('recursive', 'output'):
        for key, shape in (('A', (4, 4)), ('Wg', (3, 3)), ('Wb', (3, 3)),
                           ('conditional_m', (3,)), ('conditional_F', (3, 3)),
                           ('conditional_S', (3, 3)), ('q', (4,)), ('covariance', (3, 3)),
                           ('mean_rotation', (3, 3)), ('rotation_vector_second_moment', (9, 9))):
            name = 'q_' + suffix + '_wxyz' if key == 'q' else key + '_' + suffix
            sequence[name] = np.empty((count,) + shape)
    sequence['acceleration_updated'] = np.zeros(count, bool)
    sequence['magnetic_updated'] = np.zeros(count, bool)
    sequence['projection_moment_residual'] = np.zeros(count)
    sequence['projection_initial_moment_residual'] = np.zeros(count)
    for suffix in ('recursive','output'):
        sequence['integration_feature_error_' + suffix] = np.zeros(count)
        sequence['integration_log_normalizer_error_' + suffix] = np.zeros(count)
        sequence['integration_dense_rule_' + suffix] = np.zeros(count, bool)
    sequence['projection_integration_feature_error'] = np.zeros(count)
    sequence['projection_integration_log_normalizer_error'] = np.zeros(count)
    previous_recursive = initial.copy(); previous_output = initial.copy()
    progress = options.get('progress'); started = wall_time.monotonic()
    for k in range(count):
        if k:
            estimator.gyro_covariance = common['prediction_gyro_covariance'][k]
            estimator.acceleration_covariance = common['corrected_accel_covariance'][k]
            result = estimator.update(common['prediction_gyro'][k], common['corrected_accel'][k],
                float(stamps[k] - stamps[k - 1]), magnetic=None)
            sequence['acceleration_updated'][k] = True
            diagnostic = result['diagnostics'].get('projection', {})
            sequence['projection_moment_residual'][k] = diagnostic.get('moment_residual', 0.)
            sequence['projection_initial_moment_residual'][k] = diagnostic.get('initial_moment_residual', 0.)
            projection_quadrature = diagnostic.get('normalizer_numerical_refinement') or {}
            error = projection_quadrature.get('final_discrepancy', {})
            sequence['projection_integration_feature_error'][k] = error.get('feature_moments', 0.)
            sequence['projection_integration_log_normalizer_error'][k] = error.get('log_normalizer', 0.)
        recursive_law = estimator.law
        recursive_integral = estimator.moments
        recursive_query = _full_law_queries(recursive_integral, previous_recursive)
        output_law = recursive_law; output_query = recursive_query
        output_integral = recursive_integral
        if k and available and valid[k] and np.trace(static['mag']) > 0.:
            # Both the original reference reading and this new reading carry
            # individual-reading noise. Original static sensor noise is
            # isotropic, hence the sum remains 2*Sigma_mag in body axes.
            factor = magnetic_factor(magnetic[k], 2. * static['mag'], reference)
            output_law = ExponentialAttitudeLaw(recursive_law.A + factor['A'],
                recursive_law.Wg, recursive_law.Wb + factor['Wb'], physical_gravity, law_reference)
            output_mode = output_law.mode(recursive_query['quaternion'])
            output_proposal = output_law.proposal_covariance(output_mode, fallback=estimator.proposal)
            output_integral = output_law.integrate(output_mode, output_proposal, estimator.order,
                                                  **estimator.quadrature_options)
            output_query = _full_law_queries(output_integral, previous_output)
            sequence['magnetic_updated'][k] = True
        for suffix, law_value, query, integral in (
                ('recursive', recursive_law, recursive_query, recursive_integral),
                ('output', output_law, output_query, output_integral)):
            for key in ('A', 'Wg', 'Wb'):
                sequence[key + '_' + suffix][k] = getattr(law_value, key)
            for key in ('m', 'F', 'S'):
                sequence['conditional_' + key + '_' + suffix][k] = getattr(estimator, key)
            sequence['q_' + suffix + '_wxyz'][k] = initial if k == 0 else query['quaternion']
            sequence['covariance_' + suffix][k] = query['covariance']
            sequence['mean_rotation_' + suffix][k] = query['mean_rotation']
            sequence['rotation_vector_second_moment_' + suffix][k] = query['rotation_second_moment']
            refinement = integral.get('numerical_refinement', {})
            error = refinement.get('final_discrepancy', {})
            sequence['integration_feature_error_' + suffix][k] = error.get('feature_moments', 0.)
            sequence['integration_log_normalizer_error_' + suffix][k] = error.get('log_normalizer', 0.)
            sequence['integration_dense_rule_' + suffix][k] = refinement.get('dense_rule_used', False)
        previous_recursive = sequence['q_recursive_wxyz'][k]
        previous_output = sequence['q_output_wxyz'][k]
        if progress is not None and (k + 1 == count or (k + 1) % 200 == 0):
            progress('translation_aware', k + 1, count, wall_time.monotonic() - started)
    sequence.update(common)
    _add_rotation_aliases(sequence)
    sequence['metadata'] = dict(method='joint attitude/body-acceleration law; gyro+acceleration recurrence, output-only magnetic factor',
        initial_concentration=concentration, initial_observation_updates_skipped=True,
        initial_A_source='exact copy of historical gravity-only branch initial A',
        initial_quaternion_wxyz=initial.tolist(), only_initial_mocap_used=True,
        physical_gravity_world=physical_gravity.tolist(),
        observation_model='y_accel=a_body-R(q).T*g_physical+noise',
        magnetic_reference_available=bool(available),
        magnetic_reference_world=reference.tolist() if available else None,
        magnetic_observation_policy='actual-new-observation mask only; independent output law; no law/m/F/S/moments/reference/proposal feedback',
        magnetic_noise='2*trace(S_static)/3 I, current and initial-reference individual-reading noise',
        carried_law='p_theta(q)*N(a_body;m+F*R(q).T*g_physical,S); quadratic plus all quartic coefficients',
        attitude_prediction='shared independent Bingham gyro increment; joint required moments; finite 19-coefficient forward-KL projection',
        gyro_prediction_source=prediction_source, explicit_model_increment_used=model_prediction,
        prediction_action='right model increment acts on the current gyro+acceleration posterior; no reset to the absolute free-model quaternion',
        prediction_cross_time_dependence='aligned full source factor retained when supplied; conversion of each marginal increment covariance to an independent Bingham transition is an explicit filter approximation, not an exact shared-parameter trajectory posterior',
        acceleration_time_constant_seconds=estimator.acceleration_time_constant,
        initial_acceleration_std=float(options.get('initial_acceleration_std', .3)),
        acceleration_stationary_std=np.sqrt(np.diag(estimator.stationary_acceleration_covariance)).tolist(),
        acceleration_prior='declared AR(1) body-frame motion prior; not model-wrench acceleration feedback',
        quadrature_order=estimator.order, projection_iterations=estimator.projection_iterations,
        quadrature_options=estimator.quadrature_options,
        integration='deterministic spatial-multimode GH plus Haar MIS; no Monte Carlo',
        rotation_vectorization='column-major, vec_F(R)=[R00,R10,R20,R01,...,R22]',
        rotation_covariance='4*E(q_ref).T*Mqq*E(q_ref), right-body quaternion half-angle high-concentration second-moment proxy from full law; not modal-curvature covariance',
        selected_quaternion='principal axis of full Mqq with continuous hemisphere',
        full_rotation_moments='E[R] and E[vec_F(R)vec_F(R).T] queried from identical full law once and cached',
        static_sensor_noise='trace(S)/3 I, same declared real-data calibration as historical branch',
        corrected_noise='existing lag inversion, full physical tau covariance, smooth/derivative covariance and sparse overlap operators retained',
        corrected_noise_dependence='per-row gyro/accel laws used in recursion; shared fitted-tau and overlapping-window dependence remains an explicit approximation; cross and full operators retained for external conservative uncertainty propagation',
        wall_seconds=wall_time.monotonic() - started)
    return sequence


def estimate_orientation_comparison(data, gyro_tau, accel_tau,
                                    lag_covariance_2x2=None, options=None):
    """Return independent translation-aware and historical orientation tracks.

    ``options['progress']``, if supplied, receives (branch, done, total,
    elapsed_seconds). Original input clocks/indices/magnetic acquisition masks
    are copied, never resampled. No continuing mocap measurement is read.
    """
    options = dict(options or {})
    lag = np.zeros((2, 2)) if lag_covariance_2x2 is None else np.asarray(lag_covariance_2x2, float)
    if lag.shape != (2, 2) or not np.isfinite(lag).all():
        raise ValueError('Sensor lag covariance must be finite 2x2 in physical seconds squared')
    common, static, prediction_source, model_prediction = _prepare_signals(data, gyro_tau, accel_tau, lag, options)
    historical_options = dict(options)
    progress = options.get('progress')
    if progress is not None:
        historical_options['progress'] = lambda done, total, elapsed: progress('gravity_only', done, total, elapsed)
    gravity = estimate_orientation(data, gyro_tau, accel_tau, lag, historical_options)
    gravity.update(common)
    _attach_gravity_moments(gravity, int(options.get('rotation_moment_steps', 200)))
    translation = _translation_sequence(data, common, static, gravity['A_recursive'][0], options,
        prediction_source, model_prediction)
    return dict(translation_aware=translation, gravity_only=gravity,
        metadata=dict(schema='imu-first-order-lag/orientation-comparison/v1',
            native_clock_preserved=True, static_sensor_covariance='same isotropic trace(S)/3 I for both real-data branches',
            original_method_unchanged=True, old_publication_unchanged=True,
            both_skip_initial_observations=True, magnetic_output_only_in_both_branches=True,
            full_rotation_second_moment_order='column-major vec_F(R)',
            translation_aware_covariance='full-law quaternion-half-angle proxy',
            gravity_only_covariance='existing native Bingham modal-curvature proxy unchanged'))
