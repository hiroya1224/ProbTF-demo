"""Separate synthetic IMU inputs for attitude/translation comparison.

The motion is prescribed kinematically, not an aerodynamic plant simulation.
Random draws create measured sensor noise only; no inference samples are used.
All sensors share one calibrated body frame. Known calibration is fixed, and
there is no unmodelled gyro/accelerometer/magnetometer bias in these cases.
"""

import math

import numpy as np
from scipy.spatial.transform import Rotation


SCENARIOS = ('gravity_only', 'smooth_translation', 'sustained_translation')
MAGNETIC_SAMPLE_STRIDE = 3
GRAVITY_WORLD_M_S2 = np.array([0., 0., -9.80665])
_FIELD_DIRECTION = np.array([.8, .2, .56])
MAGNETIC_FIELD_WORLD = 36.459799094961085 * _FIELD_DIRECTION / np.linalg.norm(_FIELD_DIRECTION)


def _covariance(standard_deviations, angles_deg):
    basis = Rotation.from_euler('xyz', angles_deg, degrees=True).as_matrix()
    value = basis @ np.diag(np.asarray(standard_deviations, float)**2) @ basis.T
    return .5 * (value + value.T)


# This new experiment intentionally retains full anisotropy, as requested.
# These are declared assumptions rather than values fitted to comparison truth.
ESTIMATION_COVARIANCES = {
    'angular_velocity': _covariance([.0008, .0010, .0009], [13., -7., 21.]),
    'acceleration': _covariance([.020, .028, .023], [-11., 18., 8.]),
    'magnetic': _covariance([.12, .10, .16], [9., 12., -17.]),
}
GENERATION_COVARIANCES = {
    'angular_velocity': _covariance([.0010, .0014, .0012], [13., -7., 21.]),
    'acceleration': _covariance([.025, .036, .029], [-11., 18., 8.]),
    'magnetic': _covariance([.15, .13, .21], [9., 12., -17.]),
}


def _smooth_step_integrals(time, start, width):
    """A quintic step and its first two time integrals, zero before start."""
    u = np.clip((np.asarray(time) - start) / width, 0., 1.)
    step = 6*u**5 - 15*u**4 + 10*u**3
    first = width * (u**6 - 3*u**5 + 2.5*u**4)
    second = width**2 * (u**7/7 - .5*u**6 + .5*u**5)
    after = np.maximum(np.asarray(time) - start - width, 0.)
    first = first + after
    second = second + .5*width*after + .5*after**2
    return step, first, second


def _translation(time, duration, scenario):
    count = len(time)
    if scenario == 'gravity_only':
        return np.tile([0., 0., 1.5], (count, 1)), np.zeros((count, 3)), np.zeros((count, 3))
    if scenario == 'smooth_translation':
        amplitudes = np.array([1.2, .6, .20])
        frequency = 2*np.pi / np.array([8., 11., 13.])
        phases = time[:, None] * frequency
        position = amplitudes * np.sin(phases)
        position[:, 2] += 1.5
        velocity = amplitudes * frequency * np.cos(phases)
        acceleration = -amplitudes * frequency**2 * np.sin(phases)
        return position, velocity, acceleration
    # A positive acceleration plateau and later equal negative plateau expose
    # the persistent gravity/translation ambiguity rather than hiding it with
    # a truth-driven detector. They start/end smoothly, with analytic integrals.
    width = min(1., .05*duration)
    combined = np.zeros((3, count))
    for start, sign in ((.20*duration, 1.), (.35*duration, -1.),
                        (.60*duration, -1.), (.75*duration, 1.)):
        combined += sign * np.asarray(_smooth_step_integrals(time, start, width))
    direction = np.array([1.4, -.30, .10])
    acceleration = combined[0, :, None] * direction
    velocity = combined[1, :, None] * direction
    position = combined[2, :, None] * direction + [0., 0., 1.5]
    return position, velocity, acceleration


def generate_comparison_data(duration=60., dt=.005, seed=20261106,
                             scenario='smooth_translation'):
    """Return ``{'arrays': arrays, 'metadata': settings}`` for one scenario.

    All scenarios use identical attitude, clocks and realised noise for a given
    duration/dt/seed; only their world translation differs. Measured gyro at k
    describes the right-end increment (k-1,k], and the first ideal gyro is zero.
    Magnetic samples are newly acquired only at k=0,3,6,...; aligned arrays
    retain their original sample stamp/index between these acquisition ticks.
    """
    duration, dt = float(duration), float(dt)
    if not math.isfinite(duration) or duration <= 0.:
        raise ValueError('duration must be finite and positive')
    if not math.isfinite(dt) or dt <= 0.:
        raise ValueError('dt must be finite and positive')
    if scenario not in SCENARIOS:
        raise ValueError('Unknown synthetic scenario: ' + str(scenario))
    intervals = int(round(duration/dt))
    if intervals < 1:
        raise ValueError('duration and dt must supply at least one interval')
    time = np.linspace(0., duration, intervals + 1)
    count = len(time)
    euler_deg = np.column_stack((20.*np.sin(2*np.pi*time/17.),
        15.*np.sin(2*np.pi*time/23.),
        -20. + 30.*np.sin(2*np.pi*time/28.) + 8.*np.sin(2*np.pi*time/11.)))
    rotations = Rotation.from_euler('xyz', euler_deg, degrees=True)
    truth_wxyz = rotations.as_quat()[:, [3, 0, 1, 2]]
    position, velocity, acceleration_world = _translation(time, duration, scenario)
    acceleration_body = rotations.inv().apply(acceleration_world)
    ideal_gyro = np.zeros((count, 3))
    ideal_gyro[1:] = (rotations[:-1].inv()*rotations[1:]).as_rotvec() / np.diff(time)[:, None]
    ideal_acceleration = rotations.inv().apply(acceleration_world - GRAVITY_WORLD_M_S2)
    ideal_magnetic_current = rotations.inv().apply(np.broadcast_to(MAGNETIC_FIELD_WORLD, (count, 3)))
    sample_indices = np.arange(0, count, MAGNETIC_SAMPLE_STRIDE)
    observed = np.zeros(count, dtype=bool); observed[sample_indices] = True
    source_index = np.maximum.accumulate(np.where(observed, np.arange(count), 0))
    rng = np.random.default_rng(seed)
    noise = {}
    for name in ('angular_velocity', 'acceleration', 'magnetic'):
        n = len(sample_indices) if name == 'magnetic' else count
        noise[name] = rng.standard_normal((n, 3)) @ np.linalg.cholesky(GENERATION_COVARIANCES[name]).T
    magnetic_noise = noise['magnetic'][source_index // MAGNETIC_SAMPLE_STRIDE]
    ideal_magnetic = ideal_magnetic_current[source_index]
    arrays = dict(time_seconds=time, measurement_stamps=time.copy(),
        gyro_rad_s=ideal_gyro + noise['angular_velocity'],
        acceleration_m_s2=ideal_acceleration + noise['acceleration'],
        magnetic_raw=ideal_magnetic + magnetic_noise,
        magnetic_observed=observed, magnetic_source_index=source_index,
        magnetic_measurement_stamps=time[source_index],
        magnetic_sample_indices=sample_indices, magnetic_sample_stamps=time[sample_indices],
        truth_wxyz=truth_wxyz, truth_rpy_deg=euler_deg,
        position_m=position, velocity_m_s=velocity,
        acceleration_world_m_s2=acceleration_world, acceleration_body_m_s2=acceleration_body,
        ideal_gyro_rad_s=ideal_gyro, ideal_acceleration_m_s2=ideal_acceleration,
        ideal_magnetic_raw=ideal_magnetic,
        gyro_noise=noise['angular_velocity'], acceleration_noise=noise['acceleration'],
        magnetic_noise=magnetic_noise)
    settings = dict(schema='synthetic-calibrated-translational-imu/v1', scenario=scenario,
        simulation='prescribed kinematic trajectory, not an aerodynamic simulation',
        duration_seconds=duration, requested_dt_seconds=dt,
        effective_dt_seconds=duration/intervals, sample_count=count,
        sensor_sample_rates_hz=dict(gyro=intervals/duration, acceleration=intervals/duration,
                                   magnetic=intervals/duration/MAGNETIC_SAMPLE_STRIDE),
        magnetic_sample_stride=MAGNETIC_SAMPLE_STRIDE,
        magnetic_sample_count=len(sample_indices), seed=int(seed),
        calibration='known and fixed; all readings in common body axes; zero sensor bias, no calibration state',
        gyro_bias_rad_s=[0., 0., 0.], acceleration_bias_m_s2=[0., 0., 0.], magnetic_bias=[0., 0., 0.],
        generation_noise_covariances={k: v.tolist() for k, v in GENERATION_COVARIANCES.items()},
        estimation_noise_covariances={k: v.tolist() for k, v in ESTIMATION_COVARIANCES.items()},
        sensor_noise='independent zero-mean anisotropic Gaussian draws per new sensor acquisition; magnetic draws held at their original stamp',
        noise_units=dict(angular_velocity='rad^2/s^2', acceleration='m^2/s^4', magnetic='microtesla^2'),
        gyro_unit='rad/s', acceleration_unit='m/s^2', magnetic_unit='microtesla',
        gravity_world_m_s2=GRAVITY_WORLD_M_S2.tolist(), magnetic_field_world=MAGNETIC_FIELD_WORLD.tolist(),
        orientation_convention='Hamilton wxyz, body-to-world active rotation, scipy lowercase xyz Euler angles',
        gyro_timing='right-end interval angular velocity; k describes (k-1,k]; ideal first sample zero',
        magnetic_timing='new observation at 0,3,6,... only; held values are not new independent observations',
        initial_condition='common exact initial orientation is allowed; later orientation and translational acceleration are evaluation-only',
        attitude_formula_deg=dict(roll='20 sin(2 pi t/17)', pitch='15 sin(2 pi t/23)',
                                 yaw='-20+30 sin(2 pi t/28)+8 sin(2 pi t/11)'),
        translation_formula=('zero acceleration' if scenario == 'gravity_only' else
            ('position=[1.2 sin(2pi t/8), .6 sin(2pi t/11), 1.5+.2 sin(2pi t/13)]; analytic derivatives'
             if scenario == 'smooth_translation' else
             'a_world=[1.4,-.30,.10]*(step(.20T)-step(.35T)-step(.60T)+step(.75T)); quintic ramps of width min(1,.05T), analytic position/velocity integrals')),
        random_measurement_generation=True, monte_carlo_inference=False,
        old_isotropic_experiment_unchanged=True,
        truth_not_passed_to_estimators='only initial quaternion and known fixed world references may be passed; no ongoing true pose, acceleration or generation covariance')
    return {'arrays': arrays, 'metadata': settings}


def estimator_inputs(dataset):
    """Select the measured/common information; exclude truth and noise draws.

    Returned arrays are owned read-only copies. This boundary makes it possible
    to run all methods on exactly the same measured sequence without permitting
    a filter to consume future truth or generation-side noise settings.
    """
    arrays, settings = dataset['arrays'], dataset['metadata']
    names = ('time_seconds', 'measurement_stamps', 'gyro_rad_s', 'acceleration_m_s2',
             'magnetic_raw', 'magnetic_observed', 'magnetic_source_index',
             'magnetic_measurement_stamps', 'magnetic_sample_indices', 'magnetic_sample_stamps')
    result = {name: np.array(arrays[name], copy=True) for name in names}
    result.update(initial_wxyz=np.array(arrays['truth_wxyz'][0], copy=True),
        gravity_world_m_s2=np.array(settings['gravity_world_m_s2']),
        magnetic_field_world=np.array(settings['magnetic_field_world']),
        gyro_covariance=np.array(settings['estimation_noise_covariances']['angular_velocity']),
        acceleration_covariance=np.array(settings['estimation_noise_covariances']['acceleration']),
        magnetic_covariance=np.array(settings['estimation_noise_covariances']['magnetic']))
    for value in result.values():
        value.setflags(write=False)
    return result
