"""Generate a kinematic flight with explicit isotropic sensor-noise models.

The trajectory is prescribed; this is not an aerodynamic simulation. Random
draws generate sensor measurements only. They are not estimator integration
or a Monte Carlo approximation of any candidate distribution.
"""

import numpy as np
from scipy.spatial.transform import Rotation


ESTIMATION_VARIANCES = dict(
    angular_velocity=4.493306602258716e-7,
    acceleration=3.711202641785479e-5,
    magnetic=0.13428656459644708,
)
GENERATION_VARIANCE_RATIOS = dict(angular_velocity=4.0, acceleration=4.0, magnetic=2.25)
MAGNETIC_SAMPLE_STRIDE = 3
GYRO_BIAS_RAD_S = np.array([0.0, 0.0, 0.004])
GRAVITY_WORLD_M_S2 = np.array([0.0, 0.0, -9.80665])
MAGNETIC_REFERENCE_MAGNITUDE = 36.459799094961085


def generate_flight(duration_s=60.0, sample_rate_hz=200.0, rng=None):
    """Return measurements, saved noise draws, truth and JSON-safe settings.

    Samples include both trajectory endpoints. The requested number of
    intervals is rounded to the nearest integer and distributed uniformly
    over the requested duration; the effective rate is recorded separately.
    Angular velocity at sample k acts on (k-1, k], in body coordinates.
    Magnetic readings arrive every three samples. Their aligned arrays hold
    the last reading and its original source index/stamp between arrivals.
    """
    duration = float(duration_s)
    rate = float(sample_rate_hz)
    if not np.isfinite(duration) or duration <= 0.0:
        raise ValueError('duration_s must be finite and positive.')
    if not np.isfinite(rate) or rate <= 0.0:
        raise ValueError('sample_rate_hz must be finite and positive.')
    intervals = int(round(duration * rate))
    if intervals < 1:
        raise ValueError('The duration and rate must provide at least one interval.')
    generator = np.random.default_rng() if rng is None else rng
    times = np.linspace(0.0, duration, intervals + 1)
    count = len(times)

    roll = 20.0 * np.sin(2.0 * np.pi * times / 17.0)
    pitch = 15.0 * np.sin(2.0 * np.pi * times / 23.0)
    yaw = (-20.0 + 30.0 * np.sin(2.0 * np.pi * times / 28.0)
           + 8.0 * np.sin(2.0 * np.pi * times / 11.0))
    rotations = Rotation.from_euler('xyz', np.column_stack((roll, pitch, yaw)), degrees=True)
    xyzw = rotations.as_quat()
    truth_wxyz = xyzw[:, [3, 0, 1, 2]]

    wx, wy, wz = 2.0 * np.pi / np.array([70.0, 80.0, 90.0])
    position = np.column_stack((
        0.8 * np.sin(wx * times),
        0.5 * (1.0 - np.cos(wy * times)),
        1.5 + 0.12 * np.sin(wz * times),
    ))
    velocity = np.column_stack((
        0.8 * wx * np.cos(wx * times),
        0.5 * wy * np.sin(wy * times),
        0.12 * wz * np.cos(wz * times),
    ))
    acceleration_world = np.column_stack((
        -0.8 * wx * wx * np.sin(wx * times),
        0.5 * wy * wy * np.cos(wy * times),
        -0.12 * wz * wz * np.sin(wz * times),
    ))
    ideal_gyro = np.zeros((count, 3))
    ideal_gyro[1:] = ((rotations[:-1].inv() * rotations[1:]).as_rotvec()
                      / np.diff(times)[:, None])
    specific_force_world = acceleration_world - GRAVITY_WORLD_M_S2
    ideal_acceleration = rotations.inv().apply(specific_force_world)
    field_direction = np.array([0.8, 0.2, 0.56])
    field_direction /= np.linalg.norm(field_direction)
    field_world = MAGNETIC_REFERENCE_MAGNITUDE * field_direction
    magnetic_sample_indices = np.arange(0, count, MAGNETIC_SAMPLE_STRIDE)
    magnetic_observed = np.zeros(count, dtype=bool)
    magnetic_observed[magnetic_sample_indices] = True
    magnetic_source_index = np.maximum.accumulate(np.where(magnetic_observed, np.arange(count), 0))
    ideal_magnetic_at_ticks = rotations.inv().apply(np.broadcast_to(field_world, (count, 3)))
    ideal_magnetic = ideal_magnetic_at_ticks[magnetic_source_index]

    generation_variances = {sensor: variance * GENERATION_VARIANCE_RATIOS[sensor]
                            for sensor, variance in ESTIMATION_VARIANCES.items()}
    draws = {sensor: np.asarray(generator.normal(0.0, np.sqrt(variance),
                       size=(len(magnetic_sample_indices) if sensor == 'magnetic' else count, 3)), dtype=float)
             for sensor, variance in generation_variances.items()}
    for sensor, draw in draws.items():
        expected_count = len(magnetic_sample_indices) if sensor == 'magnetic' else count
        if draw.shape != (expected_count, 3) or not np.isfinite(draw).all():
            raise ValueError('Sensor-noise generation must return finite (sensor sample count,3) arrays.')
    draws['magnetic'] = draws['magnetic'][magnetic_source_index // MAGNETIC_SAMPLE_STRIDE]
    arrays = dict(
        time_seconds=times,
        measurement_stamps=times.copy(),
        gyro_rad_s=ideal_gyro + GYRO_BIAS_RAD_S + draws['angular_velocity'],
        acceleration_m_s2=ideal_acceleration + draws['acceleration'],
        magnetic_raw=ideal_magnetic + draws['magnetic'],
        magnetic_observed=magnetic_observed,
        magnetic_source_index=magnetic_source_index,
        magnetic_measurement_stamps=times[magnetic_source_index],
        magnetic_sample_indices=magnetic_sample_indices,
        magnetic_sample_stamps=times[magnetic_sample_indices],
        truth_wxyz=truth_wxyz,
        position_m=position,
        velocity_m_s=velocity,
        acceleration_world_m_s2=acceleration_world,
        ideal_gyro_rad_s=ideal_gyro,
        ideal_acceleration_m_s2=ideal_acceleration,
        ideal_magnetic_raw=ideal_magnetic,
        gyro_noise=draws['angular_velocity'],
        acceleration_noise=draws['acceleration'],
        magnetic_noise=draws['magnetic'],
    )
    settings = dict(
        schema='synthetic-kinematic-flight/v2',
        simulation='prescribed kinematic trajectory; not an aerodynamic simulation',
        duration_s=duration,
        sample_rate_hz=rate,
        effective_sample_rate_hz=intervals / duration,
        sample_count=count,
        sensor_sample_rates_hz=dict(angular_velocity=intervals/duration,
                                    acceleration=intervals/duration,
                                    magnetic=intervals/duration/MAGNETIC_SAMPLE_STRIDE),
        magnetic_sample_stride=MAGNETIC_SAMPLE_STRIDE,
        magnetic_sample_count=len(magnetic_sample_indices),
        magnetic_sample_timing='new observation at indices 0,3,6,...; otherwise hold raw value, ideal and noise at original source stamp; never fuse held readings again',
        sample_timing='both endpoints; right-end body angular velocity for each interval; first ideal gyro is zero',
        generation_noise_variances=generation_variances,
        estimation_noise_variances=dict(ESTIMATION_VARIANCES),
        generation_noise_covariances={sensor: (variance * np.eye(3)).tolist()
                                      for sensor, variance in generation_variances.items()},
        estimation_noise_covariances={sensor: (variance * np.eye(3)).tolist()
                                      for sensor, variance in ESTIMATION_VARIANCES.items()},
        generation_to_estimation_variance_ratios=dict(GENERATION_VARIANCE_RATIOS),
        sensor_noise='independent zero-mean isotropic Gaussian draws per newly published sensor observation; aligned magnetic arrays hold previous draws between arrivals',
        noise_covariance_units=dict(angular_velocity='rad^2/s^2', acceleration='m^2/s^4', magnetic='raw^2'),
        gyro_bias_rad_s=GYRO_BIAS_RAD_S.tolist(),
        gyro_bias_subtracted=False,
        saved_noise='zero-mean generated noise only; fixed gyro bias is recorded separately',
        gravity_world_m_s2=GRAVITY_WORLD_M_S2.tolist(),
        magnetic_field_world=field_world.tolist(),
        magnetic_reference_world_unit=field_direction.tolist(),
        reference_magnitudes=dict(acceleration=9.80665, magnetic=MAGNETIC_REFERENCE_MAGNITUDE),
        orientation_convention='quaternion wxyz; body-to-world active rotation; scipy lowercase xyz Euler convention',
        trajectory_formulas=dict(
            time_unit='s',
            roll_deg='20 sin(2 pi t / 17)',
            pitch_deg='15 sin(2 pi t / 23)',
            yaw_deg='-20 + 30 sin(2 pi t / 28) + 8 sin(2 pi t / 11)',
            position_x_m='0.8 sin(2 pi t / 70)',
            position_y_m='0.5 (1 - cos(2 pi t / 80))',
            position_z_m='1.5 + 0.12 sin(2 pi t / 90)',
            velocity_and_acceleration='analytic time derivatives of the prescribed position',
            gyro_rad_s='rotvec(R(t[k-1]).inv() * R(t[k])) / (t[k] - t[k-1]); first sample zero',
            acceleration_m_s2='R(t).T (acceleration_world(t) - gravity_world)',
            magnetic_raw='R(t_source).T magnetic_field_world; source stamp held between magnetic publications',
        ),
        random_measurement_generation=True,
        monte_carlo_inference=False,
    )
    return arrays, settings
