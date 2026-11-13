"""Actual data-flow verification, without changing an inferred law."""
import numpy as np


def verify_unused_reference_invariance(selected, profile, sigma, original_prediction, predictor, max_step):
    """Poison all redundant gyro/mocap arrays, retain only saved sparse inputs."""
    poisoned = dict(selected)
    poisoned_keys = []
    for key in selected:
        if (key.startswith('mocap_') or key.startswith('dense_')
                or key.startswith('past_calibration_') or key in ('gyro', 'gyro_noise_covariance')):
            value = np.asarray(selected[key])
            if np.issubdtype(value.dtype, np.number):
                poisoned[key] = np.full(value.shape, np.nan)
                poisoned_keys.append(key)
    result = predictor(poisoned, profile, sigma, 'nonlinear', output_dt=.02, max_step=max_step)
    equal = {key: bool(np.array_equal(original_prediction[key], result[key])) for key in
             ('time', 'quaternion_second_moment', 'rotation_mean', 'rotation_covariance_colmajor',
              'omega_mean_rad_s', 'omega_covariance_rad2_s2')}
    return dict(method='replace unused reference/dense/past-sensor arrays with NaN; rerun predictor with unchanged sparse rows, q0, frozen model and torque',
                poisoned_keys=poisoned_keys, fields_bitwise_equal=equal, all_bitwise_equal=all(equal.values()),
                sparse_readings_retained=len(selected['sparse_gyro_time']), sigma_deg_s=sigma,
                failure_policy='record result; not a scientific state-acceptance or covariance gate')
