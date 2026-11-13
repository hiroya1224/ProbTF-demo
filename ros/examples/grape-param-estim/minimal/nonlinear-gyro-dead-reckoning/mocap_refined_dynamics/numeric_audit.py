"""Read-only RK4 step and actuator interpolation sensitivity checks."""
import numpy as np


def audit_numerics(data, selected, fitted, profile, original, predictor, sigma, max_step):
    from fit import actuator_at
    refined = predictor(selected, profile, sigma, 'nonlinear', output_dt=.02, max_step=max_step/2)
    rpy_difference = (refined['rpy_mean_rad']-original['rpy_mean_rad']+np.pi)%(2*np.pi)-np.pi
    arrays = profile['arrays']
    stamps = arrays['torque_profile_time']
    midpoint = .5*(stamps[:-1]+stamps[1:])
    midpoint = midpoint[(midpoint >= data['time'][0]) & (midpoint <= data['time'][1])]
    direct, _ = actuator_at(data, fitted, midpoint)
    interpolated = profile['actuator_torque'](midpoint)
    torque_error = interpolated-direct[:, 3:]
    acceleration_error = np.linalg.solve(profile['inertia'], torque_error.T).T
    return dict(
        role='numerical sensitivity diagnostic only; does not change model, law, or accepted observations',
        original_max_step_seconds=max_step, finer_max_step_seconds=max_step/2,
        sigma_deg_s=sigma, gyro_count=len(selected['sparse_gyro_time']),
        max_quaternion_second_moment_difference=float(np.max(np.abs(refined['quaternion_second_moment']-original['quaternion_second_moment']))),
        max_rotation_mean_difference=float(np.max(np.abs(refined['rotation_mean']-original['rotation_mean']))),
        max_mean_omega_difference_rad_s=float(np.max(np.abs(refined['omega_mean_rad_s']-original['omega_mean_rad_s']))),
        max_rpy_proxy_difference_deg=np.rad2deg(np.max(np.abs(rpy_difference), axis=0)).tolist(),
        actuator_interpolation_queries=len(midpoint),
        actuator_interpolation_torque_rms_nm=np.sqrt(np.mean(torque_error**2, axis=0)).tolist(),
        actuator_interpolation_alpha_rms_rad_s2=np.sqrt(np.mean(acceleration_error**2, axis=0)).tolist(),
        direct_actuator='same exact existing ZOH first-order response, same point and measured-gimbal interpolation at profile-cell midpoints',
        heldout_gyro_or_mocap_used=False)
