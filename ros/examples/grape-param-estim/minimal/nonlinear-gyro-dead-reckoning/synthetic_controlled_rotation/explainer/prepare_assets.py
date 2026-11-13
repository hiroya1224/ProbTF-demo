"""Derive explanatory arrays from the exact saved synthetic experiment.

Only one RK4 step is recomputed as an explicitly labelled algorithm trace.
The original estimator state, measurements and plots are never overwritten.
"""
from pathlib import Path
import hashlib
import sys

import numpy as np
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parent
if str(EXPERIMENT.parent) not in sys.path:
    sys.path.append(str(EXPERIMENT.parent))
from propagation import quaternion_multiply
from torque import TorqueProfile


def prepare(data_dir, destination):
    data_dir, destination = Path(data_dir), Path(destination)
    data = dict(np.load(data_dir/'input.npz'))
    prediction = dict(np.load(data_dir/'nonlinear.npz'))
    inertia = data['inertia_body_kg_m2']
    inverse = np.linalg.solve(inertia, np.eye(3))
    torque = TorqueProfile(data['torque_time'], data['torque_body_nm'])
    stamps = data['reference_time']
    amplitudes, frequency, phases = (data['motion_amplitude_deg'],
                                   data['motion_frequency_hz'], data['motion_phase_rad'])
    components = amplitudes[None, :, :]*np.sin(
        stamps[:, None, None]*(2*np.pi*frequency)[None, :, :]+phases[None, :, :])
    source = prediction['block_0_omega'][0].copy()
    initial = np.column_stack((source, np.tile([1., 0., 0., 0.], (len(source), 1))))

    def derivative(time, state):
        omega, quaternion = state[:, :3], state[:, 3:]
        alpha = (torque(time)-np.cross(omega, omega@inertia.T))@inverse.T
        pure = np.column_stack((np.zeros(len(omega)), omega))
        return np.column_stack((alpha, .5*quaternion_multiply(quaternion, pure)))

    dt = .002
    k1 = derivative(0., initial)
    s2 = initial+.5*dt*k1
    k2 = derivative(.5*dt, s2)
    s3 = initial+.5*dt*k2
    k3 = derivative(.5*dt, s3)
    s4 = initial+dt*k3
    k4 = derivative(dt, s4)
    final_raw = initial+dt/6*(k1+2*k2+2*k3+k4)
    final = final_raw.copy()
    final[:, 3:] /= np.linalg.norm(final[:, 3:], axis=1)[:, None]
    # Repeating these exact steps to the first saved output is an independent
    # trace check. It does not replace the original trajectory.
    check = initial.copy()
    first_output = prediction['block_0_time'][1]
    step_count = int(np.ceil(first_output/.002))
    check_dt = first_output/step_count
    for index in range(step_count):
        t = index*check_dt
        a = derivative(t, check)
        b = derivative(t+check_dt/2, check+check_dt*a/2)
        c = derivative(t+check_dt/2, check+check_dt*b/2)
        d = derivative(t+check_dt, check+check_dt*c)
        check = check+check_dt/6*(a+2*b+2*c+d)
        check[:, 3:] /= np.linalg.norm(check[:, 3:], axis=1)[:, None]
    required = data['reference_alpha_body_rad_s2']@inertia.T
    covariance_parts = []
    maximum_covariance_difference = 0.
    block = 0
    while f'block_{block}_time' in prediction:
        prefix = f'block_{block}_'
        mean = prediction[prefix+'parent_rotation_mean'].reshape(9, order='F')
        covariance = prediction[prefix+'parent_rotation_covariance']
        time = prediction[prefix+'time']
        delta = prediction[prefix+'delta_wxyz']
        weights = prediction[prefix+'weights']
        for index, stamp in enumerate(time):
            rotations = Rotation.from_quat(delta[index, :, [1, 2, 3, 0]].T).as_matrix()
            actions = np.stack([np.kron(rotation.T, np.eye(3)) for rotation in rotations])
            means = np.einsum('sij,j->si', actions, mean)
            new_mean = np.einsum('s,si->i', weights, means)
            within = np.einsum('s,sai,ij,sbj->ab', weights, actions, covariance, actions, optimize=True)
            centered = means-new_mean
            between = np.einsum('s,si,sj->ij', weights, centered, centered)
            source_index = int(np.argmin(np.abs(prediction['time']-stamp)))
            maximum_covariance_difference = max(maximum_covariance_difference, float(np.max(np.abs(
                within+between-prediction['rotation_covariance_colmajor'][source_index]))))
            covariance_parts.append((stamp, np.trace(within), np.trace(between), np.trace(within+between)))
        block += 1
    covariance_parts = np.asarray(covariance_parts)
    stages, derivatives = np.stack((initial, s2, s3, s4)), np.stack((k1, k2, k3, k4))
    arrays = dict(reference_time=stamps, rpy_components_deg=components,
        rpy_component_time=stamps, rpy_components_rad=np.deg2rad(components),
        rotation_cov_time=covariance_parts[:, 0],
        rotation_cov_within_trace=covariance_parts[:, 1],
        rotation_cov_between_trace=covariance_parts[:, 2],
        rotation_cov_total_trace=covariance_parts[:, 3],
        inertial_torque_body_nm=required,
        gyroscopic_torque_body_nm=data['reference_gyroscopic_torque_body_nm'],
        rk4_time=np.array([0., dt/2, dt/2, dt]), rk4_dt=np.asarray(dt),
        rk4_stage_states=stages, rk4_derivatives=derivatives,
        rk4_stage_time=np.array([0., dt/2, dt/2, dt]),
        rk4_stage_omega=stages[:, :, :3], rk4_stage_quaternion_wxyz=stages[:, :, 3:],
        rk4_k_omega=derivatives[:, :, :3], rk4_k_quaternion=derivatives[:, :, 3:],
        rk4_final_raw=final_raw, rk4_final_normalized=final)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **arrays)
    return dict(schema='grape/synthetic-explainer-derived-assets/v1',
        files={name:hashlib.sha256((data_dir/name).read_bytes()).hexdigest()
               for name in ('input.npz', 'nonlinear.npz')},
        output_sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
        component_sum_max_difference_deg=float(np.max(np.abs(components.sum(axis=-1)-np.rad2deg(data['reference_rpy_rad'])))),
        rk4_dt_seconds=dt, trace_role='actual six initial gyro candidates and first RK4 step, recomputed read-only from same saved input and formula',
        first_saved_omega_max_difference_rad_s=float(np.max(np.abs(check[:, :3]-prediction['block_0_omega'][1]))),
        first_saved_delta_max_difference=float(np.max(np.abs(check[:, 3:]-prediction['block_0_delta_wxyz'][1]))),
        saved_rotation_covariance_max_difference=maximum_covariance_difference,
        original_results_modified=False)
