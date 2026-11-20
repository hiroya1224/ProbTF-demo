"""Frozen real-drone inputs, with plotting references kept out of inference."""
from pathlib import Path
import hashlib
import importlib.util
import json
import sys

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


HERE = Path(__file__).resolve().parent
BASE = HERE.parent
ARCHIVE = BASE/'publication/bag8_50_55s'
REFINED = BASE/'mocap_refined_dynamics/publication/bag8_40_45_fit_50_55_predict'
SYNTHETIC = BASE/'synthetic_controlled_rotation/sequential_bayes_spikes'
for directory in (BASE, BASE/'rate_comparison_5hz'):
    if str(directory) not in sys.path:
        sys.path.append(str(directory))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


historical = load('_real_student_historical_selection', BASE/'rate_comparison_5hz/run.py')
TorqueProfile = historical.TorqueProfile


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare_prediction_inputs(data, metadata, profile_arrays, rate_hz):
    """Extract only permitted estimator inputs; no noninitial pose/dense gyro."""
    old_selected, evidence = historical.acquisitions(data, metadata, rate_hz)
    selected = {name: np.asarray(old_selected[name]).copy() for name in
                ('time', 'initial_quaternion_wxyz', 'sparse_gyro_time', 'sparse_gyro')}
    index = np.asarray(evidence['source_index'], int)
    acquired = {name: np.asarray(value).copy() for name, value in evidence.items()}
    acquired.update(native_array_index=index.copy(),
        original_source_index=np.asarray(data['imu_source_index'])[index].copy(),
        requested_bag_stamp_ns=np.rint(acquired['requested_time']*1e9).astype(np.int64))
    physical = dict(inertia=np.asarray(profile_arrays['inertia_body']).copy(),
        torque=TorqueProfile(profile_arrays['torque_profile_time'],
            profile_arrays['torque_actuator_body'], profile_arrays['residual_mean_body']))
    return selected, acquired, physical


def load_inputs(rate_hz):
    """Return plot source, acquisitions, restricted data, physics, provenance."""
    data = dict(np.load(ARCHIVE/'input.npz', allow_pickle=False))
    metadata = json.loads((ARCHIVE/'input.json').read_text())
    model = dict(np.load(REFINED/'refined_torque_profile.npz', allow_pickle=False))
    selected, acquired, physical = prepare_prediction_inputs(data, metadata, model, rate_hz)
    start, end = data['time']
    native = (data['imu_time'] >= start)&(data['imu_time'] <= end)
    pose = (data['mocap_time'] > start)&(data['mocap_time'] < end)
    reference_time = np.r_[start, data['mocap_time'][pose], end]
    reference = Slerp(data['mocap_time'], Rotation.from_quat(data['mocap_quaternion_xyzw']))(reference_time)
    source = dict(time=data['time'].copy(),
        reference_time=reference_time,
        reference_quaternion_xyzw=reference.as_quat(),
        reference_rpy_rad=np.unwrap(reference.as_euler('xyz'), axis=0),
        gyro_time=data['imu_time'][native].copy(),
        gyro_body_rad_s=data['gyro'][native].copy(),
        gyro_stamp_ns=data['imu_stamp_ns'][native].copy(),
        gyro_record_ns=data['imu_record_ns'][native].copy(),
        gyro_source_index=data['imu_source_index'][native].copy(),
        original_mocap_time=data['mocap_time'].copy(),
        original_mocap_quaternion_xyzw=data['mocap_quaternion_xyzw'].copy(),
        original_mocap_stamp_ns=data['mocap_stamp_ns'].copy(),
        original_mocap_record_ns=data['mocap_record_ns'].copy(),
        original_mocap_source_index=data['mocap_source_index'].copy(),
        initial_quaternion_wxyz=data['initial_quaternion_wxyz'].copy())
    # This previously saved SG derivative is an optional read-only plot
    # reference; it never enters selected or physical.
    diagnostic = dict(np.load(REFINED/'heldout_dynamics_check.npz', allow_pickle=False))
    source.update(omega_reference_time=diagnostic['time'].copy(),
        omega_reference_body_rad_s=diagnostic['omega_body_rad_s'].copy())
    provenance = dict(schema='grape/real-student-frozen-input/v1',
        bag_path=metadata['bag_path'], time_origin_ns=metadata['time_origin_ns'],
        requested_window_seconds=metadata['requested_window_seconds'],
        actual_window_seconds=data['time'], requested_acquisition_rate_hz=float(rate_hz),
        observation_count=len(acquired['time']), last_observation_seconds=float(acquired['time'][-1]),
        prediction_tail_seconds=float(end-acquired['time'][-1]),
        initial_pose='Saved one-time mocap SLERP alignment at the first actual gyro sample',
        body_to_gyro_sensor_rotation=data['body_to_gyro_sensor_rotation'],
        training_window_seconds=[40.,45.], refit_performed=False,
        torque='Frozen refined actuator torque plus strictly past residual mean once; torque_applied_body is diagnostic only',
        fixed_residual_mean_body_nm=model['residual_mean_body'], inertia_body_kg_m2=model['inertia_body'],
        estimator_data_fields=list(selected), physical_fields=list(physical),
        evaluation_mocap_or_unacquired_gyro_supplied_to_estimator=False,
        generated_noise_or_spikes_added=False, gyro_bias_or_time_shift_corrected=False,
        model_parameter_uncertainty_propagated=False,
        source_file_sha256={str(path.relative_to(BASE)):sha(path) for path in
            (ARCHIVE/'input.npz', ARCHIVE/'input.json', REFINED/'refined_torque_profile.npz',
             REFINED/'refined_torque_profile.json', REFINED/'fit.json', REFINED/'heldout_dynamics_check.npz')},
        reference_semantics='Mocap pose and SG-derived omega are read-only; raw gyro is a noisy measured reference, not angular-velocity truth')
    return source, acquired, selected, physical, provenance
