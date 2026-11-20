#!/usr/bin/env python3
"""Independent read-only check of the frozen real-drone physical flow.

The same raw initial gyro, one initial mocap pose, full inertia tensor and
known actuator torque plus past residual mean are propagated without ANY
observation updates. Native RK4 is compared with the original Python RK4
and an independent SciPy DOP853 implementation. Mocap is evaluation only.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import sys
import time

import numpy as np
from scipy.integrate import solve_ivp
from scipy.spatial.transform import Rotation, Slerp


HERE = Path(__file__).resolve().parent
BASE = HERE.parent
SYNTHETIC = BASE/'synthetic_controlled_rotation/sequential_bayes_spikes'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def provenance():
    paths = (Path(__file__), HERE/'input.py', SYNTHETIC/'native_flow.py',
             SYNTHETIC/'native_flow.cpp', BASE/'propagation.py', BASE/'torque.py')
    return dict(
        schema='grape/real-student-independent-physics-audit/v1',
        scope='One point trajectory from raw initial gyro; no observation updates, no parameter refit',
        filter_or_measurements_modified=False, reference_is_read_only=True,
        future_gyro_or_mocap_supplied_to_prediction=False,
        coordinate_convention='Hamilton wxyz body-to-world quaternion; body angular velocity and body-CoG torque; original bag clock',
        residual_torque='Frozen past residual mean added once to stored actuator torque',
        native_and_original_python_rk4_max_step_seconds=.002,
        independent_integrator=dict(method='scipy solve_ivp DOP853',
            max_step_seconds=.0005, rtol=1e-10, atol=1e-12),
        comparison_grid_count=501,
        original_python_covariance=np.zeros((3,3)).tolist(),
        original_python_six_points='Six coincident zero-covariance initial conditions required by the old API; no random draws',
        finite_differences_are_diagnostics_not_filter_gates=True,
        source_sha256={str(path.relative_to(BASE)):sha(path) for path in paths})


def pose_difference_degrees(left, right):
    left_xyzw = np.asarray(left)[:, [1,2,3,0]]
    right_xyzw = np.asarray(right)[:, [1,2,3,0]]
    return np.rad2deg((Rotation.from_quat(left_xyzw).inv()
                      *Rotation.from_quat(right_xyzw)).magnitude())


def run(output):
    inputs = load('_physics_audit_real_input', HERE/'input.py')
    native = load('_physics_audit_native', SYNTHETIC/'native_flow.py')
    original = load('_physics_audit_original_python', BASE/'propagation.py')
    source, _acquired, data, physical, input_provenance = inputs.load_inputs(5.)
    grid = np.linspace(*data['time'], 501)
    omega0 = data['sparse_gyro'][0]
    quaternion0 = data['initial_quaternion_wxyz']
    inertia = physical['inertia']
    torque = physical['torque']
    reference = Slerp(source['original_mocap_time'],
        Rotation.from_quat(source['original_mocap_quaternion_xyzw']))(grid)
    report = dict(provenance(), initial_omega_rad_s=omega0.tolist(),
        initial_quaternion_wxyz=quaternion0.tolist(), time_bounds=data['time'].tolist(),
        inertia_body_kg_m2=inertia.tolist(), residual_mean_nm=torque.residual_mean.tolist(),
        frozen_input_source_sha256=input_provenance['source_file_sha256'], modes={})
    for mode in ('nonlinear', 'linear_forced'):
        started = time.monotonic()
        native_state = native.ControlledFlow(physical, quaternion0, *data['time'],
            max_step=.002, mode=mode).evaluate(omega0[None,:], grid)[:,0]
        old = original.integrate_six_point_increment(omega0, np.zeros((3,3)), grid,
            torque, inertia, mode=mode, max_step=.002)
        old_quaternion = original.quaternion_multiply(quaternion0, old['delta'][:,0])
        def derivative(stamp, state):
            omega, quaternion = state[:3], state[3:]
            gyroscopic = (np.cross(omega, inertia@omega) if mode=='nonlinear'
                          else np.zeros(3))
            scalar, vector = quaternion[0], quaternion[1:]
            quaternion_rate = .5*np.r_[-vector@omega,
                scalar*omega+np.cross(vector, omega)]
            return np.r_[np.linalg.solve(inertia, torque(stamp)-gyroscopic),
                         quaternion_rate]
        solution = solve_ivp(derivative, data['time'], np.r_[omega0, quaternion0],
            t_eval=grid, method='DOP853', max_step=.0005, rtol=1e-10, atol=1e-12)
        if not solution.success:
            raise RuntimeError('Independent physics audit integration failed: '+solution.message)
        scipy_state = solution.y.T
        reference_error = np.rad2deg((reference.inv()
            *Rotation.from_quat(native_state[:, [4,5,6,3]])).magnitude())
        scipy_pose_error = pose_difference_degrees(native_state[:,3:], scipy_state[:,3:])
        report['modes'][mode] = dict(scipy_success=solution.success,
            native_vs_original_python_max_omega_rad_s=float(abs(
                native_state[:,:3]-old['omega'][:,0]).max()),
            native_vs_original_python_max_pose_deg=float(pose_difference_degrees(
                native_state[:,3:], old_quaternion).max()),
            native_vs_dop853_max_omega_rad_s=float(abs(
                native_state[:,:3]-scipy_state[:,:3]).max()),
            native_vs_dop853_max_pose_deg=float(scipy_pose_error.max()),
            native_vs_dop853_final_pose_deg=float(scipy_pose_error[-1]),
            free_prediction_mocap_pose_rmse_deg=float(np.sqrt(np.mean(reference_error**2))),
            free_prediction_mocap_final_pose_deg=float(reference_error[-1]),
            final_predicted_omega_rad_s=native_state[-1,:3].tolist(),
            wall_seconds=time.monotonic()-started)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(report, indent=2, allow_nan=False))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
        default=HERE/'publication/bag8_50_55s/physics_audit.json')
    run(parser.parse_args().output)
