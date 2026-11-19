#!/usr/bin/env python3
"""Shared numerical/report helpers; CLI runs the requested two-model comparison."""
from pathlib import Path
import hashlib
import importlib.util
import json
import sys

import numpy as np
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
SHARED = HERE.parent.parent
sys.path.insert(0, str(SHARED))
from torque import TorqueProfile



def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    def encode(item):
        if isinstance(item, np.ndarray): return item.tolist()
        if isinstance(item, np.generic): return item.item()
        if isinstance(item, Path): return str(item)
        raise TypeError(type(item).__name__)
    Path(path).write_text(json.dumps(value, default=encode, ensure_ascii=False,
                                   indent=2, allow_nan=False)+'\n')


def module_from_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def score(source, case):
    """Score online outputs only; right-hand posterior is used at acquisition."""
    stamps = np.linspace(*source['time'], 501)
    def interp(name):
        values = case[name]
        shape = values.shape[1:]
        flat = values.reshape(len(values), -1)
        result = np.column_stack([np.interp(stamps, case['time'], flat[:, i])
                                  for i in range(flat.shape[1])])
        return result.reshape((len(stamps),)+shape)
    rpy = interp('rpy_proxy_rad')
    omega = interp('omega_mean_rad_s')
    truth_rpy = np.column_stack([np.interp(stamps, source['reference_time'],
                      source['reference_rpy_rad'][:, i]) for i in range(3)])
    truth_omega = np.column_stack([np.interp(stamps, source['reference_time'],
                      source['reference_omega_body_rad_s'][:, i]) for i in range(3)])
    errors = (Rotation.from_euler('xyz', truth_rpy).inv()
              *Rotation.from_euler('xyz', rpy)).magnitude()
    omega_error = omega-truth_omega
    return dict(so3_proxy_rmse_deg=float(np.rad2deg(np.sqrt(np.mean(errors**2)))),
        so3_proxy_max_deg=float(np.rad2deg(errors.max())),
        so3_proxy_final_deg=float(np.rad2deg(errors[-1])),
        omega_rmse_deg_s=np.rad2deg(np.sqrt(np.mean(omega_error**2, axis=0))),
        omega_final_error_deg_s=np.rad2deg(omega_error[-1]),
        final_omega_two_sigma_deg_s=2*np.rad2deg(np.sqrt(np.diag(case['omega_covariance_rad2_s2'][-1]))),
        final_rpy_two_sigma_deg=2*np.rad2deg(np.sqrt(np.diag(case['rpy_covariance_delta_rad2'][-1]))),
        score_count=len(stamps)), dict(error_time=stamps, error_geodesic_rad=errors,
                                       error_omega_rad_s=omega_error)


def refinement(coarse, fine):
    """Numerical integration diagnostics, never an observation acceptance gate."""
    assert np.array_equal(coarse['time'], fine['time'])
    mean_difference = fine['omega_mean_rad_s']-coarse['omega_mean_rad_s']
    covariance_difference = (fine['omega_covariance_rad2_s2']
                             -coarse['omega_covariance_rad2_s2'])
    rpy_difference = (fine['rpy_proxy_rad']-coarse['rpy_proxy_rad']+np.pi)%(2*np.pi)-np.pi
    return dict(max_omega_mean_component_difference_deg_s=float(np.max(np.abs(np.rad2deg(mean_difference)))),
        max_omega_covariance_component_difference_deg2_s2=float(np.max(np.abs(covariance_difference))*(180/np.pi)**2),
        max_rpy_proxy_component_difference_deg=float(np.max(np.abs(np.rad2deg(rpy_difference)))),
        max_rotation_covariance_component_difference=float(np.max(np.abs(
            fine['rotation_covariance_colmajor']-coarse['rotation_covariance_colmajor']))),
        finite_diagnostic_only=True)


def save_case(path, case):
    arrays = {key: value for key, value in case.items() if isinstance(value, np.ndarray)}
    np.savez_compressed(path, **arrays)


def reference_event_diagnostics(source, cases, events):
    """Read-only evaluation; these reference values never reach the filter."""
    for name, records in events.items():
        case = cases[name]
        for record in records:
            if record['initialization']:
                continue
            stamp = record['time']
            rows = np.flatnonzero(case['time']==stamp)
            truth = np.array([np.interp(stamp, source['reference_time'],
                               source['reference_rpy_rad'][:, axis]) for axis in range(3)])
            for side, label in ((-1,'prior'),(1,'posterior')):
                row = rows[case['event_side'][rows]==side][0]
                error = (Rotation.from_euler('xyz', truth).inv()
                         *Rotation.from_euler('xyz', case['rpy_proxy_rad'][row])).magnitude()
                record[label+'_so3_proxy_error_deg_reference_only'] = float(np.rad2deg(error))
                record[label+'_rpy_proxy_deg_reference_only'] = np.rad2deg(case['rpy_proxy_rad'][row])
    return events


if __name__ == '__main__':
    import runpy
    runpy.run_path(str(HERE/'frequent_run.py'), run_name='__main__')
