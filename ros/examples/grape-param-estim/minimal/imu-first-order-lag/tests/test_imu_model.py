from pathlib import Path
import json
import sys

import numpy as np

HERE = Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import imu_model as model
from grape_param_estim.dynamics import actuator_wrench
from grape_param_estim.system import ActuatorState


def synthetic_data():
    t = np.linspace(0., 2., 61) + .0006 * np.sin(np.arange(61))
    t -= t[0]
    command_time = np.linspace(-1., 2.1, 70)
    command = np.column_stack([7 + .3 * np.sin((j + 1) * command_time + j) for j in range(4)])
    gyro = np.column_stack([.2 * np.sin((j + 1) * t + .2 * j) for j in range(3)])
    accel = np.column_stack([.1 * np.cos(t), .2 * np.sin(1.5 * t), 9.8 + .3 * np.cos(.7 * t)])
    angles = np.column_stack([.12 * np.sin(.5 * t + j) for j in range(4)])
    return dict(time=t, gyro=gyro, accel=accel,
        gyro_noise_covariance=np.eye(3) * .001, accel_noise_covariance=np.eye(3) * .02,
        rotor_command_time=command_time, rotor_command=command,
        gimbal_time=t, gimbal_angle=angles, gyro_sensor_position_in_body=np.array([-.0173, -.0011, .057061]),
        controller_dt=.02)


def problem():
    return model.ImuDynamicsProblem(synthetic_data(), model.load_vehicle_model(HERE.parent / "grape_vehicle_model.json"),
        {"sg_window_seconds": .5, "sg_degree": 5})


def test_vectorized_wrench_is_the_existing_canonical_physics():
    p = problem()
    x = p.initial.copy(); x[:13] = .01 * np.cos(np.arange(13))
    value = p.evaluate(x)
    canonical = np.asarray([actuator_wrench(ActuatorState(f, g), value["parameters"], p.model.geometry)
                            for f, g in zip(value["actual_thrust"], value["actual_gimbal"])])
    np.testing.assert_allclose(value["modeled_wrench"], canonical, rtol=2e-14, atol=2e-14)


def test_all_sixteen_physical_and_lag_jacobian_columns_match_derivatives():
    p = problem()
    x = p.initial.copy(); x[:13] = .01 * np.sin(np.arange(13)); x[13:] = np.log([.13, .04, .08])
    value = p.evaluate(x)
    numerical = np.empty_like(value["jacobian"])
    step = 2e-6
    for j in range(16):
        plus, minus = x.copy(), x.copy(); plus[j] += step; minus[j] -= step
        numerical[:, :, j] = (p.evaluate(plus)["residual"] - p.evaluate(minus)["residual"]) / (2 * step)
    np.testing.assert_allclose(value["jacobian"], numerical, rtol=2e-5, atol=2e-7)


def test_gyro_noise_maps_keep_shared_omega_and_angular_acceleration_dependence():
    p = problem(); x = p.initial.copy(); x[13:] = np.log([.13, .04, .08])
    value = p.evaluate(x); mappings, _ = p.noise_maps(value)
    raw_index, component, epsilon = 30, 1, 1e-6
    expected = sum(mapping[:, :, component] * operator[:, raw_index].toarray()
                   for mapping, operator in zip(mappings, p.gyro["operators"]))
    originals = {key: p.gyro[key].copy() for key in ("smooth", "derivative", "second_derivative")}
    derivatives = []
    for sign in (1., -1.):
        for key, operator in zip(originals, p.gyro["operators"]):
            p.gyro[key] = originals[key].copy()
            p.gyro[key][:, component] += sign * epsilon * operator[:, raw_index].toarray().ravel()
        derivatives.append(p.evaluate(x)["residual"])
    p.gyro.update(originals)
    np.testing.assert_allclose((derivatives[0] - derivatives[1]) / (2 * epsilon), expected,
                              rtol=2e-5, atol=2e-8)


def test_joint_chart_removes_only_common_scale_and_keeps_all_three_taus():
    b = model.joint_chart_basis()
    assert b.shape == (17, 16)
    direction = np.r_[model._thrust_module.COMMON_SCALE_DIRECTION, np.zeros(3)]
    np.testing.assert_allclose(b.T @ direction, 0, atol=1e-14)
    np.testing.assert_array_equal(b[14:, 13:], np.eye(3))
    np.testing.assert_allclose(b.T @ b, np.eye(16), atol=1e-14)


def test_cache_restores_joint_cross_covariance_and_sensor_taus_without_refitting(tmp_path):
    p = problem()
    coordinate = p.initial.copy(); coordinate[:13] = .01 * np.cos(np.arange(13))
    coordinate[13:] = np.log([.2, .03, .06])
    basis = model.joint_chart_basis()
    factor = np.eye(16) * .02 + np.outer(np.arange(16) * .001, np.ones(16))
    covariance = factor @ factor.T
    physical_factor = np.diag(np.r_[np.ones(13), np.exp(coordinate[13:])]) @ factor
    estimate = {"parameter_distribution": {
        "reduced_log_tau_coordinate": coordinate.tolist(), "full_chart_basis": basis.tolist(),
        "full_chart_coordinate": (basis @ coordinate).tolist()}}
    (tmp_path / "estimate.json").write_text(json.dumps(estimate))
    np.savez_compressed(tmp_path / "arrays.npz",
        parameter_covariance_reduced_log_tau=covariance,
        parameter_covariance_full_chart_log_tau=basis @ covariance @ basis.T,
        parameter_covariance_factor_reduced_log_tau=factor,
        parameter_covariance_factor_full_chart_log_tau=basis @ factor,
        tau_covariance_seconds=(physical_factor @ physical_factor.T)[13:, 13:])
    restored = model.load_fit_cache(tmp_path, HERE.parent / "grape_vehicle_model.json")
    np.testing.assert_allclose(restored["taus"], [.2, .03, .06])
    np.testing.assert_array_equal(restored["parameter_covariance_reduced"], covariance)
    np.testing.assert_array_equal(restored["parameter_covariance_factor_full"], basis @ factor)
    assert np.any(restored["parameter_covariance_reduced"][:13, 13:] != 0)
    assert restored["parameter_covariance_factor_physical_si"].shape == (17, 16)


def test_identification_provenance_excludes_evaluation_mocap():
    data = synthetic_data()
    first = model.input_provenance(data, HERE.parent / "grape_vehicle_model.json")
    data["mocap_position"] = np.full((100, 3), 123.)
    second = model.input_provenance(data, HERE.parent / "grape_vehicle_model.json")
    assert first == second
    data["gyro"][0, 0] += .001
    changed = model.input_provenance(data, HERE.parent / "grape_vehicle_model.json")
    assert changed["identification_input_arrays"]["gyro"] != first["identification_input_arrays"]["gyro"]
