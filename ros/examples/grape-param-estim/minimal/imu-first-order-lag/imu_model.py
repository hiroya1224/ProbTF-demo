"""Isolated IMU Newton--Euler fit with three first-order time constants.

No attitude or mocap column enters this objective: gravity cancels in body
specific force.  The gyro/accel measurement laws are tau*y_dot + y = input.
Whitening is frozen at the nominal model, so this is quasi-GLS, not a
parameter-dependent Gaussian likelihood.  Final covariance propagates the
actual fitted lag operators and their full SG overlap without sampling.
"""
from __future__ import annotations

import importlib.util
import hashlib
import json
import math
from pathlib import Path
import sys
import time as walltime

import numpy as np
from scipy.optimize import least_squares

HERE = Path(__file__).resolve().parent
MINIMAL = HERE.parent
for path in (HERE, MINIMAL, MINIMAL.parent / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
from single_bag_savgol_core import (  # noqa: E402
    SiParameterChart, common_scale_quotient_basis, load_vehicle_model,
    physical_parameter_jacobian, physical_parameter_vector,
    PHYSICAL_CHART_LABELS,
)

_spec = importlib.util.spec_from_file_location(
    "grape_imu_first_order_thrust", MINIMAL / "first-order-lag/core.py")
_thrust_module = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _thrust_module
_spec.loader.exec_module(_thrust_module)
first_order_zoh_thrust_history = _thrust_module.first_order_zoh_thrust_history

TAU_LABELS = ("thrust_time_constant_seconds", "gyro_time_constant_seconds", "accel_time_constant_seconds")
LOG_TAU_LABELS = tuple("log_" + label for label in TAU_LABELS)
SI_LABELS = ("mass_kg", "inertia_xx_kg_m2", "inertia_yy_kg_m2", "inertia_zz_kg_m2",
             "inertia_xy_kg_m2", "inertia_xz_kg_m2", "inertia_yz_kg_m2",
             "cog_x_m", "cog_y_m", "cog_z_m", "force_effectiveness_1",
             "force_effectiveness_2", "force_effectiveness_3", "force_effectiveness_4") + TAU_LABELS

FIT_INPUT_KEYS = ("time", "gyro", "accel", "gyro_noise_covariance", "accel_noise_covariance",
    "rotor_command_time", "rotor_command", "gimbal_time", "gimbal_angle",
    "gyro_sensor_position_in_body", "controller_dt")


def input_provenance(data, model_path):
    """Hash exactly the identification inputs, separately from evaluation data."""
    arrays = {}
    for key in FIT_INPUT_KEYS:
        value = np.ascontiguousarray(np.asarray(data[key]))
        arrays[key] = {"dtype": value.dtype.str, "shape": list(value.shape),
            "sha256": hashlib.sha256(value.tobytes()).hexdigest()}
    paths = {"imu_model": Path(__file__).resolve(), "imu_signal": HERE / "imu_signal.py",
        "first_order_thrust": MINIMAL / "first-order-lag/core.py",
        "physical_chart": MINIMAL / "single_bag_savgol_core.py", "vehicle_model": Path(model_path).resolve()}
    files = {key: {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
             for key, path in paths.items()}
    metadata = data.get("metadata", {})
    return {"identification_input_arrays": arrays, "source_files": files,
        "source_bag_sha256": metadata.get("bag_sha256"),
        "ongoing_mocap_used": False,
        "mocap_policy": "excluded from identification; separate initial-condition alignment and read-only evaluation"}


def skew_batch(v):
    value = np.asarray(v, dtype=float)
    out = np.zeros(value.shape[:-1] + (3, 3))
    out[..., 0, 1], out[..., 0, 2] = -value[..., 2], value[..., 1]
    out[..., 1, 0], out[..., 1, 2] = value[..., 2], -value[..., 0]
    out[..., 2, 0], out[..., 2, 1] = -value[..., 1], value[..., 0]
    return out


def joint_chart_basis():
    """Lift 13 physical quotient coordinates and three log taus to 17-D."""
    basis = np.zeros((17, 16))
    basis[:14, :13] = common_scale_quotient_basis()
    basis[14:, 13:] = np.eye(3)
    return basis


class ImuDynamicsProblem:
    def __init__(self, data, model, options=None):
        from imu_signal import local_polynomial_signal
        self.data, self.model = data, model
        self.options = {"sg_window_seconds": 1.0, "sg_degree": 5,
                        "minimum_thrust": 1.5, "maximum_thrust": 27.6145,
                        "max_nfev": 10000, "ftol": float(np.sqrt(np.finfo(float).eps)),
                        "xtol": float(np.sqrt(np.finfo(float).eps)), "gtol": 1e-8}
        self.options.update(options or {})
        self.time = np.asarray(data["time"], dtype=float)
        self.chart = SiParameterChart(model.parameters)
        self.basis = common_scale_quotient_basis()
        self.gyro = local_polynomial_signal(self.time, np.asarray(data["gyro"]),
            np.asarray(data["gyro_noise_covariance"]),
            window_seconds=float(self.options["sg_window_seconds"]),
            degree=int(self.options["sg_degree"]), max_derivative=2)
        self.accel = local_polynomial_signal(self.time, np.asarray(data["accel"]),
            np.asarray(data["accel_noise_covariance"]),
            window_seconds=float(self.options["sg_window_seconds"]),
            degree=int(self.options["sg_degree"]), max_derivative=2)
        self.gimbal = np.column_stack([np.interp(self.time,
            np.asarray(data["gimbal_time"]), np.asarray(data["gimbal_angle"])[:, i]) for i in range(4)])
        if self.time[0] < np.asarray(data["gimbal_time"])[0] or self.time[-1] > np.asarray(data["gimbal_time"])[-1]:
            raise ValueError("IMU fit times exceed measured gimbal support")
        angle = self.gimbal
        yaw = np.asarray(model.geometry.arm_yaws)
        self.direction = np.stack((np.sin(yaw)[None, :] * np.sin(angle),
            -np.cos(yaw)[None, :] * np.sin(angle), np.cos(angle)), axis=-1)
        self.origins = np.asarray(model.geometry.rotor_origins)[None, :, :] + float(model.geometry.thrust_offset) * (
            self.direction - np.array([0., 0., 1.]))
        self.reaction = (np.asarray(model.parameters.torque_effectiveness)
            * np.asarray(model.geometry.rotor_directions) * float(model.geometry.moment_force_rate))
        self.sensor_position = np.asarray(data["gyro_sensor_position_in_body"], dtype=float)
        initial_tau = float(self.options.get("initial_tau_seconds", data["controller_dt"]))
        self.initial = np.r_[np.zeros(13), np.full(3, math.log(initial_tau))]
        self.invalid_trial_count = 0
        nominal = self.evaluate(self.initial)
        self.nominal_residual_covariance = self.residual_covariance(nominal)
        lower = np.linalg.cholesky(self.nominal_residual_covariance)
        self.whitening = np.linalg.solve(lower, np.broadcast_to(np.eye(6), lower.shape))
        self.weight = np.einsum("nji,njk->nik", self.whitening, self.whitening)
        self._cached_x = self._cached_evaluation = None

    def evaluate(self, x):
        selected = np.asarray(x, dtype=float)
        physical = self.basis @ selected[:13]
        parameters, pj = self.chart.decode_with_jacobian(physical)
        taus = np.exp(selected[13:16])
        if taus.shape != (3,) or not np.isfinite(taus).all() or np.any(taus <= 0):
            raise FloatingPointError("time constants became non-finite or non-positive")
        history = first_order_zoh_thrust_history(
            command_times=self.data["rotor_command_time"], command_values=self.data["rotor_command"],
            evaluation_times=self.time, time_constant=float(taus[0]),
            minimum_thrust=float(self.options["minimum_thrust"]),
            maximum_thrust=float(self.options["maximum_thrust"]))
        thrust, dthrust = history.thrust, history.log_tau_jacobian
        force_rotor = thrust[:, :, None] * parameters.force_effectiveness[None, :, None] * self.direction
        arms = self.origins - parameters.cog_offset
        force = force_rotor.sum(axis=1)
        torque = (np.cross(arms, force_rotor) + self.reaction[None, :, None] * force_rotor).sum(axis=1)
        rotor_j = thrust[:, :, None, None] * self.direction[:, :, :, None] * pj.force_effectiveness[None, :, None, :]
        force_j = rotor_j.sum(axis=1)
        torque_j = np.moveaxis(np.cross(arms[:, :, None, :], np.moveaxis(rotor_j, -1, -2)), -1, -2).sum(axis=1)
        torque_j += (self.reaction[None, :, None, None] * rotor_j).sum(axis=1)
        torque_j += np.einsum("nij,jp->nip", skew_batch(force), pj.cog_offset)
        rotor_tau = dthrust[:, :, None] * parameters.force_effectiveness[None, :, None] * self.direction
        force_tau = rotor_tau.sum(axis=1)
        torque_tau = (np.cross(arms, rotor_tau) + self.reaction[None, :, None] * rotor_tau).sum(axis=1)
        g, dg, ddg = (np.asarray(self.gyro[key]) for key in ("smooth", "derivative", "second_derivative"))
        a, da = (np.asarray(self.accel[key]) for key in ("smooth", "derivative"))
        omega, observed_alpha, observed_specific = g + taus[1] * dg, dg + taus[1] * ddg, a + taus[2] * da
        inertia = np.asarray(parameters.inertia)
        inverse = np.linalg.solve(inertia, np.eye(3))
        iomega = omega @ inertia.T
        alpha = (torque - np.cross(omega, iomega)) @ inverse.T
        ip_omega = np.einsum("abp,nb->nap", pj.inertia, omega)
        ip_alpha = np.einsum("abp,nb->nap", pj.inertia, alpha)
        alpha_j = np.einsum("ab,nbp->nap", inverse,
            torque_j - np.einsum("nab,nbp->nap", skew_batch(omega), ip_omega) - ip_alpha)
        alpha_tau = torque_tau @ inverse.T
        alpha_omega = np.einsum("ab,nbc->nac", inverse,
            skew_batch(iomega) - np.einsum("nab,bc->nac", skew_batch(omega), inertia))
        lever = self.sensor_position - parameters.cog_offset
        centripetal = np.cross(omega, np.cross(omega, lever))
        specific = force / parameters.mass + np.cross(alpha, lever) + centripetal
        omega_skew = skew_batch(omega)
        specific_j = force_j / parameters.mass - force[:, :, None] * pj.mass[None, None, :] / parameters.mass**2
        specific_j -= np.einsum("ab,nbp->nap", skew_batch(lever), alpha_j)
        specific_j -= np.einsum("nab,bp->nap", skew_batch(alpha)
            + np.einsum("nab,nbc->nac", omega_skew, omega_skew), pj.cog_offset)
        specific_tau = force_tau / parameters.mass - alpha_tau @ skew_batch(lever).T
        specific_omega = (-np.einsum("ab,nbc->nac", skew_batch(lever), alpha_omega)
            - skew_batch(np.cross(omega, lever)) - np.einsum("nab,bc->nac", omega_skew, skew_batch(lever)))
        residual = np.column_stack((observed_specific - specific, observed_alpha - alpha))
        joint = np.zeros((len(self.time), 6, 16))
        joint[:, :3, :13] = -specific_j @ self.basis
        joint[:, 3:, :13] = -alpha_j @ self.basis
        joint[:, :3, 13] = -specific_tau
        joint[:, 3:, 13] = -alpha_tau
        omega_tau = taus[1] * dg
        joint[:, :3, 14] = -np.einsum("nij,nj->ni", specific_omega, omega_tau)
        joint[:, 3:, 14] = taus[1] * ddg - np.einsum("nij,nj->ni", alpha_omega, omega_tau)
        joint[:, :3, 15] = taus[2] * da
        if not np.isfinite(residual).all() or not np.isfinite(joint).all():
            raise FloatingPointError("IMU Newton--Euler residual/Jacobian became non-finite")
        required_force = parameters.mass * (observed_specific - np.cross(observed_alpha, lever) - centripetal)
        required_torque = observed_alpha @ inertia.T + np.cross(omega, iomega)
        return {"parameters": parameters, "parameter_jacobian": pj, "physical_coordinate": physical,
            "taus": taus, "residual": residual, "jacobian": joint, "actual_thrust": thrust,
            "actual_thrust_log_tau_jacobian": dthrust, "actual_gimbal": self.gimbal,
            "corrected_gyro": omega, "corrected_angular_acceleration": observed_alpha,
            "corrected_specific_force": observed_specific, "predicted_specific_force": specific,
            "predicted_angular_acceleration": alpha, "specific_omega_jacobian": specific_omega,
            "alpha_omega_jacobian": alpha_omega, "modeled_wrench": np.column_stack((force, torque)),
            "required_wrench": np.column_stack((required_force, required_torque)),
            "raw_residual_wrench": np.column_stack((required_force - force, required_torque - torque))}

    def noise_maps(self, evaluation):
        n, tau_g, tau_a = len(self.time), evaluation["taus"][1], evaluation["taus"][2]
        base = np.concatenate((-evaluation["specific_omega_jacobian"], -evaluation["alpha_omega_jacobian"]), axis=1)
        gyro_maps = [base, tau_g * base, np.zeros_like(base)]
        gyro_maps[1][:, 3:, :] += np.eye(3)
        gyro_maps[2][:, 3:, :] = tau_g * np.eye(3)
        accel_zero = np.zeros((n, 6, 3)); accel_zero[:, :3] = np.eye(3)
        return gyro_maps, [accel_zero, tau_a * accel_zero, np.zeros_like(accel_zero)]

    def residual_covariance(self, evaluation):
        covariance = np.zeros((len(self.time), 6, 6))
        for signal, maps in zip((self.gyro, self.accel), self.noise_maps(evaluation)):
            local = np.asarray(signal["covariances"])
            for k in range(3):
                for ell in range(3):
                    covariance += np.einsum("nia,nab,njb->nij", maps[k], local[k, ell], maps[ell])
        return .5 * (covariance + np.swapaxes(covariance, 1, 2))

    def objective(self, x):
        if self._cached_x is not None and np.array_equal(x, self._cached_x):
            return self._cached_evaluation
        try:
            result = self.evaluate(x)
            residual = np.einsum("nij,nj->ni", self.whitening, result["residual"]).reshape(-1)
            jacobian = np.einsum("nij,njp->nip", self.whitening, result["jacobian"]).reshape(-1, 16)
        except (ValueError, FloatingPointError, OverflowError, np.linalg.LinAlgError):
            self.invalid_trial_count += 1
            residual = np.full(len(self.time) * 6, 1e12)
            jacobian = np.zeros((len(self.time) * 6, 16))
        self._cached_x, self._cached_evaluation = np.asarray(x).copy(), (residual, jacobian)
        return residual, jacobian

    def covariance(self, evaluation):
        """Full temporal SG sandwich, in physical-tau coordinates first."""
        jacobian = evaluation["jacobian"].copy()
        jacobian[:, :, 13:] /= evaluation["taus"][None, None, :]
        whitened = np.einsum("nij,njp->nip", self.whitening, jacobian).reshape(-1, 16)
        u, singular, vt = np.linalg.svd(whitened, full_matrices=False)
        if np.any(singular == 0):
            raise np.linalg.LinAlgError("exact zero information singular value; joint covariance is undefined")
        # Preserve every finite positive singular direction. Relative rank is
        # diagnostic only; it does not delete weak variances.
        inverse = (vt.T / singular**2) @ vt
        middle = np.zeros((16, 16))
        score_map = np.einsum("nip,nij->npj", jacobian, self.weight)
        least_squares_inverse = (vt.T / singular) @ u.T
        gain = np.einsum("npi,nij->npj", least_squares_inverse.reshape(16, len(self.time), 6).transpose(1, 0, 2), self.whitening)
        covariance_factors = []
        for signal, maps, noise in zip((self.gyro, self.accel), self.noise_maps(evaluation),
                (self.data["gyro_noise_covariance"], self.data["accel_noise_covariance"])):
            raw_map = np.zeros((len(self.time), 16, 3))
            raw_response = np.zeros_like(raw_map)
            for operator, mapping in zip(signal["operators"], maps):
                local = np.einsum("npi,nij->npj", score_map, mapping)
                raw_map += (operator.T @ local.reshape(len(self.time), 48)).reshape(len(self.time), 16, 3)
                local_response = np.einsum("npi,nij->npj", gain, mapping)
                raw_response += (operator.T @ local_response.reshape(len(self.time), 48)).reshape(len(self.time), 16, 3)
            middle += np.einsum("npa,ab,nqb->pq", raw_map, np.asarray(noise), raw_map)
            response = np.einsum("npa,ab->npb", raw_response, np.linalg.cholesky(np.asarray(noise)))
            covariance_factors.append(response.transpose(1, 0, 2).reshape(16, -1))
        residual_scores = np.einsum("npi,ni->np", score_map, evaluation["residual"])
        residual_middle = residual_scores.T @ residual_scores
        covariance_factors.append(np.einsum("npi,ni->np", gain, evaluation["residual"]).T)
        # Algebraically H^-1 (M_sensor + M_residual) H^-1, assembled as a
        # Gram factor through the SVD solve rather than ill-conditioned H.
        _q, compact = np.linalg.qr(np.concatenate(covariance_factors, axis=1).T, mode="reduced")
        covariance_factor = compact.T
        covariance = covariance_factor @ covariance_factor.T
        log_transform = np.diag(np.r_[np.ones(13), 1 / evaluation["taus"]])
        log_factor = log_transform @ covariance_factor
        log_covariance = log_factor @ log_factor.T
        joint_basis = joint_chart_basis()
        full_factor = joint_basis @ log_factor
        full_covariance = full_factor @ full_factor.T
        si_transform = np.zeros((17, 16))
        si_transform[:14, :13] = physical_parameter_jacobian(evaluation["parameter_jacobian"]) @ self.basis
        si_transform[14:, 13:] = np.eye(3)
        si_factor = si_transform @ covariance_factor
        si_covariance = si_factor @ si_factor.T
        if any(not np.isfinite(value).all() for value in (covariance, log_covariance, full_covariance, si_covariance)):
            raise FloatingPointError("joint IMU covariance became non-finite")
        threshold = singular[0] * max(whitened.shape) * np.finfo(float).eps
        return {"reduced_physical_tau": covariance, "reduced_log_tau": log_covariance,
                "full_chart_log_tau": full_covariance, "physical_si": si_covariance,
                "factor_reduced_physical_tau": covariance_factor,
                "factor_reduced_log_tau": log_factor,
                "factor_full_chart_log_tau": full_factor, "factor_physical_si": si_factor,
                "naive_reduced_physical_tau": inverse, "sg_sandwich_middle": middle,
                "uncentered_residual_score_middle": residual_middle,
                "singular_values": singular, "diagnostic_numerical_rank": int(np.count_nonzero(singular > threshold)),
                "relative_rank_threshold_diagnostic_only": float(threshold)}


def fit_case(data, model_path, options=None):
    started = walltime.perf_counter()
    model = load_vehicle_model(Path(model_path))
    problem = ImuDynamicsProblem(data, model, options)
    solved = least_squares(lambda x: problem.objective(x)[0], problem.initial,
        jac=lambda x: problem.objective(x)[1], method="trf", max_nfev=int(problem.options["max_nfev"]),
        ftol=float(problem.options["ftol"]), xtol=float(problem.options["xtol"]), gtol=float(problem.options["gtol"]))
    evaluation = problem.evaluate(solved.x)
    covariance = problem.covariance(evaluation)
    parameters = evaluation["parameters"]
    si_value = np.r_[physical_parameter_vector(parameters), evaluation["taus"]]
    reduced_labels = ["physical_quotient_" + str(i + 1) for i in range(13)] + list(LOG_TAU_LABELS)
    estimate = {"schema": "grape-param-estim/imu-first-order-lag-estimate/v1", "status": "completed",
        "observation_source": "native IMU gyro and accelerometer; no ongoing mocap",
        "provenance": input_provenance(data, model_path),
        "case_name": str(data.get("metadata", {}).get("case", data.get("case_name", "unspecified"))),
        "point_estimate": {"mass_kg": float(parameters.mass), "inertia_kg_m2": parameters.inertia.tolist(),
            "cog_position_body_m": parameters.cog_offset.tolist(), "force_effectiveness": parameters.force_effectiveness.tolist()},
        "time_constants_seconds": dict(zip(TAU_LABELS, map(float, evaluation["taus"]))),
        "optimization": {"success": bool(solved.success), "status": int(solved.status), "message": str(solved.message),
            "nfev": int(solved.nfev), "cost": float(solved.cost), "optimality": float(solved.optimality),
            "invalid_trial_count": problem.invalid_trial_count, "initial_reduced_log_tau_coordinate": problem.initial.tolist()},
        "parameter_distribution": {"reduced_log_tau_labels": reduced_labels,
            "reduced_log_tau_coordinate": np.asarray(solved.x).tolist(),
            "full_chart_labels": list(PHYSICAL_CHART_LABELS) + list(LOG_TAU_LABELS),
            "full_chart_coordinate": (joint_chart_basis() @ solved.x).tolist(),
            "full_chart_basis": joint_chart_basis().tolist(),
            "covariance_reduced_log_tau": covariance["reduced_log_tau"].tolist(),
            "covariance_full_chart_log_tau": covariance["full_chart_log_tau"].tolist(),
            "covariance_reduced_physical_tau": covariance["reduced_physical_tau"].tolist(),
            "physical_si_labels": list(SI_LABELS), "physical_si_value": si_value.tolist(),
            "covariance_physical_si": covariance["physical_si"].tolist(),
            "information_singular_values_physical_tau_chart": covariance["singular_values"].tolist(),
            "diagnostic_numerical_rank": covariance["diagnostic_numerical_rank"],
            "rank_threshold_is_diagnostic_only": True,
            "common_scale_gauge": "physical chart projected onto existing 13-D orthogonal common-scale section; tau coordinates unaffected"},
        "method": {"measurement_laws": "tau_g*dgyro/dt+gyro=omega; tau_a*daccel/dt+accel=specific_force",
            "whitening": "fixed at nominal physical parameters and command-period initial taus; quasi-GLS, not full parameter-dependent Gaussian likelihood",
            "covariance": "full native-SG temporal overlap sandwich plus uncentered residual score middle, all tau/physical cross retained",
            "monte_carlo": False, "options": problem.options,
            "sg_edge_policy": "same helper native local-polynomial shifted windows; no invented observations",
            "sensor_position_in_body_m": problem.sensor_position.tolist(),
            "sensor_lever_assumption": "FC origin used as nominal accelerometer origin; physical IMU origin not independently calibrated"},
        "elapsed_seconds": float(walltime.perf_counter() - started)}
    arrays = {"time": problem.time.copy(), "fit_valid_mask": np.ones(len(problem.time), dtype=bool),
        "gyro_smooth": problem.gyro["smooth"], "gyro_derivative": problem.gyro["derivative"],
        "gyro_second_derivative": problem.gyro["second_derivative"], "accel_smooth": problem.accel["smooth"],
        "accel_derivative": problem.accel["derivative"], "accel_second_derivative": problem.accel["second_derivative"],
        "residual_covariance_final": problem.residual_covariance(evaluation),
        "residual_covariance_nominal": problem.nominal_residual_covariance,
        "fixed_whitening": problem.whitening, "residual": evaluation["residual"],
        "raw_residual_wrench": evaluation["raw_residual_wrench"], "modeled_wrench": evaluation["modeled_wrench"],
        "required_wrench": evaluation["required_wrench"], "actual_thrust": evaluation["actual_thrust"],
        "actual_thrust_log_tau_jacobian": evaluation["actual_thrust_log_tau_jacobian"], "actual_gimbal": evaluation["actual_gimbal"],
        "corrected_gyro": evaluation["corrected_gyro"], "corrected_angular_acceleration": evaluation["corrected_angular_acceleration"],
        "corrected_specific_force": evaluation["corrected_specific_force"],
        "predicted_specific_force": evaluation["predicted_specific_force"],
        "predicted_angular_acceleration": evaluation["predicted_angular_acceleration"],
        "jacobian_reduced_log_tau": evaluation["jacobian"],
        "parameter_covariance_reduced_log_tau": covariance["reduced_log_tau"],
        "parameter_covariance_full_chart_log_tau": covariance["full_chart_log_tau"],
        "parameter_covariance_factor_reduced_log_tau": covariance["factor_reduced_log_tau"],
        "parameter_covariance_factor_full_chart_log_tau": covariance["factor_full_chart_log_tau"],
        "parameter_covariance_factor_physical_si": covariance["factor_physical_si"],
        "parameter_covariance_physical_si": covariance["physical_si"],
        "tau_covariance_seconds": covariance["reduced_physical_tau"][13:, 13:],
        "sg_sandwich_middle": covariance["sg_sandwich_middle"],
        "uncentered_residual_score_middle": covariance["uncentered_residual_score_middle"]}
    return {"estimate": estimate, "arrays": arrays, "model": model, "parameters": parameters,
        "chart": problem.chart, "physical_coordinate": evaluation["physical_coordinate"],
        "quotient_coordinate": np.asarray(solved.x[:13]).copy(), "log_taus": np.asarray(solved.x[13:16]).copy(),
        "taus": evaluation["taus"], "parameter_covariance_reduced": covariance["reduced_log_tau"],
        "parameter_covariance_full": covariance["full_chart_log_tau"], "joint_basis": joint_chart_basis(),
        "parameter_covariance_factor_reduced": covariance["factor_reduced_log_tau"],
        "parameter_covariance_factor_full": covariance["factor_full_chart_log_tau"],
        "parameter_covariance_factor_physical_si": covariance["factor_physical_si"],
        "tau_covariance_seconds": covariance["reduced_physical_tau"][13:, 13:],
        "problem": problem, "evaluation": evaluation, "solver": solved}


def load_fit_cache(path, model_path):
    """Restore a completed fit and its full cross-covariance without refitting.

    The cache contains numerical products, not pickled solver/model objects.
    Gram factors are retained so forward propagation need not eigendecompose a
    weakly identified covariance or alter its numerical null/gauge directions.
    """
    path = Path(path)
    estimate = json.loads((path / "estimate.json").read_text())
    with np.load(path / "arrays.npz", allow_pickle=False) as saved:
        arrays = {key: saved[key].copy() for key in saved.files}
    distribution = estimate["parameter_distribution"]
    coordinate = np.asarray(distribution["reduced_log_tau_coordinate"], dtype=float)
    basis = np.asarray(distribution["full_chart_basis"], dtype=float)
    full = np.asarray(distribution["full_chart_coordinate"], dtype=float)
    model = load_vehicle_model(Path(model_path))
    chart = SiParameterChart(model.parameters)
    parameters, parameter_jacobian = chart.decode_with_jacobian(full[:14])
    taus = np.exp(coordinate[13:])
    factor = arrays["parameter_covariance_factor_reduced_log_tau"]
    if "parameter_covariance_factor_physical_si" not in arrays:
        transform = np.zeros((17, 16))
        transform[:14, :13] = physical_parameter_jacobian(parameter_jacobian) @ basis[:14, :13]
        transform[14:, 13:] = np.diag(taus)
        arrays["parameter_covariance_factor_physical_si"] = transform @ factor
    return {"estimate": estimate, "arrays": arrays, "model": model, "parameters": parameters,
        "chart": chart, "physical_coordinate": full[:14], "quotient_coordinate": coordinate[:13],
        "log_taus": coordinate[13:], "taus": taus, "joint_basis": basis,
        "parameter_covariance_reduced": arrays["parameter_covariance_reduced_log_tau"],
        "parameter_covariance_full": arrays["parameter_covariance_full_chart_log_tau"],
        "parameter_covariance_factor_reduced": factor,
        "parameter_covariance_factor_full": arrays["parameter_covariance_factor_full_chart_log_tau"],
        "parameter_covariance_factor_physical_si": arrays["parameter_covariance_factor_physical_si"],
        "tau_covariance_seconds": arrays["tau_covariance_seconds"]}
