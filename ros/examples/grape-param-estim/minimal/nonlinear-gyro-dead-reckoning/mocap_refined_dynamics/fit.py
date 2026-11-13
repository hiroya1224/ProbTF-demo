"""Nominal-start nonlinear rotational least squares on a completed mocap prefix.

Only the rotational equation is fitted.  Its common inertia/torque scale is
unobservable, so trace(J) is held at its nominal value explicitly.  No prior,
residual bias nuisance, covariance floor, or held-out attitude enters the fit.
"""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import sys

import numpy as np
from scipy.linalg import expm, expm_frechet
from scipy.optimize import least_squares

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
_spec = importlib.util.spec_from_file_location('mocap_refinement_existing_torque', BASE / 'torque.py')
_torque = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _torque
_spec.loader.exec_module(_torque)

LABELS = ('inertia_shape_0', 'inertia_shape_1', 'inertia_shape_2',
          'inertia_shape_3', 'inertia_shape_4', 'cog_delta_x_m',
          'cog_delta_y_m', 'cog_delta_z_m', 'log_force_effectiveness_0',
          'log_force_effectiveness_1', 'log_force_effectiveness_2',
          'log_force_effectiveness_3', 'log_thrust_tau_seconds')
PHYSICAL_LABELS = ('J_xx', 'J_yy', 'J_zz', 'J_xy', 'J_xz', 'J_yz',
                   'cog_x', 'cog_y', 'cog_z', 'force_effectiveness_0',
                   'force_effectiveness_1', 'force_effectiveness_2',
                   'force_effectiveness_3', 'thrust_tau_seconds')


def _tracefree_basis():
    basis = [np.diag([1., -1., 0.]) / np.sqrt(2.),
             np.diag([1., 1., -2.]) / np.sqrt(6.)]
    for i, j in ((0, 1), (0, 2), (1, 2)):
        value = np.zeros((3, 3)); value[i, j] = value[j, i] = 1. / np.sqrt(2.)
        basis.append(value)
    return np.asarray(basis)


def _physical_vector(parameters):
    inertia = parameters['inertia']
    return np.r_[inertia[0, 0], inertia[1, 1], inertia[2, 2], inertia[0, 1],
                 inertia[0, 2], inertia[1, 2], parameters['cog_offset'],
                 parameters['force_effectiveness'], parameters['thrust_tau']]


class RotationalChart:
    """Physical inertia with fixed trace plus positive effectiveness and lag."""
    def __init__(self, model, initial_tau):
        self.model = model
        self.initial_tau = float(initial_tau)
        self.nominal_inertia = np.asarray(model.parameters.inertia, float)
        self.trace = float(np.trace(self.nominal_inertia))
        second = self.trace / 2. * np.eye(3) - self.nominal_inertia
        eig, axes = np.linalg.eigh(second)
        if np.any(eig <= 0.) or not np.isfinite(eig).all():
            raise ValueError('Nominal inertia must have positive physical second moment')
        self.root = (axes * np.sqrt(eig)) @ axes.T
        self.basis = _tracefree_basis()
        if not np.isfinite(self.initial_tau) or self.initial_tau <= 0.:
            raise ValueError('A positive finite initial tau is required')

    def initial(self):
        value = np.zeros(13); value[-1] = np.log(self.initial_tau)
        return value

    def decode(self, coordinate, *, jacobian=False):
        coordinate = np.asarray(coordinate, float)
        if coordinate.shape != (13,) or not np.isfinite(coordinate).all():
            raise ValueError('Finite 13-dimensional chart required')
        logarithm = np.einsum('i,ijk->jk', coordinate[:5], self.basis)
        raw = self.root @ expm(logarithm) @ self.root
        raw = (raw + raw.T) / 2.
        scale = self.trace / (2. * np.trace(raw))
        second = raw * scale
        inertia = self.trace / 2. * np.eye(3) - second
        effectiveness = np.asarray(self.model.parameters.force_effectiveness) * np.exp(coordinate[8:12])
        tau = float(np.exp(coordinate[-1]))
        result = dict(inertia=inertia,
                      cog_offset=np.asarray(self.model.parameters.cog_offset) + coordinate[5:8],
                      force_effectiveness=effectiveness, thrust_tau=tau,
                      torque_effectiveness=np.asarray(self.model.parameters.torque_effectiveness))
        if not all(np.isfinite(value).all() for value in result.values()):
            raise FloatingPointError('Nonfinite physical parameter decoding')
        if not jacobian:
            return result
        d_inertia = np.zeros((3, 3, 13))
        for column, basis in enumerate(self.basis):
            derivative = self.root @ expm_frechet(logarithm, basis, compute_expm=False) @ self.root
            derivative = (derivative + derivative.T) / 2.
            d_second = scale * (derivative - raw * np.trace(derivative) / np.trace(raw))
            d_inertia[:, :, column] = -d_second
        physical_jacobian = np.zeros((14, 13))
        for row, (i, j) in enumerate(((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))):
            physical_jacobian[row] = d_inertia[i, j]
        physical_jacobian[6:9, 5:8] = np.eye(3)
        physical_jacobian[9:13, 8:12] = np.diag(effectiveness)
        physical_jacobian[13, 12] = tau
        return result, d_inertia, physical_jacobian


class RotationalProblem:
    """Time-weighted torque residual, with an analytic 13-column Jacobian."""
    def __init__(self, data, kinematics, model, initial_tau):
        self.data = data
        self.time = np.asarray(kinematics['time'], float)
        self.omega = np.asarray(kinematics['omega_body_rad_s'], float)
        self.alpha = np.asarray(kinematics['alpha_body_rad_s2'], float)
        if (len(self.time) < 2 or self.omega.shape != (len(self.time), 3)
                or self.alpha.shape != self.omega.shape
                or not all(np.isfinite(x).all() for x in (self.time, self.omega, self.alpha))
                or np.any(np.diff(self.time) <= 0.)):
            raise ValueError('Finite increasing training time and Nx3 body kinematics required')
        self.chart = RotationalChart(model, initial_tau)
        self.weights = _torque._time_weights(self.time)
        self.sqrt_weights = np.sqrt(self.weights)
        self.direction, self.origins, self.reaction = _geometry_at(data, model, self.time)
        self._cached_coordinate = None
        self._cached = None
        self.evaluations = 0

    def evaluate(self, coordinate):
        coordinate = np.asarray(coordinate, float)
        if self._cached_coordinate is not None and np.array_equal(coordinate, self._cached_coordinate):
            return self._cached
        parameters, d_inertia, physical_jacobian = self.chart.decode(coordinate, jacobian=True)
        history = _history(self.data, self.time, parameters['thrust_tau'])
        force, rotor_torque, torque = _wrench_from_history(parameters, history.thrust,
            self.direction, self.origins, self.reaction)
        inertia = parameters['inertia']
        required = self.alpha @ inertia.T + np.cross(self.omega, self.omega @ inertia.T)
        residual = required - torque
        derivative = np.zeros((len(self.time), 3, 13))
        for column in range(5):
            dj = d_inertia[:, :, column]
            derivative[:, :, column] = self.alpha @ dj.T + np.cross(self.omega, self.omega @ dj.T)
        total_force = force.sum(axis=1)
        for column in range(3):
            # d tau / d CoG_j = F_total cross e_j.
            derivative[:, :, 5 + column] = -np.cross(total_force, np.eye(3)[column])
        derivative[:, :, 8:12] = -rotor_torque.transpose(0, 2, 1)
        _, _, d_tau = _wrench_from_history(parameters, history.log_tau_jacobian,
            self.direction, self.origins, self.reaction)
        derivative[:, :, 12] = -d_tau
        weighted_residual = residual * self.sqrt_weights[:, None]
        weighted_jacobian = derivative * self.sqrt_weights[:, None, None]
        self._cached_coordinate = coordinate.copy()
        self._cached = dict(parameters=parameters, physical_jacobian=physical_jacobian,
                            residual=residual, required=required, torque=torque,
                            force=force.sum(axis=1), thrust=history.thrust,
                            residual_vector=weighted_residual.ravel(),
                            jacobian=weighted_jacobian.reshape(-1, 13))
        self.evaluations += 1
        return self._cached

    def residual(self, coordinate):
        return self.evaluate(coordinate)['residual_vector']

    def jacobian(self, coordinate):
        return self.evaluate(coordinate)['jacobian']


def _geometry_at(data, model, time):
    time = np.asarray(time, float)
    stamps = np.asarray(data['gimbal_time'], float)
    gimbal = np.asarray(data['gimbal_angle'], float)
    if (time[0] < stamps[0] or time[-1] > stamps[-1]
            or np.any(np.diff(stamps) <= 0.) or gimbal.shape != (len(stamps), 4)):
        raise ValueError('Measured gimbal history must support every requested time')
    angles = np.column_stack([np.interp(time, stamps, gimbal[:, j]) for j in range(4)])
    geometry = model.body_geometry
    yaw = np.asarray(geometry.arm_yaws)
    direction = np.stack((np.sin(yaw)[None, :] * np.sin(angles),
                          -np.cos(yaw)[None, :] * np.sin(angles), np.cos(angles)), axis=-1)
    origins = np.asarray(geometry.rotor_origins)[None, :, :] + float(geometry.thrust_offset) * (
        direction - np.array([0., 0., 1.]))
    reaction = np.asarray(model.parameters.torque_effectiveness) * np.asarray(
        geometry.rotor_directions) * float(geometry.moment_force_rate)
    return direction, origins, reaction


def _history(data, time, tau):
    return _torque.first_order_zoh_thrust_history(
        command_times=np.asarray(data['rotor_command_time'], float),
        command_values=np.asarray(data['rotor_command'], float),
        evaluation_times=time, time_constant=tau,
        minimum_thrust=_torque.MINIMUM_THRUST, maximum_thrust=_torque.MAXIMUM_THRUST)


def _wrench_from_history(parameters, thrust, direction, origins, reaction):
    force = (thrust * parameters['force_effectiveness'][None, :])[:, :, None] * direction
    rotor_torque = np.cross(origins - parameters['cog_offset'], force)
    rotor_torque += reaction[None, :, None] * force
    return force, rotor_torque, rotor_torque.sum(axis=1)


def actuator_at(data, fit_result, time, *, use_initial=False):
    """Refined/initial actuator wrench on a query clock; no mocap access."""
    parameters = fit_result['initial_parameters'] if use_initial else fit_result['point_parameters']
    model = fit_result['model']
    direction, origins, reaction = _geometry_at(data, model, time)
    history = _history(data, np.asarray(time, float), parameters['thrust_tau'])
    force, _, torque = _wrench_from_history(parameters, history.thrust, direction, origins, reaction)
    return np.column_stack((force.sum(axis=1), torque)), history.thrust


def _rms(weights, value):
    return np.sqrt(weights @ (value * value))


def fit_rotational_model(data, kinematics, model_path, *, initial_tau=None, max_nfev=300):
    """Fit only supplied mocap training kinematics, starting at nominal physics.

    ``initial_tau=None`` selects the median retained command period as a
    numerical initialization (the nominal model JSON has no time constant).
    Overlapping SG residuals are correlated: saved Gauss--Newton curvature is
    a conditional local diagnostic, not a calibrated physical posterior.
    """
    model_path = Path(model_path).resolve()
    model = _torque.load_vehicle_model(model_path)
    command_time = np.asarray(data['rotor_command_time'], float)
    tau_initial = float(np.median(np.diff(command_time))) if initial_tau is None else float(initial_tau)
    problem = RotationalProblem(data, kinematics, model, tau_initial)
    if problem.time[-1] >= float(np.asarray(data['time'])[0]):
        raise ValueError('Mocap refinement must finish before prediction begins')
    initial_coordinate = problem.chart.initial()
    initial = problem.evaluate(initial_coordinate)
    solution = least_squares(problem.residual, initial_coordinate, jac=problem.jacobian,
        method='trf', x_scale='jac', loss='linear', max_nfev=int(max_nfev),
        ftol=1e-10, xtol=1e-10, gtol=1e-10)
    final = problem.evaluate(solution.x)
    _, singular, right_t = np.linalg.svd(final['jacobian'], full_matrices=False)
    tolerance = np.finfo(float).eps * max(final['jacobian'].shape) * singular[0]
    supported = singular > tolerance
    rank = int(np.count_nonzero(supported))
    variance = float(np.dot(solution.fun, solution.fun) / (solution.fun.size - rank))
    inverse_curvature = (right_t[supported].T / singular[supported]**2) @ right_t[supported]
    identifiable_covariance = variance * inverse_curvature
    physical_jacobian = final['physical_jacobian']
    initial_mean = problem.weights @ initial['residual']
    final_mean = problem.weights @ final['residual']
    initial_scatter = np.einsum('n,ni,nj->ij', problem.weights,
        initial['residual'] - initial_mean, initial['residual'] - initial_mean)
    final_scatter = np.einsum('n,ni,nj->ij', problem.weights,
        final['residual'] - final_mean, final['residual'] - final_mean)
    arrays = dict(training_time=problem.time, training_omega_body_rad_s=problem.omega,
        training_alpha_body_rad_s2=problem.alpha, training_time_weights=problem.weights,
        initial_coordinate=initial_coordinate, refined_coordinate=solution.x,
        coordinate_identifiable_covariance=identifiable_covariance,
        physical_identifiable_covariance=physical_jacobian @ identifiable_covariance @ physical_jacobian.T,
        jacobian_singular_values=singular, jacobian_right_singular_vectors=right_t,
        coordinate_null_basis=right_t[~supported].T,
        physical_parameter_jacobian=physical_jacobian,
        initial_physical_vector=_physical_vector(initial['parameters']),
        refined_physical_vector=_physical_vector(final['parameters']),
        initial_training_required_torque=initial['required'],
        refined_training_required_torque=final['required'],
        initial_training_actuator_torque=initial['torque'],
        refined_training_actuator_torque=final['torque'],
        initial_training_residual=initial['residual'], refined_training_residual=final['residual'],
        initial_residual_mean_body=initial_mean, refined_residual_mean_body=final_mean,
        initial_residual_scatter_body=initial_scatter, refined_residual_scatter_body=final_scatter,
        initial_weighted_residual_jacobian=initial['jacobian'],
        weighted_residual_jacobian=final['jacobian'])
    def json_parameters(value):
        return dict(inertia_body_kg_m2=value['inertia'].tolist(),
                    cog_offset_body_m=value['cog_offset'].tolist(),
                    force_effectiveness=value['force_effectiveness'].tolist(),
                    reaction_torque_effectiveness_fixed=value['torque_effectiveness'].tolist(),
                    thrust_tau_seconds=value['thrust_tau'])
    metadata = dict(schema='grape/mocap-refined-rotational-dynamics/v1',
        model_path=str(model_path), model_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),
        training_time_seconds=[float(problem.time[0]), float(problem.time[-1])],
        training_count=len(problem.time), objective='sum_n time_weight[n] ||J alpha + omega cross(J omega) - actuator_torque||^2',
        gyro_or_evaluation_mocap_used_in_fit=False,
        nominal_start=True, random_multistart=False, regularization=False,
        inertia_scale_gauge='trace(J) fixed at nominal; rotational equation alone cannot identify common J/torque scale',
        fixed_trace_inertia_kg_m2=float(np.trace(problem.chart.nominal_inertia)),
        inertia_chart='C=s*C0^(1/2) exp(sum_5 z_i B_i) C0^(1/2), tr(C)=tr(J_nom)/2; J=tr(C)I-C; tracefree symmetric B_i',
        mass='nominal, irrelevant to rotational-only equation; not identified',
        fixed_geometry='nominal rotor locations, arm yaws, gimbal thrust-origin offset and reaction moment/force ratio',
        coordinate_labels=list(LABELS), physical_labels=list(PHYSICAL_LABELS),
        initial=json_parameters(initial['parameters']), refined=json_parameters(final['parameters']),
        initial_tau_source=('median retained command period; numerical seed, not a physical nominal value'
                            if initial_tau is None else 'explicit numerical seed'),
        initial_jacobian_column_norm=np.linalg.norm(initial['jacobian'], axis=0).tolist(),
        initial_training_residual_rms_nm=_rms(problem.weights, initial['residual']).tolist(),
        refined_training_residual_rms_nm=_rms(problem.weights, final['residual']).tolist(),
        initial_training_residual_after_mean_rms_nm=_rms(problem.weights, initial['residual']-initial_mean).tolist(),
        refined_training_residual_after_mean_rms_nm=_rms(problem.weights, final['residual']-final_mean).tolist(),
        initial_residual_mean_body_nm=initial_mean.tolist(), refined_residual_mean_body_nm=final_mean.tolist(),
        residual_mean='computed only after physical LS from completed training prefix; frozen additive torque in subsequent prediction; not a free fit nuisance',
        optimizer=dict(method='scipy.optimize.least_squares trf', loss='ordinary quadratic',
                       success=bool(solution.success), status=int(solution.status), message=str(solution.message),
                       nfev=int(solution.nfev), njev=int(solution.njev), unique_evaluations=problem.evaluations,
                       max_nfev=int(max_nfev), cost=float(solution.cost), optimality=float(solution.optimality),
                       ftol=1e-10, xtol=1e-10, gtol=1e-10),
        jacobian_rank=rank, coordinate_dimension=13,
        jacobian_singular_values=singular.tolist(),
        numerical_rank_tolerance=float(tolerance),
        condition_number=float(singular[0]/singular[-1]) if singular[-1] > 0. else None,
        covariance_role='Gauss-Newton chart-support curvature diagnostic conditional on SG and fixed controls; overlapping SG errors are correlated and a near-boundary inertia chart can collapse, so not a calibrated physical posterior; discarded numerical chart directions do not mean physical certainty; see separate identifiability audit',
        finite_nonconvergence_policy='save optimizer residuals and continue prediction; no quality rejection',
        future_residual_or_dense_gyro_used=False)
    return dict(parameters=metadata, arrays=arrays, coordinate=solution.x,
                model=model, point_parameters=final['parameters'],
                initial_parameters=initial['parameters'], inertia=final['parameters']['inertia'],
                force_effectiveness=final['parameters']['force_effectiveness'],
                cog_offset=final['parameters']['cog_offset'], thrust_tau=final['parameters']['thrust_tau'],
                residual_mean=final_mean, initial_residual_mean=initial_mean)


def build_refined_profile(data, fit_result, *, profile_step_seconds=.001, use_initial=False):
    """Same held-out controls and frozen training residual for every noise/rate."""
    parameters = fit_result['initial_parameters'] if use_initial else fit_result['point_parameters']
    residual_mean = fit_result['initial_residual_mean'] if use_initial else fit_result['residual_mean']
    interval = np.asarray(data['time'], float)
    step = float(profile_step_seconds)
    if not np.isfinite(step) or step <= 0.:
        raise ValueError('Positive finite torque profile spacing required')
    training_time = np.asarray(fit_result['arrays']['training_time'], float)
    first = float(training_time[0]); last = float(interval[1])
    uniform = np.linspace(first, last, int(np.ceil((last-first)/step))+1)
    time = np.unique(np.r_[uniform, training_time, interval])
    wrench, thrust = actuator_at(data, fit_result, time, use_initial=use_initial)
    arrays = dict(torque_profile_time=time, torque_actuator_body=wrench[:, 3:],
                  torque_applied_body=wrench[:, 3:] + residual_mean,
                  actuator_wrench_body=wrench, profile_thrust=thrust,
                  residual_mean_body=np.asarray(residual_mean), inertia_body=parameters['inertia'])
    metadata = dict(schema='grape/mocap-refined-prediction-torque/v1',
        role='nominal-start baseline' if use_initial else 'refined nonlinear model',
        physical_parameters=fit_result['parameters']['initial' if use_initial else 'refined'],
        residual_mean_body_nm=np.asarray(residual_mean).tolist(),
        training_time_seconds=fit_result['parameters']['training_time_seconds'],
        profile_support_seconds=[float(time[0]), float(time[-1])],
        torque_interpolation='piecewise linear on bounded-spacing deterministic actuator grid',
        maximum_profile_step_seconds=float(np.max(np.diff(time))),
        command_model='same exact existing first-order ZOH thrust response and fixed physical saturation',
        gimbal_model='actual measured gimbal, linearly interpolated at original header times',
        heldout_mocap_or_gyro_used_to_construct_torque=False,
        prediction_band='conditional on fitted physical point, fixed control profile and frozen past residual mean; only requested omega law is propagated')
    return dict(torque=_torque.TorqueProfile(time, wrench[:, 3:], residual_mean),
                actuator_torque=_torque.TorqueProfile(time, wrench[:, 3:]),
                inertia=parameters['inertia'], residual_mean=residual_mean,
                arrays=arrays, metadata=metadata)
