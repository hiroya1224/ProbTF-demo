"""Structural certificate for the requested IMU-only full closed-loop criterion.

This is not a numerical eigenvalue calculation or a replacement observer.
It certifies three untranslated *physical-position error* directions in a
uniform-field, position-independent plant controlled from an IMU-only estimate.
"""
import hashlib
import inspect
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
for directory in (HERE.parent, HERE.parent.parent/'src'):
    if str(directory) not in sys.path:
        sys.path.insert(0,str(directory))

from grape_param_estim.controller import GrapeController
from grape_param_estim.dynamics import FullSixDofPlant
from grape_param_estim.geometry import quaternion_to_matrix
from grape_param_estim.system import GRAVITY


def imu_observation_for_certificate(plant, state, actuators, *, lever_body=None,
                                    magnetic_world=None, residual_wrench_body=None):
    """Physical IMU equations used only by the structural regression tests.

    Body acceleration includes the sensor lever and the same optional applied
    body residual wrench as the physical derivative. No state is estimated.
    """
    lever = np.zeros(3) if lever_body is None else np.asarray(lever_body,float)
    magnetic = np.array([.2,-.1,.4]) if magnetic_world is None else np.asarray(magnetic_world,float)
    derivative = plant.derivative(0.,state.as_vector(),actuators,residual_wrench_body)
    rotation = quaternion_to_matrix(state.orientation_xyzw)
    gravity = np.array([0.,0.,-GRAVITY])
    specific = rotation.T@(derivative[7:10]-gravity)
    specific += np.cross(derivative[10:13],lever)
    specific += np.cross(state.angular_velocity,np.cross(state.angular_velocity,lever))
    return dict(gyro=state.angular_velocity.copy(), accel=specific,
                magnetic=rotation.T@magnetic)


def _source_record(value):
    source = Path(inspect.getsourcefile(value)).resolve()
    lines, first = inspect.getsourcelines(value)
    return dict(path=str(source), first_line=int(first), last_line=int(first+len(lines)-1),
                sha256=hashlib.sha256(source.read_bytes()).hexdigest())


def imu_only_stability_certificate(bundle):
    """Return a conditional mathematical certificate, never fabricated poles.

    The user-selected model has no continuing absolute-position observations.
    A valid full closed-loop Jacobian therefore has at least three eigenvalues
    exactly one. Unknown/absent hover equilibria are not assigned eigenvalues.
    """
    if bundle.get('external_position_aiding',False):
        raise ValueError('The IMU-only translation certificate excludes external position aiding')
    if bundle.get('position_dependent_external_force',False):
        raise ValueError('Position-dependent physical forces require a different observability analysis')
    if bundle.get('position_dependent_gravity',False) or bundle.get('position_dependent_magnetic_field',False):
        raise ValueError('The translation certificate requires uniform gravity and magnetic reference fields')
    model = bundle.get('vehicle_model',bundle.get('model'))
    if model is None:
        raise ValueError('The physical model is required to bind the certificate to its source')
    source_path = Path(model.source_path).resolve()
    taus = np.asarray(bundle['nominal_taus'],float)
    if taus.shape != (3,) or not np.isfinite(taus).all() or np.any(taus <= 0.):
        raise ValueError('Three positive finite identified time constants are required')
    count = len(bundle.get('plants',()))
    if count <= 0:
        raise ValueError('The full ordered model-node denominator is required')
    return dict(schema='grape/imu-only-pid-translation-certificate/v1',
        criterion='valid hover with finite FULL augmented plant-sensor-estimator-controller Jacobian spectral radius strictly below one',
        criterion_decided=True, numeric_evaluated=False,
        exact_spectral_radius=None, rho_lower_bound=1., known_unit_modes=3,
        full_state_dimension=None, hover_validity_evaluated=False,
        hover_exists=None, hover_trim=None, jacobian_computed=False,
        lower_bound_condition='whenever a hover equilibrium and its full closed-loop Jacobian exist',
        unit_modes='shift only true world position; hold estimated state, reference, controller, actuator and sensor states fixed',
        equilibrium_absence='if no valid hover exists, the stated stable-hover event is also false',
        stable_event_possible=False, stable_fraction=0., stable_models_certified=False,
        denominator=count, all_model_nodes_retained=True,
        undecodable_nodes_do_not_become_stable=True,
        statement='For all PID gains, Q=integral 1{valid hover and full rho<1} d(mu_model)=0 under the stated IMU-only assumptions. This does not claim a computed value for rho or hover existence.',
        assumptions=dict(uniform_gravity=True, gravity_world=[0.,0.,-GRAVITY],
            uniform_magnetic_reference_if_used=True, no_external_position_aiding=True,
            causal_navigation_from_imu_and_commands=True,
            controller_uses_estimated_position_velocity=True,
            physical_position_not_fed_to_controller=True,
            external_residual_wrench='absent or position-independent time/command/IMU driven input',
            measurement_noise='position independent; same realization preserves translation symmetry',
            position_dependent_force_callback_not_supported=True),
        known_initial_frame_caveat='An exactly known initial position/velocity can restrict a noise-free perfect-model run to an observer-error-zero manifold. Nominal tracking or stability on that restricted manifold is not ruled out. The three unobservable physical-position error modes remain in the FULL augmented internal-stability criterion; this is not a proof that every physical maneuver is impossible.',
        reduced_model_caveat='Removing gauge/error modes or analyzing an observable controlled subspace defines a different criterion and is not done here.',
        sensor_time_constants_seconds=dict(thrust=float(taus[0]),gyro=float(taus[1]),accel=float(taus[2])),
        sensor_lag_equations=dict(gyro='tau_g * ydot_g = omega_body - y_g',
            accel='tau_a * ydot_a = specific_force_at_sensor_body - y_a'),
        gyro_accel_lags_included_in_structural_argument=True,
        additional_velocity_modes_claimed=False,
        zero_drag_model=bool(not np.any(model.parameters.linear_drag)
                             and not np.any(model.parameters.angular_drag)),
        proof=dict(transformation='T_a(p,chi)=(p+a,chi)',
            observation_invariance='h_IMU(T_a x)=h_IMU(x); uniform-field magnetometer also invariant',
            closed_loop_equivariance='Psi(T_a X)=T_a Psi(X) for any causal IMU-only estimator/controller',
            derivative_identity='D(Psi) E_p = E_p, E_p=[I_3;0], rank(E_p)=3',
            jacobian_block_form='[[I_3, B],[0,C]]',
            consequence='rho(D(Psi)) >= 1; no claim about additional eigenvalues'),
        source=dict(vehicle_model=dict(path=str(source_path),
                sha256=hashlib.sha256(source_path.read_bytes()).hexdigest()),
            physical_wrench=_source_record(FullSixDofPlant.total_body_wrench),
            physical_derivative=_source_record(FullSixDofPlant.derivative),
            physical_analytic_derivative=_source_record(FullSixDofPlant.derivative_with_jacobian),
            physical_RK4_step=_source_record(FullSixDofPlant.step_with_jacobian),
            estimated_state_controller=_source_record(GrapeController.step),
            certificate_module=dict(path=str(Path(__file__).resolve()),
                sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())))
