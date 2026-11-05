"""Frozen IMU model laws and deterministic Gaussian nodes for PID evaluation.

This is a separate input contract, not a relabelled mocap estimate. All sixteen
joint coordinates and all three lags remain available. A controller model may
use the physical/thrust marginal without conditioning on the other two lags.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys
import warnings

import numpy as np
import scipy
from scipy.special import ndtri
from scipy.stats import qmc
from scipy.linalg import expm

HERE = Path(__file__).resolve().parent
MINIMAL = HERE.parent
for path in (MINIMAL, MINIMAL.parent/'src'):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from single_bag_savgol_core import SiParameterChart
from grape_param_estim.gimbalrotor_pid_postprocess import ScaleFreePlant, load_vehicle_model
from grape_param_estim.system import ActuatorParameters

SOURCE = HERE/'publication/imu_native_20261031'
CONTROLLER_CONTEXT_SOURCE = (MINIMAL/'first-order-lag/publication/'
    'pid_gain_voxels_failure_only_20261028/data')
GROUPS = ('xy', 'z', 'roll_pitch', 'yaw')
TAU_NAMES = ('thrust_time_constant_seconds', 'gyro_time_constant_seconds',
             'accel_time_constant_seconds')
SCHEMA = 'grape-param-estim/imu-pid-inputs/v1'


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _readonly(value):
    result = np.asarray(value).copy()
    result.setflags(write=False)
    return result


def deterministic_gaussian_nodes(node_count, dimension=16):
    """Unscrambled Sobol order, deterministic half-stratum shift, inverse CDF.

    These are equal-weight numerical integration nodes, not random posterior
    realizations. The empirical node covariance is not substituted for the
    given Gaussian covariance. The finite node rule remains an approximation.
    """
    n = int(node_count)
    if isinstance(node_count, bool) or n != node_count or n <= 0:
        raise ValueError('node_count must be a positive integer')
    count_power = int(math.ceil(math.log2(n))) if n > 1 else 0
    unit = qmc.Sobol(d=int(dimension), scramble=False).random_base2(count_power)[:n]
    shifted = np.remainder(unit + .5/n, 1.)
    normal = ndtri(shifted)
    if not np.isfinite(normal).all():
        raise FloatingPointError('Deterministic Gaussian node transformation is non-finite')
    return _readonly(normal), _readonly(shifted)


def _scale_free(parameters):
    return ScaleFreePlant(inertia_over_mass=parameters.inertia/parameters.mass,
        cog_position_body=parameters.cog_offset,
        force_effectiveness_over_mass=parameters.force_effectiveness/parameters.mass,
        rotor_lag_seconds=0.)


def scale_free_chart_node(chart, coordinate):
    """Decode exactly the same chart after cancelling its common mass scale.

    exp(S)/exp(log_m) = exp(S-log_m*I). Cancelling before exponentiation
    avoids a spurious raw-mass/inertia overflow when their ratios are finite.
    No eigenvalue, covariance, coordinate or integration weight is changed.
    """
    value = np.asarray(coordinate,float)
    log_second = chart._symmetric_log_matrix(value[1:7])-value[0]*np.eye(3)
    base = chart.reference_second_moment_sqrt
    second = (base@expm(log_second)@base)/chart.reference.mass
    second = .5*(second+second.T)
    inertia = np.trace(second)*np.eye(3)-second
    inertia = .5*(inertia+inertia.T)
    force = (chart.reference.force_effectiveness/chart.reference.mass)*np.exp(value[10:14]-value[0])
    return ScaleFreePlant(inertia_over_mass=inertia,
        cog_position_body=chart.reference.cog_offset+value[7:10],
        force_effectiveness_over_mass=force,rotor_lag_seconds=0.)


def _actuator(tau, limits):
    return ActuatorParameters(thrust_time_constant=float(tau), delay=0., **limits)


def load_imu_pid_inputs(case, source=SOURCE, node_count=128):
    """Load one failure's unchanged full IMU law and recorded PID context.

    Returns a dictionary containing ``vehicle_model``, ordered ``plants`` and
    per-node ``actuators``, ``reference_gains``, ``controller_dt``, the full
    sixteen-dimensional law/nodes, and source metadata. No file is written and
    no sensor lag is silently discarded or relabelled as an actuator delay.
    """
    if case not in ('failure1', 'failure2'):
        raise ValueError('The failure-only PID input cases are failure1 and failure2')
    source = Path(source).expanduser().resolve()
    estimate_path = source/case/'estimate.json'
    input_path, input_metadata_path = source/'inputs'/(case+'.npz'), source/'inputs'/(case+'.json')
    estimate = json.loads(estimate_path.read_text())
    input_metadata = json.loads(input_metadata_path.read_text())
    if (estimate.get('schema') != 'grape-param-estim/imu-first-order-lag-estimate/v1'
            or estimate.get('status') != 'completed' or estimate.get('case_name') != case):
        raise ValueError('A completed IMU first-order-lag estimate for this failure is required')
    provenance = estimate['provenance']
    archived = provenance['native_inputs']
    if _sha(input_path) != archived['arrays_sha256'] or _sha(input_metadata_path) != archived['metadata_sha256']:
        raise ValueError('The archived IMU identification input hashes do not match the estimate')
    bag_sha = provenance['source_bag_sha256']
    if input_metadata['bag_sha256'] != bag_sha:
        raise ValueError('IMU estimate and archived recording refer to different bags')
    distribution = estimate['parameter_distribution']
    mean = np.asarray(distribution['reduced_log_tau_coordinate'], float)
    covariance = np.asarray(distribution['covariance_reduced_log_tau'], float)
    basis = np.asarray(distribution['full_chart_basis'], float)
    if (mean.shape != (16,) or covariance.shape != (16,16) or basis.shape != (17,16)
            or not all(np.isfinite(a).all() for a in (mean, covariance, basis))):
        raise ValueError('The IMU model requires its complete finite 16-dimensional joint law')
    # The archived covariance is retained exactly. No rank threshold, flooring,
    # jitter, diagonalisation, conditional covariance or refit enters here.
    factor = np.linalg.cholesky(covariance)
    standard, unit = deterministic_gaussian_nodes(node_count, 16)
    # Fix the reduction order as well as the Sobol order; BLAS alignment must
    # not choose different floating reductions for separately launched jobs.
    coordinates = mean[None,:] + np.einsum('nk,ik->ni', standard, factor, optimize=False)
    full_chart = np.einsum('nk,ik->ni', coordinates, basis, optimize=False)
    with np.errstate(over='ignore', under='ignore', invalid='ignore'):
        taus = np.exp(coordinates[:,13:16])

    model_file = provenance['source_files']['vehicle_model']
    model_path = Path(model_file['path'])
    if not model_path.exists():
        model_path = MINIMAL/'grape_vehicle_model.json'
    if _sha(model_path) != model_file['sha256']:
        raise ValueError('The vehicle model differs from the frozen IMU identification model')
    vehicle_model = load_vehicle_model(model_path)
    chart = SiParameterChart(vehicle_model.parameters)
    nominal_parameters = chart.decode(basis[:14,:13] @ mean[:13])
    nominal_taus = np.exp(mean[13:16])

    # The old estimate supplies recorded controller/configuration metadata
    # only. Its mocap-derived point, covariance and fitted thrust tau are not
    # inputs to this calculation.
    context_path = Path(CONTROLLER_CONTEXT_SOURCE)/case/'estimate.json'
    context = json.loads(context_path.read_text())
    if context.get('case_name') != case or context['provenance']['bag_sha256'] != bag_sha:
        raise ValueError('Recorded PID context must be from the same failure bag')
    gains = np.array([[context['controller']['gains'][group][term+'_gain']
                      for term in ('p','i','d')] for group in GROUPS], float)
    old_actuator = context['actuator_model']
    limits = {name:float(old_actuator[name]) for name in
        ('gimbal_time_constant_seconds','maximum_gimbal_angle','maximum_gimbal_rate')}
    limits['gimbal_time_constant'] = limits.pop('gimbal_time_constant_seconds')
    limits.update({name:float(estimate['method']['options'][name])
                   for name in ('minimum_thrust','maximum_thrust')})
    with np.load(input_path, allow_pickle=False) as inputs:
        controller_dt = float(np.asarray(inputs['controller_dt']).reshape(-1)[0])
    if gains.shape != (4,3) or not np.isfinite(gains).all() or np.any(gains < 0.) or not np.isfinite(controller_dt) or controller_dt <= 0.:
        raise ValueError('A finite recorded controller and positive native command period are required')
    selected_context = dict(bag_sha256=bag_sha, gains=gains.tolist(), limits=limits,
        controller_dt=controller_dt, controller= context['controller'])
    context_hash = hashlib.sha256(json.dumps(selected_context,sort_keys=True).encode()).hexdigest()
    metadata = dict(schema=SCHEMA, case=case, source_bag_sha256=bag_sha,
        sources={name:dict(path=str(path.resolve()),sha256=_sha(path)) for name,path in
            (('imu_estimate',estimate_path),('imu_input',input_path),
             ('imu_input_metadata',input_metadata_path),('vehicle_model',model_path),
             ('recorded_controller_metadata_only',context_path))},
        controller_context_selected_sha256=context_hash,
        controller_context_role='same-bag gains and fixed actuator limits only; no legacy plant point/covariance/thrust tau',
        node_rule='unscrambled Sobol in16D; U=(Sobol+0.5/N)%1; inverse standard-normal CDF',
        node_count=len(coordinates), node_order='original Sobol prefix; no pruning, reordering or resampling',
        equal_weight=1./len(coordinates), stochastic_sampling=False, monte_carlo=False,
        finite_node_rule_is_approximation=True, success_input_used=False,
        full_coordinate_labels=distribution['reduced_log_tau_labels'],
        tau_order=list(TAU_NAMES), covariance_source='unchanged covariance_reduced_log_tau',
        covariance_sha256=_array_sha(covariance), factor_sha256=_array_sha(factor),
        factor_reconstruction_max_abs=float(np.max(np.abs(factor@factor.T-covariance))),
        standard_node_sha256=_array_sha(standard), coordinate_sha256=_array_sha(coordinates),
        full_joint_covariance_preserved=True,
        physical_thrust_marginal='first14 reduced coordinates, not conditional on gyro/accel tau',
        pure_delay_seconds=0., sensor_lag_policy='all three per-node taus retained; selected closed-loop model declares their use',
        numpy_version=np.__version__, scipy_version=scipy.__version__)
    physical, plants, actuators, errors, node_warnings, raw_errors = [], [], [], [], [], []
    for index, (row, tau) in enumerate(zip(full_chart, taus)):
        parameters = plant = actuator = None
        issues = []
        raw_error = None
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter('always', RuntimeWarning)
            try:
                parameters = chart.decode(row[:14])
            except (ValueError, OverflowError, FloatingPointError, np.linalg.LinAlgError) as error:
                raw_error = 'raw_parameter_decode: '+type(error).__name__+': '+str(error)
            try:
                plant = scale_free_chart_node(chart,row[:14])
            except (ValueError, OverflowError, FloatingPointError, np.linalg.LinAlgError) as error:
                issues.append('scale_free_plant_decode: '+type(error).__name__+': '+str(error))
            try:
                if not np.isfinite(tau).all() or np.any(tau <= 0.):
                    raise FloatingPointError('mapped per-node lag is non-finite/non-positive')
                actuator = _actuator(tau[0], limits)
            except (ValueError, OverflowError, FloatingPointError) as error:
                issues.append('actuator_decode: '+type(error).__name__+': '+str(error))
        physical.append(parameters); plants.append(plant); actuators.append(actuator)
        errors.append(None if not issues else '; '.join(issues))
        raw_errors.append(raw_error)
        node_warnings.append([str(w.message) for w in recorded])
    valid = np.array([p is not None and a is not None for p,a in zip(plants,actuators)],bool)
    metadata.update(valid_node_count=int(valid.sum()),
        unresolved_node_count=int(np.count_nonzero(~valid)),
        physical_decode_invalid_count=sum(p is None for p in plants),
        physical_decode_invalid_scope='mass-normalized PID plant, after analytic common-scale cancellation',
        raw_physical_parameter_unrepresentable_count=sum(e is not None for e in raw_errors),
        actuator_decode_invalid_count=sum(a is None for a in actuators),
        invalid_node_policy='original coordinate/order/weight retained; plant or actuator=None and node_error; unresolved remains in ALL-N denominator',
        node_errors=errors,node_warnings=node_warnings,raw_parameter_decode_errors=raw_errors,
        scale_free_decode='same SiParameterChart algebra with exp(S-log_mass*I) and exp(log_effectiveness-log_mass); no clipping/floor/jitter',
        nominal_point='decoded archived Gaussian coordinate centre, not expectation of nonlinear SI parameters')
    result = dict(estimate=estimate, metadata=metadata, vehicle_model=vehicle_model,
        model=vehicle_model, nominal_parameters=nominal_parameters,
        nominal_plant=_scale_free(nominal_parameters), nominal_actuator=_actuator(nominal_taus[0],limits),
        physical_parameters=tuple(physical), plants=tuple(plants), actuators=tuple(actuators),
        node_errors=tuple(errors), node_warnings=tuple(tuple(w) for w in node_warnings),
        raw_parameter_decode_errors=tuple(raw_errors),
        controller_dt=controller_dt, actuator_limits=limits)
    for name,value in dict(mean=mean,covariance=covariance,factor=factor,
        physical_basis=basis[:14,:13], full_chart_basis=basis,
        standard_nodes=standard,unit_nodes=unit, coordinates=coordinates,
        full_chart_coordinates=full_chart, taus=taus, nominal_taus=nominal_taus,
        reference_gains=gains,controller_gains=gains,
        valid_node_mask=valid,
        weights=np.full(len(coordinates),1./len(coordinates)),
        physical_thrust_mean=mean[:14],physical_thrust_covariance=covariance[:14,:14]).items():
        result[name]=_readonly(value)
    return result
