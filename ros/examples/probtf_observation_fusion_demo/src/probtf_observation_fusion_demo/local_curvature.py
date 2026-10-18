"""Positive local curvature from the existing frozen full-objective probes.

This is a complete-component Fisher/Gauss-Newton approximation, not the exact
observed Hessian of the mixture criterion. The unchanged full-J gradient keeps
all original normalizer/covariance/mixture effects. No additional command push,
pose query, SO(3) logarithm, probability sample or candidate update occurs here.
"""
import math

import numpy as np

from probtf.geometry import quat_left_matrix
from probtf.spherical_law.support import covariance_support
from .joint_batch_fusion import log_evidence_parts_difference
from .moving_control import _relative_mixture_log_parts


METHOD = 'mixture_gaussian_bingham_tangent_fisher'


def probe_derivative(source, coordinates, getter):
    """Reuse precisely the successful full-J stencil, including one-sided ends."""
    source = tuple(float(value) for value in source)
    baseline = np.asarray(getter(source), dtype=float)
    output = []
    for coordinate in coordinates:
        index = coordinate['joint_index']
        values = {}
        for side in ('forward', 'backward'):
            probe = coordinate[side]
            if probe['status'] != 'ok':
                continue
            command = list(source)
            command[index] += probe['step']
            values[side] = (abs(probe['step']),
                np.asarray(getter(tuple(command)), dtype=float)-baseline)
        if len(values) == 2:
            hp, plus = values['forward']
            hm, minus = values['backward']
            derivative = (hm*plus/hp-hp*minus/hm)/(hp+hm)
        elif 'forward' in values:
            hp, plus = values['forward']
            derivative = plus/hp
        elif 'backward' in values:
            hm, minus = values['backward']
            derivative = -minus/hm
        else:
            raise ValueError('Local curvature has no successful task probe for joint '+str(index))
        output.append(derivative)
    result = np.stack(output, axis=-1)
    if not np.all(np.isfinite(result)):
        raise FloatingPointError('Nonfinite local sensitivity from the frozen full-J probes.')
    return result


def _fisher_rows(covariance, mean_derivative, covariance_derivative, name):
    support = covariance_support(covariance, name)
    whitening = (support.basis/np.sqrt(support.eigenvalues)).T
    mean_rows = whitening @ mean_derivative
    # The covariance contribution is .5 tr(S+ dS_i S+ dS_j), expressed as
    # a Gram matrix to retain its positive semidefinite mathematical form.
    whitened = np.einsum('ai,ijn,bj->abn', whitening, covariance_derivative, whitening)
    covariance_rows = whitened.reshape((-1, mean_derivative.shape[-1]))/math.sqrt(2.)
    return np.vstack((mean_rows, covariance_rows)), support.rank


def _responsibilities(branches):
    total = _relative_mixture_log_parts(branches)
    logs = [float(branch['log_weight'])+log_evidence_parts_difference(
        branch['log_density_parts'], total) for branch in branches]
    finite = [value for value in logs if math.isfinite(value)]
    if not finite:
        raise ValueError('No finite mixture mass for local task curvature.')
    reference = max(finite)
    scaled = [math.exp(value-reference) for value in logs]
    mass = math.fsum(scaled)
    return [value/mass for value in scaled], logs


def build_local_curvature(task, task_probes, view_probes, visibility_factor, settings,
                          *, lower=None, upper=None):
    """Build H from the same source/captured-K 13 or fewer task evaluations.

    Each component's comparison Gaussian supplies a position Fisher metric.
    Bingham's existing high-concentration covariance supplies a quaternion
    tangent Gaussian Fisher approximation. Probe quaternions are sign-aligned
    to the source; their means and tangent covariance are linearly projected
    into that source tangent. This adds no new rotation distribution or
    exponential/logarithm model. Native mixture modes remain separate.
    """
    source = tuple(task['source_command'])
    coordinates = task['finite_difference_diagnostics']['coordinates']
    branches = task_probes[source]['branches']
    count = len(source)
    absolute = float(settings.get('regularization_absolute', 1e5))
    relative = float(settings.get('regularization_relative', 1e-6))
    method = str(settings.get('method', METHOD))
    if (method != METHOD or not math.isfinite(absolute) or absolute <= 0.
            or not math.isfinite(relative) or relative < 0.):
        raise ValueError('Local curvature needs the declared Fisher method and finite positive absolute/nonnegative relative regularization.')
    weights, log_weights = _responsibilities(branches)
    task_rows, component_records = [], []
    for index, (branch, weight) in enumerate(zip(branches, weights)):
        def component(command):
            return task_probes[command]['branches'][index]
        position_mean = probe_derivative(source, coordinates,
            lambda command: component(command)['relative_position_mean'])
        position_covariance = probe_derivative(source, coordinates,
            lambda command: component(command)['relative_position_covariance'])
        position_rows, rank = _fisher_rows(branch['relative_position_covariance'],
            position_mean, position_covariance, 'local curvature relative position')
        quaternion = np.asarray(branch['relative_orientation_mean_wxyz'])
        tangent = quat_left_matrix(quaternion)[:, 1:]
        def orientation(command):
            trial = component(command)
            q = np.asarray(trial['relative_orientation_mean_wxyz'])
            if np.dot(q, quaternion) < 0.:
                q = -q
            trial_tangent = quat_left_matrix(q)[:, 1:]
            projection = tangent.T @ trial_tangent
            covariance = np.asarray(trial['query_covariance'])[3:, 3:]
            return (2.*tangent.T @ (q-quaternion),
                    projection @ covariance @ projection.T)
        rotation_mean = probe_derivative(source, coordinates,
            lambda command: orientation(command)[0])
        rotation_covariance = probe_derivative(source, coordinates,
            lambda command: orientation(command)[1])
        rotation_rows, rotation_rank = _fisher_rows(orientation(source)[1],
            rotation_mean, rotation_covariance, 'local curvature Bingham source tangent')
        rows = np.vstack((position_rows, rotation_rows))
        task_rows.append(math.sqrt(weight)*rows)
        component_records.append(dict(index=index, label=branch.get('label'),
            source_state_id=branch['state_id'], input_log_weight=branch['log_weight'],
            responsibility=weight, log_responsibility=log_weights[index],
            position_support_rank=rank, orientation_support_rank=rotation_rank,
            unweighted_metric_trace=float(np.sum(rows*rows))))
    task_rows = np.vstack(task_rows)
    task_metric = task_rows.T @ task_rows
    view_rows = []
    view_components = []
    active = visibility_factor.enabled and visibility_factor.weight > 0.
    if active:
        source_view = view_probes[source]
        for index, mode in enumerate(source_view['per_mode_record']):
            position = np.asarray(mode['position_mean'])
            radius = math.hypot(*position)
            unit = position/radius
            derivative = probe_derivative(source, coordinates,
                lambda command: view_probes[command]['per_mode_record'][index]['position_mean'])
            if visibility_factor.penalty_mode == 'tolerance_cone':
                on = not mode['inside_tolerance_cone']
                rows = ((unit[2]*unit-np.array([0., 0., 1.])) @ derivative/radius
                        if on else np.zeros(count))[None, :]
            else:
                on = True
                rows = (np.eye(3)-np.outer(unit, unit)) @ derivative/radius
            view_rows.append(math.sqrt(visibility_factor.weight*mode['normalized_weight'])*rows)
            view_components.append(dict(index=index, source_state_id=mode['source_state_id'],
                normalized_weight=mode['normalized_weight'], active_residual=on))
    view_rows = np.vstack(view_rows) if view_rows else np.zeros((0, count))
    view_metric = view_rows.T @ view_rows
    data = task_metric+view_metric
    regularization = absolute+relative*float(np.trace(data))/count
    curvature = data+regularization*np.eye(count)
    curvature = .5*curvature+.5*curvature.T
    if not np.all(np.isfinite(curvature)):
        raise FloatingPointError('Nonfinite local curvature matrix.')
    gradient = np.asarray(task['selected_gradient'])
    equilibrium, equilibrium_error = None, None
    try:
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            candidate = np.asarray(source)-np.linalg.solve(curvature, gradient)
        if np.all(np.isfinite(candidate)):
            equilibrium = candidate
        else:
            equilibrium_error = 'Nonfinite diagnostic stationary command.'
    except (np.linalg.LinAlgError, FloatingPointError) as error:
        equilibrium_error = type(error).__name__+': '+str(error)
    diagnostics = dict(method=METHOD, exact_Hessian=False,
        approximation='responsibility-weighted complete-component Fisher; Gaussian position and Bingham-derived concentrated quaternion tangent Gaussian; view residual Gauss-Newton',
        observed_mixture_negative_score_covariance_omitted=True,
        full_task_gradient_unchanged=True, candidate_law_modified=False,
        position_orientation_product_is_existing_comparison_proxy=True,
        source_quaternion_tangent='2*N(source_q).T*(sign_aligned_trial_q-source_q); covariance projected by N(source_q).T*N(trial_q)',
        task_curvature=task_metric.tolist(), view_curvature=view_metric.tolist(),
        regularization_absolute=absolute, regularization_relative=relative,
        regularization_scale_trace_over_dimension=float(np.trace(data))/count,
        regularization_added=regularization,
        regularization_role='declared local model only; not candidate covariance, K or control gain',
        component_count=len(branches), task_components=component_records,
        view_components=view_components, task_probe_count=len(task_probes),
        additional_command_pushes=0, additional_core_queries=0, monte_carlo_used=False,
        source_stamp=task['source_stamp'], source_command=list(source),
        body_state_ids=task['body_state_ids'], goal_state_ids=task['goal_state_ids'],
        parameter_density_id=task['parameter_density_id'], selected_K=task['selected_K'],
        unconstrained_equilibrium_command=None if equilibrium is None else equilibrium.tolist(),
        unconstrained_equilibrium_diagnostic_error=equilibrium_error,
        unconstrained_equilibrium_within_joint_positions=(None if equilibrium is None or lower is None or upper is None
            else bool(np.all(equilibrium >= lower) and np.all(equilibrium <= upper))),
        numerical_quality_is_acceptance_gate=False)
    # H is defined by its Gram construction plus positive declared ridge.
    # Neither this factorization nor a stationary-point solve is required by
    # the actual affine-model dynamics; their audit failure is not a gate.
    try:
        np.linalg.cholesky(curvature)
        diagnostics['positive_definite_cholesky_audit'] = True
    except np.linalg.LinAlgError as error:
        diagnostics.update(positive_definite_cholesky_audit=False,
                           cholesky_audit_error=str(error))
    try:
        eigenvalues = np.linalg.eigvalsh(curvature)
        diagnostics['eigenvalues'] = eigenvalues.tolist() if np.all(np.isfinite(eigenvalues)) else None
    except np.linalg.LinAlgError as error:
        diagnostics.update(eigenvalues=None, eigenvalue_audit_error=str(error))
    return curvature, diagnostics
