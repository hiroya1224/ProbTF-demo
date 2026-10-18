"""Deterministic transition and chart operations for change-point updating.

The reset branch is a specified redraw of the supplied body law.  It is not
variance inflation, a simulator-state observation, or a replacement drawing
law.  These operations use the existing Bingham-derived first-order proxy;
the probabilities and the observed likelihoods are evaluated by the caller.
"""

from dataclasses import replace
import math

import numpy as np
from scipy.special import logsumexp
from scipy.spatial.transform import Rotation

from probtf.geometry import quat_to_rotmat
from probtf.spherical_law.first_order import (
    JointTransformState, _right_jacobian,
)
from probtf.spherical_law.fusion import Gaussian
from probtf.spherical_law.support import covariance_support, require_in_support

from .body_projection import _partition, _require_common_body, _contraction_diagnostic
from .model import FrozenBodyBranch, WeightedState, component_log_weight
from .immutable_body_cache import reexpress_state, materialize_state_in_chart


def _same_time_geometry(left, right):
    if (left.edge_ids != right.edge_ids or left.version != right.version
            or left.stamp != right.stamp or left.command != right.command):
        raise ValueError('Transition charts must share geometry, version, time and command.')
    if any((a.bingham is None) != (b.bingham is None)
           for a, b in zip(left.transforms, right.transforms)):
        raise ValueError('Transition charts disagree on exact and uncertain rotations.')


def transition_priors(retained, direct, p=.25, marker_id='marker'):
    """Return ``(priors, provenance)`` for one common new-observation epoch.

    ``direct`` is the explicitly supplied command-dependent body family.
    The change kernel redraws those body variables independently of the
    retained environment.  The environment's actual marginal is reused,
    rather than reset to the initial unknown-marker law.  Its cross blocks
    vanish only in the reset branch, as specified by that redraw kernel.
    """
    probability = float(p)
    if not math.isfinite(probability) or not 0. < probability < 1.:
        raise ValueError('Change probability must lie strictly between zero and one.')
    source = retained.state
    _same_time_geometry(source, direct)
    if retained.label != source.component_id:
        raise ValueError('The retained prior label differs from its actual state.')
    if retained.weight != 1. or component_log_weight(retained) != 0.:
        raise ValueError('The retained internal proxy must have unit mass.')
    body, environment = _partition(source, marker_id)
    marker_index = source.edge_ids.index(marker_id)
    covariance = direct.covariance.copy()
    covariance[np.ix_(body, environment)] = 0.
    covariance[np.ix_(environment, body)] = 0.
    covariance[np.ix_(environment, environment)] = source.covariance[
        np.ix_(environment, environment)]
    transforms = tuple(old if index == marker_index else supplied
        for index, (old, supplied) in enumerate(zip(source.transforms, direct.transforms)))
    label = 'reset:{}'.format(source.version)
    if label == retained.label:
        raise ValueError('The retained label collides with the reserved reset label.')
    reset = JointTransformState(transforms, covariance,
        version=source.version, stamp=source.stamp, command=source.command,
        component_id=label)
    priors = (
        WeightedState(retained.label, 1.-probability, source, math.log1p(-probability)),
        WeightedState(label, probability, reset, math.log(probability)),
    )
    record = dict(
        change_probability=probability, change_prior_odds=probability/(1.-probability),
        retained_source_state_id=source.state_id, direct_family_state_id=direct.state_id,
        retained_label=retained.label, reset_label=label, reset_state_id=reset.state_id,
        version=source.version, stamp=source.stamp, command=list(source.command),
        marker_edge_id=marker_id, marker_marginal_reused=True,
        reset_kernel='independent redraw of the supplied command-dependent body law; retain the environment marginal',
        reset_body_environment_cross_covariance='zero by the specified independent redraw kernel',
        retained_body_environment_cross_covariance_preserved=True,
        physical_truth_received=False, sampling_used=False,
    )
    return priors, record


def common_chart_gaussian(source_ref, distribution, target_ref):
    """Transport one full joint Gaussian into a common Bingham-derived chart.

    For a source right-coordinate mean ``mu``, the target coordinate is
    ``Log(R_target.T R_source Exp(mu))``.  Its differential is
    ``J_r(target_mean)^-1 J_r(mu)``.  The same block map acts on every cross
    covariance.  This is a first-order coordinate change of the existing
    proxy, not an assumed Lie-algebra noise model.
    """
    _same_time_geometry(source_ref, target_ref)
    if (distribution.mean.shape != source_ref.mean.shape
            or distribution.covariance.shape != source_ref.covariance.shape):
        raise ValueError('A transported posterior must match its actual source chart.')
    require_in_support(distribution.mean-source_ref.mean,
        covariance_support(source_ref.covariance), 'transported posterior mean')
    if source_ref is target_ref or source_ref.state_id == target_ref.state_id:
        return distribution
    mean = distribution.mean.copy()
    chart = np.eye(mean.size)
    for index, (source, target) in enumerate(zip(source_ref.transforms, target_ref.transforms)):
        translation = slice(6*index, 6*index+3)
        rotation = slice(6*index+3, 6*index+6)
        mean[translation] += source.translation-target.translation
        source_mean = distribution.mean[rotation]
        relative = (quat_to_rotmat(target.rotation_wxyz).T
            @ quat_to_rotmat(source.rotation_wxyz)
            @ Rotation.from_rotvec(source_mean.copy()).as_matrix())
        vector = Rotation.from_matrix(relative).as_rotvec()
        # Only the actual principal-chart cut has no unique branch. A
        # finite near-pi coordinate still has an invertible right Jacobian;
        # an arbitrary angular margin must not stop later observations.
        if np.linalg.norm(vector) == math.pi:
            raise ValueError('Common-chart transport reaches the principal rotation cut.')
        mean[rotation] = vector
        chart[rotation, rotation] = np.linalg.solve(
            _right_jacobian(vector), _right_jacobian(source_mean))
    covariance = chart @ distribution.covariance @ chart.T
    covariance = .5*(covariance+covariance.T)
    return Gaussian(mean, covariance)


def projection_input(raw_result):
    """Return a temporary single-chart view for existing moment projection.

    The original result keeps its actual labelled transition priors and
    their density specifications.  Only this temporary projection view
    represents the posterior branches in the first (retained) chart.
    """
    branches = tuple(raw_result.frozen_body_branches)
    if not branches:
        raise ValueError('Projection requires the actual transition prior charts.')
    target = branches[0].reference_state
    if target.state_id != raw_result.prior.state_id:
        raise ValueError('The first transition chart must be the actual retained prior.')
    stage = raw_result.stages[-1]
    actual_by_label = {item.label: item for item in stage.components}
    if len(actual_by_label) != len(stage.components):
        raise ValueError('Posterior labels must be unique before chart transport.')
    raw, displayed, provenance = [], [], []
    for branch in branches:
        source = branch.reference_state
        _same_time_geometry(source, target)
        if (branch.label != source.component_id
                or not np.array_equal(branch.prior.mean, source.mean)
                or not np.array_equal(branch.prior.covariance, source.covariance)):
            raise ValueError('A transition branch lost its frozen prior correspondence.')
        for child in branch.posterior_components:
            if child.label not in actual_by_label:
                raise ValueError('A frozen posterior child has no actual source state.')
            actual = actual_by_label.pop(child.label)
            if (actual.state.component_id != child.label or actual.weight != child.weight
                    or component_log_weight(actual) != component_log_weight(child)
                    or actual.state.stamp != source.stamp
                    or actual.state.command != source.command):
                raise ValueError('Transported posterior provenance differs from its source.')
            identity = (np.array_equal(child.distribution.mean, source.mean)
                and np.array_equal(child.distribution.covariance, source.covariance)
                and actual.state.state_id == source.state_id)
            expected = source if identity else reexpress_state(source,
                child.distribution.mean, child.distribution.covariance,
                version=actual.state.version, component_id=actual.state.component_id)
            if expected.state_id != actual.state.state_id:
                raise ValueError('A raw posterior does not reexpress to its original source state.')
            transported = common_chart_gaussian(source, child.distribution, target)
            item = replace(child, distribution=transported)
            common = (actual.state if source.state_id == target.state_id else
                materialize_state_in_chart(target, transported.mean, transported.covariance,
                    version=actual.state.version, component_id=child.label))
            raw.append(item)
            displayed.append(replace(actual, state=common))
            provenance.append(dict(label=child.label, weight=child.weight,
                log_weight=component_log_weight(child),
                source_reference_state_id=source.state_id,
                source_posterior_state_id=actual.state.state_id,
                common_reference_state_id=target.state_id,
                common_posterior_state_id=common.state_id,
                source_mean=child.distribution.mean.tolist(),
                source_covariance=child.distribution.covariance.tolist(),
                common_mean=transported.mean.tolist(),
                common_covariance=transported.covariance.tolist()))
    if actual_by_label:
        raise ValueError('An actual posterior state lost its frozen source branch.')
    if not raw or not np.isclose(logsumexp([component_log_weight(item) for item in raw]),
                                0., atol=1e-12, rtol=0.):
        raise ValueError('The transported posterior masses must normalize globally.')
    prior = Gaussian(target.mean, target.covariance)
    temporary = FrozenBodyBranch(target.component_id, 1., target, prior, tuple(raw), 0.)
    records = dict(raw_result.records)
    records['bayes_factor_transport'] = dict(
        approximation='first-order transport of Bingham-derived proxies into the retained prior chart, followed by full joint mixture moments',
        common_reference_state_id=target.state_id, sources=provenance,
        original_prior_labels=[branch.label for branch in branches],
        original_prior_state_ids=[branch.reference_state.state_id for branch in branches],
        source_posterior_state_ids=[item.state.state_id for item in stage.components],
        source_posterior_parameter_identity_verified=True,
        common_chart_materialization=True,
        common_chart_is_probability_prior=False,
        reset_law_restricted_to_keep_prior_support=False,
        sampling_used=False, numerical_differentiation_used=False,
        density_ratio_uses_temporary_prior=False,
    )
    return replace(raw_result, records=records,
        stages=raw_result.stages[:-1]+(replace(stage, components=tuple(displayed)),),
        frozen_body_prior=prior,
        frozen_body_posterior=replace(raw_result.frozen_body_posterior, components=tuple(raw)),
        frozen_body_branches=(temporary,))


def retain_transition_goals(predicted_goals, raw_result, projection):
    """Keep each root goal marginal under a change-aware body-only update.

    The keep branch uses the declared body-side Gaussian coupling. The
    variational method transports the full marginal and supports expansion;
    historical modes retain the Schmidt contraction. The
    reset branch's independent redraw has zero environment cross covariance.
    Mixing these laws leaves every recorded marker marginal and goal mass
    unchanged, while the body's full mixture covariance includes between-
    branch uncertainty.  Its final chart map is the same as the projected
    internal body's reexpression.
    """
    goals = tuple(predicted_goals)
    branches = tuple(raw_result.frozen_body_branches)
    if not goals or len(branches) != 2:
        raise ValueError('Change-aware goal retention requires old goals and two transition branches.')
    if any(estimate.posterior is not None and estimate.detector_result is not None
           for estimate in raw_result.local_estimates):
        raise ValueError('A new marker observation must update the goal mixture instead of holding it.')
    reference = branches[0].reference_state
    marker_path = tuple(raw_result.marker_path)
    if len(marker_path) != 1 or marker_path[0][1]:
        raise ValueError('Fixed-goal retention needs the declared root marker variable.')
    marker_id = marker_path[0][0]
    body, environment = _partition(reference, marker_id)
    keep = branches[0]
    if keep.label != reference.component_id:
        raise ValueError('The first transition branch must identify the retained prior.')
    children = tuple(keep.posterior_components)
    if not children:
        raise ValueError('The retained hypothesis needs its positive posterior candidates.')
    keep_log_mass = float(logsumexp([component_log_weight(item) for item in children]))
    if not math.isfinite(keep_log_mass) or keep_log_mass > 1e-10:
        raise ValueError('The retained posterior mass must be finite and at most one.')
    weights = np.exp(np.array([component_log_weight(item) for item in children])-keep_log_mass)
    mean = sum(weight*item.distribution.mean for weight, item in zip(weights, children))
    covariance = sum(weight*(item.distribution.covariance
        + np.outer(item.distribution.mean-mean, item.distribution.mean-mean))
        for weight, item in zip(weights, children))
    prior_body = reference.covariance[np.ix_(body, body)]
    keep_body = covariance[np.ix_(body, body)]
    support = covariance_support(prior_body, 'fixed-goal retained hypothesis body')
    contraction=_contraction_diagnostic(prior_body,keep_body)
    if raw_result.records.get('observation_update_method')=='variational_moments':
        from .gaussian_marginal_transport import covariance_transport
        coupling,coupling_diagnostics=covariance_transport(prior_body,keep_body)
        coupling_method='gaussian_marginal_transport'
    else:
        coupling=keep_body@support.inverse
        coupling_diagnostics=dict(method='historical Schmidt contraction')
        coupling_method='schmidt'
    for column in keep_body.T:
        require_in_support(column, support, 'fixed-goal retained posterior body support')
    require_in_support(mean[body], support, 'fixed-goal retained posterior body mean')
    if (projection.posterior.mean.shape != reference.mean.shape
            or projection.state.edge_ids != reference.edge_ids
            or projection.state.stamp != reference.stamp
            or projection.state.command != reference.command):
        raise ValueError('The projected body differs from the current transition geometry or time.')
    chart = np.eye(reference.mean.size)
    for index, edge in enumerate(reference.transforms):
        if edge.edge_id != marker_id and edge.bingham is not None:
            block = slice(6*index+3, 6*index+6)
            chart[block, block] = _right_jacobian(projection.posterior.mean[block])
    body_map = chart[np.ix_(body, body)]
    keep_mass = float(np.exp(keep_log_mass))
    records, retained = [], []
    if not np.isclose(logsumexp([component_log_weight(item) for item in goals]),
                      0., atol=1e-12, rtol=0.):
        raise ValueError('Fixed-goal log masses must already normalize.')
    for item in goals:
        source = item.state
        if (item.label != source.component_id or source.stamp != reference.stamp
                or source.command != reference.command or source.version != reference.version):
            raise ValueError('A fixed goal lost its predicted label, time, command or version.')
        _require_common_body(reference, source, marker_id)
        old_cross = source.covariance[np.ix_(body, environment)]
        for column in old_cross.T:
            require_in_support(column, support, 'fixed-goal retained prior cross support')
        cross = keep_mass*body_map @ coupling @ old_cross
        full = projection.state.covariance.copy()
        full[np.ix_(body, environment)] = cross
        full[np.ix_(environment, body)] = cross.T
        full[np.ix_(environment, environment)] = source.covariance[
            np.ix_(environment, environment)]
        transforms = tuple(old if old.edge_id == marker_id else current
            for old, current in zip(source.transforms, projection.state.transforms))
        state = JointTransformState(transforms, full,
            version=projection.state.version, stamp=projection.state.stamp,
            command=projection.state.command, component_id=item.label)
        _require_common_body(projection.state, state, marker_id)
        retained.append(replace(item, state=state))
        records.append(dict(label=item.label, weight=item.weight,
            log_weight=component_log_weight(item), source_state_id=source.state_id,
            retained_state_id=state.state_id,
            source_body_environment_cross_covariance=old_cross.tolist(),
            retained_body_environment_cross_covariance=cross.tolist()))
    raw_result.records['bayes_factor_goal_retention'] = dict(
        policy='fixed root goal marginals; posterior-weighted declared keep-branch Gaussian coupling and independent reset branch',
        cross_formula='exp(log_keep_mass) J_body(total_mean) A_keep C_old',
        coupling_method=coupling_method,coupling_diagnostics=coupling_diagnostics,
        keep_posterior_log_mass=keep_log_mass, keep_posterior_mass=keep_mass,
        keep_conditional_body_mean=mean[body].tolist(),
        keep_conditional_body_covariance=keep_body.tolist(),
        fixed_goal_covariance_contraction=contraction,
        source_internal_state_id=reference.state_id,
        current_internal_state_id=projection.state.state_id,
        goal_components=records, goal_marginals_reconditioned=False,
        full_environment_bayes_posterior_claimed=False,
        sampling_used=False, covariance_regularization_added=False,
    )
    return tuple(retained)
