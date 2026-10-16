"""Single retained joint state and explicitly conditional goal hypotheses.

Raw observation hypotheses remain separate.  Only the retained internal state
is approximated by common-chart Gaussian moments and the existing Bingham
reexpression.  Goal construction shares that state's body marginal while
retaining each hypothesis's conditional marker law.
"""

from dataclasses import dataclass, replace

import numpy as np
from scipy.special import logsumexp

from probtf.spherical_law.first_order import JointTransformState, reexpress_state
from probtf.spherical_law.fusion import Gaussian
from probtf.spherical_law.support import covariance_support

from .model import component_log_weight


@dataclass(frozen=True)
class Projection:
    state: JointTransformState
    posterior: Gaussian
    goal_components: tuple
    record: dict


def _partition(state, marker_id):
    if marker_id not in state.edge_ids:
        raise ValueError('The declared marker variable is absent from the joint state.')
    marker_index = state.edge_ids.index(marker_id)
    environment = np.arange(6*marker_index, 6*marker_index+6)
    body = np.array([coordinate for index in range(len(state.transforms))
        if index != marker_index for coordinate in range(6*index, 6*index+6)], dtype=int)
    return body, environment


def _same_edge(left, right):
    return (left.edge_id == right.edge_id
        and np.array_equal(left.translation, right.translation)
        and np.array_equal(left.translation_covariance, right.translation_covariance)
        and np.array_equal(left.rotation_wxyz, right.rotation_wxyz)
        and ((left.bingham is None and right.bingham is None)
             or (left.bingham is not None and right.bingham is not None
                 and np.array_equal(left.bingham, right.bingham))))


def _require_common_body(left, right, marker_id):
    body, _ = _partition(left, marker_id)
    if (left.edge_ids != right.edge_ids
            or any(not _same_edge(a, b) for a, b in zip(left.transforms, right.transforms)
                   if a.edge_id != marker_id)
            or not np.array_equal(left.covariance[np.ix_(body, body)],
                                  right.covariance[np.ix_(body, body)])):
        raise ValueError('Goal hypotheses and the internal state must have bit-identical body marginals.')


def _check_weights(components):
    logs = np.array([component_log_weight(item) for item in components])
    normalizer = float(logsumexp(logs))
    if not np.isfinite(normalizer) or not np.isclose(normalizer, 0., atol=1e-12, rtol=0.):
        raise ValueError('Global hypothesis log weights must already sum to one.')
    return np.exp(logs-normalizer), normalizer


def _conditional_retarget(distribution, target, body, environment):
    """Replace B's marginal, retaining this Gaussian's E|B regression law."""
    pbb = distribution.covariance[np.ix_(body, body)]
    peb = distribution.covariance[np.ix_(environment, body)]
    pee = distribution.covariance[np.ix_(environment, environment)]
    regression = peb @ covariance_support(pbb, 'conditional goal body').inverse
    conditional = pee-regression @ peb.T
    conditional = .5*(conditional+conditional.T)
    covariance_support(conditional, 'conditional marker residual')
    target_body = target.covariance[np.ix_(body, body)]
    mean = target.mean.copy()
    mean[environment] = (distribution.mean[environment]
        + regression @ (target.mean[body]-distribution.mean[body]))
    covariance = target.covariance.copy()
    cross = target_body @ regression.T
    covariance[np.ix_(body, environment)] = cross
    covariance[np.ix_(environment, body)] = cross.T
    covariance[np.ix_(environment, environment)] = (
        conditional+regression @ target_body @ regression.T)
    return Gaussian(mean, covariance), regression, conditional


def _source_record(item, reference_id):
    return dict(label=item.label, weight=float(item.weight),
                log_weight=component_log_weight(item), source_prior_state_id=reference_id)


def project_update(result):
    """Approximate the internal joint law once, without dropping current modes."""
    branches = tuple(result.frozen_body_branches)
    if len(branches) != 1:
        raise ValueError('A projected update requires one retained internal prior chart.')
    branch = branches[0]
    reference = branch.reference_state
    if (branch.label != reference.component_id
            or not np.array_equal(branch.prior.mean, reference.mean)
            or not np.array_equal(branch.prior.covariance, reference.covariance)):
        raise ValueError('The frozen prior differs from its actual joint source.')
    marker_path = tuple(result.marker_path)
    if len(marker_path) != 1 or marker_path[0][1]:
        raise ValueError('A projected update needs the declared single marker variable.')
    marker_id = marker_path[0][0]
    body, environment = _partition(reference, marker_id)
    raw = tuple(branch.posterior_components)
    displayed = tuple(result.stages[-1].components)
    if not raw or len(raw) != len(displayed):
        raise ValueError('Every raw observation hypothesis needs its actual displayed source.')
    weights, normalizer = _check_weights(raw)
    if len({item.label for item in raw}) != len(raw):
        raise ValueError('Raw observation hypothesis labels must be unique.')
    for source, actual in zip(raw, displayed):
        if (source.label != actual.label or source.weight != actual.weight
                or component_log_weight(source) != component_log_weight(actual)
                or actual.state.component_id != actual.label
                or actual.state.stamp != reference.stamp
                or actual.state.command != reference.command):
            raise ValueError('Raw and displayed hypothesis provenance differs.')
        identity = (np.array_equal(source.distribution.mean, reference.mean)
            and np.array_equal(source.distribution.covariance, reference.covariance)
            and actual.state.state_id == reference.state_id)
        expected = reference if identity else reexpress_state(reference,
            source.distribution.mean, source.distribution.covariance,
            version=actual.state.version, component_id=actual.state.component_id)
        if expected.state_id != actual.state.state_id:
            raise ValueError('Raw posterior parameters do not reexpress to their actual source.')
    versions = {item.state.version for item in displayed}
    if len(versions) != 1:
        raise ValueError('Current observation hypotheses must share one posterior version.')

    if len(raw) == 1:
        # The already computed state, including an empty-observation identity,
        # must not undergo an extra numerical reexpression.
        posterior, state = raw[0].distribution, displayed[0].state
        within = posterior.covariance
        between = np.zeros_like(within)
        goals = displayed
        conditional_records = []
    else:
        mean = sum(weight*item.distribution.mean for weight, item in zip(weights, raw))
        within = np.zeros_like(reference.covariance)
        between = np.zeros_like(within)
        for weight, item in zip(weights, raw):
            difference = item.distribution.mean-mean
            within += weight*item.distribution.covariance
            between += weight*np.outer(difference, difference)
        posterior = Gaussian(mean, within+between)
        version = displayed[0].state.version
        state = reexpress_state(reference, posterior.mean, posterior.covariance,
            version=version, component_id='internal:{}'.format(version))
        goals, conditional_records = [], []
        for source, actual in zip(raw, displayed):
            targeted, regression, conditional = _conditional_retarget(
                source.distribution, posterior, body, environment)
            goal = reexpress_state(reference, targeted.mean, targeted.covariance,
                version=version, component_id=actual.state.component_id)
            _require_common_body(state, goal, marker_id)
            goals.append(replace(actual, state=goal))
            conditional_records.append(dict(label=source.label,
                body_to_marker_regression=regression.tolist(),
                marker_conditional_covariance=conditional.tolist(),
                retargeted_raw_mean=targeted.mean.tolist(),
                retargeted_raw_covariance=targeted.covariance.tolist()))
        goals = tuple(goals)
    record = dict(
        approximation='single internal common-chart joint Gaussian by analytic mixture moments; existing first-order Bingham reexpression once',
        coordinate_origin='Bingham-derived right rotation coordinates and parent-frame translations',
        scope='entire retained joint state, including the unknown marker variable',
        marker_edge_id=marker_id, sources=[_source_record(item, reference.state_id) for item in raw],
        raw_source_state_ids=[item.state.state_id for item in displayed],
        source_global_log_normalizer=normalizer, source_component_count=len(raw),
        raw_mean=posterior.mean.tolist(), raw_covariance=posterior.covariance.tolist(),
        raw_within_covariance=within.tolist(), raw_between_covariance=between.tolist(),
        source_prior_state_id=reference.state_id, internal_state_id=state.state_id,
        goal_state_ids=[item.state.state_id for item in goals],
        goal_labels=[item.label for item in goals],
        goal_log_weights=[component_log_weight(item) for item in goals],
        goal_construction='each current raw marker|body conditional retained under the common projected body marginal',
        original_goal_marginal_unchanged_claimed=False,
        conditional_transformations=conditional_records,
        version=state.version, stamp=state.stamp, command=list(state.command),
        sampling_used=False, numerical_differentiation_used=False,
        component_pruning_used=False, exact_group_bingham_moment_matching_claimed=False,
        single_component_identity=(len(raw) == 1),
    )
    return Projection(state, posterior, goals, record)


def retain_goal_components(prior_goal_components, predicted_internal_prior,
                           posterior_rawGaussian, effective_internal_state,
                           *, marker_edge_id='marker'):
    """With no new image, retain all old modes under the newly updated body."""
    components = tuple(prior_goal_components)
    if not components:
        raise ValueError('Retaining a goal mixture requires its existing hypotheses.')
    _check_weights(components)
    body, environment = _partition(predicted_internal_prior, marker_edge_id)
    if (posterior_rawGaussian.mean.shape != predicted_internal_prior.mean.shape
            or posterior_rawGaussian.covariance.shape != predicted_internal_prior.covariance.shape):
        raise ValueError('The updated body Gaussian must use the frozen internal prior chart.')
    result = []
    for component in components:
        source = component.state
        if source.component_id != component.label:
            raise ValueError('A retained goal label must match its actual state component ID.')
        _require_common_body(predicted_internal_prior, source, marker_edge_id)
        if (source.stamp != predicted_internal_prior.stamp
                or source.command != predicted_internal_prior.command):
            raise ValueError('Retained goal candidates must share the predicted command and stamp.')
        unchanged = (np.all(posterior_rawGaussian.mean[body] == 0.)
            and np.array_equal(posterior_rawGaussian.covariance[np.ix_(body, body)],
                               predicted_internal_prior.covariance[np.ix_(body, body)]))
        if unchanged:
            goal = (source if (source.version == effective_internal_state.version
                and source.stamp == effective_internal_state.stamp
                and source.command == effective_internal_state.command)
                else replace(source, version=effective_internal_state.version,
                    stamp=effective_internal_state.stamp, command=effective_internal_state.command))
        else:
            targeted, _, _ = _conditional_retarget(
                Gaussian(source.mean, source.covariance), posterior_rawGaussian, body, environment)
            goal = reexpress_state(source, targeted.mean, targeted.covariance,
                version=effective_internal_state.version, component_id=source.component_id)
            if goal.stamp != effective_internal_state.stamp or goal.command != effective_internal_state.command:
                raise ValueError('Retained goal and internal posterior source metadata differs.')
        _require_common_body(effective_internal_state, goal, marker_edge_id)
        result.append(component if goal is source else replace(component, state=goal))
    return tuple(result)
