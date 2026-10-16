"""Frozen-proxy distribution lifting and reference-prior corrected fusion.

Sensor adapters supply *estimated distributions*, not raw measurements, to
``fuse_sensor_posteriors``.  All ratios are evaluated in one frozen affine
support.  Mixture labels refer to an explicit common-prior component; arbitrary
Gaussian-mixture density ratios are deliberately not approximated here.
"""

from dataclasses import dataclass
from itertools import product
import math

import numpy as np

from probtf.spherical_law.support import (
    covariance_support, readonly, require_in_support, require_same_support,
)


def _checked_log_weight(weight, log_weight, *, positive=False):
    """Keep small positive mixture masses without turning underflow into pruning.

    ``weight`` remains the float used by old callers and ROS colors. The log
    mass is authoritative when that float is below the representable range.
    An explicitly zero mass still has log mass -inf and carries no evidence.
    """
    weight = float(weight)
    if not math.isfinite(weight) or weight < 0.0:
        raise ValueError("Component weight must be finite and non-negative.")
    if log_weight is None:
        value = math.log(weight) if weight > 0.0 else -math.inf
    else:
        value = float(log_weight)
        if math.isnan(value) or value == math.inf:
            raise ValueError("Component log weight must be finite or -inf.")
        represented = math.exp(value)
        if not math.isclose(represented, weight, rel_tol=1.0e-12, abs_tol=0.0):
            raise ValueError("Component weight and log weight describe different masses.")
    if positive and value == -math.inf:
        raise ValueError("A labelled prior needs a positive weight and label.")
    return weight, value


def _logsumexp(values):
    values = tuple(values)
    if not values or max(values) == -math.inf:
        return -math.inf
    maximum = max(values)
    return maximum + math.log(sum(math.exp(value-maximum) for value in values))


@dataclass(frozen=True)
class Gaussian:
    mean: np.ndarray
    covariance: np.ndarray

    def __post_init__(self):
        mean = readonly(self.mean)
        covariance = readonly(self.covariance)
        if mean.ndim != 1 or covariance.shape != (mean.size, mean.size):
            raise ValueError("Gaussian mean/covariance dimensions do not match.")
        covariance_support(covariance)
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "covariance", covariance)

    @property
    def support(self):
        return covariance_support(self.covariance)


@dataclass(frozen=True)
class FrozenGaussianQuery:
    body_prior: Gaussian
    distribution: Gaussian
    jacobian: np.ndarray
    query_id: str
    prior_version: int
    stamp: float
    command: tuple
    component_id: str
    frame: str = ""
    within: object = None
    between: object = None
    approximation_order: int = 1
    state_id: str = ""

    def __post_init__(self):
        jacobian = readonly(self.jacobian)
        if jacobian.shape != (self.distribution.mean.size, self.body_prior.mean.size):
            raise ValueError("Frozen query sensitivity dimensions do not match.")
        direct = jacobian @ self.body_prior.covariance @ jacobian.T
        if not np.allclose(direct, self.distribution.covariance,
                           rtol=1.0e-9, atol=1.0e-12):
            raise ValueError("Frozen query covariance does not equal H P H^T.")
        if (self.within is None) != (self.between is None):
            raise ValueError("Both total-covariance terms must be supplied together.")
        if self.within is not None:
            within, between = readonly(self.within), readonly(self.between)
            covariance_support(within, "within covariance")
            covariance_support(between, "between covariance")
            if not np.allclose(within + between, direct, rtol=1.0e-9, atol=1.0e-12):
                raise ValueError("Frozen query total-covariance terms are inconsistent.")
            object.__setattr__(self, "within", within)
            object.__setattr__(self, "between", between)
        if not self.query_id or not self.component_id:
            raise ValueError("Frozen query and common component IDs must be explicit.")
        if self.approximation_order != 1:
            raise ValueError("This fusion contract requires the global first-order proxy.")
        if not math.isfinite(float(self.stamp)):
            raise ValueError("Frozen observation time must be finite.")
        if not all(math.isfinite(float(value)) for value in self.command):
            raise ValueError("Frozen commands must be finite.")
        object.__setattr__(self, "jacobian", jacobian)
        object.__setattr__(self, "command", tuple(float(x) for x in self.command))

    @property
    def intercept(self):
        return readonly(self.distribution.mean - self.jacobian @ self.body_prior.mean)

    @property
    def cross_covariance(self):
        return readonly(self.body_prior.covariance @ self.jacobian.T)


def freeze_query(state, query, query_id=None):
    """Freeze a core point/transform query without changing its proxy mean."""
    if getattr(query, "version", state.version) != state.version:
        raise ValueError("Query and body state versions differ.")
    if (getattr(query, "stamp", state.stamp) != state.stamp
            or tuple(getattr(query, "command", state.command)) != tuple(state.command)
            or getattr(query, "component_id", state.component_id) != state.component_id):
        raise ValueError("Query time, command, or component differs from its body state.")
    if getattr(query, "approximation_order", 1) != 1:
        raise ValueError("Query approximation order differs from the frozen first-order proxy.")
    if getattr(query, "state_id", None) != getattr(state, "state_id", None):
        raise ValueError("Query and body state parameter identities differ.")
    return FrozenGaussianQuery(
        Gaussian(state.mean, state.covariance),
        Gaussian(query.mean, query.covariance), query.jacobian,
        query_id or query.query_id, state.version, state.stamp, tuple(state.command),
        state.component_id, getattr(query, "frame", ""),
        getattr(query, "within", None), getattr(query, "between", None),
        state_id=getattr(state, "state_id", ""),
    )


@dataclass(frozen=True)
class PosteriorComponent:
    label: str
    distribution: Gaussian
    weight: float = 1.0
    lineage: tuple = ()
    log_weight: object = None

    def __post_init__(self):
        if not self.label or not isinstance(self.distribution, Gaussian):
            raise ValueError("A posterior component needs a label and Gaussian.")
        weight, log_weight = _checked_log_weight(self.weight, self.log_weight)
        object.__setattr__(self, "weight", weight)
        object.__setattr__(self, "log_weight", log_weight)
        object.__setattr__(self, "lineage", tuple(self.lineage))


@dataclass(frozen=True)
class SensorPosterior:
    query: FrozenGaussianQuery
    sensor_id: str
    observation_id: str
    components: tuple

    def __post_init__(self):
        components = tuple(self.components)
        if not self.sensor_id or not self.observation_id or not components:
            raise ValueError("A sensor posterior requires IDs and estimated components.")
        if len({component.label for component in components}) != len(components):
            raise ValueError("Sensor component labels must be unique.")
        support = self.query.distribution.support
        for component in components:
            require_same_support(support, component.distribution.support,
                                 "sensor posterior {}".format(component.label))
            require_in_support(component.distribution.mean - self.query.distribution.mean,
                               support, "sensor posterior mean shift")
        object.__setattr__(self, "components", components)


@dataclass(frozen=True)
class LiftedComponent:
    label: str
    weight: float
    distribution: Gaussian
    within: np.ndarray
    between: np.ndarray
    sensor_id: str
    observation_id: str
    query_id: str
    prior_component_id: str
    log_weight: object = None


@dataclass(frozen=True)
class LabelledPrior:
    label: str
    distribution: Gaussian
    weight: float = 1.0
    log_weight: object = None

    def __post_init__(self):
        if not self.label:
            raise ValueError("A labelled prior needs a positive weight and label.")
        weight, log_weight = _checked_log_weight(self.weight, self.log_weight, positive=True)
        object.__setattr__(self, "weight", weight)
        object.__setattr__(self, "log_weight", log_weight)


@dataclass(frozen=True)
class FusedMixture:
    components: tuple
    individuals: tuple
    observation_ids: tuple
    prior_version: object
    stamp: object
    command: tuple


def condition_gaussian(prior, observed_mean, observed_covariance,
                       observation_matrix=None):
    """Sensor-local Gaussian observation product, including its evidence.

    This is available to adapters; the body fusion routine below never accepts
    these raw observations. Singular innovations are solved on their support.
    """
    observed_mean = np.asarray(observed_mean, dtype=float)
    noise = Gaussian(np.zeros_like(observed_mean), observed_covariance)
    h = (np.eye(prior.mean.size) if observation_matrix is None
         else np.asarray(observation_matrix, dtype=float))
    if h.shape != (observed_mean.size, prior.mean.size):
        raise ValueError("Observation sensitivity dimensions do not match.")
    innovation = h @ prior.covariance @ h.T + noise.covariance
    support = covariance_support(innovation, "innovation covariance")
    residual = observed_mean - h @ prior.mean
    require_in_support(residual, support, "observation residual")
    gain = prior.covariance @ h.T @ support.inverse
    mean = prior.mean + gain @ residual
    remaining = np.eye(prior.mean.size) - gain @ h
    # Joseph form is a sum of actual covariance terms, without regularization.
    covariance = (remaining @ prior.covariance @ remaining.T
                  + gain @ noise.covariance @ gain.T)
    covariance = 0.5 * (covariance + covariance.T)
    log_evidence = -0.5 * (support.rank * math.log(2.0 * math.pi)
                           + support.log_pdet
                           + residual @ support.inverse @ residual)
    return Gaussian(mean, covariance), float(log_evidence)


def lift_sensor_posterior(posterior):
    """Evaluate both terms of E[Cov(z|z_s)] + Cov(E[z|z_s])."""
    query = posterior.query
    prior = query.body_prior
    local_prior = query.distribution
    gain = query.cross_covariance @ local_prior.support.inverse
    within = prior.covariance - gain @ local_prior.covariance @ gain.T
    within = 0.5 * (within + within.T)
    covariance_support(within, "conditional body covariance")
    results = []
    for component in posterior.components:
        if np.array_equal(component.distribution.mean, local_prior.mean) and np.array_equal(
                component.distribution.covariance, local_prior.covariance):
            # Preserve the exact identity, including bits unaffected by algebra.
            distribution = prior
            between = gain @ local_prior.covariance @ gain.T
        else:
            mean = prior.mean + gain @ (component.distribution.mean - local_prior.mean)
            between = gain @ component.distribution.covariance @ gain.T
            covariance = within + between
            covariance = 0.5 * (covariance + covariance.T)
            distribution = Gaussian(mean, covariance)
        # Verify the required marginal equality in this same frozen proxy.
        projected_mean = query.intercept + query.jacobian @ distribution.mean
        projected_covariance = query.jacobian @ distribution.covariance @ query.jacobian.T
        if (not np.allclose(projected_mean, component.distribution.mean,
                            rtol=1.0e-8, atol=1.0e-10)
                or not np.allclose(projected_covariance, component.distribution.covariance,
                                   rtol=1.0e-8, atol=1.0e-11)):
            raise ValueError("Lifted target marginal differs from sensor posterior.")
        results.append(LiftedComponent(
            component.label, component.weight, distribution, readonly(within),
            readonly(0.5 * (between + between.T)), posterior.sensor_id,
            posterior.observation_id, query.query_id, query.component_id,
            component.log_weight))
    return tuple(results)


def gaussian_reference_product(reference, distributions):
    """Normalize prod(N_s)/N_0**(m-1), returning Gaussian and log C.

    The integrals are evaluated on the reference affine support. The common
    reference mean is used as coordinate origin, so fixed coordinates never
    acquire numerical information or variance.
    """
    distributions = tuple(distributions)
    if not distributions:
        return reference, 0.0
    support = reference.support
    basis = support.basis
    for distribution in distributions:
        require_same_support(support, distribution.support, "body posterior")
        require_in_support(distribution.mean - reference.mean, support,
                           "body posterior mean shift")
    if len(distributions) == 1:
        return distributions[0], 0.0
    if support.rank == 0:
        return reference, 0.0
    p0 = basis.T @ reference.covariance @ basis
    precision0 = np.linalg.inv(p0)
    precision = -(len(distributions) - 1) * precision0
    eta = np.zeros(support.rank)
    gamma = 0.0
    sum_logdet = 0.0
    for distribution in distributions:
        covariance = basis.T @ distribution.covariance @ basis
        mean = basis.T @ (distribution.mean - reference.mean)
        information = np.linalg.inv(covariance)
        precision += information
        eta += information @ mean
        gamma += mean @ information @ mean
        sum_logdet += np.linalg.slogdet(covariance)[1]
    precision = 0.5 * (precision + precision.T)
    precision_support = covariance_support(precision, "fused information")
    if precision_support.rank != support.rank:
        raise ValueError("Fused Gaussian information is not positive definite on its support.")
    covariance = np.linalg.inv(precision)
    mean = covariance @ eta
    log_c = 0.5 * (
        (len(distributions) - 1) * np.linalg.slogdet(p0)[1]
        - sum_logdet + np.linalg.slogdet(covariance)[1] - gamma + eta @ mean)
    result_covariance = basis @ covariance @ basis.T
    result = Gaussian(reference.mean + basis @ mean,
                      0.5 * (result_covariance + result_covariance.T))
    return result, float(log_c)


def _same_gaussian(left, right):
    return (np.array_equal(left.mean, right.mean)
            and np.array_equal(left.covariance, right.covariance))


def _same_sensor_posterior(left, right):
    return (left.sensor_id == right.sensor_id
            and left.observation_id == right.observation_id
            and left.query.query_id == right.query.query_id
            and np.array_equal(left.query.jacobian, right.query.jacobian)
            and _same_gaussian(left.query.distribution, right.query.distribution)
            and len(left.components) == len(right.components)
            and all(a.label == b.label and a.weight == b.weight and a.log_weight == b.log_weight
                    and _same_gaussian(a.distribution, b.distribution)
                    for a, b in zip(left.components, right.components)))


def fuse_labelled_sensor_posteriors(priors, posteriors):
    """Fuse finite, explicitly corresponding common-prior hypotheses.

    For each observation the input component weights are the *global* sensor
    posterior masses alpha[s,h,k], normalized across all h,k. An adapter must
    retain this correspondence; this function never guesses mixture ratios.
    """
    priors = tuple(priors)
    posteriors = tuple(posteriors)
    if not priors or len({prior.label for prior in priors}) != len(priors):
        raise ValueError("Distinct common-prior component labels are required.")
    if not math.isclose(sum(prior.weight for prior in priors), 1.0,
                        rel_tol=1.0e-10, abs_tol=1.0e-12):
        raise ValueError("Common-prior component weights must sum to one.")
    if not math.isclose(_logsumexp(prior.log_weight for prior in priors), 0.0,
                        rel_tol=0.0, abs_tol=1.0e-10):
        raise ValueError("Common-prior component log weights must normalize.")
    prior_map = {prior.label: prior for prior in priors}
    metadata = None
    state_ids = {}
    observations = {}
    for posterior in posteriors:
        query = posterior.query
        if query.component_id not in prior_map:
            raise ValueError("Sensor posterior lost its common-prior component label.")
        if not _same_gaussian(query.body_prior, prior_map[query.component_id].distribution):
            raise ValueError("Sensor posterior uses a different common body prior.")
        current_metadata = (query.prior_version, query.stamp, tuple(query.command))
        if metadata is None:
            metadata = current_metadata
        elif metadata != current_metadata:
            raise ValueError("Sensors must share one frozen version, time, and command.")
        if query.component_id in state_ids and state_ids[query.component_id] != query.state_id:
            raise ValueError("Sensors use different parameter identities for one frozen prior component.")
        state_ids[query.component_id] = query.state_id
        group = observations.setdefault(posterior.observation_id, {})
        previous = group.get(query.component_id)
        if previous is not None:
            if not _same_sensor_posterior(previous, posterior):
                raise ValueError("A repeated observation ID has inconsistent contents.")
            continue  # A retransmission is not new information.
        if group and next(iter(group.values())).sensor_id != posterior.sensor_id:
            raise ValueError("A physical observation ID was reused by another sensor.")
        group[query.component_id] = posterior
    for group in observations.values():
        if set(group) != set(prior_map):
            raise ValueError("An observation is missing explicit prior-hypothesis correspondence.")
        total_weight = sum(component.weight for posterior in group.values()
                           for component in posterior.components)
        if not math.isclose(total_weight, 1.0, rel_tol=1.0e-9, abs_tol=1.0e-12):
            raise ValueError("Sensor posterior weights must normalize across common hypotheses.")
        if not math.isclose(_logsumexp(component.log_weight for posterior in group.values()
                                      for component in posterior.components), 0.0,
                            rel_tol=0.0, abs_tol=1.0e-9):
            raise ValueError("Sensor posterior log weights must normalize across common hypotheses.")
    if not observations:
        return FusedMixture(tuple(PosteriorComponent(
            prior.label, prior.distribution, prior.weight,
            log_weight=prior.log_weight) for prior in priors),
            (), (), None, None, ())
    ordered_observations = tuple(sorted(observations))
    individuals = []
    components = []
    log_weights = []
    for prior in priors:
        lifted_groups = []
        for observation_id in ordered_observations:
            lifted = lift_sensor_posterior(observations[observation_id][prior.label])
            individuals.extend(lifted)
            lifted_groups.append(lifted)
        for selected in product(*lifted_groups):
            if any(component.log_weight == -math.inf for component in selected):
                continue
            distribution, log_c = gaussian_reference_product(
                prior.distribution, (component.distribution for component in selected))
            log_weight = ((1 - len(selected)) * prior.log_weight
                          + sum(component.log_weight for component in selected)
                          + log_c)
            lineage = tuple((component.sensor_id, component.observation_id, component.label)
                            for component in selected)
            label = prior.label + "|" + "|".join(
                "{}:{}".format(component.observation_id, component.label)
                for component in selected)
            components.append((label, distribution, lineage))
            log_weights.append(log_weight)
    if not components:
        raise ValueError("No positive-mass corresponding posterior combinations remain.")
    log_normalizer = _logsumexp(log_weights)
    result_components = tuple(PosteriorComponent(
        label, distribution, math.exp(log_weight - log_normalizer), lineage,
        log_weight=log_weight-log_normalizer)
        for (label, distribution, lineage), log_weight in zip(components, log_weights))
    return FusedMixture(result_components, tuple(individuals), ordered_observations,
                        metadata[0], metadata[1], metadata[2])


def fuse_sensor_posteriors(body_prior, posteriors):
    """Convenience entry for the common single prior-component case."""
    posteriors = tuple(posteriors)
    component_id = posteriors[0].query.component_id if posteriors else "prior"
    return fuse_labelled_sensor_posteriors(
        (LabelledPrior(component_id, body_prior),), posteriors)
