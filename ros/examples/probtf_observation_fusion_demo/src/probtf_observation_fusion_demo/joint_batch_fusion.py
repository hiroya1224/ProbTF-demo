"""Exact affine Gaussian-mixture products in one frozen batch chart.

For a normalized common Gaussian prior p0, the unnormalized joint law is
``p0(x) * product_i sum_k exp(a_ik) N(y_ik; H_ik x+b_ik, R_ik)``.
Every native combination is retained, including finite log masses whose
floating-point weights underflow. Only the final ``gaussian`` property uses
analytic mixture moments. There is no intermediate reexpression/projection,
Laplace approximation, pruning, Monte Carlo, or change/reset policy here.

Adding each raw likelihood's information once is algebraically the same
common-prior correction as ``gaussian_reference_product``: multiplying
individual posteriors and dividing by p0 to the power (factor_count-1).
The usual positive-noise path uses one common-support solve per complete
branch. Singular observation noise uses the existing core conditioning
operation without changing its support or inventing a variance floor.
Mixture enumeration is intentionally exact and can be exponential.
"""

from dataclasses import dataclass
from itertools import product
import hashlib
import json
import math

import numpy as np
from scipy.linalg import cho_solve
from scipy.special import logsumexp

from probtf.spherical_law.fusion import Gaussian, PosteriorComponent, condition_gaussian
from probtf.spherical_law.support import covariance_support


def add_log_evidence_parts(parts,values):
    """Keep a large log scale and its finite residual instead of erasing it.

    TwoSum records the rounding residual of each addition. The pair is a
    numerical representation of one log mass, never two probability modes.
    """
    high,low = parts
    for value in values:
        if high == -math.inf or float(value) == -math.inf:
            return -math.inf,0.
        total = high+float(value)
        virtual = total-high
        error = (high-(total-virtual))+(float(value)-virtual)
        residual = low+error
        high = total+residual
        virtual = high-total
        low = (total-(high-virtual))+(residual-virtual)
    return float(high),float(low)


def log_evidence_parts_difference(parts,reference):
    """Subtract common large scales before summing finite mixture masses."""
    return math.fsum((parts[0],-reference[0],parts[1],-reference[1]))


def _normalize_log_evidence_parts(parts):
    reference = max(parts,key=lambda item:(item[0],item[1]))
    if not all(math.isfinite(value) for value in reference):
        raise ValueError('Gaussian mixture has no finite positive normalizing mass.')
    shifted = np.array([log_evidence_parts_difference(item,reference) for item in parts])
    normalizer = float(logsumexp(shifted))
    total_parts = add_log_evidence_parts(reference,(normalizer,))
    if not all(math.isfinite(value) for value in total_parts):
        raise ValueError('Gaussian mixture has no finite positive normalizing mass.')
    return shifted-normalizer,total_parts


def _condition_with_evidence_parts(prior,component):
    """Use the unchanged core posterior, retaining its density-scale terms."""
    observed = component.observed-component.offset
    posterior,_ = condition_gaussian(prior,observed,component.covariance,component.affine)
    innovation = component.affine @ prior.covariance @ component.affine.T+component.covariance
    support = covariance_support(innovation,'joint-batch innovation covariance')
    residual = observed-component.affine @ prior.mean
    terms = (-.5*(support.rank*math.log(2.*math.pi)+support.log_pdet),
             -.5*float(residual @ support.inverse @ residual))
    parts = add_log_evidence_parts((0.,0.),terms)
    return posterior,parts


@dataclass(frozen=True)
class JointBatchBranch:
    label: str
    distribution: Gaussian
    weight: float
    log_weight: float
    log_evidence: float
    component_labels: tuple
    source_ids: tuple
    log_evidence_parts: tuple = ()


@dataclass(frozen=True)
class JointBatchFusionResult:
    prior: Gaussian
    factors: tuple
    branches: tuple
    gaussian: Gaussian
    log_evidence: float
    record: dict
    log_evidence_parts: tuple = ()

    @property
    def posterior_components(self):
        """Core components normalized within this one Keep/Reset parent."""
        return tuple(PosteriorComponent(branch.label, branch.distribution,
            branch.weight, lineage=branch.source_ids, log_weight=branch.log_weight)
            for branch in self.branches)


@dataclass(frozen=True)
class JointBatchPrefixComponent:
    """One exact Gaussian child, linked to its immediate prior only."""
    label: str
    distribution: Gaussian
    weight: float
    log_weight: float
    parent_label: object = None
    current_component_label: object = None
    conditional_log_evidence: float = 0.
    source_ids: tuple = ()
    conditional_log_evidence_parts: tuple = ()

    @property
    def parent_branch_id(self):
        return self.parent_label

    @property
    def lineage(self):
        return self.source_ids


@dataclass(frozen=True)
class JointBatchPrefix:
    """Before/after laws of one observation, without projecting either law."""
    factor: object
    prior_components: tuple
    posterior_components: tuple
    log_evidence: float
    cumulative_log_evidence: float
    prefix_id: str
    previous_prefix_id: str
    record: dict
    log_evidence_parts: tuple = ()
    cumulative_log_evidence_parts: tuple = ()

    @property
    def factor_id(self):
        return self.factor.factor_id

    @property
    def observation_ids(self):
        return self.factor.observation_ids

    @property
    def prefix_total_log_evidence(self):
        return self.cumulative_log_evidence


@dataclass(frozen=True)
class JointBatchPrefixFusionResult:
    fusion: JointBatchFusionResult
    prefixes: tuple
    record: dict


def _validate_factors(prior, factors):
    factor_ids, observation_ids, chart = set(), set(), None
    for factor in factors:
        if not factor.factor_id or factor.factor_id in factor_ids:
            raise ValueError('Joint-batch factor IDs must be explicit and unique.')
        factor_ids.add(factor.factor_id)
        ids = tuple(factor.observation_ids)
        if len(set(ids)) != len(ids) or observation_ids.intersection(ids):
            raise ValueError('A joint batch cannot count one observation ID twice.')
        observation_ids.update(ids)
        current_chart = (factor.common_state_id, float(factor.common_stamp),
                         tuple(factor.common_command))
        if not math.isfinite(current_chart[1]) or not all(
                math.isfinite(float(value)) for value in current_chart[2]):
            raise ValueError('Joint-batch common chart metadata must be finite.')
        if chart is None:
            chart = current_chart
        elif chart != current_chart:
            raise ValueError('Joint-batch factors must share one frozen common chart.')
        labels = set()
        for component in factor.components:
            h = np.asarray(component.affine)
            dimension = np.asarray(component.observed).size
            if (not component.label or component.label in labels
                    or h.shape != (dimension, prior.mean.size)
                    or np.asarray(component.offset).shape != (dimension,)
                    or np.asarray(component.observed).shape != (dimension,)
                    or np.asarray(component.covariance).shape != (dimension, dimension)):
                raise ValueError('Joint-batch likelihood dimensions or native labels disagree.')
            labels.add(component.label)
            if not all(np.all(np.isfinite(value)) for value in
                    (h, component.offset, component.observed, component.covariance)):
                raise ValueError('Joint-batch likelihood parameters must be finite.')
            log_weight = float(component.log_weight)
            if math.isnan(log_weight) or log_weight == math.inf:
                raise ValueError('Native likelihood log mass must be finite or -inf.')
            Gaussian(np.zeros(dimension), component.covariance)
    return chart


def _information_component(prior, root, component):
    """Whiten observation noise, retaining its exact density normalizer."""
    cholesky = np.linalg.cholesky(component.covariance)
    projected = np.linalg.solve(cholesky, component.affine @ root)
    residual = np.linalg.solve(cholesky,
        component.observed-component.offset-component.affine @ prior.mean)
    log_normalizer = (-.5*(residual.size*math.log(2.*math.pi)
             +2.*np.log(np.diag(cholesky)).sum()))
    return (projected.T @ projected,projected.T @ residual,
            float(log_normalizer),projected,residual)


def _information_product(prior, root, terms):
    rank = root.shape[1]
    precision, eta, log_normalizer = np.eye(rank), np.zeros(rank), 0.
    for information, score, normalizer, _, _ in terms:
        precision += information
        eta += score
        log_normalizer += normalizer
    if rank == 0:
        quadratic = sum(float(residual @ residual) for _,_,_,_,residual in terms)
        parts = add_log_evidence_parts((0.,0.),(log_normalizer,-.5*quadratic))
        return prior,parts
    cholesky = np.linalg.cholesky(.5*(precision+precision.T))
    delta = cho_solve((cholesky, True), eta)
    covariance = cho_solve((cholesky, True), np.eye(rank))
    # Completing the square as gamma-eta.T P^-1 eta subtracts arbitrarily
    # large near-equal values. It can also erase log_normalizer if that small
    # term is added before the subtraction. Evaluate the equal minimized
    # quadratic as a sum of nonnegative original residuals and the prior
    # whitened mean penalty instead; no change to the normalized density.
    quadratic = float(delta @ delta)
    for _, _, _, projected, residual in terms:
        remaining = residual-projected @ delta
        quadratic += float(remaining @ remaining)
    parts = add_log_evidence_parts((0.,0.),(log_normalizer,-.5*quadratic,
                -float(np.log(np.diag(cholesky)).sum())))
    if not all(np.all(np.isfinite(value)) for value in (delta, covariance)):
        raise ValueError('Joint-batch necessary information solve returned nonfinite values.')
    # A fully deterministic or unobserved prior needs no numerical rebuild.
    if all(not np.any(term[0]) for term in terms):
        distribution = prior
    else:
        full_covariance = root @ covariance @ root.T
        distribution = Gaussian(prior.mean+root @ delta,
                                 .5*(full_covariance+full_covariance.T))
    return distribution,parts


def _support_product(prior, components):
    distribution, parts = prior,(0.,0.)
    for component in components:
        distribution,evidence = _condition_with_evidence_parts(distribution,component)
        parts = add_log_evidence_parts(parts,evidence)
    return distribution,parts


def _mixture_moments(branches):
    if len(branches) == 1:
        return branches[0].distribution
    mean = sum(branch.weight*branch.distribution.mean for branch in branches)
    covariance = np.zeros_like(branches[0].distribution.covariance)
    for branch in branches:
        delta = branch.distribution.mean-mean
        covariance += branch.weight*(branch.distribution.covariance+np.outer(delta, delta))
    return Gaussian(mean, .5*(covariance+covariance.T))


def fuse_gaussian_likelihood_factors(prior, factors, *, source_ids=()):
    """Return every exact common-chart branch and one final moment Gaussian.

    Empty/ineffective factors contribute the identity and retain provenance.
    Parents' prior masses are deliberately excluded: the caller combines each
    Keep/Reset parent's ``log_evidence`` with its own prescribed prior mass.
    """
    if not isinstance(prior, Gaussian):
        raise TypeError('Joint-batch fusion requires its actual normalized Gaussian prior.')
    factors = tuple(factors)
    chart = _validate_factors(prior, factors)
    effective = tuple(factor for factor in factors if factor.components)
    support = prior.support
    root = support.basis*np.sqrt(support.eigenvalues)
    component_terms, singular_noise = {}, False
    for factor in effective:
        for index, component in enumerate(factor.components):
            try:
                component_terms[(factor.factor_id, index)] = _information_component(
                    prior, root, component)
            except np.linalg.LinAlgError:
                # PSD zero-noise constraints are an existing core domain, not
                # a reason to add jitter or discard an observation.
                singular_noise = True
    combinations = product(*(tuple(enumerate(factor.components)) for factor in effective))
    raw = []
    for number, selected in enumerate(combinations):
        components = tuple(component for _, component in selected)
        if not effective:
            distribution,parts = prior,(0.,0.)
        elif singular_noise:
            distribution,parts = _support_product(prior, components)
        else:
            distribution,parts = _information_product(prior, root,
                tuple(component_terms[(factor.factor_id, index)]
                      for factor, (index, _) in zip(effective, selected)))
        parts = add_log_evidence_parts(parts,(float(component.log_weight) for component in components))
        log_mass = math.fsum(parts)
        if math.isnan(log_mass) or log_mass == math.inf:
            raise ValueError('Joint-batch Gaussian product returned nonfinite evidence.')
        labels = tuple((factor.factor_id, component.label)
                       for factor, component in zip(effective, components))
        lineage = tuple(source_ids)+tuple(identifier for component in components
                                         for identifier in component.source_ids)
        raw.append(('joint-batch:{}'.format(number),distribution,log_mass,labels,lineage,parts))
    normalized_logs,evidence_parts = _normalize_log_evidence_parts([item[5] for item in raw])
    log_evidence = math.fsum(evidence_parts)
    branches = tuple(JointBatchBranch(label, distribution,
        math.exp(log_weight),float(log_weight),log_mass,labels,lineage,parts)
        for (label,distribution,log_mass,labels,lineage,parts),log_weight in zip(raw,normalized_logs))
    gaussian = _mixture_moments(branches)
    record = dict(method='exact common-chart affine Gaussian-mixture product',
        body_approximation='analytic full-mixture moments once at batch end',
        common_state_id=None if chart is None else chart[0],
        common_stamp=None if chart is None else chart[1],
        common_command=None if chart is None else list(chart[2]),
        prior_support_rank=support.rank, factor_ids=[factor.factor_id for factor in factors],
        observation_ids=[identifier for factor in factors for identifier in factor.observation_ids],
        native_component_counts=[len(factor.components) for factor in factors],
        effective_factor_count=len(effective), exact_branch_count=len(branches),
        log_evidence=log_evidence,log_evidence_parts=list(evidence_parts),prior_parent_mass_included=False,
        log_mass_convention='normalized common prior times original normalized Gaussian likelihood densities and native log coefficients',
        product_algorithm=('existing core support conditioning without intermediate projection'
                           if singular_noise else 'common-support whitened information sum'),
        all_cross_covariances_retained=True, intermediate_projection_count=0,
        final_moment_projection_count=int(len(branches)>1),
        component_pruning_used=False, sampling_used=False, laplace_used=False,
        generic_mixture_complexity='product of native component counts; exponential fallback is not pruned',
        branches=[dict(label=branch.label, component_labels=[list(item) for item in branch.component_labels],
                       source_ids=list(branch.source_ids), log_evidence=branch.log_evidence,
                       log_evidence_parts=list(branch.log_evidence_parts),
                       weight=branch.weight, log_weight=branch.log_weight) for branch in branches])
    return JointBatchFusionResult(prior,factors,branches,gaussian,log_evidence,record,evidence_parts)


def fuse_gaussian_likelihood_factors_with_prefixes(prior, factors, *, source_ids=()):
    """Retain exact unprojected before/after laws for every effective factor.

    Each observation conditions each previous Gaussian branch once. With
    single-mode observations this is linear in the number of factors; native
    mixtures retain the unavoidable complete Cartesian product. Prefixes
    share their preceding immutable component tuples, and provenance uses
    immediate parent IDs and a hash chain, not copied history lists.

    ``fusion`` is independently the unchanged one-shot final body result.
    Sequential support conditioning has different floating-point operation
    order from that information sum. Its actual final differences are
    reported, rather than asserting bitwise numerical equivalence. Neither
    route applies an intermediate moment or Bingham approximation.
    """
    factors, source_ids = tuple(factors), tuple(source_ids)
    # Preserve the existing final body calculation exactly. This additional
    # whole-batch call is O(N), not one whole-history call per prefix.
    fusion = fuse_gaussian_likelihood_factors(prior, factors, source_ids=source_ids)
    seed = hashlib.sha256(b'ProbTF joint Gaussian prefix v1')
    seed.update(prior.mean.tobytes())
    seed.update(prior.covariance.tobytes())
    seed.update(json.dumps(source_ids, separators=(',', ':'), allow_nan=False).encode())
    initial_id = seed.hexdigest()
    previous_id = initial_id
    previous = (JointBatchPrefixComponent('prefix-initial:'+initial_id,prior,1.,0.),)
    prefixes,factor_links,cumulative_parts = [],[],(0.,0.)
    conditioning_calls = 0
    for factor in factors:
        if not factor.components:
            factor_links.append(dict(factor_id=factor.factor_id,effective=False,
                previous_prefix_id=previous_id,prefix_id=previous_id,
                observation_ids=list(factor.observation_ids),law_identity=True))
            continue
        prefix_id = hashlib.sha256(bytes.fromhex(previous_id)+factor.factor_id.encode()).hexdigest()
        raw = []
        for parent in previous:
            for index, component in enumerate(factor.components):
                posterior,overlap_parts = _condition_with_evidence_parts(parent.distribution,component)
                conditioning_calls += 1
                conditional_parts = add_log_evidence_parts(overlap_parts,(float(component.log_weight),))
                conditional_mass = math.fsum(conditional_parts)
                raw_parts = add_log_evidence_parts(conditional_parts,(parent.log_weight,))
                raw_mass = math.fsum(raw_parts)
                if math.isnan(raw_mass) or raw_mass == math.inf:
                    raise ValueError('Joint-batch prefix conditioning returned nonfinite evidence.')
                identifier = hashlib.sha256((prefix_id+parent.label+str(index)).encode()).hexdigest()
                raw.append(('prefix-child:'+identifier,posterior,raw_mass,parent.label,
                            component.label,conditional_mass,tuple(component.source_ids),
                            raw_parts,conditional_parts))
        normalized_logs,normalizer_parts = _normalize_log_evidence_parts([item[7] for item in raw])
        normalizer = math.fsum(normalizer_parts)
        posterior = tuple(JointBatchPrefixComponent(label,distribution,
            math.exp(log_weight),float(log_weight),parent_label,native_label,
            conditional_mass,component_sources,conditional_parts)
            for (label,distribution,log_mass,parent_label,native_label,conditional_mass,
                 component_sources,_,conditional_parts),log_weight in zip(raw,normalized_logs))
        cumulative_parts = add_log_evidence_parts(cumulative_parts,normalizer_parts)
        cumulative = math.fsum(cumulative_parts)
        if not math.isfinite(cumulative):
            raise ValueError('Joint-batch cumulative prefix evidence became nonfinite.')
        record = dict(prefix_id=prefix_id,previous_prefix_id=previous_id,
            factor_id=factor.factor_id,observation_ids=list(factor.observation_ids),
            prior_component_count=len(previous),posterior_component_count=len(posterior),
            native_component_count=len(factor.components),
            conditional_log_evidence=normalizer,conditional_log_evidence_parts=list(normalizer_parts),
            cumulative_log_evidence=cumulative,cumulative_log_evidence_parts=list(cumulative_parts),
            component_links=[dict(label=item.label,parent_branch_id=item.parent_branch_id,
                native_component_label=item.current_component_label,source_ids=list(item.source_ids),
                weight=item.weight,log_weight=item.log_weight,
                conditional_log_evidence=item.conditional_log_evidence,
                conditional_log_evidence_parts=list(item.conditional_log_evidence_parts)) for item in posterior],
            common_state_id=factor.common_state_id,common_stamp=factor.common_stamp,
            gaussian_parameter_storage='immutable in-memory prefix components; not repeated in this compact journal',
            intermediate_projection_used=False,component_pruning_used=False)
        prefixes.append(JointBatchPrefix(factor,previous,posterior,normalizer,cumulative,
            prefix_id,previous_id,record,normalizer_parts,cumulative_parts))
        factor_links.append(dict(factor_id=factor.factor_id,effective=True,
            previous_prefix_id=previous_id,prefix_id=prefix_id,
            observation_ids=list(factor.observation_ids),law_identity=False))
        previous, previous_id = posterior, prefix_id
    # Itertools' final product and the prefix expansion both preserve native
    # component order; no nearest-mode matching or branch selection is used.
    if len(previous) != len(fusion.branches):
        raise ValueError('Exact prefix and one-shot native branch enumerations disagree.')
    differences = dict(
        comparison='different floating-point operation order, not a posterior approximation',
        branch_count=len(previous),
        mean_max_abs=max(float(np.max(np.abs(a.distribution.mean-b.distribution.mean),initial=0.))
                         for a,b in zip(previous,fusion.branches)),
        covariance_max_abs=max(float(np.max(np.abs(a.distribution.covariance-b.distribution.covariance),initial=0.))
                               for a,b in zip(previous,fusion.branches)),
        normalized_weight_max_abs=max(abs(a.weight-b.weight) for a,b in zip(previous,fusion.branches)),
        cumulative_log_evidence_difference=log_evidence_parts_difference(cumulative_parts,fusion.log_evidence_parts),
        affects_final_body_fusion=False,numerical_difference_is_update_gate=False)
    record = dict(method='exact unprojected Gaussian likelihood prefixes with immediate-parent provenance',
        initial_prefix_id=initial_id,final_prefix_id=previous_id,
        batch_source_ids=list(source_ids),all_raw_source_ids_stored_once=True,
        effective_prefix_count=len(prefixes),factor_links=factor_links,
        support_conditioning_call_count=conditioning_calls,
        provenance_convention='sha256 chain: initial Gaussian parameter bytes and batch source IDs, then previous prefix digest and current immutable factor ID; child digest also includes immediate parent label and native index',
        cumulative_log_evidence=math.fsum(cumulative_parts),
        cumulative_log_evidence_parts=list(cumulative_parts),
        final_one_shot_comparison=differences,
        prefix_approximation_count=0,native_component_pruning_used=False,
        sampling_used=False,laplace_used=False,
        complexity='one support-conditioning operation per new native child; linear for single-mode observations, exact Cartesian-product growth for mixtures')
    return JointBatchPrefixFusionResult(fusion,tuple(prefixes),record)
