"""One frozen-prior update shared by every observation list."""

from dataclasses import dataclass, replace
from itertools import product
from pathlib import Path
import math

import numpy as np
import yaml
from scipy.special import logsumexp

from probtf.spherical_law.first_order import (
    BinghamTransform, JointTransformState, MAX_FIRST_ORDER_ROTATION_VARIANCE,
)
from .body_native import point_query
from probtf.spherical_law.fusion import (
    Gaussian, LabelledPrior, PosteriorComponent, fuse_labelled_sensor_posteriors,
    gaussian_reference_product, lift_sensor_posterior,
)
from probtf.spherical_law.support import SUPPORT_ATOL, SUPPORT_RTOL, covariance_support

from probtf_observation_fusion_demo.scene import render_image
from probtf_observation_fusion_demo.sensors import ADAPTERS, PREPARERS
from .immutable_body_cache import reexpress_state


@dataclass(frozen=True)
class WeightedState:
    label: str
    weight: float
    state: JointTransformState
    log_weight: object = None


@dataclass(frozen=True)
class FrozenBodyBranch:
    """One labelled prior chart and its globally weighted raw posterior children.

    Different branches can have different charts. Their reference states are
    part of the density specification, not interchangeable mixture metadata.
    """

    label: str
    weight: float
    reference_state: JointTransformState
    prior: Gaussian
    posterior_components: tuple
    log_weight: object = None


@dataclass(frozen=True)
class Stage:
    label: str
    components: tuple
    sensor_ids: tuple = ()
    # Explicit current observation/goal mixture sharing the canonical body's
    # marginal. It is not the next cycle's retained internal state.
    goal_components: tuple = ()


@dataclass(frozen=True)
class DemoResult:
    prior: JointTransformState
    stages: tuple
    image: np.ndarray
    detection_image: np.ndarray
    link_paths: dict
    hand_path: tuple
    camera_path: tuple
    marker_path: tuple
    camera_marker_path: tuple
    root_frame: str
    sensor_posteriors: tuple
    local_estimates: tuple
    records: dict
    # Keep the actual common frozen chart for downstream likelihood-ratio
    # queries. Reexpressed per-link states do not share this original chart.
    frozen_body_prior: object = None
    frozen_body_posterior: object = None
    frozen_body_branches: tuple = ()
    # Optional independent numerator chart after a change-prior mixture has
    # been projected into the actual single internal law. Native branches
    # still specify the transition prior, not a tight continuation-only prior.
    projected_body_numerator: object = None
    # Historical density-ratio views retain each original acquisition q_i.
    # They are not installed as current body laws or sent to visualization.
    time_resolved_body_ratios: tuple = ()


def load_inputs(common_path, observations_path):
    with Path(common_path).open() as handle:
        common = yaml.safe_load(handle)
    with Path(observations_path).open() as handle:
        additional = yaml.safe_load(handle)
    if set(additional) != {"observations"}:
        raise ValueError("An entry may change only its observation list.")
    return common, tuple(additional["observations"])


def _candidate(data):
    return BinghamTransform(
        data["edge_id"], np.asarray(data["translation"], dtype=float),
        bingham=np.asarray(data["bingham"], dtype=float),
        translation_covariance=np.asarray(data["translation_covariance"], dtype=float),
    )


def initial_state(common):
    prior = common["prior"]
    if not prior["independence"].startswith("initial product"):
        raise ValueError("This input must explicitly declare its initial independence.")
    transforms = [_candidate(item) for item in prior["body"]]
    transforms.append(_candidate(prior["marker"]))
    for key in ("tool_mount", "camera_mount"):
        item = common["known_geometry"][key]
        transforms.append(BinghamTransform(item["edge_id"], item["translation"],
                                            rotation_wxyz=item["rotation_wxyz"]))
    update = common["update"]
    return JointTransformState.from_independent(
        transforms, version=int(update["prior_version"]), stamp=float(update["stamp"]),
        command=tuple(update["command"]), component_id="common_body_prior",
    )


def geometry_paths(common):
    body = tuple((item["edge_id"], False) for item in common["prior"]["body"])
    links = {item["edge_id"]: body[:index + 1]
             for index, item in enumerate(common["prior"]["body"])}
    camera = body + ((common["known_geometry"]["camera_mount"]["edge_id"], False),)
    marker = ((common["prior"]["marker"]["edge_id"], False),)
    camera_inverse = tuple((edge, not inverse) for edge, inverse in reversed(camera))
    return {"links": links, "camera": camera, "marker": marker,
            "hand": body + ((common["known_geometry"]["tool_mount"]["edge_id"], False),),
            "camera_marker": camera_inverse + marker}


def _gaussian_record(distribution):
    return {"mean": distribution.mean.tolist(), "covariance": distribution.covariance.tolist(),
            "support": _support_record(distribution.covariance)}


def _support_record(covariance):
    support = covariance_support(covariance)
    return {"dimension": covariance.shape[0], "rank": support.rank,
            "roundoff_tolerance": support.tolerance,
            "roundoff_eigenvalues": list(support.roundoff_eigenvalues),
            "retained_eigenvalues": support.eigenvalues.tolist(),
            "regularization_added": False}


def _joint_record(component, paths, frame):
    state = component.state
    diagonal = np.zeros_like(state.covariance)
    for index in range(len(state.transforms)):
        diagonal[6*index:6*index+6, 6*index:6*index+6] = state.covariance[6*index:6*index+6, 6*index:6*index+6]
    hand = point_query(state, paths["hand"], query_id="record:hand", frame=frame)
    return {"label": component.label, "weight": component.weight,
            "log_weight": component_log_weight(component),
            "version": state.version, "stamp": state.stamp, "command": list(state.command),
            "state_id": state.state_id,
            "component_id": state.component_id, "edge_ids": list(state.edge_ids),
            "transforms": [{"edge_id": edge.edge_id, "translation": edge.translation.tolist(),
                            "rotation_wxyz": edge.rotation_wxyz.tolist(),
                            "bingham": None if edge.bingham is None else edge.bingham.tolist()}
                           for edge in state.transforms],
            "joint_covariance": state.covariance.tolist(),
            "support": _support_record(state.covariance),
            "cross_block_frobenius_norm": float(np.linalg.norm(state.covariance - diagonal)),
            "hand_mean": hand.mean.tolist(), "hand_covariance": hand.covariance.tolist(),
            "hand_state_id": hand.state_id, "hand_support": _support_record(hand.covariance),
            "hand_within": hand.within.tolist(), "hand_between": hand.between.tolist()}


def component_log_weight(component):
    value = getattr(component, "log_weight", None)
    return float(value) if value is not None else (
        -math.inf if component.weight == 0. else math.log(component.weight))


def _posterior_parent(component, prior_components):
    matches = [parent for parent in prior_components
               if component.label == parent.label
               or component.label.startswith(parent.label + "|")]
    if len(matches) != 1:
        raise ValueError("A fused component lost its explicit labelled prior parent.")
    return matches[0]


def _global_sensor_estimates(estimates, priors):
    """Normalize p(h,branch|observation), including evidence across hypotheses."""
    terms = []
    for parent, estimate in zip(priors, estimates):
        if estimate.log_evidence is None:
            raise ValueError("A valid sensor posterior needs its common-prior evidence.")
        conditional = estimate.component_log_weights
        if len(conditional) != len(estimate.posterior.components):
            raise ValueError("Every sensor branch needs an explicit conditional log mass.")
        terms.extend(component_log_weight(parent) + float(estimate.log_evidence) + value
                     for value in conditional)
    normalizer = float(logsumexp(terms))
    if not np.isfinite(normalizer):
        raise ValueError("The observed likelihood has no finite positive common-prior mass.")
    results = []
    offset = 0
    for parent, estimate in zip(priors, estimates):
        count = len(estimate.posterior.components)
        logs = terms[offset:offset+count]
        offset += count
        components = tuple(PosteriorComponent(item.label, item.distribution,
            float(np.exp(value-normalizer)), item.lineage, log_weight=float(value-normalizer))
            for item, value in zip(estimate.posterior.components, logs))
        posterior = replace(estimate.posterior, components=components)
        diagnostics = dict(estimate.diagnostics,
            prior_component_id=parent.label, prior_component_weight=parent.weight,
            prior_component_log_weight=component_log_weight(parent),
            log_evidence=float(estimate.log_evidence),
            global_log_evidence=normalizer,
            global_posterior_weights=[item.weight for item in components],
            global_posterior_log_weights=[component_log_weight(item) for item in components])
        image_queries = tuple(replace(query, source=item)
                              for query, item in zip(estimate.image_queries, components))
        if image_queries:
            image_record = dict(diagnostics["image_xy_query"])
            image_record["components"] = [dict(original, weight=item.weight,
                log_weight=component_log_weight(item)) for original, item in zip(
                    image_record["components"], components)]
            diagnostics["image_xy_query"] = image_record
        results.append(replace(estimate, posterior=posterior, diagnostics=diagnostics,
                               image_queries=image_queries))
    return tuple(results)


def _joint_observation_evidence(priors, posteriors, estimates):
    """Normalized joint evidence with the same reference-prior fusion product.

    Sensor marginal evidences alone do not account for shared-body dependence.
    The product normalizer is evaluated in each parent's actual frozen chart.
    No state samples or independent-marginal approximation enter this query.
    """
    available = {(item.posterior.observation_id, item.posterior.query.component_id): item
                 for item in estimates if item.posterior is not None}
    observation_ids = tuple(sorted({item.observation_id for item in posteriors}))
    if not observation_ids:
        return None
    global_offset = sum(available[(identifier, priors[0].label)].diagnostics['global_log_evidence']
                        for identifier in observation_ids)
    evidence = []
    terms_all = []
    for parent in priors:
        groups = [lift_sensor_posterior(available[(identifier, parent.label)].posterior)
                  for identifier in observation_ids]
        terms = []
        for selected in product(*groups):
            _, correction = gaussian_reference_product(
                Gaussian(parent.state.mean, parent.state.covariance),
                (item.distribution for item in selected))
            terms.append((1-len(selected))*component_log_weight(parent)
                         + sum(component_log_weight(item) for item in selected) + correction)
        log_mass = float(logsumexp(terms))
        terms_all.extend(terms)
        evidence.append(dict(label=parent.label,
            log_evidence=float(global_offset+log_mass-component_log_weight(parent)),
            source_state_id=parent.state.state_id, prior_log_weight=component_log_weight(parent),
            combination_count=len(terms)))
    return dict(observation_ids=list(observation_ids), parents=evidence,
        log_evidence=float(global_offset+logsumexp(terms_all)),
        shared_body_reference_product_correction=True,
        measure='same observation-fixed IMU tangent plane and native image-Laplace likelihood with its chart-volume correction',
        approximation='normalized linearized Bingham-derived sensor Gaussian proxies; not the exact raw-pixel likelihood')


def run(common, observations, image=None, *, prior=None, paths=None, prior_components=None,
        joint_evidence=False, shared_observation_frame=False):
    """Update every retained prior hypothesis from the same new observations.

    Each labelled prior has its own frozen Bingham-derived chart. Sensor
    evidence is normalized globally over parent and observation branches;
    core reference-prior fusion is then performed within the explicit parent
    correspondence. No prior branch, cross covariance or marker memory is reset.
    """
    image = render_image(common) if image is None else np.asarray(image).copy()
    image.setflags(write=False)
    common = {key: common[key] for key in (
        "root_frame", "update", "prior", "known_geometry", "camera", "marker", "marker_board")
        if key in common}
    if prior_components is None:
        prior = initial_state(common) if prior is None else prior
        prior_components = (WeightedState(prior.component_id, 1., prior, 0.),)
    else:
        prior_components = tuple(prior_components)
        if not prior_components:
            raise ValueError("A sequential update requires the retained prior mixture.")
        prior = prior_components[0].state
    metadata = (prior.version, prior.stamp, prior.command, prior.edge_ids)
    for parent in prior_components:
        state = parent.state
        if (state.version, state.stamp, state.command, state.edge_ids) != metadata:
            raise ValueError("All prior hypotheses must share command, time, version and geometry.")
        if parent.label != state.component_id:
            raise ValueError("A prior state and its explicit hypothesis label differ.")
    if not np.isclose(logsumexp([component_log_weight(item) for item in prior_components]), 0., atol=1e-12):
        raise ValueError("Retained prior log masses must be normalized.")
    paths = geometry_paths(common) if paths is None else paths
    stages = [Stage("prior", prior_components)]
    posteriors, local_estimates = [], []
    seen = {}
    detection_image = image
    parent_map = {item.label: item for item in prior_components}
    for observation in observations:
        identifier = observation["observation_id"]
        if identifier in seen:
            if observation != seen[identifier]:
                raise ValueError("A repeated observation ID has different input data.")
            continue
        seen[identifier] = observation
        prepare = PREPARERS.get(observation["sensor"])
        extra = ({} if prepare is None else
                 {"prepared": prepare(common, image, prior.stamp)})
        if shared_observation_frame and observation['sensor'] == 'imu':
            extra['shared_observation_frame'] = True
        estimates = tuple(ADAPTERS[observation["sensor"]](
            parent.state, observation, paths, common, image, **extra) for parent in prior_components)
        available = [estimate.posterior is not None for estimate in estimates]
        if any(available) and not all(available):
            raise ValueError("Observation availability must be common to every prior hypothesis.")
        if not any(available):
            local_estimates.extend(estimates)
            if estimates[0].debug_image is not None:
                detection_image = estimates[0].debug_image
            continue
        estimates = _global_sensor_estimates(estimates, prior_components)
        local_estimates.extend(estimates)
        states = []
        for estimate in estimates:
            if estimate.debug_image is not None:
                detection_image = estimate.debug_image
            posterior = estimate.posterior
            posteriors.append(posterior)
            parent = parent_map[posterior.query.component_id]
            for item in lift_sensor_posterior(posterior):
                label = item.label if len(prior_components) == 1 else parent.label + "|" + item.label
                state = reexpress_state(parent.state, item.distribution.mean,
                    item.distribution.covariance, version=prior.version+1,
                    component_id=posterior.sensor_id + ":" + label)
                states.append(WeightedState(label, item.weight, state, component_log_weight(item)))
        stages.append(Stage("sensor:" + observation["sensor_id"], tuple(states),
                            (observation["sensor_id"],)))
    raw_priors = tuple(LabelledPrior(parent.label, Gaussian(parent.state.mean,
        parent.state.covariance), parent.weight, log_weight=component_log_weight(parent))
        for parent in prior_components)
    fused = fuse_labelled_sensor_posteriors(raw_priors, posteriors)
    # Empty observations preserve the actual prior law, including all mixture
    # masses and bits. Attach its source metadata without creating information.
    if not posteriors:
        fused = replace(fused, prior_version=prior.version, stamp=prior.stamp,
                        command=prior.command)
    components = []
    children = {item.label: [] for item in prior_components}
    for item in fused.components:
        parent = _posterior_parent(item, prior_components)
        children[parent.label].append(item)
        state = (parent.state if not posteriors else reexpress_state(
            parent.state, item.distribution.mean, item.distribution.covariance,
            version=prior.version+2, component_id=item.label))
        components.append(WeightedState(item.label, item.weight, state, component_log_weight(item)))
    components = tuple(components)
    sensor_ids = tuple(dict.fromkeys(item.sensor_id for item in posteriors))
    stages.append(Stage("fused", components, sensor_ids))
    frozen_branches = tuple(FrozenBodyBranch(parent.label, parent.weight, parent.state,
        raw.distribution, tuple(children[parent.label]), component_log_weight(parent))
        for parent, raw in zip(prior_components, raw_priors))
    body_prior = (raw_priors[0].distribution if len(raw_priors) == 1 else tuple(
        PosteriorComponent(item.label, item.distribution, item.weight,
                           log_weight=component_log_weight(item)) for item in raw_priors))
    records = {
        "root_frame": common["root_frame"],
        "valid_observation_count": len({item.observation_id for item in posteriors}),
        "observation_ids": list(fused.observation_ids),
        "approximation": "Bingham-derived S2 Gaussian proxy, global first order in covariance",
        "maximum_rotation_variance_domain_rad2": MAX_FIRST_ORDER_ROTATION_VARIANCE,
        "support_policy": {"absolute_roundoff_tolerance": SUPPORT_ATOL,
            "relative_roundoff_tolerance": SUPPORT_RTOL,
            "operation": "solve only on covariance range; no jitter or eigenvalue floor"},
        "prior_independence": common["prior"]["independence"],
        "unknown_marker": common["prior"]["marker"]["status"],
        "stages": [{"label": stage.label, "sensor_ids": list(stage.sensor_ids),
            "components": [_joint_record(component, paths, common["root_frame"])
                           for component in stage.components]} for stage in stages],
        "sensor_estimates": [estimate.diagnostics for estimate in local_estimates],
        "frozen_local_posteriors": [{
            "sensor_id": posterior.sensor_id, "observation_id": posterior.observation_id,
            "prior_component_id": posterior.query.component_id,
            "prior_version": posterior.query.prior_version, "stamp": posterior.query.stamp,
            "state_id": posterior.query.state_id, "command": list(posterior.query.command),
            "query_id": posterior.query.query_id,
            "target_prior": _gaussian_record(posterior.query.distribution),
            "jacobian": posterior.query.jacobian.tolist(),
            "components": [{"label": item.label, "weight": item.weight,
                "log_weight": component_log_weight(item), **_gaussian_record(item.distribution)}
                for item in posterior.components]} for posterior in posteriors],
        "frozen_body_charts": [{"prior_label": branch.label,
            "prior_weight": branch.weight, "prior_log_weight": branch.log_weight,
            "reference_state_id": branch.reference_state.state_id,
            "posterior_labels": [item.label for item in branch.posterior_components]}
            for branch in frozen_branches],
        "lift_total_covariance": [{"sensor_id": item.sensor_id, "label": item.label,
            "within": item.within.tolist(), "between": item.between.tolist()}
            for item in fused.individuals],
        "mixture_policy": "all labelled hypotheses retained; no pruning or approximate merging",
    }
    if joint_evidence:
        records['joint_observation_evidence'] = _joint_observation_evidence(
            prior_components, posteriors, local_estimates)
    return DemoResult(prior, tuple(stages), image, detection_image, paths["links"],
        paths["hand"], paths["camera"], paths["marker"], paths["camera_marker"],
        common["root_frame"], tuple(posteriors), tuple(local_estimates), records,
        body_prior, fused, frozen_branches)
