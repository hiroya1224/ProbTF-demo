"""One frozen-prior update shared by every observation list."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from probtf.spherical_law.first_order import (
    BinghamTransform, JointTransformState, MAX_FIRST_ORDER_ROTATION_VARIANCE,
    point_query, reexpress_state,
)
from probtf.spherical_law.fusion import (
    Gaussian, fuse_sensor_posteriors, lift_sensor_posterior,
)
from probtf.spherical_law.support import SUPPORT_ATOL, SUPPORT_RTOL, covariance_support

from probtf_observation_fusion_demo.scene import render_image
from probtf_observation_fusion_demo.sensors import ADAPTERS


@dataclass(frozen=True)
class WeightedState:
    label: str
    weight: float
    state: JointTransformState


@dataclass(frozen=True)
class Stage:
    label: str
    components: tuple
    sensor_ids: tuple = ()


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


def run(common, observations, image=None, *, prior=None, paths=None):
    """Estimate using only explicit laws/geometry and the selected observations.

    Supplying ``image`` avoids all reads of ``physical_scene``. This boundary
    permits a test that changes hidden physical values while holding the actual
    observation input fixed. Even with no sensors the displayed scene is the
    same, while no image/direction information enters the body update.
    """
    image = render_image(common) if image is None else np.asarray(image).copy()
    image.setflags(write=False)
    # Scene orchestration ends here. Neither the body estimator nor any sensor
    # adapter receives physical truth or rendering settings, even as unused data.
    common = {key: common[key] for key in (
        "root_frame", "update", "prior", "known_geometry", "camera", "marker")}
    # The moving replay supplies the directly configured command-family state
    # and its actual URDF attachment paths.  All sensor estimation, lifting,
    # fusion and reexpression below remain the same fixed-time calculation.
    prior = initial_state(common) if prior is None else prior
    paths = geometry_paths(common) if paths is None else paths
    prior_component = WeightedState("common_body_prior", 1.0, prior)
    stages = [Stage("prior", (prior_component,))]
    posteriors, local_estimates = [], []
    seen = {}
    detection_image = image
    for observation in observations:
        identifier = observation["observation_id"]
        if identifier in seen:
            if observation != seen[identifier]:
                raise ValueError("A repeated observation ID has different input data.")
            continue
        seen[identifier] = observation
        estimate = ADAPTERS[observation["sensor"]](prior, observation, paths, common, image)
        local_estimates.append(estimate)
        if estimate.debug_image is not None:
            detection_image = estimate.debug_image
        if estimate.posterior is None:
            continue
        posterior = estimate.posterior
        posteriors.append(posterior)
        lifted = lift_sensor_posterior(posterior)
        states = tuple(WeightedState(item.label, item.weight,
                       reexpress_state(prior, item.distribution.mean, item.distribution.covariance,
                                       version=prior.version + 1,
                                       component_id=posterior.sensor_id + ":" + item.label))
                       for item in lifted)
        stages.append(Stage("sensor:" + posterior.sensor_id, states, (posterior.sensor_id,)))
    body_prior = Gaussian(prior.mean, prior.covariance)
    fused = fuse_sensor_posteriors(body_prior, posteriors)
    components = tuple(WeightedState(item.label, item.weight,
                       prior if not posteriors else
                       reexpress_state(prior, item.distribution.mean, item.distribution.covariance,
                                       version=prior.version + 2, component_id=item.label))
                       for item in fused.components)
    stages.append(Stage("fused", components, tuple(p.sensor_id for p in posteriors)))
    records = {
        "root_frame": common["root_frame"], "valid_observation_count": len(posteriors),
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
                                   for component in stage.components]}
                   for stage in stages],
        "sensor_estimates": [estimate.diagnostics for estimate in local_estimates],
        "frozen_local_posteriors": [
            {"sensor_id": posterior.sensor_id, "observation_id": posterior.observation_id,
             "prior_version": posterior.query.prior_version, "stamp": posterior.query.stamp,
             "state_id": posterior.query.state_id,
             "command": list(posterior.query.command), "query_id": posterior.query.query_id,
             "target_prior": _gaussian_record(posterior.query.distribution),
             "jacobian": posterior.query.jacobian.tolist(),
             "components": [{"label": item.label, "weight": item.weight,
                             **_gaussian_record(item.distribution)} for item in posterior.components]}
            for posterior in posteriors],
        "lift_total_covariance": [
            {"sensor_id": item.sensor_id, "label": item.label,
             "within": item.within.tolist(), "between": item.between.tolist()}
            for item in fused.individuals],
    }
    return DemoResult(prior, tuple(stages), image, detection_image, paths["links"],
                      paths["hand"], paths["camera"], paths["marker"], paths["camera_marker"],
                      common["root_frame"], tuple(posteriors), tuple(local_estimates), records,
                      body_prior, fused)
