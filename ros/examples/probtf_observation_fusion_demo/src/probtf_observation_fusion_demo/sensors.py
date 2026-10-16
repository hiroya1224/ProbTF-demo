"""Sensor-local estimates in the common prior's frozen intrinsic coordinates.

Only these adapters know which sensor produced an observation. The body
lifting/fusion API receives completed local posteriors, never these raw inputs.
"""

from dataclasses import dataclass
import math

import cv2
import numpy as np
from scipy.special import logsumexp

from prob_artag_detector.detector import ArucoCornerDetector
from prob_artag_detector.estimator import PoseMixtureEstimator
from prob_artag_detector.visualization import draw_debug_image
from probtf.geometry import (
    quat_to_rotmat, rotmat_to_quat, quaternion_to_rotation_vector,
    right_perturbation_vec_rotation_jacobian, skew, tangent_basis,
)
from probtf.spherical_law.first_order import (
    bingham_rotation_coordinates, direction_query, linear_query, pose_query,
)
from probtf.spherical_law.fusion import (
    Gaussian, PosteriorComponent, SensorPosterior, condition_gaussian, freeze_query,
)

from probtf_observation_fusion_demo.scene import camera_model


@dataclass(frozen=True)
class LocalEstimate:
    posterior: object
    diagnostics: dict
    debug_image: object = None
    detector_result: object = None
    display_query: object = None
    observed_direction: object = None
    image_queries: tuple = ()
    log_evidence: object = None
    component_log_weights: tuple = ()


@dataclass(frozen=True)
class CameraXYQuery:
    """The explicit L=[I_2,0] marginal query of one completed Q_CM+ component.

    The arrays are read-only views of that very core distribution. They are
    recorded query inputs, never a replacement prior or an independently fitted law.
    """

    source: PosteriorComponent
    mean: np.ndarray
    covariance: np.ndarray


@dataclass(frozen=True)
class PreparedMarkerObservation:
    """The image-only detector/Hessian result, shared by every body hypothesis."""

    image: np.ndarray
    stamp: float
    observations: tuple
    detected: object
    camera: object
    debug_image: np.ndarray


def camera_xy_queries(posterior):
    return tuple(CameraXYQuery(component, component.distribution.mean[:2],
                               component.distribution.covariance[:2, :2])
                 for component in posterior.components)


def _camera_xy_bounds(queries):
    """Record the same Gaussian query bounds without drawing an image panel."""
    low = np.min([query.mean - 2.0 * np.sqrt(np.diag(query.covariance))
                  for query in queries], axis=0)
    high = np.max([query.mean + 2.0 * np.sqrt(np.diag(query.covariance))
                   for query in queries], axis=0)
    return low, high


def _marker_overlay(image, observations, detected, camera):
    """Draw only green detected corners and every image-only detector pose.

    The RGB axes are the detector's native image-likelihood pose modes. They
    do not represent the body posterior or replace its displayed core queries.
    Passing no observations to the existing helper suppresses its ID/weight
    text while preserving its axis projection and axis length.
    """
    output = draw_debug_image(image, (), (detected,), camera)
    for observation in observations:
        corners = np.rint(observation.corners_px).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(output, [corners], True, (0, 220, 0), 2, cv2.LINE_AA)
    return output


def _posterior_weights(log_weights):
    values = np.asarray(log_weights, dtype=float)
    values = np.exp(values - np.max(values))
    return values / np.sum(values)


def imu_estimate(state, observation, paths, common, image, *, shared_observation_frame=False):
    """Gravity constrains a direction, with exactly two observation coordinates."""
    root_gravity = np.asarray(observation["gravity_in_root"], dtype=float)
    root_gravity /= np.linalg.norm(root_gravity)
    observed = np.asarray(observation["gravity_in_sensor"], dtype=float)
    observed /= np.linalg.norm(observed)
    target_path = paths["links"][observation["target_link"]]
    inverse_path = tuple((edge, not inverse) for edge, inverse in reversed(target_path))
    target_id = observation["sensor_id"] + ":gravity"
    direction = direction_query(state, inverse_path, root_gravity,
                                query_id=target_id, frame=observation["target_link"])
    mode_direction = direction.mode
    # Change hypotheses must use the very same observation coordinates and
    # base measure. A hypothesis-specific prior-mode plane is unsuitable for
    # comparing their evidences. This observed-direction plane is shared.
    basis = tangent_basis(observed if shared_observation_frame else
                          mode_direction / np.linalg.norm(mode_direction))
    intrinsic = linear_query(direction, basis.T, query_id=target_id + ":frozen_tangent2")
    frozen = freeze_query(state, intrinsic)
    variance = float(observation["direction_sigma_rad"]) ** 2
    local, evidence = condition_gaussian(frozen.distribution, basis.T @ observed,
                                         variance * np.eye(2))
    posterior = SensorPosterior(frozen, observation["sensor_id"],
                               observation["observation_id"],
                               (PosteriorComponent("gravity", local, 1.0),))
    return LocalEstimate(posterior, {
        "sensor": "imu", "target_link": observation["target_link"],
        "coordinates": ("observation-fixed shared S2 tangent plane (2D)" if shared_observation_frame
                        else "prior-mode fixed S2 tangent plane (2D)"),
        "shared_observation_frame": bool(shared_observation_frame),
        "tangent_basis": basis.tolist(), "observed_direction": observed.tolist(),
        "information_rank": 2, "heading_observation": False,
        "measurement_variance": variance, "log_evidence": float(evidence),
    }, display_query=direction, observed_direction=observed.copy(),
        log_evidence=float(evidence), component_log_weights=(0.,))


def _rotation_log_right_jacobian_inverse(vector):
    """Derivative of log(R_ref.T R Exp(delta)), not a candidate noise model."""
    angle = float(np.linalg.norm(vector))
    cross = skew(vector)
    if angle < 1e-6:
        coefficient = 1.0 / 12.0 + angle * angle / 720.0
    else:
        coefficient = (1.0 - 0.5 * angle / math.tan(0.5 * angle)) / (angle * angle)
    return np.eye(3) + 0.5 * cross + coefficient * (cross @ cross)


def _marker_mode_chart(component, reference_rotation):
    """Recover the image estimator's original joint Laplace Gaussian.

    The native component stores that image-Hessian Gaussian through its
    Schur factors. This adapter recovers those factors and changes its chart;
    it does not claim to compute moments of the nonlinear native conditional
    Bingham law. In particular the translation entry is the Laplace mode.
    """
    quaternion = component.orientation.mode_wxyz
    rotation = quat_to_rotmat(quaternion)
    _, rotation_covariance = bingham_rotation_coordinates(component.orientation.parameter_matrix())
    coupling = (component.translation.rotation_coupling
                @ right_perturbation_vec_rotation_jacobian(quaternion))
    local_covariance = np.block([
        [component.translation.residual_covariance + coupling @ rotation_covariance @ coupling.T,
         coupling @ rotation_covariance],
        [rotation_covariance @ coupling.T, rotation_covariance],
    ])
    coordinate = quaternion_to_rotation_vector(rotmat_to_quat(reference_rotation.T @ rotation))
    coordinate_map = np.eye(6)
    coordinate_map[3:, 3:] = _rotation_log_right_jacobian_inverse(coordinate)
    covariance = coordinate_map @ local_covariance @ coordinate_map.T
    mean = np.concatenate((component.translation.mean_at_reference, coordinate))
    sign, log_volume = np.linalg.slogdet(coordinate_map)
    if sign <= 0. or not np.isfinite(log_volume):
        raise ValueError("The image likelihood chart map has invalid volume.")
    return Gaussian(mean, covariance), float(log_volume)


def _marker_mode_law(component, reference_rotation):
    return _marker_mode_chart(component, reference_rotation)[0]


def prepare_marker_observation(common, image, stamp):
    """Decode and refine an image once, before conditioning any prior branch."""
    settings = common["marker"]
    if settings["translation_prior_variance"] is not None:
        raise ValueError("This image-only adapter requires the detector translation prior to be explicitly disabled.")
    detector = ArucoCornerDetector(family=settings["family"],
        corner_sigma_px=settings["corner_sigma_px"], adaptive_covariance=False)
    observations = tuple(item for item in detector.detect(image)
                         if item.marker_id == int(settings["id"]))
    camera = camera_model(common["camera"])
    if not observations:
        return PreparedMarkerObservation(image, float(stamp), (), None, camera,
                                         np.asarray(image).copy())
    if len(observations) != 1:
        raise ValueError("The configured marker ID occurs more than once in the image.")
    estimator = PoseMixtureEstimator(settings["size"], translation_prior_variance=None)
    detected = estimator.estimate(observations[0], camera, "camera", "marker", float(stamp),
                                 camera_model_source_id="configured_calibrated_pinhole")
    return PreparedMarkerObservation(image, float(stamp), observations, detected, camera,
                                     _marker_overlay(image, observations, detected, camera))


def marker_estimate(state, observation, paths, common, image, *, prepared=None):
    """Estimate every image branch, then combine it with the common target prior.

    PoseMixtureEstimator normally includes a proper broad translation prior.
    It is explicitly disabled here: its local Laplace components represent
    image-only likelihood modes. Their normalized image masses are retained,
    and multiplying the common target prior adds overlap evidence per branch.
    """
    prepared = prepare_marker_observation(common, image, state.stamp) if prepared is None else prepared
    if prepared.image is not image or prepared.stamp != state.stamp:
        raise ValueError("Prepared image likelihood belongs to a different observation.")
    if prepared.detected is None:
        return LocalEstimate(None, {"sensor": "marker", "detected": False},
                             prepared.debug_image)
    detected, observations = prepared.detected, prepared.observations
    query = pose_query(state, paths["camera_marker"],
                       query_id=observation["sensor_id"] + ":camera_marker", frame="camera")
    frozen = freeze_query(state, query)
    reference_rotation = quat_to_rotmat(query.rotation_wxyz)
    modes, log_weights, details = [], [], []
    image_log_weights = np.asarray(detected.log_masses)-logsumexp(detected.log_masses)
    for component, image_weight, image_log_weight, seed_index in zip(
            detected.record.distribution.components, detected.weights,
            image_log_weights, detected.seed_indices):
        likelihood, chart_log_volume = _marker_mode_chart(component, reference_rotation)
        posterior, overlap = condition_gaussian(frozen.distribution,
                                                likelihood.mean, likelihood.covariance)
        label = component.component_id
        modes.append((label, posterior))
        # The native detector exposes finite log masses even if its float
        # weight underflows. Retain this likelihood branch in log coordinates.
        if image_weight < 0.0:
            raise ValueError("The detector returned a negative component weight.")
        # A scalar likelihood does not transform as a probability density.
        # N(delta)=|det J_log| N(z) preserves its native image-Hessian peak.
        log_weights.append(float(image_log_weight) + chart_log_volume + overlap)
        details.append({"label": label, "detector_seed_index": int(seed_index),
                        "image_weight": float(image_weight), "image_log_weight": float(image_log_weight),
                        "chart_log_volume": chart_log_volume,
                        "prior_overlap_log_evidence": float(overlap),
                        "image_only_mean": likelihood.mean.tolist(),
                        "image_only_covariance": likelihood.covariance.tolist()})
    weights = _posterior_weights(log_weights)
    log_evidence = float(logsumexp(log_weights))
    conditional_logs = tuple(float(value-log_evidence) for value in log_weights)
    components = tuple(PosteriorComponent(label, posterior, float(weight), log_weight=log_weight)
                       for (label, posterior), weight, log_weight in zip(modes, weights, conditional_logs))
    result = SensorPosterior(frozen, observation["sensor_id"], observation["observation_id"], components)
    debug = prepared.debug_image
    image_queries = camera_xy_queries(result)
    low, high = _camera_xy_bounds(image_queries)
    return LocalEstimate(result, {
        "sensor": "marker", "detected": True,
        "detector_prior": "translation prior explicitly disabled; image-only local likelihood mixture",
        "image_approximation": "recover original joint translation/right-rotation Laplace Gaussian from the detector Hessian Schur factors; not native conditional-law moments",
        "reference_prior": "same frozen common body push-forward as every other sensor",
        "corner_covariance": observations[0].image_covariance.tolist(),
        "detector_weights": list(detected.weights), "components": details,
        "local_posterior_weights": weights.tolist(),
        "image_overlay": {
            "corners": "green outline of detected image corners",
            "pose_axes": "all native image-only detector pose modes, not the body posterior",
            "text_or_distribution_panels": False,
        },
        "image_xy_query": {
            "operation": "L=[I_2,0] applied to the same core Q_CM+ Gaussian components",
            "state_id": result.query.state_id, "query_id": result.query.query_id,
            "frame": "camera", "units": "metres", "contour": "Mahalanobis radius 2",
            "plot_bounds_low": low.tolist(), "plot_bounds_high": high.tolist(),
            "components": [{"label": item.source.label, "weight": item.source.weight,
                            "mean": item.mean.tolist(), "covariance": item.covariance.tolist()}
                           for item in image_queries],
        },
    }, debug, detected, query, image_queries=image_queries, log_evidence=log_evidence,
        component_log_weights=conditional_logs)


ADAPTERS = {"imu": imu_estimate, "marker": marker_estimate}
PREPARERS = {"marker": prepare_marker_observation}
