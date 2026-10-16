"""Sensor-local estimates in the common prior's frozen intrinsic coordinates.

Only these adapters know which sensor produced an observation. The body
lifting/fusion API receives completed local posteriors, never these raw inputs.
"""

from dataclasses import dataclass
import math

import cv2
import numpy as np

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
from probtf.spherical_law.support import covariance_support

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


@dataclass(frozen=True)
class CameraXYQuery:
    """The explicit L=[I_2,0] marginal query of one completed Q_CM+ component.

    The arrays are read-only views of that very core distribution. They are
    drawing inputs, never a replacement prior or an independently fitted law.
    """

    source: PosteriorComponent
    mean: np.ndarray
    covariance: np.ndarray


def camera_xy_queries(posterior):
    return tuple(CameraXYQuery(component, component.distribution.mean[:2],
                               component.distribution.covariance[:2, :2])
                 for component in posterior.components)


def _camera_xy_inset(image, queries):
    """Plot exact 2-sigma XY covariance ellipses; axes are camera metres."""
    low = np.min([query.mean - 2.0 * np.sqrt(np.diag(query.covariance))
                  for query in queries], axis=0)
    high = np.max([query.mean + 2.0 * np.sqrt(np.diag(query.covariance))
                   for query in queries], axis=0)
    # The physical bounds come solely from the queried Gaussians. A collapsed
    # coordinate is drawn at the centre without manufacturing a spatial range.
    span = high - low
    box = (477, 70, 557, 150)
    def pixels(values):
        values = np.asarray(values, dtype=float)
        normalized = np.divide(values - low, span, out=np.full_like(values, .5), where=span > 0)
        return np.column_stack((box[0] + normalized[:, 0] * (box[2] - box[0]),
                                box[3] - normalized[:, 1] * (box[3] - box[1])))
    cv2.rectangle(image, (394, 40), (638, 196), (248, 248, 248), -1)
    cv2.putText(image, "Q_CM+ XY marginal: 2 sigma", (400, 55),
                cv2.FONT_HERSHEY_SIMPLEX, .36, (25, 25, 25), 1, cv2.LINE_AA)
    cv2.rectangle(image, box[:2], box[2:], (110, 110, 110), 1)
    cv2.putText(image, "y_C [m]", (397, 73), cv2.FONT_HERSHEY_SIMPLEX,
                .30, (25, 25, 25), 1, cv2.LINE_AA)
    cv2.putText(image, "{:.5f}".format(high[1]), (427, 85), cv2.FONT_HERSHEY_SIMPLEX,
                .29, (25, 25, 25), 1, cv2.LINE_AA)
    cv2.putText(image, "{:.5f}".format(low[1]), (427, 150), cv2.FONT_HERSHEY_SIMPLEX,
                .29, (25, 25, 25), 1, cv2.LINE_AA)
    cv2.putText(image, "{:.5f}".format(low[0]), (448, 164), cv2.FONT_HERSHEY_SIMPLEX,
                .29, (25, 25, 25), 1, cv2.LINE_AA)
    cv2.putText(image, "{:.5f}".format(high[0]), (556, 164), cv2.FONT_HERSHEY_SIMPLEX,
                .29, (25, 25, 25), 1, cv2.LINE_AA)
    cv2.putText(image, "x_C [m]", (499, 176), cv2.FONT_HERSHEY_SIMPLEX,
                .30, (25, 25, 25), 1, cv2.LINE_AA)
    angle = np.linspace(0.0, 2.0 * math.pi, 129)
    circle = np.column_stack((np.cos(angle), np.sin(angle)))
    colors = ((190, 75, 15), (150, 30, 170), (20, 125, 20), (20, 80, 180))
    for index, query in enumerate(queries):
        support = covariance_support(query.covariance)
        boundary = query.mean + 2.0 * circle[:, :support.rank] @ (
            support.basis * np.sqrt(support.eigenvalues)).T
        color = colors[index % len(colors)]
        cv2.polylines(image, [np.rint(pixels(boundary)).astype(np.int32)], True, color, 1, cv2.LINE_AA)
        centre = tuple(np.rint(pixels(query.mean[None, :])[0]).astype(int))
        cv2.drawMarker(image, centre, color, cv2.MARKER_CROSS, 6, 1, cv2.LINE_AA)
        cv2.putText(image, "k{} w={:.6f}".format(index, query.source.weight),
                    (400 + (index % 2) * 117, 189 + (index // 2) * 13),
                    cv2.FONT_HERSHEY_SIMPLEX, .30, color, 1, cv2.LINE_AA)
    return low, high


def _posterior_weights(log_weights):
    values = np.asarray(log_weights, dtype=float)
    values = np.exp(values - np.max(values))
    return values / np.sum(values)


def imu_estimate(state, observation, paths, common, image):
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
    basis = tangent_basis(mode_direction / np.linalg.norm(mode_direction))
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
        "coordinates": "prior-mode fixed S2 tangent plane (2D)",
        "tangent_basis": basis.tolist(), "observed_direction": observed.tolist(),
        "information_rank": 2, "heading_observation": False,
        "measurement_variance": variance, "log_evidence": float(evidence),
    }, display_query=direction, observed_direction=observed.copy())


def _rotation_log_right_jacobian_inverse(vector):
    """Derivative of log(R_ref.T R Exp(delta)), not a candidate noise model."""
    angle = float(np.linalg.norm(vector))
    cross = skew(vector)
    if angle < 1e-6:
        coefficient = 1.0 / 12.0 + angle * angle / 720.0
    else:
        coefficient = (1.0 - 0.5 * angle / math.tan(0.5 * angle)) / (angle * angle)
    return np.eye(3) + 0.5 * cross + coefficient * (cross @ cross)


def _marker_mode_law(component, reference_rotation):
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
    return Gaussian(mean, covariance)


def marker_estimate(state, observation, paths, common, image):
    """Estimate every image branch, then combine it with the common target prior.

    PoseMixtureEstimator normally includes a proper broad translation prior.
    It is explicitly disabled here: its local Laplace components represent
    image-only likelihood modes. Their normalized image masses are retained,
    and multiplying the common target prior adds overlap evidence per branch.
    """
    settings = common["marker"]
    if settings["translation_prior_variance"] is not None:
        raise ValueError("This image-only adapter requires the detector translation prior to be explicitly disabled.")
    detector = ArucoCornerDetector(family=settings["family"],
                                   corner_sigma_px=settings["corner_sigma_px"],
                                   adaptive_covariance=False)
    observations = tuple(item for item in detector.detect(image)
                         if item.marker_id == int(settings["id"]))
    if not observations:
        return LocalEstimate(None, {"sensor": "marker", "detected": False},
                             draw_debug_image(image, (), status_text="No detection: identity observation update"))
    if len(observations) != 1:
        raise ValueError("The configured marker ID occurs more than once in the image.")
    estimator = PoseMixtureEstimator(settings["size"], translation_prior_variance=None)
    camera = camera_model(common["camera"])
    detected = estimator.estimate(observations[0], camera, "camera", "marker",
                                 state.stamp, camera_model_source_id="configured_calibrated_pinhole")
    query = pose_query(state, paths["camera_marker"],
                       query_id=observation["sensor_id"] + ":camera_marker", frame="camera")
    frozen = freeze_query(state, query)
    reference_rotation = quat_to_rotmat(query.rotation_wxyz)
    modes, log_weights, details = [], [], []
    for component, image_weight, seed_index in zip(detected.record.distribution.components,
                                                  detected.weights, detected.seed_indices):
        likelihood = _marker_mode_law(component, reference_rotation)
        posterior, overlap = condition_gaussian(frozen.distribution,
                                                likelihood.mean, likelihood.covariance)
        label = component.component_id
        modes.append((label, posterior))
        # A numerically zero detector mass is still an observed component.
        # Retain its label and zero mass instead of selecting a different mode
        # or failing on log(0) during a moving image sequence.
        if image_weight < 0.0:
            raise ValueError("The detector returned a negative component weight.")
        log_weights.append((-math.inf if image_weight == 0.0 else math.log(image_weight)) + overlap)
        details.append({"label": label, "detector_seed_index": int(seed_index),
                        "image_weight": float(image_weight), "prior_overlap_log_evidence": float(overlap),
                        "image_only_mean": likelihood.mean.tolist(),
                        "image_only_covariance": likelihood.covariance.tolist()})
    weights = _posterior_weights(log_weights)
    components = tuple(PosteriorComponent(label, posterior, float(weight))
                       for (label, posterior), weight in zip(modes, weights))
    result = SensorPosterior(frozen, observation["sensor_id"], observation["observation_id"], components)
    debug = draw_debug_image(image, observations, (detected,), camera,
                            status_text="Image-only detector axes; all hypotheses retained")
    # The image panel also identifies the *estimated* camera-relative laws.
    # These values come directly from Q_s^+, not the detector likelihood or truth.
    start_y = max(90, debug.shape[0] - 28 - 52 * len(components))
    cv2.rectangle(debug, (0, start_y - 17), (debug.shape[1], debug.shape[0] - 25),
                  (248, 248, 248), thickness=-1)
    cv2.putText(debug, "Q_CM+ in frozen common-prior chart (m, rad)", (6, start_y),
                cv2.FONT_HERSHEY_SIMPLEX, .40, (25, 25, 25), 1, cv2.LINE_AA)
    for index, component in enumerate(components):
        local = component.distribution
        translation_text = "k{} w={:.6f} m_x=[{}]".format(
            index, component.weight, ",".join("{:.4f}".format(x) for x in local.mean[:3]))
        covariance_text = "diag(S)=[{}]".format(
            ",".join("{:.2g}".format(x) for x in np.diag(local.covariance)))
        orientation_text = "m_phi=[{}] rad".format(
            ",".join("{:.4f}".format(x) for x in local.mean[3:]))
        y_value = start_y + 17 + index * 52
        cv2.putText(debug, translation_text, (6, y_value), cv2.FONT_HERSHEY_SIMPLEX,
                    .40, (25, 25, 25), 1, cv2.LINE_AA)
        cv2.putText(debug, covariance_text, (6, y_value + 15), cv2.FONT_HERSHEY_SIMPLEX,
                    .37, (25, 25, 25), 1, cv2.LINE_AA)
        cv2.putText(debug, orientation_text, (6, y_value + 30), cv2.FONT_HERSHEY_SIMPLEX,
                    .37, (25, 25, 25), 1, cv2.LINE_AA)
    image_queries = camera_xy_queries(result)
    low, high = _camera_xy_inset(debug, image_queries)
    return LocalEstimate(result, {
        "sensor": "marker", "detected": True,
        "detector_prior": "translation prior explicitly disabled; image-only local likelihood mixture",
        "image_approximation": "recover original joint translation/right-rotation Laplace Gaussian from the detector Hessian Schur factors; not native conditional-law moments",
        "reference_prior": "same frozen common body push-forward as every other sensor",
        "corner_covariance": observations[0].image_covariance.tolist(),
        "detector_weights": list(detected.weights), "components": details,
        "local_posterior_weights": weights.tolist(),
        "image_xy_query": {
            "operation": "L=[I_2,0] applied to the same core Q_CM+ Gaussian components",
            "state_id": result.query.state_id, "query_id": result.query.query_id,
            "frame": "camera", "units": "metres", "contour": "Mahalanobis radius 2",
            "plot_bounds_low": low.tolist(), "plot_bounds_high": high.tolist(),
            "components": [{"label": item.source.label, "weight": item.source.weight,
                            "mean": item.mean.tolist(), "covariance": item.covariance.tolist()}
                           for item in image_queries],
        },
    }, debug, detected, query, image_queries=image_queries)


ADAPTERS = {"imu": imu_estimate, "marker": marker_estimate}
