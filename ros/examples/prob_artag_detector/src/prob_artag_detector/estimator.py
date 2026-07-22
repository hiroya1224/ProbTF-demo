"""Planar multi-mode pose inference and native ProbTF v2 construction."""

from dataclasses import dataclass
import math
from typing import Tuple

import cv2
import numpy as np

from probtf.distributions import (
    BinghamOrientation,
    ConditionalGaussianTranslation,
    TransformComponent,
    TransformDistribution,
    TransformDistributionStamped,
    trace_zero_matrix,
)
from probtf.geometry import quat_left_matrix, rotmat_to_quat
from probtf.provenance import (
    ApproximationInfo,
    ApproximationKind,
    ComponentProvenance,
    TransformProvenance,
)
from probtf_estimators import coupling_from_hessian

from prob_artag_detector.camera import (
    ippe_square_object_points,
    pose_jacobian,
    project_points,
    transform_points,
)
from prob_artag_detector.models import (
    CameraModel,
    CandidateDiagnostic,
    EstimationDiagnostics,
    MarkerObservation,
    PoseMixtureResult,
)


class PoseEstimationError(ValueError):
    """The supplied image observation has no usable local pose mode."""


@dataclass(frozen=True)
class PoseSeed:
    rotation: np.ndarray
    translation: np.ndarray
    reprojection_error: float


@dataclass(frozen=True)
class _LocalMode:
    seed_index: int
    rotation: np.ndarray
    translation: np.ndarray
    objective: float
    iterations: int
    hessian: np.ndarray
    residual_covariance: np.ndarray
    local_translation_map: np.ndarray
    rotation_precision: np.ndarray
    rotation_coupling: np.ndarray
    quaternion_wxyz: np.ndarray
    bingham_parameter: np.ndarray
    log_mass: float
    refinement_status: str


def _corners(corners_px):
    value = np.asarray(corners_px, dtype=np.float64)
    if value.shape != (4, 2) or not np.all(np.isfinite(value)):
        raise ValueError("corners_px must be a finite array with shape (4,2).")
    return value


def _positive_definite(matrix, name, tolerance):
    value = np.asarray(matrix, dtype=float)
    if value.shape[0] != value.shape[1] or not np.all(np.isfinite(value)):
        raise ValueError("{} must be a finite square matrix.".format(name))
    value = 0.5 * (value + value.T)
    eigenvalues = np.linalg.eigvalsh(value)
    scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
    if float(eigenvalues[0]) <= float(tolerance) * scale:
        raise ValueError(
            "{} must be positive definite (minimum eigenvalue {}).".format(
                name, float(eigenvalues[0])
            )
        )
    np.linalg.cholesky(value)
    return value


def image_precision(image_covariance, tolerance=1e-12):
    covariance = np.asarray(image_covariance, dtype=float)
    if covariance.shape != (8, 8) or not np.all(np.isfinite(covariance)):
        raise ValueError("image_covariance must be a finite 8x8 matrix.")
    if not np.allclose(covariance, covariance.T, rtol=0.0, atol=1e-10):
        raise ValueError("image_covariance must be symmetric.")
    covariance = _positive_definite(covariance, "image_covariance", tolerance)
    return np.linalg.solve(covariance, np.eye(8, dtype=float))


def solve_ippe_square_candidates(corners_px, camera_model, tag_size_m):
    """Return every candidate produced by ``solvePnPGeneric`` IPPE_SQUARE."""

    if not isinstance(camera_model, CameraModel):
        raise TypeError("camera_model must be CameraModel.")
    corners = _corners(corners_px)
    object_points = ippe_square_object_points(tag_size_m)
    result = cv2.solvePnPGeneric(
        object_points,
        corners,
        camera_model.camera_matrix,
        camera_model.distortion,
        flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )
    if len(result) < 3 or not bool(result[0]):
        raise PoseEstimationError("IPPE_SQUARE did not return a pose candidate.")
    rotation_vectors = tuple(result[1])
    translation_vectors = tuple(result[2])
    reported_errors = None if len(result) < 4 else np.asarray(result[3], dtype=float).reshape(-1)
    seeds = []
    for index, (rotation_vector, translation_vector) in enumerate(
        zip(rotation_vectors, translation_vectors)
    ):
        rotation, _ = cv2.Rodrigues(np.asarray(rotation_vector, dtype=float).reshape(3, 1))
        translation = np.asarray(translation_vector, dtype=float).reshape(3)
        if reported_errors is not None and index < reported_errors.size:
            error = float(reported_errors[index])
        else:
            difference = project_points(
                object_points, rotation, translation, camera_model
            ).reshape(-1) - corners.reshape(-1)
            error = float(np.sqrt(np.mean(difference * difference)))
        seeds.append(PoseSeed(rotation, translation, error))
    if not seeds:
        raise PoseEstimationError("IPPE_SQUARE returned an empty candidate set.")
    return tuple(seeds)


def local_gauss_newton_hessian(
    object_points,
    rotation,
    translation,
    camera_model,
    precision,
    finite_difference_step=1e-6,
    verify_jacobian=False,
    translation_prior_precision=0.0,
):
    jacobian = pose_jacobian(
        object_points,
        rotation,
        translation,
        camera_model,
        finite_difference_step=finite_difference_step,
        verify=verify_jacobian,
    )
    hessian = jacobian.T @ precision @ jacobian
    prior_precision = float(translation_prior_precision)
    if not np.isfinite(prior_precision) or prior_precision < 0.0:
        raise ValueError("translation_prior_precision must be finite and non-negative.")
    hessian[:3, :3] += prior_precision * np.eye(3, dtype=float)
    return 0.5 * (hessian + hessian.T)


def reconstruct_pose_hessian(residual_covariance, local_translation_map, rotation_precision):
    """Reconstruct H from ``S``, ``B`` and the Schur precision ``Lambda``."""

    covariance = np.asarray(residual_covariance, dtype=float).reshape(3, 3)
    local_map = np.asarray(local_translation_map, dtype=float).reshape(3, 3)
    rotation_precision = np.asarray(rotation_precision, dtype=float).reshape(3, 3)
    hxx = np.linalg.solve(covariance, np.eye(3, dtype=float))
    hxu = -hxx @ local_map
    huu = local_map.T @ hxx @ local_map + rotation_precision
    return np.block([[hxx, hxu], [hxu.T, huu]])


def bingham_parameter_from_tangent_precision(quaternion_wxyz, rotation_precision):
    """Embed right-chart precision as ``A=-2 E Lambda E.T`` in trace-zero gauge."""

    precision = _positive_definite(rotation_precision, "rotation_precision", 1e-14)
    tangent = quat_left_matrix(quaternion_wxyz)[:, 1:]
    parameter = -2.0 * tangent @ precision @ tangent.T
    return trace_zero_matrix(parameter)


def normalize_log_weights(log_masses):
    values = np.asarray(tuple(log_masses), dtype=float)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("log_masses must be a non-empty finite vector.")
    shifted = values - float(np.max(values))
    weights = np.exp(shifted)
    total = float(np.sum(weights))
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("log_masses could not be normalized.")
    return weights / total


def _rotation_distance(first, second):
    relative = np.asarray(first, dtype=float).T @ np.asarray(second, dtype=float)
    cosine = min(1.0, max(-1.0, 0.5 * (float(np.trace(relative)) - 1.0)))
    return math.acos(cosine)


class PoseMixtureEstimator:
    """Build a local-Laplace mixture ProbTF edge from one marker observation."""

    def __init__(
        self,
        tag_size_m,
        max_iterations=30,
        convergence_tolerance=1e-9,
        min_depth=1e-6,
        dedup_translation_tolerance=1e-7,
        dedup_rotation_tolerance_rad=1e-7,
        finite_difference_step=1e-6,
        verify_jacobian=False,
        spd_tolerance=1e-12,
        translation_prior_mean=(0.0, 0.0, 0.0),
        translation_prior_variance=1e6,
    ):
        self.object_points = ippe_square_object_points(tag_size_m)
        self.tag_size_m = float(tag_size_m)
        self.max_iterations = int(max_iterations)
        self.convergence_tolerance = float(convergence_tolerance)
        self.min_depth = float(min_depth)
        self.dedup_translation_tolerance = float(dedup_translation_tolerance)
        self.dedup_rotation_tolerance_rad = float(dedup_rotation_tolerance_rad)
        self.finite_difference_step = float(finite_difference_step)
        self.verify_jacobian = bool(verify_jacobian)
        self.spd_tolerance = float(spd_tolerance)
        prior_mean = np.asarray(translation_prior_mean, dtype=float)
        if prior_mean.shape != (3,) or not np.all(np.isfinite(prior_mean)):
            raise ValueError("translation_prior_mean must be a finite 3-vector.")
        self.translation_prior_mean = prior_mean.copy()
        if translation_prior_variance is None:
            self.translation_prior_variance = None
            self.translation_prior_precision = 0.0
        else:
            prior_variance = float(translation_prior_variance)
            if not np.isfinite(prior_variance) or prior_variance <= 0.0:
                raise ValueError(
                    "translation_prior_variance must be positive and finite or None."
                )
            self.translation_prior_variance = prior_variance
            self.translation_prior_precision = 1.0 / prior_variance
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be positive.")
        for name in (
            "convergence_tolerance",
            "min_depth",
            "dedup_translation_tolerance",
            "dedup_rotation_tolerance_rad",
            "finite_difference_step",
            "spd_tolerance",
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError("{} must be positive and finite.".format(name))

    def _cheiral(self, rotation, translation):
        return bool(
            np.all(
                transform_points(self.object_points, rotation, translation)[:, 2]
                > self.min_depth
            )
        )

    def _residual(self, rotation, translation, camera_model, observed):
        return (
            project_points(self.object_points, rotation, translation, camera_model).reshape(-1)
            - observed.reshape(-1)
        )

    def _objective(self, residual, precision, translation):
        offset = np.asarray(translation, dtype=float) - self.translation_prior_mean
        return 0.5 * float(
            residual @ precision @ residual
            + self.translation_prior_precision * (offset @ offset)
        )

    def _seed_distance(self, rotation, translation, seed):
        """Dimensionless metric used only to preserve an IPPE seed branch."""

        return (
            float(np.linalg.norm(np.asarray(translation) - seed.translation))
            / self.tag_size_m
            + _rotation_distance(rotation, seed.rotation)
        )

    def _inside_seed_voronoi_cell(self, seed_index, rotation, translation, seeds):
        own_distance = self._seed_distance(rotation, translation, seeds[seed_index])
        other_distances = [
            self._seed_distance(rotation, translation, seed)
            for index, seed in enumerate(seeds)
            if index != seed_index
        ]
        return not other_distances or own_distance <= min(other_distances) + 1e-12

    def _refine(self, seed_index, seeds, observed, camera_model, precision):
        seed = seeds[seed_index]
        rotation = np.asarray(seed.rotation, dtype=float).copy()
        translation = np.asarray(seed.translation, dtype=float).copy()
        residual = self._residual(rotation, translation, camera_model, observed)
        objective = self._objective(residual, precision, translation)
        if not np.isfinite(objective):
            raise PoseEstimationError("initial candidate objective is nonfinite.")

        iterations = 0
        refinement_status = "converged"
        for iterations in range(1, self.max_iterations + 1):
            jacobian = pose_jacobian(
                self.object_points,
                rotation,
                translation,
                camera_model,
                finite_difference_step=self.finite_difference_step,
                verify=self.verify_jacobian,
            )
            hessian = 0.5 * (
                jacobian.T @ precision @ jacobian
                + (jacobian.T @ precision @ jacobian).T
            )
            gradient = jacobian.T @ precision @ residual
            hessian[:3, :3] += self.translation_prior_precision * np.eye(3)
            gradient[:3] += self.translation_prior_precision * (
                translation - self.translation_prior_mean
            )
            try:
                step = -np.linalg.solve(hessian, gradient)
            except np.linalg.LinAlgError as exc:
                raise PoseEstimationError("local pose Hessian is singular during refinement.") from exc
            if not np.all(np.isfinite(step)):
                raise PoseEstimationError("refinement produced a nonfinite step.")
            if float(np.linalg.norm(step)) <= self.convergence_tolerance:
                break

            accepted = False
            branch_blocked = False
            scale = 1.0
            for _ in range(24):
                delta_rotation, _ = cv2.Rodrigues(scale * step[3:6])
                candidate_rotation = rotation @ delta_rotation
                candidate_translation = translation + scale * step[:3]
                if not self._inside_seed_voronoi_cell(
                    seed_index,
                    candidate_rotation,
                    candidate_translation,
                    seeds,
                ):
                    branch_blocked = True
                    scale *= 0.5
                    continue
                if self._cheiral(candidate_rotation, candidate_translation):
                    candidate_residual = self._residual(
                        candidate_rotation,
                        candidate_translation,
                        camera_model,
                        observed,
                    )
                    candidate_objective = self._objective(
                        candidate_residual, precision, candidate_translation
                    )
                    if np.isfinite(candidate_objective) and candidate_objective < objective:
                        rotation = candidate_rotation
                        translation = candidate_translation
                        residual = candidate_residual
                        objective = candidate_objective
                        accepted = True
                        break
                scale *= 0.5
            if not accepted:
                if branch_blocked:
                    refinement_status = "voronoi_guard"
                    break
                if float(np.linalg.norm(gradient, ord=np.inf)) <= max(
                    self.convergence_tolerance, 1e-7
                ):
                    break
                if len(seeds) > 1:
                    # IPPE's second planar hypothesis need not be an unconstrained
                    # stationary point of the four-corner reprojection objective.
                    # Preserve the last valid point in its seed branch rather than
                    # silently deleting that discrete hypothesis.
                    refinement_status = "seed_branch_fallback"
                    break
                raise PoseEstimationError("Mahalanobis refinement line search failed.")

        return rotation, translation, objective, iterations, refinement_status

    def _local_mode(
        self,
        seed_index,
        rotation,
        translation,
        objective,
        iterations,
        camera_model,
        precision,
        refinement_status,
    ):
        hessian = local_gauss_newton_hessian(
            self.object_points,
            rotation,
            translation,
            camera_model,
            precision,
            self.finite_difference_step,
            self.verify_jacobian,
            self.translation_prior_precision,
        )
        hxx = _positive_definite(hessian[:3, :3], "H_xx", self.spd_tolerance)
        hxu = hessian[:3, 3:6]
        hux = hessian[3:6, :3]
        huu = hessian[3:6, 3:6]
        solve_hxx_hxu = np.linalg.solve(hxx, hxu)
        rotation_precision = _positive_definite(
            huu - hux @ solve_hxx_hxu,
            "rotation Schur precision",
            self.spd_tolerance,
        )
        residual_covariance = np.linalg.solve(hxx, np.eye(3, dtype=float))
        residual_covariance = 0.5 * (residual_covariance + residual_covariance.T)
        quaternion = rotmat_to_quat(rotation)
        coupling = coupling_from_hessian(hxx, hxu, quaternion)
        bingham_parameter = bingham_parameter_from_tangent_precision(
            quaternion, rotation_precision
        )
        sign_xx, logdet_xx = np.linalg.slogdet(hxx)
        sign_rotation, logdet_rotation = np.linalg.slogdet(rotation_precision)
        if sign_xx <= 0.0 or sign_rotation <= 0.0:
            raise ValueError("local Hessian determinant is not positive.")
        log_mass = -float(objective) - 0.5 * float(logdet_xx + logdet_rotation)
        return _LocalMode(
            seed_index,
            rotation,
            translation,
            float(objective),
            iterations,
            hessian,
            residual_covariance,
            coupling.local_translation_map,
            rotation_precision,
            coupling.rotation_coupling,
            quaternion,
            bingham_parameter,
            log_mass,
            str(refinement_status),
        )

    def _deduplicate(self, modes, diagnostic_rows):
        kept = []
        duplicates = 0
        for mode in sorted(modes, key=lambda item: item.objective):
            duplicate = any(
                np.linalg.norm(mode.translation - existing.translation)
                <= self.dedup_translation_tolerance
                and _rotation_distance(mode.rotation, existing.rotation)
                <= self.dedup_rotation_tolerance_rad
                for existing in kept
            )
            if duplicate:
                duplicates += 1
                diagnostic_rows[mode.seed_index].update(
                    accepted=False,
                    reason="duplicate_mode",
                    final_objective=mode.objective,
                    iterations=mode.iterations,
                )
            else:
                kept.append(mode)
        return tuple(kept), duplicates

    def estimate(
        self,
        observation,
        camera_model,
        parent_frame_id,
        child_frame_id,
        stamp,
        edge_id=None,
        authority="prob_artag_detector",
    ):
        if not isinstance(observation, MarkerObservation):
            raise TypeError("observation must be MarkerObservation.")
        if not isinstance(camera_model, CameraModel):
            raise TypeError("camera_model must be CameraModel.")
        precision = image_precision(observation.image_covariance, self.spd_tolerance)
        seeds = solve_ippe_square_candidates(
            observation.corners_px, camera_model, self.tag_size_m
        )
        diagnostic_rows = [
            dict(
                seed_index=index,
                accepted=False,
                reason="not_processed",
                initial_error=float(seed.reprojection_error),
                final_objective=float("inf"),
                iterations=0,
            )
            for index, seed in enumerate(seeds)
        ]
        modes = []
        rejected_cheirality = 0
        rejected_refinement = 0
        rejected_spd = 0
        for index, seed in enumerate(seeds):
            if not self._cheiral(seed.rotation, seed.translation):
                rejected_cheirality += 1
                diagnostic_rows[index].update(reason="cheirality")
                continue
            try:
                refined = self._refine(
                    index,
                    seeds,
                    observation.corners_px,
                    camera_model,
                    precision,
                )
            except (PoseEstimationError, FloatingPointError, np.linalg.LinAlgError) as exc:
                rejected_refinement += 1
                diagnostic_rows[index].update(reason="refinement: {}".format(exc))
                continue
            rotation, translation, objective, iterations, refinement_status = refined
            if not self._cheiral(rotation, translation):
                rejected_cheirality += 1
                diagnostic_rows[index].update(
                    reason="cheirality_after_refinement",
                    final_objective=objective,
                    iterations=iterations,
                )
                continue
            try:
                mode = self._local_mode(
                    index,
                    rotation,
                    translation,
                    objective,
                    iterations,
                    camera_model,
                    precision,
                    refinement_status,
                )
            except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
                rejected_spd += 1
                diagnostic_rows[index].update(
                    reason="local_distribution: {}".format(exc),
                    final_objective=objective,
                    iterations=iterations,
                )
                continue
            modes.append(mode)
            diagnostic_rows[index].update(
                accepted=True,
                reason=(
                    "accepted"
                    if refinement_status == "converged"
                    else "accepted_{}".format(refinement_status)
                ),
                final_objective=objective,
                iterations=iterations,
            )

        modes, duplicate_count = self._deduplicate(modes, diagnostic_rows)
        if not modes:
            reasons = "; ".join(row["reason"] for row in diagnostic_rows)
            raise PoseEstimationError("No valid local pose mode: {}".format(reasons))
        log_masses = tuple(mode.log_mass for mode in modes)
        weights = normalize_log_weights(log_masses)
        edge_identifier = edge_id or "{}__to__{}".format(
            str(parent_frame_id).strip().strip("/"),
            str(child_frame_id).strip().strip("/"),
        )
        source_id = "{}:{}".format(observation.family, observation.marker_id)
        approximation = ApproximationInfo(
            kind=ApproximationKind.TANGENT_SURROGATE,
            lossy=True,
            detail=(
                "Each planar pose mode uses a Gauss-Newton/Laplace local Hessian; "
                "orientation is its right-chart Bingham surrogate. IPPE seed branches "
                "are kept in separate pose-space Voronoi cells. A common proper broad "
                "translation prior is included unless explicitly disabled."
            ),
            source="prob_artag_detector.PoseMixtureEstimator",
        )
        components = []
        for index, (mode, weight) in enumerate(zip(modes, weights)):
            component_approximation = ApproximationInfo(
                kind=ApproximationKind.TANGENT_SURROGATE,
                lossy=True,
                detail=(
                    "Gauss-Newton/Laplace local law within the IPPE seed branch. "
                    + (
                        "Refinement stopped at the branch Voronoi guard before it could "
                        "collapse into another seed."
                        if mode.refinement_status == "voronoi_guard"
                        else (
                            "No decreasing in-branch Gauss-Newton step was available; "
                            "the last valid IPPE-branch pose was retained explicitly."
                            if mode.refinement_status == "seed_branch_fallback"
                            else "Refinement converged without reaching the branch guard."
                        )
                    )
                ),
                source="prob_artag_detector.PoseMixtureEstimator",
            )
            components.append(
                TransformComponent(
                    component_id="{}:mode:{}".format(edge_identifier, index),
                    raw_weight=float(weight),
                    orientation=BinghamOrientation.from_parameter_matrix(
                        mode.bingham_parameter,
                        reference_quaternion_wxyz=mode.quaternion_wxyz,
                    ),
                    translation=ConditionalGaussianTranslation(
                        mean_at_reference=mode.translation,
                        residual_covariance=mode.residual_covariance,
                        rotation_coupling=mode.rotation_coupling,
                    ),
                    provenance=ComponentProvenance(
                        source_ids=(source_id,),
                        method="ippe_square_mahalanobis_laplace",
                        detail="Mode initialized by IPPE candidate {}.".format(
                            mode.seed_index
                        ),
                    ),
                    approximation=component_approximation,
                )
            )
        record = TransformDistributionStamped(
            parent_frame_id=parent_frame_id,
            child_frame_id=child_frame_id,
            stamp=float(stamp),
            edge_id=edge_identifier,
            authority=authority,
            distribution=TransformDistribution(tuple(components)),
            provenance=TransformProvenance(
                source_ids=(source_id,),
                method="opencv_aruco_ippe_square",
                detail="Ordered corners and their full 8x8 covariance produced this edge.",
            ),
            is_static=False,
            approximation=approximation,
        )
        candidate_diagnostics = tuple(
            CandidateDiagnostic(**row) for row in diagnostic_rows
        )
        diagnostics = EstimationDiagnostics(
            seed_count=len(seeds),
            accepted_count=len(modes),
            deduplicated_count=duplicate_count,
            rejected_cheirality=rejected_cheirality,
            rejected_refinement=rejected_refinement,
            rejected_spd=rejected_spd,
            candidates=candidate_diagnostics,
        )
        return PoseMixtureResult(
            record=record,
            diagnostics=diagnostics,
            rotations=tuple(mode.rotation.copy() for mode in modes),
            translations=tuple(mode.translation.copy() for mode in modes),
            log_masses=log_masses,
            weights=tuple(float(value) for value in weights),
        )
