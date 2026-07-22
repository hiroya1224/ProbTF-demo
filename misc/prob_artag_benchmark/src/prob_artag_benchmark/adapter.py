"""Narrow adapter around the independently implemented Phase-2 package."""

import inspect
from typing import Dict

import numpy as np

from .models import BenchmarkConfig, SeedPose


class ApiMismatchError(RuntimeError):
    """The installed Phase-2 package does not expose the documented API."""


def _require(value, name):
    if not hasattr(value, name):
        raise ApiMismatchError("Phase-2 API is missing {!r}".format(name))
    return getattr(value, name)


class DefaultPipelineAdapter:
    """Instantiate detector/camera/estimator objects without renderer imports.

    Phase-1 metadata is intentionally parsed as JSON rather than importing the
    renderer.  This keeps generated datasets usable even when rendering or
    OpenGL dependencies are unavailable on the benchmark host.
    """

    def __init__(self, config: BenchmarkConfig):
        try:
            import prob_artag_detector as api
        except Exception as exc:
            raise ApiMismatchError(
                "Cannot import prob_artag_detector; source/devel paths must be active: {}".format(
                    exc
                )
            ) from exc
        self.api = api
        detector_type = _require(api, "ArucoCornerDetector")
        self.camera_type = _require(api, "CameraModel")
        self.estimator_type = _require(api, "PoseMixtureEstimator")
        self.seed_type = getattr(api, "PoseSeed", None)
        self.solve_candidates_api = _require(api, "solve_ippe_square_candidates")
        self.config = config
        estimate_parameters = inspect.signature(self.estimator_type.estimate).parameters
        self.reuses_explicit_seeds = bool(
            self.seed_type is not None and "pose_seeds" in estimate_parameters
        )
        if self.reuses_explicit_seeds:
            seed_note = "The exact IPPE seed tuple is reused by the estimator."
        else:
            seed_note = (
                "Legacy detector API: the estimator resolves IPPE independently; "
                "deterministic candidate ordering is required for seed mapping."
            )
        self.api_notes = (
            "PoseMixtureResult.seed_indices is preferred for mode-to-seed mapping; "
            "component provenance and pose proximity are legacy fallbacks.",
            seed_note,
        )
        self.detector = detector_type(
            family=config.family,
            corner_sigma_px=config.corner_sigma_px,
            corner_refinement=config.corner_refinement,
        )
        self._estimators: Dict[float, object] = {}

    def camera_from_metadata(self, camera):
        matrix = camera.get("camera_matrix")
        if matrix is None:
            required = ("fx", "fy", "cx", "cy")
            if not all(name in camera for name in required):
                raise ValueError(
                    "camera metadata needs camera_matrix or fx/fy/cx/cy"
                )
            matrix = [
                [camera["fx"], 0.0, camera["cx"]],
                [0.0, camera["fy"], camera["cy"]],
                [0.0, 0.0, 1.0],
            ]
        return self.camera_type(
            np.asarray(matrix, dtype=float).reshape(3, 3),
            np.asarray(camera.get("distortion", ()), dtype=float),
            int(camera.get("width", 0)),
            int(camera.get("height", 0)),
        )

    def detect(self, bgr_image):
        observations = _require(self.detector, "detect")(bgr_image)
        return tuple(observations)

    def solve_candidates(self, observation, camera_model, tag_size_m):
        seeds = self.solve_candidates_api(
            _require(observation, "corners_px"), camera_model, float(tag_size_m)
        )
        output = []
        for seed in seeds:
            output.append(
                SeedPose(
                    _require(seed, "rotation"),
                    _require(seed, "translation"),
                    _require(seed, "reprojection_error"),
                )
            )
        return tuple(output)

    def estimator(self, tag_size_m):
        key = float(tag_size_m)
        if key not in self._estimators:
            config = self.config
            self._estimators[key] = self.estimator_type(
                tag_size_m=key,
                max_iterations=config.estimator_max_iterations,
                convergence_tolerance=config.estimator_convergence_tolerance,
                min_depth=config.estimator_min_depth,
                dedup_translation_tolerance=config.estimator_dedup_translation_tolerance,
                dedup_rotation_tolerance_rad=config.estimator_dedup_rotation_tolerance_rad,
                finite_difference_step=config.estimator_finite_difference_step,
                verify_jacobian=config.estimator_verify_jacobian,
            )
        return self._estimators[key]

    def estimate(
        self,
        observation,
        camera_model,
        tag_size_m,
        parent_frame_id,
        child_frame_id,
        stamp,
        edge_id,
        authority,
        seeds=None,
    ):
        method = _require(self.estimator(tag_size_m), "estimate")
        pose_seeds = None
        if seeds is not None and self.reuses_explicit_seeds:
            pose_seeds = tuple(
                self.seed_type(
                    seed.rotation,
                    seed.translation,
                    seed.reported_reprojection_error_px,
                )
                for seed in seeds
            )
        keyword_arguments = dict(
            parent_frame_id=parent_frame_id,
            child_frame_id=child_frame_id,
            stamp=float(stamp),
            edge_id=edge_id,
            authority=authority,
        )
        if self.reuses_explicit_seeds:
            keyword_arguments["pose_seeds"] = pose_seeds
        return method(observation, camera_model, **keyword_arguments)
