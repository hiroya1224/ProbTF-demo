"""OpenCV AprilTag dictionary detection with explicit pixel uncertainty."""

from numbers import Integral

import cv2
import numpy as np

from prob_artag_detector.models import MarkerObservation


def _dictionary_id(family):
    if isinstance(family, str):
        if not family.startswith("DICT_APRILTAG_") or not hasattr(cv2.aruco, family):
            raise ValueError("Unsupported AprilTag dictionary {!r}.".format(family))
        return int(getattr(cv2.aruco, family)), family
    value = int(family)
    return value, str(value)


def isotropic_image_covariance(corner_sigma_px):
    sigma = float(corner_sigma_px)
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("corner_sigma_px must be positive and finite.")
    return np.eye(8, dtype=float) * (sigma * sigma)


def _finite_nonnegative(value, name):
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError("{} must be non-negative and finite.".format(name))
    return result


def _bootstrap_specs(
    sample_count,
    dither_px,
    blur_sigma_px,
    seed,
):
    """Build repeatable, paired perturbations without depending on frame history."""

    random = np.random.RandomState(seed)
    specs = []
    while len(specs) < sample_count:
        shift = random.normal(0.0, dither_px, size=2)
        blur = random.uniform(0.0, 2.0 * blur_sigma_px)
        noise_seed = int(random.randint(0, np.iinfo(np.int32).max))
        specs.append((shift, blur, noise_seed, 1.0))
        if len(specs) < sample_count:
            specs.append((-shift, blur, noise_seed, -1.0))
    return tuple(specs)


class ArucoCornerDetector:
    """Detect IDs and ordered corners; pose inference is deliberately separate.

    Adaptive mode re-detects each marker in a deterministically perturbed local
    image crop.  Its full 8D sample covariance is added to the isotropic
    ``corner_sigma_px`` floor.  Supplying ``image_covariance`` always selects
    the exact fixed-covariance behavior instead.
    """

    def __init__(
        self,
        family="DICT_APRILTAG_36h11",
        corner_sigma_px=0.5,
        corner_refinement=True,
        image_covariance=None,
        adaptive_covariance=False,
        bootstrap_samples=12,
        bootstrap_noise_std=6.0,
        bootstrap_dither_px=0.35,
        bootstrap_blur_sigma_px=0.35,
        bootstrap_covariance_shrinkage=0.25,
        bootstrap_min_success_ratio=0.5,
        bootstrap_dropout_sigma_px=2.0,
        bootstrap_seed=0,
    ):
        dictionary_id, family_name = _dictionary_id(family)
        self.family = family_name
        self.dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
        if hasattr(cv2.aruco, "DetectorParameters"):
            self.parameters = cv2.aruco.DetectorParameters()
        else:
            # OpenCV 4.2, as shipped by ROS Noetic on Ubuntu 20.04.
            self.parameters = cv2.aruco.DetectorParameters_create()
        if corner_refinement:
            self.parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.detector = (
            cv2.aruco.ArucoDetector(self.dictionary, self.parameters)
            if hasattr(cv2.aruco, "ArucoDetector")
            else None
        )
        covariance = (
            isotropic_image_covariance(corner_sigma_px)
            if image_covariance is None
            else np.asarray(image_covariance, dtype=float)
        )
        if covariance.shape != (8, 8) or not np.all(np.isfinite(covariance)):
            raise ValueError("image_covariance must be a finite 8x8 matrix.")
        if not np.allclose(covariance, covariance.T, rtol=0.0, atol=1e-10):
            raise ValueError("image_covariance must be symmetric.")
        self.image_covariance = covariance.copy()
        if isinstance(bootstrap_samples, bool) or not isinstance(
            bootstrap_samples, Integral
        ):
            raise TypeError("bootstrap_samples must be an integer.")
        self.bootstrap_samples = int(bootstrap_samples)
        if self.bootstrap_samples < 2:
            raise ValueError("bootstrap_samples must be at least 2.")
        self.bootstrap_noise_std = _finite_nonnegative(
            bootstrap_noise_std, "bootstrap_noise_std"
        )
        self.bootstrap_dither_px = _finite_nonnegative(
            bootstrap_dither_px, "bootstrap_dither_px"
        )
        self.bootstrap_blur_sigma_px = _finite_nonnegative(
            bootstrap_blur_sigma_px, "bootstrap_blur_sigma_px"
        )
        shrinkage = float(bootstrap_covariance_shrinkage)
        if not np.isfinite(shrinkage) or not 0.0 <= shrinkage <= 1.0:
            raise ValueError(
                "bootstrap_covariance_shrinkage must be finite and in [0, 1]."
            )
        self.bootstrap_covariance_shrinkage = shrinkage
        success_ratio = float(bootstrap_min_success_ratio)
        if not np.isfinite(success_ratio) or not 0.0 < success_ratio <= 1.0:
            raise ValueError(
                "bootstrap_min_success_ratio must be finite and in (0, 1]."
            )
        self.bootstrap_min_success_ratio = success_ratio
        self.bootstrap_dropout_sigma_px = _finite_nonnegative(
            bootstrap_dropout_sigma_px, "bootstrap_dropout_sigma_px"
        )
        if isinstance(bootstrap_seed, bool) or not isinstance(bootstrap_seed, Integral):
            raise TypeError("bootstrap_seed must be an integer.")
        self.bootstrap_seed = int(bootstrap_seed)
        if not 0 <= self.bootstrap_seed <= np.iinfo(np.uint32).max:
            raise ValueError("bootstrap_seed must be in the uint32 range.")

        # An explicitly supplied covariance is an exact fixed override.  This keeps
        # existing calibrated/custom covariance users independent of bootstrap
        # settings, even if adaptive_covariance was also requested.
        self.adaptive_covariance = bool(
            adaptive_covariance and image_covariance is None
        )
        self._bootstrap_specs = _bootstrap_specs(
            self.bootstrap_samples,
            self.bootstrap_dither_px,
            self.bootstrap_blur_sigma_px,
            self.bootstrap_seed,
        )

    def _detect_markers(self, gray):
        if self.detector is None:
            return cv2.aruco.detectMarkers(
                gray, self.dictionary, parameters=self.parameters
            )
        return self.detector.detectMarkers(gray)

    def _perturb_image(self, gray, spec):
        shift, blur_sigma, noise_seed, noise_sign = spec
        height, width = gray.shape
        transform = np.array(
            [[1.0, 0.0, shift[0]], [0.0, 1.0, shift[1]]],
            dtype=np.float32,
        )
        perturbed = cv2.warpAffine(
            gray.astype(np.float32),
            transform,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        if blur_sigma > 1e-12:
            perturbed = cv2.GaussianBlur(
                perturbed,
                (0, 0),
                sigmaX=blur_sigma,
                sigmaY=blur_sigma,
                borderType=cv2.BORDER_REFLECT_101,
            )
        if self.bootstrap_noise_std > 0.0:
            noise = np.random.RandomState(noise_seed).normal(
                0.0,
                self.bootstrap_noise_std,
                size=gray.shape,
            )
            perturbed += noise_sign * noise.astype(np.float32)
        return np.clip(np.rint(perturbed), 0.0, 255.0).astype(np.uint8), shift

    @staticmethod
    def _match_perturbed_corners(base_corners, detected_corners, detected_ids, shift):
        if detected_ids is None:
            return (None,) * len(base_corners)
        candidates = [
            np.asarray(corners, dtype=float).reshape(4, 2) - shift
            for corners in detected_corners
        ]
        ids = np.asarray(detected_ids).reshape(-1)
        available = set(range(len(candidates)))
        matches = []
        for corners, marker_id in base_corners:
            same_id = [index for index in available if ids[index] == marker_id]
            if not same_id:
                matches.append(None)
                continue
            distances = [
                float(np.sqrt(np.mean((candidates[index] - corners) ** 2)))
                for index in same_id
            ]
            candidate_index = same_id[int(np.argmin(distances))]
            side_lengths = np.linalg.norm(
                corners - np.roll(corners, -1, axis=0), axis=1
            )
            match_gate_px = max(5.0, 0.25 * float(np.mean(side_lengths)))
            if distances[int(np.argmin(distances))] > match_gate_px:
                matches.append(None)
                continue
            available.remove(candidate_index)
            matches.append(candidates[candidate_index])
        return tuple(matches)

    def _adaptive_image_covariances(self, gray, corners, identifiers):
        base_corners = tuple(
            (
                np.asarray(marker_corners, dtype=float).reshape(4, 2),
                int(marker_id),
            )
            for marker_corners, marker_id in zip(corners, identifiers.reshape(-1))
        )
        output = []
        image_height, image_width = gray.shape
        for marker_corners, marker_id in base_corners:
            edge_lengths = np.linalg.norm(
                marker_corners - np.roll(marker_corners, -1, axis=0),
                axis=1,
            )
            roi_margin = max(12.0, 0.5 * float(np.median(edge_lengths)))
            lower = np.floor(np.min(marker_corners, axis=0) - roi_margin).astype(int)
            upper = np.ceil(np.max(marker_corners, axis=0) + roi_margin).astype(int)
            x0 = max(0, int(lower[0]))
            y0 = max(0, int(lower[1]))
            x1 = min(image_width, int(upper[0]) + 1)
            y1 = min(image_height, int(upper[1]) + 1)
            roi = gray[y0:y1, x0:x1]
            local_corners = marker_corners - np.array([x0, y0], dtype=float)
            local_base = ((local_corners, marker_id),)
            marker_samples = [local_corners.reshape(8)]
            success_count = 0

            for spec in self._bootstrap_specs:
                perturbed, shift = self._perturb_image(roi, spec)
                perturbed_corners, perturbed_ids, _ = self._detect_markers(perturbed)
                matched_corners = self._match_perturbed_corners(
                    local_base,
                    perturbed_corners,
                    perturbed_ids,
                    shift,
                )[0]
                if matched_corners is not None:
                    marker_samples.append(matched_corners.reshape(8))
                    success_count += 1

            if len(marker_samples) >= 2:
                sample_array = np.asarray(marker_samples)
                deltas = sample_array[1:] - sample_array[0]
                # The raw second moment around the reported base corners is
                # Var(delta) + bias(delta) bias(delta)^T.  It is therefore the
                # localization MSE directly, without double-counting bias.
                sample_covariance = (
                    deltas.T @ deltas / float(deltas.shape[0])
                )
            else:
                sample_covariance = np.zeros((8, 8), dtype=float)
            diagonal = np.diag(np.diag(sample_covariance))
            sample_covariance = (
                (1.0 - self.bootstrap_covariance_shrinkage) * sample_covariance
                + self.bootstrap_covariance_shrinkage * diagonal
            )

            success_ratio = float(success_count) / float(self.bootstrap_samples)
            if success_ratio < self.bootstrap_min_success_ratio:
                denominator = max(success_ratio, 1.0 / self.bootstrap_samples)
                inflation = (self.bootstrap_min_success_ratio / denominator) ** 2
                sample_covariance *= inflation
            missing_ratio = 1.0 - success_ratio
            sample_covariance += (
                missing_ratio
                * self.bootstrap_dropout_sigma_px
                * self.bootstrap_dropout_sigma_px
                * np.eye(8)
            )

            covariance = self.image_covariance + sample_covariance
            covariance = 0.5 * (covariance + covariance.T)
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            eigenvalues = np.maximum(eigenvalues, np.finfo(float).eps)
            output.append((eigenvectors * eigenvalues) @ eigenvectors.T)
        return tuple(output)

    def detect(self, image):
        value = np.asarray(image)
        if value.ndim == 3 and value.shape[2] in (3, 4):
            code = cv2.COLOR_BGRA2GRAY if value.shape[2] == 4 else cv2.COLOR_BGR2GRAY
            gray = cv2.cvtColor(value, code)
        elif value.ndim == 2:
            gray = value
        else:
            raise ValueError("image must be grayscale, BGR, or BGRA.")
        if gray.dtype != np.uint8:
            raise ValueError("OpenCV marker detection requires a uint8 image.")
        corners, identifiers, _ = self._detect_markers(gray)
        if identifiers is None:
            return ()
        covariances = (
            self._adaptive_image_covariances(gray, corners, identifiers)
            if self.adaptive_covariance
            else (self.image_covariance,) * len(corners)
        )
        observations = []
        for marker_corners, marker_id, covariance in zip(
            corners, identifiers.reshape(-1), covariances
        ):
            ordered = np.asarray(marker_corners, dtype=float).reshape(4, 2)
            observations.append(
                MarkerObservation(
                    int(marker_id),
                    ordered,
                    covariance,
                    self.family,
                )
            )
        return tuple(observations)
