"""OpenCV AprilTag dictionary detection with explicit pixel uncertainty."""

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


class ArucoCornerDetector:
    """Detect IDs and ordered corners; pose inference is deliberately separate."""

    def __init__(
        self,
        family="DICT_APRILTAG_36h11",
        corner_sigma_px=0.5,
        corner_refinement=True,
        image_covariance=None,
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
        if self.detector is None:
            corners, identifiers, _ = cv2.aruco.detectMarkers(
                gray, self.dictionary, parameters=self.parameters
            )
        else:
            corners, identifiers, _ = self.detector.detectMarkers(gray)
        if identifiers is None:
            return ()
        observations = []
        for marker_corners, marker_id in zip(corners, identifiers.reshape(-1)):
            ordered = np.asarray(marker_corners, dtype=float).reshape(4, 2)
            observations.append(
                MarkerObservation(
                    int(marker_id),
                    ordered,
                    self.image_covariance,
                    self.family,
                )
            )
        return tuple(observations)
