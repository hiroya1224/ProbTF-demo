"""AprilTag texture generation with OpenCV-version compatibility."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TagTexture:
    rgb: np.ndarray
    marker_pixels: int
    margin_pixels: int
    family: str
    marker_id: int

    @property
    def plane_to_marker_scale(self) -> float:
        return float(self.rgb.shape[0]) / float(self.marker_pixels)


def _aruco_dictionary(cv2_module: object, family: str) -> object:
    aruco = cv2_module.aruco
    if not hasattr(aruco, family):
        raise ValueError("unknown OpenCV ArUco/AprilTag family: {}".format(family))
    dictionary_id = getattr(aruco, family)
    if hasattr(aruco, "getPredefinedDictionary"):
        return aruco.getPredefinedDictionary(dictionary_id)
    return aruco.Dictionary_get(dictionary_id)


def generate_tag_texture(marker_id: int, family: str = "DICT_APRILTAG_36h11",
                         marker_pixels: int = 256, margin_pixels: int = 32,
                         border_bits: int = 1) -> TagTexture:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("opencv-contrib-python with cv2.aruco is required") from exc
    if not hasattr(cv2, "aruco"):
        raise RuntimeError("the installed OpenCV build does not include cv2.aruco")
    marker_pixels = int(marker_pixels)
    margin_pixels = int(margin_pixels)
    marker_id = int(marker_id)
    if marker_pixels < 16 or margin_pixels < 0:
        raise ValueError("marker_pixels must be >=16 and margin_pixels non-negative")
    dictionary = _aruco_dictionary(cv2, family)
    count = len(dictionary.bytesList)
    if marker_id < 0 or marker_id >= count:
        raise ValueError("marker id {} outside dictionary range [0,{})".format(marker_id, count))
    aruco = cv2.aruco
    if hasattr(aruco, "generateImageMarker"):
        marker = aruco.generateImageMarker(dictionary, marker_id, marker_pixels, borderBits=border_bits)
    else:
        marker = np.empty((marker_pixels, marker_pixels), dtype=np.uint8)
        aruco.drawMarker(dictionary, marker_id, marker_pixels, marker, border_bits)
    side = marker_pixels + 2 * margin_pixels
    canvas = np.full((side, side), 255, dtype=np.uint8)
    canvas[margin_pixels:margin_pixels + marker_pixels,
           margin_pixels:margin_pixels + marker_pixels] = marker
    rgb = np.repeat(canvas[:, :, None], 3, axis=2)
    return TagTexture(rgb, marker_pixels, margin_pixels, family, marker_id)
