#!/usr/bin/env python3

"""Regenerate the committed DICT_APRILTAG_36h11 sample PNG files."""

from pathlib import Path

import cv2
import numpy as np


FAMILY_NAME = "DICT_APRILTAG_36h11"
SAMPLE_IDS = (0, 7, 17, 21)
MARKER_SIDE_PX = 1600
QUIET_ZONE_MODULES = 1


def _marker_image(dictionary, marker_id):
    if hasattr(cv2.aruco, "generateImageMarker"):
        return cv2.aruco.generateImageMarker(
            dictionary, marker_id, MARKER_SIDE_PX
        )
    return cv2.aruco.drawMarker(dictionary, marker_id, MARKER_SIDE_PX)


def render_sample(dictionary, marker_id):
    total_marker_modules = int(dictionary.markerSize) + 2
    if MARKER_SIDE_PX % total_marker_modules:
        raise ValueError("MARKER_SIDE_PX must be divisible by the marker grid size.")
    quiet_zone_px = (
        MARKER_SIDE_PX // total_marker_modules * QUIET_ZONE_MODULES
    )
    marker = _marker_image(dictionary, marker_id)
    canvas_side = MARKER_SIDE_PX + 2 * quiet_zone_px
    canvas = np.full((canvas_side, canvas_side), 255, dtype=np.uint8)
    canvas[
        quiet_zone_px : quiet_zone_px + MARKER_SIDE_PX,
        quiet_zone_px : quiet_zone_px + MARKER_SIDE_PX,
    ] = marker
    return canvas


def main():
    output_directory = Path(__file__).resolve().parent
    family = getattr(cv2.aruco, FAMILY_NAME)
    dictionary = cv2.aruco.getPredefinedDictionary(family)
    marker_count = int(dictionary.bytesList.shape[0])
    for marker_id in SAMPLE_IDS:
        if not 0 <= marker_id < marker_count:
            raise ValueError(
                "Marker ID {} is outside {} ({} entries).".format(
                    marker_id, FAMILY_NAME, marker_count
                )
            )
        output_path = output_directory / (
            "apriltag_36h11_id_{:03d}.png".format(marker_id)
        )
        if not cv2.imwrite(str(output_path), render_sample(dictionary, marker_id)):
            raise OSError("OpenCV could not write {}.".format(output_path))
        print(output_path)


if __name__ == "__main__":
    main()
