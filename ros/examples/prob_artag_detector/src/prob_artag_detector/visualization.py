"""Small OpenCV-only debug overlays; never part of the estimation input."""

import cv2
import numpy as np


def draw_debug_image(
    image,
    observations,
    results=(),
    camera_model=None,
    axis_length=0.04,
    status_text="",
    corner_diagnostics=(),
):
    value = np.asarray(image)
    if value.ndim == 2:
        output = cv2.cvtColor(value, cv2.COLOR_GRAY2BGR)
    elif value.ndim == 3 and value.shape[2] == 3:
        output = value.copy()
    elif value.ndim == 3 and value.shape[2] == 4:
        output = cv2.cvtColor(value, cv2.COLOR_BGRA2BGR)
    else:
        raise ValueError("image must be grayscale, BGR, or BGRA.")
    result_by_id = {
        observation.marker_id: result
        for observation, result in zip(observations, results)
        if result is not None
    }
    diagnostic_by_id = {
        observation.marker_id: diagnostic
        for observation, diagnostic in zip(observations, corner_diagnostics)
        if diagnostic is not None
    }
    if status_text:
        cv2.putText(
            output,
            str(status_text),
            (12, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (20, 220, 250),
            2,
            cv2.LINE_AA,
        )
    if camera_model is not None:
        length = float(axis_length)
        if not np.isfinite(length) or length <= 0.0:
            raise ValueError("axis_length must be positive and finite.")
        for result in results:
            if result is None:
                continue
            for rotation, translation in zip(result.rotations, result.translations):
                rotation_vector, _ = cv2.Rodrigues(np.asarray(rotation, dtype=float))
                cv2.drawFrameAxes(
                    output,
                    camera_model.camera_matrix,
                    camera_model.distortion,
                    rotation_vector,
                    np.asarray(translation, dtype=float).reshape(3, 1),
                    length,
                    2,
                )
    for observation in observations:
        corners = np.rint(observation.corners_px).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(output, [corners], True, (0, 220, 0), 2, cv2.LINE_AA)
        origin = tuple(corners[0, 0])
        covariance = np.asarray(observation.image_covariance, dtype=float)
        equivalent_sigma_px = np.sqrt(max(0.0, float(np.trace(covariance))) / 8.0)
        result = result_by_id.get(observation.marker_id)
        suffix = " sigma={:.2f}px".format(equivalent_sigma_px)
        diagnostic = diagnostic_by_id.get(observation.marker_id)
        if diagnostic is not None:
            suffix += " temporal={}".format(diagnostic.status)
        if result is not None:
            suffix += " modes={} w={}".format(
                result.diagnostics.accepted_count,
                ",".join("{:.2f}".format(weight) for weight in result.weights),
            )
        cv2.putText(
            output,
            "id={}{}".format(observation.marker_id, suffix),
            origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (30, 30, 230),
            1,
            cv2.LINE_AA,
        )
    return output
