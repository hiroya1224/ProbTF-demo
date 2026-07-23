"""Exact frame annotations computed independently from rendered pixels."""

from dataclasses import dataclass, replace
from typing import Any, Dict, List

import numpy as np

from .camera_model import CameraModel
from .coordinates import invert_transform, marker_corners_ippe, projected_edge_lengths, transform_points


@dataclass(frozen=True)
class TagAnnotation:
    family: str
    marker_id: int
    instance_id: int
    size_m: float
    T_W_M: np.ndarray
    T_C_M: np.ndarray
    corners_px: np.ndarray
    corners_depth_m: np.ndarray
    front_facing: bool
    visible_fraction: float
    projected_size_px: float

    def with_visible_fraction(self, value: float) -> "TagAnnotation":
        return replace(self, visible_fraction=float(np.clip(value, 0.0, 1.0)))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "family": self.family, "id": self.marker_id,
            "instance_id": self.instance_id, "size_m": self.size_m,
            "T_W_M": self.T_W_M.tolist(), "T_C_M": self.T_C_M.tolist(),
            "corners_px": self.corners_px.tolist(),
            "corners_depth_m": self.corners_depth_m.tolist(),
            "front_facing": self.front_facing,
            "visible_fraction": self.visible_fraction,
            "projected_size_px": self.projected_size_px,
            "corner_order": "IPPE_SQUARE_TL_TR_BR_BL",
        }


@dataclass(frozen=True)
class FrameAnnotation:
    frame_id: int
    scenario: str
    seed: int
    camera: CameraModel
    T_W_C: np.ndarray
    tags: List[TagAnnotation]
    degradations: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        camera = self.camera.to_dict()
        camera["T_W_C"] = self.T_W_C.tolist()
        return {
            "schema_version": 1, "frame_id": self.frame_id,
            "scenario": self.scenario, "seed": self.seed,
            "conventions": {
                "transform": "T_A_B maps B coordinates to A",
                "camera_frame": "OpenCV optical: x-right y-down z-forward",
                "tag_frame": "x-right y-up z-front-normal",
                "length_unit": "m", "image_unit": "pixel",
            },
            "camera": camera, "tags": [tag.to_dict() for tag in self.tags],
            "degradations": self.degradations,
        }


def annotate_tags(frame_id: int, scenario: str, seed: int, camera: CameraModel,
                  T_W_C: np.ndarray, tags: List[Any],
                  degradations: Dict[str, Any]) -> FrameAnnotation:
    T_C_W = invert_transform(T_W_C)
    annotations = []
    for tag in tags:
        T_C_M = T_C_W.dot(tag.T_W_M)
        corners_C = transform_points(T_C_M, marker_corners_ippe(tag.size_m))
        corners_px = camera.project(corners_C)
        normal_C = T_C_M[:3, 2]
        to_camera = -T_C_M[:3, 3]
        front_facing = bool(np.dot(normal_C, to_camera) > 0.0)
        annotations.append(TagAnnotation(
            tag.family, tag.marker_id, tag.instance_id, tag.size_m,
            tag.T_W_M.copy(), T_C_M, corners_px, corners_C[:, 2].copy(),
            front_facing, 1.0, float(np.mean(projected_edge_lengths(corners_px))),
        ))
    return FrameAnnotation(
        int(frame_id), scenario, int(seed), camera, np.asarray(T_W_C).copy(),
        annotations, degradations,
    )


def update_visibility(annotation: FrameAnnotation, instance_image: np.ndarray) -> FrameAnnotation:
    try:
        import cv2
    except ImportError:
        return annotation
    updated = []
    height, width = instance_image.shape[:2]
    for tag in annotation.tags:
        expected = np.zeros((height, width), dtype=np.uint8)
        polygon = np.rint(tag.corners_px).astype(np.int32)
        cv2.fillConvexPoly(expected, polygon, 1)
        expected_count = int(np.count_nonzero(expected))
        visible_count = int(np.count_nonzero((instance_image == tag.instance_id) & (expected != 0)))
        fraction = 0.0 if expected_count == 0 else float(visible_count) / expected_count
        updated.append(tag.with_visible_fraction(fraction))
    return replace(annotation, tags=updated)
