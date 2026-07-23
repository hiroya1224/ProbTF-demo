"""Deterministic, visibility-first AprilTag scene sampling."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple
import zlib

import numpy as np

from .camera_model import CameraModel
from .coordinates import (invert_transform, look_at_opencv, make_transform,
                          marker_corners_ippe, normalize, projected_edge_lengths,
                          transform_points)


SCENARIOS = ("frontal", "moderate", "oblique", "small", "occluded", "multi_tag")


@dataclass(frozen=True)
class TagSpec:
    family: str
    marker_id: int
    instance_id: int
    size_m: float
    T_W_M: np.ndarray


@dataclass(frozen=True)
class OccluderSpec:
    width_m: float
    height_m: float
    T_W_O: np.ndarray
    color_rgb: Tuple[int, int, int] = (128, 128, 128)


@dataclass(frozen=True)
class SceneSample:
    frame_id: int
    scenario: str
    seed: int
    T_W_C: np.ndarray
    tags: List[TagSpec]
    occluders: List[OccluderSpec]


class SceneSampler:
    """Sample in the optical camera frame, reject, then transform to world."""

    def __init__(self, camera: CameraModel, config: Dict[str, Any], seed: int = 0):
        self.camera = camera
        self.config = config
        self.seed = int(seed)
        self.tags_config = config.get("tags", {})
        self.sampling = config.get("sampling", {})

    def _rng(self, frame_id: int, scenario: str, salt: int = 0) -> np.random.RandomState:
        scenario_code = zlib.crc32(scenario.encode("utf-8")) & 0xFFFFFFFF
        value = (self.seed * 2654435761 + int(frame_id) * 2246822519 + scenario_code + salt) & 0xFFFFFFFF
        return np.random.RandomState(value)

    @staticmethod
    def _incidence_range(scenario: str) -> Tuple[float, float]:
        if scenario == "frontal":
            return 0.0, 10.0
        if scenario == "oblique":
            return 45.0, 70.0
        return 10.0, 45.0

    @staticmethod
    def _facing_rotation(center_C: np.ndarray, incidence_deg: float,
                         azimuth: float, roll: float) -> np.ndarray:
        toward_camera = normalize(-center_C)
        reference = np.array([0.0, -1.0, 0.0])
        tangent_x = np.cross(reference, toward_camera)
        if np.linalg.norm(tangent_x) < 1e-8:
            tangent_x = np.cross((1.0, 0.0, 0.0), toward_camera)
        tangent_x = normalize(tangent_x)
        tangent_y = normalize(np.cross(toward_camera, tangent_x))
        tilt_direction = np.cos(azimuth) * tangent_x + np.sin(azimuth) * tangent_y
        angle = np.deg2rad(incidence_deg)
        normal = normalize(np.cos(angle) * toward_camera + np.sin(angle) * tilt_direction)
        x_zero = np.cross(reference, normal)
        if np.linalg.norm(x_zero) < 1e-8:
            x_zero = np.cross((1.0, 0.0, 0.0), normal)
        x_zero = normalize(x_zero)
        y_zero = normalize(np.cross(normal, x_zero))
        x_axis = np.cos(roll) * x_zero + np.sin(roll) * y_zero
        y_axis = -np.sin(roll) * x_zero + np.cos(roll) * y_zero
        return np.column_stack((x_axis, y_axis, normal))

    def _valid_projection(self, T_C_M: np.ndarray, size_m: float,
                          scenario: str) -> Optional[np.ndarray]:
        corners_C = transform_points(T_C_M, marker_corners_ippe(size_m))
        if np.any(corners_C[:, 2] <= 0.0):
            return None
        corners = self.camera.project(corners_C)
        margin = float(self.sampling.get("image_margin_px", 30))
        if (np.any(corners[:, 0] < margin) or
                np.any(corners[:, 0] >= self.camera.width - margin) or
                np.any(corners[:, 1] < margin) or
                np.any(corners[:, 1] >= self.camera.height - margin)):
            return None
        mean_edge = float(np.mean(projected_edge_lengths(corners)))
        edge_min = float(self.sampling.get("projected_edge_min_px", 24))
        edge_max = float(self.sampling.get("projected_edge_max_px", 400))
        if not edge_min <= mean_edge <= edge_max:
            return None
        if scenario == "small" and mean_edge > float(self.sampling.get("small_edge_max_px", 60)):
            return None
        return corners

    @staticmethod
    def _bbox_overlap(candidate: np.ndarray, existing: Sequence[np.ndarray]) -> bool:
        lo = candidate.min(axis=0)
        hi = candidate.max(axis=0)
        area = max(1.0, float(np.prod(hi - lo)))
        for other in existing:
            other_lo = other.min(axis=0)
            other_hi = other.max(axis=0)
            intersection = np.maximum(0.0, np.minimum(hi, other_hi) - np.maximum(lo, other_lo))
            if float(np.prod(intersection)) / area > 0.15:
                return True
        return False

    def sample(self, scenario: str = "frontal", frame_id: int = 0,
               count: Optional[int] = None,
               T_W_C: Optional[np.ndarray] = None) -> SceneSample:
        if scenario not in SCENARIOS:
            raise ValueError("scenario must be one of {}".format(", ".join(SCENARIOS)))
        T_W_C = np.eye(4) if T_W_C is None else np.asarray(T_W_C, dtype=np.float64)
        rng = self._rng(frame_id, scenario)
        if count is None:
            if scenario == "multi_tag":
                low = max(3, int(self.tags_config.get("count_min", 1)))
                high = min(5, int(self.tags_config.get("count_max", 6)))
                count = int(rng.randint(low, high + 1))
            else:
                count = 1
        if count < 1:
            raise ValueError("tag count must be positive")
        family = str(self.tags_config.get("family", "DICT_APRILTAG_36h11"))
        size_m = float(self.tags_config.get("size_m", 0.12))
        marker_ids = rng.choice(np.arange(100), size=count, replace=False)
        depth_range = self.sampling.get("depth_m", [0.4, 2.5])
        min_depth, max_depth = float(depth_range[0]), float(depth_range[1])
        max_attempts = int(self.sampling.get("max_attempts", 1000))
        incidence_low, incidence_high = self._incidence_range(scenario)
        tags = []
        projected = []
        for index in range(count):
            for _ in range(max_attempts):
                if scenario == "small":
                    approximate_min = self.camera.fx * size_m / float(self.sampling.get("small_edge_max_px", 60))
                    depth = rng.uniform(max(min_depth, approximate_min), max_depth)
                else:
                    depth = rng.uniform(min_depth, max_depth)
                margin = float(self.sampling.get("image_margin_px", 30)) + 35.0
                pixel = np.array([
                    rng.uniform(margin, self.camera.width - margin),
                    rng.uniform(margin, self.camera.height - margin),
                ])
                center_C = self.camera.unproject(pixel[None, :], [depth])[0]
                incidence = rng.uniform(incidence_low, incidence_high)
                rotation = self._facing_rotation(
                    center_C, incidence, rng.uniform(-np.pi, np.pi),
                    rng.uniform(-np.pi, np.pi),
                )
                T_C_M = make_transform(rotation, center_C)
                corners = self._valid_projection(T_C_M, size_m, scenario)
                if corners is None or (scenario == "multi_tag" and self._bbox_overlap(corners, projected)):
                    continue
                tags.append(TagSpec(
                    family, int(marker_ids[index]), index + 1, size_m,
                    T_W_C.dot(T_C_M),
                ))
                projected.append(corners)
                break
            else:
                raise RuntimeError("could not sample a valid {} tag after {} attempts".format(scenario, max_attempts))
        occluders = []
        if scenario == "occluded":
            target = tags[0]
            offset = np.array([0.28 * size_m, 0.0, 0.005])
            T_W_O = target.T_W_M.copy()
            T_W_O[:3, 3] += target.T_W_M[:3, :3].dot(offset)
            occluders.append(OccluderSpec(0.55 * size_m, 1.20 * size_m, T_W_O))
        return SceneSample(int(frame_id), scenario, self.seed, T_W_C.copy(), tags, occluders)

    def sample_sequence(self, scenario: str, frames: int,
                        count: Optional[int] = None) -> List[SceneSample]:
        frames = int(frames)
        if frames < 1:
            raise ValueError("frames must be positive")
        base = self.sample(scenario, frame_id=0, count=count, T_W_C=np.eye(4))
        target = np.mean([tag.T_W_M[:3, 3] for tag in base.tags], axis=0)
        travel = np.asarray(self.config.get("sequence", {}).get("travel_m", [0.08, 0.03, 0.0]), dtype=np.float64)
        samples = []
        for frame_id in range(frames):
            alpha = 0.0 if frames == 1 else float(frame_id) / (frames - 1) - 0.5
            position = alpha * travel
            T_W_C = look_at_opencv(position, target)
            samples.append(SceneSample(
                frame_id, scenario, self.seed, T_W_C,
                base.tags, base.occluders,
            ))
        return samples
