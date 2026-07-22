"""Headless-safe lazy pyrender scene renderer."""

from dataclasses import dataclass
import os
from pathlib import Path
import sys
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .annotations import FrameAnnotation, annotate_tags, update_visibility
from .camera_model import CameraModel
from .coordinates import opencv_camera_pose_to_pyrender
from .degradations import apply_degradations, enabled_radial_coefficients
from .scene_sampler import SceneSample
from .tag_mesh import create_colored_quad, create_tag_mesh
from .tag_texture import generate_tag_texture


@dataclass(frozen=True)
class RenderedFrame:
    rgb: np.ndarray
    depth_m: np.ndarray
    instance_id: np.ndarray
    annotation: FrameAnnotation


_MESA_EGL_VENDOR = Path("/usr/share/glvnd/egl_vendor.d/50_mesa.json")


def _configure_backend_environment(backend: Optional[str]) -> None:
    if backend and "pyrender" not in sys.modules and "OpenGL" not in sys.modules:
        os.environ.setdefault("PYOPENGL_PLATFORM", backend)
    if (backend == "egl" and "__EGL_VENDOR_LIBRARY_FILENAMES" not in os.environ
            and _MESA_EGL_VENDOR.exists()):
        os.environ["__EGL_VENDOR_LIBRARY_FILENAMES"] = str(_MESA_EGL_VENDOR)


def _load_pyrender(backend: Optional[str]):
    _configure_backend_environment(backend)
    try:
        import pyrender
        return pyrender
    except Exception as exc:
        raise RuntimeError(
            "pyrender could not initialize; set render.backend to egl or osmesa"
        ) from exc


def _instance_color(instance_id: int) -> np.ndarray:
    value = int(instance_id)
    if value <= 0 or value >= (1 << 24):
        raise ValueError("instance id must fit in non-zero RGB24")
    return np.array([value & 255, (value >> 8) & 255, (value >> 16) & 255], dtype=np.uint8)


def _decode_instances(image: np.ndarray) -> np.ndarray:
    value = image.astype(np.uint32)
    decoded = value[:, :, 0] + (value[:, :, 1] << 8) + (value[:, :, 2] << 16)
    if int(decoded.max()) > np.iinfo(np.uint16).max:
        raise ValueError("instance image exceeds uint16 dataset format")
    return decoded.astype(np.uint16)


class AprilTagRenderer:
    def __init__(self, camera: CameraModel, config: Dict[str, Any]):
        self.camera = camera
        self.config = config
        self.render_config = config.get("render", {})
        self.tag_config = config.get("tags", {})
        self.degradation_config = config.get("degradations", {})
        self._pyrender = None
        self._renderer = None
        self._tag_meshes = {}
        self._occluder_meshes = {}

    def _ensure_context(self):
        if self._pyrender is None:
            self._pyrender = _load_pyrender(self.render_config.get("backend", "egl"))
        if self._renderer is None:
            try:
                self._renderer = self._pyrender.OffscreenRenderer(
                    viewport_width=self.camera.width, viewport_height=self.camera.height
                )
            except Exception as exc:
                backend = self.render_config.get("backend", "egl")
                raise RuntimeError(
                    "pyrender backend {!r} could not create an offscreen context".format(backend)
                ) from exc

    def _tag_mesh(self, family: str, marker_id: int, size_m: float):
        key = (family, int(marker_id), float(size_m))
        if key not in self._tag_meshes:
            texture = generate_tag_texture(
                marker_id, family,
                int(self.tag_config.get("marker_pixels", 256)),
                int(self.tag_config.get("margin_pixels", 32)),
            )
            self._tag_meshes[key] = create_tag_mesh(texture, size_m)
        return self._tag_meshes[key]

    def _occluder_mesh(self, width_m: float, height_m: float,
                       color_rgb: Tuple[int, int, int]):
        key = (float(width_m), float(height_m), tuple(color_rgb))
        if key not in self._occluder_meshes:
            self._occluder_meshes[key] = create_colored_quad(width_m, height_m, color_rgb)
        return self._occluder_meshes[key]

    def _effective_camera(self) -> CameraModel:
        coefficients = enabled_radial_coefficients(self.degradation_config)
        return self.camera if coefficients is None else self.camera.with_distortion(coefficients)

    def render(self, sample: SceneSample) -> RenderedFrame:
        self._ensure_context()
        pyrender = self._pyrender
        background = np.asarray(self.render_config.get("background", [0.5, 0.5, 0.5]), dtype=float)
        if background.size == 3:
            background = np.append(background, 1.0)
        scene = pyrender.Scene(
            bg_color=background,
            ambient_light=np.asarray(self.render_config.get("ambient_light", [1, 1, 1]), dtype=float),
        )
        camera = pyrender.IntrinsicsCamera(
            self.camera.fx, self.camera.fy, self.camera.cx, self.camera.cy,
            znear=float(self.render_config.get("near", 0.05)),
            zfar=float(self.render_config.get("far", 10.0)),
        )
        scene.add(camera, pose=opencv_camera_pose_to_pyrender(sample.T_W_C), name="camera")
        segmentation = {}
        for tag in sample.tags:
            node = scene.add(
                self._tag_mesh(tag.family, tag.marker_id, tag.size_m),
                pose=tag.T_W_M, name="tag_{}".format(tag.marker_id),
            )
            segmentation[node] = _instance_color(tag.instance_id)
        for index, occluder in enumerate(sample.occluders):
            node = scene.add(
                self._occluder_mesh(occluder.width_m, occluder.height_m, occluder.color_rgb),
                pose=occluder.T_W_O, name="occluder_{}".format(index),
            )
            segmentation[node] = np.zeros(3, dtype=np.uint8)
        rgb, depth = self._renderer.render(scene)
        segment_rgb, _ = self._renderer.render(
            scene, flags=pyrender.RenderFlags.SEG, seg_node_map=segmentation
        )
        instances = _decode_instances(segment_rgb[:, :, :3])
        rng_seed = (int(sample.seed) * 1103515245 + int(sample.frame_id) * 12345 + 0xA5A5) & 0xFFFFFFFF
        rgb, depth, instances = apply_degradations(
            rgb[:, :, :3], depth, instances, self.degradation_config,
            np.random.RandomState(rng_seed), self.camera.matrix,
        )
        effective_camera = self._effective_camera()
        annotation = annotate_tags(
            sample.frame_id, sample.scenario, sample.seed, effective_camera,
            sample.T_W_C, sample.tags, self.degradation_config,
        )
        annotation = update_visibility(annotation, instances)
        return RenderedFrame(rgb, depth.astype(np.float32), instances, annotation)

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.delete()
            self._renderer = None

    def __enter__(self) -> "AprilTagRenderer":
        self._ensure_context()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
