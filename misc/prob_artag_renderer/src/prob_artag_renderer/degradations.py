"""Independently switchable, deterministic image degradations."""

from typing import Any, Dict, Optional, Tuple

import numpy as np


def enabled_radial_coefficients(config: Dict[str, Any]) -> Optional[np.ndarray]:
    item = config.get("radial_distortion", {})
    if item.get("enabled", False):
        return np.asarray(item.get("coefficients", [-0.12, 0.03, 0, 0, 0]), dtype=np.float64)
    return None


def _cv2():
    try:
        import cv2
        return cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required for the selected degradation") from exc


def _remap(image: np.ndarray, map_x: np.ndarray, map_y: np.ndarray, nearest: bool) -> np.ndarray:
    cv2 = _cv2()
    interpolation = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
    return cv2.remap(image, map_x, map_y, interpolation, borderMode=cv2.BORDER_CONSTANT)


def radial_distort(image: np.ndarray, camera_matrix: np.ndarray,
                   coefficients: np.ndarray, nearest: bool = False) -> np.ndarray:
    cv2 = _cv2()
    height, width = image.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(width, dtype=np.float32),
                                 np.arange(height, dtype=np.float32))
    distorted_pixels = np.column_stack((grid_x.ravel(), grid_y.ravel())).reshape(-1, 1, 2)
    source = cv2.undistortPoints(
        distorted_pixels, camera_matrix, coefficients, P=camera_matrix
    ).reshape(height, width, 2).astype(np.float32)
    return _remap(image, source[:, :, 0], source[:, :, 1], nearest)


def _motion_kernel(size: int, angle_deg: float) -> np.ndarray:
    cv2 = _cv2()
    size = max(1, int(size))
    if size % 2 == 0:
        size += 1
    kernel = np.zeros((size, size), dtype=np.float32)
    kernel[size // 2, :] = 1.0
    matrix = cv2.getRotationMatrix2D((size / 2.0 - 0.5, size / 2.0 - 0.5), angle_deg, 1.0)
    kernel = cv2.warpAffine(kernel, matrix, (size, size))
    total = float(kernel.sum())
    return kernel / total if total else kernel


def apply_degradations(rgb: np.ndarray, depth: np.ndarray, instance_id: np.ndarray,
                       config: Dict[str, Any], rng: np.random.RandomState,
                       camera_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rgb = np.asarray(rgb, dtype=np.uint8).copy()
    depth = np.asarray(depth, dtype=np.float32).copy()
    instance_id = np.asarray(instance_id, dtype=np.uint16).copy()
    radial = enabled_radial_coefficients(config)
    if radial is not None:
        rgb = radial_distort(rgb, camera_matrix, radial, nearest=False)
        depth = radial_distort(depth, camera_matrix, radial, nearest=True)
        instance_id = radial_distort(instance_id, camera_matrix, radial, nearest=True)
    item = config.get("background_clutter", {})
    if item.get("enabled", False):
        cv2 = _cv2()
        for _ in range(int(item.get("count", 20))):
            center = (rng.randint(0, rgb.shape[1]), rng.randint(0, rgb.shape[0]))
            radius = rng.randint(3, max(4, min(rgb.shape[:2]) // 12))
            color = tuple(int(value) for value in rng.randint(0, 256, size=3))
            clutter_mask = instance_id == 0
            layer = rgb.copy()
            cv2.circle(layer, center, radius, color, -1)
            rgb[clutter_mask] = layer[clutter_mask]
    item = config.get("brightness_contrast", {})
    if item.get("enabled", False):
        rgb = np.clip(
            float(item.get("contrast", 1.0)) * rgb.astype(np.float32) +
            float(item.get("brightness", 0.0)), 0, 255,
        ).astype(np.uint8)
    item = config.get("gaussian_noise", {})
    if item.get("enabled", False):
        noise = rng.normal(0.0, float(item.get("stddev", 5.0)), rgb.shape)
        rgb = np.clip(rgb.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    item = config.get("defocus_blur", {})
    if item.get("enabled", False):
        kernel = max(1, int(item.get("kernel_size", 5)))
        kernel += 1 - kernel % 2
        rgb = _cv2().GaussianBlur(rgb, (kernel, kernel), 0)
    item = config.get("motion_blur", {})
    if item.get("enabled", False):
        kernel = _motion_kernel(int(item.get("kernel_size", 9)), float(item.get("angle_deg", 0.0)))
        rgb = _cv2().filter2D(rgb, -1, kernel)
    item = config.get("partial_occlusion", {})
    if item.get("enabled", False):
        fraction = float(np.clip(item.get("fraction", 0.2), 0.0, 0.95))
        width = max(1, int(round(rgb.shape[1] * fraction)))
        start = int(rng.randint(0, max(1, rgb.shape[1] - width + 1)))
        color = np.asarray(item.get("color", [128, 128, 128]), dtype=np.uint8)
        rgb[:, start:start + width] = color
        depth[:, start:start + width] = 0.0
        instance_id[:, start:start + width] = 0
    item = config.get("rolling_shutter", {})
    if item.get("enabled", False):
        cv2 = _cv2()
        height, width = rgb.shape[:2]
        max_shift = float(item.get("max_shift_px", 8.0))
        grid_x, grid_y = np.meshgrid(np.arange(width, dtype=np.float32),
                                     np.arange(height, dtype=np.float32))
        shift = max_shift * (grid_y / max(1.0, height - 1.0) - 0.5)
        map_x = grid_x - shift
        rgb = _remap(rgb, map_x, grid_y, nearest=False)
        depth = _remap(depth, map_x, grid_y, nearest=True)
        instance_id = _remap(instance_id, map_x, grid_y, nearest=True)
    return rgb, depth, instance_id
