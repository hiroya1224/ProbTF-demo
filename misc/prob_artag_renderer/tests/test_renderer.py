import os
import subprocess
import sys

import numpy as np
import pytest

from prob_artag_renderer.camera_model import CameraModel
from prob_artag_renderer.config import load_config
from prob_artag_renderer.renderer import (
    AprilTagRenderer, _MESA_EGL_VENDOR, _configure_backend_environment,
    _decode_instances, _instance_color,
)
from prob_artag_renderer.scene_sampler import SceneSampler


def test_renderer_module_import_is_lazy(tmp_path):
    source = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
    code = "import sys; import prob_artag_renderer.renderer; assert 'pyrender' not in sys.modules"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = source
    subprocess.check_call([sys.executable, "-c", code], env=environment)


def test_instance_rgb24_round_trip():
    image = np.zeros((1, 3, 3), dtype=np.uint8)
    for index, value in enumerate((1, 255, 4097)):
        image[0, index] = _instance_color(value)
    np.testing.assert_array_equal(_decode_instances(image), [[1, 255, 4097]])


def test_egl_vendor_defaults_to_mesa_but_preserves_override(monkeypatch):
    monkeypatch.delenv("__EGL_VENDOR_LIBRARY_FILENAMES", raising=False)
    _configure_backend_environment("egl")
    if _MESA_EGL_VENDOR.exists():
        assert os.environ["__EGL_VENDOR_LIBRARY_FILENAMES"] == str(_MESA_EGL_VENDOR)
    monkeypatch.setenv("__EGL_VENDOR_LIBRARY_FILENAMES", "/custom/vendor.json")
    _configure_backend_environment("egl")
    assert os.environ["__EGL_VENDOR_LIBRARY_FILENAMES"] == "/custom/vendor.json"


def test_optional_headless_render_smoke():
    config = load_config()
    camera = CameraModel.from_dict(config["camera"])
    sample = SceneSampler(camera, config, seed=5).sample("frontal")
    renderer = AprilTagRenderer(camera, config)
    try:
        try:
            frame = renderer.render(sample)
        except RuntimeError as exc:
            pytest.skip(str(exc))
        repeated = renderer.render(sample)
        np.testing.assert_array_equal(repeated.rgb, frame.rgb)
        np.testing.assert_array_equal(repeated.depth_m, frame.depth_m)
        np.testing.assert_array_equal(repeated.instance_id, frame.instance_id)
        assert repeated.annotation.to_dict() == frame.annotation.to_dict()
        assert frame.rgb.shape == (camera.height, camera.width, 3)
        assert frame.depth_m.shape == (camera.height, camera.width)
        assert frame.instance_id.max() == 1
        assert frame.annotation.tags[0].visible_fraction > 0.5
        pose = frame.annotation.tags[0].T_C_M
        assert np.all(np.abs(pose[:3, 3]) > 1e-6)
        # The fixed sampler seed produces both in-plane roll and out-of-plane
        # tilt, avoiding the axis-aligned case that can hide sign mistakes.
        assert np.count_nonzero(np.abs(pose[:3, :3]) > 1e-3) >= 7
        import cv2
        aruco = cv2.aruco
        dictionary = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36h11)
        gray = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2GRAY)
        if hasattr(aruco, "ArucoDetector"):
            corners, ids, _ = aruco.ArucoDetector(dictionary).detectMarkers(gray)
        else:
            corners, ids, _ = aruco.detectMarkers(gray, dictionary)
        assert ids is not None and ids.reshape(-1).tolist() == [sample.tags[0].marker_id]
        error = np.linalg.norm(
            corners[0].reshape(4, 2) - frame.annotation.tags[0].corners_px, axis=1
        )
        assert float(error.max()) <= 2.0
    finally:
        renderer.close()
