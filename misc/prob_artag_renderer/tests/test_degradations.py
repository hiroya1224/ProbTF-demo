from copy import deepcopy

import numpy as np

from prob_artag_renderer.camera_model import CameraModel
from prob_artag_renderer.config import load_config
from prob_artag_renderer.degradations import apply_degradations


def _fixture():
    yy, xx = np.mgrid[:48, :64]
    rgb = np.stack((xx * 4, yy * 5, (xx + yy) * 2), axis=2).astype(np.uint8)
    depth = np.full((48, 64), 1.25, dtype=np.float32)
    instances = np.zeros((48, 64), dtype=np.uint16)
    instances[12:36, 16:48] = 7
    return rgb, depth, instances


def _apply(config, seed=3):
    rgb, depth, instances = _fixture()
    return apply_degradations(rgb, depth, instances, config,
                              np.random.RandomState(seed), CameraModel(width=64, height=48, fx=55, fy=55, cx=32, cy=24).matrix)


def test_all_disabled_is_identity():
    config = load_config()["degradations"]
    actual = _apply(config)
    for result, expected in zip(actual, _fixture()):
        np.testing.assert_array_equal(result, expected)


def test_each_degradation_is_independently_switchable_and_deterministic():
    base = load_config()["degradations"]
    original = _fixture()
    for name in base:
        config = deepcopy(base)
        config[name]["enabled"] = True
        first = _apply(config, seed=19)
        second = _apply(config, seed=19)
        for a, b in zip(first, second):
            np.testing.assert_array_equal(a, b)
        assert any(not np.array_equal(a, b) for a, b in zip(first, original)), name
