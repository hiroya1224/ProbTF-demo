"""Explicit paired legacy configuration for existing single-tag mode tests.

Both renderer and estimator disable the board. This preserves the original
native two-mode observation tests without changing the new moving defaults.
"""
from pathlib import Path

import yaml

from probtf_observation_fusion_demo.moving_scene import MovingSwitchScene


PACKAGE = Path(__file__).resolve().parents[1]


def legacy_marker_scene(output, **kwargs):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    scene_settings = yaml.safe_load((PACKAGE/'config/switch_press_scene.yaml').read_text())
    inference_settings = yaml.safe_load((PACKAGE/'config/moving_estimation.yaml').read_text())
    for settings in (scene_settings, inference_settings):
        settings.setdefault('marker_board', {})['enabled'] = False
    scene_path = output/'legacy_single_marker_scene.yaml'
    inference_path = output/'legacy_single_marker_inference.yaml'
    scene_path.write_text(yaml.safe_dump(scene_settings))
    inference_path.write_text(yaml.safe_dump(inference_settings))
    scene = MovingSwitchScene(output, config_path=scene_path, **kwargs)
    scene.legacy_inference_config = inference_path
    return scene
