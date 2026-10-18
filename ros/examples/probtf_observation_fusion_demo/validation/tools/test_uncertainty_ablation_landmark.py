"""A known physical landmark can no longer be injected into the map prior."""
import pytest

from uncertainty_ablation_landmark import install_known_landmark_patch
from uncertainty_ablation_policy import TruthInjectionRevoked
from probtf_observation_fusion_demo.moving_inference import MovingInference


def test_known_map_patch_is_blocked_without_changing_inference_class():
    original_init = MovingInference.__init__
    original_update = MovingInference.update_sensor_event
    with pytest.raises(TruthInjectionRevoked, match='revoked.*2026-10-19'):
        install_known_landmark_patch(None)
    assert MovingInference.__init__ is original_init
    assert MovingInference.update_sensor_event is original_update
