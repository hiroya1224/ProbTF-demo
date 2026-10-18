"""Historical image truth cannot construct or patch a new observation."""
import pytest

from probtf_observation_fusion_demo import sensors, moment_observation
from uncertainty_ablation_observations import (
    install_oracle_marker_patch, prepare_oracle_marker_observation,
)
from uncertainty_ablation_policy import TruthInjectionRevoked


def test_direct_observation_creation_is_blocked_before_consuming_truth():
    with pytest.raises(TruthInjectionRevoked, match='revoked.*2026-10-19'):
        prepare_oracle_marker_observation(None, None, None, camera_pose=None,
            marker_pose=None, observation_id=None, command=None, capture_stamp=None)


@pytest.mark.parametrize('require_detected', (True, False))
def test_patch_is_blocked_without_mutating_sensor_imports(tmp_path, require_detected):
    original = sensors.prepare_marker_observation
    original_moment = moment_observation.prepare_marker_observation
    output = tmp_path/'never-created'
    with pytest.raises(TruthInjectionRevoked):
        install_oracle_marker_patch(output, require_detected=require_detected)
    assert sensors.prepare_marker_observation is original
    assert moment_observation.prepare_marker_observation is original_moment
    assert not output.exists()
