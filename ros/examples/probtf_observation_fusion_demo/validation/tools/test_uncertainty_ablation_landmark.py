"""The known-landmark intervention changes the map prior, not body laws."""
import numpy as np

from uncertainty_ablation_landmark import install_known_landmark_patch
from probtf_observation_fusion_demo.moving_scene import MovingSwitchScene
from probtf_observation_fusion_demo.moving_inference import MovingInference


def test_known_map_initialization_preserves_body_and_restores_patch(tmp_path):
    scene=MovingSwitchScene(tmp_path/'scene')
    before=MovingInference(scene.robot_description,())
    ordinary=before.prior_at(scene.initial_command,1792080000.,0)
    restore=install_known_landmark_patch(scene.marker)
    try:
        diagnostic=MovingInference(scene.robot_description,())
        known=diagnostic.prior_at(scene.initial_command,1792080000.,0)
        assert np.array_equal(known.covariance[:-6,:-6],ordinary.covariance[:-6,:-6])
        for lhs,rhs in zip(known.transforms[:-1],ordinary.transforms[:-1]):
            assert np.array_equal(lhs.translation,rhs.translation)
            if lhs.bingham is not None:assert np.array_equal(lhs.bingham,rhs.bingham)
        assert np.array_equal(known.transforms[-1].translation,scene.marker[:3,3])
        assert np.allclose(known.covariance[-6:,-6:],np.diag([1e-8]*3+[1e-6]*3),rtol=1e-12,atol=1e-20)
        assert diagnostic.common['prior']['marker']['status'].startswith('authorized diagnostic')
    finally:
        restore()
    restored=MovingInference(scene.robot_description,()).prior_at(scene.initial_command,1792080000.,0)
    assert restored.state_id==ordinary.state_id
