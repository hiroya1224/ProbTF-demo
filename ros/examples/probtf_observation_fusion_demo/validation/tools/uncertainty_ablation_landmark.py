"""Withdrawn landmark truth patch, retained as historical source only."""
import numpy as np

from uncertainty_ablation_policy import reject_truth_injection


def install_known_landmark_patch(marker_pose,translation_sd=1e-4,rotation_sd=1e-3):
    """Anchor the map prior; all ordinary observations/updates remain active.

    Unlike control-side oracle_goal, this knowledge can inform body inference
    through the NORMAL shared unknown-marker variable and its observations.
    The finite prior is never reset after an update and is not a repeated
    independent measurement. Caller must confine this monkeypatch to its own
    diagnostic worker process and retain its explicit truth provenance.
    """
    reject_truth_injection('known-landmark inference prior patch')
    from probtf.geometry import rotmat_to_quat
    from probtf.spherical_law.first_order import BinghamTransform,bingham_from_rotation_coordinates
    from probtf_observation_fusion_demo.moving_inference import MovingInference
    pose=np.array(marker_pose,dtype=float,copy=True)
    if pose.shape!=(4,4)or not np.all(np.isfinite(pose)):
        raise ValueError('Known diagnostic landmark requires a finite 4x4 root pose.')
    if translation_sd<=0.or rotation_sd<=0.:
        raise ValueError('Known diagnostic landmark must use explicitly finite positive widths.')
    original_init=MovingInference.__init__
    original_update=MovingInference.update_sensor_event
    provenance=dict(name='known_landmark',physical_truth_received=True,
        supplied_at_initialization_only=True,repeated_measurement_added=False,
        translation_sd_m=float(translation_sd),rotation_sd_rad=float(rotation_sd),
        root_marker_pose=pose.tolist(),body_prior_changed=False,
        control_goal_forced_to_truth=False)

    def initialize(self,*args,**kwargs):
        original_init(self,*args,**kwargs)
        quaternion=rotmat_to_quat(pose[:3,:3])
        self.marker=BinghamTransform(self.marker.edge_id,pose[:3,3],
            bingham_from_rotation_coordinates(quaternion,rotation_sd**2*np.eye(3)),
            translation_sd**2*np.eye(3))
        self.common['prior']['marker']['status']='authorized diagnostic known root landmark prior'

    def update(self,*args,**kwargs):
        result=original_update(self,*args,**kwargs)
        result.records['diagnostic_known_landmark']=dict(provenance)
        return result

    MovingInference.__init__=initialize
    MovingInference.update_sensor_event=update

    def restore():
        MovingInference.__init__=original_init
        MovingInference.update_sensor_event=original_update
    return restore
