from typing import List, Optional, Sequence

import numpy as np
import pinocchio as pin
from scipy.spatial.transform import Rotation as Rsc

from deflecomp_core.robot.urdf_info import (
    infer_base_link,
    infer_imu_frames,
    infer_joint_types,
    infer_tip_link,
    load_urdf_model_info,
)

class RobotArm:
    def __init__(
        self,
        urdf_path: str,
        tip_link: Optional[str] = None,
        base_link: Optional[str] = None,
    ) -> None:
        self.urdf_path = urdf_path
        self.urdf_info = load_urdf_model_info(urdf_path)
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()
        self.nv = self.model.nv
        self.frame_names = {frame.name for frame in self.model.frames}
        self.model_joint_names = [self.model.names[idx] for idx in range(1, self.model.njoints)]
        self.model_joint_types = infer_joint_types(self.urdf_info, self.model_joint_names)
        self.base_link_name = self._resolve_frame_name(
            preferred=base_link,
            fallback=infer_base_link(self.urdf_info, preferred=base_link),
        )
        self.tip_link_name = self._resolve_frame_name(
            preferred=tip_link,
            fallback=infer_tip_link(self.urdf_info, preferred=tip_link),
        )
        self.tip_fid = self.model.getFrameId(self.tip_link_name)
        self.base_fid = self.model.getFrameId(self.base_link_name)
        if hasattr(self.model, 'gravity'):
            self.model.gravity.linear = np.array([0.0, 0.0, -9.81], dtype=float)
        self.total_mass = float(sum(inert.mass for inert in self.model.inertias))

    def _resolve_frame_name(self, preferred: Optional[str], fallback: str) -> str:
        if preferred and preferred in self.frame_names:
            return preferred
        if fallback in self.frame_names:
            return fallback
        if self.frame_names:
            return sorted(self.frame_names)[0]
        raise ValueError(f"No Pinocchio frames found for URDF: {self.urdf_path}")

    def has_frame(self, frame_name: str) -> bool:
        return frame_name in self.frame_names

    def suggest_imu_frames(self, preferred: Optional[Sequence[str]] = None, count: int = 3) -> List[str]:
        frames = infer_imu_frames(self.urdf_info, preferred=preferred, count=count)
        return [frame for frame in frames if frame in self.frame_names]

    def get_frame_id(self, frame_name: str) -> int:
        return self.model.getFrameId(frame_name)

    def _fk_update(self, theta: np.ndarray) -> None:
        pin.forwardKinematics(self.model, self.data, theta)
        pin.updateFramePlacements(self.model, self.data)

    def frame_rotation_in_base(self, theta: np.ndarray, fid: int) -> np.ndarray:
        self._fk_update(theta)
        R_wb = self.data.oMf[self.base_fid].rotation
        R_wf = self.data.oMf[fid].rotation
        return R_wb.T @ R_wf

    def frame_quaternion_wxyz_base(self, theta: np.ndarray, fid: int) -> np.ndarray:
        R_bf = self.frame_rotation_in_base(theta, fid)
        q_xyzw = Rsc.from_matrix(R_bf).as_quat()
        q_wxyz = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=float)
        n = np.linalg.norm(q_wxyz) + 1e-18
        return q_wxyz / n

    def frame_angular_jacobian_world(self, theta: np.ndarray, fid: int) -> np.ndarray:
        pin.computeJointJacobians(self.model, self.data, theta)
        pin.updateFramePlacements(self.model, self.data)
        J6 = pin.computeFrameJacobian(self.model, self.data, theta, fid, pin.ReferenceFrame.WORLD)
        return J6[3:6, :]

    def gravity_dir_in_frame(self, theta: np.ndarray, g_base: np.ndarray, fid: int) -> np.ndarray:
        # NOTE: Despite the argument name `g_base`, this function now interprets the input
        # as the gravity vector expressed in the WORLD frame and returns the unit gravity
        # direction expressed in the *frame* coordinates. This change is to ensure consistency
        # with the Bingham construction which compares WORLD gravity to the link-frame gravity.
        # (Function name is kept for compatibility; do NOT rename.)
        self._fk_update(theta)
        R_wf = self.data.oMf[fid].rotation
        gw = g_base / (np.linalg.norm(g_base) + 1e-12)
        gf = R_wf.T @ gw
        return gf / (np.linalg.norm(gf) + 1e-12)

    def tau_gravity(self, theta: np.ndarray) -> np.ndarray:
        return pin.computeGeneralizedGravity(self.model, self.data, theta)

    def d_tau_gravity(self, theta: np.ndarray) -> np.ndarray:
        return pin.computeGeneralizedGravityDerivatives(self.model, self.data, theta)

    def potential_gravity(self, theta: np.ndarray) -> float:
        com = pin.centerOfMass(self.model, self.data, theta)
        g = self.model.gravity.linear
        return -self.total_mass * float(np.dot(g, com))
    
    def fk_pose(self, theta: np.ndarray) -> pin.SE3:
        self._fk_update(theta)
        return pin.SE3(self.data.oMf[self.tip_fid])
