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
        lock_non_controllable_joints: bool = True,
    ) -> None:
        self.urdf_path = urdf_path
        self.urdf_info = load_urdf_model_info(urdf_path)
        model = pin.buildModelFromUrdf(urdf_path)
        if lock_non_controllable_joints:
            lock_joint_ids = []
            for joint_name in model.names[1:]:
                joint_info = self.urdf_info.joint_map.get(joint_name)
                if joint_info is not None and not joint_info.is_controllable:
                    lock_joint_ids.append(model.getJointId(joint_name))
            if lock_joint_ids:
                self.locked_joint_names = [model.names[joint_id] for joint_id in lock_joint_ids]
                self.model = pin.buildReducedModel(model, lock_joint_ids, np.zeros(model.nq, dtype=float))
            else:
                self.locked_joint_names = []
                self.model = model
        else:
            self.locked_joint_names = []
            self.model = model
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
        J6 = pin.getFrameJacobian(self.model, self.data, fid, pin.ReferenceFrame.WORLD)
        return J6[3:6, :]

    def frame_angular_jacobian_base(self, theta: np.ndarray, fid: int) -> np.ndarray:
        """Relative frame angular Jacobian, expressed in the base frame.

        ``frame_quaternion_wxyz_base`` represents ``R_base,frame``.  Its
        spatial quaternion tangent therefore requires the frame angular
        velocity relative to the selected base and expressed in that base,
        rather than the absolute WORLD angular velocity.
        """
        pin.computeJointJacobians(self.model, self.data, theta)
        pin.updateFramePlacements(self.model, self.data)
        J_frame_world = pin.getFrameJacobian(
            self.model,
            self.data,
            fid,
            pin.ReferenceFrame.WORLD,
        )[3:6, :]
        J_base_world = pin.getFrameJacobian(
            self.model,
            self.data,
            self.base_fid,
            pin.ReferenceFrame.WORLD,
        )[3:6, :]
        R_world_base = self.data.oMf[self.base_fid].rotation
        return R_world_base.T @ (J_frame_world - J_base_world)

    def gravity_dir_jacobian_in_frame(self, theta: np.ndarray, g_world: np.ndarray, fid: int) -> np.ndarray:
        self._fk_update(theta)
        R_wf = self.data.oMf[fid].rotation
        Jw_world = self.frame_angular_jacobian_world(theta, fid)
        Jw_frame = R_wf.T @ Jw_world
        gw = g_world / (np.linalg.norm(g_world) + 1e-12)
        gf = R_wf.T @ gw
        skew_gf = np.array(
            [
                [0.0, -gf[2], gf[1]],
                [gf[2], 0.0, -gf[0]],
                [-gf[1], gf[0], 0.0],
            ],
            dtype=float,
        )
        return skew_gf @ Jw_frame

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
        """Return Pinocchio's gravitational potential energy.

        This deliberately delegates the complete mass bookkeeping to
        Pinocchio.  ``centerOfMass`` does not include the inertia attached to
        the universe joint, while summing ``model.inertias`` does.  Combining
        those two APIs therefore makes the potential inconsistent with
        ``computeGeneralizedGravity`` whenever a fixed-base URDF gives its
        base link a non-zero mass.
        """
        return float(pin.computePotentialEnergy(self.model, self.data, theta))
    
    def fk_pose(self, theta: np.ndarray) -> pin.SE3:
        self._fk_update(theta)
        return pin.SE3(self.data.oMf[self.tip_fid])
