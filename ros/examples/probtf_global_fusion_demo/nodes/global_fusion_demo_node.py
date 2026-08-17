#!/usr/bin/env python3

import math

import numpy as np
import rospy
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

from probtf.distributions import (
    BinghamOrientation,
    ConditionalGaussianTranslation,
    TransformComponent,
    TransformDistribution,
    TransformDistributionStamped,
)
from probtf.geometry import (
    DeterministicTransform,
    quat_to_rotmat,
    relative_transform,
    rpy_to_quat,
    se3_log,
)
from probtf.probability.sampling import sample_bingham_orientation
from probtf.provenance import (
    ApproximationInfo,
    ApproximationKind,
    ComponentProvenance,
    TransformProvenance,
)
from probtf.temporal.backends import component_from_pose_covariance
from probtf_estimators.orientation_imu import vector_alignment_bingham_evidence
from probtf_msgs.msg import ProbabilisticTransformStamped
from probtf_ros.v2_conversions import transform_distribution_to_msg


PHASE_UNIFORM = 0
PHASE_ONE_VECTOR = 1
PHASE_TWO_VECTORS = 2


class GlobalFusionDemoNode:
    def __init__(self):
        self.world_frame = rospy.get_param("~world_frame", "world").lstrip("/")
        self.base_frame = rospy.get_param("~base_frame", "kasuga_base").lstrip("/")
        self.tool_frame = rospy.get_param("~tool_frame", "kasuga_tool").lstrip("/")
        self.tou_base_frame = rospy.get_param("~tou_base_frame", "tou_base").lstrip("/")
        self.tou_tool_frame = rospy.get_param("~tou_tool_frame", "tou_tool").lstrip("/")
        self.phase_duration = float(rospy.get_param("~phase_duration", 5.0))
        self.publish_rate = float(rospy.get_param("~publish_rate", 10.0))
        self.concentration = float(rospy.get_param("~concentration", 80.0))
        self.sample_count = int(rospy.get_param("~sample_count", 1200))
        self.random_seed = int(rospy.get_param("~random_seed", 19))
        self.loop = bool(rospy.get_param("~loop", True))
        self.active_view = bool(rospy.get_param("~active_view", True))
        self.active_start_delay = float(rospy.get_param("~active_start_delay", 1.5))
        self.gaze_speed = math.radians(float(rospy.get_param("~gaze_speed_deg", 10.0)))
        self.motion_cost_weight = float(rospy.get_param("~motion_cost_weight", 0.10))
        self.tou_initial_gaze_yaw_offset = math.radians(
            float(rospy.get_param("~tou_initial_gaze_yaw_offset_deg", 0.0))
        )
        self.kasuga_initial_gaze_yaw_offset = math.radians(
            float(rospy.get_param("~kasuga_initial_gaze_yaw_offset_deg", 0.0))
        )
        self.enable_joint_uncertainty = bool(
            rospy.get_param("~enable_joint_uncertainty", False)
        )
        self.joint_prior_sigma = math.radians(
            float(rospy.get_param("~joint_prior_sigma_deg", 1.5))
        )
        self.joint_sigma_floor = math.radians(
            float(rospy.get_param("~joint_sigma_floor_deg", 0.12))
        )
        self.joint_update_rate = float(rospy.get_param("~joint_update_rate", 5.0))
        self.joint_bearing_noise = math.radians(
            float(rospy.get_param("~joint_bearing_noise_deg", 0.35))
        )
        self.joint_update_start_phase = int(
            rospy.get_param("~joint_update_start_phase", PHASE_TWO_VECTORS)
        )
        self.tou_joint_bias_truth = np.radians(
            np.asarray(
                rospy.get_param(
                    "~tou_joint_bias_truth_deg",
                    [0.7, -0.5, 0.6, 0.8, -0.4, 0.5],
                ),
                dtype=float,
            )
        )
        self.kasuga_joint_bias_truth = np.radians(
            np.asarray(
                rospy.get_param(
                    "~kasuga_joint_bias_truth_deg",
                    [-0.6, 0.8, -0.5, -0.7, 0.6, -0.4],
                ),
                dtype=float,
            )
        )
        self.arm_length = float(rospy.get_param("~arm_length", 0.35))
        self.base_translation = np.asarray(
            rospy.get_param("~base_translation", [0.75, 0.0, 0.0]),
            dtype=float,
        )
        self.tou_base_position = np.asarray(
            rospy.get_param("~tou_base_position", [-0.65, -0.18, 0.35]),
            dtype=float,
        )
        self.tou_arm_joints = np.radians(
            np.asarray(
                rospy.get_param("~tou_arm_joints_deg", [15.0, -30.0, 70.0]),
                dtype=float,
            )
        )
        self.kasuga_arm_joints = np.radians(
            np.asarray(
                rospy.get_param("~kasuga_arm_joints_deg", [150.0, -25.0, 60.0]),
                dtype=float,
            )
        )
        self.camera_hfov = math.radians(float(rospy.get_param("~camera_hfov_deg", 58.0)))
        self.camera_vfov = math.radians(float(rospy.get_param("~camera_vfov_deg", 65.0)))
        true_yaw_deg = float(rospy.get_param("~true_yaw_deg", 55.0))
        self.true_quaternion = rpy_to_quat(0.0, 0.0, math.radians(true_yaw_deg))
        self.true_rotation = quat_to_rotmat(self.true_quaternion)

        if self.phase_duration <= 0.0:
            raise ValueError("~phase_duration must be positive")
        if self.publish_rate <= 0.0:
            raise ValueError("~publish_rate must be positive")
        if self.concentration <= 0.0:
            raise ValueError("~concentration must be positive")
        if self.sample_count <= 0:
            raise ValueError("~sample_count must be positive")
        if self.arm_length <= 0.0:
            raise ValueError("~arm_length must be positive")
        if self.active_start_delay < 0.0:
            raise ValueError("~active_start_delay must be non-negative")
        if self.gaze_speed <= 0.0:
            raise ValueError("~gaze_speed_deg must be positive")
        if self.motion_cost_weight < 0.0:
            raise ValueError("~motion_cost_weight must be non-negative")
        if self.joint_prior_sigma <= 0.0:
            raise ValueError("~joint_prior_sigma_deg must be positive")
        if self.joint_sigma_floor < 0.0 or self.joint_sigma_floor > self.joint_prior_sigma:
            raise ValueError("~joint_sigma_floor_deg must lie in [0, joint_prior_sigma_deg]")
        if self.joint_update_rate <= 0.0:
            raise ValueError("~joint_update_rate must be positive")
        if self.joint_bearing_noise <= 0.0:
            raise ValueError("~joint_bearing_noise_deg must be positive")
        if self.joint_update_start_phase not in (PHASE_UNIFORM, PHASE_ONE_VECTOR, PHASE_TWO_VECTORS):
            raise ValueError("~joint_update_start_phase must be 0, 1, or 2")
        if self.base_translation.shape != (3,) or not np.all(np.isfinite(self.base_translation)):
            raise ValueError("~base_translation must contain three finite values")
        if self.tou_base_position.shape != (3,) or not np.all(np.isfinite(self.tou_base_position)):
            raise ValueError("~tou_base_position must contain three finite values")
        if self.tou_arm_joints.shape != (3,) or not np.all(np.isfinite(self.tou_arm_joints)):
            raise ValueError("~tou_arm_joints_deg must contain three finite values")
        if self.kasuga_arm_joints.shape != (3,) or not np.all(np.isfinite(self.kasuga_arm_joints)):
            raise ValueError("~kasuga_arm_joints_deg must contain three finite values")
        if self.tou_joint_bias_truth.shape != (6,) or not np.all(np.isfinite(self.tou_joint_bias_truth)):
            raise ValueError("~tou_joint_bias_truth_deg must contain six finite values")
        if self.kasuga_joint_bias_truth.shape != (6,) or not np.all(np.isfinite(self.kasuga_joint_bias_truth)):
            raise ValueError("~kasuga_joint_bias_truth_deg must contain six finite values")
        if not 0.0 < self.camera_hfov < math.pi or not 0.0 < self.camera_vfov < math.pi:
            raise ValueError("camera FOV values must be in (0, 180) degrees")

        # A stylized 6-DoF arm is used for visualization and later active-view
        # control. The first three revolute joints position the wrist; a full
        # 3-DoF spherical wrist orients the eye-in-hand camera. The current demo
        # points that wrist at the shared landmark field. A later active-view
        # controller can replace only that gaze target / joint command layer.
        reach_scale = self.arm_length / 0.35
        self.shoulder_height = 0.10 * reach_scale
        self.upper_arm_length = 0.16 * reach_scale
        self.forearm_length = 0.14 * reach_scale
        self.camera_offset = 0.05 * reach_scale

        # The first observation aligns z with z. It determines tilt while leaving
        # the complete yaw circle unobservable. The second observation aligns the
        # local x axis with its true world direction and removes the remaining yaw
        # ambiguity.
        self.landmarks_world = self._make_landmarks()
        self.landmark_centroid = np.mean(self.landmarks_world, axis=0)
        self.landmarks_base = np.asarray(
            [
                self.true_rotation.T @ (point - self.base_translation)
                for point in self.landmarks_world
            ],
            dtype=float,
        )
        self.reference_vector_1 = self.landmarks_world[1] - self.landmarks_world[0]
        self.observed_vector_1 = self.landmarks_base[1] - self.landmarks_base[0]
        self.reference_vector_2 = self.landmarks_world[2] - self.landmarks_world[0]
        self.observed_vector_2 = self.landmarks_base[2] - self.landmarks_base[0]
        self.evidence_1 = vector_alignment_bingham_evidence(
            self.reference_vector_1,
            self.observed_vector_1,
            self.concentration,
        )
        self.evidence_2 = vector_alignment_bingham_evidence(
            self.reference_vector_2,
            self.observed_vector_2,
            self.concentration,
        )

        # Start both wrists by looking at the first landmark pair. With the
        # default FOV, moon_0 and moon_1 are visible while moon_2 is outside the
        # horizontal field of view. After the first Bingham ridge is acquired,
        # the active-view controller chooses the next landmark from information
        # gain and rotates both spherical wrists toward it continuously.
        self.initial_gaze_target = np.mean(self.landmarks_world[:2], axis=0)
        self.tou_arm_state = self._arm_state(
            self.tou_base_position,
            np.eye(3),
            self.tou_arm_joints,
            self.initial_gaze_target,
        )
        self.kasuga_arm_state = self._arm_state(
            self.base_translation,
            self.true_rotation,
            self.kasuga_arm_joints,
            self.initial_gaze_target,
        )
        self.tou_gaze_forward = self.tou_arm_state["camera_rotation"][:, 0].copy()
        self.kasuga_gaze_forward = self.kasuga_arm_state["camera_rotation"][:, 0].copy()
        if abs(self.tou_initial_gaze_yaw_offset) > 1.0e-12:
            self.tou_gaze_forward = self._rotation_z(self.tou_initial_gaze_yaw_offset) @ self.tou_gaze_forward
            self.tou_arm_state = self._state_with_forward(self.tou_arm_state, self.tou_gaze_forward)
        if abs(self.kasuga_initial_gaze_yaw_offset) > 1.0e-12:
            self.kasuga_gaze_forward = self._rotation_z(self.kasuga_initial_gaze_yaw_offset) @ self.kasuga_gaze_forward
            self.kasuga_arm_state = self._state_with_forward(self.kasuga_arm_state, self.kasuga_gaze_forward)

        # Demo-local latent Gaussian for the twelve joint zero offsets.  This is
        # deliberately kept outside probtf_core until the dependency-aware
        # smoother described in PROBTF_DEPENDENCY_SMOOTHER_PLAN.md is added.
        # The inference below is matrix-only; no particles are used.  The
        # resulting marginal camera-pose covariance is encoded back into the
        # native Bingham + conditional-Gaussian transform component so both
        # Tou and Kasuga publish stochastic tool edges today.
        self.joint_bias_mean = np.zeros(12, dtype=float)
        self.joint_bias_truth = np.concatenate(
            [self.tou_joint_bias_truth, self.kasuga_joint_bias_truth]
        )
        self.joint_bias_covariance = (self.joint_prior_sigma ** 2) * np.eye(12, dtype=float)
        self.joint_update_count = 0
        self.joint_landmark_cursor = 0
        self.last_joint_update_time = None

        self.evidence_by_landmark = {1: self.evidence_1, 2: self.evidence_2}
        self.active_parameter_matrix = np.zeros((4, 4), dtype=float)
        self.acquired_evidence = set()
        self.observed_landmarks = set()
        self.tou_visible = set()
        self.kasuga_visible = set()
        self.common_visible = set()
        self.active_phase = PHASE_UNIFORM
        self.active_target_landmark = None
        self.last_timer_time = None
        self._refresh_camera_locals()

        self.probtf_publisher = rospy.Publisher(
            "~probtf",
            ProbabilisticTransformStamped,
            queue_size=10,
        )
        self.marker_publisher = rospy.Publisher(
            "~markers",
            MarkerArray,
            queue_size=2,
        )

        self.start_time = rospy.Time.now().to_sec()
        self.last_phase = None
        self.sequence = 0
        self.timer = rospy.Timer(
            rospy.Duration.from_sec(1.0 / self.publish_rate),
            self._timer_callback,
        )

    def _phase(self, now):
        if self.active_view:
            return self.active_phase
        elapsed = max(0.0, now - self.start_time)
        index = int(elapsed // self.phase_duration)
        if self.loop:
            return index % 3
        return min(index, PHASE_TWO_VECTORS)

    def _orientation_for_phase(self, phase):
        if self.active_view:
            if phase == PHASE_UNIFORM:
                return BinghamOrientation.uniform()
            return BinghamOrientation.from_parameter_matrix(self.active_parameter_matrix)
        if phase == PHASE_UNIFORM:
            return BinghamOrientation.uniform()
        if phase == PHASE_ONE_VECTOR:
            return BinghamOrientation.from_parameter_matrix(self.evidence_1)
        if phase == PHASE_TWO_VECTORS:
            return BinghamOrientation.from_parameter_matrix(self.evidence_1 + self.evidence_2)
        raise ValueError("unknown phase {}".format(phase))

    def _make_landmarks(self):
        anchor = np.array([-0.05, -0.30, 0.22], dtype=float)
        vertical = anchor + 0.38 * (self.true_rotation @ np.array([0.0, 0.0, 1.0]))
        horizontal = anchor + 0.42 * (self.true_rotation @ np.array([1.0, 0.0, 0.0]))
        return np.vstack([anchor, vertical, horizontal])

    @staticmethod
    def _rotation_x(angle):
        cosine = math.cos(angle)
        sine = math.sin(angle)
        return np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, cosine, -sine],
                [0.0, sine, cosine],
            ],
            dtype=float,
        )

    @staticmethod
    def _rotation_y(angle):
        cosine = math.cos(angle)
        sine = math.sin(angle)
        return np.array(
            [
                [cosine, 0.0, sine],
                [0.0, 1.0, 0.0],
                [-sine, 0.0, cosine],
            ],
            dtype=float,
        )

    @staticmethod
    def _rotation_z(angle):
        cosine = math.cos(angle)
        sine = math.sin(angle)
        return np.array(
            [
                [cosine, -sine, 0.0],
                [sine, cosine, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )

    @staticmethod
    def _translation(mean):
        return ConditionalGaussianTranslation(
            mean_at_reference=np.asarray(mean, dtype=float),
            residual_covariance=np.zeros((3, 3), dtype=float),
            rotation_coupling=np.zeros((3, 9), dtype=float),
        )

    def _record(self, parent, child, edge_id, source_id, orientation, translation, stamp):
        component = TransformComponent(
            component_id=edge_id + "_component",
            raw_weight=1.0,
            orientation=orientation,
            translation=self._translation(translation),
        )
        return TransformDistributionStamped(
            parent_frame_id=parent,
            child_frame_id=child,
            stamp=stamp,
            edge_id=edge_id,
            authority="probtf_global_fusion_demo",
            distribution=TransformDistribution((component,)),
            provenance=TransformProvenance(
                source_ids=(source_id,),
                method="global_orientation_observability_demo",
                detail=(
                    "Synthetic posterior used to show SO(3) -> S1 -> local "
                    "observability before committing to a point estimate."
                ),
            ),
            is_static=False,
        )

    def _joint_tool_record(self, name, parent, child, edge_id, stamp):
        if not self.enable_joint_uncertainty:
            if name == "tou":
                translation = self.tou_camera_local
                rotation = self.tou_camera_rotation_local
            elif name == "kasuga":
                translation = self.kasuga_camera_local
                rotation = self.kasuga_camera_rotation_local
            else:
                raise ValueError("unknown robot {}".format(name))
            return self._record(
                parent,
                child,
                edge_id,
                "articulated_eye_in_hand_geometry",
                BinghamOrientation.dirac(self._quat_from_rotation(rotation).tolist()),
                translation,
                stamp,
            )

        transform, covariance = self._camera_pose_moments(name)
        approximation = ApproximationInfo(
            kind=ApproximationKind.TANGENT_SURROGATE,
            lossy=True,
            detail=(
                "Demo-local correlated joint-zero Gaussian is pushed through the "
                "6-DoF eye-in-hand kinematics by J Sigma J^T and encoded as a "
                "native ProbTF Bingham/conditional-Gaussian marginal."
            ),
            source="probtf_global_fusion_demo.joint_latent",
        )
        dependency_id = "latent_joint_bias:{}".format(name)
        component = component_from_pose_covariance(
            component_id=edge_id + "_component",
            raw_weight=1.0,
            transform=transform,
            covariance=covariance,
            provenance=ComponentProvenance(
                source_ids=(dependency_id,),
                method="joint_latent_marginal",
                detail=(
                    "Temporary demo-side marginal. The core plan moves this latent "
                    "Gaussian and its cross-edge dependencies into probtf_core."
                ),
            ),
            approximation=approximation,
        )
        return TransformDistributionStamped(
            parent_frame_id=parent,
            child_frame_id=child,
            stamp=stamp,
            edge_id=edge_id,
            authority="probtf_global_fusion_demo",
            distribution=TransformDistribution((component,)),
            provenance=TransformProvenance(
                source_ids=(dependency_id,),
                method="joint_latent_marginal",
                detail=(
                    "All six joint zero offsets are kept jointly Gaussian in the "
                    "demo node; this edge is their current camera-pose marginal."
                ),
            ),
            is_static=False,
            approximation=approximation,
        )

    def _publish_probtf(self, orientation, stamp):
        tou_base_record = self._record(
            self.world_frame,
            self.tou_base_frame,
            "world__to__tou_base",
            "synthetic_tou_base",
            BinghamOrientation.dirac([1.0, 0.0, 0.0, 0.0]),
            self.tou_base_position,
            stamp,
        )
        kasuga_base_record = self._record(
            self.world_frame,
            self.base_frame,
            "world__to__kasuga_base",
            "synthetic_shared_landmarks",
            orientation,
            self.base_translation,
            stamp,
        )
        tou_tool_record = self._joint_tool_record(
            "tou",
            self.tou_base_frame,
            self.tou_tool_frame,
            "tou_base__to__tou_tool",
            stamp,
        )
        kasuga_tool_record = self._joint_tool_record(
            "kasuga",
            self.base_frame,
            self.tool_frame,
            "kasuga_base__to__kasuga_tool",
            stamp,
        )
        for record in (
            tou_base_record,
            kasuga_base_record,
            tou_tool_record,
            kasuga_tool_record,
        ):
            self.probtf_publisher.publish(transform_distribution_to_msg(record))

    @staticmethod
    def _color(red, green, blue, alpha=1.0):
        return ColorRGBA(r=red, g=green, b=blue, a=alpha)

    @staticmethod
    def _point(values):
        return Point(x=float(values[0]), y=float(values[1]), z=float(values[2]))

    @staticmethod
    def _normalize(values):
        vector = np.asarray(values, dtype=float)
        norm = np.linalg.norm(vector)
        if norm <= 1e-12:
            return np.array([1.0, 0.0, 0.0], dtype=float)
        return vector / norm

    @staticmethod
    def _quat_from_rotation(rotation):
        matrix = np.asarray(rotation, dtype=float)
        trace = float(np.trace(matrix))
        if trace > 0.0:
            scale = math.sqrt(trace + 1.0) * 2.0
            qw = 0.25 * scale
            qx = (matrix[2, 1] - matrix[1, 2]) / scale
            qy = (matrix[0, 2] - matrix[2, 0]) / scale
            qz = (matrix[1, 0] - matrix[0, 1]) / scale
        elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
            scale = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            qw = (matrix[2, 1] - matrix[1, 2]) / scale
            qx = 0.25 * scale
            qy = (matrix[0, 1] + matrix[1, 0]) / scale
            qz = (matrix[0, 2] + matrix[2, 0]) / scale
        elif matrix[1, 1] > matrix[2, 2]:
            scale = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            qw = (matrix[0, 2] - matrix[2, 0]) / scale
            qx = (matrix[0, 1] + matrix[1, 0]) / scale
            qy = 0.25 * scale
            qz = (matrix[1, 2] + matrix[2, 1]) / scale
        else:
            scale = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            qw = (matrix[1, 0] - matrix[0, 1]) / scale
            qx = (matrix[0, 2] + matrix[2, 0]) / scale
            qy = (matrix[1, 2] + matrix[2, 1]) / scale
            qz = 0.25 * scale
        quaternion = np.array([qw, qx, qy, qz], dtype=float)
        quaternion /= np.linalg.norm(quaternion)
        return quaternion

    def _look_at_rotation(self, origin, target):
        forward = self._normalize(np.asarray(target, dtype=float) - np.asarray(origin, dtype=float))
        up_hint = np.array([0.0, 0.0, 1.0], dtype=float)
        if abs(float(np.dot(forward, up_hint))) > 0.96:
            up_hint = np.array([0.0, 1.0, 0.0], dtype=float)
        lateral = self._normalize(np.cross(up_hint, forward))
        upward = self._normalize(np.cross(forward, lateral))
        return np.column_stack([forward, lateral, upward])

    def _arm_state(self, base_position, base_rotation, position_joints, gaze_target):
        q1, q2, q3 = np.asarray(position_joints, dtype=float)
        base_position = np.asarray(base_position, dtype=float)
        base_rotation = np.asarray(base_rotation, dtype=float)

        shoulder = base_position + base_rotation @ np.array(
            [0.0, 0.0, self.shoulder_height],
            dtype=float,
        )
        upper_rotation = base_rotation @ self._rotation_z(q1) @ self._rotation_y(q2)
        elbow = shoulder + upper_rotation @ np.array(
            [self.upper_arm_length, 0.0, 0.0],
            dtype=float,
        )
        forearm_rotation = upper_rotation @ self._rotation_y(q3)
        wrist = elbow + forearm_rotation @ np.array(
            [self.forearm_length, 0.0, 0.0],
            dtype=float,
        )

        # The last three DoF are modeled as a spherical wrist. For this passive
        # observability demo, use them to orient the eye-in-hand camera toward the
        # landmark field without changing the wrist position.
        camera_rotation = self._look_at_rotation(wrist, gaze_target)
        camera_position = wrist + camera_rotation[:, 0] * self.camera_offset
        return {
            "base": base_position,
            "shoulder": shoulder,
            "elbow": elbow,
            "wrist": wrist,
            "camera_position": camera_position,
            "camera_rotation": camera_rotation,
        }

    def _state_with_forward(self, state, forward):
        updated = dict(state)
        forward = self._normalize(forward)
        rotation = self._look_at_rotation(state["wrist"], state["wrist"] + forward)
        updated["camera_rotation"] = rotation
        updated["camera_position"] = state["wrist"] + rotation[:, 0] * self.camera_offset
        return updated

    def _refresh_camera_locals(self):
        self.tou_camera_local = self.tou_arm_state["camera_position"] - self.tou_base_position
        self.tou_camera_rotation_local = self.tou_arm_state["camera_rotation"].copy()
        self.kasuga_camera_local = self.true_rotation.T @ (
            self.kasuga_arm_state["camera_position"] - self.base_translation
        )
        self.kasuga_camera_rotation_local = self.true_rotation.T @ self.kasuga_arm_state["camera_rotation"]

    @staticmethod
    def _wrap_angle(angle):
        return math.atan2(math.sin(angle), math.cos(angle))

    @staticmethod
    def _robot_joint_slice(name):
        if name == "tou":
            return slice(0, 6)
        if name == "kasuga":
            return slice(6, 12)
        raise ValueError("unknown robot {}".format(name))

    def _robot_position_joints(self, name):
        if name == "tou":
            return self.tou_arm_joints
        if name == "kasuga":
            return self.kasuga_arm_joints
        raise ValueError("unknown robot {}".format(name))

    def _robot_nominal_camera_rotation_local(self, name):
        if name == "tou":
            return self.tou_camera_rotation_local
        if name == "kasuga":
            return self.kasuga_camera_rotation_local
        raise ValueError("unknown robot {}".format(name))

    def _robot_base_pose(self, name):
        if name == "tou":
            return self.tou_base_position, np.eye(3, dtype=float)
        if name == "kasuga":
            return self.base_translation, self.true_rotation
        raise ValueError("unknown robot {}".format(name))

    def _camera_pose_local_from_joint_bias(self, name, bias):
        bias = np.asarray(bias, dtype=float).reshape(6)
        q1, q2, q3 = self._robot_position_joints(name) + bias[:3]

        shoulder = np.array([0.0, 0.0, self.shoulder_height], dtype=float)
        upper_rotation = self._rotation_z(q1) @ self._rotation_y(q2)
        elbow = shoulder + upper_rotation @ np.array(
            [self.upper_arm_length, 0.0, 0.0], dtype=float
        )
        forearm_rotation = upper_rotation @ self._rotation_y(q3)
        wrist = elbow + forearm_rotation @ np.array(
            [self.forearm_length, 0.0, 0.0], dtype=float
        )

        nominal_rotation = self._robot_nominal_camera_rotation_local(name)
        wrist_error = (
            self._rotation_x(bias[3])
            @ self._rotation_y(bias[4])
            @ self._rotation_z(bias[5])
        )
        camera_rotation = nominal_rotation @ wrist_error
        camera_position = wrist + camera_rotation[:, 0] * self.camera_offset
        return DeterministicTransform(
            camera_position,
            self._quat_from_rotation(camera_rotation),
        )

    def _camera_pose_jacobian(self, name, bias, epsilon=1.0e-6):
        bias = np.asarray(bias, dtype=float).reshape(6)
        nominal = self._camera_pose_local_from_joint_bias(name, bias)
        jacobian = np.zeros((6, 6), dtype=float)
        for column in range(6):
            delta = np.zeros(6, dtype=float)
            delta[column] = epsilon
            positive = self._camera_pose_local_from_joint_bias(name, bias + delta)
            negative = self._camera_pose_local_from_joint_bias(name, bias - delta)
            jacobian[:3, column] = (
                positive.translation - negative.translation
            ) / (2.0 * epsilon)
            plus_rotation = se3_log(relative_transform(nominal, positive))[3:]
            minus_rotation = se3_log(relative_transform(nominal, negative))[3:]
            jacobian[3:, column] = (
                plus_rotation - minus_rotation
            ) / (2.0 * epsilon)
        return nominal, jacobian

    def _camera_pose_moments(self, name):
        joint_slice = self._robot_joint_slice(name)
        bias = self.joint_bias_mean[joint_slice]
        covariance = self.joint_bias_covariance[joint_slice, joint_slice]
        transform, jacobian = self._camera_pose_jacobian(name, bias)
        pose_covariance = jacobian @ covariance @ jacobian.T
        pose_covariance = 0.5 * (pose_covariance + pose_covariance.T)
        return transform, pose_covariance

    def _camera_world_pose_from_joint_bias(self, name, bias):
        local = self._camera_pose_local_from_joint_bias(name, bias)
        base_translation, base_rotation = self._robot_base_pose(name)
        world_rotation = base_rotation @ quat_to_rotmat(local.rotation_wxyz)
        world_translation = base_translation + base_rotation @ local.translation
        return DeterministicTransform(
            world_translation,
            self._quat_from_rotation(world_rotation),
        )

    def _camera_state_from_joint_bias(self, name, bias):
        pose = self._camera_world_pose_from_joint_bias(name, bias)
        return {
            "camera_position": pose.translation,
            "camera_rotation": quat_to_rotmat(pose.rotation_wxyz),
        }

    def _bearing_from_joint_bias(self, name, bias, point):
        pose = self._camera_world_pose_from_joint_bias(name, bias)
        local = quat_to_rotmat(pose.rotation_wxyz).T @ (
            np.asarray(point, dtype=float) - pose.translation
        )
        if local[0] <= 1.0e-8:
            return None
        return np.array(
            [
                math.atan2(local[1], local[0]),
                math.atan2(local[2], local[0]),
            ],
            dtype=float,
        )

    def _bearing_jacobian(self, name, bias, point, epsilon=1.0e-6):
        bias = np.asarray(bias, dtype=float).reshape(6)
        nominal = self._bearing_from_joint_bias(name, bias, point)
        if nominal is None:
            return None
        jacobian = np.zeros((2, 6), dtype=float)
        for column in range(6):
            delta = np.zeros(6, dtype=float)
            delta[column] = epsilon
            positive = self._bearing_from_joint_bias(name, bias + delta, point)
            negative = self._bearing_from_joint_bias(name, bias - delta, point)
            if positive is None or negative is None:
                return None
            difference = np.array(
                [
                    self._wrap_angle(positive[0] - negative[0]),
                    self._wrap_angle(positive[1] - negative[1]),
                ],
                dtype=float,
            )
            jacobian[:, column] = difference / (2.0 * epsilon)
        return jacobian

    def _update_joint_belief_for_observation(self, name, landmark_index):
        joint_slice = self._robot_joint_slice(name)
        mean = self.joint_bias_mean[joint_slice].copy()
        truth = self.joint_bias_truth[joint_slice]
        point = self.landmarks_world[landmark_index]
        measurement = self._bearing_from_joint_bias(name, truth, point)
        prediction = self._bearing_from_joint_bias(name, mean, point)
        local_jacobian = self._bearing_jacobian(name, mean, point)
        if measurement is None or prediction is None or local_jacobian is None:
            return False

        residual = np.array(
            [
                self._wrap_angle(measurement[0] - prediction[0]),
                self._wrap_angle(measurement[1] - prediction[1]),
            ],
            dtype=float,
        )
        jacobian = np.zeros((2, 12), dtype=float)
        jacobian[:, joint_slice] = local_jacobian
        measurement_covariance = (self.joint_bearing_noise ** 2) * np.eye(2, dtype=float)
        covariance = self.joint_bias_covariance
        innovation_covariance = (
            jacobian @ covariance @ jacobian.T + measurement_covariance
        )
        try:
            gain = covariance @ jacobian.T @ np.linalg.inv(innovation_covariance)
        except np.linalg.LinAlgError:
            return False

        self.joint_bias_mean = self.joint_bias_mean + gain @ residual
        identity = np.eye(12, dtype=float)
        correction = identity - gain @ jacobian
        updated = (
            correction @ covariance @ correction.T
            + gain @ measurement_covariance @ gain.T
        )
        updated = 0.5 * (updated + updated.T)
        if self.joint_sigma_floor > 0.0:
            eigenvalues, eigenvectors = np.linalg.eigh(updated)
            floor_variance = self.joint_sigma_floor ** 2
            updated = eigenvectors @ np.diag(np.maximum(eigenvalues, floor_variance)) @ eigenvectors.T
            updated = 0.5 * (updated + updated.T)
        self.joint_bias_covariance = updated
        self.joint_update_count += 1
        return True

    def _maybe_update_joint_belief(self, now, phase):
        if not self.enable_joint_uncertainty or phase < self.joint_update_start_phase:
            return
        if self.last_joint_update_time is not None:
            if now - self.last_joint_update_time < 1.0 / self.joint_update_rate:
                return
        self.last_joint_update_time = now
        self._update_visibility()
        visible = sorted(self.common_visible)
        if not visible:
            return
        landmark_index = visible[self.joint_landmark_cursor % len(visible)]
        self.joint_landmark_cursor += 1
        self._update_joint_belief_for_observation("tou", landmark_index)
        self._update_joint_belief_for_observation("kasuga", landmark_index)

    def _joint_sigma_rms_deg(self, name):
        joint_slice = self._robot_joint_slice(name)
        covariance = self.joint_bias_covariance[joint_slice, joint_slice]
        return math.degrees(math.sqrt(max(0.0, float(np.trace(covariance)) / 6.0)))

    def _camera_visible(self, state, point):
        delta = np.asarray(point, dtype=float) - state["camera_position"]
        local = state["camera_rotation"].T @ delta
        if local[0] <= 1e-8:
            return False
        horizontal = abs(math.atan2(local[1], local[0]))
        vertical = abs(math.atan2(local[2], local[0]))
        return horizontal <= 0.5 * self.camera_hfov and vertical <= 0.5 * self.camera_vfov

    def _update_visibility(self):
        if self.enable_joint_uncertainty:
            tou_state = self._camera_state_from_joint_bias(
                "tou", self.joint_bias_truth[self._robot_joint_slice("tou")]
            )
            kasuga_state = self._camera_state_from_joint_bias(
                "kasuga", self.joint_bias_truth[self._robot_joint_slice("kasuga")]
            )
        else:
            tou_state = self.tou_arm_state
            kasuga_state = self.kasuga_arm_state
        self.tou_visible = {
            index
            for index, point in enumerate(self.landmarks_world)
            if self._camera_visible(tou_state, point)
        }
        self.kasuga_visible = {
            index
            for index, point in enumerate(self.landmarks_world)
            if self._camera_visible(kasuga_state, point)
        }
        self.common_visible = self.tou_visible & self.kasuga_visible

    @staticmethod
    def _bingham_information_score(parameter_matrix):
        symmetric = 0.5 * (parameter_matrix + parameter_matrix.T)
        eigenvalues = np.linalg.eigvalsh(symmetric)
        top = float(eigenvalues[-1])
        total_spread = float(np.sum(top - eigenvalues[:-1]))
        unique_mode_gap = float(top - eigenvalues[-2])
        return 0.25 * total_spread + 2.0 * unique_mode_gap

    @staticmethod
    def _angle_between(first, second):
        first = np.asarray(first, dtype=float)
        second = np.asarray(second, dtype=float)
        first /= np.linalg.norm(first)
        second /= np.linalg.norm(second)
        return math.acos(float(np.clip(np.dot(first, second), -1.0, 1.0)))

    def _select_informative_landmark(self):
        current_score = self._bingham_information_score(self.active_parameter_matrix)
        best_index = None
        best_score = -float("inf")
        for index, evidence in sorted(self.evidence_by_landmark.items()):
            if index in self.acquired_evidence:
                continue
            candidate_matrix = self.active_parameter_matrix + evidence
            gain = self._bingham_information_score(candidate_matrix) - current_score
            point = self.landmarks_world[index]
            tou_direction = self._normalize(point - self.tou_arm_state["wrist"])
            kasuga_direction = self._normalize(point - self.kasuga_arm_state["wrist"])
            motion_cost = (
                self._angle_between(self.tou_gaze_forward, tou_direction)
                + self._angle_between(self.kasuga_gaze_forward, kasuga_direction)
            )
            score = gain - self.motion_cost_weight * motion_cost
            if score > best_score:
                best_score = score
                best_index = index
        return best_index

    def _rotate_toward(self, current, desired, maximum_angle):
        current = self._normalize(current)
        desired = self._normalize(desired)
        angle = self._angle_between(current, desired)
        if angle <= maximum_angle or angle <= 1e-9:
            return desired
        axis = np.cross(current, desired)
        axis_norm = np.linalg.norm(axis)
        if axis_norm <= 1e-12:
            return desired
        axis /= axis_norm
        theta = maximum_angle
        return self._normalize(
            current * math.cos(theta)
            + np.cross(axis, current) * math.sin(theta)
            + axis * np.dot(axis, current) * (1.0 - math.cos(theta))
        )

    def _update_active_gaze(self, dt):
        if self.active_phase == PHASE_TWO_VECTORS:
            self.active_target_landmark = None
            return

        if self.active_phase == PHASE_UNIFORM:
            target = self.initial_gaze_target
            self.active_target_landmark = None
        else:
            target_index = self._select_informative_landmark()
            self.active_target_landmark = target_index
            if target_index is None:
                return
            target = self.landmarks_world[target_index]

        max_angle = self.gaze_speed * max(dt, 0.0)
        tou_desired = self._normalize(target - self.tou_arm_state["wrist"])
        kasuga_desired = self._normalize(target - self.kasuga_arm_state["wrist"])
        self.tou_gaze_forward = self._rotate_toward(self.tou_gaze_forward, tou_desired, max_angle)
        self.kasuga_gaze_forward = self._rotate_toward(self.kasuga_gaze_forward, kasuga_desired, max_angle)
        self.tou_arm_state = self._state_with_forward(self.tou_arm_state, self.tou_gaze_forward)
        self.kasuga_arm_state = self._state_with_forward(self.kasuga_arm_state, self.kasuga_gaze_forward)
        self._refresh_camera_locals()

    def _acquire_active_evidence(self, now):
        self._update_visibility()
        if now - self.start_time < self.active_start_delay:
            return

        self.observed_landmarks.update(self.common_visible)
        if 0 in self.observed_landmarks and 1 in self.observed_landmarks and 1 not in self.acquired_evidence:
            self.active_parameter_matrix = self.active_parameter_matrix + self.evidence_by_landmark[1]
            self.acquired_evidence.add(1)
        if 0 in self.observed_landmarks and 2 in self.observed_landmarks and 2 not in self.acquired_evidence:
            self.active_parameter_matrix = self.active_parameter_matrix + self.evidence_by_landmark[2]
            self.acquired_evidence.add(2)

        if 2 in self.acquired_evidence:
            self.active_phase = PHASE_TWO_VECTORS
        elif 1 in self.acquired_evidence:
            self.active_phase = PHASE_ONE_VECTOR
        else:
            self.active_phase = PHASE_UNIFORM

    def _active_step(self, now, dt):
        self._update_active_gaze(dt)
        self._acquire_active_evidence(now)
        return self.active_phase, self._orientation_for_phase(self.active_phase)

    def _phase_label(self, phase):
        if self.active_view:
            if phase == PHASE_UNIFORM:
                label = "Active view: seeking the initial shared moon pair; relative orientation is uniform on SO(3)"
            elif phase == PHASE_ONE_VECTOR:
                if self.active_target_landmark is None:
                    label = "Active view: one shared direction leaves S1; selecting the next informative gaze"
                else:
                    label = "Active view: S1 remains; continuous information servo turns toward moon_{}".format(
                        self.active_target_landmark
                    )
            else:
                label = "Active view: second non-collinear direction acquired; relative orientation localized"
        elif phase == PHASE_UNIFORM:
            label = "Tou and Kasuga share no common moons: Kasuga relative orientation is uniform on SO(3)"
        elif phase == PHASE_ONE_VECTOR:
            label = "Both robots observe moon_0 and moon_1: one shared direction leaves a global S1 yaw ambiguity"
        else:
            label = "Both robots also observe moon_2: a second non-collinear direction localizes Kasuga orientation"

        if self.enable_joint_uncertainty:
            label += (
                "\nJoint-bias posterior RMS: Tou {:.2f} deg | Kasuga {:.2f} deg | visual updates {}"
            ).format(
                self._joint_sigma_rms_deg("tou"),
                self._joint_sigma_rms_deg("kasuga"),
                self.joint_update_count,
            )
        return label

    def _text_marker(self, phase, stamp):
        marker = Marker()
        marker.header.frame_id = self.world_frame
        marker.header.stamp = rospy.Time.from_sec(stamp)
        marker.ns = "status"
        marker.id = 0
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        marker.pose.position.x = 0.0
        marker.pose.position.y = 0.0
        marker.pose.position.z = 0.98
        marker.pose.orientation.w = 1.0
        marker.scale.z = 0.07
        marker.color = self._color(1.0, 1.0, 1.0, 1.0)
        marker.text = self._phase_label(phase)
        return marker

    def _line_marker(self, marker_id, namespace, points, color, stamp, width=0.008):
        marker = Marker()
        marker.header.frame_id = self.world_frame
        marker.header.stamp = rospy.Time.from_sec(stamp)
        marker.ns = namespace
        marker.id = marker_id
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD
        marker.points = [self._point(point) for point in points]
        marker.scale.x = width
        marker.color = color
        return marker

    def _arrow_marker(self, marker_id, namespace, origin, direction, length, color, stamp, shaft=0.014, head=0.028):
        marker = Marker()
        marker.header.frame_id = self.world_frame
        marker.header.stamp = rospy.Time.from_sec(stamp)
        marker.ns = namespace
        marker.id = marker_id
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        direction = self._normalize(direction)
        marker.points = [
            self._point(origin),
            self._point(np.asarray(origin, dtype=float) + length * direction),
        ]
        marker.scale.x = shaft
        marker.scale.y = head
        marker.scale.z = 1.4 * head
        marker.color = color
        return marker

    def _sphere_marker(self, marker_id, namespace, center, color, stamp, diameter):
        marker = Marker()
        marker.header.frame_id = self.world_frame
        marker.header.stamp = rospy.Time.from_sec(stamp)
        marker.ns = namespace
        marker.id = marker_id
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position = self._point(center)
        marker.pose.orientation.w = 1.0
        marker.scale.x = diameter
        marker.scale.y = diameter
        marker.scale.z = diameter
        marker.color = color
        return marker

    def _box_marker(self, marker_id, namespace, center, rotation, color, stamp, scale):
        marker = Marker()
        marker.header.frame_id = self.world_frame
        marker.header.stamp = rospy.Time.from_sec(stamp)
        marker.ns = namespace
        marker.id = marker_id
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        marker.pose.position = self._point(center)
        quaternion = self._quat_from_rotation(rotation)
        marker.pose.orientation.w = float(quaternion[0])
        marker.pose.orientation.x = float(quaternion[1])
        marker.pose.orientation.y = float(quaternion[2])
        marker.pose.orientation.z = float(quaternion[3])
        marker.scale.x = float(scale[0])
        marker.scale.y = float(scale[1])
        marker.scale.z = float(scale[2])
        marker.color = color
        return marker

    def _covariance_ellipsoid_marker(
        self,
        marker_id,
        namespace,
        center,
        covariance,
        color,
        stamp,
        sigma_scale=2.5,
    ):
        covariance = 0.5 * (
            np.asarray(covariance, dtype=float) + np.asarray(covariance, dtype=float).T
        )
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        eigenvalues = np.maximum(eigenvalues, 0.0)
        if np.linalg.det(eigenvectors) < 0.0:
            eigenvectors[:, 0] *= -1.0
        diameters = 2.0 * sigma_scale * np.sqrt(eigenvalues)
        diameters = np.maximum(diameters, 0.004)

        marker = Marker()
        marker.header.frame_id = self.world_frame
        marker.header.stamp = rospy.Time.from_sec(stamp)
        marker.ns = namespace
        marker.id = marker_id
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position = self._point(center)
        quaternion = self._quat_from_rotation(eigenvectors)
        marker.pose.orientation.w = float(quaternion[0])
        marker.pose.orientation.x = float(quaternion[1])
        marker.pose.orientation.y = float(quaternion[2])
        marker.pose.orientation.z = float(quaternion[3])
        marker.scale.x = float(diameters[0])
        marker.scale.y = float(diameters[1])
        marker.scale.z = float(diameters[2])
        marker.color = color
        return marker

    def _joint_uncertainty_markers(self, stamp):
        if not self.enable_joint_uncertainty:
            return []
        markers = []
        specifications = (
            ("tou", 350, self._color(0.25, 0.75, 1.0, 0.24)),
            ("kasuga", 360, self._color(1.0, 0.45, 0.75, 0.24)),
        )
        for name, marker_id, color in specifications:
            transform, covariance = self._camera_pose_moments(name)
            base_translation, base_rotation = self._robot_base_pose(name)
            center = base_translation + base_rotation @ transform.translation
            world_translation_covariance = (
                base_rotation @ covariance[:3, :3] @ base_rotation.T
            )
            markers.append(
                self._covariance_ellipsoid_marker(
                    marker_id,
                    "{}_joint_camera_uncertainty".format(name),
                    center,
                    world_translation_covariance,
                    color,
                    stamp,
                )
            )

            label = Marker()
            label.header.frame_id = self.world_frame
            label.header.stamp = rospy.Time.from_sec(stamp)
            label.ns = "{}_joint_camera_uncertainty".format(name)
            label.id = marker_id + 1
            label.type = Marker.TEXT_VIEW_FACING
            label.action = Marker.ADD
            label.pose.position = self._point(center + np.array([0.0, 0.0, 0.10]))
            label.pose.orientation.w = 1.0
            label.scale.z = 0.034
            label.color = self._color(color.r, color.g, color.b, 0.92)
            label.text = "{} joint RMS sigma = {:.2f} deg".format(
                name, self._joint_sigma_rms_deg(name)
            )
            markers.append(label)
        return markers

    def _cylinder_marker(self, marker_id, namespace, start, end, color, stamp, radius=0.018):
        start = np.asarray(start, dtype=float)
        end = np.asarray(end, dtype=float)
        delta = end - start
        length = float(np.linalg.norm(delta))
        if length <= 1e-9:
            return self._sphere_marker(marker_id, namespace, start, color, stamp, 2.0 * radius)

        direction = delta / length
        z_axis = np.array([0.0, 0.0, 1.0], dtype=float)
        dot = float(np.clip(np.dot(z_axis, direction), -1.0, 1.0))
        if dot < -0.999999:
            quaternion = np.array([0.0, 1.0, 0.0, 0.0], dtype=float)
        else:
            cross = np.cross(z_axis, direction)
            scale = math.sqrt(2.0 * (1.0 + dot))
            quaternion = np.array(
                [0.5 * scale, cross[0] / scale, cross[1] / scale, cross[2] / scale],
                dtype=float,
            )
            quaternion /= np.linalg.norm(quaternion)

        marker = Marker()
        marker.header.frame_id = self.world_frame
        marker.header.stamp = rospy.Time.from_sec(stamp)
        marker.ns = namespace
        marker.id = marker_id
        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD
        marker.pose.position = self._point(0.5 * (start + end))
        marker.pose.orientation.w = float(quaternion[0])
        marker.pose.orientation.x = float(quaternion[1])
        marker.pose.orientation.y = float(quaternion[2])
        marker.pose.orientation.z = float(quaternion[3])
        marker.scale.x = 2.0 * radius
        marker.scale.y = 2.0 * radius
        marker.scale.z = length
        marker.color = color
        return marker

    def _camera_frustum_markers(self, marker_base_id, namespace, camera_position, camera_rotation, color, stamp):
        distance = 0.12
        half_width = math.tan(0.5 * self.camera_hfov) * distance
        half_height = math.tan(0.5 * self.camera_vfov) * distance
        forward = camera_rotation[:, 0]
        lateral = camera_rotation[:, 1]
        upward = camera_rotation[:, 2]
        center = camera_position + distance * forward
        corners = [
            center + sx * half_width * lateral + sy * half_height * upward
            for sx, sy in [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)]
        ]
        segments = []
        for corner in corners:
            segments.extend([camera_position, corner])
        for index in range(4):
            segments.extend([corners[index], corners[(index + 1) % 4]])
        return [
            self._line_marker(
                marker_base_id,
                namespace,
                segments,
                self._color(color.r, color.g, color.b, 0.34),
                stamp,
                width=0.003,
            ),
            self._arrow_marker(
                marker_base_id + 1,
                namespace,
                camera_position,
                forward,
                0.10,
                self._color(color.r, color.g, color.b, 0.85),
                stamp,
                shaft=0.006,
                head=0.015,
            ),
        ]

    def _single_robot_markers(self, name, marker_base_id, state, color, stamp):
        markers = []
        namespace = "{}_robot".format(name)
        joint_namespace = "{}_joints".format(name)
        camera_namespace = "{}_camera".format(name)

        base = state["base"]
        shoulder = state["shoulder"]
        elbow = state["elbow"]
        wrist = state["wrist"]
        camera_position = state["camera_position"]
        camera_rotation = state["camera_rotation"]

        markers.append(self._sphere_marker(marker_base_id, namespace, base, color, stamp, 0.065))
        link_color = self._color(color.r, color.g, color.b, 0.72)
        markers.append(self._cylinder_marker(marker_base_id + 1, namespace, base, shoulder, link_color, stamp, radius=0.017))
        markers.append(self._cylinder_marker(marker_base_id + 2, namespace, shoulder, elbow, link_color, stamp, radius=0.021))
        markers.append(self._cylinder_marker(marker_base_id + 3, namespace, elbow, wrist, link_color, stamp, radius=0.019))
        markers.append(self._cylinder_marker(marker_base_id + 4, namespace, wrist, camera_position, link_color, stamp, radius=0.013))

        joint_color = self._color(0.92, 0.92, 0.96, 0.92)
        for offset, point in enumerate([shoulder, elbow, wrist]):
            markers.append(
                self._sphere_marker(
                    marker_base_id + 10 + offset,
                    joint_namespace,
                    point,
                    joint_color,
                    stamp,
                    0.052 if offset < 2 else 0.045,
                )
            )

        markers.append(
            self._box_marker(
                marker_base_id + 20,
                camera_namespace,
                camera_position,
                camera_rotation,
                self._color(0.08, 0.08, 0.10, 0.98),
                stamp,
                scale=(0.055, 0.038, 0.030),
            )
        )
        markers.extend(
            self._camera_frustum_markers(
                marker_base_id + 21,
                camera_namespace,
                camera_position,
                camera_rotation,
                color,
                stamp,
            )
        )

        # Small wrist-frame axes emphasize that the camera orientation is a full
        # 3-DoF spherical wrist rather than a fixed direction on a planar stick.
        axis_colors = [
            self._color(0.95, 0.25, 0.25, 0.85),
            self._color(0.25, 0.95, 0.35, 0.85),
            self._color(0.25, 0.55, 1.0, 0.85),
        ]
        for axis_index in range(3):
            markers.append(
                self._arrow_marker(
                    marker_base_id + 30 + axis_index,
                    "{}_wrist_axes".format(name),
                    wrist,
                    camera_rotation[:, axis_index],
                    0.045,
                    axis_colors[axis_index],
                    stamp,
                    shaft=0.004,
                    head=0.010,
                )
            )

        base_label = Marker()
        base_label.header.frame_id = self.world_frame
        base_label.header.stamp = rospy.Time.from_sec(stamp)
        base_label.ns = "{}_labels".format(name)
        base_label.id = marker_base_id + 40
        base_label.type = Marker.TEXT_VIEW_FACING
        base_label.action = Marker.ADD
        base_label.pose.position = self._point(base + np.array([0.0, 0.0, 0.09]))
        base_label.pose.orientation.w = 1.0
        base_label.scale.z = 0.05
        base_label.color = self._color(1.0, 1.0, 1.0, 0.9)
        base_label.text = "{}_base".format(name)
        markers.append(base_label)

        camera_label = Marker()
        camera_label.header = base_label.header
        camera_label.ns = "{}_labels".format(name)
        camera_label.id = marker_base_id + 41
        camera_label.type = Marker.TEXT_VIEW_FACING
        camera_label.action = Marker.ADD
        camera_label.pose.position = self._point(camera_position + np.array([0.0, 0.0, 0.075]))
        camera_label.pose.orientation.w = 1.0
        camera_label.scale.z = 0.042
        camera_label.color = self._color(1.0, 1.0, 1.0, 0.85)
        camera_label.text = "{}_camera (6-DoF arm)".format(name)
        markers.append(camera_label)
        return markers

    def _robot_markers(self, stamp):
        markers = []
        markers.extend(
            self._single_robot_markers(
                "tou",
                10,
                self.tou_arm_state,
                self._color(0.25, 0.75, 1.0, 0.95),
                stamp,
            )
        )
        markers.extend(
            self._single_robot_markers(
                "kasuga",
                60,
                self.kasuga_arm_state,
                self._color(1.0, 0.45, 0.75, 0.95),
                stamp,
            )
        )
        return markers

    def _landmark_markers(self, phase, stamp):
        markers = []
        active_count = 0 if phase == PHASE_UNIFORM else (2 if phase == PHASE_ONE_VECTOR else 3)
        for index, point in enumerate(self.landmarks_world):
            marker = Marker()
            marker.header.frame_id = self.world_frame
            marker.header.stamp = rospy.Time.from_sec(stamp)
            marker.ns = "shared_landmarks"
            marker.id = 100 + index
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position = self._point(point)
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.06
            marker.scale.y = 0.06
            marker.scale.z = 0.06
            if self.active_view:
                if index in self.common_visible:
                    marker.color = self._color(1.0, 0.85, 0.15, 1.0)
                elif index in self.observed_landmarks:
                    marker.color = self._color(1.0, 0.55, 0.12, 0.72)
                else:
                    marker.color = self._color(0.45, 0.45, 0.45, 0.32)
            elif index < active_count:
                marker.color = self._color(1.0, 0.85, 0.15, 1.0)
            else:
                marker.color = self._color(0.45, 0.45, 0.45, 0.32)
            markers.append(marker)

            label = Marker()
            label.header = marker.header
            label.ns = "shared_landmark_labels"
            label.id = 120 + index
            label.type = Marker.TEXT_VIEW_FACING
            label.action = Marker.ADD
            label.pose.position = self._point(point + np.array([0.0, 0.0, 0.08]))
            label.pose.orientation.w = 1.0
            label.scale.z = 0.045
            label.color = self._color(1.0, 1.0, 1.0, 0.85)
            label.text = "moon_{}".format(index)
            markers.append(label)
        return markers

    def _observation_markers(self, phase, stamp):
        markers = []
        if self.enable_joint_uncertainty:
            tou_camera = self._camera_state_from_joint_bias(
                "tou", self.joint_bias_truth[self._robot_joint_slice("tou")]
            )["camera_position"]
            kasuga_camera = self._camera_state_from_joint_bias(
                "kasuga", self.joint_bias_truth[self._robot_joint_slice("kasuga")]
            )["camera_position"]
        else:
            tou_camera = self.tou_arm_state["camera_position"]
            kasuga_camera = self.kasuga_arm_state["camera_position"]
        tou_color = self._color(0.25, 0.75, 1.0, 0.42)
        kasuga_color = self._color(1.0, 0.45, 0.75, 0.42)

        if self.active_view:
            tou_indices = sorted(self.tou_visible)
            kasuga_indices = sorted(self.kasuga_visible)
        else:
            active_count = 0 if phase == PHASE_UNIFORM else (2 if phase == PHASE_ONE_VECTOR else 3)
            tou_indices = list(range(active_count))
            kasuga_indices = list(range(active_count))

        line_points_tou = []
        line_points_kasuga = []
        for index in tou_indices:
            line_points_tou.extend([tou_camera, self.landmarks_world[index]])
        for index in kasuga_indices:
            line_points_kasuga.extend([kasuga_camera, self.landmarks_world[index]])
        if line_points_tou:
            markers.append(self._line_marker(200, "tou_observations", line_points_tou, tou_color, stamp, width=0.0040))
        if line_points_kasuga:
            markers.append(self._line_marker(201, "kasuga_observations", line_points_kasuga, kasuga_color, stamp, width=0.0040))
        return markers

    def _camera_cloud_marker(self, orientation, phase, stamp):
        samples = sample_bingham_orientation(
            orientation,
            self.sample_count,
            rng=self.random_seed + phase,
        )
        if self.enable_joint_uncertainty:
            kasuga_camera_local = self._camera_pose_moments("kasuga")[0].translation
        else:
            kasuga_camera_local = self.kasuga_camera_local
        points = [
            self.base_translation + quat_to_rotmat(quaternion) @ kasuga_camera_local
            for quaternion in samples
        ]

        marker = Marker()
        marker.header.frame_id = self.world_frame
        marker.header.stamp = rospy.Time.from_sec(stamp)
        marker.ns = "tool_support"
        marker.id = 20
        marker.type = Marker.POINTS
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.012
        marker.scale.y = 0.012
        marker.color = self._color(1.0, 0.45, 0.1, 0.55)
        marker.points = [self._point(point) for point in points]
        return marker

    def _publish_markers(self, orientation, phase, stamp):
        markers = [self._text_marker(phase, stamp)]
        markers.extend(self._landmark_markers(phase, stamp))
        markers.extend(self._robot_markers(stamp))
        markers.extend(self._joint_uncertainty_markers(stamp))
        markers.extend(self._observation_markers(phase, stamp))
        markers.append(self._camera_cloud_marker(orientation, phase, stamp))
        self.marker_publisher.publish(MarkerArray(markers=markers))

    def _timer_callback(self, _event):
        now = rospy.Time.now().to_sec()
        if self.last_timer_time is None:
            dt = 1.0 / self.publish_rate
        else:
            dt = max(0.0, min(now - self.last_timer_time, 0.25))
        self.last_timer_time = now

        if self.active_view:
            phase, orientation = self._active_step(now, dt)
        else:
            phase = self._phase(now)
            orientation = self._orientation_for_phase(phase)
            self._update_visibility()

        self._maybe_update_joint_belief(now, phase)
        self._publish_probtf(orientation, now)
        self._publish_markers(orientation, phase, now)
        if phase != self.last_phase:
            rospy.loginfo("ProbTF global fusion demo: %s", self._phase_label(phase))
            self.last_phase = phase
        self.sequence += 1


def main():
    rospy.init_node("probtf_global_fusion_demo")
    GlobalFusionDemoNode()
    rospy.spin()


if __name__ == "__main__":
    main()
