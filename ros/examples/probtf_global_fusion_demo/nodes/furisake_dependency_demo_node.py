#!/usr/bin/env python3

import math

import numpy as np
import rospy
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

from probtf.dependency import (
    EdgeLatentBinding,
    GaussianLatentStore,
    GaussianObservationFactor,
)
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
from probtf.graph import ProbTfGraph
from probtf.probability.sampling import sample_bingham_orientation
from probtf.provenance import Provenance, TransformProvenance
from probtf_estimators.orientation_imu import vector_alignment_bingham_evidence
from probtf_msgs.msg import ProbabilisticTransformStamped
from probtf_ros.v2_conversions import transform_distribution_to_msg


PHASE_UNIFORM = 0
PHASE_ONE_VECTOR = 1
PHASE_LOCALIZED = 2


class FurisakeDependencyDemoNode:
    """Cooperative active-view demo backed by the dependency-aware ProbTF core."""

    def __init__(self):
        self.world_frame = rospy.get_param("~world_frame", "world").lstrip("/")
        self.kasuga_base_frame = rospy.get_param(
            "~kasuga_base_frame", "kasuga_base"
        ).lstrip("/")
        self.kasuga_tool_frame = rospy.get_param(
            "~kasuga_tool_frame", "kasuga_tool"
        ).lstrip("/")
        self.tou_base_frame = rospy.get_param(
            "~tou_base_frame", "tou_base"
        ).lstrip("/")
        self.tou_tool_frame = rospy.get_param(
            "~tou_tool_frame", "tou_tool"
        ).lstrip("/")

        self.publish_rate = float(rospy.get_param("~publish_rate", 20.0))
        self.concentration = float(rospy.get_param("~concentration", 45.0))
        self.sample_count = int(rospy.get_param("~sample_count", 500))
        self.random_seed = int(rospy.get_param("~random_seed", 29))
        self.landmark_count = int(rospy.get_param("~landmark_count", 9))
        self.active_start_delay = float(
            rospy.get_param("~active_start_delay", 0.8)
        )
        self.gaze_speed = math.radians(
            float(rospy.get_param("~gaze_speed_deg", 16.0))
        )
        self.motion_cost_weight = float(
            rospy.get_param("~motion_cost_weight", 0.08)
        )
        self.joint_information_weight = float(
            rospy.get_param("~joint_information_weight", 0.65)
        )
        self.tou_initial_gaze_yaw_offset = math.radians(
            float(rospy.get_param("~tou_initial_gaze_yaw_offset_deg", -55.0))
        )
        self.kasuga_initial_gaze_yaw_offset = math.radians(
            float(rospy.get_param("~kasuga_initial_gaze_yaw_offset_deg", 125.0))
        )
        self.joint_prior_sigma = math.radians(
            float(rospy.get_param("~joint_prior_sigma_deg", 1.5))
        )
        self.joint_bearing_noise = math.radians(
            float(rospy.get_param("~joint_bearing_noise_deg", 0.45))
        )
        self.camera_hfov = math.radians(
            float(rospy.get_param("~camera_hfov_deg", 56.0))
        )
        self.camera_vfov = math.radians(
            float(rospy.get_param("~camera_vfov_deg", 65.0))
        )

        self.arm_length = float(rospy.get_param("~arm_length", 0.35))
        self.kasuga_base_position = np.asarray(
            rospy.get_param("~kasuga_base_position", [0.75, 0.0, 0.0]),
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

        true_yaw = math.radians(
            float(rospy.get_param("~true_yaw_deg", 55.0))
        )
        self.true_quaternion = rpy_to_quat(0.0, 0.0, true_yaw)
        self.true_rotation = quat_to_rotmat(self.true_quaternion)

        self._validate_parameters()

        reach_scale = self.arm_length / 0.35
        self.shoulder_height = 0.10 * reach_scale
        self.upper_arm_length = 0.16 * reach_scale
        self.forearm_length = 0.14 * reach_scale
        self.camera_offset = 0.05 * reach_scale

        self.landmarks_world = self._make_landmarks()[: self.landmark_count]
        self.landmarks_kasuga_base = np.asarray(
            [
                self.true_rotation.T
                @ (point - self.kasuga_base_position)
                for point in self.landmarks_world
            ],
            dtype=float,
        )
        self.evidence_by_landmark = self._make_bingham_evidence()
        self.initial_gaze_target = np.mean(self.landmarks_world[:2], axis=0)

        self.tou_arm_state = self._arm_state(
            self.tou_base_position,
            np.eye(3, dtype=float),
            self.tou_arm_joints,
            self.initial_gaze_target,
        )
        self.kasuga_arm_state = self._arm_state(
            self.kasuga_base_position,
            self.true_rotation,
            self.kasuga_arm_joints,
            self.initial_gaze_target,
        )
        self.tou_gaze_forward = self.tou_arm_state[
            "camera_rotation"
        ][:, 0].copy()
        self.kasuga_gaze_forward = self.kasuga_arm_state[
            "camera_rotation"
        ][:, 0].copy()
        self.tou_gaze_forward = (
            self._rotation_z(self.tou_initial_gaze_yaw_offset)
            @ self.tou_gaze_forward
        )
        self.kasuga_gaze_forward = (
            self._rotation_z(self.kasuga_initial_gaze_yaw_offset)
            @ self.kasuga_gaze_forward
        )
        self.tou_arm_state = self._state_with_forward(
            self.tou_arm_state, self.tou_gaze_forward
        )
        self.kasuga_arm_state = self._state_with_forward(
            self.kasuga_arm_state, self.kasuga_gaze_forward
        )
        self._refresh_camera_locals()

        self.tou_factor_id = "tou_joint_bias"
        self.kasuga_factor_id = "kasuga_joint_bias"
        self.latent_store = GaussianLatentStore()
        initial_stamp = rospy.Time.now().to_sec()
        prior_covariance = (
            self.joint_prior_sigma ** 2
        ) * np.eye(6, dtype=float)
        self.latent_store.put_factor(
            self.tou_factor_id,
            np.zeros(6, dtype=float),
            prior_covariance,
            initial_stamp,
            Provenance(
                source_ids=("tou_joint_encoders",),
                method="joint_zero_prior",
                detail="Synthetic six-joint zero-offset prior for Tou.",
            ),
        )
        self.latent_store.put_factor(
            self.kasuga_factor_id,
            np.zeros(6, dtype=float),
            prior_covariance,
            initial_stamp,
            Provenance(
                source_ids=("kasuga_joint_encoders",),
                method="joint_zero_prior",
                detail="Synthetic six-joint zero-offset prior for Kasuga.",
            ),
        )
        self.graph = ProbTfGraph(
            max_records_per_edge=4,
            latent_store=self.latent_store,
        )

        self.active_parameter_matrix = np.zeros((4, 4), dtype=float)
        self.acquired_evidence = set()
        self.observed_landmarks = set()
        self.joint_used_landmarks = set()
        self.joint_update_queue = []
        self.tou_visible = set()
        self.kasuga_visible = set()
        self.common_visible = set()
        self.active_phase = PHASE_UNIFORM
        self.active_target_landmark = None
        self.last_tool_moments = {}
        self.relative_camera_moments = None
        self.joint_update_count = 0

        self.probtf_publisher = rospy.Publisher(
            "~probtf", ProbabilisticTransformStamped, queue_size=10
        )
        self.marker_publisher = rospy.Publisher(
            "~markers", MarkerArray, queue_size=2
        )

        self.start_time = rospy.Time.now().to_sec()
        self.last_timer_time = None
        self.last_log_signature = None
        self.timer = rospy.Timer(
            rospy.Duration.from_sec(1.0 / self.publish_rate),
            self._timer_callback,
        )

    def _validate_parameters(self):
        if self.publish_rate <= 0.0:
            raise ValueError("~publish_rate must be positive")
        if self.concentration <= 0.0:
            raise ValueError("~concentration must be positive")
        if self.sample_count <= 0:
            raise ValueError("~sample_count must be positive")
        if self.landmark_count < 4 or self.landmark_count > 11:
            raise ValueError("~landmark_count must lie in [4, 11]")
        if self.gaze_speed <= 0.0:
            raise ValueError("~gaze_speed_deg must be positive")
        if self.motion_cost_weight < 0.0:
            raise ValueError("~motion_cost_weight must be non-negative")
        if self.joint_information_weight < 0.0:
            raise ValueError("~joint_information_weight must be non-negative")
        if self.joint_prior_sigma <= 0.0:
            raise ValueError("~joint_prior_sigma_deg must be positive")
        if self.joint_bearing_noise <= 0.0:
            raise ValueError("~joint_bearing_noise_deg must be positive")
        if self.active_start_delay < 0.0:
            raise ValueError("~active_start_delay must be non-negative")
        for value, name in (
            (self.kasuga_base_position, "~kasuga_base_position"),
            (self.tou_base_position, "~tou_base_position"),
            (self.tou_arm_joints, "~tou_arm_joints_deg"),
            (self.kasuga_arm_joints, "~kasuga_arm_joints_deg"),
        ):
            if value.shape != (3,) or not np.all(np.isfinite(value)):
                raise ValueError("{} must contain three finite values".format(name))
        for value, name in (
            (self.tou_joint_bias_truth, "~tou_joint_bias_truth_deg"),
            (self.kasuga_joint_bias_truth, "~kasuga_joint_bias_truth_deg"),
        ):
            if value.shape != (6,) or not np.all(np.isfinite(value)):
                raise ValueError("{} must contain six finite values".format(name))

    @staticmethod
    def _normalize(value):
        vector = np.asarray(value, dtype=float)
        norm = float(np.linalg.norm(vector))
        if norm <= 1.0e-12:
            return np.array([1.0, 0.0, 0.0], dtype=float)
        return vector / norm

    @staticmethod
    def _rotation_x(angle):
        c = math.cos(angle)
        s = math.sin(angle)
        return np.array(
            [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]],
            dtype=float,
        )

    @staticmethod
    def _rotation_y(angle):
        c = math.cos(angle)
        s = math.sin(angle)
        return np.array(
            [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]],
            dtype=float,
        )

    @staticmethod
    def _rotation_z(angle):
        c = math.cos(angle)
        s = math.sin(angle)
        return np.array(
            [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
            dtype=float,
        )

    @staticmethod
    def _quat_from_rotation(rotation):
        matrix = np.asarray(rotation, dtype=float)
        trace = float(np.trace(matrix))
        if trace > 0.0:
            scale = math.sqrt(trace + 1.0) * 2.0
            quaternion = np.array(
                [
                    0.25 * scale,
                    (matrix[2, 1] - matrix[1, 2]) / scale,
                    (matrix[0, 2] - matrix[2, 0]) / scale,
                    (matrix[1, 0] - matrix[0, 1]) / scale,
                ],
                dtype=float,
            )
        else:
            index = int(np.argmax(np.diag(matrix)))
            if index == 0:
                scale = math.sqrt(
                    1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]
                ) * 2.0
                quaternion = np.array(
                    [
                        (matrix[2, 1] - matrix[1, 2]) / scale,
                        0.25 * scale,
                        (matrix[0, 1] + matrix[1, 0]) / scale,
                        (matrix[0, 2] + matrix[2, 0]) / scale,
                    ],
                    dtype=float,
                )
            elif index == 1:
                scale = math.sqrt(
                    1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]
                ) * 2.0
                quaternion = np.array(
                    [
                        (matrix[0, 2] - matrix[2, 0]) / scale,
                        (matrix[0, 1] + matrix[1, 0]) / scale,
                        0.25 * scale,
                        (matrix[1, 2] + matrix[2, 1]) / scale,
                    ],
                    dtype=float,
                )
            else:
                scale = math.sqrt(
                    1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]
                ) * 2.0
                quaternion = np.array(
                    [
                        (matrix[1, 0] - matrix[0, 1]) / scale,
                        (matrix[0, 2] + matrix[2, 0]) / scale,
                        (matrix[1, 2] + matrix[2, 1]) / scale,
                        0.25 * scale,
                    ],
                    dtype=float,
                )
        quaternion /= np.linalg.norm(quaternion)
        return quaternion

    def _look_at_rotation(self, origin, target):
        forward = self._normalize(
            np.asarray(target, dtype=float) - np.asarray(origin, dtype=float)
        )
        up_hint = np.array([0.0, 0.0, 1.0], dtype=float)
        if abs(float(np.dot(forward, up_hint))) > 0.96:
            up_hint = np.array([0.0, 1.0, 0.0], dtype=float)
        lateral = self._normalize(np.cross(up_hint, forward))
        upward = self._normalize(np.cross(forward, lateral))
        return np.column_stack([forward, lateral, upward])

    def _arm_state(self, base_position, base_rotation, joints, gaze_target):
        q1, q2, q3 = np.asarray(joints, dtype=float)
        shoulder = base_position + base_rotation @ np.array(
            [0.0, 0.0, self.shoulder_height], dtype=float
        )
        upper_rotation = (
            base_rotation @ self._rotation_z(q1) @ self._rotation_y(q2)
        )
        elbow = shoulder + upper_rotation @ np.array(
            [self.upper_arm_length, 0.0, 0.0], dtype=float
        )
        forearm_rotation = upper_rotation @ self._rotation_y(q3)
        wrist = elbow + forearm_rotation @ np.array(
            [self.forearm_length, 0.0, 0.0], dtype=float
        )
        camera_rotation = self._look_at_rotation(wrist, gaze_target)
        camera_position = wrist + camera_rotation[:, 0] * self.camera_offset
        return {
            "base": np.asarray(base_position, dtype=float),
            "shoulder": shoulder,
            "elbow": elbow,
            "wrist": wrist,
            "camera_position": camera_position,
            "camera_rotation": camera_rotation,
        }

    def _state_with_forward(self, state, forward):
        result = dict(state)
        rotation = self._look_at_rotation(
            state["wrist"], state["wrist"] + self._normalize(forward)
        )
        result["camera_rotation"] = rotation
        result["camera_position"] = (
            state["wrist"] + rotation[:, 0] * self.camera_offset
        )
        return result

    def _refresh_camera_locals(self):
        self.tou_camera_rotation_local = self.tou_arm_state[
            "camera_rotation"
        ].copy()
        self.kasuga_camera_rotation_local = (
            self.true_rotation.T @ self.kasuga_arm_state["camera_rotation"]
        )

    def _make_landmarks(self):
        anchor = np.array([-0.05, -0.30, 0.22], dtype=float)
        offsets = np.array(
            [
                [0.00, 0.00, 0.00],
                [0.00, 0.00, 0.38],
                [0.42, 0.00, 0.02],
                [0.30, 0.28, 0.15],
                [0.16, -0.36, 0.24],
                [-0.27, 0.25, 0.17],
                [-0.34, -0.20, 0.31],
                [0.14, 0.43, 0.33],
                [0.39, -0.31, 0.09],
                [-0.04, 0.51, 0.11],
                [-0.43, 0.08, 0.22],
            ],
            dtype=float,
        )
        return np.asarray(
            [anchor + self.true_rotation @ offset for offset in offsets],
            dtype=float,
        )

    def _make_bingham_evidence(self):
        result = {}
        for index in range(1, len(self.landmarks_world)):
            reference = self.landmarks_world[index] - self.landmarks_world[0]
            observed = (
                self.landmarks_kasuga_base[index]
                - self.landmarks_kasuga_base[0]
            )
            result[index] = vector_alignment_bingham_evidence(
                reference, observed, self.concentration
            )
        return result

    def _factor_id(self, name):
        if name == "tou":
            return self.tou_factor_id
        if name == "kasuga":
            return self.kasuga_factor_id
        raise ValueError("unknown robot {}".format(name))

    def _factor(self, name, snapshot=None):
        selected = self.latent_store.snapshot() if snapshot is None else snapshot
        return selected.factor(self._factor_id(name))

    def _truth_bias(self, name):
        if name == "tou":
            return self.tou_joint_bias_truth
        if name == "kasuga":
            return self.kasuga_joint_bias_truth
        raise ValueError("unknown robot {}".format(name))

    def _position_joints(self, name):
        if name == "tou":
            return self.tou_arm_joints
        if name == "kasuga":
            return self.kasuga_arm_joints
        raise ValueError("unknown robot {}".format(name))

    def _base_pose(self, name):
        if name == "tou":
            return self.tou_base_position, np.eye(3, dtype=float)
        if name == "kasuga":
            return self.kasuga_base_position, self.true_rotation
        raise ValueError("unknown robot {}".format(name))

    def _nominal_camera_rotation_local(self, name):
        if name == "tou":
            return self.tou_camera_rotation_local
        if name == "kasuga":
            return self.kasuga_camera_rotation_local
        raise ValueError("unknown robot {}".format(name))

    def _camera_pose_local_from_joint_bias(self, name, bias):
        bias = np.asarray(bias, dtype=float).reshape(6)
        q1, q2, q3 = self._position_joints(name) + bias[:3]
        shoulder = np.array([0.0, 0.0, self.shoulder_height], dtype=float)
        upper_rotation = self._rotation_z(q1) @ self._rotation_y(q2)
        elbow = shoulder + upper_rotation @ np.array(
            [self.upper_arm_length, 0.0, 0.0], dtype=float
        )
        forearm_rotation = upper_rotation @ self._rotation_y(q3)
        wrist = elbow + forearm_rotation @ np.array(
            [self.forearm_length, 0.0, 0.0], dtype=float
        )
        wrist_error = (
            self._rotation_x(bias[3])
            @ self._rotation_y(bias[4])
            @ self._rotation_z(bias[5])
        )
        camera_rotation = self._nominal_camera_rotation_local(name) @ wrist_error
        camera_position = wrist + camera_rotation[:, 0] * self.camera_offset
        return DeterministicTransform(
            camera_position, self._quat_from_rotation(camera_rotation)
        )

    def _camera_pose_jacobian(self, name, bias, epsilon=1.0e-6):
        bias = np.asarray(bias, dtype=float).reshape(6)
        nominal = self._camera_pose_local_from_joint_bias(name, bias)
        jacobian = np.zeros((6, 6), dtype=float)
        for column in range(6):
            delta = np.zeros(6, dtype=float)
            delta[column] = epsilon
            positive = self._camera_pose_local_from_joint_bias(
                name, bias + delta
            )
            negative = self._camera_pose_local_from_joint_bias(
                name, bias - delta
            )
            jacobian[:3, column] = (
                positive.translation - negative.translation
            ) / (2.0 * epsilon)
            plus_rotation = se3_log(
                relative_transform(nominal, positive)
            )[3:]
            minus_rotation = se3_log(
                relative_transform(nominal, negative)
            )[3:]
            jacobian[3:, column] = (
                plus_rotation - minus_rotation
            ) / (2.0 * epsilon)
        return nominal, jacobian

    def _camera_world_pose_from_joint_bias(self, name, bias):
        local = self._camera_pose_local_from_joint_bias(name, bias)
        base_translation, base_rotation = self._base_pose(name)
        world_rotation = base_rotation @ quat_to_rotmat(local.rotation_wxyz)
        world_translation = base_translation + base_rotation @ local.translation
        return DeterministicTransform(
            world_translation, self._quat_from_rotation(world_rotation)
        )

    def _camera_state_from_joint_bias(self, name, bias):
        pose = self._camera_world_pose_from_joint_bias(name, bias)
        return {
            "camera_position": pose.translation,
            "camera_rotation": quat_to_rotmat(pose.rotation_wxyz),
        }

    def _measured_local_ray(self, name, point):
        pose = self._camera_world_pose_from_joint_bias(
            name, self._truth_bias(name)
        )
        local = quat_to_rotmat(pose.rotation_wxyz).T @ (
            np.asarray(point, dtype=float) - pose.translation
        )
        if float(np.linalg.norm(local)) <= 1.0e-10:
            return None
        return self._normalize(local)

    def _epipolar_value(self, tou_bias, kasuga_bias, point, rays=None):
        if rays is None:
            tou_ray_local = self._measured_local_ray("tou", point)
            kasuga_ray_local = self._measured_local_ray("kasuga", point)
        else:
            tou_ray_local, kasuga_ray_local = rays
        if tou_ray_local is None or kasuga_ray_local is None:
            return None
        tou_pose = self._camera_world_pose_from_joint_bias("tou", tou_bias)
        kasuga_pose = self._camera_world_pose_from_joint_bias(
            "kasuga", kasuga_bias
        )
        tou_ray_world = (
            quat_to_rotmat(tou_pose.rotation_wxyz) @ tou_ray_local
        )
        kasuga_ray_world = (
            quat_to_rotmat(kasuga_pose.rotation_wxyz) @ kasuga_ray_local
        )
        baseline = kasuga_pose.translation - tou_pose.translation
        baseline_norm = float(np.linalg.norm(baseline))
        if baseline_norm <= 1.0e-8:
            return None
        baseline /= baseline_norm
        value = float(
            np.dot(tou_ray_world, np.cross(baseline, kasuga_ray_world))
        )
        return np.array([value], dtype=float)

    def _epipolar_linearization(self, landmark_index, epsilon=1.0e-6):
        snapshot = self.latent_store.snapshot()
        tou_mean = snapshot.factor(self.tou_factor_id).mean
        kasuga_mean = snapshot.factor(self.kasuga_factor_id).mean
        point = self.landmarks_world[landmark_index]
        rays = (
            self._measured_local_ray("tou", point),
            self._measured_local_ray("kasuga", point),
        )
        residual = self._epipolar_value(
            tou_mean, kasuga_mean, point, rays=rays
        )
        if residual is None:
            return None
        tou_jacobian = np.zeros((1, 6), dtype=float)
        kasuga_jacobian = np.zeros((1, 6), dtype=float)
        for column in range(6):
            delta = np.zeros(6, dtype=float)
            delta[column] = epsilon
            plus = self._epipolar_value(
                tou_mean + delta, kasuga_mean, point, rays=rays
            )
            minus = self._epipolar_value(
                tou_mean - delta, kasuga_mean, point, rays=rays
            )
            if plus is None or minus is None:
                return None
            tou_jacobian[:, column] = (plus - minus) / (2.0 * epsilon)

            plus = self._epipolar_value(
                tou_mean, kasuga_mean + delta, point, rays=rays
            )
            minus = self._epipolar_value(
                tou_mean, kasuga_mean - delta, point, rays=rays
            )
            if plus is None or minus is None:
                return None
            kasuga_jacobian[:, column] = (
                plus - minus
            ) / (2.0 * epsilon)
        return snapshot, residual, tou_jacobian, kasuga_jacobian

    def _joint_information_gain(self, landmark_index):
        linearization = self._epipolar_linearization(landmark_index)
        if linearization is None:
            return 0.0
        snapshot, _, tou_jacobian, kasuga_jacobian = linearization
        factor_ids = (self.tou_factor_id, self.kasuga_factor_id)
        _, covariance, slices = snapshot.joint_mean_covariance(factor_ids)
        jacobian = np.zeros((1, covariance.shape[0]), dtype=float)
        jacobian[:, slices[self.tou_factor_id]] = tou_jacobian
        jacobian[:, slices[self.kasuga_factor_id]] = kasuga_jacobian
        projected = float((jacobian @ covariance @ jacobian.T)[0, 0])
        noise = self.joint_bearing_noise ** 2
        return 0.5 * math.log1p(max(0.0, projected) / noise)

    def _apply_joint_observation(self, landmark_index, stamp):
        if landmark_index in self.joint_used_landmarks:
            return False
        linearization = self._epipolar_linearization(landmark_index)
        if linearization is None:
            return False
        snapshot, residual, tou_jacobian, kasuga_jacobian = linearization
        observation = GaussianObservationFactor(
            observation_id="shared_moon_{:02d}_closure".format(
                landmark_index
            ),
            latent_factor_ids=(self.tou_factor_id, self.kasuga_factor_id),
            residual=residual,
            jacobian_blocks=(tou_jacobian, kasuga_jacobian),
            noise_covariance=np.array(
                [[self.joint_bearing_noise ** 2]], dtype=float
            ),
            stamp=stamp,
            provenance=Provenance(
                source_ids=("moon_{}".format(landmark_index),),
                method="shared_landmark_epipolar_closure",
                detail=(
                    "One shared keypoint constrains both eye-in-hand chains in "
                    "one atomic dependency-aware Gaussian update."
                ),
            ),
        )
        expected_versions = dict(
            snapshot.factor_versions(
                (self.tou_factor_id, self.kasuga_factor_id)
            )
        )
        self.latent_store.apply_observation(
            observation, expected_versions=expected_versions
        )
        self.joint_used_landmarks.add(landmark_index)
        self.joint_update_count += 1
        return True

    def _translation(self, mean):
        return ConditionalGaussianTranslation(
            mean_at_reference=np.asarray(mean, dtype=float),
            residual_covariance=np.zeros((3, 3), dtype=float),
            rotation_coupling=np.zeros((3, 9), dtype=float),
        )

    def _record(
        self,
        parent,
        child,
        edge_id,
        orientation,
        translation,
        stamp,
        source_id,
        method,
    ):
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
            authority="probtf_furisake_dependency_demo",
            distribution=TransformDistribution((component,)),
            provenance=TransformProvenance(
                source_ids=(source_id,), method=method
            ),
            is_static=False,
        )

    def _tool_edge_spec(self, name):
        if name == "tou":
            return (
                self.tou_base_frame,
                self.tou_tool_frame,
                "tou_base__to__tou_tool",
            )
        if name == "kasuga":
            return (
                self.kasuga_base_frame,
                self.kasuga_tool_frame,
                "kasuga_base__to__kasuga_tool",
            )
        raise ValueError("unknown robot {}".format(name))

    def _physical_tool_record_and_binding(self, name, stamp):
        parent, child, edge_id = self._tool_edge_spec(name)
        zero_bias = np.zeros(6, dtype=float)
        transform, sensitivity = self._camera_pose_jacobian(name, zero_bias)
        factor = self._factor(name)
        record = self._record(
            parent,
            child,
            edge_id,
            BinghamOrientation.dirac(transform.rotation_wxyz.tolist()),
            transform.translation,
            stamp,
            self._factor_id(name),
            "joint_zero_linearization_pose",
        )
        binding = EdgeLatentBinding(
            edge_id=edge_id,
            factor_id=self._factor_id(name),
            sensitivity=sensitivity,
            factor_version=factor.version,
            linearization_stamp=stamp,
            linearization_pose=transform,
        )
        return record, binding

    def _marginal_record(self, name, summary, stamp):
        parent, child, edge_id = self._tool_edge_spec(name)
        component = summary.to_component(edge_id + "_marginal_component")
        return TransformDistributionStamped(
            parent_frame_id=parent,
            child_frame_id=child,
            stamp=stamp,
            edge_id=edge_id,
            authority="probtf_furisake_dependency_demo",
            distribution=TransformDistribution((component,)),
            provenance=summary.provenance,
            is_static=False,
            approximation=summary.approximation,
        )

    def _sync_core_and_publish(self, orientation, stamp):
        tou_base_record = self._record(
            self.world_frame,
            self.tou_base_frame,
            "world__to__tou_base",
            BinghamOrientation.dirac([1.0, 0.0, 0.0, 0.0]),
            self.tou_base_position,
            stamp,
            "synthetic_tou_base",
            "fixed_reference_base",
        )
        kasuga_base_record = self._record(
            self.world_frame,
            self.kasuga_base_frame,
            "world__to__kasuga_base",
            orientation,
            self.kasuga_base_position,
            stamp,
            "shared_moon_bingham",
            "global_orientation_posterior",
        )
        tou_tool_record, tou_binding = self._physical_tool_record_and_binding(
            "tou", stamp
        )
        kasuga_tool_record, kasuga_binding = (
            self._physical_tool_record_and_binding("kasuga", stamp)
        )

        for record in (
            tou_base_record,
            kasuga_base_record,
            tou_tool_record,
            kasuga_tool_record,
        ):
            self.graph.insert(record)
        self.latent_store.bind_edge(tou_binding)
        self.latent_store.bind_edge(kasuga_binding)

        tou_summary = self.graph.lookup_transform_moments(
            self.tou_base_frame, self.tou_tool_frame, stamp
        )
        kasuga_summary = self.graph.lookup_transform_moments(
            self.kasuga_base_frame, self.kasuga_tool_frame, stamp
        )
        self.last_tool_moments = {
            "tou": tou_summary,
            "kasuga": kasuga_summary,
        }

        self.relative_camera_moments = None
        if self.active_phase == PHASE_LOCALIZED:
            try:
                self.relative_camera_moments = (
                    self.graph.lookup_transform_moments(
                        self.tou_tool_frame,
                        self.kasuga_tool_frame,
                        stamp,
                    )
                )
            except Exception as error:
                rospy.logdebug_throttle(
                    2.0,
                    "relative camera moment query unavailable: %s",
                    error,
                )

        for record in (
            tou_base_record,
            kasuga_base_record,
            self._marginal_record("tou", tou_summary, stamp),
            self._marginal_record("kasuga", kasuga_summary, stamp),
        ):
            self.probtf_publisher.publish(
                transform_distribution_to_msg(record)
            )

    @staticmethod
    def _bingham_information_score(parameter_matrix):
        symmetric = 0.5 * (parameter_matrix + parameter_matrix.T)
        eigenvalues = np.linalg.eigvalsh(symmetric)
        top = float(eigenvalues[-1])
        total_spread = float(np.sum(top - eigenvalues[:-1]))
        unique_mode_gap = float(top - eigenvalues[-2])
        return 0.25 * total_spread + 2.0 * unique_mode_gap

    def _angle_between(self, first, second):
        first = self._normalize(first)
        second = self._normalize(second)
        return math.acos(
            float(np.clip(np.dot(first, second), -1.0, 1.0))
        )

    def _select_informative_landmark(self):
        current_score = self._bingham_information_score(
            self.active_parameter_matrix
        )
        best_index = None
        best_score = -float("inf")
        for index, evidence in sorted(self.evidence_by_landmark.items()):
            if index in self.acquired_evidence:
                continue
            candidate = self.active_parameter_matrix + evidence
            bingham_gain = (
                self._bingham_information_score(candidate) - current_score
            )
            joint_gain = (
                self._joint_information_gain(index)
                if self.active_phase == PHASE_LOCALIZED
                else 0.0
            )
            point = self.landmarks_world[index]
            tou_desired = self._normalize(
                point - self.tou_arm_state["wrist"]
            )
            kasuga_desired = self._normalize(
                point - self.kasuga_arm_state["wrist"]
            )
            motion_cost = self._angle_between(
                self.tou_gaze_forward, tou_desired
            ) + self._angle_between(
                self.kasuga_gaze_forward, kasuga_desired
            )
            score = (
                bingham_gain
                + self.joint_information_weight * joint_gain
                - self.motion_cost_weight * motion_cost
            )
            if score > best_score:
                best_score = score
                best_index = index
        return best_index

    def _rotate_toward(self, current, desired, angle):
        current = self._normalize(current)
        desired = self._normalize(desired)
        total = self._angle_between(current, desired)
        if total <= angle or total <= 1.0e-10:
            return desired
        axis = np.cross(current, desired)
        norm = float(np.linalg.norm(axis))
        if norm <= 1.0e-12:
            return desired
        axis /= norm
        return self._normalize(
            current * math.cos(angle)
            + np.cross(axis, current) * math.sin(angle)
            + axis * np.dot(axis, current) * (1.0 - math.cos(angle))
        )

    def _cooperative_gaze_step(self, target, dt):
        tou_desired = self._normalize(
            target - self.tou_arm_state["wrist"]
        )
        kasuga_desired = self._normalize(
            target - self.kasuga_arm_state["wrist"]
        )
        tou_error = self._angle_between(
            self.tou_gaze_forward, tou_desired
        )
        kasuga_error = self._angle_between(
            self.kasuga_gaze_forward, kasuga_desired
        )
        largest = max(tou_error, kasuga_error)
        if largest <= 1.0e-10:
            return
        fraction = min(
            1.0, self.gaze_speed * max(0.0, dt) / largest
        )
        self.tou_gaze_forward = self._rotate_toward(
            self.tou_gaze_forward,
            tou_desired,
            fraction * tou_error,
        )
        self.kasuga_gaze_forward = self._rotate_toward(
            self.kasuga_gaze_forward,
            kasuga_desired,
            fraction * kasuga_error,
        )
        self.tou_arm_state = self._state_with_forward(
            self.tou_arm_state, self.tou_gaze_forward
        )
        self.kasuga_arm_state = self._state_with_forward(
            self.kasuga_arm_state, self.kasuga_gaze_forward
        )
        self._refresh_camera_locals()

    def _camera_visible(self, state, point):
        local = state["camera_rotation"].T @ (
            np.asarray(point, dtype=float) - state["camera_position"]
        )
        if local[0] <= 1.0e-8:
            return False
        horizontal = abs(math.atan2(local[1], local[0]))
        vertical = abs(math.atan2(local[2], local[0]))
        return (
            horizontal <= 0.5 * self.camera_hfov
            and vertical <= 0.5 * self.camera_vfov
        )

    def _update_visibility(self):
        tou_state = self._camera_state_from_joint_bias(
            "tou", self.tou_joint_bias_truth
        )
        kasuga_state = self._camera_state_from_joint_bias(
            "kasuga", self.kasuga_joint_bias_truth
        )
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
        self.observed_landmarks.update(self.common_visible)

    def _phase_from_evidence(self):
        count = len(self.acquired_evidence)
        if count == 0:
            return PHASE_UNIFORM
        if count == 1:
            return PHASE_ONE_VECTOR
        return PHASE_LOCALIZED

    def _orientation(self):
        if not self.acquired_evidence:
            return BinghamOrientation.uniform()
        return BinghamOrientation.from_parameter_matrix(
            self.active_parameter_matrix
        )

    def _acquire_target_if_visible(self, now):
        if now - self.start_time < self.active_start_delay:
            return
        if not self.acquired_evidence:
            if 0 in self.common_visible and 1 in self.common_visible:
                self.active_parameter_matrix += self.evidence_by_landmark[1]
                self.acquired_evidence.add(1)
                self.active_target_landmark = None
        elif (
            self.active_target_landmark is not None
            and self.active_target_landmark in self.common_visible
            and 0 in self.observed_landmarks
        ):
            index = self.active_target_landmark
            self.active_parameter_matrix += self.evidence_by_landmark[index]
            self.acquired_evidence.add(index)
            self.active_target_landmark = None

        self.active_phase = self._phase_from_evidence()
        if self.active_phase == PHASE_LOCALIZED:
            for index in sorted(self.acquired_evidence):
                if (
                    index not in self.joint_used_landmarks
                    and index not in self.joint_update_queue
                ):
                    self.joint_update_queue.append(index)

    def _active_step(self, now, dt):
        if not self.acquired_evidence:
            target = self.initial_gaze_target
            self.active_target_landmark = None
        else:
            if self.active_target_landmark is None:
                self.active_target_landmark = (
                    self._select_informative_landmark()
                )
            target = (
                None
                if self.active_target_landmark is None
                else self.landmarks_world[self.active_target_landmark]
            )
        if target is not None:
            self._cooperative_gaze_step(target, dt)
        self._update_visibility()
        self._acquire_target_if_visible(now)
        if self.active_phase == PHASE_LOCALIZED and self.joint_update_queue:
            index = self.joint_update_queue.pop(0)
            self._apply_joint_observation(index, now)
        return self._orientation()

    def _joint_sigma_rms_deg(self, name):
        covariance = self._factor(name).covariance
        value = max(0.0, float(np.trace(covariance)) / 6.0)
        return math.degrees(math.sqrt(value))

    def _cross_covariance_norm(self):
        snapshot = self.latent_store.snapshot()
        _, covariance, slices = snapshot.joint_mean_covariance(
            (self.tou_factor_id, self.kasuga_factor_id)
        )
        block = covariance[
            slices[self.tou_factor_id], slices[self.kasuga_factor_id]
        ]
        return float(np.linalg.norm(block, ord="fro"))

    def _relative_position_sigma_mm(self):
        if self.relative_camera_moments is None:
            return None
        covariance = self.relative_camera_moments.covariance[:3, :3]
        rms = math.sqrt(max(0.0, float(np.trace(covariance)) / 3.0))
        return 1000.0 * rms

    @staticmethod
    def _color(red, green, blue, alpha=1.0):
        return ColorRGBA(r=red, g=green, b=blue, a=alpha)

    @staticmethod
    def _point(value):
        return Point(x=float(value[0]), y=float(value[1]), z=float(value[2]))

    def _sphere_marker(
        self, marker_id, namespace, center, color, stamp, diameter
    ):
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

    def _line_marker(
        self, marker_id, namespace, points, color, stamp, width=0.004
    ):
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

    def _cylinder_marker(
        self,
        marker_id,
        namespace,
        start,
        end,
        color,
        stamp,
        radius=0.018,
    ):
        start = np.asarray(start, dtype=float)
        end = np.asarray(end, dtype=float)
        delta = end - start
        length = float(np.linalg.norm(delta))
        if length <= 1.0e-9:
            return self._sphere_marker(
                marker_id, namespace, start, color, stamp, 2.0 * radius
            )
        direction = delta / length
        z_axis = np.array([0.0, 0.0, 1.0], dtype=float)
        dot = float(np.clip(np.dot(z_axis, direction), -1.0, 1.0))
        if dot < -0.999999:
            quaternion = np.array([0.0, 1.0, 0.0, 0.0], dtype=float)
        else:
            cross = np.cross(z_axis, direction)
            scale = math.sqrt(2.0 * (1.0 + dot))
            quaternion = np.array(
                [
                    0.5 * scale,
                    cross[0] / scale,
                    cross[1] / scale,
                    cross[2] / scale,
                ],
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

    def _box_marker(
        self, marker_id, namespace, center, rotation, color, stamp
    ):
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
        marker.scale.x = 0.055
        marker.scale.y = 0.038
        marker.scale.z = 0.030
        marker.color = color
        return marker

    def _ellipsoid_marker(
        self, marker_id, namespace, center, covariance, color, stamp
    ):
        covariance = 0.5 * (covariance + covariance.T)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        eigenvalues = np.maximum(eigenvalues, 0.0)
        if np.linalg.det(eigenvectors) < 0.0:
            eigenvectors[:, 0] *= -1.0
        diameters = np.maximum(5.0 * np.sqrt(eigenvalues), 0.004)
        marker = self._sphere_marker(
            marker_id, namespace, center, color, stamp, 1.0
        )
        quaternion = self._quat_from_rotation(eigenvectors)
        marker.pose.orientation.w = float(quaternion[0])
        marker.pose.orientation.x = float(quaternion[1])
        marker.pose.orientation.y = float(quaternion[2])
        marker.pose.orientation.z = float(quaternion[3])
        marker.scale.x = float(diameters[0])
        marker.scale.y = float(diameters[1])
        marker.scale.z = float(diameters[2])
        return marker

    def _robot_markers(self, name, state, base_id, color, stamp):
        markers = []
        points = [
            state["base"],
            state["shoulder"],
            state["elbow"],
            state["wrist"],
            state["camera_position"],
        ]
        for index in range(len(points) - 1):
            markers.append(
                self._cylinder_marker(
                    base_id + index,
                    "{}_links".format(name),
                    points[index],
                    points[index + 1],
                    self._color(color.r, color.g, color.b, 0.75),
                    stamp,
                    radius=0.017,
                )
            )
        for index, point in enumerate(points[:-1]):
            markers.append(
                self._sphere_marker(
                    base_id + 10 + index,
                    "{}_joints".format(name),
                    point,
                    self._color(0.92, 0.92, 0.96, 0.92),
                    stamp,
                    0.045,
                )
            )
        markers.append(
            self._box_marker(
                base_id + 20,
                "{}_camera".format(name),
                state["camera_position"],
                state["camera_rotation"],
                self._color(0.08, 0.08, 0.10, 0.98),
                stamp,
            )
        )
        forward = state["camera_rotation"][:, 0]
        markers.append(
            self._line_marker(
                base_id + 21,
                "{}_camera_axis".format(name),
                [
                    state["camera_position"],
                    state["camera_position"] + 0.12 * forward,
                ],
                self._color(color.r, color.g, color.b, 0.85),
                stamp,
                width=0.008,
            )
        )
        return markers

    def _landmark_markers(self, stamp):
        markers = []
        for index, point in enumerate(self.landmarks_world):
            if index == self.active_target_landmark:
                color = self._color(0.25, 1.0, 0.35, 1.0)
                diameter = 0.082
            elif index in self.common_visible:
                color = self._color(1.0, 0.88, 0.12, 1.0)
                diameter = 0.065
            elif index in self.acquired_evidence or index == 0:
                color = self._color(1.0, 0.52, 0.10, 0.78)
                diameter = 0.060
            else:
                color = self._color(0.48, 0.48, 0.50, 0.40)
                diameter = 0.055
            markers.append(
                self._sphere_marker(
                    100 + index,
                    "shared_keypoints",
                    point,
                    color,
                    stamp,
                    diameter,
                )
            )
            label = Marker()
            label.header.frame_id = self.world_frame
            label.header.stamp = rospy.Time.from_sec(stamp)
            label.ns = "shared_keypoint_labels"
            label.id = 150 + index
            label.type = Marker.TEXT_VIEW_FACING
            label.action = Marker.ADD
            label.pose.position = self._point(
                point + np.array([0.0, 0.0, 0.075], dtype=float)
            )
            label.pose.orientation.w = 1.0
            label.scale.z = 0.034
            label.color = self._color(1.0, 1.0, 1.0, 0.86)
            label.text = "moon_{}".format(index)
            markers.append(label)
        return markers

    def _observation_markers(self, stamp):
        markers = []
        tou_state = self._camera_state_from_joint_bias(
            "tou", self.tou_joint_bias_truth
        )
        kasuga_state = self._camera_state_from_joint_bias(
            "kasuga", self.kasuga_joint_bias_truth
        )
        for marker_id, namespace, state, visible, color in (
            (
                210,
                "tou_observations",
                tou_state,
                self.tou_visible,
                self._color(0.25, 0.75, 1.0, 0.38),
            ),
            (
                211,
                "kasuga_observations",
                kasuga_state,
                self.kasuga_visible,
                self._color(1.0, 0.45, 0.75, 0.38),
            ),
        ):
            points = []
            for index in sorted(visible):
                points.extend(
                    [state["camera_position"], self.landmarks_world[index]]
                )
            if points:
                markers.append(
                    self._line_marker(
                        marker_id, namespace, points, color, stamp
                    )
                )
        return markers

    def _uncertainty_markers(self, stamp):
        markers = []
        for name, marker_id, color in (
            ("tou", 300, self._color(0.25, 0.75, 1.0, 0.24)),
            ("kasuga", 310, self._color(1.0, 0.45, 0.75, 0.24)),
        ):
            summary = self.last_tool_moments.get(name)
            if summary is None:
                continue
            base_translation, base_rotation = self._base_pose(name)
            center = base_translation + base_rotation @ summary.mean.translation
            covariance = (
                base_rotation
                @ summary.covariance[:3, :3]
                @ base_rotation.T
            )
            markers.append(
                self._ellipsoid_marker(
                    marker_id,
                    "{}_core_joint_uncertainty".format(name),
                    center,
                    covariance,
                    color,
                    stamp,
                )
            )
        return markers

    def _support_marker(self, orientation, stamp):
        samples = sample_bingham_orientation(
            orientation,
            self.sample_count,
            rng=self.random_seed + len(self.acquired_evidence),
        )
        summary = self.last_tool_moments.get("kasuga")
        local = (
            self._camera_pose_local_from_joint_bias(
                "kasuga", np.zeros(6, dtype=float)
            ).translation
            if summary is None
            else summary.mean.translation
        )
        marker = Marker()
        marker.header.frame_id = self.world_frame
        marker.header.stamp = rospy.Time.from_sec(stamp)
        marker.ns = "kasuga_global_tool_support"
        marker.id = 400
        marker.type = Marker.POINTS
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.011
        marker.scale.y = 0.011
        marker.color = self._color(1.0, 0.42, 0.08, 0.48)
        marker.points = [
            self._point(
                self.kasuga_base_position
                + quat_to_rotmat(quaternion) @ local
            )
            for quaternion in samples
        ]
        return marker

    def _status_marker(self, stamp):
        if self.active_phase == PHASE_UNIFORM:
            state = "SO(3) uniform: cooperatively seeking the vertical pair"
        elif self.active_phase == PHASE_ONE_VECTOR:
            state = "S1 ridge: seeking a non-collinear shared keypoint"
        else:
            state = "localized Bingham: continuing keypoints for joint smoothing"
        target = (
            "none"
            if self.active_target_landmark is None
            else "moon_{}".format(self.active_target_landmark)
        )
        relative_sigma = self._relative_position_sigma_mm()
        relative_text = (
            "n/a"
            if relative_sigma is None
            else "{:.1f} mm".format(relative_sigma)
        )
        text = (
            "{}\n"
            "target={} | Bingham evidence {}/{} | joint closures={}\n"
            "joint RMS: Tou {:.2f} deg | Kasuga {:.2f} deg | "
            "cross-cov Fro={:.2e} | relative camera sigma={}"
        ).format(
            state,
            target,
            len(self.acquired_evidence),
            len(self.evidence_by_landmark),
            self.joint_update_count,
            self._joint_sigma_rms_deg("tou"),
            self._joint_sigma_rms_deg("kasuga"),
            self._cross_covariance_norm(),
            relative_text,
        )
        marker = Marker()
        marker.header.frame_id = self.world_frame
        marker.header.stamp = rospy.Time.from_sec(stamp)
        marker.ns = "status"
        marker.id = 0
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        marker.pose.position = self._point([0.0, 0.0, 1.03])
        marker.pose.orientation.w = 1.0
        marker.scale.z = 0.052
        marker.color = self._color(1.0, 1.0, 1.0, 1.0)
        marker.text = text
        return marker

    def _publish_markers(self, orientation, stamp):
        markers = [self._status_marker(stamp)]
        markers.extend(self._landmark_markers(stamp))
        markers.extend(
            self._robot_markers(
                "tou",
                self.tou_arm_state,
                10,
                self._color(0.25, 0.75, 1.0, 0.95),
                stamp,
            )
        )
        markers.extend(
            self._robot_markers(
                "kasuga",
                self.kasuga_arm_state,
                50,
                self._color(1.0, 0.45, 0.75, 0.95),
                stamp,
            )
        )
        markers.extend(self._observation_markers(stamp))
        markers.extend(self._uncertainty_markers(stamp))
        markers.append(self._support_marker(orientation, stamp))
        self.marker_publisher.publish(MarkerArray(markers=markers))

    def _timer_callback(self, _event):
        now = rospy.Time.now().to_sec()
        if self.last_timer_time is None:
            dt = 1.0 / self.publish_rate
        else:
            dt = max(0.0, min(0.25, now - self.last_timer_time))
        self.last_timer_time = now

        orientation = self._active_step(now, dt)
        self._sync_core_and_publish(orientation, now)
        self._publish_markers(orientation, now)

        signature = (
            self.active_phase,
            self.active_target_landmark,
            len(self.acquired_evidence),
            self.joint_update_count,
        )
        if signature != self.last_log_signature:
            rospy.loginfo(
                "Furisake core demo: phase=%d target=%s evidence=%d/%d "
                "joint_updates=%d",
                self.active_phase,
                self.active_target_landmark,
                len(self.acquired_evidence),
                len(self.evidence_by_landmark),
                self.joint_update_count,
            )
            self.last_log_signature = signature


def main():
    rospy.init_node("probtf_furisake_dependency_demo")
    FurisakeDependencyDemoNode()
    rospy.spin()


if __name__ == "__main__":
    main()
