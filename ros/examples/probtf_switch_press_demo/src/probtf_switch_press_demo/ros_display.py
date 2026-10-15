"""ROS publication only. True poses enter here solely to place the visualization."""
import cv2
from pathlib import Path
import numpy as np
import rospy
import rospkg
from geometry_msgs.msg import Point, TransformStamped
from sensor_msgs.msg import CameraInfo, Image, JointState
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray
from probtf_msgs.msg import ProbabilisticTransformArray, ProbabilisticTransformStamped
from probtf_ros import transform_distribution_to_msg
from probtf_ros.v2_conversions import transform_array_to_msg
from scipy.spatial.transform import Rotation
from tf2_ros import StaticTransformBroadcaster

from probtf_switch_press_demo.scene import apply
from probtf_switch_press_demo.beliefs import body_frame_records, observation_pose_record
from symaware_grasp.probtf_config import load_prob_tf_mapping


class Display:
    def __init__(self, trial, texture_path):
        self.joints = rospy.Publisher("joint_states", JointState, queue_size=1, latch=True)
        self.rigid_joints = rospy.Publisher("rigid_joint_states", JointState, queue_size=1, latch=True)
        self.rigid_root = StaticTransformBroadcaster()
        root = TransformStamped()
        root.header.stamp, root.header.frame_id = rospy.Time.now(), "base_link"
        root.child_frame_id, root.transform.rotation.w = "rigid/base_link", 1.
        base = TransformStamped()
        base.header.stamp, base.header.frame_id = rospy.Time.now(), "world"
        base.child_frame_id, base.transform.rotation.w = "base_link", 1.
        base.transform.translation.x, base.transform.translation.y, base.transform.translation.z = trial.config["base_translation"]
        self.rigid_root.sendTransform([root, base])
        self.markers = rospy.Publisher("markers", MarkerArray, queue_size=1, latch=True)
        self.predictions = rospy.Publisher("predictions", MarkerArray, queue_size=1, latch=True)
        self.image = rospy.Publisher("image_raw", Image, queue_size=1, latch=True)
        self.debug = rospy.Publisher("detection_image", Image, queue_size=1, latch=True)
        self.info = rospy.Publisher("camera_info", CameraInfo, queue_size=1, latch=True)
        self.status = rospy.Publisher("status", String, queue_size=1, latch=True)
        self.marker_belief = rospy.Publisher("marker_belief", ProbabilisticTransformStamped, queue_size=1, latch=True)
        self.goal_belief = rospy.Publisher("goal_belief", ProbabilisticTransformStamped, queue_size=1, latch=True)
        self.probtf = rospy.Publisher("probtf_batch", ProbabilisticTransformArray, queue_size=1, latch=True)
        self.marker_display = rospy.Publisher("marker_display", ProbabilisticTransformStamped, queue_size=1)
        self.joint_mapping = load_prob_tf_mapping(Path(rospkg.RosPack().get_path("symaware_grasp")) /
                                                  "configs/simple_six_dof_prob_tf.yaml")
        self.tag_texture = cv2.imread(str(texture_path), cv2.IMREAD_GRAYSCALE)
        self.trial = trial
        self.marker_id = 0

    def marker(self, namespace, kind, color, scale, position=(0, 0, 0), pose=None, stamp=None):
        marker = Marker()
        marker.header.frame_id = "base_link"
        marker.header.stamp = rospy.Time.now() if stamp is None else rospy.Time.from_sec(stamp)
        marker.ns, marker.id, marker.type = namespace, self.marker_id, kind
        self.marker_id += 1
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.pose.position.x, marker.pose.position.y, marker.pose.position.z = position
        if pose is not None:
            marker.pose.position.x, marker.pose.position.y, marker.pose.position.z = pose[:3, 3]
            q = Rotation.from_matrix(pose[:3, :3]).as_quat()
            marker.pose.orientation.x, marker.pose.orientation.y, marker.pose.orientation.z, marker.pose.orientation.w = q
        marker.scale.x, marker.scale.y, marker.scale.z = scale
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = color
        return marker

    @staticmethod
    def points(marker, points):
        marker.points = [Point(*map(float, point)) for point in points]
        return marker

    def scene_markers(self):
        trial, config = self.trial, self.trial.config
        self.marker_id = 0
        output = []
        reset = Marker(action=Marker.DELETEALL)
        output.append(reset)
        for obstacle in config["obstacles"]:
            pose = trial.marker.copy()
            pose[:3, 3] = apply(trial.marker, obstacle["center"])
            color = {"panel": (.26, .29, .34, 1), "button": (.85, .13, .10, 1),
                     "lamp": ((.05, 1, .12, 1) if trial.status == "reached" and trial.monitor.reached and not trial.monitor.collided else (.07, .12, .08, 1))}[obstacle["name"]]
            output.append(self.marker("scene", Marker.CUBE, color, obstacle["size"], pose=pose))
        size = config["marker"]["size"]
        board = trial.marker.copy()
        board[:3, 3] = apply(trial.marker, [0, 0, .0005])
        output.append(self.marker("tag", Marker.CUBE, (1, 1, 1, 1), (size * 1.25, size * 1.25, .001), pose=board))
        black = self.marker("tag", Marker.TRIANGLE_LIST, (.01, .01, .01, 1), (1, 1, 1))
        triangles = []
        height = self.tag_texture.shape[0]
        module = height / 10
        for row in range(8):
            for column in range(8):
                pixel = self.tag_texture[int((row + 1.5) * module), int((column + 1.5) * module)]
                if pixel > 128:
                    continue
                x0, x1 = (column / 8 - .5) * size, ((column + 1) / 8 - .5) * size
                y0, y1 = (.5 - (row + 1) / 8) * size, (.5 - row / 8) * size
                triangles.extend(((x0, y0, .0015), (x1, y0, .0015), (x1, y1, .0015),
                                  (x0, y0, .0015), (x1, y1, .0015), (x0, y1, .0015)))
        output.append(self.points(black, apply(trial.marker, triangles)))
        diameter = config["reach_tolerance"] * 2
        output.append(self.marker("true_goal", Marker.SPHERE, (.1, 1, .25, .25), (diameter,) * 3, position=trial.goal))
        actual_tip = apply(trial.plant.flange(trial.pose), trial.tip_offset)
        output.append(self.marker("actual_tip", Marker.SPHERE, (.05, .75, .8, 1), (.012,) * 3, position=actual_tip))
        trajectory = self.marker("actual_path", Marker.LINE_STRIP, (.02, .65, .75, 1), (.003, 0, 0))
        if len(trial.trajectory) >= 2:
            output.append(self.points(trajectory, [row["tip"] for row in trial.trajectory]))
        error = np.linalg.norm(actual_tip - trial.goal)
        label = self.marker("status", Marker.TEXT_VIEW_FACING, (.16, .18, .22, 1), (0, 0, .033), position=(.72, 0, 1.02))
        phase = {"planning": "INITIAL IK", "tracking": "CONTINUOUS APPROACH + IK",
                 "initial": "INITIAL POSE", "ready": "READY"}.get(trial.phase, trial.phase.upper())
        label.text = "{} | update {} | IK iteration {}\nActual error {:.1f} mm\nOpaque: flexible | Translucent: rigid".format(
            phase, trial.cycle + 1, trial.ik_iteration, error * 1000)
        output.append(label)
        return MarkerArray(markers=output)

    def prediction_markers(self):
        """Optional IK diagnostics, in a separate display disabled by default."""
        self.marker_id = 0
        output = [Marker(action=Marker.DELETEALL)]
        preview = self.trial.preview
        if preview is not None and self.trial.status == "running":
            for index, (rod, color) in enumerate(zip(preview["prediction"].rods,
                        ((.18, .48, 1, .8), (1, .64, .02, .8), (.65, .15, 1, .8)))):
                line = self.marker("next_motion_rod_" + str(index), Marker.LINE_LIST, color, (.002, 0, 0), stamp=preview["stamp"])
                output.append(self.points(line, apply(preview["hand"], rod)))
        return MarkerArray(markers=output)

    @staticmethod
    def image_message(array, stamp):
        array = np.ascontiguousarray(array)
        message = Image()
        message.header.stamp = rospy.Time.from_sec(stamp)
        message.header.frame_id = "camera_optical_frame"
        message.height, message.width = array.shape[:2]
        message.encoding = "mono8" if array.ndim == 2 else "bgr8"
        message.step = array.strides[0]
        message.data = array.tobytes()
        return message

    def publish(self, _trial=None):
        trial = self.trial
        state = JointState()
        state.header.stamp = rospy.Time.now()
        state.name = ["joint_{}".format(index + 1) for index in range(6)]
        state.position = trial.pose.tolist()
        self.joints.publish(state)
        rigid_state = JointState()
        rigid_state.header, rigid_state.name = state.header, state.name
        rigid_state.position = trial.command.tolist()
        self.rigid_joints.publish(rigid_state)
        self.markers.publish(self.scene_markers())
        self.predictions.publish(self.prediction_markers())
        self.status.publish(String(data="{} cycle={} ik_iteration={} ik_pending={}".format(
            trial.phase, trial.cycle + 1, trial.ik_iteration, int(trial.ik_pending))))
        if trial.raw_image is not None:
            self.image.publish(self.image_message(trial.raw_image, trial.observation_stamp))
            if trial.debug_image is not None:
                debug = trial.debug_image.copy()
                height = debug.shape[0]
                cv2.rectangle(debug, (0, height - 72), (debug.shape[1], height), (28, 32, 38), -1)
                text = "{} | cycle {} | IK iteration {}".format(trial.phase.upper(), trial.cycle + 1, trial.ik_iteration)
                cv2.putText(debug, text, (15, height - 42), cv2.FONT_HERSHEY_SIMPLEX, .8, (240, 240, 240), 2, cv2.LINE_AA)
                cv2.putText(debug, "Replay: rosservice call /switch_press/replay", (15, height - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, .6, (190, 210, 230), 1, cv2.LINE_AA)
                self.debug.publish(self.image_message(debug, trial.debug_stamp))
            camera = trial.camera.camera
            info = CameraInfo()
            info.header.frame_id, info.header.stamp = "camera_optical_frame", rospy.Time.from_sec(trial.observation_stamp)
            info.width, info.height = camera.width, camera.height
            info.K = camera.camera_matrix.reshape(-1).tolist()
            info.D, info.distortion_model = [0.] * 5, "plumb_bob"
            info.R = np.eye(3).reshape(-1).tolist()
            info.P = np.column_stack((camera.camera_matrix, np.zeros(3))).reshape(-1).tolist()
            self.info.publish(info)
        if trial.observation_preview is not None:
            self.marker_belief.publish(transform_distribution_to_msg(trial.observation_preview["marker_record"]))
            self.goal_belief.publish(transform_distribution_to_msg(trial.observation_preview["goal_record"]))
        now = state.header.stamp.to_sec()
        records = body_frame_records(self.joint_mapping, trial.pose, trial.config["rod"]["length"], now)
        snapshot = transform_array_to_msg(records)
        snapshot.header.stamp, snapshot.header.frame_id = state.header.stamp, "base_link"
        self.probtf.publish(snapshot)
        if trial.observation_preview is not None:
            self.marker_display.publish(transform_distribution_to_msg(
                observation_pose_record(trial.observation_preview, trial.hand_camera, now)))
