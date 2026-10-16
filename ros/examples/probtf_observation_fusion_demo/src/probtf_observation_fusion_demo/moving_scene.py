"""The switch-press physical world, shared verbatim by all sensor cases.

Sent commands are supplied by the observation-driven controller.  Every step
solves the same gravity/spring equilibrium; no successful recording is loaded.
``SensorInputs`` is the only input type intended for the estimation pipeline.
``PhysicalFrame`` is reserved for robot/scene drawing and physical validation.
"""

from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np
import rospkg
import yaml

from probtf_switch_press_demo.body import EquilibriumBody, ReachMonitor, clearance, motion_samples
from probtf_switch_press_demo.scene import apply, inverse, transform, write_robot
from probtf_switch_press_demo.vision import PlanarCamera


def _readonly(value):
    result = np.array(value, dtype=float, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class SensorInputs:
    """Observations and command metadata; no equilibrium or truth transforms."""

    time: float
    stamp: float
    command: np.ndarray
    image: np.ndarray
    gravity_in_sensor: np.ndarray
    imu_target_link: str
    imu_sigma_rad: float


@dataclass(frozen=True)
class PhysicalFrame:
    """Simulator output for physical drawing, explicitly separate from inference."""

    time: float
    stamp: float
    command: np.ndarray
    joints: np.ndarray
    link_poses: dict
    rigid_link_poses: dict
    camera_pose: np.ndarray
    tip: np.ndarray
    rigid_tip: np.ndarray
    image: np.ndarray
    gravity_in_sensor: np.ndarray
    clearance: float
    closest_pair: str
    reached: bool
    collided: bool
    maximum_deflection_rad: float
    camera_diagnostics: dict


class MovingSwitchScene:
    """Reuse the original arm, camera, obstacles and physical success condition.

    All cases share this physical model, geometry and initial command.  Their
    commands may differ because the common controller uses their respective
    estimated states.  The opaque arm is the gravity/spring physical arm; the
    transparent robot is FK of the command sent at the same instant.
    """

    def __init__(self, output_dir, config_path=None, *, imu_target_link=None):
        packages = rospkg.RosPack()
        package = Path(packages.get_path("probtf_observation_fusion_demo"))
        config_path = Path(config_path or package / "config/switch_press_scene.yaml")
        self.config_sha256 = hashlib.sha256(config_path.read_bytes()).hexdigest()
        self.config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        source = Path(packages.get_path("symaware_grasp")) / "urdf/simple_six_dof_arm.urdf"
        self.urdf = write_robot(source, self.output_dir / "robot.urdf", self.config)
        self.robot_description = self.urdf.read_text(encoding="utf-8")
        self.plant = EquilibriumBody(self.urdf, self.config["plant"]["stiffness"])
        self.initial_command = _readonly(self.config["initial_command"])
        self.duration = float(self.config["motion"].get("duration_s", 36.0))
        if not np.isfinite(self.duration) or self.duration <= 0.:
            raise ValueError("The common experiment duration must be finite and positive.")
        self.world_base = transform(self.config["base_translation"])
        marker = self.config["marker"]
        self.marker = inverse(self.world_base) @ transform(marker["world_translation"], marker["world_rpy"])
        self.goal = apply(self.marker, marker["goal_offset"])
        self.tip_offset = np.array([self.config["rod"]["length"], 0.0, 0.0])
        self.hand_camera = transform(self.config["camera"]["translation"], self.config["camera"]["rpy"])
        self.texture_path = (Path(packages.get_path("prob_artag_detector")) / "samples"
                             / "apriltag_36h11_id_{:03d}.png".format(marker["id"]))
        self.imu_target_link = (self.config["imu"]["target_link"]
                                if imu_target_link is None else str(imu_target_link))
        imu_links = {"link_{}".format(index+1) for index in range(6)} | {"tool0"}
        if self.imu_target_link not in imu_links:
            raise ValueError("IMU attachment must be an active link or the known tool0 flange.")
        self.imu_sigma_rad = float(self.config["imu"]["direction_sigma_rad"])
        self.reset()

    def reset(self):
        self.pose = None
        self.camera_history = []
        self.trajectory = []
        self.last_time = None
        self.command = self.initial_command
        self.monitor = ReachMonitor(self.goal, self.config["reach_tolerance"])
        self.camera = PlanarCamera(self.texture_path, self.config["marker"]["size"],
                                   self.config["camera"], self.config["seed"])

    def _link_poses(self, joints):
        robot = self.plant.robot
        robot._fk_update(joints)
        base_inverse = np.linalg.inv(np.asarray(robot.data.oMf[robot.base_fid].homogeneous))
        return {name: _readonly(base_inverse @ np.asarray(
            robot.data.oMf[robot.get_frame_id(name)].homogeneous)) for name in
            ["base_link"] + ["link_{}".format(index + 1) for index in range(6)]
            + ["tool0", "press_rod", "rod_tip", "camera_optical_frame"]}

    def step(self, time, stamp=None, command=None):
        time = float(np.clip(time, 0.0, self.duration))
        if self.last_time is not None and time < self.last_time:
            raise ValueError("Replay time moved backwards; reset the scene first.")
        stamp = float(time if stamp is None else stamp)
        command = np.array(self.command if command is None else command, dtype=float, copy=True)
        if command.shape != self.initial_command.shape or not np.all(np.isfinite(command)):
            raise ValueError("The sent command must contain one finite position per joint.")
        if self.pose is None:
            self.pose = self.plant.equilibrium(command)
        else:
            # Rendering/update work can skip wall-time ticks.  Check the entire
            # interpolated command interval with the original joint/spatial
            # subdivision instead of checking only its two drawn endpoints.
            for _sample_command, sample_pose in motion_samples(
                    self.plant, self.command, command,
                    self.pose, self.config["motion"]):
                sample_tip = apply(self.plant.flange(sample_pose), self.tip_offset)
                sample_gap, _pair = clearance(self.plant.boxes(sample_pose), self.marker,
                                               self.config["obstacles"])
                self.monitor.update(sample_tip, sample_gap, 0.0)
                self.pose = sample_pose
        physical = self._link_poses(self.pose)
        rigid = self._link_poses(command)
        camera_pose = physical["camera_optical_frame"]
        if not self.camera_history or time > self.camera_history[-1][0]:
            self.camera_history.append((time, camera_pose))
        else:
            self.camera_history[-1] = (time, camera_pose)
        self.camera_history = self.camera_history[-32:]
        image = self.camera.render(camera_pose, self.marker, self.camera_history)
        image.setflags(write=False)
        # This is a simulated accelerometer reading.  Its physical link
        # rotation remains entirely on this side of the sensor boundary.
        gravity = physical[self.imu_target_link][:3, :3].T @ np.array([0.0, 0.0, -1.0])
        # Frame-indexed noise makes the same time produce the same observation
        # in all four cases, independently of which adapters are enabled.
        index = int(round(time * 1e6))
        rng = np.random.default_rng(np.random.SeedSequence([int(self.config["seed"]), index, 1701]))
        noise = rng.normal(0.0, self.imu_sigma_rad, 3)
        gravity += noise - gravity * np.dot(gravity, noise)
        gravity /= np.linalg.norm(gravity)
        tip, rigid_tip = physical["rod_tip"][:3, 3], rigid["rod_tip"][:3, 3]
        gap, pair = clearance(self.plant.boxes(self.pose), self.marker, self.config["obstacles"])
        reached = self.monitor.update(tip, gap, 0.0)
        frame = PhysicalFrame(time, stamp, _readonly(command), _readonly(self.pose), physical,
            rigid, camera_pose, _readonly(tip), _readonly(rigid_tip), image, _readonly(gravity),
            float(gap), pair, reached, self.monitor.collided,
            float(np.max(np.abs(self.pose - command))), dict(self.camera.diagnostics))
        self.last_time = time
        self.command = frame.command
        self.trajectory.append(self.physical_record(frame))
        return frame

    def sensor_inputs(self, frame):
        return SensorInputs(frame.time, frame.stamp, frame.command, frame.image,
                            frame.gravity_in_sensor, self.imu_target_link, self.imu_sigma_rad)

    def physical_record(self, frame):
        """Truth diagnostics for validation/display only, never estimator input."""
        return dict(time=frame.time, stamp=frame.stamp, command=frame.command.tolist(),
            equilibrium=frame.joints.tolist(), tip=frame.tip.tolist(), rigid_tip=frame.rigid_tip.tolist(),
            physical_error_m=float(np.linalg.norm(frame.tip - self.goal)),
            rigid_command_error_m=float(np.linalg.norm(frame.rigid_tip - self.goal)),
            clearance_m=frame.clearance, closest_pair=frame.closest_pair,
            reached=frame.reached, collided=frame.collided,
            maximum_deflection_rad=frame.maximum_deflection_rad,
            image_sha256=hashlib.sha256(frame.image.tobytes()).hexdigest(),
            gravity_in_sensor=frame.gravity_in_sensor.tolist())
