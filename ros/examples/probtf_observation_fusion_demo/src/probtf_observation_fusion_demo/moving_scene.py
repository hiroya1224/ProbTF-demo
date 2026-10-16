"""The switch-press physical world, shared verbatim by all sensor cases.

The command recording contains commands alone.  Every replay solves the same
gravity/spring equilibrium afresh; recorded actual poses are never used.
``SensorInputs`` is the only input type intended for the estimation pipeline.
``PhysicalFrame`` is reserved for robot/scene drawing and physical validation.
"""

import csv
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

    The replay is an observation comparison under one common sent command
    trajectory, not four new controllers.  The opaque arm is the gravity/spring
    physical arm; the transparent robot is FK of the same sent command.
    """

    def __init__(self, output_dir, config_path=None):
        packages = rospkg.RosPack()
        package = Path(packages.get_path("probtf_observation_fusion_demo"))
        config_path = Path(config_path or package / "config/switch_press_scene.yaml")
        self.config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        source = Path(packages.get_path("symaware_grasp")) / "urdf/simple_six_dof_arm.urdf"
        self.urdf = write_robot(source, self.output_dir / "robot.urdf", self.config)
        self.robot_description = self.urdf.read_text(encoding="utf-8")
        self.plant = EquilibriumBody(self.urdf, self.config["plant"]["stiffness"])
        commands_path = config_path.parent / self.config["command_trajectory"]["file"]
        self.command_file_sha256 = hashlib.sha256(commands_path.read_bytes()).hexdigest()
        with commands_path.open(newline="", encoding="utf-8") as stream:
            samples = list(csv.DictReader(stream))
        # Several adaptive collision samples share a wall time.  The last
        # command at that time is the command actually reached by that tick.
        by_time = {float(row["time"]): [float(row["command_{}".format(j + 1)])
                                       for j in range(6)] for row in samples}
        self.command_times = np.array(sorted(by_time), dtype=float)
        self.commands = np.array([by_time[time] for time in self.command_times])
        if self.command_times[0] != 0.0 or np.any(np.diff(self.command_times) <= 0.0):
            raise ValueError("Command trajectory requires strictly increasing nonnegative times.")
        self.duration = float(self.command_times[-1])
        self.world_base = transform(self.config["base_translation"])
        marker = self.config["marker"]
        self.marker = inverse(self.world_base) @ transform(marker["world_translation"], marker["world_rpy"])
        self.goal = apply(self.marker, marker["goal_offset"])
        self.tip_offset = np.array([self.config["rod"]["length"], 0.0, 0.0])
        self.hand_camera = transform(self.config["camera"]["translation"], self.config["camera"]["rpy"])
        self.texture_path = (Path(packages.get_path("prob_artag_detector")) / "samples"
                             / "apriltag_36h11_id_{:03d}.png".format(marker["id"]))
        self.imu_target_link = self.config["imu"]["target_link"]
        self.imu_sigma_rad = float(self.config["imu"]["direction_sigma_rad"])
        self.reset()

    def reset(self):
        self.pose = None
        self.camera_history = []
        self.trajectory = []
        self.last_time = None
        self.monitor = ReachMonitor(self.goal, self.config["reach_tolerance"])
        self.camera = PlanarCamera(self.texture_path, self.config["marker"]["size"],
                                   self.config["camera"], self.config["seed"])

    def command_at(self, time):
        time = np.clip(float(time), 0.0, self.duration)
        return np.array([np.interp(time, self.command_times, self.commands[:, index])
                         for index in range(6)])

    def _link_poses(self, joints):
        robot = self.plant.robot
        robot._fk_update(joints)
        base_inverse = np.linalg.inv(np.asarray(robot.data.oMf[robot.base_fid].homogeneous))
        return {name: _readonly(base_inverse @ np.asarray(
            robot.data.oMf[robot.get_frame_id(name)].homogeneous)) for name in
            ["base_link"] + ["link_{}".format(index + 1) for index in range(6)]
            + ["tool0", "press_rod", "rod_tip", "camera_optical_frame"]}

    def step(self, time, stamp=None):
        time = float(np.clip(time, 0.0, self.duration))
        if self.last_time is not None and time < self.last_time:
            raise ValueError("Replay time moved backwards; reset the scene first.")
        stamp = float(time if stamp is None else stamp)
        command = self.command_at(time)
        if self.pose is None:
            self.pose = self.plant.equilibrium(command)
        else:
            # Rendering/update work can skip wall-time ticks.  Check the entire
            # interpolated command interval with the original joint/spatial
            # subdivision instead of checking only its two drawn endpoints.
            for _sample_command, sample_pose in motion_samples(
                    self.plant, self.command_at(self.last_time), command,
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
