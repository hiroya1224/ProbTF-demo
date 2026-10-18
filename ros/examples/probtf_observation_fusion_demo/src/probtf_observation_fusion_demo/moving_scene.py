"""The switch-press physical world, shared verbatim by all sensor cases.

Sent commands are supplied by the observation-driven controller.  A native
inertia/spring/damper model retains the realized joint positions and velocities
between steps; no successful recording is loaded.  Static equilibrium is used
only to initialize the already-sagged arm, with zero initial velocity.
``SensorInputs`` is the only input type intended for the estimation pipeline.
``PhysicalFrame`` is reserved for robot/scene drawing and physical validation.
"""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Optional

import numpy as np
import rospkg
import yaml

from probtf_switch_press_demo.body import EquilibriumBody, ReachMonitor
from probtf_switch_press_demo.scene import apply, inverse, transform, write_robot
from probtf_switch_press_demo.vision import PlanarCamera
from symaware_grasp.box_geometry import box_box_gap

from .board_camera import BoardCamera, make_camera
from .dynamic_physics import NativeDynamics


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
    clearance: Optional[float]
    closest_pair: str
    reached: bool
    collided: Optional[bool]
    maximum_deflection_rad: float
    camera_diagnostics: dict
    joint_velocities: np.ndarray
    joint_accelerations: np.ndarray
    physics_diagnostics: dict


class _PhysicalClearance:
    """The existing box/sphere test with fixed obstacle geometry cached.

    These are physical diagnostics, not collision forces or an estimator input.
    The separating-axis calculation and its 6 cm broad-phase cutoff are the
    same as the original switch scene.  Batch the cheap sphere tests because
    enabled dynamics checks the internal path, not just drawn frames.
    """
    def __init__(self, marker, obstacles, bodies):
        self.centers = np.asarray([apply(marker, item['center']) for item in obstacles])
        self.half = np.asarray([item['size'] for item in obstacles], dtype=float)/2.
        self.rotation = marker[:3, :3]
        self.radii = np.linalg.norm(self.half, axis=1)
        self.body_radii = np.asarray([np.linalg.norm(half) for _, _, _, half in bodies])
        self.obstacles = obstacles
    def __call__(self, boxes):
        poses = np.asarray([pose for _, pose, _ in boxes])
        gaps = (np.linalg.norm(poses[:, None, :3, 3]-self.centers[None], axis=2)
                - self.body_radii[:, None]-self.radii[None])
        for i, j in np.argwhere(gaps <= .06):
            gaps[i, j] = box_box_gap(poses[i, :3, 3], poses[i, :3, :3], boxes[i][2],
                                     self.centers[j], self.rotation, self.half[j])
        i, j = np.unravel_index(np.argmin(gaps), gaps.shape)
        return float(gaps[i, j]), boxes[i][0]+'/'+self.obstacles[j]['name']

    def displacement_bound(self, first, second):
        a, b = (np.asarray([pose for _, pose, _ in boxes]) for boxes in (first, second))
        # Frobenius bounds the spectral norm used by the original monitor.
        return float(np.max(np.linalg.norm(a[:, :3, 3]-b[:, :3, 3], axis=1)
            + np.linalg.norm(a[:, :3, :3]-b[:, :3, :3], axis=(1, 2))*self.body_radii))


class MovingSwitchScene:
    """Reuse the original arm, camera, obstacles and physical success condition.

    All cases share this physical model, geometry and initial command.  Their
    commands may differ because the common controller uses their respective
    estimated states.  The opaque arm is the gravity/spring physical arm; the
    transparent robot is FK of the command sent at the same instant.
    """

    def __init__(self, output_dir, config_path=None, *, imu_target_link=None,
                 marker_occlusion_enabled=None):
        packages = rospkg.RosPack()
        package = Path(packages.get_path("probtf_observation_fusion_demo"))
        config_path = Path(config_path or package / "config/switch_press_scene.yaml")
        self.config_sha256 = hashlib.sha256(config_path.read_bytes()).hexdigest()
        self.config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        self.physical_scene_overrides = {}
        if marker_occlusion_enabled is not None:
            if not isinstance(marker_occlusion_enabled, (bool, np.bool_)):
                raise ValueError('marker_occlusion_enabled must be a boolean or None.')
            self.config['camera'].setdefault('marker_occluder', {})['enabled'] = bool(marker_occlusion_enabled)
            self.physical_scene_overrides['marker_occlusion_enabled'] = bool(marker_occlusion_enabled)
        self.resolved_config_sha256 = hashlib.sha256(json.dumps(
            self.config, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        self.collision_checking = self.config['plant'].get('collision_checking', True)
        if not isinstance(self.collision_checking, bool):
            raise ValueError('Physical collision_checking must be an explicit Boolean.')
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        source = Path(packages.get_path("symaware_grasp")) / "urdf/simple_six_dof_arm.urdf"
        self.urdf = write_robot(source, self.output_dir / "robot.urdf", self.config)
        self.robot_description = self.urdf.read_text(encoding="utf-8")
        # Kept for initial equilibrium, FK and collision shapes only.  It does
        # not generate the physical trajectory after initialization.
        self.plant = EquilibriumBody(self.urdf, self.config["plant"]["stiffness"])
        dynamics = dict(self.config['plant'].get('dynamics', {}))
        self.dynamics = NativeDynamics(self.urdf, self.config['plant']['stiffness'], **dynamics)
        self.physics_source = dict(backend_path=str(self.dynamics.backend_path),
            backend_sha256=hashlib.sha256(Path(self.dynamics.backend_path).read_bytes()).hexdigest(),
            urdf_sha256=hashlib.sha256(self.urdf.read_bytes()).hexdigest(),
            scene_config_sha256=self.config_sha256,
            resolved_scene_config_sha256=self.resolved_config_sha256,
            physical_scene_overrides=dict(self.physical_scene_overrides))
        self.initial_command = _readonly(self.config["initial_command"])
        # This scene's root is fixed to the world. Resolve its invariant
        # transform and all frame names once; internal integration points
        # need only the camera transform, not eleven detached link poses.
        robot = self.plant.robot
        if robot.model.frames[robot.base_fid].parentJoint != 0:
            raise ValueError('The physical scene requires a fixed root frame.')
        robot._fk_update(self.initial_command)
        self._base_inverse = _readonly(np.linalg.inv(
            np.asarray(robot.data.oMf[robot.base_fid].homogeneous)))
        names = (["base_link"] + ["link_{}".format(index + 1) for index in range(6)]
                 + ["tool0", "press_rod", "rod_tip", "camera_optical_frame"])
        self._frame_ids = {name: robot.get_frame_id(name) for name in names}
        self.duration = float(self.config["motion"].get("duration_s", 36.0))
        if not np.isfinite(self.duration) or self.duration <= 0.:
            raise ValueError("The common experiment duration must be finite and positive.")
        self.world_base = transform(self.config["base_translation"])
        marker = self.config["marker"]
        self.marker = inverse(self.world_base) @ transform(marker["world_translation"], marker["world_rpy"])
        self.goal = apply(self.marker, marker["goal_offset"])
        self._clearance = (_PhysicalClearance(self.marker, self.config['obstacles'],
            self.plant.geometry.bodies) if self.collision_checking else None)
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
        self.joint_velocity = np.zeros_like(self.initial_command)
        self.camera_history = []
        self.trajectory = []
        self.last_time = None
        self.command = self.initial_command
        self.monitor = ReachMonitor(self.goal, self.config["reach_tolerance"])
        if not self.collision_checking:
            self.monitor.collided = None
            self.monitor.minimum_clearance = None
        board = self.config.get('marker_board', {})
        self.camera = make_camera(self.texture_path, board, self.config['camera'],
            self.config['seed'], marker_size=self.config['marker']['size'])

    def _link_poses(self, joints):
        robot = self.plant.robot
        robot._fk_update(joints)
        return {name: _readonly(self._base_inverse @ np.asarray(
            robot.data.oMf[fid].homogeneous)) for name, fid in self._frame_ids.items()}

    def _camera_pose(self, joints):
        robot = self.plant.robot
        robot._fk_update(joints)
        return _readonly(self._base_inverse @ np.asarray(
            robot.data.oMf[self._frame_ids['camera_optical_frame']].homogeneous))

    def _remember_camera(self, time, pose):
        item = (float(time), _readonly(pose))
        if not self.camera_history or time > self.camera_history[-1][0]:
            self.camera_history.append(item)
        else:
            self.camera_history[-1] = item
        # Retain the entire exposure and its preceding interpolation point.
        # A fixed count would drop most of the exposure at 1 kHz substeps.
        cutoff = time-float(self.config['camera']['exposure_time'])
        first = np.searchsorted([t for t, _ in self.camera_history], cutoff, side='left')
        if first > 1:
            self.camera_history = self.camera_history[first-1:]

    def _monitor_segment(self, q0, boxes0, q1, boxes1, depth=0):
        """Check the realized integration path, without resolving equilibrium.

        The integrator's within-substep position path is linear.  Subdivide
        that path further if its joint/shape displacement exceeds the existing
        collision sampling resolutions.  Obstacles remain monitored forbidden
        intersections; they do not supply contact forces to the arm.
        """
        settings = self.config['motion']
        if (np.max(np.abs(q1-q0)) > settings['joint_resolution']
                or self._clearance.displacement_bound(boxes0, boxes1)
                   > settings['spatial_resolution']):
            if depth >= 16:
                raise RuntimeError('Dynamic collision-path subdivision did not resolve the motion.')
            middle = (q0+q1)/2.
            middle_boxes = self.plant.boxes(middle)
            self._monitor_segment(q0, boxes0, middle, middle_boxes, depth+1)
            self._monitor_segment(middle, middle_boxes, q1, boxes1, depth+1)
            return
        gap, _ = self._clearance(boxes1)
        self.monitor.update(apply(self.plant.flange(q1), self.tip_offset), gap, 0.)

    def step(self, time, stamp=None, command=None, *, render_camera=True):
        time = float(np.clip(time, 0.0, self.duration))
        if self.last_time is not None and time < self.last_time:
            raise ValueError("Replay time moved backwards; reset the scene first.")
        stamp = float(time if stamp is None else stamp)
        command = np.array(self.command if command is None else command, dtype=float, copy=True)
        if command.shape != self.initial_command.shape or not np.all(np.isfinite(command)):
            raise ValueError("The sent command must contain one finite position per joint.")
        if np.any(command < self.plant.lower) or np.any(command > self.plant.upper):
            raise ValueError('The sent spring command exceeds joint limits.')
        if self.pose is None:
            self.pose = self.plant.equilibrium(self.initial_command)
            self.dynamics.reset(self.pose, self.joint_velocity, command=self.initial_command)
            self._remember_camera(0., self._camera_pose(self.pose))
            if self.collision_checking:
                initial_gap, _ = self._clearance(self.plant.boxes(self.pose))
                self.monitor.update(apply(self.plant.flange(self.pose), self.tip_offset), initial_gap, 0.)
        start_time = 0. if self.last_time is None else self.last_time
        start_pose = self.pose.copy()
        path = self.dynamics.advance(time-start_time, self.command, command)
        previous_pose = start_pose
        previous_boxes = self.plant.boxes(start_pose) if self.collision_checking else None
        previous_offset, stationary = 0., False
        for offset, sample_pose in zip(path['times'], path['positions']):
            sample_pose = np.asarray(sample_pose)
            if float(offset) <= 0.:
                continue
            if np.array_equal(sample_pose, previous_pose):
                previous_offset, stationary = float(offset), True
                continue
            if stationary:
                self._remember_camera(start_time+previous_offset,
                    self._camera_pose(previous_pose))
            if self.collision_checking:
                sample_boxes = self.plant.boxes(sample_pose)
                self._monitor_segment(previous_pose, previous_boxes, sample_pose, sample_boxes)
                previous_boxes = sample_boxes
            camera = self._camera_pose(sample_pose)
            self._remember_camera(start_time+float(offset), camera)
            previous_pose = sample_pose
            previous_offset, stationary = float(offset), False
        state = self.dynamics.state()
        self.pose = np.array(state['position'], copy=True)
        self.joint_velocity = np.array(state['velocity'], copy=True)
        physical = self._link_poses(self.pose)
        rigid = self._link_poses(command)
        camera_pose = physical["camera_optical_frame"]
        self._remember_camera(time, camera_pose)
        image = (self.camera.render(camera_pose, self.marker, self.camera_history)
                 if render_camera else np.empty((0, 0), dtype=np.uint8))
        image.setflags(write=False)
        # Retain the existing gravity-direction sensor definition.  This is
        # not a raw accelerometer specific-force measurement: translational
        # acceleration and gyroscope signals are not part of this adapter.
        # Its physical link rotation stays on this side of the sensor boundary.
        gravity = physical[self.imu_target_link][:3, :3].T @ np.array([0.0, 0.0, -1.0])
        # Frame-indexed noise makes the same time produce the same observation
        # in all four cases, independently of which adapters are enabled.
        index = int(round(time * 1e6))
        rng = np.random.default_rng(np.random.SeedSequence([int(self.config["seed"]), index, 1701]))
        noise = rng.normal(0.0, self.imu_sigma_rad, 3)
        gravity += noise - gravity * np.dot(gravity, noise)
        gravity /= np.linalg.norm(gravity)
        tip, rigid_tip = physical["rod_tip"][:3, 3], rigid["rod_tip"][:3, 3]
        if self.collision_checking:
            gap, pair = self._clearance(self.plant.boxes(self.pose))
            reached = self.monitor.update(tip, gap, 0.0)
        else:
            gap, pair = None, 'unchecked'
            reached = bool(np.linalg.norm(tip-self.goal) <= self.config['reach_tolerance'])
            self.monitor.reached = reached
        frame = PhysicalFrame(time, stamp, _readonly(command), _readonly(self.pose), physical,
            rigid, camera_pose, _readonly(tip), _readonly(rigid_tip), image, _readonly(gravity),
            gap, pair, reached, self.monitor.collided,
            float(np.max(np.abs(self.pose - command))), dict(self.camera.diagnostics,
                rendered=bool(render_camera), image_stamp=stamp if render_camera else None),
            _readonly(self.joint_velocity), _readonly(state['acceleration']),
            dict(state['diagnostics'], model='inertial_spring_damper',
                integrator='implicit_midpoint', command_interpolation='linear_between_actual_sent_positions',
                imu_measurement='unit_gravity_direction_with_tangent_noise', source=self.physics_source,
                collision_checking_enabled=self.collision_checking,
                collision_status='checked' if self.collision_checking else 'unchecked',
                reach_condition=('distance_and_no_detected_collision' if self.collision_checking
                                 else 'distance_only_collision_unchecked')))
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
            actual_joints=frame.joints.tolist(), joint_velocities=frame.joint_velocities.tolist(),
            joint_accelerations=frame.joint_accelerations.tolist(), physics=frame.physics_diagnostics,
            tip=frame.tip.tolist(), rigid_tip=frame.rigid_tip.tolist(),
            physical_error_m=float(np.linalg.norm(frame.tip - self.goal)),
            rigid_command_error_m=float(np.linalg.norm(frame.rigid_tip - self.goal)),
            clearance_m=frame.clearance, closest_pair=frame.closest_pair,
            collision_checking_enabled=self.collision_checking,
            reach_condition=frame.physics_diagnostics['reach_condition'],
            reached=frame.reached, collided=frame.collided,
            maximum_deflection_rad=frame.maximum_deflection_rad,
            image_sha256=hashlib.sha256(frame.image.tobytes()).hexdigest(),
            gravity_in_sensor=frame.gravity_in_sensor.tolist())
