"""Physical spring/gravity simulation and known collision geometry."""
from dataclasses import dataclass

import numpy as np

from deflecomp_core.model.equilibrium import EquilibriumConfig, EquilibriumSolver
from deflecomp_core.model.spring import LinearSpringModel
from deflecomp_core.robot.pinocchio_robot import RobotArm
from symaware_grasp.grasp_collision import BottleCollisionModel

from probtf_switch_press_demo.scene import apply


class EquilibriumBody:
    def __init__(self, urdf, stiffness):
        self.robot = RobotArm(str(urdf), tip_link="tool0", base_link="base_link")
        self.geometry = BottleCollisionModel(urdf)
        self.stiffness = np.asarray(stiffness, dtype=float)
        self.spring = LinearSpringModel()
        self.solver = EquilibriumSolver(self.robot, self.spring,
                                       EquilibriumConfig(n_lambda=1, maxiter=100, ftol=1e-14,
                                                         refine=True, refine_tol=1e-14))
        self.collision_frames = [self.robot.get_frame_id(link) for link, _name, _local, _half in self.geometry.bodies]
        self.lower = np.asarray(self.robot.model.lowerPositionLimit)
        self.upper = np.asarray(self.robot.model.upperPositionLimit)

    def equilibrium(self, command, warm=None):
        command = np.asarray(command, dtype=float)
        if np.any(command < self.lower) or np.any(command > self.upper):
            raise ValueError("spring command exceeds joint limits")
        # Nearby commands share a stable spring/gravity branch. Newton solves
        # that same torque balance on the physical trajectory;
        # the bounded potential minimizer remains the fallback.
        pose = self._newton_equilibrium(command, command if warm is None else warm)
        if pose is None:
            pose = self.solver.solve(command, self.stiffness,
                                     theta_init=command if warm is None else warm,
                                     lambdas=np.array([0.0]))
        residual = self.robot.tau_gravity(pose) + self.spring.torque(pose, command, self.stiffness)
        active = ((pose <= self.lower + 1e-8) & (residual >= 0)) | ((pose >= self.upper - 1e-8) & (residual <= 0))
        residual[active] = 0.0
        if not np.all(np.isfinite(pose)) or np.max(np.abs(residual)) > 1e-4:
            raise RuntimeError("equilibrium did not satisfy the bounded torque balance")
        if np.linalg.eigvalsh(self.robot.d_tau_gravity(pose) + np.diag(self.stiffness))[0] <= 0:
            raise RuntimeError("equilibrium is not on a stable branch")
        return pose

    def _newton_equilibrium(self, command, warm):
        pose = np.clip(np.asarray(warm, dtype=float), self.lower, self.upper)
        for _iteration in range(16):
            gradient = self.robot.tau_gravity(pose) + self.stiffness * (pose - command)
            active = ((pose <= self.lower + 1e-10) & (gradient >= 0)) | ((pose >= self.upper - 1e-10) & (gradient <= 0))
            free = np.flatnonzero(~active)
            if not len(free) or np.max(np.abs(gradient[free])) < 1e-9:
                return pose
            hessian = self.robot.d_tau_gravity(pose) + np.diag(self.stiffness)
            reduced = hessian[np.ix_(free, free)]
            if np.linalg.eigvalsh(reduced)[0] <= 0:
                return None
            direction = np.zeros(6)
            direction[free] = np.linalg.solve(reduced, -gradient[free])
            energy = self.robot.potential_gravity(pose) + .5 * np.dot(self.stiffness, (pose - command) ** 2)
            accepted = False
            for fraction in (.5 ** np.arange(16)):
                trial = np.clip(pose + fraction * direction, self.lower, self.upper)
                candidate_energy = self.robot.potential_gravity(trial) + .5 * np.dot(self.stiffness, (trial - command) ** 2)
                if candidate_energy <= energy + 1e-4 * np.dot(gradient, trial - pose) + 1e-13:
                    pose, accepted = trial, True
                    break
            if not accepted:
                return None
        return None

    def flange(self, pose):
        return np.asarray(self.robot.fk_pose(pose).homogeneous).copy()

    def boxes(self, pose, reference=None):
        self.robot._fk_update(pose)
        boxes = []
        for frame, (link, _name, local, half) in zip(self.collision_frames, self.geometry.bodies):
            pose = np.asarray(self.robot.data.oMf[frame].homogeneous) @ local
            boxes.append((link, pose if reference is None else reference @ pose, half))
        return boxes


def clearance(boxes, marker_pose, obstacles):
    if not boxes or not obstacles:
        return float("inf"), ""
    poses = np.array([body for _link, body, _half in boxes])
    half = np.array([half for _link, _body, half in boxes])
    centers = apply(marker_pose, [obstacle["center"] for obstacle in obstacles])
    other_half = np.array([obstacle["size"] for obstacle in obstacles]) / 2
    differences = centers[None] - poses[:, None, :3, 3]
    bounds = (np.linalg.norm(differences, axis=-1) - np.linalg.norm(half, axis=-1)[:, None]
              - np.linalg.norm(other_half, axis=-1)[None])
    # The same fifteen-axis separating lower bound, batched over nearby pairs.
    # All rotations and dimensions come from the already validated URDF/config.
    rows, columns = np.nonzero(bounds <= .06)
    if len(rows):
        rotations = poses[rows, :3, :3]
        axes_a = rotations.transpose(0, 2, 1)
        axes_b = np.broadcast_to(marker_pose[:3, :3].T, axes_a.shape)
        axes = np.concatenate((axes_a, axes_b,
            np.cross(axes_a[:, :, None], axes_b[:, None]).reshape(-1, 9, 3)), axis=1)
        lengths = np.linalg.norm(axes, axis=-1)
        axes /= np.where(lengths > 1e-12, lengths, 1.)[:, :, None]
        extent_a = np.einsum("pij,pj->pi", np.abs(axes @ rotations), half[rows])
        extent_b = np.einsum("pij,pj->pi", np.abs(axes @ marker_pose[:3, :3]), other_half[columns])
        gaps = np.abs(np.einsum("pij,pj->pi", axes, differences[rows, columns])) - extent_a - extent_b
        gaps[lengths <= 1e-12] = -np.inf
        gaps = gaps.max(axis=1)
        tolerance = 32 * np.finfo(float).eps * np.maximum(1., np.maximum(
            np.linalg.norm(differences[rows, columns], axis=1), (half[rows] + other_half[columns]).max(axis=1)))
        bounds[rows, columns] = np.where(np.abs(gaps) <= tolerance, 0., gaps)
    row, column = np.unravel_index(np.argmin(bounds), bounds.shape)
    return float(bounds[row, column]), boxes[row][0] + "/" + obstacles[column]["name"]


def shape_displacement(first, second):
    """Upper bound on each corresponding box point's endpoint displacement."""
    a = np.array([pose for _name, pose, _half in first])
    b = np.array([pose for _name, pose, _half in second])
    radii = np.linalg.norm([half for _name, _pose, half in first], axis=1)
    # Difference of two SO(3) matrices has equal nonzero singular values.
    rotation = np.linalg.norm(a[:, :3, :3] - b[:, :3, :3], axis=(1, 2)) / np.sqrt(2.)
    return float(np.max(np.linalg.norm(a[:, :3, 3] - b[:, :3, 3], axis=1) + rotation * radii))


def motion_samples(body, start, end, warm, settings):
    """Short command intervals, subdivided again by collision-shape displacement."""
    start, end = np.asarray(start), np.asarray(end)
    count = max(1, int(np.ceil(np.max(np.abs(end - start)) / settings["joint_resolution"])))
    pose = np.array(warm, copy=True)
    yield start.copy(), pose.copy()

    def segment(c0, q0, boxes0, c1, q1, boxes1, depth=0):
        if shape_displacement(boxes0, boxes1) <= settings["spatial_resolution"]:
            yield c1.copy(), q1.copy()
            return
        if depth >= 12:
            raise RuntimeError("motion subdivision did not resolve a continuous equilibrium path")
        middle = (c0 + c1) / 2
        middle_pose = body.equilibrium(middle, q0)
        middle_boxes = body.boxes(middle_pose)
        yield from segment(c0, q0, boxes0, middle, middle_pose, middle_boxes, depth + 1)
        yield from segment(middle, middle_pose, middle_boxes, c1, q1, boxes1, depth + 1)

    previous = start
    previous_boxes = body.boxes(pose)
    for fraction in np.linspace(0, 1, count + 1)[1:]:
        command = start + fraction * (end - start)
        next_pose = body.equilibrium(command, pose)
        next_boxes = body.boxes(next_pose)
        yield from segment(previous, pose, previous_boxes, command, next_pose, next_boxes)
        previous, pose = command, next_pose
        previous_boxes = next_boxes


@dataclass
class ReachMonitor:
    goal: np.ndarray
    tolerance: float
    collided: bool = False
    reached: bool = False
    minimum_clearance: float = float("inf")

    def update(self, tip, gap, required_clearance):
        self.minimum_clearance = min(self.minimum_clearance, float(gap))
        self.collided = self.collided or gap < required_clearance
        self.reached = bool(not self.collided and np.linalg.norm(tip - self.goal) <= self.tolerance)
        return self.reached
