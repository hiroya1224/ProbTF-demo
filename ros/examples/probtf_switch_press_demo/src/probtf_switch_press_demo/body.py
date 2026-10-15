"""The physical gravity/spring simulator and actual-path collision monitor."""
from dataclasses import dataclass

import numpy as np
import pinocchio as pin

from deflecomp_core.model.equilibrium import EquilibriumConfig, EquilibriumSolver
from deflecomp_core.model.spring import LinearSpringModel
from deflecomp_core.robot.pinocchio_robot import RobotArm
from symaware_grasp.grasp_collision import BottleCollisionModel
from symaware_grasp.box_geometry import box_box_gap

from probtf_switch_press_demo.scene import apply


class EquilibriumBody:
    def __init__(self, urdf, stiffness):
        self.robot = RobotArm(str(urdf), tip_link="tool0", base_link="base_link")
        self.stiffness = np.asarray(stiffness, dtype=float)
        self.spring = LinearSpringModel()
        self.solver = EquilibriumSolver(self.robot, self.spring,
                                       EquilibriumConfig(n_lambda=1, maxiter=100, ftol=1e-14,
                                                         refine=True, refine_tol=1e-14))
        self.geometry = BottleCollisionModel(urdf)
        self.lower = np.asarray(self.robot.model.lowerPositionLimit)
        self.upper = np.asarray(self.robot.model.upperPositionLimit)

    def equilibrium(self, command, warm=None):
        command = np.asarray(command, dtype=float)
        if np.any(command < self.lower) or np.any(command > self.upper):
            raise ValueError("spring command exceeds joint limits")
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

    def flange(self, pose):
        return np.asarray(self.robot.fk_pose(pose).homogeneous).copy()

    def tip_command_jacobian(self, pose, reference_rotation):
        """Differentiate equilibrium and FK on this body's current stable branch."""
        hessian = self.robot.d_tau_gravity(pose) + np.diag(self.stiffness)
        free = np.flatnonzero((pose > self.lower + 1e-7) & (pose < self.upper - 1e-7))
        sensitivity = np.zeros((6, 6))
        sensitivity[np.ix_(free, free)] = np.linalg.solve(
            hessian[np.ix_(free, free)], np.diag(self.stiffness[free]))
        jacobian = pin.computeFrameJacobian(self.robot.model, self.robot.data, pose,
            self.robot.get_frame_id("rod_tip"), pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[:3]
        return reference_rotation @ jacobian @ sensitivity

    def boxes(self, pose, reference=None):
        frames = self.geometry.transforms(pose, 0.185)
        return [(link, (frames[link] @ local if reference is None else reference @ frames[link] @ local), half)
                for link, _name, local, half in self.geometry.bodies]


def clearance(boxes, marker_pose, obstacles):
    smallest, pair = float("inf"), ""
    for link, body, half in boxes:
        for obstacle in obstacles:
            center = apply(marker_pose, obstacle["center"])
            other_half = np.asarray(obstacle["size"]) / 2
            # Bounding spheres cheaply certify distant pairs before the SAT.
            bound = np.linalg.norm(body[:3, 3] - center) - np.linalg.norm(half) - np.linalg.norm(other_half)
            gap = (float(bound) if bound > 0.06 else
                   box_box_gap(body[:3, 3], body[:3, :3], half, center, marker_pose[:3, :3], other_half))
            if gap < smallest:
                smallest, pair = gap, link + "/" + obstacle["name"]
    return smallest, pair


def shape_displacement(first, second):
    """Upper bound on each corresponding box point's endpoint displacement."""
    return max(np.linalg.norm(a[:3, 3] - b[:3, 3]) +
               np.linalg.norm(a[:3, :3] - b[:3, :3], ord=2) * np.linalg.norm(half)
               for (_name, a, half), (_other, b, _half) in zip(first, second))


def motion_samples(body, start, end, warm, settings):
    """Short command intervals, subdivided again by collision-shape displacement."""
    start, end = np.asarray(start), np.asarray(end)
    count = max(1, int(np.ceil(np.max(np.abs(end - start)) / settings["joint_resolution"])))
    pose = np.array(warm, copy=True)
    yield start.copy(), pose.copy()

    def segment(c0, q0, c1, q1, depth=0):
        if shape_displacement(body.boxes(q0), body.boxes(q1)) <= settings["spatial_resolution"]:
            yield c1.copy(), q1.copy()
            return
        if depth >= 12:
            raise RuntimeError("motion subdivision did not resolve a continuous equilibrium path")
        middle = (c0 + c1) / 2
        middle_pose = body.equilibrium(middle, q0)
        yield from segment(c0, q0, middle, middle_pose, depth + 1)
        yield from segment(middle, middle_pose, c1, q1, depth + 1)

    previous = start
    for fraction in np.linspace(0, 1, count + 1)[1:]:
        command = start + fraction * (end - start)
        next_pose = body.equilibrium(command, pose)
        yield from segment(previous, pose, command, next_pose)
        previous, pose = command, next_pose


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
