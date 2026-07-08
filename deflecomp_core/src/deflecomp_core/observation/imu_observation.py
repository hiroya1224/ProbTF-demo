from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import numpy as np

from deflecomp_core.observation.bingham import simple_bingham_unit
from deflecomp_core.robot.pinocchio_robot import RobotArm
from deflecomp_core.utils.linalg import normalize


@dataclass(frozen=True)
class FrameImuObservation:
    frame_name: str
    gravity_dir: np.ndarray
    stamp: Optional[float] = None


class ImuObservationBuilder:
    def __init__(
        self,
        robot: RobotArm,
        g_world: Optional[np.ndarray] = None,
        parameter_A: float = 100.0,
    ) -> None:
        self.robot = robot
        self.g_world = normalize(
            np.array([0.0, 0.0, -9.81], dtype=float) if g_world is None else np.asarray(g_world, dtype=float)
        )
        self.parameter_A = float(parameter_A)

    def build_A_map(self, observations: Iterable[FrameImuObservation]) -> Dict[int, np.ndarray]:
        a_map: Dict[int, np.ndarray] = {}
        for observation in observations:
            g_frame = normalize(np.asarray(observation.gravity_dir, dtype=float))
            fid = self.robot.get_frame_id(observation.frame_name)
            a_map[fid] = simple_bingham_unit(g_frame, self.g_world, parameter=self.parameter_A)
        return a_map
