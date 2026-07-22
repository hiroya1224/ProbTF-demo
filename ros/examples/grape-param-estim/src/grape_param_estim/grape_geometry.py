"""Grape rotor geometry and calibrated-command wrench reconstruction.

All vectors are expressed about the ``fc`` origin.  The recorded
``FourAxisCommand.base_thrust`` values are treated as force only when the
caller explicitly opts into that calibration assumption; they are not thrust
sensors.  This distinction is surfaced by the offline pipeline provenance.
"""

from typing import Tuple

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


GIMBAL_ORIGINS_MAIN = np.array(
    [
        [-0.22309, -0.22309, 0.0],
        [0.22309, -0.22309, 0.0],
        [0.22309, 0.22309, 0.0],
        [-0.22309, 0.22309, 0.0],
    ],
    dtype=float,
)
MAIN_TO_FC_TRANSLATION = np.array(
    [-0.0172999968682441, -0.00110000084294132, 0.05706099896], dtype=float
)
GIMBAL_ORIGINS_FC = GIMBAL_ORIGINS_MAIN - MAIN_TO_FC_TRANSLATION
ARM_YAWS = np.array([-2.3562, -0.7854, 0.7854, 2.3562], dtype=float)
ROTOR_DIRECTIONS = np.array([-1.0, 1.0, -1.0, 1.0], dtype=float)
THRUST_OFFSET = 0.056
MOMENT_FORCE_RATE = -0.0181


def _rotation_z(angle: float) -> np.ndarray:
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.array([[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]])


ARM_ROTATIONS = np.stack([_rotation_z(value) for value in ARM_YAWS])


def reconstruct_actuator_wrench(
    thrust: np.ndarray,
    gimbal_angle: np.ndarray,
    include_thrust_offset: bool = True,
) -> np.ndarray:
    """Map four thrust magnitudes and measured gimbal angles to ``[F,tau]``.

    The force convention matches ``GimbalrotorController``:
    ``f_arm=[0,-lambda*sin(q),lambda*cos(q)]``.  Reaction moment signs are
    read from the four URDF rotor axes and use the current ``m_f_rate``.
    """

    thrust_values = np.asarray(thrust, dtype=float)
    angle_values = np.asarray(gimbal_angle, dtype=float)
    if thrust_values.shape[-1:] != (4,) or angle_values.shape != thrust_values.shape:
        raise ValueError("thrust and gimbal_angle must have matching (..., 4) shapes.")
    if not np.all(np.isfinite(thrust_values)) or not np.all(np.isfinite(angle_values)):
        raise ValueError("thrust and gimbal angles must be finite.")
    flat_thrust = thrust_values.reshape(-1, 4)
    flat_angle = angle_values.reshape(-1, 4)
    result = np.zeros((flat_thrust.shape[0], 6), dtype=float)
    offset_local = np.array([0.0, 0.0, THRUST_OFFSET])
    for sample in range(flat_thrust.shape[0]):
        for rotor in range(4):
            angle = flat_angle[sample, rotor]
            force_arm = np.array(
                [0.0, -flat_thrust[sample, rotor] * np.sin(angle), flat_thrust[sample, rotor] * np.cos(angle)]
            )
            force = ARM_ROTATIONS[rotor] @ force_arm
            point = GIMBAL_ORIGINS_FC[rotor]
            if include_thrust_offset:
                point = point + ARM_ROTATIONS[rotor] @ Rotation.from_rotvec(
                    np.array([angle, 0.0, 0.0])
                ).apply(offset_local)
            result[sample, :3] += force
            result[sample, 3:] += np.cross(point, force)
            result[sample, 3:] += ROTOR_DIRECTIONS[rotor] * MOMENT_FORCE_RATE * force
    shaped = result.reshape(thrust_values.shape[:-1] + (6,))
    return shaped if thrust_values.ndim > 1 else shaped.reshape(6)


def allocate_wrench(
    actuator_wrench: np.ndarray,
    initial_thrust: np.ndarray = None,
    initial_angle: np.ndarray = None,
    maximum_thrust: float = 15.0,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Numerically allocate a six-axis wrench for synthetic bag generation.

    This helper is not used by the estimator.  It merely makes the synthetic
    bag resemble the Grape command/joint interface while the exact calibrated
    wrench remains the estimator input for the no-gauge sanity check.
    """

    target = np.asarray(actuator_wrench, dtype=float).reshape(6)
    if initial_thrust is None:
        initial_thrust = np.full(4, max(0.1, np.linalg.norm(target[:3]) / 4.0))
    if initial_angle is None:
        initial_angle = np.zeros(4)
    initial = np.concatenate((np.asarray(initial_thrust), np.asarray(initial_angle)))

    force_scale = max(1.0, np.linalg.norm(target[:3]))
    torque_scale = max(0.1, np.linalg.norm(target[3:]))
    scale = np.array([force_scale] * 3 + [torque_scale] * 3)

    def residual(value: np.ndarray) -> np.ndarray:
        return (reconstruct_actuator_wrench(value[:4], value[4:]) - target) / scale

    solution = least_squares(
        residual,
        initial,
        bounds=(
            np.concatenate((np.zeros(4), np.full(4, -1.45))),
            np.concatenate((np.full(4, maximum_thrust), np.full(4, 1.45))),
        ),
        max_nfev=100,
        ftol=1.0e-10,
        xtol=1.0e-10,
        gtol=1.0e-10,
    )
    return solution.x[:4], solution.x[4:], float(np.linalg.norm(residual(solution.x)))
