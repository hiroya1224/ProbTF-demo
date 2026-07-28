"""Small controller-side examples used by the offline estimator.

This module is intentionally not an exact copy of the live controller.  It
contains only:

* the source-assumed Grape geometry needed to interpret recorded commands;
* a compact, ROS-independent PID example for readers of the sample.
"""

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np


GIMBAL_ORIGINS_MAIN = np.asarray(
    (
        (-0.22309, -0.22309, 0.0),
        (0.22309, -0.22309, 0.0),
        (0.22309, 0.22309, 0.0),
        (-0.22309, 0.22309, 0.0),
    ),
    dtype=float,
)
MAIN_TO_FC_TRANSLATION = np.asarray(
    (-0.0172999968682441, -0.00110000084294132, 0.05706099896),
    dtype=float,
)
GIMBAL_ORIGINS_FC = GIMBAL_ORIGINS_MAIN - MAIN_TO_FC_TRANSLATION
ARM_YAWS = np.asarray((-2.3562, -0.7854, 0.7854, 2.3562), dtype=float)
ROTOR_DIRECTIONS = np.asarray((-1.0, 1.0, -1.0, 1.0), dtype=float)
THRUST_OFFSET = 0.056
MOMENT_FORCE_RATE = -0.0181


def _rotate_z(vector: np.ndarray, angle: float) -> np.ndarray:
    cosine = float(np.cos(angle))
    sine = float(np.sin(angle))
    x_value, y_value, z_value = vector
    return np.asarray(
        (
            cosine * x_value - sine * y_value,
            sine * x_value + cosine * y_value,
            z_value,
        ),
        dtype=float,
    )


def command_to_wrench(
    base_thrust: Sequence[float],
    gimbal_angle: Sequence[float],
) -> np.ndarray:
    """Convert one recorded command into ``[Fx,Fy,Fz,Tx,Ty,Tz]``.

    ``base_thrust`` remains a command unit.  Therefore the returned wrench is
    a *command wrench*, not a calibrated physical wrench.  The geometry is a
    fixed source-code assumption and is recorded as such in estimator output.
    """

    thrust = np.asarray(base_thrust, dtype=float)
    angle = np.asarray(gimbal_angle, dtype=float)
    if thrust.shape != (4,) or angle.shape != (4,):
        raise ValueError("base_thrust and gimbal_angle must contain 4 values")
    if not np.all(np.isfinite(thrust)) or not np.all(np.isfinite(angle)):
        raise ValueError("recorded command values must be finite")

    result = np.zeros(6, dtype=float)
    for rotor in range(4):
        sine = float(np.sin(angle[rotor]))
        cosine = float(np.cos(angle[rotor]))
        force_arm = np.asarray(
            (0.0, -thrust[rotor] * sine, thrust[rotor] * cosine),
            dtype=float,
        )
        force = _rotate_z(force_arm, ARM_YAWS[rotor])
        offset_arm = np.asarray(
            (0.0, -THRUST_OFFSET * sine, THRUST_OFFSET * cosine),
            dtype=float,
        )
        point = (
            GIMBAL_ORIGINS_FC[rotor]
            + _rotate_z(offset_arm, ARM_YAWS[rotor])
        )
        result[:3] += force
        result[3:] += np.cross(point, force)
        result[3:] += (
            ROTOR_DIRECTIONS[rotor] * MOMENT_FORCE_RATE * force
        )
    return result


@dataclass
class SamplePidAxis:
    """Minimal PID example with integral clamp and simple anti-windup."""

    proportional_gain: float
    integral_gain: float
    derivative_gain: float
    integral_limit: float
    output_limit: float
    integral: float = 0.0
    previous_error: Optional[float] = None

    def reset(self) -> None:
        self.integral = 0.0
        self.previous_error = None

    def step(
        self,
        error: float,
        dt: float,
        *,
        derivative: Optional[float] = None,
        integrate: bool = True,
    ) -> Tuple[float, Tuple[float, float, float]]:
        """Advance one sample and return output plus ``(P, I, D)`` terms."""

        value = float(error)
        step = float(dt)
        if not np.isfinite(value) or not np.isfinite(step) or step <= 0.0:
            raise ValueError("error must be finite and dt must be positive")
        if derivative is None:
            rate = (
                0.0
                if self.previous_error is None
                else (value - self.previous_error) / step
            )
        else:
            rate = float(derivative)
            if not np.isfinite(rate):
                raise ValueError("derivative must be finite")

        previous_integral = self.integral
        if integrate:
            limit = abs(float(self.integral_limit))
            self.integral = float(
                np.clip(self.integral + value * step, -limit, limit)
            )
        p_term = float(self.proportional_gain) * value
        i_term = float(self.integral_gain) * self.integral
        d_term = float(self.derivative_gain) * rate
        raw = p_term + i_term + d_term
        output_limit = abs(float(self.output_limit))
        output = float(np.clip(raw, -output_limit, output_limit))

        if (
            integrate
            and raw != output
            and np.sign(value) == np.sign(raw)
        ):
            self.integral = previous_integral
            i_term = float(self.integral_gain) * self.integral
            raw = p_term + i_term + d_term
            output = float(np.clip(raw, -output_limit, output_limit))
        self.previous_error = value
        return output, (p_term, i_term, d_term)


__all__ = [
    "SamplePidAxis",
    "command_to_wrench",
]
