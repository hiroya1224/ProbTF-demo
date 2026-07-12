from dataclasses import dataclass

import numpy as np


@dataclass
class JointFeedbackCompensator:
    gain: np.ndarray

    def apply(self, theta_cmd_ff: np.ndarray, theta_ref: np.ndarray, theta_meas: np.ndarray) -> np.ndarray:
        theta_cmd_ff = np.asarray(theta_cmd_ff, dtype=float)
        theta_ref = np.asarray(theta_ref, dtype=float)
        theta_meas = np.asarray(theta_meas, dtype=float)
        gain = np.asarray(self.gain, dtype=float)
        return theta_cmd_ff + gain * (theta_ref - theta_meas)
