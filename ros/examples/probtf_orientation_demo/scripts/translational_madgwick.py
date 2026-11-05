"""Plain Madgwick controls for the standalone translational-acceleration study.

The update itself is the existing, independently checked implementation in
``compare_madgwick_rolling``.  This wrapper supplies only the fixed heading
frame conversion required by a known world magnetic reference.  It introduces
no gain tuning, acceleration rejection, bias estimator or truth feedback.

Quaternions are Hamilton (w,x,y,z), rotating IMU vectors into the world frame.
The accelerometer measures specific force: with world gravity (0,0,-g), its
stationary reading is R(q).T (0,0,g), matching Madgwick's positive-Z reference.

Primary sources: https://x-io.co.uk/downloads/madgwick_internal_report.pdf
and https://x-io.co.uk/downloads/madgwick_algorithm_matlab.zip .
"""
import numpy as np
from scipy.spatial.transform import Rotation

from compare_madgwick_rolling import (
    heading_frame, madgwick_update, quaternion_product, REPORT_URL, MATLAB_URL,
)


DEFAULT_BETA = 0.1
ALGORITHM_METADATA = {
    'algorithm': 'original Madgwick normalized-gradient IMU/MARG',
    'integration': 'forward Euler quaternion derivative followed by normalization',
    'default_beta': DEFAULT_BETA,
    'beta_source': 'same fixed value as the existing author-example baseline',
    'report': REPORT_URL,
    'matlab_source': MATLAB_URL,
    'magnetic_reference_use': 'fixed world-to-magnetic-north heading conversion',
    'magnetic_dip': 'estimated from the current rotated measurement, as in original MARG',
    'adaptation': 'none: no bias estimation, measurement gating or gain schedule',
    'noise_weighting': 'standard algorithm normalizes vectors; no covariance weighting',
}


def _wxyz(rotation):
    return rotation.as_quat()[[3, 0, 1, 2]]


class MadgwickBaseline:
    """One causal IMU/MARG state, initialized by one supplied orientation.

    ``magnetic_reference_world`` is a known calibrated reference, not a series
    of true attitudes.  A missing magnetic sample means an IMU-only update;
    callers must pass held low-rate magnetometer values only on new-observation
    ticks.  ``acceleration_m_s2=None`` means no usable accelerometer observation
    and invokes the existing algorithm's gyro-only branch.
    """
    def __init__(self, initial_wxyz, magnetic_reference_world=None,
                 beta=DEFAULT_BETA):
        q = np.asarray(initial_wxyz, dtype=float)
        beta = float(beta)
        if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) == 0:
            raise ValueError('initial_wxyz must be a finite, nonzero quaternion')
        if not np.isfinite(beta) or beta < 0:
            raise ValueError('beta must be finite and nonnegative')
        self.beta = beta
        self._has_magnetic_reference = magnetic_reference_world is not None
        if self._has_magnetic_reference:
            reference = np.asarray(magnetic_reference_world, dtype=float)
            if reference.shape != (3,) or not np.isfinite(reference).all():
                raise ValueError('magnetic_reference_world must be a finite vector (3,)')
            self._world_from_north, self.heading_radians = heading_frame(reference)
        else:
            self._world_from_north = Rotation.identity()
            self.heading_radians = 0.0
        self._q_world_from_north = _wxyz(self._world_from_north)
        north_from_world = self._q_world_from_north * np.array([1., -1., -1., -1.])
        self._q_north = quaternion_product(north_from_world, q / np.linalg.norm(q))

    @property
    def quaternion_wxyz(self):
        """Current IMU-to-world quaternion; a caller-owned array."""
        return quaternion_product(self._q_world_from_north, self._q_north)

    def update(self, gyro_rad_s, acceleration_m_s2, dt, magnetic=None):
        """Apply this sample once and return the current world quaternion."""
        if magnetic is not None and not self._has_magnetic_reference:
            raise ValueError('A world magnetic reference is required for MARG updates')
        acceleration = (np.zeros(3) if acceleration_m_s2 is None
                        else np.asarray(acceleration_m_s2, dtype=float))
        self._q_north = madgwick_update(
            self._q_north, gyro_rad_s, acceleration, dt,
            magnetic=magnetic, beta=self.beta)
        return self.quaternion_wxyz
