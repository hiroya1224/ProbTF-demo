"""ROS-independent port of the rigid-pair IMU extended least-squares estimator.

The source is ``urdf_estimation_with_imus`` (develop, 291f66f), especially
``estimator/extended_leastsq.py`` and ``utils/sequential_leastsq.py``.  Only
the unknown, fixed transform between two IMUs on ONE rigid body is ported.
There is no joint model, URDF, convergence latch, or point-pose substitution.

The convention throughout is ``x_parent = R(q) x_child + p``, q in wxyz.
The source posterior consists of a Bingham orientation and a Gaussian
position, without their cross-dependence.  That approximation, including the
source's unit residual covariance for position, is carried in the native
ProbTF record rather than being presented as an exact joint posterior.
"""

import numpy as np

from probtf.bingham import (
    bingham_second_moment,
    canonical_bingham_parameter,
    rotation_first_moment,
)
from probtf.distributions import (
    BinghamOrientation,
    ConditionalGaussianTranslation,
    TransformComponent,
    TransformDistribution,
    TransformDistributionStamped,
)
from probtf.geometry import quat_left_matrix, quat_right_matrix
from probtf.provenance import (
    ApproximationInfo,
    ApproximationKind,
    ComponentProvenance,
    TransformProvenance,
)
from relative_coords_from_paired_imu.imu_kinematics import ImuKinematics
from relative_coords_from_paired_imu.imu_relative_pose import skew


def _positive(value, name):
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("{} must be positive and finite.".format(name))
    return value


def _factor(value, name):
    value = _positive(value, name)
    if value > 1.0:
        raise ValueError("{} must be in (0, 1].".format(name))
    return value


def expected_acceleration_operator(observation):
    """E[[omega]x squared] + [alpha]x, including gyro covariance."""

    omega_cross = skew(observation.angular_velocity)
    covariance = observation.angular_velocity_covariance
    return (
        omega_cross @ omega_cross
        + covariance - np.trace(covariance) * np.eye(3)
        + skew(observation.angular_acceleration)
    )


def gyro_residual_covariance(parent_covariance, child_covariance):
    """The source's N (diag(C_parent, C_child) kron I/4) N.T.

    Right/left multiplication here deliberately follows ``get_Nmat`` in
    the source.  The observation H below follows ``get_Hmat``.  Keeping
    these two conventions explicit matches the source's uniform-quaternion
    second-moment approximation (Q ultimately depends on noise traces).
    """

    basis = np.column_stack((np.zeros(3), np.eye(3)))
    noise_matrix = np.hstack(
        [quat_right_matrix(axis, normalize_input=False) for axis in basis]
        + [-quat_left_matrix(axis, normalize_input=False) for axis in basis]
    )
    covariance = np.zeros((6, 6))
    covariance[:3, :3] = parent_covariance
    covariance[3:, 3:] = child_covariance
    result = noise_matrix @ np.kron(covariance, np.eye(4) * 0.25) @ noise_matrix.T
    return 0.5 * (result + result.T)


def gyro_bingham_likelihood(parent, child, residual_precision):
    """-H.T Q^-1 H / 2 for R(q) omega_child = omega_parent."""

    parent_pure = np.concatenate(([0.0], parent.angular_velocity))
    child_pure = np.concatenate(([0.0], child.angular_velocity))
    observation = (
        quat_left_matrix(parent_pure, normalize_input=False)
        - quat_right_matrix(child_pure, normalize_input=False)
    )
    result = -0.5 * observation.T @ residual_precision @ observation
    return 0.5 * (result + result.T)


class ExtendedLeastSquaresImuRelativePoseEstimator:
    """Keep updating a Gaussian/Bingham relative-transform distribution.

    ``use_child_gyro=True`` preserves the source's two 3-D position fits and
    its precision-weighted combination.  ``False`` uses its averaged 3-D
    operator. Q is recomputed from each aligned pair's gyro covariance,
    including the covariance changes caused by asynchronous SG interpolation.

    The finite position prior keeps the source's 1e30 m^2 per fit.  Position
    updates accumulate information and use ordinary solves rather than the
    source's repeated pseudoinverses.  An actually unsolvable update raises;
    it does not replace a poorly determined direction by zero variance.
    ``position_observation_weight`` has the source's gamma meaning: it
    weights NEW position information; it does not forget old information.
    """

    orientation_type = BinghamOrientation

    def __init__(
        self,
        parent_frame_id,
        child_frame_id,
        use_child_gyro=True,
        rotation_forgetting_factor=1.0,
        position_observation_weight=1.0,
        position_residual_variance=1.0,
        prior_position_variance=1e30,
        integration_steps=50,
        source_id="imu_extended_leastsq",
        edge_id=None,
        authority=None,
    ):
        self.parent_frame_id = str(parent_frame_id).strip().lstrip("/")
        self.child_frame_id = str(child_frame_id).strip().lstrip("/")
        if not self.parent_frame_id or not self.child_frame_id:
            raise ValueError("Both IMU frame identifiers are required.")
        if self.parent_frame_id == self.child_frame_id:
            raise ValueError("The two IMU frame identifiers must differ.")
        self.use_child_gyro = bool(use_child_gyro)
        self.rotation_forgetting_factor = _factor(
            rotation_forgetting_factor, "rotation_forgetting_factor"
        )
        self.position_observation_weight = _factor(
            position_observation_weight, "position_observation_weight"
        )
        self.position_residual_variance = _positive(
            position_residual_variance, "position_residual_variance"
        )
        self.prior_position_variance = _positive(
            prior_position_variance, "prior_position_variance"
        )
        self.integration_steps = int(integration_steps)
        if self.integration_steps < 1 or self.integration_steps != integration_steps:
            raise ValueError("integration_steps must be a positive integer.")
        self.source_id = str(source_id).strip()
        self.edge_id = (
            "{}__to__{}".format(self.parent_frame_id, self.child_frame_id)
            if edge_id is None else str(edge_id).strip()
        )
        self.authority = self.source_id if authority is None else str(authority).strip()
        if not self.source_id or not self.edge_id or not self.authority:
            raise ValueError("source_id, edge_id, and authority must not be empty.")
        self.reset()

    def reset(self):
        self.rotation_parameter = np.zeros((4, 4))
        self.rotation_residual_precision = None
        dimension = 6 if self.use_child_gyro else 3
        self.position_information = np.eye(dimension) / self.prior_position_variance
        self.position_information_vector = np.zeros(dimension)
        self.position_mean = np.zeros(3)
        self.position_covariance = np.eye(3) * self.prior_position_variance / (
            2.0 if self.use_child_gyro else 1.0
        )
        self.last_stamp = None
        self.update_count = 0
        self._result = None

    def _check_pair(self, parent, child):
        if not isinstance(parent, ImuKinematics) or not isinstance(child, ImuKinematics):
            raise TypeError("Both observations must be ImuKinematics.")
        if (parent.frame_id, child.frame_id) != (self.parent_frame_id, self.child_frame_id):
            raise ValueError("Unexpected IMU frame identifiers.")
        if parent.stamp is None or child.stamp is None:
            raise ValueError("Both observations need acquisition timestamps.")
        if parent.stamp != child.stamp:
            raise ValueError("Evaluate both IMUs at the same acquisition time first.")
        if self.last_stamp is not None and parent.stamp <= self.last_stamp:
            raise ValueError("Pair timestamps must be strictly increasing; do not reuse observations.")

    def _position_coefficients(self, parent, child, mean_rotation):
        parent_operator = expected_acceleration_operator(parent)
        child_operator = expected_acceleration_operator(child)
        rotated_child_operator = mean_rotation @ child_operator @ mean_rotation.T
        force_delta = mean_rotation @ child.specific_force - parent.specific_force
        if not self.use_child_gyro:
            return 0.5 * (parent_operator + rotated_child_operator), force_delta
        coefficient = np.zeros((6, 6))
        coefficient[:3, :3] = parent_operator
        coefficient[3:, 3:] = rotated_child_operator
        return coefficient, np.tile(force_delta, 2)

    def _record(self, stamp, parameter, position, covariance):
        approximation = ApproximationInfo(
            kind=ApproximationKind.PRODUCER_SUPPLIED,
            lossy=True,
            detail=(
                "Port of urdf_estimation_with_imus 291f66f rigid-pair extended least squares. "
                "Position uses E[R], covariance-corrected gyro operators, and fixed "
                "residual variance={:g}, observation weight={:g}, prior variance={:g}. "
                "The source factors Gaussian position and Bingham orientation (B=0); "
                "rotation-position, repeated-force/two-fit, and temporal observation "
                "correlations are not estimated. Gyro residual covariance is computed "
                "from each aligned pair. No convergence latch or pose projection."
            ).format(
                self.position_residual_variance,
                self.position_observation_weight,
                self.prior_position_variance,
            ),
            source="relative_coords_from_paired_imu.imu_extended_leastsq",
        )
        method = "imu_extended_leastsq_{}".format("two_gyro" if self.use_child_gyro else "averaged_gyro")
        component = TransformComponent(
            component_id=self.edge_id + ":imu_extended_leastsq",
            raw_weight=1.0,
            orientation=self.orientation_type.from_parameter_matrix(parameter),
            translation=ConditionalGaussianTranslation(
                mean_at_reference=position,
                residual_covariance=covariance,
                rotation_coupling=np.zeros((3, 9)),
            ),
            provenance=ComponentProvenance(
                source_ids=(self.source_id,),
                method=method,
                detail="Rigid IMU pair; child coordinates map into parent coordinates.",
            ),
            approximation=approximation,
        )
        return TransformDistributionStamped(
            parent_frame_id=self.parent_frame_id,
            child_frame_id=self.child_frame_id,
            stamp=stamp,
            edge_id=self.edge_id,
            authority=self.authority,
            distribution=TransformDistribution((component,)),
            provenance=TransformProvenance(
                source_ids=(self.source_id,), method=method,
                detail="Sequential IMU Gaussian/Bingham posterior, update {}.".format(self.update_count + 1),
            ),
            approximation=approximation,
        )

    def update(self, parent, child):
        self._check_pair(parent, child)
        # The source's set_Qinv_once is called on every observation, and
        # BinghamHolder.update_Amat clears CovInv after each update. SG
        # interpolation changes these covariances even for constant raw noise.
        covariance = gyro_residual_covariance(
            parent.angular_velocity_covariance, child.angular_velocity_covariance
        )
        # An actually singular noise model cannot define this likelihood.
        # No variance floor, jitter, or pseudoinverse changes its meaning.
        np.linalg.cholesky(covariance)
        precision = np.linalg.solve(covariance, np.eye(4))
        parameter = canonical_bingham_parameter(
            self.rotation_forgetting_factor * self.rotation_parameter
            + gyro_bingham_likelihood(parent, child, precision)
        )
        # E[R] needs only E[qq.T]; contracting the source's fourth moments
        # with q.T q=1 gives precisely this same first rotation moment.
        mean_rotation = rotation_first_moment(
            bingham_second_moment(parameter, integration_steps=self.integration_steps)
        )
        coefficient, observation = self._position_coefficients(parent, child, mean_rotation)
        weight = self.position_observation_weight / self.position_residual_variance
        information = self.position_information + weight * coefficient.T @ coefficient
        information_vector = self.position_information_vector + weight * coefficient.T @ observation
        information = 0.5 * (information + information.T)
        if self.use_child_gyro:
            # Equivalent to the source's two Gaussian estimates followed by
            # its covariance-weighted get_estimates(), without two inverses.
            combined_information = information[:3, :3] + information[3:, 3:]
            combined_vector = information_vector[:3] + information_vector[3:]
        else:
            combined_information = information
            combined_vector = information_vector
        position = np.linalg.solve(combined_information, combined_vector)
        covariance = np.linalg.solve(combined_information, np.eye(3))
        covariance = 0.5 * (covariance + covariance.T)
        record = self._record(parent.stamp, parameter, position, covariance)
        # Commit a complete finite update together, after constructing the
        # actual immutable record that will also be sent over ROS.
        self.rotation_parameter = parameter
        self.rotation_residual_precision = precision
        self.position_information = information
        self.position_information_vector = information_vector
        self.position_mean = position
        self.position_covariance = covariance
        self.last_stamp = parent.stamp
        self.update_count += 1
        self._result = record
        return record

    def result(self):
        if self._result is None:
            raise ValueError("No timestamped IMU pair has been processed yet.")
        return self._result
