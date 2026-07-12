"""IMU quaternion-Bingham prediction and vector-alignment evidence.

Quaternions use ``[w, x, y, z]`` order.  The orientation maps vectors from
the child/body frame into the parent/reference frame, and body-frame angular
velocity increments compose on the right: ``q_next = q_current * delta_q``.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from probtf.bingham import (
    bingham_second_moment,
    canonical_bingham_parameter,
    match_bingham_to_second_moment,
    quaternion_product_second_moment,
)


_MATRIX_TOLERANCE = 1e-10


def _finite_vector3(values, name):
    try:
        vector = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("{} must contain numeric values.".format(name)) from exc
    if vector.shape != (3,):
        raise ValueError("{} must have shape (3,).".format(name))
    if not np.all(np.isfinite(vector)):
        raise ValueError("{} must contain only finite values.".format(name))
    return np.array(vector, dtype=float, copy=True)


def _unit_vector3(values, name):
    vector = _finite_vector3(values, name)
    norm = float(np.linalg.norm(vector))
    if norm <= _MATRIX_TOLERANCE:
        raise ValueError("{} must have non-zero norm.".format(name))
    return vector / norm


def _angular_velocity_covariance(values):
    if values is None:
        return np.zeros((3, 3), dtype=float)
    try:
        covariance = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "angular_velocity_covariance must contain numeric values."
        ) from exc
    if covariance.shape != (3, 3):
        raise ValueError("angular_velocity_covariance must have shape (3, 3).")
    if not np.all(np.isfinite(covariance)):
        raise ValueError(
            "angular_velocity_covariance must contain only finite values."
        )
    if not np.allclose(
        covariance,
        covariance.T,
        rtol=0.0,
        atol=_MATRIX_TOLERANCE,
    ):
        raise ValueError("angular_velocity_covariance must be symmetric.")
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    scale = max(1.0, float(np.linalg.norm(covariance, ord=np.inf)))
    if float(eigenvalues[0]) < -_MATRIX_TOLERANCE * scale:
        raise ValueError(
            "angular_velocity_covariance must be positive semidefinite."
        )
    eigenvalues = np.maximum(eigenvalues, 0.0)
    return eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T


def _time_step(value):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("dt must be a finite non-negative number.")
    try:
        dt = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("dt must be a finite non-negative number.") from exc
    if not np.isfinite(dt) or dt < 0.0:
        raise ValueError("dt must be a finite non-negative number.")
    return dt


def _positive_integer(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("{} must be a positive integer.".format(name))
    try:
        integer = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("{} must be a positive integer.".format(name)) from exc
    if integer < 1 or integer != value:
        raise ValueError("{} must be a positive integer.".format(name))
    return integer


def _delta_quaternion(angular_velocity, dt):
    rotation_vector = np.asarray(angular_velocity, dtype=float) * dt
    angle = float(np.linalg.norm(rotation_vector))
    if angle <= 1e-15:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    half_angle = 0.5 * angle
    return np.hstack(
        [np.cos(half_angle), rotation_vector * (np.sin(half_angle) / angle)]
    )


def delta_quaternion_second_moment(
    angular_velocity,
    dt,
    angular_velocity_covariance=None,
):
    """Approximate ``E[delta_q delta_q.T]`` for a Gaussian gyro sample.

    A third-degree spherical-radial cubature rule propagates the full 3x3
    angular-velocity covariance through the quaternion exponential.  It is
    exact for deterministic angular velocity and always returns a symmetric,
    positive-semidefinite, trace-one second moment.
    """

    mean = _finite_vector3(angular_velocity, "angular_velocity")
    covariance = _angular_velocity_covariance(angular_velocity_covariance)
    time_step = _time_step(dt)

    if time_step == 0.0 or not np.any(covariance):
        quaternion = _delta_quaternion(mean, time_step)
        return np.outer(quaternion, quaternion)

    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    square_root = eigenvectors @ np.diag(np.sqrt(np.maximum(eigenvalues, 0.0)))
    scale = np.sqrt(3.0)
    moment = np.zeros((4, 4), dtype=float)
    for column in range(3):
        offset = scale * square_root[:, column]
        for sample in (mean + offset, mean - offset):
            quaternion = _delta_quaternion(sample, time_step)
            moment += np.outer(quaternion, quaternion) / 6.0
    moment = 0.5 * (moment + moment.T)
    moment /= float(np.trace(moment))
    return moment


def predict_orientation_bingham(
    parameter_matrix,
    angular_velocity,
    dt,
    angular_velocity_covariance=None,
    integration_steps=120,
    max_iterations=80,
):
    """Predict a quaternion Bingham law from a body-frame gyro increment."""

    prior = canonical_bingham_parameter(parameter_matrix)
    mean = _finite_vector3(angular_velocity, "angular_velocity")
    covariance = _angular_velocity_covariance(angular_velocity_covariance)
    time_step = _time_step(dt)
    steps = _positive_integer(integration_steps, "integration_steps")
    iterations = _positive_integer(max_iterations, "max_iterations")

    if time_step == 0.0 or (
        not np.any(covariance) and not np.any(mean)
    ):
        return prior

    prior_moment = bingham_second_moment(prior, integration_steps=steps)
    increment_moment = delta_quaternion_second_moment(
        mean,
        time_step,
        angular_velocity_covariance=covariance,
    )
    predicted_moment = quaternion_product_second_moment(
        prior_moment,
        increment_moment,
    )
    return canonical_bingham_parameter(
        match_bingham_to_second_moment(
            predicted_moment,
            integration_steps=steps,
            max_iterations=iterations,
        )
    )


def vector_alignment_bingham_evidence(
    reference_vector_parent,
    observed_vector_body,
    concentration,
):
    """Return Bingham evidence for aligning one observed direction.

    The likelihood is proportional to
    ``exp(concentration * reference.T @ R(q) @ observed)``.  Both vectors are
    normalized, so ``concentration`` is the directional inverse-noise scale.
    A single vector intentionally leaves rotation about that vector
    unobservable.
    """

    reference = _unit_vector3(reference_vector_parent, "reference_vector_parent")
    observed = _unit_vector3(observed_vector_body, "observed_vector_body")
    try:
        weight = float(concentration)
    except (TypeError, ValueError) as exc:
        raise ValueError("concentration must be a finite positive number.") from exc
    if not np.isfinite(weight) or weight <= 0.0:
        raise ValueError("concentration must be a finite positive number.")

    dot_product = float(reference @ observed)
    parameter = np.zeros((4, 4), dtype=float)
    parameter[0, 0] = dot_product
    parameter[0, 1:] = np.cross(observed, reference)
    parameter[1:, 0] = parameter[0, 1:]
    parameter[1:, 1:] = (
        np.outer(reference, observed)
        + np.outer(observed, reference)
        - dot_product * np.eye(3, dtype=float)
    )
    return canonical_bingham_parameter(weight * parameter)


def gravity_bingham_evidence(
    reference_gravity_parent,
    observed_gravity_body,
    concentration,
):
    """Return gravity-direction alignment evidence without assuming its sign."""

    return vector_alignment_bingham_evidence(
        reference_gravity_parent,
        observed_gravity_body,
        concentration,
    )


def magnetic_bingham_evidence(
    reference_magnetic_parent,
    observed_magnetic_body,
    concentration,
):
    """Return magnetic-field direction alignment evidence."""

    return vector_alignment_bingham_evidence(
        reference_magnetic_parent,
        observed_magnetic_body,
        concentration,
    )


@dataclass(frozen=True, eq=False)
class OrientationEvidence:
    """One explicitly identified independent Bingham likelihood."""

    source_id: str
    parameter: np.ndarray
    kind: str = "other"

    def __post_init__(self):
        source_id = str(self.source_id).strip()
        kind = str(self.kind).strip()
        if not source_id:
            raise ValueError("source_id must be a non-empty string.")
        if not kind:
            raise ValueError("kind must be a non-empty string.")
        parameter = canonical_bingham_parameter(self.parameter)
        parameter = np.array(parameter, dtype=float, copy=True)
        parameter.setflags(write=False)
        object.__setattr__(self, "source_id", source_id)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "parameter", parameter)

    @property
    def parameter_matrix(self):
        return self.parameter


@dataclass(frozen=True, eq=False)
class OrientationFilterUpdate:
    """Separately exposed components from one predict/fuse cycle."""

    prior_parameter: np.ndarray
    delta_second_moment: np.ndarray
    prediction_parameter: np.ndarray
    gravity_evidence: Optional[np.ndarray]
    magnetic_evidence: Optional[np.ndarray]
    independent_evidence: Tuple[OrientationEvidence, ...]
    posterior_parameter: np.ndarray

    @property
    def evidence_source_ids(self):
        source_ids = []
        if self.gravity_evidence is not None:
            source_ids.append("gravity")
        if self.magnetic_evidence is not None:
            source_ids.append("magnetic")
        source_ids.extend(item.source_id for item in self.independent_evidence)
        return tuple(source_ids)


def _frozen_parameter(parameter_matrix):
    parameter = np.array(
        canonical_bingham_parameter(parameter_matrix),
        dtype=float,
        copy=True,
    )
    parameter.setflags(write=False)
    return parameter


def _frozen_moment(second_moment):
    moment = np.array(second_moment, dtype=float, copy=True)
    moment.setflags(write=False)
    return moment


class OrientationBinghamFilter:
    """Stateful Bingham filter with explicit independent evidence inputs."""

    def __init__(self, initial_parameter, integration_steps=120, max_iterations=80):
        self.integration_steps = _positive_integer(
            integration_steps,
            "integration_steps",
        )
        self.max_iterations = _positive_integer(max_iterations, "max_iterations")
        initial = _frozen_parameter(initial_parameter)
        self._posterior_parameter = initial
        self._prior_parameter = initial
        self._prediction_parameter = initial
        self._delta_second_moment = _frozen_moment(
            np.diag([1.0, 0.0, 0.0, 0.0])
        )
        self._gravity_evidence = None
        self._magnetic_evidence = None
        self._independent_evidence = ()
        self.last_update = None

    @property
    def parameter(self):
        return self._posterior_parameter

    @property
    def posterior_parameter(self):
        return self._posterior_parameter

    @property
    def prediction_parameter(self):
        return self._prediction_parameter

    @property
    def gravity_evidence(self):
        return self._gravity_evidence

    @property
    def magnetic_evidence(self):
        return self._magnetic_evidence

    @property
    def independent_evidence(self):
        return self._independent_evidence

    def predict(self, angular_velocity, dt, angular_velocity_covariance=None):
        """Advance the posterior through an independent gyro increment."""

        mean = _finite_vector3(angular_velocity, "angular_velocity")
        covariance = _angular_velocity_covariance(angular_velocity_covariance)
        time_step = _time_step(dt)
        self._prior_parameter = self._posterior_parameter
        self._delta_second_moment = _frozen_moment(
            delta_quaternion_second_moment(
                mean,
                time_step,
                angular_velocity_covariance=covariance,
            )
        )
        self._prediction_parameter = _frozen_parameter(
            predict_orientation_bingham(
                self._prior_parameter,
                mean,
                time_step,
                angular_velocity_covariance=covariance,
                integration_steps=self.integration_steps,
                max_iterations=self.max_iterations,
            )
        )
        self._posterior_parameter = self._prediction_parameter
        self._gravity_evidence = None
        self._magnetic_evidence = None
        self._independent_evidence = ()
        self.last_update = None
        return self._prediction_parameter

    def fuse_independent_evidence(
        self,
        gravity_evidence=None,
        magnetic_evidence=None,
        independent_evidence=(),
    ):
        """Fuse likelihoods once against the most recent prediction.

        Additional evidence must be supplied as :class:`OrientationEvidence`.
        Duplicate source IDs, including the reserved ``gravity`` and
        ``magnetic`` component IDs, are rejected to prevent double counting.
        Repeating this method replaces the previous evidence update rather
        than accumulating it a second time.
        """

        gravity = (
            None if gravity_evidence is None else _frozen_parameter(gravity_evidence)
        )
        magnetic = (
            None if magnetic_evidence is None else _frozen_parameter(magnetic_evidence)
        )
        additional = tuple(independent_evidence)
        for item in additional:
            if not isinstance(item, OrientationEvidence):
                raise TypeError(
                    "independent_evidence must contain OrientationEvidence instances."
                )

        source_ids = set()
        if gravity is not None:
            source_ids.add("gravity")
        if magnetic is not None:
            source_ids.add("magnetic")
        for item in additional:
            if item.source_id in source_ids:
                raise ValueError(
                    "duplicate evidence source_id {!r} would double count evidence.".format(
                        item.source_id
                    )
                )
            source_ids.add(item.source_id)

        posterior = np.array(self._prediction_parameter, dtype=float, copy=True)
        if gravity is not None:
            posterior += gravity
        if magnetic is not None:
            posterior += magnetic
        for item in additional:
            posterior += item.parameter
        posterior = _frozen_parameter(posterior)

        self._gravity_evidence = gravity
        self._magnetic_evidence = magnetic
        self._independent_evidence = additional
        self._posterior_parameter = posterior
        self.last_update = OrientationFilterUpdate(
            prior_parameter=self._prior_parameter,
            delta_second_moment=self._delta_second_moment,
            prediction_parameter=self._prediction_parameter,
            gravity_evidence=gravity,
            magnetic_evidence=magnetic,
            independent_evidence=additional,
            posterior_parameter=posterior,
        )
        return self.last_update

    def update(
        self,
        angular_velocity,
        dt,
        angular_velocity_covariance=None,
        gravity_evidence=None,
        magnetic_evidence=None,
        independent_evidence=(),
    ):
        """Predict from gyro data, then fuse explicitly independent evidence."""

        self.predict(
            angular_velocity,
            dt,
            angular_velocity_covariance=angular_velocity_covariance,
        )
        return self.fuse_independent_evidence(
            gravity_evidence=gravity_evidence,
            magnetic_evidence=magnetic_evidence,
            independent_evidence=independent_evidence,
        )


# Descriptive aliases for callers that prefer the vector modality in the name.
gravity_vector_bingham_evidence = gravity_bingham_evidence
magnetic_vector_bingham_evidence = magnetic_bingham_evidence
predict_bingham_from_gyro = predict_orientation_bingham


__all__ = [
    "OrientationBinghamFilter",
    "OrientationEvidence",
    "OrientationFilterUpdate",
    "delta_quaternion_second_moment",
    "gravity_bingham_evidence",
    "gravity_vector_bingham_evidence",
    "magnetic_bingham_evidence",
    "magnetic_vector_bingham_evidence",
    "predict_bingham_from_gyro",
    "predict_orientation_bingham",
    "vector_alignment_bingham_evidence",
]
