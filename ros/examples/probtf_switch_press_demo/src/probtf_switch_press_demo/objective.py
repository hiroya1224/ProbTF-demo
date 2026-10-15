"""Closed-form overlap of the specified Gaussian--Bingham comparison laws."""
from functools import lru_cache

import numpy as np
from scipy.special import logsumexp

from probtf.bingham import bingham_log_normalizer
from probtf.distributions import OrientationKind


def _array(value, shape, name):
    result = np.array(value, dtype=float, copy=True)
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise ValueError(name + " must have shape " + str(shape) + " and finite entries")
    result.setflags(write=False)
    return result


def _symmetric(value, size, name):
    matrix = _array(value, (size, size), name)
    if not np.allclose(matrix, matrix.T, rtol=0., atol=1e-8):
        raise ValueError(name + " must be symmetric")
    result = (matrix + matrix.T)/2
    result.setflags(write=False)
    return result


def _covariance(value, name):
    covariance = _symmetric(value, 3, name)
    if np.linalg.eigvalsh(covariance)[0] < -1e-12*max(1., np.linalg.norm(covariance)):
        raise ValueError(name + " must be positive semidefinite")
    return covariance


def _orientation(parameter, orientation=None):
    if parameter is None:
        if orientation is None or orientation.kind is not OrientationKind.DIRAC:
            raise ValueError("a finite orientation parameter or an explicit Dirac law is required")
        return None, _array(orientation.reference_quaternion_wxyz, (4,), "Dirac quaternion")
    if orientation is not None and orientation.kind is OrientationKind.DIRAC:
        raise ValueError("a Dirac law must not be assigned a finite concentration")
    matrix = _symmetric(parameter, 4, "orientation parameter")
    # Gauge shifts cancel in every normalized density. Remove them before
    # adding parameters, avoiding cancellation of unnecessarily large shifts.
    canonical = matrix-np.linalg.eigvalsh(matrix)[-1]*np.eye(4)
    canonical = (canonical+canonical.T)/2
    canonical.setflags(write=False)
    return canonical, None


def _steps(value):
    if isinstance(value, bool) or int(value) != value or value < 1:
        raise ValueError("integration_steps must be a positive integer")
    return int(value)


@lru_cache(maxsize=4096)
def _diagonal_log_normalizer(eigenvalues, integration_steps):
    # Spectral invariance is exact; eigenvalues are never rounded or binned.
    return float(bingham_log_normalizer(np.diag(eigenvalues), integration_steps=integration_steps))


def _log_normalizer(parameter, integration_steps):
    values = np.linalg.eigvalsh(parameter)
    shift = float(values[-1])
    return _diagonal_log_normalizer(tuple(values-shift), integration_steps)+shift


class PoseOverlapObjective:
    """Keep one observation-time goal fixed while comparing Hand laws.

    The overlap formula is exact for the specified final product families.
    Existing deterministic Bingham normalizer numerics evaluate its Z terms;
    no pose samples, density quadrature or additional closure is introduced.
    """
    def __init__(self, goal, integration_steps=480):
        self.goal = goal
        self.integration_steps = _steps(integration_steps)
        self._mean = _array(goal.mean, (3,), "goal mean")
        self._covariance = _covariance(goal.covariance, "goal covariance")
        self._metadata = dict(getattr(goal, "metadata", {}))
        parameters = tuple(goal.orientation_parameters)
        orientations = tuple(getattr(goal, "orientations", (None,)*len(parameters)))
        weights = np.asarray(goal.weights, dtype=float)
        if (not parameters or len(orientations) != len(parameters) or weights.shape != (len(parameters),)
                or not np.all(np.isfinite(weights)) or np.any(weights < 0.) or not np.any(weights > 0.)):
            raise ValueError("goal mixture requires matching parameters and nonnegative weights with positive mass")
        if not np.isclose(weights.sum(), 1., rtol=0., atol=1e-12):
            raise ValueError("goal mixture weights must already be normalized by candidate construction")
        self._weights = weights.copy()
        self._weights.setflags(write=False)
        self._orientations = tuple(_orientation(parameter, orientation)
                                   for parameter, orientation in zip(parameters, orientations))
        with np.errstate(divide="ignore"):
            self._log_weights = np.log(self._weights)
        # The source goal object may belong to a caller. All quantities used
        # for comparison are frozen copies from this observation snapshot.
        self._goal_normalizers = self._normalizers(self.integration_steps)

    def _normalizers(self, steps):
        return tuple(None if parameter is None else _log_normalizer(parameter, steps)
                     for parameter, _dirac in self._orientations)

    def evaluate(self, hand):
        return self._evaluate(hand, self.integration_steps, self._goal_normalizers)

    def _evaluate(self, hand, steps, goal_normalizers):
        hand_metadata = dict(getattr(hand, "metadata", {}))
        hand_frame, goal_frame = hand_metadata.get("frame_id"), self._metadata.get("frame_id")
        if hand_frame is not None and goal_frame is not None and hand_frame != goal_frame:
            raise ValueError("Hand and goal must use the same comparison frame")
        mean = _array(hand.mean, (3,), "Hand mean")
        covariance = _covariance(hand.covariance, "Hand covariance")
        combined = covariance+self._covariance
        try:
            cholesky = np.linalg.cholesky(combined)
        except np.linalg.LinAlgError as error:
            raise ValueError("combined positional covariance must be positive definite") from error
        whitened = np.linalg.solve(cholesky, mean-self._mean)
        mahalanobis = float(whitened @ whitened)
        log_determinant = float(2*np.log(np.diag(cholesky)).sum())
        log_position = -.5*(3*np.log(2*np.pi)+log_determinant+mahalanobis)
        parameter, hand_dirac = _orientation(hand.orientation_parameter, getattr(hand, "orientation", None))
        hand_normalizer = None if parameter is None else _log_normalizer(parameter, steps)
        overlaps, product_normalizers = [], []
        for weight, (goal_parameter, goal_dirac), goal_normalizer in zip(
                self._weights, self._orientations, goal_normalizers):
            if weight == 0.:
                overlaps.append(float("-inf"))
                product_normalizers.append(None)
            elif hand_dirac is not None and goal_dirac is not None:
                raise ValueError("Dirac--Dirac orientation overlap has no density product with respect to S3 surface measure")
            elif hand_dirac is not None:
                overlaps.append(float(hand_dirac @ goal_parameter @ hand_dirac-goal_normalizer))
                product_normalizers.append(None)
            elif goal_dirac is not None:
                overlaps.append(float(goal_dirac @ parameter @ goal_dirac-hand_normalizer))
                product_normalizers.append(None)
            else:
                product = _log_normalizer(parameter+goal_parameter, steps)
                overlaps.append(product-hand_normalizer-goal_normalizer)
                product_normalizers.append(product)
        terms = self._log_weights+np.asarray(overlaps)
        log_orientation = float(logsumexp(terms))
        value = log_position+log_orientation
        if not np.isfinite(value):
            raise FloatingPointError("Gaussian--Bingham overlap must have finite positive mass")
        diagnostics = dict(method="gaussian_bingham_closed_form_overlap", log_overlap=value,
            log_position_overlap=log_position, position_mahalanobis_squared=mahalanobis,
            position_log_determinant=log_determinant, log_orientation_overlap=log_orientation,
            log_hand_normalizer=hand_normalizer, log_goal_normalizers=goal_normalizers,
            log_product_normalizers=tuple(product_normalizers),
            log_weighted_orientation_terms=tuple(float(term) for term in terms),
            orientation_component_responsibilities=tuple(float(weight) for weight in np.exp(terms-log_orientation)),
            normalizer_integration_steps=steps, orientation_measure="S3 surface measure; Z(0)=2*pi^2",
            frame_id=goal_frame or hand_frame, goal_source=dict(self._metadata), hand_source=hand_metadata)
        return float(value), diagnostics

    def finite_difference_gradient(self, hand_at, command, step=1e-5):
        command = np.asarray(command, dtype=float)
        if command.ndim != 1 or not np.all(np.isfinite(command)) or not np.isfinite(step) or step <= 0.:
            raise ValueError("finite differences require a finite command vector and positive step")
        gradient = np.empty_like(command)
        for index in range(len(command)):
            delta = np.zeros_like(command)
            delta[index] = step
            gradient[index] = (self.evaluate(hand_at(command+delta))[0]
                               - self.evaluate(hand_at(command-delta))[0])/(2*step)
        return gradient

    def refinement_diagnostics(self, hand, refined_steps=None):
        """Compare existing normalizer evaluations; this is not an error bound."""
        refined_steps = _steps(2*self.integration_steps if refined_steps is None else refined_steps)
        base, _base_diagnostics = self.evaluate(hand)
        refined, _refined_diagnostics = self._evaluate(hand, refined_steps, self._normalizers(refined_steps))
        return dict(base_steps=self.integration_steps, refined_steps=refined_steps,
                    base_log_overlap=base, refined_log_overlap=refined,
                    log_overlap_change=refined-base, absolute_log_overlap_change=abs(refined-base),
                    error_bound=None, method="existing Bingham normalizer refinement")
