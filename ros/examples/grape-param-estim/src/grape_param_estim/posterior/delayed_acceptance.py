"""Ridge-aware delayed-acceptance Metropolis-Hastings primitives."""

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import numpy as np


STATIC_PARAMETER_DIMENSION = 18
POSTERIOR_DIMENSION = 19


def _finite_vector(value: np.ndarray, size: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        raise ValueError(
            "{} must contain {} finite values".format(name, size)
        )
    result = result.copy()
    result.setflags(write=False)
    return result


def _positive_scalar(value: float, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError("{} must be finite and positive".format(name))
    return result


@dataclass(frozen=True)
class PosteriorPoint:
    """One unconstrained 18-D plant coordinate and bounded delay value."""

    static_coordinate: np.ndarray
    delay: float

    def __post_init__(self) -> None:
        coordinate = _finite_vector(
            self.static_coordinate,
            STATIC_PARAMETER_DIMENSION,
            "static_coordinate",
        )
        delay = float(self.delay)
        if not np.isfinite(delay):
            raise ValueError("delay must be finite")
        object.__setattr__(self, "static_coordinate", coordinate)
        object.__setattr__(self, "delay", delay)

    @classmethod
    def from_vector(cls, value: np.ndarray) -> "PosteriorPoint":
        vector = _finite_vector(value, POSTERIOR_DIMENSION, "posterior point")
        return cls(vector[:STATIC_PARAMETER_DIMENSION], vector[-1])

    @property
    def vector(self) -> np.ndarray:
        result = np.concatenate(
            (self.static_coordinate, np.asarray((self.delay,)))
        )
        result.setflags(write=False)
        return result

    @property
    def exact_cache_key(self) -> bytes:
        """Return the exact float64 bit pattern without numerical rounding."""

        return np.asarray(self.vector, dtype="<f8").tobytes(order="C")


@dataclass(frozen=True)
class TargetEvaluation:
    """One full nonlinear target evaluation and its warm-start payload."""

    point: PosteriorPoint
    log_density: float
    successful: bool
    failure_reason: str
    inner_iterations: int
    warm_start: Any = None
    graph_objective: Optional[float] = None
    local_log_determinant: Optional[float] = None
    delay_log_prior: Optional[float] = None

    def __post_init__(self) -> None:
        if not isinstance(self.point, PosteriorPoint):
            raise TypeError("point must be a PosteriorPoint")
        if not isinstance(self.successful, (bool, np.bool_)):
            raise TypeError("successful must be boolean")
        successful = bool(self.successful)
        log_density = float(self.log_density)
        if successful:
            if not np.isfinite(log_density):
                raise ValueError("a successful target log density must be finite")
            if self.failure_reason != "":
                raise ValueError("a successful evaluation cannot have a failure reason")
        else:
            if log_density != float("-inf"):
                raise ValueError("a failed target evaluation must have log density -inf")
            if not isinstance(self.failure_reason, str) or not self.failure_reason:
                raise ValueError("a failed evaluation needs a failure reason")
        if (
            isinstance(self.inner_iterations, (bool, np.bool_))
            or not isinstance(self.inner_iterations, (int, np.integer))
            or self.inner_iterations < 0
        ):
            raise ValueError("inner_iterations must be a non-negative integer")
        object.__setattr__(self, "successful", successful)
        object.__setattr__(self, "log_density", log_density)
        object.__setattr__(self, "inner_iterations", int(self.inner_iterations))
        component_names = (
            "graph_objective",
            "local_log_determinant",
            "delay_log_prior",
        )
        component_values = tuple(getattr(self, name) for name in component_names)
        supplied = tuple(value is not None for value in component_values)
        if any(supplied) and not all(supplied):
            raise ValueError(
                "target density components must be supplied together"
            )
        if all(supplied):
            if not successful:
                raise ValueError(
                    "a failed target evaluation cannot have density components"
                )
            converted = tuple(float(value) for value in component_values)
            if not np.all(np.isfinite(converted)):
                raise ValueError("target density components must be finite")
            objective, log_determinant, delay_prior = converted
            reconstructed = (
                delay_prior - objective - 0.5 * log_determinant
            )
            if not np.isclose(
                log_density,
                reconstructed,
                rtol=1.0e-12,
                atol=1.0e-12,
            ):
                raise ValueError(
                    "target components do not reconstruct log_density"
                )
            for name, value in zip(component_names, converted):
                object.__setattr__(self, name, value)

    @classmethod
    def failure(
        cls,
        point: PosteriorPoint,
        reason: str,
        inner_iterations: int = 0,
    ) -> "TargetEvaluation":
        return cls(
            point=point,
            log_density=float("-inf"),
            successful=False,
            failure_reason=reason,
            inner_iterations=inner_iterations,
            warm_start=None,
        )

    @property
    def has_density_components(self) -> bool:
        """Whether the exact target exposed its auditable decomposition."""

        return self.graph_objective is not None


class ExactTargetCache:
    """Cache only bit-identical posterior points, never rounded neighbours."""

    def __init__(self) -> None:
        self._evaluations: Dict[bytes, TargetEvaluation] = {}

    def get(self, point: PosteriorPoint) -> Optional[TargetEvaluation]:
        if not isinstance(point, PosteriorPoint):
            raise TypeError("point must be a PosteriorPoint")
        return self._evaluations.get(point.exact_cache_key)

    def store(self, evaluation: TargetEvaluation) -> None:
        if not isinstance(evaluation, TargetEvaluation):
            raise TypeError("evaluation must be a TargetEvaluation")
        key = evaluation.point.exact_cache_key
        existing = self._evaluations.get(key)
        if existing is not None and (
            existing.log_density != evaluation.log_density
            or existing.successful != evaluation.successful
            or existing.graph_objective != evaluation.graph_objective
            or existing.local_log_determinant
            != evaluation.local_log_determinant
            or existing.delay_log_prior != evaluation.delay_log_prior
        ):
            raise ValueError(
                "bit-identical target point produced inconsistent evaluations"
            )
        self._evaluations[key] = evaluation

    def evaluate(
        self,
        point: PosteriorPoint,
        evaluator: Callable[[PosteriorPoint, Any], TargetEvaluation],
        warm_start: Any,
    ) -> Tuple[TargetEvaluation, bool]:
        cached = self.get(point)
        if cached is not None:
            return cached, True
        evaluation = evaluator(point, warm_start)
        if not isinstance(evaluation, TargetEvaluation):
            raise TypeError("target evaluator must return TargetEvaluation")
        if evaluation.point.exact_cache_key != point.exact_cache_key:
            raise ValueError("target evaluator returned a different point")
        self.store(evaluation)
        return evaluation, False

    def __len__(self) -> int:
        return len(self._evaluations)


@dataclass(frozen=True)
class QuadraticSurrogate:
    """Local log-density surrogate with a positive-semidefinite information."""

    center: PosteriorPoint
    center_log_density: float
    information: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.center, PosteriorPoint):
            raise TypeError("center must be a PosteriorPoint")
        center_log_density = float(self.center_log_density)
        if not np.isfinite(center_log_density):
            raise ValueError("center_log_density must be finite")
        information = np.asarray(self.information, dtype=float)
        if information.shape != (POSTERIOR_DIMENSION, POSTERIOR_DIMENSION):
            raise ValueError("information must be a finite 19 by 19 matrix")
        if not np.all(np.isfinite(information)):
            raise ValueError("information must be a finite 19 by 19 matrix")
        symmetric = 0.5 * (information + information.T)
        asymmetry = float(np.max(np.abs(information - information.T)))
        scale = max(1.0, float(np.max(np.abs(symmetric))))
        if asymmetry > 1.0e-10 * scale:
            raise ValueError("information must be symmetric up to roundoff")
        eigenvalues = np.linalg.eigvalsh(symmetric)
        if eigenvalues[0] < -1.0e-10 * max(1.0, float(eigenvalues[-1])):
            raise ValueError("information must be positive semidefinite")
        symmetric.setflags(write=False)
        object.__setattr__(self, "center_log_density", center_log_density)
        object.__setattr__(self, "information", symmetric)

    def log_density(self, point: PosteriorPoint) -> float:
        if not isinstance(point, PosteriorPoint):
            raise TypeError("point must be a PosteriorPoint")
        difference = point.vector - self.center.vector
        return float(
            self.center_log_density
            - 0.5 * difference @ self.information @ difference
        )


class SymmetricProposalKernel:
    """Interface for state-independent symmetric random-walk proposals."""

    @property
    def name(self) -> str:
        raise NotImplementedError

    def propose(self, point: PosteriorPoint, random_state: Any) -> PosteriorPoint:
        raise NotImplementedError


@dataclass(frozen=True)
class GaussianSubspaceKernel(SymmetricProposalKernel):
    """Gaussian random walk in a fixed orthonormal subspace."""

    kernel_name: str
    basis: np.ndarray
    scales: np.ndarray

    def __post_init__(self) -> None:
        if (
            not isinstance(self.kernel_name, str)
            or not self.kernel_name
            or self.kernel_name.strip() != self.kernel_name
        ):
            raise ValueError("kernel_name must be a canonical non-empty string")
        basis = np.asarray(self.basis, dtype=float)
        scales = np.asarray(self.scales, dtype=float)
        if (
            basis.ndim != 2
            or basis.shape[0] != POSTERIOR_DIMENSION
            or basis.shape[1] == 0
            or not np.all(np.isfinite(basis))
        ):
            raise ValueError("basis must be a finite non-empty 19 by r matrix")
        if scales.shape != (basis.shape[1],) or not np.all(np.isfinite(scales)):
            raise ValueError("scales must have one finite value per basis vector")
        if np.any(scales <= 0.0):
            raise ValueError("proposal scales must be positive")
        if not np.allclose(
            basis.T @ basis,
            np.eye(basis.shape[1]),
            rtol=0.0,
            atol=1.0e-9,
        ):
            raise ValueError("proposal basis columns must be orthonormal")
        basis = basis.copy()
        scales = scales.copy()
        basis.setflags(write=False)
        scales.setflags(write=False)
        object.__setattr__(self, "basis", basis)
        object.__setattr__(self, "scales", scales)

    @property
    def name(self) -> str:
        return self.kernel_name

    def propose(self, point: PosteriorPoint, random_state: Any) -> PosteriorPoint:
        if not isinstance(point, PosteriorPoint):
            raise TypeError("point must be a PosteriorPoint")
        standard = np.asarray(
            random_state.normal(size=self.scales.size), dtype=float
        )
        if standard.shape != self.scales.shape or not np.all(np.isfinite(standard)):
            raise ValueError("random source returned invalid normal draws")
        increment = self.basis @ (self.scales * standard)
        return PosteriorPoint.from_vector(point.vector + increment)


@dataclass(frozen=True)
class ProposalMixture:
    """Fixed-weight mixture of symmetric state-independent kernels."""

    kernels: Tuple[SymmetricProposalKernel, ...]
    weights: np.ndarray

    def __post_init__(self) -> None:
        if type(self.kernels) is not tuple or not self.kernels:
            raise TypeError("kernels must be a non-empty tuple")
        names = []
        for kernel in self.kernels:
            if not isinstance(kernel, SymmetricProposalKernel):
                raise TypeError("kernels must implement SymmetricProposalKernel")
            names.append(kernel.name)
        if len(set(names)) != len(names):
            raise ValueError("proposal kernel names must be unique")
        weights = np.asarray(self.weights, dtype=float)
        if weights.shape != (len(self.kernels),) or not np.all(np.isfinite(weights)):
            raise ValueError("weights must have one finite value per kernel")
        if np.any(weights <= 0.0):
            raise ValueError("proposal mixture weights must be positive")
        weights = weights / np.sum(weights)
        weights.setflags(write=False)
        object.__setattr__(self, "weights", weights)

    def select(self, random_state: Any) -> SymmetricProposalKernel:
        index = int(random_state.choice(len(self.kernels), p=self.weights))
        if index < 0 or index >= len(self.kernels):
            raise ValueError("random source returned an invalid mixture index")
        return self.kernels[index]


def _embedded_static_basis(static_basis: np.ndarray) -> np.ndarray:
    basis = np.asarray(static_basis, dtype=float)
    result = np.zeros((POSTERIOR_DIMENSION, basis.shape[1]), dtype=float)
    result[:STATIC_PARAMETER_DIMENSION] = basis
    return result


def build_ridge_aware_proposal(
    static_information: np.ndarray,
    exact_ridge_direction: np.ndarray,
    delay_scale: float,
    *,
    local_scale: float = 0.5,
    exact_ridge_scale: float = 0.25,
    near_ridge_scale: float = 0.25,
    identified_scale: float = 0.1,
    near_relative_threshold: float = 1.0e-6,
    weights: Sequence[float] = (0.4, 0.2, 0.15, 0.15, 0.1),
) -> ProposalMixture:
    """Build the five fixed proposal components required by the v1 plan."""

    information = np.asarray(static_information, dtype=float)
    if information.shape != (
        STATIC_PARAMETER_DIMENSION,
        STATIC_PARAMETER_DIMENSION,
    ) or not np.all(np.isfinite(information)):
        raise ValueError("static_information must be a finite 18 by 18 matrix")
    symmetric = 0.5 * (information + information.T)
    if not np.allclose(information, symmetric, rtol=0.0, atol=1.0e-10):
        raise ValueError("static_information must be symmetric")
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    maximum = max(0.0, float(eigenvalues[-1]))
    if eigenvalues[0] < -1.0e-10 * max(1.0, maximum):
        raise ValueError("static_information must be positive semidefinite")
    threshold = _positive_scalar(
        near_relative_threshold, "near_relative_threshold"
    )
    floor = max(np.finfo(float).eps, maximum * threshold)
    relative = eigenvalues / maximum if maximum > 0.0 else np.zeros(18)
    near_mask = relative <= threshold
    if not np.any(near_mask):
        near_mask[0] = True
    identified_mask = ~near_mask
    if not np.any(identified_mask):
        identified_mask[-1] = True

    ridge = _finite_vector(
        exact_ridge_direction,
        STATIC_PARAMETER_DIMENSION,
        "exact_ridge_direction",
    )
    ridge_norm = float(np.linalg.norm(ridge))
    if ridge_norm == 0.0:
        raise ValueError("exact_ridge_direction cannot be zero")
    ridge_basis = _embedded_static_basis((ridge / ridge_norm)[:, None])

    local_basis = np.zeros((POSTERIOR_DIMENSION, POSTERIOR_DIMENSION))
    local_basis[:STATIC_PARAMETER_DIMENSION, :STATIC_PARAMETER_DIMENSION] = (
        eigenvectors
    )
    local_basis[-1, -1] = 1.0
    local_scales = np.concatenate(
        (
            _positive_scalar(local_scale, "local_scale")
            / np.sqrt(np.maximum(eigenvalues, floor)),
            np.asarray((_positive_scalar(delay_scale, "delay_scale"),)),
        )
    )
    near_basis = _embedded_static_basis(eigenvectors[:, near_mask])
    identified_basis = _embedded_static_basis(
        eigenvectors[:, identified_mask]
    )
    delay_basis = np.zeros((POSTERIOR_DIMENSION, 1), dtype=float)
    delay_basis[-1, 0] = 1.0

    kernels = (
        GaussianSubspaceKernel("local_gaussian", local_basis, local_scales),
        GaussianSubspaceKernel(
            "exact_ridge",
            ridge_basis,
            np.asarray((_positive_scalar(exact_ridge_scale, "exact_ridge_scale"),)),
        ),
        GaussianSubspaceKernel(
            "near_ridge",
            near_basis,
            np.full(
                near_basis.shape[1],
                _positive_scalar(near_ridge_scale, "near_ridge_scale"),
            ),
        ),
        GaussianSubspaceKernel(
            "identified_subspace",
            identified_basis,
            np.full(
                identified_basis.shape[1],
                _positive_scalar(identified_scale, "identified_scale"),
            ),
        ),
        GaussianSubspaceKernel(
            "delay_only",
            delay_basis,
            np.asarray((_positive_scalar(delay_scale, "delay_scale"),)),
        ),
    )
    return ProposalMixture(kernels, np.asarray(tuple(weights), dtype=float))


def delayed_acceptance_log_probabilities(
    current_exact_log_density: float,
    candidate_exact_log_density: float,
    current_surrogate_log_density: float,
    candidate_surrogate_log_density: float,
) -> Tuple[float, float]:
    """Return stage-one and exact correction log acceptance probabilities."""

    values = np.asarray(
        (
            current_exact_log_density,
            candidate_exact_log_density,
            current_surrogate_log_density,
            candidate_surrogate_log_density,
        ),
        dtype=float,
    )
    if not np.all(np.isfinite(values)):
        raise ValueError("acceptance log densities must be finite")
    surrogate_change = (
        candidate_surrogate_log_density - current_surrogate_log_density
    )
    exact_change = candidate_exact_log_density - current_exact_log_density
    return (
        min(0.0, float(surrogate_change)),
        min(0.0, float(exact_change - surrogate_change)),
    )


def _accept(log_probability: float, random_state: Any) -> bool:
    uniform = float(random_state.uniform())
    if not np.isfinite(uniform) or uniform < 0.0 or uniform > 1.0:
        raise ValueError("random source returned an invalid uniform draw")
    if uniform == 0.0:
        return True
    return bool(np.log(uniform) <= log_probability)


@dataclass(frozen=True)
class DelayedAcceptanceStep:
    """One proposal outcome with distinct stage-one and stage-two status."""

    current: TargetEvaluation
    candidate: PosteriorPoint
    kernel_name: str
    accepted: bool
    stage_one_accepted: bool
    stage_two_attempted: bool
    stage_two_accepted: bool
    full_target_cache_hit: bool
    inner_solve_failed: bool
    candidate_evaluation: Optional[TargetEvaluation]


class DelayedAcceptanceSampler:
    """Two-stage MH transition with an exact nonlinear second-stage target."""

    def __init__(
        self,
        surrogate: QuadraticSurrogate,
        proposal: ProposalMixture,
        delay_bounds: Tuple[float, float],
        target_evaluator: Callable[[PosteriorPoint, Any], TargetEvaluation],
        cache: Optional[ExactTargetCache] = None,
    ) -> None:
        if not isinstance(surrogate, QuadraticSurrogate):
            raise TypeError("surrogate must be a QuadraticSurrogate")
        if not isinstance(proposal, ProposalMixture):
            raise TypeError("proposal must be a ProposalMixture")
        if type(delay_bounds) is not tuple or len(delay_bounds) != 2:
            raise TypeError("delay_bounds must be a (lower, upper) tuple")
        lower, upper = (float(value) for value in delay_bounds)
        if not np.all(np.isfinite((lower, upper))) or lower >= upper:
            raise ValueError("delay_bounds must be finite and increasing")
        if not callable(target_evaluator):
            raise TypeError("target_evaluator must be callable")
        if cache is not None and not isinstance(cache, ExactTargetCache):
            raise TypeError("cache must be an ExactTargetCache")
        self.surrogate = surrogate
        self.proposal = proposal
        self.delay_bounds = (lower, upper)
        self.target_evaluator = target_evaluator
        self.cache = cache if cache is not None else ExactTargetCache()

    def step(
        self,
        current: TargetEvaluation,
        random_state: Any,
    ) -> DelayedAcceptanceStep:
        if not isinstance(current, TargetEvaluation) or not current.successful:
            raise ValueError("current target evaluation must be successful")
        kernel = self.proposal.select(random_state)
        candidate = kernel.propose(current.point, random_state)
        lower, upper = self.delay_bounds
        if candidate.delay < lower or candidate.delay > upper:
            return DelayedAcceptanceStep(
                current=current,
                candidate=candidate,
                kernel_name=kernel.name,
                accepted=False,
                stage_one_accepted=False,
                stage_two_attempted=False,
                stage_two_accepted=False,
                full_target_cache_hit=False,
                inner_solve_failed=False,
                candidate_evaluation=None,
            )

        current_surrogate = self.surrogate.log_density(current.point)
        candidate_surrogate = self.surrogate.log_density(candidate)
        stage_one_log_probability = min(
            0.0, candidate_surrogate - current_surrogate
        )
        if not _accept(stage_one_log_probability, random_state):
            return DelayedAcceptanceStep(
                current=current,
                candidate=candidate,
                kernel_name=kernel.name,
                accepted=False,
                stage_one_accepted=False,
                stage_two_attempted=False,
                stage_two_accepted=False,
                full_target_cache_hit=False,
                inner_solve_failed=False,
                candidate_evaluation=None,
            )

        candidate_evaluation, cache_hit = self.cache.evaluate(
            candidate,
            self.target_evaluator,
            current.warm_start,
        )
        if not candidate_evaluation.successful:
            return DelayedAcceptanceStep(
                current=current,
                candidate=candidate,
                kernel_name=kernel.name,
                accepted=False,
                stage_one_accepted=True,
                stage_two_attempted=True,
                stage_two_accepted=False,
                full_target_cache_hit=cache_hit,
                inner_solve_failed=True,
                candidate_evaluation=candidate_evaluation,
            )
        _, stage_two_log_probability = delayed_acceptance_log_probabilities(
            current.log_density,
            candidate_evaluation.log_density,
            current_surrogate,
            candidate_surrogate,
        )
        accepted = _accept(stage_two_log_probability, random_state)
        next_evaluation = candidate_evaluation if accepted else current
        return DelayedAcceptanceStep(
            current=next_evaluation,
            candidate=candidate,
            kernel_name=kernel.name,
            accepted=accepted,
            stage_one_accepted=True,
            stage_two_attempted=True,
            stage_two_accepted=accepted,
            full_target_cache_hit=cache_hit,
            inner_solve_failed=False,
            candidate_evaluation=candidate_evaluation,
        )


__all__ = [
    "DelayedAcceptanceSampler",
    "DelayedAcceptanceStep",
    "ExactTargetCache",
    "GaussianSubspaceKernel",
    "POSTERIOR_DIMENSION",
    "PosteriorPoint",
    "ProposalMixture",
    "QuadraticSurrogate",
    "STATIC_PARAMETER_DIMENSION",
    "SymmetricProposalKernel",
    "TargetEvaluation",
    "build_ridge_aware_proposal",
    "delayed_acceptance_log_probabilities",
]
