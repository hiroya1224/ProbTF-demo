"""Exact induced spherical density with a numerical Bingham normalizer."""

import numpy as np

from probtf.bingham import bingham_log_normalizer
from probtf.distributions import BinghamOrientation, OrientationKind
from probtf.provenance import ApproximationInfo, ApproximationKind

from probtf.spherical_law.induced_direction import (
    FiniteBinghamInducedDirectionLaw, special_case_direction_law,
)
from probtf.spherical_law.induced_vector import vector_law_from_direction_backend
from probtf.spherical_law.protocol import IslEvaluationOptions


class UnavailableExactIslBackend:
    """Compatibility backend that deliberately rejects finite Bingham laws."""

    def rotate_direction(self, orientation, direction, options):
        return special_case_direction_law(orientation, direction)

    def rotate_vector(self, orientation, vector, options):
        return vector_law_from_direction_backend(self, orientation, vector, options)


class ExactIslBackend:
    """Chapter-4 finite-Bingham fiber formula, plus exact singular cases.

    The normalizer uses the existing Chen--Tanaka Fourier calculation, with
    resolution doubling. The recorded difference is a convergence diagnostic,
    not a rigorous integration-error bound or a distribution approximation.
    """

    def __init__(self, bingham_integration_steps=120):
        steps = int(bingham_integration_steps)
        if isinstance(bingham_integration_steps, bool) or steps < 1 or steps != bingham_integration_steps:
            raise ValueError("bingham_integration_steps must be a positive integer.")
        self.bingham_integration_steps = steps

    def rotate_direction(self, orientation, direction, options):
        if not isinstance(orientation, BinghamOrientation):
            raise TypeError("orientation must be BinghamOrientation.")
        if not isinstance(options, IslEvaluationOptions):
            raise TypeError("options must be IslEvaluationOptions.")
        if orientation.kind is not OrientationKind.FINITE_BINGHAM:
            return special_case_direction_law(orientation, direction)
        parameter = orientation.backend_parameter_matrix()
        steps = self.bingham_integration_steps
        previous = bingham_log_normalizer(parameter, steps)
        for _refinement in range(5):
            steps *= 2
            current = bingham_log_normalizer(parameter, steps)
            relative_change = abs(float(np.expm1(current - previous)))
            # Require both absolute and relative resolution convergence. The
            # floating-point floor is scaled to this canonical normalizer.
            constant = float(np.exp(current))
            floor = 32. * np.finfo(float).eps
            if (relative_change <= max(options.relative_tolerance, floor) and
                    constant * relative_change <= max(options.absolute_tolerance, floor * constant)):
                approximation = ApproximationInfo(kind=ApproximationKind.NUMERICAL_INTEGRATION,
                    lossy=False, source="probtf.spherical_law.ExactIslBackend",
                    detail=("Exact induced-direction fiber formula; Chen--Tanaka normalizer, "
                            "integration_steps={}; relative_resolution_change={:.6g}; "
                            "resolution comparison is not a certified error bound.").format(steps, relative_change))
                return FiniteBinghamInducedDirectionLaw(approximation=approximation,
                    orientation=orientation, direction=direction, log_normalizer=current,
                    normalizer_integration_steps=steps)
            previous = current
        raise FloatingPointError("Bingham normalizer resolution did not converge at the requested tolerance.")

    def rotate_vector(self, orientation, vector, options):
        return vector_law_from_direction_backend(self, orientation, vector, options)


class NumericalIslBackend(ExactIslBackend):
    """Compatibility name for the implemented finite-Bingham density backend."""

