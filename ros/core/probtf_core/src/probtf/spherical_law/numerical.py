"""Exact/numerical backend insertion point.

The JMAA manuscript contains the exact global density, but no source evaluator
was present when this architecture was implemented.  This backend therefore
implements only the analytically safe Dirac, uniform, and zero-vector cases.
"""

from probtf.spherical_law.induced_direction import special_case_direction_law
from probtf.spherical_law.induced_vector import vector_law_from_direction_backend


class UnavailableExactIslBackend:
    def rotate_direction(self, orientation, direction, options):
        return special_case_direction_law(orientation, direction)

    def rotate_vector(self, orientation, vector, options):
        return vector_law_from_direction_backend(self, orientation, vector, options)


class NumericalIslBackend:
    """Interface placeholder for a future quadrature implementation."""

    def rotate_direction(self, orientation, direction, options):
        raise NotImplementedError("Numerical ISL integration is not implemented.")

    def rotate_vector(self, orientation, vector, options):
        return vector_law_from_direction_backend(self, orientation, vector, options)

