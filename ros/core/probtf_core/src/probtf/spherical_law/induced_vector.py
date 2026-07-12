import numpy as np

from probtf.spherical_law.protocol import (
    DiracDirectionLaw,
    DiracVectorLaw,
    ScaledDirectionVectorLaw,
)


def vector_law_from_direction_backend(backend, orientation, vector, options):
    value = np.asarray(vector, dtype=float)
    if value.shape != (3,) or not np.all(np.isfinite(value)):
        raise ValueError("vector must be a finite vector with shape (3,).")
    radius = float(np.linalg.norm(value))
    if radius <= 1e-12:
        return DiracVectorLaw(vector=np.zeros(3))
    direction_law = backend.rotate_direction(orientation, value / radius, options)
    if isinstance(direction_law, DiracDirectionLaw):
        return DiracVectorLaw(vector=radius * direction_law.direction)
    return ScaledDirectionVectorLaw(
        approximation=direction_law.approximation,
        direction_law=direction_law,
        radius=radius,
    )

