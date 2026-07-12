import numpy as np

from probtf.distributions import OrientationKind
from probtf.geometry import quat_to_rotmat
from probtf.provenance import ApproximationInfo, ApproximationKind
from probtf.spherical_law.protocol import (
    DiracDirectionLaw,
    DiracVectorLaw,
    ScaledDirectionVectorLaw,
    TangentInducedDirectionLaw,
    TangentInducedVectorLaw,
    UniformDirectionLaw,
)
from probtf.spherical_law.tangent import induced_vector_moments_tangent


class TangentSurrogateIslBackend:
    """Explicitly approximate adapter for the existing JMAA tangent surrogate."""

    def rotate_direction(self, orientation, direction, options):
        value = np.asarray(direction, dtype=float)
        if value.shape != (3,) or not np.isclose(np.linalg.norm(value), 1.0, atol=1e-8):
            raise ValueError("direction must be a unit 3-vector.")
        if orientation.kind is OrientationKind.UNIFORM:
            return UniformDirectionLaw()
        vector_law = self.rotate_vector(orientation, direction, options)
        if isinstance(vector_law, DiracVectorLaw):
            return DiracDirectionLaw(
                approximation=vector_law.approximation,
                direction=vector_law.vector,
            )
        return TangentInducedDirectionLaw(
            approximation=vector_law.approximation,
            mean=vector_law.mean,
            covariance=vector_law.covariance,
            mode=vector_law.mode,
        )

    def rotate_vector(self, orientation, vector, options):
        value = np.asarray(vector, dtype=float)
        if value.shape != (3,) or not np.all(np.isfinite(value)):
            raise ValueError("vector must be a finite vector with shape (3,).")
        if np.linalg.norm(value) <= 1e-12:
            return DiracVectorLaw(vector=np.zeros(3))
        if orientation.kind is OrientationKind.DIRAC:
            return DiracVectorLaw(vector=quat_to_rotmat(orientation.mode_wxyz) @ value)
        if orientation.kind is OrientationKind.UNIFORM:
            return ScaledDirectionVectorLaw(
                direction_law=UniformDirectionLaw(),
                radius=float(np.linalg.norm(value)),
            )
        result = induced_vector_moments_tangent(value, orientation.parameter_matrix())
        approximation = ApproximationInfo(
            kind=ApproximationKind.TANGENT_SURROGATE,
            lossy=True,
            detail="JMAA leading-exponent tangent moment surrogate; not an exact ISL density.",
            source="symaware_grasp.prob_tf.tangent_surrogate",
        )
        return TangentInducedVectorLaw(
            approximation=approximation,
            mean=result.mean,
            covariance=result.cov,
            mode=result.mode,
            radius=result.radius,
        )
