"""ROS-independent likelihood evidence and fusion for probabilistic transforms.

The Bingham parameter matrices in this module act on quaternions ordered as
``[w, x, y, z]``.  Gaussian position likelihoods use canonical information
form, ``exp(-0.5 * x.T @ information @ x + information_vector.T @ x)``.
Independent likelihoods fuse by adding their natural parameters.
"""

from dataclasses import dataclass
from numbers import Integral
from typing import Optional, Tuple

import numpy as np


_SYMMETRY_TOLERANCE = 1e-10
_PSD_TOLERANCE = 1e-10


def _required_identifier(value, name):
    identifier = str(value).strip()
    if not identifier:
        raise ValueError("{} must be a non-empty string.".format(name))
    return identifier


def _optional_timestamp(value):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("timestamp must be a finite non-negative number.")
    timestamp = float(value)
    if not np.isfinite(timestamp) or timestamp < 0.0:
        raise ValueError("timestamp must be a finite non-negative number.")
    return timestamp


def _optional_sequence(value):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError("sequence must be a non-negative integer.")
    sequence = int(value)
    if sequence < 0:
        raise ValueError("sequence must be a non-negative integer.")
    return sequence


def _finite_array(values, shape, name):
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("{} must contain numeric values.".format(name)) from exc
    if array.shape != shape:
        raise ValueError("{} must have shape {}.".format(name, shape))
    if not np.all(np.isfinite(array)):
        raise ValueError("{} must contain only finite values.".format(name))
    return np.array(array, dtype=float, copy=True)


def _symmetric_matrix(values, size, name):
    matrix = _finite_array(values, (size, size), name)
    if not np.allclose(
        matrix,
        matrix.T,
        rtol=_SYMMETRY_TOLERANCE,
        atol=_SYMMETRY_TOLERANCE,
    ):
        raise ValueError("{} must be symmetric.".format(name))
    return 0.5 * (matrix + matrix.T)


def _psd_matrix(values, size, name):
    matrix = _symmetric_matrix(values, size, name)
    eigenvalues = np.linalg.eigvalsh(matrix)
    scale = max(1.0, float(np.linalg.norm(matrix, ord=np.inf)))
    if float(eigenvalues[0]) < -_PSD_TOLERANCE * scale:
        raise ValueError("{} must be positive semidefinite.".format(name))
    return matrix


def _readonly(array):
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class EvidenceProvenance:
    """Identity and ordering metadata retained for one fused contribution."""

    source_id: str
    evidence_kind: str
    timestamp: Optional[float]
    sequence: Optional[int]
    contributes_orientation: bool
    contributes_position: bool


@dataclass(frozen=True, eq=False)
class TransformEvidence:
    """One source's likelihood evidence for a directed transform.

    ``orientation_bingham`` is a symmetric 4x4 natural-parameter matrix in
    ``wxyz`` basis.  Position evidence must provide both canonical Gaussian
    parameters, or neither.
    """

    source_id: str
    parent_frame_id: str
    child_frame_id: str
    evidence_kind: str = "likelihood"
    orientation_bingham: Optional[np.ndarray] = None
    position_information: Optional[np.ndarray] = None
    position_information_vector: Optional[np.ndarray] = None
    timestamp: Optional[float] = None
    sequence: Optional[int] = None

    def __post_init__(self):
        source_id = _required_identifier(self.source_id, "source_id")
        evidence_kind = _required_identifier(self.evidence_kind, "evidence_kind")
        parent_frame_id = _required_identifier(self.parent_frame_id, "parent_frame_id")
        child_frame_id = _required_identifier(self.child_frame_id, "child_frame_id")
        if parent_frame_id == child_frame_id:
            raise ValueError("parent_frame_id and child_frame_id must differ.")

        has_orientation = self.orientation_bingham is not None
        has_information = self.position_information is not None
        has_information_vector = self.position_information_vector is not None
        if has_information != has_information_vector:
            raise ValueError(
                "position_information and position_information_vector must be provided together."
            )
        if not has_orientation and not has_information:
            raise ValueError("evidence must contain an orientation or position likelihood.")

        orientation_bingham = None
        if has_orientation:
            orientation_bingham = _readonly(
                _symmetric_matrix(self.orientation_bingham, 4, "orientation_bingham")
            )

        position_information = None
        position_information_vector = None
        if has_information:
            position_information = _psd_matrix(
                self.position_information,
                3,
                "position_information",
            )
            position_information_vector = _finite_array(
                self.position_information_vector,
                (3,),
                "position_information_vector",
            )
            projected_vector = position_information @ (
                np.linalg.pinv(position_information) @ position_information_vector
            )
            scale = max(1.0, float(np.linalg.norm(position_information_vector)))
            if not np.allclose(
                projected_vector,
                position_information_vector,
                rtol=_PSD_TOLERANCE,
                atol=_PSD_TOLERANCE * scale,
            ):
                raise ValueError(
                    "position_information_vector must lie in the range of "
                    "position_information."
                )
            position_information = _readonly(position_information)
            position_information_vector = _readonly(position_information_vector)

        object.__setattr__(self, "source_id", source_id)
        object.__setattr__(self, "evidence_kind", evidence_kind)
        object.__setattr__(self, "parent_frame_id", parent_frame_id)
        object.__setattr__(self, "child_frame_id", child_frame_id)
        object.__setattr__(self, "orientation_bingham", orientation_bingham)
        object.__setattr__(self, "position_information", position_information)
        object.__setattr__(
            self,
            "position_information_vector",
            position_information_vector,
        )
        object.__setattr__(self, "timestamp", _optional_timestamp(self.timestamp))
        object.__setattr__(self, "sequence", _optional_sequence(self.sequence))

    @property
    def orientation_bingham_likelihood(self):
        """Alias that makes the likelihood interpretation explicit."""

        return self.orientation_bingham

    @classmethod
    def from_gaussian_position(
        cls,
        source_id,
        parent_frame_id,
        child_frame_id,
        position_mean,
        position_covariance,
        orientation_bingham=None,
        evidence_kind="likelihood",
        timestamp=None,
        sequence=None,
    ):
        """Construct evidence from a proper Gaussian mean and covariance."""

        mean = _finite_array(position_mean, (3,), "position_mean")
        covariance = _psd_matrix(position_covariance, 3, "position_covariance")
        eigenvalues = np.linalg.eigvalsh(covariance)
        scale = max(1.0, float(np.linalg.norm(covariance, ord=np.inf)))
        if float(eigenvalues[0]) <= _PSD_TOLERANCE * scale:
            raise ValueError("position_covariance must be positive definite.")
        information = np.linalg.inv(covariance)
        return cls(
            source_id=source_id,
            parent_frame_id=parent_frame_id,
            child_frame_id=child_frame_id,
            orientation_bingham=orientation_bingham,
            evidence_kind=evidence_kind,
            position_information=information,
            position_information_vector=information @ mean,
            timestamp=timestamp,
            sequence=sequence,
        )


@dataclass(frozen=True, eq=False)
class FusedTransformEvidence:
    """Natural parameters and provenance produced by evidence fusion."""

    parent_frame_id: str
    child_frame_id: str
    orientation_bingham: Optional[np.ndarray]
    position_information: Optional[np.ndarray]
    position_information_vector: Optional[np.ndarray]
    provenance: Tuple[EvidenceProvenance, ...]

    @property
    def source_ids(self):
        return tuple(item.source_id for item in self.provenance)

    @property
    def orientation_bingham_likelihood(self):
        return self.orientation_bingham

    def gaussian_position(self):
        """Return fused ``(mean, covariance)`` when position is proper.

        Singular information is valid likelihood evidence, but it does not
        define a normalized Gaussian by itself and therefore raises here.
        """

        if self.position_information is None:
            return None
        eigenvalues = np.linalg.eigvalsh(self.position_information)
        scale = max(1.0, float(np.linalg.norm(self.position_information, ord=np.inf)))
        if float(eigenvalues[0]) <= _PSD_TOLERANCE * scale:
            raise ValueError(
                "fused position information is singular and has no finite covariance."
            )
        covariance = np.linalg.inv(self.position_information)
        mean = covariance @ self.position_information_vector
        return mean, covariance

    @property
    def position_mean(self):
        gaussian = self.gaussian_position()
        return None if gaussian is None else gaussian[0]

    @property
    def position_covariance(self):
        gaussian = self.gaussian_position()
        return None if gaussian is None else gaussian[1]


def fuse_transform_evidence(evidence, allow_duplicate_sources=False):
    """Fuse independent transform likelihoods by adding natural parameters.

    Duplicate source IDs are rejected unless explicitly allowed because two
    records from one source are not evidence of statistical independence by
    default.
    """

    contributions = tuple(evidence)
    if not contributions:
        raise ValueError("at least one evidence contribution is required.")
    for contribution in contributions:
        if not isinstance(contribution, TransformEvidence):
            raise TypeError("all contributions must be TransformEvidence instances.")

    parent_frame_id = contributions[0].parent_frame_id
    child_frame_id = contributions[0].child_frame_id
    seen_sources = set()
    orientation_bingham = None
    position_information = None
    position_information_vector = None
    provenance = []

    for contribution in contributions:
        if (
            contribution.parent_frame_id != parent_frame_id
            or contribution.child_frame_id != child_frame_id
        ):
            raise ValueError("all evidence must describe the same directed frame pair.")
        if not allow_duplicate_sources and contribution.source_id in seen_sources:
            raise ValueError(
                "duplicate source_id {!r} would double count evidence.".format(
                    contribution.source_id
                )
            )
        seen_sources.add(contribution.source_id)

        if contribution.orientation_bingham is not None:
            if orientation_bingham is None:
                orientation_bingham = np.zeros((4, 4), dtype=float)
            orientation_bingham += contribution.orientation_bingham
        if contribution.position_information is not None:
            if position_information is None:
                position_information = np.zeros((3, 3), dtype=float)
                position_information_vector = np.zeros(3, dtype=float)
            position_information += contribution.position_information
            position_information_vector += contribution.position_information_vector

        provenance.append(
            EvidenceProvenance(
                source_id=contribution.source_id,
                evidence_kind=contribution.evidence_kind,
                timestamp=contribution.timestamp,
                sequence=contribution.sequence,
                contributes_orientation=contribution.orientation_bingham is not None,
                contributes_position=contribution.position_information is not None,
            )
        )

    if orientation_bingham is not None:
        orientation_bingham = _readonly(0.5 * (orientation_bingham + orientation_bingham.T))
    if position_information is not None:
        position_information = _readonly(
            0.5 * (position_information + position_information.T)
        )
        position_information_vector = _readonly(position_information_vector)

    return FusedTransformEvidence(
        parent_frame_id=parent_frame_id,
        child_frame_id=child_frame_id,
        orientation_bingham=orientation_bingham,
        position_information=position_information,
        position_information_vector=position_information_vector,
        provenance=tuple(provenance),
    )


def fuse_evidence(evidence, allow_duplicate_sources=False):
    """Short alias for :func:`fuse_transform_evidence`."""

    return fuse_transform_evidence(
        evidence,
        allow_duplicate_sources=allow_duplicate_sources,
    )
