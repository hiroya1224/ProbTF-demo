"""Passive chapter-4 directional tangent covariance for a learned Bingham A.

This is the finite leading-exponent (LE) Gaussian proxy in spherical logarithmic
coordinates, including the chapter's I/3 coordinate-Jacobian term.  It is not
the exact finite S2 covariance.  It changes neither A nor a training objective.
No sampling, concentration rescaling, variance floor, or pseudoinverse is used.
"""
from __future__ import annotations

import numpy as np

from figure_geometry import quaternion_left_matrix, quaternion_matrix


CHAPTER4_SOURCE = "/home/leus/dthesis/chapter/4-transform-law-propagation.tex"
CHAPTER4_LABELS = ("jmaa:eq:general_precision_operator",
                   "jmaa:eq:finite-local-covariance", "jmaa:eq:LE-covariance-by-type",
                   "jmaa:eq:LE-covariance-high-concentration",
                   "jmaa:prop:unique_maximizer")


def _tangent_basis(direction):
    # v=X gives [e_y,e_z], while v=Z gives [e_x,e_y].
    first = np.eye(3)[np.argmin(np.abs(direction))]
    first = first - (first @ direction)*direction
    first /= np.linalg.norm(first)
    return np.column_stack((first, np.cross(direction, first)))


def _geometry(matrix, v):
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("A must be a finite symmetric 4 by 4 matrix")
    if not np.allclose(matrix, matrix.T, rtol=0.,
                       atol=1e-12*max(1., float(np.max(np.abs(matrix))))):
        raise ValueError("A must be symmetric")
    direction = np.asarray(v, dtype=np.float64)
    if direction.shape != (3,) or not np.isfinite(direction).all() or np.linalg.norm(direction) == 0.:
        raise ValueError("v must be a finite nonzero three-vector")
    direction = direction/np.linalg.norm(direction)
    values, vectors = np.linalg.eigh(matrix)
    values, vectors = values[::-1], vectors[:, ::-1].copy()
    if vectors[np.argmax(np.abs(vectors[:, 0])), 0] < 0.:
        vectors[:, 0] *= -1.
    if np.linalg.det(vectors) < 0.:
        vectors[:, -1] *= -1.
    quaternion = vectors[:, 0]/np.linalg.norm(vectors[:, 0])
    rotation = quaternion_matrix(quaternion)
    frame = (quaternion_left_matrix(quaternion).T @ vectors[:, 1:])[1:]
    w = frame.T @ direction
    gaps = values[0]-values[1:]
    # These mu combinations account only for the irrelevant scalar trace
    # gauge.  The original learned A and its raw eigenvalues are not changed.
    mu = .5*np.array([values[0]+values[1]-values[2]-values[3],
                       values[0]+values[2]-values[1]-values[3],
                       values[1]+values[2]-values[0]-values[3]])
    # Equivalent to the chapter's B(mu) and tr(D)-w.T@D@w, without
    # cancellation in the latter near a weak eigenvalue gap.
    b = np.array([gaps[2]*gaps[1], gaps[2]*gaps[0], gaps[1]*gaps[0]])
    d = float(gaps @ (w*w))
    basis_input = _tangent_basis(direction)
    basis_output = rotation @ basis_input
    valid = False
    if gaps[0] > 0.:
        case = "unique_mode_nonpolar"
        valid = True
    elif gaps[1] == 0.:
        case = "uniform_law" if gaps[2] == 0. else "spherical_leading_exponent"
    elif np.array_equal(w[1:], np.zeros(2)):
        case = "polar_aligned_LE_only_nonGaussian_exact_limit"
        valid = True
    else:
        case = "latitude_ring_no_isolated_surrogate_mode"
    precision = None
    if d > 0.:
        rows = frame.T @ basis_input
        precision = sum((b[i]*np.outer(rows[i], rows[i]) for i in range(3)),
                        np.zeros((2, 2)))/(2.*d)
    elif valid and gaps[0] == 0.:
        # Exactly aligned polar case: the explicit chapter-4 formula.
        precision = .5*gaps[1]*np.eye(2)
    return dict(matrix=matrix, direction=direction, values=values, quaternion=quaternion,
                frame=frame, w=w, gaps=gaps, mu=mu, b=b, d=d,
                basis_input=basis_input, basis_output=basis_output,
                mode=rotation @ direction, valid=valid, case=case, precision=precision)


def _inverse(precision, basis, structural_valid):
    result = {"status": "structurally_undefined", "covariance_defined": False,
              "covariance_2x2_rad2": None, "covariance_3x3_rad2": None,
              "principal_variance_rad2": None, "principal_sigma_deg": None}
    if not structural_valid or precision is None:
        return result
    try:
        covariance = np.linalg.inv(precision)
    except np.linalg.LinAlgError:
        result["status"] = "numerical_inverse_failure"
        return result
    if not np.isfinite(covariance).all():
        result["status"] = "numerical_inverse_nonfinite"
        return result
    variance = np.linalg.eigvalsh(covariance)
    result.update(covariance_2x2_rad2=covariance.tolist(),
                  covariance_3x3_rad2=(basis @ covariance @ basis.T).tolist(),
                  principal_variance_rad2=variance.tolist(),
                  principal_sigma_deg=[float(np.degrees(np.sqrt(value))) if value >= 0. else None
                                       for value in variance])
    if np.all(variance > 0.):
        result.update(status="defined_le_proxy", covariance_defined=True)
    else:
        result["status"] = "numerical_inverse_nonpositive"
    return result


def direction_tangent_covariance(A, v=(1., 0., 0.)) -> dict:
    """Lightweight JSON-compatible covariance of R(Q)v in the frame of A.

    v is the input/object direction, not an output direction at which to
    evaluate a scalar density.  For A_camera, the returned surrogate mode and
    3D covariance are in camera coordinates; principal variances are invariant
    under a fixed change of destination frame.

    For a unique top eigenvalue, the chapter's raw 2D tangent precision L is
    formed from the full learned eigensystem, and covariance=(L+I2/3)^(-1).
    Uniform and nonisolated ridge/sphere cases return undefined covariance
    rather than a finite result obtained solely from I2/3.  The exact polar
    aligned exception is explicitly marked as an LE proxy; its exact local
    limit can be non-Gaussian.  No finite quality threshold selects cases.
    """
    geometry = _geometry(A, v)
    leading = geometry["precision"]
    finite = None if leading is None else leading+np.eye(2)/3.
    result = _inverse(finite, geometry["basis_output"], geometry["valid"])
    result.update(
        structural_case=geometry["case"],
        input_direction=geometry["direction"].tolist(),
        surrogate_mode=geometry["mode"].tolist(),
        surrogate_mode_unique=geometry["valid"],
        basis_input=geometry["basis_input"].tolist(),
        basis_output=geometry["basis_output"].tolist(),
        leading_precision_2x2=None if leading is None else leading.tolist(),
        finite_precision_2x2=None if finite is None else finite.tolist(),
        raw_precision_eigenvalues=None if leading is None else np.linalg.eigvalsh(leading).tolist(),
        eigenvalues_descending=geometry["values"].tolist(),
        mode_eigenvalue_gaps=geometry["gaps"].tolist(),
        shape_mu=geometry["mu"].tolist(),
        diagonal_frame_H=geometry["frame"].tolist(),
        input_direction_diagonal=geometry["w"].tolist(),
        B_diagonal=geometry["b"].tolist(),
        d=geometry["d"],
        matrix_trace=float(np.trace(geometry["matrix"])),
        matrix_modified=False,
        covariance_floor=False,
        pseudoinverse_cutoff=False,
        interpretation="finite LE proxy covariance in spherical logarithmic coordinates; includes I2/3 Jacobian precision",
        source=CHAPTER4_SOURCE,
        source_labels=list(CHAPTER4_LABELS),
    )
    return result


def audit_direction(A, logZ=None, v=(1., 0., 0.)) -> dict:
    """Compatibility audit with finite LE and leading inverses, no quadrature.

    logZ is recorded for caller convenience; it has no role in tangent
    curvature.  This function is passive and does not change training/NLL.
    """
    result = direction_tangent_covariance(A, v)
    leading = result["leading_precision_2x2"]
    high = _inverse(None if leading is None else np.asarray(leading),
                    np.asarray(result["basis_output"]), result["surrogate_mode_unique"])

    def previous_names(value):
        return dict(status=value["status"],
                    covariance_2x2=value["covariance_2x2_rad2"],
                    covariance_3x3=value["covariance_3x3_rad2"],
                    covariance_eigenvalues_2d=value["principal_variance_rad2"],
                    principal_sigma_deg=value["principal_sigma_deg"])

    result["finite_LE"] = previous_names(result)
    result["high_concentration"] = previous_names(high)
    result["log_normalizer_s3"] = None if logZ is None else float(logZ)
    return result
