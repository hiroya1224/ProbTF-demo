"""Rotation and sphere geometry helpers."""

import math

import numpy as np

from probtf.geometry.quaternion import normalize_vec


def complete_orthonormal_basis(first_vec):
    basis_vectors = [normalize_vec(first_vec)]
    dimension = basis_vectors[0].shape[0]
    for candidate in np.eye(dimension, dtype=float):
        work = candidate.copy()
        for vector in basis_vectors:
            work -= float(np.dot(work, vector)) * vector
        norm = float(np.linalg.norm(work))
        if norm > 1e-8:
            basis_vectors.append(work / norm)
        if len(basis_vectors) == dimension:
            break
    if len(basis_vectors) != dimension:
        raise ValueError("Could not construct an orthonormal basis.")
    return np.column_stack(basis_vectors)


def tangent_projector(v):
    unit_v = normalize_vec(v)
    return np.eye(unit_v.shape[0], dtype=float) - np.outer(unit_v, unit_v)


def tangent_basis(v):
    unit_v = normalize_vec(v)
    projector = tangent_projector(unit_v)
    eigenvalues, eigenvectors = np.linalg.eigh(projector)
    order = np.argsort(eigenvalues)[::-1]
    basis = eigenvectors[:, order[:2]]
    basis[:, 0] = normalize_vec(basis[:, 0])
    residual = basis[:, 1] - float(np.dot(basis[:, 0], basis[:, 1])) * basis[:, 0]
    basis[:, 1] = normalize_vec(residual)
    return basis


def exp_s2(v, u, eps=1e-12):
    base = normalize_vec(v)
    tangent = np.asarray(u, dtype=float)
    norm = float(np.linalg.norm(tangent))
    if norm < eps:
        return base.copy()
    return normalize_vec(math.cos(norm) * base + math.sin(norm) * tangent / norm)


def skew(vector):
    x_value, y_value, z_value = np.asarray(vector, dtype=float).reshape(3)
    return np.array(
        [
            [0.0, -z_value, y_value],
            [z_value, 0.0, -x_value],
            [-y_value, x_value, 0.0],
        ],
        dtype=float,
    )

