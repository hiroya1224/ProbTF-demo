#!/usr/bin/env python3
"""Small regression tests for the flat dimensionless SG experiment."""

from pathlib import Path

import numpy as np

import dimensionless_savgol_experiment as experiment


def _model_path() -> Path:
    return Path(__file__).resolve().with_name("grape_vehicle_model.json")


def test_parameterization_self_test() -> None:
    model = experiment.load_vehicle_model(_model_path())
    scales = experiment.ReferenceScales.from_reference(model.parameters)
    parameterization = experiment.DimensionlessParameterization(
        model.parameters,
        scales,
    )
    result = parameterization.self_test()
    assert result["physical_jacobian_relative_error"] < 2.0e-7


def test_common_scale_direction_is_straight() -> None:
    model = experiment.load_vehicle_model(_model_path())
    scales = experiment.ReferenceScales.from_reference(model.parameters)
    parameterization = experiment.DimensionlessParameterization(
        model.parameters,
        scales,
    )
    coordinate = np.zeros(experiment.PHYSICAL_DIMENSION)
    direction = parameterization.common_scale_direction()
    shift = 0.4

    first = parameterization.decode(coordinate)
    second = parameterization.decode(coordinate + shift * direction)
    factor = np.exp(shift)

    assert np.isclose(
        second.parameters.mass,
        factor * first.parameters.mass,
        rtol=1.0e-12,
        atol=0.0,
    )
    assert np.allclose(
        second.parameters.inertia,
        factor * first.parameters.inertia,
        rtol=1.0e-11,
        atol=1.0e-13,
    )
    assert np.allclose(
        second.parameters.force_effectiveness,
        factor * first.parameters.force_effectiveness,
        rtol=1.0e-12,
        atol=1.0e-14,
    )
    assert np.allclose(
        second.parameters.cog_offset,
        first.parameters.cog_offset,
        rtol=0.0,
        atol=1.0e-14,
    )


def test_random_chart_points_remain_physical() -> None:
    model = experiment.load_vehicle_model(_model_path())
    scales = experiment.ReferenceScales.from_reference(model.parameters)
    parameterization = experiment.DimensionlessParameterization(
        model.parameters,
        scales,
    )
    generator = np.random.default_rng(982451653)
    for _index in range(32):
        coordinate = generator.uniform(-1.5, 1.5, size=14)
        coordinate[7:10] *= 0.2
        decoded = parameterization.decode(coordinate)
        principal = np.linalg.eigvalsh(decoded.parameters.inertia)
        assert np.all(np.isfinite(principal))
        assert np.all(principal > 0.0)
        assert principal[0] + principal[1] > principal[2]

def test_gauge_constrained_lm_keeps_exact_gauge_and_does_not_threshold_weak_mode() -> None:
    from types import SimpleNamespace

    direction = experiment._normalized_scale_gauge(
        experiment.PHYSICAL_DIMENSION
    )
    projector = (
        np.eye(experiment.PHYSICAL_DIMENSION)
        - np.outer(direction, direction)
    )
    u, _s, _vt = np.linalg.svd(projector)
    basis = u[
        :,
        : experiment.PHYSICAL_DIMENSION - 1,
    ]
    singular = np.geomspace(
        1.0,
        1.0e-7,
        experiment.PHYSICAL_DIMENSION - 1,
    )
    jacobian = (
        np.diag(singular) @ basis.T
    )
    target = basis @ np.linspace(
        -0.4,
        0.4,
        experiment.PHYSICAL_DIMENSION - 1,
    )

    def evaluator(coordinate):
        residual = jacobian @ (
            np.asarray(
                coordinate,
                dtype=float,
            )
            - target
        )
        return SimpleNamespace(
            residual=residual,
            jacobian=jacobian,
        )

    arguments = SimpleNamespace(
        numeric_coordinate_guard=50.0,
        lm_initial_damping_relative=1.0e-3,
        lm_initial_trust_radius=1.0,
        lm_maximum_trust_radius=8.0,
        lm_minimum_trust_radius=1.0e-10,
        lm_acceptance_ratio=1.0e-4,
        ftol=1.0e-12,
        xtol=1.0e-12,
        gtol=1.0e-10,
    )
    initial = np.zeros(
        experiment.PHYSICAL_DIMENSION
    )
    lower = np.full_like(
        initial,
        -np.inf,
    )
    upper = np.full_like(
        initial,
        np.inf,
    )
    (
        coordinate,
        evaluation,
        payload,
    ) = experiment._adaptive_lm(
        evaluator,
        initial,
        lower,
        upper,
        160,
        True,
        arguments,
    )

    assert abs(
        direction @ coordinate
    ) < 1.0e-10
    assert payload[
        "ridge_threshold_used"
    ] is None
    assert not payload[
        "near_ridge_handling"
    ][
        "unknown_weak_modes_removed"
    ]
    assert 0.5 * float(
        evaluation.residual
        @ evaluation.residual
    ) < 1.0e-8
