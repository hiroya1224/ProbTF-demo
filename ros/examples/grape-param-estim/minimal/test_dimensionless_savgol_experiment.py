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
