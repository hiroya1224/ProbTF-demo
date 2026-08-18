#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
FIRST_ORDER = HERE.parent
if str(FIRST_ORDER) not in sys.path:
    sys.path.insert(0, str(FIRST_ORDER))

from pid_gain_contour import (  # noqa: E402
    ProjectionEvaluator,
    QuantileEvaluation,
    _normal_newton_projection,
)


class _SyntheticRobust:
    def __init__(self) -> None:
        self.base_gain = np.ones(3)

    def quantile(self, coordinate, *, want_gradient=True):
        value = np.asarray(coordinate, dtype=float)
        q = float(value @ value - 1.0)
        gradient = 2.0 * value if want_gradient else None
        return QuantileEvaluation(
            value=q,
            stable_fraction=float(q < 0.0),
            active_sample=0,
            log_spectral_radius=np.asarray((q,)),
            gradient=gradient,
            analytic_gradient=bool(want_gradient),
        )


def test_projection_hidden_minimum_uses_full_three_gain_scalar() -> None:
    evaluator = ProjectionEvaluator(
        _SyntheticRobust(),
        first_axis=0,
        second_axis=1,
        hidden_axis=2,
        lower=-2.0,
        upper=2.0,
        hidden_seed_count=9,
    )
    point = evaluator.evaluate((0.6, 0.8), want_gradient=True)
    assert abs(point.value) < 1.0e-10
    assert abs(point.hidden_coordinate) < 1.0e-7
    assert np.allclose(point.projected_gradient, (1.2, 1.6), atol=1.0e-7)


def test_normal_newton_projection_hits_implicit_boundary() -> None:
    evaluator = ProjectionEvaluator(
        _SyntheticRobust(),
        first_axis=0,
        second_axis=1,
        hidden_axis=2,
        lower=-2.0,
        upper=2.0,
        hidden_seed_count=9,
    )
    point = _normal_newton_projection(
        evaluator,
        (0.9, 0.9),
        -2.0,
        2.0,
    )
    assert abs(point.value) < 1.0e-8
    assert np.isclose(np.linalg.norm(point.coordinate), 1.0, atol=1.0e-7)
