#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
FIRST_ORDER = HERE.parent
if str(FIRST_ORDER) not in sys.path:
    sys.path.insert(0, str(FIRST_ORDER))

from core import first_order_zoh_thrust_history  # noqa: E402


def test_constant_command_is_exact_fixed_point() -> None:
    result = first_order_zoh_thrust_history(
        command_times=np.asarray((0.0, 1.0)),
        command_values=np.asarray(((3.0, 4.0, 5.0, 6.0), (3.0, 4.0, 5.0, 6.0))),
        evaluation_times=np.asarray((0.25, 0.5, 1.0, 1.5)),
        time_constant=0.2,
        minimum_thrust=0.0,
        maximum_thrust=10.0,
    )
    expected = np.repeat(np.asarray(((3.0, 4.0, 5.0, 6.0),)), 4, axis=0)
    assert np.array_equal(result.thrust, expected)
    assert np.array_equal(result.log_tau_jacobian, np.zeros_like(expected))


def test_step_response_matches_closed_form() -> None:
    tau = 0.4
    result = first_order_zoh_thrust_history(
        command_times=np.asarray((0.0, 1.0)),
        command_values=np.asarray(((2.0, 2.0, 2.0, 2.0), (6.0, 6.0, 6.0, 6.0))),
        evaluation_times=np.asarray((1.0, 1.2, 1.7)),
        time_constant=tau,
        minimum_thrust=0.0,
        maximum_thrust=10.0,
    )
    expected_12 = 6.0 + np.exp(-0.2 / tau) * (2.0 - 6.0)
    expected_17 = 6.0 + np.exp(-0.7 / tau) * (2.0 - 6.0)
    assert np.allclose(result.thrust[0], 2.0)
    assert np.allclose(result.thrust[1], expected_12)
    assert np.allclose(result.thrust[2], expected_17)


def test_log_tau_derivative_matches_central_difference() -> None:
    tau = 0.31
    h = 1.0e-6
    kwargs = dict(
        command_times=np.asarray((0.0, 0.4, 0.9, 1.3)),
        command_values=np.asarray(
            (
                (2.0, 3.0, 4.0, 5.0),
                (6.0, 5.0, 4.0, 3.0),
                (4.0, 7.0, 2.0, 6.0),
                (5.0, 2.0, 6.0, 4.0),
            )
        ),
        evaluation_times=np.asarray((0.5, 0.8, 1.0, 1.5)),
        minimum_thrust=0.0,
        maximum_thrust=10.0,
    )
    center = first_order_zoh_thrust_history(time_constant=tau, **kwargs)
    plus = first_order_zoh_thrust_history(
        time_constant=tau * np.exp(h), **kwargs
    )
    minus = first_order_zoh_thrust_history(
        time_constant=tau * np.exp(-h), **kwargs
    )
    finite_difference = (plus.thrust - minus.thrust) / (2.0 * h)
    assert np.allclose(
        center.log_tau_jacobian,
        finite_difference,
        rtol=2.0e-7,
        atol=2.0e-8,
    )
