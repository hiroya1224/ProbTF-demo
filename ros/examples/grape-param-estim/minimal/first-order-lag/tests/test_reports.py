#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
FIRST_ORDER = HERE.parent
if str(FIRST_ORDER) not in sys.path:
    sys.path.insert(0, str(FIRST_ORDER))

from reports import residual_wrench_summary, wrench_lomb_scargle  # noqa: E402


def test_lomb_scargle_recovers_known_frequency() -> None:
    time = np.linspace(0.0, 8.0, 801)
    frequency = 0.63
    wrench = np.zeros((time.size, 6), dtype=float)
    wrench[:, 4] = 0.2 * time + np.sin(2.0 * np.pi * frequency * time)
    _grid, _power, peaks = wrench_lomb_scargle(time, wrench)
    assert abs(peaks[4] - frequency) < 0.01


def test_residual_summary_has_force_and_torque_vector_rms() -> None:
    time = np.linspace(0.0, 1.0, 101)
    wrench = np.zeros((time.size, 6), dtype=float)
    wrench[:, 0] = 3.0
    wrench[:, 3] = 2.0
    summary = residual_wrench_summary(time, wrench)
    assert np.isclose(summary["force_vector_rms_n"], 3.0)
    assert np.isclose(summary["torque_vector_rms_nm"], 2.0)
