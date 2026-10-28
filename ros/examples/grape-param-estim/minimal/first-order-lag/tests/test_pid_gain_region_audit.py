"""The historical local-surrogate entrypoint must not leak success information."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

FIRST_ORDER = Path(__file__).resolve().parents[1]
if str(FIRST_ORDER) not in sys.path:
    sys.path.insert(0, str(FIRST_ORDER))

import pid_gain_region as region  # noqa: E402
import run_all_pid_gain_regions as batch  # noqa: E402


def test_failed_surrogate_fit_retains_its_original_sample_identity():
    baseline = np.array([0.9, 0.8, 0.7])
    all_valid = np.ones(3, dtype=bool)
    outcomes = [(baseline, all_valid),
                (np.array([1.0, np.nan, 0.9]), np.array([True, False, True])),
                (np.array([0.8, 0.8, 0.5]), all_valid)]
    outcomes += [(baseline, all_valid)] * 4
    gains = {group: region.GainTriple(2.0, 1.0, 3.0) for group in region.PID_GROUPS}
    with patch.object(region, "_spectral_radii", side_effect=outcomes):
        radii, gradient, valid = region._local_radius_surrogate(
            group="xy", baseline_gains=gains, plants=(object(),) * 3,
            vehicle_model=object(), actuator_parameters=object(),
            controller_dt=0.01, fd_ratio=1.12,
        )
    assert np.array_equal(radii, baseline)
    assert gradient.shape == (3, 3)
    assert np.array_equal(valid, [True, False, True])
    fraction = region._stable_fraction_tensor(radii, gradient, np.zeros(1), valid)
    assert fraction.item() == pytest.approx(2.0 / 3.0)


def test_all_unresolved_surrogates_produce_zero_over_full_population():
    fraction = region._stable_fraction_tensor(
        np.array([0.9, 0.8, np.nan]), np.zeros((3, 3)), np.zeros(1),
        np.zeros(3, dtype=bool),
    )
    assert fraction.item() == 0.0
    exact = region._exact_stability_summary(
        np.array([0.9, 0.8, np.nan]), np.ones(3, dtype=bool),
    )
    assert exact == {"sample_count": 3, "stable_count": 2,
                     "stable_fraction": 2.0 / 3.0,
                     "pole_valid_count": 2, "unresolved_count": 1}


def test_success_is_only_loaded_after_all_maps_and_exact_comparison_uses_all_n(tmp_path, monkeypatch):
    events = []
    plotted = []
    failure_gains = [[3.0, 0.1, 1.0], [5.0, 1.0, 2.5],
                     [20.0, 1.0, 8.0], [4.0, 1.0, 2.0]]

    def estimate(rows):
        return {"case_name": "failure-audit", "input": {"vehicle_model": "unused.urdf"},
                "controller_timing": {"median_seconds": 0.01},
                "actuator_model": {"thrust_time_constant_seconds": 0.1},
                "controller": {"gains": {
                    group: dict(zip(("p_gain", "i_gain", "d_gain"), row))
                    for group, row in zip(region.PID_GROUPS, rows)}}}

    def load(path):
        success = Path(path).name == "success.json"
        events.append("load_success" if success else "load_failure")
        return estimate(np.full((4, 3), 100.0) if success else failure_gains)

    def surrogate(**kwargs):
        events.append("surrogate_" + kwargs["group"])
        assert [kwargs["baseline_gains"][group].array().tolist()
                for group in region.PID_GROUPS] == failure_gains
        return np.array([0.9, 0.8, np.nan]), np.zeros((3, 3)), np.array([True, False, False])

    def plot(**kwargs):
        events.append("plot_" + kwargs["group"])
        plotted.append(kwargs)

    monkeypatch.setattr(region, "load_estimate_json", load)
    monkeypatch.setattr(region, "load_vehicle_model", lambda _: SimpleNamespace(parameters=object()))
    monkeypatch.setattr(region, "actuator_parameters_from_estimate", lambda _: object())
    monkeypatch.setattr(region, "draw_quotient_coordinates", lambda *_: np.zeros((3, 13)))
    monkeypatch.setattr(region, "quotient_to_scale_free_plants", lambda *_: (object(),) * 3)
    monkeypatch.setattr(region, "_local_radius_surrogate", surrogate)
    monkeypatch.setattr(region, "_spectral_radii", lambda **_: (
        np.array([0.9, 0.8, np.nan]), np.array([True, True, False])))
    monkeypatch.setattr(region, "_plot", plot)
    args = argparse.Namespace(
        estimate_json=tmp_path / "failure.json", success_json=tmp_path / "success.json",
        output_dir=tmp_path / "output", covariance="likelihood", samples=3, seed=0,
        scale_min=0.5, scale_max=2.0, grid_size=3, fd_ratio=1.12,
        group=["xy", "yaw"], alpha=0.95,
    )
    summary = region.analyze(args)
    assert events.index("load_success") > events.index("plot_xy")
    assert events.index("load_success") > events.index("plot_yaw")
    assert summary["full_controller_references"]["recorded_success"]["stable_fraction"] == 2.0 / 3.0
    for row in summary["rows"]:
        assert row["recorded_exact_stable_fraction"] == 2.0 / 3.0
        assert row["surrogate_stable_fraction_min"] == 1.0 / 3.0
        assert row["sample_count"] == 3 and row["surrogate_valid_samples"] == 1
        assert row["success_overlay"] is None
        assert row["ratio_grid"] == [0.5, 1.0, 2.0]
        with np.load(args.output_dir / row["group"] / "gain_region.npz") as data:
            assert data["surrogate_baseline_spectral_radius"].shape == (3,)
            assert data["quotient_coordinates"].shape == (3, 13)
    assert all("success" not in kwargs for kwargs in plotted)


def test_batch_entrypoint_does_not_inject_a_success_dataset():
    commands = []
    with patch.object(batch.subprocess, "run", side_effect=lambda command, **_: (
            commands.append(command) or SimpleNamespace(returncode=0))):
        assert batch._run_group("failure1", "xy", []) == 0
    assert "--success-json" not in commands[0]
