"""Regression checks for slice semantics, checkpoint recovery and VTK axis order."""

from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace
from xml.etree import ElementTree as ET

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pid_gain_voxels import (GAIN_CAPS, Volume, axes_for_group, coordinates_for_nodes)


def result(delta: float, radii: np.ndarray, valid: np.ndarray) -> SimpleNamespace:
    return SimpleNamespace(delta=delta, worst_spectral_radius=1 - delta,
                           worst_sample_index=0 if np.all(valid) else None,
                           pole_valid_mask=valid, near_kink_mask=np.zeros(len(radii), dtype=bool),
                           spectral_radius=radii)


def test_success_anchor_is_actual_controller_and_other_nine_gains_stay_fixed():
    base = np.array([[4, .1, 2], [5, 1, 2.5], [13, 1, 20], [6, 1, 2]], dtype=float)
    group = 2
    axes = axes_for_group(3, GAIN_CAPS[group], [base[group], [20, 1, 8]])
    index = np.array([[np.flatnonzero(axis == value)[0] for axis, value in zip(axes, base[group])],
                      [0, 1, 2]])
    q = coordinates_for_nodes(index, axes, base, group)
    recovered = q.reshape(-1, 4, 3) * GAIN_CAPS
    assert np.allclose(recovered[0], base, rtol=0, atol=1e-15)
    for other in (0, 1, 3):
        assert np.allclose(recovered[:, other], base[other], rtol=0, atol=1e-15)
    assert np.array_equal(recovered[1, group], [axes[j][index[1, j]] for j in range(3)])


def test_checkpoint_keeps_pending_unresolved_and_red_distinct_and_resumes(tmp_path):
    axes = [np.arange(2.0)] * 3
    volume = Volume(tmp_path / "v", axes, 2, "one", "xy", np.ones((4, 3)))
    volume.store(np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]), [
        result(.1, np.array([.9, .8]), np.ones(2, dtype=bool)),
        result(-.2, np.array([1.2, .9]), np.ones(2, dtype=bool)),
        result(np.nan, np.array([.9, np.nan]), np.array([True, False]))])
    reopened = Volume(tmp_path / "v", axes, 2, "one", "xy", np.ones((4, 3)))
    assert reopened.counts() == {"total": 8, "evaluated": 3, "pending": 5,
                                 "unresolved": 1, "non_positive_margin": 1, "positive_margin": 1}
    assert reopened.classes()[1, 1, 1] == 0
    assert np.isnan(reopened.fields["spectral_radius"][0, 1, 0, 1])
    with pytest.raises(ValueError, match="incompatible checkpoint"):
        Volume(tmp_path / "v", axes, 2, "two", "xy", np.ones((4, 3)))


def test_refinement_reuses_only_exact_nested_nodes(tmp_path):
    coarse_axes = axes_for_group(3, np.array([12, 12, 12]), [[4, .1, 2]])
    fine_axes = axes_for_group(5, np.array([12, 12, 12]), [[4, .1, 2]])
    coarse = Volume(tmp_path / "c", coarse_axes, 2, "one", "xy", np.ones((4, 3)))
    index = np.array([[1, 2, 3]])
    coarse.store(index, [result(.05, np.array([.95, .8]), np.ones(2, dtype=bool))])
    fine = Volume(tmp_path / "f", fine_axes, 2, "one", "xy", np.ones((4, 3)))
    assert fine.inherit(coarse) == 1
    fine_index = tuple(np.flatnonzero(fine_axes[j] == coarse_axes[j][index[0, j]])[0] for j in range(3))
    assert fine.fields["delta"][fine_index] == .05
    assert np.count_nonzero(fine.fields["evaluated"]) == 1
    assert np.array_equal(fine.fields["spectral_radius"][fine_index], [.95, .8])


def test_vtk_values_follow_p_fastest_axis_and_npz_preserves_array_order(tmp_path):
    axes = [np.array([0, 2]), np.array([0, .1, 1]), np.array([0, 4, 8, 12])]
    volume = Volume(tmp_path / "v", axes, 2, "one", "xy", np.ones((4, 3)))
    for index in np.ndindex(volume.shape):
        volume.fields["delta"][index] = index[0] + 10 * index[1] + 100 * index[2]
    volume.export(complete=True)
    tree = ET.parse(tmp_path / "v/volume.vtr")
    item = tree.find(".//PointData/DataArray[@Name='delta']")
    values = np.fromstring(item.text, sep=" ")
    assert np.array_equal(values[:6], [0, 1, 10, 11, 20, 21])
    assert values[6] == 100
    payload = np.load(tmp_path / "v/volume.npz")
    assert np.array_equal(payload["p"], axes[0])
    assert np.array_equal(payload["i"], axes[1])
    assert np.array_equal(payload["d"], axes[2])
    assert np.array_equal(payload["delta"], volume.fields["delta"])


def test_committed_checkpoint_cannot_silently_recreate_missing_data(tmp_path):
    directory = tmp_path / "v"
    volume = Volume(directory, [np.arange(2.0)] * 3, 2, "one", "xy", np.ones((4, 3)))
    volume.store(np.array([[0, 0, 0]]), [result(.1, np.array([.9, .8]), np.ones(2, dtype=bool))])
    (directory / "delta.npy").unlink()
    with pytest.raises(ValueError, match="missing fields"):
        Volume(directory, [np.arange(2.0)] * 3, 2, "one", "xy", np.ones((4, 3)))
