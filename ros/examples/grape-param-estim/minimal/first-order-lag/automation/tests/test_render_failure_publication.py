from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pytest

AUTOMATION = Path(__file__).resolve().parents[1]
if str(AUTOMATION) not in sys.path:
    sys.path.insert(0, str(AUTOMATION))
import render_failure_publication as paper


def source_grid(root, case="failure1", group="roll_pitch", size=33):
    folder = root / case
    directory = folder / group / f"grid_{size}"
    directory.mkdir(parents=True)
    axes = (np.array([.25, 1., 2.]),) * 3
    n, shape = 4, (3, 3, 3)
    gains = np.ones((4, 3))
    metadata = {"schema": "grape-param-estim/exact-pid-gain-volume/v2-unconditional-failure-context",
                "group": group, "shape": list(shape), "axes": [a.tolist() for a in axes],
                "plant_samples": n, "full_fixed_controller_gains": gains.tolist()}
    (folder / "references.json").write_text(json.dumps({"recorded_case": {"gains": gains.tolist()}}))
    (directory / "metadata.json").write_text(json.dumps(metadata))
    (directory / "status.json").write_text(json.dumps({"complete": True, "pending": 0}))
    stable = np.full(shape, 2)
    valid = np.full(shape, 3)
    np.savez_compressed(directory / "stability_fraction.npz", p=axes[0], i=axes[1], d=axes[2],
                        stable_fraction=stable / n, stable_count=stable, valid_count=valid,
                        unresolved_count=n - valid, sample_count=n)
    np.save(directory / "spectral_radius.npy", np.broadcast_to([.9, .99, 1.02, np.nan], shape + (n,)).copy())
    np.save(directory / "evaluated.npy", np.ones(shape, dtype=bool))
    return directory


def test_load_preserves_full_denominator_and_source_bytes(tmp_path):
    directory = source_grid(tmp_path)
    source = directory / "stability_fraction.npz"
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    grid = paper.load_grid(tmp_path, "failure1", "roll_pitch", 33)
    assert grid.sample_count == 4
    assert grid.mark()["stable_fraction"] == .5
    assert grid.mark()["unresolved_count"] == 1
    assert grid.provenance["per_sample_check"] == "per-sample memmaps"
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before


def test_conditioning_away_unresolved_sample_is_rejected(tmp_path):
    directory = source_grid(tmp_path)
    source = directory / "stability_fraction.npz"
    with np.load(source) as saved:
        arrays = {name: saved[name].copy() for name in saved.files}
    arrays["stable_fraction"][:] = 2 / 3
    np.savez_compressed(source, **arrays)
    with pytest.raises(ValueError, match="ALL original"):
        paper.load_grid(tmp_path, "failure1", "roll_pitch", 33)


def test_other_nine_gain_context_cannot_be_replaced(tmp_path):
    directory = source_grid(tmp_path)
    path = directory / "metadata.json"
    data = json.loads(path.read_text())
    data["full_fixed_controller_gains"][0][0] = 19
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="other nine gains"):
        paper.load_grid(tmp_path, "failure1", "roll_pitch", 33)


def test_rate_is_independently_recounted_from_all_source_poles(tmp_path):
    directory = source_grid(tmp_path)
    path = directory / "spectral_radius.npy"
    values = np.load(path)
    values[0, 0, 0, 2] = .8
    np.save(path, values)
    with pytest.raises(ValueError, match="per-sample poles"):
        paper.load_grid(tmp_path, "failure1", "roll_pitch", 33)


def save_bits_in_place_of_radii(directory):
    radii = np.load(directory / "spectral_radius.npy")
    done = np.load(directory / "evaluated.npy")
    valid = np.isfinite(radii)
    stable = valid & (radii < 1)
    axes = np.array([.25, 1., 2.])
    np.savez_compressed(directory / "sample_bits.npz", p=axes, i=axes, d=axes,
                        stable_bits=np.packbits(stable, axis=-1, bitorder="little"),
                        valid_bits=np.packbits(valid, axis=-1, bitorder="little"),
                        evaluated=done, sample_count=4, bitorder="little",
                        stable_count=np.count_nonzero(stable, axis=-1),
                        valid_count=np.count_nonzero(valid, axis=-1))
    (directory / "spectral_radius.npy").unlink()
    (directory / "evaluated.npy").unlink()


def test_portable_sample_bits_recount_all_n_and_check_axes(tmp_path):
    directory = source_grid(tmp_path)
    save_bits_in_place_of_radii(directory)
    grid = paper.load_grid(tmp_path, "failure1", "roll_pitch", 33)
    assert grid.mark()["stable_fraction"] == .5
    assert grid.mark()["sample_count"] == 4 and grid.mark()["valid_count"] == 3
    assert grid.provenance["per_sample_check"] == "complete per-sample stable/valid bit archive"
    path = directory / "sample_bits.npz"
    with np.load(path) as saved:
        arrays = {key: saved[key].copy() for key in saved.files}
    arrays["p"][0] = .3
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError, match="bit archive axes"):
        paper.load_grid(tmp_path, "failure1", "roll_pitch", 33)


def test_stable_sample_bit_cannot_be_an_invalid_sample(tmp_path):
    directory = source_grid(tmp_path)
    save_bits_in_place_of_radii(directory)
    path = directory / "sample_bits.npz"
    with np.load(path) as saved:
        arrays = {key: saved[key].copy() for key in saved.files}
    # The fourth sample is invalid. Do not admit a falsely declared stable bit.
    arrays["stable_bits"][0, 0, 0, 0] |= np.uint8(1 << 3)
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError, match="stable sample bit"):
        paper.load_grid(tmp_path, "failure1", "roll_pitch", 33)


def test_faces_preserve_exact_rates_and_remove_only_equal_internal_faces():
    axes = (np.array([0., 2.]), np.array([0., 1.]), np.array([0., 3.]))
    values = np.full((2, 2, 2), .5)
    faces, rates = paper.exposed_faces(values, axes)
    assert faces.shape == (24, 4, 3)
    assert np.all(rates == .5)
    values[0, 0, 0] = .25
    faces, rates = paper.exposed_faces(values, axes)
    assert faces.shape == (30, 4, 3)
    assert np.count_nonzero(rates == .25) == 6
    assert set(rates) == {.25, .5}


def test_merged_pd_cells_preserve_nonuniform_geometry_and_values():
    values = np.array([[.25, .25], [.75, .75]])
    vertices, rates, rectangles = paper.merge_uniform_cells(values, [0., 1., 3.], [0., 2., 5.])
    assert len(rectangles) == 2
    areas = (vertices[:, 1, 0] - vertices[:, 0, 0]) * (vertices[:, 2, 1] - vertices[:, 1, 1])
    assert areas.sum() == 15
    assert np.dot(areas, rates) == 8.25


def test_old_visual_settings_and_eight_panel_layout_are_preserved(tmp_path):
    grids = {}
    for case in paper.CASES:
        for group in paper.GROUPS:
            source_grid(tmp_path, case, group)
            grids[case, group] = paper.load_grid(tmp_path, case, group, 33)
    settings = json.loads(paper.DEFAULT_SETTINGS.read_text())
    assert settings["white_mix"] == .75 and settings["voxel_opacity"] == .30
    assert settings["camera_oblique"] == {"elevation": 24, "azimuth": -58}
    figure = paper.comparison_slice(grids, settings)
    try:
        assert len(figure.axes) == 8
        assert np.array_equal(figure.get_size_inches(), [16, 8.6])
        assert [ax.get_title() for ax in figure.axes] == [paper.LABELS[g] for _ in paper.CASES for g in paper.GROUPS]
        assert not any("Success" in text.get_text() for ax in figure.axes for text in ax.texts)
    finally:
        plt.close(figure)


def test_publication_index_has_explicit_grid_filenames_and_actual_n(tmp_path):
    root = tmp_path / "source"
    for case in paper.CASES:
        for group in paper.GROUPS:
            source_grid(root, case, group)
    settings = json.loads(paper.DEFAULT_SETTINGS.read_text())
    settings.update(formats=["png"], dpi=25)
    style = tmp_path / "settings.json"
    style.write_text(json.dumps(settings))
    output = tmp_path / "figures"
    records = paper.render(root, output, 33, style, "PD")
    assert len(records) == 1
    record = records[0]
    assert record["grid_size"] == 33 and record["panel_count"] == 8
    assert record["files"]["png"]["path"] == "all_pid_failure_PD_grid_33.png"
    assert (output / record["files"]["png"]["path"]).stat().st_size > 0
    assert record["references"]["failure1"]["roll_pitch"]["stable_fraction"] == .5
    assert record["references"]["failure1"]["roll_pitch"]["sample_count"] == 4
    assert json.loads((output / "figure_index.json").read_text())[0]["color_encoding"]["continuous"]
