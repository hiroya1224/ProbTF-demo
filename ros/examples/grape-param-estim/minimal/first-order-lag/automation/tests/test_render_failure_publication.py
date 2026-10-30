from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pytest

AUTOMATION = Path(__file__).resolve().parents[1]
if str(AUTOMATION) not in sys.path:
    sys.path.insert(0, str(AUTOMATION))
import render_failure_publication as paper


@pytest.fixture
def historical_rendering(monkeypatch):
    """Read the archived visual helpers without using their numerical data."""
    scripts = paper.DEFAULT_SETTINGS.parent / "scripts"

    def load(name, filename):
        spec = importlib.util.spec_from_file_location(name, scripts / filename)
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        return module

    data = load("data_io", "data_io.py")
    palette = load("rate_colors", "rate_colors.py")
    plots = load("historical_pid_paper_figures", "plot_paper_figures.py")
    return SimpleNamespace(data=data, palette=palette, plots=plots)


def test_all_4096_palette_colors_match_historical_hex_anchor_rounding(historical_rendering):
    settings = json.loads(paper.DEFAULT_SETTINGS.read_text())
    rates = np.linspace(0., 1., 4096)
    expected = historical_rendering.palette.rate_colormap(settings)(rates)
    actual = paper.rate_colormap(settings)(rates)
    np.testing.assert_array_equal(actual, expected)


def test_nonuniform_field_faces_and_pd_cells_match_historical_mapping(historical_rendering):
    axes = (np.array([.2, 1., 3., 7.]), np.array([.1, .7, 2.]),
            np.array([.3, 1., 2.5]))
    # Adjacent equal cells and several distinct rates exercise both face removal
    # and exact rectangle merging; no thresholding or interpolation is allowed.
    values = np.array([0., 0., .25, .5, .75, 1., .5, .5, .25,
                       .25, .25, .5, 1., 1., .75, .25, .25, 0.,
                       .5, .5, .75, .25, .25, .5, 1., 1., .75,
                       1., .75, .5, .5, .25, 0., 0., .25, .5]).reshape(4, 3, 3)
    old_grid = historical_rendering.data.Grid(
        "failure1", "roll_pitch", 65,
        dict(zip(("p", "i", "d"), axes), survival_fraction=values), {}, {})
    old_faces, old_rates = historical_rendering.data.exposed_faces(old_grid)
    faces, rates = paper.exposed_faces(values, axes)
    np.testing.assert_array_equal(faces, old_faces)
    np.testing.assert_array_equal(rates, old_rates)
    _, old_slice, old_edges = historical_rendering.data.slice_at(old_grid, 1, 1)
    np.testing.assert_array_equal(values[:, 1, :].T, old_slice)
    new_edges = (paper.midpoint_edges(axes[0]), paper.midpoint_edges(axes[2]))
    old_vertices, old_slice_rates, old_rectangles = historical_rendering.plots.merge_uniform_cells(
        old_slice, *old_edges)
    vertices, slice_rates, rectangles = paper.merge_uniform_cells(values[:, 1, :].T, *new_edges)
    np.testing.assert_array_equal(vertices, old_vertices)
    np.testing.assert_array_equal(slice_rates, old_slice_rates)
    assert rectangles == old_rectangles
    settings = json.loads(paper.DEFAULT_SETTINGS.read_text())
    np.testing.assert_array_equal(paper.rate_colormap(settings)(rates),
                                  historical_rendering.palette.rate_colormap(settings)(old_rates))
    np.testing.assert_array_equal(paper.rate_colormap(settings)(slice_rates),
                                  historical_rendering.palette.rate_colormap(settings)(old_slice_rates))


def test_2d_and_3d_use_same_historical_rate_color_and_fixed_normalization(tmp_path, historical_rendering):
    grids = {}
    for case in paper.CASES:
        for group in paper.GROUPS:
            source_grid(tmp_path, case, group)
            grids[case, group] = paper.load_grid(tmp_path, case, group, 33)
    settings = json.loads(paper.DEFAULT_SETTINGS.read_text())
    expected_2d = historical_rendering.palette.rate_colormap(settings)(.5)
    expected_3d = np.array(expected_2d)
    expected_3d[3] = settings["voxel_opacity"]
    slice_figure = paper.comparison_slice(grids, settings)
    volume_figure = paper.comparison_volume(grids, settings)
    try:
        for ax in slice_figure.axes:
            field = ax.collections[0]
            assert (field.norm.vmin, field.norm.vmax) == (0., 1.)
            np.testing.assert_array_equal(field.to_rgba(field.get_array()),
                                          np.broadcast_to(expected_2d, (len(field.get_array()), 4)))
        for ax in volume_figure.axes:
            faces = np.asarray(ax.collections[0].get_facecolor())
            np.testing.assert_array_equal(faces, np.broadcast_to(expected_3d, faces.shape))
    finally:
        plt.close(slice_figure)
        plt.close(volume_figure)


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


def successful_reference(path, gains=None):
    gains = np.ones((4, 3)) if gains is None else np.asarray(gains)
    payload = {"schema": paper.SUCCESS_REFERENCE_SCHEMA, "gain_order": list(paper.SOURCE_GROUPS),
               "gains": gains.tolist(), "source": {"path": "successful_estimate.json", "sha256": "abc"}}
    path.write_text(json.dumps(payload))
    return paper.load_success_reference(path)


def test_successful_reference_schema_and_source_hash_are_explicit(tmp_path):
    path = tmp_path / "successful_gain_reference.json"
    reference = successful_reference(path)
    assert reference["role"] == "coordinates_only"
    assert reference["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert reference["gain_order"] == list(paper.SOURCE_GROUPS)
    assert reference["source"] == {"path": "successful_estimate.json", "sha256": "abc"}
    payload = json.loads(path.read_text())
    payload["gain_order"] = list(reversed(paper.SOURCE_GROUPS))
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="finite 4x3"):
        paper.load_success_reference(path)


@pytest.mark.parametrize("view", ["PD", "3D"])
def test_successful_coordinates_are_visible_without_modifying_grid_or_colors(tmp_path, view):
    grids = {}
    for case in paper.CASES:
        for group in paper.GROUPS:
            source_grid(tmp_path, case, group)
            grids[case, group] = paper.load_grid(tmp_path, case, group, 33)
    gains = np.ones((4, 3))
    gains[paper.SOURCE_GROUPS.index("roll_pitch")] = [13., 1., 20.]
    gains[paper.SOURCE_GROUPS.index("xy")] = [1.5, 1., 2.]
    reference = successful_reference(tmp_path / "successful_gain_reference.json", gains)
    before = {key: tuple(array.copy() for array in (*grid.axes, grid.stable_fraction,
                                                    grid.stable_count, grid.valid_count, grid.unresolved_count))
              for key, grid in grids.items()}
    settings = json.loads(paper.DEFAULT_SETTINGS.read_text())
    draw = paper.comparison_slice if view == "PD" else paper.comparison_volume
    figure = draw(grids, settings, reference)
    try:
        for ax, (case, group) in zip(figure.axes, ((case, group) for case in paper.CASES for group in paper.GROUPS)):
            grid = grids[case, group]
            success = paper.reference_gain(reference, group)
            point = next(item for item in ax.collections if item.get_gid() == "success_reference")
            failure = next(item for item in ax.collections if item.get_gid() == "failure_reference")
            if view == "PD":
                np.testing.assert_array_equal(point.get_offsets(), success[[0, 2]][None, :])
                assert (ax.collections[0].norm.vmin, ax.collections[0].norm.vmax) == (0, 1)
                np.testing.assert_array_equal(ax.collections[0].get_array(), [.5])
                limits = [ax.get_xlim(), ax.get_ylim()]
                shown = success[[0, 2]]
                if group == "z":
                    np.testing.assert_array_equal(failure.get_offsets(), point.get_offsets())
            else:
                np.testing.assert_array_equal(np.asarray(point._offsets3d).reshape(3), success)
                limits = [ax.get_xlim(), ax.get_ylim(), ax.get_zlim()]
                shown = success
                expected = np.array(paper.rate_colormap(settings)(.5))
                expected[3] = settings["voxel_opacity"]
                facecolors = np.asarray(ax.collections[0].get_facecolor())
                np.testing.assert_array_equal(facecolors, np.broadcast_to(expected, facecolors.shape))
                if group == "z":
                    np.testing.assert_array_equal(np.asarray(failure._offsets3d), np.asarray(point._offsets3d))
            assert all(lower < value < upper for (lower, upper), value in zip(limits, shown))
            if group == "z":
                assert len(failure.get_facecolors()) == 0  # purple outer ring and blue diamond
            annotation = next(text for text in ax.texts if text.get_gid() == "success_reference_label")
            assert "position only" not in annotation.get_text() and "rate" not in annotation.get_text()
            metadata = paper.success_panel_metadata(grid, reference, settings, view)
            assert metadata["role"] == "coordinates_only" and not metadata["stability_value_assigned"]
            assert metadata["inside_computed_box"] is (group != "roll_pitch")
            for previous, current in zip(before[case, group], (*grid.axes, grid.stable_fraction,
                                                              grid.stable_count, grid.valid_count, grid.unresolved_count)):
                np.testing.assert_array_equal(current, previous)
            if group == "roll_pitch":
                assert metadata["computed_axis_limits"][2] == [.25, 2.]
                assert metadata["display_axis_limits"][2][1] > 20
                # The colored geometry remains inside the original node domain.
                faces, _ = paper.exposed_faces(grid.stable_fraction, grid.axes)
                assert faces[:, :, 2].max() == 2
    finally:
        plt.close(figure)


def test_successful_reference_is_loaded_after_all_grids_and_saved_with_provenance(tmp_path, monkeypatch):
    root = tmp_path / "source"
    for case in paper.CASES:
        for group in paper.GROUPS:
            source_grid(root, case, group)
    reference_path = tmp_path / "successful_gain_reference.json"
    successful_reference(reference_path)
    calls = []
    load_grid, load_reference = paper.load_grid, paper.load_success_reference

    def tracked_grid(*args):
        calls.append("grid")
        return load_grid(*args)

    def tracked_reference(*args):
        calls.append("success")
        return load_reference(*args)

    monkeypatch.setattr(paper, "load_grid", tracked_grid)
    monkeypatch.setattr(paper, "load_success_reference", tracked_reference)
    settings = json.loads(paper.DEFAULT_SETTINGS.read_text())
    settings.update(formats=["png"], dpi=25)
    style = tmp_path / "settings.json"
    style.write_text(json.dumps(settings))
    records = paper.render(root, tmp_path / "figures", 33, style, "PD", reference_path)
    assert calls == ["grid"] * 8 + ["success"]
    recorded = records[0]["success_reference"]
    assert recorded["sha256"] == hashlib.sha256(reference_path.read_bytes()).hexdigest()
    assert recorded["role"] == "coordinates_only"
    assert recorded["panels"]["failure1"]["z"]["coincides_with_failure"]
    assert records[0]["references"]["failure1"]["z"]["stable_fraction"] == .5
    assert "with no assigned stability value" in records[0]["caption"]


def test_off_plane_success_reference_is_not_silently_drawn_as_a_slice_point(tmp_path):
    source_grid(tmp_path)
    grid = paper.load_grid(tmp_path, "failure1", "roll_pitch", 33)
    gains = np.ones((4, 3))
    gains[paper.SOURCE_GROUPS.index("roll_pitch"), 1] = 1.5
    reference = successful_reference(tmp_path / "successful_gain_reference.json", gains)
    settings = json.loads(paper.DEFAULT_SETTINGS.read_text())
    with pytest.raises(ValueError, match="does not lie"):
        paper.success_panel_metadata(grid, reference, settings, "PD")


def successful_comparison(path, root, reference):
    cases = {}
    for case in paper.CASES:
        coordinates = np.arange(4 * 13, dtype=np.float64).reshape(4, 13)
        sample_file = root / case / "plant_samples.npz"
        np.savez_compressed(sample_file, quotient_coordinates=coordinates)
        radii = [.9, .99, .8, 1.1] if case == "failure1" else [.9, 1.02, None, 1.5]
        cases[case] = {"sample_count": 4, "stable_count": 3 if case == "failure1" else 1,
                       "valid_count": 4 if case == "failure1" else 3,
                       "unresolved_count": 0 if case == "failure1" else 1,
                       "stable_fraction": .75 if case == "failure1" else .25,
                       "spectral_radius": radii,
                       "provenance": {"plant_samples_sha256": paper.sha256(sample_file),
                           "quotient_coordinates_sha256": hashlib.sha256(coordinates.tobytes()).hexdigest()}}
    payload = {"schema": paper.SUCCESS_COMPARISON_SCHEMA, "role": "independent_full_controller",
               "gain_order": list(paper.SOURCE_GROUPS), "gains": np.array(reference["gains"]).tolist(), "cases": cases}
    path.write_text(json.dumps(payload))
    return payload


def comparison_fixture(root, reference_path, comparison_path):
    grids = {}
    for case in paper.CASES:
        for group in paper.GROUPS:
            source_grid(root, case, group)
            grids[case, group] = paper.load_grid(root, case, group, 33)
    reference = successful_reference(reference_path)
    payload = successful_comparison(comparison_path, root, reference)
    return grids, reference, payload


@pytest.mark.parametrize("change,match", [
    ("gains", "full gains differ"), ("denominator", "ALL original"),
    ("valid_only_rate", "ALL original"), ("pole_recount", "ALL original"),
    ("sample_order", "same original plant samples"), ("file_hash", "same original plant samples")])
def test_independent_success_comparison_rejects_wrong_population_or_score(tmp_path, change, match):
    root = tmp_path / "source"
    path = tmp_path / "success_comparison.json"
    grids, reference, payload = comparison_fixture(root, tmp_path / "reference.json", path)
    if change == "gains":
        payload["gains"][0][0] = 2
    elif change == "denominator":
        payload["cases"]["failure2"]["sample_count"] = 3
    elif change == "valid_only_rate":
        payload["cases"]["failure2"]["stable_fraction"] = 1 / 3
    elif change == "pole_recount":
        payload["cases"]["failure1"]["spectral_radius"][0] = 1.1
    elif change == "sample_order":
        payload["cases"]["failure1"]["provenance"]["quotient_coordinates_sha256"] = "wrong"
    elif change == "file_hash":
        payload["cases"]["failure1"]["provenance"]["plant_samples_sha256"] = "wrong"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match=match):
        paper.load_success_comparison(path, reference, grids, root)


@pytest.mark.parametrize("view", ["PD", "3D"])
def test_independent_success_rates_are_distinct_from_unchanged_background(tmp_path, view):
    root = tmp_path / "source"
    path = tmp_path / "success_comparison.json"
    grids, reference, _ = comparison_fixture(root, tmp_path / "reference.json", path)
    comparison = paper.load_success_comparison(path, reference, grids, root)
    settings = json.loads(paper.DEFAULT_SETTINGS.read_text())
    before = {key: grid.stable_fraction.copy() for key, grid in grids.items()}
    draw = paper.comparison_slice if view == "PD" else paper.comparison_volume
    figure = draw(grids, settings, reference, comparison)
    try:
        for ax, (case, group) in zip(figure.axes, ((c, g) for c in paper.CASES for g in paper.GROUPS)):
            label = next(text for text in ax.texts if text.get_gid() == "success_reference_label")
            assert label.get_text().endswith("rate: " + ("75.0%" if case == "failure1" else "25.0%"))
            assert "position only" not in label.get_text()
            grid = grids[case, group]
            np.testing.assert_array_equal(grid.stable_fraction, before[case, group])
            assert grid.mark()["stable_fraction"] == .5
            metadata = paper.success_panel_metadata(grid, reference, settings, view, comparison)
            assert metadata["role"] == "coordinates_plus_independent_full_controller_comparison"
            assert not metadata["background_stability_value_assigned"]
            assert metadata["stability_value_scope"] == "independent_full_controller"
            assert metadata["independent_full_controller"]["sample_count"] == 4
            assert metadata["independent_full_controller"]["stable_fraction"] == (.75 if case == "failure1" else .25)
    finally:
        plt.close(figure)


def test_comparison_loaded_after_all_grids_and_reference_and_saved_with_separate_scope(tmp_path, monkeypatch):
    root = tmp_path / "source"
    reference_path, comparison_path = tmp_path / "reference.json", tmp_path / "comparison.json"
    comparison_fixture(root, reference_path, comparison_path)
    calls = []
    load_grid, load_reference, load_comparison = paper.load_grid, paper.load_success_reference, paper.load_success_comparison

    def tracked_grid(*args):
        calls.append("grid")
        return load_grid(*args)

    def tracked_reference(*args):
        calls.append("reference")
        return load_reference(*args)

    def tracked_comparison(*args):
        calls.append("comparison")
        return load_comparison(*args)

    monkeypatch.setattr(paper, "load_grid", tracked_grid)
    monkeypatch.setattr(paper, "load_success_reference", tracked_reference)
    monkeypatch.setattr(paper, "load_success_comparison", tracked_comparison)
    settings = json.loads(paper.DEFAULT_SETTINGS.read_text())
    settings.update(formats=["png"], dpi=25)
    style = tmp_path / "settings.json"
    style.write_text(json.dumps(settings))
    records = paper.render(root, tmp_path / "figures", 33, style, "PD", reference_path, comparison_path)
    assert calls == ["grid"] * 8 + ["reference", "comparison"]
    record = records[0]
    assert record["success_reference"]["role"] == "coordinates_plus_independent_full_controller_comparison"
    assert record["success_reference"]["independent_full_controller_comparison"]["sha256"] == paper.sha256(comparison_path)
    assert "not stability values of the background slice" in record["caption"]
    assert record["references"]["failure1"]["z"]["stable_fraction"] == .5
