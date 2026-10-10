"""Scientific decisions, geometry and reference-slice invariants."""
import sys
import json
from pathlib import Path
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from data_io import (CASES, GROUPS, Grid, load_grid, midpoint_edges, unpack_samples,
                     survival_from_bits, exposed_faces, slice_at, marks_in_slice)
from plot_paper_figures import merge_uniform_cells
from rate_colors import rate_colormap, rate_colorscale


class PublicationDataTests(unittest.TestCase):
    def test_every_node_decision_recomputes_exactly(self):
        for case in CASES:
            for group in GROUPS:
                for size in (33, 65):
                    with self.subTest(case=case, group=group, size=size):
                        grid = load_grid(case, group, size)
                        values = grid.arrays
                        recovered = survival_from_bits(values["stable_bits"], values["valid_bits"], values["baseline_stable_mask"])
                        for key, array in zip(("survival_fraction", "caused_break_count", "conditional_unresolved_count"), recovered):
                            self.assertTrue(np.array_equal(array, values[key]))
                        self.assertTrue(np.array_equal(grid.classes(), values["classification"]))
                        self.assertEqual(grid.baseline_count, 127 if case == "failure1" else 128)
                        self.assertEqual(int(values["sample_count"]), 128)

    def test_reference_nodes_and_clearance(self):
        for case in CASES:
            for group in GROUPS:
                grid = load_grid(case, group)
                for name in ("success", "recorded"):
                    gain = grid.gain(name)
                    index = grid.index(gain)
                    for k, axis in enumerate(grid.axes):
                        self.assertGreater(index[k], 0)
                        self.assertLess(index[k], len(axis) - 1)
                        clearance = min(gain[k] - axis[0], axis[-1] - gain[k]) / (axis[-1] - axis[0])
                        self.assertGreaterEqual(clearance, 1 / 6 - 1e-12)
                    stable = unpack_samples(grid.arrays["stable_bits"][index], 128)
                    finite = unpack_samples(grid.arrays["valid_bits"][index], 128)
                    key = "success_spectral_radius" if name == "success" else "recorded_case_spectral_radius"
                    radii = grid.arrays[key]
                    self.assertTrue(np.array_equal(stable, np.isfinite(radii) & (radii < 1)))
                    self.assertTrue(np.array_equal(finite, np.isfinite(radii)))
                self.assertEqual(grid.mark("success")["survival"], 1)
                self.assertNotIn("classification", grid.mark("success"))
        self.assertEqual(load_grid("failure1", "roll_pitch").mark("recorded")["breaks"], 56)
        self.assertEqual(load_grid("failure2", "roll_pitch").mark("recorded")["breaks"], 110)
        self.assertEqual(load_grid("failure1", "yaw").mark("recorded")["breaks"], 4)
        self.assertEqual(load_grid("failure2", "yaw").mark("recorded")["breaks"], 2)

    def test_fixed_conditioning_set_and_unresolved(self):
        baseline = np.array([True, False, True, True])
        stable = np.array([[True, False, False, False]], dtype=bool)
        finite = np.array([[True, True, True, False]], dtype=bool)
        recovered = survival_from_bits(np.packbits(stable, axis=-1, bitorder="little"),
                                       np.packbits(finite, axis=-1, bitorder="little"), baseline)
        self.assertEqual(recovered[0][0], 1 - 2 / 3)
        self.assertEqual(recovered[1][0], 2)
        self.assertEqual(recovered[2][0], 1)

    def test_nonuniform_cells_and_internal_face_removal(self):
        axes = [np.array([0., 2.]), np.array([0., 1.]), np.array([0., 3.])]
        grid = Grid("synthetic", "xy", 2, {"p": axes[0], "i": axes[1], "d": axes[2],
                    "survival_fraction": np.ones((2, 2, 2))}, {}, {})
        faces, rates = exposed_faces(grid)
        self.assertEqual(len(faces), 24)
        for k, axis in enumerate(axes):
            self.assertGreaterEqual(faces[..., k].min(), axis[0])
            self.assertLessEqual(faces[..., k].max(), axis[-1])
        grid.arrays["survival_fraction"][:] = .99
        grid.arrays["survival_fraction"][0, 0, 0] = .96
        faces, rates = exposed_faces(grid)
        self.assertEqual(np.count_nonzero(rates == .96), 6)
        self.assertEqual(np.count_nonzero(rates == .99), 24)
        edges = midpoint_edges([0, .2, 1, 3])
        np.testing.assert_array_equal(edges, [0, .1, .6, 2, 3])
        for bad in ([0, 0, 1], [1, 0], [1]):
            with self.assertRaises(ValueError):
                midpoint_edges(bad)

    def test_slices_do_not_project_off_plane_markers(self):
        for case in CASES:
            grid = load_grid(case, "roll_pitch")
            index = grid.index(grid.gain("success"))
            self.assertEqual(marks_in_slice(grid, 2, index[2]), ["success"])
            index_r = grid.index(grid.gain("recorded"))
            self.assertEqual(marks_in_slice(grid, 2, index_r[2]), ["recorded"])
            self.assertEqual(marks_in_slice(grid, 1, index[1]), ["success", "recorded"])
            for axis in range(3):
                free, values, edges = slice_at(grid, axis, index[axis])
                self.assertEqual(values.shape, (len(grid.axes[free[1]]), len(grid.axes[free[0]])))
                self.assertEqual(values[index[free[1]], index[free[0]]], 1)

    def test_rate_palette_is_continuous_across_95_percent(self):
        settings = json.loads((ROOT / 'figure_settings.json').read_text())
        cmap = rate_colormap(settings)
        palette = rate_colorscale(settings)
        self.assertNotIn('alpha', settings)
        self.assertEqual(palette[0][0], 0)
        self.assertEqual(palette[-1][0], 1)
        rgb = cmap([.94, .95, .96])[:, :3]
        self.assertTrue(np.all(np.linalg.norm(np.diff(rgb, axis=0), axis=1) > 0))
        self.assertLess(np.max(np.abs(np.diff(rgb, axis=0))), .02)
        for q in (.141, .559, .969, .984):
            self.assertFalse(np.array_equal(cmap(q), cmap(0)))
            self.assertFalse(np.array_equal(cmap(q), cmap(1)))

    def test_vector_cell_merging_preserves_every_node_value(self):
        for case in CASES:
            for group in GROUPS:
                grid = load_grid(case, group)
                for axis in range(3):
                    index = grid.index(grid.gain("success"))[axis]
                    _, data, edges = slice_at(grid, axis, index)
                    for values in (data,):
                        vertices, colors, rectangles = merge_uniform_cells(values, *edges)
                        restored = np.full(values.shape, np.nan)
                        coverage = np.zeros(values.shape, dtype=int)
                        for first, last, start, stop, value in rectangles:
                            restored[first:last, start:stop] = value
                            coverage[first:last, start:stop] += 1
                        self.assertTrue(np.array_equal(values, restored))
                        self.assertTrue(np.all(coverage == 1))
                        self.assertEqual(len(vertices), len(colors))


if __name__ == "__main__":
    unittest.main()
