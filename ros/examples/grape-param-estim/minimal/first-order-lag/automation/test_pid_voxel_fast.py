"""Check equilibrium invariance, exact cohort agreement and refinement identity."""
from pathlib import Path
import unittest
import numpy as np

from pid_gain_voxels_fast import (
    HERE, GAIN_CAPS, PID_GROUPS, parser, input_fingerprint, load_configuration,
    load_estimate_json, _recorded_gain_matrix, axes_for_group,
)
from pid_voxel_fast_evaluator import scale_trim_seed, FastGainEvaluator


class FastChecks(unittest.TestCase):
    def test_controller_integral_outputs_and_orientation_are_preserved(self):
        base = _recorded_gain_matrix(load_estimate_json(HERE/'outputs/success/estimate.json'))
        warm = np.random.default_rng(42).normal(size=(128, 10))
        original = warm.copy()
        target = base.copy()
        target[:, 1] = [.025, .25, 3, .25]
        scaled = scale_trim_seed(warm, base, target)
        # Physical invariant: the six hover-control integral terms are equal.
        old_i = np.repeat(base[:, 1], [2, 1, 2, 1])
        new_i = np.repeat(target[:, 1], [2, 1, 2, 1])
        np.testing.assert_allclose(scaled[:, :6]*new_i, warm[:, :6]*old_i,
                                   rtol=2e-15, atol=0)
        np.testing.assert_array_equal(scaled[:, 6:], warm[:, 6:])
        np.testing.assert_array_equal(warm, original)
        self.assertFalse(np.shares_memory(warm, scaled))
        np.testing.assert_array_equal(scale_trim_seed(warm, base, base), warm)

    def test_zero_integral_keeps_original_conservative_seed(self):
        base = np.ones((4, 3))
        warm = np.arange(30).reshape(3, 10).astype(float)
        target = base.copy()
        target[:, 1] = 0
        np.testing.assert_array_equal(scale_trim_seed(warm, base, target), warm)

    def test_every_recorded_anchor_stays_inside_nested_65_axes(self):
        config = Path(__file__).with_name('pid_voxel_domains.json')
        bounds, anchors, report, _, clearance = load_configuration(
            config, HERE, load_estimate_json, _recorded_gain_matrix,
            [HERE/'outputs/failure1/estimate.json'])
        totals = {}
        for group in PID_GROUPS:
            coarse = axes_for_group(33, bounds[group], anchors[group], clearance)
            fine = axes_for_group(65, bounds[group], anchors[group], clearance)
            for old, new in zip(coarse, fine):
                self.assertTrue(np.all(np.isin(old, new)))
            self.assertGreaterEqual(report[group]['minimum_relative_clearance'], 1/6-1e-15)
            totals[group] = int(np.prod([len(axis) for axis in fine]))
        self.assertGreater(sum(totals.values()), 0)
        self.assertTrue(all(len(anchors[group])==1 for group in PID_GROUPS))

    def test_defaults_are_separate_and_fast_source_enters_fingerprint(self):
        args = parser().parse_args([])
        self.assertEqual(args.grid_sizes, [3,5,9,17,33,65])
        self.assertTrue(args.output.endswith('pid_gain_voxels_fast_interior_failure_context'))
        estimate=Path(args.estimate_json)
        success=None
        model=Path(load_estimate_json(estimate)['input']['vehicle_model'])
        _, hashes=input_fingerprint(args,estimate,success,model)
        self.assertIn(str(Path(__file__).with_name('pid_voxel_fast_evaluator.py')),hashes)
        self.assertIn(str(Path(__file__).with_name('pid_gain_voxels_interior.py')),hashes)
        # The expanded P/D coordinates are valid even past the old scale.
        self.assertGreater(np.array([25,1,28])[2]/GAIN_CAPS[2,2],1)
        q=np.ones(12)*.5
        q[8]=1.4
        self.assertEqual(FastGainEvaluator._key(q)[8],1.4)


if __name__ == '__main__':
    unittest.main()
