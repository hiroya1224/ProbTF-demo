"""Validate real recorded gains, strict interiors, physical scales and resume identities."""
import json
import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path
import numpy as np
from pid_voxel_domains import GROUPS, axes_for_box, load_configuration, validate_box
from pid_gain_voxels_interior import (HERE, GAIN_CAPS, ExpandedGainEvaluator,
                                    coordinates_for_nodes, input_fingerprint,
                                    load_estimate_json, _recorded_gain_matrix, parser,
                                    Volume, plot_volume)

class DomainChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = Path(__file__).with_name('pid_voxel_domains.json')
        cls.bounds, cls.anchors, cls.report, cls.paths, cls.clearance = load_configuration(
            cls.config, HERE, load_estimate_json, _recorded_gain_matrix,
            [HERE/'outputs/failure1/estimate.json'])

    def test_every_recorded_gain_has_real_clearance_at_all_refinement_levels(self):
        for group in GROUPS:
            self.assertGreaterEqual(self.report[group]['minimum_relative_clearance'], 1/6-1e-15)
            previous = None
            for size in (3,5,9,17,33):
                axes = axes_for_box(size,self.bounds[group],self.anchors[group],self.clearance)
                for k,axis in enumerate(axes):
                    self.assertEqual(axis[0],self.bounds[group][0,k])
                    self.assertEqual(axis[-1],self.bounds[group][1,k])
                    for anchor in self.anchors[group][:,k]:
                        self.assertIn(anchor,axis)
                        self.assertGreater(anchor,axis[0])
                        self.assertLess(anchor,axis[-1])
                    if previous is not None:
                        self.assertTrue(np.all(np.isin(previous[k],axis)))
                previous=axes

    def test_boundary_anchors_are_rejected_even_when_clearance_is_zero(self):
        bounds=np.array([[1,1,1],[5,5,5]],float)
        for candidate in ([1,3,3],[5,3,3],[3,1,3],[3,3,5]):
            with self.assertRaisesRegex(ValueError,'strictly inside'):
                axes_for_box(3,bounds,[candidate],0)

    def test_tiny_positive_interior_gap_is_not_enough(self):
        with self.assertRaisesRegex(ValueError,'too close'):
            axes_for_box(3,[[0,0,0],[10,10,10]],[[1e-10,5,5]])

    def test_invalid_boxes_and_anchors_are_rejected(self):
        for bounds in ([[1,1,1],[1,5,5]], [[-1,1,1],[5,5,5]], [[1,1,1],[5,np.nan,5]]):
            with self.assertRaises(ValueError):
                validate_box(bounds,[[3,3,3]])
        with self.assertRaises(ValueError):
            validate_box([[1,1,1],[5,5,5]],[[np.inf,3,3]])

    def test_larger_gain_is_not_clipped_to_old_term_clamp_scale(self):
        base=_recorded_gain_matrix(load_estimate_json(HERE/'outputs/failure1/estimate.json'))
        group=2
        axes=axes_for_box(33,self.bounds['roll_pitch'],self.anchors['roll_pitch'],self.clearance)
        index=np.array([[len(axes[0])-1,len(axes[1])-1,len(axes[2])-1]])
        q=coordinates_for_nodes(index,axes,base,group)
        self.assertGreater(q.max(),1)
        self.assertEqual(ExpandedGainEvaluator._key(q[0]),tuple(q[0]))
        recovered=q.reshape(4,3)*GAIN_CAPS
        np.testing.assert_array_equal(recovered[group],self.bounds['roll_pitch'][1])
        np.testing.assert_array_equal(recovered[[0,1,3]],base[[0,1,3]])

    def test_domain_change_changes_the_checkpoint_fingerprint(self):
        with tempfile.TemporaryDirectory() as directory:
            config=Path(directory)/'domain.json'
            value=json.loads(self.config.read_text())
            config.write_text(json.dumps(value))
            args=parser().parse_args(['--domain-config',str(config)])
            estimate=Path(args.estimate_json);success=None
            model=Path(load_estimate_json(estimate)['input']['vehicle_model'])
            first,hashes=input_fingerprint(args,estimate,success,model)
            self.assertIn(str(config),hashes)
            value['upper_factor']=3
            config.write_text(json.dumps(value))
            second,_=input_fingerprint(args,estimate,success,model)
            self.assertNotEqual(first,second)

    def test_new_default_output_does_not_reinterpret_the_old_checkpoint(self):
        args=parser().parse_args([])
        self.assertTrue(args.output.endswith('pid_gain_voxels_interior_failure_context'))
        self.assertEqual(args.domain_config,str(self.config))

    def test_plot_retains_every_sample_and_uses_only_recorded_failure_marker(self):
        base=_recorded_gain_matrix(load_estimate_json(HERE/'outputs/failure1/estimate.json'))
        refs={'recorded_case':{'gains':base.tolist()}}
        with tempfile.TemporaryDirectory() as directory:
            axes=[np.array([5,20,40]),np.array([.25,1,2]),np.array([2,8,16])]
            volume=Volume(Path(directory),axes,128,'all-fixed-samples','roll_pitch',base)
            values=np.full(128,.9);values[26]=1.0000195565456116
            row=SimpleNamespace(delta=1-values.max(),worst_spectral_radius=values.max(),
                                worst_sample_index=26,spectral_radius=values,
                                pole_valid_mask=np.isfinite(values),near_kink_mask=np.zeros(128,bool))
            volume.store(np.array([[1,1,1]]),[row])
            plot_volume(volume,'roll_pitch','failure1',refs)
            with np.load(Path(directory)/'stability_fraction.npz') as saved:
                self.assertEqual(saved['sample_count'],128)
                self.assertEqual(saved['stable_count'][1,1,1],127)
                self.assertEqual(saved['stable_fraction'][1,1,1],127/128)
                self.assertNotIn('baseline_stable_mask',saved.files)
            html=(Path(directory)/'pid_3d.html').read_text()
            self.assertNotIn('Successful flight (',html)
            self.assertIn('recorded failure',html)

if __name__=='__main__':
    unittest.main()
