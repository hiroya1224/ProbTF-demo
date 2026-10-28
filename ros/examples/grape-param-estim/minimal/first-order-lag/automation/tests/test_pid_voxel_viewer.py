"""Viewer must display the actual all-sample fraction in the failure context."""
import base64
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import build_pid_voxel_viewer as viewer
from pid_voxel_survival import sample_stability


class ViewerChecks(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='pid-all-sample-viewer-')
        self.source = Path(self.temporary.name)
        self.root = self.source/'failure1'/viewer.DATASET
        self.grid = self.root/'roll_pitch/grid_3'
        self.grid.mkdir(parents=True)
        self.recorded = np.ones((4, 3))
        self.radii = np.full((3, 3, 3, 4), .9)
        self.radii[..., 1] = 1.1
        self.radii[1, 1, 1] = [.9, .9, 1.2, np.nan]
        done = np.ones((3, 3, 3), dtype=bool)
        checked = sample_stability(self.radii, done)
        np.save(self.grid/'spectral_radius.npy', self.radii)
        np.save(self.grid/'evaluated.npy', done)
        np.savez(self.grid/'stability_fraction.npz',
                 p=[0., 1., 2.], i=[0., 1., 2.], d=[0., 1., 2.], sample_count=4,
                 stable_fraction=checked.stable_fraction, stable_count=checked.stable_count,
                 valid_count=checked.valid_count, unresolved_count=checked.unresolved_count,
                 classification=checked.alpha_classification)
        self.write('status.json', {'status': 'complete', 'updated_utc': 'test'})
        self.write('manifest.json', {'arguments': {'grid_sizes': [3]}, 'plant_samples': 4,
                                     'fingerprint': 'fixture'})
        self.write('references.json', {
            'recorded_case': {'gains': self.recorded.tolist(), 'spectral_radius': [.9, .9, 1.2, None]},
            'success_full_controller': {'gains': (2*self.recorded).tolist(),
                                        'spectral_radius': [.9]*4}})
        self.write('roll_pitch/grid_3/status.json', {'complete': True, 'pending': 0})
        self.write('roll_pitch/grid_3/metadata.json', {
            'full_fixed_controller_gains': self.recorded.tolist()})

    def write(self, relative, value):
        (self.root/relative).write_text(json.dumps(value))

    def tearDown(self):
        self.temporary.cleanup()

    def test_uses_all_samples_and_never_makes_success_a_section_anchor(self):
        payload, report = viewer.load_payload(self.source, cases=['failure1'], groups=['roll_pitch'])
        case = payload['cases']['failure1']
        grid = case['groups']['roll_pitch']['grids']['3']
        self.assertEqual(payload['sizes'], [3])
        self.assertEqual(case['sample_count'], 4)
        self.assertEqual(set(grid['marks']), {'failure'})
        self.assertEqual(grid['marks']['failure']['stable_fraction'], .5)
        self.assertEqual(grid['marks']['failure']['unresolved'], 1)
        counts = np.frombuffer(base64.b64decode(grid['nonstable']), dtype='<u4').reshape(3, 3, 3)
        self.assertEqual(int(counts[0, 0, 0]), 1)
        self.assertEqual(int(counts[1, 1, 1]), 2)
        self.assertEqual(case['full_controller_references']['success_full_controller']['stable_fraction'], 1.)
        self.assertIn('all original plant samples', report['criterion'])

    def test_refuses_to_relabel_a_success_fixed_volume_as_failure_fixed(self):
        self.write('roll_pitch/grid_3/metadata.json', {
            'full_fixed_controller_gains': (2*self.recorded).tolist()})
        with self.assertRaisesRegex(ValueError, 'not a recorded-failure controller section'):
            viewer.load_payload(self.source, cases=['failure1'], groups=['roll_pitch'])

    def test_count_encoding_does_not_drop_samples_above_255(self):
        values = np.array([0, 128, 256, 1000], dtype=np.int32)
        actual = np.frombuffer(base64.b64decode(viewer.encode_bytes(values)), dtype='<u4')
        np.testing.assert_array_equal(actual, values)

    def test_success_reference_cannot_change_the_displayed_section(self):
        before, _ = viewer.load_payload(self.source, cases=['failure1'], groups=['roll_pitch'])
        references = json.loads((self.root/'references.json').read_text())
        references['success_full_controller']['spectral_radius'] = [1.1, None, 1.3, 1.4]
        self.write('references.json', references)
        after, _ = viewer.load_payload(self.source, cases=['failure1'], groups=['roll_pitch'])
        self.assertEqual(before['cases']['failure1']['groups'], after['cases']['failure1']['groups'])

    def test_final_archive_is_sufficient_without_intermediate_checkpoint_arrays(self):
        np.savez(self.grid/'volume.npz', spectral_radius=self.radii,
                 evaluated=np.ones((3, 3, 3), dtype=bool))
        (self.grid/'spectral_radius.npy').unlink()
        (self.grid/'evaluated.npy').unlink()
        payload, _ = viewer.load_payload(self.source, cases=['failure1'], groups=['roll_pitch'])
        mark = payload['cases']['failure1']['groups']['roll_pitch']['grids']['3']['marks']['failure']
        self.assertEqual(mark['stable_count'], 2)
        self.assertEqual(mark['stable_fraction'], .5)

    def test_html_contains_only_the_actually_computed_sizes(self):
        from unittest.mock import patch
        output = self.source/'viewer'
        arguments = ['viewer', '--source', str(self.source), '--output', str(output),
                     '--cases', 'failure1', '--groups', 'roll_pitch', '--sizes', '3']
        with patch.object(sys, 'argv', arguments):
            viewer.main()
        html = (output/'index.html').read_text()
        self.assertIn('id="plot3"', html)
        self.assertNotIn('id="plot33"', html)
        self.assertNotIn('id="plot65"', html)
        self.assertNotIn('__VOLUME_CARDS__', html)
        self.assertNotIn('__SLICE_CARDS__', html)


if __name__ == '__main__':
    unittest.main()
