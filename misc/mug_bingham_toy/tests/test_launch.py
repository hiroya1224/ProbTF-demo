"""Fresh-launch guards for the corrected cylindrical-center experiment."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch


ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('toy_mug_status',ROOT/'status.py')
status=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(status)


class FreshLaunchTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.folder=Path(self.temporary.name)
        self.data=self.folder/'data'
        self.run=self.folder/'run'
        self.assets=self.folder/'assets'
        self.assets.mkdir()
        (self.assets/'mug_original.stl').write_bytes(b'retained original STL fixture')
        (self.assets/status.COLORED_MESH).write_bytes(b'corrected cylinder-centered PLY fixture')
        self.mesh={'center_native':[40.,40.,45.],
                   'colored_mesh':status.COLORED_MESH,
                   'colored_mesh_sha256':status.sha256(self.assets/status.COLORED_MESH),
                   'original_stl':'mug_original.stl',
                   'original_sha256':status.sha256(self.assets/'mug_original.stl')}
        self.write_json(self.assets/status.MESH_METADATA,self.mesh)
        self.data.mkdir()
        self.write_json(self.data/'render_metadata.json',{'image_size':[640,360],'mesh':self.mesh,
                                                        'image_blur_sigma_pixels':0.0})
        # Small fixtures exercise the same count/seed guard without creating
        # 12,000 files or invoking rendering/training during this test.
        self.splits=(('train',2,1234),('val',1,1235),('test',1,1236))
        for split,count,seed in self.splits:
            folder=self.data/split
            (folder/'images').mkdir(parents=True)
            (folder/'poses.npz').write_bytes(f'{split} exact label bytes'.encode())
            for index in range(count):
                (folder/'images'/f'{index:06d}.png').touch()
            self.write_json(folder/'metadata.json',{'count':count,'seed':seed,'mesh':self.mesh,
                                                    'poses_sha256':status.sha256(folder/'poses.npz')})
        patcher=patch.object(status,'SPLITS',self.splits)
        patcher.start()
        self.addCleanup(patcher.stop)

    @staticmethod
    def write_json(path,value):
        path.write_text(json.dumps(value))

    def validate(self):
        return status.validate_launch(self.data,self.run,self.assets)

    def test_corrected_fresh_data_passes_and_defaults_select_new_run(self):
        result=self.validate()
        self.assertEqual(result['center_native'],[40.,40.,45.])
        self.assertEqual((result['epochs'],result['batch_size'],result['training_seed']),(50,16,20261126))
        self.assertEqual(status.DEFAULT_DATA,ROOT/'data/cylinder_center_no_blur_20261127')
        self.assertEqual(status.DEFAULT_RUN,ROOT/'runs/vgg16_10000_cylinder_center_no_blur_20261127')

    def test_wrong_bounding_box_center_is_rejected(self):
        wrong=deepcopy(self.mesh)
        wrong['center_native']=[56.056610107421875,40.,45.]
        self.write_json(self.assets/status.MESH_METADATA,wrong)
        with self.assertRaisesRegex(ValueError,'cylindrical center'):
            self.validate()

    def test_blurred_input_is_rejected(self):
        path=self.data/'render_metadata.json'
        metadata=json.loads(path.read_text())
        metadata['image_blur_sigma_pixels']=1.2
        self.write_json(path,metadata)
        with self.assertRaisesRegex(ValueError,'without blur'):
            self.validate()

    def test_old_split_cannot_hide_behind_new_global_metadata(self):
        path=self.data/'train/metadata.json'
        metadata=json.loads(path.read_text())
        metadata['mesh']['center_native']=[56.056610107421875,40.,45.]
        self.write_json(path,metadata)
        with self.assertRaisesRegex(ValueError,'train: mesh/origin'):
            self.validate()

    def test_changed_asset_or_labels_are_rejected(self):
        (self.assets/status.COLORED_MESH).write_bytes(b'other PLY')
        with self.assertRaisesRegex(ValueError,'asset bytes'):
            self.validate()
        (self.assets/status.COLORED_MESH).write_bytes(b'corrected cylinder-centered PLY fixture')
        (self.data/'val/poses.npz').write_bytes(b'changed labels')
        with self.assertRaisesRegex(ValueError,'val: pose-label bytes'):
            self.validate()

    def test_existing_run_and_incomplete_images_are_rejected(self):
        self.run.mkdir()
        (self.run/'last.pt').touch()
        with self.assertRaisesRegex(ValueError,'prior training artifacts'):
            self.validate()
        (self.run/'last.pt').unlink()
        (self.data/'test/images/000000.png').unlink()
        with self.assertRaisesRegex(ValueError,'test: complete image count'):
            self.validate()

    def test_launcher_rejects_resume_and_config_overrides_before_any_work(self):
        for arguments in (['--resume','old.pt'],['--data','old-data'],['--config','old.json']):
            result=subprocess.run(['bash',str(ROOT/'run.sh'),*arguments],capture_output=True,text=True)
            self.assertEqual(result.returncode,2)
            self.assertIn('from scratch',result.stderr)


if __name__=='__main__':
    unittest.main()
