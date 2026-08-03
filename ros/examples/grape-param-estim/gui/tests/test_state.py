from pathlib import Path
import tempfile
import unittest

try:
    from PySide6.QtCore import QCoreApplication
except ImportError as error:  # exactly the supported optional-test guard
    raise unittest.SkipTest("PySide6 is unavailable: {}".format(error))

from grape_param_estim_gui.project_io import (
    freshness_fingerprint,
    new_project_manifest,
)
from grape_param_estim_gui.state import BagRecord, ProjectStore


class ProjectStateQtTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.application = QCoreApplication.instance() or QCoreApplication([])

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        bag = self.root / "bags" / "bag-a.bag"
        bag.parent.mkdir()
        bag.write_bytes(b"bag")
        self.manifest = new_project_manifest("state-test")
        self.manifest["bags"] = [
            {
                "bag_id": "bag-a",
                "source_path": "/provenance/bag-a.bag",
                "relative_path": "bags/bag-a.bag",
                "sha256": "a" * 64,
            }
        ]
        self.store = ProjectStore(self.root, self.manifest)
        self.record = BagRecord(
            bag_id="bag-a", path=bag, source_path=Path("/provenance/bag-a.bag"),
            sha256="a" * 64, included=True, auto_interval=(1.0, 5.0),
            selected_interval=(1.0, 5.0), configuration_fingerprint="complete:abc",
            controller_snapshot={"gains": [1.0]},
        )
        self.store.add(self.record)
        self.store.manifest["run_request_fingerprint"] = freshness_fingerprint(self.store.manifest)
        self.store._refresh_stale()

    def tearDown(self):
        self.temporary.cleanup()

    def test_selection_interval_fingerprint_and_settings_mark_results_stale(self):
        self.assertFalse(self.store.results_stale)
        self.store.update_interval("bag-a", (1.2, 4.8), "MODIFIED")
        self.assertTrue(self.store.results_stale)

        self.store.manifest["run_request_fingerprint"] = freshness_fingerprint(self.store.manifest)
        self.store._refresh_stale()
        self.assertFalse(self.store.results_stale)
        self.store.set_included("bag-a", False)
        self.assertTrue(self.store.results_stale)

        self.store.set_included("bag-a", True)
        self.store.manifest["run_request_fingerprint"] = freshness_fingerprint(self.store.manifest)
        self.store._refresh_stale()
        self.record.configuration_fingerprint = "complete:def"
        self.store._sync_manifest_inputs()
        self.assertTrue(self.store.results_stale)

        self.store.manifest["run_request_fingerprint"] = freshness_fingerprint(self.store.manifest)
        self.store._refresh_stale()
        self.store.set_estimator_settings({"ensemble_size": 64})
        self.assertTrue(self.store.results_stale)

    def test_interval_state_transitions_are_explicit(self):
        self.store.update_interval("bag-a", (1.1, 4.9), "MODIFIED")
        self.assertEqual(self.record.interval_state, "MODIFIED")
        self.store.update_interval("bag-a", self.record.selected_range, "LOCKED")
        self.assertEqual(self.record.interval_state, "LOCKED")
        self.store.restore_auto_interval("bag-a")
        self.assertEqual(self.record.interval_state, "AUTO")
        self.assertEqual(self.record.selected_range, (1.0, 5.0))


if __name__ == "__main__":
    unittest.main()
