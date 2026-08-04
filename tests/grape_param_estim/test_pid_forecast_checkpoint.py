from pathlib import Path
import tempfile
import unittest

import numpy as np

from grape_param_estim.artifact_io import ArtifactValidationError, load_npz_strict
from grape_param_estim.pid.checkpoint import (
    PidForecastCheckpointIdentity,
    PidForecastCheckpointStore,
)
from grape_param_estim.pid.metrics import ForecastMetricRecord, ForecastMetrics


def _record(index):
    return ForecastMetricRecord(
        candidate_id="candidate-{}".format(index),
        sample_id="sample-a",
        bag_id="bag-a",
        replicate_index=0,
        discrepancy_seed=2 ** 63 + index,
        metrics=ForecastMetrics(
            position_rmse=1.0 + index,
            orientation_rmse=2.0 + index,
            maximum_position_error=3.0 + index,
            maximum_orientation_error=4.0 + index,
            forecast_completion=1.0,
            numerical_failure_count=0,
            actuator_saturation_duration=0.1 * index,
            actuator_saturation_rate=0.01 * index,
        ),
    )


class PidForecastCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.identity = PidForecastCheckpointIdentity(
            evaluation_id="evaluation-a",
            estimation_run_id="run-a",
            request_fingerprint="sha256:" + "a" * 64,
            estimation_request_fingerprint="sha256:" + "b" * 64,
        )

    def test_atomic_content_addressed_pickle_free_round_trip(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "checkpoint"
            store = PidForecastCheckpointStore.open(
                root, self.identity, resume=False, flush_size=2
            )
            store.record_completed(_record(0))
            self.assertEqual(store.resumed_record_count, 0)
            store.record_completed(_record(1))
            manifest = __import__("json").loads(
                (root / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["record_count"], 2)
            descriptor = manifest["record_batch"]
            self.assertEqual(
                descriptor["path"],
                "objects/{}.npz".format(descriptor["content_sha256"][7:]),
            )
            arrays = load_npz_strict(root / descriptor["path"])
            self.assertTrue(all(not value.dtype.hasobject for value in arrays.values()))

            resumed = PidForecastCheckpointStore.open(
                root, self.identity, resume=True
            )
            self.assertEqual(resumed.resumed_record_count, 2)
            self.assertEqual(resumed.records, (_record(0), _record(1)))

    def test_resume_rejects_request_and_estimation_fingerprint_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "checkpoint"
            store = PidForecastCheckpointStore.open(
                root, self.identity, resume=False
            )
            store.record_completed(_record(0))
            store.flush()
            for field in (
                "request_fingerprint",
                "estimation_request_fingerprint",
            ):
                values = dict(self.identity.__dict__)
                values[field] = "sha256:" + "c" * 64
                with self.subTest(field=field):
                    with self.assertRaisesRegex(
                        ArtifactValidationError, "{} mismatch".format(field)
                    ):
                        PidForecastCheckpointStore.open(
                            root,
                            PidForecastCheckpointIdentity(**values),
                            resume=True,
                        )

    def test_tampered_content_object_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "checkpoint"
            store = PidForecastCheckpointStore.open(
                root, self.identity, resume=False
            )
            store.record_completed(_record(0))
            store.flush()
            manifest = __import__("json").loads(
                (root / "manifest.json").read_text(encoding="utf-8")
            )
            with (root / manifest["record_batch"]["path"]).open("ab") as stream:
                stream.write(b"tampered")
            with self.assertRaisesRegex(ArtifactValidationError, "digest"):
                PidForecastCheckpointStore.open(
                    root, self.identity, resume=True
                )


if __name__ == "__main__":
    unittest.main()
