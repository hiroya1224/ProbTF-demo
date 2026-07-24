import importlib.util
from pathlib import Path
import unittest

import numpy as np
import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = (
    REPOSITORY_ROOT
    / "ros/examples/grape-param-estim/scripts/estimate_grape_bag.py"
)
SPEC = importlib.util.spec_from_file_location(
    "estimate_grape_bag_for_test",
    SCRIPT_PATH,
)
ESTIMATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ESTIMATOR)


class OfflineEstimatorFlightStateTests(unittest.TestCase):
    def setUp(self):
        self.grid = np.array([0.0, 1.0, 2.0, 3.0])
        self.data = {
            "flight_event": [0.0, 1.0, 2.0, 3.0],
            "flight_state": [3, 3, 5, 4],
            "pose_event": [0.0, 1.0, 2.0, 3.0],
            "position": [
                np.array([0.0, 0.0, 0.18]),
                np.array([0.0, 0.0, 0.24]),
                np.array([0.0, 0.0, 0.19]),
                np.array([0.0, 0.0, 0.18]),
            ],
            "ground_reference_z": 0.18,
        }
        self.config = {
            "real_bag": {
                "allowed_flight_states": [3, 5],
                "takeoff_state": 3,
                "minimum_takeoff_clearance_m": 0.05,
            }
        }

    def test_takeoff_requires_clearance_but_hover_does_not(self):
        np.testing.assert_array_equal(
            ESTIMATOR._flight_state_valid_mask(
                self.data,
                self.grid,
                self.config,
            ),
            [False, True, True, False],
        )

    def test_takeoff_fails_closed_without_ground_reference(self):
        self.data["ground_reference_z"] = None
        np.testing.assert_array_equal(
            ESTIMATOR._flight_state_valid_mask(
                self.data,
                self.grid,
                self.config,
            ),
            [False, False, True, False],
        )

    def test_timestamp_before_first_state_is_not_assigned_future_state(self):
        np.testing.assert_array_equal(
            ESTIMATOR._flight_state_valid_mask(
                self.data,
                np.array([-1.0, 0.0]),
                self.config,
            ),
            [False, False],
        )


class OfflineEstimatorBatchingTests(unittest.TestCase):
    def test_final_partial_batch_is_retained(self):
        batches = ESTIMATOR._observation_batches(np.arange(39), 5)

        self.assertEqual(len(batches), 8)
        np.testing.assert_array_equal(batches[-1], [35, 36, 37, 38])
        np.testing.assert_array_equal(np.concatenate(batches), np.arange(39))

    def test_unit_batches_produce_one_update_per_selected_observation(self):
        batches = ESTIMATOR._observation_batches(np.arange(39), 1)

        self.assertEqual(len(batches), 39)
        self.assertTrue(all(batch.size == 1 for batch in batches))

    def test_non_positive_batch_size_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "batch_size must be positive"):
            ESTIMATOR._observation_batches(np.arange(3), 0)

    def test_default_config_updates_and_outputs_every_evidence_sample(self):
        config_path = (
            REPOSITORY_ROOT
            / "ros/examples/grape-param-estim/config/estimator.yaml"
        )
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

        self.assertEqual(config["particle_filter"]["batch_size"], 1)
        self.assertEqual(
            config["particle_filter"]["output_every_observations"],
            1,
        )
        self.assertEqual(config["synchronization"]["estimation_stride"], 5)


if __name__ == "__main__":
    unittest.main()
