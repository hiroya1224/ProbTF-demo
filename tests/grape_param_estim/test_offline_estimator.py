import importlib.util
from pathlib import Path
import unittest

import numpy as np


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


if __name__ == "__main__":
    unittest.main()
