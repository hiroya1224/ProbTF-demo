import tempfile
import unittest
from pathlib import Path

import numpy as np

from grape_param_estim.data import (
    BagRecording,
    load_yaml,
    save_yaml,
    scan_bag_paths,
)


def make_recording() -> BagRecording:
    times = np.linspace(0.0, 2.0, 21)
    quaternion = np.zeros((times.size, 4))
    quaternion[:, 3] = 1.0
    return BagRecording(
        bag_path="/tmp/synthetic.bag",
        bag_start_time=100.0,
        bag_duration=2.0,
        command_times=times,
        base_thrust=np.column_stack(
            [times + float(rotor) for rotor in range(4)]
        ),
        gimbal_times=times,
        gimbal_angle=np.column_stack(
            [0.1 * times + float(rotor) for rotor in range(4)]
        ),
        imu_times=times,
        specific_force=np.column_stack((times, 2.0 * times, 3.0 * times)),
        angular_velocity=np.column_stack(
            (0.1 * times, 0.2 * times, 0.3 * times)
        ),
        state_times=times,
        position=np.column_stack((times, times**2, -times)),
        orientation_xyzw=quaternion,
        linear_velocity=np.column_stack(
            (np.ones_like(times), 2.0 * times, -np.ones_like(times))
        ),
        flight_state_times=np.asarray((0.0, 0.8, 1.4)),
        flight_state=np.asarray((0, 3, 5)),
    )


class BagRecordingTest(unittest.TestCase):
    def test_select_interval_aligns_streams_and_assigns_segments(self):
        recording = make_recording()

        selected = recording.select_interval(0.2, 1.8, 0.5)

        np.testing.assert_allclose(selected.times[[0, -1]], (0.2, 1.8))
        np.testing.assert_allclose(
            selected.base_thrust[:, 0], selected.times
        )
        np.testing.assert_allclose(
            selected.angular_velocity[:, 2], 0.3 * selected.times
        )
        self.assertEqual(selected.segment_count, 4)
        self.assertEqual(
            sum(
                segment.stop - segment.start
                for _, segment in selected.segments()
            ),
            selected.times.size,
        )
        np.testing.assert_array_equal(
            selected.flight_state[selected.times < 0.8], 0
        )
        np.testing.assert_array_equal(
            selected.flight_state[selected.times >= 1.4], 5
        )

    def test_analysis_bounds_are_common_stream_overlap(self):
        recording = make_recording()

        self.assertEqual(recording.analysis_bounds, (0.0, 2.0))
        with self.assertRaisesRegex(ValueError, "analysis interval"):
            recording.select_interval(-0.1, 1.0, 0.5)


class ConfigurationTest(unittest.TestCase):
    def test_yaml_round_trip_and_recursive_bag_scan(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            nested = root / "trial"
            nested.mkdir()
            first = root / "a.bag"
            second = nested / "b.bag"
            ignored = nested / "notes.txt"
            first.touch()
            second.touch()
            ignored.touch()

            paths = scan_bag_paths(str(root))
            self.assertEqual(paths, (str(first.resolve()), str(second.resolve())))

            config_path = root / "config" / "analysis.yaml"
            written = save_yaml(
                str(config_path),
                {
                    "schema": "grape_param_estim/phase0",
                    "analysis": {"segment_duration": 0.75},
                },
            )
            self.assertEqual(written, str(config_path.resolve()))
            self.assertEqual(
                load_yaml(written)["analysis"]["segment_duration"], 0.75
            )


if __name__ == "__main__":
    unittest.main()
