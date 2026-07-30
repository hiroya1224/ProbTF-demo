from pathlib import Path
import tempfile
import unittest

import numpy as np

from grape_param_estim.effective_estimator import (
    EstimatorSettings,
    effective_parameter_trace,
    estimate_effective_parameters,
    evaluate_effective_parameters,
    prepared_timestamps,
    write_result,
)
from grape_param_estim.failure_bag import FailureBagData


def _synthetic_data():
    step = 0.01
    timestamps = np.arange(0.0, 12.0 + 0.5 * step, step)
    wrench = np.column_stack(
        (
            2.0 * np.sin(2.0 * np.pi * 0.43 * timestamps),
            1.5 * np.sin(2.0 * np.pi * 0.57 * timestamps + 0.2),
            20.0 + 3.0 * np.sin(2.0 * np.pi * 0.31 * timestamps),
            0.8 * np.sin(2.0 * np.pi * 0.47 * timestamps + 0.4),
            0.7 * np.sin(2.0 * np.pi * 0.61 * timestamps + 0.1),
            0.4 * np.sin(2.0 * np.pi * 0.73 * timestamps + 0.6),
        )
    )
    delay = 0.04
    delayed_indices = np.searchsorted(
        timestamps, timestamps - delay, side="right"
    ) - 1
    delayed_indices = np.clip(delayed_indices, 0, timestamps.size - 1)
    delayed = wrench[delayed_indices]
    velocity = np.column_stack(
        (
            0.2 * np.sin(2.0 * np.pi * 0.19 * timestamps),
            0.15 * np.cos(2.0 * np.pi * 0.23 * timestamps),
            0.1 * np.sin(2.0 * np.pi * 0.17 * timestamps + 0.3),
        )
    )
    force_gain = np.asarray((1.2, 0.8, 0.5))
    feedback = np.asarray((-0.3, -0.2, -0.4))
    bias = np.asarray((0.1, -0.2, 0.3))
    specific_force = (
        bias
        + delayed[:, :3] * force_gain
        + velocity * feedback
    )
    angular_gain = np.asarray((1.1, 0.9, 0.7))
    angular_acceleration = delayed[:, 3:] * angular_gain
    angular_velocity = np.cumsum(
        angular_acceleration * step, axis=0
    )
    rng = np.random.default_rng(123)
    specific_force += rng.normal(0.0, 0.01, specific_force.shape)
    angular_velocity += rng.normal(
        0.0, 0.0005, angular_velocity.shape
    )
    return FailureBagData(
        bag_path="/synthetic/failure.bag",
        bag_sha256="a" * 64,
        bag_start_time=0.0,
        start_offset_s=1.0,
        end_offset_s=11.0,
        command_times=timestamps,
        command_wrench=wrench,
        imu_times=timestamps,
        specific_force=specific_force,
        angular_velocity=angular_velocity,
        state_times=timestamps,
        linear_velocity=velocity,
    )


def _settings():
    return EstimatorSettings(
        start_offset_s=1.0,
        end_offset_s=11.0,
        sample_rate_hz=100.0,
        smoothing_window_s=0.03,
        maximum_delay_s=0.08,
        delay_step_s=0.01,
        bootstrap_samples=40,
        bootstrap_block_s=0.2,
        huber_delta=1.5,
        ridge=1.0e-8,
        seed=7,
        minimum_input_std=0.001,
        minimum_r2=0.05,
        maximum_relative_interval_width=4.0,
    )


class EffectiveEstimatorTests(unittest.TestCase):
    def test_recovers_delay_and_effective_force_gains(self):
        result = estimate_effective_parameters(
            _synthetic_data(), _settings()
        )
        self.assertAlmostEqual(
            result["selected_alignment_lag_s"], 0.04, delta=0.02
        )
        expected = {
            "specific_force_x_gain": 1.2,
            "specific_force_y_gain": 0.8,
            "specific_force_z_gain": 0.5,
        }
        for name, truth in expected.items():
            self.assertAlmostEqual(
                result["parameters"][name]["estimate"],
                truth,
                delta=0.08,
            )
        self.assertEqual(
            result["channels"]["x"]["information_grade"],
            "informative",
        )

    def test_result_is_deterministic_for_a_fixed_seed(self):
        first = estimate_effective_parameters(
            _synthetic_data(), _settings()
        )
        second = estimate_effective_parameters(
            _synthetic_data(), _settings()
        )
        self.assertEqual(first, second)

    def test_writer_does_not_overwrite_by_default(self):
        result = {"schema": "test", "value": 1}
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "result.json"
            write_result(destination, result)
            with self.assertRaises(FileExistsError):
                write_result(destination, result)
            write_result(destination, result, overwrite=True)

    def test_explicit_fit_mask_drives_estimate_and_parameter_trace(self):
        data = _synthetic_data()
        settings = _settings()
        timestamps = prepared_timestamps(data, settings)
        fit_mask = timestamps <= 9.0

        result = estimate_effective_parameters(
            data, settings, fit_mask=fit_mask
        )
        evaluation = evaluate_effective_parameters(
            data, settings, result
        )
        trace = effective_parameter_trace(
            data,
            settings,
            result,
            fit_mask=fit_mask,
            minimum_duration_s=1.0,
            step_s=1.0,
        )

        self.assertEqual(
            result["fit_sample_count"], int(np.sum(fit_mask))
        )
        self.assertEqual(
            evaluation["timestamps"].shape, timestamps.shape
        )
        self.assertEqual(
            evaluation["residual"].shape, (timestamps.size, 6)
        )
        self.assertGreater(len(trace), 2)
        self.assertAlmostEqual(
            trace[-1]["parameters"]["specific_force_x_gain"],
            1.2,
            delta=0.08,
        )

    def test_point_fit_can_skip_bootstrap_during_mask_search(self):
        result = estimate_effective_parameters(
            _synthetic_data(), _settings(), bootstrap=False
        )

        self.assertEqual(result["bootstrap"]["samples"], 0)
        self.assertEqual(
            result["bootstrap"]["method"], "disabled_point_fit"
        )
        self.assertAlmostEqual(
            result["parameters"]["specific_force_x_gain"][
                "estimate"
            ],
            1.2,
            delta=0.08,
        )


if __name__ == "__main__":
    unittest.main()
