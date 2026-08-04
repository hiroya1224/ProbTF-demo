from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from grape_param_estim.batch_artifact import BatchEstimationRun, file_sha256
from grape_param_estim.controller import ControllerConfig
from grape_param_estim.controller_config import PidGainConfiguration
from grape_param_estim.pid.cli import execute_pid_evaluation
from grape_param_estim.pid.metrics import ForecastMetrics
from grape_param_estim.pid.proposal import PhysicalPlantPosterior
from grape_param_estim.pid.request import (
    PID_EVALUATION_REQUEST_SCHEMA,
    validate_pid_evaluation_request,
)
from grape_param_estim.progress import CancellationToken, ProgressCancelled
from grape_param_estim.system import VehicleParameters


def _request(root, bag_path, bag_sha):
    run_path = root / "run"
    run_path.mkdir()
    payload = {
        "schema": PID_EVALUATION_REQUEST_SCHEMA,
        "evaluation_id": "pid-evaluation",
        "estimation_run": str(run_path),
        "output_directory": str(root / "pid-output"),
        "resume": False,
        "baseline_bag_id": "bag-a",
        "selected_mode_id": "mode-map",
        "bags": [
            {
                "bag_id": "bag-a",
                "path": str(bag_path),
                "sha256": bag_sha,
                "roll_pitch_integration_active": True,
            }
        ],
        "fixed_plant_parameters": {
            "linear_drag": [0.0, 0.0, 0.0],
            "angular_drag": [0.0, 0.0, 0.0],
        },
        "model_discrepancy": {
            "policy": "zero_model_discrepancy",
            "base_seed": 42,
            "replicates": 1,
        },
        "plant_sample_subset": {
            "method": "all_equal_weight_mcmc_samples",
            "sample_ids": None,
        },
        "candidates": [
            {
                "candidate_id": "current",
                "source": "current",
                "source_sample_id": None,
                "gain_values": None,
            },
            {
                "candidate_id": "user-a",
                "source": "user",
                "source_sample_id": None,
                "gain_values": [[1.0, 0.1, 2.0]] * 4,
            },
        ],
        "quantile_level": 0.95,
        "cvar_level": 0.9,
        "selected_candidate_id": None,
        "maximum_reference_age_seconds": 0.05,
    }
    return validate_pid_evaluation_request(payload)


class PidCliTests(unittest.TestCase):
    def setUp(self):
        self.nominal = VehicleParameters.nominal()
        self.posterior = PhysicalPlantPosterior.from_aligned_values(
            ("chain-a:1",),
            (self.nominal,),
            (0.012,),
            ("mode-map",),
        )
        self.current = PidGainConfiguration(
            np.asarray(
                (
                    (4.0, 0.1, 2.0),
                    (5.0, 1.0, 2.5),
                    (13.0, 1.0, 20.0),
                    (6.0, 1.0, 2.0),
                )
            )
        )

    def _run(self, request, bag_sha, progress):
        run = BatchEstimationRun(
            root=request.estimation_run,
            manifest={
                "selected_bag_ids": ["bag-a"],
                "selected_bag_sha256": {"bag-a": bag_sha},
                "selected_intervals": {"bag-a": [18.0, 24.0]},
                "controller_snapshot_fingerprint": "sha256:" + "c" * 64,
                "q_definition": {
                    "definition": "body_wrench/continuous_spectral_density"
                },
                "actuator_model": {
                    "source": "calibrated_test",
                    "thrust_time_constant_seconds": 0.03,
                    "gimbal_time_constant_seconds": 0.02,
                    "minimum_thrust_newtons": 1.5,
                    "maximum_thrust_newtons": 27.6145,
                    "maximum_gimbal_angle_radians": 3.14,
                    "maximum_gimbal_rate_radians_per_second": 6.0,
                },
                "run_id": "batch-run",
                "request_fingerprint": "sha256:" + "d" * 64,
            },
            map_static={"q_diagonal": np.arange(1.0, 7.0)},
            q_em={},
            laplace={},
            diagnostics={},
            bags={},
            mcmc_samples={},
            trajectories={},
        )

        def evaluator(_candidate, _sample, _bag_id, _realization):
            return ForecastMetrics(
                position_rmse=1.0,
                orientation_rmse=1.0,
                maximum_position_error=1.0,
                maximum_orientation_error=1.0,
                forecast_completion=1.0,
                numerical_failure_count=0,
                actuator_saturation_duration=0.0,
                actuator_saturation_rate=0.0,
            )

        written = SimpleNamespace(root=request.output_directory)
        flight = SimpleNamespace(
            controller_snapshot=object(),
            controller_configuration=ControllerConfig.grape(),
        )
        with ExitStack() as stack:
            stack.enter_context(patch(
                "grape_param_estim.pid.cli.load_batch_estimation_run",
                return_value=run,
            ))
            stack.enter_context(patch(
                "grape_param_estim.pid.cli.physical_posterior_from_batch_run",
                return_value=self.posterior,
            ))
            stack.enter_context(patch(
                "grape_param_estim.pid.cli.load_flight_data", return_value=flight
            ))
            stack.enter_context(patch(
                "grape_param_estim.pid.cli.controller_snapshot_fingerprint",
                return_value="sha256:" + "c" * 64,
            ))
            stack.enter_context(patch(
                "grape_param_estim.pid.cli.configuration_from_controller_snapshot",
                return_value=self.current,
            ))
            stack.enter_context(patch(
                "grape_param_estim.pid.cli.forecast_scenarios_from_batch_run",
                return_value=(object(),),
            ))
            stack.enter_context(patch(
                "grape_param_estim.pid.cli.ClosedLoopPidForecastEvaluator",
                return_value=evaluator,
            ))
            writer = stack.enter_context(patch(
                "grape_param_estim.pid.cli.write_pid_proposal_evaluation",
                return_value=written,
            ))
            result = execute_pid_evaluation(
                request, progress_callback=progress.append
            )
        return result, writer

    def test_one_command_uses_artifact_q_actuator_model_and_progress(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bag = root / "flight.bag"
            bag.write_bytes(b"strict test bag")
            bag_sha = file_sha256(bag)
            request = _request(root, bag, bag_sha)
            progress = []
            result, writer = self._run(request, bag_sha, progress)
            self.assertEqual(result.root, request.output_directory)
            self.assertEqual(progress[-1].fraction, 1.0)
            self.assertEqual(
                {event.stage_id for event in progress},
                {
                    "preparing_trajectory",
                    "optimizing_full_trajectory",
                    "writing_artifacts",
                },
            )
            evaluation = writer.call_args.kwargs["evaluation"]
            self.assertEqual(len(evaluation.records), 2)
            self.assertEqual(
                evaluation.discrepancy.interval_model,
                "continuous_spectral_density",
            )
            np.testing.assert_array_equal(
                evaluation.discrepancy.diagonal_q, np.arange(1.0, 7.0)
            )
            identity = writer.call_args.kwargs["identity"]
            self.assertEqual(identity.request_fingerprint, request.fingerprint)

    def test_pre_cancel_stops_before_any_forecast(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bag = root / "flight.bag"
            bag.write_bytes(b"strict test bag")
            request = _request(root, bag, file_sha256(bag))
            token = CancellationToken()
            token.cancel("test")
            with self.assertRaises(ProgressCancelled):
                execute_pid_evaluation(request, cancellation_token=token)


if __name__ == "__main__":
    unittest.main()
