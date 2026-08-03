import copy
import hashlib
import io
import json
import os
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

import numpy as np

from grape_param_estim.artifact_io import request_fingerprint
from grape_param_estim.diagonal_q_stage_cli import (
    DIAGONAL_Q_STAGE_REQUEST_SCHEMA,
    _q_e_step_units,
    resolved_forecast_workers,
    run_request,
    validate_diagonal_q_stage_request,
)
from grape_param_estim.progress import ProgressCancelled, ProgressEvent


class DiagonalQStageCliTest(unittest.TestCase):
    def _request(self, bag: Path):
        digest = hashlib.sha256(bag.read_bytes()).hexdigest()
        return {
            "schema": DIAGONAL_Q_STAGE_REQUEST_SCHEMA,
            "run_id": "q-attempt-1",
            "project_fingerprint": "sha256:" + "1" * 64,
            "stage_id": "diagonal_q",
            "stage_input_fingerprint": "sha256:" + "2" * 64,
            "bags": [
                {
                    "bag_id": "bag-a",
                    "path": str(bag),
                    "sha256": digest,
                    "episode_index": 0,
                    "selected_interval_local_seconds": [1.0, 1.2],
                    "configuration_fingerprint": (
                        "manual-group:sha256:" + "3" * 64
                    ),
                }
            ],
            "settings": {
                "sample_period": 0.1,
                "ensemble_size": 39,
                "maximum_em_iterations": 1,
                "log_q_tolerance": 0.01,
                "component_floor": [
                    1.0e-6,
                    2.0e-6,
                    3.0e-6,
                    4.0e-8,
                    5.0e-8,
                    6.0e-8,
                ],
                "fixed_initial_delay_seconds": 0.02,
                "seed": 23,
                "forecast_workers": "auto",
            },
        }

    def test_request_schema_is_exact_and_requires_six_q_floors(self):
        with tempfile.TemporaryDirectory() as directory:
            bag = Path(directory) / "flight.bag"
            bag.write_bytes(b"authenticated fake bag")
            request = self._request(bag)
            self.assertIs(validate_diagonal_q_stage_request(request), request)

            invalid_values = []
            extra = copy.deepcopy(request)
            extra["unexpected"] = True
            invalid_values.append(extra)
            bad_stage = copy.deepcopy(request)
            bad_stage["stage_id"] = "static_parameters"
            invalid_values.append(bad_stage)
            short_floor = copy.deepcopy(request)
            short_floor["settings"]["component_floor"] = [1.0] * 5
            invalid_values.append(short_floor)
            scalar_floor = copy.deepcopy(request)
            scalar_floor["settings"]["component_floor"] = [1.0] * 5 + [0.0]
            invalid_values.append(scalar_floor)
            too_small = copy.deepcopy(request)
            too_small["settings"]["ensemble_size"] = 38
            invalid_values.append(too_small)
            unsafe_configuration = copy.deepcopy(request)
            unsafe_configuration["bags"][0]["configuration_fingerprint"] = (
                "vehicle-a"
            )
            invalid_values.append(unsafe_configuration)
            for invalid in invalid_values:
                with self.subTest(invalid=invalid):
                    with self.assertRaises(ValueError):
                        validate_diagonal_q_stage_request(invalid)

    def test_selected_bags_must_be_sorted_unique_and_same_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.bag"
            second = root / "second.bag"
            first.write_bytes(b"first")
            second.write_bytes(b"second")
            request = self._request(first)
            other = copy.deepcopy(request["bags"][0])
            other.update(
                {
                    "bag_id": "bag-b",
                    "path": str(second),
                    "sha256": hashlib.sha256(second.read_bytes()).hexdigest(),
                }
            )
            request["bags"].append(other)
            validate_diagonal_q_stage_request(request)

            reversed_request = copy.deepcopy(request)
            reversed_request["bags"].reverse()
            with self.assertRaisesRegex(ValueError, "sorted, unique"):
                validate_diagonal_q_stage_request(reversed_request)
            mismatch = copy.deepcopy(request)
            mismatch["bags"][1]["configuration_fingerprint"] = (
                "complete:" + "4" * 64
            )
            with self.assertRaisesRegex(ValueError, "share one"):
                validate_diagonal_q_stage_request(mismatch)

    def test_auto_workers_use_half_affinity_and_cap_at_32(self):
        with mock.patch("os.sched_getaffinity", return_value=set(range(64))):
            self.assertEqual(resolved_forecast_workers("auto", 39), 32)
        with mock.patch("os.sched_getaffinity", return_value=set(range(6))):
            self.assertEqual(resolved_forecast_workers("auto", 39), 3)
        self.assertEqual(resolved_forecast_workers(100, 39), 39)
        for value in (True, 0, 257, 1.5, "many"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    resolved_forecast_workers(value, 39)

    def test_run_request_wires_real_adapter_progress_and_audited_writer(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bag = root / "flight.bag"
            bag.write_bytes(b"fake")
            request = self._request(bag)
            request_path = root / "request.json"
            request_path.write_text(json.dumps(request), encoding="utf-8")
            output = root / "q-artifact"
            digest = hashlib.sha256(bag.read_bytes()).hexdigest()
            arrays = SimpleNamespace(
                bag_sha256=digest,
                bag_size_bytes=bag.stat().st_size,
                bag_path=str(bag.resolve()),
                bag_record_start=100.0,
                bag_record_end=102.0,
            )
            episode = object()
            observations = SimpleNamespace(times=np.asarray((0.0, 0.1, 0.2)))
            prepared = SimpleNamespace(
                bag_id="bag-a",
                problem=SimpleNamespace(observations=observations),
            )
            terminal_expectations = (object(),)
            em_result = SimpleNamespace(
                final_expectations=terminal_expectations
            )
            result = SimpleNamespace(em_result=em_result)
            units = _q_e_step_units(3, 39)

            def fake_estimate(
                _prepared,
                _configuration,
                *,
                progress_callback,
                bag_progress_callback,
                run_id,
                **kwargs
            ):
                self.assertEqual(kwargs["ensemble_size"], 39)
                self.assertEqual(kwargs["forecast_workers"], 32)
                progress_callback(
                    ProgressEvent(
                        run_id=run_id,
                        stage_id="diagonal_q_expectation",
                        stage_label="Diagonal Q expectation",
                        completed_units=0,
                        total_units=1,
                        fraction=0.0,
                        elapsed_seconds=0.0,
                        eta_seconds=None,
                        iteration=1,
                        maximum_iterations=1,
                    )
                )
                for completed in (0, 1, units):
                    bag_progress_callback(
                        1,
                        "bag-a",
                        ProgressEvent(
                            run_id="nested",
                            stage_id="q_only_filter",
                            stage_label="Q-only ensemble filtering",
                            completed_units=completed,
                            total_units=units,
                            fraction=float(completed) / float(units),
                            elapsed_seconds=float(completed),
                            eta_seconds=None,
                            bag_id="bag-a",
                        ),
                    )
                progress_callback(
                    ProgressEvent(
                        run_id=run_id,
                        stage_id="diagonal_q_maximization",
                        stage_label="Diagonal Q maximization",
                        completed_units=1,
                        total_units=1,
                        fraction=1.0,
                        elapsed_seconds=1.0,
                        eta_seconds=0.0,
                        iteration=1,
                        maximum_iterations=1,
                    )
                )
                return result

            artifact_input = object()
            stream = io.StringIO()
            with mock.patch(
                "grape_param_estim.diagonal_q_stage_cli.read_grape_rosbag_arrays",
                return_value=arrays,
            ) as read_bag, mock.patch(
                "grape_param_estim.diagonal_q_stage_cli.build_real_flight_episode",
                return_value=episode,
            ) as build_episode, mock.patch(
                "grape_param_estim.diagonal_q_stage_cli.prepare_real_diagonal_q_bag",
                return_value=prepared,
            ), mock.patch(
                "grape_param_estim.diagonal_q_stage_cli._artifact_bag_input",
                return_value=artifact_input,
            ), mock.patch(
                "grape_param_estim.diagonal_q_stage_cli.run_real_diagonal_q_em",
                side_effect=fake_estimate,
            ), mock.patch(
                "grape_param_estim.diagonal_q_stage_cli.write_diagonal_q_artifact"
            ) as writer, mock.patch(
                "os.sched_getaffinity", return_value=set(range(64))
            ), mock.patch("sys.stdout", stream):
                self.assertEqual(run_request(str(request_path), str(output)), output)

            self.assertTrue(callable(read_bag.call_args.kwargs["checkpoint"]))
            self.assertIsNone(build_episode.call_args.kwargs["window_state"])
            writer.assert_called_once()
            written = writer.call_args.kwargs
            self.assertEqual(written["run_id"], request["run_id"])
            self.assertEqual(written["stage_id"], "diagonal_q")
            self.assertEqual(
                written["request_fingerprint"], request_fingerprint(request)
            )
            self.assertEqual(written["bag_inputs"], [artifact_input])
            self.assertIs(written["result"], em_result)
            self.assertEqual(written["expectations"], terminal_expectations)
            self.assertEqual(
                written["implementation_provenance"]["algorithm_version"],
                "diagonal-q-em-v1",
            )

            events = [
                ProgressEvent.from_json(line)
                for line in stream.getvalue().splitlines()
            ]
            self.assertGreaterEqual(len(events), 4)
            self.assertTrue(
                all(
                    left.fraction <= right.fraction
                    for left, right in zip(events[:-1], events[1:])
                )
            )
            self.assertEqual(events[-1].fraction, 1.0)
            self.assertEqual(events[-1].stage_id, "complete")
            self.assertTrue(all(event.run_id == request["run_id"] for event in events))

    def test_sha_mismatch_stops_before_episode_construction(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bag = root / "flight.bag"
            bag.write_bytes(b"fake")
            request = self._request(bag)
            request_path = root / "request.json"
            request_path.write_text(json.dumps(request), encoding="utf-8")
            arrays = SimpleNamespace(
                bag_sha256="0" * 64,
                bag_size_bytes=bag.stat().st_size,
                bag_path=str(bag),
            )
            with mock.patch(
                "grape_param_estim.diagonal_q_stage_cli.read_grape_rosbag_arrays",
                return_value=arrays,
            ), mock.patch(
                "grape_param_estim.diagonal_q_stage_cli.build_real_flight_episode"
            ) as build:
                with self.assertRaisesRegex(ValueError, "SHA256 changed"):
                    run_request(str(request_path), str(root / "output"))
            build.assert_not_called()

    def test_cancelled_run_marks_an_existing_writing_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bag = root / "flight.bag"
            bag.write_bytes(b"fake")
            request = self._request(bag)
            request_path = root / "request.json"
            request_path.write_text(json.dumps(request), encoding="utf-8")
            digest = hashlib.sha256(bag.read_bytes()).hexdigest()
            arrays = SimpleNamespace(
                bag_sha256=digest,
                bag_size_bytes=bag.stat().st_size,
                bag_path=str(bag.resolve()),
                bag_record_start=0.0,
                bag_record_end=1.0,
            )
            prepared = SimpleNamespace(
                bag_id="bag-a",
                problem=SimpleNamespace(
                    observations=SimpleNamespace(times=np.asarray((0.0, 0.1)))
                ),
            )
            output = root / "output"
            with mock.patch(
                "grape_param_estim.diagonal_q_stage_cli.read_grape_rosbag_arrays",
                return_value=arrays,
            ), mock.patch(
                "grape_param_estim.diagonal_q_stage_cli.build_real_flight_episode",
                return_value=object(),
            ), mock.patch(
                "grape_param_estim.diagonal_q_stage_cli.prepare_real_diagonal_q_bag",
                return_value=prepared,
            ), mock.patch(
                "grape_param_estim.diagonal_q_stage_cli._artifact_bag_input",
                return_value=object(),
            ), mock.patch(
                "grape_param_estim.diagonal_q_stage_cli.run_real_diagonal_q_em",
                side_effect=ProgressCancelled("signal_2"),
            ), mock.patch(
                "grape_param_estim.diagonal_q_stage_cli._cancel_writing_artifact_if_present",
                return_value=True,
            ) as marker, mock.patch("sys.stdout", io.StringIO()):
                with self.assertRaises(ProgressCancelled):
                    run_request(str(request_path), str(output))
            marker.assert_called_once_with(output, "signal_2")

    def test_executable_sets_blas_defaults_before_importing_worker(self):
        script = (
            Path(__file__).resolve().parents[2]
            / "ros/examples/grape-param-estim/scripts/grape_estimate_diagonal_q.py"
        )
        fake_module = SimpleNamespace(main=lambda: None)
        with mock.patch.dict(os.environ, {}, clear=True), mock.patch.dict(
            sys.modules,
            {"grape_param_estim.diagonal_q_stage_cli": fake_module},
        ):
            runpy.run_path(str(script), run_name="diagonal_q_script_test")
            self.assertEqual(os.environ["OPENBLAS_NUM_THREADS"], "1")
            self.assertEqual(os.environ["OMP_NUM_THREADS"], "1")
            self.assertEqual(os.environ["MKL_NUM_THREADS"], "1")


if __name__ == "__main__":
    unittest.main()
