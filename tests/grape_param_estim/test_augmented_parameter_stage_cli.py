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
from grape_param_estim.augmented_parameter_stage_cli import (
    AUGMENTED_PARAMETER_STAGE_REQUEST_SCHEMA,
    _filter_work_units,
    _validate_rebuilt_bag_against_upstream,
    resolved_forecast_workers,
    run_request,
    validate_augmented_parameter_stage_request,
)
from grape_param_estim.diagonal_q import BodyWrenchDiagonalCovariance
from grape_param_estim.progress import ProgressCancelled, ProgressEvent


class AugmentedParameterStageCliTest(unittest.TestCase):
    def _request(self, bag: Path, q_root: Path):
        return {
            "schema": AUGMENTED_PARAMETER_STAGE_REQUEST_SCHEMA,
            "run_id": "parameter-attempt-1",
            "project_fingerprint": "sha256:" + "1" * 64,
            "stage_id": "static_parameters",
            "stage_input_fingerprint": "sha256:" + "2" * 64,
            "upstream_diagonal_q": {
                "path": str(q_root),
                "artifact_fingerprint": "sha256:" + "3" * 64,
            },
            "bags": [
                {
                    "bag_id": "bag-a",
                    "path": str(bag),
                    "sha256": hashlib.sha256(bag.read_bytes()).hexdigest(),
                    "episode_index": 0,
                    "selected_interval_local_seconds": [1.0, 1.2],
                    "configuration_fingerprint": (
                        "manual-group:sha256:" + "4" * 64
                    ),
                }
            ],
            "settings": {
                "sample_period": 0.1,
                "ensemble_size": 58,
                "delay_prior_mean_seconds": 0.02,
                "delay_prior_standard_deviation_seconds": 0.01,
                "maximum_delay_seconds": 0.2,
                "covariance_rcond": 1.0e-12,
                "seed": 23,
                "forecast_workers": "auto",
            },
        }

    def test_request_contract_is_exact_and_delay_prior_is_interior(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bag = root / "flight.bag"
            bag.write_bytes(b"fake")
            q_root = root / "q"
            q_root.mkdir()
            request = self._request(bag, q_root)
            self.assertIs(
                validate_augmented_parameter_stage_request(request), request
            )

            invalid_values = []
            extra = copy.deepcopy(request)
            extra["execution"] = {}
            invalid_values.append(extra)
            wrong_stage = copy.deepcopy(request)
            wrong_stage["stage_id"] = "diagonal_q"
            invalid_values.append(wrong_stage)
            too_small = copy.deepcopy(request)
            too_small["settings"]["ensemble_size"] = 57
            invalid_values.append(too_small)
            boundary_mean = copy.deepcopy(request)
            boundary_mean["settings"]["delay_prior_mean_seconds"] = 0.2
            invalid_values.append(boundary_mean)
            invalid_workers = copy.deepcopy(request)
            invalid_workers["settings"]["forecast_workers"] = True
            invalid_values.append(invalid_workers)
            for invalid in invalid_values:
                with self.subTest(invalid=invalid):
                    with self.assertRaises(ValueError):
                        validate_augmented_parameter_stage_request(invalid)

    def test_auto_workers_use_half_affinity_and_cap_at_32(self):
        with mock.patch("os.sched_getaffinity", return_value=set(range(64))):
            self.assertEqual(resolved_forecast_workers("auto", 58), 32)
        with mock.patch("os.sched_getaffinity", return_value=set(range(8))):
            self.assertEqual(resolved_forecast_workers("auto", 58), 4)
        self.assertEqual(resolved_forecast_workers(100, 58), 58)

    def test_rebuilt_stage1_inputs_must_match_r_tau_model_and_pilot(self):
        common = {
            "source_path": "/flight/bag-a.bag",
            "source_sha256": "a" * 64,
            "source_size_bytes": 123,
            "selected_interval_local_seconds": (1.0, 2.0),
            "effective_interval_local_seconds": (1.01, 1.91),
            "episode_index": 0,
            "configuration_fingerprint": "complete:" + "b" * 64,
            "constant_delay_seconds": 0.02,
            "fixed_model_fingerprint": "sha256:" + "c" * 64,
            "observation_covariance_fingerprint": "sha256:" + "d" * 64,
            "translation_covariance": np.eye(3),
            "rotation_covariance": 2.0 * np.eye(3),
            "fixed_model_provenance": {"model": "fixed"},
            "fixed_r_provenance": {"method": "robust"},
        }
        upstream_input = SimpleNamespace(**common)
        rebuilt = SimpleNamespace(**copy.deepcopy(common))
        times = np.asarray((0.0, 0.1, 0.2))
        sigma = np.arange(1.0, 7.0)
        prepared = SimpleNamespace(
            problem=SimpleNamespace(
                observations=SimpleNamespace(times=times.copy())
            ),
            calibration=SimpleNamespace(
                correlation_time=0.3,
                stationary_standard_deviation=sigma.copy(),
            ),
        )
        expectation = SimpleNamespace(times=times.copy(), correlation_time=0.3)
        pilot = SimpleNamespace(stationary_standard_deviation=sigma.copy())
        _validate_rebuilt_bag_against_upstream(
            "bag-a",
            rebuilt,
            prepared,
            upstream_input,
            expectation,
            pilot,
        )

        relocated_values = copy.deepcopy(common)
        relocated_values["source_path"] = "/restored/project/bags/bag-a.bag"
        _validate_rebuilt_bag_against_upstream(
            "bag-a",
            SimpleNamespace(**relocated_values),
            prepared,
            upstream_input,
            expectation,
            pilot,
        )

        changed = SimpleNamespace(**copy.deepcopy(common))
        changed.translation_covariance[0, 0] = 9.0
        with self.assertRaisesRegex(ValueError, "translation_covariance"):
            _validate_rebuilt_bag_against_upstream(
                "bag-a",
                changed,
                prepared,
                upstream_input,
                expectation,
                pilot,
            )
        changed = SimpleNamespace(**copy.deepcopy(common))
        changed.fixed_model_provenance["model"] = "changed"
        with self.assertRaisesRegex(ValueError, "fixed model"):
            _validate_rebuilt_bag_against_upstream(
                "bag-a",
                changed,
                prepared,
                upstream_input,
                expectation,
                pilot,
            )
        changed_prepared = SimpleNamespace(
            problem=prepared.problem,
            calibration=SimpleNamespace(
                correlation_time=0.31,
                stationary_standard_deviation=sigma,
            ),
        )
        with self.assertRaisesRegex(ValueError, "correlation time"):
            _validate_rebuilt_bag_against_upstream(
                "bag-a",
                rebuilt,
                changed_prepared,
                upstream_input,
                expectation,
                pilot,
            )

    def _fake_upstream(self, request):
        q_input = SimpleNamespace(
            bag_id="bag-a",
            source_path=str(Path(request["bags"][0]["path"]).resolve()),
            source_sha256=request["bags"][0]["sha256"],
            source_size_bytes=4,
            selected_interval_local_seconds=(1.0, 1.2),
            effective_interval_local_seconds=(1.0, 1.2),
            episode_index=0,
            configuration_fingerprint=request["bags"][0][
                "configuration_fingerprint"
            ],
            constant_delay_seconds=0.02,
            fixed_model_fingerprint="sha256:" + "5" * 64,
            observation_covariance_fingerprint="sha256:" + "6" * 64,
            translation_covariance=np.eye(3),
            rotation_covariance=np.eye(3),
            fixed_model_provenance={"fixed_model": "stage-1"},
            fixed_r_provenance={"method": "robust"},
        )
        times = np.asarray((0.0, 0.1, 0.2))
        manifest = {
            "project_fingerprint": request["project_fingerprint"],
            "stage_id": "diagonal_q",
            "run_id": "q-run",
        }
        bundle = SimpleNamespace(
            manifest=manifest,
            bag_ids=("bag-a",),
            bag_inputs=(q_input,),
            expectations=(
                SimpleNamespace(
                    bag_id="bag-a", times=times, correlation_time=0.3
                ),
            ),
            pilots=(
                SimpleNamespace(
                    bag_id="bag-a",
                    stationary_standard_deviation=np.ones(6),
                ),
            ),
            covariance=BodyWrenchDiagonalCovariance(np.ones(6)),
        )
        return bundle, q_input, times

    def test_run_request_wires_fixed_q_filter_progress_and_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bag = root / "flight.bag"
            bag.write_bytes(b"fake")
            q_root = root / "q"
            q_root.mkdir()
            request = self._request(bag, q_root)
            upstream, q_input, times = self._fake_upstream(request)
            q_fingerprint = request_fingerprint(upstream.manifest)
            request["upstream_diagonal_q"][
                "artifact_fingerprint"
            ] = q_fingerprint
            request_path = root / "request.json"
            request_path.write_text(json.dumps(request), encoding="utf-8")
            output = root / "stage2"
            arrays = SimpleNamespace(
                bag_sha256=request["bags"][0]["sha256"],
                bag_size_bytes=bag.stat().st_size,
                bag_path=str(bag.resolve()),
            )
            problem = SimpleNamespace(
                observations=SimpleNamespace(times=times),
                nominal_trajectory=object(),
            )
            diagonal_bag = SimpleNamespace(
                problem=problem,
                calibration=SimpleNamespace(
                    correlation_time=0.3,
                    stationary_standard_deviation=np.ones(6),
                ),
            )
            augmented_bag = SimpleNamespace(
                bag_id="bag-a", problem=problem
            )
            stage_result = object()
            artifact_input = object()
            compute_units = _filter_work_units(58, 3)

            def fake_filter(
                _bags,
                covariance,
                *,
                progress_callback,
                run_id,
                **kwargs
            ):
                np.testing.assert_array_equal(
                    covariance.stationary_variance, np.ones(6)
                )
                self.assertEqual(kwargs["ensemble_size"], 58)
                self.assertEqual(kwargs["forecast_workers"], 32)
                for completed in (0, 1, compute_units):
                    progress_callback(
                        ProgressEvent(
                            run_id=run_id,
                            stage_id="multi_bag_augmented_parameter",
                            stage_label="Augmented parameter filter",
                            completed_units=completed,
                            total_units=compute_units,
                            fraction=float(completed) / float(compute_units),
                            elapsed_seconds=float(completed),
                            eta_seconds=None,
                            bag_id="bag-a",
                        )
                    )
                return stage_result

            stream = io.StringIO()
            with mock.patch(
                "grape_param_estim.augmented_parameter_stage_cli.diagonal_q_artifact_fingerprint",
                return_value=q_fingerprint,
            ), mock.patch(
                "grape_param_estim.augmented_parameter_stage_cli.load_diagonal_q_artifact",
                return_value=upstream,
            ), mock.patch(
                "grape_param_estim.augmented_parameter_stage_cli.read_grape_rosbag_arrays",
                return_value=arrays,
            ), mock.patch(
                "grape_param_estim.augmented_parameter_stage_cli.build_real_flight_episode",
                return_value=object(),
            ) as build_episode, mock.patch(
                "grape_param_estim.augmented_parameter_stage_cli.prepare_real_diagonal_q_bag",
                return_value=diagonal_bag,
            ) as prepare, mock.patch(
                "grape_param_estim.augmented_parameter_stage_cli._stage1_artifact_bag_input",
                return_value=q_input,
            ), mock.patch(
                "grape_param_estim.augmented_parameter_stage_cli.PreparedAugmentedParameterBag.from_diagonal_q_bag",
                return_value=augmented_bag,
            ), mock.patch(
                "grape_param_estim.augmented_parameter_stage_cli.AugmentedParameterArtifactBagInput",
                return_value=artifact_input,
            ), mock.patch(
                "grape_param_estim.augmented_parameter_stage_cli.run_multi_bag_augmented_parameter_filter",
                side_effect=fake_filter,
            ), mock.patch(
                "grape_param_estim.augmented_parameter_stage_cli.write_augmented_parameter_artifact"
            ) as writer, mock.patch(
                "os.sched_getaffinity", return_value=set(range(64))
            ), mock.patch("sys.stdout", stream):
                self.assertEqual(run_request(str(request_path), str(output)), output)

            self.assertIsNone(build_episode.call_args.kwargs["window_state"])
            self.assertEqual(
                prepare.call_args.kwargs["initial_delay"],
                q_input.constant_delay_seconds,
            )
            writer.assert_called_once()
            written = writer.call_args.kwargs
            self.assertEqual(written["upstream_diagonal_q_path"], q_root)
            self.assertEqual(
                written["upstream_diagonal_q_fingerprint"], q_fingerprint
            )
            self.assertEqual(written["bag_inputs"], [artifact_input])
            self.assertIs(written["result"], stage_result)
            events = [
                ProgressEvent.from_json(line)
                for line in stream.getvalue().splitlines()
            ]
            self.assertTrue(
                all(
                    left.fraction <= right.fraction
                    for left, right in zip(events[:-1], events[1:])
                )
            )
            self.assertEqual(events[-1].fraction, 1.0)
            self.assertEqual(events[-1].stage_id, "complete")

    def test_upstream_fingerprint_mismatch_stops_before_bag_read(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bag = root / "flight.bag"
            bag.write_bytes(b"fake")
            q_root = root / "q"
            q_root.mkdir()
            request = self._request(bag, q_root)
            request_path = root / "request.json"
            request_path.write_text(json.dumps(request), encoding="utf-8")
            with mock.patch(
                "grape_param_estim.augmented_parameter_stage_cli.diagonal_q_artifact_fingerprint",
                return_value="sha256:" + "9" * 64,
            ), mock.patch(
                "grape_param_estim.augmented_parameter_stage_cli.read_grape_rosbag_arrays"
            ) as read_bag:
                with self.assertRaisesRegex(ValueError, "fingerprint changed"):
                    run_request(str(request_path), str(root / "output"))
            read_bag.assert_not_called()

    def test_cancelled_filter_marks_a_writing_stage2_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bag = root / "flight.bag"
            bag.write_bytes(b"fake")
            q_root = root / "q"
            q_root.mkdir()
            request = self._request(bag, q_root)
            upstream, q_input, times = self._fake_upstream(request)
            q_fingerprint = request_fingerprint(upstream.manifest)
            request["upstream_diagonal_q"][
                "artifact_fingerprint"
            ] = q_fingerprint
            request_path = root / "request.json"
            request_path.write_text(json.dumps(request), encoding="utf-8")
            problem = SimpleNamespace(
                observations=SimpleNamespace(times=times),
                nominal_trajectory=object(),
            )
            diagonal_bag = SimpleNamespace(
                problem=problem,
                calibration=SimpleNamespace(
                    correlation_time=0.3,
                    stationary_standard_deviation=np.ones(6),
                ),
            )
            arrays = SimpleNamespace(
                bag_sha256=request["bags"][0]["sha256"],
                bag_size_bytes=bag.stat().st_size,
                bag_path=str(bag.resolve()),
            )
            output = root / "output"
            with mock.patch(
                "grape_param_estim.augmented_parameter_stage_cli.diagonal_q_artifact_fingerprint",
                return_value=q_fingerprint,
            ), mock.patch(
                "grape_param_estim.augmented_parameter_stage_cli.load_diagonal_q_artifact",
                return_value=upstream,
            ), mock.patch(
                "grape_param_estim.augmented_parameter_stage_cli.read_grape_rosbag_arrays",
                return_value=arrays,
            ), mock.patch(
                "grape_param_estim.augmented_parameter_stage_cli.build_real_flight_episode",
                return_value=object(),
            ), mock.patch(
                "grape_param_estim.augmented_parameter_stage_cli.prepare_real_diagonal_q_bag",
                return_value=diagonal_bag,
            ), mock.patch(
                "grape_param_estim.augmented_parameter_stage_cli._stage1_artifact_bag_input",
                return_value=q_input,
            ), mock.patch(
                "grape_param_estim.augmented_parameter_stage_cli.PreparedAugmentedParameterBag.from_diagonal_q_bag",
                return_value=SimpleNamespace(bag_id="bag-a", problem=problem),
            ), mock.patch(
                "grape_param_estim.augmented_parameter_stage_cli.AugmentedParameterArtifactBagInput",
                return_value=object(),
            ), mock.patch(
                "grape_param_estim.augmented_parameter_stage_cli.run_multi_bag_augmented_parameter_filter",
                side_effect=ProgressCancelled("signal_2"),
            ), mock.patch(
                "grape_param_estim.augmented_parameter_stage_cli._cancel_writing_artifact_if_present",
                return_value=True,
            ) as marker, mock.patch("sys.stdout", io.StringIO()):
                with self.assertRaises(ProgressCancelled):
                    run_request(str(request_path), str(output))
            marker.assert_called_once_with(output, "signal_2")

    def test_executable_sets_blas_defaults_before_worker_import(self):
        script = (
            Path(__file__).resolve().parents[2]
            / "ros/examples/grape-param-estim/scripts/"
            "grape_estimate_augmented_parameters.py"
        )
        fake_module = SimpleNamespace(main=lambda: None)
        with mock.patch.dict(os.environ, {}, clear=True), mock.patch.dict(
            sys.modules,
            {"grape_param_estim.augmented_parameter_stage_cli": fake_module},
        ):
            runpy.run_path(str(script), run_name="augmented_stage_script_test")
            self.assertEqual(os.environ["OPENBLAS_NUM_THREADS"], "1")
            self.assertEqual(os.environ["OMP_NUM_THREADS"], "1")
            self.assertEqual(os.environ["MKL_NUM_THREADS"], "1")


if __name__ == "__main__":
    unittest.main()
