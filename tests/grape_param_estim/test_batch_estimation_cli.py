from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from grape_param_estim.batch_estimation_cli import (
    BatchProgressReporter,
    configuration_fingerprint,
    execute_batch_estimation,
    main,
    planned_progress_units,
)
from grape_param_estim.batch_request import validate_batch_estimation_request
from grape_param_estim.progress import (
    ProgressEvent,
    STAGE_OPTIMIZING_FULL_TRAJECTORY,
    STAGE_PREPARING_TRAJECTORY,
    STAGE_WRITING_ARTIFACTS,
)
from tests.grape_param_estim.test_batch_preparation import (
    _request_payload,
)


class BatchEstimationCliTests(unittest.TestCase):
    def _request(self, root):
        from grape_param_estim.batch_artifact import file_sha256

        bag = root / "flight.bag"
        bag.write_bytes(b"batch cli request")
        return validate_batch_estimation_request(
            _request_payload(root, (("flight-a", bag, file_sha256(bag)),))
        )

    def test_progress_adapter_has_stable_total_and_terminal_fraction(self):
        import tempfile

        with tempfile.TemporaryDirectory() as temporary:
            request = self._request(Path(temporary))
            events = []
            reporter = BatchProgressReporter(request, events.append)
            expected_total = planned_progress_units(request)
            reporter(STAGE_PREPARING_TRAJECTORY, 1, 1, "prepared")
            reporter(
                STAGE_OPTIMIZING_FULL_TRAJECTORY,
                1,
                3,
                "iteration 1",
            )
            reporter(
                STAGE_OPTIMIZING_FULL_TRAJECTORY,
                2,
                3,
                "iteration 2",
            )
            reporter(STAGE_WRITING_ARTIFACTS, 0, 1, "publishing")
            reporter(STAGE_WRITING_ARTIFACTS, 1, 1, "run complete")
            self.assertTrue(all(v.total_units == expected_total for v in events))
            self.assertEqual(events[-1].completed_units, expected_total)
            self.assertEqual(events[-1].fraction, 1.0)
            self.assertEqual(events[-1].eta_seconds, 0.0)
            self.assertEqual(
                [v.fraction for v in events],
                sorted(v.fraction for v in events),
            )

    def test_main_reserves_stdout_for_strict_progress_jsonl(self):
        import tempfile

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            request = self._request(root)
            request_path = root / "request.json"
            request_path.write_text(
                json.dumps(
                    request.payload,
                    default=lambda value: (
                        dict(value)
                        if hasattr(value, "items")
                        else list(value)
                    ),
                ),
                encoding="utf-8",
            )

            def fake_execute(selected, **kwargs):
                callback = kwargs["progress"]
                callback(
                    STAGE_PREPARING_TRAJECTORY, 1, 1, "prepared flight-a"
                )
                callback(STAGE_WRITING_ARTIFACTS, 0, 1, "publishing strict run")
                callback(STAGE_WRITING_ARTIFACTS, 1, 1, "run complete")
                return SimpleNamespace(root=selected.output_directory)

            stdout = io.StringIO()
            stderr = io.StringIO()
            with patch(
                "grape_param_estim.batch_estimation_cli.execute_batch_estimation",
                side_effect=fake_execute,
            ), patch(
                "grape_param_estim.batch_estimation_cli.discover_estimator_revision",
                return_value="test-revision",
            ), redirect_stdout(stdout), redirect_stderr(stderr):
                exit_code = main(("--request", str(request_path)))
            self.assertEqual(exit_code, 0)
            events = [
                ProgressEvent.from_json(line)
                for line in stdout.getvalue().splitlines()
            ]
            self.assertEqual(events[-1].fraction, 1.0)
            self.assertIn("batch estimation complete", stderr.getvalue())

    def test_configuration_fingerprint_ignores_output_identity_not_science(self):
        import tempfile

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bag = root / "flight.bag"
            bag.write_bytes(b"batch cli")
            from grape_param_estim.batch_artifact import file_sha256

            payload = _request_payload(
                root, (("flight-a", bag, file_sha256(bag)),)
            )
            first = validate_batch_estimation_request(payload)
            changed = dict(payload)
            changed["run_id"] = "another-run"
            changed["output_directory"] = str(root / "another-output")
            second = validate_batch_estimation_request(changed)
            self.assertEqual(
                configuration_fingerprint(first),
                configuration_fingerprint(second),
            )
            changed_q = dict(payload)
            changed_q["q"] = dict(payload["q"])
            changed_q["q"]["initial_diagonal"] = [2.0] * 6
            third = validate_batch_estimation_request(changed_q)
            self.assertNotEqual(
                configuration_fingerprint(first),
                configuration_fingerprint(third),
            )

    @patch("grape_param_estim.batch_estimation_cli.write_batch_estimation_run")
    @patch("grape_param_estim.batch_estimation_cli.export_batch_estimation_artifact_payload")
    @patch("grape_param_estim.batch_estimation_cli.measure_run_performance")
    @patch("grape_param_estim.batch_estimation_cli.run_real_estimation")
    @patch("grape_param_estim.batch_estimation_cli.prepare_real_estimation_inputs")
    def test_executes_one_strict_run_and_passes_writer_arguments_exactly(
        self,
        prepare,
        run,
        measure,
        export,
        write,
    ):
        import tempfile
        from grape_param_estim.batch_artifact import file_sha256

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bag = root / "flight.bag"
            bag.write_bytes(b"batch cli orchestration")
            request = validate_batch_estimation_request(
                _request_payload(
                    root, (("flight-a", bag, file_sha256(bag)),)
                )
            )
            from grape_param_estim.controller import ControllerConfig

            controller = ControllerConfig.grape()
            inputs = SimpleNamespace(
                flight_data=(
                    SimpleNamespace(
                        bag_id="flight-a",
                        controller_configuration=controller,
                    ),
                ),
                initializations=(object(),),
            )
            prepare.return_value = inputs
            em = SimpleNamespace(converged=True)
            selected = SimpleNamespace(
                final_solution=object(),
                em=em,
                static_geometry=object(),
                lag_profile_history=(object(),),
                final_q_lag_profile_history=(object(),),
                delay_uncertainty=SimpleNamespace(
                    standard_deviation_seconds=0.01,
                    curvature=10000.0,
                    source="profile",
                ),
            )
            run.return_value = SimpleNamespace(
                modes=(SimpleNamespace(mode_id="recorded-mode", em=em),),
                selected_mode=selected,
                mcmc=None,
            )
            performance = object()
            measure.return_value = performance
            export.return_value = SimpleNamespace(
                writer_arguments={"manifest_metadata": {"run": "strict"}}
            )
            written = SimpleNamespace(root=request.output_directory)
            write.return_value = written
            progress = []
            result = execute_batch_estimation(
                request,
                estimator_revision="test-revision",
                progress=lambda *value: progress.append(value),
            )
            self.assertIs(result, written)
            export.assert_called_once()
            call = export.call_args.kwargs
            self.assertIs(call["request"], request)
            self.assertIs(call["performance"], performance)
            self.assertEqual(call["mcmc_chains"], ())
            write.assert_called_once_with(
                request.output_directory,
                manifest_metadata={"run": "strict"},
            )
            self.assertEqual(progress[-1][0], "writing_artifacts")
            self.assertEqual(progress[-1][1:3], (1, 1))


if __name__ == "__main__":
    unittest.main()
