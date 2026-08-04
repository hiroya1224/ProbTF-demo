from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from grape_param_estim.batch_estimation_cli import (
    configuration_fingerprint,
    execute_batch_estimation,
)
from grape_param_estim.batch_request import validate_batch_estimation_request
from tests.grape_param_estim.test_batch_preparation import (
    _request_payload,
)


class BatchEstimationCliTests(unittest.TestCase):
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
