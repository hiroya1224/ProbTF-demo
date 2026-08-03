import contextlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

import numpy as np

from grape_param_estim.ensemble_solver import (
    InitialPriorForecastDiagnostics,
    InitialPriorForecastFailure,
)
from grape_param_estim.joint_assimilation_cli import (
    _iteration_progress_metadata,
    run_request,
)
from grape_param_estim.progress import ProgressEvent


class JointAssimilationCliTest(unittest.TestCase):
    def test_iteration_label_parser_leaves_noniteration_stages_null(self):
        self.assertEqual(
            _iteration_progress_metadata("iteration 2/5 line search"),
            (2, 5),
        )
        self.assertEqual(
            _iteration_progress_metadata(
                "iteration 1/3 initial-prior backoff 1/8"
            ),
            (1, 3),
        )
        self.assertEqual(
            _iteration_progress_metadata("posterior diagnostics"),
            (None, None),
        )
        self.assertEqual(
            _iteration_progress_metadata("iteration 4/3 invalid"),
            (None, None),
        )

    def test_backoff_padding_and_unused_reserve_reach_fixed_progress_total(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bag = root / "flight.bag"
            bag.write_bytes(b"fake")
            request = {
                "schema": "grape-param-estim/assimilation-request/v1",
                "run_id": "progress-backoff",
                "project_id": "project-a",
                "project_request_fingerprint": "sha256:" + "b" * 64,
                "baseline_bag_id": "bag-a",
                "bags": [
                    {
                        "bag_id": "bag-a",
                        "path": str(bag),
                        "sha256": "a" * 64,
                        "episode_index": 0,
                        "selected_interval": [1.0, 2.0],
                        "configuration_fingerprint": "vehicle-a",
                    }
                ],
                "settings": {
                    "sample_period": 0.1,
                    "maximum_knots": 2,
                    "ensemble_size": 3,
                    "maximum_iterations": 1,
                    "seed": 4,
                    "delay_prior_mean": 0.02,
                    "delay_prior_standard_deviation": 0.01,
                    "allow_configuration_mismatch": False,
                    "maximum_initial_prior_backoff_trials": 2,
                },
            }
            request_path = root / "request.json"
            request_path.write_text(json.dumps(request), encoding="utf-8")
            output_path = root / "run"

            audit = InitialPriorForecastDiagnostics(
                radial_scale=0.5,
                backoff_trials=1,
                maximum_backoff_trials=2,
                requested_rank=2,
                effective_rank=2,
                failures=(
                    InitialPriorForecastFailure(
                        radial_scale=1.0,
                        exception_type="ValueError",
                        reason="synthetic divergence",
                    ),
                ),
            )
            posterior = SimpleNamespace(
                member_id=np.arange(3, dtype=np.int64),
                initial_prior_forecast=audit,
                termination_reason="maximum_iterations",
                converged=False,
            )
            result = SimpleNamespace(posterior=posterior)

            def fake_assimilate(
                _prepared,
                configuration,
                progress_callback,
                member_bag_callback,
                **_kwargs
            ):
                progress_callback(
                    "prior_ensemble_generation", 0, 1, "generating prior"
                )
                progress_callback(
                    "prior_ensemble_generation", 1, 1, "prior generated"
                )
                progress_callback("initial_forecast", 0, 20, "initial center")
                member_bag_callback(0, "bag-a", 1, 1)

                progress_callback(
                    "ensemble_forecast", 1, 20, "iteration 1/1 ensemble"
                )
                member_bag_callback(0, "bag-a", 1, 3)
                member_bag_callback(1, "bag-a", 2, 3)
                progress_callback(
                    "initial_prior_forecast_failed",
                    2,
                    20,
                    (
                        "iteration 1/1 initial-prior attempt failed at "
                        "radial scale 1: ValueError: synthetic divergence"
                    ),
                )
                progress_callback(
                    "initial_prior_backoff_forecast",
                    2,
                    20,
                    (
                        "iteration 1/1 initial-prior backoff 1/2 "
                        "(radial scale 0.5)"
                    ),
                )
                for member in range(configuration.ensemble_size):
                    member_bag_callback(member, "bag-a", member + 1, 3)

                progress_callback(
                    "line_search_trial", 3, 20, "iteration 1/1 line search"
                )
                member_bag_callback(0, "bag-a", 1, 1)
                for label in ("posterior linearization", "posterior ensemble"):
                    progress_callback(
                        "posterior_ensemble_forecast", 4, 20, label
                    )
                    for member in range(configuration.ensemble_size):
                        member_bag_callback(member, "bag-a", member + 1, 3)
                progress_callback(
                    "posterior_ensemble_forecast", 5, 20, "posterior center"
                )
                member_bag_callback(0, "bag-a", 1, 1)
                progress_callback(
                    "posterior_diagnostics", 0, 1, "computing diagnostics"
                )
                for replay in range(2):
                    for member in range(configuration.ensemble_size):
                        member_bag_callback(member, "bag-a", member + 1, 3)
                progress_callback(
                    "posterior_diagnostics", 1, 1, "diagnostics complete"
                )
                return result

            stream = io.StringIO()
            patches = (
                mock.patch(
                    "grape_param_estim.joint_assimilation_cli.read_grape_rosbag_arrays",
                    return_value=SimpleNamespace(bag_sha256="a" * 64),
                ),
                mock.patch(
                    "grape_param_estim.joint_assimilation_cli."
                    "build_real_flight_episode",
                    return_value=SimpleNamespace(),
                ),
                mock.patch(
                    "grape_param_estim.joint_assimilation_cli.prepare_joint_flight",
                    return_value=SimpleNamespace(),
                ),
                mock.patch(
                    "grape_param_estim.joint_assimilation_cli.assimilate_joint_flights",
                    side_effect=fake_assimilate,
                ),
                mock.patch(
                    "grape_param_estim.joint_assimilation_cli."
                    "write_joint_assimilation_payloads"
                ),
                mock.patch(
                    "grape_param_estim.joint_assimilation_cli.mark_bundle_complete"
                ),
            )
            with contextlib.ExitStack() as stack:
                for patch in patches:
                    stack.enter_context(patch)
                with contextlib.redirect_stdout(stream):
                    run_request(str(request_path), str(output_path))

            events = tuple(
                ProgressEvent.from_json(line)
                for line in stream.getvalue().splitlines()
            )
            completed = [event.completed_units for event in events]
            self.assertEqual(completed, sorted(completed))
            self.assertEqual(events[-1].stage_id, "complete")
            self.assertEqual(
                events[-1].completed_units, events[-1].total_units
            )
            self.assertTrue(
                all(event.total_units == events[-1].total_units for event in events)
            )
            skipped = next(
                event
                for event in events
                if event.stage_id == "initial_prior_backoff_skipped"
            )
            self.assertEqual((skipped.iteration, skipped.maximum_iterations), (1, 1))
            self.assertIn("1 member-bag", skipped.message)
            unused = next(
                event
                for event in events
                if event.stage_id == "initial_prior_backoff_unused_reserve"
            )
            self.assertIsNone(unused.iteration)
            iteration_events = [
                event for event in events if "iteration 1/1" in event.stage_label
            ]
            self.assertTrue(iteration_events)
            self.assertTrue(
                all(
                    (event.iteration, event.maximum_iterations) == (1, 1)
                    for event in iteration_events
                )
            )
            request_event = next(
                event for event in events if event.stage_id == "request_validation"
            )
            self.assertIsNone(request_event.iteration)


if __name__ == "__main__":
    unittest.main()
