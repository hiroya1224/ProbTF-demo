import copy
from pathlib import Path
import tempfile
import unittest

from grape_param_estim_gui import stage_requests
from grape_param_estim_gui.stage_requests import (
    BATCH_ESTIMATION_REQUEST_SCHEMA,
    BATCH_ESTIMATION_STAGE_ID,
    batch_estimation_settings,
    build_batch_estimation_request,
    stage_bag_requests,
    workflow_mode_run_mode,
)
from grape_param_estim_gui.state import BagRecord
from grape_param_estim_gui.workflow import WorkflowMode, canonical_fingerprint


def _settings(*, mcmc_enabled=True):
    return {
        "q": {"definition": "explicit-six-axis"},
        "parameter_prior": {"kind": "gaussian"},
        "delay": {"prior_kind": "uniform"},
        "actuator_model": {"source": "calibration-a"},
        "knot_policy": {"origin": "interval_start"},
        "interpolation_policy": {"orientation": "so3_geodesic"},
        "controller_snapshot_policy": {
            "source": "bag_startup_parameter_updates"
        },
        "mode_hypotheses": [{"mode_id": "recorded"}],
        "solver_settings": {"maximum_iterations": 50},
        "em_settings": {"maximum_iterations": 10},
        "mcmc_settings": {"enabled": mcmc_enabled},
    }


def _bag_configuration():
    return {
        "observation_factors": {"pose": {"enabled": True}},
        "fixed_factor_covariances": {"position_kinematic": {}},
        "initial_state_prior_covariances": {"position": {}},
    }


class StageRequestTests(unittest.TestCase):
    def test_workflow_mode_selects_estimate_or_estimate_and_sample(self):
        self.assertEqual(
            workflow_mode_run_mode(WorkflowMode.STEP), "estimate_only"
        )
        self.assertEqual(
            workflow_mode_run_mode(WorkflowMode.ALL), "estimate_and_sample"
        )

    def test_settings_are_strict_and_never_supply_scientific_defaults(self):
        settings = _settings()
        self.assertEqual(
            batch_estimation_settings(
                settings, run_mode="estimate_and_sample"
            ),
            settings,
        )
        for key in tuple(settings):
            candidate = copy.deepcopy(settings)
            del candidate[key]
            with self.subTest(key=key), self.assertRaisesRegex(
                ValueError, "missing"
            ):
                batch_estimation_settings(
                    candidate, run_mode="estimate_and_sample"
                )

        with self.assertRaisesRegex(ValueError, "match run_mode"):
            batch_estimation_settings(
                _settings(mcmc_enabled=True), run_mode="estimate_only"
            )

    def test_selected_bags_bind_exact_interval_sha_and_factor_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = []
            for bag_id in ("bag-z", "bag-a"):
                path = root / (bag_id + ".bag")
                path.write_bytes(b"bag")
                records.append(
                    BagRecord(
                        bag_id=bag_id,
                        path=path,
                        source_path=path,
                        sha256="1" * 64,
                        inspection={"sensor_contract": {}},
                        selected_interval=(18.0, 24.0),
                    )
                )
            bags = stage_bag_requests(
                records,
                {bag_id: _bag_configuration() for bag_id in ("bag-a", "bag-z")},
            )

        self.assertEqual([value["bag_id"] for value in bags], ["bag-a", "bag-z"])
        self.assertEqual(bags[0]["sha256"], "sha256:" + "1" * 64)
        self.assertEqual(bags[0]["interval_seconds"], [18.0, 24.0])
        self.assertEqual(
            set(bags[0]),
            {
                "bag_id",
                "path",
                "sha256",
                "interval_seconds",
                "observation_factors",
                "fixed_factor_covariances",
                "initial_state_prior_covariances",
            },
        )

    def test_one_command_request_is_exact_stable_and_resumable(self):
        settings = _settings()
        bag = {
            "bag_id": "bag-a",
            "path": "/tmp/a.bag",
            "sha256": "sha256:" + "1" * 64,
            "interval_seconds": [18.0, 24.0],
            **_bag_configuration(),
        }
        first = build_batch_estimation_request(
            run_id="run-a",
            run_mode="estimate_and_sample",
            resume=True,
            output_directory="/tmp/project/runs/run-a/estimation_run",
            bags=[bag],
            settings=settings,
        )
        second = build_batch_estimation_request(
            run_id="run-a",
            run_mode="estimate_and_sample",
            resume=True,
            output_directory="/tmp/project/runs/run-a/estimation_run",
            bags=[bag],
            settings=settings,
        )
        self.assertEqual(first, second)
        self.assertEqual(first["schema"], BATCH_ESTIMATION_REQUEST_SCHEMA)
        self.assertEqual(first["resume"], True)
        self.assertEqual(
            set(first),
            {
                "schema",
                "run_id",
                "run_mode",
                "resume",
                "output_directory",
                "bags",
                *settings.keys(),
            },
        )
        self.assertEqual(canonical_fingerprint(first), canonical_fingerprint(second))

    def test_old_two_stage_api_is_absent(self):
        self.assertEqual(BATCH_ESTIMATION_STAGE_ID, "batch_estimation")
        for name in (
            "build_diagonal_q_stage_request",
            "build_augmented_parameter_stage_request",
            "diagonal_q_stage_settings",
            "augmented_parameter_stage_settings",
        ):
            self.assertFalse(hasattr(stage_requests, name))


if __name__ == "__main__":
    unittest.main()
