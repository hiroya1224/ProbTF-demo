from pathlib import Path
import tempfile
import unittest

from grape_param_estim_gui.stage_requests import (
    MINIMUM_STAGED_ENSEMBLE_SIZE,
    augmented_parameter_stage_settings,
    build_augmented_parameter_stage_request,
    build_diagonal_q_stage_request,
    diagonal_q_stage_settings,
    stage_bag_requests,
)
from grape_param_estim_gui.state import BagRecord
from grape_param_estim_gui.workflow import canonical_fingerprint


class StageRequestTests(unittest.TestCase):
    def _settings(self):
        return {
            "sample_period": 0.04,
            "ensemble_size": 64,
            "maximum_iterations": 4,
            "convergence_tolerance": 2.0e-3,
            "seed": 23,
            "delay_prior_mean": 0.02,
        }

    def test_q_settings_keep_all_six_diagonal_floors(self):
        source = self._settings()
        source["q_component_floor"] = [
            1.0e-8,
            2.0e-8,
            3.0e-8,
            4.0e-8,
            5.0e-8,
            6.0e-8,
        ]
        source["q_maximum_em_iterations"] = 7
        source["q_log_q_tolerance"] = 4.0e-4
        source["forecast_workers"] = 12
        resolved = diagonal_q_stage_settings(source)
        self.assertEqual(resolved["component_floor"], source["q_component_floor"])
        self.assertEqual(resolved["maximum_em_iterations"], 7)
        self.assertEqual(resolved["log_q_tolerance"], 4.0e-4)
        self.assertEqual(resolved["forecast_workers"], 12)

    def test_staged_workflow_rejects_an_undersized_ensemble(self):
        source = self._settings()
        source["ensemble_size"] = MINIMUM_STAGED_ENSEMBLE_SIZE - 1
        with self.assertRaisesRegex(ValueError, "at least 58"):
            diagonal_q_stage_settings(source)

    def test_augmented_settings_are_explicit_and_bounded(self):
        source = self._settings()
        source.update(
            delay_prior_standard_deviation=0.015,
            maximum_delay=0.2,
            covariance_rcond=1.0e-10,
            forecast_workers="auto",
        )
        resolved = augmented_parameter_stage_settings(source)
        self.assertEqual(resolved["ensemble_size"], 64)
        self.assertEqual(resolved["delay_prior_mean_seconds"], 0.02)
        self.assertEqual(
            resolved["delay_prior_standard_deviation_seconds"], 0.015
        )
        self.assertEqual(resolved["maximum_delay_seconds"], 0.2)
        self.assertEqual(resolved["covariance_rcond"], 1.0e-10)
        self.assertEqual(resolved["forecast_workers"], "auto")

        source["delay_prior_mean"] = 0.2
        with self.assertRaisesRegex(ValueError, "below maximum_delay"):
            augmented_parameter_stage_settings(source)

    def test_selected_bags_are_canonical_and_preserve_manual_group(self):
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
                        inspection={
                            "recommended_interval": {"episode_index": 2}
                        },
                        selected_interval=(1.0, 3.0),
                        configuration_fingerprint=(
                            "manual-group:sha256:" + "2" * 64
                        ),
                    )
                )
            bags = stage_bag_requests(records)
        self.assertEqual([value["bag_id"] for value in bags], ["bag-a", "bag-z"])
        self.assertEqual(
            bags[0]["configuration_fingerprint"],
            "manual-group:sha256:" + "2" * 64,
        )

    def test_request_has_exact_worker_contract_and_stable_fingerprint(self):
        stage_input = canonical_fingerprint({"stage": "q"})
        project = canonical_fingerprint({"project": "a"})
        bag = {
            "bag_id": "bag-a",
            "path": "/tmp/a.bag",
            "sha256": "1" * 64,
            "episode_index": 0,
            "selected_interval_local_seconds": [1.0, 2.0],
            "configuration_fingerprint": "complete:" + "2" * 64,
        }
        settings = diagonal_q_stage_settings(self._settings())
        first = build_diagonal_q_stage_request(
            run_id="q-a",
            project_fingerprint=project,
            stage_input_fingerprint=stage_input,
            bags=[bag],
            settings=settings,
        )
        second = build_diagonal_q_stage_request(
            run_id="q-a",
            project_fingerprint=project,
            stage_input_fingerprint=stage_input,
            bags=[bag],
            settings=settings,
        )
        self.assertEqual(first, second)
        self.assertEqual(
            set(first),
            {
                "schema",
                "run_id",
                "project_fingerprint",
                "stage_id",
                "stage_input_fingerprint",
                "bags",
                "settings",
            },
        )
        self.assertEqual(canonical_fingerprint(first), canonical_fingerprint(second))

    def test_augmented_request_binds_the_exact_q_artifact(self):
        source = self._settings()
        source["delay_prior_standard_deviation"] = 0.015
        request = build_augmented_parameter_stage_request(
            run_id="parameters-a",
            project_fingerprint=canonical_fingerprint({"project": "a"}),
            stage_input_fingerprint=canonical_fingerprint(
                {"stage": "parameters"}
            ),
            upstream_diagonal_q_path="/tmp/project/runs/q-a/diagonal_q",
            upstream_diagonal_q_fingerprint=canonical_fingerprint(
                {"q": "artifact"}
            ),
            bags=[
                {
                    "bag_id": "bag-a",
                    "path": "/tmp/a.bag",
                    "sha256": "1" * 64,
                    "episode_index": 0,
                    "selected_interval_local_seconds": [1.0, 2.0],
                    "configuration_fingerprint": "complete:" + "2" * 64,
                }
            ],
            settings=augmented_parameter_stage_settings(source),
        )
        self.assertEqual(
            set(request),
            {
                "schema",
                "run_id",
                "project_fingerprint",
                "stage_id",
                "stage_input_fingerprint",
                "upstream_diagonal_q",
                "bags",
                "settings",
            },
        )
        self.assertTrue(
            request["upstream_diagonal_q"]["artifact_fingerprint"].startswith(
                "sha256:"
            )
        )


if __name__ == "__main__":
    unittest.main()
