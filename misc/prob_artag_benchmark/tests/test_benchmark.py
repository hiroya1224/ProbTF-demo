import csv
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest

import cv2
import numpy as np

from prob_artag_benchmark.aggregate import _compare_trees, aggregate_reports
from prob_artag_benchmark.cli import main as benchmark_main
from prob_artag_benchmark.metrics import (
    _minimum_cost_pairs,
    associate_by_id_and_geometry,
    corner_metrics,
)
from prob_artag_benchmark.models import BenchmarkConfig, SeedPose
from prob_artag_benchmark.runner import (
    _complete_mode_seed_mapping,
    _ippe_candidate_metrics,
    evaluate_dataset,
)


ROTATION = np.diag([1.0, -1.0, -1.0])
TRANSLATION = np.array([0.0, 0.0, 1.0])
CORNERS = np.array(
    [[270.0, 190.0], [370.0, 190.0], [370.0, 290.0], [270.0, 290.0]]
)


class FakeObservation:
    def __init__(self, marker_id=7, corners=CORNERS):
        self.marker_id = int(marker_id)
        self.corners_px = np.asarray(corners, dtype=float)
        self.family = "DICT_APRILTAG_36h11"


class FakeAdapter:
    def __init__(self, observations=(), fail_estimation=False, empty_seeds=False):
        self.observations = tuple(observations)
        self.fail_estimation = bool(fail_estimation)
        self.empty_seeds = bool(empty_seeds)

    def camera_from_metadata(self, camera):
        return SimpleNamespace(camera=camera)

    def detect(self, image):
        return self.observations

    def solve_candidates(self, observation, camera_model, tag_size_m):
        if self.empty_seeds:
            return ()
        tilted, _ = cv2.Rodrigues(np.array([0.03, -0.01, 0.0]))
        return (
            SeedPose(ROTATION, TRANSLATION, 0.0),
            SeedPose(ROTATION @ tilted, np.array([0.002, -0.001, 1.01]), 1.5),
        )

    def estimate(
        self,
        observation,
        camera_model,
        tag_size_m,
        parent_frame_id,
        child_frame_id,
        stamp,
        edge_id,
        authority,
        seeds=None,
    ):
        if self.fail_estimation:
            raise ValueError("synthetic degenerate Hessian")
        seeds = (
            self.solve_candidates(observation, camera_model, tag_size_m)
            if seeds is None
            else tuple(seeds)
        )
        diagnostics = SimpleNamespace(
            accepted_count=len(seeds),
            deduplicated_count=0,
            candidates=tuple(
                SimpleNamespace(
                    seed_index=index,
                    accepted=True,
                    reason="accepted",
                    initial_error=seed.reported_reprojection_error_px,
                    final_objective=float(index) + 0.25,
                    iterations=index + 1,
                )
                for index, seed in enumerate(seeds)
            ),
        )
        components = tuple(
            SimpleNamespace(
                provenance=SimpleNamespace(
                    detail="Mode initialized by IPPE candidate {}.".format(
                        len(seeds) - 1 - index
                    )
                )
            )
            for index in range(len(seeds))
        )
        record = SimpleNamespace(
            distribution=SimpleNamespace(components=components)
        )
        return SimpleNamespace(
            record=record,
            diagnostics=diagnostics,
            rotations=tuple(seed.rotation for seed in seeds),
            translations=tuple(seed.translation for seed in seeds),
            seed_indices=tuple(range(len(seeds))),
            log_masses=(-0.1, -1.0),
            weights=(0.7, 0.3),
        )


def write_frame(root, marker_id=7):
    frame = Path(root) / "frames" / "000000"
    frame.mkdir(parents=True)
    image = np.full((480, 640, 3), 245, dtype=np.uint8)
    cv2.polylines(
        image,
        [np.rint(CORNERS).astype(np.int32).reshape(-1, 1, 2)],
        True,
        (0, 0, 0),
        3,
    )
    assert cv2.imwrite(str(frame / "rgb.png"), image)
    transform = np.eye(4)
    transform[:3, :3] = ROTATION
    transform[:3, 3] = TRANSLATION
    metadata = {
        "schema_version": 1,
        "frame_id": 0,
        "scenario": "frontal",
        "seed": 11,
        "camera": {
            "width": 640,
            "height": 480,
            "camera_matrix": [[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]],
            "distortion": [0.0, 0.0, 0.0, 0.0, 0.0],
            "T_W_C": np.eye(4).tolist(),
        },
        "tags": [
            {
                "family": "DICT_APRILTAG_36h11",
                "id": marker_id,
                "instance_id": 1,
                "size_m": 0.2,
                "T_C_M": transform.tolist(),
                "corners_px": CORNERS.tolist(),
                "front_facing": True,
                "visible_fraction": 1.0,
                "projected_size_px": 100.0,
            }
        ],
    }
    with (frame / "metadata.json").open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream)


class MetricTest(unittest.TestCase):
    def test_corner_metric_is_zero_for_equal_ordered_corners(self):
        value = corner_metrics(CORNERS, CORNERS)
        self.assertEqual(value["mean_px"], 0.0)
        self.assertEqual(value["rmse_px"], 0.0)
        self.assertEqual(value["per_corner_px"], [0.0] * 4)

        moved = CORNERS.copy()
        moved[0, 0] += 5.0
        value = corner_metrics(CORNERS, moved)
        self.assertEqual(value["rmse_px"], 2.5)

    def test_wrong_id_requires_geometric_proximity(self):
        truth = (
            {
                "id": 7,
                "family": "DICT_APRILTAG_36h11",
                "corners_px": CORNERS,
            },
        )
        near = (FakeObservation(99, CORNERS + 1.0),)
        result = associate_by_id_and_geometry(truth, near, 5.0)
        self.assertEqual(len(result["pairs"]), 1)
        self.assertFalse(result["pairs"][0]["id_correct"])
        far = (FakeObservation(99, CORNERS + 100.0),)
        result = associate_by_id_and_geometry(truth, far, 5.0)
        self.assertEqual(result["pairs"], ())

        other_family = (FakeObservation(7, CORNERS),)
        other_family[0].family = "DICT_APRILTAG_16h5"
        result = associate_by_id_and_geometry(truth, other_family, 5.0)
        self.assertEqual(result["pairs"], ())

        competing = (
            FakeObservation(7, CORNERS + 100.0),
            FakeObservation(99, CORNERS + 1.0),
        )
        result = associate_by_id_and_geometry(truth, competing, 5.0)
        self.assertEqual(len(result["pairs"]), 1)
        self.assertFalse(result["pairs"][0]["id_correct"])
        self.assertEqual(result["pairs"][0]["detection_index"], 1)

    def test_global_assignment_beats_greedy_pair_selection(self):
        costs = ((1.0, 0, 0), (2.0, 0, 1), (2.0, 1, 0), (100.0, 1, 1))
        selected = _minimum_cost_pairs(costs, (0, 1), (0, 1))
        self.assertEqual(selected, ((2.0, 0, 1), (2.0, 1, 0)))

    def test_two_ippe_metrics_distinguish_coverage_and_threshold_boundary(self):
        config = BenchmarkConfig()

        def candidate(index, translation, rotation, reprojection):
            return {
                "candidate_index": index,
                "initial_pose_error": {
                    "translation_m": translation,
                    "rotation_deg": rotation,
                },
                "initial_reprojection_rmse_px": reprojection,
            }

        for candidates in (
            (),
            (candidate(0, 0.01, 1.0, 0.2),),
            (
                candidate(0, 0.01, 1.0, 0.2),
                candidate(1, 0.03, 7.0, 0.5),
                candidate(2, 0.04, 8.0, 0.8),
            ),
        ):
            metrics = _ippe_candidate_metrics(candidates, config)
            self.assertFalse(metrics["two_ippe_candidates_available"])
            self.assertIsNone(metrics["gt_near_ippe_candidate_exists"])
            self.assertIsNone(metrics["ippe_reprojection_rmse_gap_px"])

        metrics = _ippe_candidate_metrics(
            (
                candidate(0, 0.02, 5.0, 0.2),
                candidate(1, 0.04, 8.0, 0.7),
            ),
            config,
        )
        self.assertTrue(metrics["two_ippe_candidates_available"])
        self.assertTrue(metrics["gt_near_ippe_candidate_exists"])
        self.assertAlmostEqual(metrics["ippe_reprojection_rmse_gap_px"], 0.5)


class EndToEndTest(unittest.TestCase):
    def test_writes_deterministic_json_csv_and_overlay(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "first-dataset"
            regenerated_dataset = Path(directory) / "second-dataset"
            first_output = Path(directory) / "first"
            second_output = Path(directory) / "second"
            write_frame(dataset)
            write_frame(regenerated_dataset)
            adapter = FakeAdapter((FakeObservation(7),))
            first = evaluate_dataset(dataset, first_output, BenchmarkConfig(), adapter)
            second = evaluate_dataset(
                regenerated_dataset, second_output, BenchmarkConfig(), adapter
            )
            self.assertEqual(first["summary"]["recall"], 1.0)
            self.assertEqual(first["summary"]["correct_id_count"], 1)
            self.assertEqual(first["summary"]["candidate_status_counts"], {"accepted": 2})
            self.assertEqual(
                first["summary"]["nearest_mode_translation_error_m_mean"], 0.0
            )
            self.assertEqual(
                first["summary"]["nearest_mode_translation_error_m_max"], 0.0
            )
            self.assertEqual(
                first["summary"]["nearest_mode_rotation_error_deg_mean"], 0.0
            )
            self.assertEqual(first["summary"]["gt_near_ippe_candidate_count"], 1)
            self.assertEqual(first["summary"]["gt_near_ippe_candidate_rate"], 1.0)
            self.assertGreater(
                first["summary"]["ippe_reprojection_rmse_gap_px_mean"], 0.0
            )
            self.assertEqual(
                (first_output / "metrics.json").read_bytes(),
                (second_output / "metrics.json").read_bytes(),
            )
            self.assertEqual(
                (first_output / "overlays" / "000000.png").read_bytes(),
                (second_output / "overlays" / "000000.png").read_bytes(),
            )
            with (first_output / "candidates.csv").open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["accepted"], "True")
            self.assertAlmostEqual(float(rows[0]["translation_error_m"]), 0.0)

    def test_mode_seed_mapping_is_one_to_one_and_validates_result_shapes(self):
        adapter = FakeAdapter()
        seeds = adapter.solve_candidates(None, None, 0.12)
        components = tuple(
            SimpleNamespace(provenance=SimpleNamespace(detail="")) for _ in seeds
        )
        reverse_result = SimpleNamespace(
            record=SimpleNamespace(
                distribution=SimpleNamespace(components=components)
            ),
            rotations=tuple(seed.rotation for seed in reversed(seeds)),
            translations=tuple(seed.translation for seed in reversed(seeds)),
            seed_indices=(1, 0),
            weights=(0.6, 0.4),
            log_masses=(-0.2, -0.8),
        )
        self.assertEqual(
            _complete_mode_seed_mapping(reverse_result, seeds), (1, 0)
        )

        reverse_result.seed_indices = (0, 0)
        self.assertEqual(
            _complete_mode_seed_mapping(reverse_result, seeds), (1, 0)
        )

        malformed = SimpleNamespace(
            record=reverse_result.record,
            rotations=reverse_result.rotations,
            translations=reverse_result.translations[:1],
            seed_indices=(1, 0),
            weights=reverse_result.weights,
            log_masses=reverse_result.log_masses,
        )
        with self.assertRaisesRegex(ValueError, "inconsistent lengths"):
            _complete_mode_seed_mapping(malformed, seeds)

    def test_missing_detection_is_a_miss_not_an_exception(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "dataset"
            output = Path(directory) / "output"
            write_frame(dataset)
            report = evaluate_dataset(dataset, output, BenchmarkConfig(), FakeAdapter(()))
            self.assertEqual(report["summary"]["recall"], 0.0)
            self.assertEqual(report["summary"]["missed_count"], 1)
            self.assertTrue((output / "overlays" / "000000.png").is_file())

    def test_missing_image_failure_report_is_root_independent(self):
        with tempfile.TemporaryDirectory() as directory:
            first_dataset = Path(directory) / "first-dataset"
            second_dataset = Path(directory) / "second-dataset"
            write_frame(first_dataset)
            write_frame(second_dataset)
            (first_dataset / "frames" / "000000" / "rgb.png").unlink()
            (second_dataset / "frames" / "000000" / "rgb.png").unlink()
            first_output = Path(directory) / "first-output"
            second_output = Path(directory) / "second-output"
            evaluate_dataset(
                first_dataset, first_output, BenchmarkConfig(), FakeAdapter(())
            )
            evaluate_dataset(
                second_dataset, second_output, BenchmarkConfig(), FakeAdapter(())
            )
            self.assertEqual(
                (first_output / "metrics.json").read_bytes(),
                (second_output / "metrics.json").read_bytes(),
            )
            report = json.loads((first_output / "metrics.json").read_text())
            self.assertEqual(
                report["frames"][0]["error"],
                "cannot read frames/000000/rgb.png",
            )

    def test_aggregate_report_is_derived_from_scenario_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "frontal" / "dataset"
            output = root / "frontal" / "report"
            write_frame(dataset)
            evaluate_dataset(
                dataset,
                output,
                BenchmarkConfig(),
                FakeAdapter((FakeObservation(7),)),
            )
            comparison_roots = []
            for name in ("comparison-a", "comparison-b"):
                comparison_root = root / name
                shutil.copytree(dataset, comparison_root / "dataset")
                shutil.copytree(output, comparison_root / "report")
                comparison_roots.append(comparison_root)
            aggregate = aggregate_reports(
                root,
                scenarios=("frontal",),
                projected_size_threshold_px=50.0,
                comparison_roots=comparison_roots,
            )
            self.assertEqual(aggregate["fixture"]["seed"], 11)
            self.assertEqual(aggregate["scenarios"]["frontal"]["candidate_count"], 2)
            self.assertEqual(aggregate["derived"]["metrics"]["tag_count"], 1)
            self.assertTrue(aggregate["determinism"]["comparison_performed"])
            self.assertTrue(aggregate["determinism"]["byte_identical"])
            self.assertEqual(aggregate["determinism"]["scenario"], "frontal")

    def test_tree_comparison_computes_evidence_and_rejects_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first"
            second = Path(directory) / "second"
            first.mkdir()
            second.mkdir()
            (first / "value.bin").write_bytes(b"deterministic")
            (second / "value.bin").write_bytes(b"deterministic")
            evidence = _compare_trees((first, second))
            self.assertTrue(evidence["comparison_performed"])
            self.assertTrue(evidence["byte_identical"])
            self.assertEqual(evidence["file_count"], 1)
            self.assertEqual(len(evidence["tree_sha256"]), 64)

            (second / "value.bin").write_bytes(b"different")
            with self.assertRaisesRegex(ValueError, "not byte-identical"):
                _compare_trees((first, second))

    def test_degenerate_estimation_keeps_detection_and_status(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "dataset"
            output = Path(directory) / "output"
            write_frame(dataset)
            adapter = FakeAdapter((FakeObservation(7),), fail_estimation=True)
            report = evaluate_dataset(dataset, output, BenchmarkConfig(), adapter)
            detection = report["frames"][0]["detections"][0]
            self.assertEqual(detection["estimation_status"], "error")
            self.assertIn("degenerate Hessian", detection["estimation_error"])
            self.assertEqual(report["summary"]["recall"], 1.0)
            self.assertEqual(report["summary"]["error_frame_count"], 1)
            self.assertEqual(report["summary"]["estimation_error_count"], 1)

    def test_empty_and_malformed_datasets_are_not_successful(self):
        with tempfile.TemporaryDirectory() as directory:
            empty = Path(directory) / "empty"
            empty.mkdir()
            with self.assertRaisesRegex(ValueError, "no frame metadata"):
                evaluate_dataset(
                    empty,
                    Path(directory) / "output",
                    BenchmarkConfig(),
                    FakeAdapter(()),
                )

            frame = Path(directory) / "bad" / "frames" / "000000"
            frame.mkdir(parents=True)
            (frame / "metadata.json").write_text("{bad", encoding="utf-8")
            (frame.parent / "000001").mkdir()
            report = evaluate_dataset(
                Path(directory) / "bad",
                Path(directory) / "bad-output",
                BenchmarkConfig(),
                FakeAdapter(()),
            )
            self.assertEqual(report["summary"]["frame_count"], 2)
            self.assertEqual(report["summary"]["error_frame_count"], 2)
            self.assertIn("JSONDecodeError", report["frames"][0]["error"])
            self.assertIn("missing frames/000001", report["frames"][1]["error"])

            self.assertEqual(
                benchmark_main(
                    [str(empty), str(Path(directory) / "cli-output"), "--corner-sigma-px", "0"]
                ),
                2,
            )

    def test_wrong_id_is_reported_separately_from_recall(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "dataset"
            output = Path(directory) / "output"
            write_frame(dataset, marker_id=7)
            adapter = FakeAdapter((FakeObservation(99),))
            report = evaluate_dataset(dataset, output, BenchmarkConfig(), adapter)
            self.assertEqual(report["summary"]["recall"], 0.0)
            self.assertEqual(report["summary"]["geometric_recall"], 1.0)
            self.assertEqual(report["summary"]["wrong_id_count"], 1)


if __name__ == "__main__":
    unittest.main()
