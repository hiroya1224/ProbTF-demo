import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from grape_param_estim.artifacts import (
    EXPERIMENTAL,
    AnalysisBagRecord,
    AnalysisArtifactWriter,
    ArtifactProvenance,
    merge_analysis_bag,
)
from grape_param_estim.controller_replay import ControllerParameters, PidLimits
from grape_param_estim.counterfactual import (
    SUPPORTED,
    CounterfactualCandidate,
    CounterfactualResult,
    SupportDiagnostics,
    TrajectoryRollout,
    TubeEvaluation,
)
from grape_param_estim.episode import sha256_file, stable_hash


def _result():
    controller = ControllerParameters(
        p_gain=np.ones(6),
        i_gain=np.zeros(6),
        d_gain=np.full(6, 0.5),
        limits=PidLimits.unbounded(),
    )
    tube = TubeEvaluation(
        success=True,
        violations=(),
        diagnostic_exceedances=(),
        outside_duration_s=0.0,
        maximum_continuous_saturation_s=0.0,
        maximum_position_ratio=0.5,
        maximum_velocity_ratio=0.5,
    )
    zeros = np.zeros((3, 6))
    rollout = TrajectoryRollout(
        rollout_id=0,
        initial_sample_id=10,
        response_sample_id=20,
        noise_sample_id=30,
        weight=1.0,
        position=zeros,
        velocity=zeros,
        command=zeros,
        saturation=np.zeros_like(zeros, dtype=bool),
        tube=tube,
    )
    support = SupportDiagnostics(
        label=SUPPORTED,
        candidate_distance=0.1,
        state_action_distance_p95=0.2,
        importance_weight_ess=8.0,
        maximum_predictive_std=0.1,
        reasons=(),
    )
    content_hash = stable_hash({"fixture": "counterfactual-result"})
    return CounterfactualResult(
        candidate=CounterfactualCandidate("candidate-1", controller),
        success_probability=1.0,
        credible_lower=0.7,
        credible_upper=0.9,
        lower_credible_bound=0.7,
        support=support,
        violation_probability={},
        effective_rollout_sample_size=1.0,
        rollouts=(rollout,),
        run_id=content_hash[:20],
        provenance={
            "seed": 5,
            "counterfactual_content_hash": content_hash,
            "recommendation_threshold": 0.6,
            "exact_controller_gate_passed": False,
            "probability_calibration_gate_passed": False,
            "integrator_state_gate_passed": False,
            "dependence_handling": (
                "DEPENDENCE_APPROXIMATED_INDEPENDENT_PRODUCT"
            ),
            "workflow_status": "EXPERIMENTAL",
        },
        recommendation_threshold=0.6,
    )


class ArtifactTests(unittest.TestCase):
    def test_counterfactual_result_rejects_truthy_non_boolean_gates(self):
        result = _result()
        invalid_values = ("false", 0, 1, np.bool_(False), np.bool_(True))
        gate_fields = (
            "exact_controller_gate_passed",
            "probability_calibration_gate_passed",
            "integrator_state_gate_passed",
        )
        for field in gate_fields:
            for invalid in invalid_values:
                with self.subTest(field=field, invalid=repr(invalid)):
                    with self.assertRaisesRegex(TypeError, field):
                        replace(result, **{field: invalid})

    def test_counterfactual_result_rejects_status_provenance_mismatch(self):
        result = _result()
        with self.assertRaisesRegex(ValueError, "status fields"):
            replace(
                result,
                provenance=dict(
                    result.provenance,
                    workflow_status="MANUAL_REVIEW_REQUIRED",
                ),
            )
        with self.assertRaisesRegex(ValueError, "workflow_status"):
            replace(
                result,
                workflow_status="MANUAL_REVIEW_REQUIRED",
                provenance=dict(
                    result.provenance,
                    workflow_status="MANUAL_REVIEW_REQUIRED",
                ),
            )

    def test_online_prefix_provenance_rejects_post_cutoff_source_data(self):
        common = {
            "source_bag_sha256": ("a" * 64,),
            "normalized_dataset_sha256": ("b" * 64,),
            "source_topics": ("/imu",),
            "interval_start_s": 0.0,
            "config_sha256": stable_hash({}),
            "source_commit": "abc123",
            "model_version": "model/v1",
            "seed": 1,
            "analysis_mode": "online_prefix",
            "prefix_cutoff_s": 2.0,
        }
        with self.assertRaisesRegex(ValueError, "end at or before"):
            ArtifactProvenance(interval_end_s=2.1, **common)
        valid = ArtifactProvenance(interval_end_s=2.0, **common)
        self.assertEqual(valid.interval_end_s, valid.prefix_cutoff_s)

    def test_analysis_bag_merge_preserves_source_and_exact_record_order(self):
        try:
            import rosbag
            import rospy
            from std_msgs.msg import Float32
        except ImportError as error:
            self.skipTest("ROS 1 messages unavailable: {}".format(error))

        with tempfile.TemporaryDirectory(prefix="grape-analysis-bag-") as directory:
            source = Path(directory) / "source.bag"
            output = Path(directory) / "analysis.bag"
            with rosbag.Bag(str(source), "w") as bag:
                bag.write(
                    "/raw",
                    Float32(data=1.0),
                    t=rospy.Time.from_sec(1.0),
                )
                bag.write(
                    "/raw",
                    Float32(data=2.0),
                    t=rospy.Time.from_sec(2.0),
                )
            source_hash = sha256_file(source)
            metadata = merge_analysis_bag(
                source,
                output,
                (
                    AnalysisBagRecord(
                        "/analysis/grape/candidate",
                        Float32(data=1.5),
                        1_500_000_000,
                    ),
                    AnalysisBagRecord(
                        "/analysis/grape/candidate",
                        Float32(data=0.5),
                        500_000_000,
                    ),
                ),
                source_hash,
            )
            self.assertEqual(sha256_file(source), source_hash)
            self.assertTrue(metadata["source_bag_unchanged"])
            self.assertEqual(metadata["analysis_record_count"], 2)
            with rosbag.Bag(str(output), "r") as bag:
                records = [
                    (topic, stamp.to_nsec())
                    for topic, _, stamp in bag.read_messages()
                ]
            self.assertEqual(
                records,
                [
                    ("/analysis/grape/candidate", 500_000_000),
                    ("/raw", 1_000_000_000),
                    ("/analysis/grape/candidate", 1_500_000_000),
                    ("/raw", 2_000_000_000),
                ],
            )
            with self.assertRaises(FileExistsError):
                merge_analysis_bag(
                    source,
                    output,
                    (
                        AnalysisBagRecord(
                            "/analysis/grape/candidate",
                            Float32(data=0.5),
                            500_000_000,
                        ),
                    ),
                    source_hash,
                )

    def test_bundle_is_hashed_non_overwriting_and_explicitly_experimental(self):
        config = {"target_tube": {"position_m": 0.2}, "seed": 5}
        provenance = ArtifactProvenance(
            source_bag_sha256=("a" * 64,),
            normalized_dataset_sha256=("b" * 64,),
            source_topics=("/vicon/grape/odom", "/imu"),
            interval_start_s=1.0,
            interval_end_s=1.2,
            config_sha256=stable_hash(config),
            source_commit="abc123",
            model_version="effective_response/v1",
            seed=5,
            analysis_mode="retrospective",
        )
        with tempfile.TemporaryDirectory(prefix="grape-artifact-") as directory:
            writer = AnalysisArtifactWriter(directory)
            output = writer.write(
                [_result()],
                [1.0, 1.1, 1.2],
                provenance,
                config,
                recommendation_threshold=0.6,
                exact_controller_gate_passed=False,
                run_id="fixed-run",
            )
            self.assertEqual(output.name, "fixed-run")
            expected = {
                "artifact_manifest.json",
                "counterfactual_candidates.csv",
                "counterfactual_candidates.json",
                "provenance.json",
                "report.md",
                "trajectory_particles.npz",
            }
            self.assertEqual({item.name for item in output.iterdir()}, expected)
            candidate_payload = json.loads(
                (output / "counterfactual_candidates.json").read_text()
            )
            candidate = candidate_payload["candidates"][0]
            self.assertEqual(candidate["proposal_status"], EXPERIMENTAL)
            self.assertFalse(
                candidate["proposal_eligible_after_statistical_gates"]
            )
            self.assertTrue(candidate["manual_review_required"])
            self.assertIn(
                "MANUAL REVIEW REQUIRED", (output / "report.md").read_text()
            )
            with np.load(
                str(output / "trajectory_particles.npz"), allow_pickle=False
            ) as archive:
                self.assertEqual(archive["position"].shape, (1, 3, 6))
                np.testing.assert_array_equal(
                    archive["initial_sample_id"], [10]
                )
                np.testing.assert_array_equal(
                    archive["joint_sample_id"], [-1]
                )
            manifest = json.loads(
                (output / "artifact_manifest.json").read_text()
            )
            self.assertEqual(
                set(manifest["files"]), expected - {"artifact_manifest.json"}
            )
            self.assertTrue(
                all(
                    len(item["sha256"]) == 64
                    for item in manifest["files"].values()
                )
            )
            for invalid in ("false", 0, 1, np.bool_(False)):
                with self.subTest(invalid_exact_gate=repr(invalid)):
                    with self.assertRaisesRegex(
                        TypeError, "exact_controller_gate_passed"
                    ):
                        writer.write(
                            [_result()],
                            [1.0, 1.1, 1.2],
                            provenance,
                            config,
                            recommendation_threshold=0.6,
                            exact_controller_gate_passed=invalid,
                        )
            with self.assertRaises(FileExistsError):
                writer.write(
                    [_result()],
                    [1.0, 1.1, 1.2],
                    provenance,
                    config,
                    recommendation_threshold=0.6,
                    run_id="fixed-run",
                )
            bare_exact = writer.write(
                [_result()],
                [1.0, 1.1, 1.2],
                provenance,
                config,
                recommendation_threshold=0.6,
                exact_controller_gate_passed=True,
                run_id="bare-exact-flag",
            )
            bare_payload = json.loads(
                (bare_exact / "counterfactual_candidates.json").read_text()
            )["candidates"][0]
            self.assertEqual(bare_payload["proposal_status"], EXPERIMENTAL)
            self.assertFalse(
                bare_payload["proposal_eligible_after_statistical_gates"]
            )

    def test_config_hash_mismatch_is_rejected_before_writing(self):
        provenance = ArtifactProvenance(
            source_bag_sha256=("a" * 64,),
            normalized_dataset_sha256=("b" * 64,),
            source_topics=("/imu",),
            interval_start_s=0.0,
            interval_end_s=0.2,
            config_sha256=stable_hash({"expected": True}),
            source_commit="abc123",
            model_version="model/v1",
            seed=1,
            analysis_mode="retrospective",
        )
        with tempfile.TemporaryDirectory(prefix="grape-artifact-") as directory:
            with self.assertRaisesRegex(ValueError, "config content"):
                AnalysisArtifactWriter(directory).write(
                    [_result()],
                    [0.0, 0.1, 0.2],
                    provenance,
                    {"expected": False},
                    recommendation_threshold=0.6,
                )
            self.assertEqual(list(Path(directory).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
