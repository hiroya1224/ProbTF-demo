import copy
from hashlib import sha256
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

from grape_param_estim.selection import (
    DEFAULT_CANDIDATE,
    EXPERIMENTAL,
    OPTIONAL_CANDIDATE,
    PRUNE,
    SelectionObservation,
    episode_bootstrap_mean,
    load_selection_protocol,
    run_selection,
    write_selection_outputs,
)


REPOSITORY = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = (
    REPOSITORY
    / "ros/examples/grape-param-estim/config/selection_protocol.yaml"
)


def _hash(text):
    return sha256(text.encode("utf-8")).hexdigest()


def _observations(protocol, candidate_id, metric, values, extra_gates=()):
    gates = {
        item: True for item in protocol["evaluation"]["required_hard_gates"]
    }
    gates.update({item: True for item in extra_gates})
    result = []
    for index, (fold, value) in enumerate(zip(protocol["outer_folds"], values)):
        fold_id = fold["fold_id"]
        result.append(
            SelectionObservation(
                candidate_id=candidate_id,
                fold_id=fold_id,
                held_out_episode=fold["held_out_episode"],
                stratum=fold["stratum"],
                metrics={metric: value},
                hard_gates=gates,
                trajectory_sample_bundle_sha256=_hash("samples-" + fold_id),
                candidate_grid_sha256=_hash("grid-" + fold_id),
                random_stream_sha256=_hash("random-" + fold_id),
                run_sha256=_hash("{}-{}".format(candidate_id, fold_id)),
            )
        )
    return result


class SelectionTests(unittest.TestCase):
    def test_observation_rejects_non_boolean_hard_gate_values(self):
        protocol = load_selection_protocol(PROTOCOL_PATH)
        observation = _observations(
            protocol,
            "error_state_ekf_rts",
            "held_out_log_predictive_density",
            [1.0],
        )[0]
        for invalid in ("false", 0, 1, np.bool_(False)):
            with self.subTest(value=repr(invalid)):
                payload = observation.__dict__.copy()
                payload["hard_gates"] = dict(payload["hard_gates"])
                gate = next(iter(payload["hard_gates"]))
                payload["hard_gates"][gate] = invalid
                with self.assertRaisesRegex(
                    ValueError, "must be JSON booleans"
                ):
                    SelectionObservation.from_mapping(payload)

    def test_repository_protocol_has_explicit_leak_free_lobo_folds(self):
        protocol = load_selection_protocol(PROTOCOL_PATH)
        self.assertEqual(len(protocol["episodes"]), 12)
        self.assertEqual(len(protocol["outer_folds"]), 12)
        self.assertEqual(
            {item["held_out_episode"] for item in protocol["outer_folds"]},
            set(protocol["episodes"]),
        )
        manifest = yaml.safe_load(
            (
                REPOSITORY
                / "ros/examples/grape-param-estim/config/bag_manifest.yaml"
            ).read_text()
        )
        self.assertEqual(protocol["manifest_hash"], manifest["manifest_hash"])
        manifest_hashes = {
            item["episode_id"]: item["sha256"] for item in manifest["bags"]
        }
        self.assertEqual(
            {
                episode: item["bag_sha256"]
                for episode, item in protocol["episodes"].items()
            },
            manifest_hashes,
        )
        for fold in protocol["outer_folds"]:
            train = set(fold["train_episodes"])
            validation = set(fold["inner_validation"])
            held_out = {fold["held_out_episode"]}
            self.assertFalse(train & validation)
            self.assertFalse(train & held_out)
            self.assertFalse(validation & held_out)
            self.assertEqual(
                train | validation | held_out, set(protocol["episodes"])
            )

    def test_episode_bootstrap_is_reproducible_and_counts_bags(self):
        first = episode_bootstrap_mean([1.0, 2.0, 4.0], seed=9, draws=500)
        second = episode_bootstrap_mean([1.0, 2.0, 4.0], seed=9, draws=500)
        self.assertEqual(first, second)
        self.assertEqual(first["episode_count"], 3)
        self.assertLess(first["bootstrap_lower"], first["bootstrap_upper"])

    def test_one_standard_error_rule_selects_simpler_complete_candidate(self):
        protocol = load_selection_protocol(PROTOCOL_PATH)
        metric = protocol["candidate_groups"]["trajectory_smoother"][
            "primary_metric"
        ]
        simple = _observations(
            protocol,
            "error_state_ekf_rts",
            metric,
            [1.0] * 12,
        )
        complex_values = [0.92, 1.12] * 6
        complex_candidate = _observations(
            protocol,
            "factor_graph_imu_preintegration",
            metric,
            complex_values,
        )
        result = run_selection(
            protocol, simple + complex_candidate, source_commit="test"
        )
        self.assertEqual(
            result["candidates"]["error_state_ekf_rts"]["status"],
            DEFAULT_CANDIDATE,
        )
        self.assertEqual(
            result["candidates"]["factor_graph_imu_preintegration"]["status"],
            OPTIONAL_CANDIDATE,
        )
        self.assertEqual(
            result["groups"]["trajectory_smoother"]["selected_default"],
            "error_state_ekf_rts",
        )

    def test_failed_hard_gate_cannot_be_default(self):
        protocol = load_selection_protocol(PROTOCOL_PATH)
        metric = protocol["candidate_groups"]["trajectory_smoother"][
            "primary_metric"
        ]
        observations = _observations(
            protocol,
            "error_state_ekf_rts",
            metric,
            [10.0] * 12,
        )
        failed = list(observations)
        payload = failed[0].__dict__.copy()
        payload["hard_gates"] = dict(payload["hard_gates"])
        payload["hard_gates"]["online_prefix_no_future_data"] = False
        failed[0] = SelectionObservation(**payload)
        result = run_selection(protocol, failed, source_commit="test")
        candidate = result["candidates"]["error_state_ekf_rts"]
        self.assertEqual(candidate["status"], PRUNE)
        self.assertIn(
            "online_prefix_no_future_data", candidate["failed_hard_gates"]
        )

    def test_exact_oracle_absence_blocks_python_controller_promotion(self):
        protocol = load_selection_protocol(PROTOCOL_PATH)
        group = protocol["candidate_groups"]["controller_replay"]
        surrogate = next(
            item
            for item in group["candidates"]
            if item["candidate_id"] == "python_vector_pid_surrogate"
        )
        observations = _observations(
            protocol,
            "python_vector_pid_surrogate",
            group["primary_metric"],
            [0.005] * 12,
            surrogate["required_hard_gates"],
        )
        result = run_selection(protocol, observations, source_commit="test")
        candidate = result["candidates"]["python_vector_pid_surrogate"]
        self.assertEqual(candidate["status"], EXPERIMENTAL)
        self.assertIn(
            "blocking_candidate_exact_cpp_pc_mcu_replay_not_validated",
            candidate["reasons"],
        )
        self.assertIsNone(
            result["groups"]["controller_replay"]["selected_default"]
        )

    def test_selection_is_incomplete_when_competitors_are_unmeasured(self):
        protocol = load_selection_protocol(PROTOCOL_PATH)
        observations = []
        for group in protocol["candidate_groups"].values():
            candidate = group["candidates"][0]
            observations.extend(
                _observations(
                    protocol,
                    candidate["candidate_id"],
                    group["primary_metric"],
                    [0.1] * 12,
                    candidate.get("required_hard_gates", ()),
                )
            )
        result = run_selection(
            protocol, observations, source_commit="test"
        )
        self.assertTrue(
            all(
                group["selected_default"] is not None
                for group in result["groups"].values()
            )
        )
        self.assertFalse(result["selection_complete"])

    def test_candidates_must_share_fold_samples_grid_and_random_stream(self):
        protocol = load_selection_protocol(PROTOCOL_PATH)
        metric = protocol["candidate_groups"]["trajectory_smoother"][
            "primary_metric"
        ]
        first = _observations(
            protocol, "error_state_ekf_rts", metric, [1.0] * 12
        )
        second = _observations(
            protocol,
            "factor_graph_imu_preintegration",
            metric,
            [1.0] * 12,
        )
        payload = second[0].__dict__.copy()
        payload["random_stream_sha256"] = _hash("different")
        second[0] = SelectionObservation(**payload)
        with self.assertRaisesRegex(ValueError, "common samples/grid/random"):
            run_selection(protocol, first + second, source_commit="test")

    def test_different_candidate_groups_may_use_different_comparison_contracts(self):
        protocol = load_selection_protocol(PROTOCOL_PATH)
        smoother_metric = protocol["candidate_groups"]["trajectory_smoother"][
            "primary_metric"
        ]
        inference_metric = protocol["candidate_groups"]["inference"][
            "primary_metric"
        ]
        smoother = _observations(
            protocol, "error_state_ekf_rts", smoother_metric, [1.0] * 12
        )
        inference = _observations(
            protocol, "modular_tempered_smc", inference_metric, [2.0] * 12
        )
        changed = []
        for item in inference:
            payload = item.__dict__.copy()
            payload["trajectory_sample_bundle_sha256"] = _hash(
                "inference-samples-" + item.fold_id
            )
            payload["candidate_grid_sha256"] = _hash(
                "inference-grid-" + item.fold_id
            )
            payload["random_stream_sha256"] = _hash(
                "inference-random-" + item.fold_id
            )
            changed.append(SelectionObservation(**payload))
        result = run_selection(
            protocol, smoother + changed, source_commit="test"
        )
        self.assertEqual(
            result["candidates"]["error_state_ekf_rts"]["status"],
            DEFAULT_CANDIDATE,
        )
        self.assertEqual(
            result["candidates"]["modular_tempered_smc"]["status"],
            DEFAULT_CANDIDATE,
        )

    def test_empty_results_are_honestly_experimental_and_non_overwriting(self):
        protocol = load_selection_protocol(PROTOCOL_PATH)
        result = run_selection(protocol, (), source_commit="test")
        self.assertFalse(result["selection_complete"])
        self.assertTrue(
            all(
                item["status"] == EXPERIMENTAL
                for item in result["candidates"].values()
            )
        )
        with tempfile.TemporaryDirectory(prefix="grape-selection-") as directory:
            markdown = Path(directory) / "results.md"
            machine = Path(directory) / "results.json"
            write_selection_outputs(result, markdown, machine)
            self.assertIn("Submitted observations: `0`", markdown.read_text())
            self.assertEqual(
                json.loads(machine.read_text())["result_hash"],
                result["result_hash"],
            )
            with self.assertRaises(FileExistsError):
                write_selection_outputs(result, markdown, machine)


if __name__ == "__main__":
    unittest.main()
