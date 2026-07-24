import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import yaml

from probtf.temporal import (
    benchmark_callable,
    bootstrap_mean_confidence_interval,
    energy_distance_samples,
    environment_manifest,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = REPOSITORY_ROOT / "ros/core/probtf_core"
RUNNER_PATH = PACKAGE_ROOT / "test/run_temporal_selection.py"
SPEC = importlib.util.spec_from_file_location("run_temporal_selection", RUNNER_PATH)
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def test_benchmark_helpers_are_deterministic_and_validate_inputs():
    first = bootstrap_mean_confidence_interval(
        [1.0, 2.0, 4.0],
        resamples=200,
        seed=17,
    )
    second = bootstrap_mean_confidence_interval(
        [1.0, 2.0, 4.0],
        resamples=200,
        seed=17,
    )
    assert first == second
    assert first[1] <= first[0] <= first[2]
    assert energy_distance_samples(np.zeros((4, 2)), np.zeros((3, 2))) == 0.0
    assert energy_distance_samples(np.zeros((4, 2)), np.ones((3, 2))) > 0.0
    with pytest.raises(ValueError):
        benchmark_callable(lambda: None, repetitions=0)
    manifest = environment_manifest([3, 5], packages=("numpy",))
    assert manifest["random_seeds"] == [3, 5]
    assert "numpy" in manifest["packages"]
    assert manifest["cpu_count"] >= 1


def test_frozen_selection_config_hash_and_split_cover_the_corpus():
    config_path = PACKAGE_ROOT / "test/temporal_selection.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    corpus_path = config_path.parent / config["corpus"]["path"]
    assert hashlib.sha256(corpus_path.read_bytes()).hexdigest() == config["corpus"][
        "sha256"
    ]
    corpus = yaml.safe_load(corpus_path.read_text(encoding="utf-8"))
    corpus_ids = {episode["id"] for episode in corpus["episodes"]}
    training = set(config["corpus"]["training_episode_ids"])
    held_out = set(config["corpus"]["held_out_episode_ids"])
    assert not training.intersection(held_out)
    assert training | held_out == corpus_ids
    assert config["comparison_commit"] == "dee44b6"
    assert config["selection_rules"]["primary_metric"] == "episode_pose_rmse"
    assert config["selection_rules"]["motion_resampling_unit"] == "episode"
    assert (
        config["selection_rules"]["uncertainty_resampling_unit"]
        == "distribution_case_seed"
    )


def test_selection_source_hash_excludes_generated_result_artifacts(
    tmp_path,
    monkeypatch,
):
    repository = tmp_path / "repository"
    package = repository / "ros/core/probtf_core"
    package_source = package / "src"
    package_test = package / "test"
    shared_tests = repository / "tests/probtf"
    for directory in (package_source, package_test, shared_tests):
        directory.mkdir(parents=True, exist_ok=True)
    (package_source / "model.py").write_text("MODEL = 1\n", encoding="utf-8")
    (package_test / "runner.py").write_text("RUNNER = 1\n", encoding="utf-8")
    (shared_tests / "test_model.py").write_text(
        "def test_model(): pass\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(RUNNER, "REPOSITORY_ROOT", repository)
    monkeypatch.setattr(RUNNER, "PACKAGE_ROOT", package)
    monkeypatch.setattr(RUNNER, "HERE", package_test)

    baseline = RUNNER._evaluated_core_source_hash()
    generated = package_test / "temporal_selection_results_repeat.json"
    generated.write_text('{"run": 1}\n', encoding="utf-8")
    assert RUNNER._evaluated_core_source_hash() == baseline

    arbitrary_output = package_test / "custom_output.json"
    arbitrary_output.write_text('{"run": 1}\n', encoding="utf-8")
    assert RUNNER._evaluated_core_source_hash((arbitrary_output,)) == baseline
    assert RUNNER._evaluated_core_source_hash() != baseline


def test_log_predictive_density_contains_one_log_determinant_term():
    error = np.array([0.2, -0.1, 0.3, 0.0, 0.1, -0.2])
    spectral_density = np.diag([0.5, 0.8, 1.1, 0.4, 0.7, 0.9])
    horizon = 0.25
    covariance = spectral_density * horizon + np.eye(6) * 1.0e-6
    _, logdet = np.linalg.slogdet(covariance)
    expected = -0.5 * (
        error @ np.linalg.solve(covariance, error)
        + logdet
        + 6.0 * np.log(2.0 * np.pi)
    )
    assert RUNNER._log_predictive_density(
        error,
        spectral_density,
        horizon,
    ) == pytest.approx(expected)


def test_motion_prediction_uses_the_fixture_anchor_interval_not_one_second():
    initial_twist = np.array([0.8, -0.2, 0.15, 0.25, -0.1, 0.45])
    source_duration = 0.1
    previous = RUNNER.DeterministicTransform.identity()
    anchor = RUNNER.compose_transforms(
        previous,
        RUNNER.integrate_linear_body_twist(
            initial_twist,
            np.zeros(6),
            source_duration,
        ),
    )
    model = RUNNER.ConstantBodyTwistModel(
        np.zeros((6, 6)),
        0.5,
        model_id="interval_regression",
    )
    result = RUNNER._predict(
        model,
        previous,
        anchor,
        0.2,
        7,
        source_duration=source_duration,
    )
    truth = RUNNER.compose_transforms(
        anchor,
        RUNNER.se3_exp(initial_twist * 0.2),
    )
    actual = result.record.distribution.deterministic_transform()
    np.testing.assert_allclose(
        RUNNER.se3_log(RUNNER.relative_transform(actual, truth)),
        np.zeros(6),
        atol=1.0e-10,
    )


def test_qd_adapter_is_measured_at_two_sampling_rates():
    result = RUNNER.evaluate_qd_sampling_rate_equivalence()
    assert result["sample_periods"] == [0.05, 0.1]
    assert result["diagnostics"] == [
        "discrete_process_noise_adapted",
        "discrete_process_noise_adapted",
    ]
    assert result["passed"]


def test_empirical_sample_coverage_uses_a_held_out_evaluation_set():
    generator = np.random.default_rng(123)
    candidate = generator.normal(size=(256, 6))
    evaluation = generator.normal(size=(4096, 6))
    mean, covariance = RUNNER._empirical_moments(candidate)
    successes, total, threshold = RUNNER._empirical_sample_coverage(
        candidate,
        evaluation,
        mean,
        covariance,
    )
    assert total == 4096
    assert threshold > 0.0
    assert successes / total == pytest.approx(0.95, abs=0.03)


def test_one_standard_error_rule_uses_episodes_not_horizons():
    rows = []
    episode_improvements = {}
    for episode_id, twist_rmse, acceleration_rmse in (
        ("held_out_a", 1.0, 0.7),
        ("held_out_b", 0.8, 0.9),
    ):
        for horizon in (0.1, 0.2, 0.3, 0.4):
            rows.append(
                {
                    "episode_id": episode_id,
                    "split": "held_out",
                    "horizon": horizon,
                    "constant_twist": {"se3_error_norm": twist_rmse},
                    "constant_acceleration": {
                        "se3_error_norm": acceleration_rmse
                    },
                }
            )
        episode_improvements[episode_id] = {
            "constant_twist_rmse": twist_rmse,
            "constant_acceleration_rmse": acceleration_rmse,
        }
    result = RUNNER._one_standard_error_model_rule(
        {
            "rows": rows,
            "episode_improvements": episode_improvements,
        },
        {
            "selection_rules": {
                "primary_metric": "episode_pose_rmse",
                "motion_resampling_unit": "episode",
            }
        },
    )
    assert result["selection_unit"] == "episode"
    assert result["held_out_episode_ids"] == ["held_out_a", "held_out_b"]
    for metrics in result["models"].values():
        assert metrics["selection_unit_count"] == 2
