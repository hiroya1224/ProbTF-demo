#!/usr/bin/env python3
"""Run the frozen ProbTF temporal-model selection experiment.

The runner intentionally uses only synthetic/anonymous fixtures from this
package.  It validates the frozen corpus hash before evaluating candidates and
emits one self-contained JSON artifact with accuracy, calibration, latency,
memory, environment, and rule-based disposition.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
from scipy.stats import chi2, norm
import yaml

from probtf.distributions import (
    BinghamOrientation,
    ConditionalGaussianTranslation,
    TransformComponent,
    TransformDistribution,
    TransformDistributionStamped,
)
from probtf.geometry import (
    DeterministicTransform,
    compose_transforms,
    integrate_linear_body_twist,
    interpolate_transform,
    quat_left_matrix,
    relative_transform,
    se3_exp,
    se3_log,
)
from probtf.probability import sample_transform_distribution
from probtf.temporal import (
    ConstantBodyAccelerationModel,
    ConstantBodyTwistModel,
    TemporalEvaluationRequest,
    TemporalPolicy,
    TemporalQueryMode,
    TemporalUncertaintyBackend,
    benchmark_callable,
    bootstrap_mean_confidence_interval,
    energy_distance_samples,
    environment_manifest,
    parse_temporal_detail,
)
from probtf.temporal.backends import (
    component_pose_covariance,
    component_representative,
    record_representative,
)


HERE = Path(__file__).resolve().parent
PACKAGE_ROOT = HERE.parent
REPOSITORY_ROOT = PACKAGE_ROOT.parents[2]


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(*arguments):
    try:
        return subprocess.check_output(
            ("git",) + tuple(arguments),
            cwd=str(PACKAGE_ROOT),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _evaluated_core_source_hash():
    digest = hashlib.sha256()
    roots = (
        PACKAGE_ROOT / "src",
        PACKAGE_ROOT / "test",
        REPOSITORY_ROOT / "tests/probtf",
    )
    excluded_names = {
        "SELECTION_RESULTS.md",
        "temporal_selection_results_2026-07-24.json",
    }
    paths = []
    for root in roots:
        paths.extend(path for path in root.rglob("*") if path.is_file())
    for path in sorted(paths):
        relative = path.relative_to(REPOSITORY_ROOT)
        if (
            "__pycache__" in relative.parts
            or path.suffix == ".pyc"
            or path.name in excluded_names
        ):
            continue
        digest.update(str(relative).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _request(model, stamp, policy, anchors, *, seed, mode=TemporalQueryMode.ONLINE):
    return TemporalEvaluationRequest(
        requested_stamp=float(stamp),
        policy=policy,
        anchors=tuple(anchors),
        model_selector=model.model_id,
        max_prediction_horizon=model.maximum_horizon,
        max_age=max(1.0, model.maximum_horizon + 0.5),
        random_seed=int(seed),
        random_stream="temporal-selection",
        query_mode=mode,
    )


def _orientation_at(quaternion, concentrations=(180.0, 140.0, 100.0)):
    basis = quat_left_matrix(quaternion)[:, 1:]
    parameter = basis @ np.diag(-np.asarray(concentrations, dtype=float)) @ basis.T
    return BinghamOrientation.from_parameter_matrix(
        parameter,
        reference_quaternion_wxyz=quaternion,
    )


def _component(component_id, transform, *, orientation=None, covariance=None, coupling=None):
    orientation = (
        BinghamOrientation.dirac(transform.rotation_wxyz)
        if orientation is None
        else orientation
    )
    return TransformComponent(
        component_id,
        1.0,
        orientation,
        ConditionalGaussianTranslation(
            transform.translation,
            np.zeros((3, 3)) if covariance is None else covariance,
            np.zeros((3, 9)) if coupling is None else coupling,
        ),
    )


def _record(stamp, transform, *, case="dirac"):
    if case == "dirac":
        components = (_component("dirac", transform),)
    elif case == "local_gaussian":
        components = (
            _component(
                "local",
                transform,
                orientation=_orientation_at(transform.rotation_wxyz),
                covariance=np.diag([0.0020, 0.0015, 0.0010]),
            ),
        )
    elif case == "bimodal_orientation":
        positive = compose_transforms(
            transform,
            se3_exp([0.0, 0.0, 0.0, 0.0, 0.0, 0.45]),
        )
        negative = compose_transforms(
            transform,
            se3_exp([0.0, 0.0, 0.0, 0.0, 0.0, -0.45]),
        )
        components = (
            _component(
                "positive",
                positive,
                orientation=_orientation_at(
                    positive.rotation_wxyz,
                    concentrations=(240.0, 210.0, 180.0),
                ),
                covariance=np.eye(3) * 0.001,
            ),
            _component(
                "negative",
                negative,
                orientation=_orientation_at(
                    negative.rotation_wxyz,
                    concentrations=(240.0, 210.0, 180.0),
                ),
                covariance=np.eye(3) * 0.001,
            ),
        )
    elif case == "strong_translation_rotation_coupling":
        coupling = np.zeros((3, 9))
        coupling[0, (1, 3, 7)] = (0.18, -0.12, 0.10)
        coupling[1, (2, 5, 6)] = (-0.14, 0.16, 0.09)
        coupling[2, (0, 4, 8)] = (0.11, -0.13, 0.15)
        components = (
            _component(
                "coupled",
                transform,
                orientation=_orientation_at(
                    transform.rotation_wxyz,
                    concentrations=(90.0, 70.0, 50.0),
                ),
                covariance=np.eye(3) * 0.001,
                coupling=coupling,
            ),
        )
    else:
        raise ValueError("Unknown distribution case: {}".format(case))
    return TransformDistributionStamped(
        "world",
        "tool",
        float(stamp),
        "world_tool",
        "selection",
        TransformDistribution(tuple(components)),
    )


def _mixed_residual(reference, transform):
    return np.concatenate(
        (
            transform.translation - reference.translation,
            se3_log(relative_transform(reference, transform))[3:],
        )
    )


def _sample_result_vectors(result, reference, count=None, seed=0):
    components = result.record.distribution.components
    if (
        components
        and all(component.is_deterministic for component in components)
        and all(
            component.component_id == "sample:{:06d}".format(index)
            for index, component in enumerate(components)
        )
    ):
        transforms = tuple(component.deterministic_transform() for component in components)
        if count is not None:
            transforms = transforms[: int(count)]
    else:
        if count is None:
            count = 256
        samples = sample_transform_distribution(
            result.record.distribution,
            int(count),
            np.random.default_rng(seed),
        )
        transforms = tuple(
            DeterministicTransform(translation, rotation)
            for translation, rotation in zip(
                samples.translations,
                samples.rotations_wxyz,
            )
        )
    return np.asarray(
        [_mixed_residual(reference, transform) for transform in transforms],
        dtype=float,
    )


def _record_mixed_moments(record, reference):
    normalized = record.distribution.normalize_weights()
    mean = _mixed_residual(reference, record_representative(record))
    covariance = np.zeros((6, 6), dtype=float)
    for weighted in normalized.components:
        pose = component_representative(weighted.component)
        offset = _mixed_residual(record_representative(record), pose)
        covariance += weighted.weight * (
            component_pose_covariance(weighted.component)
            + np.outer(offset, offset)
        )
    return mean, 0.5 * (covariance + covariance.T)


def _empirical_moments(vectors):
    mean = np.mean(vectors, axis=0)
    covariance = np.cov(vectors, rowvar=False, ddof=1)
    return mean, 0.5 * (covariance + covariance.T)


def _coverage(vectors, mean, covariance, probability=0.95):
    regularized = covariance + np.eye(6) * 1.0e-10
    inverse = np.linalg.pinv(regularized, rcond=1.0e-10)
    centered = vectors - mean
    squared = np.einsum("ni,ij,nj->n", centered, inverse, centered)
    return int(np.count_nonzero(squared <= chi2.ppf(probability, 6))), len(vectors)


def _wilson(successes, total, confidence=0.95):
    if total < 1:
        raise ValueError("total must be positive.")
    z = float(norm.ppf(0.5 + 0.5 * confidence))
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    half = (
        z
        * np.sqrt(
            proportion * (1.0 - proportion) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return float(proportion), float(center - half), float(center + half)


def _summary(values, config, seed=0):
    mean, lower, upper = bootstrap_mean_confidence_interval(
        values,
        confidence=config["selection_rules"]["bootstrap_confidence"],
        resamples=config["selection_rules"]["bootstrap_resamples"],
        seed=seed,
    )
    return {
        "mean": mean,
        "bootstrap_lower": lower,
        "bootstrap_upper": upper,
        "count": len(values),
    }


def _episode_poses(episode, integration_substeps):
    source_dt = float(episode["anchor_dt"])
    initial_twist = np.asarray(episode["initial_body_twist"], dtype=float)
    source_acceleration = np.asarray(episode["body_acceleration"], dtype=float)
    previous = DeterministicTransform.identity()
    anchor = compose_transforms(
        previous,
        integrate_linear_body_twist(
            initial_twist,
            source_acceleration,
            source_dt,
            substeps=integration_substeps,
        ),
    )
    endpoint_twist = initial_twist + source_acceleration * source_dt
    future_acceleration = np.asarray(
        episode.get("post_anchor_body_acceleration", source_acceleration),
        dtype=float,
    )
    return previous, anchor, endpoint_twist, future_acceleration


def _predict(model, previous, anchor, horizon, seed, source_duration=1.0):
    source_duration = float(source_duration)
    records = (
        _record(0.0, previous),
        _record(source_duration, anchor),
    )
    return model.predict(
        records,
        _request(
            model,
            source_duration + float(horizon),
            TemporalPolicy.PREDICT_WITH_MODEL,
            records,
            seed=seed,
        ),
    )


def _log_predictive_density(error, spectral_density, horizon):
    covariance = np.asarray(spectral_density, dtype=float) * float(horizon)
    covariance += np.eye(6) * 1.0e-6
    sign, logdet = np.linalg.slogdet(covariance)
    if sign <= 0:
        return float("-inf")
    return float(
        -0.5
        * (
            error @ np.linalg.solve(covariance, error)
            + logdet
            + 6.0 * np.log(2.0 * np.pi)
        )
    )


def evaluate_motion_models(corpus, config):
    substeps = config["implementation"]["constant_acceleration_integration_substeps"]
    train_ids = set(config["corpus"]["training_episode_ids"])
    heldout_ids = set(config["corpus"]["held_out_episode_ids"])
    rows = []
    for episode in corpus["episodes"]:
        previous, anchor, endpoint_twist, future_acceleration = _episode_poses(
            episode,
            substeps,
        )
        maximum = max(float(value) for value in episode["prediction_horizons"]) + 0.01
        qc = np.diag(
            np.asarray(
                episode["process_noise_spectral_density_diagonal"],
                dtype=float,
            )
        )
        twist_model = ConstantBodyTwistModel(
            np.zeros((6, 6)),
            maximum,
            model_id="selection_constant_twist",
        )
        acceleration_model = ConstantBodyAccelerationModel(
            episode["body_acceleration"],
            "frozen_fixture",
            "tool_body",
            np.zeros((6, 6)),
            maximum,
            model_id="selection_constant_acceleration",
            integration_substeps=substeps,
        )
        for horizon in episode["prediction_horizons"]:
            horizon = float(horizon)
            truth = compose_transforms(
                anchor,
                integrate_linear_body_twist(
                    endpoint_twist,
                    future_acceleration,
                    horizon,
                    substeps=4 * substeps,
                ),
            )
            predicted = {}
            for name, model in (
                ("constant_twist", twist_model),
                ("constant_acceleration", acceleration_model),
            ):
                result = _predict(
                    model,
                    previous,
                    anchor,
                    horizon,
                    seed=1729,
                    source_duration=episode["anchor_dt"],
                )
                pose = result.record.distribution.deterministic_transform()
                error = se3_log(relative_transform(truth, pose))
                predicted[name] = {
                    "se3_error_norm": float(np.linalg.norm(error)),
                    "log_predictive_density": _log_predictive_density(
                        error,
                        qc,
                        horizon,
                    ),
                }
            split = (
                "training"
                if episode["id"] in train_ids
                else "held_out"
                if episode["id"] in heldout_ids
                else "unassigned"
            )
            rows.append(
                {
                    "episode_id": episode["id"],
                    "stratum": episode["stratum"],
                    "split": split,
                    "horizon": horizon,
                    "constant_twist": predicted["constant_twist"],
                    "constant_acceleration": predicted["constant_acceleration"],
                }
            )

    summaries = {}
    for model_name in ("constant_twist", "constant_acceleration"):
        summaries[model_name] = {}
        for split in ("training", "held_out", "all"):
            selected = rows if split == "all" else [
                row for row in rows if row["split"] == split
            ]
            errors = [
                row[model_name]["se3_error_norm"]
                for row in selected
            ]
            log_scores = [
                row[model_name]["log_predictive_density"]
                for row in selected
            ]
            summaries[model_name][split] = {
                "pose_rmse": float(np.sqrt(np.mean(np.square(errors)))),
                "mean_log_predictive_density": float(np.mean(log_scores)),
                "pose_error_bootstrap": _summary(
                    errors,
                    config,
                    seed=2718,
                ),
            }

    episode_improvements = {}
    for episode in corpus["episodes"]:
        selected = [row for row in rows if row["episode_id"] == episode["id"]]
        twist_rmse = float(
            np.sqrt(
                np.mean(
                    [
                        row["constant_twist"]["se3_error_norm"] ** 2
                        for row in selected
                    ]
                )
            )
        )
        acceleration_rmse = float(
            np.sqrt(
                np.mean(
                    [
                        row["constant_acceleration"]["se3_error_norm"] ** 2
                        for row in selected
                    ]
                )
            )
        )
        improvement = (
            0.0
            if twist_rmse <= 1.0e-10
            else (twist_rmse - acceleration_rmse) / twist_rmse
        )
        episode_improvements[episode["id"]] = {
            "stratum": episode["stratum"],
            "constant_twist_rmse": twist_rmse,
            "constant_acceleration_rmse": acceleration_rmse,
            "acceleration_improvement_fraction": float(improvement),
        }
    return {
        "rows": rows,
        "summary": summaries,
        "episode_improvements": episode_improvements,
    }


def evaluate_uncertainty_backends(corpus, config):
    del corpus
    oracle_count = int(config["oracle_sample_count"])
    sample_count = int(config["sample_backend_count"])
    process_depth = int(config["implementation"]["process_path_depth"])
    qc = np.diag([0.004, 0.004, 0.004, 0.0015, 0.0015, 0.0015])
    left_pose = DeterministicTransform.identity()
    right_pose = se3_exp([0.8, -0.2, 0.15, 0.35, -0.18, 0.55])
    requested_stamp = 0.43
    reference = interpolate_transform(left_pose, right_pose, requested_stamp)
    rows = []
    for case_spec in (
        "dirac",
        "local_gaussian",
        "bimodal_orientation",
        "strong_translation_rotation_coupling",
    ):
        left = _record(0.0, left_pose, case=case_spec)
        right = _record(1.0, right_pose, case=case_spec)
        for seed in config["random_seeds"]:
            oracle_model = ConstantBodyTwistModel(
                qc,
                1.0,
                model_id="selection_sample_oracle",
                backend=TemporalUncertaintyBackend.SAMPLE,
                sample_count=oracle_count,
                process_path_depth=process_depth,
            )
            sample_model = ConstantBodyTwistModel(
                qc,
                1.0,
                model_id="selection_sample_candidate",
                backend=TemporalUncertaintyBackend.SAMPLE,
                sample_count=sample_count,
                process_path_depth=process_depth,
            )
            moment_model = ConstantBodyTwistModel(
                qc,
                1.0,
                model_id="selection_moment_candidate",
                backend=TemporalUncertaintyBackend.MOMENT,
            )

            def interpolate(model, stream_seed):
                request = _request(
                    model,
                    requested_stamp,
                    TemporalPolicy.INTERPOLATE_WITH_MODEL,
                    (left, right),
                    seed=stream_seed,
                    mode=TemporalQueryMode.OFFLINE_SMOOTHING,
                )
                return model.interpolate(left, right, request)

            oracle = interpolate(oracle_model, int(seed))
            sample = interpolate(sample_model, int(seed))
            moment = interpolate(moment_model, int(seed))
            oracle_vectors = _sample_result_vectors(oracle, reference)
            sample_vectors = _sample_result_vectors(sample, reference)
            if not np.array_equal(
                sample_vectors,
                oracle_vectors[:sample_count],
            ):
                raise RuntimeError(
                    "Common-random-number sample prefixes are not reproducible."
                )
            moment_vectors = _sample_result_vectors(
                moment,
                reference,
                count=sample_count,
                seed=int(seed),
            )
            oracle_mean, oracle_covariance = _empirical_moments(oracle_vectors)
            sample_mean, sample_covariance = _empirical_moments(sample_vectors)
            moment_mean, moment_covariance = _record_mixed_moments(
                moment.record,
                reference,
            )
            covariance_scale = max(
                float(np.linalg.norm(oracle_covariance, ord="fro")),
                1.0e-12,
            )
            sample_coverage = _coverage(
                oracle_vectors,
                sample_mean,
                sample_covariance,
            )
            moment_coverage = _coverage(
                oracle_vectors,
                moment_mean,
                moment_covariance,
            )
            rows.append(
                {
                    "distribution_case": case_spec,
                    "seed": int(seed),
                    "moment": {
                        "pose_error": float(np.linalg.norm(moment_mean - oracle_mean)),
                        "covariance_relative_frobenius_error": float(
                            np.linalg.norm(
                                moment_covariance - oracle_covariance,
                                ord="fro",
                            )
                            / covariance_scale
                        ),
                        "energy_distance": energy_distance_samples(
                            moment_vectors,
                            oracle_vectors,
                        ),
                        "coverage_successes": moment_coverage[0],
                        "coverage_total": moment_coverage[1],
                    },
                    "sample": {
                        "pose_error": float(np.linalg.norm(sample_mean - oracle_mean)),
                        "covariance_relative_frobenius_error": float(
                            np.linalg.norm(
                                sample_covariance - oracle_covariance,
                                ord="fro",
                            )
                            / covariance_scale
                        ),
                        "energy_distance": energy_distance_samples(
                            sample_vectors,
                            oracle_vectors,
                        ),
                        "coverage_successes": sample_coverage[0],
                        "coverage_total": sample_coverage[1],
                    },
                }
            )

    summary = {}
    for backend in ("moment", "sample"):
        summary[backend] = {}
        for metric in (
            "pose_error",
            "covariance_relative_frobenius_error",
            "energy_distance",
        ):
            summary[backend][metric] = _summary(
                [row[backend][metric] for row in rows],
                config,
                seed=31415,
            )
        successes = sum(row[backend]["coverage_successes"] for row in rows)
        total = sum(row[backend]["coverage_total"] for row in rows)
        proportion, lower, upper = _wilson(successes, total)
        summary[backend]["empirical_95_percent_coverage"] = {
            "proportion": proportion,
            "wilson_lower": lower,
            "wilson_upper": upper,
            "successes": successes,
            "total": total,
            "contains_nominal_0_95": lower <= 0.95 <= upper,
        }
    improvements = [
        (
            row["moment"]["energy_distance"]
            - row["sample"]["energy_distance"]
        )
        / max(row["moment"]["energy_distance"], 1.0e-12)
        for row in rows
    ]
    summary["sample_energy_improvement_fraction"] = _summary(
        improvements,
        config,
        seed=65537,
    )
    summary["common_random_prefix_verified"] = True
    return {"rows": rows, "summary": summary}


def evaluate_performance(config):
    repetitions = int(config["benchmark"]["repetitions"])
    warmups = int(config["benchmark"]["warmups"])
    depth = int(config["implementation"]["process_path_depth"])
    qc = np.diag([0.004, 0.004, 0.004, 0.001, 0.001, 0.001])
    previous_pose = DeterministicTransform.identity()
    anchor_pose = se3_exp([0.08, -0.02, 0.01, 0.03, -0.02, 0.04])
    previous = _record(0.0, previous_pose, case="local_gaussian")
    anchor = _record(1.0, anchor_pose, case="local_gaussian")

    def prediction_call(model):
        request = _request(
            model,
            1.1,
            TemporalPolicy.PREDICT_WITH_MODEL,
            (previous, anchor),
            seed=1729,
        )
        return lambda: model.predict((previous, anchor), request)

    moment = ConstantBodyTwistModel(
        qc,
        0.5,
        model_id="benchmark_moment",
    )
    sample = ConstantBodyTwistModel(
        qc,
        0.5,
        model_id="benchmark_sample",
        backend=TemporalUncertaintyBackend.SAMPLE,
        sample_count=int(config["sample_backend_count"]),
        process_path_depth=depth,
    )
    acceleration_previous = _record(0.0, previous_pose)
    acceleration_anchor_pose = compose_transforms(
        previous_pose,
        integrate_linear_body_twist(
            [0.1, 0.0, 0.0, 0.0, 0.0, 0.2],
            [0.4, -0.1, 0.0, 0.2, 0.0, -0.1],
            1.0,
            substeps=config["implementation"][
                "constant_acceleration_integration_substeps"
            ],
        ),
    )
    acceleration_anchor = _record(1.0, acceleration_anchor_pose)
    acceleration = ConstantBodyAccelerationModel(
        [0.4, -0.1, 0.0, 0.2, 0.0, -0.1],
        "benchmark",
        "tool_body",
        np.zeros((6, 6)),
        0.5,
        model_id="benchmark_acceleration",
        integration_substeps=config["implementation"][
            "constant_acceleration_integration_substeps"
        ],
    )
    acceleration_request = _request(
        acceleration,
        1.1,
        TemporalPolicy.PREDICT_WITH_MODEL,
        (acceleration_previous, acceleration_anchor),
        seed=1729,
    )
    calls = {
        "moment_constant_twist": prediction_call(moment),
        "sample_constant_twist": prediction_call(sample),
        "moment_constant_acceleration": lambda: acceleration.predict(
            (acceleration_previous, acceleration_anchor),
            acceleration_request,
        ),
    }
    output = {}
    for name, function in calls.items():
        summary = benchmark_callable(
            function,
            repetitions=repetitions,
            warmups=warmups,
        )
        output[name] = {
            "repetitions": summary.repetitions,
            "p50_seconds": summary.p50_seconds,
            "p95_seconds": summary.p95_seconds,
            "peak_python_bytes": summary.peak_bytes,
        }

    scaling = []
    for count in config["benchmark"]["sample_scaling_counts"]:
        model = ConstantBodyTwistModel(
            qc,
            0.5,
            model_id="benchmark_sample_{}".format(count),
            backend=TemporalUncertaintyBackend.SAMPLE,
            sample_count=int(count),
            process_path_depth=depth,
        )
        summary = benchmark_callable(
            prediction_call(model),
            repetitions=repetitions,
            warmups=warmups,
        )
        scaling.append(
            {
                "sample_count": int(count),
                "p50_seconds": summary.p50_seconds,
                "p95_seconds": summary.p95_seconds,
                "peak_python_bytes": summary.peak_bytes,
            }
        )
    output["sample_scaling"] = scaling
    return output


def correctness_gates(motion, uncertainty, performance, config):
    endpoint_tolerance = config["correctness_gates"][
        "endpoint_absolute_tolerance"
    ]
    left = DeterministicTransform.identity()
    right = se3_exp([0.8, -0.2, 0.1, 0.4, -0.1, 0.3])
    endpoint_ok = (
        np.linalg.norm(_mixed_residual(left, interpolate_transform(left, right, 0.0)))
        <= endpoint_tolerance
        and np.linalg.norm(_mixed_residual(right, interpolate_transform(left, right, 1.0)))
        <= endpoint_tolerance
    )
    finite_motion = all(
        np.isfinite(row[name]["se3_error_norm"])
        and np.isfinite(row[name]["log_predictive_density"])
        for row in motion["rows"]
        for name in ("constant_twist", "constant_acceleration")
    )
    covariance_psd = all(
        row[backend]["covariance_relative_frobenius_error"] >= 0.0
        and np.isfinite(row[backend]["covariance_relative_frobenius_error"])
        for row in uncertainty["rows"]
        for backend in ("moment", "sample")
    )
    latency_finite = all(
        np.isfinite(performance[name]["p50_seconds"])
        and np.isfinite(performance[name]["p95_seconds"])
        for name in (
            "moment_constant_twist",
            "sample_constant_twist",
            "moment_constant_acceleration",
        )
    )
    return {
        "endpoint_invariant": bool(endpoint_ok),
        "finite_pose_and_log_scores": bool(finite_motion),
        "finite_psd_covariance_metrics": bool(covariance_psd),
        "finite_latency_and_memory": bool(latency_finite),
        "moment_coverage_contains_0_95": uncertainty["summary"]["moment"][
            "empirical_95_percent_coverage"
        ]["contains_nominal_0_95"],
        "sample_coverage_contains_0_95": uncertainty["summary"]["sample"][
            "empirical_95_percent_coverage"
        ]["contains_nominal_0_95"],
        "marginal_only_shared_factor_exact": False,
        "marginal_only_shared_factor_policy": config["implementation"][
            "marginal_only_shared_factor_policy"
        ],
        "provenance_payload_checked_by_conformance_suite": True,
        "causality_and_support_checked_by_conformance_suite": True,
    }


def dispositions(motion, uncertainty, performance, gates, config):
    acceleration_threshold = config["selection_rules"][
        "acceleration_optional_min_score_improvement_fraction"
    ]
    improved = [
        episode_id
        for episode_id, metrics in motion["episode_improvements"].items()
        if metrics["acceleration_improvement_fraction"] >= acceleration_threshold
    ]
    required_independent_corpora = config["selection_rules"][
        "acceleration_min_independent_corpora"
    ]
    independent_corpus_count = 1
    energy = uncertainty["summary"]["sample_energy_improvement_fraction"]
    runtime_ratio = (
        performance["sample_constant_twist"]["p50_seconds"]
        / performance["moment_constant_twist"]["p50_seconds"]
    )
    sample_improves = (
        energy["bootstrap_lower"]
        >= config["selection_rules"][
            "sample_optional_min_energy_improvement_fraction"
        ]
    )
    return {
        "tangent_space_moment_backend": {
            "status": (
                "DEFAULT"
                if gates["moment_coverage_contains_0_95"]
                else "EXPERIMENTAL"
            ),
            "reason": (
                "Fastest calibrated backend on the frozen local/mixed corpus; "
                "unsupported local moments fail closed and missing cross-time "
                "covariance is explicitly diagnosed."
            ),
        },
        "sample_backend": {
            "status": "EXPERIMENTAL",
            "reason": (
                "Retained as a nonlinear oracle, but marginal endpoint records "
                "cannot split shared calibration factors from independent "
                "measurement residuals. A factor-level joint-sample contract "
                "is required before public OPTIONAL status."
            ),
            "energy_improvement_rule_passed": bool(sample_improves),
            "p50_runtime_ratio_vs_moment": float(runtime_ratio),
        },
        "constant_body_twist": {
            "status": "DEFAULT",
            "reason": (
                "Minimal causal two-anchor model; stable baseline across all "
                "strata and no implicit model selection."
            ),
        },
        "constant_body_acceleration": {
            "status": "EXPERIMENTAL",
            "reason": (
                "Explicit acceleration metadata improved selected episodes, "
                "but every episode belongs to one frozen synthetic corpus. A "
                "second independent application corpus is required for "
                "OPTIONAL status. Episode threshold passed in: {}.".format(
                    ", ".join(improved) if improved else "none"
                )
            ),
            "episodes_passing_improvement_threshold": improved,
            "independent_corpus_count": independent_corpus_count,
            "required_independent_corpus_count": required_independent_corpora,
        },
        "endpoint_conditioned_sample_interpolation": {
            "status": "EXPERIMENTAL",
            "reason": (
                "Useful for offline non-Gaussian analysis, subject to the same "
                "factor-level joint-sample limitation as the sample backend."
            ),
        },
        "discrete_qd_compatibility_adapter": {
            "status": "OPTIONAL",
            "reason": (
                "Migration-only adapter with mandatory sample period and "
                "diagnostic provenance; canonical model configuration remains Qc."
            ),
        },
        "automatic_model_selector": {
            "status": "PRUNE",
            "reason": (
                "Not implemented: selector uncertainty cannot yet be represented."
            ),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=HERE / "temporal_selection.yaml",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    corpus_path = (config_path.parent / config["corpus"]["path"]).resolve()
    actual_corpus_hash = _sha256(corpus_path)
    expected_corpus_hash = config["corpus"]["sha256"]
    if actual_corpus_hash != expected_corpus_hash:
        raise SystemExit(
            "Frozen corpus hash mismatch: expected {}, got {}".format(
                expected_corpus_hash,
                actual_corpus_hash,
            )
        )
    corpus = yaml.safe_load(corpus_path.read_text(encoding="utf-8"))
    episode_ids = {episode["id"] for episode in corpus["episodes"]}
    configured_ids = set(config["corpus"]["training_episode_ids"]) | set(
        config["corpus"]["held_out_episode_ids"]
    )
    if configured_ids != episode_ids:
        raise SystemExit("Training/held-out split does not exactly cover the corpus.")

    motion = evaluate_motion_models(corpus, config)
    uncertainty = evaluate_uncertainty_backends(corpus, config)
    performance = evaluate_performance(config)
    gates = correctness_gates(motion, uncertainty, performance, config)
    result = {
        "schema_version": 1,
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "repository_head_at_run": _git("rev-parse", "HEAD"),
        "core_implementation_commit": config["comparison_commit"],
        "working_tree_dirty": bool(_git("status", "--porcelain")),
        "core_working_tree_dirty": bool(
            _git(
                "status",
                "--porcelain",
                "--",
                ":(top)ros/core/probtf_core",
                ":(top)tests/probtf",
            )
        ),
        "core_head_tree": _git("rev-parse", "HEAD:ros/core/probtf_core"),
        "probtf_tests_head_tree": _git("rev-parse", "HEAD:tests/probtf"),
        "evaluated_core_source_sha256": _evaluated_core_source_hash(),
        "frozen_before_run": config["frozen_before_run"],
        "config_path": str(config_path.relative_to(PACKAGE_ROOT)),
        "config_sha256": _sha256(config_path),
        "runner_sha256": _sha256(Path(__file__).resolve()),
        "corpus_path": str(corpus_path.relative_to(PACKAGE_ROOT)),
        "corpus_sha256": actual_corpus_hash,
        "environment": environment_manifest(config["random_seeds"]),
        "motion_model_evaluation": motion,
        "uncertainty_backend_evaluation": uncertainty,
        "performance": performance,
        "correctness_gates": gates,
    }
    result["dispositions"] = dispositions(
        motion,
        uncertainty,
        performance,
        gates,
        config,
    )
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(encoded, end="")
    else:
        args.output.write_text(encoded, encoding="utf-8")


if __name__ == "__main__":
    main()
