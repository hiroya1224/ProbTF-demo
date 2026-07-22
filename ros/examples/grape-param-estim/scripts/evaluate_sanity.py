#!/usr/bin/env python3
"""Evaluate the final synthetic posterior without feeding truth to estimation."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import rosbag

from grape_param_estim.dynamics import PARAMETER_NAMES, parameters_to_inertia


DEFAULT_ESTIMATE_TOPIC = "/grape_param_estim/estimate"
DEFAULT_TRUTH_TOPIC = "/grape_param_estim/ground_truth"


class EvaluationError(RuntimeError):
    """Raised when an analysis bag cannot be evaluated safely."""


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Compare the final parameter posterior with synthetic ground truth."
    )
    parser.add_argument("--analysis-bag", required=True, help="Estimator analysis bag.")
    parser.add_argument(
        "--estimate-topic", default=DEFAULT_ESTIMATE_TOPIC, help="Posterior topic."
    )
    parser.add_argument(
        "--truth-topic", default=DEFAULT_TRUTH_TOPIC, help="Evaluation-only truth topic."
    )
    parser.add_argument(
        "--mass-relative-threshold",
        type=float,
        default=0.05,
        help="Maximum relative mass error (default: %(default)s).",
    )
    parser.add_argument(
        "--cog-threshold-m",
        type=float,
        default=0.02,
        help="Maximum CoG Euclidean error in meters (default: %(default)s).",
    )
    parser.add_argument(
        "--inertia-relative-threshold",
        type=float,
        default=0.15,
        help="Maximum relative inertia Frobenius error (default: %(default)s).",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Always return success after producing a valid report.",
    )
    return parser.parse_args(argv)


def _validate_threshold(value, name):
    value = float(value)
    if not np.isfinite(value) or value < 0.0:
        raise EvaluationError("{} must be finite and non-negative".format(name))
    return value


def _last_messages(path, estimate_topic, truth_topic):
    estimate = None
    truth = None
    try:
        with rosbag.Bag(str(path), "r") as bag:
            for topic, message, _ in bag.read_messages(
                topics=[estimate_topic, truth_topic]
            ):
                if topic == estimate_topic:
                    estimate = message
                elif topic == truth_topic:
                    truth = message
    except (OSError, rosbag.ROSBagException) as exc:
        raise EvaluationError("failed to read {}: {}".format(path, exc)) from exc
    if estimate is None:
        raise EvaluationError("no estimate messages found on {}".format(estimate_topic))
    if truth is None:
        raise EvaluationError("no truth messages found on {}".format(truth_topic))
    return estimate, truth


def _named_vector(message, field, label):
    names = [str(name) for name in message.parameter_names]
    if not names or len(set(names)) != len(names):
        raise EvaluationError("{} parameter_names must be non-empty and unique".format(label))
    values = np.asarray(getattr(message, field), dtype=float)
    if values.shape != (len(names),) or not np.all(np.isfinite(values)):
        raise EvaluationError(
            "{}.{} must be a finite vector matching parameter_names".format(label, field)
        )
    return dict(zip(names, values))


def _ordered(mapping, label):
    missing = [name for name in PARAMETER_NAMES if name not in mapping]
    if missing:
        raise EvaluationError("{} is missing parameters {}".format(label, missing))
    return np.asarray([mapping[name] for name in PARAMETER_NAMES], dtype=float)


def evaluate(args):
    bag_path = Path(args.analysis_bag).expanduser().resolve()
    if not bag_path.is_file():
        raise EvaluationError("analysis bag does not exist: {}".format(bag_path))
    mass_threshold = _validate_threshold(
        args.mass_relative_threshold, "--mass-relative-threshold"
    )
    cog_threshold = _validate_threshold(args.cog_threshold_m, "--cog-threshold-m")
    inertia_threshold = _validate_threshold(
        args.inertia_relative_threshold, "--inertia-relative-threshold"
    )

    estimate_message, truth_message = _last_messages(
        bag_path, args.estimate_topic, args.truth_topic
    )
    estimate = _ordered(
        _named_vector(estimate_message, "mean", "estimate"), "estimate"
    )
    truth = _ordered(_named_vector(truth_message, "mean", "truth"), "truth")
    lower = _ordered(
        _named_vector(estimate_message, "lower_95", "estimate"), "estimate lower_95"
    )
    upper = _ordered(
        _named_vector(estimate_message, "upper_95", "estimate"), "estimate upper_95"
    )
    if np.any(lower > upper):
        raise EvaluationError("estimate has lower_95 values above upper_95")

    if abs(truth[0]) <= np.finfo(float).tiny:
        raise EvaluationError("truth mass is zero, so relative error is undefined")
    truth_inertia = parameters_to_inertia(truth)
    estimate_inertia = parameters_to_inertia(estimate)
    inertia_norm = float(np.linalg.norm(truth_inertia, ord="fro"))
    if inertia_norm <= np.finfo(float).tiny:
        raise EvaluationError("truth inertia norm is zero, so relative error is undefined")

    mass_error = float(abs(estimate[0] - truth[0]) / abs(truth[0]))
    cog_error = float(np.linalg.norm(estimate[1:4] - truth[1:4]))
    inertia_error = float(
        np.linalg.norm(estimate_inertia - truth_inertia, ord="fro") / inertia_norm
    )
    inside = (truth >= lower) & (truth <= upper)

    checks = {
        "mass_relative_error": mass_error <= mass_threshold,
        "cog_error_m": cog_error <= cog_threshold,
        "inertia_frobenius_relative_error": inertia_error <= inertia_threshold,
    }
    passed = bool(all(checks.values()))
    report = {
        "analysis_bag": str(bag_path),
        "estimate_topic": args.estimate_topic,
        "truth_topic": args.truth_topic,
        "estimate": {
            "model": str(estimate_message.model),
            "source_bag": str(estimate_message.source_bag),
            "update_index": int(estimate_message.update_index),
            "observation_count": int(estimate_message.observation_count),
            "parameter_names": list(PARAMETER_NAMES),
            "mean": [float(value) for value in estimate],
        },
        "truth": {
            "model": str(truth_message.model),
            "parameter_names": list(PARAMETER_NAMES),
            "mean": [float(value) for value in truth],
        },
        "metrics": {
            "mass_relative_error": mass_error,
            "cog_error_m": cog_error,
            "inertia_frobenius_relative_error": inertia_error,
        },
        "thresholds": {
            "mass_relative_error": mass_threshold,
            "cog_error_m": cog_threshold,
            "inertia_frobenius_relative_error": inertia_threshold,
        },
        "credible_interval_95": {
            "inside_by_parameter": {
                name: bool(value) for name, value in zip(PARAMETER_NAMES, inside)
            },
            "inside_count": int(np.sum(inside)),
            "parameter_count": len(PARAMETER_NAMES),
            "all_inside": bool(np.all(inside)),
        },
        "checks": checks,
        "passed": passed,
        "report_only": bool(args.report_only),
    }
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0 if passed or args.report_only else 1


def main(argv=None):
    try:
        args = _parse_args(argv)
        return evaluate(args)
    except (AttributeError, EvaluationError, ValueError, TypeError) as exc:
        print(
            json.dumps(
                {"error": str(exc), "passed": False},
                indent=2,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    sys.exit(main())
