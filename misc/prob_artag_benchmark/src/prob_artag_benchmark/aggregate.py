"""Deterministically aggregate per-scenario Phase-3 reports."""

import argparse
import csv
import hashlib
from importlib import metadata
import json
from pathlib import Path

import cv2
import numpy as np
import scipy

from .io import write_json


DEFAULT_SCENARIOS = (
    "frontal",
    "moderate",
    "oblique",
    "small",
    "occluded",
    "multi_tag",
)


def _distribution_version(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _mean_max(rows, field):
    values = [float(row[field]) for row in rows if row.get(field) not in (None, "")]
    return {
        "{}_mean".format(field): None if not values else float(np.mean(values)),
        "{}_max".format(field): None if not values else float(np.max(values)),
    }


def _tree_manifest(root):
    root = Path(root)
    if not root.is_dir():
        raise ValueError("comparison tree does not exist: {}".format(root))
    manifest = {}
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        manifest[path.relative_to(root).as_posix()] = {
            "sha256": digest.hexdigest(),
            "size": path.stat().st_size,
        }
    return manifest


def _compare_trees(comparison_roots):
    if comparison_roots is None:
        return {
            "comparison_performed": False,
            "byte_identical": None,
            "file_count": 0,
            "tree_sha256": None,
        }
    roots = tuple(comparison_roots)
    if len(roots) != 2:
        raise ValueError("comparison_roots must contain exactly two directories")
    first = _tree_manifest(roots[0])
    second = _tree_manifest(roots[1])
    if first != second:
        raise ValueError("regenerated trees are not byte-identical")
    digest = hashlib.sha256()
    for relative_path, entry in sorted(first.items()):
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(entry["sha256"].encode("ascii"))
        digest.update(b"\0")
        digest.update(str(entry["size"]).encode("ascii"))
        digest.update(b"\n")
    return {
        "comparison_performed": True,
        "byte_identical": True,
        "file_count": len(first),
        "tree_sha256": digest.hexdigest(),
    }


def _validate_comparison_provenance(
    comparison_roots, scenario_reports, reference_config, fixture_seed
):
    if comparison_roots is None:
        return None
    observed_scenario = None
    for comparison_root in comparison_roots:
        root = Path(comparison_root)
        metrics_path = root / "report" / "metrics.json"
        with metrics_path.open("r", encoding="utf-8") as stream:
            report = json.load(stream)
        if report.get("schema_version") != 1 or report.get("config") != reference_config:
            raise ValueError("comparison report schema/config does not match aggregate input")
        frame_scenarios = {
            str(frame.get("scenario")) for frame in report.get("frames", ())
        }
        if len(frame_scenarios) != 1:
            raise ValueError("comparison report must contain exactly one scenario")
        scenario = frame_scenarios.pop()
        if scenario not in scenario_reports:
            raise ValueError("comparison scenario is absent from aggregate input")
        expected_summary = dict(scenario_reports[scenario])
        expected_candidate_count = expected_summary.pop("candidate_count")
        if report.get("summary") != expected_summary:
            raise ValueError("comparison summary does not match aggregate input")
        with (root / "report" / "candidates.csv").open(
            "r", encoding="utf-8", newline=""
        ) as stream:
            candidate_count = sum(1 for _ in csv.DictReader(stream))
        if candidate_count != expected_candidate_count:
            raise ValueError("comparison candidate count does not match aggregate input")
        metadata_paths = sorted((root / "dataset" / "frames").glob("*/metadata.json"))
        if not metadata_paths:
            raise ValueError("comparison dataset contains no metadata")
        for metadata_path in metadata_paths:
            with metadata_path.open("r", encoding="utf-8") as stream:
                frame_metadata = json.load(stream)
            if int(frame_metadata.get("seed", -1)) != fixture_seed:
                raise ValueError("comparison dataset seed does not match aggregate input")
            if str(frame_metadata.get("scenario")) != scenario:
                raise ValueError("comparison dataset scenario does not match its report")
        if observed_scenario is None:
            observed_scenario = scenario
        elif observed_scenario != scenario:
            raise ValueError("comparison trees contain different scenarios")
    return observed_scenario


def aggregate_reports(
    report_root,
    scenarios=DEFAULT_SCENARIOS,
    projected_size_threshold_px=50.0,
    comparison_roots=None,
):
    root = Path(report_root)
    scenario_reports = {}
    large_tag_rows = []
    reference_config = None
    fixture_seeds = set()
    for scenario in scenarios:
        metadata_paths = sorted(
            (root / scenario / "dataset" / "frames").glob("*/metadata.json")
        )
        if not metadata_paths:
            raise ValueError("scenario {} has no dataset metadata".format(scenario))
        for metadata_path in metadata_paths:
            with metadata_path.open("r", encoding="utf-8") as stream:
                frame_metadata = json.load(stream)
            frame_seed = frame_metadata.get("seed")
            if isinstance(frame_seed, bool) or not isinstance(frame_seed, int):
                raise ValueError("frame seed must be an integer: {}".format(metadata_path))
            if str(frame_metadata.get("scenario")) != str(scenario):
                raise ValueError("frame scenario does not match directory: {}".format(metadata_path))
            fixture_seeds.add(int(frame_seed))
        report_dir = root / scenario / "report"
        with (report_dir / "metrics.json").open("r", encoding="utf-8") as stream:
            report = json.load(stream)
        config = report["config"]
        if reference_config is None:
            reference_config = config
        elif config != reference_config:
            raise ValueError("benchmark config differs for scenario {}".format(scenario))
        with (report_dir / "candidates.csv").open(
            "r", encoding="utf-8", newline=""
        ) as stream:
            candidate_count = sum(1 for _ in csv.DictReader(stream))
        with (report_dir / "tags.csv").open(
            "r", encoding="utf-8", newline=""
        ) as stream:
            tag_rows = list(csv.DictReader(stream))
        for row in tag_rows:
            size = row.get("projected_size_px", "")
            if size and float(size) >= float(projected_size_threshold_px):
                large_tag_rows.append(row)
        scenario_summary = dict(report["summary"])
        scenario_summary["candidate_count"] = candidate_count
        scenario_reports[str(scenario)] = scenario_summary

    if reference_config is None:
        raise ValueError("at least one scenario is required")
    if len(fixture_seeds) != 1:
        raise ValueError("all scenario frames must use one deterministic seed")
    fixture_seed = next(iter(fixture_seeds))
    derived = {"tag_count": len(large_tag_rows)}
    for field in (
        "corner_rmse_px",
        "nearest_ippe_translation_error_m",
        "nearest_ippe_rotation_error_deg",
        "nearest_mode_translation_error_m",
        "nearest_mode_rotation_error_deg",
    ):
        derived.update(_mean_max(large_tag_rows, field))

    determinism = _compare_trees(comparison_roots)
    determinism["scenario"] = _validate_comparison_provenance(
        comparison_roots, scenario_reports, reference_config, fixture_seed
    )
    return {
        "schema_version": 1,
        "benchmark": "prob_artag_benchmark",
        "fixture": {
            "seed": fixture_seed,
            "family": reference_config["family"],
            "corner_sigma_px": reference_config["corner_sigma_px"],
            "gt_near_translation_threshold_m": reference_config[
                "gt_near_translation_threshold_m"
            ],
            "gt_near_rotation_threshold_deg": reference_config[
                "gt_near_rotation_threshold_deg"
            ],
        },
        "environment": {
            "numpy": np.__version__,
            "opencv": cv2.__version__,
            "pyrender": _distribution_version("pyrender"),
            "scipy": scipy.__version__,
        },
        "determinism": determinism,
        "derived": {
            "projected_size_at_least_px": float(projected_size_threshold_px),
            "metrics": derived,
        },
        "scenarios": scenario_reports,
    }


def build_parser():
    parser = argparse.ArgumentParser(
        description="Aggregate deterministic prob_artag_benchmark scenario reports."
    )
    parser.add_argument("report_root")
    parser.add_argument("output")
    parser.add_argument("--projected-size-threshold-px", type=float, default=50.0)
    parser.add_argument(
        "--compare-tree",
        nargs=2,
        metavar=("FIRST", "SECOND"),
        help="verify two regenerated dataset+report trees byte-for-byte",
    )
    return parser


def main(argv=None):
    arguments = build_parser().parse_args(argv)
    try:
        report = aggregate_reports(
            arguments.report_root,
            projected_size_threshold_px=arguments.projected_size_threshold_px,
            comparison_roots=arguments.compare_tree,
        )
        write_json(arguments.output, report)
    except (KeyError, OSError, TypeError, ValueError) as exc:
        print("prob-artag-aggregate: {}".format(exc))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
