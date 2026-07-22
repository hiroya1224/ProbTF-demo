"""Command-line entry point for the Phase-3 offline benchmark."""

import argparse
import json
import sys

from .adapter import ApiMismatchError
from .models import BenchmarkConfig
from .runner import evaluate_dataset


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Run Phase-2 AprilTag detection/mixture estimation on a Phase-1 "
            "synthetic dataset and write deterministic metrics."
        )
    )
    parser.add_argument("dataset", help="dataset root or its frames directory")
    parser.add_argument("output", help="output directory for metrics and overlays")
    parser.add_argument("--family", default="DICT_APRILTAG_36h11")
    parser.add_argument("--corner-sigma-px", type=float, default=0.5)
    parser.add_argument("--no-corner-refinement", action="store_true")
    parser.add_argument("--default-tag-size-m", type=float, default=0.12)
    parser.add_argument("--camera-frame-id", default="camera_optical_frame")
    parser.add_argument("--tag-frame-prefix", default="apriltag_")
    parser.add_argument("--min-visible-fraction", type=float, default=0.0)
    parser.add_argument("--include-back-facing", action="store_true")
    parser.add_argument("--association-max-corner-rmse-px", type=float, default=25.0)
    parser.add_argument("--gt-near-translation-threshold-m", type=float, default=0.02)
    parser.add_argument("--gt-near-rotation-threshold-deg", type=float, default=5.0)
    parser.add_argument("--max-iterations", type=int, default=30)
    parser.add_argument("--verify-jacobian", action="store_true")
    return parser


def main(argv=None):
    arguments = build_parser().parse_args(argv)
    try:
        config = BenchmarkConfig(
            family=arguments.family,
            corner_sigma_px=arguments.corner_sigma_px,
            corner_refinement=not arguments.no_corner_refinement,
            default_tag_size_m=arguments.default_tag_size_m,
            camera_frame_id=arguments.camera_frame_id,
            tag_frame_prefix=arguments.tag_frame_prefix,
            min_visible_fraction=arguments.min_visible_fraction,
            front_facing_only=not arguments.include_back_facing,
            association_max_corner_rmse_px=arguments.association_max_corner_rmse_px,
            gt_near_translation_threshold_m=arguments.gt_near_translation_threshold_m,
            gt_near_rotation_threshold_deg=arguments.gt_near_rotation_threshold_deg,
            estimator_max_iterations=arguments.max_iterations,
            estimator_verify_jacobian=arguments.verify_jacobian,
        )
        report = evaluate_dataset(arguments.dataset, arguments.output, config)
    except (ApiMismatchError, OSError, ValueError) as exc:
        print("prob-artag-benchmark: {}".format(exc), file=sys.stderr)
        return 2
    summary = report["summary"]
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2))
    return 0 if summary["error_frame_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
