from __future__ import annotations

import argparse
import os
from pathlib import Path
import signal
import shutil
import sys
from typing import Callable

import numpy as np

os.environ.setdefault("QT_API", "pyside6")

import pyqtgraph as pg
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from .main_window import MainWindow
from .project_io import (
    create_project_directory,
    new_project_manifest,
    unique_project_id,
)
from .state import ProjectStore


def default_batch_estimator_settings() -> dict[str, object]:
    """Return explicit, auditable starting settings for a new project.

    Numeric values are provisional project configuration, not inferred sensor
    statistics.  The request preserves that provenance and users may edit the
    project JSON before a run.
    """

    return {
        "q": {
            "update_policy": "fixed",
            "residual_quantity": "body_wrench",
            "interval_model": "continuous_spectral_density",
            "component_names": ["x", "y", "z", "roll", "pitch", "yaw"],
            "component_units": ["N", "N", "N", "N*m", "N*m", "N*m"],
            "initial_diagonal": [25.0, 25.0, 25.0, 1.0, 1.0, 1.0],
            "floor_diagonal": [1.0e-8] * 6,
        },
        "parameter_prior": {
            "kind": "gaussian",
            "mean_coordinate": [0.0] * 18,
            "covariance": np.eye(18).tolist(),
        },
        "delay": {
            "prior_kind": "uniform",
            "bounds_seconds": [0.0, 0.08],
            "initial_seconds": 0.035,
            "coarse_grid_points": 5,
            "refinement_tolerance_seconds": 1.0e-4,
            "maximum_refinement_evaluations": 8,
        },
        "actuator_model": {
            "source": (
                "gimbalrotor MotorInfo.yaml and gimbal_limits.urdf; "
                "gimbal time constant provisional pending system identification"
            ),
            "thrust_time_constant_seconds": 0.01,
            "gimbal_time_constant_seconds": 0.02,
            "minimum_thrust_newtons": 1.5,
            "maximum_thrust_newtons": 27.6145,
            "maximum_gimbal_angle_radians": 3.14,
            "maximum_gimbal_rate_radians_per_second": 6.0,
        },
        "knot_policy": {
            "period_seconds": 0.05,
            "origin": "interval_start",
            "maximum_measurement_gap_seconds": 0.06,
        },
        "interpolation_policy": {
            "euclidean": "linear",
            "orientation": "so3_geodesic",
            "command": "zoh_record_issue_time",
            "allow_extrapolation": False,
        },
        "controller_snapshot_policy": {
            "source": "bag_startup_parameter_updates",
            "require_constant_within_interval": True,
        },
        "mode_hypotheses": [],
        "solver_settings": {
            "method": "sparse_lm",
            "maximum_iterations": 30,
            "maximum_factorization_retries": 4,
            "maximum_model_evaluation_retries": 4,
            "acceptance_ratio": 1.0e-4,
            "gradient_tolerance": 1.0e-6,
            "scaled_step_tolerance": 1.0e-7,
            "relative_objective_tolerance": 1.0e-8,
            "initial_damping": 1.0e-3,
            "minimum_damping": 1.0e-12,
            "maximum_damping": 1.0e12,
        },
        "em_settings": {
            "maximum_iterations": 5,
            "minimum_iterations": 2,
            "maximum_repeated_q_rejections": 3,
            "maximum_repeated_lag_profile_failures": 3,
            "log_q_tolerance": 1.0e-3,
            "lag_tolerance": 1.0e-5,
            "map_objective_tolerance": 1.0e-5,
            "marginal_objective_tolerance": 1.0e-5,
            "q_acceptance_objective_tolerance": 0.0,
            "q_minimum_alpha": 1.0 / 64.0,
        },
        "mcmc_settings": {
            "enabled": True,
            "chain_count": 4,
            "warmup_steps": 100,
            "retained_draws": 200,
            "thinning": 1,
            "random_seed": 42,
            "local_scale": 0.5,
            "exact_ridge_scale": 0.25,
            "near_ridge_scale": 0.25,
            "identified_scale": 0.1,
            "delay_scale_seconds": 0.002,
            "near_relative_threshold": 1.0e-6,
            "rhat_threshold": 1.01,
            "minimum_effective_sample_size": 100.0,
        },
    }


def _arguments(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grape sparse batch estimation desktop GUI")
    parser.add_argument("--package-root", type=Path)
    parser.add_argument("--projects-root", type=Path)
    parser.add_argument(
        "--bag",
        action="append",
        default=[],
        type=Path,
        help="rosbag to copy into the new project and inspect (repeatable)",
    )
    return parser.parse_args(argv)


def _validated_bag_paths(values: list[Path]) -> tuple[Path, ...]:
    paths = tuple(path.expanduser().resolve() for path in values)
    missing = tuple(path for path in paths if not path.is_file())
    if missing:
        raise FileNotFoundError(
            "rosbag is not a file: {}".format(
                ", ".join(str(path) for path in missing)
            )
        )
    return paths


def _install_interrupt_handler(
    application: QApplication,
    close_window: Callable[[], object],
) -> tuple[QTimer, object]:
    """Bridge terminal SIGINT into the Qt event loop and window close path."""

    if not callable(close_window):
        raise TypeError("close_window must be callable")
    previous_handler = signal.getsignal(signal.SIGINT)
    poll_timer = QTimer(application)
    poll_timer.setInterval(100)
    # Entering Python regularly lets its pending-signal machinery run while
    # QApplication.exec() otherwise remains inside the Qt C++ event loop.
    poll_timer.timeout.connect(lambda: None)
    poll_timer.start()

    def request_close(_signum: int, _frame: object) -> None:
        QTimer.singleShot(0, close_window)

    signal.signal(signal.SIGINT, request_close)
    return poll_timer, previous_handler


def main(argv: list[str] | None = None) -> int:
    arguments = _arguments(list(sys.argv[1:] if argv is None else argv))
    bag_paths = _validated_bag_paths(arguments.bag)
    configured_root = arguments.package_root
    if configured_root is None:
        environment_root = os.environ.get("GRAPE_PARAM_ESTIM_PACKAGE_ROOT")
        configured_root = (
            Path(environment_root)
            if environment_root
            else Path(__file__).resolve().parents[3]
        )
    package_root = configured_root.expanduser().resolve()
    source_worker = package_root / "scripts" / "grape_inspect_flights.py"
    installed_worker = (
        package_root.parent.parent
        / "lib"
        / "grape_param_estim"
        / "grape_inspect_flights.py"
    )
    if (
        not source_worker.is_file()
        and not installed_worker.is_file()
        and shutil.which("grape_inspect_flights.py") is None
    ):
        raise RuntimeError(
            "cannot locate estimator workers; pass --package-root or set "
            "GRAPE_PARAM_ESTIM_PACKAGE_ROOT"
        )
    projects_root = (
        package_root / "projects"
        if arguments.projects_root is None
        else arguments.projects_root.expanduser().resolve()
    )
    application = QApplication.instance() or QApplication(sys.argv[:1])
    application.setApplicationName("Grape sparse batch estimation")
    application.setOrganizationName("ProbTF demo")
    pg.setConfigOptions(
        background="w",
        foreground=(35, 35, 35),
        antialias=False,
        useOpenGL=False,
    )
    manifest = new_project_manifest(
        unique_project_id("grape"),
        gui_revision="1.0.0",
        estimator_revision=os.environ.get("GRAPE_PARAM_ESTIM_REVISION", "workspace"),
    )
    manifest["estimator_settings"] = default_batch_estimator_settings()
    project_path = create_project_directory(projects_root, manifest)
    store = ProjectStore(project_path, manifest)
    window = MainWindow(store, package_root)
    window.show()
    if bag_paths:
        QTimer.singleShot(0, lambda: window.add_bag_files(bag_paths))
    interrupt_timer, previous_interrupt_handler = _install_interrupt_handler(
        application, window.close
    )
    try:
        return int(application.exec())
    finally:
        interrupt_timer.stop()
        signal.signal(signal.SIGINT, previous_interrupt_handler)


__all__ = ["main"]
