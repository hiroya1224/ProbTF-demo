from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import sys

os.environ.setdefault("QT_API", "pyside6")

import pyqtgraph as pg
from PySide6.QtWidgets import QApplication

from .main_window import MainWindow
from .project_io import (
    create_project_directory,
    new_project_manifest,
    unique_project_id,
)
from .state import ProjectStore


def _arguments(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grape parameter assimilation desktop GUI")
    parser.add_argument("--package-root", type=Path)
    parser.add_argument("--projects-root", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    arguments = _arguments(list(sys.argv[1:] if argv is None else argv))
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
    application.setApplicationName("Grape parameter assimilation")
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
    manifest["estimator_settings"] = {
        "sample_period": 0.04,
        "maximum_knots": 12,
        "ensemble_size": 128,
        "maximum_iterations": 5,
        "convergence_tolerance": 1.0e-3,
        "minimum_line_search_step": 1.0 / 64.0,
        "seed": 23,
        "delay_prior_mean": 0.02,
        "delay_prior_standard_deviation": 0.015,
        "allow_configuration_mismatch": False,
    }
    project_path = create_project_directory(projects_root, manifest)
    store = ProjectStore(project_path, manifest)
    window = MainWindow(store, package_root)
    window.show()
    return int(application.exec())


__all__ = ["main"]
