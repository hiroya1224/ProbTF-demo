"""Render the production GUI and real artifact paths for visual acceptance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PySide6.QtGui import QImageReader
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from grape_param_estim_gui.artifact_loader import (
    load_assimilation,
    load_pid_evaluation,
)
from grape_param_estim_gui.main_window import MainWindow
from grape_param_estim_gui.project_io import new_project_manifest
from grape_param_estim_gui.state import BagRecord, ProjectStore


def _first_text(value: object, fallback: str) -> str:
    array = np.asarray(value).reshape(-1)
    return fallback if array.size == 0 else str(array[0])


def _capture_widget(
    application: QApplication,
    widget: object,
    destination: Path,
) -> None:
    widget.raise_()
    widget.activateWindow()
    application.processEvents()
    QTest.qWait(250)
    screen = widget.screen() or application.primaryScreen()
    if screen is None:
        raise RuntimeError("no screen is available for visual acceptance")
    pixmap = screen.grabWindow(int(widget.winId()))
    if pixmap.isNull() or not pixmap.save(str(destination), "PNG"):
        raise RuntimeError("could not capture {}".format(destination))


def _capture_plotter(
    application: QApplication,
    plotter: object,
    destination: Path,
) -> None:
    application.processEvents()
    QTest.qWait(250)
    plotter.render()
    plotter.screenshot(str(destination))


def render_acceptance(
    assimilation_path: Path,
    pid_path: Path,
    output: Path,
) -> dict[str, object]:
    run = load_assimilation(assimilation_path)
    evaluation = load_pid_evaluation(pid_path)
    if evaluation.manifest["source_run_id"] != run.manifest["run_id"]:
        raise ValueError("PID evaluation and assimilation run IDs differ")
    bag_id = str(run.manifest["selected_bag_ids"][0])
    result = run.bag_results[bag_id]
    if bag_id not in evaluation.bags:
        raise ValueError("PID evaluation does not contain the assimilation bag")
    member_id = int(run.shared_posterior.member_id[0])
    candidate_id = "member-{}".format(member_id)
    candidates = set(np.asarray(evaluation.summary["candidate_id"]).astype(str))
    if candidate_id not in candidates:
        raise ValueError("PID evaluation lacks {}".format(candidate_id))

    output.mkdir(parents=True, exist_ok=True)
    application = QApplication.instance() or QApplication(sys.argv[:1])
    project_path = output / "working-project"
    project_path.mkdir(exist_ok=True)
    manifest = new_project_manifest("visual-acceptance")
    provenance = result.provenance
    bag_path = Path(
        _first_text(provenance.get("bag_path"), "/missing/flight.bag")
    )
    bag_sha = _first_text(provenance.get("bag_sha256"), "0" * 64)
    manifest["bags"] = [
        {
            "bag_id": bag_id,
            "source_path": str(bag_path),
            "relative_path": "bags/visual-source.bag",
            "sha256": bag_sha,
        }
    ]
    store = ProjectStore(project_path, manifest)
    interval = (float(result.time[0]), float(result.time[-1]))
    gains = np.asarray(evaluation.summary["current_pid"], dtype=float)
    record = BagRecord(
        bag_id=bag_id,
        path=bag_path,
        source_path=bag_path,
        sha256=bag_sha,
        inspection={
            "status": "ready",
            "topic_contract": provenance.get("topic_names", []),
            "warnings": list(run.warnings),
        },
        preview=result,
        result=result,
        included=True,
        auto_interval=interval,
        selected_interval=interval,
        interval_state="LOCKED",
        status="complete",
        configuration_fingerprint=str(
            run.manifest["configuration_fingerprint"]
        ),
        controller_snapshot={
            "gains": gains.tolist(),
            "source": "recorded dynamic-reconfigure snapshot",
        },
        current_time=float(result.time[result.time.size // 2]),
        view_range=interval,
    )
    store.add(record)
    request_fingerprint = store.request_fingerprint()
    run.manifest["project_request_fingerprint"] = request_fingerprint
    store.manifest["run_request_fingerprint"] = request_fingerprint
    store.apply_assimilation(run)
    store.apply_pid_evaluation(evaluation)
    store.set_selected_member(member_id)

    window = MainWindow(store, Path(__file__).resolve().parents[2])
    window.master_view.set_run(run)
    store.set_selected_pid_proposal(candidate_id)
    window.resize(1800, 1120)
    window.show()
    application.processEvents()
    QTest.qWait(500)

    screenshots: list[Path] = []
    window.tabs.setCurrentWidget(window.master_view)
    master_path = output / "master.png"
    _capture_widget(application, window, master_path)
    screenshots.append(master_path)

    window.tabs.setCurrentWidget(window.bag_browser)
    if window.bag_browser.scene.plotter is None:
        raise RuntimeError("production world-trajectory 3D plotter is unavailable")
    window.bag_browser.view_mode_combo.setCurrentIndex(
        window.bag_browser.view_mode_combo.findData("world")
    )
    bag_world_path = output / "bag_browser_world.png"
    _capture_widget(application, window, bag_world_path)
    screenshots.append(bag_world_path)
    world_vtk_path = output / "bag_world_vtk.png"
    _capture_plotter(
        application, window.bag_browser.scene.plotter, world_vtk_path
    )
    screenshots.append(world_vtk_path)

    window.bag_browser.view_mode_combo.setCurrentIndex(
        window.bag_browser.view_mode_combo.findData("correction")
    )
    bag_correction_path = output / "bag_browser_correction.png"
    _capture_widget(application, window, bag_correction_path)
    screenshots.append(bag_correction_path)
    correction_vtk_path = output / "bag_correction_vtk.png"
    _capture_plotter(
        application,
        window.bag_browser.scene.plotter,
        correction_vtk_path,
    )
    screenshots.append(correction_vtk_path)

    window.tabs.setCurrentWidget(window.next_experiment)
    comparison = window.next_experiment.comparison_scene
    if comparison.plotter is None:
        raise RuntimeError("production PID-comparison 3D plotter is unavailable")
    comparison.set_selection(bag_id, member_id, candidate_id)
    detail_tabs = window.next_experiment.detail_tabs
    for tab_name, file_name in (
        ("Aggregate metrics", "pid_aggregate_metrics.png"),
        ("Correction paths", "pid_correction_paths.png"),
        ("Proposed YAML", "pid_proposed_yaml.png"),
    ):
        for index in range(detail_tabs.count()):
            if detail_tabs.tabText(index) == tab_name:
                detail_tabs.setCurrentIndex(index)
                break
        else:
            raise RuntimeError("missing Next experiment tab {}".format(tab_name))
        widget_path = output / file_name
        _capture_widget(application, window, widget_path)
        screenshots.append(widget_path)
    for index in range(detail_tabs.count()):
        if detail_tabs.tabText(index) == "3D comparison":
            detail_tabs.setCurrentIndex(index)
            break
    else:
        raise RuntimeError("missing Next experiment 3D comparison tab")
    for mode, file_name in (
        ("trajectory", "pid_trajectory"),
        ("translation", "pid_translation"),
        ("rotation", "pid_rotation"),
    ):
        comparison.view_combo.setCurrentIndex(
            comparison.view_combo.findData(mode)
        )
        widget_path = output / (file_name + ".png")
        _capture_widget(application, window, widget_path)
        screenshots.append(widget_path)
        vtk_path = output / (file_name + "_vtk.png")
        _capture_plotter(application, comparison.plotter, vtk_path)
        screenshots.append(vtk_path)

    dimensions = {}
    for path in screenshots:
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError("empty screenshot {}".format(path))
        image_size = QImageReader(str(path)).size()
        if not image_size.isValid():
            raise RuntimeError("invalid screenshot {}".format(path))
        dimensions[path.name] = {
            "size_bytes": path.stat().st_size,
            "width": int(image_size.width()),
            "height": int(image_size.height()),
        }
    summary = {
        "schema": "grape-param-estim/gui-visual-acceptance/v1",
        "assimilation_run_id": run.manifest["run_id"],
        "pid_evaluation_id": evaluation.manifest["evaluation_id"],
        "bag_id": bag_id,
        "member_id": member_id,
        "candidate_id": candidate_id,
        "window_capture_method": "QScreen.grabWindow",
        "world_plotter_available": True,
        "pid_plotter_available": True,
        "screenshots": dimensions,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    window.close()
    application.processEvents()
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assimilation", required=True, type=Path)
    parser.add_argument("--pid-evaluation", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    print(
        json.dumps(
            render_acceptance(
                arguments.assimilation,
                arguments.pid_evaluation,
                arguments.output,
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
