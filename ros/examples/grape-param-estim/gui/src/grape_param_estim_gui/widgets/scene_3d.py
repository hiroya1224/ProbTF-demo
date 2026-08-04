"""Small PyVista views for strict batch-estimation trajectories."""

from __future__ import annotations

import os

import numpy as np
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from ..artifact_loader import BatchEstimationRun

try:
    import pyvista as pv
    from pyvistaqt import QtInteractor
except ImportError:  # pragma: no cover - optional visual dependencies
    pv = None  # type: ignore[assignment]
    QtInteractor = None  # type: ignore[assignment]


class PidComparisonScene3DWidget(QWidget):
    """Show recorded-control rollouts for the selected posterior plant sample.

    The PID evaluation v2 artifact stores scalar cross-evaluation metrics, not
    candidate forecast trajectories.  This view therefore presents the exact
    estimation-run context used by the selected plant sample: reference,
    estimated-parameter rollout, and the selected posterior rollout.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.run: BatchEstimationRun | None = None
        self.selected_bag_id: str | None = None
        self.selected_sample_id: str | None = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        disabled = os.environ.get("GRAPE_PARAM_ESTIM_DISABLE_3D") == "1"
        if pv is None or QtInteractor is None or disabled:
            self.plotter = None
            self.status_label = QLabel(
                "3D viewer disabled. Reference, MAP, and selected conditional "
                "trajectory identities remain available in the artifact."
            )
            self.status_label.setWordWrap(True)
            self.status_label.setStyleSheet("padding:24px;font-size:14px;")
            layout.addWidget(self.status_label)
        else:
            self.status_label = None
            self.plotter = QtInteractor(self)
            layout.addWidget(self.plotter.interactor)
            self.plotter.set_background("#20242b")
            self.plotter.add_axes()

    def set_context(
        self,
        run: BatchEstimationRun | None,
        bag_id: str | None,
        sample_id: str | None,
    ) -> None:
        self.run = run
        self.selected_bag_id = None if bag_id is None else str(bag_id)
        self.selected_sample_id = None if sample_id is None else str(sample_id)
        self.rebuild_scene()

    def _paths(self) -> list[tuple[str, np.ndarray, str, float]]:
        if self.run is None or self.selected_bag_id not in self.run.bags:
            return []
        result = self.run.bags[self.selected_bag_id]
        paths = [
            ("Reference", result.reference.position, "#aaaaaa", 2.0),
            (
                "Estimated-parameter rollout",
                result.estimated_parameter_rollout.position[
                    result.estimated_parameter_rollout.valid
                ],
                "#34b77b",
                3.2,
            ),
        ]
        if self.selected_sample_id is not None:
            selected = self.run.selected_trajectory(
                self.selected_bag_id, self.selected_sample_id
            )
            if selected is not None:
                paths.append(
                    (
                        "Selected posterior rollout",
                        selected.recorded_control_rollout.position[
                            selected.recorded_control_rollout.valid
                        ],
                        "#d45bd4",
                        4.0,
                    )
                )
        paths.append(
            (
                "Observed",
                result.pose.position,
                "#1e5abe",
                3.8,
            )
        )
        return paths

    def rebuild_scene(self) -> None:
        paths = self._paths()
        if self.plotter is None:
            if self.status_label is not None:
                sample = self.selected_sample_id or "none"
                self.status_label.setText(
                    "3D viewer disabled. {} path(s) loaded for sample {}. "
                    "PID forecast paths are not part of the v1 evaluation artifact.".format(
                        len(paths), sample
                    )
                )
            return
        self.plotter.clear()
        self.plotter.add_axes()
        legend = []
        for name, points, color, width in paths:
            values = np.asarray(points, dtype=float)
            values = values[np.all(np.isfinite(values), axis=1)]
            if values.shape[0] < 2:
                continue
            self.plotter.add_mesh(
                pv.lines_from_points(values),
                color=color,
                line_width=width,
                render_lines_as_tubes=True,
                name=name,
            )
            legend.append((name, color))
        self.plotter.add_text(
            "Estimation trajectory context | sample {}".format(
                self.selected_sample_id or "—"
            ),
            position="upper_left",
            font_size=11,
        )
        if legend:
            self.plotter.add_legend(legend, bcolor="#20242b")
            self.plotter.reset_camera()
            self.plotter.view_isometric()
        self.plotter.render()

    def reset_camera(self) -> None:
        if self.plotter is not None:
            self.plotter.reset_camera()
            self.plotter.view_isometric()
            self.plotter.render()

    def close_scene(self) -> None:
        if self.plotter is not None:
            self.plotter.close()


class Scene3DWidget(PidComparisonScene3DWidget):
    """General batch trajectory scene kept as a small reusable widget."""


__all__ = ["PidComparisonScene3DWidget", "Scene3DWidget"]
