from __future__ import annotations

import numpy as np
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from ..artifact_loader import FlightResult, PidProposalEvaluation, SharedPosterior

try:
    import pyvista as pv
    from pyvistaqt import QtInteractor
except ImportError:  # pragma: no cover - optional 3D dependencies
    pv = None
    QtInteractor = None


def _rpy_matrix(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = (float(value) for value in rpy)
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=float,
    )


def _rotvec_to_rpy(rotvec: np.ndarray) -> np.ndarray:
    vector = np.asarray(rotvec, dtype=float)
    angle = float(np.linalg.norm(vector))
    if angle < 1.0e-14:
        matrix = np.eye(3)
    else:
        axis = vector / angle
        skew = np.array(
            [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]]
        )
        matrix = np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)
    pitch = np.arcsin(np.clip(-matrix[2, 0], -1.0, 1.0))
    return np.array(
        [np.arctan2(matrix[2, 1], matrix[2, 2]), pitch, np.arctan2(matrix[1, 0], matrix[0, 0])]
    )


def _segmented_polyline(points: np.ndarray):
    if pv is None:
        return None
    points = np.asarray(points, dtype=float)
    poly = pv.PolyData()
    poly.points = points
    count = int(points.shape[0])
    if count >= 2:
        indices = np.arange(count - 1, dtype=np.int64)
        poly.lines = np.column_stack(
            [
                np.full(count - 1, 2, dtype=np.int64),
                indices,
                indices + 1,
            ]
        ).ravel()
    return poly


class Scene3DWidget(QWidget):
    """Embedded PyVista scene synchronized with time and ensemble-member selection."""

    def __init__(
        self,
        parameter_ensemble: SharedPosterior | None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.parameter_ensemble = parameter_ensemble
        self.session: FlightResult | None = None
        self.selected_member_id = (
            None if parameter_ensemble is None or parameter_ensemble.size == 0
            else int(parameter_ensemble.member_id[0])
        )
        self.current_time = 0.0
        self.view_mode = "world"
        self.layers = {
            "reference": True,
            "observed": True,
            "nominal": True,
            "posterior": True,
            "selected": True,
        }
        self._marker_actors: dict[str, object] = {}
        self._frame_actor_names: list[str] = []
        self._full_path_states: list[tuple[object, np.ndarray]] = []
        self._path_states: list[tuple[object, np.ndarray, float]] = []
        self._last_path_index = -1

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        if QtInteractor is None or pv is None:
            message = QLabel(
                "PyVistaQt is not installed.\n"
                "Install the project dependencies to enable the 3D viewer."
            )
            message.setStyleSheet("padding: 24px; font-size: 14px;")
            layout.addWidget(message)
            self.plotter = None
            return

        self.plotter = QtInteractor(self)
        layout.addWidget(self.plotter.interactor)
        self.plotter.set_background("#20242b")
        self.plotter.add_axes()

    def set_session(self, session: FlightResult | None) -> None:
        self.session = session
        if session is not None:
            self.current_time = float(session.time[0])
        self.rebuild_scene()

    def set_parameter_ensemble(self, ensemble: SharedPosterior | None) -> None:
        self.parameter_ensemble = ensemble
        if ensemble is None or ensemble.size == 0:
            self.selected_member_id = None
        elif self.selected_member_id not in set(ensemble.member_id.tolist()):
            self.selected_member_id = int(ensemble.member_id[0])
        self.rebuild_scene(preserve_camera=True)

    def set_selected_member(self, member_id: int | None) -> None:
        self.selected_member_id = None if member_id is None else int(member_id)
        self.rebuild_scene()

    def set_current_time(self, value: float) -> None:
        self.current_time = float(value)
        self._update_markers()

    def set_view_mode(self, mode: str) -> None:
        if mode not in {"world", "correction"}:
            raise ValueError(f"Unknown 3D view mode: {mode}")
        if mode == self.view_mode:
            return
        self.view_mode = mode
        self.rebuild_scene()

    def set_layer_visible(self, layer: str, visible: bool) -> None:
        if layer not in self.layers:
            return
        self.layers[layer] = bool(visible)
        self.rebuild_scene(preserve_camera=True)

    def reset_camera(self) -> None:
        if self.plotter is None or self.session is None:
            return
        if self.view_mode == "world":
            observed_path = self.session.observed_position
        elif self.session.observed_correction_translation is not None:
            observed_path = self.session.observed_correction_translation
        else:
            observed_path = self.session.observed_position
        self._reset_camera(observed_path)
        self.plotter.reset_camera_clipping_range()
        self.plotter.render()

    def rebuild_scene(self, preserve_camera: bool = False) -> None:
        if self.plotter is None:
            return

        camera_state = self._capture_camera_state() if preserve_camera else None

        self.plotter.clear()
        self._marker_actors.clear()
        self._frame_actor_names.clear()
        self._full_path_states.clear()
        self._path_states.clear()
        self._last_path_index = -1
        self.plotter.add_axes()

        if self.session is None:
            self.plotter.add_text("No bag selected", position="upper_left", font_size=12)
            return

        session = self.session
        has_posterior = (
            self.parameter_ensemble is not None
            and session.member_position is not None
            and session.nominal_position is not None
        )
        selected_index = self._member_index() if has_posterior else 0
        center_position = (
            np.mean(session.member_position, axis=0) if has_posterior else None
        )

        if self.view_mode == "world":
            reference_path = session.reference_position
            observed_path = session.observed_position
            nominal_path = session.nominal_position
            member_paths = session.member_position
            center_path = center_position
        else:
            if session.observed_correction_translation is None:
                self.plotter.add_text(
                    "Correction paths are available after smoothing",
                    position="upper_left",
                    font_size=12,
                )
                return
            reference_path = np.zeros_like(session.observed_correction_translation)
            observed_path = session.observed_correction_translation
            nominal_path = np.zeros_like(observed_path)
            member_paths = session.correction_translation
            center_path = (
                None if member_paths is None else np.mean(member_paths, axis=0)
            )

        if self.layers["reference"] and self.view_mode == "world":
            self._add_path(reference_path, "reference", "#aaaaaa", width=2.0)
        if self.layers["observed"]:
            self._add_path(observed_path, "observed", "#3d8bf2", width=4.0)
        if self.layers["nominal"] and nominal_path is not None:
            self._add_path(nominal_path, "nominal", "#ec8a2f", width=3.0)
        if self.layers["posterior"] and member_paths is not None and center_path is not None:
            sample_indices = self._display_member_indices(selected_index)
            for member_index in sample_indices:
                self._add_path(
                    member_paths[member_index],
                    f"member-{member_index}",
                    "#69b98a",
                    width=1.0,
                    opacity=0.22,
                )
            self._add_path(center_path, "posterior-center", "#3db879", width=3.0)
        if self.layers["selected"] and member_paths is not None:
            self._add_path(
                member_paths[selected_index],
                "selected-member",
                "#d45bd4",
                width=4.0,
            )

        sphere = pv.Sphere(radius=max(self._scene_span(observed_path) * 0.012, 0.015))
        if self.layers["observed"]:
            self._marker_actors["observed"] = self.plotter.add_mesh(
                sphere.copy(), color="#3d8bf2", name="marker-observed", render=False
            )
        if self.layers["nominal"] and nominal_path is not None:
            self._marker_actors["nominal"] = self.plotter.add_mesh(
                sphere.copy(), color="#ec8a2f", name="marker-nominal", render=False
            )
        if self.layers["selected"] and member_paths is not None:
            self._marker_actors["selected"] = self.plotter.add_mesh(
                sphere.copy(), color="#d45bd4", name="marker-selected", render=False
            )

        title = "World trajectory" if self.view_mode == "world" else "Nominal-to-real correction"
        self.plotter.add_text(
            title,
            position="upper_left",
            font_size=11,
            name="scene-title",
            render=False,
        )
        self.plotter.add_text(
            "thin dashed: full path  |  solid: elapsed",
            position="upper_right",
            font_size=9,
            name="path-style-label",
            render=False,
        )
        if camera_state is None:
            self._reset_camera(observed_path)
        else:
            self._restore_camera_state(camera_state)
        self._update_markers()

    def _add_path(
        self,
        points: np.ndarray,
        name: str,
        color: str,
        width: float,
        opacity: float = 1.0,
    ) -> None:
        assert self.plotter is not None and pv is not None
        full_poly = _segmented_polyline(points)
        elapsed_poly = _segmented_polyline(points)
        if full_poly is None or elapsed_poly is None:
            return

        rgb = np.asarray(pv.Color(color).int_rgb, dtype=np.uint8)
        full_rgba = np.empty((full_poly.n_cells, 4), dtype=np.uint8)
        full_rgba[:, :3] = rgb
        self._set_dashed_path_alpha(full_rgba, opacity)
        full_poly.cell_data["full_path_rgba"] = full_rgba
        mapped_full_rgba = full_poly.cell_data["full_path_rgba"]
        self.plotter.add_mesh(
            full_poly,
            scalars="full_path_rgba",
            rgb=True,
            preference="cell",
            show_scalar_bar=False,
            line_width=max(1.0, 0.55 * width),
            render_lines_as_tubes=False,
            lighting=False,
            name=f"{name}-full-path",
            render=False,
        )
        self._full_path_states.append((full_poly, mapped_full_rgba))

        elapsed_rgba = np.empty((elapsed_poly.n_cells, 4), dtype=np.uint8)
        elapsed_rgba[:, :3] = rgb
        self._set_elapsed_path_alpha(
            elapsed_rgba,
            opacity,
            self._current_sample_index(),
        )
        elapsed_poly.cell_data["elapsed_path_rgba"] = elapsed_rgba
        mapped_elapsed_rgba = elapsed_poly.cell_data["elapsed_path_rgba"]
        self.plotter.add_mesh(
            elapsed_poly,
            scalars="elapsed_path_rgba",
            rgb=True,
            preference="cell",
            show_scalar_bar=False,
            line_width=width,
            render_lines_as_tubes=True,
            lighting=False,
            name=name,
            render=False,
        )
        self._path_states.append((elapsed_poly, mapped_elapsed_rgba, opacity))

    def _update_markers(self) -> None:
        if self.plotter is None or self.session is None:
            return
        camera_state = self._capture_camera_state()
        session = self.session
        index = self._current_sample_index()
        self._update_path_progress(index)
        has_posterior = (
            self.parameter_ensemble is not None
            and session.member_position is not None
            and session.nominal_position is not None
        )
        selected_index = self._member_index() if has_posterior else 0

        if self.view_mode == "world":
            observed_position = session.observed_position[index]
            observed_rpy = session.observed_rpy[index]
            nominal_position = (
                None if session.nominal_position is None else session.nominal_position[index]
            )
            nominal_rpy_array = session.nominal_rpy
            nominal_rpy = None if nominal_rpy_array is None else nominal_rpy_array[index]
            member_rpy_array = session.member_rpy
            selected_position = (
                None if session.member_position is None
                else session.member_position[selected_index, index]
            )
            selected_rpy = (
                None if member_rpy_array is None else member_rpy_array[selected_index, index]
            )
        else:
            if (
                session.observed_correction_translation is None
                or session.observed_correction_rotation_vector is None
            ):
                return
            observed_position = session.observed_correction_translation[index]
            nominal_position = np.zeros(3, dtype=float)
            selected_position = None if session.correction_translation is None else (
                session.correction_translation[selected_index, index]
            )
            observed_rpy = _rotvec_to_rpy(
                session.observed_correction_rotation_vector[index]
            )
            nominal_rpy = np.zeros(3, dtype=float)
            selected_rpy = None if session.correction_rotation_vector is None else _rotvec_to_rpy(
                session.correction_rotation_vector[selected_index, index]
            )

        marker_data = {"observed": observed_position}
        if nominal_position is not None:
            marker_data["nominal"] = nominal_position
        if selected_position is not None:
            marker_data["selected"] = selected_position
        for name, position in marker_data.items():
            actor = self._marker_actors.get(name)
            if actor is not None:
                actor.SetPosition(*(float(value) for value in position))

        for actor_name in self._frame_actor_names:
            self.plotter.remove_actor(actor_name, render=False)
        self._frame_actor_names.clear()

        span = self._scene_span(session.observed_position)
        axis_scale = max(span * 0.045, 0.08)
        if self.layers["observed"]:
            self._add_frame(observed_position, observed_rpy, axis_scale, "observed-frame")
        if self.layers["nominal"] and nominal_position is not None and nominal_rpy is not None:
            self._add_frame(nominal_position, nominal_rpy, axis_scale, "nominal-frame")
        if self.layers["selected"] and selected_position is not None and selected_rpy is not None:
            self._add_frame(selected_position, selected_rpy, axis_scale, "selected-frame")

        self.plotter.add_text(
            "t = {:.3f} s{}".format(
                session.time[index],
                "" if self.selected_member_id is None else " | member {}".format(self.selected_member_id),
            ),
            position="lower_left",
            font_size=10,
            name="time-label",
            render=False,
        )
        self._restore_camera_state(camera_state)
        self.plotter.render()

    def _capture_camera_state(
        self,
    ) -> tuple[
        tuple[tuple[float, ...], ...],
        float,
        float,
        bool,
        tuple[float, float],
    ]:
        assert self.plotter is not None
        camera_position = tuple(
            tuple(float(value) for value in vector)
            for vector in self.plotter.camera_position
        )
        return (
            camera_position,
            float(self.plotter.camera.parallel_scale),
            float(self.plotter.camera.view_angle),
            bool(self.plotter.camera.parallel_projection),
            tuple(float(value) for value in self.plotter.camera.clipping_range),
        )

    def _restore_camera_state(
        self,
        state: tuple[
            tuple[tuple[float, ...], ...],
            float,
            float,
            bool,
            tuple[float, float],
        ],
    ) -> None:
        assert self.plotter is not None
        (
            camera_position,
            parallel_scale,
            view_angle,
            parallel_projection,
            clipping_range,
        ) = state
        self.plotter.camera_position = camera_position
        self.plotter.camera.parallel_scale = parallel_scale
        self.plotter.camera.view_angle = view_angle
        self.plotter.camera.parallel_projection = parallel_projection
        self.plotter.camera.clipping_range = clipping_range

    def _current_sample_index(self) -> int:
        if self.session is None:
            return 0
        return int(np.argmin(np.abs(self.session.time - self.current_time)))

    def _update_path_progress(self, current_index: int) -> None:
        if current_index == self._last_path_index:
            return
        for poly, rgba, opacity in self._path_states:
            self._set_elapsed_path_alpha(rgba, opacity, current_index)
            vtk_array = getattr(rgba, "VTKObject", None)
            if vtk_array is not None:
                vtk_array.Modified()
            poly.GetCellData().Modified()
            poly.Modified()
        self._last_path_index = current_index

    @staticmethod
    def _set_dashed_path_alpha(
        rgba: np.ndarray,
        opacity: float,
    ) -> None:
        """Draw one stable, thin dashed guide over the complete trajectory."""
        segment_count = int(rgba.shape[0])
        if segment_count == 0:
            return

        alpha = int(round(255.0 * float(np.clip(opacity, 0.0, 1.0))))
        dash_span = max(2, int(np.ceil(segment_count / 120.0)))
        segment_indices = np.arange(segment_count, dtype=np.int64)
        dash_mask = (segment_indices // dash_span) % 2 == 0
        rgba[:, 3] = 0
        rgba[dash_mask, 3] = int(round(0.62 * alpha))

    @staticmethod
    def _set_elapsed_path_alpha(
        rgba: np.ndarray,
        opacity: float,
        current_index: int,
    ) -> None:
        """Draw the elapsed portion as a solid overlay on the full guide."""
        segment_count = int(rgba.shape[0])
        if segment_count == 0:
            return

        alpha = int(round(255.0 * float(np.clip(opacity, 0.0, 1.0))))
        elapsed_count = int(np.clip(current_index, 0, segment_count))
        rgba[:, 3] = 0
        rgba[:elapsed_count, 3] = alpha

    def _add_frame(
        self,
        origin: np.ndarray,
        rpy: np.ndarray,
        scale: float,
        prefix: str,
    ) -> None:
        assert self.plotter is not None and pv is not None
        rotation = _rpy_matrix(rpy)
        colors = ("#e35d5d", "#63c174", "#5a86e8")
        for axis in range(3):
            name = f"{prefix}-{axis}"
            arrow = pv.Arrow(
                start=np.asarray(origin, dtype=float),
                direction=rotation[:, axis],
                scale=scale,
                tip_length=0.22,
                tip_radius=0.07,
                shaft_radius=0.025,
            )
            self.plotter.add_mesh(arrow, color=colors[axis], name=name, render=False)
            self._frame_actor_names.append(name)

    def _display_member_indices(self, selected_index: int) -> np.ndarray:
        if self.parameter_ensemble is None:
            return np.empty(0, dtype=int)
        count = self.parameter_ensemble.size
        display_count = min(20, count)
        indices = np.linspace(0, count - 1, display_count, dtype=int)
        if selected_index not in indices:
            indices = np.unique(np.append(indices, selected_index))
        return indices

    def _member_index(self) -> int:
        if self.parameter_ensemble is None or self.selected_member_id is None:
            return 0
        matches = np.flatnonzero(self.parameter_ensemble.member_id == self.selected_member_id)
        return int(matches[0]) if matches.size else 0

    @staticmethod
    def _scene_span(points: np.ndarray) -> float:
        extent = np.max(points, axis=0) - np.min(points, axis=0)
        return float(max(np.max(extent), 1.0e-3))

    def _reset_camera(self, observed_path: np.ndarray) -> None:
        if self.plotter is None:
            return
        minimum = np.min(observed_path, axis=0)
        maximum = np.max(observed_path, axis=0)
        center = 0.5 * (minimum + maximum)
        span = max(float(np.max(maximum - minimum)), 0.5) * 1.18
        half = 0.5 * span
        bounds = (
            float(center[0] - half),
            float(center[0] + half),
            float(center[1] - half),
            float(center[1] + half),
            float(center[2] - half),
            float(center[2] + half),
        )
        self.plotter.reset_camera(bounds=bounds, render=False)
        self.plotter.view_isometric()

    def close_scene(self) -> None:
        if self.plotter is not None:
            self.plotter.close()


class PidComparisonScene3DWidget(QWidget):
    """Current/selected PID trajectory comparison from one evaluation bundle."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.evaluation: PidProposalEvaluation | None = None
        self.selected_bag_id: str | None = None
        self.selected_member_id: int | None = None
        self.selected_candidate_id: str | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        controls = QHBoxLayout()
        self.status_label = QLabel("Load an evaluation and select a source bag/member.")
        self.status_label.setWordWrap(True)
        controls.addWidget(self.status_label, 1)
        self.view_combo = QComboBox()
        self.view_combo.addItem("Trajectory cloud", "trajectory")
        self.view_combo.addItem("Correction translation cloud", "translation")
        self.view_combo.addItem("Correction rotation-vector cloud", "rotation")
        self.view_combo.currentIndexChanged.connect(
            lambda _index: self.rebuild_scene()
        )
        controls.addWidget(self.view_combo)
        layout.addLayout(controls)
        if QtInteractor is None or pv is None:
            self.plotter = None
            self.status_label.setText(
                "PyVistaQt is not installed. The selected real artifact paths "
                "remain available in the correction plots."
            )
        else:
            self.plotter = QtInteractor(self)
            layout.addWidget(self.plotter.interactor, 1)
            self.plotter.set_background("#20242b")
            self.plotter.add_axes()

    def set_evaluation(self, evaluation: PidProposalEvaluation | None) -> None:
        self.evaluation = evaluation
        self.rebuild_scene()

    def set_selection(
        self,
        bag_id: str | None,
        member_id: int | None,
        candidate_id: str | None,
    ) -> None:
        self.selected_bag_id = None if bag_id is None else str(bag_id)
        self.selected_member_id = None if member_id is None else int(member_id)
        self.selected_candidate_id = (
            None if candidate_id is None else str(candidate_id)
        )
        self.rebuild_scene()

    def _selection_indices(self) -> tuple[object, int, int, int] | None:
        if (
            self.evaluation is None
            or self.selected_bag_id is None
            or self.selected_member_id is None
            or self.selected_candidate_id is None
        ):
            return None
        arrays = self.evaluation.bags.get(self.selected_bag_id)
        if arrays is None:
            return None
        members = np.asarray(arrays["member_id"])
        candidates = np.asarray(arrays["candidate_id"]).astype(str)
        member = np.flatnonzero(members == self.selected_member_id)
        current = np.flatnonzero(candidates == "current")
        selected = np.flatnonzero(candidates == self.selected_candidate_id)
        if not member.size or not current.size or not selected.size:
            return None
        return arrays, int(member[0]), int(current[0]), int(selected[0])

    def rebuild_scene(self) -> None:
        selection = self._selection_indices()
        if selection is None:
            self.status_label.setText(
                "The current GUI bag/member/candidate selection is not present "
                "in the loaded evaluation."
            )
            if self.plotter is not None:
                self.plotter.clear()
                self.plotter.add_axes()
                self.plotter.add_text(
                    "No exact evaluation selection",
                    position="upper_left",
                    font_size=11,
                )
            return
        arrays, member_index, current_index, selected_index = selection
        success = np.asarray(arrays["forecast_success"])
        reference = np.asarray(arrays["reference_position"], dtype=float)
        prediction = np.asarray(arrays["prediction_position"], dtype=float)
        view_mode = str(self.view_combo.currentData())
        if view_mode == "trajectory":
            paths = prediction
        elif view_mode == "translation":
            paths = np.asarray(arrays["correction_translation"], dtype=float)
        else:
            paths = np.asarray(arrays["correction_rotation_vector"], dtype=float)
        current_ok = bool(success[current_index, member_index])
        selected_ok = bool(success[selected_index, member_index])
        self.status_label.setText(
            "Bag {} | member {} | {} | current={} | {}={}".format(
                self.selected_bag_id,
                self.selected_member_id,
                self.view_combo.currentText(),
                "complete" if current_ok else "failed",
                self.selected_candidate_id,
                "complete" if selected_ok else "failed",
            )
        )
        if self.plotter is None:
            return
        self.plotter.clear()
        self.plotter.add_axes()
        if view_mode == "trajectory":
            self._add_path(reference, "Reference", "#aaaaaa", 2.0)
        for raw_member_index in np.flatnonzero(success[current_index]):
            if int(raw_member_index) != member_index:
                self._add_path(
                    paths[current_index, raw_member_index],
                    "current-member-{}".format(int(raw_member_index)),
                    "#3d8bf2",
                    1.0,
                    opacity=0.16,
                    legend=False,
                )
        if selected_index != current_index:
            for raw_member_index in np.flatnonzero(success[selected_index]):
                if int(raw_member_index) != member_index:
                    self._add_path(
                        paths[selected_index, raw_member_index],
                        "candidate-member-{}".format(int(raw_member_index)),
                        "#d45bd4",
                        1.0,
                        opacity=0.16,
                        legend=False,
                    )
        if current_ok:
            self._add_path(
                paths[current_index, member_index],
                "Current PID selected member",
                "#3d8bf2",
                4.0,
            )
        if selected_ok and selected_index != current_index:
            self._add_path(
                paths[selected_index, member_index],
                "Candidate selected member",
                "#d45bd4",
                4.0,
            )
        self.plotter.add_text(
            "Exact posterior-predictive {}".format(
                self.view_combo.currentText().lower()
            ),
            position="upper_left",
            font_size=11,
            name="pid-comparison-title",
        )
        legend_entries = [
            ("Current PID ensemble / selected", "#3d8bf2"),
        ]
        if selected_index != current_index:
            legend_entries.append(
                ("Candidate ensemble / selected", "#d45bd4")
            )
        if view_mode == "trajectory":
            legend_entries.insert(0, ("Reference", "#aaaaaa"))
        self.plotter.add_legend(
            legend_entries,
            bcolor="#20242b",
        )
        self.plotter.reset_camera()
        self.plotter.view_isometric()
        self.plotter.render()

    def _add_path(
        self,
        points: np.ndarray,
        label: str,
        color: str,
        width: float,
        *,
        opacity: float = 1.0,
        legend: bool = True,
    ) -> None:
        assert self.plotter is not None
        polyline = _segmented_polyline(points)
        if polyline is None:
            return
        self.plotter.add_mesh(
            polyline,
            color=color,
            line_width=width,
            render_lines_as_tubes=True,
            lighting=False,
            opacity=opacity,
            label=label if legend else None,
        )

    def close_scene(self) -> None:
        if self.plotter is not None:
            self.plotter.close()
