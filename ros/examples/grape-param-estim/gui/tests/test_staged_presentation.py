import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtWidgets import QApplication

from grape_param_estim_gui.artifact_loader import (
    AssimilationRun,
    FlightResult,
    SharedPosterior,
)
from grape_param_estim_gui.project_io import new_project_manifest
from grape_param_estim_gui.state import BagRecord, ProjectStore
from grape_param_estim_gui.widgets.bag_browser import BagBrowserView
from grape_param_estim_gui.widgets.master_view import MasterView
from grape_param_estim_gui.widgets.scene_3d import Scene3DWidget


STAGE2_SCHEMA = "grape-param-estim/fixed-q-augmented-parameter-estimate/v1"


def _shared_posterior() -> SharedPosterior:
    return SharedPosterior(
        member_id=np.asarray((7, 11), dtype=np.int64),
        parameter_coordinate=np.zeros((2, 19)),
        mass=np.asarray((2.0, 2.2)),
        inertia=np.asarray((np.eye(3), 1.1 * np.eye(3))),
        cog=np.zeros((2, 3)),
        force_effectiveness=np.asarray(((0.9,) * 4, (1.0,) * 4)),
        torque_effectiveness=np.asarray(((0.9,) * 4, (1.0,) * 4)),
        constant_delay=np.asarray((0.01, 0.02)),
        ridge={},
        mode={},
        iteration_diagnostics={},
    )


def _flight_result(*, correction_available: bool = True) -> FlightResult:
    times = np.asarray((0.0, 1.0))
    quaternion = np.tile(
        np.asarray((0.0, 0.0, 0.0, 1.0)), (times.size, 1)
    )
    member_quaternion = np.tile(quaternion, (2, 1, 1))
    correction = np.zeros((times.size, 3)) if correction_available else None
    return FlightResult(
        bag_id="bag-a",
        time=times,
        record_time=times + 100.0,
        reference_position=np.zeros((times.size, 3)),
        reference_rpy=np.zeros((times.size, 3)),
        observed_position=np.zeros((times.size, 3)),
        observed_orientation_xyzw=quaternion,
        nominal_position=np.zeros((times.size, 3)),
        nominal_orientation_xyzw=quaternion,
        member_position=np.zeros((2, times.size, 3)),
        member_orientation_xyzw=member_quaternion,
        correction_translation=(
            np.zeros((2, times.size, 3))
            if correction_available
            else None
        ),
        correction_rotation_vector=(
            np.zeros((2, times.size, 3))
            if correction_available
            else None
        ),
        observed_correction_translation=correction,
        observed_correction_rotation_vector=correction,
        residual_wrench=np.zeros((2, times.size, 6)),
        flight_state=None,
        q_resolution_sufficient=None,
        provenance={},
        calibration={
            "fixed_q_stationary_variance": np.arange(1.0, 7.0),
        },
        coverage={},
        objective_contribution=-9.0,
    )


class StagedPresentationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.application = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        manifest = new_project_manifest("staged-presentation")
        manifest["bags"] = [
            {
                "bag_id": "bag-a",
                "source_path": "/source/bag-a.bag",
                "relative_path": "bags/bag-a.bag",
                "sha256": "a" * 64,
            }
        ]
        self.store = ProjectStore(self.root, manifest)
        self.record = BagRecord(
            bag_id="bag-a",
            path=self.root / "bag-a.bag",
            source_path=Path("/source/bag-a.bag"),
            sha256="a" * 64,
            inspection={"warnings": [], "topic_contract": {}},
            included=True,
            auto_interval=(0.0, 1.0),
            selected_interval=(0.0, 1.0),
            status="ready",
            configuration_fingerprint="complete:shared",
        )
        self.store.add(self.record)
        flight = _flight_result()
        run = AssimilationRun(
            root=self.root / "stage2",
            manifest={
                "schema": STAGE2_SCHEMA,
                "status": "complete",
                "run_id": "stage2-run",
                "project_request_fingerprint": self.store.request_fingerprint(),
            },
            shared_posterior=_shared_posterior(),
            bag_results={"bag-a": flight},
            diagnostics={},
            warnings=(),
        )
        self.store.apply_assimilation(run)

    def tearDown(self):
        self.temporary.cleanup()

    def test_master_view_presents_stage2_values_without_legacy_diagnostics(self):
        view = MasterView(self.store)
        diagnostic = view.diagnostic_label.text()

        self.assertIn("Fixed-Q staged", view.diagnostic_group.title())
        self.assertNotIn("IEnKS-Q", view.diagnostic_group.title())
        self.assertIn(
            "body order [Fx, Fy, Fz, tau_x, tau_y, tau_z]",
            diagnostic,
        )
        self.assertIn("[1., 2., 3., 4., 5., 6.]", diagnostic)
        self.assertIn("artifact status: complete", diagnostic)
        self.assertIn("convergence: not reported", diagnostic)
        self.assertIn("iterations: not reported", diagnostic)
        self.assertIn("Q time resolution: not applicable", diagnostic)
        self.assertIn("correction-path coverage: not reported", diagnostic)
        self.assertNotIn("insufficient", diagnostic)
        self.assertEqual(
            view.bag_table.horizontalHeaderItem(8).text(), "Log likelihood"
        )
        self.assertEqual(view.bag_table.item(0, 9).text(), "not reported")
        self.assertEqual(view.iteration_plot.listDataItems(), [])
        view.close()

    def test_bag_browser_does_not_treat_absent_q_resolution_as_failure(self):
        with mock.patch(
            "grape_param_estim_gui.widgets.scene_3d.pv", None
        ), mock.patch(
            "grape_param_estim_gui.widgets.scene_3d.QtInteractor", None
        ):
            view = BagBrowserView(self.store)

        self.assertIn("staged parameter estimation", view.include_checkbox.text())
        self.assertIn(
            "residual-wrench Q time resolution: not applicable",
            view.inspection_details.text(),
        )
        self.assertNotIn("insufficient", view.inspection_details.text())
        view.close()

    def test_missing_correction_paths_use_staged_estimation_wording(self):
        with mock.patch(
            "grape_param_estim_gui.widgets.scene_3d.pv", None
        ), mock.patch(
            "grape_param_estim_gui.widgets.scene_3d.QtInteractor", None
        ):
            scene = Scene3DWidget(None)
        scene.plotter = mock.Mock()
        scene.session = _flight_result(correction_available=False)
        scene.view_mode = "correction"

        scene.rebuild_scene()

        message = scene.plotter.add_text.call_args.args[0]
        self.assertIn("staged parameter estimation", message)
        self.assertNotIn("smoothing", message)
        scene.close()


if __name__ == "__main__":
    unittest.main()
