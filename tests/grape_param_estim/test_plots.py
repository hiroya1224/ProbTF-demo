import importlib.util
import unittest

from test_data import make_recording
from test_model import make_analysis, nominal_parameters


@unittest.skipUnless(
    importlib.util.find_spec("plotly") is not None,
    "Plotly is an optional GUI dependency",
)
class PlotTest(unittest.TestCase):
    def test_phase_zero_figures_contain_selected_data(self):
        from grape_param_estim.plots import (
            make_bag_overview_figure,
            make_command_figure,
            make_state_figure,
            make_trajectory_figure,
        )

        recording = make_recording()
        data = recording.select_interval(0.2, 1.8, 0.5)

        overview = make_bag_overview_figure(recording, (0.2, 1.8))
        trajectory = make_trajectory_figure(data)
        state = make_state_figure(data)
        command = make_command_figure(data)

        self.assertEqual(len(overview.data), 13)
        self.assertEqual(len(trajectory.data), data.segment_count)
        self.assertEqual(len(state.data), 13)
        self.assertEqual(len(command.data), 12)

    def test_phase_one_figures_show_replay_and_residuals(self):
        from grape_param_estim.model import (
            GrapeRigidBodyModel,
            replay_segments,
        )
        from grape_param_estim.plots import (
            make_correction_figure,
            make_replay_pose_figure,
            make_replay_trajectory_figure,
            make_segment_residual_figure,
        )

        data = make_analysis(two_segments=True)
        replay = replay_segments(
            data, GrapeRigidBodyModel(), nominal_parameters()
        )

        trajectory = make_replay_trajectory_figure(data, replay)
        pose = make_replay_pose_figure(data, replay)
        correction = make_correction_figure(data, replay)
        residual = make_segment_residual_figure(data, replay)

        self.assertEqual(len(trajectory.data), 2 * data.segment_count)
        self.assertEqual(
            tuple(trajectory.layout.scene.xaxis.range), (-0.12, 0.12)
        )
        self.assertEqual(len(pose.data), 12)
        self.assertEqual(len(correction.data), 6)
        self.assertEqual(len(residual.data), 2 * data.segment_count)


if __name__ == "__main__":
    unittest.main()
