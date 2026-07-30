import importlib.util
import unittest

from test_data import make_recording


@unittest.skipUnless(
    importlib.util.find_spec("plotly") is not None,
    "Plotly is an optional GUI dependency",
)
class PlotTest(unittest.TestCase):
    def test_phase_zero_figures_contain_selected_data(self):
        from grape_param_estim.plots import (
            make_command_figure,
            make_state_figure,
            make_trajectory_figure,
        )

        data = make_recording().select_interval(0.2, 1.8, 0.5)

        trajectory = make_trajectory_figure(data)
        state = make_state_figure(data)
        command = make_command_figure(data)

        self.assertEqual(len(trajectory.data), data.segment_count)
        self.assertEqual(len(state.data), 13)
        self.assertEqual(len(command.data), 8)


if __name__ == "__main__":
    unittest.main()
