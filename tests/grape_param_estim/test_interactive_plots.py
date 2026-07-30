import unittest

from matplotlib.figure import Figure

from grape_param_estim.interactive_plots import (
    draw_parameter_trace,
    draw_timeline,
    parameter_rows,
)


def _episode():
    prefix = "specific_force_x"
    return {
        "episode_index": 0,
        "start_s": 1.0,
        "end_s": 4.0,
        "liftoff_s": 2.0,
        "support": {"height_m": 0.5},
        "selection": {
            "fit_intervals": [{"start_s": 2.0, "end_s": 3.5}],
            "failure_diagnostic_intervals": [
                {
                    "start_s": 1.0,
                    "end_s": 2.0,
                    "reason": "controlled_supported",
                }
            ],
        },
        "parameter_trace": [
            {
                "sequence_time_s": 2.5,
                "parameters": {
                    "{}_bias".format(prefix): 0.1,
                    "{}_gain".format(prefix): 1.2,
                    "{}_velocity_feedback".format(prefix): -0.2,
                },
            },
            {
                "sequence_time_s": 3.5,
                "parameters": {
                    "{}_bias".format(prefix): 0.2,
                    "{}_gain".format(prefix): 1.3,
                    "{}_velocity_feedback".format(prefix): -0.1,
                },
            },
        ],
        "estimate": {
            "parameters": {
                "{}_bias".format(prefix): {
                    "estimate": 0.2,
                    "ci95": [0.1, 0.3],
                },
                "{}_gain".format(prefix): {
                    "estimate": 1.3,
                    "ci95": [1.1, 1.5],
                },
                "{}_velocity_feedback".format(prefix): {
                    "estimate": -0.1,
                    "ci95": [-0.2, 0.0],
                },
            },
            "channels": {
                "x": {
                    "gain_parameter": "{}_gain".format(prefix),
                    "information_grade": "informative",
                }
            },
        },
    }


def _bag():
    return {
        "path": "/tmp/trial.bag",
        "episodes": [_episode()],
        "plot": {
            "time_s": [0.0, 1.0, 2.0, 3.0, 4.0],
            "z_m": [0.5, 0.5, 0.5, 0.7, 0.8],
            "speed_m_s": [0.0, 0.0, 0.1, 0.2, 0.1],
            "specific_force_norm_m_s2": [9.8] * 5,
            "angular_velocity_norm_rad_s": [0.0] * 5,
            "vertical_command": [None, 0.0, 10.0, 10.0, 0.0],
            "flight_state": [0, 3, 3, 3, 6],
        },
    }


class InteractivePlotTests(unittest.TestCase):
    def test_timeline_has_six_linked_signal_axes(self):
        figure = Figure()

        draw_timeline(figure, _bag())

        self.assertEqual(len(figure.axes), 6)
        self.assertEqual(figure.axes[-1].get_xlabel(), "bag-local time [s]")
        self.assertTrue(figure.axes[0].patches)

    def test_parameter_trace_and_final_parameter_rows(self):
        figure = Figure()
        result = {"bags": [_bag()]}

        draw_parameter_trace(figure, result)
        rows = parameter_rows(_episode())

        self.assertEqual(len(figure.axes), 6)
        self.assertTrue(figure.axes[0].lines)
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[1][1], "specific_force_x_gain")
        self.assertEqual(rows[1][-1], "informative")


if __name__ == "__main__":
    unittest.main()
