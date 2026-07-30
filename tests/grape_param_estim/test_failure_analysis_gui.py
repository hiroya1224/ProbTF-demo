from pathlib import Path
import unittest

from grape_param_estim.failure_analysis_gui import FailureAnalysisApp


class _TextValue:
    def __init__(self):
        self.value = None

    def set(self, value):
        self.value = value


class FailureAnalysisGuiHelperTests(unittest.TestCase):
    def test_duration_and_advice_status_are_user_facing(self):
        self.assertEqual(
            FailureAnalysisApp._format_duration(65.0), "1分05秒"
        )
        self.assertEqual(
            FailureAnalysisApp._advice_status_text(
                "proposal_available"
            ),
            "提案あり",
        )
        self.assertEqual(
            FailureAnalysisApp._advice_status_text(
                "nominal_within_uncertainty"
            ),
            "現状維持",
        )

    def test_eta_uses_observed_fraction_and_is_determinate(self):
        path = Path("/tmp/trial.bag")
        app = object.__new__(FailureAnalysisApp)
        app._run_prefix = {path: 0}
        app._run_sizes = {path: 100}
        app._run_total_size = 100
        app._run_started = 100.0
        app.progress = {}
        app.progress_text = _TextValue()

        app._update_eta(
            path,
            bag_fraction=0.25,
            phase="block bootstrap",
            now=105.0,
        )

        self.assertAlmostEqual(app.progress["value"], 25.0)
        self.assertEqual(
            app.progress_text.value,
            "25% / 残り約 15秒 / block bootstrap",
        )


if __name__ == "__main__":
    unittest.main()
