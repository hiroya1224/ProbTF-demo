from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

try:
    from PySide6.QtCore import QCoreApplication, QEventLoop, QTimer
except ImportError as error:  # exactly the supported optional-test guard
    raise unittest.SkipTest("PySide6 is unavailable: {}".format(error))

from grape_param_estim_gui.process_runner import EstimatorProcessRunner


def _event(run_id="run-a", completed=1, total=2, eta=1.0):
    return {
        "schema": "grape-param-estim/progress-event/v1",
        "run_id": run_id,
        "stage_id": "forecast",
        "stage_label": "Forecast",
        "completed_units": completed,
        "total_units": total,
        "fraction": completed / total,
        "elapsed_seconds": 0.1,
        "eta_seconds": eta,
        "iteration": None,
        "maximum_iterations": None,
        "bag_id": None,
        "member_id": None,
        "message": "working",
    }


class ProcessRunnerQtTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.application = QCoreApplication.instance() or QCoreApplication([])

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def _run_until_terminal(self, runner, timeout_ms=3000):
        loop = QEventLoop()
        result = []
        runner.finished.connect(lambda path: (result.append(("finished", path)), loop.quit()))
        runner.failed.connect(lambda message: (result.append(("failed", message)), loop.quit()))
        runner.cancelled.connect(lambda: (result.append(("cancelled", None)), loop.quit()))
        QTimer.singleShot(timeout_ms, loop.quit)
        loop.exec()
        self.assertTrue(result, "runner did not emit a terminal signal")
        return result[0]

    def test_stdout_is_progress_only_and_stderr_is_log(self):
        worker = self.root / "worker.py"
        worker.write_text(
            "import json,sys\n"
            "events=" + repr([_event(completed=1), _event(completed=2, eta=0.0)]) + "\n"
            "[print(json.dumps(v), flush=True) for v in events]\n"
            "print('diagnostic log', file=sys.stderr, flush=True)\n"
        )
        runner = EstimatorProcessRunner()
        progress = []
        logs = []
        runner.progress.connect(progress.append)
        runner.stderrLog.connect(logs.append)
        runner.start(sys.executable, [worker], output_directory=self.root / "out", run_id="run-a")
        terminal = self._run_until_terminal(runner)
        self.assertEqual(terminal[0], "finished")
        self.assertEqual([value.fraction for value in progress], [0.5, 1.0])
        self.assertIn("diagnostic log", logs)

    def test_non_json_stdout_is_protocol_error(self):
        worker = self.root / "worker.py"
        worker.write_text("print('ordinary log on stdout', flush=True)\n")
        runner = EstimatorProcessRunner()
        runner.start(sys.executable, [worker], output_directory=self.root / "out", run_id="run-a")
        terminal = self._run_until_terminal(runner)
        self.assertEqual(terminal[0], "failed")
        self.assertIn("progress JSONL", terminal[1])

    def test_cancel_escalates_from_cooperative_request_to_terminate_and_kill(self):
        worker = self.root / "worker.py"
        worker.write_text(
            "import signal,time\n"
            "signal.signal(signal.SIGINT, signal.SIG_IGN)\n"
            "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
            "while True: time.sleep(0.05)\n"
        )
        runner = EstimatorProcessRunner(terminate_after_ms=40, kill_after_ms=40)
        runner.start(sys.executable, [worker], output_directory=self.root / "out", run_id="run-a")
        QTimer.singleShot(100, runner.request_cancel)
        with mock.patch(
            "grape_param_estim_gui.process_runner.finalize_cancelled_bundle",
            return_value=True,
        ) as finalise:
            terminal = self._run_until_terminal(runner)
        self.assertEqual(terminal[0], "cancelled")
        finalise.assert_called_once_with(self.root / "out", "user_requested")

    @unittest.skipUnless(sys.platform != "win32", "POSIX signal test")
    def test_cancel_reaches_worker_sigint_handler_before_grace_timeout(self):
        marker = self.root / "cancelled.txt"
        worker = self.root / "cooperative_worker.py"
        worker.write_text(
            "import pathlib,signal,time\n"
            "marker=pathlib.Path(" + repr(str(marker)) + ")\n"
            "def cancel(_signum,_frame):\n"
            " marker.write_text('sigint')\n"
            " raise SystemExit(130)\n"
            "signal.signal(signal.SIGINT,cancel)\n"
            "while True: time.sleep(0.05)\n"
        )
        runner = EstimatorProcessRunner(
            terminate_after_ms=1500, kill_after_ms=1500
        )
        runner.start(
            sys.executable,
            [worker],
            output_directory=self.root / "out",
            run_id="run-a",
        )
        QTimer.singleShot(150, runner.request_cancel)
        terminal = self._run_until_terminal(runner)
        self.assertEqual(terminal[0], "cancelled")
        self.assertEqual(marker.read_text(), "sigint")


if __name__ == "__main__":
    unittest.main()
