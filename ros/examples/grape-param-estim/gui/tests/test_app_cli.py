from pathlib import Path
import signal
import tempfile
import unittest
from unittest import mock

try:
    from PySide6.QtWidgets import QApplication

    from grape_param_estim_gui.app import (
        _arguments,
        _install_interrupt_handler,
        _validated_bag_paths,
    )
except ImportError as error:  # exactly the supported optional-test guard
    raise unittest.SkipTest("GUI dependencies are unavailable: {}".format(error))


class GuiAppCliTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.application = QApplication.instance() or QApplication([])

    def test_bag_option_is_repeatable(self):
        arguments = _arguments(
            ["--bag", "/tmp/failed.bag", "--bag", "/tmp/success.bag"]
        )
        self.assertEqual(
            arguments.bag,
            [Path("/tmp/failed.bag"), Path("/tmp/success.bag")],
        )

    def test_bag_paths_are_resolved_and_must_be_files(self):
        with tempfile.TemporaryDirectory() as directory:
            bag = Path(directory) / "sample.bag"
            bag.write_bytes(b"bag")
            self.assertEqual(_validated_bag_paths([bag]), (bag.resolve(),))
            with self.assertRaisesRegex(FileNotFoundError, "missing.bag"):
                _validated_bag_paths([Path(directory) / "missing.bag"])

    def test_sigint_is_queued_through_qt_window_close(self):
        closed = []
        previous_handler = object()
        with mock.patch(
            "grape_param_estim_gui.app.signal.getsignal",
            return_value=previous_handler,
        ), mock.patch("grape_param_estim_gui.app.signal.signal") as setter:
            timer, previous = _install_interrupt_handler(
                self.application, lambda: closed.append(True)
            )
            installed_handler = setter.call_args.args[1]
            installed_handler(signal.SIGINT, None)
            self.application.processEvents()
            timer.stop()
        self.assertIs(previous, previous_handler)
        self.assertEqual(closed, [True])


if __name__ == "__main__":
    unittest.main()
