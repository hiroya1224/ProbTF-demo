from pathlib import Path
import tempfile
import unittest

try:
    from grape_param_estim_gui.app import _arguments, _validated_bag_paths
except ImportError as error:  # exactly the supported optional-test guard
    raise unittest.SkipTest("GUI dependencies are unavailable: {}".format(error))


class GuiAppCliTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
