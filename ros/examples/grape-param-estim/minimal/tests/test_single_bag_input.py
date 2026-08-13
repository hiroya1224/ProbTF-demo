from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
import unittest

from single_bag_input import load_single_bag_input
from single_bag_savgol_core import (
    DEFAULT_SMOOTH_MAX_NFEV,
    DEFAULT_STRICT_MAX_NFEV,
)
from single_bag_savgol_estimator import (
    build_argument_parser,
    resolve_bag_arguments,
)


class SingleBagInputTests(unittest.TestCase):
    def test_repository_bag_jsons_have_only_the_minimal_members(self):
        directory = Path(__file__).resolve().parents[1] / "bag_jsons"
        paths = sorted(directory.glob("*.json"))
        self.assertEqual(len(paths), 3)
        for path in paths:
            raw = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(
                set(raw), {"bag_path", "start_seconds", "end_seconds"}
            )
            bag_input = load_single_bag_input(path)
            self.assertTrue(bag_input.bag_path.is_absolute())
            self.assertLess(bag_input.start_seconds, bag_input.end_seconds)

    def test_loader_ignores_unrelated_members(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bag.json"
            path.write_text(
                json.dumps(
                    {
                        "bag_path": "flight.bag",
                        "start_seconds": 1.0,
                        "end_seconds": 2.0,
                        "initial_delay_seconds": 999.0,
                        "spline": {"must_not_be_used": True},
                    }
                ),
                encoding="utf-8",
            )
            value = load_single_bag_input(path)
            self.assertEqual(value.bag_path, Path(temporary) / "flight.bag")
            self.assertEqual(value.start_seconds, 1.0)
            self.assertEqual(value.end_seconds, 2.0)

    def test_json_replaces_only_bag_fields(self):
        source = self._repository_json("single_rosbag_1.json")
        arguments = argparse.Namespace(
            bag=None,
            bag_json=source,
            bag_id=None,
            bag_start=None,
            bag_end=None,
            sg_window=0.5,
        )
        result = resolve_bag_arguments(arguments)
        repeated = resolve_bag_arguments(result)
        self.assertEqual(result.bag_id, "single_rosbag_1")
        self.assertEqual(result.bag_start, 19.0)
        self.assertEqual(result.bag_end, 25.0)
        self.assertEqual(result.sg_window, 0.5)
        self.assertEqual(vars(result), vars(repeated))

    def test_json_cannot_be_mixed_with_direct_bag_fields(self):
        source = self._repository_json("single_rosbag_1.json")
        arguments = argparse.Namespace(
            bag=Path("different.bag"),
            bag_json=source,
            bag_id=None,
            bag_start=None,
            bag_end=None,
        )
        with self.assertRaisesRegex(ValueError, "cannot be used together"):
            resolve_bag_arguments(arguments)

    def test_cli_uses_high_optimizer_safety_ceilings(self):
        arguments = build_argument_parser().parse_args(
            [
                "--bag-json",
                str(self._repository_json("single_rosbag_1.json")),
                "--vehicle-model",
                "model.json",
                "--sg-window",
                "0.5",
            ]
        )
        self.assertEqual(arguments.smooth_max_nfev, DEFAULT_SMOOTH_MAX_NFEV)
        self.assertEqual(arguments.strict_max_nfev, DEFAULT_STRICT_MAX_NFEV)
        self.assertEqual(DEFAULT_SMOOTH_MAX_NFEV, 2000)
        self.assertEqual(DEFAULT_STRICT_MAX_NFEV, 2000)

    @staticmethod
    def _repository_json(name):
        return Path(__file__).resolve().parents[1] / "bag_jsons" / name


if __name__ == "__main__":
    unittest.main()
