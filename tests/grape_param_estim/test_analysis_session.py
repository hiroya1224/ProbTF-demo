from copy import deepcopy
from datetime import datetime
from pathlib import Path
import tempfile
import unittest

from grape_param_estim.analysis_session import (
    IncrementalAnalysisSession,
    default_session_directory,
)
from grape_param_estim.automatic_analysis import (
    RESULT_SCHEMA,
    merge_analysis_results,
)
from grape_param_estim.effective_estimator import canonical_sha256


def _result(path, duration, trace_time=1.0, config_sha="a" * 64):
    result = {
        "schema": RESULT_SCHEMA,
        "config_sha256": config_sha,
        "interpretation": "independent episode fits",
        "bag_count": 1,
        "sequence_duration_s": duration,
        "bags": [
            {
                "bag_index": 0,
                "path": str(path),
                "sha256": "b" * 64,
                "bag_start_time": 100.0,
                "duration_s": duration,
                "sequence_offset_s": 0.0,
                "episode_count": 1,
                "episodes": [
                    {
                        "episode_index": 0,
                        "sequence_start_s": 0.5,
                        "sequence_end_s": duration - 0.5,
                        "parameter_trace": [
                            {
                                "time_s": trace_time,
                                "sequence_time_s": trace_time,
                                "parameters": {"gain": 1.0},
                            }
                        ],
                    }
                ],
                "plot": {},
            }
        ],
    }
    result["result_sha256"] = canonical_sha256(result)
    return result


class MergeAnalysisResultsTests(unittest.TestCase):
    def test_rebases_added_bag_and_parameter_times(self):
        first = _result("/tmp/first.bag", 12.0)
        second = _result("/tmp/second.bag", 7.0, trace_time=2.0)
        untouched = deepcopy(second)

        merged = merge_analysis_results(first, second)

        self.assertEqual(merged["bag_count"], 2)
        self.assertEqual(merged["sequence_duration_s"], 19.0)
        added = merged["bags"][1]
        self.assertEqual(added["bag_index"], 1)
        self.assertEqual(added["sequence_offset_s"], 12.0)
        self.assertEqual(
            added["episodes"][0]["sequence_start_s"], 12.5
        )
        self.assertEqual(
            added["episodes"][0]["sequence_end_s"], 18.5
        )
        self.assertEqual(
            added["episodes"][0]["parameter_trace"][0][
                "sequence_time_s"
            ],
            14.0,
        )
        self.assertEqual(second, untouched)
        without_hash = deepcopy(merged)
        digest = without_hash.pop("result_sha256")
        self.assertEqual(digest, canonical_sha256(without_hash))

    def test_rejects_different_configuration(self):
        first = _result("/tmp/first.bag", 12.0)
        second = _result(
            "/tmp/second.bag", 7.0, config_sha="c" * 64
        )

        with self.assertRaisesRegex(
            ValueError, "different configurations"
        ):
            merge_analysis_results(first, second)


class IncrementalAnalysisSessionTests(unittest.TestCase):
    def test_only_new_bags_are_analyzed_and_result_is_saved(self):
        calls = []

        def analyzer(paths, config):
            path = tuple(paths)[0]
            calls.append(path)
            return _result(path, 5.0)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.bag"
            second = root / "second.bag"
            first.touch()
            second.touch()
            session = IncrementalAnalysisSession(
                object(), root / "output", analyzer=analyzer
            )

            self.assertEqual(
                session.add_bags([first, first, second]),
                (first.resolve(), second.resolve()),
            )
            session.analyze(first)
            self.assertEqual(session.pending_paths, (second.resolve(),))
            session.analyze(second)

            self.assertEqual(
                calls, [first.resolve(), second.resolve()]
            )
            self.assertEqual(session.result["bag_count"], 2)
            self.assertTrue(session.analysis_path.is_file())
            with self.assertRaisesRegex(
                ValueError, "already analyzed"
            ):
                session.analyze(first)

    def test_default_output_is_under_ros_home(self):
        path = default_session_directory(
            datetime(2026, 7, 30, 12, 34, 56)
        )

        self.assertEqual(
            path.name, "20260730-123456"
        )
        self.assertEqual(
            path.parts[-3:-1],
            ("grape_param_estim", "failure_analysis"),
        )


if __name__ == "__main__":
    unittest.main()
