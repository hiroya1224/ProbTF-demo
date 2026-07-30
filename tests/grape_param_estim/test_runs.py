import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from grape_param_estim.data import load_yaml
from grape_param_estim.runs import (
    atomic_json,
    latest_completed_run_directory,
    latest_run_directory,
    start_run,
)


class RunDirectoryTest(unittest.TestCase):
    def test_start_run_persists_config_and_spawns_worker(self):
        configuration = {
            "schema": "grape_param_estim/phase2",
            "analysis": {"datasets": [{"bag_path": "/tmp/example.bag"}]},
        }
        with tempfile.TemporaryDirectory() as temporary:
            process = mock.Mock(pid=12345)
            with mock.patch(
                "grape_param_estim.runs.subprocess.Popen",
                return_value=process,
            ) as popen:
                run_path = start_run(
                    temporary,
                    configuration,
                    worker_script="/tmp/grape_param_estim_worker.py",
                )

            self.assertEqual(latest_run_directory(temporary), run_path)
            self.assertEqual(
                load_yaml(str(run_path / "config.yaml")), configuration
            )
            status = json.loads(
                (run_path / "status.json").read_text(encoding="utf-8")
            )
            self.assertEqual(status["state"], "queued")
            self.assertEqual(status["pid"], 12345)
            command = popen.call_args.args[0]
            self.assertEqual(command[-2:], ["--run-dir", str(run_path)])

    def test_latest_completed_run_requires_result(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            older = root / "20260101-000000-000000"
            newer = root / "20260102-000000-000000"
            older.mkdir()
            newer.mkdir()
            atomic_json(older / "status.json", {"state": "completed"})
            (older / "result.npz").write_bytes(b"npz")
            atomic_json(newer / "status.json", {"state": "running"})

            self.assertEqual(latest_run_directory(temporary), newer)
            self.assertEqual(
                latest_completed_run_directory(temporary), older
            )


if __name__ == "__main__":
    unittest.main()
