import json
import os
from pathlib import Path
import subprocess
import sys
import unittest


PACKAGE_ROOT = (
    Path(__file__).resolve().parents[2]
    / "ros"
    / "examples"
    / "grape-param-estim"
)


class PackageImportBoundaryTest(unittest.TestCase):
    def test_root_import_is_solver_and_ros_free(self):
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(PACKAGE_ROOT / "src")
        script = (
            "import json,sys; import grape_param_estim; "
            "print(json.dumps(sorted(name for name in sys.modules "
            "if name == 'scipy' or name.startswith('scipy.') "
            "or name == 'rospy' or name.startswith('rospy.') "
            "or name.startswith('grape_param_estim.'))))"
        )
        completed = subprocess.run(
            [sys.executable, "-c", script],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            env=environment,
        )
        self.assertEqual(json.loads(completed.stdout), [])

    def test_legacy_symbols_are_not_package_root_aliases(self):
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(PACKAGE_ROOT / "src")
        script = (
            "import grape_param_estim; "
            "assert not hasattr(grape_param_estim, 'WeakConstraintIEnKSQ'); "
            "assert not hasattr(grape_param_estim, 'StrongConstraintIEnKS'); "
            "assert not hasattr(grape_param_estim, 'RealFlightEpisode')"
        )
        subprocess.run(
            [sys.executable, "-c", script],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            env=environment,
        )

    def test_removed_real_episode_api_is_absent_from_adapter_module(self):
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(PACKAGE_ROOT / "src")
        script = (
            "from grape_param_estim import real_rosbag; "
            "removed = ('RealFlightEpisode', 'build_real_flight_episode', "
            "'load_grape_rosbag_episode', 'save_real_flight_episode', "
            "'EpisodeProvenance', 'robust_covariance', "
            "'robust_pose_covariances'); "
            "assert all(not hasattr(real_rosbag, name) for name in removed)"
        )
        subprocess.run(
            [sys.executable, "-c", script],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            env=environment,
        )

    def test_removed_geometry_wrappers_are_absent(self):
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(PACKAGE_ROOT / "src")
        script = (
            "from grape_param_estim import geometry; "
            "removed = ('rotation_matrix_from_vector', "
            "'rotation_vector_from_matrix'); "
            "assert all(not hasattr(geometry, name) for name in removed)"
        )
        subprocess.run(
            [sys.executable, "-c", script],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            env=environment,
        )


if __name__ == "__main__":
    unittest.main()
