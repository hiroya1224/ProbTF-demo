import ast
import unittest
from pathlib import Path


ROS_MODULES = {
    "geometry_msgs",
    "probik_msgs",
    "roslib",
    "rospkg",
    "rospy",
    "sensor_msgs",
    "std_msgs",
    "tf2_ros",
    "visualization_msgs",
}


class RosBoundaryTest(unittest.TestCase):
    def test_root_python_packages_do_not_import_ros(self):
        source_root = Path(__file__).resolve().parents[1] / "src"
        violations = []
        for path in sorted(source_root.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    modules = [alias.name.split(".", 1)[0] for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    modules = [node.module.split(".", 1)[0]]
                else:
                    continue
                for module in modules:
                    if module in ROS_MODULES:
                        violations.append(f"{path.relative_to(source_root)} imports {module}")

        self.assertEqual(violations, [])


if __name__ == "__main__":
    unittest.main()
