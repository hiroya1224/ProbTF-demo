import ast
import unittest
from pathlib import Path


ROS_MODULES = {
    "geometry_msgs",
    "message_filters",
    "probtf_msgs",
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

    def test_probtf_foundation_does_not_import_producers_ros_or_examples(self):
        source_root = Path(__file__).resolve().parents[1] / "src"
        foundation_root = source_root / "probtf"
        forbidden = ROS_MODULES | {
            "deflecomp_examples",
            "probtf_estimators",
            "symaware_grasp",
        }
        violations = []
        for path in sorted(foundation_root.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    modules = [alias.name.split(".", 1)[0] for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    modules = [node.module.split(".", 1)[0]]
                else:
                    continue
                for module in modules:
                    if module in forbidden:
                        violations.append(
                            f"{path.relative_to(source_root)} imports forbidden dependency {module}"
                        )
        self.assertEqual(violations, [])

    def test_generic_ros_bridge_does_not_import_estimators(self):
        root = Path(__file__).resolve().parents[1]
        bridge_root = root / "ros" / "core" / "probtf_core" / "src" / "probtf_ros"
        violations = []
        for path in sorted(bridge_root.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    modules = [alias.name.split(".", 1)[0] for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    modules = [node.module.split(".", 1)[0]]
                else:
                    continue
                if "probtf_estimators" in modules:
                    violations.append(str(path.relative_to(root)))
        self.assertEqual(violations, [])

    def test_ros_setup_does_not_repackage_foundation_or_bingham(self):
        root = Path(__file__).resolve().parents[1]
        setup_text = (root / "ros" / "core" / "probtf_core" / "setup.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn('"probtf":', setup_text)
        self.assertNotIn('"probtf_estimators":', setup_text)
        self.assertNotIn('"bingham":', setup_text)
        self.assertIn('packages=["probtf_ros"]', setup_text)


if __name__ == "__main__":
    unittest.main()
