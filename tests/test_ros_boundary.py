import ast
import importlib
import re
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

LEGACY_RUNTIME_MODULES = (
    "probtf.models",
    "probtf.compatibility",
    "probtf_ros.conversions",
    "probtf_ros.legacy_conversions",
)
LEGACY_RUNTIME_SYMBOLS = {
    "BinghamDistribution",
    "BinghamRotation",
    "GaussianPosition",
    "LEGACY_ADAPTER",
    "LegacyConversionResult",
    "LegacyProjectionPolicy",
    "LegacyRosConversionResult",
    "ProbabilisticTF",
    "ProbabilisticTFArray",
    "ProbabilisticTransform",
    "distribution_to_legacy_transform",
    "legacy_message_to_v2_record",
    "legacy_transform_to_distribution",
    "legacy_transform_to_stamped",
    "probabilistic_transform_to_msg",
    "v2_record_to_legacy_message",
}
LEGACY_MESSAGE_FILES = {
    "BinghamDistribution.msg",
    "GaussianPosition.msg",
    "ProbabilisticTF.msg",
    "ProbabilisticTFArray.msg",
}


def _ros_free_package_roots(repository_root):
    return (
        repository_root / "ros" / "core" / "probtf_core" / "src" / "probtf",
        repository_root / "ros" / "core" / "probtf_core" / "src" / "probtf_estimators",
        repository_root / "ros" / "examples" / "deflecomp" / "deflecomp_core" / "src" / "deflecomp_core",
        repository_root
        / "ros"
        / "examples"
        / "deflecomp"
        / "deflecomp_examples"
        / "src"
        / "deflecomp_examples",
        repository_root / "ros" / "examples" / "deflecomp" / "deflecomp_sim" / "src" / "deflecomp_sim",
        repository_root / "ros" / "examples" / "symaware_grasp" / "src" / "symaware_grasp",
    )


class RosBoundaryTest(unittest.TestCase):
    def test_legacy_v1_cannot_reenter_runtime_source_or_message_generation(self):
        root = Path(__file__).resolve().parents[1]
        core_root = root / "ros" / "core" / "probtf_core"
        source_paths = sorted((core_root / "src").rglob("*.py"))
        source_paths.extend(sorted((core_root / "nodes").glob("*.py")))
        pattern = re.compile(
            r"\b(?:{})\b".format(
                "|".join(re.escape(symbol) for symbol in sorted(LEGACY_RUNTIME_SYMBOLS))
            )
        )
        violations = []
        for path in source_paths:
            for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if pattern.search(line) or any(module in line for module in LEGACY_RUNTIME_MODULES):
                    violations.append("{}:{}".format(path.relative_to(root), line_number))
        self.assertEqual(violations, [])

        probtf = importlib.import_module("probtf")
        probtf_ros = importlib.import_module("probtf_ros")
        for symbol in ("BinghamRotation", "GaussianPosition", "ProbabilisticTransform"):
            self.assertFalse(hasattr(probtf, symbol), symbol)
        self.assertFalse(hasattr(probtf_ros, "probabilistic_transform_to_msg"))
        for module in LEGACY_RUNTIME_MODULES:
            with self.subTest(module=module):
                with self.assertRaises(ModuleNotFoundError):
                    importlib.import_module(module)

        message_root = root / "ros" / "core" / "probtf_msgs"
        cmake = (message_root / "CMakeLists.txt").read_text(encoding="utf-8")
        declared_messages = {
            line.strip() for line in cmake.splitlines() if line.strip().endswith(".msg")
        }
        source_messages = {path.name for path in (message_root / "msg").glob("*.msg")}
        self.assertEqual(declared_messages, source_messages)
        self.assertTrue(LEGACY_MESSAGE_FILES.isdisjoint(declared_messages))
        for filename in LEGACY_MESSAGE_FILES:
            self.assertFalse((message_root / "msg" / filename).exists(), filename)

    def test_root_python_packages_do_not_import_ros(self):
        root = Path(__file__).resolve().parents[1]
        violations = []
        for source_root in _ros_free_package_roots(root):
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
                            violations.append(f"{path.relative_to(root)} imports {module}")

        self.assertEqual(violations, [])

    def test_probtf_foundation_does_not_import_producers_ros_or_examples(self):
        root = Path(__file__).resolve().parents[1]
        foundation_root = root / "ros" / "core" / "probtf_core" / "src" / "probtf"
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
                            f"{path.relative_to(root)} imports forbidden dependency {module}"
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

    def test_ros_setup_packages_first_party_sources_without_parent_path_relays(self):
        root = Path(__file__).resolve().parents[1]
        setup_text = (root / "ros" / "core" / "probtf_core" / "setup.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('find_packages(where="src")', setup_text)
        self.assertIn('package_dir={"": "src"}', setup_text)
        self.assertNotIn("../", setup_text)
        self.assertTrue((root / "ros" / "core" / "probtf_core" / "src" / "probtf").is_dir())
        self.assertTrue(
            (root / "ros" / "core" / "probtf_core" / "src" / "probtf_estimators").is_dir()
        )

    def test_root_aggregate_contains_only_first_party_package_roots(self):
        root = Path(__file__).resolve().parents[1]
        setup_path = root / "setup.py"
        tree = ast.parse(setup_path.read_text(encoding="utf-8"), filename=str(setup_path))
        package_roots = None
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "PACKAGE_ROOTS"
                for target in node.targets
            ):
                package_roots = ast.literal_eval(node.value)
                break
        self.assertIsNotNone(package_roots)
        self.assertTrue(package_roots)
        self.assertTrue(all(path.startswith("ros/") for path in package_roots))
        self.assertNotIn("third_party/BinghamNLL/src", package_roots)

        setup_config = (root / "setup.cfg").read_text(encoding="utf-8")
        self.assertNotIn("numpy-quaternion", setup_config)

    def test_tests_do_not_modify_sys_path_for_first_party_imports(self):
        root = Path(__file__).resolve().parents[1]
        violations = []
        for path in sorted((root / "tests").rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                    continue
                owner = node.func.value
                if (
                    isinstance(owner, ast.Attribute)
                    and isinstance(owner.value, ast.Name)
                    and owner.value.id == "sys"
                    and owner.attr == "path"
                    and node.func.attr in ("append", "insert")
                ):
                    violations.append(str(path.relative_to(root)))
        self.assertEqual(violations, [])

    def test_ros_core_installs_only_bridge_nodes_and_has_no_estimator_import(self):
        root = Path(__file__).resolve().parents[1]
        core_root = root / "ros" / "core" / "probtf_core"
        cmake = (core_root / "CMakeLists.txt").read_text(encoding="utf-8")
        self.assertIn("nodes/probtf_bridge_node.py", cmake)
        for producer in (
            "imu_kinematics_node.py",
            "imu_relative_pose_node.py",
            "orientation_filter_node.py",
            "probtf_fusion_node.py",
        ):
            self.assertNotIn("nodes/{}".format(producer), cmake)
        violations = []
        for path in sorted((core_root / "nodes").glob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                module = None
                if isinstance(node, ast.ImportFrom):
                    module = node.module
                elif isinstance(node, ast.Import):
                    module = node.names[0].name
                if module and module.split(".", 1)[0] == "probtf_estimators":
                    violations.append(str(path.relative_to(root)))
        self.assertEqual(violations, [])


if __name__ == "__main__":
    unittest.main()
