from pathlib import Path
import re
import unittest


class FunctionalNamingTest(unittest.TestCase):
    def test_runtime_tree_uses_functional_names(self):
        repository = Path(__file__).resolve().parents[2]
        package = repository / "ros" / "examples" / "grape-param-estim"
        sources = [package / "CMakeLists.txt"]
        sources.extend(
            path
            for path in package.rglob("*")
            if path.is_file()
            and ".venv" not in path.parts
            and path.suffix in {".md", ".py", ".toml", ".xml"}
        )
        sources.extend((repository / "tests" / "grape_param_estim").glob("*.py"))

        numbered_stage = re.compile(
            "p" + r"hase(?:[ _-]?[0-9]|[0-9]_)", re.IGNORECASE
        )
        violations = []
        for path in sources:
            relative = path.relative_to(repository)
            if numbered_stage.search(str(relative)):
                violations.append(str(relative))
            for line_number, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), start=1
            ):
                if numbered_stage.search(line):
                    violations.append("{}:{}".format(relative, line_number))

        self.assertEqual(violations, [])


if __name__ == "__main__":
    unittest.main()
