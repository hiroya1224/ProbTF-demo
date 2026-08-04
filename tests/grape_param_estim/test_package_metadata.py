import unittest
import xml.etree.ElementTree as ET
from pathlib import Path


EXPECTED_DESCRIPTION = (
    "Sparse batch trajectory estimation, Laplace-EM, and posterior parameter "
    "sampling for the Grape closed-loop system."
)
PACKAGE_XML = (
    Path(__file__).resolve().parents[2]
    / "ros"
    / "examples"
    / "grape-param-estim"
    / "package.xml"
)


class PackageMetadataTest(unittest.TestCase):
    def test_sparse_batch_description_and_scipy_runtime_dependency(self):
        package = ET.parse(str(PACKAGE_XML)).getroot()

        descriptions = package.findall("description")
        self.assertEqual(len(descriptions), 1)
        self.assertEqual((descriptions[0].text or "").strip(), EXPECTED_DESCRIPTION)

        scipy_dependencies = [
            (element.tag, (element.text or "").strip())
            for element in package
            if element.tag.endswith("depend")
            and (element.text or "").strip() == "python3-scipy"
        ]
        self.assertEqual(scipy_dependencies, [("exec_depend", "python3-scipy")])


if __name__ == "__main__":
    unittest.main()
