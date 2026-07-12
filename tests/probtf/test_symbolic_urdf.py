import xml.etree.ElementTree as ET

import pytest

from probtf.symbolic_urdf import (
    find_symbolic_urdf_placeholders,
    materialize_symbolic_urdf,
    parse_symbolic_urdf,
)


SYMBOLIC_URDF = """<?xml version="1.0"?>
<!-- ================= SYMBOLIC URDF ================= -->
<!-- indefinite variables are written in #|this_form|# -->
<robot name="example">
  <!-- A marker in an ordinary comment is documentation, not a variable: #|ignored|# -->
  <link name="base" />
  <link name="tip" />
  <joint name="joint" type="fixed">
    <origin xyz="#|joint_xyz|#" rpy="#|joint_rpy|#" />
    <parent link="base" />
    <child link="tip" />
  </joint>
</robot>
"""


def test_parse_finds_real_placeholders_in_first_seen_order():
    template = parse_symbolic_urdf(SYMBOLIC_URDF)

    assert template.placeholder_names == ("joint_xyz", "joint_rpy")
    assert find_symbolic_urdf_placeholders(SYMBOLIC_URDF) == template.placeholder_names


def test_parse_supports_legacy_xacro_placeholder_names_and_deduplicates_them():
    document = """<robot name="example" xmlns:xacro="http://www.ros.org/wiki/xacro">
  <xacro:macro name="m" params="joint_name">
    <origin xyz="#|${joint_name}_xyz|#" rpy="#|${joint_name}_xyz|#" />
  </xacro:macro>
</robot>"""

    assert find_symbolic_urdf_placeholders(document) == ("${joint_name}_xyz",)


def test_materialize_replaces_vectors_and_removes_legacy_notes():
    realized = materialize_symbolic_urdf(
        SYMBOLIC_URDF,
        {"joint_xyz": [0.1, 0, -0.25], "joint_rpy": "0 1.5 0"},
    )

    root = ET.fromstring(realized)
    origin = root.find("joint/origin")
    assert origin.attrib == {"xyz": "0.1 0 -0.25", "rpy": "0 1.5 0"}
    assert "SYMBOLIC URDF" not in realized
    assert "#|this_form|#" not in realized
    assert "#|ignored|#" in realized


def test_materialize_can_keep_symbolic_notes_without_treating_comments_as_variables():
    realized = materialize_symbolic_urdf(
        SYMBOLIC_URDF,
        {"joint_xyz": [0, 0, 0], "joint_rpy": [0, 0, 0]},
        keep_symbolic_notes=True,
    )

    assert "#|this_form|#" in realized
    assert "#|joint_xyz|#" not in realized


@pytest.mark.parametrize(
    "substitutions, message",
    [
        ({"joint_xyz": [0, 0, 0]}, "missing substitutions: joint_rpy"),
        (
            {"joint_xyz": [0, 0, 0], "joint_rpy": [0, 0, 0], "typo": [1]},
            "unexpected substitutions: typo",
        ),
    ],
)
def test_materialize_requires_an_exact_substitution_mapping(substitutions, message):
    with pytest.raises(ValueError, match=message):
        materialize_symbolic_urdf(SYMBOLIC_URDF, substitutions)


def test_materialize_rejects_non_string_substitution_names():
    substitutions = {"joint_xyz": [0, 0, 0], "joint_rpy": [0, 0, 0], 1: [0]}

    with pytest.raises(TypeError, match="substitution names must be strings"):
        materialize_symbolic_urdf(SYMBOLIC_URDF, substitutions)


@pytest.mark.parametrize(
    "bad_value, error_type",
    [
        ([0, float("nan"), 0], ValueError),
        ("0 &quot; 0", TypeError),
        ([0, True, 0], TypeError),
        ([], ValueError),
    ],
)
def test_materialize_rejects_unsafe_or_non_numeric_values(bad_value, error_type):
    substitutions = {"joint_xyz": bad_value, "joint_rpy": [0, 0, 0]}

    with pytest.raises(error_type):
        materialize_symbolic_urdf(SYMBOLIC_URDF, substitutions)


@pytest.mark.parametrize(
    "document",
    [
        '<robot name="bad"><origin xyz="#|joint_xyz" /></robot>',
        '<!DOCTYPE robot [<!ENTITY x "1">]><robot name="bad" />',
        '<robot name="bad">',
    ],
)
def test_parse_rejects_malformed_or_unsafe_documents(document):
    with pytest.raises(ValueError):
        parse_symbolic_urdf(document)
