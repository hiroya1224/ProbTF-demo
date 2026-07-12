from pathlib import Path

import numpy as np
import pytest
import yaml

from probtf.sensor_config import load_sensor_mounts, parse_sensor_mounts
from probtf.sensor_mount import SensorMount


def test_loads_generic_sensor_list_with_explicit_wxyz_and_rpy():
    mounts = load_sensor_mounts(
        {
            "sensors": [
                {
                    "source_id": "wrist_imu",
                    "frame_id": "wrist_imu_frame",
                    "parent_frame_id": "wrist_link",
                    "position_xyz": [0.01, 0.02, 0.03],
                    "orientation_wxyz": [1, 0, 0, 0],
                },
                {
                    "source_id": "tool_marker",
                    "frame_id": "tool_marker_frame",
                    "parent_frame_id": "tool0",
                    "rpy": [0, 0, np.pi / 2],
                },
            ]
        }
    )

    assert all(isinstance(mount, SensorMount) for mount in mounts)
    assert mounts[0].source_id == "wrist_imu"
    np.testing.assert_allclose(mounts[0].position_xyz, [0.01, 0.02, 0.03])
    np.testing.assert_allclose(mounts[1].orientation_wxyz, [np.sqrt(0.5), 0, 0, np.sqrt(0.5)])


def test_loads_generic_sensor_mapping_and_nested_transform():
    mounts = parse_sensor_mounts(
        {
            "sensors": {
                "overhead_camera": {
                    "frame_id": "camera_optical",
                    "transform": {
                        "parent_frame_id": "world",
                        "translation": [1, 2, 3],
                        "orientation_wxyz": [0, 1, 0, 0],
                    },
                }
            }
        }
    )

    assert mounts[0].source_id == "overhead_camera"
    assert mounts[0].parent_frame_id == "world"
    np.testing.assert_allclose(mounts[0].position_xyz, [1, 2, 3])
    np.testing.assert_allclose(mounts[0].orientation_wxyz, [0, 1, 0, 0])


def test_loads_existing_deflecomp_file_convention():
    path = (
        Path(__file__).resolve().parents[2]
        / "ros"
        / "examples"
        / "deflecomp"
        / "deflecomp_ros"
        / "config"
        / "imu_frames.yaml"
    )

    mounts = load_sensor_mounts(path)

    assert [mount.source_id for mount in mounts] == ["link6", "link3", "link2"]
    assert [mount.parent_frame_id for mount in mounts] == ["link6", "link3", "link2"]


def test_loads_legacy_static_tf_and_xyzw_quaternion():
    mounts = parse_sensor_mounts(
        {
            "imu_frames": [
                {
                    "frame_id": "imu",
                    "model_frame": "link",
                    "static_tf": {
                        "parent_frame": "mount",
                        "xyz": "0.1, 0.2, 0.3",
                        "quat": [0, 0, np.sqrt(0.5), np.sqrt(0.5)],
                    },
                    "publish_static_tf": True,
                }
            ]
        }
    )

    assert mounts[0].source_id == "imu"
    assert mounts[0].parent_frame_id == "mount"
    np.testing.assert_allclose(mounts[0].position_xyz, [0.1, 0.2, 0.3])
    np.testing.assert_allclose(mounts[0].orientation_wxyz, [np.sqrt(0.5), 0, 0, np.sqrt(0.5)])


def test_legacy_static_tf_overrides_top_level_mount_and_compact_entries_work():
    mounts = parse_sensor_mounts(
        {
            "imu_frames": [
                {
                    "frame_id": "top_level_imu",
                    "parent_frame": "top_level_parent",
                    "xyz": [9, 9, 9],
                    "rpy": [1, 1, 1],
                    "static_tf": {
                        "child_frame": "nested_imu",
                        "parent_frame": "nested_parent",
                        "xyz": [1, 2, 3],
                        "rpy": [0, 0, 0],
                    },
                },
                "compact_imu@0.1,0.2,0.3@0,0,0",
            ]
        }
    )

    assert mounts[0].frame_id == "nested_imu"
    assert mounts[0].parent_frame_id == "nested_parent"
    np.testing.assert_allclose(mounts[0].position_xyz, [1, 2, 3])
    assert mounts[1].source_id == "compact_imu"
    np.testing.assert_allclose(mounts[1].position_xyz, [0.1, 0.2, 0.3])


def test_loads_yaml_path_using_safe_loader(tmp_path):
    path = tmp_path / "sensors.yaml"
    path.write_text(
        "sensors:\n"
        "  encoder:\n"
        "    frame_id: encoder_frame\n"
        "    parent_frame_id: joint\n",
        encoding="utf-8",
    )

    mounts = load_sensor_mounts(path)

    assert mounts[0].source_id == "encoder"
    malicious = tmp_path / "unsafe.yaml"
    malicious.write_text("!!python/object/apply:os.system ['echo unsafe']\n", encoding="utf-8")
    with pytest.raises(yaml.constructor.ConstructorError):
        load_sensor_mounts(malicious)


@pytest.mark.parametrize(
    "config, message",
    [
        (
            {
                "sensors": [
                    {"source_id": "imu", "frame_id": "imu_a", "parent_frame_id": "base"},
                    {"source_id": "imu", "frame_id": "imu_b", "parent_frame_id": "base"},
                ]
            },
            "duplicate sensor source_id: imu",
        ),
        (
            {
                "sensors": [
                    {"source_id": "imu_a", "frame_id": "/imu", "parent_frame_id": "base"},
                    {"source_id": "imu_b", "frame_id": "imu", "parent_frame_id": "base"},
                ]
            },
            "duplicate sensor frame_id: imu",
        ),
    ],
)
def test_rejects_duplicate_source_or_normalized_frame_ids(config, message):
    with pytest.raises(ValueError, match=message):
        parse_sensor_mounts(config)


@pytest.mark.parametrize(
    "entry, message",
    [
        (
            {"source_id": "imu", "frame_id": "imu", "parent_frame_id": "base", "xyz": [0, 0]},
            "position_xyz must contain exactly 3 numbers",
        ),
        (
            {
                "source_id": "imu",
                "frame_id": "imu",
                "parent_frame_id": "base",
                "orientation_wxyz": [0, 0, 0, 0],
            },
            "zero quaternion",
        ),
        (
            {
                "source_id": "imu",
                "frame_id": "imu",
                "parent_frame_id": "base",
                "orientation_wxyz": [1, 0, 0, 0],
                "rpy": [0, 0, 0],
            },
            "orientation is ambiguous",
        ),
        (
            {
                "source_id": "imu",
                "frame_id": "imu",
                "parent_frame_id": "base",
                "position_xyz": [0, float("nan"), 0],
            },
            "finite numbers",
        ),
    ],
)
def test_rejects_malformed_or_ambiguous_transforms(entry, message):
    with pytest.raises(ValueError, match=message):
        parse_sensor_mounts({"sensors": [entry]})


def test_generic_list_requires_source_id_and_rejects_unknown_fields():
    with pytest.raises(ValueError, match="source_id is required"):
        parse_sensor_mounts({"sensors": [{"frame_id": "imu", "parent_frame_id": "base"}]})
    with pytest.raises(ValueError, match="unsupported keys: postion_xyz"):
        parse_sensor_mounts(
            {
                "sensors": [
                    {
                        "source_id": "imu",
                        "frame_id": "imu",
                        "parent_frame_id": "base",
                        "postion_xyz": [0, 0, 0],
                    }
                ]
            }
        )
