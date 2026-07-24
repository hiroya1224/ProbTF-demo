import tempfile
import unittest
from pathlib import Path

from grape_param_estim.manifest import (
    MANIFEST_SCHEMA,
    build_manifest,
    dynamic_reconfigure_values,
    inspect_bag,
)


class _Parameter:
    def __init__(self, name, value):
        self.name = name
        self.value = value


class _Config:
    bools = [_Parameter("enabled", True)]
    ints = [_Parameter("count", 3)]
    strs = [_Parameter("mode", "hover")]
    doubles = [_Parameter("p_gain", 13.0)]


class ManifestTests(unittest.TestCase):
    def test_dynamic_reconfigure_is_flattened_without_current_defaults(self):
        self.assertEqual(
            dynamic_reconfigure_values(_Config()),
            {
                "enabled": True,
                "count": 3,
                "mode": "hover",
                "p_gain": 13.0,
            },
        )

    def test_topic_event_range_uses_min_max_and_clock_diagnostics(self):
        try:
            import rosbag
            import rospy
            from geometry_msgs.msg import PoseStamped
            from std_msgs.msg import Float32
        except ImportError as error:
            self.skipTest("ROS 1 messages unavailable: {}".format(error))

        with tempfile.TemporaryDirectory(prefix="grape-manifest-") as directory:
            path = Path(directory) / "clock.bag"
            with rosbag.Bag(str(path), "w") as bag:
                # Header stamps intentionally arrive out of order.
                for record_time, header_time in (
                    (10.0, 5.0),
                    (11.0, 7.0),
                    (12.0, 6.0),
                    (13.0, 8.0),
                ):
                    message = PoseStamped()
                    message.header.stamp = rospy.Time.from_sec(header_time)
                    bag.write(
                        "/pose", message, t=rospy.Time.from_sec(record_time)
                    )
                bag.write(
                    "/record_only",
                    Float32(data=1.0),
                    t=rospy.Time.from_sec(10.5),
                )

            inventory = inspect_bag(
                path,
                metadata={"episode_id": "clock", "split": "lobo"},
            )
            pose = next(
                item for item in inventory["topics"] if item["topic"] == "/pose"
            )
            self.assertEqual(pose["first_record_time"], 10.0)
            self.assertEqual(pose["last_record_time"], 13.0)
            self.assertEqual(pose["first_event_time"], 5.0)
            self.assertEqual(pose["last_event_time"], 8.0)
            self.assertEqual(pose["header_time_count"], 4)
            self.assertEqual(pose["record_time_count"], 0)
            self.assertIsNotNone(pose["header_record_offset_median_s"])
            self.assertIsNotNone(pose["header_record_offset_drift_ppm"])

            record_only = next(
                item
                for item in inventory["topics"]
                if item["topic"] == "/record_only"
            )
            self.assertEqual(record_only["header_time_count"], 0)
            self.assertEqual(record_only["record_time_count"], 1)
            self.assertIsNone(record_only["header_record_offset_median_s"])

            manifest = build_manifest(
                [path],
                metadata_by_name={
                    path.name: {"episode_id": "clock", "split": "lobo"}
                },
            )
            self.assertEqual(manifest["schema"], MANIFEST_SCHEMA)
            self.assertEqual(manifest["bag_count"], 1)
            self.assertEqual(len(manifest["manifest_hash"]), 64)


if __name__ == "__main__":
    unittest.main()
