#!/usr/bin/env python3
from typing import Dict, Iterable, List

import rospy
import tf2_ros
from geometry_msgs.msg import TransformStamped

from deflecomp_core.observation.imu_frame_config import (
    ImuFrameConfig,
    parse_imu_frame_configs,
    quat_xyzw_from_matrix,
)


def _prefixed(frame: str, prefix: str) -> str:
    name = str(frame).strip().strip("/")
    pref = str(prefix or "").strip().strip("/")
    if not pref or name.startswith(pref + "/"):
        return name
    return f"{pref}/{name}"


def _explicit_static_configs(entries) -> List[ImuFrameConfig]:
    if not entries:
        return []
    patched = []
    for entry in list(entries):
        if isinstance(entry, dict):
            item: Dict = dict(entry)
            item["publish_static_tf"] = True
            patched.append(item)
    return parse_imu_frame_configs(patched)


def _make_transform(cfg: ImuFrameConfig, parent_prefix: str, child_prefix: str) -> TransformStamped:
    quat = quat_xyzw_from_matrix(cfg.R_model_imu)
    msg = TransformStamped()
    msg.header.stamp = rospy.Time.now()
    msg.header.frame_id = _prefixed(cfg.parent_frame, parent_prefix)
    msg.child_frame_id = _prefixed(cfg.frame_id, child_prefix)
    msg.transform.translation.x = float(cfg.xyz[0])
    msg.transform.translation.y = float(cfg.xyz[1])
    msg.transform.translation.z = float(cfg.xyz[2])
    msg.transform.rotation.x = float(quat[0])
    msg.transform.rotation.y = float(quat[1])
    msg.transform.rotation.z = float(quat[2])
    msg.transform.rotation.w = float(quat[3])
    return msg


def _transforms_from_configs(configs: Iterable[ImuFrameConfig], parent_prefix: str, child_prefix: str) -> List[TransformStamped]:
    transforms: List[TransformStamped] = []
    seen = set()
    for cfg in configs:
        if not cfg.publish_static_tf:
            continue
        msg = _make_transform(cfg, parent_prefix=parent_prefix, child_prefix=child_prefix)
        key = (msg.header.frame_id, msg.child_frame_id)
        if msg.header.frame_id == msg.child_frame_id or key in seen:
            continue
        seen.add(key)
        transforms.append(msg)
    return transforms


def main() -> None:
    rospy.init_node("deflecomp_static_imu_tf_publisher", anonymous=False)
    parent_prefix = rospy.get_param("~parent_prefix", "")
    child_prefix = rospy.get_param("~child_prefix", "")
    imu_configs = parse_imu_frame_configs(rospy.get_param("~imu_frames", rospy.get_param("~frames", [])))
    explicit_configs = _explicit_static_configs(rospy.get_param("~static_transforms", []))
    transforms = _transforms_from_configs(
        list(imu_configs) + list(explicit_configs),
        parent_prefix=parent_prefix,
        child_prefix=child_prefix,
    )
    if not transforms:
        rospy.loginfo("deflecomp_static_imu_tf_publisher: no static IMU transforms to publish")
        return

    broadcaster = tf2_ros.StaticTransformBroadcaster()
    broadcaster.sendTransform(transforms)
    rospy.loginfo(
        "deflecomp_static_imu_tf_publisher: published %d static transforms: %s",
        len(transforms),
        ", ".join(f"{msg.header.frame_id}->{msg.child_frame_id}" for msg in transforms),
    )
    rospy.spin()


if __name__ == "__main__":
    main()
