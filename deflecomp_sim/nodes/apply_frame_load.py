#!/usr/bin/env python3
import time
from typing import Any, List

import rospy
from geometry_msgs.msg import WrenchStamped


def _float_list(value: Any, size: int, default: List[float]) -> List[float]:
    if value is None:
        return list(default)
    if isinstance(value, str):
        text = value.strip()
        if text[:1] in ("[", "(") and text[-1:] in ("]", ")"):
            text = text[1:-1]
        text = text.replace(";", ",")
        items = text.split(",") if "," in text else text.split()
    else:
        items = list(value)
    values = [float(item) for item in items if str(item).strip()]
    if len(values) != size:
        raise ValueError(f"expected {size} values, got {len(values)}: {value}")
    return values


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() not in ("0", "false", "no", "off", "")
    return bool(value)


def _reference_frame(value: Any) -> str:
    reference_frame = str(value).strip().lower()
    if reference_frame in ("", "world", "base", "map"):
        return "world"
    if reference_frame in ("local", "frame", "target"):
        return "local"
    raise ValueError("~reference_frame must be 'world' or 'local'")


def _encoded_frame_id(frame: str, reference_frame: str) -> str:
    if not frame:
        return ""
    if reference_frame == "local":
        return f"{frame}@local"
    return frame


def _prefixed_frame_id(frame: str, tf_prefix: str) -> str:
    clean_frame = frame.strip().lstrip("/")
    clean_prefix = str(tf_prefix).strip().strip("/")
    if not clean_frame or not clean_prefix:
        return clean_frame
    if clean_frame == clean_prefix or clean_frame.startswith(f"{clean_prefix}/"):
        return clean_frame
    return f"{clean_prefix}/{clean_frame}"


def _wrench_msg(frame_id: str, force: List[float], torque: List[float]) -> WrenchStamped:
    msg = WrenchStamped()
    msg.header.stamp = rospy.Time.now()
    msg.header.frame_id = frame_id
    msg.wrench.force.x = float(force[0])
    msg.wrench.force.y = float(force[1])
    msg.wrench.force.z = float(force[2])
    msg.wrench.torque.x = float(torque[0])
    msg.wrench.torque.y = float(torque[1])
    msg.wrench.torque.z = float(torque[2])
    return msg


def main() -> None:
    rospy.init_node("apply_frame_load", anonymous=True)

    topic = rospy.get_param("~topic", "/deflecomp_sim/external_wrench")
    frame = str(rospy.get_param("~frame", "")).strip()
    tf_prefix = str(rospy.get_param("~tf_prefix", "equil")).strip()
    reference_frame = _reference_frame(rospy.get_param("~reference_frame", "world"))
    mass_kg = float(rospy.get_param("~mass_kg", 0.1))
    gravity = float(rospy.get_param("~gravity", -9.81))
    duration = float(rospy.get_param("~duration", 0.0))
    rate_hz = float(rospy.get_param("~rate", 20.0))
    clear = _as_bool(rospy.get_param("~clear", False))
    clear_on_exit = _as_bool(rospy.get_param("~clear_on_exit", True))

    if not frame and not clear:
        raise ValueError("~frame is required unless ~clear:=true")

    if clear:
        frame_id = _encoded_frame_id(_prefixed_frame_id(frame, tf_prefix), reference_frame) if frame else ""
        force = [0.0, 0.0, 0.0]
        torque = [0.0, 0.0, 0.0]
    else:
        frame_id = _encoded_frame_id(_prefixed_frame_id(frame, tf_prefix), reference_frame)
        if rospy.has_param("~force"):
            force = _float_list(rospy.get_param("~force"), 3, [0.0, 0.0, mass_kg * gravity])
        else:
            force = [0.0, 0.0, mass_kg * gravity]
        torque = _float_list(rospy.get_param("~torque", None), 3, [0.0, 0.0, 0.0])

    pub = rospy.Publisher(topic, WrenchStamped, queue_size=1, latch=True)
    # Give the subscriber connection a short window before the first publish.
    rospy.sleep(0.2)

    did_clear = False

    def publish_wrench() -> None:
        pub.publish(_wrench_msg(frame_id, force, torque))

    def publish_clear() -> None:
        nonlocal did_clear
        if did_clear:
            return
        did_clear = True
        for _ in range(3):
            pub.publish(_wrench_msg(frame_id, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]))
            time.sleep(0.05)

    rospy.loginfo(
        "apply_frame_load: topic=%s frame=%s tf_frame=%s reference_frame=%s force=[%.6g, %.6g, %.6g] torque=[%.6g, %.6g, %.6g] duration=%.6g clear_on_exit=%s",
        topic,
        frame or "(clear)",
        frame_id or "(clear)",
        reference_frame,
        force[0],
        force[1],
        force[2],
        torque[0],
        torque[1],
        torque[2],
        duration,
        clear_on_exit,
    )

    if clear:
        publish_clear()
        return

    rospy.on_shutdown(publish_clear if clear_on_exit else (lambda: None))

    try:
        rate = rospy.Rate(max(rate_hz, 1e-6))
        if duration <= 0.0:
            while not rospy.is_shutdown():
                publish_wrench()
                rate.sleep()
        else:
            t_end = rospy.get_time() + duration
            while not rospy.is_shutdown() and rospy.get_time() <= t_end:
                publish_wrench()
                rate.sleep()
    finally:
        if clear_on_exit:
            publish_clear()


if __name__ == "__main__":
    main()
