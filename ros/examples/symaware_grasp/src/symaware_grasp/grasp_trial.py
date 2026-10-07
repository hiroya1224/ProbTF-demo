"""Kinematic grasp animation and independent bag evaluation.

The final grasp is selected by the existing Bingham IK. Cartesian approach
planning uses that selected hand pose; it never replaces the target belief
with an arbitrary object orientation. No contact dynamics are simulated.
"""
import csv
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from symaware_grasp.arm_kinematics import ToyArm6DOF


GRIPPER_JOINT_NAMES = ["gripper_left_finger_joint", "gripper_right_finger_joint"]
REST_OPENING = 0.105
MIN_OPENING = 0.100
MAX_OPENING = 0.225
EVENT_TOPIC = "/symaware_grasp/trial_event"
MARKER_TOPIC = "/symaware_grasp/trial_markers"


def gripper_joint_position(opening):
    opening = float(opening)
    if not math.isfinite(opening) or not MIN_OPENING <= opening <= MAX_OPENING:
        raise ValueError("gripper opening must be between 0.100 and 0.225 m")
    return 0.5 * (opening - REST_OPENING)


def gripper_opening(joint_position):
    return REST_OPENING + 2.0 * float(joint_position)


def animated_gripper_urdf(source):
    """Keep the arm joints/TCP, shorten the flange and animate both jaws."""
    root = ET.fromstring(source)
    tool = root.find("./link[@name='tool0']")
    if tool is None:
        raise ValueError("tool0 is missing from the demo model")
    # The original link_6 cylinder ended at the TCP, inside the bottle.
    # Stop the mounting flange inside the palm instead; joint frames stay fixed.
    for element in root.findall("./link[@name='link_6']/visual") + root.findall("./link[@name='link_6']/collision"):
        element.find("origin").set("xyz", "0.0125 0 0")
        element.find("geometry/cylinder").set("length", "0.025")
    for element in list(tool):
        if element.get("name", "").startswith("gripper_palm"):
            element.find("geometry/box").set("size", "0.04 0.29 0.12")
    for side, sign, name in zip(("left", "right"), (1, -1), GRIPPER_JOINT_NAMES):
        link = ET.SubElement(root, "link", name="gripper_" + side + "_link")
        for element in list(tool):
            if element.get("name", "").startswith("gripper_" + side + "_"):
                tool.remove(element)
                link.append(element)
        if len(link.findall("visual")) != 2:
            raise ValueError("expected a finger and pad on each side")
        pad = link.find("./visual[@name='gripper_{}_pad']".format(side))
        collision = ET.SubElement(link, "collision", name="gripper_{}_pad_collision".format(side))
        for tag in ("origin", "geometry"):
            collision.append(ET.fromstring(ET.tostring(pad.find(tag))))
        joint = ET.SubElement(root, "joint", name=name, type="prismatic")
        ET.SubElement(joint, "parent", link="tool0")
        ET.SubElement(joint, "child", link=link.get("name"))
        ET.SubElement(joint, "origin", xyz="0 0 0", rpy="0 0 0")
        ET.SubElement(joint, "axis", xyz="0 {} 0".format(sign))
        ET.SubElement(joint, "limit", lower="-0.0025", upper="0.06", effort="20", velocity="0.04")
    return "\n".join(line.rstrip() for line in ET.tostring(root, encoding="unicode").splitlines()) + "\n"


def solve_cartesian_pose(robot, position, rotation, seed):
    """Track a hand pose after the symmetry-aware IK selected its free yaw."""
    position, rotation = np.asarray(position), np.asarray(rotation)
    def residual(theta):
        transform = robot.forward_transform(theta)
        return np.concatenate(((transform[:3, 3] - position) / 0.1,
                               Rotation.from_matrix(rotation.T @ transform[:3, :3]).as_rotvec()))
    result = least_squares(residual, robot.clip_to_limits(seed),
                           bounds=(robot.lower_limits + 1e-8, robot.upper_limits - 1e-8),
                           max_nfev=300, ftol=1e-10, xtol=1e-10, gtol=1e-10)
    theta = result.x
    transform = robot.forward_transform(theta)
    position_error = float(np.linalg.norm(transform[:3, 3] - position))
    rotation_error = float(np.linalg.norm(Rotation.from_matrix(rotation.T @ transform[:3, :3]).as_rotvec()))
    if not result.success or position_error > 0.001 or rotation_error > math.radians(0.5):
        raise RuntimeError("Cartesian approach waypoint is not reachable")
    return theta


def plan_approach(robot, grasp_theta, distance=0.12, count=25):
    if not math.isfinite(distance) or distance <= 0 or count < 2:
        raise ValueError("approach distance/count must be positive")
    grasp = robot.forward_transform(grasp_theta)
    position, rotation = grasp[:3, 3], grasp[:3, :3]
    pregrasp_position = position - distance * rotation[:, 0]
    pregrasp = solve_cartesian_pose(robot, pregrasp_position, rotation, grasp_theta)
    waypoints, seed = [], pregrasp
    for fraction in np.linspace(0, 1, count):
        seed = solve_cartesian_pose(robot, pregrasp_position + fraction * distance * rotation[:, 0], rotation, seed)
        waypoints.append(seed.copy())
    return pregrasp, np.asarray(waypoints)


def smooth_fraction(elapsed, duration):
    u = np.clip(float(elapsed) / duration, 0.0, 1.0)
    return u ** 3 * (10 + u * (-15 + 6 * u))


def trajectory_reference(waypoints, fraction):
    waypoints = np.asarray(waypoints)
    index = np.clip(float(fraction), 0, 1) * (len(waypoints) - 1)
    left = min(int(index), len(waypoints) - 2)
    return waypoints[left] + (index - left) * (waypoints[left + 1] - waypoints[left])


def evaluate_grasp(theta, opening, object_position, object_axis, diameter=0.1, robot=None, object_rotation=None):
    robot = robot or ToyArm6DOF()
    theta = np.asarray(theta)
    axis = np.asarray(object_axis, dtype=float)
    if theta.shape != (6,) or not np.all(np.isfinite(theta)) or not robot.within_limits(theta):
        raise ValueError("invalid arm joint state")
    if not np.all(np.isfinite(axis)) or np.linalg.norm(axis) < 1e-8:
        raise ValueError("invalid bottle axis")
    gripper_joint_position(opening)
    transform = robot.forward_transform(theta)
    dot = float(np.dot(transform[:3, 2], axis / np.linalg.norm(axis)))
    metrics = {
        "tcp_x": float(transform[0, 3]), "tcp_y": float(transform[1, 3]), "tcp_z": float(transform[2, 3]),
        "tcp_object_distance_mm": 1000 * float(np.linalg.norm(transform[:3, 3] - object_position)),
        "axis_tilt_deg": math.degrees(math.acos(np.clip(dot, -1, 1))),
        "gripper_opening_mm": 1000 * opening,
        "gripper_clearance_mm": 1000 * (opening - diameter),
        "wrist_bend_deg": math.degrees(abs(float(theta[4]))),
        "elbow_flexion_deg": math.degrees(abs(float(theta[2]))),
    }
    if object_rotation is not None:
        metrics["orientation_mode_error_deg"] = math.degrees(float(
            Rotation.from_matrix(np.asarray(object_rotation).T @ transform[:3, :3]).magnitude()))
    return metrics


def analyze_trial(directory, output=None):
    import rosbag
    directory = Path(directory)
    output = Path(output) if output else directory
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((directory / "manifest.json").read_text())
    from probtf.geometry import quat_to_rotmat
    from symaware_grasp_ros import record_from_app_message
    from symaware_grasp.beliefs import representative_component
    from symaware_grasp.grasp_collision import BottleCollisionModel
    collision = BottleCollisionModel(directory / manifest["model"])
    start = manifest["start_stamp"]
    robot, rows, events, results = ToyArm6DOF(), [], [], []
    belief, stage, target, previous_stamp, previous_theta = None, "startup", None, None, None
    topics = ["/joint_states", "/target_joint_states", EVENT_TOPIC,
              "/symaware_grasp/object_belief", "/symaware_grasp/symmetry_aware_ik_result"]
    with rosbag.Bag(str(directory / "experiment.bag")) as bag:
        recorded = bag.get_type_and_topic_info().topics
        if any(topic not in recorded for topic in topics + ["/probtf", "/probtf_static", "/tf", "/tf_static"]):
            raise ValueError("bag is missing required belief, IK or motion topics")
        for topic, message, stamp in bag.read_messages(topics=topics):
            stamp = stamp.to_sec()
            if topic == EVENT_TOPIC:
                event = json.loads(message.data)
                events.append(event)
                stage = event["stage"]
            elif topic == "/symaware_grasp/object_belief":
                belief = (stamp, record_from_app_message(message))
            elif topic == "/target_joint_states":
                mapping = dict(zip(message.name, message.position))
                if all(name in mapping for name in robot.joint_names):
                    target = [mapping[name] for name in robot.joint_names]
            elif topic == "/symaware_grasp/symmetry_aware_ik_result":
                results.append({key: getattr(message, key) for key in
                    ("solver_name", "grasp_id", "success", "total_cost", "position_cost",
                     "orientation_cost", "motion_cost", "joint_limit_cost")})
                results[-1]["theta_solution"] = list(message.theta_solution)
            elif topic == "/joint_states" and stamp >= start and belief is not None:
                if not 0 <= stamp - belief[0] <= 0.5:
                    continue
                mapping = dict(zip(message.name, message.position))
                if not all(name in mapping for name in robot.joint_names + GRIPPER_JOINT_NAMES):
                    continue
                theta = np.array([mapping[name] for name in robot.joint_names])
                _, position, quaternion = representative_component(belief[1])
                rotation = quat_to_rotmat(quaternion)
                axis = rotation[:, 2]
                opening = gripper_opening(mapping[GRIPPER_JOINT_NAMES[0]])
                row = dict(time_s=stamp-start, stage=stage)
                row.update(evaluate_grasp(theta, opening, position, axis, robot=robot, object_rotation=rotation))
                gap, pair = collision.clearance(theta, opening, position, axis)
                row.update(bottle_clearance_lower_bound_mm=1000*gap, closest_collision_pair=pair)
                row["joint_speed_rad_s"] = (float(np.max(np.abs(theta-previous_theta))) / (stamp-previous_stamp)
                    if previous_stamp is not None and stamp > previous_stamp else 0.0)
                for i, value in enumerate(theta):
                    row["joint_{}".format(i+1)] = float(value)
                    row["target_{}".format(i+1)] = float(target[i]) if target is not None else float(value)
                previous_stamp, previous_theta = stamp, theta.copy()
                rows.append(row)
    if not rows or not results:
        raise ValueError("bag has no grasp measurements or IK result")
    with (output / "samples.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    tail = [row for row in rows if row["stage"] == "hold_grasp"]
    if not tail:
        raise ValueError("bag has no final grasp observation")
    tail = [row for row in tail if row["time_s"] >= tail[-1]["time_s"]-1]
    final = tail[-1]
    smallest = min(rows, key=lambda row: row["bottle_clearance_lower_bound_mm"])
    collision_free = smallest["bottle_clearance_lower_bound_mm"] >= manifest.get("minimum_required_clearance_mm", .5)
    orientation_reached = manifest.get("distribution") != "single_pose" or max(row["orientation_mode_error_deg"] for row in tail) <= 2
    summary = {
        "completed": bool(manifest["completed"]), "kinematic_grasp_reached": bool(manifest["completed"]
            and collision_free and orientation_reached
            and max(row["tcp_object_distance_mm"] for row in tail) <= 2
            and max(row["axis_tilt_deg"] for row in tail) <= 2
            and max(abs(row["gripper_opening_mm"]-1000*manifest["closed_opening_m"]) for row in tail) <= 0.2),
        "ik_results": results, "samples": len(rows), "final": final,
        "collision_free": bool(collision_free), "minimum_bottle_clearance_lower_bound_mm": smallest["bottle_clearance_lower_bound_mm"],
        "closest_collision_pair": smallest["closest_collision_pair"],
        "distribution": manifest.get("distribution", "axisymmetric"), "object_yaw_pi": manifest.get("object_yaw_pi"),
        "events": events, "interpretation": "Kinematic approach and jaw closure; contact, friction and holding force are not simulated.",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    (output / "events.json").write_text(json.dumps(events, indent=2)+"\n")
    plot_trial(output, rows, events)
    return summary


def plot_trial(output, rows, events):
    import os
    os.environ.setdefault("MPLCONFIGDIR", str(output / "matplotlib_cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = [row["time_s"] for row in rows]
    fig, axes = plt.subplots(5, 1, figsize=(12, 12), sharex=True, constrained_layout=True)
    for ax, key, label in zip(axes, ("tcp_object_distance_mm", "axis_tilt_deg", "orientation_mode_error_deg", "gripper_opening_mm", "bottle_clearance_lower_bound_mm"),
                              ("TCP to bottle [mm]", "Tool/bottle axis tilt [deg]", "Error from bottle mode [deg]", "Jaw opening [mm]", "Certified clearance [mm]")):
        ax.plot(t, [row[key] for row in rows])
        for event in events:
            ax.axvline(event["time_s"], color="0.6", alpha=0.4)
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
    axes[3].axhline(100, color="k", ls=":", label="Bottle diameter")
    axes[3].legend()
    axes[4].axhline(0, color="r", ls=":", label="Interference boundary")
    axes[4].legend()
    axes[4].set_xlabel("Time [s]")
    fig.savefig(output / "overview.png", dpi=150)
    fig.savefig(output / "overview.pdf")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    for i in range(6):
        ax.plot(t, np.degrees([row["joint_{}".format(i+1)] for row in rows]), label="joint_{}".format(i+1))
    ax.set(xlabel="Time [s]", ylabel="Joint angle [deg]")
    ax.legend(ncol=3)
    ax.grid(alpha=0.3)
    fig.savefig(output / "joint_states.png", dpi=150)
    fig.savefig(output / "joint_states.pdf")
    plt.close(fig)
