#!/usr/bin/python3
"""Compare one four-component box-grasp target with one concentrated target.

Reuse the bottle demo's arm, controllers, IK node, Cartesian approach and
synchronized screen recording. Only the object and orientation law differ.
"""
import argparse
import datetime
from fractions import Fraction
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time

import numpy as np
import yaml

PACKAGE = Path(__file__).resolve().parents[1]
if not (PACKAGE / "urdf/simple_six_dof_arm.urdf").is_file():
    import rospkg
    PACKAGE = Path(rospkg.RosPack().get_path("symaware_grasp"))
sys.path.insert(0, str(PACKAGE / "src"))
spec = importlib.util.spec_from_file_location("bottle_demo", PACKAGE / "scripts/run_bottle_grasp_demo.py")
bottle_demo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bottle_demo)

from probtf.geometry import quat_mul, quat_to_rotmat, rpy_to_quat
from symaware_grasp.box_geometry import BOX_DIMENSIONS, BOX_POSITION, BoxCollisionModel, box_mode_quaternions, evaluate_box_grasp
from symaware_grasp.box_trial_analysis import analyze_box_trial
from symaware_grasp.grasp_library import load_grasp_library
from symaware_grasp.grasp_trial import animated_gripper_urdf, plan_approach, smooth_fraction, trajectory_reference

Processes = bottle_demo.Processes


def verify_publisher_parameters(manifest, parameters, grasp_orientation):
    """Check the actual publisher state and retain its exact component values."""
    node = parameters.get("object_pose_node", {})
    expected = manifest["bingham_components"]
    actual = node.get("components", [])
    if node.get("distribution") != manifest["distribution"]:
        raise RuntimeError("box publisher distribution differs from the trial manifest")
    if node.get("component_count") != len(expected) or len(actual) != len(expected):
        raise RuntimeError("box publisher component count differs from the trial manifest")

    def matches(values, reference):
        values, reference = np.asarray(values, dtype=float), np.asarray(reference, dtype=float)
        return values.shape == reference.shape and np.all(np.isfinite(values)) and np.allclose(
            values, reference, rtol=0, atol=1e-8)

    def same_quaternion(values, reference):
        values, reference = np.asarray(values, dtype=float), np.asarray(reference, dtype=float)
        return (values.shape == (4,) and np.all(np.isfinite(values))
                and abs(np.linalg.norm(values) - 1) < 1e-10
                and abs(abs(float(np.dot(values, reference))) - 1) < 1e-10)

    physical = manifest["physical_box"]
    if not matches(node.get("dimensions"), physical["size"]):
        raise RuntimeError("box publisher dimensions differ from the collision geometry")
    if not matches(node.get("true_position"), physical["position"]):
        raise RuntimeError("box publisher true position differs from the collision geometry")
    true_quaternion = node.get("true_quaternion_wxyz")
    if not same_quaternion(true_quaternion, expected[0]["quaternion_wxyz"]) or not matches(
            quat_to_rotmat(true_quaternion), physical["rotation"]):
        raise RuntimeError("box publisher true orientation differs from the collision geometry")
    concentration = float(node.get("concentration", float("nan")))
    if not np.isfinite(concentration) or concentration != 5000:
        raise RuntimeError("box publisher concentration differs from the trial configuration")
    if not matches(node.get("position_mean"), physical["position"]) or not matches(
            node.get("position_covariance"), np.zeros(9)):
        raise RuntimeError("box publisher translation law differs from the trial configuration")
    if manifest["target_candidate_count"] != 1:
        raise RuntimeError("the box trial requires one complete mixed grasp target")

    verified_components, target_quaternions = [], []
    for source, reference, target_reference in zip(actual, expected, manifest["target_quaternions"]):
        weight = float(source.get("weight", float("nan")))
        mode = source.get("mode_wxyz")
        matrix = source.get("parameter_matrix")
        if not np.isfinite(weight) or abs(weight - reference["weight"]) > 1e-12:
            raise RuntimeError("box publisher component weights differ from the trial manifest")
        if not same_quaternion(mode, reference["quaternion_wxyz"]) or not matches(
                matrix, reference["parameter_matrix"]):
            raise RuntimeError("box publisher Bingham components differ from the trial manifest")
        target = quat_mul(mode, grasp_orientation)
        if not same_quaternion(target, target_reference):
            raise RuntimeError("the composed grasp mode differs from the trial target quaternion")
        verified_components.append({"component_id": source["component_id"], "weight": weight,
                                    "quaternion_wxyz": list(mode), "parameter_matrix": matrix,
                                    "concentrations": [concentration] * 3})
        target_quaternions.append(target.tolist())
    if len(manifest["target_quaternions"]) != len(expected) or abs(sum(
            item["weight"] for item in verified_components) - 1) > 1e-12:
        raise RuntimeError("the trial modes or normalized component weights are incomplete")
    manifest["bingham_components"] = verified_components
    manifest["target_quaternions"] = target_quaternions
    manifest["publisher_parameters_verified"] = True
    manifest["publisher_components_source"] = "/object_pose_node/components"


def make_rviz_config(destination, view):
    bottle_demo.make_rviz_config(destination, view)
    config = yaml.safe_load(destination.read_text())
    for display in config["Visualization Manager"]["Displays"]:
        if display.get("Name") == "Cylinder Object":
            display["Name"] = "Box Object"
        if display.get("Marker Topic") == bottle_demo.MARKER_TOPIC:
            display["Name"] = "Grasp frame"
            display["Namespaces"] = {"trial_progress": False, "trial": False, "tcp_axes": False}
    destination.write_text(yaml.safe_dump(config, sort_keys=False))


def start_dual_capture(directory, env, processes, windows):
    rectangles = {view: bottle_demo.capture_rectangle(env, processes, child.pid)
                  for view, child in windows.items()}
    oblique, top = rectangles["oblique"], rectangles["top"]
    if min(oblique["x"] + oblique["width"], top["x"] + top["width"]) > max(oblique["x"], top["x"]) and min(
            oblique["y"] + oblique["height"], top["y"] + top["height"]) > max(oblique["y"], top["y"]):
        raise RuntimeError("the two RViz capture areas overlap")
    x, y = min(r["x"] for r in rectangles.values()), min(r["y"] for r in rectangles.values())
    width = max(r["x"] + r["width"] for r in rectangles.values()) - x
    height = max(r["y"] + r["height"] for r in rectangles.values()) - y
    output_width = min(r["width"] for r in rectangles.values())
    output_height = min(r["height"] for r in rectangles.values())
    filters = ["[0:v]split=2[oblique_raw][top_raw]"]
    for view, rectangle in rectangles.items():
        filters.append("[{}_raw]crop={}:{}:{}:{},scale={}:{},setsar=1,split=2[{}][{}_combined]".format(
            view, rectangle["width"], rectangle["height"], rectangle["x"] - x, rectangle["y"] - y,
            output_width, output_height, view, view))
    filters.append("[oblique_combined][top_combined]hstack=inputs=2[combined]")
    command = ["ffmpeg", "-hide_banner", "-loglevel", "warning", "-nostdin", "-filter_complex_threads", "2",
               "-f", "x11grab", "-framerate", "20", "-draw_mouse", "0", "-video_size", f"{width}x{height}",
               "-i", f"{env['DISPLAY']}+{x},{y}", "-filter_complex", ";".join(filters)]
    for stream, name in (("oblique", "rviz_oblique.mp4"), ("top", "rviz_top.mp4"), ("combined", "rviz_side_by_side.mp4")):
        command += ["-map", f"[{stream}]", "-c:v", "libx264", "-threads", "2", "-preset", "veryfast",
                    "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(directory / name)]
    import rospy
    stamp = rospy.get_time()
    processes.start("ffmpeg", command)
    return dict(views=rectangles, x=x, y=y, width=width, height=height, fps=20,
                capture_command_start_stamp=stamp,
                synchronization="single X11 frame split into both views before encoding")


class Trial(bottle_demo.Trial):
    def __init__(self, directory, processes, manifest):
        super().__init__(directory, processes)
        physical = manifest["physical_box"]
        self.physical_position = np.asarray(physical["position"])
        self.physical_rotation = np.asarray(physical["rotation"])
        self.collision = BoxCollisionModel(directory / manifest["model"],
                                          dimensions=physical["size"], object_rotation=self.physical_rotation)

    def object_transform(self):
        return self.physical_position.copy(), self.physical_rotation.copy()

    def display(self):
        # The fixed true box is published independently by box_pose_node.
        # The reused RViz Axes display already shows the hand coordinate frame.
        pass

    def execute(self, manifest):
        import rospy
        from std_msgs.msg import Float64
        self.start_stamp = rospy.get_time()
        manifest["start_stamp"] = self.start_stamp
        manifest_path = self.directory / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2)+"\n")
        self.set_stage("observe_box")
        self.tick(2)
        self.set_stage("solve_symmetry_aware_ik")
        begin = time.monotonic()
        self.processes.start("ik", ["rosrun", "--prefix", "/usr/bin/python3", "symaware_grasp",
                            "symmetry_aware_ik_node.py", "_publish_command:=false"], finite=True)
        while self.snapshot()[2] is None:
            if time.monotonic()-begin > 30:
                raise RuntimeError("IK did not publish a result")
            self.tick(0.1)
        result = self.snapshot()[2]
        manifest["ik_solve_observed_s"] = time.monotonic()-begin
        if not result.success:
            raise RuntimeError("IK reported failure")
        grasp_q = np.array(result.theta_solution)
        position, rotation = self.object_transform()
        axis = rotation[:, 2]
        metrics = evaluate_box_grasp(grasp_q, 0.185, position, manifest["target_quaternions"],
                                     robot=self.robot, dimensions=manifest["physical_box"]["size"], object_rotation=rotation)
        if metrics["tcp_object_distance_mm"] > 2 or metrics["nearest_mode_error_deg"] > 2:
            raise RuntimeError("IK result does not reach the box with an allowed gripper orientation")
        pregrasp, waypoints = plan_approach(self.robot, grasp_q)
        q, _ = self.current()
        high = np.array([.3, -2, 1.5, 0, .5, 0])
        transfer = np.array([q, high, pregrasp])
        manifest["path_clearance"] = {
            "transfer": self.collision.validate_path(transfer, manifest["open_opening_m"], position, axis, minimum=.002),
            "approach": self.collision.validate_path(waypoints, manifest["open_opening_m"], position, axis, minimum=.002)}
        for opening in np.linspace(manifest["open_opening_m"], manifest["closed_opening_m"], 31):
            if self.collision.clearance(waypoints[-1], opening, position, axis)[0] < .002:
                raise RuntimeError("planned jaw closure has insufficient clearance")
        manifest["selected_theta"] = grasp_q.tolist()
        manifest["approach_waypoints"] = waypoints.tolist()
        manifest["transfer_waypoints"] = transfer.tolist()
        manifest_path.write_text(json.dumps(manifest, indent=2)+"\n")
        self.set_stage("move_to_pregrasp")
        for start, end in zip(transfer[:-1], transfer[1:]):
            self.tick(4, lambda elapsed: start+(end-start)*smooth_fraction(elapsed, 4))
            self.publish_reference(end)
            self.wait_motion(end)
        self.publish_reference(pregrasp)
        self.wait_motion(pregrasp)
        self.set_stage("pregrasp_pause")
        self.tick(1)
        self.set_stage("approach_box")
        self.tick(2, lambda elapsed: trajectory_reference(waypoints, smooth_fraction(elapsed, 2)))
        self.publish_reference(waypoints[-1])
        self.wait_motion(waypoints[-1])
        self.set_stage("close_gripper")
        self.gripper.publish(Float64(data=manifest["closed_opening_m"]))
        deadline = time.monotonic()+5
        while abs(self.current()[1]-manifest["closed_opening_m"]) > 0.0001:
            if time.monotonic() > deadline:
                raise RuntimeError("gripper did not close")
            self.tick(0.05)
        self.set_stage("hold_grasp")
        self.tick(5)
        self.set_stage("finished")
        manifest["end_stamp"], manifest["completed"] = rospy.get_time(), True
        manifest_path.write_text(json.dumps(manifest, indent=2)+"\n")

def run(args):
    import rospy
    directory = Path(args.output).resolve()
    if not args.no_video:
        if not os.environ.get("DISPLAY"):
            raise ValueError("DISPLAY is required for RViz capture")
        for executable in ("ffmpeg", "ffprobe", "xwininfo", "xprop"):
            if not shutil.which(executable):
                raise ValueError(executable + " is required")
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", args.port))
    directory.mkdir(parents=True, exist_ok=False)
    original = PACKAGE / "urdf/simple_six_dof_arm.urdf"
    model = directory / "animated_gripper.urdf"
    model.write_text(animated_gripper_urdf(original.read_text()))
    shutil.copy2(original, directory / "arm.urdf")
    shutil.copy2(PACKAGE / "config/box_grasp_library.yaml", directory / "grasp_library.yaml")
    rviz_configs = {view: directory / ("box_grasp_" + view + ".rviz") for view in ("oblique", "top")}
    for view, config in rviz_configs.items():
        make_rviz_config(config, view)
    yaw_fraction = str(Fraction(args.object_yaw_pi))
    yaw_degrees = 180 * float(Fraction(yaw_fraction))
    modes = box_mode_quaternions(yaw_degrees)
    if args.distribution == "single_pose":
        modes = modes[:1]
    physical_rotation = quat_to_rotmat(rpy_to_quat(0, 0, np.radians(yaw_degrees)))
    grasp = load_grasp_library(directory / "grasp_library.yaml", "demo_box")[0]
    target_quaternions = [quat_mul(mode, grasp.object_to_grasp_orientation_wxyz).tolist() for mode in modes]
    env = dict(os.environ, ROS_MASTER_URI="http://127.0.0.1:{}".format(args.port), ROS_IP="127.0.0.1",
               ROS_HOME=str(directory / "ros_home"), ROS_LOG_DIR=str(directory / "ros_logs"), QT_QPA_PLATFORM="xcb")
    env.pop("ROS_HOSTNAME", None)
    os.environ.update({key: env[key] for key in ("ROS_MASTER_URI", "ROS_IP", "ROS_HOME", "ROS_LOG_DIR")})
    os.environ.pop("ROS_HOSTNAME", None)
    modules = ["arm_kinematics.py", "grasp_trial.py", "symmetry_aware_ik.py", "grasp_targets.py"]
    scripts = ["arm_joint_state_controller.py", "symmetry_aware_ik_node.py", "run_bottle_grasp_demo.py"]
    manifest = {
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "completed": False,
        "closed_opening_m": 0.106, "open_opening_m": 0.185, "approach_distance_m": 0.12,
        "ros_master_uri": env["ROS_MASTER_URI"], "model": "animated_gripper.urdf",
        "model_sha256": hashlib.sha256(model.read_bytes()).hexdigest(), "video": None,
        "common_module_sha256": {name: hashlib.sha256((PACKAGE / "src/symaware_grasp" / name).read_bytes()).hexdigest()
                                 for name in modules},
        "common_script_sha256": {name: hashlib.sha256((PACKAGE / "scripts" / name).read_bytes()).hexdigest()
                                 for name in scripts},
        "mode": "kinematic grasp animation; no contact dynamics", "distribution": args.distribution,
        "object_yaw_pi": yaw_fraction, "object_yaw_deg": yaw_degrees,
        "physical_box": {"size": list(BOX_DIMENSIONS), "position": list(BOX_POSITION), "rotation": physical_rotation.tolist()},
        "target_quaternions": target_quaternions,
        "bingham_components": [{"weight": 1 / len(modes), "quaternion_wxyz": mode.tolist(),
                                 "parameter_matrix": (5000 * (np.outer(mode, mode) - np.eye(4))).tolist(),
                                 "concentrations": [5000, 5000, 5000]} for mode in modes],
        "target_candidate_count": 1, "orientation_objective": "negative log-sum-exp of normalized mixture density",
        "collision_check": "URDF collision boxes (cylinders conservatively boxed) against a fixed exact cuboid; 15 SAT axes",
        "minimum_required_clearance_mm": 0.5,
    }
    manifest_path = directory / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    processes, trial, error = Processes(directory, env), None, None
    print("Output: " + str(directory), flush=True)
    try:
        processes.start("roscore", ["roscore", "-p", str(args.port)])
        bottle_demo.wait_master(env["ROS_MASTER_URI"], processes)
        processes.start("scene", ["roslaunch", "symaware_grasp", "box_grasp_demo.launch",
                                 "model:=" + str(model), "gripper_enabled:=true", "object_yaw_deg:=" + str(yaw_degrees),
                                 "distribution:=" + args.distribution, "rviz:=false"])
        windows = {}
        if not args.no_video:
            for view, config in rviz_configs.items():
                windows[view] = processes.start("rviz_" + view, ["rviz", "-d", str(config), "__name:=rviz_" + view])
        processes.start("rosbag", ["rosbag", "record", "--all", "--buffsize", "256", "--output-name", str(directory / "experiment.bag")])
        rospy.init_node("box_grasp_trial", disable_signals=True)
        trial = Trial(directory, processes, manifest)
        trial.ready()
        parameters = rospy.get_param("/")
        (directory / "parameters.yaml").write_text(yaml.safe_dump(parameters, sort_keys=False))
        verify_publisher_parameters(manifest, parameters, grasp.object_to_grasp_orientation_wxyz)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        if not args.no_video:
            manifest["video"] = start_dual_capture(directory, env, processes, windows)
            time.sleep(0.5)
        trial.execute(manifest)
    except BaseException as exc:
        error = exc
        manifest["error"] = "{}: {}".format(type(exc).__name__, exc)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(manifest["error"], file=sys.stderr, flush=True)
    finally:
        if trial is not None:
            trial.event_log.close()
        processes.stop({"rosbag", "ffmpeg"})
        processes.stop()
        if trial is not None:
            rospy.signal_shutdown("box grasp trial finished")
    bottle_demo.finalize_videos(directory, manifest)
    if error is not None:
        raise error
    summary = analyze_box_trial(directory)
    if not summary["kinematic_grasp_reached"]:
        raise RuntimeError("recorded final pose does not meet the kinematic grasp criteria")
    print("Box grasp reached; synchronized views and measurements saved: " + str(directory), flush=True)
    return summary


def compare(args):
    directory = Path(args.output).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    summaries = {}
    for distribution in ("mixture", "single_pose"):
        command = [sys.executable, str(PACKAGE / "scripts/run_box_grasp_demo.py"), "run",
                   "--distribution", distribution, "--object-yaw-pi", args.object_yaw_pi,
                   "--output", str(directory / distribution), "--port", str(args.port)]
        if args.no_video:
            command.append("--no-video")
        subprocess.run(command, check=True)
        summaries[distribution] = json.loads((directory / distribution / "summary.json").read_text())
    (directory / "comparison.json").write_text(json.dumps(summaries, indent=2) + "\n")
    print("Completed both box-grasp conditions: " + str(directory), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name, default in (("run", "box_grasp_trial"), ("compare", "box_grasp_distribution_comparison")):
        command = commands.add_parser(name)
        command.add_argument("--output", default=str(PACKAGE / "outputs" / default))
        command.add_argument("--object-yaw-pi", default="3/8")
        command.add_argument("--port", type=int, default=11981)
        command.add_argument("--no-video", action="store_true")
        if name == "run":
            command.add_argument("--distribution", choices=("mixture", "single_pose"), default="mixture")
    analyzer = commands.add_parser("analyze")
    analyzer.add_argument("directory")
    analyzer.add_argument("--output")
    args = parser.parse_args()
    if args.command == "run":
        def interrupt(signum, frame):
            raise KeyboardInterrupt()
        signal.signal(signal.SIGTERM, interrupt)
        run(args)
    elif args.command == "compare":
        compare(args)
    else:
        summary = analyze_box_trial(args.directory, args.output)
        print(json.dumps(summary["final"], indent=2))


if __name__ == "__main__":
    main()
