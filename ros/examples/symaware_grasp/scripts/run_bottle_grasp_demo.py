#!/usr/bin/python3
"""Record one symmetry-aware bottle approach and kinematic jaw closure."""
import argparse
import ctypes
import datetime
from fractions import Fraction
import json
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import xmlrpc.client

import numpy as np
import yaml

PACKAGE = Path(__file__).resolve().parents[1]
if not (PACKAGE / "urdf/simple_six_dof_arm.urdf").is_file():
    import rospkg
    PACKAGE = Path(rospkg.RosPack().get_path("symaware_grasp"))
if (PACKAGE / "src").is_dir():
    sys.path.insert(0, str(PACKAGE / "src"))

from symaware_grasp.arm_kinematics import ToyArm6DOF
from symaware_grasp.grasp_collision import BottleCollisionModel
from symaware_grasp.grasp_trial import (
    EVENT_TOPIC, GRIPPER_JOINT_NAMES, MARKER_TOPIC, analyze_trial,
    animated_gripper_urdf, evaluate_grasp, gripper_opening, plan_approach,
    smooth_fraction, trajectory_reference,
)


class Processes:
    def __init__(self, directory, env):
        self.directory, self.env, self.children = directory, env, []

    def start(self, name, command, finite=False):
        log = (self.directory / (name + ".log")).open("w")
        child = subprocess.Popen(command, env=self.env, stdout=log,
                                 stderr=subprocess.STDOUT, start_new_session=True)
        self.children.append((name, child, log, finite))
        return child

    def check(self):
        for name, child, _, finite in self.children:
            if child.poll() is not None and (not finite or child.returncode != 0):
                raise RuntimeError("{} exited ({}); see {}.log".format(name, child.returncode, name))

    def stop(self, names=None):
        remaining = []
        for name, child, log, finite in reversed(self.children):
            if names is not None and name not in names:
                remaining.append((name, child, log, finite))
                continue
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGINT)
                try:
                    child.wait(timeout=12)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait(timeout=3)
            log.close()
        self.children = list(reversed(remaining))


def wait_master(uri, processes):
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        processes.check()
        try:
            if xmlrpc.client.ServerProxy(uri).getPid("/bottle_trial_probe")[0] == 1:
                return
        except OSError:
            pass
        time.sleep(0.1)
    raise RuntimeError("isolated ROS master did not start")


def make_rviz_config(destination, view="oblique"):
    config = yaml.safe_load((PACKAGE / "rviz/probabilistic_tf_demo.rviz").read_text())
    config["Panels"] = []
    config["Preferences"]["PromptSaveOnExit"] = False
    geometry = config["Window Geometry"]
    geometry.pop("QMainWindow State", None)
    geometry.update({"X": 60 if view == "oblique" else 1360, "Y": 80,
                     "Width": 1200, "Height": 950, "Hide Left Dock": True, "Hide Right Dock": True})
    current = config["Visualization Manager"]["Views"]["Current"]
    current.update({"Distance": 1.9, "Pitch": 0.45, "Yaw": 0.9,
                    "Focal Point": {"X": 0.45, "Y": 0.0, "Z": 0.43}})
    if view == "top":
        config["Visualization Manager"]["Views"]["Current"] = {
            "Class": "rviz/TopDownOrtho", "Name": "Current View", "Target Frame": "base_link",
            "Scale": 650, "Angle": 0, "X": .45, "Y": .1, "Near Clip Distance": .01}
    config["Visualization Manager"]["Displays"].append({
        "Class": "rviz/MarkerArray", "Enabled": True, "Value": True,
        "Name": "Bottle grasp progress", "Marker Topic": MARKER_TOPIC, "Queue Size": 10,
        "Namespaces": {"trial_progress": view != "top", "trial": True, "tcp_axes": True}})
    config["Visualization Manager"]["Displays"].append({
        "Class": "rviz/Axes", "Enabled": True, "Value": True, "Name": "TCP coordinate frame",
        "Reference Frame": "tool0", "Length": .14, "Radius": .006})
    destination.write_text(yaml.safe_dump(config, sort_keys=False))


def capture_rectangle(env, processes, process_id):
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        processes.check()
        tree = subprocess.check_output(["xwininfo", "-root", "-tree"], env=env, text=True)
        for line in tree.splitlines():
            match = re.search(r'(0x[0-9a-f]+) "[^\"]*(?:rviz|RViz)[^\"]*"', line)
            if not match:
                continue
            window = match.group(1)
            prop = subprocess.check_output(["xprop", "-id", window, "_NET_WM_PID"], env=env, text=True)
            pid = re.search(r"= (\d+)", prop)
            if not pid:
                continue
            if int(pid.group(1)) != process_id:
                continue
            try:
                entries = Path("/proc/{}/environ".format(pid.group(1))).read_bytes().split(b"\0")
            except OSError:
                continue
            if ("ROS_MASTER_URI=" + env["ROS_MASTER_URI"]).encode() not in entries:
                continue
            application_window = window
            subtree = subprocess.check_output(["xwininfo", "-id", window, "-tree"], env=env, text=True)
            candidates = []
            for child_line in subtree.splitlines():
                child = re.search(r'(0x[0-9a-f]+).*?\s(\d+)x(\d+)\+', child_line)
                if child and int(child.group(2)) >= 800 and int(child.group(3)) >= 500:
                    candidates.append((int(child.group(2))*int(child.group(3)), child.group(1)))
            if candidates:
                # The smallest large native child is the render widget; omit
                # menus, toolbars, dock handles and the status bar.
                window = min(candidates)[1]
            info = subprocess.check_output(["xwininfo", "-id", window], env=env, text=True)
            def field(name):
                return int(re.search(name + r":\s+(-?\d+)", info).group(1))
            x, y, width, height = (field(name) for name in
                ("Absolute upper-left X", "Absolute upper-left Y", "Width", "Height"))
            if width < 800 or height < 500:
                continue
            root = subprocess.check_output(["xwininfo", "-root"], env=env, text=True)
            root_width = int(re.search(r"Width:\s+(\d+)", root).group(1))
            root_height = int(re.search(r"Height:\s+(\d+)", root).group(1))
            if x < 0 or y < 0 or x+width > root_width or y+height > root_height:
                raise RuntimeError("RViz window extends outside the screen")
            xlib = ctypes.CDLL("libX11.so.6")
            xlib.XOpenDisplay.argtypes, xlib.XOpenDisplay.restype = [ctypes.c_char_p], ctypes.c_void_p
            xlib.XRaiseWindow.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
            xlib.XFlush.argtypes = [ctypes.c_void_p]
            xlib.XCloseDisplay.argtypes = [ctypes.c_void_p]
            display = xlib.XOpenDisplay(env["DISPLAY"].encode())
            if not display:
                raise RuntimeError("cannot open X display")
            xlib.XRaiseWindow(display, int(application_window, 16))
            xlib.XFlush(display)
            xlib.XCloseDisplay(display)
            return {"window_id": window, "application_window_id": application_window, "x": x, "y": y,
                    "width": width-width%2, "height": height-height%2}
        time.sleep(0.2)
    raise RuntimeError("RViz window for the trial was not found")


def start_dual_capture(directory, env, processes, windows, distribution, yaw_fraction):
    rectangles = {view: capture_rectangle(env, processes, child.pid) for view, child in windows.items()}
    oblique, top = rectangles["oblique"], rectangles["top"]
    if min(oblique["x"]+oblique["width"], top["x"]+top["width"]) > max(oblique["x"], top["x"]) and min(
            oblique["y"]+oblique["height"], top["y"]+top["height"]) > max(oblique["y"], top["y"]):
        raise RuntimeError("the two RViz capture areas overlap")
    x, y = min(r["x"] for r in rectangles.values()), min(r["y"] for r in rectangles.values())
    width = max(r["x"]+r["width"] for r in rectangles.values())-x
    height = max(r["y"]+r["height"] for r in rectangles.values())-y
    output_width, output_height = min(r["width"] for r in rectangles.values()), min(r["height"] for r in rectangles.values())
    filters = ["[0:v]split=2[oblique_raw][top_raw]"]
    label = "AXISYMMETRIC" if distribution == "axisymmetric" else "SINGLE POSE"
    for view, rectangle in rectangles.items():
        title = "{} - {} - bottle yaw {} pi".format(label, view.upper(), yaw_fraction)
        filters.append("[{}_raw]crop={}:{}:{}:{},scale={}:{},setsar=1,".format(
            view, rectangle["width"], rectangle["height"], rectangle["x"]-x, rectangle["y"]-y, output_width, output_height)
            + "drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf:text='{}':".format(title)
            + "x=16:y=12:fontsize=20:fontcolor=black:box=1:boxcolor=white@0.85:boxborderw=8,"
            + "split=2[{}][{}_combined]".format(view, view))
    filters.append("[oblique_combined][top_combined]hstack=inputs=2[combined]")
    command = ["ffmpeg", "-hide_banner", "-loglevel", "warning", "-nostdin", "-filter_complex_threads", "2",
        "-f", "x11grab", "-framerate", "20", "-draw_mouse", "0", "-video_size", "{}x{}".format(width, height),
        "-i", "{}+{},{}".format(env["DISPLAY"], x, y), "-filter_complex", ";".join(filters)]
    for stream, name in (("oblique", "rviz_oblique.mp4"), ("top", "rviz_top.mp4"), ("combined", "rviz_side_by_side.mp4")):
        command += ["-map", "[{}]".format(stream), "-c:v", "libx264", "-threads", "2", "-preset", "veryfast",
            "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(directory/name)]
    import rospy
    stamp = rospy.get_time()
    processes.start("ffmpeg", command)
    return dict(views=rectangles, x=x, y=y, width=width, height=height, fps=20,
        capture_command_start_stamp=stamp, synchronization="single X11 frame split into both views before encoding")


def finalize_videos(directory, manifest):
    probes = {}
    for view, filename in (("oblique", "rviz_oblique.mp4"), ("top", "rviz_top.mp4"), ("side_by_side", "rviz_side_by_side.mp4")):
        video = directory/filename
        if not video.exists():
            continue
        probe = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-show_entries",
            "format=duration:stream=codec_name,width,height,avg_frame_rate,nb_read_frames", "-of", "json", str(video)],
            text=True, capture_output=True, check=True)
        probes[view] = json.loads(probe.stdout)
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-sseof", "-2", "-i", str(video),
            "-frames:v", "1", str(directory/("grasp_"+view+".png"))], check=True)
    if probes:
        if len(probes) != 3 or len({p["streams"][0]["nb_read_frames"] for p in probes.values()}) != 1:
            raise RuntimeError("dual-view recordings do not have matching frame counts")
        manifest["video_probe"] = probes
        shutil.copy2(directory/"grasp_side_by_side.png", directory/"grasp_preview.png")
        (directory/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")


class Trial:
    def __init__(self, directory, processes):
        import rospy
        from sensor_msgs.msg import JointState
        from std_msgs.msg import Float64, String
        from symaware_grasp.msg import IKResult, ObjectBelief
        from visualization_msgs.msg import MarkerArray
        self.directory, self.processes = directory, processes
        self.robot, self.lock = ToyArm6DOF(), threading.Lock()
        self.collision = BottleCollisionModel(directory / "animated_gripper.urdf")
        self.state, self.belief, self.result = None, None, None
        self.stage, self.start_stamp, self.command_q = "startup", None, None
        self.reference = rospy.Publisher("/target_joint_states", JointState, queue_size=1, latch=True)
        self.gripper = rospy.Publisher("/gripper_opening_command", Float64, queue_size=1, latch=True)
        self.events = rospy.Publisher(EVENT_TOPIC, String, queue_size=10, latch=True)
        self.markers = rospy.Publisher(MARKER_TOPIC, MarkerArray, queue_size=1, latch=True)
        self.subscriptions = [
            rospy.Subscriber("/joint_states", JointState, lambda m: self.store("state", m), queue_size=1),
            rospy.Subscriber("/symaware_grasp/object_belief", ObjectBelief, lambda m: self.store("belief", m), queue_size=1),
            rospy.Subscriber("/symaware_grasp/symmetry_aware_ik_result", IKResult, lambda m: self.store("result", m), queue_size=1)]
        self.event_log = (directory / "events.jsonl").open("w")

    def store(self, key, message):
        with self.lock:
            setattr(self, key, message)

    def snapshot(self):
        with self.lock:
            return self.state, self.belief, self.result

    def ready(self):
        import rospy
        from sensor_msgs.msg import PointCloud2
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            self.processes.check()
            state, belief, _ = self.snapshot()
            if state is not None and belief is not None and self.reference.get_num_connections() >= 2 and self.events.get_num_connections() >= 1:
                break
            time.sleep(0.05)
        else:
            raise RuntimeError("trial publishers and recorder did not become ready")
        rospy.wait_for_message("/object_axes_cloud", PointCloud2, timeout=15)
        q, _ = self.current()
        self.publish_reference(q)

    def current(self):
        import rospy
        state, _, _ = self.snapshot()
        if state is None or abs(rospy.get_time()-state.header.stamp.to_sec()) > 0.5:
            raise RuntimeError("joint state is stale")
        mapping = dict(zip(state.name, state.position))
        q = np.array([mapping[name] for name in self.robot.joint_names])
        return q, gripper_opening(mapping[GRIPPER_JOINT_NAMES[0]])

    def object_geometry(self):
        position, rotation = self.object_transform()
        return position, rotation[:, 2]

    def object_transform(self):
        from probtf.geometry import quat_to_rotmat
        from symaware_grasp.beliefs import representative_component
        from symaware_grasp_ros import record_from_app_message
        _, belief, _ = self.snapshot()
        _, position, quaternion = representative_component(record_from_app_message(belief))
        return position, quat_to_rotmat(quaternion)

    def publish_reference(self, theta):
        import rospy
        from sensor_msgs.msg import JointState
        self.command_q = np.array(theta).copy()
        message = JointState()
        message.header.stamp = rospy.Time.now()
        message.name, message.position = self.robot.joint_names, self.command_q.tolist()
        self.reference.publish(message)

    def set_stage(self, stage, details=None):
        import rospy
        from std_msgs.msg import String
        self.stage = stage
        event = {"stage": stage, "stamp": rospy.get_time(), "time_s": rospy.get_time()-self.start_stamp,
                 "details": details or {}}
        text = json.dumps(event)
        self.events.publish(String(data=text))
        self.event_log.write(text+"\n")
        self.event_log.flush()
        print("{:5.1f}s {}".format(event["time_s"], stage), flush=True)

    def display(self):
        import rospy
        from visualization_msgs.msg import Marker, MarkerArray
        q, opening = self.current()
        position, axis = self.object_geometry()
        values = evaluate_grasp(q, opening, position, axis, robot=self.robot)
        array = MarkerArray()
        text = Marker()
        text.header.frame_id, text.header.stamp = "base_link", rospy.Time.now()
        text.ns, text.id, text.type = "trial_progress", 0, Marker.TEXT_VIEW_FACING
        text.pose.orientation.w, text.pose.position.x, text.pose.position.z = 1, 0.45, 1.03
        text.scale.z, text.color.r, text.color.g, text.color.b, text.color.a = 0.026, 0.1, 0.1, 0.1, 1
        text.text = "{:.1f} s | {}\nTCP to bottle: {:.1f} mm | Axis tilt: {:.2f} deg\nGripper opening: {:.1f} mm".format(
            rospy.get_time()-self.start_stamp, self.stage.replace("_", " "),
            values["tcp_object_distance_mm"], values["axis_tilt_deg"], values["gripper_opening_mm"])
        array.markers.append(text)
        for index, (label, coordinate) in enumerate((("TCP X", (.16, 0, 0)), ("TCP Y", (0, .16, 0)), ("TCP Z", (0, 0, .16))), 10):
            marker = Marker()
            marker.header.frame_id, marker.header.stamp = "tool0", rospy.Time.now()
            marker.ns, marker.id, marker.type = "tcp_axes", index, Marker.TEXT_VIEW_FACING
            marker.pose.orientation.w = 1
            marker.pose.position.x, marker.pose.position.y, marker.pose.position.z = coordinate
            marker.text, marker.scale.z, marker.color.a = label, .018, 1
            marker.color.r, marker.color.g, marker.color.b = {"TCP X": (1, 0, 0), "TCP Y": (0, .65, 0), "TCP Z": (0, 0, 1)}[label]
            array.markers.append(marker)
        # Add a neck and cap to the existing cylindrical bottle body. They share
        # its upright symmetry axis; the grasp target remains the body center.
        for index, (height, radius, length, color) in enumerate(
                ((0.105, 0.025, 0.03, (0.35, 0.65, 0.85)), (0.128, 0.029, 0.016, (0.9, 0.9, 0.9))), 1):
            marker = Marker()
            marker.header.frame_id, marker.header.stamp = "base_link", rospy.Time.now()
            marker.ns, marker.id, marker.type = "trial", index, Marker.CYLINDER
            marker.pose.orientation.w = 1
            marker.pose.position.x, marker.pose.position.y, marker.pose.position.z = (position + height*axis).tolist()
            marker.scale.x = marker.scale.y = 2*radius
            marker.scale.z = length
            marker.color.r, marker.color.g, marker.color.b = color
            marker.color.a = 1
            array.markers.append(marker)
        self.markers.publish(array)

    def tick(self, duration, update=None):
        import rospy
        begin, next_display = time.monotonic(), 0
        while time.monotonic()-begin < duration:
            if rospy.is_shutdown():
                raise KeyboardInterrupt()
            self.processes.check()
            q, opening = self.current()
            position, axis = self.object_geometry()
            gap, pair = self.collision.clearance(q, opening, position, axis)
            if gap < .0005:
                raise RuntimeError("observed bottle clearance is insufficient ({:.2f} mm): {}".format(1000*gap, pair))
            elapsed = time.monotonic()-begin
            if update is not None:
                self.publish_reference(update(elapsed))
            elif self.command_q is not None:
                self.publish_reference(self.command_q)
            if time.monotonic() >= next_display:
                self.display()
                next_display = time.monotonic()+0.1
            time.sleep(min(1/30, max(0, duration-(time.monotonic()-begin))))

    def wait_motion(self, target):
        deadline, settled = time.monotonic()+15, None
        while time.monotonic() < deadline:
            q, _ = self.current()
            if np.max(np.abs(q-target)) <= 0.001:
                settled = time.monotonic() if settled is None else settled
                if time.monotonic()-settled >= 0.2:
                    return
            else:
                settled = None
            self.tick(0.05)
        raise RuntimeError("arm did not reach the planned waypoint")

    def execute(self, manifest):
        import rospy
        from std_msgs.msg import Float64
        self.start_stamp = rospy.get_time()
        manifest["start_stamp"] = self.start_stamp
        manifest_path = self.directory / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2)+"\n")
        self.set_stage("observe_bottle")
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
        metrics = evaluate_grasp(grasp_q, 0.185, position, axis, robot=self.robot, object_rotation=rotation)
        if metrics["tcp_object_distance_mm"] > 2 or metrics["axis_tilt_deg"] > 2:
            raise RuntimeError("IK result does not reach the bottle with an aligned gripper")
        if manifest["distribution"] == "single_pose" and metrics["orientation_mode_error_deg"] > 2:
            raise RuntimeError("IK does not respect the constrained bottle orientation")
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
        self.set_stage("approach_bottle")
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
                raise ValueError(executable+" is required")
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", args.port))
    directory.mkdir(parents=True, exist_ok=False)
    original = PACKAGE / "urdf/simple_six_dof_arm.urdf"
    model = directory / "animated_gripper.urdf"
    model.write_text(animated_gripper_urdf(original.read_text()))
    shutil.copy2(original, directory / "arm.urdf")
    shutil.copy2(PACKAGE / "config/grasp_library.yaml", directory / "grasp_library.yaml")
    rviz_configs = {view: directory/("bottle_grasp_"+view+".rviz") for view in ("oblique", "top")}
    for view, config in rviz_configs.items():
        make_rviz_config(config, view)
    yaw_fraction = str(Fraction(args.object_yaw_pi))
    yaw_degrees = 180*float(Fraction(yaw_fraction))
    yaw_concentration = 0 if args.distribution == "axisymmetric" else 5000
    env = dict(os.environ, ROS_MASTER_URI="http://127.0.0.1:{}".format(args.port), ROS_IP="127.0.0.1",
               ROS_HOME=str(directory / "ros_home"), ROS_LOG_DIR=str(directory / "ros_logs"), QT_QPA_PLATFORM="xcb")
    env.pop("ROS_HOSTNAME", None)
    os.environ.update({key: env[key] for key in ("ROS_MASTER_URI", "ROS_IP", "ROS_HOME", "ROS_LOG_DIR")})
    os.environ.pop("ROS_HOSTNAME", None)
    manifest = {"created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "completed": False, "closed_opening_m": 0.106, "open_opening_m": 0.185,
                "approach_distance_m": 0.12, "ros_master_uri": env["ROS_MASTER_URI"],
                "model": "animated_gripper.urdf", "video": None,
                "mode": "kinematic grasp animation; no contact dynamics",
                "distribution": args.distribution, "object_yaw_pi": yaw_fraction,
                "object_yaw_deg": yaw_degrees, "bingham_concentrations": [5000, 5000, yaw_concentration],
                "collision_check": "URDF collision boxes (cylinders conservatively boxed) against exact bottle cylinders",
                "minimum_required_clearance_mm": .5}
    manifest_path = directory / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2)+"\n")
    processes, trial, error = Processes(directory, env), None, None
    print("Output: "+str(directory), flush=True)
    try:
        processes.start("roscore", ["roscore", "-p", str(args.port)])
        wait_master(env["ROS_MASTER_URI"], processes)
        processes.start("scene", ["roslaunch", "symaware_grasp", "probabilistic_tf_demo.launch",
            "model:="+str(model), "gripper_enabled:=true", "object_yaw_deg:="+str(yaw_degrees),
            "yaw_concentration:="+str(yaw_concentration), "rviz:=false"])
        windows = {}
        if not args.no_video:
            for view, config in rviz_configs.items():
                windows[view] = processes.start("rviz_"+view, ["rviz", "-d", str(config), "__name:=rviz_"+view])
        processes.start("rosbag", ["rosbag", "record", "--all", "--buffsize", "256", "--output-name", str(directory / "experiment.bag")])
        rospy.init_node("bottle_grasp_trial", disable_signals=True)
        trial = Trial(directory, processes)
        trial.ready()
        (directory / "parameters.yaml").write_text(yaml.safe_dump(rospy.get_param("/"), sort_keys=False))
        if not args.no_video:
            manifest["video"] = start_dual_capture(directory, env, processes, windows, args.distribution, yaw_fraction)
            time.sleep(0.5)
        trial.execute(manifest)
    except BaseException as exc:
        error = exc
        manifest["error"] = "{}: {}".format(type(exc).__name__, exc)
        manifest_path.write_text(json.dumps(manifest, indent=2)+"\n")
        print(manifest["error"], file=sys.stderr, flush=True)
    finally:
        if trial is not None:
            trial.event_log.close()
        processes.stop({"rosbag", "ffmpeg"})
        processes.stop()
        if trial is not None:
            rospy.signal_shutdown("bottle grasp trial finished")
    finalize_videos(directory, manifest)
    if error is not None:
        raise error
    summary = analyze_trial(directory)
    if not summary["kinematic_grasp_reached"]:
        raise RuntimeError("recorded final pose does not meet the kinematic grasp criteria")
    print("Collision-free grasp reached; saved synchronized videos, bag, CSV, graphs and summary: "+str(directory), flush=True)
    return summary


def compare(args):
    directory = Path(args.output).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    summaries = {}
    for distribution in ("axisymmetric", "single_pose"):
        command = [sys.executable, str(PACKAGE/"scripts/run_bottle_grasp_demo.py"), "run",
            "--distribution", distribution, "--object-yaw-pi", args.object_yaw_pi,
            "--output", str(directory/distribution), "--port", str(args.port)]
        if args.no_video:
            command.append("--no-video")
        subprocess.run(command, check=True)
        summaries[distribution] = json.loads((directory/distribution/"summary.json").read_text())
    (directory/"comparison.json").write_text(json.dumps(summaries, indent=2)+"\n")
    print("Completed both distribution conditions: "+str(directory), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    runner = commands.add_parser("run")
    runner.add_argument("--output", default=str(PACKAGE / "outputs/bottle_grasp_trial"))
    runner.add_argument("--port", type=int, default=11765)
    runner.add_argument("--no-video", action="store_true")
    runner.add_argument("--distribution", choices=("axisymmetric", "single_pose"), default="axisymmetric")
    runner.add_argument("--object-yaw-pi", default="3/8")
    comparison = commands.add_parser("compare")
    comparison.add_argument("--output", default=str(PACKAGE / "outputs/bottle_grasp_distribution_comparison"))
    comparison.add_argument("--object-yaw-pi", default="3/8")
    comparison.add_argument("--port", type=int, default=11765)
    comparison.add_argument("--no-video", action="store_true")
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
        summary = analyze_trial(args.directory, args.output)
        print(json.dumps(summary["final"], indent=2))


if __name__ == "__main__":
    main()
