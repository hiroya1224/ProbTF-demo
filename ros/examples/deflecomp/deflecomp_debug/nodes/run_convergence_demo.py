#!/usr/bin/python3
"""Run an isolated ROS simulation, record RViz/bag, and evaluate predictions."""
import argparse
import ctypes
import datetime
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import threading
import time
import xmlrpc.client

import numpy as np
import yaml

# Also support execution directly from the checkout before a catkin build.
PACKAGE = Path(__file__).resolve().parents[1]
if not (PACKAGE / "config/convergence_demo.yaml").is_file():
    import rospkg
    PACKAGE = Path(rospkg.RosPack().get_path("deflecomp_debug"))
if (PACKAGE / "src").is_dir():
    sys.path.insert(0, str(PACKAGE / "src"))
from deflecomp_debug.convergence_demo import (
    EVENT_TOPIC, JOINT_TOPICS, MARKER_TOPIC, VECTOR_TOPICS,
    PredictionConvergenceMonitor,
    analyze_recording, evaluate_snapshot, interpolate_reference,
    validate_scenario,
)


class Processes:
    def __init__(self, directory, env):
        self.directory, self.env, self.children = directory, env, []

    def start(self, name, command):
        log = (self.directory / f"{name}.log").open("w")
        try:
            child = subprocess.Popen(command, env=self.env, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
        except BaseException:
            log.close()
            raise
        self.children.append((name, child, log))
        return child

    def check(self):
        for name, child, _ in self.children:
            if child.poll() is not None:
                raise RuntimeError(f"{name} exited ({child.returncode}); see {name}.log")

    def stop(self, names=None):
        remaining = []
        for name, child, log in reversed(self.children):
            if names is not None and name not in names:
                remaining.append((name, child, log))
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
            if xmlrpc.client.ServerProxy(uri).getPid("/convergence_demo_probe")[0] == 1:
                return
        except OSError:
            pass
        time.sleep(0.1)
    raise RuntimeError("isolated ROS master did not start")


def make_rviz_config(destination, view=None):
    import rospkg
    path = Path(rospkg.RosPack().get_path("deflecomp_description")) / "rviz/deflecomp.rviz"
    config = yaml.safe_load(path.read_text())
    config["Preferences"]["PromptSaveOnExit"] = False
    geometry = config["Window Geometry"]
    geometry.pop("QMainWindow State", None)
    geometry.update({"X": 60, "Y": 60, "Width": 1500, "Height": 950,
                     "Hide Left Dock": True, "Hide Right Dock": True})
    current = config["Visualization Manager"]["Views"]["Current"]
    current.update({"Distance": 1.65, "Pitch": 0.3, "Yaw": 2.45,
                    "Focal Point": {"X": 0.0, "Y": 0.0, "Z": 0.32}})
    if view:
        current.update(view)
    config["Visualization Manager"]["Displays"].append({
        "Class": "rviz/MarkerArray", "Enabled": True, "Value": True,
        "Name": "Demo progress and prediction", "Marker Topic": MARKER_TOPIC,
        "Queue Size": 10,
    })
    destination.write_text(yaml.safe_dump(config, sort_keys=False))


def capture_rectangle(env, processes):
    """Find only an RViz window belonging to this isolated ROS master."""
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
            pid_match = re.search(r"= (\d+)", prop)
            if not pid_match:
                continue
            try:
                proc_env = Path(f"/proc/{pid_match.group(1)}/environ").read_bytes().split(b"\0")
            except OSError:
                continue
            if f"ROS_MASTER_URI={env['ROS_MASTER_URI']}".encode() not in proc_env:
                continue
            xlib = ctypes.CDLL("libX11.so.6")
            xlib.XOpenDisplay.argtypes = [ctypes.c_char_p]
            xlib.XOpenDisplay.restype = ctypes.c_void_p
            xlib.XRaiseWindow.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
            xlib.XFlush.argtypes = [ctypes.c_void_p]
            xlib.XCloseDisplay.argtypes = [ctypes.c_void_p]
            display = xlib.XOpenDisplay(env["DISPLAY"].encode())
            if not display:
                raise RuntimeError("cannot open X display")
            xlib.XRaiseWindow(display, int(window, 16))
            xlib.XFlush(display)
            xlib.XCloseDisplay(display)
            details = subprocess.check_output(["xwininfo", "-id", window], env=env, text=True)
            def field(name):
                return int(re.search(rf"{name}:\s+(-?\d+)", details).group(1))
            x, y = field("Absolute upper-left X"), field("Absolute upper-left Y")
            width, height = field("Width"), field("Height")
            # RViz briefly maps a small splash window with the same PID/title.
            # Wait for the main rendering window rather than recording it.
            if width < 800 or height < 500:
                continue
            root = subprocess.check_output(["xwininfo", "-root"], env=env, text=True)
            root_width = int(re.search(r"Width:\s+(\d+)", root).group(1))
            root_height = int(re.search(r"Height:\s+(\d+)", root).group(1))
            if x < 0 or y < 0 or x + width > root_width or y + height > root_height:
                raise RuntimeError("RViz window extends outside the screen; adjust its saved geometry")
            return {"window_id": window, "x": x, "y": y,
                    "width": width - width % 2, "height": height - height % 2}
        time.sleep(0.2)
    raise RuntimeError("RViz window for this master was not found")


class Observer:
    def __init__(self, robot):
        import rospy
        from sensor_msgs.msg import JointState
        from std_msgs.msg import Float64MultiArray, String
        self.robot, self.latest, self.lock, self.subscriptions = robot, {}, threading.Lock(), []
        for topic, key in VECTOR_TOPICS.items():
            self.subscriptions.append(rospy.Subscriber(topic, Float64MultiArray,
                lambda msg, name=key: self.store(name, list(msg.data)), queue_size=50))
        for topic, key in JOINT_TOPICS.items():
            self.subscriptions.append(rospy.Subscriber(topic, JointState,
                lambda msg, name=key: self.joints(name, msg), queue_size=50))
        self.subscriptions.append(rospy.Subscriber("/deflecomp/estimation_gate_status", String,
            lambda msg: self.store("gate", msg.data), queue_size=50))

    def store(self, key, value):
        import rospy
        with self.lock:
            self.latest[key] = (rospy.get_time(), value)

    def joints(self, key, msg):
        mapping = dict(zip(msg.name, msg.position))
        if all(name in mapping for name in self.robot.model_joint_names):
            self.store(key, [mapping[name] for name in self.robot.model_joint_names])

    def snapshot(self):
        with self.lock:
            return dict(self.latest)


class Demo:
    def __init__(self, config, robot, directory):
        import rospy
        from geometry_msgs.msg import WrenchStamped
        from sensor_msgs.msg import JointState
        from std_msgs.msg import String
        from visualization_msgs.msg import MarkerArray
        self.config, self.robot, self.directory = config, robot, directory
        self.observer = Observer(robot)
        self.reference = rospy.Publisher("/ref/joint_states", JointState, queue_size=10)
        self.load = rospy.Publisher("/deflecomp_sim/external_wrench", WrenchStamped, queue_size=1, latch=True)
        self.event = rospy.Publisher(EVENT_TOPIC, String, queue_size=20, latch=True)
        self.markers = rospy.Publisher(MARKER_TOPIC, MarkerArray, queue_size=2, latch=True)
        self.events = (directory / "events.jsonl").open("w")
        self.start_stamp = None

    def publish_reference(self, q):
        import rospy
        from sensor_msgs.msg import JointState
        msg = JointState()
        msg.header.stamp = rospy.Time.now()
        positions = dict(zip(self.config["joint_names"], np.asarray(q).tolist()))
        msg.name = self.robot.urdf_info.movable_joint_names
        msg.position = [positions.get(name, 0.0) for name in msg.name]
        self.reference.publish(msg)

    def publish_load(self, mass):
        import rospy
        from geometry_msgs.msg import WrenchStamped
        msg = WrenchStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "equil/" + self.config["load_frame"]
        msg.wrench.force.z = -9.81 * mass
        self.load.publish(msg)

    def publish_event(self, kind, phase, stamp=None, details=None):
        import rospy
        from std_msgs.msg import String
        stamp = rospy.get_time() if stamp is None else stamp
        data = {"kind": kind, "phase": phase["name"], "label": phase["label"],
                "stamp": stamp, "time_s": stamp - self.start_stamp,
                "mass_kg": phase["mass_kg"], "target_deg": phase["target_deg"]}
        if details is not None:
            data["details"] = details
        text = json.dumps(data)
        self.event.publish(String(data=text))
        self.events.write(text + "\n")
        self.events.flush()
        print(f"{data['time_s']:6.1f}s {kind:8s} {phase['name']} load={phase['mass_kg']:.2f}kg", flush=True)

    def display(self, phase, mode):
        import rospy
        from geometry_msgs.msg import Point
        from visualization_msgs.msg import Marker, MarkerArray
        stamp = rospy.get_time()
        latest = self.observer.snapshot()
        elapsed = stamp - self.start_stamp if self.start_stamp else 0
        array = MarkerArray()
        text = Marker()
        text.header.frame_id = "base_link"
        text.header.stamp = rospy.Time.now()
        text.ns, text.id, text.type = "demo_status", 0, Marker.TEXT_VIEW_FACING
        text.pose.orientation.w = 1
        text.pose.position.z = 0.78
        text.scale.z = 0.031
        text.color.r, text.color.g, text.color.b, text.color.a = 0.12, 0.15, 0.2, 1.0
        text.text = (f"{elapsed:5.1f} s | {phase['label']} | {mode}\n"
                     f"Tip load: {phase['mass_kg']:.2f} kg ({9.81 * phase['mass_kg']:.2f} N downward)\n"
                     "Green: reference   Pink: plant   Blue: predicted equilibrium")
        row = evaluate_snapshot(latest, self.robot, stamp, self.start_stamp or stamp)
        if row:
            text.text += (f"\nPrediction: {row['prediction_joint_rms_deg']:.2f} deg RMS, "
                          f"{row['prediction_tip_mm']:.1f} mm | estimator ready: {row['gate_ready']}")
        array.markers.append(text)
        for index, (key, color) in enumerate((("reference", (0.0, 0.65, 0.35)),
                    ("actual", (0.85, 0.2, 0.6)), ("prediction", (0.0, 0.45, 0.85)))):
            if key not in latest:
                continue
            pose = self.robot.fk_pose(np.asarray(latest[key][1]))
            marker = Marker()
            marker.header.frame_id, marker.header.stamp = "base_link", rospy.Time.now()
            marker.ns, marker.id, marker.type = "demo_tips", index, Marker.SPHERE
            marker.pose.orientation.w = 1
            marker.pose.position = Point(*pose.translation.tolist())
            marker.scale.x = marker.scale.y = marker.scale.z = 0.025
            marker.color.r, marker.color.g, marker.color.b = color
            marker.color.a = 1
            array.markers.append(marker)
        self.markers.publish(array)

    def ready(self, processes):
        import rospy
        deadline = time.monotonic() + 40
        q = np.zeros(self.robot.nv)
        while time.monotonic() < deadline:
            processes.check()
            self.publish_reference(q)
            self.publish_load(0)
            latest = self.observer.snapshot()
            if (self.reference.get_num_connections() >= 2 and self.load.get_num_connections() >= 2
                    and self.event.get_num_connections() >= 1
                    and evaluate_snapshot(latest, self.robot, rospy.get_time(), rospy.get_time()) is not None):
                return
            time.sleep(0.05)
        raise RuntimeError("required simulation/estimator/recorder connections did not become ready")

    def execute(self, processes):
        import rospy
        q = np.zeros(self.robot.nv)
        self.start_stamp = rospy.get_time()
        manifest_path = self.directory / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["start_stamp"] = self.start_stamp
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        last_display = 0
        for phase in self.config["phases"]:
            processes.check()
            self.publish_load(phase["mass_kg"])
            self.publish_event("phase", phase)
            target = np.radians(phase["target_deg"])
            for mode, duration in (("move", phase["move_s"]), ("hold", phase["hold_s"])):
                monitor = None
                if mode == "hold":
                    q = target.copy()
                    self.publish_reference(q)
                    self.publish_event("hold", phase)
                    if "hold_until" in phase:
                        monitor = PredictionConvergenceMonitor(self.config["criteria"],
                            rospy.get_time() - self.start_stamp,
                            phase.get("min_hold_s", self.config["criteria"]["window_s"]),
                            # Offline samples may end up to one allowed gap
                            # before the phase boundary. Cover that interval too.
                            confirmation_s=self.config["criteria"]["max_sample_gap_s"])
                begin = time.monotonic()
                deadline = begin + duration
                next_evaluation = begin
                converged = False
                while time.monotonic() < deadline:
                    if rospy.is_shutdown():
                        raise KeyboardInterrupt()
                    processes.check()
                    now = time.monotonic()
                    self.publish_reference(interpolate_reference(q, target, now - begin, duration) if mode == "move" else target)
                    if monitor is not None and now >= next_evaluation:
                        stamp = rospy.get_time()
                        row = evaluate_snapshot(self.observer.snapshot(), self.robot, stamp,
                                                self.start_stamp, self.config["criteria"]["max_sample_gap_s"])
                        monitor.observe(row)
                        next_evaluation = now + 1.0 / self.config["analysis_rate_hz"]
                        if row is not None and monitor.converged(stamp - self.start_stamp):
                            self.publish_event("converged", phase, details={
                                "hold_elapsed_s": now - begin,
                                "required_window_s": monitor.criteria["window_s"],
                                "prediction_joint_rms_deg": row["prediction_joint_rms_deg"],
                                "prediction_tip_mm": row["prediction_tip_mm"]})
                            converged = True
                            break
                    if now - last_display >= 0.1:
                        self.publish_load(phase["mass_kg"])
                        self.display(phase, mode)
                        last_display = now
                    time.sleep(min(1.0 / self.config["rate_hz"], max(0, deadline - time.monotonic())))
                if monitor is not None and not converged:
                    self.publish_event("hold_timeout", phase, details={"hold_elapsed_s": duration})
            q = target.copy()
        self.publish_event("finished", self.config["phases"][-1])

    def close(self):
        for _ in range(3):
            self.publish_load(0)
            time.sleep(0.05)
        self.events.close()


def run(args):
    import shutil
    import rospy
    from deflecomp_core.robot.pinocchio_robot import RobotArm

    model, imu = Path(args.model).resolve(), Path(args.imu_config).resolve()
    if not model.is_file() or not imu.is_file():
        raise ValueError("model and IMU configuration must exist; pass --model and --imu-config")
    config = yaml.safe_load(Path(args.scenario).read_text())
    for phase in config["phases"]:
        phase["move_s"] *= args.time_scale
        phase["hold_s"] *= args.time_scale
        if "min_hold_s" in phase:
            phase["min_hold_s"] *= args.time_scale
    robot = RobotArm(str(model), tip_link=config["load_frame"])
    validate_scenario(config, robot)
    if not args.no_video:
        for executable in ("ffmpeg", "ffprobe", "xwininfo", "xprop"):
            if not shutil.which(executable):
                raise ValueError(f"{executable} is required for recording")
        if not os.environ.get("DISPLAY"):
            raise ValueError("DISPLAY is required for RViz recording")
    # Refuse to reuse a live master or overwrite a prior experiment.
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", args.port))
    directory = Path(args.output).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    for source, name in ((model, "model.urdf"), (imu, "imu_frames.yaml")):
        shutil.copy2(source, directory / name)
    env = dict(os.environ, ROS_MASTER_URI=f"http://127.0.0.1:{args.port}",
               ROS_IP="127.0.0.1", ROS_HOME=str(directory / "ros_home"),
               QT_QPA_PLATFORM="xcb")
    env.pop("ROS_HOSTNAME", None)
    env["ROS_LOG_DIR"] = str(directory / "ros_logs")
    os.environ.update({key: env[key] for key in ("ROS_MASTER_URI", "ROS_IP", "ROS_HOME", "ROS_LOG_DIR")})
    os.environ.pop("ROS_HOSTNAME", None)
    manifest = {"schema_version": 1, "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "launcher_pid": os.getpid(),
                "model_source": str(model), "imu_source": str(imu), "scenario": config,
                "ros_master_uri": env["ROS_MASTER_URI"], "completed": False,
                "video": None, "time_basis": "ROS receipt time / wall time"}
    manifest_path = directory / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    processes, demo = Processes(directory, env), None
    print(f"Output: {directory}", flush=True)
    timing = "at most " if any("hold_until" in p for p in config["phases"]) else ""
    print(f"Scenario: {timing}{sum(p['move_s'] + p['hold_s'] for p in config['phases']):.0f}s; isolated master {args.port}", flush=True)
    error = None
    try:
        processes.start("roscore", ["roscore", "-p", str(args.port)])
        wait_master(env["ROS_MASTER_URI"], processes)
        rviz_config = directory / "convergence_demo.rviz"
        make_rviz_config(rviz_config, config.get("rviz_view"))
        processes.start("simulation", ["roslaunch", "deflecomp_ros", "deflecomp_frames.launch",
            f"model:={model}", f"imu_config:={imu}",
            f"viewer:={'false' if args.no_video else 'true'}",
            "start_reference_publisher:=false",
            f"start_kp_plotter:={'true' if args.live_plot and not args.no_video else 'false'}",
            f"rviz_config:={rviz_config}", "particle_scan_enabled:=false"])
        processes.start("rosbag", ["rosbag", "record", "--all", "--buffsize", "256",
                                   "--output-name", str(directory / "experiment.bag")])
        rospy.init_node("deflecomp_convergence_demo", disable_signals=True)
        demo = Demo(config, robot, directory)
        demo.ready(processes)
        (directory / "parameters.yaml").write_text(yaml.safe_dump(rospy.get_param("/"), sort_keys=False))
        if not args.no_video:
            rectangle = capture_rectangle(env, processes)
            video_start = rospy.get_time()
            processes.start("ffmpeg", ["ffmpeg", "-hide_banner", "-loglevel", "warning",
                "-nostdin", "-f", "x11grab", "-framerate", str(args.fps), "-draw_mouse", "0",
                "-video_size", f"{rectangle['width']}x{rectangle['height']}",
                "-i", f"{env['DISPLAY']}+{rectangle['x']},{rectangle['y']}",
                "-c:v", "libx264", "-threads", "2", "-preset", "veryfast", "-crf", "20",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(directory / "rviz.mp4")])
            manifest["video"] = dict(rectangle, fps=args.fps, capture_command_start_stamp=video_start,
                                     note="Use the in-scene time and events for alignment; process startup adds a small video offset.")
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
            time.sleep(0.5)
            processes.check()
        demo.execute(processes)
        manifest = json.loads(manifest_path.read_text())
        manifest["completed"] = True
        manifest["end_stamp"] = rospy.get_time()
        time.sleep(0.5)
    except BaseException as exc:
        error = exc
        manifest = json.loads(manifest_path.read_text())
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        print(f"Recording stopped: {manifest['error']}", file=sys.stderr, flush=True)
    finally:
        if demo is not None:
            demo.close()
        processes.stop({"rosbag", "ffmpeg"})
        if not args.no_video and (directory / "rviz.mp4").is_file():
            result = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                "format=duration:stream=codec_name,width,height,avg_frame_rate", "-of", "json",
                str(directory / "rviz.mp4")], text=True, capture_output=True)
            manifest["video_probe"] = json.loads(result.stdout) if result.returncode == 0 else {"error": result.stderr}
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        processes.stop()
        if demo is not None:
            rospy.signal_shutdown("demo finished")
    report = None
    if (directory / "experiment.bag").is_file() and "start_stamp" in manifest:
        report = analyze_recording(directory)
        print_summary(report)
    if error:
        raise error
    if report is None:
        raise RuntimeError("the recorder did not finalize an analyzable bag")
    print(f"Saved video, bag, CSV, plots and summary: {directory}", flush=True)
    return report


def print_summary(report):
    for phase in report["phases"]:
        if not phase.get("convergence_checked", True):
            state = "brief observation (convergence not checked)"
        elif phase["prediction_stable_at_end"]:
            state = "prediction converged; " + ("stiffness stable" if phase["stable_at_end"] else "stiffness still adapting")
        else:
            state = "prediction has not met the sustained accuracy criterion"
        error = phase["tail_median_prediction_joint_rms_deg"]
        value = "no data" if error is None else f"{error:.3f} deg"
        print(f"{phase['phase']}: {state}; prediction RMS {value}; updates={phase['updates']}", flush=True)
        if phase.get("hold_termination") == "hold_timeout":
            print("  waiting reached its maximum duration; load release was not triggered by convergence", flush=True)


def main():
    default_src = next((parent for parent in PACKAGE.parents
                        if (parent / "nejineji-urdfs/yamaguchi_arm_nejineji").is_dir()), PACKAGE.parent)
    arm = default_src / "nejineji-urdfs/yamaguchi_arm_nejineji"
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run_parser = commands.add_parser("run", help="start isolated simulation and record a complete experiment")
    run_parser.add_argument("--model", default=str(arm / "urdf/yamaguchi_6axis_arm_nejineji.urdf"))
    run_parser.add_argument("--imu-config", default=str(arm / "config/deflecomp_imu_frames.yaml"))
    run_parser.add_argument("--scenario", default=str(PACKAGE / "config/convergence_demo.yaml"))
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_parser.add_argument("--output", default=str(PACKAGE.parent / "outputs/convergence_demo" / stamp))
    run_parser.add_argument("--port", type=int, default=11763)
    run_parser.add_argument("--time-scale", type=float, default=1.0, help="multiply move/hold durations; recorded in manifest")
    run_parser.add_argument("--fps", type=int, default=20)
    run_parser.add_argument("--no-video", action="store_true", help="run without GUI/capture for automated smoke checks")
    run_parser.add_argument("--live-plot", action="store_true", help="also open the existing live stiffness plotter")
    run_parser.add_argument("--require-convergence", action="store_true", help="exit with status 2 if a hold fails convergence criteria")
    analyze_parser = commands.add_parser("analyze", help="regenerate CSV, graphs and summary from a recorded bag")
    analyze_parser.add_argument("directory", help="experiment directory containing manifest and parameter/model snapshots")
    analyze_parser.add_argument("--bag", help="alternate bag path, e.g. an exported or repaired bag")
    analyze_parser.add_argument("--output", help="write regenerated analysis to a separate directory")
    analyze_parser.add_argument("--require-convergence", action="store_true", help="exit with status 2 if a hold fails convergence criteria")
    args = parser.parse_args()
    if args.command == "run":
        if not math_is_positive(args.time_scale) or args.fps <= 0:
            parser.error("--time-scale and --fps must be positive")
        def interrupt(signum, frame):
            raise KeyboardInterrupt()
        signal.signal(signal.SIGTERM, interrupt)
        report = run(args)
    else:
        report = analyze_recording(args.directory, args.bag, args.output)
        print_summary(report)
    if args.require_convergence and not report["all_holds_stable_at_end"]:
        raise SystemExit(2)


def math_is_positive(value):
    return bool(np.isfinite(value) and value > 0)


if __name__ == "__main__":
    main()
