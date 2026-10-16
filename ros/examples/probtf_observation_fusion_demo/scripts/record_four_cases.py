#!/usr/bin/python3
"""Record four live RViz motion replays on an otherwise unused ROS master.

Run after sourcing catkin's setup.bash. Nothing in RViz's settings is edited;
the same window rectangle, capture duration and encoder are used in every case.
The generated manifest records the actual ROS metadata and ffprobe results.
"""

import argparse
import ctypes
import hashlib
import io
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import time
import xmlrpc.client


CASES = ("no_sensors", "imu", "marker", "imu_marker")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stop_owned(process):
    """Only signal a process group started by this recorder."""
    if process is None or process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGINT)
    try:
        process.wait(timeout=12)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=8)


def wait_until(predicate, timeout, message, processes=()):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for process in processes:
            if process.poll() is not None:
                raise RuntimeError("A recorder-owned process exited early: " + message)
        result = predicate()
        if result:
            return result
        time.sleep(0.15)
    raise RuntimeError(message)


def rviz_rectangle(config_name):
    tree = subprocess.check_output(["xwininfo", "-root", "-tree"], text=True)
    matches = [line for line in tree.splitlines() if config_name in line and "RViz" in line]
    if len(matches) != 1:
        return None
    window = re.search(r"\b(0x[0-9a-fA-F]+)\b", matches[0]).group(1)
    detail = subprocess.check_output(["xwininfo", "-id", window], text=True)
    if "Map State: IsViewable" not in detail:
        return None
    values = []
    for field in ("Absolute upper-left X", "Absolute upper-left Y", "Width", "Height"):
        values.append(int(re.search(re.escape(field) + r":\s*(-?\d+)", detail).group(1)))
    x, y, width, height = values
    # yuv420p requires even dimensions; this only omits the outermost pixel.
    return {"window_id": window, "x": x, "y": y,
            "width": width - width % 2, "height": height - height % 2}


def raise_window(window):
    """Bring only the selected RViz window forward, without changing its layout."""
    xlib = ctypes.CDLL("libX11.so.6")
    xlib.XOpenDisplay.restype = ctypes.c_void_p
    xlib.XOpenDisplay.argtypes = [ctypes.c_char_p]
    display = xlib.XOpenDisplay(None)
    if not display:
        raise RuntimeError("Cannot connect to the configured X11 display.")
    for name in ("XRaiseWindow", "XSetInputFocus", "XFlush", "XCloseDisplay"):
        getattr(xlib, name).argtypes = ([ctypes.c_void_p, ctypes.c_ulong]
                                      if name == "XRaiseWindow" else
                                      [ctypes.c_void_p, ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
                                      if name == "XSetInputFocus" else [ctypes.c_void_p])
    xlib.XQueryTree.argtypes = [ctypes.c_void_p, ctypes.c_ulong,
                               ctypes.POINTER(ctypes.c_ulong), ctypes.POINTER(ctypes.c_ulong),
                               ctypes.POINTER(ctypes.POINTER(ctypes.c_ulong)),
                               ctypes.POINTER(ctypes.c_uint)]
    xlib.XFree.argtypes = [ctypes.c_void_p]
    root, parent = ctypes.c_ulong(), ctypes.c_ulong()
    children = ctypes.POINTER(ctypes.c_ulong)()
    count = ctypes.c_uint()
    target = int(window, 16)
    xlib.XQueryTree(display, target, ctypes.byref(root), ctypes.byref(parent),
                    ctypes.byref(children), ctypes.byref(count))
    if children:
        xlib.XFree(children)
    # GNOME reparents the client into its own frame; raise that frame as well.
    xlib.XRaiseWindow(display, parent.value if parent.value != root.value else target)
    xlib.XSetInputFocus(display, target, 2, 0)
    xlib.XFlush(display)
    xlib.XCloseDisplay(display)


def ffprobe(path):
    return json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
        "stream=codec_name,width,height,r_frame_rate,nb_frames:format=duration,size",
        "-of", "json", str(path)], text=True))


def verify_received(run_dir, snapshots, motion, wire):
    """Compare live ROS subscriptions to the immutable estimator export."""
    final = motion[-1]["data"]
    if not (final["finished"] and final["sample_index"] == final["frames"] - 1
            and abs(final["scene_elapsed"] - final["duration"]) < 1e-9):
        raise AssertionError("The recording ended before the complete physical motion.")
    expected_wire = {sha256(path) for path in (run_dir / "node").glob("estimate_*.markers.bin")}
    received_wire = {row["sha256"] for row in wire}
    if not expected_wire or expected_wire != received_wire:
        raise AssertionError("Actual ROS MarkerArrays differ from the estimator exports: "
                             "expected {}, received {}, missing {}".format(
                                 len(expected_wire), len(received_wire),
                                 len(expected_wire - received_wire)))
    archive_only = {"render_seed", "point_size", "display_roundoff_tolerance",
                    "wire_identity_verified"}
    expected_snapshots = set()
    for path in (run_dir / "node").glob("estimate_*.json"):
        record = json.loads(path.read_text())
        if not record["wire_identity_verified"]:
            raise AssertionError("Core-to-adapter identity was not verified.")
        expected_snapshots.add(json.dumps({key: value for key, value in record.items()
                                          if key not in archive_only}, sort_keys=True))
    actual_snapshots = {json.dumps(row["data"], sort_keys=True) for row in snapshots}
    if expected_snapshots != actual_snapshots:
        raise AssertionError("Actual ROS snapshot parameters/provenance differ from core exports.")
    return {"final_motion_observed": True, "snapshot_identity_verified": True,
            "published_wire_identity_verified": True,
            "unique_snapshots": len(actual_snapshots), "unique_wire_messages": len(received_wire)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True,
                        help="New directory for four MP4s and the small manifest.")
    parser.add_argument("--runtime-output", type=Path, required=True,
                        help="New directory for launch logs and full ROS metadata.")
    parser.add_argument("--port", type=int, default=11766)
    parser.add_argument("--duration", type=float, default=42.5)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--config-name", default="observation_fusion_motion.rviz")
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    args = parser.parse_args()
    if args.port in (11311, 11764, 11765):
        parser.error("Existing experiment masters must not be used by this recorder.")
    probe = socket.socket()
    probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        probe.bind(("127.0.0.1", args.port))
    except OSError as exc:
        raise RuntimeError("Recorder master port is already occupied; refusing to use it.") from exc
    finally:
        probe.close()
    args.output.mkdir(parents=True, exist_ok=False)
    args.runtime_output.mkdir(parents=True, exist_ok=False)
    package = Path(__file__).resolve().parent.parent
    config = package / "rviz" / args.config_name
    protected = (config, package / "rviz/observation_fusion_demo.rviz",
                 package.parent / "probtf_switch_press_demo/rviz/switch_press_demo.rviz")
    original_bytes = {path: path.read_bytes() for path in protected}
    settings_hashes = {str(path.relative_to(package.parent.parent.parent)): sha256(path)
                       for path in protected}
    env = dict(os.environ, ROS_MASTER_URI="http://localhost:{}/".format(args.port),
               ROS_HOME=str(args.runtime_output / "ros_home"),
               OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    os.environ.update({key: env[key] for key in
                       ("ROS_MASTER_URI", "ROS_HOME", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")})
    records = []
    manifest = {"master_uri": env["ROS_MASTER_URI"], "settings_sha256": settings_hashes,
                "capture": {"fps": args.fps, "duration_s": args.duration,
                            "encoder": "libx264", "crf": 18, "pixel_format": "yuv420p"},
                "cases": records}
    master = None
    launch = None
    capture = None
    master_log = (args.runtime_output / "roscore.log").open("w")
    try:
        master = subprocess.Popen(["roscore", "-p", str(args.port)], env=env,
                                  stdout=master_log, stderr=subprocess.STDOUT,
                                  start_new_session=True)
        server = xmlrpc.client.ServerProxy(env["ROS_MASTER_URI"])

        def master_ready():
            try:
                return server.getPid("/four_case_recorder")[0] == 1
            except (OSError, xmlrpc.client.Error):
                return False

        wait_until(master_ready, 20, "Recorder-owned ROS master did not start.", (master,))
        import rospy
        from std_msgs.msg import String
        from std_srvs.srv import Trigger
        from visualization_msgs.msg import MarkerArray
        rospy.init_node("four_case_recorder", anonymous=True, disable_signals=True)
        for case in args.cases:
            run_dir = args.runtime_output / case
            run_dir.mkdir()
            snapshot_messages = []
            motion_messages = []
            wire_messages = []
            wire_dir = run_dir / "received_markers"
            wire_dir.mkdir()
            started = time.monotonic()

            def receive(destination, message, origin=started):
                destination.append({"wall_elapsed_s": time.monotonic() - origin,
                                    "data": json.loads(message.data)})

            snapshot_sub = rospy.Subscriber(
                "/observation_fusion/snapshot", String,
                lambda message, target=snapshot_messages, handler=receive: handler(target, message),
                queue_size=100)
            motion_sub = rospy.Subscriber(
                "/observation_fusion/motion_metadata", String,
                lambda message, target=motion_messages, handler=receive: handler(target, message),
                queue_size=100)

            def receive_markers(message, target=wire_messages, directory=wire_dir, origin=started):
                buffer = io.BytesIO()
                message.serialize(buffer)
                payload = buffer.getvalue()
                index = len(target)
                filename = "markers_{:04d}.bin".format(index)
                (directory / filename).write_bytes(payload)
                target.append({"wall_elapsed_s": time.monotonic() - origin,
                                      "file": filename, "sha256": hashlib.sha256(payload).hexdigest(),
                                      "stamp": message.markers[0].header.stamp.to_sec()
                                      if message.markers else None,
                                      "marker_count": len(message.markers)})

            marker_sub = rospy.Subscriber("/observation_fusion/distributions", MarkerArray,
                                          receive_markers, queue_size=100)
            launch_log = (run_dir / "launch.log").open("w")
            try:
                launch = subprocess.Popen([
                    "roslaunch", "probtf_observation_fusion_demo", case + ".launch",
                    "rviz:=true", "autostart:=false", "output_dir:=" + str(run_dir / "node")],
                    env=env, stdout=launch_log, stderr=subprocess.STDOUT, start_new_session=True)
                rospy.wait_for_service("/observation_fusion/replay", timeout=180)
                rectangle = wait_until(lambda: rviz_rectangle(args.config_name), 45,
                                       "The new motion RViz window was not found.", (master, launch))
                wait_until(lambda: snapshot_messages and motion_messages, 180,
                           "The motion node did not publish both metadata topics.", (master, launch))
                roi = {key: rectangle[key] for key in ("x", "y", "width", "height")}
                if records and roi != records[0]["rectangle"]:
                    raise RuntimeError("RViz capture rectangle differs between cases.")
                raise_window(rectangle["window_id"])
                time.sleep(1)
                video = args.output / (case + ".mp4")
                ffmpeg_log = (run_dir / "ffmpeg.log").open("w")
                capture = subprocess.Popen([
                    "ffmpeg", "-hide_banner", "-loglevel", "warning", "-n", "-f", "x11grab",
                    "-draw_mouse", "0", "-framerate", str(args.fps), "-video_size",
                    "{width}x{height}".format(**roi), "-i",
                    "{}+{},{}".format(env.get("DISPLAY", ":1"), roi["x"], roi["y"]),
                    "-t", str(args.duration), "-an", "-c:v", "libx264", "-preset", "veryfast",
                    "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(video)],
                    env=env, stdout=ffmpeg_log, stderr=subprocess.STDOUT, start_new_session=True)
                time.sleep(1.5)
                replay_time = time.monotonic() - started
                response = rospy.ServiceProxy("/observation_fusion/replay", Trigger)()
                if not response.success:
                    raise RuntimeError("Replay was rejected: " + response.message)
                capture.wait(timeout=args.duration + 60)
                if capture.returncode:
                    raise RuntimeError("ffmpeg failed; see " + str(run_dir / "ffmpeg.log"))
                ffmpeg_log.close()
                probe_result = ffprobe(video)
                for label, data in (("snapshots", snapshot_messages), ("motion", motion_messages),
                                    ("received_markers", wire_messages)):
                    with (run_dir / (label + ".jsonl")).open("w") as output:
                        for row in data:
                            output.write(json.dumps(row) + "\n")
                verification = verify_received(run_dir, snapshot_messages, motion_messages, wire_messages)
                final = motion_messages[-1]["data"]
                if records:
                    reference = records[0]["last_motion"]
                    for key in ("command_sha256", "physics_sha256", "duration", "frames", "updates",
                                "reached", "collided", "final_physical_error_m", "final_rigid_error_m"):
                        if final[key] != reference[key]:
                            raise AssertionError("The physical comparison differs between cases: " + key)
                records.append({"case": case, "video": video.name, "sha256": sha256(video),
                                "rectangle": roi, "ffprobe": probe_result,
                                "replay_wall_elapsed_s": replay_time,
                                "snapshot_count": len(snapshot_messages),
                                "motion_count": len(motion_messages),
                                "received_marker_arrays": len(wire_messages),
                                "last_motion": final, "verification": verification,
                                "runtime_record": str(run_dir)})
                (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
                # These frames are extracted from the actual capture for visual review.
                for label, at in (("initial", 2.0), ("middle", args.duration / 2),
                                  ("final", args.duration - 1)):
                    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-n", "-ss",
                                    str(at), "-i", str(video), "-frames:v", "1",
                                    str(run_dir / (label + ".png"))], check=True)
                print(json.dumps({"case": case, "video": str(video),
                                  "motion_messages": len(motion_messages)}), flush=True)
            finally:
                stop_owned(capture)
                capture = None
                stop_owned(launch)
                launch = None
                snapshot_sub.unregister()
                motion_sub.unregister()
                marker_sub.unregister()
                launch_log.close()
                for path, before in original_bytes.items():
                    if path.read_bytes() != before:
                        raise RuntimeError("RViz settings were changed during playback: " + str(path))
                time.sleep(1)
        manifest["settings_byte_comparison_passed"] = True
        manifest["same_commands_and_physics_verified"] = True
        (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    finally:
        stop_owned(capture)
        stop_owned(launch)
        stop_owned(master)
        master_log.close()


if __name__ == "__main__":
    main()
