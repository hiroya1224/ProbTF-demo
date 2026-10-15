#!/usr/bin/env python3
"""One trial, optionally publishing its live state to ROS/RViz."""
import argparse
from datetime import datetime
from pathlib import Path
import sys
from threading import Event

import rospkg

from probtf_switch_press_demo.scene import load_config, write_robot
from probtf_switch_press_demo.trial import Trial


def main():
    packages = rospkg.RosPack()
    package = Path(packages.get_path("probtf_switch_press_demo"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true", help="run and record without ROS")
    parser.add_argument("--robot-description", action="store_true", help="print generated URDF for roslaunch")
    parser.add_argument("--config", default=str(package / "config/default.yaml"))
    parser.add_argument("--output", default=None)
    args, _ros_arguments = parser.parse_known_args()
    source = Path(packages.get_path("symaware_grasp")) / "urdf/simple_six_dof_arm.urdf"
    config = load_config(args.config)
    if args.robot_description:
        # roslaunch invokes this once before starting robot_state_publisher.
        import tempfile
        with tempfile.TemporaryDirectory(prefix="switch-press-urdf-") as directory:
            path = write_robot(source, Path(directory) / "robot.urdf", config)
            sys.stdout.write(path.read_text())
        return 0
    if not args.headless:
        import rospy
        from std_srvs.srv import Trigger, TriggerResponse
        rospy.init_node("switch_press_demo")
        config = load_config(rospy.get_param("~config", args.config))
        args.output = rospy.get_param("~output_dir", args.output) or None
    output = args.output or "/tmp/probtf_switch_press_demo/" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    texture = Path(packages.get_path("prob_artag_detector")) / "samples/apriltag_36h11_id_{:03d}.png".format(config["marker"]["id"])
    trial = Trial(config, source, texture, output, package)
    print("Recording switch demo to " + str(trial.output), flush=True)
    if args.headless:
        summary = trial.run()
        print("{}: error {:.2f} mm, minimum clearance {:.2f} mm".format(
            summary["result"], 1000 * summary["final_error_m"], 1000 * summary["minimum_clearance_m"]), flush=True)
        return 0 if summary["reached"] else 1
    from probtf_switch_press_demo.ros_display import Display
    display = Display(trial, texture)
    requested = Event()
    state = dict(running=False, runs=0)

    def replay(_request):
        if state["running"] or requested.is_set():
            return TriggerResponse(success=False, message="A trial is already running or scheduled")
        requested.set()
        return TriggerResponse(success=True, message="Starting from the initial pose after the display countdown")

    service = rospy.Service("replay", Trigger, replay)
    trial.phase = "ready"
    if rospy.get_param("~autostart", True):
        requested.set()
    while not rospy.is_shutdown():
        if requested.is_set():
            state["running"] = True
            requested.clear()
            if state["runs"]:
                replay_output = str(output) + "_replay_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                trial = Trial(config, source, texture, replay_output, package)
                display.trial = trial
            trial.phase = "initial"
            start = rospy.Time.now()
            while not rospy.is_shutdown() and (rospy.Time.now() - start).to_sec() < rospy.get_param("~startup_delay", 5.0):
                display.publish()
                rospy.sleep(.1)
            trial.run(display.publish, rospy.is_shutdown)
            state["runs"] += 1
            state["running"] = False
            rospy.loginfo("Switch demo: %s; records: %s; replay: rosservice call /switch_press/replay", trial.status, trial.output)
        display.publish()
        rospy.sleep(.2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
