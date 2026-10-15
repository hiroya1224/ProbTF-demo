"""Trial orchestration and small, inspectable records; no ROS dependency."""
import csv
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
from queue import Empty, SimpleQueue
import subprocess
import time

import cv2
import numpy as np
import yaml

from prob_artag_detector.estimator import PoseEstimationError
from probtf.distributions import OrientationKind
from probtf_switch_press_demo.body import EquilibriumBody, ReachMonitor, clearance, motion_samples
from probtf_switch_press_demo.control import CommandServo, Controller, GeometryGuard
from probtf_switch_press_demo.scene import apply, inverse, transform, write_robot
from probtf_switch_press_demo.vision import PlanarCamera


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    raise TypeError("cannot serialize " + type(value).__name__)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, default=json_value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def record_parameters(record):
    return dict(parent=record.parent_frame_id, child=record.child_frame_id, stamp=record.stamp,
        provenance=dict(method=record.provenance.method, detail=record.provenance.detail,
                        source_ids=record.provenance.source_ids),
        components=[dict(weight=item.weight, id=item.component.component_id,
                         orientation_kind=item.component.orientation.kind.value,
                         orientation_parameter=(None if item.component.orientation.kind is OrientationKind.DIRAC
                                                else item.component.orientation.backend_parameter_matrix()),
                         reference_wxyz=item.component.orientation.reference_quaternion_wxyz,
                         mean_at_reference=item.component.translation.mean_at_reference,
                         residual_covariance=item.component.translation.residual_covariance,
                         rotation_coupling=item.component.translation.rotation_coupling)
                    for item in record.distribution.normalized_components()])


def solve_observation(controller, image, command, stamp, updates):
    """Worker owns its controller; no simulated pose or world transform enters it."""
    started = time.monotonic()
    def progress(prediction, goal, result, debug, iteration):
        updates.put((stamp, prediction, goal, result, debug, iteration))
    try:
        selected, goal, result, debug = controller.choose(image, command, stamp, progress)
        error = "" if result is not None else "marker_not_detected"
    except (PoseEstimationError, np.linalg.LinAlgError) as failure:
        selected, goal, result = None, None, None
        debug = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        error = str(failure)
    return dict(selected=selected, goal=goal, result=result, debug=debug, error=error,
                solve_wall_s=time.monotonic() - started,
                pixel_covariance=(None if controller.observation.last_observation is None else
                                  controller.observation.last_observation.image_covariance),
                source_command=np.asarray(command).copy())


def source_version(package_path):
    package_path = Path(package_path).resolve()
    root = next((path for path in package_path.parents if (path / ".git").exists()), None)
    version = dict(git_head="unavailable", source_sha256={})
    if root is not None:
        version["git_head"] = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
        files = [path for path in package_path.rglob("*")
                 if path.is_file() and path.suffix in (".py", ".yaml", ".launch", ".rviz", ".xml", ".txt")
                 and "outputs" not in path.parts]
        files += [root / path for path in (
            "ros/examples/symaware_grasp/urdf/simple_six_dof_arm.urdf",
            "ros/examples/symaware_grasp/configs/simple_six_dof_prob_tf.yaml",
            "ros/examples/symaware_grasp/src/symaware_grasp/probtf_config.py",
            "ros/examples/deflecomp/deflecomp_core/src/deflecomp_core/model/equilibrium.py",
            "ros/examples/deflecomp/deflecomp_core/src/deflecomp_core/model/spring.py",
            "ros/examples/deflecomp/deflecomp_core/src/deflecomp_core/robot/pinocchio_robot.py",
            "ros/examples/prob_artag_detector/src/prob_artag_detector/estimator.py",
            "ros/examples/prob_artag_detector/src/prob_artag_detector/detector.py",
            "ros/core/probtf_core/src/probtf/distributions/composition.py",
            "ros/core/probtf_core/src/probtf/probability/sampling.py",
            "ros/core/probtf_core/src/sampling.cpp",
            "ros/core/probtf_rviz/src/probabilistic_tf_display.cpp",
            "ros/core/probtf_rviz/src/probabilistic_pose_display.cpp",
            "AGENTS.md")]
        files += [path for folder in (root / "ros/core/probtf_rviz/src", root / "ros/core/probtf_rviz/include")
                  for path in folder.rglob("*") if path.is_file() and path.suffix in (".cpp", ".hpp")]
        files += [root / path for path in (
            "ros/core/probtf_core/src/latest_snapshot.cpp",
            "ros/core/probtf_core/include/probtf_core/latest_snapshot.hpp",
            "ros/core/probtf_core/src/probtf/probability/moments.py",
            "ros/core/probtf_core/src/probtf/bingham.py")]
        version["source_sha256"] = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                                    for path in sorted(set(files))}
    return version


class Trial:
    def __init__(self, config, source_urdf, texture_path, output, package_path):
        self.config = config
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=False)
        self.urdf = write_robot(source_urdf, self.output / "robot.urdf", config)
        (self.output / "settings.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        self.version = source_version(package_path)
        self.plant = EquilibriumBody(self.urdf, config["plant"]["stiffness"])
        self.command = np.array(config["initial_command"], dtype=float)
        self.pose = self.plant.equilibrium(self.command)
        self.world_base = transform(config["base_translation"])
        self.marker = inverse(self.world_base) @ transform(config["marker"]["world_translation"], config["marker"]["world_rpy"])
        self.goal = apply(self.marker, config["marker"]["goal_offset"])
        self.tip_offset = np.array([config["rod"]["length"], 0.0, 0.0])
        self.hand_camera = transform(config["camera"]["translation"], config["camera"]["rpy"])
        self.camera = PlanarCamera(texture_path, config["marker"]["size"], config["camera"], config["seed"])
        marker_model = {key: config["marker"][key] for key in
                        ("id", "size", "goal_offset", "goal_orientation_wxyz", "corner_sigma_px", "translation_prior_variance", "adaptive_covariance")}
        self.controller = Controller(self.urdf, config["control"], config["motion"], marker_model,
                                     config["camera"], config["rod"]["length"], config["obstacles"], config["seed"])
        # The guard only queries immutable native-law representatives; no plant K.
        self.guard = GeometryGuard(self.controller.model)
        self.belief_records = self.controller.model.display_records(self.command, time.time())
        self.servo = CommandServo(config["motion"])
        self.monitor = ReachMonitor(self.goal, config["reach_tolerance"])
        self.trajectory, self.cycles = [], []
        self.preview = None
        self.observation_preview = None
        self.raw_image, self.debug_image = None, None
        self.observation_stamp = 0.0
        self.debug_stamp = 0.0
        self.time = 0.0
        self.cycle = 0
        self.status = "running"
        self.observation_count = 0
        self.phase = "initial"
        self.ik_iteration = 0
        self.phases = []
        self.servo_ticks = []
        self.ik_pending = False
        self.camera_history = [(0., self.plant.flange(self.pose) @ self.hand_camera)]

    def set_phase(self, phase):
        self.phase = phase
        self.phases.append(dict(phase=phase, cycle=self.cycle, time=self.time, wall_time=time.time()))

    def sample_actual(self):
        flange = self.plant.flange(self.pose)
        tip = apply(flange, self.tip_offset)
        gap, pair = clearance(self.plant.boxes(self.pose), self.marker, self.config["obstacles"])
        self.monitor.update(tip, gap, 0.)
        self.trajectory.append(dict(time=self.time, cycle=self.cycle, command=self.command.copy(),
            equilibrium=self.pose.copy(), tip=tip, gap=gap, pair=pair,
            error=float(np.linalg.norm(tip - self.goal)), reached=self.monitor.reached,
            collided=self.monitor.collided))
        camera_pose = flange @ self.hand_camera
        if self.camera_history and self.time == self.camera_history[-1][0]:
            self.camera_history[-1] = (self.time, camera_pose)
        else:
            self.camera_history.append((self.time, camera_pose))
        self.camera_history = self.camera_history[-32:]

    def observe(self):
        self.observation_stamp = time.time()
        camera = self.plant.flange(self.pose) @ self.hand_camera
        if self.time > self.camera_history[-1][0]:
            self.camera_history.append((self.time, camera))
            self.camera_history = self.camera_history[-32:]
        self.raw_image = self.camera.render(camera, self.marker, self.camera_history)
        self.observation_count += 1

    def show_prediction(self, request, prediction, goal, result, debug, iteration):
        self.debug_image, self.debug_stamp = debug, request["source_stamp"]
        self.ik_iteration = iteration
        self.preview = dict(prediction=prediction, goal=goal,
                            marker_record=goal.marker_record, goal_record=goal.record,
                            sensor_record=result.record, stamp=request["source_stamp"],
                            source_command=request["command"], cycle=request["cycle"])
        self.observation_preview = self.preview
        # These records are made from the very Hand/Goal objects evaluated by J.
        self.belief_records = prediction.records + (goal.record,)

    def record_plan(self, request, outcome, applied=True):
        selected, goal, result = (outcome[key] for key in ("selected", "goal", "result"))
        self.debug_image, self.debug_stamp = outcome["debug"], request["source_stamp"]
        cv2.imwrite(str(self.output / "camera_{:02d}.png".format(request["cycle"])), self.debug_image)
        entry = dict(cycle=request["cycle"], time=request["source_time"], source_stamp=request["source_stamp"],
            command_before=request["command"], observation_error=outcome["error"],
            image_degradation=request["image_degradation"], pixel_covariance=outcome["pixel_covariance"],
            solve_wall_s=outcome["solve_wall_s"], received_time=self.time,
            command_at_receipt=self.command.copy(),
            moving_ticks_during_ik=sum(np.max(np.abs(tick["velocity"])) > 1e-6
                                      for tick in self.servo_ticks[request["first_tick"]:]))
        if result is not None:
            entry["marker_record"] = record_parameters(result.record)
        if selected is not None:
            self.show_prediction(request, selected, goal, result, self.debug_image, selected.ik["iterations"])
            entry.update(action="retarget" if applied else "discarded_after_stop", increment=selected.increment,
                absolute_target=request["command"] + selected.increment, log_expected_density=selected.score,
                predicted_clearance=selected.minimum_clearance,
                predicted_tips=selected.tips, predicted_rods=selected.rods,
                hand_record=record_parameters(next(record for record in selected.records
                                                  if record.child_frame_id == "rod_tip")),
                root_records=[record_parameters(record) for record in selected.records],
                local_laws=self.controller.model.descriptor,
                goal_record=record_parameters(goal.record),
                propagated_marker_record=record_parameters(goal.marker_record),
                overlap=selected.diagnostics, ik=selected.ik)
            print("IK update {}: {} iterations while {} moving ticks; actual error {:.2f} mm".format(
                request["cycle"] + 1, selected.ik["iterations"], entry["moving_ticks_during_ik"],
                1000 * self.trajectory[-1]["error"]), flush=True)
        else:
            entry["action"] = "hold" if applied else "discarded_after_stop"
            if goal is not None:
                entry["selection_failure"] = getattr(goal, "failure", {})
        self.cycles.append(entry)

    def run(self, on_frame=lambda trial: None, stopped=lambda: False):
        misses = 0
        self.sample_actual()
        initial_error = self.trajectory[0]["error"]
        if self.monitor.collided:
            self.status = "initial_collision"
        elif self.monitor.reached:
            self.status = "initially_reached_scene_invalid"
        worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="expected-likelihood-ik")
        updates = SimpleQueue()
        pending, request, active = None, None, None
        target = self.command.copy()
        period = self.config["motion"]["frame_period"]
        origin = time.monotonic()
        previous_tick, next_tick = origin, origin
        submitted = 0
        try:
            self.set_phase("initial")
            on_frame(self)
            self.set_phase("planning")
            while self.status == "running" and not stopped():
                now = time.monotonic()
                self.time = now - origin
                dt = min(now - previous_tick, 2 * period)
                previous_tick = now
                latest = None
                while True:
                    try:
                        update = updates.get_nowait()
                        if request is not None and update[0] == request["source_stamp"]:
                            latest = update[1:]
                    except Empty:
                        break
                if latest is not None:
                    prediction, goal, result, debug, iteration = latest
                    # An accepted optimizer iterate can already drive the servo.
                    # Its source command and goal remain frozen for this solve.
                    safe, _gap = self.guard.safe_motion(self.command, prediction.command,
                        goal.marker_modes, self.config["obstacles"], self.config["motion"])
                    if safe:
                        self.show_prediction(request, *latest)
                        target = prediction.command.copy()
                        active = dict(command=request["command"], source_time=request["source_time"],
                                      marker_modes=goal.marker_modes, cycle=request["cycle"])
                        if self.phase != "tracking":
                            self.set_phase("tracking")
                if pending is not None and pending.done():
                    outcome = pending.result()
                    self.record_plan(request, outcome)
                    selected, goal = outcome["selected"], outcome["goal"]
                    if selected is None:
                        misses += 1
                        if misses >= self.config["control"]["max_observation_misses"]:
                            self.status = "observation_unavailable" if goal is None else "no_safe_command"
                    else:
                        misses = 0
                        # The old observation's increment is NEVER added to the latest command.
                        target = request["command"] + selected.increment
                        active = dict(command=request["command"], source_time=request["source_time"],
                                      marker_modes=goal.marker_modes, cycle=request["cycle"])
                        if self.phase != "tracking":
                            self.set_phase("tracking")
                    pending = None
                if self.status != "running":
                    break
                if self.time >= self.config["control"]["max_duration"]:
                    self.status = "time_limit"
                    break
                if pending is None:
                    if submitted >= self.config["control"]["max_cycles"]:
                        self.status = "cycle_limit"
                        break
                    self.cycle, self.ik_iteration = submitted, 0
                    self.observe()
                    request = dict(cycle=submitted, source_stamp=self.observation_stamp, source_time=self.time,
                                   command=self.command.copy(),
                                   first_tick=len(self.servo_ticks), image_degradation=self.camera.diagnostics.copy())
                    cv2.imwrite(str(self.output / "camera_clean_{:02d}.png".format(submitted)), self.camera.clean_image)
                    cv2.imwrite(str(self.output / "camera_input_{:02d}.png".format(submitted)), self.raw_image)
                    pending = worker.submit(solve_observation, self.controller, self.raw_image.copy(),
                                            request["command"], request["source_stamp"], updates)
                    submitted += 1
                self.ik_pending = pending is not None
                age = self.time - active["source_time"] if active is not None else None
                live_target = target if age is not None and age <= self.config["control"]["max_plan_age"] else self.command
                start = self.command.copy()
                end = self.servo.step(start, live_target, dt)
                predicted_gap = None
                if np.max(np.abs(end - start)) > 1e-12:
                    safe, predicted_gap = self.guard.safe_motion(start, end, active["marker_modes"], self.config["obstacles"], self.config["motion"])
                    if not safe:
                        self.status = "no_safe_motion"
                        break
                    samples = list(motion_samples(self.plant, start, end, self.pose, self.config["motion"]))[1:]
                    for command, pose in samples:
                        self.command, self.pose = command.copy(), pose.copy()
                        self.sample_actual()
                        if self.monitor.collided:
                            self.status = "collision"
                        elif self.monitor.reached:
                            self.status = "reached"
                        if self.status != "running":
                            break
                self.servo_ticks.append(dict(time=self.time, command=self.command.copy(),
                    velocity=self.servo.velocity.copy(), target=np.asarray(live_target).copy(), ik_pending=self.ik_pending,
                    plan_cycle=active["cycle"] if active else None, plan_age_s=age,
                    predicted_clearance=predicted_gap))
                on_frame(self)
                next_tick = max(next_tick + period, time.monotonic())
                time.sleep(max(0., next_tick - time.monotonic()))
            if self.status == "running":
                self.status = "interrupted"
            self.set_phase(self.status)
            self.ik_pending = False
            on_frame(self)
            if pending is not None:
                self.record_plan(request, pending.result(), applied=False)
            # Final sensor pixels are recorded; comparison laws retain their source.
            self.time = time.monotonic() - origin
            self.observe()
            self.debug_stamp = self.observation_stamp
            try:
                _final_result, self.debug_image = self.controller.observation.estimate(self.raw_image, self.observation_stamp)
            except (PoseEstimationError, np.linalg.LinAlgError):
                self.debug_image = cv2.cvtColor(self.raw_image, cv2.COLOR_GRAY2BGR)
            on_frame(self)
        except BaseException:
            self.status = "execution_error"
            raise
        finally:
            worker.shutdown(wait=True)
            self.save(initial_error)
        return self.summary

    def save(self, initial_error):
        self.summary = dict(result=self.status, reached=self.status == "reached" and self.monitor.reached,
            collided=self.monitor.collided, seed=self.config["seed"], cycles=len(self.cycles),
            image_observations=self.observation_count, actual_samples=len(self.trajectory),
            initial_error_m=initial_error, final_error_m=self.trajectory[-1]["error"],
            minimum_clearance_m=self.monitor.minimum_clearance,
            reached_time_s=self.trajectory[-1]["time"] if self.monitor.reached else None,
            rod_mass_kg=self.config["rod"]["mass"], total_robot_mass_kg=self.plant.robot.total_mass,
            maximum_deflection_rad=max(float(np.max(np.abs(row["equilibrium"] - row["command"]))) for row in self.trajectory),
            nonlinear_ik_solves=sum("ik" in row for row in self.cycles),
            nonlinear_ik_iterations=sum(row.get("ik", {}).get("iterations", 0) for row in self.cycles),
            control_mode="continuous_async_Gaussian_Bingham_expected_likelihood",
            comparison_laws="same immutable Hand/Goal objects for objective and native ROS records",
            actual_collision_check="conservative URDF boxes; adaptively sampled physical path",
            feasibility_check="nominal native-law representatives; no Gaussian-support guarantee",
            trajectory_frame="base_link",
            moving_replans=sum(row["moving_ticks_during_ik"] > 0 for row in self.cycles),
            servo_ticks=len(self.servo_ticks), nominal_servo_hz=1 / self.config["motion"]["frame_period"],
            final_command=self.command, final_equilibrium=self.pose, true_goal_base=self.goal,
            true_goal_world=apply(self.world_base, self.goal), base_translation=self.config["base_translation"],
            **self.version)
        write_json(self.output / "summary.json", self.summary)
        write_json(self.output / "cycles.json", self.cycles)
        write_json(self.output / "phases.json", self.phases)
        write_json(self.output / "servo.json", self.servo_ticks)
        with (self.output / "trajectory.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["time", "cycle"] + ["command_{}".format(i + 1) for i in range(6)] +
                ["equilibrium_{}".format(i + 1) for i in range(6)] +
                ["tip_x", "tip_y", "tip_z", "clearance", "closest_pair", "goal_error", "reached", "collided"])
            for row in self.trajectory:
                writer.writerow([row["time"], row["cycle"], *row["command"], *row["equilibrium"],
                    *row["tip"], row["gap"], row["pair"], row["error"], row["reached"], row["collided"]])
        if self.raw_image is not None:
            cv2.imwrite(str(self.output / "camera_final_raw.png"), self.raw_image)
            cv2.imwrite(str(self.output / "camera_final_detection.png"), self.debug_image)
