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
from probtf.distributions import OrientationKind, compose_with_deterministic_right
from probtf.geometry import DeterministicTransform
from probtf_switch_press_demo.body import EquilibriumBody, ReachMonitor, clearance, motion_samples
from probtf_switch_press_demo.beliefs import left_compose_record
from probtf_switch_press_demo.control import CommandServo, Controller
from probtf_switch_press_demo.kinematics import CandidateKinematics
from probtf.probability.point_density import DensityIntegrationError
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
        components=[dict(weight=item.weight, id=item.component.component_id,
                         orientation_kind=item.component.orientation.kind.value,
                         orientation_parameter=(item.component.orientation.backend_parameter_matrix()
                             if item.component.orientation.kind is OrientationKind.FINITE_BINGHAM else None),
                         inverse_concentration=item.component.orientation.inverse_concentration,
                         orientation_shape=item.component.orientation.shape_matrix,
                         reference_wxyz=item.component.orientation.reference_quaternion_wxyz,
                         mean_at_reference=item.component.translation.mean_at_reference,
                         residual_covariance=item.component.translation.residual_covariance,
                         rotation_coupling=item.component.translation.rotation_coupling)
                    for item in record.distribution.normalized_components()])


def density_diagnostics_summary(diagnostics, metadata):
    """Keep integration diagnostics without copying per-point proposal tables."""
    components = []
    count = max((len(point.get("components", ())) for point in diagnostics), default=0)
    for index in range(count):
        entries = [point["components"][index] for point in diagnostics
                   if index < len(point.get("components", ()))]
        stationary = [item["maximum_stationary_gradient"] for item in entries
                      if "maximum_stationary_gradient" in item]
        components.append(dict(component_index=index,
            maximum_modes=max(item.get("modes", 0) for item in entries),
            maximum_stationary_gradient=max(stationary) if stationary else None))
    return dict(integration_method=metadata.get("method"), order=metadata.get("order"),
                error_bound=None, evaluated_point_count=len(diagnostics), components=components)


def solve_observation(controller, image, command, stamp, updates):
    """Worker owns its controller; no simulated pose or world transform enters it."""
    started = time.monotonic()
    error_diagnostics = None
    def progress(prediction, goal, result, debug, iteration):
        prediction.root_records
        prediction.hand_record
        updates.put((stamp, prediction, goal, result, debug, iteration))
    try:
        selected, goal, result, debug = controller.choose(image, command, stamp, progress)
        error = "" if result is not None else "marker_not_detected"
    except (PoseEstimationError, np.linalg.LinAlgError, DensityIntegrationError) as failure:
        selected, goal, result = None, None, None
        debug = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        error = str(failure)
        error_diagnostics = getattr(failure, "diagnostics", None)
    # Complete the selected law's native views in the worker as well. Their
    # validation/serialization must not pause the motion loop at each retarget.
    native_links = None
    if selected is not None:
        native_links = [record_parameters(record) for record in selected.link_records]
        selected.root_records
    return dict(selected=selected, goal=goal, result=result, debug=debug, error=error,
                error_diagnostics=error_diagnostics,
                solve_wall_s=time.monotonic() - started,
                native_link_records=native_links,
                pixel_covariance=(None if controller.observation.last_observation is None else
                                  controller.observation.last_observation.image_covariance),
                current_transform_candidates=controller.candidates.current_measure.local_transforms)


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
            "ros/examples/deflecomp/deflecomp_core/src/deflecomp_core/model/equilibrium.py",
            "ros/examples/deflecomp/deflecomp_core/src/deflecomp_core/model/spring.py",
            "ros/examples/deflecomp/deflecomp_core/src/deflecomp_core/robot/pinocchio_robot.py",
            "ros/examples/prob_artag_detector/src/prob_artag_detector/estimator.py",
            "ros/examples/prob_artag_detector/src/prob_artag_detector/detector.py",
            "ros/core/probtf_core/src/probtf/distributions/composition.py",
            "ros/core/probtf_core/src/probtf/probability/sampling.py")]
        files += list((root / "ros/core/probtf_core/src/probtf").rglob("*.py"))
        files += [path for path in (root / "ros/core/probtf_rviz/src").rglob("*")
                  if path.is_file() and path.suffix in (".cpp", ".hpp", ".h")]
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
                        ("id", "size", "goal_offset", "corner_sigma_px", "translation_prior_variance", "adaptive_covariance")}
        self.controller = Controller(self.urdf, config["control"], config["motion"], marker_model,
                                     config["camera"], config["rod"]["length"], config["obstacles"], config["seed"])
        # Worker and motion guard share the source law, with separate query caches.
        self.guard = CandidateKinematics(self.urdf, self.controller.candidates.source_law,
            config["rod"]["length"], self.controller.candidates.source_law.reference_command)
        write_json(self.output / "initial_transform_law.json", dict(
            law=self.controller.candidates.law.descriptor(),
            meaning="configured native joint transform law; no stiffness or additive joint-noise assumption in IK"))
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
        self.applied_ik_updates = []
        self.ik_pending = False
        self.camera_history = [(0., self.plant.flange(self.pose) @ self.hand_camera)]

    def set_phase(self, phase):
        self.phase = phase
        self.phases.append(dict(phase=phase, cycle=self.cycle, time=self.time, wall_time=time.time()))

    def sample_actual(self):
        flange = self.plant.flange(self.pose)
        tip = apply(flange, self.tip_offset)
        gap, pair = clearance(self.plant.boxes(self.pose), self.marker, self.config["obstacles"])
        self.monitor.update(tip, gap, self.config["motion"]["clearance"])
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

    def show_prediction(self, request, prediction, goal, result, debug, iteration, selected=False):
        self.debug_image, self.debug_stamp = debug, request["source_stamp"]
        self.ik_iteration = iteration
        preview = dict(hand=request["hand"], prediction=prediction,
                            marker_record=result.record, goal_record=goal.record,
                            stamp=request["source_stamp"], cycle=request["cycle"])
        self.observation_preview = preview
        # Accepted safe progress is already an active command. Its native
        # views were prepared by the worker, so display the same query.
        self.preview = preview

    def record_plan(self, request, outcome, applied=True, discard_reason=None):
        selected, goal, result = (outcome[key] for key in ("selected", "goal", "result"))
        self.debug_image, self.debug_stamp = outcome["debug"], request["source_stamp"]
        cv2.imwrite(str(self.output / "camera_{:02d}.png".format(request["cycle"])), self.debug_image)
        entry = dict(cycle=request["cycle"], time=request["source_time"], source_stamp=request["source_stamp"],
            command_before=request["command"], observation_error=outcome["error"],
            observation_diagnostics=outcome.get("error_diagnostics"),
            image_degradation=request["image_degradation"], pixel_covariance=outcome["pixel_covariance"],
            solve_wall_s=outcome["solve_wall_s"], received_time=self.time,
            command_at_receipt=self.command.copy(),
            applied_updates=list(request.get("applied_updates", ())),
            moving_ticks_during_ik=sum(np.max(np.abs(tick["velocity"])) > 1e-6
                                      for tick in self.servo_ticks[request["first_tick"]:]))
        discard_action = "discarded_stale" if discard_reason == "stale" else "discarded_after_stop"
        if discard_reason is not None:
            entry["discard_reason"] = discard_reason
        if result is not None:
            entry["marker_record"] = record_parameters(result.record)
        if selected is not None:
            if applied:
                self.show_prediction(request, selected, goal, result, self.debug_image, selected.ik["iterations"], selected=True)
            entry.update(action="retarget" if applied else discard_action, increment=selected.increment,
                absolute_target=request["command"] + selected.increment, log_expected_density=selected.score,
                predicted_clearance=selected.minimum_clearance, body_ids=self.guard.ids, body_weights=self.guard.weights,
                predicted_tips=selected.tips, predicted_rods=selected.rods,
                hand_record=outcome["native_link_records"][-1],
                joint_measure_before=dict(body_ids=selected.joint_before.body_ids,
                    query=dict(selected.joint_before.metadata),
                    weights=selected.joint_before.weights, local_transforms=selected.joint_before.local_transforms),
                joint_measure_after=dict(body_ids=selected.joint_after.body_ids,
                    query=dict(selected.joint_after.metadata),
                    weights=selected.joint_after.weights, local_transforms=selected.joint_after.local_transforms,
                    root_transforms=selected.joint_after.root_transforms),
                link_records=outcome["native_link_records"],
                reference_hand_base=request["hand"],
                reference_hand_world=self.world_base @ request["hand"],
                current_transform_candidates=outcome["current_transform_candidates"], next_transform_candidates=selected.poses,
                goal_record=record_parameters(goal.record),
                goal_integration_metadata=goal.metadata, ik=selected.ik)
            print("IK update {}: {} iterations while {} moving ticks; actual error {:.2f} mm".format(
                request["cycle"] + 1, selected.ik["iterations"], entry["moving_ticks_during_ik"],
                1000 * self.trajectory[-1]["error"]), flush=True)
        else:
            entry["action"] = (("continue_last_target" if entry["applied_updates"] else "hold")
                               if applied else discard_action)
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
                    prediction, goal = latest[:2]
                    if self.time - request["source_time"] <= self.config["control"]["max_plan_age"]:
                        self.show_prediction(request, *latest)
                        # Each callback passed the same full candidate-path
                        # feasibility check as the final selection. The target
                        # stays absolute in its original observation frame.
                        target = request["command"] + prediction.increment
                        active = dict(command=request["command"], source_time=request["source_time"],
                                      marker_modes=goal.marker_modes, cycle=request["cycle"])
                        request["applied_updates"].append(len(self.applied_ik_updates))
                        self.applied_ik_updates.append(dict(
                            source_stamp=request["source_stamp"], source_time=request["source_time"],
                            applied_time=self.time, cycle=request["cycle"], iteration=latest[-1],
                            reference_command=request["command"].copy(), increment=prediction.increment.copy(),
                            display_reference_hand_base=request["hand"].copy(),
                            absolute_target=target.copy(), log_expected_density=prediction.score,
                            predicted_clearance=prediction.minimum_clearance,
                            joint_measure_before_query=dict(prediction.joint_before.metadata),
                            joint_measure_after_query=dict(prediction.joint_after.metadata),
                            goal_record=record_parameters(goal.record), goal_integration_metadata=goal.metadata,
                            expected_density_diagnostics=prediction.ik["expected_density_diagnostics"],
                            goal_density_diagnostics=density_diagnostics_summary(
                                prediction.ik["goal_density_diagnostics"], goal.metadata)))
                        if self.phase != "tracking":
                            self.set_phase("tracking")
                if pending is not None and pending.done():
                    outcome = pending.result()
                    fresh = self.time - request["source_time"] <= self.config["control"]["max_plan_age"]
                    self.record_plan(request, outcome, applied=fresh,
                                     discard_reason=None if fresh else "stale")
                    selected, goal = outcome["selected"], outcome["goal"]
                    if fresh and selected is None:
                        misses += 1
                        if misses >= self.config["control"]["max_observation_misses"]:
                            self.status = "observation_unavailable" if goal is None else "no_safe_command"
                    elif fresh:
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
                                   command=self.command.copy(), hand=self.plant.flange(self.pose),
                                   first_tick=len(self.servo_ticks), image_degradation=self.camera.diagnostics.copy(),
                                   applied_updates=[])
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
                    safe, predicted_gap = self.guard.safe_motion(start, end, active["command"],
                        active["marker_modes"], self.config["obstacles"], self.config["motion"])
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
            # Fresh final pixels and detection for display; no further command is selected.
            self.time = time.monotonic() - origin
            self.observe()
            self.debug_stamp = self.observation_stamp
            try:
                _final_result, self.debug_image = self.controller.observation.estimate(self.raw_image, self.observation_stamp)
            except (PoseEstimationError, np.linalg.LinAlgError):
                self.debug_image = cv2.cvtColor(self.raw_image, cv2.COLOR_GRAY2BGR)
                _final_result = None
            if _final_result is not None:
                goal_record = compose_with_deterministic_right(_final_result.record,
                    DeterministicTransform(np.asarray(self.config["marker"]["goal_offset"]), [1, 0, 0, 0]),
                    child_frame_id="switch_goal", edge_id="observed_switch_goal")
                goal_record = left_compose_record(goal_record, self.hand_camera, "observation_hand")
                self.observation_preview = dict(hand=self.plant.flange(self.pose), marker_record=_final_result.record,
                    goal_record=goal_record, stamp=self.observation_stamp)
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
            control_mode="continuous_async_expected_likelihood",
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
        write_json(self.output / "applied_ik_updates.json", self.applied_ik_updates)
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
        if self.debug_image is not None:
            cv2.imwrite(str(self.output / "camera_final_detection.png"), self.debug_image)
